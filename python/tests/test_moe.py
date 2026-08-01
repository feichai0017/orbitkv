from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    moe_combine,
    moe_permute,
    reset_launch_count,
    supports_moe_combine,
    supports_moe_permute,
)


def reference_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_local_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = topk_ids.numel()
    flattened_ids = topk_ids.flatten()
    if expert_map is None:
        local_ids = flattened_ids
    else:
        local_ids = expert_map[flattened_ids.long()]
    valid = (local_ids >= 0) & (local_ids < num_local_experts)
    remote_sort_keys = flattened_ids + (
        expert_map.numel() if expert_map is not None else num_local_experts
    )
    sort_keys = torch.where(valid, local_ids, remote_sort_keys)
    order = torch.argsort(sort_keys, stable=True)
    sorted_keys = sort_keys[order]
    sorted_valid = sorted_keys < num_local_experts
    top_k = topk_ids.shape[1]
    gathered = hidden_states[torch.div(order, top_k, rounding_mode="floor")]
    permuted = torch.where(sorted_valid[:, None], gathered, torch.zeros_like(gathered))
    offsets = torch.stack(
        [(sorted_keys < expert).sum() for expert in range(num_local_experts + 1)]
    ).to(torch.int64)
    inverse = torch.empty(assignments, dtype=torch.int32, device=topk_ids.device)
    inverse[order] = torch.arange(
        assignments, dtype=torch.int32, device=topk_ids.device
    )
    assignment_ids = torch.where(
        sorted_valid,
        order.to(torch.int32),
        torch.full_like(order, assignments, dtype=torch.int32),
    )
    return permuted, offsets, inverse.view_as(topk_ids), assignment_ids


def reference_combine(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    valid_assignments = expert_offsets[-1].clamp(
        min=0, max=expert_outputs.shape[0]
    )
    rows = inverse_permutation.long()
    valid = rows < valid_assignments
    selected = expert_outputs[rows.clamp(min=0, max=expert_outputs.shape[0] - 1)]
    weighted = selected.float() * routing_weights[..., None]
    return torch.where(valid[..., None], weighted, torch.zeros_like(weighted)).sum(
        dim=1
    ).to(expert_outputs.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moe_permute_matches_stable_reference_on_external_stream(dtype):
    hidden_states = torch.arange(20, device="cuda", dtype=torch.float32).reshape(5, 4)
    hidden_states = hidden_states.to(dtype)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    expected = reference_permute(
        hidden_states, topk_ids, num_local_experts=3
    )

    reset_launch_count(Operator.MOE_PERMUTE)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = moe_permute(hidden_states, topk_ids, num_experts=3)
    stream.synchronize()

    assert launch_count(Operator.MOE_PERMUTE) == 1
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_permute_preserves_fp8_storage_bytes():
    hidden_storage = torch.arange(32, dtype=torch.uint8, device="cuda").reshape(4, 8)
    hidden_states = hidden_storage.view(torch.float8_e4m3fn)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    expected = reference_permute(
        hidden_states, topk_ids, num_local_experts=3
    )

    actual = moe_permute(hidden_states, topk_ids, num_experts=3)
    torch.cuda.synchronize()

    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    for actual_tensor, expected_tensor in zip(actual[1:], expected[1:]):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_permute_expert_parallel_zeroes_remote_tail():
    hidden_states = torch.arange(
        32, device="cuda", dtype=torch.bfloat16
    ).reshape(4, 8)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int32, device="cuda")
    expected = reference_permute(
        hidden_states,
        topk_ids,
        num_local_experts=2,
        expert_map=expert_map,
    )

    actual = moe_permute(
        hidden_states,
        topk_ids,
        num_experts=3,
        num_local_experts=2,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)
    valid_assignments = int(actual[1][-1].item())
    assert torch.count_nonzero(actual[0][valid_assignments:]) == 0
    assert torch.all(actual[3][valid_assignments:] == topk_ids.numel())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moe_combine_matches_weighted_inverse_reference(dtype):
    torch.manual_seed(701)
    hidden_states = torch.randn(7, 32, device="cuda", dtype=dtype)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1], [0, 2], [1, 0], [2, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    routing_weights = torch.softmax(
        torch.randn(7, 2, device="cuda", dtype=torch.float32), dim=-1
    )
    permuted, offsets, inverse, _ = moe_permute(
        hidden_states, topk_ids, num_experts=3
    )
    expert_outputs = (permuted.float() * 0.75 + 0.125).to(dtype)
    expected = reference_combine(
        expert_outputs, routing_weights, inverse, offsets
    )

    reset_launch_count(Operator.MOE_COMBINE)
    actual = moe_combine(expert_outputs, routing_weights, inverse, offsets)
    torch.cuda.synchronize()

    assert launch_count(Operator.MOE_COMBINE) == 1
    tolerance = {
        torch.float32: (1.0e-6, 1.0e-6),
        torch.float16: (2.0e-3, 2.0e-3),
        torch.bfloat16: (2.0e-2, 2.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_combine_skips_remote_expert_parallel_assignments():
    hidden_states = torch.arange(
        16, device="cuda", dtype=torch.float32
    ).reshape(4, 4)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int32, device="cuda")
    routing_weights = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    permuted, offsets, inverse, _ = moe_permute(
        hidden_states,
        topk_ids,
        num_experts=3,
        num_local_experts=2,
        expert_map=expert_map,
    )
    valid_assignments = int(offsets[-1].item())
    expert_outputs = permuted.clone()
    expert_outputs[valid_assignments:] = 10_000.0

    actual = moe_combine(expert_outputs, routing_weights, inverse, offsets)
    expected = reference_combine(
        expert_outputs, routing_weights, inverse, offsets
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_dispatcher_contract_and_fake_tensor():
    hidden_states = torch.randn(4, 32, device="cuda", dtype=torch.float16)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.library.opcheck(
        torch.ops.loom_kernels.moe_permute.default,
        (hidden_states, topk_ids, 3, 3, None),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )
    permuted, offsets, inverse, _ = moe_permute(
        hidden_states, topk_ids, num_experts=3
    )
    weights = torch.full((4, 2), 0.5, device="cuda", dtype=torch.float32)
    torch.library.opcheck(
        torch.ops.loom_kernels.moe_combine.default,
        (permuted, weights, inverse, offsets),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_caller_owned_overloads_match_allocating_operators():
    hidden_states = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 1], [2, 1], [1, 0], [2, 0], [1, 2], [0, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    expected_permuted, expected_offsets, expected_inverse, expected_ids = moe_permute(
        hidden_states, topk_ids, num_experts=3
    )
    permuted = torch.empty_like(expected_permuted)
    offsets = torch.empty_like(expected_offsets)
    inverse = torch.empty_like(expected_inverse)
    assignment_ids = torch.empty_like(expected_ids)
    torch.ops.loom_kernels.moe_permute.out(
        hidden_states,
        topk_ids,
        3,
        3,
        None,
        permuted,
        offsets,
        inverse,
        assignment_ids,
    )
    assert torch.equal(permuted, expected_permuted)
    assert torch.equal(offsets, expected_offsets)
    assert torch.equal(inverse, expected_inverse)
    assert torch.equal(assignment_ids, expected_ids)

    weights = torch.softmax(torch.randn(6, 2, device="cuda"), dim=-1)
    expected_output = moe_combine(permuted, weights, inverse, offsets)
    output = torch.empty_like(expected_output)
    torch.ops.loom_kernels.moe_combine.out(
        permuted, weights, inverse, offsets, output
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_ops_survive_torch_compile():
    @torch.compile(fullgraph=True)
    def compiled(hidden_states, topk_ids, weights):
        permuted, offsets, inverse, assignment_ids = moe_permute(
            hidden_states, topk_ids, num_experts=3
        )
        combined = moe_combine(permuted, weights, inverse, offsets)
        return combined, offsets, assignment_ids

    hidden_states = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 1], [2, 1], [1, 0], [2, 0], [1, 2], [0, 2], [2, 1], [0, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.softmax(torch.randn(8, 2, device="cuda"), dim=-1)
    actual = compiled(hidden_states, topk_ids, weights)
    expected_permuted, expected_offsets, expected_inverse, expected_ids = (
        reference_permute(hidden_states, topk_ids, num_local_experts=3)
    )
    expected = reference_combine(
        expected_permuted, weights, expected_inverse, expected_offsets
    )
    torch.testing.assert_close(actual[0], expected, rtol=2.0e-2, atol=2.0e-2)
    assert torch.equal(actual[1], expected_offsets)
    assert torch.equal(actual[2], expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_ops_can_be_captured_and_replayed():
    hidden_states = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    topk_ids = torch.tensor(
        [[0, 1], [2, 1], [1, 0], [2, 0], [1, 2], [0, 2], [2, 1], [0, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.full((8, 2), 0.5, device="cuda", dtype=torch.float32)
    for _ in range(3):
        permuted, offsets, inverse, _ = moe_permute(
            hidden_states, topk_ids, num_experts=3
        )
        output = moe_combine(permuted, weights, inverse, offsets)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        permuted, offsets, inverse, _ = moe_permute(
            hidden_states, topk_ids, num_experts=3
        )
        output = moe_combine(permuted, weights, inverse, offsets)
    hidden_states.add_(1)
    graph.replay()
    torch.cuda.synchronize()

    expected_permuted, expected_offsets, expected_inverse, _ = reference_permute(
        hidden_states, topk_ids, num_local_experts=3
    )
    expected = reference_combine(
        expected_permuted, weights, expected_inverse, expected_offsets
    )
    torch.testing.assert_close(output, expected, rtol=2.0e-3, atol=2.0e-3)


def test_moe_public_contract_rejects_cpu_and_missing_expert_map():
    hidden_states = torch.randn(4, 8)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]], dtype=torch.int32)
    assert not supports_moe_permute(
        hidden_states, topk_ids, num_experts=3
    )
    with pytest.raises(ValueError, match="contiguous inference CUDA tensors"):
        moe_permute(hidden_states, topk_ids, num_experts=3)
    with pytest.raises(ValueError, match="expert map"):
        moe_permute(
            hidden_states.cuda() if torch.cuda.is_available() else hidden_states,
            topk_ids.cuda() if torch.cuda.is_available() else topk_ids,
            num_experts=3,
            num_local_experts=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_permute_matches_vllm_metadata_when_available():
    pytest.importorskip("vllm")
    try:
        import vllm._custom_ops  # noqa: F401

        vllm_permute = torch.ops._moe_C.moe_permute.default
    except (AttributeError, ImportError, RuntimeError):
        pytest.skip("installed vLLM does not expose _moe_C::moe_permute")

    hidden_states = torch.arange(
        32, device="cuda", dtype=torch.bfloat16
    ).reshape(4, 8)
    topk_ids = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    assignments = topk_ids.numel()
    token_expert_indices = torch.arange(
        assignments, dtype=torch.int32, device="cuda"
    ).view_as(topk_ids)
    expected_permuted = torch.empty(
        assignments, hidden_states.shape[1], dtype=hidden_states.dtype, device="cuda"
    )
    expected_offsets = torch.empty(4, dtype=torch.int64, device="cuda")
    expected_inverse = torch.empty_like(topk_ids)
    expected_ids = torch.empty(assignments, dtype=torch.int32, device="cuda")
    vllm_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        None,
        3,
        3,
        2,
        expected_permuted,
        expected_offsets,
        expected_inverse,
        expected_ids,
    )
    actual = moe_permute(hidden_states, topk_ids, num_experts=3)
    torch.cuda.synchronize()

    assert torch.equal(actual[0], expected_permuted)
    assert torch.equal(actual[1], expected_offsets)
    assert torch.equal(actual[2], expected_inverse)
    assert torch.equal(actual[3], expected_ids)

    routing_weights = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    expected_combined = torch.empty_like(hidden_states)
    torch.ops._moe_C.moe_unpermute.default(
        expected_permuted,
        routing_weights,
        expected_inverse,
        expected_offsets,
        2,
        expected_combined,
    )
    actual_combined = moe_combine(actual[0], routing_weights, actual[2], actual[1])
    torch.testing.assert_close(
        actual_combined, expected_combined, rtol=2.0e-2, atol=2.0e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_permute_matches_vllm_expert_parallel_scratch_tail():
    pytest.importorskip("vllm")
    try:
        from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
            MoEPermuteScratch,
            moe_permute as vllm_moe_permute,
            moe_unpermute as vllm_moe_unpermute,
        )
    except (AttributeError, ImportError, RuntimeError):
        pytest.skip("installed vLLM does not expose MoEPermuteScratch")

    hidden_states = torch.arange(
        48, device="cuda", dtype=torch.bfloat16
    ).reshape(6, 8)
    topk_ids = torch.tensor(
        [[4, 0], [1, 5], [3, 2], [5, 1], [0, 4], [2, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    expert_map = torch.tensor(
        [0, 1, 2, -1, -1, -1], dtype=torch.int32, device="cuda"
    )
    scratch = MoEPermuteScratch(
        max_num_tokens=6,
        topk=2,
        num_experts=6,
        num_local_experts=3,
        device=torch.device("cuda"),
        hidden_size=8,
        hidden_dtype=torch.bfloat16,
    )
    expected_permuted, _, expected_offsets, expected_inverse, expected_ids = (
        vllm_moe_permute(
            hidden_states,
            None,
            topk_ids,
            6,
            3,
            expert_map,
            scratch=scratch,
        )
    )
    actual = moe_permute(
        hidden_states,
        topk_ids,
        num_experts=6,
        num_local_experts=3,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()

    valid_assignments = int(expected_offsets[-1].item())
    assert torch.equal(
        actual[0][:valid_assignments], expected_permuted[:valid_assignments]
    )
    assert torch.count_nonzero(actual[0][valid_assignments:]) == 0
    assert torch.equal(actual[1], expected_offsets)
    assert torch.equal(actual[2].flatten(), expected_inverse)
    assert torch.equal(actual[3], expected_ids)

    routing_weights = torch.softmax(
        torch.randn(6, 2, device="cuda", dtype=torch.float32), dim=-1
    )
    expected_combined = torch.empty_like(hidden_states)
    vllm_moe_unpermute(
        expected_combined,
        expected_permuted,
        routing_weights,
        expected_inverse,
        expected_offsets,
    )
    actual_combined = moe_combine(actual[0], routing_weights, actual[2], actual[1])
    torch.testing.assert_close(
        actual_combined, expected_combined, rtol=2.0e-2, atol=2.0e-2
    )
