from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import Operator, launch_count, reset_launch_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_moe_movement_override_preserves_vendor_grouped_gemm():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe import moe_permute_unpermute
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe

    from loom_kernels.vllm import (
        MOE_MOVEMENT_OVERRIDE_KEY,
        provider_metadata,
        register_vllm_moe_movement,
    )

    original_permute = moe_permute_unpermute.moe_permute
    original_unpermute = moe_permute_unpermute.moe_unpermute
    vendor_grouped_gemm = cutlass_moe.ops.cutlass_moe_mm

    tokens = 8
    hidden_size = 64
    top_k = 2
    experts = 4
    hidden_storage = torch.arange(
        tokens * hidden_size, dtype=torch.uint8, device="cuda"
    ).reshape(tokens, hidden_size)
    hidden_states = hidden_storage.view(torch.float8_e4m3fn)
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 1],
            [3, 0],
            [1, 2],
            [0, 3],
            [2, 0],
            [3, 1],
            [1, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    scales = torch.arange(1, tokens + 1, dtype=torch.float32, device="cuda")[
        :, None
    ]
    baseline_scratch = moe_permute_unpermute.MoEPermuteScratch(
        max_num_tokens=tokens,
        topk=top_k,
        num_experts=experts,
        num_local_experts=experts,
        device=torch.device("cuda"),
    )
    expected_permuted_storage = torch.empty(
        (tokens * top_k, hidden_size), dtype=torch.uint8, device="cuda"
    )
    expected = original_permute(
        hidden_states,
        scales,
        topk_ids,
        experts,
        experts,
        None,
        expected_permuted_storage.view(torch.float8_e4m3fn),
        baseline_scratch,
    )

    assert register_vllm_moe_movement() == MOE_MOVEMENT_OVERRIDE_KEY
    assert cutlass_moe.ops.cutlass_moe_mm is vendor_grouped_gemm
    actual_scratch = moe_permute_unpermute.MoEPermuteScratch(
        max_num_tokens=tokens,
        topk=top_k,
        num_experts=experts,
        num_local_experts=experts,
        device=torch.device("cuda"),
    )
    actual_permuted_storage = torch.empty_like(expected_permuted_storage)
    reset_launch_count(Operator.MOE_PERMUTE)
    actual = cutlass_moe.moe_permute(
        hidden_states,
        scales,
        topk_ids,
        experts,
        experts,
        None,
        actual_permuted_storage.view(torch.float8_e4m3fn),
        actual_scratch,
    )
    torch.cuda.synchronize()

    assert launch_count(Operator.MOE_PERMUTE) == 1
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
    assert torch.equal(actual[4], expected[4])

    expert_outputs = torch.randn(
        tokens * top_k, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.softmax(
        torch.randn(tokens, top_k, dtype=torch.float32, device="cuda"), dim=-1
    )
    expected_output = torch.empty(
        tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    original_unpermute(
        expected_output,
        expert_outputs,
        weights,
        expected[3],
        expected[2],
    )
    actual_output = torch.empty_like(expected_output)
    reset_launch_count(Operator.MOE_COMBINE)
    cutlass_moe.moe_unpermute(
        actual_output,
        expert_outputs,
        weights,
        actual[3],
        actual[2],
    )
    torch.cuda.synchronize()

    assert launch_count(Operator.MOE_COMBINE) == 1
    torch.testing.assert_close(
        actual_output, expected_output, rtol=2.0e-2, atol=2.0e-2
    )
    metadata = provider_metadata()
    assert metadata["moe_movement_override"] is True
    assert metadata["moe_movement_permute_hits"] >= 1
    assert metadata["moe_movement_combine_hits"] >= 1
    assert metadata["moe_grouped_gemm_owner"] == "vllm_vendor_backend"
