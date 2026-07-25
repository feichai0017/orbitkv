from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    topk_sampled_logprobs,
)


def reference(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_f32 = logits.float()
    all_logprobs = logits_f32.log_softmax(dim=-1)
    _, sorted_token_ids = torch.sort(
        logits_f32, dim=-1, descending=True, stable=True
    )
    top_token_ids = sorted_token_ids[:, :top_k]
    top_logprobs = all_logprobs.gather(-1, top_token_ids)
    sampled_values = all_logprobs.gather(
        -1, sampled_token_ids.unsqueeze(-1)
    )
    sampled_logits = logits_f32.gather(
        -1, sampled_token_ids.unsqueeze(-1)
    )
    ranks = (logits_f32 >= sampled_logits).sum(dim=-1)
    return (
        torch.cat(
            (sampled_token_ids.unsqueeze(-1), top_token_ids), dim=-1
        ).to(torch.int32),
        torch.cat((sampled_values, top_logprobs), dim=-1),
        ranks,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("shape", "top_k"),
    [
        ((1, 17), 1),
        ((7, 4096), 5),
        ((8, 151936), 20),
        ((4, 257), 32),
        ((1, 524289), 1),
    ],
)
def test_topk_sampled_logprobs_matches_pytorch(dtype, shape, top_k):
    torch.manual_seed(211)
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    sampled_token_ids = torch.randint(
        0, shape[1], (shape[0],), device="cuda", dtype=torch.int64
    )
    expected = reference(logits, sampled_token_ids, top_k)
    actual = topk_sampled_logprobs(logits, sampled_token_ids, top_k)
    torch.cuda.synchronize()

    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=3.0e-5, atol=3.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_sampled_logprobs_has_deterministic_tie_order_and_rank():
    logits = torch.tensor(
        [[4.0, 1.0, 4.0, 4.0, -2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    sampled_token_ids = torch.tensor([1], device="cuda", dtype=torch.int64)
    token_ids, logprobs, ranks = topk_sampled_logprobs(
        logits, sampled_token_ids, 3
    )
    torch.cuda.synchronize()

    assert token_ids.tolist() == [[1, 0, 2, 3]]
    expected_logprobs = logits.log_softmax(dim=-1)[:, [1, 0, 2, 3]]
    torch.testing.assert_close(logprobs, expected_logprobs)
    assert ranks.tolist() == [4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_sampled_logprobs_accepts_padded_vocabulary_rows():
    storage = torch.randn((8, 152064), device="cuda", dtype=torch.bfloat16)
    logits = storage[:, :151936]
    sampled_token_ids = torch.tensor(
        [0, 17, 4096, 70000, 100000, 151935, 11, 99],
        device="cuda",
        dtype=torch.int64,
    )
    assert logits.stride() == (152064, 1)

    actual = topk_sampled_logprobs(logits, sampled_token_ids, 20)
    expected = reference(logits, sampled_token_ids, 20)
    torch.cuda.synchronize()
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=3.0e-5, atol=3.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_sampled_logprobs_uses_current_external_stream():
    logits = torch.randn((4, 8192), device="cuda", dtype=torch.float16)
    sampled_token_ids = torch.tensor([0, 9, 1024, 8191], device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = topk_sampled_logprobs(logits, sampled_token_ids, 8)
    stream.synchronize()
    expected = reference(logits, sampled_token_ids, 8)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=3.0e-5, atol=3.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_sampled_logprobs_dispatcher_contract_and_fake_tensor():
    logits = torch.randn((3, 257), device="cuda", dtype=torch.bfloat16)
    sampled_token_ids = torch.tensor(
        [0, 128, 256], device="cuda", dtype=torch.int64
    )
    torch.library.opcheck(
        torch.ops.loom_kernels.topk_sampled_logprobs.default,
        (logits, sampled_token_ids, 8),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_sampled_logprobs_survives_torch_compile():
    @torch.compile(fullgraph=True)
    def compiled(logits: torch.Tensor, sampled_token_ids: torch.Tensor):
        return torch.ops.loom_kernels.topk_sampled_logprobs(
            logits, sampled_token_ids, 16
        )

    logits = torch.randn((5, 4096), device="cuda", dtype=torch.bfloat16)
    sampled_token_ids = torch.tensor([0, 1, 255, 2048, 4095], device="cuda")
    actual = compiled(logits, sampled_token_ids)
    expected = reference(logits, sampled_token_ids, 16)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=3.0e-5, atol=3.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_sampled_logprobs_cuda_graph_replay_and_telemetry():
    logits = torch.randn((8, 4096), device="cuda", dtype=torch.float16)
    sampled_token_ids = torch.arange(8, device="cuda", dtype=torch.int64)
    for _ in range(3):
        outputs = topk_sampled_logprobs(logits, sampled_token_ids, 8)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    reset_launch_count(Operator.TOPK_SAMPLED_LOGPROBS)
    with torch.cuda.graph(graph):
        outputs = topk_sampled_logprobs(logits, sampled_token_ids, 8)
    logits.copy_(torch.randn_like(logits))
    sampled_token_ids.copy_(
        torch.arange(8, device="cuda", dtype=torch.int64) + 8
    )
    graph.replay()
    torch.cuda.synchronize()

    expected = reference(logits, sampled_token_ids, 8)
    assert torch.equal(outputs[0], expected[0])
    torch.testing.assert_close(
        outputs[1], expected[1], rtol=3.0e-5, atol=3.0e-5
    )
    assert torch.equal(outputs[2], expected[2])
    assert launch_count(Operator.TOPK_SAMPLED_LOGPROBS) == 1


def test_topk_sampled_logprobs_rejects_invalid_inputs():
    logits = torch.empty((2, 4), device="cpu")
    sampled_token_ids = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="CUDA logits"):
        topk_sampled_logprobs(logits, sampled_token_ids, 2)
    if torch.cuda.is_available():
        logits = torch.empty((2, 4), device="cuda")
        sampled_token_ids = torch.zeros(2, device="cuda", dtype=torch.int64)
        with pytest.raises(ValueError, match=r"1 <= top_k <= 4"):
            topk_sampled_logprobs(logits, sampled_token_ids, 0)
        with pytest.raises(ValueError, match=r"1 <= top_k <= 4"):
            topk_sampled_logprobs(logits, sampled_token_ids, 5)
