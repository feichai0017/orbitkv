from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    top_k_filter_,
)


def reference(logits: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    sorted_values = torch.sort(logits.float(), dim=-1, descending=True).values
    thresholds = sorted_values.gather(
        -1, (top_ks.to(torch.int64) - 1).unsqueeze(-1)
    )
    return logits.clone().masked_fill_(logits.float() < thresholds, -float("inf"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 17), (7, 4096), (8, 151936)])
def test_top_k_filter_matches_pytorch(dtype, shape):
    torch.manual_seed(317)
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    top_ks = torch.linspace(
        1, shape[1], shape[0], device="cuda", dtype=torch.int32
    )
    expected = reference(logits, top_ks)

    returned = top_k_filter_(logits, top_ks)
    torch.cuda.synchronize()

    assert returned is logits
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))
    assert torch.equal(
        logits[~torch.isneginf(logits)],
        expected[~torch.isneginf(expected)],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_filter_preserves_threshold_ties_and_signed_zero():
    logits = torch.tensor(
        [
            [5.0, 4.0, 4.0, 1.0, -1.0],
            [2.0, 1.0, 0.0, -0.0, -1.0],
            [3.0, 2.0, 1.0, 0.0, -1.0],
        ],
        device="cuda",
    )
    top_ks = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int32)
    expected = reference(logits, top_ks)

    top_k_filter_(logits, top_ks)
    torch.cuda.synchronize()

    assert torch.equal(logits, expected)
    assert torch.isfinite(logits[0]).sum().item() == 3
    assert torch.isfinite(logits[1]).sum().item() == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_filter_accepts_padded_rows_and_current_stream():
    storage = torch.randn((8, 152064), device="cuda", dtype=torch.bfloat16)
    logits = storage[:, :151936]
    top_ks = torch.tensor(
        [1, 5, 20, 64, 1024, 4096, 70000, 151936],
        device="cuda",
        dtype=torch.int32,
    )
    expected = reference(logits, top_ks)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        returned = top_k_filter_(logits, top_ks)
    stream.synchronize()

    assert returned is logits
    assert logits.stride() == (152064, 1)
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_filter_schema_survives_opcheck_and_torch_compile():
    logits = torch.randn((3, 257), device="cuda", dtype=torch.float32)
    top_ks = torch.tensor([1, 17, 257], device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        torch.ops.loom_kernels.top_k_filter_.default,
        (logits.clone(), top_ks),
        test_utils=("test_schema", "test_faketensor"),
    )

    @torch.compile(fullgraph=True)
    def compiled(values: torch.Tensor, per_row_top_k: torch.Tensor):
        torch.ops.loom_kernels.top_k_filter_(values, per_row_top_k)
        return values

    source = torch.randn((3, 257), device="cuda", dtype=torch.float32)
    expected = reference(source, top_ks)
    actual = compiled(source, top_ks)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_filter_cuda_graph_replay_and_telemetry():
    logits = torch.randn((8, 4096), device="cuda", dtype=torch.float16)
    top_ks = torch.arange(1, 9, device="cuda", dtype=torch.int32)
    for _ in range(3):
        warmup = logits.clone()
        top_k_filter_(warmup, top_ks)
    torch.cuda.synchronize()

    graph_logits = logits.clone()
    graph = torch.cuda.CUDAGraph()
    reset_launch_count(Operator.TOP_K_FILTER)
    with torch.cuda.graph(graph):
        top_k_filter_(graph_logits, top_ks)
    source = torch.randn_like(logits)
    graph_logits.copy_(source)
    top_ks.copy_(torch.arange(9, 17, device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(graph_logits, reference(source, top_ks))
    assert launch_count(Operator.TOP_K_FILTER) == 1


def test_top_k_filter_rejects_invalid_tensor_contracts():
    with pytest.raises(ValueError, match="CUDA logits"):
        top_k_filter_(
            torch.empty((2, 4), device="cpu"),
            torch.ones(2, dtype=torch.int32),
        )
    if torch.cuda.is_available():
        logits = torch.empty((2, 4), device="cuda")
        with pytest.raises(ValueError, match="int32 top-k"):
            top_k_filter_(
                logits,
                torch.ones(2, device="cuda", dtype=torch.int64),
            )
        with pytest.raises(ValueError, match="per row"):
            top_k_filter_(
                logits,
                torch.ones(3, device="cuda", dtype=torch.int32),
            )
