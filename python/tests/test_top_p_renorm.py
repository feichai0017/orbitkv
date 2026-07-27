from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    top_p_renorm_,
)


def reference(
    logits: torch.Tensor,
    top_ps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab_size = logits.shape
    token_ids = torch.arange(
        vocab_size - 1,
        -1,
        -1,
        device=logits.device,
        dtype=torch.int64,
    ).expand(rows, -1)
    descending_token_values = logits.float().gather(-1, token_ids)
    order = torch.argsort(
        descending_token_values,
        dim=-1,
        descending=True,
        stable=True,
    )
    sorted_token_ids = token_ids.gather(-1, order)
    sorted_values = logits.float().gather(-1, sorted_token_ids)
    sorted_masses = torch.exp(sorted_values - sorted_values[:, :1])
    cumulative = sorted_masses.cumsum(dim=-1)
    targets = top_ps[:, None] * cumulative[:, -1:]
    reaches_target = cumulative >= targets
    cutoffs = reaches_target.to(torch.int64).argmax(dim=-1)
    cutoffs = torch.where(
        top_ps == 1.0,
        torch.full_like(cutoffs, vocab_size - 1),
        cutoffs,
    )
    sorted_keep = (
        torch.arange(vocab_size, device=logits.device)[None, :]
        <= cutoffs[:, None]
    )
    keep = torch.zeros_like(sorted_keep).scatter_(
        -1, sorted_token_ids, sorted_keep
    )
    filtered = logits.clone().masked_fill_(~keep, -float("inf"))
    probabilities = torch.where(
        keep,
        torch.exp(logits.float() - sorted_values[:, :1]),
        0.0,
    )
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    return filtered, probabilities


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_p_renorm_matches_deterministic_reference(dtype):
    torch.manual_seed(419)
    logits = torch.randn((3, 4097), device="cuda", dtype=dtype)
    top_ps = torch.tensor([0.1, 0.9, 1.0], device="cuda")
    expected_logits, expected_probabilities = reference(logits, top_ps)

    probabilities = top_p_renorm_(logits, top_ps)
    torch.cuda.synchronize()

    assert probabilities.dtype == torch.float32
    assert probabilities.shape == logits.shape
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected_logits))
    assert torch.equal(
        logits[~torch.isneginf(logits)],
        expected_logits[~torch.isneginf(expected_logits)],
    )
    torch.testing.assert_close(
        probabilities,
        expected_probabilities,
        rtol=3.0e-5,
        atol=3.0e-7,
    )
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(3, device="cuda"),
        rtol=3.0e-5,
        atol=3.0e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_p_renorm_orders_ties_by_descending_token_id():
    logits = torch.tensor(
        [
            [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
            [5.0, 4.0, 4.0, 4.0, 1.0, -float("inf"), -1.0, -2.0],
        ],
        device="cuda",
    )
    top_ps = torch.tensor([0.5, 0.75], device="cuda")
    expected_logits, expected_probabilities = reference(logits, top_ps)

    probabilities = top_p_renorm_(logits, top_ps)
    torch.cuda.synchronize()

    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected_logits))
    assert torch.equal(
        torch.nonzero(torch.isfinite(logits[0]), as_tuple=False).flatten(),
        torch.tensor([4, 5, 6, 7], device="cuda"),
    )
    torch.testing.assert_close(
        probabilities,
        expected_probabilities,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_p_renorm_accepts_padded_rows_and_current_stream():
    torch.manual_seed(421)
    storage = torch.randn((7, 152064), device="cuda")
    logits = storage[:, :151936]
    top_ps = torch.linspace(0.5, 1.0, 7, device="cuda")
    expected_logits, expected_probabilities = reference(logits, top_ps)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        probabilities = top_p_renorm_(logits, top_ps)
    stream.synchronize()

    assert logits.stride() == (152064, 1)
    assert probabilities.is_contiguous()
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected_logits))
    torch.testing.assert_close(
        probabilities,
        expected_probabilities,
        rtol=4.0e-5,
        atol=4.0e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_p_renorm_schema_survives_opcheck_and_torch_compile():
    logits = torch.randn((3, 257), device="cuda")
    top_ps = torch.tensor([0.1, 0.9, 1.0], device="cuda")
    torch.library.opcheck(
        torch.ops.loom_kernels.top_p_renorm_.default,
        (logits.clone(), top_ps),
        test_utils=("test_schema", "test_faketensor"),
    )

    @torch.compile(fullgraph=True)
    def compiled(values: torch.Tensor, per_row_top_p: torch.Tensor):
        probabilities = torch.ops.loom_kernels.top_p_renorm_(
            values, per_row_top_p
        )
        return values, probabilities

    source = torch.randn((3, 257), device="cuda")
    expected_logits, expected_probabilities = reference(source, top_ps)
    actual_logits, actual_probabilities = compiled(source, top_ps)
    torch.cuda.synchronize()
    assert torch.equal(
        torch.isneginf(actual_logits), torch.isneginf(expected_logits)
    )
    torch.testing.assert_close(
        actual_probabilities,
        expected_probabilities,
        rtol=3.0e-5,
        atol=3.0e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_p_renorm_cuda_graph_replay_and_telemetry():
    logits = torch.randn((3, 4097), device="cuda")
    top_ps = torch.tensor([0.25, 0.75, 1.0], device="cuda")
    for _ in range(3):
        warmup = logits.clone()
        top_p_renorm_(warmup, top_ps)
    torch.cuda.synchronize()

    graph_logits = logits.clone()
    graph = torch.cuda.CUDAGraph()
    reset_launch_count(Operator.TOP_P_RENORM)
    with torch.cuda.graph(graph):
        graph_probabilities = top_p_renorm_(graph_logits, top_ps)
    source = torch.randn_like(logits)
    expected_logits, expected_probabilities = reference(source, top_ps)
    graph_logits.copy_(source)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(
        torch.isneginf(graph_logits), torch.isneginf(expected_logits)
    )
    torch.testing.assert_close(
        graph_probabilities,
        expected_probabilities,
        rtol=3.0e-5,
        atol=3.0e-7,
    )
    assert launch_count(Operator.TOP_P_RENORM) == 1


def test_top_p_renorm_rejects_invalid_tensor_contracts():
    with pytest.raises(ValueError, match="CUDA logits"):
        top_p_renorm_(
            torch.empty((2, 4), device="cpu"),
            torch.ones(2),
        )
    if torch.cuda.is_available():
        logits = torch.empty((2, 4), device="cuda")
        with pytest.raises(ValueError, match="F32 top-p"):
            top_p_renorm_(
                logits,
                torch.ones(2, device="cuda", dtype=torch.float16),
            )
        with pytest.raises(ValueError, match="per row"):
            top_p_renorm_(
                logits,
                torch.ones(3, device="cuda"),
            )
