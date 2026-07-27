from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels import logits_preprocess_
from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
)


def reference(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    blocked_mask: torch.Tensor | None,
    bias_row_ids: torch.Tensor | None,
    bias_token_ids: torch.Tensor | None,
    bias_values: torch.Tensor | None,
    suppressed_row_ids: torch.Tensor | None,
    suppressed_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    output = logits.clone()
    if blocked_mask is not None:
        output.masked_fill_(blocked_mask, -float("inf"))
    if bias_row_ids is not None:
        assert bias_token_ids is not None
        assert bias_values is not None
        output[bias_row_ids.long(), bias_token_ids.long()] += bias_values
    if suppressed_row_ids is not None:
        assert suppressed_token_ids is not None
        output[
            suppressed_row_ids.long(), suppressed_token_ids.long()
        ] = -float("inf")
    divisors = torch.where(
        temperatures < 1.0e-5,
        torch.ones_like(temperatures),
        temperatures,
    )
    return output.div_(divisors.unsqueeze(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 17), (8, 4096), (7, 151936)])
def test_logits_preprocess_matches_composed_pytorch(shape):
    torch.manual_seed(307)
    rows, vocab_size = shape
    logits = torch.randn(shape, device="cuda", dtype=torch.float32)
    temperatures = torch.linspace(0.0, 1.4, rows, device="cuda")
    blocked_mask = torch.zeros(shape, device="cuda", dtype=torch.bool)
    blocked_mask[:, 3::97] = True
    bias_row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    bias_token_ids = (
        torch.arange(rows, device="cuda", dtype=torch.int32) * 101 + 5
    ) % vocab_size
    bias_values = torch.linspace(-0.5, 0.5, rows, device="cuda")
    suppressed_row_ids = torch.arange(
        rows, device="cuda", dtype=torch.int32
    )
    suppressed_token_ids = (
        torch.arange(rows, device="cuda", dtype=torch.int32) * 113 + 7
    ) % vocab_size
    expected = reference(
        logits,
        temperatures,
        blocked_mask,
        bias_row_ids,
        bias_token_ids,
        bias_values,
        suppressed_row_ids,
        suppressed_token_ids,
    )

    reset_launch_count(Operator.LOGITS_PREPROCESS)
    returned = logits_preprocess_(
        logits,
        temperatures,
        blocked_mask,
        bias_row_ids,
        bias_token_ids,
        bias_values,
        suppressed_row_ids,
        suppressed_token_ids,
    )
    torch.cuda.synchronize()

    assert returned is logits
    assert launch_count(Operator.LOGITS_PREPROCESS) == 1
    torch.testing.assert_close(logits, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_logits_preprocess_accepts_absent_optional_groups_and_padded_rows():
    storage = torch.randn((4, 1031), device="cuda")
    logits = storage[:, :1025]
    temperatures = torch.tensor([0.0, 1.0, 0.5, 2.0], device="cuda")
    expected = reference(
        logits, temperatures, None, None, None, None, None, None
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        logits_preprocess_(logits, temperatures)
    stream.synchronize()

    assert logits.stride() == (1031, 1)
    torch.testing.assert_close(logits, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_logits_preprocess_schema_survives_opcheck_compile_and_graph_replay():
    logits = torch.randn((3, 257), device="cuda")
    temperatures = torch.tensor([0.0, 0.7, 1.2], device="cuda")
    blocked_mask = torch.zeros_like(logits, dtype=torch.bool)
    blocked_mask[:, 11] = True
    torch.library.opcheck(
        torch.ops.loom_kernels.logits_preprocess_.default,
        (
            logits.clone(),
            temperatures,
            blocked_mask,
            None,
            None,
            None,
            None,
            None,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )

    @torch.compile(fullgraph=True)
    def compiled(
        values: torch.Tensor,
        row_temperatures: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.loom_kernels.logits_preprocess_(
            values, row_temperatures, mask
        )
        return values

    compile_input = torch.randn_like(logits)
    expected = reference(
        compile_input,
        temperatures,
        blocked_mask,
        None,
        None,
        None,
        None,
        None,
    )
    actual = compiled(compile_input, temperatures, blocked_mask)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    graph_input = torch.randn_like(logits)
    static_input = graph_input.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        logits_preprocess_(static_input, temperatures, blocked_mask)
    replay_input = torch.randn_like(logits)
    static_input.copy_(replay_input)
    expected_replay = reference(
        replay_input,
        temperatures,
        blocked_mask,
        None,
        None,
        None,
        None,
        None,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        static_input, expected_replay, rtol=1.0e-6, atol=1.0e-6
    )


def test_logits_preprocess_rejects_partial_optional_groups():
    logits = torch.empty((2, 4))
    temperatures = torch.ones(2)
    with pytest.raises(ValueError, match="optional non-empty"):
        logits_preprocess_(
            logits,
            temperatures,
            bias_row_ids=torch.zeros(1, dtype=torch.int32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_logits_preprocess_rejects_partial_optional_groups_before_launch():
    logits = torch.empty((2, 4), device="cuda")
    temperatures = torch.ones(2, device="cuda")
    bias_row_ids = torch.zeros(1, device="cuda", dtype=torch.int32)
    reset_launch_count(Operator.LOGITS_PREPROCESS)

    with pytest.raises(RuntimeError, match="must be supplied together"):
        torch.ops.loom_kernels.logits_preprocess_(
            logits,
            temperatures,
            None,
            bias_row_ids,
            None,
            None,
            None,
            None,
        )

    assert launch_count(Operator.LOGITS_PREPROCESS) == 0
