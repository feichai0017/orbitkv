from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    apply_token_penalties_,
    launch_count,
    reset_launch_count,
    token_penalties_workspace_capacity,
)


def reference(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    rows, vocab_size = logits.shape

    def counts(tokens: torch.Tensor) -> torch.Tensor:
        valid = tokens.clone()
        valid.masked_fill_((valid < 0) | (valid >= vocab_size), vocab_size)
        result = torch.zeros(
            (rows, vocab_size + 1),
            dtype=torch.int64,
            device=logits.device,
        )
        result.scatter_add_(1, valid, torch.ones_like(valid))
        return result[:, :vocab_size]

    prompt_counts = counts(prompt_token_ids)
    output_counts = counts(output_token_ids)
    repeated = (prompt_counts > 0) | (output_counts > 0)
    penalties = repetition_penalties.unsqueeze(1)
    penalized = torch.where(logits > 0, logits / penalties, logits * penalties)
    expected = torch.where(repeated, penalized, logits)
    expected -= frequency_penalties.unsqueeze(1) * output_counts
    expected -= presence_penalties.unsqueeze(1) * (output_counts > 0)
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_penalties_match_vllm_semantics_on_external_stream():
    torch.manual_seed(401)
    rows, vocab_size = 8, 4096
    logits = torch.randn((rows, vocab_size), device="cuda")
    prompt = torch.randint(0, vocab_size, (rows, 97), device="cuda")
    output = torch.randint(0, vocab_size, (rows, 33), device="cuda")
    prompt[0, :4] = torch.tensor([-1, vocab_size, 17, 17], device="cuda")
    output[0, :5] = torch.tensor(
        [-1, vocab_size, 17, 17, 17], device="cuda"
    )
    presence = torch.linspace(-0.4, 0.6, rows, device="cuda")
    frequency = torch.linspace(0.7, -0.2, rows, device="cuda")
    repetition = torch.linspace(0.8, 1.3, rows, device="cuda")
    expected = reference(
        logits.clone(),
        prompt,
        output,
        presence,
        frequency,
        repetition,
    )
    capacity = token_penalties_workspace_capacity(
        prompt.shape[1], output.shape[1]
    )
    workspace = torch.empty(
        (rows, capacity), dtype=torch.int64, device="cuda"
    )
    pointer = logits.data_ptr()

    reset_launch_count(Operator.TOKEN_PENALTIES)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        returned = apply_token_penalties_(
            logits,
            prompt,
            output,
            presence,
            frequency,
            repetition,
            workspace,
        )
    stream.synchronize()

    assert returned is logits
    assert logits.data_ptr() == pointer
    assert launch_count(Operator.TOKEN_PENALTIES) == 1
    torch.testing.assert_close(logits, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_penalties_accept_padded_rows_and_reuse_workspace():
    rows, vocab_size = 3, 257
    logits_storage = torch.randn((rows, vocab_size + 11), device="cuda")
    logits = logits_storage[:, :vocab_size]
    prompt_storage = torch.full(
        (rows, 12), vocab_size, dtype=torch.int64, device="cuda"
    )
    prompt = prompt_storage[:, :7]
    output_storage = torch.full(
        (rows, 8), vocab_size, dtype=torch.int64, device="cuda"
    )
    output = output_storage[:, :3]
    prompt[:, 0] = torch.tensor([1, 2, 3], device="cuda")
    output[:, 0] = torch.tensor([1, 2, 3], device="cuda")
    penalties = torch.ones(rows, device="cuda")
    presence = torch.full((rows,), 0.25, device="cuda")
    frequency = torch.full((rows,), 0.5, device="cuda")
    capacity = token_penalties_workspace_capacity(7, 3)
    workspace_storage = torch.empty(
        (rows, capacity + 16), dtype=torch.int64, device="cuda"
    )
    workspace = workspace_storage[:, :capacity]
    expected = reference(
        logits.clone(),
        prompt,
        output,
        presence,
        frequency,
        penalties,
    )

    apply_token_penalties_(
        logits,
        prompt,
        output,
        presence,
        frequency,
        penalties,
        workspace,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_penalties_survive_torch_compile_and_cuda_graph_replay():
    rows, vocab_size = 2, 1024
    prompt = torch.randint(0, vocab_size, (rows, 16), device="cuda")
    output = torch.randint(0, vocab_size, (rows, 8), device="cuda")
    presence = torch.full((rows,), 0.1, device="cuda")
    frequency = torch.full((rows,), 0.2, device="cuda")
    repetition = torch.full((rows,), 1.1, device="cuda")
    workspace = torch.empty(
        (rows, token_penalties_workspace_capacity(16, 8)),
        dtype=torch.int64,
        device="cuda",
    )

    def target(logits):
        torch.ops.loom_kernels.apply_token_penalties_(
            logits,
            prompt,
            output,
            presence,
            frequency,
            repetition,
            workspace,
        )
        return logits

    compiled = torch.compile(target, fullgraph=True)
    compiled_logits = torch.randn((rows, vocab_size), device="cuda")
    expected = reference(
        compiled_logits.clone(),
        prompt,
        output,
        presence,
        frequency,
        repetition,
    )
    actual = compiled(compiled_logits)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    graph_logits = torch.randn((rows, vocab_size), device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target(graph_logits)
    source = torch.randn_like(graph_logits)
    expected = reference(
        source.clone(),
        prompt,
        output,
        presence,
        frequency,
        repetition,
    )
    graph_logits.copy_(source)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        graph_logits, expected, rtol=1.0e-6, atol=1.0e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_penalties_reject_short_workspace_before_launch():
    rows, vocab_size = 2, 128
    logits = torch.randn((rows, vocab_size), device="cuda")
    prompt = torch.zeros((rows, 8), dtype=torch.int64, device="cuda")
    output = torch.zeros((rows, 4), dtype=torch.int64, device="cuda")
    penalties = torch.ones(rows, device="cuda")
    workspace = torch.empty((rows, 16), dtype=torch.int64, device="cuda")

    reset_launch_count(Operator.TOKEN_PENALTIES)
    with pytest.raises(ValueError, match="workspace"):
        apply_token_penalties_(
            logits,
            prompt,
            output,
            penalties,
            penalties,
            penalties,
            workspace,
        )
    assert launch_count(Operator.TOKEN_PENALTIES) == 0
