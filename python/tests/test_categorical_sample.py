from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    categorical_sample,
    launch_count,
    reset_launch_count,
)


_MASK32 = 0xFFFF_FFFF


def philox_word(seed: int, counter: int) -> int:
    value = [counter & _MASK32, (counter >> 32) & _MASK32, 0, 0]
    key = [seed & _MASK32, (seed >> 32) & _MASK32]
    for _ in range(10):
        product0 = 0xD251_1F53 * value[0]
        product1 = 0xCD9E_8D57 * value[2]
        value = [
            ((product1 >> 32) ^ value[1] ^ key[0]) & _MASK32,
            product1 & _MASK32,
            ((product0 >> 32) ^ value[3] ^ key[1]) & _MASK32,
            product0 & _MASK32,
        ]
        key[0] = (key[0] + 0x9E37_79B9) & _MASK32
        key[1] = (key[1] + 0xBB67_AE85) & _MASK32
    return value[0]


def reference(
    probabilities: torch.Tensor,
    state: torch.Tensor,
) -> list[int]:
    rows = probabilities.detach().cpu().tolist()
    host_state = state.detach().cpu().tolist()
    output = []
    for values, (seed, counter) in zip(rows, host_state, strict=True):
        uniform = (philox_word(seed, counter) + 0.5) / 2**32
        cumulative = 0.0
        last_positive = 0
        selected = None
        for token, probability in enumerate(values):
            if probability > 0.0:
                last_positive = token
            cumulative += probability
            if cumulative > uniform:
                selected = token
                break
        output.append(last_positive if selected is None else selected)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_categorical_sample_replays_exact_stream_and_advances_counters():
    probabilities = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.75, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    initial_state = torch.tensor(
        [[0, 0], [17, 41], [torch.iinfo(torch.int64).max, 57]],
        device="cuda",
        dtype=torch.int64,
    )
    expected = reference(probabilities, initial_state)
    first_state = initial_state.clone()
    replay_state = initial_state.clone()

    reset_launch_count(Operator.CATEGORICAL_SAMPLE)
    first = categorical_sample(probabilities, first_state)
    replay = categorical_sample(probabilities, replay_state)
    torch.cuda.synchronize()

    assert first.tolist() == expected
    assert torch.equal(first, replay)
    assert torch.equal(first_state, replay_state)
    assert torch.equal(first_state[:, 0], initial_state[:, 0])
    assert torch.equal(first_state[:, 1], initial_state[:, 1] + 1)
    assert launch_count(Operator.CATEGORICAL_SAMPLE) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_categorical_sample_matches_distribution_and_zero_mass_boundary():
    samples = 65_536
    probabilities = (
        torch.tensor([0.0, 0.125, 0.375, 0.5], device="cuda")
        .expand(samples, -1)
        .clone()
    )
    state = torch.empty((samples, 2), device="cuda", dtype=torch.int64)
    state[:, 0] = 31
    state[:, 1] = torch.arange(samples, device="cuda", dtype=torch.int64)

    tokens = categorical_sample(probabilities, state)
    counts = torch.bincount(tokens, minlength=4).cpu()
    observed = counts.to(torch.float64) / samples

    assert counts[0].item() == 0
    torch.testing.assert_close(
        observed,
        torch.tensor([0.0, 0.125, 0.375, 0.5], dtype=torch.float64),
        rtol=0.0,
        atol=0.008,
    )
    assert state[0].tolist() == [31, 1]
    assert state[-1].tolist() == [31, samples]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_categorical_sample_uses_current_stream_and_survives_compile():
    probabilities = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4]],
        device="cuda",
    )
    initial_state = torch.tensor([[3, 11], [5, 17]], device="cuda")
    expected = reference(probabilities, initial_state)
    state = initial_state.clone()
    stream = torch.cuda.Stream()

    @torch.compile(fullgraph=True)
    def compiled(values: torch.Tensor, mutable_state: torch.Tensor):
        return torch.ops.loom_kernels.categorical_sample(
            values, mutable_state
        )

    with torch.cuda.stream(stream):
        tokens = compiled(probabilities, state)
    stream.synchronize()

    assert tokens.tolist() == expected
    assert torch.equal(state[:, 1], initial_state[:, 1] + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_categorical_sample_schema_fake_tensor_and_cuda_graph_replay():
    probabilities = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.0, 0.25, 0.75, 0.0]],
        device="cuda",
    )
    state = torch.tensor([[7, 0], [11, 9]], device="cuda")
    schema = str(torch.ops.loom_kernels.categorical_sample.default._schema)
    assert "Tensor(a!) rng_state" in schema
    torch.library.opcheck(
        torch.ops.loom_kernels.categorical_sample.default,
        (probabilities, state.clone()),
        test_utils=("test_schema", "test_faketensor"),
    )

    for _ in range(3):
        categorical_sample(probabilities, state.clone())
    torch.cuda.synchronize()

    graph_state = state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_tokens = categorical_sample(probabilities, graph_state)
    graph_state.copy_(state)
    expected = reference(probabilities, state)
    graph.replay()
    torch.cuda.synchronize()

    assert graph_tokens.tolist() == expected
    assert torch.equal(graph_state[:, 1], state[:, 1] + 1)


def test_categorical_sample_rejects_invalid_tensor_contracts():
    with pytest.raises(ValueError, match="F32 CUDA probabilities"):
        categorical_sample(
            torch.tensor([[0.25, 0.75]]),
            torch.tensor([[1, 0]], dtype=torch.int64),
        )
    if torch.cuda.is_available():
        probabilities = torch.tensor([[0.25, 0.75]], device="cuda")
        with pytest.raises(ValueError, match=r"\[rows, 2\]"):
            categorical_sample(
                probabilities,
                torch.tensor([1, 0], device="cuda"),
            )
        with pytest.raises(ValueError, match="F32 CUDA probabilities"):
            categorical_sample(
                probabilities.to(torch.float16),
                torch.tensor([[1, 0]], device="cuda"),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_categorical_sample_rejects_aliasing_before_launch():
    storage = torch.empty(8, device="cuda", dtype=torch.float32)
    probabilities = storage.view(2, 4)
    rng_state = storage.view(torch.int64).view(2, 2)

    reset_launch_count(Operator.CATEGORICAL_SAMPLE)
    with pytest.raises(RuntimeError, match="must not overlap"):
        torch.ops.loom_kernels.categorical_sample(probabilities, rng_state)
    assert launch_count(Operator.CATEGORICAL_SAMPLE) == 0
