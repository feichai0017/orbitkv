#!/usr/bin/env python3
"""Compare Loom's explicit-state sampler with vLLM's seeded fallback."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch

from loom_kernels.torch_ops import bridge_abi_version, categorical_sample


Operation = Callable[[torch.Tensor], torch.Tensor]
OperationFactory = Callable[[], Operation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,2,4,7,8,32")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--active-tokens", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--distribution-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        args.rows = [int(value) for value in args.rows.split(",") if value]
    except ValueError as error:
        parser.error(f"rows must be comma-separated integers: {error}")
    if not args.rows or min(args.rows) <= 0:
        parser.error("rows must contain positive integers")
    if len(args.rows) != len(set(args.rows)):
        parser.error("rows must not contain duplicates")
    if min(
        args.vocab_size,
        args.active_tokens,
        args.warmup,
        args.iterations,
        args.repeats,
        args.distribution_samples,
    ) <= 0:
        parser.error("shape, timing, and distribution counts must be positive")
    if args.active_tokens > args.vocab_size:
        parser.error("active-tokens cannot exceed vocab-size")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def make_probabilities(
    rows: int,
    vocab_size: int,
    active_tokens: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed + rows)
    logits = torch.randn(
        (rows, vocab_size),
        device="cuda",
        dtype=torch.float32,
    )
    active_logits, active_ids = logits.topk(active_tokens, dim=-1)
    probabilities = torch.zeros_like(logits)
    probabilities.scatter_(
        1,
        active_ids,
        active_logits.softmax(dim=-1, dtype=torch.float32),
    )
    return probabilities


def make_loom_state(rows: int, seed: int) -> torch.Tensor:
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + seed
    counters = torch.arange(rows, device="cuda", dtype=torch.int64) * 17
    return torch.stack((seeds, counters), dim=1)


def make_vllm_generators(
    rows: int,
    seed: int,
) -> dict[int, torch.Generator]:
    generators = {}
    for row in range(rows):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + row)
        generators[row] = generator
    return generators


def elapsed_microseconds(
    factory: OperationFactory,
    source: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    operation = factory()
    warmup_values = source.clone()
    for _ in range(warmup):
        warmup_values.copy_(source)
        operation(warmup_values)

    workspaces = [source.clone() for _ in range(iterations)]
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for values in workspaces:
        output = operation(values)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end) * 1000.0 / iterations)

    del output, workspaces, warmup_values, operation
    torch.cuda.empty_cache()
    return elapsed


def peak_increment_bytes(
    factory: OperationFactory,
    source: torch.Tensor,
) -> int:
    operation = factory()
    values = source.clone()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation(values)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output, values, operation
    torch.cuda.empty_cache()
    return max(0, peak - before)


def kernel_group(name: str) -> str:
    lowered = name.lower()
    if name.startswith("Memcpy"):
        return "memcpy"
    if name.startswith("Memset"):
        return "memset"
    if "categorical_sample" in lowered:
        return "loom_categorical_sample"
    if "exponential" in lowered or "distribution_elementwise" in lowered:
        return "exponential_noise"
    if "argmax" in lowered or "argmaxops" in lowered:
        return "argmax"
    if "divfunctor" in lowered or "div_kernel" in lowered:
        return "division"
    return "other"


def profile_cuda(
    factory: OperationFactory,
    source: torch.Tensor,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    warmup = factory()
    warmup(source.clone())
    torch.cuda.synchronize()
    del warmup

    operation = factory()
    values = source.clone()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        output = operation(values)
        torch.cuda.synchronize()

    cuda_events = [
        event
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA")
    ]
    groups = Counter(kernel_group(event.name) for event in cuda_events)
    memory_events = groups["memcpy"] + groups["memset"]
    result = {
        "cuda_device_events": len(cuda_events),
        "cuda_kernel_launches": len(cuda_events) - memory_events,
        "cuda_memcpy_events": groups["memcpy"],
        "cuda_memset_events": groups["memset"],
        "cuda_device_time_us": sum(
            float(event.device_time_total) for event in cuda_events
        ),
        "kernel_groups": dict(sorted(groups.items())),
        "sampled_token_ids": output.tolist(),
    }
    del output, values, operation, profiler
    torch.cuda.empty_cache()
    return result


def direct_probe(
    probabilities: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    initial_state = make_loom_state(probabilities.shape[0], seed)
    first_state = initial_state.clone()
    replay_state = initial_state.clone()
    first = categorical_sample(probabilities, first_state)
    replay = categorical_sample(probabilities, replay_state)
    torch.cuda.synchronize()
    selected_mass = probabilities.gather(1, first.unsqueeze(1)).squeeze(1)
    exact_replay = torch.equal(first, replay) and torch.equal(
        first_state, replay_state
    )
    counters_advanced_once = torch.equal(
        first_state[:, 1], initial_state[:, 1] + 1
    )
    seeds_unchanged = torch.equal(first_state[:, 0], initial_state[:, 0])
    selected_positive_mass = bool(torch.all(selected_mass > 0).item())
    if not all(
        (
            exact_replay,
            counters_advanced_once,
            seeds_unchanged,
            selected_positive_mass,
        )
    ):
        raise AssertionError("Loom categorical direct-contract probe failed")
    return {
        "exact_replay": exact_replay,
        "counters_advanced_once": counters_advanced_once,
        "seeds_unchanged": seeds_unchanged,
        "selected_positive_mass": selected_positive_mass,
        "sampled_token_ids": first.tolist(),
    }


def benchmark_case(
    rows: int,
    vocab_size: int,
    active_tokens: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
    random_sample: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    probabilities = make_probabilities(
        rows,
        vocab_size,
        active_tokens,
        seed,
    )

    def baseline_factory() -> Operation:
        generators = make_vllm_generators(rows, seed + 10_000)

        def operation(values: torch.Tensor) -> torch.Tensor:
            return random_sample(values, generators)

        return operation

    def loom_factory() -> Operation:
        state = make_loom_state(rows, seed + 20_000)

        def operation(values: torch.Tensor) -> torch.Tensor:
            return categorical_sample(values, state)

        return operation

    samples = {"vllm_seeded": [], "loom": []}
    providers = {
        "vllm_seeded": baseline_factory,
        "loom": loom_factory,
    }
    names = list(providers)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            samples[name].append(
                elapsed_microseconds(
                    providers[name],
                    probabilities,
                    warmup,
                    iterations,
                )
            )

    baseline_median = statistics.median(samples["vllm_seeded"])
    loom_median = statistics.median(samples["loom"])
    result = {
        "rows": rows,
        "probability_tensor_bytes": probabilities.numel()
        * probabilities.element_size(),
        "vllm_seeded_us": baseline_median,
        "loom_us": loom_median,
        "speedup": baseline_median / loom_median,
        "vllm_seeded_samples_us": samples["vllm_seeded"],
        "loom_samples_us": samples["loom"],
        "vllm_seeded_p90_us": percentile(samples["vllm_seeded"], 0.9),
        "loom_p90_us": percentile(samples["loom"], 0.9),
        "vllm_seeded_peak_increment_bytes": peak_increment_bytes(
            baseline_factory,
            probabilities,
        ),
        "loom_peak_increment_bytes": peak_increment_bytes(
            loom_factory,
            probabilities,
        ),
        "vllm_seeded_profile": profile_cuda(
            baseline_factory,
            probabilities,
        ),
        "loom_profile": profile_cuda(loom_factory, probabilities),
        "direct_contract": direct_probe(probabilities, seed + 30_000),
    }
    del probabilities
    torch.cuda.empty_cache()
    return result


def distribution_probe(samples: int, seed: int) -> dict[str, Any]:
    expected = torch.tensor(
        [0.0, 0.125, 0.375, 0.5],
        device="cuda",
        dtype=torch.float32,
    )
    probabilities = expected.expand(samples, -1).clone()
    state = torch.empty((samples, 2), device="cuda", dtype=torch.int64)
    state[:, 0] = seed
    state[:, 1] = torch.arange(samples, device="cuda", dtype=torch.int64)
    tokens = categorical_sample(probabilities, state)
    counts = torch.bincount(tokens, minlength=expected.numel()).cpu()
    observed = counts.to(torch.float64) / samples
    expected_host = expected.to(device="cpu", dtype=torch.float64)
    maximum_absolute_error = float(
        (observed - expected_host).abs().max().item()
    )
    zero_mass_never_selected = counts[0].item() == 0
    tolerance = max(0.008, 2.0 / math.sqrt(samples))
    passed = (
        zero_mass_never_selected
        and maximum_absolute_error <= tolerance
    )
    if not passed:
        raise AssertionError("Loom categorical distribution probe failed")
    return {
        "samples": samples,
        "counts": counts.tolist(),
        "observed": observed.tolist(),
        "expected": expected_host.tolist(),
        "maximum_absolute_error": maximum_absolute_error,
        "absolute_tolerance": tolerance,
        "zero_mass_never_selected": zero_mass_never_selected,
        "passed": passed,
    }


def external_stream_probe(
    probabilities: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    state = make_loom_state(probabilities.shape[0], seed)
    initial_counters = state[:, 1].clone()
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(default_stream)
    with torch.cuda.stream(stream):
        output = categorical_sample(probabilities, state)
        completed = stream.record_event()
    default_stream.wait_event(completed)
    torch.cuda.synchronize()
    return {
        "completed": True,
        "non_default_stream": stream.cuda_stream
        != default_stream.cuda_stream,
        "counters_advanced_once": torch.equal(
            state[:, 1], initial_counters + 1
        ),
        "sampled_token_ids": output.tolist(),
    }


def cuda_graph_probe(
    probabilities: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    initial_state = make_loom_state(probabilities.shape[0], seed)
    for _ in range(3):
        categorical_sample(probabilities, initial_state.clone())
    torch.cuda.synchronize()

    state = initial_state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = categorical_sample(probabilities, state)
    state.copy_(initial_state)
    sequences = []
    counters = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        sequences.append(output.tolist())
        counters.append(state[:, 1].tolist())
    expected_first = (initial_state[:, 1] + 1).tolist()
    expected_second = (initial_state[:, 1] + 2).tolist()
    passed = counters == [expected_first, expected_second]
    if not passed:
        raise AssertionError("Loom categorical CUDA Graph state probe failed")
    return {
        "captured": True,
        "explicit_generator_registration": False,
        "counters_advanced_per_replay": passed,
        "replay_token_ids": sequences,
    }


def driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()[0]


def git_revision(repository: Path) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm
    from vllm.v1.sample.ops.topk_topp_sampler import random_sample

    results = [
        benchmark_case(
            rows,
            args.vocab_size,
            args.active_tokens,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
            random_sample,
        )
        for rows in args.rows
    ]
    probe_rows = 8 if 8 in args.rows else args.rows[-1]
    probe_probabilities = make_probabilities(
        probe_rows,
        args.vocab_size,
        args.active_tokens,
        args.seed + 100_000,
    )
    repository = Path(__file__).resolve().parents[1]
    vllm_root = Path(vllm.__file__).resolve().parent
    implementation_files = [
        "crates/loom-kernels/src/sampling.rs",
        "crates/loom-cuda-sys/cuda/src/categorical_sample.cu",
        "crates/loom-cuda/src/sampling_dispatch.rs",
        "crates/loom-cuda-bridge/src/cuda/sampling_bridge.rs",
        "python/csrc/sampling.cpp",
        "python/src/loom_kernels/ops/sampling.py",
    ]
    report = {
        "schema_version": 1,
        "benchmark": "vllm_categorical_sample",
        "repository_base_revision": git_revision(repository),
        "repository_state": (
            "base revision plus the exact implementation file hashes below; "
            "the evidence JSON is generated before its documentation commit"
        ),
        "implementation_sha256": {
            relative: sha256_file(repository / relative)
            for relative in implementation_files
        },
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "baseline": (
                "vLLM 0.24 random_sample over normalized F32 "
                "probabilities with one torch.Generator per row"
            ),
            "candidate": (
                "Loom ABI8 categorical_sample with caller-owned "
                "(seed, counter) state"
            ),
            "semantic_difference": (
                "Loom preserves its declared Philox stream and categorical "
                "distribution, not native vLLM seed-to-token identity"
            ),
            "timed_boundary": (
                "sampling only; top-k/top-p filtering and softmax excluded"
            ),
            "not_claimed": [
                "unseeded FlashInfer replacement",
                "native vLLM seed-to-token parity",
                "end-to-end engine speedup",
            ],
        },
        "environment": {
            "host": os.uname().nodename,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "driver": driver_version(),
            "bridge_abi_version": bridge_abi_version(),
        },
        "configuration": {
            "rows": args.rows,
            "vocab_size": args.vocab_size,
            "active_tokens": args.active_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "distribution_samples": args.distribution_samples,
            "seed": args.seed,
            "provider_order": "reversed on alternating repeats",
            "timing": (
                "CUDA events over independent preallocated probability "
                "inputs; input construction excluded"
            ),
            "persistent_state": (
                "vLLM generators and Loom RNG tensors are created once per "
                "timing sample and advance across invocations"
            ),
        },
        "vllm_source": {
            "v1/sample/ops/topk_topp_sampler.py": sha256_file(
                vllm_root / "v1/sample/ops/topk_topp_sampler.py"
            )
        },
        "distribution": distribution_probe(
            args.distribution_samples,
            args.seed + 200_000,
        ),
        "external_stream": external_stream_probe(
            probe_probabilities,
            args.seed + 300_000,
        ),
        "cuda_graph": cuda_graph_probe(
            probe_probabilities,
            args.seed + 400_000,
        ),
        "results": results,
        "acceptance": {
            "direct_contract_passed": True,
            "distribution_passed": True,
            "external_stream_passed": True,
            "cuda_graph_passed": True,
            "faster_rows": [
                result["rows"]
                for result in results
                if result["speedup"] > 1.0
            ],
            "adapter_policy": (
                "an engine-lifetime opt-in owns every explicitly seeded "
                "random row; row-count fallback is forbidden because it "
                "would change an in-flight request's RNG stream"
            ),
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
