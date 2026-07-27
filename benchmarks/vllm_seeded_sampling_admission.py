#!/usr/bin/env python3
"""Profile vLLM's seeded CUDA sampling fallback before admitting ABI8-A."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch


Operation = Callable[[torch.Tensor], Any]
OperationFactory = Callable[[], Operation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,2,4,7,8,32")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sequence-draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=461)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        rows = [int(value) for value in args.rows.split(",") if value]
    except ValueError as error:
        parser.error(f"rows must be comma-separated integers: {error}")
    if not rows or min(rows) <= 0:
        parser.error("rows must contain positive integers")
    if len(rows) != len(set(rows)):
        parser.error("rows must not contain duplicates")
    if min(
        args.vocab_size,
        args.top_k,
        args.warmup,
        args.iterations,
        args.repeats,
        args.sequence_draws,
    ) <= 0:
        parser.error("shape, sampling, and timing counts must be positive")
    if args.top_k > args.vocab_size:
        parser.error("top-k cannot exceed vocab-size")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("top-p must be in (0, 1]")
    if len(args.repository_revision) != 40:
        parser.error("repository-revision must be a full 40-character Git SHA")
    args.rows = rows
    return args


def prepare_environment(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    current_path = os.environ.get("PATH", "").split(os.pathsep)
    required_path = [
        str(Path(sys.executable).absolute().parent),
        str(cuda_home / "bin"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required_path if entry not in current_path]
        + current_path
    )


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


def generator_indices(mode: str, rows: int) -> range | tuple[int, ...]:
    if mode == "unseeded":
        return ()
    if mode == "one_seeded":
        return (0,)
    if mode == "all_seeded":
        return range(rows)
    raise ValueError(f"unknown generator mode: {mode}")


def make_generators(
    mode: str,
    rows: int,
    seed: int,
) -> dict[int, torch.Generator]:
    generators: dict[int, torch.Generator] = {}
    for row in generator_indices(mode, rows):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + row)
        generators[row] = generator
    return generators


def token_tensor(result: Any) -> torch.Tensor:
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"operation returned {type(result)!r}, expected Tensor")
    return result


def elapsed_microseconds(
    operation: Operation,
    source: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    warmup_values = source.clone()
    warmup_output: Any = None
    for _ in range(warmup):
        warmup_values.copy_(source)
        warmup_output = operation(warmup_values)
    del warmup_output, warmup_values

    workspaces = [source.clone() for _ in range(iterations)]
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output: Any = None
    for values in workspaces:
        output = operation(values)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end) * 1000.0 / iterations)

    del output, workspaces
    torch.cuda.empty_cache()
    return elapsed


def peak_increment_bytes(
    operation: Operation,
    source: torch.Tensor,
) -> int:
    values = source.clone()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation(values)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output, values
    torch.cuda.empty_cache()
    return max(0, peak - before)


def kernel_group(name: str) -> str:
    lowered = name.lower()
    if name.startswith("Memcpy"):
        return "memcpy"
    if name.startswith("Memset"):
        return "memset"
    if "exponential" in lowered or "distribution_elementwise" in lowered:
        return "exponential_noise"
    if "flashinfer" in lowered or "sampling_from" in lowered:
        return "flashinfer_sampling"
    if "softmax" in lowered:
        return "softmax"
    if "argmax" in lowered or "argmaxops" in lowered:
        return "argmax"
    if "divfunctor" in lowered or "div_kernel" in lowered:
        return "division"
    if "radix" in lowered or "sort" in lowered:
        return "sort"
    if "cumsum" in lowered or "scan" in lowered:
        return "scan"
    if "scatter" in lowered:
        return "scatter"
    if "masked_fill" in lowered or "mask" in lowered:
        return "mask"
    if "reduce" in lowered or "reduction" in lowered:
        return "reduction"
    return "other"


def profile_cuda_events(
    operation: Operation,
    source: torch.Tensor,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    warmup_values = source.clone()
    operation(warmup_values)
    del warmup_values
    torch.cuda.synchronize()

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
    summary = {
        "cuda_device_events": len(cuda_events),
        "cuda_kernel_launches": len(cuda_events) - memory_events,
        "cuda_memcpy_events": groups["memcpy"],
        "cuda_memset_events": groups["memset"],
        "cuda_device_time_us": sum(
            float(event.device_time_total) for event in cuda_events
        ),
        "kernel_groups": dict(sorted(groups.items())),
        "device_resource_ids": sorted(
            {int(event.device_resource_id) for event in cuda_events}
        ),
        "sampled_token_ids": token_tensor(output).tolist(),
    }
    del output, values, profiler
    torch.cuda.empty_cache()
    return summary


def benchmark_variants(
    source: torch.Tensor,
    factories: dict[str, OperationFactory],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    samples = {name: [] for name in factories}
    names = list(factories)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            samples[name].append(
                elapsed_microseconds(
                    factories[name](),
                    source,
                    warmup,
                    iterations,
                )
            )

    result: dict[str, Any] = {}
    for name, factory in factories.items():
        variant_samples = samples[name]
        result[name] = {
            "median_us": statistics.median(variant_samples),
            "p90_us": percentile(variant_samples, 0.9),
            "samples_us": variant_samples,
            "peak_increment_bytes": peak_increment_bytes(factory(), source),
            "profile": profile_cuda_events(factory(), source),
        }
    return result


def deterministic_sequence_probe(
    source: torch.Tensor,
    rows: int,
    seed: int,
    draws: int,
    random_sample: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    def sequence() -> list[list[int]]:
        generators = make_generators("all_seeded", rows, seed)
        return [
            random_sample(source.clone(), generators).tolist()
            for _ in range(draws)
        ]

    first = sequence()
    second = sequence()
    return {
        "draws": draws,
        "exact_replay": first == second,
        "first_sequence": first,
        "second_sequence": second,
    }


def external_stream_probe(
    source: torch.Tensor,
    rows: int,
    seed: int,
    random_sample: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    values = source.clone()
    generators = make_generators("all_seeded", rows, seed)
    default_stream = torch.cuda.current_stream()
    external_stream = torch.cuda.Stream()
    external_stream.wait_stream(default_stream)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        with torch.cuda.stream(external_stream):
            output = random_sample(values, generators)
            completed = external_stream.record_event()
        default_stream.wait_event(completed)
        torch.cuda.synchronize()

    cuda_events = [
        event
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA")
    ]
    result = {
        "completed": True,
        "non_default_stream": (
            external_stream.cuda_stream != default_stream.cuda_stream
        ),
        "default_stream_handle": int(default_stream.cuda_stream),
        "external_stream_handle": int(external_stream.cuda_stream),
        "profiler_device_resource_ids": sorted(
            {int(event.device_resource_id) for event in cuda_events}
        ),
        "sampled_token_ids": output.tolist(),
    }
    del output, values, profiler
    torch.cuda.empty_cache()
    return result


def cuda_graph_attempt(
    source: torch.Tensor,
    rows: int,
    seed: int,
    random_sample: Callable[..., torch.Tensor],
    register_generators: bool,
) -> dict[str, Any]:
    generators = make_generators("all_seeded", rows, seed)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            random_sample(source.clone(), generators)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    static_values = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph()
    registration_supported = hasattr(graph, "register_generator_state")
    if register_generators:
        if not registration_supported:
            return {
                "captured": False,
                "registration_supported": False,
                "error_type": "MissingGeneratorRegistrationAPI",
                "error": "torch.cuda.CUDAGraph.register_generator_state is absent",
            }
        for generator in generators.values():
            graph.register_generator_state(generator)

    try:
        with torch.cuda.graph(graph, stream=capture_stream):
            static_values.copy_(source)
            output = random_sample(static_values, generators)
        sequences: list[list[int]] = []
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            sequences.append(output.tolist())
        result = {
            "captured": True,
            "registration_supported": registration_supported,
            "registered_generators": register_generators,
            "replay_token_ids": sequences,
        }
        del output
    except Exception as error:
        result = {
            "captured": False,
            "registration_supported": registration_supported,
            "registered_generators": register_generators,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    del graph, static_values
    torch.cuda.empty_cache()
    return result


def sampling_factories(
    rows: int,
    seed: int,
    random_sample: Callable[..., torch.Tensor],
) -> dict[str, OperationFactory]:
    modes = ["unseeded", "all_seeded"]
    if rows > 1:
        modes.insert(1, "one_seeded")

    factories: dict[str, OperationFactory] = {}
    for mode in modes:

        def factory(mode: str = mode) -> Operation:
            generators = make_generators(mode, rows, seed)

            def operation(values: torch.Tensor) -> torch.Tensor:
                return random_sample(values, generators)

            return operation

        factories[mode] = factory
    return factories


def full_forward_factories(
    rows: int,
    seed: int,
    sampler: Any,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> dict[str, OperationFactory]:
    modes = ["unseeded", "all_seeded"]
    if rows > 1:
        modes.insert(1, "one_seeded")

    factories: dict[str, OperationFactory] = {}
    for mode in modes:

        def factory(mode: str = mode) -> Operation:
            generators = make_generators(mode, rows, seed)

            def operation(values: torch.Tensor) -> tuple[torch.Tensor, Any]:
                return sampler.forward_cuda(
                    values,
                    generators,
                    top_k,
                    top_p,
                )

            return operation

        factories[mode] = factory
    return factories


def benchmark_case(
    rows: int,
    vocab_size: int,
    top_k_value: int,
    top_p_value: float,
    warmup: int,
    iterations: int,
    repeats: int,
    sequence_draws: int,
    seed: int,
    sampler: Any,
    random_sample: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    torch.manual_seed(seed + rows)
    logits = torch.randn(
        (rows, vocab_size),
        device="cuda",
        dtype=torch.float32,
    )
    probabilities = logits.softmax(dim=-1, dtype=torch.float32)
    top_k = torch.full(
        (rows,),
        top_k_value,
        device="cuda",
        dtype=torch.int32,
    )
    top_p = torch.full(
        (rows,),
        top_p_value,
        device="cuda",
        dtype=torch.float32,
    )

    direct = benchmark_variants(
        probabilities,
        sampling_factories(rows, seed, random_sample),
        warmup,
        iterations,
        repeats,
    )
    full_forward = benchmark_variants(
        logits,
        full_forward_factories(rows, seed, sampler, top_k, top_p),
        warmup,
        iterations,
        repeats,
    )

    unseeded_direct = float(direct["unseeded"]["median_us"])
    all_seeded_direct = float(direct["all_seeded"]["median_us"])
    unseeded_forward = float(full_forward["unseeded"]["median_us"])
    all_seeded_forward = float(full_forward["all_seeded"]["median_us"])
    result = {
        "rows": rows,
        "vocab_size": vocab_size,
        "probability_tensor_bytes": rows * vocab_size * 4,
        "sampling_only": direct,
        "full_top_k_top_p_forward": full_forward,
        "diagnostic_ratios": {
            "sampling_all_seeded_over_unseeded": (
                all_seeded_direct / unseeded_direct
            ),
            "sampling_all_seeded_minus_unseeded_us": (
                all_seeded_direct - unseeded_direct
            ),
            "full_seeded_fallback_over_unseeded_flashinfer": (
                all_seeded_forward / unseeded_forward
            ),
            "full_seeded_fallback_minus_unseeded_flashinfer_us": (
                all_seeded_forward - unseeded_forward
            ),
        },
        "deterministic_sequence": deterministic_sequence_probe(
            probabilities,
            rows,
            seed + 100_000,
            sequence_draws,
            random_sample,
        ),
        "external_stream": external_stream_probe(
            probabilities,
            rows,
            seed + 200_000,
            random_sample,
        ),
        "cuda_graph": {
            "without_explicit_generator_registration": cuda_graph_attempt(
                probabilities,
                rows,
                seed + 300_000,
                random_sample,
                register_generators=False,
            ),
            "with_explicit_generator_registration": cuda_graph_attempt(
                probabilities,
                rows,
                seed + 400_000,
                random_sample,
                register_generators=True,
            ),
        },
    }
    del logits, probabilities, top_k, top_p
    torch.cuda.empty_cache()
    return result


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


def source_provenance(vllm_root: Path) -> dict[str, str]:
    relatives = [
        "v1/sample/ops/topk_topp_sampler.py",
        "v1/sample/ops/topk_topp_triton.py",
        "v1/sample/sampler.py",
        "v1/worker/gpu_input_batch.py",
        "v1/worker/gpu_model_runner.py",
    ]
    return {
        relative: sha256_file(vllm_root / relative) for relative in relatives
    }


def preliminary_observations(cases: list[dict[str, Any]]) -> dict[str, Any]:
    largest = max(cases, key=lambda case: int(case["rows"]))
    direct = largest["sampling_only"]
    unseeded_launches = int(
        direct["unseeded"]["profile"]["cuda_kernel_launches"]
    )
    seeded_launches = int(
        direct["all_seeded"]["profile"]["cuda_kernel_launches"]
    )
    probability_bytes = int(largest["probability_tensor_bytes"])
    seeded_peak = int(direct["all_seeded"]["peak_increment_bytes"])
    return {
        "largest_rows": largest["rows"],
        "all_seeded_kernel_launches": seeded_launches,
        "unseeded_kernel_launches": unseeded_launches,
        "additional_seeded_kernel_launches": (
            seeded_launches - unseeded_launches
        ),
        "all_seeded_peak_increment_bytes": seeded_peak,
        "probability_tensor_bytes": probability_bytes,
        "peak_increment_over_probability_tensor": (
            seeded_peak / probability_bytes
        ),
        "automatic_admission_decision": None,
        "reason": (
            "The raw profiler reports source-pinned cost only. Admission "
            "requires an explicit interpretation that separates the "
            "sampling-only boundary from the larger native-filter fallback."
        ),
    }


def main() -> None:
    args = parse_args()
    prepare_environment(args.cache_root.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm
    from vllm.v1.sample.ops.topk_topp_sampler import (
        TopKTopPSampler,
        random_sample,
    )

    sampler = TopKTopPSampler()
    if sampler.forward != sampler.forward_cuda:
        raise RuntimeError(
            "vLLM did not select its FlashInfer CUDA sampler for the "
            "unseeded control"
        )

    cases = [
        benchmark_case(
            rows,
            args.vocab_size,
            args.top_k,
            args.top_p,
            args.warmup,
            args.iterations,
            args.repeats,
            args.sequence_draws,
            args.seed,
            sampler,
            random_sample,
        )
        for rows in args.rows
    ]
    vllm_root = Path(vllm.__file__).resolve().parent
    report = {
        "schema_version": 1,
        "benchmark": "vllm_seeded_sampling_admission",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_base_revision": args.repository_revision,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "scope": {
            "candidate": "ABI8-A explicit-state categorical sampling",
            "sampling_only_boundary": (
                "vLLM random_sample over normalized contiguous F32 "
                "probabilities"
            ),
            "full_fallback_boundary": (
                "TopKTopPSampler.forward_cuda with top-k and top-p"
            ),
            "unseeded_full_control": "FlashInfer CUDA sampler",
            "seeded_full_path": "PyTorch-native fallback",
            "not_claimed": [
                "performance of an unimplemented Loom operator",
                "native seed-to-token compatibility for ABI8-A",
                "an end-to-end model or serving speedup",
            ],
        },
        "environment": {
            "host": os.uname().nodename,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": importlib.metadata.version("vllm"),
            "gpu": torch.cuda.get_device_name(0),
            "driver": driver_version(),
            "flashinfer_sampler_enabled": True,
            "torchinductor_cache_dir": os.environ["TORCHINDUCTOR_CACHE_DIR"],
            "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
        },
        "configuration": {
            "rows": args.rows,
            "vocab_size": args.vocab_size,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "sequence_draws": args.sequence_draws,
            "seed": args.seed,
            "provider_order": (
                "variants reverse on alternating repeats within each case"
            ),
        },
        "vllm_source_provenance": source_provenance(vllm_root),
        "cases": cases,
        "preliminary_observations": preliminary_observations(cases),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
