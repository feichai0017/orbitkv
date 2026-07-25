#!/usr/bin/env python3
"""Compare sparse Loom token penalties with vLLM's vocabulary-wide path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import (
    apply_token_penalties_,
    bridge_abi_version,
    token_penalties_workspace_capacity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,8,32,128")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.vocab_size,
        args.prompt_tokens,
        args.output_tokens,
        args.warmup,
        args.iterations,
        args.repeats,
    ) <= 0:
        parser.error("vocabulary, history widths, and timing counts must be positive")
    return args


def elapsed_microseconds(
    operation: Callable[[torch.Tensor], object],
    source: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation(source.clone())
    workspaces = [source.clone() for _ in range(iterations)]
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for values in workspaces:
        operation(values)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end) * 1000.0 / iterations)
    del workspaces
    torch.cuda.empty_cache()
    return elapsed


def peak_temporary_bytes(
    operation: Callable[[torch.Tensor], object],
    source: torch.Tensor,
) -> int:
    values = source.clone()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation(values)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del values
    torch.cuda.empty_cache()
    return max(0, peak - before)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def benchmark_case(
    rows: int,
    vocab_size: int,
    prompt_tokens: int,
    output_tokens: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    from vllm.model_executor.layers.utils import apply_penalties

    torch.manual_seed(seed + rows)
    source = torch.randn((rows, vocab_size), device="cuda")
    prompt_token_ids = torch.randint(
        0,
        vocab_size,
        (rows, prompt_tokens),
        device="cuda",
        dtype=torch.int64,
    )
    output_token_ids = torch.randint(
        0,
        vocab_size,
        (rows, output_tokens),
        device="cuda",
        dtype=torch.int64,
    )
    output_token_ids[:, : min(8, output_tokens)] = output_token_ids[:, :1]
    presence_penalties = torch.linspace(-0.4, 0.6, rows, device="cuda")
    frequency_penalties = torch.linspace(0.7, -0.2, rows, device="cuda")
    repetition_penalties = torch.linspace(0.8, 1.3, rows, device="cuda")
    workspace_capacity = token_penalties_workspace_capacity(
        prompt_tokens, output_tokens
    )
    workspace = torch.empty(
        (rows, workspace_capacity),
        device="cuda",
        dtype=torch.int64,
    )

    def baseline(values: torch.Tensor):
        return apply_penalties(
            values,
            prompt_token_ids,
            output_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
        )

    def loom(values: torch.Tensor):
        return apply_token_penalties_(
            values,
            prompt_token_ids,
            output_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            workspace,
        )

    expected = baseline(source.clone())
    actual = loom(source.clone())
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    baseline_temporary_bytes = peak_temporary_bytes(baseline, source)
    loom_temporary_bytes = peak_temporary_bytes(loom, source)
    baseline_samples: list[float] = []
    loom_samples: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_samples.append(
                elapsed_microseconds(baseline, source, warmup, iterations)
            )
            loom_samples.append(
                elapsed_microseconds(loom, source, warmup, iterations)
            )
        else:
            loom_samples.append(
                elapsed_microseconds(loom, source, warmup, iterations)
            )
            baseline_samples.append(
                elapsed_microseconds(baseline, source, warmup, iterations)
            )

    baseline_median = statistics.median(baseline_samples)
    loom_median = statistics.median(loom_samples)
    return {
        "rows": rows,
        "baseline_us": baseline_median,
        "loom_us": loom_median,
        "speedup": baseline_median / loom_median,
        "latency_reduction_percent": (
            (baseline_median - loom_median) / baseline_median * 100.0
        ),
        "baseline_samples_us": baseline_samples,
        "loom_samples_us": loom_samples,
        "baseline_p90_us": percentile(baseline_samples, 0.9),
        "loom_p90_us": percentile(loom_samples, 0.9),
        "baseline_peak_temporary_bytes": baseline_temporary_bytes,
        "loom_peak_temporary_bytes": loom_temporary_bytes,
        "loom_workspace_capacity_per_row": workspace_capacity,
        "loom_caller_workspace_bytes": workspace.numel() * workspace.element_size(),
        "maximum_absolute_error": float((actual - expected).abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm

    rows = [int(value) for value in args.rows.split(",") if value]
    if not rows or min(rows) <= 0:
        raise ValueError("rows must contain positive integers")
    results = [
        benchmark_case(
            row_count,
            args.vocab_size,
            args.prompt_tokens,
            args.output_tokens,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
        )
        for row_count in rows
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "bridge_abi_version": bridge_abi_version(),
        "baseline": (
            "vLLM apply_penalties with two [rows, vocab+1] int64 bin-count "
            "tensors plus full-vocabulary masks and arithmetic"
        ),
        "candidate": "Loom apply_token_penalties_ with caller-owned sparse hash",
        "dtype": "f32 (vLLM sampling processor contract)",
        "vocab_size": args.vocab_size,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "timing_method": (
            "CUDA events over independent preallocated logits; input creation "
            "and reset copies excluded; provider order alternated by repeat; "
            "Loom caller workspace preallocated and reported separately"
        ),
        "acceptance": {
            "passed": True,
            "semantics": (
                "exact repetition over prompt/output union followed by exact "
                "output frequency and presence updates"
            ),
            "rtol": 1.0e-6,
            "atol": 1.0e-6,
            "maximum_absolute_error": max(
                result["maximum_absolute_error"] for result in results
            ),
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
