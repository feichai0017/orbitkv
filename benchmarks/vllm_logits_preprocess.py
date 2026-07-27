#!/usr/bin/env python3
"""Compare Loom's fused logits preprocessing with vLLM's PyTorch sequence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Callable

import torch

from loom_kernels.torch_ops import bridge_abi_version, logits_preprocess_


def clone_preserving_strides(source: torch.Tensor) -> torch.Tensor:
    values = torch.empty_strided(
        source.shape,
        source.stride(),
        dtype=source.dtype,
        device=source.device,
    )
    values.copy_(source)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,2,4,8,16,32")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=0,
        help="zero uses vocab-size; larger values model padded vocabulary rows",
    )
    parser.add_argument("--biases-per-row", type=int, default=4)
    parser.add_argument("--suppressions-per-row", type=int, default=4)
    parser.add_argument(
        "--scenario",
        choices=(
            "full",
            "mask-bias",
            "mask-suppression",
            "bias-suppression",
            "mask-only",
            "bias-only",
            "suppression-only",
            "temperature-only",
        ),
        default="full",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=461)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.vocab_size,
        args.biases_per_row,
        args.suppressions_per_row,
        args.warmup,
        args.iterations,
        args.repeats,
    ) <= 0:
        parser.error("shape, sparse counts, and timing counts must be positive")
    if args.row_stride == 0:
        args.row_stride = args.vocab_size
    if args.row_stride < args.vocab_size:
        parser.error("row-stride must be at least vocab-size")
    return args


def elapsed_microseconds(
    operation: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation(clone_preserving_strides(source))
    workspaces = [
        clone_preserving_strides(source) for _ in range(iterations)
    ]
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = [operation(values) for values in workspaces]
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end) * 1000.0 / iterations)
    del outputs
    del workspaces
    torch.cuda.empty_cache()
    return elapsed


def peak_temporary_bytes(
    operation: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
) -> int:
    values = clone_preserving_strides(source)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation(values)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
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
    row_stride: int,
    biases_per_row: int,
    suppressions_per_row: int,
    scenario: str,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed + rows)
    storage = torch.randn(
        (rows, row_stride), device="cuda", dtype=torch.float32
    )
    source = storage[:, :vocab_size]
    temperatures = torch.linspace(0.0, 1.4, rows, device="cuda")
    blocked_mask = torch.zeros(
        (rows, vocab_size), device="cuda", dtype=torch.bool
    )
    blocked_mask[:, 17::257] = True

    bias_row_ids = torch.arange(
        rows, device="cuda", dtype=torch.int32
    ).repeat_interleave(biases_per_row)
    bias_offsets = torch.arange(
        biases_per_row, device="cuda", dtype=torch.int32
    ).repeat(rows)
    bias_token_ids = (
        bias_row_ids * 101 + bias_offsets * 37 + 5
    ) % vocab_size
    bias_values = torch.linspace(
        -0.75,
        0.75,
        rows * biases_per_row,
        device="cuda",
        dtype=torch.float32,
    )

    suppressed_row_ids = torch.arange(
        rows, device="cuda", dtype=torch.int32
    ).repeat_interleave(suppressions_per_row)
    suppression_offsets = torch.arange(
        suppressions_per_row, device="cuda", dtype=torch.int32
    ).repeat(rows)
    suppressed_token_ids = (
        suppressed_row_ids * 113 + suppression_offsets * 41 + 7
    ) % vocab_size
    use_mask = scenario in ("full", "mask-bias", "mask-suppression", "mask-only")
    use_bias = scenario in ("full", "mask-bias", "bias-suppression", "bias-only")
    use_suppression = scenario in (
        "full",
        "mask-suppression",
        "bias-suppression",
        "suppression-only",
    )
    mask_input = blocked_mask if use_mask else None
    bias_row_input = bias_row_ids if use_bias else None
    bias_token_input = bias_token_ids if use_bias else None
    bias_value_input = bias_values if use_bias else None
    suppressed_row_input = suppressed_row_ids if use_suppression else None
    suppressed_token_input = suppressed_token_ids if use_suppression else None

    def baseline(values: torch.Tensor) -> torch.Tensor:
        if mask_input is not None:
            values.masked_fill_(mask_input, -float("inf"))
        if bias_row_input is not None:
            assert bias_token_input is not None
            assert bias_value_input is not None
            values[bias_row_input, bias_token_input] += bias_value_input
        if suppressed_row_input is not None:
            assert suppressed_token_input is not None
            values[suppressed_row_input, suppressed_token_input] = -float("inf")
        divisors = torch.where(
            temperatures < 1.0e-5,
            torch.ones_like(temperatures),
            temperatures,
        )
        values.div_(divisors.unsqueeze(1))
        return values

    def loom(values: torch.Tensor) -> torch.Tensor:
        return logits_preprocess_(
            values,
            temperatures,
            mask_input,
            bias_row_input,
            bias_token_input,
            bias_value_input,
            suppressed_row_input,
            suppressed_token_input,
        )

    expected = baseline(clone_preserving_strides(source))
    actual = loom(clone_preserving_strides(source))
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
        "maximum_absolute_error": float(
            torch.nan_to_num(actual - expected).abs().max().item()
        ),
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
            args.row_stride,
            args.biases_per_row,
            args.suppressions_per_row,
            args.scenario,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
        )
        for row_count in rows
    ]
    all_shapes_faster = all(
        result["loom_us"] < result["baseline_us"] for result in results
    )
    repository = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "tested_revision": subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            cwd=repository,
        ).stdout.strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "bridge_abi_version": bridge_abi_version(),
        "baseline": (
            "vLLM-order PyTorch masked_fill_, sparse additive bias, sparse "
            "suppression, mixed-row temperature where, and div_"
        ),
        "candidate": (
            "one Loom logits_preprocess_ CUDA pass through the checked ABI"
        ),
        "dtype": "f32 (vLLM sampling contract)",
        "vocab_size": args.vocab_size,
        "row_stride": args.row_stride,
        "biases_per_row": args.biases_per_row,
        "suppressions_per_row": args.suppressions_per_row,
        "scenario": args.scenario,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "timing_method": (
            "CUDA events over independent preallocated logits; sparse metadata "
            "and dense mask construction excluded; provider order alternated "
            "by repeat"
        ),
        "acceptance": {
            "passed": all_shapes_faster,
            "semantics": (
                "mask then unique sparse bias then sparse suppression then "
                "temperature; mixed greedy rows use divisor one"
            ),
            "rtol": 1.0e-6,
            "atol": 1.0e-6,
            "performance": "lower median latency for every measured row count",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if not all_shapes_faster:
        failures = [
            result["rows"]
            for result in results
            if result["loom_us"] >= result["baseline_us"]
        ]
        raise RuntimeError(
            f"Loom did not beat the composed baseline for rows {failures}"
        )


if __name__ == "__main__":
    main()
