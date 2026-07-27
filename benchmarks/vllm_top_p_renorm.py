#!/usr/bin/env python3
"""Compare fused Loom top-p renormalization with vLLM's native fallback."""

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

from loom_kernels.torch_ops import bridge_abi_version, top_p_renorm_
from loom_kernels.vllm import (
    TOP_P_RENORM_MAX_ROWS,
    TOP_P_RENORM_MIN_ROWS,
    TOP_P_RENORM_MIN_VOCAB_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="2,4,7")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=0,
        help="zero uses vocab-size; larger values model padded vocabulary rows",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=439)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.vocab_size,
        args.warmup,
        args.iterations,
        args.repeats,
    ) <= 0:
        parser.error("shape and timing counts must be positive")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("top-p must be in (0, 1]")
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
        operation(source.clone())
    workspaces = [source.clone() for _ in range(iterations)]
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
    values = source.clone()
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
    top_p: float,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    torch.manual_seed(seed + rows)
    storage = torch.randn(
        (rows, row_stride), device="cuda", dtype=torch.float32
    )
    source = storage[:, :vocab_size]
    top_ps = torch.full(
        (rows,), top_p, device="cuda", dtype=torch.float32
    )

    def baseline(values: torch.Tensor) -> torch.Tensor:
        apply_top_k_top_p(values, None, top_ps)
        return values.softmax(dim=-1, dtype=torch.float32)

    def loom(values: torch.Tensor) -> torch.Tensor:
        return top_p_renorm_(values, top_ps)

    expected_logits = source.clone()
    expected_probabilities = baseline(expected_logits)
    actual_logits = source.clone()
    actual_probabilities = loom(actual_logits)
    torch.cuda.synchronize()
    expected_mask = torch.isneginf(expected_logits)
    actual_mask = torch.isneginf(actual_logits)
    boundary_mismatches = (actual_mask != expected_mask).sum(dim=-1)
    finite_count_delta = (
        (~actual_mask).sum(dim=-1) - (~expected_mask).sum(dim=-1)
    )
    if int(boundary_mismatches.max().item()) > 1 or int(
        finite_count_delta.abs().max().item()
    ) > 1:
        mismatch = torch.nonzero(actual_mask != expected_mask)[0]
        raise AssertionError(
            "Loom top-p boundary differs from vLLM by more than one token: "
            f"row={int(mismatch[0].item())}, "
            f"column={int(mismatch[1].item())}"
        )
    common_retained = ~actual_mask & ~expected_mask
    if not torch.equal(
        actual_logits[common_retained], expected_logits[common_retained]
    ):
        raise AssertionError("Loom changed a retained logit")
    probability_l1 = (
        actual_probabilities - expected_probabilities
    ).abs().sum(dim=-1)
    if float(probability_l1.max().item()) > 1.0e-4:
        raise AssertionError(
            "Loom top-p probabilities exceed the F32 boundary tolerance: "
            f"maximum L1={float(probability_l1.max().item())}"
        )

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
        "vllm_path": "PyTorch full sort plus softmax",
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
        "boundary_mismatches_per_row": boundary_mismatches.tolist(),
        "finite_count_delta_per_row": finite_count_delta.tolist(),
        "maximum_probability_l1": float(probability_l1.max().item()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm

    rows = [int(value) for value in args.rows.split(",") if value]
    if not rows or min(rows) < TOP_P_RENORM_MIN_ROWS:
        raise ValueError(
            "rows must stay inside the registered top-p boundary: "
            f"{TOP_P_RENORM_MIN_ROWS}..{TOP_P_RENORM_MAX_ROWS}"
        )
    if max(rows) > TOP_P_RENORM_MAX_ROWS:
        raise ValueError(
            "rows must stay inside the registered top-p boundary: "
            f"{TOP_P_RENORM_MIN_ROWS}..{TOP_P_RENORM_MAX_ROWS}"
        )
    if args.vocab_size < TOP_P_RENORM_MIN_VOCAB_SIZE:
        raise ValueError(
            "vocab-size must be at least the registered top-p boundary of "
            f"{TOP_P_RENORM_MIN_VOCAB_SIZE}"
        )
    results = [
        benchmark_case(
            row_count,
            args.vocab_size,
            args.row_stride,
            args.top_p,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
        )
        for row_count in rows
    ]
    all_registered_shapes_faster = all(
        result["loom_us"] < result["baseline_us"] for result in results
    )
    report = {
        "schema_version": 1,
        "tested_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "bridge_abi_version": bridge_abi_version(),
        "baseline": (
            "vLLM apply_top_k_top_p PyTorch full-sort path plus F32 softmax"
        ),
        "candidate": (
            "Loom partition radix sort, device threshold selection, in-place "
            "filtering, and F32 retained-prefix renormalization"
        ),
        "dtype": "f32 (vLLM sampling contract)",
        "vocab_size": args.vocab_size,
        "row_stride": args.row_stride,
        "top_p": args.top_p,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "vllm_fast_path_gate": {
            "minimum_rows": TOP_P_RENORM_MIN_ROWS,
            "maximum_rows": TOP_P_RENORM_MAX_ROWS,
            "minimum_vocab_size": TOP_P_RENORM_MIN_VOCAB_SIZE,
            "contract": "top-p only, F32 logits, F32 per-row top-p values",
            "fallbacks": (
                "row one, smaller vocabulary, eight or more rows, non-F32 "
                "logits, and joint top-k/top-p stay native"
            ),
        },
        "timing_method": (
            "CUDA events over independent preallocated logits; input creation "
            "excluded; internal output/workspace allocation included; "
            "provider order alternated by repeat"
        ),
        "acceptance": {
            "passed": all_registered_shapes_faster,
            "mask": (
                "at most one F32 cutoff-boundary token per row versus vLLM"
            ),
            "retained_logits": "exact",
            "probabilities": "per-row L1 difference at most 1e-4",
            "performance": "lower median latency for every registered shape",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if not all_registered_shapes_faster:
        failures = [
            result["rows"]
            for result in results
            if result["loom_us"] >= result["baseline_us"]
        ]
        raise RuntimeError(
            f"Loom did not beat vLLM for registered row counts {failures}"
        )


if __name__ == "__main__":
    main()
