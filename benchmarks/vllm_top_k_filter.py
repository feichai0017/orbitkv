#!/usr/bin/env python3
"""Compare Loom's exact top-k filter with vLLM's native fallback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import bridge_abi_version, top_k_filter_
from loom_kernels.vllm import TOP_K_FILTER_MAX_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,2,4,7")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=0,
        help="zero uses vocab-size; larger values model padded vocabulary rows",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.vocab_size,
        args.top_k,
        args.warmup,
        args.iterations,
        args.repeats,
    ) <= 0:
        parser.error("shape, top-k, and timing counts must be positive")
    if args.top_k > args.vocab_size:
        parser.error("top-k must not exceed vocab-size")
    if args.row_stride == 0:
        args.row_stride = args.vocab_size
    if args.row_stride < args.vocab_size:
        parser.error("row-stride must be at least vocab-size")
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
    row_stride: int,
    top_k: int,
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
    top_ks = torch.full(
        (rows,), top_k, device="cuda", dtype=torch.int32
    )

    def baseline(values: torch.Tensor):
        return apply_top_k_top_p(values, top_ks, None)

    def loom(values: torch.Tensor):
        return top_k_filter_(values, top_ks)

    expected = baseline(source.clone())
    actual = loom(source.clone())
    torch.cuda.synchronize()
    expected_mask = torch.isneginf(expected)
    actual_mask = torch.isneginf(actual)
    if not torch.equal(actual_mask, expected_mask):
        mismatch = torch.nonzero(actual_mask != expected_mask)[0]
        row = int(mismatch[0].item())
        column = int(mismatch[1].item())
        expected_threshold = float(source[row][~expected_mask[row]].min().item())
        actual_threshold = float(source[row][~actual_mask[row]].min().item())
        raise AssertionError(
            "Loom top-k mask differs from vLLM: "
            f"row={row}, column={column}, value={float(source[row, column])}, "
            f"expected_finite={int((~expected_mask[row]).sum().item())}, "
            f"actual_finite={int((~actual_mask[row]).sum().item())}, "
            f"expected_threshold={expected_threshold}, "
            f"actual_threshold={actual_threshold}"
        )
    if not torch.equal(actual[~actual_mask], expected[~expected_mask]):
        raise AssertionError("Loom changed a retained logit")

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
        "vllm_path": "PyTorch full sort",
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
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm

    rows = [int(value) for value in args.rows.split(",") if value]
    if not rows or min(rows) <= 0:
        raise ValueError("rows must contain positive integers")
    if max(rows) > TOP_K_FILTER_MAX_ROWS:
        raise ValueError(
            "vLLM top-k filter benchmarking is limited to the registered "
            f"full-sort fallback boundary of {TOP_K_FILTER_MAX_ROWS} rows"
        )
    results = [
        benchmark_case(
            row_count,
            args.vocab_size,
            args.row_stride,
            args.top_k,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
        )
        for row_count in rows
    ]
    all_registered_rows_faster = all(
        result["loom_us"] < result["baseline_us"] for result in results
    )
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "bridge_abi_version": bridge_abi_version(),
        "baseline": (
            "vLLM apply_top_k_top_p PyTorch full-sort path below eight rows"
        ),
        "candidate": (
            "Loom top_k_filter_ partition radix sort plus exact parallel "
            "binary-count selection"
        ),
        "dtype": "f32 (vLLM sampling contract)",
        "vocab_size": args.vocab_size,
        "row_stride": args.row_stride,
        "top_k": args.top_k,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "vllm_fast_path_gate": {
            "maximum_rows": TOP_K_FILTER_MAX_ROWS,
            "top_k_contract": (
                "every value in [1, vocab_size] stays on device; no "
                "top_k>256 fallback or host synchronization"
            ),
            "eight_or_more_rows": (
                "fall back to vLLM Qrita Triton because it selects exactly k "
                "entries when the threshold is tied"
            ),
        },
        "timing_method": (
            "CUDA events over independent preallocated logits; input creation "
            "excluded; each operator's internal workspace allocation included; "
            "provider order alternated by repeat"
        ),
        "acceptance": {
            "passed": all_registered_rows_faster,
            "mask": "exact, including threshold ties",
            "retained_logits": "exact",
            "performance": "lower median latency for every registered row",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if not all_registered_rows_faster:
        failed_rows = [
            result["rows"]
            for result in results
            if result["loom_us"] >= result["baseline_us"]
        ]
        raise RuntimeError(
            f"Loom did not beat vLLM for registered rows {failed_rows}"
        )


if __name__ == "__main__":
    main()
