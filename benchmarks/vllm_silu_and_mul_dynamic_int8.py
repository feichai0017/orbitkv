#!/usr/bin/env python3
"""Compare Loom fused SwiGLU+INT8 with vLLM's two-kernel boundary."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import statistics
import time
from typing import Any


Operation = Callable[[], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--width", type=int, default=4864)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 128, 256, 512, 4096],
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--provider-order", choices=("forward", "reverse"), default="forward"
    )
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.width <= 0 or any(rows <= 0 for rows in args.rows):
        parser.error("width and every row count must be positive")
    if args.warmup <= 0 or args.iterations <= 0 or args.samples <= 0:
        parser.error("warmup, iterations, and samples must be positive")
    if args.gpu_warmup_seconds < 0.0:
        parser.error("gpu-warmup-seconds must be non-negative")
    return args


def latency_summary(samples_us: list[float]) -> dict[str, Any]:
    return {
        "minimum_us": min(samples_us),
        "median_us": statistics.median(samples_us),
        "maximum_us": max(samples_us),
        "samples_us": samples_us,
    }


def warm_gpu(torch: Any, seconds: float) -> None:
    if seconds == 0.0:
        return
    side = 4096
    left = torch.randn((side, side), device="cuda", dtype=torch.bfloat16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    deadline = time.perf_counter() + seconds
    while True:
        for _ in range(8):
            torch.mm(left, right, out=output)
        torch.cuda.synchronize()
        if time.perf_counter() >= deadline:
            break


def benchmark_operation(
    torch: Any,
    operation: Operation,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    eager_samples_us = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        eager_samples_us.append(start.elapsed_time(end) * 1000.0 / iterations)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    graph_samples_us = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples_us.append(start.elapsed_time(end) * 1000.0 / iterations)

    return {
        "eager_latency": latency_summary(eager_samples_us),
        "cuda_graph_replay_latency": latency_summary(graph_samples_us),
    }


def benchmark_shape(
    torch: Any,
    rows: int,
    width: int,
    dtype: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    input_tensor = torch.randn(
        (rows, width * 2), device="cuda", dtype=dtype
    )
    native_output = torch.empty(
        (rows, width), device="cuda", dtype=torch.int8
    )
    native_scales = torch.empty(
        (rows, 1), device="cuda", dtype=torch.float32
    )
    loom_output = torch.empty_like(native_output)
    loom_scales = torch.empty_like(native_scales)

    def native_graph(
        source: Any,
        result: Any,
        scales: Any,
    ) -> tuple[Any, Any]:
        activated = (
            torch.nn.functional.silu(source[..., :width])
            * source[..., width:]
        )
        torch.ops._C.dynamic_scaled_int8_quant(
            result, activated, scales, None
        )
        return result, scales

    def loom_fused() -> None:
        torch.ops.loom_kernels.silu_and_mul_dynamic_per_token_int8(
            loom_output, input_tensor, loom_scales
        )

    compiled_native = torch.compile(native_graph, fullgraph=True)

    def native_compiled_two_kernel() -> None:
        compiled_native(input_tensor, native_output, native_scales)

    native_compiled_two_kernel()
    loom_fused()
    torch.cuda.synchronize()
    integer_delta = (
        loom_output.to(torch.int16) - native_output.to(torch.int16)
    ).abs()
    scale_delta = (loom_scales - native_scales).abs()

    operations = {
        "vllm_compiled_two_kernel": native_compiled_two_kernel,
        "loom_fused": loom_fused,
    }
    forward_order = ("vllm_compiled_two_kernel", "loom_fused")
    provider_order = (
        forward_order
        if args.provider_order == "forward"
        else tuple(reversed(forward_order))
    )
    providers = {
        name: benchmark_operation(
            torch,
            operations[name],
            args.warmup,
            args.iterations,
            args.samples,
        )
        for name in provider_order
    }
    native = providers["vllm_compiled_two_kernel"]
    loom = providers["loom_fused"]
    native_eager = native["eager_latency"]["median_us"]
    loom_eager = loom["eager_latency"]["median_us"]
    native_graph = native["cuda_graph_replay_latency"]["median_us"]
    loom_graph = loom["cuda_graph_replay_latency"]["median_us"]
    return {
        "rows": rows,
        "width": width,
        "correctness": {
            "output_mismatch_count": int(torch.count_nonzero(integer_delta).item()),
            "maximum_absolute_int8_delta": int(integer_delta.max().item()),
            "scale_byte_exact": bool(
                torch.equal(
                    loom_scales.view(torch.uint8),
                    native_scales.view(torch.uint8),
                )
            ),
            "maximum_absolute_scale_delta": float(scale_delta.max().item()),
        },
        "providers": providers,
        "loom_eager_speedup": native_eager / loom_eager,
        "loom_cuda_graph_speedup": native_graph / loom_graph,
    }


def main() -> None:
    args = parse_args()

    import torch
    import vllm

    from loom_kernels.torch_ops import bridge_abi_version

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(83)
    warm_gpu(torch, args.gpu_warmup_seconds)
    shapes = [
        benchmark_shape(torch, rows, args.width, dtype, args)
        for rows in args.rows
    ]
    report = {
        "schema_version": 1,
        "benchmark": "silu_and_mul_dynamic_per_token_int8",
        "candidate": "Loom fused SwiGLU plus symmetric dynamic per-token INT8",
        "baseline": (
            "vLLM compiled native SwiGLU graph materializing FP16/BF16 output "
            "followed by _C.dynamic_scaled_int8_quant"
        ),
        "dtype": args.dtype,
        "output_dtype": "int8",
        "scale_dtype": "f32",
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "gpu_warmup_seconds": args.gpu_warmup_seconds,
        "provider_order": args.provider_order,
        "shapes": shapes,
        "claim_boundary": [
            "This is an operator-boundary diagnostic, not engine latency.",
            "The baseline uses its Inductor-lowered graph; Loom is the direct "
            "custom-op call emitted by the engine graph.",
            "CUDA Graph replay is the primary decode-oriented launch comparison.",
            "Eager timing includes unequal Python dispatcher boundaries and "
            "is diagnostic only.",
            "Large prefill rows are reported and must not be inferred from decode rows.",
            "GEMM is unchanged and remains vendor-owned.",
        ],
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi_version": bridge_abi_version(),
        },
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="")
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
