#!/usr/bin/env python3
"""Compare Loom's fused RMSNorm+INT8 with vLLM's native-IR boundary."""

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
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="bf16")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--with-residual",
        action="store_true",
        help=(
            "Benchmark fused Add+RMSNorm+INT8. Timed input is zero so repeated "
            "calls leave the residual value unchanged."
        ),
    )
    parser.add_argument(
        "--provider-order",
        choices=("loom-first", "vllm-first"),
        default="loom-first",
    )
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    for name in ("rows", "hidden_size", "warmup", "iterations", "samples"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.epsilon <= 0.0:
        parser.error("epsilon must be positive")
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


def vllm_native_ir_operation(
    torch: Any,
    input_tensor: Any,
    weight: Any,
    output: Any,
    scales: Any,
    epsilon: float,
    residual: Any | None,
) -> Operation:
    """Return the exact tensor semantics used by vLLM's native layernorm IR."""

    def operation() -> None:
        value = input_tensor.float()
        if residual is not None:
            value = value + residual.float()
            residual.copy_(value.to(input_tensor.dtype))
        variance = value.pow(2).mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + epsilon)
        normalized = (
            normalized.to(weight.dtype) * weight
        ).to(input_tensor.dtype)
        torch.ops._C.dynamic_scaled_int8_quant(
            output, normalized, scales, None
        )

    return operation


def profile_operation(torch: Any, operation: Operation) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    operation()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        operation()
        torch.cuda.synchronize()
    device_events = [
        event
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA")
    ]
    names = [event.name for event in device_events]
    return {
        "cuda_device_events": len(device_events),
        "cuda_device_time_us": sum(
            float(event.device_time_total) for event in device_events
        ),
        "cuda_event_names": names,
    }


def benchmark_operation(
    torch: Any,
    operation: Operation,
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile = profile_operation(torch, operation)
    warm_gpu(torch, args.gpu_warmup_seconds)
    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()

    eager_samples_us: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            operation()
        end.record()
        end.synchronize()
        eager_samples_us.append(
            start.elapsed_time(end) * 1000.0 / args.iterations
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    graph_samples_us: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples_us.append(
            start.elapsed_time(end) * 1000.0 / args.iterations
        )

    return {
        "eager_latency": latency_summary(eager_samples_us),
        "cuda_graph_replay_latency": latency_summary(graph_samples_us),
        "single_call_profile": profile,
    }


def correctness_report(
    torch: Any,
    actual_output: Any,
    actual_scales: Any,
    actual_residual: Any | None,
    expected_output: Any,
    expected_scales: Any,
    expected_residual: Any | None,
) -> dict[str, Any]:
    integer_delta = (
        actual_output.to(torch.int16) - expected_output.to(torch.int16)
    ).abs()
    scale_delta = (actual_scales - expected_scales).abs()
    per_row_mismatches = torch.count_nonzero(integer_delta, dim=-1)
    return {
        "output_mismatch_count": int(
            torch.count_nonzero(integer_delta).item()
        ),
        "maximum_absolute_int8_delta": int(integer_delta.max().item()),
        "maximum_mismatches_per_row": int(per_row_mismatches.max().item()),
        "scale_byte_exact": bool(
            torch.equal(
                actual_scales.view(torch.uint8),
                expected_scales.view(torch.uint8),
            )
        ),
        "maximum_absolute_scale_delta": float(scale_delta.max().item()),
        "maximum_relative_scale_delta": float(
            (
                scale_delta
                / expected_scales.abs().clamp_min(1.0e-30)
            )
            .max()
            .item()
        ),
        "residual_byte_exact": (
            None
            if actual_residual is None
            else bool(torch.equal(actual_residual, expected_residual))
        ),
    }


def main() -> None:
    args = parse_args()

    import torch
    import vllm

    from loom_kernels.torch_ops import bridge_abi_version

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    shape = (args.rows, args.hidden_size)
    torch.manual_seed(43)
    correctness_input = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
    initial_residual = (
        torch.randn(shape, device="cuda", dtype=dtype)
        if args.with_residual
        else None
    )

    expected_output = torch.empty_like(correctness_input, dtype=torch.int8)
    expected_scales = torch.empty(
        (args.rows, 1), device="cuda", dtype=torch.float32
    )
    expected_residual = (
        initial_residual.clone() if initial_residual is not None else None
    )
    vllm_native_ir_operation(
        torch,
        correctness_input,
        weight,
        expected_output,
        expected_scales,
        args.epsilon,
        expected_residual,
    )()

    actual_output = torch.empty_like(correctness_input, dtype=torch.int8)
    actual_scales = torch.empty_like(expected_scales)
    actual_residual = (
        initial_residual.clone() if initial_residual is not None else None
    )
    torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8(
        actual_output,
        correctness_input,
        weight,
        actual_scales,
        args.epsilon,
        actual_residual,
    )
    torch.cuda.synchronize()
    correctness = correctness_report(
        torch,
        actual_output,
        actual_scales,
        actual_residual,
        expected_output,
        expected_scales,
        expected_residual,
    )

    timed_input = (
        torch.zeros_like(correctness_input)
        if args.with_residual
        else correctness_input
    )
    provider_names = (
        ("loom_cuda", "vllm_native_ir")
        if args.provider_order == "loom-first"
        else ("vllm_native_ir", "loom_cuda")
    )
    providers: dict[str, Any] = {}
    for provider in provider_names:
        output = torch.empty_like(timed_input, dtype=torch.int8)
        scales = torch.empty_like(expected_scales)
        residual = (
            initial_residual.clone() if initial_residual is not None else None
        )
        if provider == "loom_cuda":

            def operation(
                output: Any = output,
                scales: Any = scales,
                residual: Any | None = residual,
            ) -> None:
                torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8(
                    output,
                    timed_input,
                    weight,
                    scales,
                    args.epsilon,
                    residual,
                )

        else:
            operation = vllm_native_ir_operation(
                torch,
                timed_input,
                weight,
                output,
                scales,
                args.epsilon,
                residual,
            )
        providers[provider] = benchmark_operation(torch, operation, args)

    loom_eager = providers["loom_cuda"]["eager_latency"]["median_us"]
    vllm_eager = providers["vllm_native_ir"]["eager_latency"]["median_us"]
    loom_graph = providers["loom_cuda"]["cuda_graph_replay_latency"]["median_us"]
    vllm_graph = providers["vllm_native_ir"][
        "cuda_graph_replay_latency"
    ]["median_us"]
    report = {
        "schema_version": 1,
        "benchmark": "rms_norm_dynamic_per_token_int8",
        "candidate": "Loom fused RMSNorm plus symmetric dynamic per-token INT8",
        "baseline": (
            "uncompiled vLLM native RMSNorm IR tensor semantics followed by "
            "_C.dynamic_scaled_int8_quant"
        ),
        "dtype": args.dtype,
        "output_dtype": "int8",
        "scale_dtype": "f32",
        "scale_shape": [args.rows, 1],
        "with_residual": args.with_residual,
        "residual_replay_input": "zero" if args.with_residual else None,
        "rows": args.rows,
        "hidden_size": args.hidden_size,
        "epsilon": args.epsilon,
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "gpu_warmup_seconds_per_provider": args.gpu_warmup_seconds,
        "provider_order": args.provider_order,
        "correctness": correctness,
        "providers": providers,
        "diagnostic_loom_eager_ratio_vs_uncompiled_native_ir": (
            vllm_eager / loom_eager
        ),
        "diagnostic_loom_graph_ratio_vs_uncompiled_native_ir": (
            vllm_graph / loom_graph
        ),
        "claim_boundary": [
            "This is an uncompiled boundary diagnostic, not engine latency.",
            "The baseline intentionally preserves vLLM native-IR rounding.",
            "Real compiled-engine A/B evidence is authoritative for performance.",
            "GEMM is outside this benchmark and remains vendor-owned.",
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
