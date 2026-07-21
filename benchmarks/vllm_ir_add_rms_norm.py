"""Compare Loom and vLLM C kernels through the same vLLM IR eager dispatch."""

from __future__ import annotations

import argparse
import json
import statistics
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="bf16")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--gpu-warmup-seconds",
        type=float,
        default=0.5,
        help="Run BF16 GEMMs before each provider to stabilize GPU clocks.",
    )
    parser.add_argument(
        "--provider-order",
        choices=("loom-first", "vllm-first"),
        default="loom-first",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def max_errors(expected, actual) -> tuple[float, float]:
    difference = (expected.float() - actual.float()).abs()
    absolute = difference.max().item()
    relative = (difference / expected.float().abs().clamp_min(1.0e-8)).max().item()
    return absolute, relative


def reference(input_tensor, residual, weight, epsilon):
    import torch

    summed = input_tensor.float() + residual.float()
    inverse_rms = torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + epsilon)
    output = (summed * inverse_rms * weight.float()).to(input_tensor.dtype)
    return output, summed.to(input_tensor.dtype)


def warm_gpu(torch, seconds: float) -> None:
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


def benchmark_provider(operation, provider, weight, spec, args):
    import torch

    dtype, shape = spec
    torch.manual_seed(17)
    original_input = torch.randn(shape, device="cuda", dtype=dtype)
    original_residual = torch.randn(shape, device="cuda", dtype=dtype)
    expected_output, expected_residual = reference(
        original_input, original_residual, weight, args.epsilon
    )
    actual_input = original_input.clone()
    actual_residual = original_residual.clone()

    with operation.set_priority([provider, "native"]):
        selected = operation.dispatch(
            actual_input, actual_residual, weight, args.epsilon
        ).provider
        if selected != provider:
            raise RuntimeError(f"requested {provider}, IR selected {selected}")
        output, residual_output = operation.maybe_inplace(
            actual_input, actual_residual, weight, args.epsilon
        )
    torch.cuda.synchronize()

    output_abs, output_rel = max_errors(expected_output, output)
    residual_abs, residual_rel = max_errors(expected_residual, residual_output)

    warm_gpu(torch, args.gpu_warmup_seconds)

    # Zero input/residual are a stable fixed point for repeated in-place calls.
    timing_input = torch.zeros(shape, device="cuda", dtype=dtype)
    timing_residual = torch.zeros_like(timing_input)
    samples_us: list[float] = []
    with operation.set_priority([provider, "native"]):
        for _ in range(args.warmup):
            operation.maybe_inplace(
                timing_input, timing_residual, weight, args.epsilon
            )
        torch.cuda.synchronize()

        for _ in range(args.samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                operation.maybe_inplace(
                    timing_input, timing_residual, weight, args.epsilon
                )
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000.0 / args.iterations)

    graph_input = torch.zeros(shape, device="cuda", dtype=dtype)
    graph_residual = torch.zeros_like(graph_input)
    graph = torch.cuda.CUDAGraph()
    with operation.set_priority([provider, "native"]):
        with torch.cuda.graph(graph):
            operation.maybe_inplace(
                graph_input, graph_residual, weight, args.epsilon
            )
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
        "selected_provider": selected,
        "latency": {
            "minimum_us": min(samples_us),
            "median_us": statistics.median(samples_us),
            "maximum_us": max(samples_us),
            "samples_us": samples_us,
        },
        "cuda_graph_replay_latency": {
            "minimum_us": min(graph_samples_us),
            "median_us": statistics.median(graph_samples_us),
            "maximum_us": max(graph_samples_us),
            "samples_us": graph_samples_us,
        },
        "max_output_abs_error": output_abs,
        "max_output_rel_error": output_rel,
        "max_residual_abs_error": residual_abs,
        "max_residual_rel_error": residual_rel,
    }


def main() -> None:
    args = parse_args()
    for name in ("rows", "hidden_size", "warmup", "iterations", "samples"):
        require_positive(name, getattr(args, name))
    if args.epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {args.epsilon}")
    if args.gpu_warmup_seconds < 0.0:
        raise ValueError(
            "gpu_warmup_seconds must be non-negative, "
            f"got {args.gpu_warmup_seconds}"
        )

    import torch
    import vllm
    from vllm import ir
    from vllm.platforms import current_platform

    from loom_kernels.vllm import DEFAULT_PROVIDER, register_vllm_ir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    required_providers = (
        (DEFAULT_PROVIDER, "vllm_c")
        if args.provider_order == "loom-first"
        else ("vllm_c", DEFAULT_PROVIDER)
    )
    for provider in required_providers:
        if provider not in operation.impls or not operation.impls[provider].supported:
            raise RuntimeError(f"vLLM IR provider {provider!r} is unavailable")

    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    shape = (args.rows, args.hidden_size)
    torch.manual_seed(23)
    weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
    providers = {
        provider: benchmark_provider(
            operation, provider, weight, (dtype, shape), args
        )
        for provider in required_providers
    }
    loom_median = providers[DEFAULT_PROVIDER]["latency"]["median_us"]
    vllm_median = providers["vllm_c"]["latency"]["median_us"]
    loom_graph_median = providers[DEFAULT_PROVIDER]["cuda_graph_replay_latency"][
        "median_us"
    ]
    vllm_graph_median = providers["vllm_c"]["cuda_graph_replay_latency"][
        "median_us"
    ]

    report = {
        "benchmark": "vllm_ir_fused_add_rms_norm",
        "dispatch": "vllm IR eager maybe_inplace",
        "dtype": args.dtype,
        "rows": args.rows,
        "hidden_size": args.hidden_size,
        "epsilon": args.epsilon,
        "correctness_fixture": "seeded nonzero single launch",
        "timing_fixture": "stable zero in-place",
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "gpu_warmup_seconds_per_provider": args.gpu_warmup_seconds,
        "provider_order": args.provider_order,
        "providers": providers,
        "loom_speedup_vs_vllm_c": vllm_median / loom_median,
        "loom_cuda_graph_speedup_vs_vllm_c": (
            vllm_graph_median / loom_graph_median
        ),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
