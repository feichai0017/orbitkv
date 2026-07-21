"""Measure PyTorch RMSNorm with the same CUDA-event protocol as Loom Kernels."""

from __future__ import annotations

import argparse
import json
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def main() -> None:
    args = parse_args()
    for name in ("rows", "hidden_size", "warmup", "iterations", "samples"):
        require_positive(name, getattr(args, name))
    if args.epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {args.epsilon}")

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    input_tensor = torch.randn(
        (args.rows, args.hidden_size), device=device, dtype=dtype
    )
    weight = torch.randn((args.hidden_size,), device=device, dtype=dtype)

    for _ in range(args.warmup):
        functional.rms_norm(
            input_tensor, (args.hidden_size,), weight=weight, eps=args.epsilon
        )
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            functional.rms_norm(
                input_tensor, (args.hidden_size,), weight=weight, eps=args.epsilon
            )
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / args.iterations)

    report = {
        "backend": "pytorch-native",
        "operator": "rms_norm",
        "dtype": args.dtype,
        "rows": args.rows,
        "hidden_size": args.hidden_size,
        "epsilon": args.epsilon,
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "latency": {
            "minimum_us": min(samples_us),
            "median_us": statistics.median(samples_us),
            "maximum_us": max(samples_us),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
