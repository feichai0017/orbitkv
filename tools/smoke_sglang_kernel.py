#!/usr/bin/env python3
"""Run a small numerical smoke test against the source-built SGLang kernel."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the SGLang kernel smoke test requires a GPU")

    import sgl_kernel

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    input_tensor = torch.randn(
        args.batch, args.hidden_size, dtype=torch.bfloat16, device=device
    )
    weight = torch.randn(args.hidden_size, dtype=torch.bfloat16, device=device)
    reference = (
        input_tensor.float()
        * torch.rsqrt(input_tensor.float().square().mean(dim=-1, keepdim=True) + args.eps)
        * weight.float()
    ).to(torch.bfloat16)

    output = sgl_kernel.rmsnorm(
        input_tensor, weight, eps=args.eps, enable_pdl=False
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=2e-2)

    difference = (output.float() - reference.float()).abs()
    agreement = torch.isclose(output, reference, rtol=1e-2, atol=2e-2)
    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "sglang_kernel": getattr(sgl_kernel, "__version__", None),
                "operation": "rmsnorm",
                "dtype": "bfloat16",
                "shape": [args.batch, args.hidden_size],
                "max_abs_error": difference.max().item(),
                "output_agreement": agreement.float().mean().item(),
                "atol": 2e-2,
                "rtol": 1e-2,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
