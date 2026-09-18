"""Qualify source-pinned FlashInfer BF16 RMSNorm for Aletheia."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .probe import probe
from .qualification import canonical_sha256, summarize_samples

ROOT = Path(__file__).resolve().parents[4]
FLASHINFER_SOURCE = ROOT / "third-party" / "flashinfer"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def driver_version() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0].strip()


def semantics(batch: int, hidden_size: int, eps: float, dtype: str) -> dict[str, Any]:
    return {
        "operation": "rmsnorm",
        "equation": "out = x * rsqrt(mean(x^2) + eps) * weight",
        "shape": [batch, hidden_size],
        "weight_shape": [hidden_size],
        "dtype": dtype,
        "accumulation": "float32",
        "eps": eps,
    }


def exact_domain(batch: int, context_tokens: int) -> dict[str, Any]:
    return {
        "phases": ["decode"],
        "batch": {"min": batch, "max": batch},
        "rows_per_sequence": {"min": 1, "max": 1},
        "context_tokens": {"min": context_tokens, "max": context_tokens},
    }


def locate_norm_artifact() -> Path:
    from flashinfer.jit import env as jit_env

    expected = jit_env.FLASHINFER_WORKSPACE_DIR / "cached_ops" / "norm" / "norm.so"
    if expected.is_file():
        return expected
    matches = sorted(jit_env.FLASHINFER_WORKSPACE_DIR.glob("**/norm.so"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one FlashInfer norm.so artifact, found {matches}")
    return matches[0]


def qualify(
    output: Path,
    *,
    batch: int,
    hidden_size: int,
    context_tokens: int,
    cases: int,
    samples: int,
    warmup: int,
    eps: float,
    atol: float,
    rtol: float,
) -> Path:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if min(batch, hidden_size, cases, samples) <= 0 or samples < 12:
        raise ValueError("batch, hidden size and cases must be positive; samples must be at least 12")

    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeError(f"the first qualified target is SM90, got sm{capability[0]}{capability[1]}")
    provenance = probe("flashinfer")
    if not provenance["from_pinned_source"]:
        raise RuntimeError("FlashInfer was not imported from the pinned source checkout")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    # Trigger source JIT before resetting memory statistics or timing.
    warm_x = torch.ones((batch, hidden_size), dtype=dtype, device=device)
    warm_w = torch.ones((hidden_size,), dtype=dtype, device=device)
    warm_out = torch.empty_like(warm_x)
    flashinfer.norm.rmsnorm(warm_x, warm_w, out=warm_out, eps=eps, enable_pdl=False)
    torch.cuda.synchronize(device)
    module_path = locate_norm_artifact()

    max_abs_error = 0.0
    agreed = 0
    elements = 0
    torch.cuda.reset_peak_memory_stats(device)
    for seed in range(cases):
        generator = torch.Generator(device=device).manual_seed(seed)
        input_tensor = torch.randn((batch, hidden_size), generator=generator, device=device, dtype=dtype)
        weight = torch.randn((hidden_size,), generator=generator, device=device, dtype=dtype)
        output_tensor = torch.empty_like(input_tensor)
        flashinfer.norm.rmsnorm(input_tensor, weight, out=output_tensor, eps=eps, enable_pdl=False)
        reference = (
            input_tensor.float()
            * torch.rsqrt(input_tensor.float().pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight.float()
        ).to(dtype)
        error = (output_tensor.float() - reference.float()).abs()
        tolerance = atol + rtol * reference.float().abs()
        max_abs_error = max(max_abs_error, float(error.max().item()))
        agreed += int((error <= tolerance).sum().item())
        elements += error.numel()
    torch.cuda.synchronize(device)

    input_tensor = torch.randn((batch, hidden_size), device=device, dtype=dtype)
    weight = torch.randn((hidden_size,), device=device, dtype=dtype)
    output_tensor = torch.empty_like(input_tensor)
    for _ in range(warmup):
        flashinfer.norm.rmsnorm(input_tensor, weight, out=output_tensor, eps=eps, enable_pdl=False)
    torch.cuda.synchronize(device)
    samples_micros = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        flashinfer.norm.rmsnorm(input_tensor, weight, out=output_tensor, eps=eps, enable_pdl=False)
        end.record()
        end.synchronize()
        samples_micros.append(float(start.elapsed_time(end)) * 1_000.0)

    properties = torch.cuda.get_device_properties(device)
    artifact_digest = sha256_file(module_path)
    plan_id = f"flashinfer-rmsnorm-sm90-b{batch}-h{hidden_size}-c{context_tokens}"
    semantic_contract = semantics(batch, hidden_size, eps, "bfloat16")
    resource_peak = int(torch.cuda.max_memory_allocated(device))
    plan = {
        "schema_version": 1,
        "id": plan_id,
        "model": {
            "name": "operator:rmsnorm",
            "semantics_sha256": canonical_sha256(semantic_contract),
        },
        "hardware": {
            "accelerator": "nvidia",
            "architecture": "sm90",
            "device_count": 1,
            "min_device_memory_bytes": resource_peak,
        },
        "workload": exact_domain(batch, context_tokens),
        "resources": {
            "peak_device_memory_bytes": resource_peak,
            "workspace_bytes": 0,
            "stable_addresses": False,
            "capture": "eager",
        },
        "steps": [
            {
                "id": "rmsnorm",
                "provider": "flashinfer",
                "operation": "rmsnorm",
                "artifact_sha256": artifact_digest,
                "inputs": ["input", "weight"],
                "outputs": ["output"],
                "config": {
                    "batch": str(batch),
                    "hidden_size": str(hidden_size),
                    "dtype": "bfloat16",
                    "eps": repr(eps),
                    "enable_pdl": "false",
                },
            }
        ],
        "fallback_plan_ids": [],
        "labels": {"scope": "operator", "requires_runtime_trace": "true"},
    }
    evidence = {
        "schema_version": 1,
        "hardware": {
            "accelerator": "nvidia",
            "architecture": "sm90",
            "device_count": 1,
            "device_memory_bytes": int(properties.total_memory),
            "driver": driver_version(),
            "runtime": f"cuda-{torch.version.cuda};torch-{torch.__version__}",
        },
        "numerical": {
            "oracle": "pytorch-fp32-rmsnorm",
            "cases": cases,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs_error,
            "min_output_agreement": agreed / elements,
        },
        "timing": {"tokens_per_sample": batch, "samples_micros": samples_micros},
        "resources": {
            "peak_device_memory_bytes": resource_peak,
            "peak_workspace_bytes": 0,
        },
        "resilience": {"soak_seconds": 0, "passed_faults": []},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "flashinfer": flashinfer.__version__,
            "flashinfer_revision": git_revision(FLASHINFER_SOURCE),
            "device_name": properties.name,
            "provider_provenance": provenance["provenance"],
        },
        "artifacts": [{"path": "norm.so", "sha256": artifact_digest}],
        "semantic_contract": semantic_contract,
        "performance_summary": summarize_samples(samples_micros, batch),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copy2(module_path, temporary / "norm.so")
        (temporary / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        (temporary / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--context-tokens", type=int, default=32)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    result = qualify(
        args.out,
        batch=args.batch,
        hidden_size=args.hidden_size,
        context_tokens=args.context_tokens,
        cases=args.cases,
        samples=args.samples,
        warmup=args.warmup,
        eps=args.eps,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(result)


if __name__ == "__main__":
    main()
