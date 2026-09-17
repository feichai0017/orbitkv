#!/usr/bin/env python3
"""Build and qualify one pinned OrbitKV DeepGEMM H20 contract.

The script is intentionally outside serving. It produces a source file, a host
qualification library, and a JSON evidence record. A successful run proves one
concrete contract; it does not admit untested shapes or row buckets.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Any

def checked_output(arguments: list[str], cwd: pathlib.Path | None = None) -> str:
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require_revision(root: pathlib.Path, revision: str, name: str) -> None:
    actual = checked_output(["git", "rev-parse", "HEAD"], root)
    if actual != revision:
        raise RuntimeError(f"{name} checkout is {actual}, expected {revision}")


def build(
    repo: pathlib.Path,
    provider: pathlib.Path,
    output: pathlib.Path,
    nvcc: pathlib.Path,
    rows: int,
    n: int,
    k: int,
) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path, pathlib.Path, str]:
    output.mkdir(parents=True, exist_ok=True)
    stem = f"deepgemm-h20-m{rows}-n{n}-k{k}"
    source = output / f"{stem}.cu"
    library = output / f"{stem}.so"
    cubin = output / f"{stem}.cubin"
    contract = json.loads(
        checked_output(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "--bin",
                "orbitkv",
                "--",
                "provider",
                "deepgemm-h20",
                "plan",
                str(rows),
                str(n),
                str(k),
            ],
            repo,
        )
    )
    require_revision(provider, contract["provider_revision"], "DeepGEMM")
    cutlass_revision = checked_output(["git", "ls-tree", "HEAD", "third-party/cutlass"], provider).split()[2]
    require_revision(provider / "third-party/cutlass", cutlass_revision, "CUTLASS")
    for marker in [
        provider / "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh",
        provider / "third-party/cutlass/include/cutlass/arch/synclog.hpp",
    ]:
        if not marker.is_file():
            raise RuntimeError(f"provider checkout is incomplete: missing {marker}")
    rendered = checked_output(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--bin",
            "orbitkv",
            "--",
            "provider",
            "deepgemm-h20",
            "source",
            str(rows),
            str(n),
            str(k),
        ],
        repo,
    )
    source.write_text(rendered + "\n")
    subprocess.run(
        [
            str(nvcc),
            "-x",
            "cu",
            "-shared",
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--diag-suppress=39,161,174,177,186,940",
            "--ptxas-options=--register-usage-level=10",
            "-Xcompiler=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi",
            "-gencode",
            "arch=compute_90a,code=sm_90a",
            f"-I{provider / 'deep_gemm/include'}",
            f"-I{provider / 'third-party/cutlass/include'}",
            "-lcuda",
            "-lcudart",
            "-o",
            str(library),
            str(source),
        ],
        check=True,
        cwd=repo,
    )
    subprocess.run(
        [
            str(nvcc),
            "-x",
            "cu",
            "-cubin",
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--diag-suppress=39,161,174,177,186,940",
            "--ptxas-options=--register-usage-level=10",
            "-gencode",
            "arch=compute_90a,code=sm_90a",
            f"-I{provider / 'deep_gemm/include'}",
            f"-I{provider / 'third-party/cutlass/include'}",
            "-o",
            str(cubin),
            str(source),
        ],
        check=True,
        cwd=repo,
    )
    return contract, source, library, cubin, cutlass_revision


def qualify(library_path: pathlib.Path, contract: dict[str, Any], warmup: int, samples: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"expected SM90, found SM{properties.major}{properties.minor}")
    if properties.multi_processor_count != contract["num_sms"]:
        raise RuntimeError(
            f"expected {contract['num_sms']} SMs, found {properties.multi_processor_count}"
        )

    rows = contract["row_limit"]
    n = contract["shape"]["output_features"]
    k = contract["shape"]["input_features"]
    layout = contract["layout"]
    input_cpu = (((torch.arange(rows * k) % 31).float() - 15.0) / 16.0).reshape(rows, k).to(torch.bfloat16)
    input_values = input_cpu.to(device="cuda")
    weight_row = torch.full((k,), 0.5, dtype=torch.float8_e4m3fn, device="cuda")
    weight_row[::3] = -0.5
    weight = weight_row.repeat(n, 1)
    k_blocks = k // 128
    n_blocks = (n + 127) // 128
    weight_scale = torch.empty((n_blocks, k_blocks), dtype=torch.float32, device="cuda")
    host_scale = torch.empty((n_blocks, k_blocks), dtype=torch.float32)
    for n_block in range(n_blocks):
        for k_block in range(k_blocks):
            host_scale[n_block, k_block] = 0.25 * (1 + (n_block + k_block) % 4)
    weight_scale.copy_(host_scale)
    quantized = torch.empty(layout["scratch"]["quantized_bytes"], dtype=torch.uint8, device="cuda")
    activation_scale = torch.empty(tuple(layout["activation_scale_f32"]), dtype=torch.float32, device="cuda")
    output = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")

    library = ctypes.CDLL(str(library_path))
    run = library.orbitkv_deepgemm_run
    run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
    run.restype = ctypes.c_int
    last_error = library.orbitkv_deepgemm_last_error
    last_error.restype = ctypes.c_char_p
    stream = torch.cuda.current_stream()

    def launch() -> None:
        status = run(
            input_values.data_ptr(),
            weight.data_ptr(),
            weight_scale.data_ptr(),
            quantized.data_ptr(),
            activation_scale.data_ptr(),
            output.data_ptr(),
            rows,
            stream.cuda_stream,
        )
        if status != 0:
            message = last_error()
            raise RuntimeError(message.decode() if message else f"provider returned {status}")

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    input_blocks = input_cpu.float().reshape(rows, k_blocks, 128)
    expected_scales = input_blocks.abs().amax(dim=2).clamp_min(1.0e-4) * (1.0 / 448.0)
    inverse_scales = torch.reciprocal(expected_scales)
    expected_quantized = (input_blocks * inverse_scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    actual_scales = activation_scale[:, :rows].transpose(0, 1).cpu()
    scale_error = (actual_scales - expected_scales).abs().max().item()
    quantized_codes_match = torch.equal(
        quantized.view(torch.uint8).reshape(rows, k).cpu(),
        expected_quantized.view(torch.uint8).reshape(rows, k),
    )
    if scale_error != 0.0 or not quantized_codes_match:
        raise RuntimeError(
            f"quantizer gate failed: scale_max_abs={scale_error}, codes_match={quantized_codes_match}"
        )
    weight_pattern = weight_row.float().cpu().reshape(k_blocks, 128)
    block_sums = (expected_quantized.float() * expected_scales.unsqueeze(2) * weight_pattern.unsqueeze(0)).sum(2)
    expected_by_block = block_sums @ host_scale.transpose(0, 1)
    expected = expected_by_block.repeat_interleave(128, dim=1)[:, :n].to(torch.bfloat16)
    actual = output.cpu()
    maximum_absolute_error = (actual.float() - expected.float()).abs().max().item()
    exact_fraction = (actual == expected).float().mean().item()
    if maximum_absolute_error > 0.5:
        raise RuntimeError(
            f"numerical gate failed: max_abs={maximum_absolute_error}, exact_fraction={exact_fraction}"
        )

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        launch()
        end.record(stream)
        end.synchronize()
        timings.append(start.elapsed_time(end))

    median_ms = statistics.median(timings)
    return {
        "device": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
            "torch": torch.__version__,
        },
        "numerical": {
            "oracle": "patterned-activation block-scale analytical reference",
            "activation_scale_maximum_absolute_error": scale_error,
            "quantized_activation_codes_match": quantized_codes_match,
            "maximum_allowed_absolute_error": 0.5,
            "maximum_absolute_error": maximum_absolute_error,
            "exact_fraction": exact_fraction,
        },
        "timing": {
            "scope": "combined BF16 activation quantization plus FP8 GEMM",
            "warmup": warmup,
            "samples": samples,
            "median_ms": median_ms,
            "minimum_ms": min(timings),
            "maximum_ms": max(timings),
        },
        "benchmark": {
            "median_nanoseconds": round(median_ms * 1_000_000),
            "samples": samples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--nvcc", type=pathlib.Path, default=pathlib.Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--output-features", type=int, default=14_336)
    parser.add_argument("--input-features", type=int, default=5_120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if min(args.rows, args.output_features, args.input_features, args.warmup, args.samples) <= 0:
        parser.error("dimensions, warmup, and samples must be positive")

    repo = pathlib.Path(__file__).resolve().parents[1]
    contract, source, library, cubin, cutlass_revision = build(
        repo,
        args.provider_dir.resolve(),
        args.output_dir.resolve(),
        args.nvcc.resolve(),
        args.rows,
        args.output_features,
        args.input_features,
    )
    evidence = qualify(library, contract, args.warmup, args.samples)
    benchmark = evidence.pop("benchmark")
    contract["qualification"] = {
        "state": "benchmarked",
        "source_sha256": digest(source),
        "cubin_sha256": digest(cubin),
        **benchmark,
    }
    evidence.update(
        {
            "contract": contract,
            "provider": {
                "deepgemm_revision": contract["provider_revision"],
                "cutlass_revision": cutlass_revision,
                "numerical_abi": contract["numerical_abi"],
            },
            "artifacts": {
                "source": str(source),
                "source_sha256": digest(source),
                "library": str(library),
                "library_sha256": digest(library),
                "cubin": str(cubin),
                "cubin_sha256": digest(cubin),
            },
            "nvcc": checked_output([str(args.nvcc), "--version"]).splitlines()[-1],
        }
    )
    report = args.output_dir.resolve() / "qualification.json"
    report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        raise
