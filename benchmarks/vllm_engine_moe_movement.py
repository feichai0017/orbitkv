#!/usr/bin/env python3
"""Run an isolated real-engine A/B for Loom MoE movement admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


PROVIDERS = ("vllm", "loom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-provider", choices=PROVIDERS, help=argparse.SUPPRESS
    )
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.input_len,
        args.output_len,
        args.warmup,
        args.repeats,
    ) <= 0:
        parser.error("workload dimensions and timing counts must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be between zero and one")
    if args.internal_provider is not None and args.internal_result is None:
        parser.error("internal provider runs require --internal-result")
    return args


def prompts(batch_size: int, input_len: int) -> list[dict[str, list[int]]]:
    return [
        {
            "prompt_token_ids": [
                3 + ((batch * 37 + position * 19) % 1000)
                for position in range(input_len)
            ]
        }
        for batch in range(batch_size)
    ]


def run_provider(args: argparse.Namespace) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["LOOM_KERNELS_ENABLE_MOE_MOVEMENT"] = (
        "1" if args.internal_provider == "loom" else "0"
    )
    venv_bin = str(Path(sys.executable).absolute().parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from loom_kernels.torch_ops import bridge_abi_version
    from loom_kernels.vllm import provider_metadata, register_vllm_ir

    register_vllm_ir()
    engine = LLM(
        model=str(args.model.resolve()),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        quantization="fp8_per_channel",
        moe_backend="cutlass",
        max_model_len=args.input_len + args.output_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=97,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    workload = prompts(args.batch_size, args.input_len)
    for _ in range(args.warmup):
        engine.generate(workload, sampling, use_tqdm=False)

    samples_ms: list[float] = []
    token_ids: list[list[int]] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(workload, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        token_ids = [list(request.outputs[0].token_ids) for request in outputs]
    metadata = provider_metadata()
    if args.internal_provider == "loom":
        if metadata["moe_movement_permute_hits"] <= 0:
            raise RuntimeError("real engine never reached Loom MoE permutation")
        if metadata["moe_movement_combine_hits"] <= 0:
            raise RuntimeError("real engine never reached Loom MoE combine")
        first_contract = metadata["moe_movement_first_contract"]
        if first_contract is None or first_contract["hidden_dtype"] != "torch.float8_e4m3fn":
            raise RuntimeError("real engine did not admit the Cutlass FP8 movement path")

    report = {
        "provider": args.internal_provider,
        "token_ids": token_ids,
        "latency_ms": {
            "minimum": min(samples_ms),
            "median": statistics.median(samples_ms),
            "maximum": max(samples_ms),
            "samples": samples_ms,
        },
        "provider_metadata": metadata,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi": bridge_abi_version(),
            "quantization": "fp8_per_channel",
            "moe_backend": "cutlass",
            "v1_multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
        },
    }
    args.internal_result.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def run_isolated(args: argparse.Namespace, provider: str, result: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        str(args.model.resolve()),
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--internal-provider",
        provider,
        "--internal-result",
        str(result),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.internal_provider is not None:
        run_provider(args)
        return

    with tempfile.TemporaryDirectory(prefix="loom-moe-engine-") as temporary:
        temporary_path = Path(temporary)
        reports: dict[str, dict[str, Any]] = {}
        for provider in PROVIDERS:
            result = temporary_path / f"{provider}.json"
            run_isolated(args, provider, result)
            reports[provider] = json.loads(result.read_text(encoding="utf-8"))
    if reports["vllm"]["token_ids"] != reports["loom"]["token_ids"]:
        raise RuntimeError("vLLM and Loom providers generated different token IDs")
    baseline_ms = reports["vllm"]["latency_ms"]["median"]
    candidate_ms = reports["loom"]["latency_ms"]["median"]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "vllm_engine_moe_movement",
        "model": str(args.model.resolve()),
        "model_kind": "synthetic-random-qwen2-moe",
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "providers": reports,
        "semantic_gate": {
            "token_ids_exact": True,
            "loom_permute_reached": True,
            "loom_combine_reached": True,
            "cutlass_fp8_movement_contract_reached": True,
            "grouped_gemm_owner": "vllm_vendor_backend",
        },
        "median_speedup": baseline_ms / candidate_ms,
        "claim_boundary": [
            "This is a real vLLM generate loop over a synthetic local MoE checkpoint.",
            "Loom replaces only FP8 permutation and BF16 weighted combine.",
            "Cutlass grouped GEMM selection, weights, and matrix multiplication remain vLLM-owned.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
