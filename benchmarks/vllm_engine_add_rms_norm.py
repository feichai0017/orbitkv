"""Run a real vLLM generate loop with one fused Add+RMSNorm provider."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=("loom_cuda", "vllm_c"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Write the machine-readable report to this path in addition to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("batch_size", "input_len", "output_len", "warmup", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    # vLLM launches worker subprocesses which JIT-compile a few helper kernels.
    # Preserve the active virtual environment in their PATH so tools installed
    # there (notably ninja) remain discoverable.
    # Do not resolve the interpreter symlink: a venv commonly points its
    # `python` at /usr/bin/python, while its helper executables live beside the
    # symlink in <venv>/bin.
    venv_bin = str(Path(sys.executable).absolute().parent)
    current_path = os.environ.get("PATH", "")
    if venv_bin not in current_path.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + current_path

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    prompts = [
        {
            "prompt_token_ids": [
                3 + ((batch_index * 17 + position * 13) % 1000)
                for position in range(args.input_len)
            ]
        }
        for batch_index in range(args.batch_size)
    ]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    engine = LLM(
        model=str(args.model.resolve()),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=args.input_len + args.output_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        ir_op_priority={"fused_add_rms_norm": [args.provider]},
        seed=31,
    )

    for _ in range(args.warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    samples_ms: list[float] = []
    token_ids: list[list[int]] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)
        token_ids = [request.outputs[0].token_ids for request in outputs]
        if any(len(tokens) != args.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")

    median_ms = statistics.median(samples_ms)
    report = {
        "benchmark": "vllm_engine_generate",
        "model": str(args.model.resolve()),
        "model_kind": "synthetic-random-qwen2",
        "provider": args.provider,
        "dtype": "bf16",
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "latency_ms": {
            "minimum": min(samples_ms),
            "median": median_ms,
            "maximum": max(samples_ms),
            "samples": samples_ms,
        },
        "median_decode_step_ms": median_ms / args.output_len,
        "median_output_tokens_per_second": (
            args.batch_size * args.output_len / (median_ms / 1000.0)
        ),
        "token_ids": token_ids,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
