#!/usr/bin/env python3
"""Run provider-isolated vLLM W8A8 fused-boundary A/B evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


MODEL_DEFAULT = "RedHatAI/Qwen2.5-0.5B-quantized.w8a8"
MODEL_REVISION_DEFAULT = "0d1951533baeec89dd257df86d4d10c3bd5429f5"
LINEAR_BACKEND_DEFAULT = "cutlass"
PROVIDERS = ("vllm", "loom")
FUSIONS = ("rms-norm", "silu-and-mul")
FUSION_ENVIRONMENTS = {
    "rms-norm": "LOOM_KERNELS_ENABLE_RMS_NORM_INT8",
    "silu-and-mul": "LOOM_KERNELS_ENABLE_SILU_AND_MUL_INT8",
}
FUSION_METADATA_KEYS = {
    "rms-norm": "rms_norm_int8_override",
    "silu-and-mul": "silu_and_mul_int8_override",
}
FUSION_OPERATOR_NAMES = {
    "rms-norm": "RMS_NORM_DYNAMIC_INT8",
    "silu-and-mul": "SILU_AND_MUL_DYNAMIC_INT8",
}
FUSION_SOURCE_KEYS = {
    "rms-norm": "loom_rms_norm_int8",
    "silu-and-mul": "loom_silu_and_mul_int8",
}
FUSION_BOUNDARIES = {
    "rms-norm": "RMSNorm-to-INT8",
    "silu-and-mul": "SiLU-and-Mul-to-INT8",
}
QUALITY_PROMPTS = (
    ("english_fact", "The capital of France is"),
    ("english_science", "Water freezes at a temperature of"),
    ("english_history", "The printing press was invented by"),
    ("english_summary", "Summarize why regular exercise is beneficial:"),
    ("english_reasoning", "If all roses are flowers and some flowers fade,"),
    ("english_writing", "Write one sentence describing a quiet library:"),
    ("english_translation", "Translate 'good morning' into Spanish:"),
    ("english_definition", "In machine learning, overfitting means"),
    ("chinese_fact", "中国的首都是"),
    ("chinese_science", "水在标准大气压下的沸点是"),
    ("chinese_summary", "用一句话说明为什么要保护环境："),
    ("chinese_reasoning", "如果所有的猫都是动物，那么"),
    ("chinese_writing", "写一句描绘清晨公园的话："),
    ("chinese_translation", "把“人工智能”翻译成英文："),
    ("chinese_definition", "在计算机科学中，算法是"),
    ("chinese_advice", "保持专注的一个简单方法是"),
    ("python_function", "def reverse_list(values):\n    "),
    ("python_loop", "for index in range(10):\n    "),
    ("python_docstring", 'def add(a, b):\n    """Return'),
    ("rust_function", "fn add(left: i32, right: i32) -> i32 {\n    "),
    ("rust_result", "fn parse_number(text: &str) -> Result<i32,"),
    ("sql_query", "SELECT name FROM users WHERE"),
    ("shell_command", "To list all files in the current directory, run"),
    ("json_shape", '{"name": "loom", "language":'),
    ("arithmetic", "The value of 17 multiplied by 6 is"),
    ("algebra", "Solve for x: 3x + 5 = 20. x ="),
    ("sequence", "Complete the sequence: 2, 4, 8, 16,"),
    ("probability", "A fair coin is flipped once. The probability of heads is"),
    ("logic", "A is taller than B, and B is taller than C. Therefore"),
    ("instruction", "List three primary colors, separated by commas:"),
    ("comparison", "Compared with a hard disk, RAM is generally"),
    ("llm_inference", "During autoregressive language model decoding, the KV cache"),
)


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    input_len: int
    output_len: int

    @property
    def label(self) -> str:
        return f"b{self.batch_size}-in{self.input_len}-out{self.output_len}"

    @property
    def argument(self) -> str:
        return f"{self.batch_size}x{self.input_len}x{self.output_len}"


def parse_case(value: str) -> BenchmarkCase:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT")
    try:
        batch_size, input_len, output_len = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case dimensions must be positive integers"
        ) from error
    if min(batch_size, input_len, output_len) <= 0:
        raise argparse.ArgumentTypeError("case dimensions must be positive")
    return BenchmarkCase(batch_size, input_len, output_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--model-revision",
        default=MODEL_REVISION_DEFAULT,
        help="Pinned checkpoint revision; recorded for local snapshots too.",
    )
    parser.add_argument(
        "--repository-revision",
        help="Full repository base SHA for an uncommitted candidate.",
    )
    parser.add_argument("--linear-backend", default=LINEAR_BACKEND_DEFAULT)
    parser.add_argument("--fusion", choices=FUSIONS, default="rms-norm")
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        help="Repeatable BATCHxINPUTxOUTPUT performance workload.",
    )
    parser.add_argument("--quality-prompts", type=int, default=32)
    parser.add_argument("--quality-logprobs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--provider-order",
        choices=("baseline-first", "loom-first"),
        default="baseline-first",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-provider", choices=PROVIDERS, help=argparse.SUPPRESS
    )
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-cache-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 128, 32),
            BenchmarkCase(8, 128, 32),
            BenchmarkCase(32, 128, 16),
        ]
    if min(
        args.quality_prompts,
        args.quality_logprobs,
        args.warmup,
        args.repeats,
    ) <= 0:
        parser.error("quality and timing counts must be positive")
    if args.quality_logprobs > 64:
        parser.error("quality-logprobs must be at most 64")
    if args.quality_prompts > len(QUALITY_PROMPTS):
        parser.error(
            f"quality-prompts must be at most {len(QUALITY_PROMPTS)}"
        )
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.repository_revision is not None and (
        len(args.repository_revision) != 40
    ):
        parser.error("repository-revision must be a full 40-character Git SHA")
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error(
            "internal provider runs require --internal-result and "
            "--internal-cache-root"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def make_prompt(batch_index: int, length: int, salt: int) -> list[int]:
    return [
        3 + ((salt * 101 + batch_index * 17 + position * 13) % 4093)
        for position in range(length)
    ]


def make_performance_prompts(
    case: BenchmarkCase,
) -> list[dict[str, list[int]]]:
    return [
        {"prompt_token_ids": make_prompt(batch_index, case.input_len, 7)}
        for batch_index in range(case.batch_size)
    ]


def prompt_descriptor(
    prompt_name: str,
    prompt_text: str,
    token_ids: list[int],
    index: int,
) -> dict[str, Any]:
    token_digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "prompt_index": index,
        "prompt_name": prompt_name,
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "input_len": len(token_ids),
        "token_ids_sha256": token_digest,
    }


def request_metrics(outputs: list[Any]) -> tuple[list[float], list[float]]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            continue
        if metrics.first_token_latency > 0.0:
            ttft_ms.append(metrics.first_token_latency * 1000.0)
        generated = metrics.num_generation_tokens
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        if generated > 1 and decode_seconds >= 0.0:
            tpot_ms.append(decode_seconds * 1000.0 / (generated - 1))
    return ttft_ms, tpot_ms


def serialize_logprobs(step: dict[int, Any]) -> list[dict[str, Any]]:
    entries = [
        {
            "token_id": int(token_id),
            "logprob": float(logprob.logprob),
            "rank": None if logprob.rank is None else int(logprob.rank),
        }
        for token_id, logprob in step.items()
    ]
    return sorted(
        entries,
        key=lambda entry: (
            entry["rank"] is None,
            entry["rank"] if entry["rank"] is not None else sys.maxsize,
            entry["token_id"],
        ),
    )


def top1_margin(entries: list[dict[str, Any]]) -> float | None:
    ranked = [entry for entry in entries if entry["rank"] is not None]
    ranked.sort(key=lambda entry: (entry["rank"], entry["token_id"]))
    if len(ranked) < 2:
        return None
    return float(ranked[0]["logprob"] - ranked[1]["logprob"])


def run_quality(
    engine: Any,
    sampling_type: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    named_prompts = QUALITY_PROMPTS[: args.quality_prompts]
    prompts = [text for _, text in named_prompts]
    sampling = sampling_type(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        logprobs=args.quality_logprobs,
    )
    for _ in range(args.warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    requests: list[dict[str, Any]] = []
    for index, ((prompt_name, prompt_text), output) in enumerate(
        zip(named_prompts, outputs, strict=True)
    ):
        completion = output.outputs[0]
        if len(completion.token_ids) != 1:
            raise RuntimeError("quality request did not return exactly one token")
        if completion.logprobs is None or len(completion.logprobs) != 1:
            raise RuntimeError("quality request did not return one logprob step")
        entries = serialize_logprobs(completion.logprobs[0])
        generated_token = int(completion.token_ids[0])
        if generated_token not in {
            int(entry["token_id"]) for entry in entries
        }:
            raise RuntimeError("generated token is absent from returned logprobs")
        prompt_token_ids = output.prompt_token_ids
        if prompt_token_ids is None:
            raise RuntimeError("tokenized quality prompt IDs are unavailable")
        requests.append(
            {
                **prompt_descriptor(
                    prompt_name,
                    prompt_text,
                    list(prompt_token_ids),
                    index,
                ),
                "generated_token_id": generated_token,
                "top1_margin": top1_margin(entries),
                "returned_logprobs": entries,
            }
        )
    return {
        "prompt_count": len(prompts),
        "max_tokens": 1,
        "num_logprobs": args.quality_logprobs,
        "batch_latency_ms": elapsed_ms,
        "requests": requests,
    }


def run_performance_case(
    engine: Any,
    sampling_type: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    prompts = make_performance_prompts(case)
    sampling = sampling_type(
        temperature=0.0,
        max_tokens=case.output_len,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    latency_ms: list[float] = []
    throughput: list[float] = []
    all_ttft_ms: list[float] = []
    all_tpot_ms: list[float] = []
    token_ids: list[list[int]] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)
        throughput.append(
            case.batch_size * case.output_len / (elapsed_ms / 1000.0)
        )
        ttft_ms, tpot_ms = request_metrics(outputs)
        all_ttft_ms.extend(ttft_ms)
        all_tpot_ms.extend(tpot_ms)
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        if any(len(tokens) != case.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")

    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "batch_latency_ms": summary(latency_ms),
        "request_ttft_ms": summary(all_ttft_ms),
        "request_tpot_ms": summary(all_tpot_ms),
        "output_tokens_per_second": summary(throughput),
        "token_ids": token_ids,
    }


def audit_generated_sources(cache_root: Path) -> dict[str, Any]:
    patterns = {
        "loom_rms_norm_int8": (
            "torch.ops.loom_kernels."
            "rms_norm_dynamic_per_token_int8.default"
        ),
        "loom_silu_and_mul_int8": (
            "torch.ops.loom_kernels."
            "silu_and_mul_dynamic_per_token_int8.default"
        ),
        "vllm_silu_and_mul": "torch.ops._C.silu_and_mul.default",
        "vllm_dynamic_int8_quant": (
            "torch.ops._C.dynamic_scaled_int8_quant.default"
        ),
        "cutlass_scaled_mm": "torch.ops._C.cutlass_scaled_mm.default",
    }
    call_counts = {name: 0 for name in patterns}
    matching_files = {name: [] for name in patterns}
    python_files = 0
    for source in sorted(cache_root.rglob("*.py")):
        python_files += 1
        contents = source.read_text(encoding="utf-8", errors="replace")
        for name, pattern in patterns.items():
            count = contents.count(pattern)
            if count > 0:
                call_counts[name] += count
                matching_files[name].append(str(source.relative_to(cache_root)))
    return {
        "python_files": python_files,
        "patterns": patterns,
        "call_counts": call_counts,
        "matching_files": matching_files,
    }


def driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()[0]


def prepare_environment(
    provider: str,
    cache_root: Path,
    fusion: str,
) -> None:
    for environment in FUSION_ENVIRONMENTS.values():
        os.environ[environment] = "0"
    if provider == "loom":
        os.environ[FUSION_ENVIRONMENTS[fusion]] = "1"
    os.environ["LOOM_KERNELS_ENABLE_RMS_NORM_FP8"] = "0"
    os.environ["LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc was not found at {nvcc}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    current_path = os.environ.get("PATH", "").split(os.pathsep)
    required_path = [
        str(Path(sys.executable).absolute().parent),
        str(cuda_home / "bin"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required_path if entry not in current_path]
        + current_path
    )


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None
    assert args.internal_cache_root is not None
    cache_root = args.internal_cache_root.resolve()
    prepare_environment(provider, cache_root, args.fusion)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import provider_metadata, register_vllm_ir
    from vllm.compilation.passes.vllm_inductor_pass import get_match_table

    registered_provider = register_vllm_ir()
    if registered_provider is None:
        raise RuntimeError("Loom's vLLM IR provider did not register")
    metadata_before_engine = provider_metadata()
    metadata_key = FUSION_METADATA_KEYS[args.fusion]
    int8_registered = bool(metadata_before_engine[metadata_key])
    if int8_registered != (provider == "loom"):
        raise RuntimeError(
            "INT8 fusion registration does not match selected provider"
        )

    model_path = Path(args.model).expanduser()
    model_is_local = model_path.exists()
    model = str(model_path.resolve()) if model_is_local else args.model
    maximum_input = max(
        max(case.input_len for case in args.cases),
        128,
    )
    maximum_output = max(case.output_len for case in args.cases)
    maximum_batch = max(
        max(case.batch_size for case in args.cases),
        args.quality_prompts,
    )
    maximum_model_len = maximum_input + maximum_output
    quality_batched_tokens = args.quality_prompts * 128
    engine_arguments: dict[str, Any] = {
        "model": model,
        "skip_tokenizer_init": False,
        "dtype": "bfloat16",
        "linear_backend": args.linear_backend,
        "max_model_len": maximum_model_len,
        "max_num_seqs": maximum_batch,
        "max_num_batched_tokens": max(
            maximum_model_len,
            quality_batched_tokens,
            max(
                case.batch_size * case.input_len for case in args.cases
            ),
        ),
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": False,
        "compilation_config": {
            "pass_config": {
                "fuse_norm_quant": args.fusion == "rms-norm",
                "fuse_act_quant": args.fusion == "silu-and-mul",
            }
        },
        "disable_log_stats": False,
    }
    if args.model_revision and not model_is_local:
        engine_arguments["revision"] = args.model_revision
    engine = LLM(**engine_arguments)
    vllm_config = getattr(engine.llm_engine, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    detected_quantization = getattr(model_config, "quantization", None)
    if detected_quantization is None:
        raise RuntimeError("checkpoint did not resolve to a quantized vLLM model")
    operator = getattr(Operator, FUSION_OPERATOR_NAMES[args.fusion])
    reset_launch_count(operator)
    matches_after_engine = get_match_table()
    try:
        quality = run_quality(engine, SamplingParams, args)
        performance = [
            run_performance_case(engine, SamplingParams, case, args)
            for case in args.cases
        ]
        torch.cuda.synchronize()
        host_launch_count = launch_count(operator)
        matches_after_workloads = get_match_table()
        generated_sources = audit_generated_sources(cache_root)
        local_model_files: dict[str, str] = {}
        if model_is_local:
            for name in (
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "quantize_config.json",
            ):
                candidate = Path(model) / name
                if candidate.is_file():
                    local_model_files[name] = sha256_file(candidate)
        report = {
            "provider": provider,
            "registered_provider": registered_provider,
            "model": model,
            "model_source": args.model,
            "model_revision": args.model_revision,
            "model_kind": (
                "local-checkpoint" if model_is_local else "huggingface"
            ),
            "local_model_file_sha256": local_model_files,
            "dtype": "bfloat16",
            "weight_activation_quantization": "W8A8 checkpoint metadata",
            "detected_vllm_quantization": detected_quantization,
            "linear_backend": args.linear_backend,
            "fusion": args.fusion,
            "fusion_boundary": FUSION_BOUNDARIES[args.fusion],
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "quality": quality,
            "performance": performance,
            "loom_path": {
                "metadata_before_engine": metadata_before_engine,
                "host_launch_count": host_launch_count,
                "fusion_matches_after_engine": matches_after_engine,
                "fusion_matches_after_workloads": matches_after_workloads,
                "generated_sources": generated_sources,
                "counter_semantics": (
                    "host submissions during graph construction or eager "
                    "execution; CUDA Graph replays do not increment this counter"
                ),
            },
            "environment": {
                "host": os.uname().nodename,
                "gpu": torch.cuda.get_device_name(0),
                "driver": driver_version(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "vllm": vllm.__version__,
                "cuda_home": os.environ["CUDA_HOME"],
                "v1_multiprocessing": os.environ[
                    "VLLM_ENABLE_V1_MULTIPROCESSING"
                ],
                "vllm_cache_root": os.environ["VLLM_CACHE_ROOT"],
            },
        }
    finally:
        engine.llm_engine.engine_core.shutdown()

    assert args.internal_result is not None
    args.internal_result.parent.mkdir(parents=True, exist_ok=True)
    args.internal_result.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"provider={provider} int8_registered={int8_registered} "
        f"fusion={args.fusion} loom_host_launches={host_launch_count}",
        file=sys.stderr,
    )
    return report


def child_command(
    args: argparse.Namespace,
    provider: str,
    result: Path,
    cache_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--model-revision",
        args.model_revision,
        "--linear-backend",
        args.linear_backend,
        "--fusion",
        args.fusion,
        "--quality-prompts",
        str(args.quality_prompts),
        "--quality-logprobs",
        str(args.quality_logprobs),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--internal-provider",
        provider,
        "--internal-result",
        str(result),
        "--internal-cache-root",
        str(cache_root),
    ]
    if args.repository_revision is not None:
        command.extend(("--repository-revision", args.repository_revision))
    for case in args.cases:
        command.extend(("--case", case.argument))
    return command


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def median_metric(case: dict[str, Any], name: str) -> float | None:
    metric = case[name]
    return None if metric is None else float(metric["median"])


def compare_quality(
    baseline: dict[str, Any],
    loom: dict[str, Any],
) -> dict[str, Any]:
    baseline_requests = baseline["requests"]
    loom_requests = loom["requests"]
    if len(baseline_requests) != len(loom_requests):
        raise RuntimeError("providers returned different quality request counts")

    comparisons: list[dict[str, Any]] = []
    common_errors: list[float] = []
    jaccards: list[float] = []
    token_agreements = 0
    returned_sets_equal = 0
    for baseline_request, loom_request in zip(
        baseline_requests, loom_requests, strict=True
    ):
        for key in (
            "prompt_index",
            "prompt_name",
            "prompt_text_sha256",
            "input_len",
            "token_ids_sha256",
        ):
            if baseline_request[key] != loom_request[key]:
                raise RuntimeError("providers used different quality prompts")
        baseline_entries = {
            int(entry["token_id"]): entry
            for entry in baseline_request["returned_logprobs"]
        }
        loom_entries = {
            int(entry["token_id"]): entry
            for entry in loom_request["returned_logprobs"]
        }
        baseline_ids = set(baseline_entries)
        loom_ids = set(loom_entries)
        intersection = baseline_ids & loom_ids
        union = baseline_ids | loom_ids
        request_errors = [
            abs(
                float(baseline_entries[token_id]["logprob"])
                - float(loom_entries[token_id]["logprob"])
            )
            for token_id in intersection
        ]
        common_errors.extend(request_errors)
        jaccard = len(intersection) / len(union)
        jaccards.append(jaccard)
        tokens_match = (
            baseline_request["generated_token_id"]
            == loom_request["generated_token_id"]
        )
        token_agreements += int(tokens_match)
        sets_equal = baseline_ids == loom_ids
        returned_sets_equal += int(sets_equal)
        baseline_top1 = int(baseline_request["generated_token_id"])
        loom_top1 = int(loom_request["generated_token_id"])
        comparisons.append(
            {
                "prompt_index": baseline_request["prompt_index"],
                "prompt_name": baseline_request["prompt_name"],
                "input_len": baseline_request["input_len"],
                "generated_tokens_match": tokens_match,
                "baseline_generated_token_id": baseline_top1,
                "loom_generated_token_id": loom_top1,
                "baseline_top1_margin": baseline_request["top1_margin"],
                "loom_top1_margin": loom_request["top1_margin"],
                "baseline_top1_rank_under_loom": (
                    loom_entries.get(baseline_top1, {}).get("rank")
                ),
                "loom_top1_rank_under_baseline": (
                    baseline_entries.get(loom_top1, {}).get("rank")
                ),
                "returned_token_sets_equal": sets_equal,
                "returned_token_intersection": len(intersection),
                "returned_token_union": len(union),
                "returned_token_jaccard": jaccard,
                "maximum_common_logprob_error": (
                    max(request_errors) if request_errors else None
                ),
                "mean_common_logprob_error": (
                    statistics.mean(request_errors)
                    if request_errors
                    else None
                ),
            }
        )

    count = len(comparisons)
    maximum_common_error = max(common_errors) if common_errors else None
    median_common_error = (
        statistics.median(common_errors) if common_errors else None
    )
    mean_common_error = (
        statistics.mean(common_errors) if common_errors else None
    )
    return {
        "prompt_count": count,
        "top1_token_agreement_count": token_agreements,
        "top1_token_agreement_rate": token_agreements / count,
        "all_top1_tokens_match": token_agreements == count,
        "returned_token_sets_equal_count": returned_sets_equal,
        "all_returned_token_sets_equal": returned_sets_equal == count,
        "minimum_returned_token_jaccard": min(jaccards),
        "median_returned_token_jaccard": statistics.median(jaccards),
        "mean_returned_token_jaccard": statistics.mean(jaccards),
        "maximum_common_logprob_error": maximum_common_error,
        "median_common_logprob_error": median_common_error,
        "mean_common_logprob_error": mean_common_error,
        "requests": comparisons,
    }


def compare_performance(
    baseline: list[dict[str, Any]],
    loom: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for baseline_case, loom_case in zip(baseline, loom, strict=True):
        if baseline_case["case"] != loom_case["case"]:
            raise RuntimeError("providers returned different performance cases")
        baseline_batch = median_metric(baseline_case, "batch_latency_ms")
        loom_batch = median_metric(loom_case, "batch_latency_ms")
        baseline_ttft = median_metric(baseline_case, "request_ttft_ms")
        loom_ttft = median_metric(loom_case, "request_ttft_ms")
        baseline_tpot = median_metric(baseline_case, "request_tpot_ms")
        loom_tpot = median_metric(loom_case, "request_tpot_ms")
        baseline_throughput = median_metric(
            baseline_case, "output_tokens_per_second"
        )
        loom_throughput = median_metric(
            loom_case, "output_tokens_per_second"
        )
        comparisons.append(
            {
                "case": baseline_case["case"],
                "token_ids_match": (
                    baseline_case["token_ids"] == loom_case["token_ids"]
                ),
                "baseline_over_loom_batch_latency": ratio(
                    baseline_batch, loom_batch
                ),
                "baseline_over_loom_ttft": ratio(baseline_ttft, loom_ttft),
                "baseline_over_loom_tpot": ratio(baseline_tpot, loom_tpot),
                "loom_over_baseline_output_throughput": ratio(
                    loom_throughput, baseline_throughput
                ),
            }
        )
    return comparisons


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    order = (
        ("vllm", "loom")
        if args.provider_order == "baseline-first"
        else ("loom", "vllm")
    )
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="loom-vllm-int8-ab-") as directory:
        root = Path(directory)
        for provider in order:
            result = root / f"{provider}.json"
            subprocess.run(
                child_command(args, provider, result, root / f"{provider}-cache"),
                check=True,
            )
            reports[provider] = json.loads(
                result.read_text(encoding="utf-8")
            )

    quality = compare_quality(
        reports["vllm"]["quality"], reports["loom"]["quality"]
    )
    performance = compare_performance(
        reports["vllm"]["performance"],
        reports["loom"]["performance"],
    )
    baseline_metadata = reports["vllm"]["loom_path"][
        "metadata_before_engine"
    ]
    loom_metadata = reports["loom"]["loom_path"]["metadata_before_engine"]
    baseline_launches = reports["vllm"]["loom_path"]["host_launch_count"]
    loom_launches = reports["loom"]["loom_path"]["host_launch_count"]
    baseline_sources = reports["vllm"]["loom_path"]["generated_sources"][
        "call_counts"
    ]
    loom_sources = reports["loom"]["loom_path"]["generated_sources"][
        "call_counts"
    ]
    metadata_key = FUSION_METADATA_KEYS[args.fusion]
    source_key = FUSION_SOURCE_KEYS[args.fusion]
    path_evidence = {
        "selected_fusion": args.fusion,
        "selected_metadata_key": metadata_key,
        "selected_generated_source_key": source_key,
        "baseline_selected_override_disabled": (
            not baseline_metadata[metadata_key]
        ),
        "loom_selected_override_enabled": bool(loom_metadata[metadata_key]),
        "baseline_loom_host_launches_zero": baseline_launches == 0,
        "loom_host_launch_observed": loom_launches > 0,
        "baseline_generated_source_has_no_selected_loom_op": (
            baseline_sources[source_key] == 0
        ),
        "loom_generated_source_has_selected_loom_op": (
            loom_sources[source_key] > 0
        ),
        "baseline_generated_source_has_cutlass_gemm": (
            baseline_sources["cutlass_scaled_mm"] > 0
        ),
        "loom_generated_source_has_cutlass_gemm": (
            loom_sources["cutlass_scaled_mm"] > 0
        ),
        "source_call_counts": {
            "baseline": baseline_sources,
            "loom": loom_sources,
        },
        "host_launch_counts": {
            "baseline": baseline_launches,
            "loom": loom_launches,
        },
    }
    path_gate_passed = all(
        value
        for key, value in path_evidence.items()
        if key
        not in {
            "selected_fusion",
            "selected_metadata_key",
            "selected_generated_source_key",
            "source_call_counts",
            "host_launch_counts",
        }
    )
    maximum_common_error = quality["maximum_common_logprob_error"]
    exact_quality_gate = (
        quality["all_top1_tokens_match"]
        and quality["all_returned_token_sets_equal"]
        and maximum_common_error is not None
        and maximum_common_error <= 1.0e-3
    )
    report = {
        "schema_version": 1,
        "benchmark": f"vllm_engine_{args.fusion.replace('-', '_')}_dynamic_int8_ab",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_base_revision": args.repository_revision,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "model": args.model,
        "model_revision": args.model_revision,
        "linear_backend": args.linear_backend,
        "fusion": args.fusion,
        "fusion_boundary": FUSION_BOUNDARIES[args.fusion],
        "provider_order": list(order),
        "decision": {
            "path_gate_passed": path_gate_passed,
            "exact_quality_gate_passed": exact_quality_gate,
            "recommended_runtime_mode": (
                "eligible-for-broader-quality-validation"
                if exact_quality_gate
                else "explicit-opt-in-only"
            ),
            "automatic_default_enable_decision": None,
            "reason": (
                "This run proves path isolation and measures one-step quality. "
                "Default enablement still requires a broader task/model suite."
            ),
        },
        "path_evidence": path_evidence,
        "quality_comparison": quality,
        "performance_comparisons": performance,
        "claim_boundary": {
            "accepted": [
                "Both providers use the same W8A8 checkpoint and Cutlass GEMM.",
                (
                    "Only the "
                    f"{FUSION_BOUNDARIES[args.fusion]} memory-bound boundary changes."
                ),
                "One-step top-logprob evidence compares identical prompt states.",
            ],
            "excluded": [
                "A single run is not a default-on quality decision.",
                "Token identity is not expected after autoregressive divergence.",
                "Order-stable performance requires both provider orders.",
                "No GEMM implementation or GEMM speedup is claimed.",
            ],
        },
        "providers": reports,
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="")
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(payload, encoding="utf-8")
    if not path_gate_passed:
        raise SystemExit("vLLM INT8 path-evidence gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is None:
        run_controller(args)
    else:
        run_provider(args)


if __name__ == "__main__":
    main()
