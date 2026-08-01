#!/usr/bin/env python3
"""Qualify Loom MoE movement in an isolated pretrained vLLM engine.

The default controller runs an ABBA process order: native vLLM then Loom,
followed by Loom then native vLLM. Every child constructs a fresh engine and
owns a fresh compiler cache. Timed samples exclude engine construction and
warmup. Exact prompts, generated tokens, operator hits, unchanged vendor GEMM,
request latency, throughput, and CUDA memory are persisted in one result.

This benchmark deliberately measures only Loom's movement substitution around
vLLM-selected Cutlass grouped GEMM. It never replaces or wraps grouped GEMM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


PROVIDERS = ("vllm", "loom")
ORDER_MODES = {
    "baseline-first": ("vllm", "loom"),
    "loom-first": ("loom", "vllm"),
}
NATURAL_PROMPT_TEMPLATES = (
    (
        "Explain why mixture-of-experts inference can be limited by token "
        "routing, permutation, communication, and memory movement even when "
        "grouped matrix multiplication is highly optimized."
    ),
    (
        "Compare expert parallelism with tensor parallelism for serving a "
        "sparse language model, including load balance and communication."
    ),
    (
        "Write an engineering note about profiling a production inference "
        "optimization without confusing microbenchmarks with model latency."
    ),
    (
        "Describe how stable expert-major permutation and inverse permutation "
        "surround a vendor grouped GEMM in a mixture-of-experts layer."
    ),
    (
        "Analyze how batch size and decode length affect memory-bound CUDA "
        "operators in an online large language model serving engine."
    ),
    (
        "Explain why a fair GPU provider comparison uses isolated processes, "
        "identical token IDs, warmup, and reversed execution order."
    ),
    (
        "Summarize the correctness evidence needed before replacing an "
        "inference engine operator with a custom Rust and CUDA backend."
    ),
    (
        "Discuss when a fused memory-bound operator should be admitted into "
        "production and when the engine should retain its native path."
    ),
)
CONFIG_FIELDS = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "num_selected_experts",
    "vocab_size",
)
METADATA_FILENAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
LOOM_ENGINE_OVERRIDE_ENVIRONMENTS = (
    "LOOM_KERNELS_ENABLE_MOE_MOVEMENT",
    "LOOM_KERNELS_ENABLE_RMS_NORM_FP8",
    "LOOM_KERNELS_ENABLE_RMS_NORM_INT8",
    "LOOM_KERNELS_ENABLE_SILU_AND_MUL",
    "LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8",
    "LOOM_KERNELS_ENABLE_SILU_AND_MUL_INT8",
    "LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION",
    "LOOM_KERNELS_ENABLE_MIN_P",
    "LOOM_KERNELS_ENABLE_LOGITS_PREPROCESS",
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
    try:
        dimensions = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be BATCHxINPUTxOUTPUT"
        ) from error
    if len(dimensions) != 3 or min(dimensions) <= 0:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT")
    return BenchmarkCase(*dimensions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument(
        "--model-kind",
        choices=("production", "synthetic"),
        default="production",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("natural", "synthetic"),
        default="natural",
    )
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument(
        "--order-mode",
        choices=("both", *ORDER_MODES),
        default="both",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tested-revision", required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-provider",
        choices=PROVIDERS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-cache-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 128, 128),
            BenchmarkCase(8, 128, 64),
            BenchmarkCase(32, 128, 32),
        ]
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.model_kind == "production" and args.prompt_mode != "natural":
        parser.error("production qualification requires natural prompts")
    if not re.fullmatch(r"[0-9a-f]{40}", args.tested_revision):
        parser.error("tested-revision must be a full lowercase Git SHA")
    model_path = Path(args.model).expanduser()
    if model_path.exists() and args.model_revision:
        parser.error("a local checkpoint is pinned by manifest, not revision")
    if not model_path.exists() and not re.fullmatch(
        r"[0-9a-f]{40}", args.model_revision
    ):
        parser.error(
            "a Hugging Face model requires a pinned 40-character revision"
        )
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error("internal runs require result and cache paths")
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def resolve_model(value: str) -> tuple[str, str]:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve()), "local-checkpoint"
    return value, "huggingface"


def local_checkpoint_manifest(root: Path) -> dict[str, Any]:
    metadata_paths = sorted(
        {
            path
            for name in METADATA_FILENAMES
            if (path := root / name).is_file()
        }
        | set(root.glob("*.index.json"))
    )
    weights = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix in WEIGHT_SUFFIXES
    )
    weight_manifest = [
        {"name": path.name, "bytes": path.stat().st_size} for path in weights
    ]
    metadata_sha256 = {
        path.name: sha256_file(path) for path in metadata_paths
    }
    manifest_payload = json.dumps(
        {
            "metadata_sha256": metadata_sha256,
            "weights": weight_manifest,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "metadata_sha256": metadata_sha256,
        "weights": weight_manifest,
        "manifest_sha256": sha256_bytes(manifest_payload),
    }


def model_identity(
    model: str,
    model_source: str,
    model_kind: str,
    model_revision: str,
) -> tuple[dict[str, Any], Any]:
    from transformers import AutoConfig

    source_kind = "local-checkpoint" if Path(model).is_dir() else "huggingface"
    arguments: dict[str, Any] = {
        "local_files_only": source_kind == "local-checkpoint"
    }
    if model_revision and source_kind == "huggingface":
        arguments["revision"] = model_revision
    config = AutoConfig.from_pretrained(model, **arguments)
    config_dict = config.to_dict()
    resolved_commit = getattr(config, "_commit_hash", None)
    if (
        source_kind == "huggingface"
        and resolved_commit is not None
        and resolved_commit != model_revision
    ):
        raise RuntimeError(
            "resolved Hugging Face commit differs from the requested revision"
        )
    fixture = config_dict.get("loom_fixture")
    if model_kind == "production" and fixture is not None:
        raise RuntimeError(
            "production qualification rejected a Loom synthetic fixture"
        )
    if model_kind == "synthetic" and fixture != "synthetic-random-qwen2-moe":
        raise RuntimeError("synthetic qualification requires the Loom MoE fixture")
    expert_count = config_dict.get(
        "num_experts",
        config_dict.get("num_local_experts"),
    )
    experts_per_token = config_dict.get(
        "num_experts_per_tok",
        config_dict.get("num_selected_experts"),
    )
    if (
        not isinstance(expert_count, int)
        or expert_count <= 1
        or not isinstance(experts_per_token, int)
        or not 0 < experts_per_token <= expert_count
    ):
        raise RuntimeError("checkpoint does not expose a valid sparse MoE config")
    canonical_config = json.dumps(
        config_dict,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    identity: dict[str, Any] = {
        "source": model_source,
        "resolved": model,
        "source_kind": source_kind,
        "declared_kind": model_kind,
        "requested_revision": model_revision or None,
        "resolved_commit_hash": resolved_commit,
        "configuration_sha256": sha256_bytes(canonical_config),
        "configuration": {
            key: config_dict[key]
            for key in CONFIG_FIELDS
            if key in config_dict
        },
        "loom_fixture": fixture,
    }
    if source_kind == "local-checkpoint":
        identity["local_checkpoint"] = local_checkpoint_manifest(Path(model))
    fingerprint_payload = json.dumps(
        identity,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    identity["identity_sha256"] = sha256_bytes(fingerprint_payload)
    return identity, config


def make_prompts(
    case: BenchmarkCase,
    prompt_mode: str,
    tokenizer: Any | None,
    vocab_size: int,
) -> tuple[list[dict[str, list[int]]], str]:
    token_rows: list[list[int]] = []
    for batch_index in range(case.batch_size):
        if prompt_mode == "synthetic":
            token_range = min(max(vocab_size - 3, 1), 1000)
            tokens = [
                3 + ((batch_index * 37 + position * 19) % token_range)
                for position in range(case.input_len)
            ]
        else:
            if tokenizer is None:
                raise RuntimeError("natural prompts require a tokenizer")
            text = (
                NATURAL_PROMPT_TEMPLATES[
                    batch_index % len(NATURAL_PROMPT_TEMPLATES)
                ]
                + f"\nDeterministic request index: {batch_index}."
            )
            unit = list(tokenizer.encode(text, add_special_tokens=False))
            if not unit:
                raise RuntimeError("tokenizer returned an empty natural prompt")
            repetitions = (case.input_len + len(unit) - 1) // len(unit)
            tokens = (unit * repetitions)[: case.input_len]
        token_rows.append(tokens)
    serialized = json.dumps(
        token_rows,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        [{"prompt_token_ids": tokens} for tokens in token_rows],
        sha256_bytes(serialized),
    )


def request_metrics(outputs: list[Any]) -> dict[str, list[float]]:
    collected = {
        "ttft_ms": [],
        "tpot_ms": [],
        "e2e_ms": [],
        "queue_ms": [],
        "prefill_ms": [],
        "decode_ms": [],
    }
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            continue
        ttft_seconds = float(metrics.first_token_latency)
        first_token_ts = float(metrics.first_token_ts)
        last_token_ts = float(metrics.last_token_ts)
        queued_ts = float(metrics.queued_ts)
        scheduled_ts = float(metrics.scheduled_ts)
        decode_seconds = last_token_ts - first_token_ts
        generated = int(metrics.num_generation_tokens)
        if ttft_seconds > 0.0:
            collected["ttft_ms"].append(ttft_seconds * 1000.0)
        if first_token_ts > 0.0 and last_token_ts >= first_token_ts:
            collected["decode_ms"].append(decode_seconds * 1000.0)
            if generated > 1:
                collected["tpot_ms"].append(
                    decode_seconds * 1000.0 / (generated - 1)
                )
            if ttft_seconds > 0.0:
                collected["e2e_ms"].append(
                    (ttft_seconds + decode_seconds) * 1000.0
                )
        if queued_ts > 0.0 and scheduled_ts >= queued_ts:
            collected["queue_ms"].append(
                (scheduled_ts - queued_ts) * 1000.0
            )
        if scheduled_ts > 0.0 and first_token_ts >= scheduled_ts:
            collected["prefill_ms"].append(
                (first_token_ts - scheduled_ts) * 1000.0
            )
    return collected


def run_case(
    engine: Any,
    sampling_type: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
    tokenizer: Any | None,
    vocab_size: int,
    launch_count_fn: Callable[[Any], int],
    permute_operator: Any,
    combine_operator: Any,
    metadata_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    import torch

    workload, prompt_fingerprint = make_prompts(
        case,
        args.prompt_mode,
        tokenizer,
        vocab_size,
    )
    sampling = sampling_type(
        temperature=0.0,
        max_tokens=case.output_len,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        engine.generate(workload, sampling, use_tqdm=False)
    torch.cuda.synchronize()

    metadata_before = metadata_fn()
    permute_before = launch_count_fn(permute_operator)
    combine_before = launch_count_fn(combine_operator)
    torch.cuda.reset_peak_memory_stats()
    steady_allocated = torch.cuda.memory_allocated()
    steady_reserved = torch.cuda.memory_reserved()
    free_before, total_memory = torch.cuda.mem_get_info()

    latency_ms: list[float] = []
    output_throughput: list[float] = []
    total_throughput: list[float] = []
    collected_metrics: dict[str, list[float]] = {
        "ttft_ms": [],
        "tpot_ms": [],
        "e2e_ms": [],
        "queue_ms": [],
        "prefill_ms": [],
        "decode_ms": [],
    }
    replay_tokens: list[list[int]] | None = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(workload, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)
        output_throughput.append(
            case.batch_size * case.output_len / (elapsed_ms / 1000.0)
        )
        total_throughput.append(
            case.batch_size
            * (case.input_len + case.output_len)
            / (elapsed_ms / 1000.0)
        )
        metrics = request_metrics(outputs)
        for name, values in metrics.items():
            collected_metrics[name].extend(values)
        token_ids = [
            list(request.outputs[0].token_ids) for request in outputs
        ]
        if any(len(tokens) != case.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")
        if replay_tokens is None:
            replay_tokens = token_ids
        elif replay_tokens != token_ids:
            raise RuntimeError(
                f"{args.internal_provider} changed greedy output across repeats "
                f"for {case.label}"
            )

    free_after, _ = torch.cuda.mem_get_info()
    metadata_after = metadata_fn()
    permute_launches = launch_count_fn(permute_operator) - permute_before
    combine_launches = launch_count_fn(combine_operator) - combine_before
    permute_hits = (
        metadata_after["moe_movement_permute_hits"]
        - metadata_before["moe_movement_permute_hits"]
    )
    combine_hits = (
        metadata_after["moe_movement_combine_hits"]
        - metadata_before["moe_movement_combine_hits"]
    )
    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "prompt_mode": args.prompt_mode,
        "prompt_token_ids_sha256": prompt_fingerprint,
        "batch_latency_ms": summary(latency_ms),
        "request_ttft_ms": summary(collected_metrics["ttft_ms"]),
        "request_tpot_ms": summary(collected_metrics["tpot_ms"]),
        "request_e2e_ms": summary(collected_metrics["e2e_ms"]),
        "request_queue_ms": summary(collected_metrics["queue_ms"]),
        "request_prefill_ms": summary(collected_metrics["prefill_ms"]),
        "request_decode_ms": summary(collected_metrics["decode_ms"]),
        "output_tokens_per_second": summary(output_throughput),
        "total_tokens_per_second": summary(total_throughput),
        "token_ids": replay_tokens,
        "movement": {
            "measured_permute_host_launches": permute_launches,
            "measured_combine_host_launches": combine_launches,
            "measured_permute_adapter_hits": permute_hits,
            "measured_combine_adapter_hits": combine_hits,
        },
        "cuda_memory": {
            "device_total_bytes": total_memory,
            "steady_allocated_bytes": steady_allocated,
            "steady_reserved_bytes": steady_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_before_measured_bytes": free_before,
            "free_after_measured_bytes": free_after,
        },
    }


def prepare_environment(cache_root: Path) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in LOOM_ENGINE_OVERRIDE_ENVIRONMENTS:
        os.environ[name] = "0"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    required = [
        str(Path(sys.executable).absolute().parent),
        str(cuda_home / "bin"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required if entry not in current_entries]
        + current_entries
    )


def callable_identity(function: Any) -> dict[str, Any]:
    return {
        "module": getattr(function, "__module__", None),
        "qualified_name": getattr(function, "__qualname__", None),
        "name": getattr(function, "__name__", None),
    }


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None and args.internal_cache_root is not None
    prepare_environment(args.internal_cache_root.resolve())

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe

    from loom_kernels import native_build_info
    from loom_kernels.torch_ops import (
        Operator,
        bridge_abi_version,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import (
        MOE_MOVEMENT_OVERRIDE_KEY,
        provider_metadata,
        register_vllm_moe_movement,
    )

    vendor_grouped_gemm = cutlass_moe.ops.cutlass_moe_mm
    vendor_identity = callable_identity(vendor_grouped_gemm)
    explicit_registration = None
    if provider == "loom":
        os.environ["LOOM_KERNELS_ENABLE_MOE_MOVEMENT"] = "1"
        explicit_registration = register_vllm_moe_movement()
        if explicit_registration != MOE_MOVEMENT_OVERRIDE_KEY:
            raise RuntimeError("Loom MoE movement registration failed")
    vendor_grouped_gemm_unchanged = (
        cutlass_moe.ops.cutlass_moe_mm is vendor_grouped_gemm
    )
    if not vendor_grouped_gemm_unchanged:
        raise RuntimeError("Loom changed vLLM's vendor grouped GEMM callable")

    native_build = native_build_info()
    if args.model_kind == "production" and native_build is None:
        raise RuntimeError("production qualification requires a native wheel")

    model, source_kind = resolve_model(args.model)
    identity, config = model_identity(
        model,
        args.model,
        args.model_kind,
        args.model_revision,
    )
    tokenizer = None
    if args.prompt_mode == "natural":
        tokenizer_arguments: dict[str, Any] = {
            "local_files_only": source_kind == "local-checkpoint"
        }
        if args.model_revision and source_kind == "huggingface":
            tokenizer_arguments["revision"] = args.model_revision
        tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_arguments)

    max_model_len = max(
        case.input_len + case.output_len for case in args.cases
    )
    max_num_seqs = max(case.batch_size for case in args.cases)
    max_num_batched_tokens = max(
        max_model_len,
        max(case.batch_size * case.input_len for case in args.cases),
    )
    engine_arguments: dict[str, Any] = {
        "model": model,
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "quantization": "fp8_per_channel",
        "moe_backend": "cutlass",
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "seed": args.seed,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
    }
    if args.model_revision and source_kind == "huggingface":
        engine_arguments["revision"] = args.model_revision
        engine_arguments["tokenizer_revision"] = args.model_revision
    engine = LLM(**engine_arguments)

    permute_operator = Operator.MOE_PERMUTE
    combine_operator = Operator.MOE_COMBINE
    reset_launch_count(permute_operator)
    reset_launch_count(combine_operator)
    vocab_size = int(config.vocab_size)
    cases = [
        run_case(
            engine,
            SamplingParams,
            case,
            args,
            tokenizer,
            vocab_size,
            launch_count,
            permute_operator,
            combine_operator,
            provider_metadata,
        )
        for case in args.cases
    ]
    metadata = provider_metadata()
    report = {
        "provider": provider,
        "tested_revision": args.tested_revision,
        "model_identity": identity,
        "model_kind": args.model_kind,
        "prompt_mode": args.prompt_mode,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "engine_seed": args.seed,
        },
        "engine": {
            "dtype": "bfloat16",
            "quantization": "fp8_per_channel",
            "moe_backend": "cutlass",
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": False,
            "enforce_eager": args.enforce_eager,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
        "loom_path": {
            "explicit_registration": explicit_registration,
            "provider_metadata": metadata,
            "total_permute_host_launches": launch_count(permute_operator),
            "total_combine_host_launches": launch_count(combine_operator),
        },
        "vendor_grouped_gemm": {
            "owner": "vllm_vendor_backend",
            "callable": vendor_identity,
            "same_object_after_loom_registration": (
                vendor_grouped_gemm_unchanged
            ),
            "instrumented_or_wrapped_by_benchmark": False,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi": bridge_abi_version(),
            "native_build": native_build,
            "cuda_home": os.environ["CUDA_HOME"],
            "v1_multiprocessing": os.environ[
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ],
            "vllm_cache_root": os.environ["VLLM_CACHE_ROOT"],
        },
    }
    assert args.internal_result is not None
    args.internal_result.parent.mkdir(parents=True, exist_ok=True)
    args.internal_result.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    measured_permute = sum(
        case["movement"]["measured_permute_adapter_hits"] for case in cases
    )
    measured_combine = sum(
        case["movement"]["measured_combine_adapter_hits"] for case in cases
    )
    print(
        f"provider={provider} permute_hits={measured_permute} "
        f"combine_hits={measured_combine}",
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
        "--model-kind",
        args.model_kind,
        "--prompt-mode",
        args.prompt_mode,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--tested-revision",
        args.tested_revision,
        "--internal-provider",
        provider,
        "--internal-result",
        str(result),
        "--internal-cache-root",
        str(cache_root),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    for case in args.cases:
        command.extend(("--case", case.argument))
    return command


def ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def median_metric(case: dict[str, Any], name: str) -> float | None:
    metric = case[name]
    return None if metric is None else float(metric["median"])


def runtime_identity(provider: dict[str, Any]) -> dict[str, Any]:
    environment = provider["environment"]
    return {
        "engine": provider["engine"],
        "gpu": environment["gpu"],
        "compute_capability": environment["compute_capability"],
        "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
        "torch": environment["torch"],
        "torch_cuda": environment["torch_cuda"],
        "vllm": environment["vllm"],
        "bridge_abi": environment["bridge_abi"],
    }


def compare_round(
    label: str,
    provider_order: tuple[str, str],
    providers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = providers["vllm"]
    loom = providers["loom"]
    baseline_cases = {case["case"]: case for case in baseline["cases"]}
    loom_cases = {case["case"]: case for case in loom["cases"]}
    if baseline_cases.keys() != loom_cases.keys():
        raise RuntimeError("provider reports contain different cases")

    comparisons: list[dict[str, Any]] = []
    prompts_match = True
    tokens_match = True
    baseline_isolated = True
    loom_reached = True
    for case_label, native_case in baseline_cases.items():
        loom_case = loom_cases[case_label]
        case_prompts_match = (
            native_case["prompt_token_ids_sha256"]
            == loom_case["prompt_token_ids_sha256"]
        )
        case_tokens_match = native_case["token_ids"] == loom_case["token_ids"]
        prompts_match = prompts_match and case_prompts_match
        tokens_match = tokens_match and case_tokens_match
        native_movement = native_case["movement"]
        loom_movement = loom_case["movement"]
        case_baseline_isolated = all(
            value == 0 for value in native_movement.values()
        )
        case_loom_reached = all(value > 0 for value in loom_movement.values())
        baseline_isolated = baseline_isolated and case_baseline_isolated
        loom_reached = loom_reached and case_loom_reached
        native_latency = median_metric(native_case, "batch_latency_ms")
        loom_latency = median_metric(loom_case, "batch_latency_ms")
        native_tpot = median_metric(native_case, "request_tpot_ms")
        loom_tpot = median_metric(loom_case, "request_tpot_ms")
        native_throughput = median_metric(
            native_case,
            "output_tokens_per_second",
        )
        loom_throughput = median_metric(
            loom_case,
            "output_tokens_per_second",
        )
        comparisons.append(
            {
                "case": case_label,
                "prompt_token_ids_match": case_prompts_match,
                "token_ids_match": case_tokens_match,
                "baseline_movement_isolated": case_baseline_isolated,
                "loom_movement_reached": case_loom_reached,
                "native_over_loom_batch_latency": ratio(
                    native_latency,
                    loom_latency,
                ),
                "native_over_loom_tpot": ratio(native_tpot, loom_tpot),
                "loom_over_native_output_throughput": ratio(
                    loom_throughput,
                    native_throughput,
                ),
                "loom_minus_native_peak_allocated_bytes": (
                    loom_case["cuda_memory"]["peak_allocated_bytes"]
                    - native_case["cuda_memory"]["peak_allocated_bytes"]
                ),
                "performance_is_semantic_acceptance_gate": False,
            }
        )

    metadata = loom["loom_path"]["provider_metadata"]
    registered = (
        loom["loom_path"]["explicit_registration"] == "moe_movement"
        and metadata["moe_movement_override"]
    )
    no_rejection = metadata["moe_movement_first_rejection"] is None
    contract_observed = metadata["moe_movement_first_contract"] is not None
    vendor_unchanged = all(
        provider["vendor_grouped_gemm"][
            "same_object_after_loom_registration"
        ]
        for provider in providers.values()
    )
    model_identity_matches = (
        baseline["model_identity"]["identity_sha256"]
        == loom["model_identity"]["identity_sha256"]
    )
    native_build_matches = (
        baseline["environment"]["native_build"]
        == loom["environment"]["native_build"]
    )
    native_wheel_present = (
        baseline["model_kind"] != "production"
        or baseline["environment"]["native_build"] is not None
    )
    runtime_identity_matches = (
        runtime_identity(baseline) == runtime_identity(loom)
    )
    passed = all(
        (
            prompts_match,
            tokens_match,
            baseline_isolated,
            loom_reached,
            registered,
            no_rejection,
            contract_observed,
            vendor_unchanged,
            model_identity_matches,
            native_build_matches,
            native_wheel_present,
            runtime_identity_matches,
        )
    )
    return {
        "label": label,
        "provider_order": list(provider_order),
        "acceptance": {
            "passed": passed,
            "prompt_token_ids_match": prompts_match,
            "token_ids_match": tokens_match,
            "baseline_has_zero_loom_movement": baseline_isolated,
            "loom_movement_reached_in_every_case": loom_reached,
            "loom_registered": registered,
            "loom_contract_observed": contract_observed,
            "loom_has_no_contract_rejection": no_rejection,
            "vendor_grouped_gemm_unchanged": vendor_unchanged,
            "model_identity_matches": model_identity_matches,
            "native_build_matches": native_build_matches,
            "native_wheel_present_for_production": native_wheel_present,
            "runtime_identity_matches": runtime_identity_matches,
            "performance_is_acceptance_gate": False,
        },
        "comparisons": comparisons,
        "providers": providers,
    }


def performance_decisions(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [comparison["case"] for comparison in rounds[0]["comparisons"]]
    decisions: list[dict[str, Any]] = []
    for case_label in labels:
        order_speedups = {
            round_report["label"]: next(
                comparison["native_over_loom_batch_latency"]
                for comparison in round_report["comparisons"]
                if comparison["case"] == case_label
            )
            for round_report in rounds
        }
        measured = [
            speedup for speedup in order_speedups.values() if speedup is not None
        ]
        dual_order = len(rounds) == 2 and len(measured) == 2
        qualified = dual_order and all(speedup > 1.0 for speedup in measured)
        decisions.append(
            {
                "case": case_label,
                "order_speedups": order_speedups,
                "minimum_order_speedup": min(measured) if measured else None,
                "maximum_order_speedup": max(measured) if measured else None,
                "dual_order_model_latency_reduction": qualified,
            }
        )
    return decisions


def cross_round_outputs_match(rounds: list[dict[str, Any]]) -> bool:
    reference = round_reports_by_provider(rounds[0])
    for round_report in rounds[1:]:
        current = round_reports_by_provider(round_report)
        for provider in PROVIDERS:
            if reference[provider] != current[provider]:
                return False
    return True


def cross_round_execution_identity_matches(
    rounds: list[dict[str, Any]],
) -> bool:
    def identity(provider: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_identity_sha256": provider["model_identity"][
                "identity_sha256"
            ],
            "native_build": provider["environment"]["native_build"],
            "runtime": runtime_identity(provider),
        }

    reference = identity(rounds[0]["providers"]["vllm"])
    return all(
        identity(round_report["providers"][provider]) == reference
        for round_report in rounds
        for provider in PROVIDERS
    )


def round_reports_by_provider(
    round_report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        provider: [
            {
                "case": case["case"],
                "prompt_token_ids_sha256": case[
                    "prompt_token_ids_sha256"
                ],
                "token_ids": case["token_ids"],
            }
            for case in round_report["providers"][provider]["cases"]
        ]
        for provider in PROVIDERS
    }


def repository_state(repository: Path, tested_revision: str) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=repository,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        cwd=repository,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if head != tested_revision:
        raise RuntimeError(
            f"tested revision {tested_revision} does not match HEAD {head}"
        )
    if status:
        raise RuntimeError("formal MoE engine qualification requires a clean tree")
    return {"head": head, "clean": True}


def selected_orders(mode: str) -> list[tuple[str, tuple[str, str]]]:
    if mode == "both":
        return list(ORDER_MODES.items())
    return [(mode, ORDER_MODES[mode])]


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    git = repository_state(repository, args.tested_revision)
    round_reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="loom-vllm-moe-production-"
    ) as temporary:
        temporary_root = Path(temporary)
        for round_label, order in selected_orders(args.order_mode):
            providers: dict[str, dict[str, Any]] = {}
            for provider in order:
                run_root = temporary_root / round_label / provider
                result = run_root / "result.json"
                subprocess.run(
                    child_command(
                        args,
                        provider,
                        result,
                        run_root / "cache",
                    ),
                    check=True,
                    cwd=repository,
                )
                providers[provider] = json.loads(
                    result.read_text(encoding="utf-8")
                )
            round_reports.append(compare_round(round_label, order, providers))

    cross_round_tokens_match = cross_round_outputs_match(round_reports)
    cross_round_identity_matches = cross_round_execution_identity_matches(
        round_reports
    )
    decisions = performance_decisions(round_reports)
    semantic_and_path_passed = all(
        round_report["acceptance"]["passed"]
        for round_report in round_reports
    )
    dual_order_completed = len(round_reports) == 2
    production_speedup_qualified = (
        args.model_kind == "production"
        and semantic_and_path_passed
        and cross_round_tokens_match
        and cross_round_identity_matches
        and dual_order_completed
        and all(
            decision["dual_order_model_latency_reduction"]
            for decision in decisions
        )
    )
    sources = (
        "benchmarks/vllm_engine_moe_movement.py",
        "python/src/loom_kernels/vllm/moe.py",
        "crates/loom-cuda-sys/cuda/src/moe.cu",
    )
    report = {
        "schema_version": 2,
        "benchmark": "vllm_pretrained_moe_movement",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tested_revision": args.tested_revision,
        "repository": git,
        "source_sha256": {
            source: sha256_file(repository / source) for source in sources
        },
        "model": args.model,
        "model_revision": args.model_revision or None,
        "model_kind": args.model_kind,
        "prompt_mode": args.prompt_mode,
        "cases": [case.argument for case in args.cases],
        "order_mode": args.order_mode,
        "acceptance": {
            "semantic_and_path_passed": semantic_and_path_passed,
            "cross_round_prompts_and_tokens_exact": cross_round_tokens_match,
            "cross_round_execution_identity_matches": (
                cross_round_identity_matches
            ),
            "dual_order_completed": dual_order_completed,
            "production_speedup_qualified": production_speedup_qualified,
            "performance_is_semantic_acceptance_gate": False,
        },
        "performance_decisions": decisions,
        "rounds": round_reports,
        "claim_boundary": [
            (
                "The workload uses a pinned pretrained MoE checkpoint only "
                "when model_kind is production."
            ),
            "Loom replaces FP8 permutation and BF16 weighted combine only.",
            (
                "Cutlass grouped GEMM remains vLLM-owned and is not wrapped "
                "by the benchmark."
            ),
            (
                "A production speedup is qualified only when every declared "
                "case improves in both process orders."
            ),
            (
                "Serving concurrency and goodput are outside this offline "
                "LLM.generate gate."
            ),
        ],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if (
        not semantic_and_path_passed
        or not cross_round_tokens_match
        or not cross_round_identity_matches
    ):
        raise SystemExit("pretrained MoE movement semantic/path gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is None:
        run_controller(args)
    else:
        run_provider(args)


if __name__ == "__main__":
    main()
