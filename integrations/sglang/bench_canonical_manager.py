from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from checkpoint_identity import checkpoint_identity, sha256_file


INTEGRATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
ADAPTER_SOURCE_ROOT = INTEGRATION_ROOT / "src"
ADAPTER_PACKAGE_ROOT = ADAPTER_SOURCE_ROOT / "orbitkv_sglang"
SUPPORTED_SGLANG_RELEASE = "v0.5.17"
SUPPORTED_SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
MANAGER_ENTRYPOINT = "orbitkv_manager"
STOCK_PLUGIN_SENTINEL = "orbitkv_stock_baseline_no_plugins"
PAGE_TOKENS = 16
ATTENTION_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": "flashinfer",
    "GptOssForCausalLM": "fa3",
}
MOE_RUNNER_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": None,
    # The v0.5.17 auto route selects the external triton_kernel MXFP4 MoE
    # path on H20.  That path is not batch invariant.  The release's built-in
    # Triton runner fixes its tiling under deterministic inference and first
    # expands the checkpoint's MXFP4 expert weights to BF16.
    "GptOssForCausalLM": "triton",
}
SGLANG_LOADER_PATCH_PATH = "python/sglang/srt/plugins/__init__.py"
SGLANG_LOADER_BASE_GIT_BLOB = "00ae1acd18266765c006d87ba5eec51e9f113d8d"
SGLANG_LOADER_PATCHED_GIT_BLOB = "7c20ccb51e46942f0bbdfdbcaf88c3148939cb55"
SGLANG_LOADER_BASE_SHA256 = (
    "3a975a73f1a7887e68c81ea7a2530250597ac8ae978efc0b0f70f038a99a3164"
)
MANAGER_LOADER_PATCH_SHA256 = (
    "6de7acab246b299386b5d6557154a7aa237c8bdb498b8b885bed1c72e849745d"
)
MANAGER_LOADER_BLOB_SHA256 = (
    "1fc2e2472e8fd55f564826509b2afa1f8f0d86a4b2ee3a3986c3209e3c09c934"
)
SUPPORTED_ARCHITECTURES = ("Qwen2ForCausalLM", "GptOssForCausalLM")
QUALIFICATION_BATCH_SIZES = (1, 4)
PREFIX_SEED_BATCH_SIZE = 1
RECORD_SCHEMA = "orbitkv.sglang-v0517-prefix-cow-single-run.v6"
MANAGER_RADIX_CACHE_BACKEND = "orbitkv"
PAIR_IMPLEMENTATION_DIFFERENCE = {
    "field": "radix_cache_backend",
    "manager": {"present": True, "value": MANAGER_RADIX_CACHE_BACKEND},
    "stock": {"present": False, "value": None},
    "scope": "implementation selection only",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one official SGLang v0.5.17 Full/Hybrid qualification sample "
            "with either the canonical OrbitKV manager or the pristine stock "
            "allocator. Run this program once per mode and pair records only "
            "when their complete comparison contracts match."
        )
    )
    parser.add_argument("--mode", choices=("manager", "stock"), required=True)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", help="canonical KvPlanInput; manager mode only")
    parser.add_argument("--library", help="canonical OrbitKV cdylib; manager mode only")
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        required=True,
        help=(
            "explicit SGLang tensor-arena sizing cap; use the same value in "
            "the independent manager and stock runs"
        ),
    )
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument(
        "--attention-backend",
        choices=tuple(ATTENTION_BACKENDS_BY_ARCHITECTURE.values()),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def _regular_file(value: str, name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"invalid {name} {value!r}: {error}") from error
    if not path.is_file():
        raise ValueError(f"{name} must name a regular file")
    return path


def _directory(value: str, name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"invalid {name} {value!r}: {error}") from error
    if not path.is_dir():
        raise ValueError(f"{name} must name a directory")
    return path


def validate_arguments(args: argparse.Namespace) -> dict[str, Path | None]:
    for name in (
        "requests",
        "max_running_requests",
        "prompt_tokens",
        "decode_tokens",
        "iterations",
        "chunked_prefill_size",
        "context_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.requests not in QUALIFICATION_BATCH_SIZES:
        raise ValueError("--requests must be exactly 1 or 4 for ABI6 qualification")
    if args.decode_tokens != 33:
        raise ValueError(
            "--decode-tokens must be exactly 33 so measured requests do not "
            "publish a second page-aligned Prefix endpoint"
        )
    if (args.prompt_tokens + args.decode_tokens - 1) % PAGE_TOKENS == 0:
        raise ValueError("measured final KV boundary must fall between pages")
    if args.max_running_requests < args.requests:
        raise ValueError("--max-running-requests must cover the complete B1/B4 batch")
    prefix_seed_tokens = (args.prompt_tokens - 1) // PAGE_TOKENS * PAGE_TOKENS
    if prefix_seed_tokens < PAGE_TOKENS:
        raise ValueError(
            "--prompt-tokens must leave one page for the measured Prefix extend"
        )
    aligned_prompt_tokens = (
        (args.prompt_tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
    ) * PAGE_TOKENS
    if args.requests * aligned_prompt_tokens != args.chunked_prefill_size:
        raise ValueError(
            "--chunked-prefill-size must equal the page-aligned complete "
            "B1/B4 prompt batch"
        )
    if args.chunked_prefill_size % PAGE_TOKENS:
        raise ValueError("--chunked-prefill-size must be divisible by 16")
    if args.prompt_tokens + args.decode_tokens >= args.context_length:
        raise ValueError(
            "prompt plus decode tokens must leave one context slot unused"
        )
    if args.attention_backend not in ATTENTION_BACKENDS_BY_ARCHITECTURE.values():
        raise ValueError("--attention-backend must be flashinfer or fa3")
    if args.mem_fraction_static is not None and not (
        0.0 < args.mem_fraction_static <= 1.0
    ):
        raise ValueError("--mem-fraction-static must be in (0, 1]")
    if (
        isinstance(args.max_total_tokens, bool)
        or not isinstance(args.max_total_tokens, int)
        or args.max_total_tokens <= 0
    ):
        raise ValueError("--max-total-tokens must be positive")
    if args.max_total_tokens % PAGE_TOKENS:
        raise ValueError("--max-total-tokens must be divisible by 16")
    required_batch_tokens = args.requests * (
        args.prompt_tokens + args.decode_tokens
    )
    if required_batch_tokens > args.max_total_tokens:
        raise ValueError(
            "--max-total-tokens must hold the complete B1/B4 qualification batch"
        )

    sglang_root = _directory(args.sglang_root, "--sglang-root")
    if not (sglang_root / "python/sglang/__init__.py").is_file():
        raise ValueError("--sglang-root is not an SGLang source checkout")
    model = _directory(args.model, "--model")
    _regular_file(str(model / "config.json"), "checkpoint config")

    plan: Path | None = None
    library: Path | None = None
    if args.mode == "manager":
        if not args.plan or not args.library:
            raise ValueError("manager mode requires --plan and --library")
        plan = _regular_file(args.plan, "--plan")
        library = _regular_file(args.library, "--library")
    else:
        if args.plan is not None or args.library is not None:
            raise ValueError("stock mode forbids --plan and --library")

    return {
        "sglang_root": sglang_root,
        "model": model,
        "plan": plan,
        "library": library,
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot verify source checkout {root}") from error


def verify_sglang_source(root: Path, mode: str) -> dict[str, Any]:
    from orbitkv_sglang.pinned import (
        validate_base_checkout,
        validate_patched_checkout,
    )

    if mode == "manager":
        validate_patched_checkout(root)
    elif mode == "stock":
        validate_base_checkout(root)
    else:
        raise RuntimeError(f"unknown source verification mode: {mode}")
    revision = _git(root, "rev-parse", "HEAD").strip()
    if revision != SUPPORTED_SGLANG_REVISION:
        raise RuntimeError(
            f"SGLang revision {revision!r} is not {SUPPORTED_SGLANG_REVISION}"
        )
    dirty = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "python/sglang",
    )
    dirty_paths = {
        line[3:]
        for line in dirty.splitlines()
        if len(line) >= 4 and line[3:]
    }
    loader = root / SGLANG_LOADER_PATCH_PATH
    loader_bytes = loader.read_bytes()
    loader_blob_sha256 = hashlib.sha256(loader_bytes).hexdigest()
    loader_head_blob = _git(
        root, "rev-parse", f"HEAD:{SGLANG_LOADER_PATCH_PATH}"
    ).strip()
    loader_worktree_blob = _git(
        root, "hash-object", SGLANG_LOADER_PATCH_PATH
    ).strip()
    patch = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--full-index",
            "--binary",
            "--",
            SGLANG_LOADER_PATCH_PATH,
        ],
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    if mode == "stock":
        if dirty_paths:
            raise RuntimeError("stock mode requires a clean pinned SGLang source tree")
        if loader_head_blob != SGLANG_LOADER_BASE_GIT_BLOB:
            raise RuntimeError("stock SGLang loader HEAD blob is not canonical")
        if loader_worktree_blob != SGLANG_LOADER_BASE_GIT_BLOB:
            raise RuntimeError("stock SGLang loader worktree blob is not canonical")
        if loader_blob_sha256 != SGLANG_LOADER_BASE_SHA256:
            raise RuntimeError("stock SGLang loader SHA-256 is not canonical")
        patch_contract = "clean_pinned_head"
    elif mode == "manager":
        if dirty_paths != {SGLANG_LOADER_PATCH_PATH}:
            raise RuntimeError(
                "manager mode requires exactly the fail-closed SGLang loader patch"
            )
        if patch_sha256 != MANAGER_LOADER_PATCH_SHA256:
            raise RuntimeError("SGLang loader patch SHA-256 is not canonical")
        if loader_blob_sha256 != MANAGER_LOADER_BLOB_SHA256:
            raise RuntimeError("SGLang loader worktree blob SHA-256 is not canonical")
        if loader_head_blob != SGLANG_LOADER_BASE_GIT_BLOB:
            raise RuntimeError("manager SGLang loader HEAD blob is not canonical")
        if loader_worktree_blob != SGLANG_LOADER_PATCHED_GIT_BLOB:
            raise RuntimeError("manager SGLang loader patched blob is not canonical")
        patch_contract = "pinned_head_plus_canonical_loader_patch"
    else:
        raise RuntimeError(f"unknown source verification mode: {mode}")
    return {
        "root": str(root),
        "release": SUPPORTED_SGLANG_RELEASE,
        "revision": revision,
        "python_source_contract": patch_contract,
        "dirty_paths": sorted(dirty_paths),
        "loader": {
            "path": SGLANG_LOADER_PATCH_PATH,
            "head_git_blob": loader_head_blob,
            "worktree_git_blob": loader_worktree_blob,
            "worktree_sha256": loader_blob_sha256,
            "patch_sha256": patch_sha256,
        },
    }


def configure_environment(
    args: argparse.Namespace, paths: dict[str, Path | None]
) -> dict[str, str]:
    if any(name == "sglang" or name.startswith("sglang.") for name in sys.modules):
        raise RuntimeError("SGLang was imported before the benchmark environment froze")
    for name in tuple(os.environ):
        if name.startswith("ORBITKV_"):
            os.environ.pop(name)

    sglang_python = paths["sglang_root"] / "python"  # type: ignore[operator]
    python_paths = [str(sglang_python), str(ADAPTER_SOURCE_ROOT)]
    previous_python_path = os.environ.get("PYTHONPATH")
    if previous_python_path:
        python_paths.append(previous_python_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)
    for path in reversed((sglang_python, ADAPTER_SOURCE_ROOT)):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    if args.mode == "manager":
        environment = {
            "SGLANG_PLUGINS": MANAGER_ENTRYPOINT,
            "SGLANG_USE_HND_KVCACHE": "0",
            "ORBITKV_PLAN": str(paths["plan"]),
            "ORBITKV_LIBRARY": str(paths["library"]),
            "ORBITKV_SGLANG_ROOT": str(paths["sglang_root"]),
        }
    else:
        environment = {
            "SGLANG_PLUGINS": STOCK_PLUGIN_SENTINEL,
            "SGLANG_USE_HND_KVCACHE": "0",
        }
    environment.update(
        SGLANG_EXPERIMENTAL_CPP_RADIX_TREE="0",
        SGLANG_ENABLE_UNIFIED_RADIX_TREE="0",
        SGLANG_RADIX_FORCE_MISS="0",
    )
    os.environ.update(environment)
    environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
    environment["PATH"] = os.environ.get("PATH", "")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return environment


def build_tool_identity() -> dict[str, str]:
    executable = shutil.which("ninja")
    if executable is None:
        raise RuntimeError("the pinned FlashInfer path requires ninja on PATH")
    path = Path(executable).resolve(strict=True)
    expected_directory = Path(sys.executable).absolute().parent
    if path.parent != expected_directory:
        raise RuntimeError("ninja must come from the active Python environment")
    try:
        version = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot execute the pinned ninja build tool") from error
    if not version:
        raise RuntimeError("ninja returned an empty version")
    return {"path": str(path), "version": version, "sha256": sha256_file(path)}


def _adapter_identity() -> dict[str, Any]:
    files = [
        INTEGRATION_ROOT / "pyproject.toml",
        INTEGRATION_ROOT / "prepare_pinned_checkout.py",
        INTEGRATION_ROOT
        / "patches/v0.5.17-orbitkv-fail-closed.patch",
        *sorted(ADAPTER_PACKAGE_ROOT.rglob("*.py")),
    ]
    return {
        "files": [
            {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    }


def verify_manager_entrypoint() -> dict[str, Any]:
    matches = [
        entry
        for entry in importlib.metadata.entry_points(group="sglang.srt.plugins")
        if entry.name == MANAGER_ENTRYPOINT
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "exactly one installed orbitkv_manager entry point is required"
        )
    entry = matches[0]
    if entry.value != "orbitkv_sglang.plugin:register":
        raise RuntimeError(
            "orbitkv_manager entry point does not target the canonical adapter"
        )
    spec = importlib.util.find_spec("orbitkv_sglang.plugin")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot resolve the canonical adapter module")
    origin = Path(spec.origin).resolve(strict=True)
    if not origin.is_relative_to(ADAPTER_PACKAGE_ROOT.resolve(strict=True)):
        raise RuntimeError(
            "installed orbitkv_manager does not resolve to this checkout"
        )
    return {
        "name": entry.name,
        "value": entry.value,
        "distribution": entry.dist.name if entry.dist else None,
        "distribution_version": entry.dist.version if entry.dist else None,
        "module": str(origin),
    }


def verify_stock_plugin_selection() -> dict[str, Any]:
    matches = [
        entry
        for entry in importlib.metadata.entry_points(group="sglang.srt.plugins")
        if entry.name == STOCK_PLUGIN_SENTINEL
    ]
    if matches:
        raise RuntimeError("stock no-plugin sentinel unexpectedly resolves to a plugin")
    return {
        "selection": STOCK_PLUGIN_SENTINEL,
        "matching_entrypoints": 0,
        "contract": "pristine_loader_whitelist_selects_no_installed_plugin",
    }


def verify_pinned_module_constants() -> dict[str, str]:
    from orbitkv_sglang import pinned

    expected = {
        "release": SUPPORTED_SGLANG_RELEASE,
        "revision": SUPPORTED_SGLANG_REVISION,
        "loader_path": SGLANG_LOADER_PATCH_PATH,
        "base_source_sha256": SGLANG_LOADER_BASE_SHA256,
        "patched_source_sha256": MANAGER_LOADER_BLOB_SHA256,
        "patch_diff_sha256": MANAGER_LOADER_PATCH_SHA256,
    }
    actual = {
        "release": pinned.SUPPORTED_SGLANG_RELEASE,
        "revision": pinned.SUPPORTED_SGLANG_REVISION,
        "loader_path": pinned.PATCHED_SOURCE_PATH,
        "base_source_sha256": pinned.BASE_SOURCE_SHA256,
        "patched_source_sha256": pinned.PATCHED_SOURCE_SHA256,
        "patch_diff_sha256": pinned.PATCH_DIFF_SHA256,
    }
    if actual != expected:
        raise RuntimeError("runner and canonical pinned-source contract differ")
    return actual


def _positive_checkpoint_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"checkpoint has invalid {name}")
    return value


def _validate_plan_attention(
    manager_config: Any, expected_classes: Sequence[dict[str, Any]], layers: int
) -> None:
    if manager_config.page_tokens != PAGE_TOKENS:
        raise RuntimeError("manager plan does not use page_tokens=16")
    if manager_config.num_hidden_layers != layers:
        raise RuntimeError("manager plan layer count differs from the checkpoint")
    if len(manager_config.classes) != len(expected_classes):
        raise RuntimeError("manager plan attention classes differ from the checkpoint")
    for actual, expected in zip(
        manager_config.classes, expected_classes, strict=True
    ):
        fields = {
            "name": actual.name,
            "retention": actual.retention,
            "layers": list(actual.layers),
            "window_tokens": actual.window_tokens,
        }
        if fields != expected:
            raise RuntimeError(
                "manager plan attention class differs from the checkpoint: "
                f"expected={expected} actual={fields}"
            )


def checkpoint_contract(
    model: Path, manager_config: Any | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read checkpoint config") from error
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint config must be a JSON object")
    architectures = config.get("architectures")
    if architectures not in ([SUPPORTED_ARCHITECTURES[0]], [SUPPORTED_ARCHITECTURES[1]]):
        raise RuntimeError(
            "qualification supports only Qwen2ForCausalLM or GptOssForCausalLM"
        )
    architecture = architectures[0]
    num_hidden_layers = _positive_checkpoint_int(config, "num_hidden_layers")
    vocab_size = _positive_checkpoint_int(config, "vocab_size")
    max_position_embeddings = _positive_checkpoint_int(
        config, "max_position_embeddings"
    )

    if architecture == "Qwen2ForCausalLM":
        if config.get("use_sliding_window") is not False:
            raise RuntimeError(
                "Qwen2 qualification requires use_sliding_window=false"
            )
        if "layer_types" in config:
            raise RuntimeError("Qwen2 Full qualification forbids layer_types")
        expected_classes = [
            {
                "name": "full",
                "retention": "full",
                "layers": list(range(num_hidden_layers)),
                "window_tokens": None,
            }
        ]
        sliding_window = None
        profile = "full"
    else:
        raw_layer_types = config.get("layer_types")
        if not isinstance(raw_layer_types, list) or len(raw_layer_types) != num_hidden_layers:
            raise RuntimeError(
                "GptOss qualification requires one explicit layer_type per layer"
            )
        allowed = {"full_attention", "sliding_attention"}
        if any(value not in allowed for value in raw_layer_types):
            raise RuntimeError("GptOss checkpoint has an unsupported layer_type")
        full_layers = [
            index
            for index, value in enumerate(raw_layer_types)
            if value == "full_attention"
        ]
        sliding_layers = [
            index
            for index, value in enumerate(raw_layer_types)
            if value == "sliding_attention"
        ]
        if not full_layers or not sliding_layers:
            raise RuntimeError("GptOss qualification requires both Full and SWA layers")
        sliding_window = _positive_checkpoint_int(config, "sliding_window")
        expected_classes = [
            {
                "name": "full",
                "retention": "full",
                "layers": full_layers,
                "window_tokens": None,
            },
            {
                "name": "swa",
                "retention": "sliding",
                "layers": sliding_layers,
                "window_tokens": sliding_window,
            },
        ]
        profile = "hybrid_full_swa"

    if manager_config is not None:
        _validate_plan_attention(
            manager_config, expected_classes, num_hidden_layers
        )

    values = {
        "architecture": architecture,
        "attention_profile": profile,
        "attention_backend": ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture],
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "sliding_window": sliding_window,
        "classes": expected_classes,
    }
    identity = checkpoint_identity(model, "auto")
    if identity["weight_bytes"] <= 0 or not identity["indexed_weights_complete"]:
        raise RuntimeError("checkpoint weights are missing or incomplete")
    return values, identity


def artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def manager_plan_identity(config: Any) -> dict[str, Any]:
    return {
        "artifact": artifact_identity(config.plan_path),
        "plan_fingerprint": config.plan_fingerprint,
        "page_tokens": config.page_tokens,
        "classes": [
            {
                "class_id": item.class_id,
                "pool_id": item.pool_id,
                "backend_domain": item.backend_domain,
                "name": item.name,
                "retention": item.retention,
                "layers": list(item.layers),
                "bytes_per_token_per_layer": item.bytes_per_token_per_layer,
                "window_tokens": item.window_tokens,
                "period_blocks": item.period_blocks,
            }
            for item in config.classes
        ],
    }


def _required_attention_backend(contract: dict[str, Any]) -> str:
    architecture = contract.get("architecture")
    try:
        expected = ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "checkpoint contract has an unsupported architecture"
        ) from error
    if contract.get("attention_backend") != expected:
        raise RuntimeError("checkpoint attention backend contract is inconsistent")
    return expected


def _required_moe_runner_backend(contract: dict[str, Any]) -> str | None:
    architecture = contract.get("architecture")
    try:
        return MOE_RUNNER_BACKENDS_BY_ARCHITECTURE[architecture]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "checkpoint contract has an unsupported architecture"
        ) from error


def engine_arguments(
    args: argparse.Namespace, model: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    attention_backend = _required_attention_backend(contract)
    if args.attention_backend != attention_backend:
        raise RuntimeError(
            f"{contract['architecture']} requires --attention-backend "
            f"{attention_backend}"
        )
    values: dict[str, Any] = {
        "model_path": str(model),
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": args.context_length,
        "page_size": PAGE_TOKENS,
        "attention_backend": attention_backend,
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": False,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": args.chunked_prefill_size,
        "max_running_requests": args.max_running_requests,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "enable_dp_attention": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
        "enable_hierarchical_cache": False,
        "enable_streaming_session": False,
        "enable_unified_memory": False,
        "enable_pdmux": False,
        "enable_lmcache": False,
        "enable_flexkv": False,
        "enable_session_radix_cache": False,
        "enable_hisparse": False,
        "enable_page_major_kv_layout": False,
        "random_seed": args.seed,
        "log_level": "error",
        "max_total_tokens": args.max_total_tokens,
    }
    moe_runner_backend = _required_moe_runner_backend(contract)
    if moe_runner_backend is not None:
        values["moe_runner_backend"] = moe_runner_backend
    if args.mem_fraction_static is not None:
        values["mem_fraction_static"] = args.mem_fraction_static
    if args.mode == "manager":
        values["radix_cache_backend"] = MANAGER_RADIX_CACHE_BACKEND
    elif args.mode != "stock":
        raise RuntimeError(f"unknown qualification mode: {args.mode}")
    return values


def pair_engine_arguments(
    mode: str, engine_args: dict[str, Any]
) -> dict[str, Any]:
    """Normalize only the declared implementation-selection difference."""

    values = dict(engine_args)
    if mode == "manager":
        if values.pop("radix_cache_backend", None) != MANAGER_RADIX_CACHE_BACKEND:
            raise RuntimeError(
                "manager pair contract requires radix_cache_backend='orbitkv'"
            )
    elif mode == "stock":
        if "radix_cache_backend" in values:
            raise RuntimeError(
                "stock pair contract must omit radix_cache_backend"
            )
    else:
        raise RuntimeError(f"unknown qualification mode: {mode}")
    return values


def expected_prefix_cache(mode: str, checkpoint: dict[str, Any]) -> str:
    if mode == "manager":
        return "OrbitKvPrefixCache"
    if mode != "stock":
        raise RuntimeError(f"unknown qualification mode: {mode}")
    if checkpoint.get("attention_profile") == "full":
        return "RadixCache"
    if checkpoint.get("attention_profile") == "hybrid_full_swa":
        return "UnifiedRadixCache"
    raise RuntimeError("checkpoint has an unsupported Prefix-cache profile")


def deterministic_input_ids(
    *, requests: int, prompt_tokens: int, vocab_size: int, seed: int
) -> list[list[int]]:
    if vocab_size - 3 < requests:
        raise RuntimeError("checkpoint vocabulary is too small")
    shared_count = (prompt_tokens - 1) // PAGE_TOKENS * PAGE_TOKENS
    shared_material = hashlib.shake_256(
        f"orbitkv-prefix-shared-v1:{seed}".encode("ascii")
    ).digest(shared_count * 4)
    shared = [
        3
        + int.from_bytes(shared_material[offset : offset + 4], "little")
        % (vocab_size - 3)
        for offset in range(0, len(shared_material), 4)
    ]
    result: list[list[int]] = []
    for request in range(requests):
        material = hashlib.shake_256(
            f"orbitkv-canonical-v1:{seed}:{request}".encode("ascii")
        ).digest((prompt_tokens - shared_count) * 4)
        suffix = [
            3
            + int.from_bytes(material[offset : offset + 4], "little")
            % (vocab_size - 3)
            for offset in range(0, len(material), 4)
        ]
        suffix[0] = 3 + (seed + request) % (vocab_size - 3)
        result.append(shared + suffix)
    return result


def token_digest(outputs: Sequence[Sequence[dict[str, Any]]]) -> str:
    payload = [
        [output["output_ids"] for output in iteration]
        for iteration in outputs
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def input_digest(inputs: Sequence[Sequence[int]]) -> str:
    encoded = json.dumps(inputs, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def request_token_digests(
    outputs: Sequence[Sequence[dict[str, Any]]],
) -> list[list[str]]:
    return [
        [canonical_digest(output["output_ids"]) for output in iteration]
        for iteration in outputs
    ]


def request_traces(
    *,
    outputs: Sequence[Sequence[dict[str, Any]]],
    submitted_rids: Sequence[Sequence[str]],
    submitted_input_digests: Sequence[Sequence[str]],
) -> list[list[dict[str, Any]]]:
    if not (
        len(outputs) == len(submitted_rids) == len(submitted_input_digests)
    ):
        raise RuntimeError("request trace iteration cardinality is inconsistent")
    traces: list[list[dict[str, Any]]] = []
    for iteration_outputs, iteration_rids, iteration_inputs in zip(
        outputs, submitted_rids, submitted_input_digests, strict=True
    ):
        if not (
            len(iteration_outputs) == len(iteration_rids) == len(iteration_inputs)
        ):
            raise RuntimeError("request trace batch cardinality is inconsistent")
        row: list[dict[str, Any]] = []
        for request_index, (output, rid, input_sha256) in enumerate(
            zip(
                iteration_outputs,
                iteration_rids,
                iteration_inputs,
                strict=True,
            )
        ):
            ids = list(output["output_ids"])
            meta_info = output.get("meta_info")
            if not isinstance(meta_info, dict) or meta_info.get("id") != rid:
                raise RuntimeError("SGLang returned a foreign request id")
            row.append(
                {
                    "request_index": request_index,
                    "submitted_rid": rid,
                    "submitted_input_ids_sha256": input_sha256,
                    "returned_rid": meta_info["id"],
                    "output_ids": ids,
                    "output_ids_sha256": canonical_digest(ids),
                }
            )
        traces.append(row)
    return traces


def verify_request_trace_stability(
    traces: Sequence[Sequence[dict[str, Any]]],
) -> None:
    if not traces:
        raise RuntimeError("qualification produced no request traces")
    width = len(traces[0])
    if width <= 0 or any(len(row) != width for row in traces):
        raise RuntimeError("request trace matrix is not rectangular")
    for request_index in range(width):
        expected = traces[0][request_index]["output_ids"]
        if any(row[request_index]["output_ids"] != expected for row in traces[1:]):
            raise RuntimeError(
                "deterministic inference changed output tokens across iterations "
                f"for request index {request_index}"
            )


def gpu_snapshot(label: str) -> dict[str, Any]:
    fields = (
        "index",
        "name",
        "uuid",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    )
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("nvidia-smi GPU snapshot failed") from error
    rows = []
    for values in csv.reader(output.splitlines()):
        if len(values) != len(fields):
            raise RuntimeError("nvidia-smi returned an unexpected row")
        rows.append(
            {
                field: value.strip()
                for field, value in zip(fields, values, strict=True)
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return {"label": label, "time_ns": time.time_ns(), "gpus": rows}


def _state(info: dict[str, Any]) -> dict[str, Any]:
    states = info.get("internal_states", [])
    if isinstance(states, list) and states and isinstance(states[0], dict):
        return states[0]
    if isinstance(states, dict):
        return states
    return {}


def server_memory(info: dict[str, Any]) -> dict[str, Any]:
    memory = _state(info).get("memory_usage")
    if not isinstance(memory, dict):
        raise RuntimeError("SGLang did not report server memory")
    return memory


def verify_runtime_contract(
    args: argparse.Namespace,
    info: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    state = _state(info)
    attention_backend = _required_attention_backend(checkpoint)
    required = {
        "page_size": PAGE_TOKENS,
        "max_total_tokens": args.max_total_tokens,
        "attention_backend": attention_backend,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "chunked_prefill_size": args.chunked_prefill_size,
        "max_running_requests": args.max_running_requests,
        "disable_overlap_schedule": True,
        "disable_radix_cache": False,
        "disable_cuda_graph": True,
        "disable_hybrid_swa_memory": False,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "enable_page_major_kv_layout": False,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "enable_dp_attention": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
        "enable_hierarchical_cache": False,
        "enable_streaming_session": False,
        "enable_unified_memory": False,
        "enable_pdmux": False,
        "enable_lmcache": False,
        "enable_flexkv": False,
        "enable_session_radix_cache": False,
        "enable_hisparse": False,
    }
    if "radix_cache_backend" not in state:
        raise RuntimeError("resolved SGLang contract omitted radix_cache_backend")
    required["radix_cache_backend"] = (
        MANAGER_RADIX_CACHE_BACKEND if args.mode == "manager" else None
    )
    moe_runner_backend = _required_moe_runner_backend(checkpoint)
    if moe_runner_backend is not None:
        required["moe_runner_backend"] = moe_runner_backend
    mismatches = {
        name: {"expected": expected, "actual": state.get(name)}
        for name, expected in required.items()
        if state.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"resolved SGLang contract mismatch: {mismatches}")
    memory = server_memory(info)
    full_capacity = memory.get("token_capacity")
    if (
        isinstance(full_capacity, bool)
        or not isinstance(full_capacity, int)
        or full_capacity != args.max_total_tokens
        or full_capacity % PAGE_TOKENS
    ):
        raise RuntimeError(
            "resolved Full capacity differs from the explicit same-cap parameter"
        )
    swa_capacity = memory.get("token_capacity_swa")
    if checkpoint["attention_profile"] == "full":
        if swa_capacity is not None:
            raise RuntimeError("Full-only execution unexpectedly exposed an SWA arena")
    elif (
        isinstance(swa_capacity, bool)
        or not isinstance(swa_capacity, int)
        or swa_capacity <= 0
        or swa_capacity % PAGE_TOKENS
    ):
        raise RuntimeError("Hybrid execution did not expose a page-aligned SWA capacity")
    return {
        "requested_max_total_tokens": args.max_total_tokens,
        "full_tokens": full_capacity,
        "swa_tokens": swa_capacity,
        "unit": "tokens_per_attention_class",
        "interpretation": (
            "SGLang tensor-arena sizing only; this single-run record makes no "
            "KV compression or memory-saving claim"
        ),
    }


_IDENTITY_FIELDS = (
    "engine_epoch",
    "pool_epoch",
    "pool_id",
    "class_id",
    "backend_domain",
    "page_count",
    "page_tokens",
    "backend_base_index",
    "first_page_id",
)
_ARENA_PHASE_FIELDS = (
    "free_pages",
    "reserved_pages",
    "writing_pages",
    "active_pages",
    "retiring_pages",
    "quarantined_pages",
    "exhausted_pages",
)
_ARENA_STATS_FIELDS = (
    "engine_epoch",
    "pool_epoch",
    "pool_id",
    "page_count",
    "class_id",
    "backend_domain",
    "first_page_id",
    *_ARENA_PHASE_FIELDS,
    "request_page_refs",
    "prefix_page_refs",
    "reader_pins",
)
_MANAGER_STATS_FIELDS = (
    "active_requests",
    "active_snapshots",
    "active_prefixes",
    "evicted_prefixes",
    "prepared_steps",
    "submitted_steps",
    *_ARENA_PHASE_FIELDS,
    "pending_reclamations",
    "total_request_page_refs",
    "total_prefix_page_refs",
    "total_reader_pins",
)
_BATCH_COUNTER_FIELDS = (
    "request_acquire_batch_calls",
    "request_fork_batch_calls",
    "prepare_batch_calls",
    "submit_batch_calls",
    "complete_batch_calls",
    "abort_steps_batch_calls",
    "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls",
    "release_batch_calls",
    "acknowledge_reclamations_batch_calls",
    "recycle_requests_batch_calls",
    "prefix_lookup_batch_calls",
    "prefix_attach_batch_calls",
    "prefix_publish_batch_calls",
    "prefix_publish_release_batch_calls",
    "prefix_evict_batch_calls",
    "prefix_recycle_batch_calls",
    "buffer_too_small_preflights",
    "retryable_conflicts",
    "fail_stops",
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "cold_workspace_allocations",
    "materialized_page_objects",
    "forward_events",
    "completion_values",
    "event_queries",
    "event_waits",
    "quarantine_count",
    "fail_stop_count",
    "prefix_matches",
    "prefix_hits",
    "prefix_publishes",
    "prefix_evictions",
    "prefix_evicted_full_tokens",
    "prefix_evicted_swa_tokens",
    "prefix_global_alias_scans",
    "cow_copy_intents",
    "cow_move_calls",
    "cow_copied_tokens",
    "mirror_validation_calls",
    "mirror_syncs",
)
_FORBIDDEN_COUNTER_FIELDS = (
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "abort_steps_batch_calls",
    "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls",
    "retryable_conflicts",
    "fail_stops",
    "quarantine_count",
    "fail_stop_count",
)


def _counter_record(value: Any, fields: Sequence[str], label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise RuntimeError(f"{label} has a noncanonical field set")
    if any(
        isinstance(value[name], bool)
        or not isinstance(value[name], int)
        or value[name] < 0
        for name in fields
    ):
        raise RuntimeError(f"{label} contains an invalid integer")
    return {name: int(value[name]) for name in fields}


def _validate_batch_counter_contract(
    counters: dict[str, int],
    *,
    batch_size: int,
    completed_iterations: int,
    decode_tokens: int,
    hybrid: bool,
    swa_activity: dict[str, Any],
    stage: str,
    prefix_seeded: bool,
    global_cleanup: bool,
) -> None:
    if batch_size not in QUALIFICATION_BATCH_SIZES:
        raise RuntimeError(f"OrbitKV census has an unsupported batch size at {stage}")
    if (
        isinstance(completed_iterations, bool)
        or not isinstance(completed_iterations, int)
        or completed_iterations < 0
    ):
        raise RuntimeError(f"OrbitKV census has an invalid iteration count at {stage}")
    if decode_tokens != 33:
        raise RuntimeError(f"OrbitKV census has a noncanonical decode length at {stage}")
    seed_batches = int(prefix_seeded)
    cleanup_batches = int(global_cleanup)
    forward_batches = seed_batches + completed_iterations * decode_tokens
    release_batches = completed_iterations
    warm_request_calls = completed_iterations * batch_size
    expected = {
        "request_acquire_batch_calls": seed_batches + warm_request_calls,
        "request_fork_batch_calls": 0,
        "prepare_batch_calls": forward_batches,
        "submit_batch_calls": forward_batches,
        "complete_batch_calls": forward_batches,
        "release_batch_calls": release_batches,
        "recycle_requests_batch_calls": release_batches + seed_batches,
        "prefix_lookup_batch_calls": warm_request_calls,
        "prefix_attach_batch_calls": warm_request_calls,
        "prefix_publish_batch_calls": 0,
        "prefix_publish_release_batch_calls": seed_batches,
        "prefix_evict_batch_calls": cleanup_batches,
        "prefix_recycle_batch_calls": cleanup_batches,
        "buffer_too_small_preflights": (
            warm_request_calls + release_batches + seed_batches + cleanup_batches
        ),
        "cold_workspace_allocations": (
            warm_request_calls + release_batches + seed_batches + cleanup_batches
        ),
        "forward_events": forward_batches,
        "completion_values": forward_batches,
        "prefix_matches": seed_batches + completed_iterations * batch_size,
        "prefix_hits": completed_iterations * batch_size,
        "prefix_publishes": seed_batches,
        "prefix_evictions": cleanup_batches,
        "prefix_global_alias_scans": cleanup_batches * int(hybrid),
        "cow_copy_intents": 0,
        "cow_move_calls": 0,
        "cow_copied_tokens": 0,
    }
    identities = {
        name: {"expected": expected, "actual": counters[name]}
        for name, expected in expected.items()
        if counters[name] != expected
    }
    if identities:
        raise RuntimeError(
            f"OrbitKV ABI6 B{batch_size} batch identities disagree at {stage}: "
            f"{identities}"
        )
    full_evicted = counters["prefix_evicted_full_tokens"]
    swa_evicted = counters["prefix_evicted_swa_tokens"]
    if global_cleanup:
        invalid_eviction = (
            full_evicted <= 0
            or full_evicted % PAGE_TOKENS
            or (
                hybrid
                and (
                    swa_evicted <= 0
                    or swa_evicted % PAGE_TOKENS
                    or swa_evicted > full_evicted
                )
            )
            or (not hybrid and swa_evicted != 0)
        )
    else:
        invalid_eviction = full_evicted != 0 or swa_evicted != 0
    if invalid_eviction:
        raise RuntimeError(f"OrbitKV Prefix eviction spans are invalid at {stage}")
    if (
        counters["mirror_validation_calls"] != counters["mirror_syncs"]
        or bool(counters["mirror_validation_calls"]) is not prefix_seeded
    ):
        raise RuntimeError(f"OrbitKV global mirror cleanup is invalid at {stage}")

    event_queries = counters["event_queries"]
    event_waits = counters["event_waits"]
    if event_waits > forward_batches or event_queries + event_waits < forward_batches:
        raise RuntimeError(
            f"OrbitKV ABI6 event observation counters are inconsistent at {stage}"
        )

    base_acknowledgements = completed_iterations + cleanup_batches
    acknowledgement_calls = counters["acknowledge_reclamations_batch_calls"]
    online_acknowledgements = acknowledgement_calls - base_acknowledgements
    retirement_certificates = swa_activity["swa_retirement_certificates"]
    if hybrid:
        if not 0 <= online_acknowledgements <= forward_batches:
            raise RuntimeError(
                f"OrbitKV Hybrid acknowledgement count is invalid at {stage}"
            )
        if (online_acknowledgements > 0) != (retirement_certificates > 0):
            raise RuntimeError(
                f"OrbitKV Hybrid acknowledgements disagree with SWA activity at {stage}"
            )
    elif acknowledgement_calls != base_acknowledgements:
        raise RuntimeError(
            f"OrbitKV Full global acknowledgements are not grouped at {stage}"
        )


def _swa_activity(reported: dict[str, Any], hybrid: bool) -> dict[str, Any]:
    raw = reported.get("swa_activity")
    fields = (
        "swa_retirement_certificates",
        "swa_pages_reclaimed",
        "swa_wrap_events",
    )
    if (
        not isinstance(raw, dict)
        or set(raw) != {"status", "applicable", *fields}
        or raw.get("status") != "exposed"
        or not isinstance(raw.get("applicable"), bool)
        or raw["applicable"] is not hybrid
    ):
        raise RuntimeError("OrbitKV SWA activity has a noncanonical schema")
    metrics = _counter_record(
        {name: raw[name] for name in fields},
        fields,
        "OrbitKV SWA activity",
    )
    if not hybrid and any(metrics.values()):
        raise RuntimeError("Full-only manager reported SWA activity")
    return {
        "status": "exposed" if hybrid else "not_applicable",
        "source": "SGLang orbitkv_manager internal state",
        "derived": False,
        "applicable": hybrid,
        **metrics,
    }


def manager_census(
    info: dict[str, Any],
    config: Any,
    capacities: dict[str, Any],
    stage: str,
    *,
    batch_size: int,
    completed_iterations: int,
    decode_tokens: int,
    prefix_seeded: bool = False,
    global_cleanup: bool = False,
) -> dict[str, Any]:
    reported = _state(info).get("orbitkv_manager")
    if not isinstance(reported, dict):
        raise RuntimeError(f"OrbitKV manager census is missing at {stage}")
    allowed = {
        "abi_version",
        "identities",
        "arena_stats",
        "manager_stats",
        "swa_activity",
        "batch_counters",
    }
    if set(reported) != allowed or reported.get("abi_version") != 6:
        raise RuntimeError(f"OrbitKV manager top-level schema is invalid at {stage}")
    raw_identities = reported["identities"]
    raw_arena_stats = reported["arena_stats"]
    if not isinstance(raw_identities, list) or not isinstance(raw_arena_stats, list):
        raise RuntimeError(f"OrbitKV arena records are malformed at {stage}")
    if len(raw_identities) != len(config.classes) or len(raw_arena_stats) != len(
        config.classes
    ):
        raise RuntimeError(f"OrbitKV arena count differs from the plan at {stage}")

    identities = [
        _counter_record(value, _IDENTITY_FIELDS, f"OrbitKV identity[{index}]")
        for index, value in enumerate(raw_identities)
    ]
    arena_stats = [
        _counter_record(value, _ARENA_STATS_FIELDS, f"OrbitKV arena_stats[{index}]")
        for index, value in enumerate(raw_arena_stats)
    ]
    manager_stats = _counter_record(
        reported["manager_stats"], _MANAGER_STATS_FIELDS, "OrbitKV manager_stats"
    )
    batch_counters = _counter_record(
        reported["batch_counters"],
        _BATCH_COUNTER_FIELDS,
        "OrbitKV batch_counters",
    )
    hybrid = any(item.retention == "sliding" for item in config.classes)
    swa_activity = _swa_activity(reported, hybrid)
    forbidden = {
        name: batch_counters[name]
        for name in _FORBIDDEN_COUNTER_FIELDS
        if batch_counters[name]
    }
    if forbidden:
        raise RuntimeError(
            f"OrbitKV ABI6 failure counters are nonzero at {stage}: {forbidden}"
        )
    _validate_batch_counter_contract(
        batch_counters,
        batch_size=batch_size,
        completed_iterations=completed_iterations,
        decode_tokens=decode_tokens,
        hybrid=hybrid,
        swa_activity=swa_activity,
        stage=stage,
        prefix_seeded=prefix_seeded,
        global_cleanup=global_cleanup,
    )
    batch_call_counts = tuple(
        batch_counters[name]
        for name in (
            "prepare_batch_calls",
            "submit_batch_calls",
            "complete_batch_calls",
            "forward_events",
            "completion_values",
        )
    )
    if len(set(batch_call_counts)) != 1:
        raise RuntimeError(
            f"OrbitKV ABI6 batch call identities disagree at {stage}"
        )
    engine_epochs: set[int] = set()
    page_ranges: list[tuple[int, int]] = []
    for index, (identity, stats, class_config) in enumerate(
        zip(identities, arena_stats, config.classes, strict=True)
    ):
        if identity["class_id"] != index:
            raise RuntimeError(f"OrbitKV identities are not in class order at {stage}")
        expected = {
            "class_id": class_config.class_id,
            "pool_id": class_config.pool_id,
            "backend_domain": class_config.backend_domain,
            "page_tokens": PAGE_TOKENS,
        }
        mismatches = {
            name: {"expected": value, "actual": identity[name]}
            for name, value in expected.items()
            if identity[name] != value
        }
        if mismatches:
            raise RuntimeError(
                f"OrbitKV identity differs from plan at {stage}: {mismatches}"
            )
        if identity["engine_epoch"] <= 0 or identity["pool_epoch"] <= 0:
            raise RuntimeError(f"OrbitKV arena epochs are invalid at {stage}")
        if identity["page_count"] <= 0 or identity["first_page_id"] <= 0:
            raise RuntimeError(f"OrbitKV arena geometry is invalid at {stage}")
        engine_epochs.add(identity["engine_epoch"])
        page_range = (
            identity["first_page_id"],
            identity["first_page_id"] + identity["page_count"],
        )
        if any(
            page_range[0] < end and begin < page_range[1]
            for begin, end in page_ranges
        ):
            raise RuntimeError(f"OrbitKV arena page-id ranges overlap at {stage}")
        page_ranges.append(page_range)
        echoed = {
            name: identity[name]
            for name in (
                "engine_epoch",
                "pool_epoch",
                "pool_id",
                "page_count",
                "class_id",
                "backend_domain",
                "first_page_id",
            )
        }
        if any(stats[name] != value for name, value in echoed.items()):
            raise RuntimeError(f"OrbitKV arena stats changed identity at {stage}")
        if sum(stats[name] for name in _ARENA_PHASE_FIELDS) != identity["page_count"]:
            raise RuntimeError(f"OrbitKV arena phase census is incomplete at {stage}")
        capacity_name = (
            "full_tokens" if class_config.retention == "full" else "swa_tokens"
        )
        if identity["page_count"] * PAGE_TOKENS != capacities[capacity_name]:
            raise RuntimeError(
                f"OrbitKV arena capacity differs from SGLang readback at {stage}"
            )
    if len(engine_epochs) != 1:
        raise RuntimeError(f"OrbitKV arenas have different engine epochs at {stage}")

    for name in _ARENA_PHASE_FIELDS:
        if manager_stats[name] != sum(item[name] for item in arena_stats):
            raise RuntimeError(
                f"OrbitKV aggregate {name} differs from arena census at {stage}"
            )
    for aggregate, arena_field in (
        ("total_request_page_refs", "request_page_refs"),
        ("total_prefix_page_refs", "prefix_page_refs"),
        ("total_reader_pins", "reader_pins"),
    ):
        if manager_stats[aggregate] != sum(
            item[arena_field] for item in arena_stats
        ):
            raise RuntimeError(
                f"OrbitKV aggregate {aggregate} differs from arena census at {stage}"
            )
    steady_zero = (
        "active_requests",
        "active_snapshots",
        "evicted_prefixes",
        "prepared_steps",
        "submitted_steps",
        "reserved_pages",
        "writing_pages",
        "retiring_pages",
        "quarantined_pages",
        "exhausted_pages",
        "pending_reclamations",
        "total_request_page_refs",
        "total_reader_pins",
    )
    dirty = {
        name: manager_stats[name]
        for name in steady_zero
        if manager_stats[name] != 0
    }
    if dirty:
        raise RuntimeError(f"OrbitKV manager did not settle at {stage}: {dirty}")
    page_capacity = sum(item["page_count"] for item in identities)
    expect_live_prefix = prefix_seeded and not global_cleanup
    if expect_live_prefix:
        prefix_dirty = {
            "active_prefixes": manager_stats["active_prefixes"],
            "active_pages": manager_stats["active_pages"],
            "total_prefix_page_refs": manager_stats["total_prefix_page_refs"],
        }
        if (
            prefix_dirty["active_prefixes"] != 1
            or prefix_dirty["active_pages"] <= 0
            or prefix_dirty["total_prefix_page_refs"]
            != prefix_dirty["active_pages"]
            or manager_stats["free_pages"] + manager_stats["active_pages"]
            != page_capacity
        ):
            raise RuntimeError(
                f"OrbitKV Prefix census is not singular and settled at {stage}: "
                f"{prefix_dirty}"
            )
    elif any(
        manager_stats[name]
        for name in ("active_prefixes", "active_pages", "total_prefix_page_refs")
    ) or manager_stats["free_pages"] != page_capacity:
        raise RuntimeError(
            f"OrbitKV manager leaked pages at {stage}: "
            f"free={manager_stats['free_pages']} expected={page_capacity}"
        )
    return {
        "abi_version": 6,
        "identities": identities,
        "arena_stats": arena_stats,
        "manager_stats": manager_stats,
        "batch_counters": batch_counters,
        "swa_activity": swa_activity,
    }


def stock_census_absent(info: dict[str, Any], stage: str) -> None:
    if "orbitkv_manager" in _state(info):
        raise RuntimeError(f"pristine stock run loaded OrbitKV at {stage}")


def verify_swa_activity_transition(
    after_load: dict[str, Any], after_workload: dict[str, Any]
) -> None:
    before = after_load["swa_activity"]
    after = after_workload["swa_activity"]
    if before["status"] != after["status"]:
        raise RuntimeError("SWA activity exposure changed during the run")
    fields = (
        "swa_retirement_certificates",
        "swa_pages_reclaimed",
        "swa_wrap_events",
    )
    if before["status"] == "not_applicable":
        if any(before[name] or after[name] for name in fields):
            raise RuntimeError("Full-only run reported SWA activity")
        return
    if before["status"] != "exposed":
        raise RuntimeError("SWA activity was not exposed for a Hybrid run")
    for name in fields:
        if after[name] <= before[name]:
            raise RuntimeError(
                f"Hybrid workload did not advance SWA activity counter {name}"
            )


def _normalize_outputs(value: Any, requests: int) -> list[dict[str, Any]]:
    outputs = [value] if isinstance(value, dict) else list(value)
    if len(outputs) != requests:
        raise RuntimeError(
            f"SGLang returned {len(outputs)} outputs for {requests} requests"
        )
    for output in outputs:
        ids = output.get("output_ids")
        if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
            raise RuntimeError("SGLang output_ids are missing or invalid")
    return outputs


def _flush_cache_record(value: Any) -> dict[str, Any]:
    success = getattr(value, "success", None)
    message = getattr(value, "message", None)
    if not isinstance(success, bool) or not isinstance(message, str):
        raise RuntimeError("SGLang flush_cache returned a noncanonical result")
    if not success:
        raise RuntimeError(f"SGLang refused the global Prefix cleanup: {message}")
    return {"success": True, "message": message}


def run(args: argparse.Namespace, paths: dict[str, Path | None]) -> dict[str, Any]:
    run_started = time.perf_counter()
    environment = configure_environment(args, paths)
    build_tool = build_tool_identity()
    pinned_contract = verify_pinned_module_constants()
    source = verify_sglang_source(
        paths["sglang_root"], args.mode  # type: ignore[arg-type]
    )
    source["pinned_contract"] = pinned_contract
    plugin_selection = (
        verify_manager_entrypoint()
        if args.mode == "manager"
        else verify_stock_plugin_selection()
    )
    manager_config = None
    manager_plan = None
    manager_library = None
    if args.mode == "manager":
        from orbitkv_sglang.config import load_config

        manager_config = load_config()
        manager_plan = manager_plan_identity(manager_config)
        manager_library = artifact_identity(paths["library"])  # type: ignore[arg-type]
    contract, checkpoint = checkpoint_contract(
        paths["model"], manager_config  # type: ignore[arg-type]
    )
    if args.context_length > contract["max_position_embeddings"]:
        raise RuntimeError("--context-length exceeds checkpoint position capacity")
    canonical_prompts = tuple(
        tuple(prompt)
        for prompt in deterministic_input_ids(
            requests=args.requests,
            prompt_tokens=args.prompt_tokens,
            vocab_size=int(contract["vocab_size"]),
            seed=args.seed,
        )
    )
    canonical_prompt_digests = tuple(
        canonical_digest(list(prompt)) for prompt in canonical_prompts
    )
    engine_args = engine_arguments(
        args, paths["model"], contract  # type: ignore[arg-type]
    )
    source.update(
        {
            "sglang_python_sha256": sha256_file(
                paths["sglang_root"]  # type: ignore[operator]
                / "python/sglang/__init__.py"
            ),
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint_identity_helper_sha256": sha256_file(
                INTEGRATION_ROOT / "checkpoint_identity.py"
            ),
            "adapter": _adapter_identity(),
            "plugin_selection": plugin_selection,
            "plan": manager_plan["artifact"] if manager_plan else None,
            "library": manager_library,
            "build_tool": build_tool,
        }
    )

    before = gpu_snapshot("before_engine")
    setup_started = time.perf_counter()
    import sglang as sgl
    from sglang.srt.environ import envs

    if envs.SGLANG_USE_HND_KVCACHE.get():
        raise RuntimeError("SGLang resolved HND instead of the required NHD layout")

    imported = Path(sgl.__file__).resolve(strict=True)
    expected_package = (
        paths["sglang_root"] / "python/sglang"  # type: ignore[operator]
    ).resolve(strict=True)
    if not imported.is_relative_to(expected_package):
        raise RuntimeError("imported SGLang is outside --sglang-root")
    if sgl.__version__ != SUPPORTED_SGLANG_RELEASE.removeprefix("v"):
        raise RuntimeError("imported SGLang package version is not exactly 0.5.17")

    outputs_by_iteration: list[list[dict[str, Any]]] = []
    submitted_rids_by_iteration: list[list[str]] = []
    submitted_input_digests_by_iteration: list[list[str]] = []
    iteration_seconds: list[float] = []
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.decode_tokens,
        "min_new_tokens": args.decode_tokens,
        "ignore_eos": True,
        "sampling_seed": args.seed,
    }
    prefix_seed_sampling_params = {
        **sampling_params,
        "max_new_tokens": 1,
        "min_new_tokens": 1,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    engine_started = time.perf_counter()
    with sgl.Engine(**engine_args) as engine:
        load_seconds = time.perf_counter() - engine_started
        setup_and_load_seconds = time.perf_counter() - setup_started
        after_load = gpu_snapshot("after_load")
        info_after_load = engine.get_server_info()
        capacity_after_load = verify_runtime_contract(
            args, info_after_load, contract
        )
        if manager_config is None:
            stock_census_absent(info_after_load, "after_load")
            manager_after_load = None
        else:
            manager_after_load = manager_census(
                info_after_load,
                manager_config,
                capacity_after_load,
                "after_load",
                batch_size=args.requests,
                completed_iterations=0,
                decode_tokens=args.decode_tokens,
            )
        prefix_seed_tokens = (args.prompt_tokens - 1) // PAGE_TOKENS * PAGE_TOKENS
        seed_inputs = [list(canonical_prompts[0][:prefix_seed_tokens])]
        seed_rids = [f"orbitkv-prefix-seed-{args.seed}"]
        seed_started = time.perf_counter()
        seed_output = engine.generate(
            input_ids=seed_inputs,
            rid=seed_rids,
            sampling_params=prefix_seed_sampling_params,
        )
        prefix_seed_seconds = time.perf_counter() - seed_started
        seed_outputs = _normalize_outputs(seed_output, PREFIX_SEED_BATCH_SIZE)
        if any(
            len(output["output_ids"]) != 1
            or not isinstance(output.get("meta_info"), dict)
            or output["meta_info"].get("id") != rid
            for output, rid in zip(seed_outputs, seed_rids, strict=True)
        ):
            raise RuntimeError("untimed Prefix seed did not complete the exact B1 batch")
        if seed_inputs != [list(canonical_prompts[0][:prefix_seed_tokens])]:
            raise RuntimeError("SGLang mutated the untimed Prefix seed inputs")
        info_after_prefix_seed = engine.get_server_info()
        capacity_after_prefix_seed = verify_runtime_contract(
            args, info_after_prefix_seed, contract
        )
        if capacity_after_prefix_seed != capacity_after_load:
            raise RuntimeError("SGLang capacity changed during the Prefix seed")
        if manager_config is None:
            stock_census_absent(info_after_prefix_seed, "after_prefix_seed")
            manager_after_prefix_seed = None
        else:
            manager_after_prefix_seed = manager_census(
                info_after_prefix_seed,
                manager_config,
                capacity_after_prefix_seed,
                "after_prefix_seed",
                batch_size=args.requests,
                completed_iterations=0,
                decode_tokens=args.decode_tokens,
                prefix_seeded=True,
            )
        for iteration in range(args.iterations):
            submitted_prompts = [list(prompt) for prompt in canonical_prompts]
            submitted_input_digests = [
                canonical_digest(prompt) for prompt in submitted_prompts
            ]
            if tuple(submitted_input_digests) != canonical_prompt_digests:
                raise RuntimeError("qualification input changed before submission")
            submitted_rids = [
                f"orbitkv-canonical-{args.seed}-{iteration}-{request}"
                for request in range(args.requests)
            ]
            iteration_started = time.perf_counter()
            output = engine.generate(
                input_ids=submitted_prompts,
                rid=submitted_rids,
                sampling_params=sampling_params,
            )
            iteration_seconds.append(time.perf_counter() - iteration_started)
            if [canonical_digest(prompt) for prompt in submitted_prompts] != list(
                canonical_prompt_digests
            ):
                raise RuntimeError("SGLang mutated qualification input ids")
            normalized = _normalize_outputs(output, args.requests)
            cached = [item.get("meta_info", {}).get("cached_tokens") for item in normalized]
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != prefix_seed_tokens
                for value in cached
            ):
                raise RuntimeError(
                    "measured Prefix hit boundary differs from the untimed seed: "
                    f"expected={prefix_seed_tokens} actual={cached}"
                )
            outputs_by_iteration.append(normalized)
            submitted_rids_by_iteration.append(submitted_rids)
            submitted_input_digests_by_iteration.append(submitted_input_digests)
        info_after_workload = engine.get_server_info()
        capacity_after_workload = verify_runtime_contract(
            args, info_after_workload, contract
        )
        if capacity_after_workload != capacity_after_load:
            raise RuntimeError("SGLang Full/SWA capacity changed during the run")
        if manager_config is None:
            stock_census_absent(info_after_workload, "after_workload")
            manager_after_workload = None
        else:
            manager_after_workload = manager_census(
                info_after_workload,
                manager_config,
                capacity_after_workload,
                "after_workload",
                batch_size=args.requests,
                completed_iterations=args.iterations,
                decode_tokens=args.decode_tokens,
                prefix_seeded=True,
            )
            verify_swa_activity_transition(
                manager_after_load, manager_after_workload
            )
        after_workload = gpu_snapshot("after_workload")
        flush_cache = _flush_cache_record(engine.flush_cache())
        info_after_global_cleanup = engine.get_server_info()
        capacity_after_global_cleanup = verify_runtime_contract(
            args, info_after_global_cleanup, contract
        )
        if capacity_after_global_cleanup != capacity_after_workload:
            raise RuntimeError("SGLang capacity changed during global Prefix cleanup")
        if manager_config is None:
            stock_census_absent(info_after_global_cleanup, "after_global_cleanup")
            manager_after_global_cleanup = None
        else:
            manager_after_global_cleanup = manager_census(
                info_after_global_cleanup,
                manager_config,
                capacity_after_global_cleanup,
                "after_global_cleanup",
                batch_size=args.requests,
                completed_iterations=args.iterations,
                decode_tokens=args.decode_tokens,
                prefix_seeded=True,
                global_cleanup=True,
            )
        after_global_cleanup = gpu_snapshot("after_global_cleanup")
    after_shutdown = gpu_snapshot("after_shutdown")

    completion_tokens = sum(
        len(output["output_ids"])
        for iteration in outputs_by_iteration
        for output in iteration
    )
    expected_completion_tokens = (
        args.requests * args.iterations * args.decode_tokens
    )
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            f"completed {completion_tokens} tokens, "
            f"expected {expected_completion_tokens}"
        )
    traces = request_traces(
        outputs=outputs_by_iteration,
        submitted_rids=submitted_rids_by_iteration,
        submitted_input_digests=submitted_input_digests_by_iteration,
    )
    verify_request_trace_stability(traces)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    total_seconds = time.perf_counter() - run_started
    workload = {
        "requests": args.requests,
        "max_running_requests": args.max_running_requests,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "iterations": args.iterations,
        "seed": args.seed,
        "input_token_digest_sha256": input_digest(canonical_prompts),
    }
    prefix_seed_contract = {
        "requests": PREFIX_SEED_BATCH_SIZE,
        "prompt_tokens": prefix_seed_tokens,
        "input_token_digest_sha256": input_digest(seed_inputs),
        "sampling_params": prefix_seed_sampling_params,
        "included_in_iteration_timing": False,
        "included_in_measured_output_pairing": False,
    }
    comparison_contract = {
        "checkpoint_identity_sha256": canonical_digest(checkpoint),
        "attention_contract_sha256": canonical_digest(contract),
        "engine_args": pair_engine_arguments(args.mode, engine_args),
        "allowed_implementation_difference": PAIR_IMPLEMENTATION_DIFFERENCE,
        "sampling_params": sampling_params,
        "prefix_seed": prefix_seed_contract,
        "workload": workload,
        "capacity_readback": capacity_after_workload,
    }
    manager_record = None
    if manager_config is not None:
        manager_record = {
            "plan": manager_plan,
            "library": manager_library,
            "counter_contract": {
                "required_fields": list(_BATCH_COUNTER_FIELDS),
                "forbidden_nonzero_fields": list(_FORBIDDEN_COUNTER_FIELDS),
                "page_aligned_prefix_cow_expected": 0,
                "mirror_validation_equals_sync": True,
            },
            "after_load": manager_after_load,
            "after_prefix_seed": manager_after_prefix_seed,
            "after_workload": manager_after_workload,
            "final_census": manager_after_global_cleanup,
        }
    return {
        "schema": RECORD_SCHEMA,
        "mode": args.mode,
        "started_at_utc": started_at,
        "command": command,
        "command_sha256": canonical_digest(command),
        "environment": environment,
        "environment_sha256": canonical_digest(environment),
        "source_identity": source,
        "source_identity_sha256": canonical_digest(source),
        "runtime_identity": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sglang_version": sgl.__version__,
            "sglang_package": str(imported),
            "kv_layout": "nhd",
            "attention_backend": engine_args["attention_backend"],
            "deterministic_inference": engine_args[
                "enable_deterministic_inference"
            ],
            "sampling_backend": engine_args["sampling_backend"],
            "moe_runner_backend": _state(info_after_workload).get(
                "moe_runner_backend"
            ),
            "prefix_cache": {
                "enabled_readback": not _state(info_after_workload)[
                    "disable_radix_cache"
                ],
                "backend_readback": _state(info_after_workload)[
                    "radix_cache_backend"
                ],
                "expected_implementation": expected_prefix_cache(
                    args.mode, contract
                ),
            },
            "execution": "eager",
        },
        "model": str(paths["model"]),
        "checkpoint": checkpoint,
        "checkpoint_identity_sha256": canonical_digest(checkpoint),
        "checkpoint_contract": contract,
        "engine_args": engine_args,
        "sampling_params": sampling_params,
        "prefix_seed": {
            **prefix_seed_contract,
            "published_boundary_tokens": prefix_seed_tokens,
            "measured_prefix_limit_tokens": args.prompt_tokens - 1,
            "observed_hit_boundary_tokens": prefix_seed_tokens,
            "seconds": prefix_seed_seconds,
        },
        "workload": workload,
        "pairing": {
            "pair_key_sha256": canonical_digest(comparison_contract),
            "contract": comparison_contract,
            "valid_comparison_requires_equal_pair_key": True,
            "single_run_memory_claim": "none",
            "max_total_tokens_is_storage_sizing_not_compression": True,
        },
        "load_seconds": load_seconds,
        "setup_and_load_seconds": setup_and_load_seconds,
        "iteration_seconds": iteration_seconds,
        "iteration_total_seconds": sum(iteration_seconds),
        "total_seconds": total_seconds,
        "output_token_digest_sha256": token_digest(outputs_by_iteration),
        "output_request_digests_sha256": request_token_digests(
            outputs_by_iteration
        ),
        "request_traces": traces,
        "completed_requests": args.requests * args.iterations,
        "completion_tokens": completion_tokens,
        "server_memory": {
            "after_load": server_memory(info_after_load),
            "after_workload": server_memory(info_after_workload),
        },
        "capacity_readback": capacity_after_workload,
        "manager": manager_record,
        "global_prefix_cleanup": flush_cache,
        "gpu_snapshots": [
            before,
            after_load,
            after_workload,
            after_global_cleanup,
            after_shutdown,
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = validate_arguments(args)
        result = run(args, paths)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
