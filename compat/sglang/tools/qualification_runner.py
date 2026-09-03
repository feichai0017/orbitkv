#!/usr/bin/env python3
"""Run one manifest-bound qualification observation against SGLang.

This runner emits observations only.  Qualification is deliberately owned by
the independent pair verifier; every claim gate in a single-run record is
therefore fixed to false.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


_SGLANG_TOOLS_ROOT = Path(__file__).resolve().parent
_SGLANG_ROOT = _SGLANG_TOOLS_ROOT.parent
_REPOSITORY_ROOT = _SGLANG_ROOT.parents[1]
_ADAPTER_SOURCE_ROOT = _SGLANG_ROOT / "bridge/src"
_TOOLS_ROOT = _REPOSITORY_ROOT / "tools"
if str(_ADAPTER_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_SOURCE_ROOT))
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

import qualification_runtime as runtime_support
from checkpoint_identity import checkpoint_identity
from orbitkv_sglang.benchmark_profiles import (
    canonical_digest,
    fresh_input_ids,
    input_digest,
    request_token_digests,
    request_traces,
    token_digest,
)
from orbitkv_sglang.ffi import WIRE_VERSION
from orbitkv_sglang.runtime_admission import runtime_binding_from_manifest
from orbitkv_sglang.ffi.library import LoadedLibrary
from orbitkv_sglang.qualification_primitives import (
    parse_strict_json_object,
    sha256_file,
)
from orbitkv_sglang import pinned
from orbitkv_sglang.runtime_manifest import validate_runtime_manifest
from qualification_source import direct_source_identity
from engine_e2e_verifier_contract import manager_input_fingerprint


RECORD_SCHEMA = "orbitkv.sglang-exact-chunked-single-run.v1"
NATIVE_RECORD_SCHEMA = "orbitkv.sglang-native-session-single-run.v1"
EXACT_TOPOLOGY = "whole_domain_chunked_token_kv"
FULL_SLIDING_TOPOLOGY = "whole_domain_full_sliding_token_kv"
SLIDING_TOPOLOGY = "whole_domain_sliding_token_kv"
QUALIFICATION_PROFILES = (
    EXACT_TOPOLOGY,
    FULL_SLIDING_TOPOLOGY,
    SLIDING_TOPOLOGY,
)
_PROFILE_CACHE_POLICIES = {
    EXACT_TOPOLOGY: "request_private",
    FULL_SLIDING_TOPOLOGY: "shared_prefix",
    SLIDING_TOPOLOGY: "request_private",
}
PAGE_TOKENS = 16
RUNTIME_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
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
_ARENA_FIELDS = (
    "engine_epoch",
    "pool_epoch",
    "pool_id",
    "page_count",
    "class_id",
    "backend_domain",
    "first_page_id",
    "free_pages",
    "reserved_pages",
    "writing_pages",
    "active_pages",
    "retiring_pages",
    "quarantined_pages",
    "exhausted_pages",
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
    "free_pages",
    "reserved_pages",
    "writing_pages",
    "active_pages",
    "retiring_pages",
    "quarantined_pages",
    "exhausted_pages",
    "pending_reclamations",
    "total_request_page_refs",
    "total_prefix_page_refs",
    "total_reader_pins",
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "mode",
        "started_at_utc",
        "command",
        "command_sha256",
        "environment",
        "environment_sha256",
        "source_identity",
        "source_identity_sha256",
        "runtime_identity",
        "model",
        "checkpoint",
        "checkpoint_identity_sha256",
        "runtime_manifest",
        "runtime_binding",
        "engine_args",
        "sampling_params",
        "workload",
        "timings",
        "outputs",
        "server_capacity",
        "manager",
        "gpu_snapshots",
        "claims",
    }
)
NATIVE_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS | {"profile"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one manifest-bound SGLang Engine observation in manager or "
            "stock mode. The emitted record is not a qualification result."
        )
    )
    parser.add_argument("--mode", choices=("manager", "stock"), required=True)
    parser.add_argument(
        "--profile", choices=QUALIFICATION_PROFILES, default=EXACT_TOPOLOGY
    )
    parser.add_argument("--case", choices=("roomy", "exact-floor"), required=True)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-manifest", "--manifest", dest="manifest")
    parser.add_argument("--library")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--chunked-prefill-size", type=int)
    parser.add_argument("--swa-full-tokens-ratio", type=float)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _profile(args: argparse.Namespace) -> str:
    return runtime_support.selected_profile(args, QUALIFICATION_PROFILES)


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


def validate_chunked_workload(
    *, prompt_tokens: int, decode_tokens: int, chunk_tokens: int, iterations: int
) -> dict[str, int]:
    prompt = _positive_integer("prompt tokens", prompt_tokens)
    decode = _positive_integer("decode tokens", decode_tokens)
    chunk = _positive_integer("chunk tokens", chunk_tokens)
    count = _positive_integer("iterations", iterations)
    if prompt > chunk:
        raise ValueError("prompt tokens must not exceed the compiled chunk")
    final_kv = prompt + decode - 1
    if final_kv <= chunk:
        raise ValueError("prompt plus materialized decode must cross a chunk boundary")
    return {
        "prompt_tokens": prompt,
        "decode_tokens": decode,
        "final_kv_tokens": final_kv,
        "chunk_tokens": chunk,
        "chunk_epoch_count": (final_kv + chunk - 1) // chunk,
        "iterations": count,
    }


def validate_arguments(args: argparse.Namespace) -> dict[str, Path | None]:
    profile = _profile(args)
    for name in (
        "prompt_tokens",
        "decode_tokens",
        "iterations",
        "context_length",
        "max_total_tokens",
        "max_running_requests",
    ):
        _positive_integer(f"--{name.replace('_', '-')}", getattr(args, name))
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.max_running_requests != 1:
        raise ValueError("qualification execution requires --max-running-requests=1")
    if args.max_total_tokens % PAGE_TOKENS:
        raise ValueError("--max-total-tokens must be divisible by 16")
    if args.prompt_tokens + args.decode_tokens >= args.context_length:
        raise ValueError("prompt plus decode must leave one context slot unused")
    if args.mem_fraction_static is not None and not (
        0.0 < args.mem_fraction_static <= 1.0
    ):
        raise ValueError("--mem-fraction-static must be in (0, 1]")
    prefill = getattr(args, "chunked_prefill_size", None)
    if prefill is not None:
        _positive_integer("--chunked-prefill-size", prefill)
        if prefill % PAGE_TOKENS:
            raise ValueError("--chunked-prefill-size must be divisible by 16")
    elif profile != EXACT_TOPOLOGY:
        raise ValueError(
            "Sliding qualification requires --chunked-prefill-size"
        )
    ratio = getattr(args, "swa_full_tokens_ratio", None)
    if ratio is not None and (
        isinstance(ratio, bool) or not isinstance(ratio, (int, float))
        or not 0.0 < float(ratio) <= 1.0
    ):
        raise ValueError("--swa-full-tokens-ratio must be in (0, 1]")
    sglang_root = _directory(args.sglang_root, "--sglang-root")
    if not (sglang_root / "python/sglang/__init__.py").is_file():
        raise ValueError("--sglang-root is not an SGLang source checkout")
    model = _directory(args.model, "--model")
    _regular_file(str(model / "config.json"), "checkpoint config")
    manifest = None
    library = None
    if args.mode == "manager":
        if not args.manifest or not args.library:
            raise ValueError("manager mode requires --runtime-manifest and --library")
        manifest = _regular_file(args.manifest, "--runtime-manifest")
        library = _regular_file(args.library, "--library")
    elif args.mode == "stock":
        if args.library is not None:
            raise ValueError("stock mode forbids --library")
        if profile == EXACT_TOPOLOGY and args.manifest is not None:
            raise ValueError("stock Chunked mode forbids --runtime-manifest")
        if profile != EXACT_TOPOLOGY and not args.manifest:
            raise ValueError(
                "stock Sliding mode requires --runtime-manifest as a profile artifact"
            )
        if args.manifest:
            manifest = _regular_file(args.manifest, "--runtime-manifest")
    else:
        raise ValueError(f"unknown mode {args.mode!r}")
    return {
        "sglang_root": sglang_root,
        "model": model,
        "manifest": manifest,
        "library": library,
    }


def validate_case_capacity(
    *, case: str, max_total_tokens: int, chunk_tokens: int, final_kv_tokens: int
) -> None:
    capacity = _positive_integer("max total tokens", max_total_tokens)
    chunk = _positive_integer("chunk tokens", chunk_tokens)
    logical = _positive_integer("materialized KV tokens", final_kv_tokens)
    if case == "roomy":
        if capacity < logical:
            raise ValueError(
                "roomy case requires --max-total-tokens to cover materialized KV"
            )
        return
    if case == "exact-floor":
        if capacity != chunk:
            raise ValueError(
                "exact-floor case requires --max-total-tokens == chunk_tokens"
            )
        if logical <= capacity:
            raise ValueError("exact-floor case must cross the physical chunk floor")
        return
    raise ValueError("--case must be roomy or exact-floor")


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(RUNTIME_MANIFEST_MAX_BYTES + 1)
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read RuntimeManifest {path}: {error}") from error
    if len(encoded) > RUNTIME_MANIFEST_MAX_BYTES:
        raise ValueError("RuntimeManifest exceeds the 16777216-byte limit")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"cannot read RuntimeManifest {path}: {error}") from error
    return parse_strict_json_object(text)


def _chunk_geometry(binding: Mapping[str, Any]) -> dict[str, int]:
    signature = binding.get("execution_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("target binding omitted its execution signature")
    classes = signature.get("token_classes")
    if not isinstance(classes, list) or len(classes) != 1:
        raise ValueError("exact chunked binding requires one token class")
    item = classes[0]
    if not isinstance(item, Mapping):
        raise ValueError("exact chunked token class is malformed")
    address = item.get("address")
    if not isinstance(address, Mapping):
        raise ValueError("exact chunked address is malformed")
    page_tokens = _positive_integer("manifest page tokens", signature.get("page_tokens"))
    blocks = _positive_integer("manifest blocks per epoch", address.get("blocks_per_epoch"))
    return {
        "page_tokens": page_tokens,
        "chunk_tokens": page_tokens * blocks,
        "blocks_per_epoch": blocks,
    }


def load_qualification_admission(
    path: Path, profile: str = EXACT_TOPOLOGY
) -> dict[str, Any]:
    manifest = _read_strict_json(path)
    validate_runtime_manifest(manifest)
    binding = dict(runtime_binding_from_manifest(manifest))
    if profile not in QUALIFICATION_PROFILES:
        raise ValueError("qualification profile is unsupported")
    if binding.get("execution_topology") != profile:
        raise ValueError(
            "RuntimeManifest topology differs from the selected qualification profile"
        )
    geometry = (
        _chunk_geometry(binding)
        if profile == EXACT_TOPOLOGY
        else runtime_support.sliding_geometry(binding)
    )
    if geometry["page_tokens"] != PAGE_TOKENS:
        raise ValueError("qualification requires page_tokens=16")
    result = {
        "runtime_manifest": manifest,
        "runtime_binding": binding,
        "chunk_geometry": geometry,
        "lifecycle_route": "native_session",
        "cache_policy": _PROFILE_CACHE_POLICIES[profile],
    }
    if profile != EXACT_TOPOLOGY:
        source = manifest.get("source")
        if not isinstance(source, Mapping) or source.get("kind") != "attention_state":
            raise ValueError("Sliding qualification requires an attention-state source")
        result.update(
            profile=profile,
            class_ids=tuple(
                range(len(binding["execution_signature"]["token_classes"]))
            ),
            plan_fingerprint=manager_input_fingerprint(
                manifest, "RuntimeManifest"
            ),
        )
    return result


def configure_environment(
    args: argparse.Namespace, paths: Mapping[str, Path | None]
) -> dict[str, str]:
    if any(name == "sglang" or name.startswith("sglang.") for name in sys.modules):
        raise RuntimeError("SGLang was imported before the benchmark environment froze")
    for name in tuple(os.environ):
        if name.startswith("ORBITKV_") or name == "SGLANG_PLUGINS":
            os.environ.pop(name)

    root = paths["sglang_root"]
    if not isinstance(root, Path):
        raise RuntimeError("resolved SGLang root is missing")
    sglang_python = root / "python"
    python_paths = [str(sglang_python), str(_ADAPTER_SOURCE_ROOT)]
    previous = os.environ.get("PYTHONPATH")
    if previous:
        python_paths.append(previous)
    os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)
    for path in reversed((sglang_python, _ADAPTER_SOURCE_ROOT)):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    environment = {
        "SGLANG_USE_HND_KVCACHE": "0",
        "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
        "SGLANG_RADIX_FORCE_MISS": "0",
    }
    if args.mode == "manager":
        manifest = paths.get("manifest")
        library = paths.get("library")
        if not isinstance(manifest, Path) or not isinstance(library, Path):
            raise RuntimeError("manager artifact paths are missing")
        environment.update(
            ORBITKV_RUNTIME_MANIFEST=str(manifest),
            ORBITKV_LIBRARY=str(library),
            ORBITKV_SGLANG_ROOT=str(root),
        )
        if _profile(args) == EXACT_TOPOLOGY:
            environment["ORBITKV_PRESSURE_TELEMETRY"] = "1"
    os.environ.update(environment)
    recorded = dict(environment)
    recorded["PYTHONPATH"] = os.environ["PYTHONPATH"]
    recorded["PATH"] = os.environ.get("PATH", "")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        recorded["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return recorded


def verify_chunked_source(root: Path, mode: str) -> dict[str, Any]:
    """Bind the run to the current reviewed chunked-source contract."""

    contract = pinned.pinned_source_contract()
    if mode == "manager":
        checkout = pinned.validate_patched_checkout(root)
        patch_path = _REPOSITORY_ROOT / contract["patch_path"]
        reviewed_patch = {
            "status": "applied",
            "sha256": sha256_file(patch_path),
            "bytes": patch_path.stat().st_size,
        }
    elif mode == "stock":
        checkout = pinned.validate_base_checkout(root)
        reviewed_patch = {
            "status": "absent",
            "sha256": None,
            "bytes": 0,
        }
    else:
        raise RuntimeError(f"unknown source verification mode: {mode}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot read the pinned SGLang revision") from error
    if revision != contract["revision"]:
        raise RuntimeError("validated SGLang revision differs from pinned contract")
    return {
        "root": str(checkout),
        "release": contract["release"],
        "revision": revision,
        "contract": contract,
        "contract_sha256": canonical_digest(contract),
        "reviewed_patch": reviewed_patch,
    }


def engine_arguments(
    args: argparse.Namespace,
    model: Path,
    chunk_geometry: Mapping[str, int],
    *,
    cache_policy: str | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    profile = _profile(args)
    chunk_tokens = (
        _positive_integer(
            "chunk geometry chunk_tokens", chunk_geometry.get("chunk_tokens")
        )
        if profile == EXACT_TOPOLOGY
        else _positive_integer(
            "--chunked-prefill-size", getattr(args, "chunked_prefill_size", None)
        )
    )
    page_tokens = _positive_integer(
        "chunk geometry page_tokens", chunk_geometry.get("page_tokens")
    )
    values: dict[str, Any] = {
        "model_path": str(model),
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": args.context_length,
        "page_size": page_tokens,
        "attention_backend": "fa3",
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": (
            True if cache_policy is None else cache_policy == "request_private"
        ),
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": chunk_tokens,
        "prefill_max_requests": 1,
        "max_prefill_tokens": chunk_tokens,
        "enable_dynamic_chunking": False,
        "enable_mixed_chunk": False,
        "max_running_requests": 1,
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
    if args.mem_fraction_static is not None:
        values["mem_fraction_static"] = args.mem_fraction_static
    ratio = getattr(args, "swa_full_tokens_ratio", None)
    if ratio is not None:
        values["swa_full_tokens_ratio"] = float(ratio)
    if architecture == "GptOssForCausalLM":
        # The pinned SGLang exposes this setting directly. Pin it rather than
        # accepting architecture/driver-dependent "auto" resolution.
        values.update(
            moe_runner_backend="triton",
            moe_a2a_backend="none",
            ep_size=1,
        )
    if args.mode == "manager":
        values["radix_cache_backend"] = "orbitkv"
    elif args.mode != "stock":
        raise RuntimeError(f"unknown mode {args.mode!r}")
    return values


def _state(info: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sole authoritative scheduler state for this DP=1 runner."""

    states = info.get("internal_states")
    if (
        not isinstance(states, list)
        or len(states) != 1
        or not isinstance(states[0], dict)
    ):
        raise RuntimeError(
            "SGLang must return exactly one scheduler internal state"
        )
    return states[0]


def stock_census_absent(info: Mapping[str, Any], stage: str) -> None:
    if "orbitkv_manager" in _state(info):
        raise RuntimeError(f"stock run loaded OrbitKV at {stage}")


def _nonnegative_mapping(
    value: Any, label: str, fields: Sequence[str]
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise RuntimeError(f"{label} has a noncanonical field set")
    result: dict[str, int] = {}
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
        ):
            raise RuntimeError(f"{label} contains an invalid counter")
        result[name] = item
    return result


def manager_census(
    info: Mapping[str, Any],
    stage: str,
    *,
    expected_manifest_fingerprint: str,
    expected_runtime_binding_fingerprint: str,
    expected_chunk_geometry: Mapping[str, int],
    expected_engine_args: Mapping[str, Any],
    expected_lifecycle_route: str,
    expected_cache_policy: str,
    expected_profile: str = EXACT_TOPOLOGY,
    expected_class_ids: Sequence[int] = (0,),
    expected_plan_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate actual worker readback without inventing provenance fields."""

    if (
        not expected_manifest_fingerprint
        or not expected_runtime_binding_fingerprint
    ):
        raise RuntimeError("parent admission fingerprints are missing")
    raw = _state(info).get("orbitkv_manager")
    if not isinstance(raw, dict):
        raise RuntimeError(f"OrbitKV manager census is missing at {stage}")
    if raw.get("wire_version") != WIRE_VERSION:
        raise RuntimeError(f"OrbitKV wire-version manager census is missing at {stage}")
    if raw.get("runtime_manifest_fingerprint") != expected_manifest_fingerprint:
        raise RuntimeError(f"OrbitKV worker manifest fingerprint changed at {stage}")
    if (
        raw.get("runtime_binding_fingerprint")
        != expected_runtime_binding_fingerprint
    ):
        raise RuntimeError(f"OrbitKV worker binding fingerprint changed at {stage}")
    lifecycle_route = raw.get("lifecycle_route")
    if lifecycle_route != expected_lifecycle_route:
        raise RuntimeError(
            f"OrbitKV lifecycle route differs from admission at {stage}"
        )
    cache_policy = raw.get("cache_policy")
    if cache_policy != expected_cache_policy:
        raise RuntimeError(
            f"OrbitKV cache policy differs from admission at {stage}"
        )
    runtime_proof = (
        _validate_runtime_proof(
            raw.get("runtime_proof"),
            stage,
            chunk_geometry=expected_chunk_geometry,
            engine_args=expected_engine_args,
        )
        if expected_profile == EXACT_TOPOLOGY
        else runtime_support.validate_native_runtime_proof(
            raw.get("runtime_proof"), stage, expected_engine_args
        )
    )
    plan_field = (
        "plan_fingerprint"
        if expected_profile == EXACT_TOPOLOGY
        else "manager_input_fingerprint"
    )
    has_runtime_proof = raw.get("runtime_proof") is not None
    if (
        expected_plan_fingerprint is not None
        and (expected_profile != EXACT_TOPOLOGY or has_runtime_proof)
        and raw.get(plan_field) != expected_plan_fingerprint
    ):
        raise RuntimeError(f"OrbitKV worker plan fingerprint changed at {stage}")
    identities = raw.get("identities")
    arenas = raw.get("arena_stats")
    class_ids = tuple(expected_class_ids)
    if (
        not class_ids
        or not isinstance(identities, list)
        or len(identities) != len(class_ids)
        or not isinstance(arenas, list)
        or len(arenas) != len(class_ids)
    ):
        raise RuntimeError(f"OrbitKV arena census is malformed at {stage}")
    manager_stats = _nonnegative_mapping(
        raw.get("manager_stats"), "manager_stats", _MANAGER_STATS_FIELDS
    )
    counter_fields = (
        runtime_support.BATCH_COUNTER_FIELDS
        if expected_profile == EXACT_TOPOLOGY
        else runtime_support.SESSION_COUNTER_FIELDS
    )
    batch_counters = _nonnegative_mapping(
        raw.get("batch_counters"),
        "batch_counters",
        counter_fields,
    )
    pressure = raw.get("pressure")
    if not isinstance(pressure, dict):
        raise RuntimeError(f"OrbitKV pressure readback is missing at {stage}")
    if expected_profile == EXACT_TOPOLOGY:
        _validate_pressure_readback(
            pressure, stage=stage,
            require_activity=stage in ("after_workload", "final"),
        )
        swa_activity = None
    else:
        expected_pressure = {
            "schema": "orbitkv.runtime-pressure.v1",
            "enabled": False,
            "mode": "event_driven_high_water",
            "sample_count": 0,
        }
        if pressure != expected_pressure:
            raise RuntimeError(
                f"OrbitKV native-session pressure must be disabled at {stage}"
            )
        swa_activity = runtime_support.validate_swa_activity(
            raw.get("swa_activity"), stage
        )
        completion_evidence = runtime_support.validate_completion_evidence(
            raw.get("completion_evidence"), stage,
            require_activity=stage in ("after_workload", "final"),
        )
    normalized_identities = [
        _nonnegative_mapping(item, "arena identity", _IDENTITY_FIELDS)
        for item in identities
    ]
    normalized_arenas = [
        _nonnegative_mapping(item, "arena stats", _ARENA_FIELDS)
        for item in arenas
    ]
    if [item["class_id"] for item in normalized_identities] != list(class_ids):
        raise RuntimeError(f"OrbitKV class identity changed at {stage}")
    if [item["class_id"] for item in normalized_arenas] != list(class_ids):
        raise RuntimeError(f"OrbitKV arena class identity changed at {stage}")
    if any(
        item["page_tokens"] != expected_chunk_geometry.get("page_tokens")
        for item in normalized_identities
    ):
        raise RuntimeError(f"OrbitKV arena page geometry changed at {stage}")
    phases = (
        "free_pages",
        "reserved_pages",
        "writing_pages",
        "active_pages",
        "retiring_pages",
        "quarantined_pages",
        "exhausted_pages",
    )
    for identity, arena in zip(
        normalized_identities, normalized_arenas, strict=True
    ):
        page_count = arena["page_count"]
        if sum(arena[name] for name in phases) != page_count:
            raise RuntimeError(f"OrbitKV arena phase census is incomplete at {stage}")
        for name in (
            "engine_epoch", "pool_epoch", "pool_id", "page_count",
            "class_id", "backend_domain", "first_page_id",
        ):
            if arena[name] != identity[name]:
                raise RuntimeError(f"OrbitKV arena identity echo changed at {stage}")
    aggregate_fields = (
        "free_pages",
        "reserved_pages",
        "writing_pages",
        "active_pages",
        "retiring_pages",
        "quarantined_pages",
        "exhausted_pages",
    )
    if any(
        manager_stats[name] != sum(arena[name] for arena in normalized_arenas)
        for name in aggregate_fields
    ):
        raise RuntimeError(f"OrbitKV manager/arena page census differs at {stage}")
    if (
        manager_stats["total_request_page_refs"]
        != sum(item["request_page_refs"] for item in normalized_arenas)
        or manager_stats["total_prefix_page_refs"]
        != sum(item["prefix_page_refs"] for item in normalized_arenas)
        or manager_stats["total_reader_pins"]
        != sum(item["reader_pins"] for item in normalized_arenas)
    ):
        raise RuntimeError(f"OrbitKV manager/arena reference census differs at {stage}")
    failure_fields = (
        "retryable_conflicts",
        "fail_stops",
        "fail_stop_count",
        "quarantine_count",
    )
    nonzero_failures = {
        name: batch_counters[name]
        for name in failure_fields
        if batch_counters.get(name, 0)
    }
    if nonzero_failures:
        raise RuntimeError(f"OrbitKV failure counters are nonzero at {stage}: {nonzero_failures}")
    must_drain = (
        stage in ("after_load", "final")
        or expected_profile == EXACT_TOPOLOGY
        or expected_cache_policy == "request_private"
    )
    if must_drain:
        dirty = {
            name: manager_stats.get(name)
            for name in (
                "active_requests",
                "active_snapshots",
                "active_prefixes",
                "prepared_steps",
                "submitted_steps",
                "reserved_pages",
                "writing_pages",
                "active_pages",
                "retiring_pages",
                "quarantined_pages",
                "pending_reclamations",
                "total_request_page_refs",
                "total_prefix_page_refs",
                "total_reader_pins",
            )
            if manager_stats.get(name, 0)
        }
        if dirty or any(
            arena["free_pages"] != arena["page_count"]
            for arena in normalized_arenas
        ):
            raise RuntimeError(f"OrbitKV manager did not drain at {stage}: {dirty}")
    elif stage == "after_workload":
        forbidden_live = {
            name: manager_stats[name]
            for name in (
                "active_requests", "prepared_steps", "submitted_steps",
                "reserved_pages", "writing_pages", "retiring_pages",
                "quarantined_pages", "pending_reclamations",
                "total_request_page_refs", "total_reader_pins",
            )
            if manager_stats[name]
        }
        if (
            forbidden_live
            or manager_stats["active_pages"]
            != manager_stats["total_prefix_page_refs"]
        ):
            raise RuntimeError(
                f"OrbitKV shared-prefix manager did not settle at {stage}: "
                f"{forbidden_live}"
            )
    result = {
        "stage": stage,
        "lifecycle_route": lifecycle_route,
        "cache_policy": cache_policy,
        "identities": normalized_identities,
        "manager_stats": manager_stats,
        "arena_stats": normalized_arenas,
        "batch_counters": batch_counters,
        "pressure": dict(pressure),
        "runtime_proof": runtime_proof,
    }
    if expected_profile != EXACT_TOPOLOGY:
        result["swa_activity"] = swa_activity
        result["completion_evidence"] = completion_evidence
    return result


def _positive_runtime_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} is not a positive integer")
    return value


def _validate_runtime_proof(
    value: Any,
    stage: str,
    *,
    chunk_geometry: Mapping[str, int],
    engine_args: Mapping[str, Any],
) -> dict[str, Any]:
    admitted_chunk = _positive_runtime_integer(
        chunk_geometry.get("chunk_tokens"),
        "admitted chunk geometry chunk_tokens",
    )
    admitted_page = _positive_runtime_integer(
        chunk_geometry.get("page_tokens"),
        "admitted chunk geometry page_tokens",
    )
    engine_chunk = _positive_runtime_integer(
        engine_args.get("chunked_prefill_size"),
        "engine argument chunked_prefill_size",
    )
    engine_page = _positive_runtime_integer(
        engine_args.get("page_size"), "engine argument page_size"
    )
    engine_prefill = _positive_runtime_integer(
        engine_args.get("max_prefill_tokens"),
        "engine argument max_prefill_tokens",
    )
    engine_running = _positive_runtime_integer(
        engine_args.get("max_running_requests"),
        "engine argument max_running_requests",
    )
    expected = {"actual_attention_backend", "effective_scheduler"}
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"OrbitKV runtime proof is malformed at {stage}")
    backend = value.get("actual_attention_backend")
    backend_fields = {
        "backend_class",
        "backend_module",
        "prefill_backend",
        "decode_backend",
        "has_local_attention",
        "attention_chunk_size",
        "page_size",
        "compiled_layer_ids",
        "use_irope_layer_ids",
    }
    if not isinstance(backend, dict) or set(backend) != backend_fields:
        raise RuntimeError(f"OrbitKV backend proof is malformed at {stage}")
    if (
        backend.get("backend_class") != "FlashAttentionBackend"
        or backend.get("backend_module")
        != "sglang.srt.layers.attention.flashattention_backend"
        or backend.get("prefill_backend") != "fa3"
        or backend.get("decode_backend") != "fa3"
        or backend.get("has_local_attention") is not True
    ):
        raise RuntimeError(f"OrbitKV loaded backend proof changed at {stage}")
    for field in ("attention_chunk_size", "page_size"):
        _positive_runtime_integer(backend.get(field), f"backend proof {field}")
    if (
        backend["attention_chunk_size"] != admitted_chunk
        or engine_chunk != admitted_chunk
    ):
        raise RuntimeError(
            f"OrbitKV attention chunk geometry changed at {stage}"
        )
    if backend["page_size"] != admitted_page or engine_page != admitted_page:
        raise RuntimeError(f"OrbitKV attention page geometry changed at {stage}")
    compiled = backend.get("compiled_layer_ids")
    local = backend.get("use_irope_layer_ids")
    if (
        not isinstance(compiled, list)
        or not compiled
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in compiled
        )
        or compiled != sorted(set(compiled))
        or local != compiled
    ):
        raise RuntimeError(f"OrbitKV layer execution proof changed at {stage}")
    scheduler = value.get("effective_scheduler")
    scheduler_fields = {
        "max_prefill_tokens",
        "max_running_requests",
        "effective_max_running_requests_per_dp",
    }
    if not isinstance(scheduler, dict) or set(scheduler) != scheduler_fields:
        raise RuntimeError(f"OrbitKV scheduler proof is malformed at {stage}")
    normalized_scheduler = {
        field: _positive_runtime_integer(
            scheduler.get(field), f"scheduler proof {field}"
        )
        for field in scheduler_fields
    }
    if (
        normalized_scheduler["max_prefill_tokens"] != admitted_chunk
        or engine_prefill != admitted_chunk
    ):
        raise RuntimeError(
            f"OrbitKV scheduler prefill geometry changed at {stage}"
        )
    if (
        normalized_scheduler["max_running_requests"] != engine_running
        or normalized_scheduler["effective_max_running_requests_per_dp"]
        != engine_running
    ):
        raise RuntimeError(
            f"OrbitKV scheduler concurrency geometry changed at {stage}"
        )
    return {
        "actual_attention_backend": dict(backend),
        "effective_scheduler": normalized_scheduler,
    }


def claim_gates() -> dict[str, dict[str, Any]]:
    reasons = {
        "correctness_qualified": (
            "one single-run observation is not an independently verified manager/stock correctness pair"
        ),
        "stream_event_qualified": (
            "server counters do not independently prove CUDA stream/event ordering"
        ),
        "capacity_qualified": (
            "configured and observed capacity is not a qualified memory-saving comparison"
        ),
        "throughput_go": (
            "one process-local run does not satisfy paired multi-epoch throughput thresholds"
        ),
    }
    return {
        name: {"qualified": False, "reasons": [reason]}
        for name, reason in reasons.items()
    }


_CAPACITY_FAILURE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bkv cache pool is full\b",
        r"\bprefill out of memory\b",
        r"\bdecode out of memory\b",
        r"\bout of memory even after retracting all other requests\b",
        r"\btry to allocate [0-9]+ tokens\b",
    )
)
_NON_KV_OOM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"cuda out of memory",
        r"cublas.*alloc",
        r"hip out of memory",
        r"allocator.*device memory",
    )
)


def classify_capacity_failure(error: BaseException) -> str:
    """Classify only explicit SGLang KV-capacity failures.

    The caller must propagate every other exception.  In particular, a device
    allocator OOM is not evidence that the configured KV-token floor was hit.
    """

    message = str(error).strip()
    if not message or any(pattern.search(message) for pattern in _NON_KV_OOM_PATTERNS):
        raise error
    if any(pattern.search(message) for pattern in _CAPACITY_FAILURE_PATTERNS):
        return "capacity_exhausted"
    raise error


def gpu_snapshot(stage: str) -> dict[str, Any]:
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
        import subprocess

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
    gpus = []
    for values in csv.reader(output.splitlines()):
        if len(values) != len(fields):
            raise RuntimeError("nvidia-smi returned an unexpected row")
        gpus.append(
            {name: item.strip() for name, item in zip(fields, values, strict=True)}
        )
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return {"stage": stage, "time_ns": time.time_ns(), "gpus": gpus}


def _checkpoint_config(model: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = parse_strict_json_object((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"cannot read checkpoint config: {error}") from error
    architectures = raw.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise RuntimeError("checkpoint must expose exactly one architecture")
    integers = {}
    for name in ("num_hidden_layers", "vocab_size", "max_position_embeddings"):
        integers[name] = _positive_integer(f"checkpoint {name}", raw.get(name))
    chunk_value = raw.get("attention_chunk_size")
    chunk = (
        None
        if chunk_value is None
        else _positive_integer("checkpoint attention_chunk_size", chunk_value)
    )
    sliding_value = raw.get("sliding_window")
    sliding = (
        None
        if sliding_value is None
        else _positive_integer("checkpoint sliding_window", sliding_value)
    )
    layer_types = raw.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list)
        or len(layer_types) != integers["num_hidden_layers"]
        or any(not isinstance(item, str) or not item for item in layer_types)
    ):
        raise RuntimeError("checkpoint layer_types are malformed")
    control_names = (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    )
    controls = {
        name: raw[name]
        for name in control_names
        if isinstance(raw.get(name), int) and not isinstance(raw.get(name), bool)
    }
    config: dict[str, Any] = {
        "architectures": architectures,
        **integers,
        "control_token_ids": controls,
    }
    if chunk is not None:
        config["attention_chunk_size"] = chunk
    if sliding is not None:
        config["sliding_window"] = sliding
    if layer_types is not None:
        config["layer_types"] = layer_types
    identity = checkpoint_identity(model, "auto")
    if identity["weight_bytes"] <= 0 or not identity["indexed_weights_complete"]:
        raise RuntimeError("checkpoint weights are missing or incomplete")
    return config, identity


def _manifest_record(
    path: Path, admission: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = admission["runtime_manifest"]
    binding = admission["runtime_binding"]
    manager = manifest["token_manager_plan"]
    layout = manager["layout"]
    source = manifest["source"]
    profile = admission.get("profile", EXACT_TOPOLOGY)
    result = {
        "artifact": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "schema": manifest["schema"],
        "version": manifest["version"],
        "manifest_fingerprint": manifest["fingerprint"],
        "layout_plan_fingerprint": layout["plan_fingerprint"],
        "execution_signature_fingerprint": binding["execution_signature"][
            "fingerprint"
        ],
        "chunk_geometry": dict(admission["chunk_geometry"]),
    }
    if profile == EXACT_TOPOLOGY:
        retention = source["program"]
        canonical_retention = json.dumps(
            retention, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result["retention_program_fingerprint"] = (
            "sha256:" + hashlib.sha256(canonical_retention).hexdigest()
        )
    else:
        result = {
            "artifact": result["artifact"],
            "document": manifest,
            "manifest_fingerprint": result["manifest_fingerprint"],
            "layout_plan_fingerprint": result["layout_plan_fingerprint"],
            "execution_signature_fingerprint": result["execution_signature_fingerprint"],
            "profile_geometry": dict(admission["chunk_geometry"]),
        }
    return result


def _capacity_observation(
    info: Mapping[str, Any], requested_tokens: int, *, profile: str = EXACT_TOPOLOGY
) -> dict[str, Any]:
    try:
        memory = _state(info).get("memory_usage")
        if not isinstance(memory, dict):
            raise ValueError("server memory readback is missing")
        available = memory.get("token_capacity")
        if isinstance(available, bool) or not isinstance(available, int) or available <= 0:
            raise ValueError("server token capacity is invalid")
        if available != requested_tokens:
            raise ValueError(
                f"server token capacity {available} differs from requested {requested_tokens}"
            )
        result = {
            "status": "observed",
            "requested_tokens": requested_tokens,
            "available_tokens": available,
            "failure": None,
        }
        if profile != EXACT_TOPOLOGY:
            swa = memory.get("token_capacity_swa")
            if isinstance(swa, bool) or not isinstance(swa, int) or swa <= 0:
                raise ValueError("server SWA token capacity is invalid")
            result["class_capacities"] = {
                "full_tokens": (0 if profile == SLIDING_TOPOLOGY else available),
                "sliding_tokens": swa,
            }
        return result
    except ValueError as error:
        result = {
            "status": "failed",
            "requested_tokens": requested_tokens,
            "available_tokens": None,
            "failure": {"type": "invalid", "message": str(error)},
        }
        if profile != EXACT_TOPOLOGY:
            result["class_capacities"] = None
        return result


def _capacity_failure_observation(
    error: BaseException,
    requested_tokens: int,
    before: Mapping[str, Any] | None,
    *,
    profile: str = EXACT_TOPOLOGY,
) -> dict[str, Any]:
    failure_type = classify_capacity_failure(error)
    available = None
    if before is not None and before.get("status") == "observed":
        available = before.get("available_tokens")
    result = {
        "status": "failed",
        "requested_tokens": requested_tokens,
        "available_tokens": available,
        "failure": {"type": failure_type, "message": str(error).strip()},
    }
    if profile != EXACT_TOPOLOGY:
        result["class_capacities"] = (
            None if before is None else before.get("class_capacities")
        )
    return result


def _validate_pressure_readback(
    pressure: Mapping[str, Any], *, stage: str, require_activity: bool
) -> None:
    if pressure.get("enabled") is not True:
        raise RuntimeError(f"OrbitKV pressure telemetry is not enabled at {stage}")
    expected = {
        "schema",
        "enabled",
        "mode",
        "scope",
        "sample_count",
        "last_event",
        "event_counts",
        "active_requests",
        "max_active_requests",
        "global",
        "classes",
    }
    if set(pressure) != expected:
        raise RuntimeError(f"OrbitKV pressure schema changed at {stage}")
    if (
        pressure.get("schema") != "orbitkv.runtime-pressure.v1"
        or pressure.get("mode") != "event_driven_high_water"
    ):
        raise RuntimeError(f"OrbitKV pressure telemetry is not enabled at {stage}")
    samples = pressure.get("sample_count")
    counts = pressure.get("event_counts")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
        or not isinstance(counts, dict)
        or not counts
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for name, count in counts.items()
        )
        or sum(counts.values()) != samples
        or pressure.get("last_event") not in counts
    ):
        raise RuntimeError(f"OrbitKV pressure event census is invalid at {stage}")
    active = pressure.get("active_requests")
    maximum = pressure.get("max_active_requests")
    if (
        isinstance(active, bool)
        or not isinstance(active, int)
        or active != 0
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 0
        or maximum > 1
        or (require_activity and maximum != 1)
    ):
        raise RuntimeError(f"OrbitKV pressure request census is invalid at {stage}")
    classes = pressure.get("classes")
    if not isinstance(pressure.get("global"), dict) or not isinstance(classes, list):
        raise RuntimeError(f"OrbitKV pressure values are malformed at {stage}")
    if len(classes) != 1 or not isinstance(classes[0], dict):
        raise RuntimeError(f"OrbitKV pressure class census is malformed at {stage}")


def _verify_runtime_readback(
    info: Mapping[str, Any], args: argparse.Namespace, engine_args: Mapping[str, Any]
) -> None:
    state = _state(info)
    required = {
        "page_size": engine_args["page_size"],
        "max_total_tokens": args.max_total_tokens,
        "attention_backend": "fa3",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "chunked_prefill_size": engine_args["chunked_prefill_size"],
        "max_prefill_tokens": engine_args["max_prefill_tokens"],
        "prefill_max_requests": 1,
        "max_running_requests": engine_args["max_running_requests"],
        "effective_max_running_requests_per_dp": engine_args[
            "max_running_requests"
        ],
        "enable_dynamic_chunking": False,
        "enable_mixed_chunk": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": engine_args["disable_radix_cache"],
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
    }
    for name in ("moe_runner_backend", "moe_a2a_backend", "ep_size"):
        if name in engine_args:
            required[name] = engine_args[name]
    mismatch = {
        name: {"expected": expected, "actual": state.get(name)}
        for name, expected in required.items()
        if state.get(name) != expected
    }
    if mismatch:
        raise RuntimeError(f"resolved SGLang qualification contract mismatch: {mismatch}")


def _normalized_outputs(value: Any) -> list[dict[str, Any]]:
    outputs = [value] if isinstance(value, dict) else list(value)
    if len(outputs) != 1:
        raise RuntimeError("exact chunked workload must return one request")
    ids = outputs[0].get("output_ids")
    if not isinstance(ids, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in ids
    ):
        raise RuntimeError("SGLang output_ids are missing or invalid")
    return outputs


def _output_record(traces: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    iterations = []
    for index, row in enumerate(traces):
        requests = []
        for trace in row:
            requests.append(
                {
                    "request_id": trace["submitted_rid"],
                    "input_sha256": trace["submitted_input_ids_sha256"],
                    "output_ids": trace["output_ids"],
                    "output_sha256": trace["output_ids_sha256"],
                    "cached_tokens": trace["cached_tokens"],
                }
            )
        iterations.append({"iteration": index, "requests": requests})
    return {"iterations": iterations, "aggregate_sha256": canonical_digest(iterations)}


def _combine_capacity_observations(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    if before["status"] != "observed":
        return dict(before)
    if after["status"] != "observed":
        return dict(after)
    if before["available_tokens"] != after["available_tokens"] or (
        before.get("class_capacities") != after.get("class_capacities")
    ):
        result = {
            "status": "failed",
            "requested_tokens": after["requested_tokens"],
            "available_tokens": after["available_tokens"],
            "failure": {
                "type": "drift",
                "message": "server token capacity changed during the workload",
            },
        }
        if "class_capacities" in after:
            result["class_capacities"] = after["class_capacities"]
        return result
    return dict(after)


def _flush_succeeded(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return value.get("success") is True
    return getattr(value, "success", None) is True


def _final_server_info(engine: Any, *, flush_cache: bool) -> Mapping[str, Any]:
    if flush_cache:
        flush = engine.flush_cache()
        if not _flush_succeeded(flush):
            raise RuntimeError(
                "SGLang cache flush failed before final drain proof"
            )
    return engine.get_server_info()


def _canonical_fingerprint(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RuntimeError(f"{label} is not a canonical fingerprint")
    return value


def _runtime_fingerprints(
    admission: Mapping[str, Any], manifest_record: Mapping[str, Any]
) -> tuple[str, str, str]:
    manifest = _canonical_fingerprint(
        manifest_record.get("manifest_fingerprint"), "manifest fingerprint"
    )
    binding_value = admission["runtime_binding"].get("fingerprint")
    binding = _canonical_fingerprint(binding_value, "binding fingerprint")
    if admission.get("profile", EXACT_TOPOLOGY) == EXACT_TOPOLOGY:
        plan = _canonical_fingerprint(
            manifest_record.get("retention_program_fingerprint"),
            "retention program fingerprint",
        )
    else:
        plan = _canonical_fingerprint(
            admission.get("plan_fingerprint"), "manager input fingerprint"
        )
    return manifest, binding, plan


def run(
    args: argparse.Namespace, paths: Mapping[str, Path | None]
) -> dict[str, Any]:
    profile = _profile(args)
    run_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid.uuid4())
    environment = configure_environment(args, paths)
    runtime_support.reject_legacy_manager_entrypoint()
    root = paths["sglang_root"]
    model = paths["model"]
    if not isinstance(root, Path) or not isinstance(model, Path):
        raise RuntimeError("resolved benchmark paths are incomplete")

    source_contract = verify_chunked_source(root, args.mode)
    source = {
        "contract": source_contract,
        "pinned_contract": source_contract["contract"],
        "direct_source": direct_source_identity(args.mode),
        "sglang_python_sha256": sha256_file(
            root / "python/sglang/__init__.py"
        ),
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "checkpoint_identity_helper_sha256": sha256_file(
            _SGLANG_TOOLS_ROOT / "checkpoint_identity.py"
        ),
        "adapter": runtime_support.adapter_identity(),
        "build_tool": runtime_support.build_tool_identity(),
    }

    checkpoint_config, checkpoint = _checkpoint_config(model)
    checkpoint_chunk = checkpoint_config.get("attention_chunk_size")
    admission = None
    manifest_record = None
    runtime_binding = None
    if args.mode == "manager":
        manifest_path = paths["manifest"]
        library_path = paths["library"]
        if not isinstance(manifest_path, Path) or not isinstance(library_path, Path):
            raise RuntimeError("manager artifacts are missing")
        admission = load_qualification_admission(manifest_path, profile)
        geometry = admission["chunk_geometry"]
        if (
            profile == EXACT_TOPOLOGY
            and geometry["chunk_tokens"] != checkpoint_chunk
        ):
            raise RuntimeError(
                "RuntimeManifest chunk size differs from the checkpoint"
            )
        if profile != EXACT_TOPOLOGY:
            sliding_class = next(
                item for item in geometry["classes"]
                if item["retention"] == "sliding"
            )
            if checkpoint_config.get("sliding_window") != sliding_class["window_tokens"]:
                raise RuntimeError(
                    "RuntimeManifest sliding window differs from the checkpoint"
                )
            checkpoint_geometry = runtime_support.checkpoint_sliding_geometry(
                checkpoint_config, profile, page_tokens=PAGE_TOKENS
            )
            runtime_support.validate_profile_geometry_match(
                geometry, checkpoint_geometry
            )
        manifest_record = _manifest_record(manifest_path, admission)
        runtime_binding = admission["runtime_binding"]
        LoadedLibrary(library_path)
        source["library"] = {
            "path": str(library_path),
            "bytes": library_path.stat().st_size,
            "sha256": sha256_file(library_path),
            "wire_version": WIRE_VERSION,
        }
    else:
        if profile == EXACT_TOPOLOGY:
            if not isinstance(checkpoint_chunk, int) or checkpoint_chunk % PAGE_TOKENS:
                raise RuntimeError("checkpoint attention chunk must be page aligned")
            geometry = {
                "page_tokens": PAGE_TOKENS,
                "chunk_tokens": checkpoint_chunk,
                "blocks_per_epoch": checkpoint_chunk // PAGE_TOKENS,
            }
        else:
            manifest_path = paths["manifest"]
            if not isinstance(manifest_path, Path):
                raise RuntimeError("stock Sliding profile artifact is missing")
            profile_admission = load_qualification_admission(manifest_path, profile)
            geometry = profile_admission["chunk_geometry"]
            checkpoint_geometry = runtime_support.checkpoint_sliding_geometry(
                checkpoint_config, profile, page_tokens=PAGE_TOKENS
            )
            runtime_support.validate_profile_geometry_match(
                geometry, checkpoint_geometry
            )
        source["library"] = None
    profile_artifact = (
        None
        if profile == EXACT_TOPOLOGY
        else {
            "path": str(paths["manifest"]),
            "bytes": paths["manifest"].stat().st_size,
            "sha256": sha256_file(paths["manifest"]),
        }
    )

    if profile == EXACT_TOPOLOGY:
        capacity_contract = None
        validated_workload = validate_chunked_workload(
            prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens,
            chunk_tokens=geometry["chunk_tokens"], iterations=args.iterations,
        )
        validate_case_capacity(
            case=args.case, max_total_tokens=args.max_total_tokens,
            chunk_tokens=geometry["chunk_tokens"],
            final_kv_tokens=validated_workload["final_kv_tokens"],
        )
    else:
        validated_workload = runtime_support.validate_native_workload(
            prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens,
            iterations=args.iterations, profile_geometry=geometry,
        )
        capacity_contract = runtime_support.native_capacity_contract(
            case=args.case, profile=profile,
            max_total_tokens=args.max_total_tokens,
            chunked_prefill_tokens=args.chunked_prefill_size,
            final_kv_tokens=validated_workload["final_kv_tokens"],
            profile_geometry=geometry,
            swa_full_tokens_ratio=getattr(args, "swa_full_tokens_ratio", None),
        )
    if args.context_length > checkpoint_config["max_position_embeddings"]:
        raise RuntimeError("--context-length exceeds checkpoint position capacity")
    if validated_workload["final_kv_tokens"] >= args.context_length:
        raise RuntimeError("materialized KV boundary must fit inside context length")

    forbidden = tuple(checkpoint_config["control_token_ids"].values())
    prompts = tuple(
        tuple(tuple(prompt) for prompt in fresh_input_ids(
            requests=1, prompt_tokens=args.prompt_tokens,
            vocab_size=checkpoint_config["vocab_size"], seed=args.seed,
            iteration=(0 if profile == EXACT_TOPOLOGY else iteration),
            forbidden_token_ids=forbidden,
        ))
        for iteration in range(args.iterations)
    )
    if any(len(row) != 1 for row in prompts):
        raise RuntimeError("deterministic qualification prompt construction changed")
    if profile == EXACT_TOPOLOGY and any(row != prompts[0] for row in prompts):
        raise RuntimeError("deterministic replay prompt construction changed")
    if profile != EXACT_TOPOLOGY and len(
        {tuple(row[0][:PAGE_TOKENS]) for row in prompts}
    ) != len(prompts):
        raise RuntimeError("native qualification prompt pages are not unique")
    prompt_digests = [
        [canonical_digest(list(row[0]))] for row in prompts
    ]

    engine_args = engine_arguments(
        args, model, geometry,
        cache_policy=(
            None
            if profile == EXACT_TOPOLOGY
            else _PROFILE_CACHE_POLICIES[profile]
        ),
        architecture=checkpoint_config["architectures"][0],
    )
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.decode_tokens,
        "min_new_tokens": args.decode_tokens,
        "ignore_eos": True,
        "sampling_seed": args.seed,
    }
    before_engine = gpu_snapshot("before_engine")
    import sglang as sgl
    from sglang.srt.environ import envs

    if envs.SGLANG_USE_HND_KVCACHE.get():
        raise RuntimeError("SGLang resolved HND instead of the required NHD layout")
    package = Path(sgl.__file__).resolve(strict=True)
    expected_package = (root / "python/sglang").resolve(strict=True)
    if not package.is_relative_to(expected_package):
        raise RuntimeError("imported SGLang is outside --sglang-root")
    if sgl.__version__ != source_contract["contract"]["release"].removeprefix(
        "v"
    ):
        raise RuntimeError("imported SGLang version differs from the pinned source")

    outputs_by_iteration: list[list[dict[str, Any]]] = []
    submitted_rids: list[list[str]] = []
    submitted_digests: list[list[str]] = []
    iteration_seconds: list[float] = []
    manager_snapshots: list[dict[str, Any]] = []
    manager_runtime_proof: dict[str, Any] | None = None
    gpu_snapshots = [before_engine]
    capacity_before: dict[str, Any] | None = None
    capacity_after: dict[str, Any] | None = None
    capacity_failure: dict[str, Any] | None = None
    load_seconds = 0.0
    engine_started = time.perf_counter()
    try:
        with sgl.Engine(**engine_args) as engine:
            load_seconds = time.perf_counter() - engine_started
            gpu_snapshots.append(gpu_snapshot("after_load"))
            after_load = engine.get_server_info()
            _verify_runtime_readback(after_load, args, engine_args)
            capacity_before = _capacity_observation(
                after_load, args.max_total_tokens, profile=profile
            )
            if capacity_before["status"] != "observed":
                raise RuntimeError(
                    "SGLang capacity readback is invalid before the workload: "
                    + capacity_before["failure"]["message"]
                )
            if (
                profile != EXACT_TOPOLOGY
                and capacity_contract is not None
                and capacity_before["class_capacities"]
                != {
                    "full_tokens": capacity_contract["expected_full_tokens"],
                    "sliding_tokens": capacity_contract["expected_sliding_tokens"],
                }
            ):
                raise RuntimeError(
                    "SGLang per-class capacity differs from the input contract"
                )
            if args.mode == "manager":
                assert admission is not None and manifest_record is not None
                manifest_fp, runtime_binding_fp, plan_fp = _runtime_fingerprints(
                    admission, manifest_record
                )
                initial_census = manager_census(
                    after_load,
                    "after_load",
                    expected_manifest_fingerprint=manifest_fp,
                    expected_runtime_binding_fingerprint=runtime_binding_fp,
                    expected_chunk_geometry=geometry,
                    expected_engine_args=engine_args,
                    expected_lifecycle_route=admission["lifecycle_route"],
                    expected_cache_policy=admission["cache_policy"],
                    expected_profile=profile,
                    expected_class_ids=admission.get("class_ids", (0,)),
                    expected_plan_fingerprint=plan_fp,
                )
                manager_snapshots.append(initial_census)
                manager_runtime_proof = initial_census["runtime_proof"]
            else:
                stock_census_absent(after_load, "after_load")

            for iteration, row in enumerate(prompts):
                input_ids = [list(row[0])]
                rid_prefix = (
                    "orbitkv-chunked"
                    if profile == EXACT_TOPOLOGY
                    else "orbitkv-native-session"
                )
                rid = [f"{rid_prefix}-{args.seed}-{iteration}-0"]
                before_digest = canonical_digest(input_ids[0])
                iteration_started = time.perf_counter()
                raw_output = engine.generate(
                    input_ids=input_ids,
                    rid=rid,
                    sampling_params=sampling_params,
                )
                elapsed = time.perf_counter() - iteration_started
                if canonical_digest(input_ids[0]) != before_digest:
                    raise RuntimeError("SGLang mutated the submitted input ids")
                normalized = _normalized_outputs(raw_output)
                if len(normalized[0]["output_ids"]) != args.decode_tokens:
                    meta = normalized[0].get("meta_info")
                    finish_reason = (
                        meta.get("finish_reason")
                        if isinstance(meta, Mapping)
                        else None
                    )
                    raise RuntimeError(
                        "SGLang returned an incomplete decode: "
                        f"expected={args.decode_tokens} "
                        f"actual={len(normalized[0]['output_ids'])} "
                        f"finish_reason={finish_reason!r}"
                    )
                cached = normalized[0].get("meta_info", {}).get("cached_tokens")
                if cached != 0:
                    raise RuntimeError(
                        "fresh qualification request observed cached prompt tokens"
                    )
                iteration_seconds.append(elapsed)
                outputs_by_iteration.append(normalized)
                submitted_rids.append(rid)
                submitted_digests.append(prompt_digests[iteration])

            after_workload = engine.get_server_info()
            _verify_runtime_readback(after_workload, args, engine_args)
            capacity_after = _capacity_observation(
                after_workload, args.max_total_tokens, profile=profile
            )
            if capacity_after["status"] != "observed":
                raise RuntimeError(
                    "SGLang capacity readback is invalid after the workload: "
                    + capacity_after["failure"]["message"]
                )
            if args.mode == "manager":
                workload_census = manager_census(
                    after_workload,
                    "after_workload",
                    expected_manifest_fingerprint=manifest_fp,
                    expected_runtime_binding_fingerprint=runtime_binding_fp,
                    expected_chunk_geometry=geometry,
                    expected_engine_args=engine_args,
                    expected_lifecycle_route=admission["lifecycle_route"],
                    expected_cache_policy=admission["cache_policy"],
                    expected_profile=profile,
                    expected_class_ids=admission.get("class_ids", (0,)),
                    expected_plan_fingerprint=plan_fp,
                )
                if workload_census["runtime_proof"] != manager_runtime_proof:
                    raise RuntimeError("OrbitKV runtime proof changed during workload")
                manager_snapshots.append(workload_census)
            else:
                stock_census_absent(after_workload, "after_workload")
            gpu_snapshots.append(gpu_snapshot("after_workload"))

            final_info = _final_server_info(
                engine,
                flush_cache=(args.mode == "manager" and profile != EXACT_TOPOLOGY),
            )
            _verify_runtime_readback(final_info, args, engine_args)
            if args.mode == "manager":
                final_census = manager_census(
                    final_info,
                    "final",
                    expected_manifest_fingerprint=manifest_fp,
                    expected_runtime_binding_fingerprint=runtime_binding_fp,
                    expected_chunk_geometry=geometry,
                    expected_engine_args=engine_args,
                    expected_lifecycle_route=admission["lifecycle_route"],
                    expected_cache_policy=admission["cache_policy"],
                    expected_profile=profile,
                    expected_class_ids=admission.get("class_ids", (0,)),
                    expected_plan_fingerprint=plan_fp,
                )
                if final_census["runtime_proof"] != manager_runtime_proof:
                    raise RuntimeError("OrbitKV runtime proof changed before shutdown")
                manager_snapshots.append(final_census)
            else:
                stock_census_absent(final_info, "final")
    except Exception as error:
        if (
            args.mode != "stock"
            or args.case != "exact-floor"
            or outputs_by_iteration
            or iteration_seconds
        ):
            raise
        capacity_failure = _capacity_failure_observation(
            error, args.max_total_tokens, capacity_before, profile=profile
        )
        load_seconds = max(load_seconds, time.perf_counter() - engine_started)
    gpu_snapshots.append(gpu_snapshot("after_shutdown"))
    if args.mode == "manager" and profile != EXACT_TOPOLOGY:
        runtime_support.require_swa_progress(
            manager_snapshots[0]["swa_activity"],
            manager_snapshots[1]["swa_activity"],
        )

    traces = request_traces(
        outputs=outputs_by_iteration,
        submitted_rids=submitted_rids,
        submitted_input_digests=submitted_digests,
    )
    # Exercise and retain the standard digest definitions before projecting
    # into the smaller frozen record schema.
    if request_token_digests(outputs_by_iteration) != [
        [trace["output_ids_sha256"] for trace in row] for row in traces
    ]:
        raise RuntimeError("request output digest projection changed")
    if token_digest(outputs_by_iteration) != canonical_digest(
        [[trace["output_ids"] for trace in row] for row in traces]
    ):
        raise RuntimeError("aggregate output digest projection changed")

    materialized = validated_workload["final_kv_tokens"]
    workload = {
        "case": args.case,
        "requests": 1,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "materialized_kv_tokens_per_request": materialized,
        "iterations": args.iterations,
        "seed": args.seed,
        "fresh_prompts": True,
        "input_token_digest_sha256": input_digest(
            [prompt for row in prompts for prompt in row]
        ),
        "input_token_digests_by_iteration_sha256": prompt_digests,
    }
    if profile == EXACT_TOPOLOGY:
        chunk_tokens = geometry["chunk_tokens"]
        workload["chunk_geometry"] = {
            **dict(geometry),
            "chunk_epoch_count_per_request": validated_workload["chunk_epoch_count"],
            "epoch_end_crossings_per_request": (
                materialized // chunk_tokens
                - (args.prompt_tokens - 1) // chunk_tokens
            ),
        }
    else:
        workload["profile_geometry"] = dict(geometry)
        workload["retirement_boundary_crossings_per_request"] = (
            validated_workload["retirement_boundary_crossings"]
        )
    if profile == EXACT_TOPOLOGY:
        checkpoint_record = {"identity": checkpoint, "config": checkpoint_config}
    else:
        checkpoint_record = {
            "identity": checkpoint,
            "config": checkpoint_config,
            "backend_profile": {
                "attention_backend": "fa3",
                "moe_runner_backend": engine_args.get("moe_runner_backend"),
                "moe_a2a_backend": engine_args.get("moe_a2a_backend"),
                "ep_size": engine_args.get("ep_size"),
            },
        }
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    total_seconds = time.perf_counter() - run_started
    record = {
        "schema": (RECORD_SCHEMA if profile == EXACT_TOPOLOGY else NATIVE_RECORD_SCHEMA),
        "mode": args.mode,
        "started_at_utc": started_at,
        "command": command,
        "command_sha256": canonical_digest(command),
        "environment": environment,
        "environment_sha256": canonical_digest(environment),
        "source_identity": source,
        "source_identity_sha256": canonical_digest(source),
        "runtime_identity": runtime_support.runtime_identity(
            sgl,
            package,
            engine_args,
            run_id=run_id,
            runtime_proof=(
                manager_runtime_proof if args.mode == "manager" else None
            ),
        ),
        "model": str(model),
        "checkpoint": checkpoint_record,
        "checkpoint_identity_sha256": canonical_digest(checkpoint_record),
        "runtime_manifest": manifest_record,
        "runtime_binding": runtime_binding,
        "engine_args": engine_args,
        "sampling_params": sampling_params,
        "workload": workload,
        "timings": {
            "load_seconds": load_seconds,
            "total_seconds": total_seconds,
            "iteration_seconds": iteration_seconds,
        },
        "outputs": _output_record(traces),
        "server_capacity": (
            capacity_failure
            if capacity_failure is not None
            else _combine_capacity_observations(
                capacity_before or {}, capacity_after or {}
            )
        ),
        "manager": (
            {"wire_version": WIRE_VERSION, "snapshots": manager_snapshots}
            if args.mode == "manager"
            else None
        ),
        "gpu_snapshots": gpu_snapshots,
        "claims": claim_gates(),
    }
    if profile != EXACT_TOPOLOGY:
        assert capacity_contract is not None
        record["profile"] = profile
        record["source_identity"]["profile_artifact"] = profile_artifact
        record["source_identity_sha256"] = canonical_digest(
            record["source_identity"]
        )
        capacity = record["server_capacity"]
        if capacity.get("status") == "observed":
            if capacity["class_capacities"] != {
                "full_tokens": capacity_contract["expected_full_tokens"],
                "sliding_tokens": capacity_contract["expected_sliding_tokens"],
            }:
                raise RuntimeError("SGLang per-class capacity differs from the input contract")
            capacity["floor"] = capacity_contract
    expected_keys = (
        TOP_LEVEL_KEYS if profile == EXACT_TOPOLOGY else NATIVE_TOP_LEVEL_KEYS
    )
    if set(record) != expected_keys:
        raise RuntimeError("qualification record construction changed its top-level schema")
    return record


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
