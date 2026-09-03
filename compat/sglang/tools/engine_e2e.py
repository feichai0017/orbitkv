#!/usr/bin/env python3
"""Run a deterministic synchronous SGLang Engine workload.

The manager mode exercises the reviewed direct-source OrbitKV integration.
The stock mode exercises the pristine checkout.  The program writes exactly
one JSON document to stdout and never persists benchmark results itself.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
SGLANG_ROOT = TOOLS_ROOT.parent
REPOSITORY_ROOT = SGLANG_ROOT.parents[1]
ADAPTER_SOURCE_ROOT = SGLANG_ROOT / "bridge/src"
if str(ADAPTER_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_SOURCE_ROOT))

import qualification_runtime as runtime_support
import engine_e2e_contract as contract
from checkpoint_identity import checkpoint_identity
from orbitkv_sglang import pinned
from orbitkv_sglang.ffi import WIRE_VERSION
from orbitkv_sglang.ffi.library import LoadedLibrary
from orbitkv_sglang.qualification_primitives import (
    canonical_json_sha256,
    parse_strict_json_object,
    sha256_file,
)
from orbitkv_sglang.runtime_admission import runtime_binding_from_manifest
from orbitkv_sglang.runtime_manifest import (
    RUNTIME_MANIFEST_MAX_BYTES,
    load_runtime_manifest,
    validate_runtime_manifest,
)
from orbitkv_sglang.runtime import disabled_pressure_report


RECORD_SCHEMA = "orbitkv.sglang-engine-e2e.v3"
NATIVE_SESSION_TOPOLOGIES = {
    "whole_domain_full_token_kv": (("full", "token_kv"),),
    "whole_domain_full_latent_kv": (("full", "latent_kv"),),
    "whole_domain_full_sliding_token_kv": (
        ("full", "token_kv"),
        ("sliding", "token_kv"),
    ),
    "whole_domain_sliding_token_kv": (("sliding", "token_kv"),),
    "whole_domain_chunked_token_kv": (("chunked", "token_kv"),),
}
NATIVE_SESSION_CACHE_POLICIES = {
    "whole_domain_full_token_kv": "shared_prefix",
    "whole_domain_full_latent_kv": "request_private",
    "whole_domain_full_sliding_token_kv": "shared_prefix",
    "whole_domain_sliding_token_kv": "request_private",
    "whole_domain_chunked_token_kv": "request_private",
}
_CHUNKED_TOPOLOGY = "whole_domain_chunked_token_kv"
_OWNER_MODULE = "orbitkv_sglang.engine"
_OWNER_TYPE = "OrbitKvLifecycleOwner"
_OWNER_FIELDS = frozenset(
    {
        "module",
        "type",
        "owner_is_process_singleton",
        "config_is_canonical",
        "allocator_owned",
        "tree_cache_owned",
    }
)
_SESSION_COUNTER_FIELDS = contract.SESSION_COUNTER_FIELDS
_POST_WORKLOAD_RESIDENCY_FIELDS = contract.POST_WORKLOAD_RESIDENCY_FIELDS
_ACCELERATOR_FIELDS = frozenset(
    {
        "device_type",
        "device_name",
        "compute_capability",
        "total_memory_bytes",
        "runtime_version",
        "driver_version",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic SGLang Engine E2E workload in stock or "
            "direct-source manager mode and emit one JSON record."
        )
    )
    parser.add_argument("--mode", choices=("stock", "manager"), required=True)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-manifest", "--manifest", dest="manifest")
    parser.add_argument("--library")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--attention-backend", choices=("fa3", "flashinfer"), default="fa3"
    )
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _directory(value: str, name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"invalid {name} {value!r}: {error}") from error
    if not path.is_dir():
        raise ValueError(f"{name} must name a directory")
    return path


def _regular_file(value: str, name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"invalid {name} {value!r}: {error}") from error
    if not path.is_file():
        raise ValueError(f"{name} must name a regular file")
    return path


def validate_arguments(args: argparse.Namespace) -> dict[str, Path | None]:
    for name in (
        "prompt_tokens",
        "decode_tokens",
        "iterations",
        "context_length",
        "max_total_tokens",
        "chunked_prefill_size",
        "page_size",
    ):
        _integer(f"--{name.replace('_', '-')}", getattr(args, name))
    _integer("--warmups", args.warmups, allow_zero=True)
    _integer("--seed", args.seed, allow_zero=True)
    if args.page_size != 16:
        raise ValueError("direct-source OrbitKV requires --page-size=16")
    if args.chunked_prefill_size % args.page_size:
        raise ValueError("--chunked-prefill-size must be page aligned")
    if args.max_total_tokens % args.page_size:
        raise ValueError("--max-total-tokens must be page aligned")
    if args.prompt_tokens + args.decode_tokens >= args.context_length:
        raise ValueError("prompt plus decode must leave one context slot unused")
    if args.prompt_tokens > args.chunked_prefill_size:
        raise ValueError(
            "--chunked-prefill-size must cover the complete prompt"
        )
    if args.mem_fraction_static is not None and not (
        0.0 < args.mem_fraction_static <= 1.0
    ):
        raise ValueError("--mem-fraction-static must be in (0, 1]")

    root = _directory(args.sglang_root, "--sglang-root")
    if not (root / "python/sglang/__init__.py").is_file():
        raise ValueError("--sglang-root is not an SGLang source checkout")
    model = _directory(args.model, "--model")
    _regular_file(str(model / "config.json"), "checkpoint config")

    manifest = None
    library = None
    if args.mode == "manager":
        if not args.manifest or not args.library:
            raise ValueError(
                "manager mode requires --runtime-manifest and --library"
            )
        manifest = _regular_file(args.manifest, "--runtime-manifest")
        library = _regular_file(args.library, "--library")
    elif args.mode == "stock":
        if args.manifest is not None or args.library is not None:
            raise ValueError(
                "stock mode forbids --runtime-manifest and --library"
            )
    else:
        raise ValueError(f"unknown mode {args.mode!r}")
    return {
        "sglang_root": root,
        "model": model,
        "manifest": manifest,
        "library": library,
    }


def verify_source(root: Path, mode: str) -> dict[str, Any]:
    contract = pinned.pinned_source_contract()
    if mode == "manager":
        checkout = pinned.validate_patched_checkout(root)
        patch = REPOSITORY_ROOT / str(contract["patch_path"])
        patch_record: dict[str, Any] = {
            "status": "applied",
            "sha256": sha256_file(patch),
        }
    elif mode == "stock":
        checkout = pinned.validate_base_checkout(root)
        patch_record = {"status": "absent", "sha256": None}
    else:
        raise RuntimeError(f"unknown mode {mode!r}")
    return {
        "root": str(checkout),
        "release": contract["release"],
        "revision": contract["revision"],
        "patch": patch_record,
    }


def configure_environment(
    args: argparse.Namespace, paths: Mapping[str, Path | None]
) -> dict[str, str]:
    if any(name == "sglang" or name.startswith("sglang.") for name in sys.modules):
        raise RuntimeError("SGLang was imported before the E2E environment froze")
    if "SGLANG_PLUGINS" in os.environ:
        raise RuntimeError("SGLANG_PLUGINS must be unset for direct-source E2E")
    for name in tuple(os.environ):
        if name.startswith("ORBITKV_"):
            os.environ.pop(name)

    root = paths.get("sglang_root")
    if not isinstance(root, Path):
        raise RuntimeError("resolved SGLang root is missing")
    sglang_python = root / "python"
    python_paths = [str(sglang_python), str(ADAPTER_SOURCE_ROOT)]
    prior = os.environ.get("PYTHONPATH")
    if prior:
        python_paths.append(prior)
    os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)
    for path in reversed((sglang_python, ADAPTER_SOURCE_ROOT)):
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
    os.environ.update(environment)
    recorded = dict(environment)
    recorded["PYTHONPATH"] = os.environ["PYTHONPATH"]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        recorded["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return recorded


def reject_installed_sglang_plugins() -> None:
    try:
        entries = list(
            importlib.metadata.entry_points(group="sglang.srt.plugins")
        )
    except Exception as error:
        raise RuntimeError(
            "cannot prove the SGLang plugin entry-point group is empty"
        ) from error
    if entries:
        names = sorted(
            str(getattr(entry, "name", "<unnamed>")) for entry in entries
        )
        raise RuntimeError(
            "direct-source E2E forbids installed SGLang plugins: "
            + ", ".join(names)
        )


def _cuda_driver_version() -> str:
    library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        library = ctypes.CDLL(library_name)
        query = library.cuDriverGetVersion
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        version = ctypes.c_int()
        status = int(query(ctypes.byref(version)))
    except (AttributeError, OSError) as error:
        raise RuntimeError(
            "cannot read the accelerator driver version"
        ) from error
    if status != 0 or version.value <= 0:
        raise RuntimeError(
            f"accelerator driver version query failed with status {status}"
        )
    raw = int(version.value)
    return f"{raw // 1000}.{raw % 1000 // 10}"


def accelerator_provenance(torch_module: Any) -> dict[str, Any]:
    """Return exact, vendor-neutral accelerator identity for this run."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise RuntimeError("PyTorch CUDA support is unavailable")
    if not cuda.is_available():
        raise RuntimeError("Engine E2E requires an available CUDA accelerator")
    try:
        device = int(cuda.current_device())
        properties = cuda.get_device_properties(device)
        capability = cuda.get_device_capability(device)
    except Exception as error:
        raise RuntimeError("cannot read accelerator properties") from error
    name = getattr(properties, "name", None)
    total_memory = getattr(properties, "total_memory", None)
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("accelerator device name is unavailable")
    if (
        isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory <= 0
    ):
        raise RuntimeError("accelerator total memory is unavailable")
    if (
        not isinstance(capability, (tuple, list))
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in capability
        )
    ):
        raise RuntimeError("accelerator compute capability is unavailable")
    runtime_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if not isinstance(runtime_version, str) or not runtime_version.strip():
        raise RuntimeError("accelerator runtime version is unavailable")
    record = {
        "device_type": "cuda",
        "device_name": name.strip(),
        "compute_capability": {
            "major": capability[0],
            "minor": capability[1],
        },
        "total_memory_bytes": total_memory,
        "runtime_version": runtime_version.strip(),
        "driver_version": _cuda_driver_version(),
    }
    if set(record) != _ACCELERATOR_FIELDS:
        raise AssertionError("accelerator provenance schema changed")
    return record


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read RuntimeManifest {path}: {error}") from error
    if len(encoded) > RUNTIME_MANIFEST_MAX_BYTES:
        raise ValueError("RuntimeManifest exceeds the byte limit")
    try:
        manifest = parse_strict_json_object(encoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid RuntimeManifest: {error}") from error
    return validate_runtime_manifest(manifest)


def _chunked_geometry(
    signature: Mapping[str, Any], class_spec: Mapping[str, Any]
) -> dict[str, int]:
    """Derive the exact resettable-arena geometry carried by a binding."""

    page_tokens = _integer(
        "RuntimeManifest page_tokens", signature.get("page_tokens")
    )
    address = class_spec.get("address")
    retirement = class_spec.get("retirement")
    if not isinstance(address, Mapping) or set(address) != {
        "kind",
        "blocks_per_epoch",
    }:
        raise ValueError(
            "exact Chunked requires resettable-arena address geometry"
        )
    blocks = _integer(
        "Chunked blocks_per_epoch", address.get("blocks_per_epoch")
    )
    if dict(address) != {
        "kind": "resettable_arena",
        "blocks_per_epoch": blocks,
    }:
        raise ValueError(
            "exact Chunked requires resettable-arena address geometry"
        )
    expected_retirement = {
        "kind": "epoch_end",
        "blocks_per_epoch": blocks,
    }
    if not isinstance(retirement, Mapping) or dict(retirement) != expected_retirement:
        raise ValueError(
            "exact Chunked requires matching EpochEnd retirement geometry"
        )
    if class_spec.get("minimum_slots_per_request") != blocks:
        raise ValueError(
            "exact Chunked minimum_slots_per_request differs from its epoch"
        )
    return {
        "page_tokens": page_tokens,
        "blocks_per_epoch": blocks,
        "chunk_tokens": page_tokens * blocks,
    }


def load_native_session_admission(
    path: Path, library_path: Path
) -> dict[str, Any]:
    manifest = _read_manifest(path)
    binding = dict(runtime_binding_from_manifest(manifest))
    config = load_runtime_manifest(
        {
            "ORBITKV_RUNTIME_MANIFEST": str(path),
            "ORBITKV_LIBRARY": str(library_path),
        }
    )
    signature = binding.get("execution_signature")
    classes = signature.get("token_classes") if isinstance(signature, Mapping) else None
    states = signature.get("token_states") if isinstance(signature, Mapping) else None
    topology = binding.get("execution_topology")
    expected_shape = NATIVE_SESSION_TOPOLOGIES.get(topology)
    if config.runtime_binding != binding:
        raise ValueError(
            "loaded RuntimeManifest binding differs from admission"
        )
    if (
        expected_shape is None
        or
        not isinstance(classes, list)
        or len(classes) != len(expected_shape)
        or not isinstance(states, list)
        or len(states) != len(expected_shape)
    ):
        raise ValueError("manager RuntimeManifest is not a native-session profile")
    actual_shape = tuple(
        (
            state.get("backend", {}).get("retention"),
            state.get("backend", {}).get("storage"),
        )
        if isinstance(state, Mapping)
        else (None, None)
        for state in states
    )
    if actual_shape != expected_shape:
        raise ValueError(
            "manager RuntimeManifest topology and token-state shape differ"
        )
    if signature.get("fixed_states") != []:
        raise ValueError(
            "manager RuntimeManifest native-session profile contains fixed state"
        )
    expected_plan_format = (
        "retention_ir" if topology == _CHUNKED_TOPOLOGY else "kv_plan"
    )
    source = manifest.get("source")
    expected_source_kind = (
        "retention_ir" if topology == _CHUNKED_TOPOLOGY else "attention_state"
    )
    if (
        config.manager_plan_format != expected_plan_format
        or config.token_reclamation.mode != "off"
        or config.fixed_states
        or not isinstance(source, Mapping)
        or source.get("kind") != expected_source_kind
    ):
        raise ValueError(
            "manager RuntimeManifest plan format differs from its native-session profile"
        )
    for index, (class_spec, state_spec, (retention, storage)) in enumerate(
        zip(classes, states, expected_shape, strict=True)
    ):
        if not isinstance(class_spec, Mapping) or not isinstance(
            state_spec, Mapping
        ):
            raise ValueError(
                f"manager RuntimeManifest token class {index} is malformed"
            )
        backend = state_spec.get("backend")
        window = backend.get("window_tokens") if isinstance(backend, Mapping) else None
        if (
            class_spec.get("name") != state_spec.get("name")
            or class_spec.get("layers") != state_spec.get("layers")
            or not isinstance(backend, Mapping)
            or backend.get("kind") != "token_slots"
            or backend.get("retention") != retention
            or backend.get("storage") != storage
            or retention == "full"
            and window is not None
            or retention == "sliding"
            and (
                isinstance(window, bool)
                or not isinstance(window, int)
                or window <= 0
            )
        ):
            raise ValueError(
                "manager RuntimeManifest topology and token-state details differ"
            )
    chunk_geometry = None
    if topology == _CHUNKED_TOPOLOGY:
        chunk_geometry = _chunked_geometry(signature, classes[0])
        chunked_class = config.chunked_class
        if (
            chunked_class is None
            or chunked_class.blocks_per_epoch
            != chunk_geometry["blocks_per_epoch"]
            or chunked_class.chunk_tokens != chunk_geometry["chunk_tokens"]
        ):
            raise ValueError(
                "exact Chunked binding and loaded runtime geometry differ"
            )
    return {
        "runtime_manifest": manifest,
        "runtime_binding": binding,
        "manager_input_fingerprint": config.plan_fingerprint,
        "lifecycle_route": "native_session",
        "cache_policy": NATIVE_SESSION_CACHE_POLICIES[topology],
        "class_ids": tuple(item.class_id for item in config.classes),
        "chunk_geometry": chunk_geometry,
    }


def _checkpoint_vocabulary(
    model: Path,
) -> tuple[int, int | None, int | None, tuple[int, ...]]:
    try:
        config = parse_strict_json_object(
            (model / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"cannot read checkpoint config: {error}") from error
    vocab_size = _integer("checkpoint vocab_size", config.get("vocab_size"))
    context = config.get("max_position_embeddings")
    if context is not None:
        context = _integer("checkpoint max_position_embeddings", context)
    attention_chunk_size = config.get("attention_chunk_size")
    if attention_chunk_size is not None:
        attention_chunk_size = _integer(
            "checkpoint attention_chunk_size", attention_chunk_size
        )
    controls = []
    for name, value in config.items():
        if not name.endswith("_token_id"):
            continue
        values = value if isinstance(value, list) else [value]
        controls.extend(
            item
            for item in values
            if isinstance(item, int) and not isinstance(item, bool)
        )
    return vocab_size, context, attention_chunk_size, tuple(controls)


def validate_chunked_execution(
    args: argparse.Namespace,
    geometry: Mapping[str, int],
    checkpoint_chunk_tokens: int | None,
) -> dict[str, int]:
    """Bind the exact Chunked workload to compiled and checkpoint geometry."""

    page_tokens = _integer(
        "compiled Chunked page_tokens", geometry.get("page_tokens")
    )
    blocks = _integer(
        "compiled Chunked blocks_per_epoch",
        geometry.get("blocks_per_epoch"),
    )
    chunk_tokens = _integer(
        "compiled Chunked chunk_tokens", geometry.get("chunk_tokens")
    )
    if chunk_tokens != page_tokens * blocks:
        raise RuntimeError(
            "compiled Chunked token and block geometry are inconsistent"
        )
    if args.attention_backend != "fa3":
        raise RuntimeError("exact Chunked E2E requires --attention-backend=fa3")
    if args.page_size != page_tokens:
        raise RuntimeError(
            "RuntimeManifest page size differs from --page-size"
        )
    if args.chunked_prefill_size != chunk_tokens:
        raise RuntimeError(
            "--chunked-prefill-size differs from compiled Chunked geometry"
        )
    if checkpoint_chunk_tokens != chunk_tokens:
        raise RuntimeError(
            "checkpoint attention_chunk_size differs from compiled Chunked geometry"
        )
    final_kv_tokens = args.prompt_tokens + args.decode_tokens - 1
    if args.prompt_tokens > chunk_tokens:
        raise RuntimeError(
            "exact Chunked prompt exceeds the compiled chunk"
        )
    if final_kv_tokens <= chunk_tokens:
        raise RuntimeError(
            "exact Chunked E2E workload must cross an epoch boundary"
        )
    return {
        "page_tokens": page_tokens,
        "blocks_per_epoch": blocks,
        "chunk_tokens": chunk_tokens,
        "final_kv_tokens": final_kv_tokens,
        "chunk_epoch_count": (final_kv_tokens + chunk_tokens - 1)
        // chunk_tokens,
    }


def deterministic_input_ids(
    *,
    prompt_tokens: int,
    vocab_size: int,
    seed: int,
    iteration: int = 0,
    forbidden: Sequence[int] = (),
) -> list[int]:
    if vocab_size <= 3:
        raise RuntimeError("checkpoint vocabulary is too small")
    blocked = {value for value in forbidden if 3 <= value < vocab_size}
    allowed = vocab_size - 3 - len(blocked)
    if allowed <= 0:
        raise RuntimeError("checkpoint vocabulary has no usable token IDs")
    if iteration >= allowed:
        raise RuntimeError(
            "requested iterations exceed the collision-free token domain"
        )

    def token_at(ordinal: int) -> int:
        token = 3 + ordinal % allowed
        for value in sorted(blocked):
            if value > token:
                break
            token += 1
        return token

    material = hashlib.shake_256(
        f"orbitkv-engine-e2e-v2:{seed}:{iteration}".encode("ascii")
    ).digest(prompt_tokens * 8)
    result = [
        token_at(int.from_bytes(material[offset : offset + 8], "little"))
        for offset in range(0, len(material), 8)
    ]
    result[0] = token_at(seed + iteration)
    return result


def engine_arguments(
    args: argparse.Namespace, model: Path, *, cache_policy: str | None = None
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_path": str(model),
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": args.context_length,
        "page_size": args.page_size,
        "attention_backend": args.attention_backend,
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": False,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": args.chunked_prefill_size,
        "prefill_max_requests": 1,
        "max_prefill_tokens": args.chunked_prefill_size,
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
    if args.mode == "manager":
        if cache_policy not in ("shared_prefix", "request_private"):
            raise ValueError(
                "manager mode requires an admitted OrbitKV cache policy"
            )
        values["disable_radix_cache"] = cache_policy == "request_private"
        values["radix_cache_backend"] = "orbitkv"
    elif cache_policy is not None:
        raise ValueError("stock mode does not accept an OrbitKV cache policy")
    return values


def _scheduler_state(info: Any, stage: str) -> dict[str, Any]:
    return contract.scheduler_state(info, stage)


def verify_engine_readback(
    info: Mapping[str, Any], engine_args: Mapping[str, Any], stage: str
) -> None:
    state = _scheduler_state(info, stage)
    fields = (
        "page_size",
        "max_total_tokens",
        "attention_backend",
        "dtype",
        "kv_cache_dtype",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "prefill_max_requests",
        "max_running_requests",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "disable_cuda_graph",
        "enable_torch_compile",
        "enable_dynamic_chunking",
        "enable_mixed_chunk",
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
    )
    mismatch = {
        name: {"expected": engine_args[name], "actual": state.get(name)}
        for name in fields
        if state.get(name) != engine_args[name]
    }
    if state.get("effective_max_running_requests_per_dp") != 1:
        mismatch["effective_max_running_requests_per_dp"] = {
            "expected": 1,
            "actual": state.get("effective_max_running_requests_per_dp"),
        }
    if mismatch:
        raise RuntimeError(f"resolved SGLang Engine contract mismatch at {stage}: {mismatch}")


def engine_readback_snapshot(
    info: Mapping[str, Any], engine_args: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    verify_engine_readback(info, engine_args, stage)
    state = _scheduler_state(info, stage)
    fields = (
        "page_size",
        "max_total_tokens",
        "attention_backend",
        "dtype",
        "kv_cache_dtype",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "prefill_max_requests",
        "max_running_requests",
        "effective_max_running_requests_per_dp",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "disable_cuda_graph",
        "enable_torch_compile",
        "enable_dynamic_chunking",
        "enable_mixed_chunk",
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
    )
    return {
        "stage": stage,
        "resolved_engine": {name: state[name] for name in fields},
        "orbitkv_manager_present": "orbitkv_manager" in state,
    }


def _nonnegative_mapping(value: Any, label: str) -> dict[str, int]:
    return contract.nonnegative_mapping(value, label)


def _session_swa_activity(
    value: Any, stage: str, *, sliding: bool
) -> dict[str, Any]:
    fields = (
        "swa_retirement_certificates", "swa_pages_reclaimed",
        "swa_wrap_events", "swa_page_reuse_events",
    )
    if not isinstance(value, Mapping) or set(value) != {
        "status", "applicable", "source", "derived", *fields
    }:
        raise RuntimeError(f"OrbitKV native-session SWA activity changed at {stage}")
    expected = {
        "status": "exposed" if sliding else "not_applicable",
        "applicable": sliding,
        "source": "native_runtime_session",
        "derived": False,
    }
    if any(
        type(value.get(name)) is not type(item) or value.get(name) != item
        for name, item in expected.items()
    ):
        raise RuntimeError(f"OrbitKV native-session SWA activity changed at {stage}")
    counters = _nonnegative_mapping(
        {name: value[name] for name in fields}, "OrbitKV SWA activity"
    )
    if not sliding and any(counters.values()):
        raise RuntimeError(f"OrbitKV non-Sliding profile reported SWA activity at {stage}")
    return {**expected, **counters}


def _disabled_pressure(value: Any, stage: str) -> dict[str, Any]:
    expected = disabled_pressure_report()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"OrbitKV session pressure must be disabled at {stage}")
    return expected


def _owner_proof(value: Any, stage: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OWNER_FIELDS:
        raise RuntimeError(f"direct-source owner proof is malformed at {stage}")
    proof = dict(value)
    if proof["module"] != _OWNER_MODULE or proof["type"] != _OWNER_TYPE:
        raise RuntimeError(f"direct-source owner identity changed at {stage}")
    for field in (
        "owner_is_process_singleton",
        "config_is_canonical",
        "allocator_owned",
        "tree_cache_owned",
    ):
        if proof[field] is not True:
            raise RuntimeError(f"direct-source owner proof failed at {stage}: {field}")
    return proof


def _runtime_proof(
    value: Any, stage: str, engine_args: Mapping[str, Any]
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"OrbitKV runtime proof is malformed at {stage}")
    scheduler = value.get("effective_scheduler")
    if not isinstance(scheduler, Mapping):
        raise RuntimeError(f"OrbitKV scheduler proof is malformed at {stage}")
    expected = {
        "max_prefill_tokens": engine_args["max_prefill_tokens"],
        "max_running_requests": 1,
        "effective_max_running_requests_per_dp": 1,
    }
    if any(scheduler.get(name) != item for name, item in expected.items()):
        raise RuntimeError(f"OrbitKV scheduler proof changed at {stage}")
    return dict(value)


def _completion_evidence(
    value: Any, stage: str, *, require_activity: bool
) -> dict[str, Any]:
    fields = {"event_backend", "pending_events", "completion_high_water"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError(f"OrbitKV completion evidence is malformed at {stage}")
    if value.get("event_backend") != "cuda_event_current_forward_stream":
        raise RuntimeError(
            f"OrbitKV session completion backend changed at {stage}"
        )
    pending = value.get("pending_events")
    if isinstance(pending, bool) or not isinstance(pending, int) or pending != 0:
        raise RuntimeError(f"OrbitKV completion events did not drain at {stage}")
    raw_high_water = value.get("completion_high_water")
    if not isinstance(raw_high_water, list):
        raise RuntimeError(f"OrbitKV completion high-water is malformed at {stage}")
    high_water = []
    domains = set()
    for item in raw_high_water:
        if not isinstance(item, Mapping) or set(item) != {"domain", "value"}:
            raise RuntimeError(f"OrbitKV completion point is malformed at {stage}")
        domain = item.get("domain")
        point = item.get("value")
        if (
            isinstance(domain, bool)
            or not isinstance(domain, int)
            or domain <= 0
            or domain in domains
            or isinstance(point, bool)
            or not isinstance(point, int)
            or point <= 0
        ):
            raise RuntimeError(f"OrbitKV completion point is invalid at {stage}")
        domains.add(domain)
        high_water.append({"domain": domain, "value": point})
    if high_water != sorted(high_water, key=lambda item: item["domain"]):
        raise RuntimeError(f"OrbitKV completion domains are not ordered at {stage}")
    if require_activity and not high_water:
        raise RuntimeError(f"OrbitKV completion frontier is empty at {stage}")
    return {
        "event_backend": value["event_backend"],
        "pending_events": pending,
        "completion_high_water": high_water,
    }


def manager_snapshot(
    info: Mapping[str, Any],
    stage: str,
    *,
    manifest_fingerprint: str,
    binding_fingerprint: str,
    manager_input_fingerprint: str,
    expected_lifecycle_route: str,
    expected_cache_policy: str,
    expected_class_ids: Sequence[int],
    expected_sliding: bool | None = None,
    require_activity: bool,
    engine_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return contract.validate_manager_snapshot(
        info, stage, wire_version=WIRE_VERSION,
        manifest_fingerprint=manifest_fingerprint,
        binding_fingerprint=binding_fingerprint,
        manager_input_fingerprint=manager_input_fingerprint,
        expected_lifecycle_route=expected_lifecycle_route,
        expected_cache_policy=expected_cache_policy,
        expected_class_ids=expected_class_ids,
        require_activity=require_activity, engine_args=engine_args or {},
        owner_proof=_owner_proof, runtime_proof=_runtime_proof,
        completion_evidence=_completion_evidence,
        session_swa_activity=lambda value, current_stage: _session_swa_activity(
            value, current_stage,
            sliding=(
                value.get("applicable") is True
                if expected_sliding is None and isinstance(value, Mapping)
                else bool(expected_sliding)
            ),
        ),
        disabled_pressure=_disabled_pressure,
    )


def require_counter_progress(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    before_counters = before.get("batch_counters")
    after_counters = after.get("batch_counters")
    if not isinstance(before_counters, Mapping) or not isinstance(
        after_counters, Mapping
    ):
        raise RuntimeError("OrbitKV counter snapshots are malformed")
    required = ("forward_events", "completion_values")
    stalled = [
        name
        for name in required
        if after_counters.get(name, -1) <= before_counters.get(name, -1)
    ]
    before_observed = before_counters.get("event_queries", 0) + before_counters.get(
        "event_waits", 0
    )
    after_observed = after_counters.get("event_queries", 0) + after_counters.get(
        "event_waits", 0
    )
    if after_observed <= before_observed:
        stalled.append("event_query_or_wait")
    if stalled:
        raise RuntimeError(
            "measured workload did not advance OrbitKV counters: "
            + ", ".join(stalled)
        )
    before_frontier = {
        item["domain"]: item["value"]
        for item in before["completion_evidence"]["completion_high_water"]
    }
    after_frontier = {
        item["domain"]: item["value"]
        for item in after["completion_evidence"]["completion_high_water"]
    }
    if not any(
        value > before_frontier.get(domain, 0)
        for domain, value in after_frontier.items()
    ):
        raise RuntimeError(
            "measured workload did not advance the OrbitKV completion frontier"
        )


def require_session_monotonic(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    for name in ("lifecycle_route", "cache_policy"):
        if before.get(name) != after.get(name):
            raise RuntimeError(f"OrbitKV {name.replace('_', ' ')} changed")
    before_counters = before.get("batch_counters")
    after_counters = after.get("batch_counters")
    if not isinstance(before_counters, Mapping) or not isinstance(
        after_counters, Mapping
    ):
        raise RuntimeError("OrbitKV counter snapshots are malformed")
    decreased = [
        name
        for name in _SESSION_COUNTER_FIELDS
        if after_counters.get(name, -1) < before_counters.get(name, -1)
    ]
    if decreased:
        raise RuntimeError(
            "OrbitKV session counters decreased: " + ", ".join(decreased)
        )
    if before.get("identities") != after.get("identities"):
        raise RuntimeError("OrbitKV session arena identity changed")
    before_frontier = {
        item["domain"]: item["value"]
        for item in before["completion_evidence"]["completion_high_water"]
    }
    after_frontier = {
        item["domain"]: item["value"]
        for item in after["completion_evidence"]["completion_high_water"]
    }
    regressed = {
        domain: {"before": value, "after": after_frontier.get(domain)}
        for domain, value in before_frontier.items()
        if after_frontier.get(domain, -1) < value
    }
    if regressed:
        raise RuntimeError(
            f"OrbitKV session completion frontier regressed: {regressed}"
        )


def require_session_baseline(snapshot: Mapping[str, Any]) -> None:
    counters = snapshot.get("batch_counters")
    frontier = snapshot.get("completion_evidence", {}).get(
        "completion_high_water"
    )
    if not isinstance(counters, Mapping) or any(counters.values()) or frontier != []:
        raise RuntimeError(
            "OrbitKV runtime session was not empty immediately after load"
        )


def require_manager_drained(snapshot: Mapping[str, Any]) -> None:
    contract.require_manager_drained(snapshot)


def post_workload_residency_for_policy(
    snapshot: Mapping[str, Any]
) -> dict[str, int]:
    return contract.post_workload_residency(snapshot)


def require_stock_absent(info: Mapping[str, Any], stage: str) -> None:
    if "orbitkv_manager" in _scheduler_state(info, stage):
        raise RuntimeError(f"stock run loaded OrbitKV at {stage}")


def _normalize_output(
    value: Any, request_id: str, decode_tokens: int
) -> tuple[list[int], int]:
    outputs = [value] if isinstance(value, Mapping) else value
    if (
        not isinstance(outputs, (list, tuple))
        or len(outputs) != 1
        or not isinstance(outputs[0], Mapping)
    ):
        raise RuntimeError("SGLang Engine must return exactly one request")
    output = outputs[0]
    ids = output.get("output_ids")
    metadata = output.get("meta_info")
    if (
        not isinstance(ids, list)
        or len(ids) != decode_tokens
        or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
    ):
        raise RuntimeError("SGLang returned invalid or incomplete output_ids")
    if not isinstance(metadata, Mapping) or metadata.get("id") != request_id:
        raise RuntimeError("SGLang returned a foreign request id")
    cached_tokens = metadata.get("cached_tokens")
    if (
        isinstance(cached_tokens, bool)
        or not isinstance(cached_tokens, int)
        or cached_tokens != 0
    ):
        raise RuntimeError("fresh E2E request observed cached prompt tokens")
    return list(ids), cached_tokens


def _flush_succeeded(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return value.get("success") is True
    return getattr(value, "success", None) is True


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency sample is empty")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_timings(
    latencies: Sequence[float], *, prompt_tokens: int, decode_tokens: int
) -> dict[str, Any]:
    if not latencies or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        for value in latencies
    ):
        raise ValueError("measured latencies must be positive")
    total_seconds = sum(latencies)
    iterations = len(latencies)
    return {
        "iteration_seconds": list(latencies),
        "median_seconds": statistics.median(latencies),
        "p95_seconds": _percentile(latencies, 0.95),
        "measured_seconds": total_seconds,
        "output_tokens_per_second": iterations * decode_tokens / total_seconds,
        "total_tokens_per_second": (
            iterations * (prompt_tokens + decode_tokens) / total_seconds
        ),
    }


def run(args: argparse.Namespace, paths: Mapping[str, Path | None]) -> dict[str, Any]:
    environment = configure_environment(args, paths)
    runtime_support.reject_legacy_manager_entrypoint()
    reject_installed_sglang_plugins()
    root = paths.get("sglang_root")
    model = paths.get("model")
    if not isinstance(root, Path) or not isinstance(model, Path):
        raise RuntimeError("resolved E2E paths are incomplete")
    source = verify_source(root, args.mode)

    admission = None
    manager_expected = None
    if args.mode == "manager":
        manifest_path = paths.get("manifest")
        library_path = paths.get("library")
        if not isinstance(manifest_path, Path) or not isinstance(library_path, Path):
            raise RuntimeError("manager artifact paths are missing")
        admission = load_native_session_admission(manifest_path, library_path)
        signature = admission["runtime_binding"]["execution_signature"]
        if signature["page_tokens"] != args.page_size:
            raise RuntimeError("RuntimeManifest page size differs from --page-size")
        manager_plan = admission["runtime_manifest"]["token_manager_plan"]
        manager_expected = {
            "manifest_fingerprint": admission["runtime_manifest"]["fingerprint"],
            "binding_fingerprint": admission["runtime_binding"]["fingerprint"],
            "manager_input_fingerprint": admission[
                "manager_input_fingerprint"
            ],
            "expected_lifecycle_route": admission["lifecycle_route"],
            "expected_cache_policy": admission["cache_policy"],
            "expected_class_ids": admission["class_ids"],
            "expected_sliding": any(
                state.get("backend", {}).get("retention") == "sliding"
                for state in signature.get("token_states", [])
                if isinstance(state, Mapping)
            ),
        }
        if not isinstance(manager_plan, Mapping):
            raise RuntimeError("RuntimeManifest token-manager plan is missing")
        LoadedLibrary(library_path)

    (
        vocab_size,
        checkpoint_context,
        checkpoint_chunk_tokens,
        forbidden,
    ) = _checkpoint_vocabulary(model)
    checkpoint = checkpoint_identity(model, "auto")
    if checkpoint["weight_bytes"] <= 0 or not checkpoint["indexed_weights_complete"]:
        raise RuntimeError("checkpoint weights are missing or incomplete")
    if checkpoint_context is not None and args.context_length > checkpoint_context:
        raise RuntimeError("--context-length exceeds checkpoint position capacity")
    if admission is not None and admission["chunk_geometry"] is not None:
        validate_chunked_execution(
            args, admission["chunk_geometry"], checkpoint_chunk_tokens
        )
    request_count = args.warmups + args.iterations
    prompts = [
        deterministic_input_ids(
            prompt_tokens=args.prompt_tokens,
            vocab_size=vocab_size,
            seed=args.seed,
            iteration=index,
            forbidden=forbidden,
        )
        for index in range(request_count)
    ]
    if len({tuple(prompt) for prompt in prompts}) != request_count:
        raise RuntimeError("deterministic request prompts are not unique")
    engine_args = engine_arguments(
        args,
        model,
        cache_policy=(
            None if admission is None else admission["cache_policy"]
        ),
    )
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.decode_tokens,
        "min_new_tokens": args.decode_tokens,
        "ignore_eos": True,
        "sampling_seed": args.seed,
    }

    import sglang as sgl
    import torch
    from sglang.srt.environ import envs

    if envs.SGLANG_USE_HND_KVCACHE.get():
        raise RuntimeError("SGLang resolved HND instead of NHD KV layout")
    package = Path(sgl.__file__).resolve(strict=True)
    expected_package = (root / "python/sglang").resolve(strict=True)
    if not package.is_relative_to(expected_package):
        raise RuntimeError("imported SGLang is outside --sglang-root")
    if sgl.__version__ != str(source["release"]).removeprefix("v"):
        raise RuntimeError("imported SGLang version differs from pinned source")
    accelerator = accelerator_provenance(torch)

    warmup_outputs = []
    outputs = []
    latencies = []
    snapshots = []
    server_snapshots = []
    request_ids = []
    post_workload_residency = None
    warmup_snapshot = None
    with sgl.Engine(**engine_args) as engine:
        after_load = engine.get_server_info()
        server_snapshots.append(
            engine_readback_snapshot(after_load, engine_args, "after_load")
        )
        if args.mode == "manager":
            assert manager_expected is not None
            load_snapshot = manager_snapshot(
                after_load,
                "after_load",
                require_activity=False,
                engine_args=engine_args,
                **manager_expected,
            )
            require_manager_drained(load_snapshot)
            require_session_baseline(load_snapshot)
            snapshots.append(load_snapshot)
        else:
            require_stock_absent(after_load, "after_load")

        for index in range(args.warmups):
            request_id = f"orbitkv-engine-e2e-warmup-{args.seed}-{index}"
            prompt = prompts[index]
            input_digest = canonical_json_sha256(prompt)
            submitted = list(prompt)
            raw = engine.generate(
                input_ids=[submitted],
                rid=[request_id],
                sampling_params=sampling_params,
            )
            if canonical_json_sha256(submitted) != input_digest:
                raise RuntimeError("SGLang mutated warmup input token IDs")
            output_ids, cached_tokens = _normalize_output(
                raw, request_id, args.decode_tokens
            )
            warmup_outputs.append(
                {
                    "warmup": index,
                    "request_id": request_id,
                    "input_ids_sha256": input_digest,
                    "output_ids": output_ids,
                    "output_ids_sha256": canonical_json_sha256(output_ids),
                    "cached_tokens": cached_tokens,
                }
            )

        after_warmup = engine.get_server_info()
        server_snapshots.append(
            engine_readback_snapshot(after_warmup, engine_args, "after_warmup")
        )
        if args.mode == "manager":
            warmup_snapshot = manager_snapshot(
                after_warmup,
                "after_warmup",
                require_activity=bool(args.warmups),
                engine_args=engine_args,
                **manager_expected,
            )
            post_workload_residency_for_policy(warmup_snapshot)
            require_session_monotonic(snapshots[-1], warmup_snapshot)
            if args.warmups == 0:
                require_session_baseline(warmup_snapshot)
            snapshots.append(warmup_snapshot)
        else:
            require_stock_absent(after_warmup, "after_warmup")

        for index in range(args.iterations):
            request_id = f"orbitkv-engine-e2e-measured-{args.seed}-{index}"
            prompt = prompts[args.warmups + index]
            input_digest = canonical_json_sha256(prompt)
            submitted = list(prompt)
            started = time.perf_counter()
            raw = engine.generate(
                input_ids=[submitted],
                rid=[request_id],
                sampling_params=sampling_params,
            )
            elapsed = time.perf_counter() - started
            if canonical_json_sha256(submitted) != input_digest:
                raise RuntimeError("SGLang mutated measured input token IDs")
            output_ids, cached_tokens = _normalize_output(
                raw, request_id, args.decode_tokens
            )
            latencies.append(elapsed)
            request_ids.append(request_id)
            outputs.append(
                {
                    "iteration": index,
                    "request_id": request_id,
                    "input_ids_sha256": input_digest,
                    "output_ids": output_ids,
                    "output_ids_sha256": canonical_json_sha256(output_ids),
                    "cached_tokens": cached_tokens,
                }
            )

        after_workload = engine.get_server_info()
        server_snapshots.append(
            engine_readback_snapshot(
                after_workload, engine_args, "after_workload"
            )
        )
        if args.mode == "manager":
            workload_snapshot = manager_snapshot(
                after_workload,
                "after_workload",
                require_activity=True,
                engine_args=engine_args,
                **manager_expected,
            )
            assert warmup_snapshot is not None
            require_session_monotonic(warmup_snapshot, workload_snapshot)
            require_counter_progress(warmup_snapshot, workload_snapshot)
            if manager_expected["expected_sliding"]:
                runtime_support.require_swa_progress(
                    warmup_snapshot["swa_activity"],
                    workload_snapshot["swa_activity"],
                )
            post_workload_residency = post_workload_residency_for_policy(
                workload_snapshot
            )
            snapshots.append(workload_snapshot)
        else:
            require_stock_absent(after_workload, "after_workload")

        flush_result = engine.flush_cache()
        if not _flush_succeeded(flush_result):
            raise RuntimeError("SGLang cache flush failed before final drain proof")
        final_info = engine.get_server_info()
        server_snapshots.append(
            engine_readback_snapshot(final_info, engine_args, "final")
        )
        if args.mode == "manager":
            final_snapshot = manager_snapshot(
                final_info,
                "final",
                require_activity=True,
                engine_args=engine_args,
                **manager_expected,
            )
            require_manager_drained(final_snapshot)
            require_session_monotonic(snapshots[-1], final_snapshot)
            snapshots.append(final_snapshot)
        else:
            require_stock_absent(final_info, "final")

    return {
        "schema": RECORD_SCHEMA,
        "mode": args.mode,
        "source": source,
        "environment": environment,
        "accelerator": accelerator,
        "model": str(model),
        "checkpoint": checkpoint,
        "runtime_manifest": (
            None if admission is None else admission["runtime_manifest"]
        ),
        "runtime_binding": (
            None if admission is None else admission["runtime_binding"]
        ),
        "engine_args": engine_args,
        "server_snapshots": server_snapshots,
        "sampling_params": sampling_params,
        "workload": {
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "seed": args.seed,
            "input_ids": {
                "warmup": prompts[: args.warmups],
                "measured": prompts[args.warmups :],
            },
            "input_ids_sha256": {
                "warmup": [
                    canonical_json_sha256(prompt)
                    for prompt in prompts[: args.warmups]
                ],
                "measured": [
                    canonical_json_sha256(prompt)
                    for prompt in prompts[args.warmups :]
                ],
            },
            "request_ids": {
                "warmup": [item["request_id"] for item in warmup_outputs],
                "measured": request_ids,
            },
        },
        "outputs": {
            "warmups": warmup_outputs,
            "iterations": outputs,
            "aggregate_sha256": canonical_json_sha256(
                {"warmups": warmup_outputs, "iterations": outputs}
            ),
        },
        "timings": summarize_timings(
            latencies,
            prompt_tokens=args.prompt_tokens,
            decode_tokens=args.decode_tokens,
        ),
        "manager": (
            None
            if args.mode == "stock"
            else {
                "wire_version": WIRE_VERSION,
                "post_workload_residency": post_workload_residency,
                "snapshots": snapshots,
            }
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # SGLang and its dependencies occasionally print progress messages.
        # Reserve stdout for the machine-readable record.
        with redirect_stdout(sys.stderr):
            paths = validate_arguments(args)
            record = run(args, paths)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
