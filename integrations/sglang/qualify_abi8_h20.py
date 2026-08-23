#!/usr/bin/env python3
"""Exact-source ABI8 H20 qualification orchestration.

The default command is host-only preflight.  GPU execution requires both the
``run`` subcommand and an explicit execution token.  This module intentionally
keeps record verification independent from the benchmark that emits records.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


INTEGRATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "src"
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import bench_canonical_manager as benchmark  # noqa: E402
from orbitkv_sglang import pinned  # noqa: E402
from orbitkv_sglang.ffi.library import (  # noqa: E402
    ABI_VERSION,
    EXACT_SYMBOL_ALLOWLIST,
    LoadedLibrary,
)

PRECHECK_SCHEMA = "orbitkv.abi8-h20-preflight.v1"
PAIR_SCHEMA = "orbitkv.abi8-h20-pair-verification.v1"
SUMMARY_SCHEMA = "orbitkv.abi8-h20-multi-epoch-summary.v1"
MANIFEST_SCHEMA = "orbitkv.abi8-h20-sealed-manifest.v1"
EXECUTION_TOKEN = "ABI8_H20_QUALIFICATION"
SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
REQUIREMENTS_SHA256 = "472d8f63cad22cd7ac4908059562bebde5e54b8d2432f750640a14d525d2fa97"
MODEL_HASHES = {
    "qwen2.5-7b": {
        "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    },
    "gpt-oss-20b": {
        "config.json": "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce",
        "model.safetensors.index.json": "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1",
    },
}
PLAN_HASHES = {
    "qwen2.5-7b": "415db5596d4bb6943c930d3cc159471e0f8911ed5572707d527152460acca130",
    "gpt-oss-20b": "cf870e5f8191f8bffd9b7bc4eac5d8c3aa8c6bc1b46f04b35c3c738c3d12e5e5",
}
FIXED_STATE_COUNTERS = frozenset(
    {
        "fixed_state_prepares",
        "fixed_state_clears",
        "fixed_state_copies",
        "fixed_state_events",
        "fixed_state_retirements",
        "fixed_state_acks",
    }
)


@dataclass(frozen=True)
class Case:
    model: str
    batch: int
    iterations: int
    chunk_tokens: int
    capacity_tokens: int
    backend: str
    profile: str

    @property
    def slug(self) -> str:
        return f"{self.model}-b{self.batch}"


CASES = (
    Case("qwen2.5-7b", 1, 1, 528, 1024, "flashinfer", "full"),
    Case("gpt-oss-20b", 1, 1, 528, 1024, "fa3", "hybrid_full_swa"),
    Case("qwen2.5-7b", 4, 5, 2112, 4096, "flashinfer", "full"),
    Case("gpt-oss-20b", 4, 5, 2112, 4096, "fa3", "hybrid_full_swa"),
)


def execution_order(epoch: int) -> tuple[str, str]:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    return ("stock", "manager") if epoch % 2 else ("manager", "stock")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _run(arguments: Sequence[str], *, cwd: Path = REPOSITORY_ROOT,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(arguments), cwd=cwd, env=env, check=True, capture_output=True,
            text=True, timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(f"command failed: {' '.join(arguments)}: {detail}") from error


def _git(*arguments: str) -> str:
    return _run(("git", *arguments)).stdout


def source_identity() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("ABI8 qualification requires a clean Git worktree")
    commit = _git("rev-parse", "HEAD").strip()
    paths = [Path(value) for value in _git("ls-files").splitlines()]
    inventory = []
    for relative in paths:
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"tracked source is missing or not regular: {relative}")
        inventory.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    return {
        "commit": commit,
        "clean": True,
        "tracked_file_count": len(inventory),
        "inventory_sha256": canonical_digest(inventory),
        "inventory": inventory,
    }


def _symbols(path: Path) -> list[str]:
    output = _run(("nm", "-D", "--defined-only", str(path))).stdout
    return sorted(
        fields[-1] for line in output.splitlines()
        if (fields := line.split()) and fields[-1].startswith("orbitkv_")
    )


def library_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    LoadedLibrary(path)
    library = ctypes.CDLL(str(path))
    library.orbitkv_abi_version.restype = ctypes.c_uint32
    actual_abi = int(library.orbitkv_abi_version())
    symbols = _symbols(path)
    expected = sorted(EXACT_SYMBOL_ALLOWLIST)
    if actual_abi != ABI_VERSION or ABI_VERSION != 8:
        raise RuntimeError(f"qualification library is ABI{actual_abi}, expected ABI8")
    if symbols != expected or len(symbols) != 40:
        raise RuntimeError("qualification library does not export exact ABI8 40 symbols")
    return {
        "path": str(path), "sha256": sha256_file(path),
        "bytes": path.stat().st_size, "abi_version": actual_abi,
        "symbols": symbols,
    }


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {observed}")
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def _check_benchmark_schema() -> None:
    from orbitkv_sglang.plugin.state import _COUNTER_NAMES

    missing = set(_COUNTER_NAMES) - set(benchmark._BATCH_COUNTER_FIELDS)
    if missing:
        raise RuntimeError(
            "benchmark census omits adapter counters: " + ", ".join(sorted(missing))
        )
    if not FIXED_STATE_COUNTERS <= set(benchmark._BATCH_COUNTER_FIELDS):
        raise RuntimeError("benchmark does not bind the ABI8 fixed-state counter schema")


def _build_library(work_dir: Path, cargo: str) -> tuple[Path, dict[str, Any]]:
    target = work_dir / "build"
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target)
    command = (cargo, "build", "--release", "--locked", "--manifest-path",
               str(REPOSITORY_ROOT / "crates/orbitkv-ffi/Cargo.toml"))
    _run(command, env=env)
    return target / "release/liborbitkv_ffi.so", {
        "command": list(command),
        "cargo_version": _run((cargo, "--version")).stdout.strip(),
        "cargo_target_dir": str(target),
    }


def _verify_python_environment(python: Path, requirements: Path) -> dict[str, Any]:
    _run((str(python), "-m", "pip", "check"))
    frozen = _run((str(python), "-m", "pip", "freeze", "--all")).stdout
    expected = requirements.read_text(encoding="utf-8")

    def normalize(lines: list[str]) -> tuple[list[str], str]:
        editable = [line for line in lines if line.startswith("-e git+") and "#egg=orbitkv_sglang" in line]
        if len(editable) != 1:
            raise RuntimeError(
                "Python environment must contain exactly one editable orbitkv-sglang"
            )
        return ["-e <active-orbitkv-source>#egg=orbitkv_sglang" if line == editable[0] else line for line in lines], editable[0]

    expected_lines, expected_editable = normalize(expected.splitlines())
    frozen_lines, active_editable = normalize(frozen.splitlines())
    if frozen_lines != expected_lines:
        difference = "\n".join(
            difflib.unified_diff(
                expected_lines, frozen_lines,
                fromfile=str(requirements), tofile="pip freeze --all", lineterm="",
            )
        )
        raise RuntimeError(
            "active Python environment differs from requirements lock:\n" + difference
        )
    return {
        "executable": str(python.resolve(strict=True)),
        "normalized_freeze_sha256": canonical_digest(frozen_lines),
        "active_editable": active_editable,
        "locked_editable": expected_editable,
    }


def _sglang_checkout_identity(root: Path, mode: str) -> dict[str, str]:
    root = root.resolve(strict=True)
    identity = benchmark.verify_sglang_source(root, mode)
    revision = _run(("git", "-C", str(root), "rev-parse", "HEAD")).stdout.strip()
    tag = _run(("git", "-C", str(root), "describe", "--tags", "--exact-match", "HEAD")).stdout.strip()
    remote = _run(("git", "-C", str(root), "remote", "get-url", "origin")).stdout.strip()
    if revision != SGLANG_REVISION or tag != "v0.5.17":
        raise RuntimeError(f"{mode} checkout is not exact official SGLang v0.5.17")
    if remote != "https://github.com/sgl-project/sglang.git":
        raise RuntimeError(f"{mode} checkout does not use the official SGLang remote")
    return {**identity, "tag": tag, "remote": remote}


def _input_identity(args: argparse.Namespace) -> dict[str, Any]:
    requirements = _require_hash(args.requirements, REQUIREMENTS_SHA256, "requirements lock")
    models: dict[str, Any] = {}
    plans: dict[str, Any] = {}
    for name, model in (("qwen2.5-7b", args.qwen_model), ("gpt-oss-20b", args.gpt_model)):
        model = model.resolve(strict=True)
        models[name] = {
            filename: _require_hash(model / filename, digest, f"{name} {filename}")
            for filename, digest in MODEL_HASHES[name].items()
        }
        index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if not shard_names:
            raise RuntimeError(f"{name} index contains no weight shards")
        shards = []
        for filename in shard_names:
            if Path(filename).name != filename:
                raise RuntimeError(f"{name} index contains an unsafe shard path")
            shard = model / filename
            shards.append({
                "filename": filename, "size": shard.stat().st_size,
                "sha256": sha256_file(shard),
            })
        models[name]["weight_shards"] = shards
        models[name]["weight_shards_sha256"] = canonical_digest(shards)
        models[name]["root"] = str(model)
    for name, plan in (("qwen2.5-7b", args.qwen_plan), ("gpt-oss-20b", args.gpt_plan)):
        plans[name] = _require_hash(plan, PLAN_HASHES[name], f"{name} plan")
    return {"requirements": requirements, "models": models, "plans": plans}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing work directory: {work_dir}")
    source = source_identity()
    _check_benchmark_schema()
    if pinned.SUPPORTED_SGLANG_REVISION != SGLANG_REVISION:
        raise RuntimeError("adapter pinned SGLang revision drifted")
    stock = pinned.validate_base_checkout(args.stock_root)
    manager = pinned.validate_patched_checkout(args.manager_root)
    stock_identity = _sglang_checkout_identity(stock, "stock")
    manager_identity = _sglang_checkout_identity(manager, "manager")
    pinned_contract = benchmark.verify_pinned_module_constants()
    manager_entrypoint = benchmark.verify_manager_entrypoint()
    stock_selection = benchmark.verify_stock_plugin_selection()
    inputs = _input_identity(args)
    python_identity = _verify_python_environment(args.python, args.requirements)
    work_dir.mkdir(parents=True)
    library, build = _build_library(work_dir, args.cargo)
    record = {
        "schema": PRECHECK_SCHEMA, "source": source,
        "benchmark": {"path": str(Path(benchmark.__file__).resolve()),
                      "sha256": sha256_file(Path(benchmark.__file__).resolve()),
                      "record_schema": benchmark.RECORD_SCHEMA},
        "library": library_identity(library),
        "build": build,
        "sglang": {
            "release": pinned.SUPPORTED_SGLANG_RELEASE, "revision": SGLANG_REVISION,
            "stock_root": str(stock), "manager_root": str(manager),
            "stock": stock_identity, "manager": manager_identity,
            "pinned_contract": pinned_contract,
            "manager_entrypoint": manager_entrypoint,
            "stock_plugin_selection": stock_selection,
        },
        "inputs": inputs,
        "python": python_identity,
        "status": "host_preflight_passed_gpu_not_initialized",
    }
    path = work_dir / "preflight.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load JSON record {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON record is not an object: {path}")
    return value


def _normalized_engine_args(record: dict[str, Any]) -> dict[str, Any]:
    values = dict(record["engine_args"])
    mode = record.get("mode")
    if mode == "manager":
        if values.pop("radix_cache_backend", None) != "orbitkv":
            raise RuntimeError("manager record does not select OrbitKV radix backend")
    elif mode == "stock":
        if "radix_cache_backend" in values:
            raise RuntimeError("stock record unexpectedly selects a radix backend")
    else:
        raise RuntimeError("pair has invalid mode")
    return values


def _record_pair_contract(record: dict[str, Any]) -> dict[str, Any]:
    seed = record.get("prefix_seed", {})
    if not isinstance(seed, dict):
        raise RuntimeError("record prefix seed is malformed")
    prefix_contract = {
        name: seed.get(name)
        for name in (
            "requests", "prompt_tokens", "input_token_digest_sha256",
            "sampling_params", "included_in_iteration_timing",
            "included_in_measured_output_pairing",
        )
    }
    return {
        "checkpoint_identity_sha256": record.get("checkpoint_identity_sha256"),
        "attention_contract_sha256": canonical_digest(record.get("checkpoint_contract")),
        "engine_args": _normalized_engine_args(record),
        "allowed_implementation_difference": benchmark.PAIR_IMPLEMENTATION_DIFFERENCE,
        "sampling_params": record.get("sampling_params"),
        "prefix_seed": prefix_contract,
        "workload": record.get("workload"),
        "capacity_readback": record.get("capacity_readback"),
    }


def verify_pair_records(stock: dict[str, Any], manager: dict[str, Any]) -> dict[str, Any]:
    if stock.get("schema") != benchmark.RECORD_SCHEMA or manager.get("schema") != benchmark.RECORD_SCHEMA:
        raise RuntimeError("pair does not use the active benchmark record schema")
    if stock.get("mode") != "stock" or manager.get("mode") != "manager":
        raise RuntimeError("pair ordering must be stock then manager")
    if stock.get("manager") is not None or not isinstance(manager.get("manager"), dict):
        raise RuntimeError("pair manager presence is invalid")
    current_harness = sha256_file(Path(benchmark.__file__).resolve())
    current_adapter = benchmark._adapter_identity()
    for mode, record in (("stock", stock), ("manager", manager)):
        source = record.get("source_identity")
        if not isinstance(source, dict):
            raise RuntimeError(f"{mode} source identity is missing")
        if source.get("release") != "v0.5.17" or source.get("revision") != SGLANG_REVISION:
            raise RuntimeError(f"{mode} source is not pinned SGLang v0.5.17")
        if source.get("harness_sha256") != current_harness:
            raise RuntimeError(f"{mode} record does not bind the active harness")
        if source.get("adapter") != current_adapter:
            raise RuntimeError(f"{mode} record does not bind the active adapter inventory")
    if stock["source_identity"].get("python_source_contract") != "clean_pinned_head":
        raise RuntimeError("stock record does not bind the pristine checkout")
    if manager["source_identity"].get("python_source_contract") != "pinned_head_plus_canonical_loader_patch":
        raise RuntimeError("manager record does not bind the reviewed loader patch")
    if stock["source_identity"].get("library") is not None or stock["source_identity"].get("plan") is not None:
        raise RuntimeError("stock record unexpectedly binds manager artifacts")
    manager_library = manager["source_identity"].get("library")
    manager_plan = manager["source_identity"].get("plan")
    if manager_library != manager["manager"].get("library"):
        raise RuntimeError("manager library identities disagree")
    if manager_plan != manager["manager"].get("plan", {}).get("artifact"):
        raise RuntimeError("manager plan identities disagree")
    equal_fields = (
        "checkpoint", "checkpoint_identity_sha256", "checkpoint_contract", "sampling_params",
        "workload", "capacity_readback",
        "output_token_digest_sha256", "output_request_digests_sha256",
        "request_traces", "completed_requests", "completion_tokens",
    )
    mismatches = [name for name in equal_fields if stock.get(name) != manager.get(name)]
    if mismatches:
        raise RuntimeError("pair fields differ: " + ", ".join(mismatches))
    if any(
        record.get("checkpoint_identity_sha256")
        != canonical_digest(record.get("checkpoint"))
        for record in (stock, manager)
    ):
        raise RuntimeError("checkpoint identity digest is invalid")
    if _normalized_engine_args(stock) != _normalized_engine_args(manager):
        raise RuntimeError("normalized engine arguments differ")
    stock_pair = stock.get("pairing", {})
    manager_pair = manager.get("pairing", {})
    expected_contract = _record_pair_contract(stock)
    if expected_contract != _record_pair_contract(manager):
        raise RuntimeError("reconstructed normalized pairing contracts differ")
    if (stock_pair.get("contract") != expected_contract
            or manager_pair.get("contract") != expected_contract):
        raise RuntimeError("normalized pairing contracts differ")
    expected_key = canonical_digest(expected_contract)
    if stock_pair.get("pair_key_sha256") != expected_key or manager_pair.get("pair_key_sha256") != expected_key:
        raise RuntimeError("pair key does not bind the normalized contract")
    final = manager["manager"].get("final_census")
    if not isinstance(final, dict) or final.get("abi_version") != 8:
        raise RuntimeError("manager final census is not ABI8")
    stats = final.get("manager_stats", {})
    dirty = {name: stats.get(name) for name in (
        "active_requests", "active_snapshots", "active_prefixes",
        "active_pages", "pending_reclamations", "quarantined_pages",
    ) if stats.get(name) != 0}
    if dirty:
        raise RuntimeError(f"manager final census did not drain: {dirty}")
    counters = final.get("batch_counters", {})
    if any(counters.get(name) != 0 for name in FIXED_STATE_COUNTERS):
        raise RuntimeError("TokenKV qualification exercised fixed-state counters")
    profile = manager["checkpoint_contract"].get("attention_profile")
    swa = final.get("swa_activity", {})
    swa_fields = ("swa_retirement_certificates", "swa_pages_reclaimed", "swa_wrap_events")
    if profile == "full":
        if swa.get("status") != "not_applicable" or any(swa.get(name) != 0 for name in swa_fields):
            raise RuntimeError("Full pair has invalid SWA telemetry")
    elif profile == "hybrid_full_swa":
        after_load = manager["manager"].get("after_load", {}).get("swa_activity", {})
        if swa.get("status") != "exposed" or any(swa.get(name, 0) <= after_load.get(name, 0) for name in swa_fields):
            raise RuntimeError("Hybrid pair did not advance all SWA counters")
    else:
        raise RuntimeError("ABI8 Full/Full+SWA verifier rejects this profile")
    return {
        "schema": PAIR_SCHEMA, "status": "passed", "profile": profile,
        "batch_size": manager["workload"]["requests"],
        "iterations": manager["workload"]["iterations"],
        "pair_key_sha256": expected_key,
        "stock_output_sha256": stock["output_token_digest_sha256"],
        "manager_output_sha256": manager["output_token_digest_sha256"],
        "stock_iteration_seconds": stock["iteration_seconds"],
        "manager_iteration_seconds": manager["iteration_seconds"],
    }


def verify_pair_files(
    stock_path: Path, manager_path: Path, preflight: dict[str, Any] | None = None
) -> dict[str, Any]:
    stock = _load(stock_path)
    manager = _load(manager_path)
    result = verify_pair_records(stock, manager)
    result["stock_record"] = str(stock_path.resolve())
    result["manager_record"] = str(manager_path.resolve())
    if preflight is not None:
        model_name = (
            "qwen2.5-7b"
            if manager["checkpoint_contract"]["attention_profile"] == "full"
            else "gpt-oss-20b"
        )
        expected_plan = preflight["inputs"]["plans"][model_name]
        library_fields = ("path", "sha256", "bytes")
        if any(
            manager["source_identity"]["library"].get(name)
            != preflight["library"].get(name)
            for name in library_fields
        ):
            raise RuntimeError("pair library differs from preflight")
        if any(
            manager["source_identity"]["plan"].get(name) != expected_plan.get(name)
            for name in library_fields
        ):
            raise RuntimeError("pair plan differs from preflight")
        model_identity = preflight["inputs"]["models"][model_name]
        for mode, record in (("stock", stock), ("manager", manager)):
            source = record["source_identity"]
            expected_root = preflight["sglang"][f"{mode}_root"]
            expected_contract = (
                "clean_pinned_head"
                if mode == "stock"
                else "pinned_head_plus_canonical_loader_patch"
            )
            if (
                source.get("root") != expected_root
                or source.get("release") != preflight["sglang"]["release"]
                or source.get("revision") != preflight["sglang"]["revision"]
                or source.get("pinned_contract")
                != preflight["sglang"]["pinned_contract"]
                or source.get("python_source_contract") != expected_contract
                or source.get("loader")
                != preflight["sglang"][mode].get("loader")
                or record.get("runtime_identity", {}).get("python_executable")
                != preflight["python"]["executable"]
            ):
                raise RuntimeError(f"{mode} record differs from preflight identity")
        checkpoint = manager["checkpoint"]
        if checkpoint["config_sha256"] != model_identity["config.json"]["sha256"]:
            raise RuntimeError("pair checkpoint differs from preflight")
        index_files = checkpoint.get("index_files", [])
        if (len(index_files) != 1
                or index_files[0].get("name") != "model.safetensors.index.json"
                or index_files[0].get("sha256") != model_identity["model.safetensors.index.json"]["sha256"]):
            raise RuntimeError("pair checkpoint index differs from preflight")
        expected_shards = model_identity["weight_shards"]
        if checkpoint.get("indexed_weight_files") != [item["filename"] for item in expected_shards]:
            raise RuntimeError("pair indexed shard names differ from preflight")
        recorded_shards = [
            {
                "filename": item.get("name"),
                "size": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for item in checkpoint.get("weight_files", [])
        ]
        if recorded_shards != expected_shards:
            raise RuntimeError("pair shard inventory differs from preflight")
        result["input_identity"] = {
            "source_commit": preflight["source"]["commit"],
            "source_inventory_sha256": preflight["source"]["inventory_sha256"],
            "library_sha256": preflight["library"]["sha256"],
            "plan_sha256": expected_plan["sha256"],
            "model_config_sha256": preflight["inputs"]["models"][model_name]["config.json"]["sha256"],
            "model_index_sha256": preflight["inputs"]["models"][model_name]["model.safetensors.index.json"]["sha256"],
            "model_weight_shards_sha256": preflight["inputs"]["models"][model_name]["weight_shards_sha256"],
            "python_environment_sha256": canonical_digest(preflight["python"]),
            "sglang_checkouts_sha256": canonical_digest(
                {name: preflight["sglang"][name] for name in ("stock", "manager")}
            ),
        }
    return result


def _write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_environment(python: str) -> dict[str, str]:
    python_bin = str(Path(python).resolve().parent)
    return {
        "HOME": os.environ.get("HOME", "/root"), "USER": os.environ.get("USER", "root"),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "root")),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": f"{python_bin}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CUDA_HOME": "/usr/local/cuda", "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "SGLANG_SWA_EVICTION_INTERVAL": "128",
    }


def _assert_idle_h20() -> None:
    gpu = _run(("nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader,nounits")).stdout.strip()
    if "H20" not in gpu or "\n" in gpu:
        raise RuntimeError(f"CUDA device 0 is not exactly one H20: {gpu!r}")
    processes = _run(("nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits")).stdout.strip()
    if processes:
        raise RuntimeError(f"H20 has active compute processes: {processes}")


def _validate_preflight(record: dict[str, Any]) -> None:
    if record.get("schema") != PRECHECK_SCHEMA:
        raise RuntimeError("work directory has no ABI8 preflight record")
    current = source_identity()
    if current != record.get("source"):
        raise RuntimeError("active source differs from preflight identity")
    library = library_identity(Path(record["library"]["path"]))
    if library != record["library"]:
        raise RuntimeError("ABI8 library differs from preflight identity")
    benchmark_path = Path(record["benchmark"]["path"])
    if sha256_file(benchmark_path) != record["benchmark"]["sha256"]:
        raise RuntimeError("benchmark differs from preflight identity")
    python_identity = _verify_python_environment(
        Path(record["python"]["executable"]),
        Path(record["inputs"]["requirements"]["path"]),
    )
    if python_identity != record["python"]:
        raise RuntimeError("Python environment differs from preflight identity")
    for name, model in record["inputs"]["models"].items():
        root = Path(model["root"])
        for filename in ("config.json", "model.safetensors.index.json"):
            _require_hash(root / filename, model[filename]["sha256"], f"{name} {filename}")
        observed = []
        for shard in model["weight_shards"]:
            path = root / shard["filename"]
            if path.stat().st_size != shard["size"] or sha256_file(path) != shard["sha256"]:
                raise RuntimeError(f"{name} shard differs from preflight: {path.name}")
            observed.append(shard)
        if canonical_digest(observed) != model["weight_shards_sha256"]:
            raise RuntimeError(f"{name} shard inventory digest differs from preflight")
    for name, plan in record["inputs"]["plans"].items():
        _require_hash(Path(plan["path"]), plan["sha256"], f"{name} plan")
    requirements = record["inputs"]["requirements"]
    _require_hash(Path(requirements["path"]), requirements["sha256"], "requirements lock")
    stock = pinned.validate_base_checkout(record["sglang"]["stock_root"])
    manager = pinned.validate_patched_checkout(record["sglang"]["manager_root"])
    if _sglang_checkout_identity(stock, "stock") != record["sglang"]["stock"]:
        raise RuntimeError("stock SGLang checkout differs from preflight")
    if _sglang_checkout_identity(manager, "manager") != record["sglang"]["manager"]:
        raise RuntimeError("manager SGLang checkout differs from preflight")


def _case_command(args: argparse.Namespace, pre: dict[str, Any], case: Case, mode: str) -> list[str]:
    inputs = pre["inputs"]
    command = [
        pre["python"]["executable"], str(Path(pre["benchmark"]["path"])), "--mode", mode,
        "--sglang-root", pre["sglang"][f"{mode}_root"],
        "--model", inputs["models"][case.model]["root"],
        "--requests", str(case.batch), "--max-running-requests", str(case.batch),
        "--prompt-tokens", "513", "--decode-tokens", "33",
        "--iterations", str(case.iterations), "--chunked-prefill-size", str(case.chunk_tokens),
        "--context-length", "1024", "--max-total-tokens", str(case.capacity_tokens),
        "--attention-backend", case.backend, "--seed", str(args.seed),
    ]
    if mode == "manager":
        command.extend(("--plan", inputs["plans"][case.model]["path"],
                        "--library", pre["library"]["path"]))
    return command


def _run_record(command: Sequence[str], output: Path, stderr: Path) -> None:
    if output.exists() or stderr.exists() or output.with_suffix(".json.partial").exists():
        raise RuntimeError(f"refusing to overwrite record for {output.stem}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".json.partial")
    with partial.open("w", encoding="utf-8") as stdout, stderr.open("w", encoding="utf-8") as errors:
        subprocess.run(list(command), check=True, env=_fresh_environment(command[0]), stdout=stdout, stderr=errors)
    _load(partial)
    partial.rename(output)


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute != EXECUTION_TOKEN:
        raise RuntimeError(f"run requires --execute {EXECUTION_TOKEN}")
    work_dir = args.work_dir.resolve(strict=True)
    pre = _load(work_dir / "preflight.json")
    _validate_preflight(pre)
    _assert_idle_h20()
    pair_results = []
    selected = [case for case in CASES if args.phase == "all" or case.batch == int(args.phase[1:])]
    for epoch in range(1, args.epochs + 1):
        if args.phase == "b4":
            b1_root = work_dir / "records" / f"epoch-{epoch:03d}"
            for case in (item for item in CASES if item.batch == 1):
                verify_pair_files(
                    b1_root / f"{case.slug}-stock.json",
                    b1_root / f"{case.slug}-manager.json",
                )
        for batch in (1, 4):
            for case in (item for item in selected if item.batch == batch):
                records = work_dir / "records" / f"epoch-{epoch:03d}"
                paths = {mode: records / f"{case.slug}-{mode}.json" for mode in ("stock", "manager")}
                order = execution_order(epoch)
                for mode in order:
                    _validate_preflight(pre)
                    output = paths[mode]
                    _run_record(
                        _case_command(args, pre, case, mode), output,
                        output.with_suffix(".stderr.log"),
                    )
                    _assert_idle_h20()
                _validate_preflight(pre)
                stock, manager = paths["stock"], paths["manager"]
                pair = verify_pair_files(stock, manager, pre)
                pair["epoch"] = epoch
                pair["execution_order"] = list(order)
                pair_path = records / f"{case.slug}-pair.json"
                _write_new(pair_path, pair)
                pair_results.append(pair)
                _assert_idle_h20()
    summary = summarize_pairs(pair_results)
    _write_new(work_dir / f"summary-{args.phase}-{args.epochs}-epochs.json", summary)
    return summary


def _completed_pairs(work_dir: Path, preflight: dict[str, Any]) -> list[dict[str, Any]]:
    epoch_dirs = sorted((work_dir / "records").glob("epoch-*"))
    if not epoch_dirs:
        raise RuntimeError("seal requires at least one completed epoch")
    pairs: list[dict[str, Any]] = []
    for expected_epoch, epoch_dir in enumerate(epoch_dirs, 1):
        if epoch_dir.name != f"epoch-{expected_epoch:03d}" or not epoch_dir.is_dir():
            raise RuntimeError("qualification epochs must be contiguous from epoch-001")
        for case in CASES:
            stock = epoch_dir / f"{case.slug}-stock.json"
            manager = epoch_dir / f"{case.slug}-manager.json"
            pair_path = epoch_dir / f"{case.slug}-pair.json"
            pair = verify_pair_files(stock, manager, preflight)
            pair["epoch"] = expected_epoch
            pair["execution_order"] = list(execution_order(expected_epoch))
            if _load(pair_path) != pair:
                raise RuntimeError(f"stored pair verification differs: {pair_path}")
            pairs.append(pair)
    return pairs


def _relative_artifact_hashes(root: Path, excluded: frozenset[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def seal(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing seal directory: {output_dir}")
    pre = _load(work_dir / "preflight.json")
    _validate_preflight(pre)
    for name, identity in pre["inputs"]["plans"].items():
        _require_hash(Path(identity["path"]), identity["sha256"], f"{name} plan")
    requirements_identity = pre["inputs"]["requirements"]
    _require_hash(
        Path(requirements_identity["path"]),
        requirements_identity["sha256"],
        "requirements lock",
    )
    if sha256_file(Path(__file__).resolve()) != next(
        item["sha256"]
        for item in pre["source"]["inventory"]
        if item["path"] == "integrations/sglang/qualify_abi8_h20.py"
    ):
        raise RuntimeError("qualification runner differs from preflight source inventory")
    pairs = _completed_pairs(work_dir, pre)
    calculated_summary = summarize_pairs(pairs)
    summaries = sorted(work_dir.glob("summary-*.json"))
    if not summaries or not any(_load(path) == calculated_summary for path in summaries):
        raise RuntimeError("no completed summary matches independently verified pairs")

    output_dir.mkdir(parents=True)
    shutil.copy2(work_dir / "preflight.json", output_dir / "preflight.json")
    shutil.copytree(work_dir / "records", output_dir / "records")
    (output_dir / "summary.json").write_text(
        json.dumps(calculated_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    qualification = output_dir / "qualification"
    (qualification / "build").mkdir(parents=True)
    (qualification / "plans").mkdir()
    (qualification / "source").mkdir()
    shutil.copy2(pre["library"]["path"], qualification / "build/liborbitkv_ffi.so")
    for name, identity in pre["inputs"]["plans"].items():
        shutil.copy2(identity["path"], qualification / "plans" / f"{name}.json")
    shutil.copy2(pre["inputs"]["requirements"]["path"], qualification / "requirements.lock.txt")
    shutil.copy2(Path(__file__).resolve(), qualification / "source/qualify_abi8_h20.py")
    shutil.copy2(Path(benchmark.__file__).resolve(), qualification / "source/bench_canonical_manager.py")
    readme = (
        "# OrbitKV ABI8 SGLang Full and Full+SWA Prefix qualification\n\n"
        "Status: ABI8 SGLang Full/Full+SWA Prefix correctness qualified; "
        "performance pending.\n\n"
        f"Source commit: `{pre['source']['commit']}`\n\n"
        f"Verified pairs: {len(pairs)} across {len(pairs) // len(CASES)} epoch(s).\n\n"
        "Cases: Qwen2.5-7B Full/FlashInfer B1+B4; GPT-OSS-20B "
        "Full+SWA/FA3 B1+B4.\n\n"
        "Excluded: token relocation, MLA, fixed-state, overlap scheduling, "
        "CUDA Graphs, speculation, distributed execution, and performance qualification.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    artifacts = _relative_artifact_hashes(
        output_dir, frozenset({"manifest.json", "SHA256SUMS"})
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "qualification_status": "abi8_sglang_full_full_swa_prefix_correctness_qualified_performance_pending",
        "scope": {
            "cases": [
                {"model": case.model, "profile": case.profile,
                 "attention_backend": case.backend, "batch_size": case.batch}
                for case in CASES
            ],
            "excluded": [
                "token_relocation", "mla", "fixed_state",
                "overlap_scheduling", "cuda_graphs", "speculation",
                "distributed_execution", "performance_qualification",
            ],
        },
        "abi_version": 8, "exact_symbol_count": 40,
        "source_commit": pre["source"]["commit"],
        "source_inventory_sha256": pre["source"]["inventory_sha256"],
        "library_sha256": pre["library"]["sha256"],
        "input_hashes": {
            "requirements_sha256": pre["inputs"]["requirements"]["sha256"],
            "plans": {name: value["sha256"] for name, value in pre["inputs"]["plans"].items()},
            "models": {
                name: {
                    "config_sha256": value["config.json"]["sha256"],
                    "index_sha256": value["model.safetensors.index.json"]["sha256"],
                    "weight_shards_sha256": value["weight_shards_sha256"],
                }
                for name, value in pre["inputs"]["models"].items()
            },
        },
        "epoch_count": len(pairs) // len(CASES), "pair_count": len(pairs),
        "performance_go": False, "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums = _relative_artifact_hashes(output_dir, frozenset({"SHA256SUMS"}))
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sums.items()), encoding="utf-8"
    )
    return manifest


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize_pairs(pairs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for pair in pairs:
        if pair.get("schema") != PAIR_SCHEMA or pair.get("status") != "passed":
            raise RuntimeError("summary input contains an unverified pair")
        groups.setdefault((pair["profile"], pair["batch_size"]), []).append(pair)
    if not groups:
        raise RuntimeError("summary requires at least one verified pair")
    summaries = []
    for (profile, batch), values in sorted(groups.items()):
        stock = [x for item in values for x in item["stock_iteration_seconds"]]
        manager = [x for item in values for x in item["manager_iteration_seconds"]]
        if not stock or not manager or any(x <= 0 for x in stock + manager):
            raise RuntimeError("summary timings must be positive")
        stock_mean = statistics.fmean(stock)
        manager_mean = statistics.fmean(manager)
        summaries.append({
            "profile": profile, "batch_size": batch, "epoch_count": len(values),
            "sample_count_per_mode": len(stock),
            "stock_seconds": {"mean": stock_mean, "p50": statistics.median(stock), "p95": _percentile(stock, .95), "p99": _percentile(stock, .99)},
            "manager_seconds": {"mean": manager_mean, "p50": statistics.median(manager), "p95": _percentile(manager, .95), "p99": _percentile(manager, .99)},
            "manager_over_stock_percent": (manager_mean / stock_mean - 1.0) * 100.0,
            "performance_go": False,
        })
    return {"schema": SUMMARY_SCHEMA, "pair_count": sum(len(v) for v in groups.values()), "groups": summaries, "performance_go": False}


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")
    pre = sub.add_parser("preflight")
    _common_paths(pre)
    pre.add_argument("--python", type=Path, default=REPOSITORY_ROOT / ".venv-sglang-v0517/bin/python")
    pre.add_argument("--cargo", default="cargo")
    pre.add_argument("--stock-root", type=Path, default=REPOSITORY_ROOT / ".qualification/sglang-v0517-stock")
    pre.add_argument("--manager-root", type=Path, default=REPOSITORY_ROOT / ".qualification/sglang-v0517-manager")
    pre.add_argument("--requirements", type=Path, default=REPOSITORY_ROOT / ".qualification/requirements-v0.5.17.lock.txt")
    pre.add_argument("--qwen-model", type=Path, default=Path("/workspace/models/qwen2.5-7b-instruct"))
    pre.add_argument("--gpt-model", type=Path, default=Path("/workspace/models/gpt-oss-20b"))
    pre.add_argument("--qwen-plan", type=Path, default=REPOSITORY_ROOT / ".qualification/plans/qwen2.5-7b-full-page16-bf16.json")
    pre.add_argument("--gpt-plan", type=Path, default=REPOSITORY_ROOT / ".qualification/plans/gpt-oss-20b-hybrid-page16-bf16.json")
    run = sub.add_parser("run")
    _common_paths(run)
    run.add_argument("--execute", required=True)
    run.add_argument("--phase", choices=("b1", "b4", "all"), default="all")
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--seed", type=int, default=20260820)
    pair = sub.add_parser("verify-pair")
    pair.add_argument("stock", type=Path)
    pair.add_argument("manager", type=Path)
    pair.add_argument("--output", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("pairs", nargs="+", type=Path)
    summary.add_argument("--output", type=Path)
    seal_parser = sub.add_parser("seal")
    _common_paths(seal_parser)
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] not in {"preflight", "run", "verify-pair", "summarize", "seal"}:
        raw.insert(0, "preflight")
    args = parser.parse_args(raw)
    try:
        if args.action == "preflight":
            result = preflight(args)
        elif args.action == "run":
            if args.epochs <= 0:
                raise RuntimeError("--epochs must be positive")
            result = run_matrix(args)
        elif args.action == "verify-pair":
            result = verify_pair_files(args.stock, args.manager)
            if args.output:
                _write_new(args.output, result)
        elif args.action == "summarize":
            result = summarize_pairs(_load(path) for path in args.pairs)
            if args.output:
                _write_new(args.output, result)
        elif args.action == "seal":
            result = seal(args)
        else:
            raise RuntimeError(f"unknown action: {args.action}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
