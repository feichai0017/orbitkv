#!/usr/bin/env python3
"""Fail-closed verifier for the Qwen3.8 FP8 H20 diagnostic archive.

This archive is deliberately not a seal or a qualification result.  Pair
semantics are re-evaluated by the current trusted checkout's
``integrations/sglang/qualify_abi8_h20.py``.  The run-bound harness hash and
adapter inventory are supplied through that verifier's offline-evidence
injection points because the OrbitKV worktree was dirty when these records
were produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

sys.dont_write_bytecode = True

ROOT = Path(__file__).absolute().parent
REPOSITORY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.sglang import qualify_abi8_h20 as qualification  # noqa: E402

MANIFEST_SCHEMA = "orbitkv.abi8-h20-qwen38-fp8-diagnostic-manifest.v1"
SUMMARY_SCHEMA = "orbitkv.abi8-h20-qwen38-fp8-diagnostic-summary.v1"
VERIFICATION_SCHEMA = "orbitkv.abi8-h20-qwen38-fp8-diagnostic-verification.v1"
RECORD_SCHEMA = "orbitkv.sglang-v0517-abi8-single-run.v3"
PAIR_SCHEMA = "orbitkv.abi8-h20-pair-verification.v1"

MODEL_REPOSITORY = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
MODEL_CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
MODEL_INDEX_SHA256 = "f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2"
MODEL_CHECKPOINT_SHA256 = "20dc2e14e5b2718f8d9bae6af8bf9be585f8b1a7a0a248562271d48d08fc249e"
MODEL_WEIGHT_INVENTORY_SHA256 = "0174512f33206947585ff17ad2e7e5507da30ab06fd103a20f298ed94df90cfe"
MODEL_SHARDS = 66
MODEL_WEIGHT_BYTES = 30_866_866_928
MODEL_TENSORS = 1_606
SGLANG_RELEASE = "v0.5.17"
SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
GPU_NAME = "NVIDIA H20"
GPU_UUID = "GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3"
LIBRARY_SHA256 = "fcff90dabd79e3ef69725cfff93f350db922acad9f3c58f6fb6086da7145ae23"
LIBRARY_BYTES = 1_719_056
TOKEN_PLAN_SHA256 = "521ff25f034bcabcac2d408174c84c2129dc778a0e2a006f424b4fd21c1e8a9c"
STATE_PLAN_SHA256 = "34edf5dfd0e2c3ca96cd7b41d909242ee0e4005e7520de0efc6b182638e9fa3e"
HARNESS_SHA256 = "f47afa653f7a59d5854b68e7778901aba782c93ef100950257bf8f3c2a903c75"
ADAPTER_INVENTORY_SHA256 = "e5b513aa0d0d2d8dae3ef5907f88d566d4cc50d4ee3c415cc35ff690c87d507a"
PAIR_VERIFIER_SHA256 = "3292a57649b59c4f8872a9caf4ff19131cb42a25251c5bb713940e3e6fd80e1b"
REPOSITORY_HEAD_AT_PACKAGING = "0eb74430b7e3228c45b137a539e6d266b51a5bed"

EPOCHS = (1, 2, 3, 4)
BATCHES = (1, 4)
MODES = ("stock", "manager")
EXPECTED_ORDERS = {
    1: ["manager", "stock"],
    2: ["stock", "manager"],
    3: ["manager", "stock"],
    4: ["stock", "manager"],
}
DEEPGEMM_DISCLOSURE = {
    "requested_backend": "auto",
    "weight_shards_loaded": "66/66",
    "precompile_completed": False,
    "termination": "external_SIGQUIT",
    "end_to_end_record_produced": False,
    "included_in_archive": False,
    "included_in_steady_analysis": False,
    "result": "no_result",
}
RAW_KEYS = frozenset(
    {
        "capacity_readback", "checkpoint", "checkpoint_contract",
        "checkpoint_identity_sha256", "command", "command_sha256",
        "completed_requests", "completion_tokens", "engine_args",
        "environment", "environment_sha256", "global_prefix_cleanup",
        "gpu_snapshots", "iteration_seconds", "iteration_total_seconds",
        "load_seconds", "manager", "mode", "model",
        "output_request_digests_sha256", "output_token_digest_sha256",
        "pairing", "prefix_seed", "request_traces", "runtime_identity",
        "sampling_params", "schema", "server_memory",
        "setup_and_load_seconds", "source_identity",
        "source_identity_sha256", "started_at_utc", "total_seconds",
        "workload", "workload_profile",
    }
)
PAIR_KEYS = frozenset(
    {
        "batch_size", "epoch", "execution_order",
        "hardware_attested", "iterations", "manager_iteration_seconds",
        "manager_output_sha256", "manager_record", "pair_key_sha256",
        "preflight_bound", "profile", "qualification_claim",
        "qualified", "schema", "status", "stock_iteration_seconds",
        "stock_output_sha256", "stock_record", "workload_profile",
    }
)
MANIFEST_KEYS = frozenset(
    {
        "artifact_count", "artifacts", "deepgemm_attempt",
        "directories", "evidence_class", "hardware_attested",
        "identity", "integrity_scope",
        "pair_schema", "performance_go", "preflight_bound",
        "qualified", "record_schema", "schema", "sealed",
        "source_clean", "source_dirty", "source_inputs",
        "summary_schema",
    }
)
SUMMARY_KEYS = frozenset(
    {
        "all_pairs_passed", "deepgemm_attempt", "epoch_count",
        "evidence_class", "exact_token_equality", "groups",
        "hardware", "hardware_attested", "manager_census", "memory",
        "memory_claim", "model", "pair_count", "performance_go",
        "preflight_bound",
        "qualified", "qualification_claim", "record_count", "runtime",
        "schema", "sealed", "source", "source_clean",
        "source_dirty",
        "status", "stderr", "stderr_log_count",
    }
)
TOKEN_DRAIN_FIELDS = (
    "active_requests", "active_snapshots", "active_prefixes",
    "active_pages", "reserved_pages", "writing_pages",
    "retiring_pages", "quarantined_pages", "exhausted_pages",
    "pending_reclamations", "total_request_page_refs",
    "total_prefix_page_refs", "total_reader_pins", "prepared_steps",
    "submitted_steps",
)
ARENA_DRAIN_FIELDS = (
    "active_pages", "reserved_pages", "writing_pages",
    "retiring_pages", "quarantined_pages", "exhausted_pages",
    "request_page_refs", "prefix_page_refs", "reader_pins",
)
FIXED_DRAIN_FIELDS = (
    "reserved_slots", "relocating_slots", "live_slots",
    "retiring_slots", "quarantined_slots", "active_owners",
    "pending_transitions", "pending_retirements",
)
FAILURE_COUNTER_FIELDS = (
    "abort_relocations_batch_calls", "abort_steps_batch_calls",
    "fail_stop_count", "fail_stops", "quarantine_count",
    "quarantine_steps_batch_calls", "quarantine_submissions_batch_calls",
    "retryable_conflicts",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def _reject_constant(value: str) -> None:
    fail(f"non-finite JSON constant is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"non-finite JSON number in {label}")
    if isinstance(value, dict):
        for child in value.values():
            _reject_non_finite(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_non_finite(child, label)


def strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    _reject_non_finite(value, str(path))
    return value


def exact_keys(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    require(actual == expected, f"{label} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: Any) -> PurePosixPath:
    require(isinstance(value, str) and value, "artifact path must be a non-empty string")
    require("\\" not in value, f"artifact path uses a backslash: {value!r}")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"artifact path must be relative: {value!r}")
    require(path.as_posix() == value, f"artifact path is not normalized: {value!r}")
    require(all(part not in ("", ".", "..") for part in path.parts), f"unsafe artifact path: {value!r}")
    return path


def inspect_tree() -> tuple[set[str], set[str]]:
    require(not Path(__file__).is_symlink(), "verify.py must not be a symlink")
    require(ROOT.is_dir() and not ROOT.is_symlink(), "archive root must be a real directory")
    files: set[str] = set()
    directories: set[str] = set()
    for current, names, filenames in os.walk(ROOT, followlinks=False):
        current_path = Path(current)
        for name in names:
            path = current_path / name
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), f"symlink is forbidden: {path}")
            require(stat.S_ISDIR(mode), f"non-directory tree entry: {path}")
            directories.add(path.relative_to(ROOT).as_posix())
        for name in filenames:
            path = current_path / name
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), f"symlink is forbidden: {path}")
            require(stat.S_ISREG(mode), f"non-regular artifact: {path}")
            files.add(path.relative_to(ROOT).as_posix())
    return files, directories


def verify_artifacts(manifest: dict[str, Any]) -> None:
    exact_keys(manifest, MANIFEST_KEYS, "manifest")
    require(manifest["schema"] == MANIFEST_SCHEMA, "manifest schema differs")
    require(manifest["record_schema"] == RECORD_SCHEMA, "record schema differs")
    require(manifest["pair_schema"] == PAIR_SCHEMA, "pair schema differs")
    require(manifest["summary_schema"] == SUMMARY_SCHEMA, "summary schema differs")
    for field in ("sealed", "source_clean", "preflight_bound", "hardware_attested", "qualified", "performance_go"):
        require(manifest[field] is False, f"diagnostic boundary must keep {field}=false")
    require(manifest["source_dirty"] is True, "diagnostic boundary must keep source_dirty=true")
    require(manifest["evidence_class"] == "diagnostic_only", "manifest evidence class differs")
    require(manifest["integrity_scope"] == "all_archive_files_except_self_referential_manifest", "manifest integrity scope differs")

    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "manifest artifacts must be an object")
    require(manifest.get("artifact_count") == len(artifacts), "manifest artifact count differs")
    declared_dirs = manifest.get("directories")
    require(isinstance(declared_dirs, list) and all(isinstance(x, str) for x in declared_dirs), "manifest directories must be strings")
    require(len(declared_dirs) == len(set(declared_dirs)), "manifest directories contain duplicates")

    actual_files, actual_dirs = inspect_tree()
    expected_files = set(artifacts) | {"manifest.json"}
    require(actual_files == expected_files, f"archive file inventory differs: missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}")
    require(actual_dirs == set(declared_dirs), f"archive directory inventory differs: missing={sorted(set(declared_dirs) - actual_dirs)}, extra={sorted(actual_dirs - set(declared_dirs))}")

    root_resolved = ROOT.resolve(strict=True)
    for name, identity in artifacts.items():
        relative = safe_relative_path(name)
        require(isinstance(identity, dict) and set(identity) == {"bytes", "sha256"}, f"artifact identity keys differ: {name}")
        size = identity["bytes"]
        digest = identity["sha256"]
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"artifact byte count is invalid: {name}")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"artifact hash is invalid: {name}")
        path = ROOT.joinpath(*relative.parts)
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(root_resolved), f"artifact escapes archive root: {name}")
        require(not path.is_symlink() and path.is_file(), f"artifact is not a regular file: {name}")
        require(path.stat().st_size == size, f"artifact byte count differs: {name}")
        require(sha256_file(path) == digest, f"artifact hash differs: {name}")


def record_path(epoch: int, batch: int, kind: str) -> Path:
    return ROOT / "records" / f"epoch-{epoch:03d}" / f"b{batch}-{kind}.json"


def log_path(epoch: int, batch: int, mode: str) -> Path:
    return ROOT / "records" / f"epoch-{epoch:03d}" / f"b{batch}-{mode}.stderr.log"


def artifact_fields(value: Any) -> tuple[Any, Any]:
    require(isinstance(value, dict), "artifact identity must be an object")
    return value.get("sha256"), value.get("bytes")


def verify_record_identity(record: dict[str, Any], epoch: int, batch: int, mode: str, adapter: dict[str, Any]) -> None:
    label = f"epoch-{epoch:03d}/b{batch}-{mode}"
    exact_keys(record, RAW_KEYS, label)
    require(record["schema"] == RECORD_SCHEMA, f"{label} record schema differs")
    require(record["mode"] == mode, f"{label} mode differs")
    require(record["workload_profile"] == "fresh_prompt", f"{label} workload profile differs")
    workload = record["workload"]
    require(isinstance(workload, dict), f"{label} workload is missing")
    require(workload.get("iterations") == 5, f"{label} must contain five iterations")
    require(workload.get("requests") == batch, f"{label} batch differs")
    require(workload.get("prompt_tokens") == 513 and workload.get("decode_tokens") == 33, f"{label} workload shape differs")
    timings = record["iteration_seconds"]
    require(isinstance(timings, list) and len(timings) == 5, f"{label} timing count differs")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in timings), f"{label} has invalid timings")
    require(record["completed_requests"] == 5 * batch, f"{label} completed request count differs")
    require(record["completion_tokens"] == 5 * batch * 33, f"{label} completion token count differs")

    checkpoint = record["checkpoint"]
    require(record["checkpoint_identity_sha256"] == MODEL_CHECKPOINT_SHA256, f"{label} checkpoint identity differs")
    require(qualification.canonical_digest(checkpoint) == MODEL_CHECKPOINT_SHA256, f"{label} checkpoint digest is invalid")
    require(checkpoint.get("config_sha256") == MODEL_CONFIG_SHA256, f"{label} config hash differs")
    indexes = checkpoint.get("index_files")
    require(isinstance(indexes, list) and len(indexes) == 1 and indexes[0].get("sha256") == MODEL_INDEX_SHA256, f"{label} model index differs")
    shards = checkpoint.get("weight_files")
    require(isinstance(shards, list) and len(shards) == MODEL_SHARDS, f"{label} shard count differs")
    require(sum(x.get("bytes", -1) for x in shards if isinstance(x, dict)) == MODEL_WEIGHT_BYTES, f"{label} weight byte count differs")
    require(checkpoint.get("weight_bytes") == MODEL_WEIGHT_BYTES and checkpoint.get("observed_indexed_weight_bytes") == MODEL_WEIGHT_BYTES, f"{label} checkpoint byte totals differ")
    require(qualification.canonical_digest(shards) == MODEL_WEIGHT_INVENTORY_SHA256, f"{label} weight inventory differs")

    source = record["source_identity"]
    require(source.get("release") == SGLANG_RELEASE and source.get("revision") == SGLANG_REVISION, f"{label} SGLang identity differs")
    require(source.get("harness_sha256") == HARNESS_SHA256, f"{label} harness differs")
    require(source.get("adapter") == adapter, f"{label} adapter inventory differs")
    require(qualification.canonical_digest(adapter) == ADAPTER_INVENTORY_SHA256, f"{label} adapter digest differs")
    if mode == "stock":
        require(source.get("python_source_contract") == "clean_pinned_head" and source.get("dirty_paths") == [], f"{label} stock source contract differs")
        require(source.get("library") is None and source.get("plan") is None and source.get("state_plan") is None, f"{label} stock binds manager artifacts")
        require(record["manager"] is None, f"{label} stock manager field is not null")
    else:
        require(source.get("python_source_contract") == "pinned_head_plus_canonical_loader_patch", f"{label} manager source contract differs")
        require(source.get("dirty_paths") == ["python/sglang/srt/plugins/__init__.py"], f"{label} manager dirty path differs")
        require(artifact_fields(source.get("library")) == (LIBRARY_SHA256, LIBRARY_BYTES), f"{label} library identity differs")
        require(artifact_fields(source.get("plan"))[0] == TOKEN_PLAN_SHA256, f"{label} token plan differs")
        require(artifact_fields(source.get("state_plan"))[0] == STATE_PLAN_SHA256, f"{label} state plan differs")
        require(isinstance(record["manager"], dict), f"{label} manager payload is missing")

    runtime = record["runtime_identity"]
    backend = runtime.get("backend_profile", {})
    require(runtime.get("attention_backend") == "fa3" and runtime.get("execution") == "eager" and runtime.get("kv_layout") == "nhd", f"{label} runtime profile differs")
    require(runtime.get("deterministic_inference") is True, f"{label} runtime is not deterministic")
    require(runtime.get("sampling_backend") == "pytorch", f"{label} sampling backend differs")
    expected_backend = {
        "attention_backend": "fa3",
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "linear_attn_prefill_backend": "triton",
        "mamba_radix_cache_strategy": "no_buffer",
        "mamba_ssm_dtype": "float32",
    }
    require(backend == expected_backend, f"{label} backend profile differs")
    engine = record["engine_args"]
    require(engine.get("fp8_gemm_runner_backend") == "triton", f"{label} FP8 GEMM backend is not explicit Triton")
    require(all(backend.get(name) == "triton" for name in ("linear_attn_backend", "linear_attn_prefill_backend", "linear_attn_decode_backend")), f"{label} linear backend is not explicit Triton")
    expected_engine = {
        "attention_backend": "fa3",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "page_size": 16,
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "linear_attn_prefill_backend": "triton",
        "mamba_radix_cache_strategy": "no_buffer",
        "mamba_ssm_dtype": "float32",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
    }
    require(all(engine.get(name) == value for name, value in expected_engine.items()), f"{label} engine profile differs")
    require(engine.get("disable_cuda_graph") is True, f"{label} execution is not eager")
    require(engine.get("disable_overlap_schedule") is True, f"{label} overlap scheduling is enabled")
    require(engine.get("disable_radix_cache") is True, f"{label} Radix cache is enabled")
    require(engine.get("enable_deterministic_inference") is True, f"{label} deterministic inference is disabled")
    require(record["capacity_readback"].get("full_tokens") == (1024 if batch == 1 else 4096), f"{label} capacity differs")

    snapshots = record["gpu_snapshots"]
    require(isinstance(snapshots, list) and len(snapshots) == 5, f"{label} GPU snapshot count differs")
    for snapshot in snapshots:
        gpus = snapshot.get("gpus") if isinstance(snapshot, dict) else None
        require(isinstance(gpus, list) and len(gpus) == 1, f"{label} GPU snapshot shape differs")
        require(gpus[0].get("name") == GPU_NAME and gpus[0].get("uuid") == GPU_UUID, f"{label} observed GPU differs")


def verify_final_census(record: dict[str, Any], batch: int, label: str) -> None:
    manager = record["manager"]
    final = manager.get("final_census")
    require(isinstance(final, dict) and final.get("abi_version") == 8, f"{label} final census is not ABI8")
    require(manager.get("after_workload") == final, f"{label} final census changed after workload drain")
    require(record.get("global_prefix_cleanup", {}).get("success") is True, f"{label} global cleanup failed")
    stats = final.get("manager_stats", {})
    require(all(stats.get(name) == 0 for name in TOKEN_DRAIN_FIELDS), f"{label} token state did not drain")
    expected_pages = 64 if batch == 1 else 256
    require(stats.get("free_pages") == expected_pages, f"{label} free page count differs")
    arenas = final.get("arena_stats")
    require(isinstance(arenas, list) and len(arenas) == 1, f"{label} arena census differs")
    arena = arenas[0]
    require(all(arena.get(name) == 0 for name in ARENA_DRAIN_FIELDS), f"{label} arena did not drain")
    require(arena.get("free_pages") == expected_pages and arena.get("page_count") == expected_pages, f"{label} arena free page count differs")
    fixed = final.get("fixed_state", {})
    require(all(fixed.get(name) == 0 for name in FIXED_DRAIN_FIELDS), f"{label} fixed state did not drain")
    slots = 2 if batch == 1 else 4
    require(fixed.get("free_slots") == slots and fixed.get("identity", {}).get("slot_count") == slots, f"{label} fixed-state slots did not drain")
    require(final.get("fixed_state_byte_count") == 153_944_064, f"{label} fixed-state byte count differs")
    counters = final.get("batch_counters", {})
    require(all(counters.get(name) == 0 for name in FAILURE_COUNTER_FIELDS), f"{label} failure counter is nonzero")


def percentile_inclusive(values: list[float], percentile: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[percentile - 1]


def mode_statistics(values: list[float], output_tokens: int) -> dict[str, Any]:
    mean = statistics.fmean(values)
    return {
        "sample_count": len(values),
        "mean_seconds": mean,
        "median_seconds": statistics.median(values),
        "p95_seconds_inclusive": percentile_inclusive(values, 95),
        "output_tokens_per_second": output_tokens / mean,
    }


def build_summary(records: dict[tuple[int, int, str], dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    groups = []
    memory = []
    census_batches = []
    for batch in BATCHES:
        samples = {mode: [] for mode in MODES}
        epochs = []
        capacities = {mode: set() for mode in MODES}
        memories = {mode: set() for mode in MODES}
        for epoch in EPOCHS:
            hot: dict[str, list[float]] = {}
            for mode in MODES:
                record = records[(epoch, batch, mode)]
                hot[mode] = list(record["iteration_seconds"][1:])
                samples[mode].extend(hot[mode])
                capacities[mode].add(record["capacity_readback"]["full_tokens"])
                server_memory = record["server_memory"]
                require(server_memory["after_load"]["kvcache"] == server_memory["after_workload"]["kvcache"], f"B{batch} {mode} KV memory changed during workload")
                memories[mode].add(server_memory["after_load"]["kvcache"])
            stock_mean = statistics.fmean(hot["stock"])
            manager_mean = statistics.fmean(hot["manager"])
            epochs.append({
                "epoch": epoch,
                "execution_order": EXPECTED_ORDERS[epoch],
                "stock_mean_seconds": stock_mean,
                "manager_mean_seconds": manager_mean,
                "manager_over_stock_percent": (manager_mean / stock_mean - 1.0) * 100.0,
            })
        require(len(samples["stock"]) == 16 and len(samples["manager"]) == 16, f"B{batch} hot sample count differs")
        stock_stats = mode_statistics(samples["stock"], batch * 33)
        manager_stats = mode_statistics(samples["manager"], batch * 33)
        groups.append({
            "batch_size": batch,
            "epoch_count": 4,
            "iterations_per_process": 5,
            "excluded_iteration_indices": [0],
            "hot_iteration_indices": [1, 2, 3, 4],
            "hot_sample_count_per_mode": 16,
            "percentile_method": "statistics.quantiles(n=100, method=inclusive)",
            "stock": stock_stats,
            "manager": manager_stats,
            "manager_over_stock_latency_percent": (manager_stats["mean_seconds"] / stock_stats["mean_seconds"] - 1.0) * 100.0,
            "manager_throughput_delta_percent": (stock_stats["mean_seconds"] / manager_stats["mean_seconds"] - 1.0) * 100.0,
            "epochs": epochs,
        })
        require(all(len(capacities[mode]) == 1 and len(memories[mode]) == 1 for mode in MODES), f"B{batch} capacity or memory is unstable")
        stock_capacity = next(iter(capacities["stock"]))
        manager_capacity = next(iter(capacities["manager"]))
        stock_memory = next(iter(memories["stock"]))
        manager_memory = next(iter(memories["manager"]))
        require(stock_capacity == manager_capacity and stock_memory == manager_memory, f"B{batch} stock/manager capacity differs")
        memory.append({
            "batch_size": batch,
            "full_tokens": {"stock": stock_capacity, "manager": manager_capacity},
            "reported_kv_cache_gb": {"stock": stock_memory, "manager": manager_memory},
            "manager_capacity_saving_percent": 0.0,
            "manager_reported_kv_cache_saving_percent": 0.0,
        })
        census_batches.append({
            "batch_size": batch,
            "manager_record_count": 4,
            "free_pages": 64 if batch == 1 else 256,
            "page_count": 64 if batch == 1 else 256,
            "free_fixed_state_slots": 2 if batch == 1 else 4,
            "fixed_state_slot_count": 2 if batch == 1 else 4,
        })

    stderr = {
        "all_completed_66_of_66_shards": True,
        "cuda_ipc_shutdown_warning_log_count": 16,
        "fatal_error_log_count": 0,
        "transformers_use_fast_deprecation_line_count": 32,
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "diagnostic_pair_verification_passed",
        "evidence_class": "diagnostic_only",
        "qualification_claim": "diagnostic_only_not_qualified",
        "sealed": False,
        "source_clean": False,
        "source_dirty": True,
        "preflight_bound": False,
        "hardware_attested": False,
        "qualified": False,
        "performance_go": False,
        "epoch_count": 4,
        "record_count": 16,
        "stderr_log_count": 16,
        "pair_count": 8,
        "all_pairs_passed": all(pair.get("status") == "passed" for pair in pairs),
        "exact_token_equality": all(pair.get("stock_output_sha256") == pair.get("manager_output_sha256") for pair in pairs),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "index_sha256": MODEL_INDEX_SHA256,
            "checkpoint_identity_sha256": MODEL_CHECKPOINT_SHA256,
            "weight_inventory_sha256": MODEL_WEIGHT_INVENTORY_SHA256,
            "weight_shard_count": MODEL_SHARDS,
            "weight_bytes": MODEL_WEIGHT_BYTES,
            "tensor_count": MODEL_TENSORS,
            "external_provenance_note": "repository revision and tensor count are declared provenance; raw records bind config, index, and every weight shard hash and byte count",
        },
        "runtime": {
            "sglang_release": SGLANG_RELEASE,
            "sglang_revision": SGLANG_REVISION,
            "record_schema": RECORD_SCHEMA,
            "pair_schema": PAIR_SCHEMA,
            "attention_backend": "fa3",
            "linear_attention_backend": "triton",
            "linear_attention_prefill_backend": "triton",
            "linear_attention_decode_backend": "triton",
            "fp8_gemm_runner_backend": "triton",
            "execution": "eager",
            "kv_layout": "nhd",
            "page_tokens": 16,
            "library": {"sha256": LIBRARY_SHA256, "bytes": LIBRARY_BYTES, "abi_version": 8, "exact_symbol_count": 40},
            "token_plan_sha256": TOKEN_PLAN_SHA256,
            "state_plan_sha256": STATE_PLAN_SHA256,
            "harness_sha256": HARNESS_SHA256,
        },
        "hardware": {
            "observed_name": GPU_NAME,
            "observed_uuid": GPU_UUID,
            "snapshot_count": 80,
            "attestation": "recorded_observation_only",
        },
        "source": {
            "repository_head_at_packaging": REPOSITORY_HEAD_AT_PACKAGING,
            "worktree_clean": False,
            "source_bundle_included": False,
            "adapter_inventory_sha256": ADAPTER_INVENTORY_SHA256,
            "pair_verifier_path": "integrations/sglang/qualify_abi8_h20.py",
            "pair_verifier_sha256_at_packaging": PAIR_VERIFIER_SHA256,
        },
        "groups": groups,
        "manager_census": {
            "manager_record_count": 8,
            "all_after_workload_equal_final_census": True,
            "all_global_cleanup_successful": True,
            "all_token_state_drained": True,
            "all_fixed_state_drained": True,
            "all_failure_counters_zero": True,
            "fixed_state_byte_count": 153_944_064,
            "batches": census_batches,
        },
        "memory": memory,
        "memory_claim": {
            "metric": "observed_equal_configured_tensor_arena_reservation_difference_percent",
            "value": 0.0,
            "qualified_end_to_end_memory_saving_result": False,
        },
        "stderr": stderr,
        "deepgemm_attempt": {
            **DEEPGEMM_DISCLOSURE,
            "evidence_basis": "declared_external_observation_not_portably_verified",
        },
    }


def verify_logs() -> None:
    use_fast_lines = 0
    cuda_ipc_logs = 0
    fatal_logs = 0
    completed_logs = 0
    fatal_markers = (
        "traceback (most recent call last)", "cuda out of memory",
        "outofmemoryerror", "[error]", "exception",
    )
    for epoch in EPOCHS:
        for batch in BATCHES:
            for mode in MODES:
                path = log_path(epoch, batch, mode)
                text = path.read_text(encoding="utf-8", errors="strict").replace("\r", "\n")
                if "100% Completed | 66/66" in text:
                    completed_logs += 1
                use_fast_lines += sum("The `use_fast` parameter is deprecated" in line for line in text.splitlines())
                if "Producer process has been terminated before all shared CUDA tensors released" in text:
                    cuda_ipc_logs += 1
                if any(marker in text.lower() for marker in fatal_markers):
                    fatal_logs += 1
    require(completed_logs == 16, "not every stderr log records a completed 66/66 shard load")
    require(use_fast_lines == 32, "stderr deprecation warning count differs")
    require(cuda_ipc_logs == 16, "stderr CUDA IPC shutdown warning count differs")
    require(fatal_logs == 0, "stderr contains a fatal-error marker")


def verify_timestamp_order(records: dict[tuple[int, int, str], dict[str, Any]]) -> None:
    for epoch in EPOCHS:
        expected = EXPECTED_ORDERS[epoch]
        for batch in BATCHES:
            first = records[(epoch, batch, expected[0])]
            second = records[(epoch, batch, expected[1])]
            first_start = datetime.fromisoformat(first["started_at_utc"])
            second_start = datetime.fromisoformat(second["started_at_utc"])
            require(first_start < second_start, f"epoch {epoch} B{batch} execution order differs")
            first_end = first_start + timedelta(seconds=first["total_seconds"])
            require(first_end <= second_start, f"epoch {epoch} B{batch} processes overlap")


def verify_pairs() -> tuple[dict[tuple[int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    first = strict_json(record_path(1, 1, "stock"))
    adapter = first.get("source_identity", {}).get("adapter")
    require(isinstance(adapter, dict), "run-bound adapter inventory is missing")
    pairs = []
    for epoch in EPOCHS:
        for batch in BATCHES:
            for mode in MODES:
                record = strict_json(record_path(epoch, batch, mode))
                verify_record_identity(record, epoch, batch, mode, adapter)
                records[(epoch, batch, mode)] = record
            verify_final_census(records[(epoch, batch, "manager")], batch, f"epoch-{epoch:03d}/b{batch}-manager")
            calculated = qualification.verify_pair_files(
                record_path(epoch, batch, "stock"),
                record_path(epoch, batch, "manager"),
                harness_sha256=HARNESS_SHA256,
                adapter_identity=adapter,
            )
            calculated.update(epoch=epoch, execution_order=EXPECTED_ORDERS[epoch])
            stored = strict_json(record_path(epoch, batch, "pair"))
            exact_keys(stored, PAIR_KEYS, f"epoch-{epoch:03d}/b{batch}-pair")
            require(stored == calculated, f"epoch-{epoch:03d}/b{batch} stored pair differs from current verifier")
            require(stored["preflight_bound"] is False and stored["hardware_attested"] is False and stored["qualified"] is False, f"epoch-{epoch:03d}/b{batch} pair crosses diagnostic boundary")
            require(stored["stock_output_sha256"] == stored["manager_output_sha256"], f"epoch-{epoch:03d}/b{batch} output tokens differ")
            pairs.append(stored)
    verify_timestamp_order(records)
    return records, pairs


def verify_static_identity(manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    expected = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "config_sha256": MODEL_CONFIG_SHA256,
        "index_sha256": MODEL_INDEX_SHA256,
        "weight_shard_count": MODEL_SHARDS,
        "weight_bytes": MODEL_WEIGHT_BYTES,
        "tensor_count": MODEL_TENSORS,
        "sglang_release": SGLANG_RELEASE,
        "sglang_revision": SGLANG_REVISION,
        "gpu_uuid": GPU_UUID,
        "library_sha256": LIBRARY_SHA256,
        "library_bytes": LIBRARY_BYTES,
        "abi_version": 8,
        "exact_symbol_count": 40,
        "token_plan_sha256": TOKEN_PLAN_SHA256,
        "state_plan_sha256": STATE_PLAN_SHA256,
        "harness_sha256": HARNESS_SHA256,
    }
    require(manifest.get("identity") == expected, "manifest static identity differs")
    require(len(qualification.EXACT_SYMBOL_ALLOWLIST) == 40 and qualification.ABI_VERSION == 8, "current verifier does not define exact ABI8/40-symbol contract")
    verifier = REPOSITORY_ROOT / "integrations/sglang/qualify_abi8_h20.py"
    require(verifier.is_file() and not verifier.is_symlink(), "current pair verifier is missing or symlinked")
    require(sha256_file(verifier) == PAIR_VERIFIER_SHA256, "current pair verifier differs from the packaging-time dirty source")
    for field in ("sealed", "source_clean", "preflight_bound", "hardware_attested", "qualified", "performance_go"):
        require(summary[field] is False, f"summary must keep {field}=false")
    require(summary["source_dirty"] is True, "summary must keep source_dirty=true")
    require(manifest.get("deepgemm_attempt") == DEEPGEMM_DISCLOSURE, "manifest DeepGEMM disclosure differs")
    source_inputs = manifest.get("source_inputs")
    require(
        source_inputs == {
            "path_at_packaging": "/tmp/orbitkv-qwen38-h20-20260824/steady",
            "file_count": 32,
            "inventory_sha256": "13d0237eabf0151d0512acb46a11eb17217a1b042920a2e9773a07b9875e8da0",
            "copied_byte_for_byte_at_packaging": True,
            "portable_verification": "manifest hashes verify retained copies; origin comparison requires the original temporary tree",
        },
        "manifest source-input provenance differs",
    )
    origin_root = Path(source_inputs["path_at_packaging"])
    if origin_root.is_dir():
        origin_inventory = []
        for path in sorted(origin_root.rglob("*")):
            require(not path.is_symlink(), f"source input is symlinked: {path}")
            if path.is_file():
                relative = path.relative_to(origin_root).as_posix()
                require(
                    (ROOT / "records" / relative).read_bytes() == path.read_bytes(),
                    f"retained input differs byte-for-byte from source: {relative}",
                )
                origin_inventory.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                )
        require(len(origin_inventory) == source_inputs["file_count"], "source input file count differs")
        require(qualification.canonical_digest(origin_inventory) == source_inputs["inventory_sha256"], "source input inventory digest differs")


def main() -> int:
    try:
        manifest = strict_json(ROOT / "manifest.json")
        verify_artifacts(manifest)
        summary = strict_json(ROOT / "summary.json")
        exact_keys(summary, SUMMARY_KEYS, "summary")
        verify_static_identity(manifest, summary)
        verify_logs()
        records, pairs = verify_pairs()
        calculated_summary = build_summary(records, pairs)
        require(summary == calculated_summary, "stored summary differs from raw-record recomputation")
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    result = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "evidence_class": "diagnostic_only",
        "artifact_count": manifest["artifact_count"],
        "record_count": 16,
        "stderr_log_count": 16,
        "pair_count": 8,
        "hot_sample_count_per_mode_per_batch": 16,
        "sealed": False,
        "source_clean": False,
        "source_dirty": True,
        "preflight_bound": False,
        "hardware_attested": False,
        "qualified": False,
        "performance_go": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
