#!/usr/bin/env python3
"""Verify untrusted paired Engine E2E records without writing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import engine_e2e_verifier_contract as _verifier_contract
from engine_e2e_verifier_contract import (
    U64_MAX as _U64_MAX,
    layout_fingerprint_from_manager_input as _layout_fingerprint_from_manager_input,
    manager_input_fingerprint as _manager_input_fingerprint,
    manager_input_from_attention_source as _manager_input_from_attention_source,
    native_session_contract as _native_session_contract,
    validate_native_session_profile as _validate_native_session_profile,
)
from engine_e2e_schema import (
    ACCELERATOR_KEYS,
    ACTUAL_BACKEND_KEYS,
    ARENA_KEYS,
    BATCH_COUNTER_KEYS,
    BINDING_KEYS,
    CHECKPOINT_KEYS,
    COMPLETION_KEYS,
    COMPLETION_POINT_KEYS,
    COMPUTE_CAPABILITY_KEYS,
    DRAIN_FIELDS,
    EFFECTIVE_SCHEDULER_KEYS,
    ENGINE_OPTIONAL_KEYS,
    ENGINE_REQUIRED_KEYS,
    EXECUTION_SIGNATURE_KEYS,
    FILE_IDENTITY_KEYS,
    IDENTITY_KEYS,
    ITERATION_OUTPUT_KEYS,
    MANAGER_KEYS,
    MANAGER_SNAPSHOT_KEYS,
    MANAGER_STATS_KEYS,
    MANIFEST_KEYS,
    OUTPUT_KEYS,
    OWNER_KEYS,
    PAGE_PHASE_FIELDS,
    PATCH_KEYS,
    POST_WORKLOAD_RESIDENCY_KEYS,
    PRESSURE_KEYS,
    PROGRESS_COUNTER_FIELDS,
    RECORD_KEYS,
    RESOLVED_ENGINE_KEYS,
    RUNTIME_PROOF_KEYS,
    SAMPLING_KEYS,
    SERVER_SNAPSHOT_KEYS,
    SOURCE_KEYS,
    SWA_ACTIVITY_KEYS,
    SWA_COUNTER_FIELDS,
    TIMING_KEYS,
    WARMUP_OUTPUT_KEYS,
    WORKLOAD_KEYS,
    WORKLOAD_PARTITION_KEYS,
)


sys.dont_write_bytecode = True

FULL_TOKEN_KV_TOPOLOGY = _verifier_contract.FULL_TOKEN_KV_TOPOLOGY
FULL_SLIDING_TOKEN_KV_TOPOLOGY = (
    _verifier_contract.FULL_SLIDING_TOKEN_KV_TOPOLOGY
)
FULL_LATENT_KV_TOPOLOGY = _verifier_contract.FULL_LATENT_KV_TOPOLOGY
RECORD_SCHEMA = "orbitkv.sglang-engine-e2e.v3"
VERIFICATION_SCHEMA = "orbitkv.sglang-engine-e2e-pairs-verification.v3"
WIRE_VERSION = 14
TARGET_CONTRACT_VERSION = 4
TARGET_CONTRACT_FINGERPRINT = (
    "sha256:ac915458195e757e477cf04866dae76147cd71a7472661e0d791e9c4474173ba"
)
def _exact(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(expected_keys - actual_keys)} "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    return value


def _same(left: Any, right: Any) -> bool:
    """Compare JSON values without treating booleans as integers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _object_with_keys(
    value: Any, required: Iterable[str], optional: Iterable[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    required_keys = set(required)
    allowed_keys = required_keys | set(optional)
    actual_keys = set(value)
    if not required_keys <= actual_keys or not actual_keys <= allowed_keys:
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(required_keys - actual_keys)} "
            f"extra={sorted(actual_keys - allowed_keys)}"
        )
    return value


def _json_tree(value: Any, label: str = "value") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        if not -(1 << 63) < value < (1 << 63):
            raise RuntimeError(f"{label} integer is outside the supported range")
        return
    if type(value) is float:
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
            raise RuntimeError(f"{label} contains a non-finite or negative-zero number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{label} contains a non-string key")
            _json_tree(item, f"{label}.{key}")
        return
    raise RuntimeError(f"{label} is not a JSON value")


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite JSON number {value}")
        return result

    try:
        if path.is_symlink():
            raise RuntimeError(f"record must not be a symlink: {path}")
        if not path.is_file():
            raise RuntimeError(f"record is not a regular file: {path}")
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except RuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"cannot load strict JSON object {path}: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"JSON record is not an object: {path}")
    _json_tree(result, str(path))
    return result


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RuntimeError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(value: Mapping[str, Any], label: str) -> str:
    observed = value.get("fingerprint")
    if (
        not isinstance(observed, str)
        or len(observed) != 71
        or not observed.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in observed[7:])
    ):
        raise RuntimeError(f"{label}.fingerprint is not canonical")
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RuntimeError(f"{label} is not canonical JSON") from error
    expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if observed != expected:
        raise RuntimeError(f"{label}.fingerprint does not match its payload")
    return observed


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(mark in value for mark in ("\x00", "\r", "\n"))
    ):
        raise RuntimeError(f"{label} must be a nonempty single-line string")
    return value


def _sha256(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a canonical SHA-256 digest")
    return value


def _integer(
    value: Any, label: str, *, positive: bool = False, expected: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    if expected is not None and value != expected:
        raise RuntimeError(f"{label} must be {expected}")
    if positive and value <= 0:
        raise RuntimeError(f"{label} must be positive")
    if not positive and value < 0:
        raise RuntimeError(f"{label} must be nonnegative")
    return value


def _bounded_positive_integer(value: Any, label: str, maximum: int) -> int:
    result = _integer(value, label, positive=True)
    if result > maximum:
        raise RuntimeError(f"{label} exceeds the supported integer range")
    return result


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (result == 0.0 and math.copysign(1.0, result) < 0):
        raise RuntimeError(f"{label} must be finite and not negative zero")
    if positive and result <= 0:
        raise RuntimeError(f"{label} must be positive")
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("latency sample is empty")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _expect_close(actual: Any, expected: float, label: str) -> float:
    result = _number(actual, label, positive=True)
    if not math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{label} differs from its samples")
    return result


def _validate_checkpoint(value: Any, label: str) -> Mapping[str, Any]:
    checkpoint = _exact(value, CHECKPOINT_KEYS, label)
    if checkpoint["load_format"] != "auto":
        raise RuntimeError(f"{label}.load_format must be 'auto'")
    _sha256(checkpoint["config_sha256"], f"{label}.config_sha256")
    total_weight_bytes = 0
    seen_names: set[str] = set()
    for field in ("index_files", "weight_files"):
        files = checkpoint[field]
        if not isinstance(files, list):
            raise RuntimeError(f"{label}.{field} must be an array")
        names = []
        for index, raw in enumerate(files):
            item = _exact(raw, FILE_IDENTITY_KEYS, f"{label}.{field}[{index}]")
            name = _text(item["name"], f"{label}.{field}[{index}].name")
            if "/" in name or "\\" in name or name in seen_names:
                raise RuntimeError(f"{label}.{field} contains an invalid file name")
            seen_names.add(name)
            names.append(name)
            size = _integer(item["bytes"], f"{label}.{field}[{index}].bytes", positive=True)
            _sha256(item["sha256"], f"{label}.{field}[{index}].sha256")
            if field == "weight_files":
                total_weight_bytes += size
        if names != sorted(names):
            raise RuntimeError(f"{label}.{field} must be sorted by name")
    weight_bytes = _integer(
        checkpoint["weight_bytes"], f"{label}.weight_bytes", positive=True
    )
    if weight_bytes != total_weight_bytes:
        raise RuntimeError(f"{label}.weight_bytes differs from weight_files")
    indexed_names = checkpoint["indexed_weight_files"]
    missing = checkpoint["missing_indexed_weights"]
    if (
        not isinstance(indexed_names, list)
        or any(not isinstance(item, str) or not item for item in indexed_names)
        or indexed_names != sorted(set(indexed_names))
        or not isinstance(missing, list)
        or any(item not in indexed_names for item in missing)
        or missing != sorted(set(missing))
    ):
        raise RuntimeError(f"{label} indexed weight inventory is invalid")
    observed = _integer(
        checkpoint["observed_indexed_weight_bytes"],
        f"{label}.observed_indexed_weight_bytes",
    )
    indexed = checkpoint["indexed_weight_bytes"]
    overhead = checkpoint["indexed_weight_container_overhead_bytes"]
    if indexed is None:
        if overhead is not None:
            raise RuntimeError(f"{label} indexed weight byte accounting is invalid")
    else:
        _integer(indexed, f"{label}.indexed_weight_bytes")
        if isinstance(overhead, bool) or not isinstance(overhead, int):
            raise RuntimeError(f"{label} indexed weight overhead must be an integer")
        if overhead != observed - indexed:
            raise RuntimeError(f"{label} indexed weight byte accounting differs")
    complete = checkpoint["indexed_weights_complete"]
    if not isinstance(complete, bool):
        raise RuntimeError(f"{label}.indexed_weights_complete must be Boolean")
    expected_complete = not missing and (indexed is None or observed >= indexed)
    if not complete or complete != expected_complete:
        raise RuntimeError(f"{label} checkpoint weights are incomplete")
    return checkpoint


def _validate_source(value: Any, mode: str, label: str) -> Mapping[str, Any]:
    source = _exact(value, SOURCE_KEYS, label)
    _text(source["root"], f"{label}.root")
    _text(source["release"], f"{label}.release")
    revision = _text(source["revision"], f"{label}.revision")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError(f"{label}.revision is not a canonical commit ID")
    patch = _exact(source["patch"], PATCH_KEYS, f"{label}.patch")
    expected_status = "absent" if mode == "stock" else "applied"
    if patch["status"] != expected_status:
        raise RuntimeError(f"{label}.patch.status does not match mode {mode!r}")
    if mode == "stock":
        if patch["sha256"] is not None:
            raise RuntimeError(f"{label} stock patch SHA-256 must be null")
    else:
        _sha256(patch["sha256"], f"{label}.patch.sha256")
    return source


def _validate_environment(value: Any, mode: str, label: str) -> Mapping[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(name, str) or not isinstance(item, str)
        for name, item in value.items()
    ):
        raise RuntimeError(f"{label} must map strings to strings")
    required = {
        "SGLANG_USE_HND_KVCACHE": "0",
        "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
        "SGLANG_RADIX_FORCE_MISS": "0",
        "PYTHONPATH": None,
    }
    optional = {"CUDA_VISIBLE_DEVICES"}
    if mode == "manager":
        required.update(
            ORBITKV_RUNTIME_MANIFEST=None,
            ORBITKV_LIBRARY=None,
            ORBITKV_SGLANG_ROOT=None,
        )
    _object_with_keys(value, required, optional, label)
    for name, expected in required.items():
        if expected is not None and value.get(name) != expected:
            raise RuntimeError(f"{label}.{name} must be {expected!r}")
    if "SGLANG_PLUGINS" in value:
        raise RuntimeError(f"{label} must not select SGLang plugins")
    manager_names = {name for name in value if name.startswith("ORBITKV_")}
    required_manager = {
        "ORBITKV_RUNTIME_MANIFEST",
        "ORBITKV_LIBRARY",
        "ORBITKV_SGLANG_ROOT",
    }
    if mode == "stock" and manager_names:
        raise RuntimeError(f"{label} stock environment contains ORBITKV settings")
    if mode == "manager" and manager_names != required_manager:
        raise RuntimeError(f"{label} manager ORBITKV settings differ")
    return value


def _version(value: Any, label: str) -> str:
    version = _text(value, label)
    parts = version.split(".")
    if len(parts) < 2 or any(not part.isascii() or not part.isdigit() for part in parts):
        raise RuntimeError(f"{label} must be a dotted numeric version")
    return version


def _validate_accelerator(value: Any, label: str) -> Mapping[str, Any]:
    accelerator = _exact(value, ACCELERATOR_KEYS, label)
    if accelerator["device_type"] != "cuda":
        raise RuntimeError(f"{label}.device_type must be 'cuda'")
    _text(accelerator["device_name"], f"{label}.device_name")
    capability = _exact(
        accelerator["compute_capability"],
        COMPUTE_CAPABILITY_KEYS,
        f"{label}.compute_capability",
    )
    _integer(
        capability["major"],
        f"{label}.compute_capability.major",
        positive=True,
    )
    _integer(
        capability["minor"],
        f"{label}.compute_capability.minor",
    )
    _integer(
        accelerator["total_memory_bytes"],
        f"{label}.total_memory_bytes",
        positive=True,
    )
    _version(accelerator["runtime_version"], f"{label}.runtime_version")
    _version(accelerator["driver_version"], f"{label}.driver_version")
    return accelerator


def _validate_engine(value: Any, mode: str, label: str) -> Mapping[str, Any]:
    optional = set(ENGINE_OPTIONAL_KEYS)
    if mode == "manager":
        optional.add("radix_cache_backend")
    engine = _object_with_keys(value, ENGINE_REQUIRED_KEYS, optional, label)
    fixed = {
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "page_size": 16,
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "prefill_max_requests": 1,
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
        "log_level": "error",
    }
    mismatches = {
        name: engine.get(name)
        for name, expected in fixed.items()
        if type(engine.get(name)) is not type(expected) or engine.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"{label} fixed contract differs: {sorted(mismatches)}")
    if type(engine["disable_radix_cache"]) is not bool:
        raise RuntimeError(f"{label}.disable_radix_cache must be Boolean")
    if mode == "stock" and engine["disable_radix_cache"] is not False:
        raise RuntimeError(f"{label} stock radix cache must remain enabled")
    if engine["attention_backend"] not in ("fa3", "flashinfer"):
        raise RuntimeError(f"{label}.attention_backend is unsupported")
    for name in (
        "context_length",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "max_total_tokens",
    ):
        _integer(engine[name], f"{label}.{name}", positive=True)
    _integer(engine["random_seed"], f"{label}.random_seed")
    if engine["chunked_prefill_size"] != engine["max_prefill_tokens"]:
        raise RuntimeError(f"{label} prefill limits differ")
    if engine["chunked_prefill_size"] % 16 or engine["max_total_tokens"] % 16:
        raise RuntimeError(f"{label} token capacities must be page aligned")
    if "mem_fraction_static" in engine:
        fraction = _number(engine["mem_fraction_static"], f"{label}.mem_fraction_static", positive=True)
        if fraction > 1.0:
            raise RuntimeError(f"{label}.mem_fraction_static must be at most one")
    if mode == "manager":
        if engine.get("radix_cache_backend") != "orbitkv":
            raise RuntimeError(f"{label} manager radix backend is not OrbitKV")
    elif "radix_cache_backend" in engine:
        raise RuntimeError(f"{label} stock engine selects a manager radix backend")
    return engine


def _validate_sampling(
    value: Any, workload: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    sampling = _exact(value, SAMPLING_KEYS, label)
    for name in ("temperature", "max_new_tokens", "min_new_tokens", "sampling_seed"):
        _integer(
            sampling[name],
            f"{label}.{name}",
            expected=(
                0
                if name == "temperature"
                else workload["decode_tokens"]
                if name in ("max_new_tokens", "min_new_tokens")
                else workload["seed"]
            ),
        )
    if sampling["ignore_eos"] is not True:
        raise RuntimeError(f"{label} differs from deterministic workload settings")
    return sampling


def _integer_array(value: Any, label: str, *, nonempty: bool = True) -> list[int]:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"{label} must be an integer array")
    return value


def _validate_workload(
    value: Any, engine: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    workload = _exact(value, WORKLOAD_KEYS, label)
    prompt_tokens = _integer(workload["prompt_tokens"], f"{label}.prompt_tokens", positive=True)
    decode_tokens = _integer(workload["decode_tokens"], f"{label}.decode_tokens", positive=True)
    warmups = _integer(workload["warmups"], f"{label}.warmups")
    iterations = _integer(workload["iterations"], f"{label}.iterations", positive=True)
    seed = _integer(workload["seed"], f"{label}.seed")
    if (
        prompt_tokens + decode_tokens >= engine["context_length"]
        or prompt_tokens > engine["chunked_prefill_size"]
        or engine["random_seed"] != seed
    ):
        raise RuntimeError(f"{label} differs from engine limits or seed")
    inputs = _exact(workload["input_ids"], WORKLOAD_PARTITION_KEYS, f"{label}.input_ids")
    digests = _exact(
        workload["input_ids_sha256"],
        WORKLOAD_PARTITION_KEYS,
        f"{label}.input_ids_sha256",
    )
    request_ids = _exact(
        workload["request_ids"], WORKLOAD_PARTITION_KEYS, f"{label}.request_ids"
    )
    counts = {"warmup": warmups, "measured": iterations}
    for partition, count in counts.items():
        prompts = inputs[partition]
        hashes = digests[partition]
        ids = request_ids[partition]
        if not isinstance(prompts, list) or not isinstance(hashes, list) or not isinstance(ids, list):
            raise RuntimeError(f"{label}.{partition} arrays are malformed")
        if len(prompts) != count or len(hashes) != count or len(ids) != count:
            raise RuntimeError(f"{label}.{partition} arrays have the wrong length")
        expected_prefix = "warmup" if partition == "warmup" else "measured"
        for index, prompt in enumerate(prompts):
            tokens = _integer_array(prompt, f"{label}.input_ids.{partition}[{index}]")
            if len(tokens) != prompt_tokens:
                raise RuntimeError(f"{label}.input_ids.{partition}[{index}] length differs")
            expected_digest = _canonical_digest(tokens)
            if hashes[index] != expected_digest:
                raise RuntimeError(f"{label}.input_ids_sha256.{partition}[{index}] differs")
            expected_id = f"orbitkv-engine-e2e-{expected_prefix}-{seed}-{index}"
            if ids[index] != expected_id:
                raise RuntimeError(f"{label}.request_ids.{partition}[{index}] differs")
    all_prompts = inputs["warmup"] + inputs["measured"]
    if len({tuple(item) for item in all_prompts}) != len(all_prompts):
        raise RuntimeError(f"{label} input token IDs are not unique")
    return workload


def _validate_outputs(
    value: Any, workload: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    outputs = _exact(value, OUTPUT_KEYS, label)
    partitions = (
        ("warmups", "warmup", WARMUP_OUTPUT_KEYS),
        ("iterations", "measured", ITERATION_OUTPUT_KEYS),
    )
    for output_name, workload_name, keys in partitions:
        rows = outputs[output_name]
        count = workload["warmups" if output_name == "warmups" else "iterations"]
        if not isinstance(rows, list) or len(rows) != count:
            raise RuntimeError(f"{label}.{output_name} has the wrong length")
        for index, raw in enumerate(rows):
            item = _exact(raw, keys, f"{label}.{output_name}[{index}]")
            ordinal_name = "warmup" if output_name == "warmups" else "iteration"
            _integer(item[ordinal_name], f"{label}.{output_name}[{index}].{ordinal_name}", expected=index)
            if item["request_id"] != workload["request_ids"][workload_name][index]:
                raise RuntimeError(f"{label}.{output_name}[{index}] request ID differs")
            if item["input_ids_sha256"] != workload["input_ids_sha256"][workload_name][index]:
                raise RuntimeError(f"{label}.{output_name}[{index}] input digest differs")
            token_ids = _integer_array(
                item["output_ids"], f"{label}.{output_name}[{index}].output_ids"
            )
            if len(token_ids) != workload["decode_tokens"]:
                raise RuntimeError(f"{label}.{output_name}[{index}] output length differs")
            if item["output_ids_sha256"] != _canonical_digest(token_ids):
                raise RuntimeError(f"{label}.{output_name}[{index}] output digest differs")
            _integer(
                item["cached_tokens"],
                f"{label}.{output_name}[{index}].cached_tokens",
                expected=0,
            )
    payload = {"warmups": outputs["warmups"], "iterations": outputs["iterations"]}
    if outputs["aggregate_sha256"] != _canonical_digest(payload):
        raise RuntimeError(f"{label}.aggregate_sha256 differs from outputs")
    return outputs


def _validate_timings(
    value: Any, workload: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    timings = _exact(value, TIMING_KEYS, label)
    samples = timings["iteration_seconds"]
    if not isinstance(samples, list) or len(samples) != workload["iterations"]:
        raise RuntimeError(f"{label}.iteration_seconds has the wrong length")
    normalized = [
        _number(item, f"{label}.iteration_seconds[{index}]", positive=True)
        for index, item in enumerate(samples)
    ]
    total = sum(normalized)
    _expect_close(timings["median_seconds"], statistics.median(normalized), f"{label}.median_seconds")
    _expect_close(timings["p95_seconds"], _percentile(normalized, 0.95), f"{label}.p95_seconds")
    _expect_close(timings["measured_seconds"], total, f"{label}.measured_seconds")
    _expect_close(
        timings["output_tokens_per_second"],
        workload["iterations"] * workload["decode_tokens"] / total,
        f"{label}.output_tokens_per_second",
    )
    _expect_close(
        timings["total_tokens_per_second"],
        workload["iterations"]
        * (workload["prompt_tokens"] + workload["decode_tokens"])
        / total,
        f"{label}.total_tokens_per_second",
    )
    return timings


def _validate_server_snapshots(
    value: Any, mode: str, engine: Mapping[str, Any], label: str
) -> list[Mapping[str, Any]]:
    stages = ("after_load", "after_warmup", "after_workload", "final")
    if not isinstance(value, list) or len(value) != len(stages):
        raise RuntimeError(f"{label} must contain exactly four snapshots")
    expected_engine = {name: engine[name] for name in RESOLVED_ENGINE_KEYS if name != "effective_max_running_requests_per_dp"}
    expected_engine["effective_max_running_requests_per_dp"] = 1
    snapshots = []
    for index, (raw, stage) in enumerate(zip(value, stages, strict=True)):
        snapshot = _exact(raw, SERVER_SNAPSHOT_KEYS, f"{label}[{index}]")
        if snapshot["stage"] != stage:
            raise RuntimeError(f"{label}[{index}] stage differs")
        resolved = _exact(
            snapshot["resolved_engine"], RESOLVED_ENGINE_KEYS, f"{label}[{index}].resolved_engine"
        )
        if not _same(resolved, expected_engine):
            raise RuntimeError(f"{label}[{index}] engine readback differs")
        expected_present = mode == "manager"
        if snapshot["orbitkv_manager_present"] is not expected_present:
            raise RuntimeError(f"{label}[{index}] manager presence differs from mode")
        snapshots.append(snapshot)
    return snapshots


def _validate_runtime_admission(
    manifest_value: Any, binding_value: Any, engine: Mapping[str, Any], label: str
) -> tuple[
    Mapping[str, Any], Mapping[str, Any], str, tuple[Mapping[str, Any], ...], str
]:
    manifest = _exact(manifest_value, MANIFEST_KEYS, f"{label}.runtime_manifest")
    if manifest["schema"] != "orbitkv.runtime-manifest":
        raise RuntimeError(f"{label}.runtime_manifest schema or version differs")
    _integer(
        manifest["version"],
        f"{label}.runtime_manifest.version",
        expected=3,
    )
    manifest_fingerprint = _fingerprint(manifest, f"{label}.runtime_manifest")
    manager_input = _manager_input_from_attention_source(
        manifest, f"{label}.runtime_manifest"
    )
    manager_input_fingerprint = _manager_input_fingerprint(
        manifest, f"{label}.runtime_manifest"
    )
    binding = _exact(binding_value, BINDING_KEYS, f"{label}.runtime_binding")
    if binding["schema"] != "orbitkv.runtime-binding":
        raise RuntimeError(f"{label}.runtime_binding schema or version differs")
    _integer(
        binding["version"], f"{label}.runtime_binding.version", expected=1
    )
    binding_fingerprint = _fingerprint(binding, f"{label}.runtime_binding")
    if binding["manifest_fingerprint"] != manifest_fingerprint:
        raise RuntimeError(f"{label} runtime binding does not bind its manifest")
    signature = _exact(
        binding["execution_signature"],
        EXECUTION_SIGNATURE_KEYS,
        f"{label}.runtime_binding.execution_signature",
    )
    signature_fingerprint = _fingerprint(
        signature, f"{label}.runtime_binding.execution_signature"
    )
    if (
        signature["schema"] != "orbitkv.execution-signature"
        or type(signature["manifest_version"]) is not int
        or signature["manifest_schema"] != manifest["schema"]
        or signature["manifest_version"] != manifest["version"]
        or signature["manifest_fingerprint"] != manifest_fingerprint
        or signature["page_tokens"] != engine["page_size"]
    ):
        raise RuntimeError(f"{label} runtime admission identity differs")
    _integer(
        signature["version"],
        f"{label}.runtime_binding.execution_signature.version",
        expected=1,
    )
    target = _exact(
        binding["target"],
        {"id", "contract_version"},
        f"{label}.runtime_binding.target",
    )
    if target != {"id": "sglang", "contract_version": TARGET_CONTRACT_VERSION}:
        raise RuntimeError(f"{label} runtime binding target identity differs")
    admission_profile = _exact(
        binding["admission_profile"],
        {"id", "version"},
        f"{label}.runtime_binding.admission_profile",
    )
    _text(
        admission_profile["id"],
        f"{label}.runtime_binding.admission_profile.id",
    )
    _integer(
        admission_profile["version"],
        f"{label}.runtime_binding.admission_profile.version",
        positive=True,
    )
    target_fingerprint = binding["target_contract_fingerprint"]
    if target_fingerprint != TARGET_CONTRACT_FINGERPRINT:
        raise RuntimeError(
            f"{label}.runtime_binding target fingerprint differs from the current target"
        )
    _integer(
        binding["required_wire_version"],
        f"{label}.runtime_binding.required_wire_version",
        expected=WIRE_VERSION,
    )
    profile = _validate_native_session_profile(
        manifest, signature, manager_input, label
    )
    expected_topology, expected_cache_policy = _native_session_contract(
        signature, label
    )
    if binding["execution_topology"] != expected_topology:
        raise RuntimeError(f"{label} runtime admission topology differs")
    layout = manifest.get("token_manager_plan")
    if not isinstance(layout, Mapping) or not isinstance(layout.get("layout"), Mapping):
        raise RuntimeError(f"{label} token-manager layout is missing")
    layout_fingerprint = layout["layout"].get("plan_fingerprint")
    if (
        not isinstance(layout_fingerprint, str)
        or len(layout_fingerprint) != 71
        or not layout_fingerprint.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in layout_fingerprint[7:])
    ):
        raise RuntimeError(f"{label} plan fingerprint is malformed")
    expected_layout_fingerprint = _layout_fingerprint_from_manager_input(
        manager_input, f"{label}.runtime_manifest manager input"
    )
    if layout_fingerprint != expected_layout_fingerprint:
        raise RuntimeError(
            f"{label} layout fingerprint differs from the attention source"
        )
    if signature_fingerprint != signature["fingerprint"] or binding_fingerprint != binding["fingerprint"]:
        raise AssertionError("validated fingerprints changed")
    return manifest, binding, manager_input_fingerprint, profile, expected_cache_policy


def _nonnegative_counter_map(
    value: Any, required: Iterable[str], label: str, *, exact: bool = False
) -> Mapping[str, int]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    required_keys = set(required)
    if not required_keys <= set(value) or (exact and set(value) != required_keys):
        raise RuntimeError(f"{label} counter keys differ")
    if any(
        not isinstance(name, str)
        or isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        for name, item in value.items()
    ):
        raise RuntimeError(f"{label} contains an invalid counter")
    return value


def _validate_swa_activity(
    value: Any, *, sliding: bool, label: str
) -> Mapping[str, Any]:
    activity = _exact(value, SWA_ACTIVITY_KEYS, label)
    expected_metadata = {
        "status": "exposed" if sliding else "not_applicable",
        "applicable": sliding,
        "source": "native_runtime_session",
        "derived": False,
    }
    observed_metadata = {name: activity[name] for name in expected_metadata}
    if not _same(observed_metadata, expected_metadata):
        raise RuntimeError(f"{label} native-session SWA telemetry differs")
    counters = _nonnegative_counter_map(
        {name: activity[name] for name in SWA_COUNTER_FIELDS},
        SWA_COUNTER_FIELDS,
        label,
        exact=True,
    )
    if not sliding and any(counters.values()):
        raise RuntimeError(f"{label} non-Sliding SWA counters are nonzero")
    return activity


def _validate_arena_identity(
    value: Any, expected_class_id: int, page_tokens: int, label: str
) -> Mapping[str, int]:
    identity = _nonnegative_counter_map(value, IDENTITY_KEYS, label, exact=True)
    for name in ("engine_epoch", "pool_epoch", "pool_id", "backend_domain", "page_count", "first_page_id"):
        _bounded_positive_integer(identity[name], f"{label}.{name}", _U64_MAX)
    _integer(identity["class_id"], f"{label}.class_id", expected=expected_class_id)
    _integer(identity["page_tokens"], f"{label}.page_tokens", expected=page_tokens)
    _integer(identity["backend_base_index"], f"{label}.backend_base_index")
    if any(item > _U64_MAX for item in identity.values()):
        raise RuntimeError(f"{label} exceeds the supported integer range")
    if identity["backend_base_index"] > _U64_MAX - identity["page_count"]:
        raise RuntimeError(f"{label} backend index range overflows")
    return identity


def _validate_runtime_proof(
    value: Any,
    engine: Mapping[str, Any],
    expected_layers: Sequence[int],
    label: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    proof = _exact(value, RUNTIME_PROOF_KEYS, label)
    backend = _exact(proof["actual_attention_backend"], ACTUAL_BACKEND_KEYS, f"{label}.actual_attention_backend")
    scheduler = _exact(proof["effective_scheduler"], EFFECTIVE_SCHEDULER_KEYS, f"{label}.effective_scheduler")
    layer_ids = backend["compiled_layer_ids"]
    irope_ids = backend["use_irope_layer_ids"]
    if (
        backend["prefill_backend"] != engine["attention_backend"]
        or backend["decode_backend"] != engine["attention_backend"]
        or backend["page_size"] != engine["page_size"]
        or not isinstance(backend["has_local_attention"], bool)
        or isinstance(backend["attention_chunk_size"], bool)
        or not isinstance(backend["attention_chunk_size"], int)
        or not isinstance(layer_ids, list)
        or layer_ids != list(expected_layers)
        or not isinstance(irope_ids, list)
        or any(item not in layer_ids for item in irope_ids)
    ):
        raise RuntimeError(f"{label} backend proof differs from the admitted execution")
    _text(backend["backend_class"], f"{label}.actual_attention_backend.backend_class")
    _text(backend["backend_module"], f"{label}.actual_attention_backend.backend_module")
    expected_scheduler = {
        "max_prefill_tokens": engine["max_prefill_tokens"],
        "max_running_requests": 1,
        "effective_max_running_requests_per_dp": 1,
    }
    if scheduler != expected_scheduler:
        raise RuntimeError(f"{label} scheduler proof differs")
    return proof


def _validate_manager_snapshot(
    value: Any,
    stage: str,
    engine: Mapping[str, Any],
    expected_classes: Sequence[Mapping[str, Any]],
    expected_cache_policy: str,
    manifest_fingerprint: str,
    binding_fingerprint: str,
    manager_input_fingerprint: str,
    expected_sliding: bool,
    require_activity: bool,
    label: str,
) -> Mapping[str, Any]:
    snapshot = _exact(value, MANAGER_SNAPSHOT_KEYS, label)
    if snapshot["stage"] != stage:
        raise RuntimeError(f"{label}.stage differs")
    expected_provenance = {
        "wire_version": WIRE_VERSION,
        "runtime_manifest_fingerprint": manifest_fingerprint,
        "runtime_binding_fingerprint": binding_fingerprint,
        "manager_input_fingerprint": manager_input_fingerprint,
    }
    if any(snapshot[name] != expected for name, expected in expected_provenance.items()):
        raise RuntimeError(f"{label} runtime-session provenance differs")
    lifecycle_route = snapshot["lifecycle_route"]
    if lifecycle_route != "native_session":
        raise RuntimeError(f"{label} lifecycle route is not native_session")
    cache_policy = snapshot["cache_policy"]
    if cache_policy not in ("request_private", "shared_prefix"):
        raise RuntimeError(f"{label} cache policy is invalid")
    owner = _exact(snapshot["direct_source_owner"], OWNER_KEYS, f"{label}.direct_source_owner")
    if not _same(owner, {
        "module": "orbitkv_sglang.engine",
        "type": "OrbitKvLifecycleOwner",
        "owner_is_process_singleton": True,
        "config_is_canonical": True,
        "allocator_owned": True,
        "tree_cache_owned": True,
    }):
        raise RuntimeError(f"{label} direct-source owner proof failed")
    completion = _exact(snapshot["completion_evidence"], COMPLETION_KEYS, f"{label}.completion_evidence")
    if completion["event_backend"] != "cuda_event_current_forward_stream":
        raise RuntimeError(f"{label} completion backend is not current-stream CUDA event")
    _integer(
        completion["pending_events"],
        f"{label}.completion_evidence.pending_events",
        expected=0,
    )
    frontier = completion["completion_high_water"]
    if not isinstance(frontier, list):
        raise RuntimeError(f"{label} completion frontier is malformed")
    normalized_frontier = []
    domains = set()
    for index, raw in enumerate(frontier):
        point = _exact(raw, COMPLETION_POINT_KEYS, f"{label}.completion_evidence.completion_high_water[{index}]")
        domain = _integer(point["domain"], f"{label}.completion domain", positive=True)
        counter = _integer(point["value"], f"{label}.completion value", positive=True)
        if domain in domains:
            raise RuntimeError(f"{label} completion domain is duplicated")
        domains.add(domain)
        normalized_frontier.append((domain, counter))
    if normalized_frontier != sorted(normalized_frontier):
        raise RuntimeError(f"{label} completion frontier is not ordered and unique")
    if require_activity and not normalized_frontier:
        raise RuntimeError(f"{label} completion frontier does not prove activity")
    _validate_runtime_proof(
        snapshot["runtime_proof"],
        engine,
        sorted(layer for item in expected_classes for layer in item["layers"]),
        f"{label}.runtime_proof",
    )
    identities = snapshot["identities"]
    arenas = snapshot["arena_stats"]
    expected_class_ids = list(range(len(expected_classes)))
    if (
        not isinstance(identities, list)
        or not isinstance(arenas, list)
        or len(identities) != len(expected_class_ids)
        or len(arenas) != len(expected_class_ids)
    ):
        raise RuntimeError(
            f"{label} must contain one ordered arena identity and census per token class"
        )
    normalized_identities = [
        _validate_arena_identity(
            raw, class_id, engine["page_size"], f"{label}.identities[{class_id}]"
        )
        for class_id, raw in zip(expected_class_ids, identities, strict=True)
    ]
    normalized_arenas = [
        _nonnegative_counter_map(
            raw, ARENA_KEYS, f"{label}.arena_stats[{class_id}]", exact=True
        )
        for class_id, raw in zip(expected_class_ids, arenas, strict=True)
    ]
    expected_domains = {identity["backend_domain"] for identity in normalized_identities}
    if any(domain not in expected_domains for domain, _counter in normalized_frontier):
        raise RuntimeError(f"{label} completion frontier contains an unknown domain")
    stats = _nonnegative_counter_map(snapshot["manager_stats"], MANAGER_STATS_KEYS, f"{label}.manager_stats", exact=True)
    prefix_state = {
        name: stats[name]
        for name in (
            "active_prefixes",
            "evicted_prefixes",
            "total_prefix_page_refs",
        )
        if stats[name] != 0
    }
    if cache_policy == "request_private" and prefix_state:
        raise RuntimeError(
            f"{label} request-private prefix state is nonzero: {prefix_state}"
        )
    counters = _nonnegative_counter_map(
        snapshot["batch_counters"],
        BATCH_COUNTER_KEYS,
        f"{label}.batch_counters",
        exact=True,
    )
    _validate_swa_activity(
        snapshot["swa_activity"],
        sliding=expected_sliding,
        label=f"{label}.swa_activity",
    )
    pressure = _exact(snapshot["pressure"], PRESSURE_KEYS, f"{label}.pressure")
    expected_pressure = {
        "schema": "orbitkv.runtime-pressure.v1",
        "enabled": False,
        "mode": "event_driven_high_water",
        "sample_count": 0,
    }
    if not _same(pressure, expected_pressure):
        raise RuntimeError(f"{label} native-session pressure is not disabled")
    seen_pools: set[tuple[int, int, int]] = set()
    domain_ranges: dict[int, list[tuple[int, int]]] = {}
    for index, (identity, arena) in enumerate(
        zip(normalized_identities, normalized_arenas, strict=True)
    ):
        pool_key = (identity["engine_epoch"], identity["pool_epoch"], identity["pool_id"])
        if pool_key in seen_pools:
            raise RuntimeError(f"{label} arena pool identity is duplicated")
        seen_pools.add(pool_key)
        for name in ("engine_epoch", "pool_epoch", "pool_id", "class_id", "backend_domain", "page_count", "first_page_id"):
            if arena[name] != identity[name]:
                raise RuntimeError(f"{label} arena identity echo differs at class {index}")
        if sum(arena[name] for name in PAGE_PHASE_FIELDS) != arena["page_count"]:
            raise RuntimeError(f"{label} arena page conservation failed at class {index}")
        start = identity["backend_base_index"]
        end = start + identity["page_count"]
        ranges = domain_ranges.setdefault(identity["backend_domain"], [])
        if any(start < other_end and other_start < end for other_start, other_end in ranges):
            raise RuntimeError(f"{label} backend index ranges overlap")
        ranges.append((start, end))
    for name in PAGE_PHASE_FIELDS:
        if stats[name] != sum(arena[name] for arena in normalized_arenas):
            raise RuntimeError(f"{label} aggregate {name} differs from its arenas")
    references = {
        "total_request_page_refs": "request_page_refs",
        "total_prefix_page_refs": "prefix_page_refs",
        "total_reader_pins": "reader_pins",
    }
    if any(
        stats[total] != sum(arena[item] for arena in normalized_arenas)
        for total, item in references.items()
    ):
        raise RuntimeError(f"{label} aggregate reference census differs")
    if counters["fail_stop_count"] != 0:
        raise RuntimeError(f"{label} fail-stop counter is nonzero")
    if counters["forward_events"] != counters["completion_values"]:
        raise RuntimeError(f"{label} session completion counters differ")
    if require_activity:
        inactive = [name for name in PROGRESS_COUNTER_FIELDS if counters.get(name, 0) <= 0]
        if counters.get("event_queries", 0) + counters.get("event_waits", 0) <= 0:
            inactive.append("event_query_or_wait")
        if inactive:
            raise RuntimeError(f"{label} lifecycle activity is not proven: {inactive}")
    allowed_shared = (
        {"active_prefixes", "active_pages", "total_prefix_page_refs"}
        if cache_policy == "shared_prefix" and stage != "final"
        else set()
    )
    dirty = {
        name: stats[name]
        for name in DRAIN_FIELDS
        if name not in allowed_shared and stats[name] != 0
    }
    if dirty:
        raise RuntimeError(f"{label} lifecycle did not drain: {dirty}")
    if stage == "final" and any(
        arena["free_pages"] != arena["page_count"] for arena in normalized_arenas
    ):
        raise RuntimeError(f"{label} arenas did not drain")
    if cache_policy == "shared_prefix" and stage != "final" and any(
        arena[name] != 0
        for arena in normalized_arenas
        for name in (
            "reserved_pages",
            "writing_pages",
            "retiring_pages",
            "quarantined_pages",
            "request_page_refs",
            "reader_pins",
        )
    ):
        raise RuntimeError(f"{label} shared-prefix arena has unsafe transient state")
    return snapshot


def _validate_manager(
    value: Any,
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    expected_classes: Sequence[Mapping[str, Any]],
    expected_cache_policy: str,
    manager_input_fingerprint: str,
    engine: Mapping[str, Any],
    warmups: int,
    label: str,
) -> Mapping[str, Any]:
    manager = _exact(value, MANAGER_KEYS, label)
    wire = _integer(manager["wire_version"], f"{label}.wire_version", positive=True)
    if wire != WIRE_VERSION or wire != binding["required_wire_version"]:
        raise RuntimeError(f"{label} wire version differs from runtime binding")
    residency = _nonnegative_counter_map(
        manager["post_workload_residency"],
        POST_WORKLOAD_RESIDENCY_KEYS,
        f"{label}.post_workload_residency",
        exact=True,
    )
    stages = ("after_load", "after_warmup", "after_workload", "final")
    raw_snapshots = manager["snapshots"]
    if not isinstance(raw_snapshots, list) or len(raw_snapshots) != len(stages):
        raise RuntimeError(f"{label}.snapshots must contain four stages")
    snapshots = []
    has_sliding = any(
        state["backend"]["retention"] == "sliding"
        for state in binding["execution_signature"]["token_states"]
    )
    for index, (raw, stage) in enumerate(zip(raw_snapshots, stages, strict=True)):
        snapshot = _validate_manager_snapshot(
            raw,
            stage,
            engine,
            expected_classes,
            expected_cache_policy,
            manifest["fingerprint"],
            binding["fingerprint"],
            manager_input_fingerprint,
            has_sliding,
            stage in ("after_workload", "final")
            or (stage == "after_warmup" and warmups > 0),
            f"{label}.snapshots[{index}]",
        )
        snapshots.append(snapshot)
    if any(snapshots[0]["batch_counters"].values()) or snapshots[0][
        "completion_evidence"
    ]["completion_high_water"]:
        raise RuntimeError(f"{label} runtime session was not empty after load")
    if warmups == 0 and (
        any(snapshots[1]["batch_counters"].values())
        or snapshots[1]["completion_evidence"]["completion_high_water"]
    ):
        raise RuntimeError(f"{label} zero-warmup snapshot contains activity")
    if any(
        not _same(
            snapshot["direct_source_owner"],
            snapshots[0]["direct_source_owner"],
        )
        for snapshot in snapshots[1:]
    ):
        raise RuntimeError(f"{label} direct-source owner changed across lifecycle")
    if any(
        not _same(snapshot["identities"], snapshots[0]["identities"])
        for snapshot in snapshots[1:]
    ):
        raise RuntimeError(f"{label} arena identity changed across lifecycle")
    for field in ("lifecycle_route", "cache_policy"):
        if any(snapshot[field] != snapshots[0][field] for snapshot in snapshots[1:]):
            raise RuntimeError(f"{label} {field} changed across lifecycle")
    if snapshots[0]["lifecycle_route"] != "native_session":
        raise RuntimeError(f"{label} lifecycle route is not native_session")
    if snapshots[0]["cache_policy"] != expected_cache_policy:
        raise RuntimeError(
            f"{label} cache policy differs from the admitted native-session topology"
        )
    expected_radix_disabled = snapshots[0]["cache_policy"] == "request_private"
    if engine["disable_radix_cache"] is not expected_radix_disabled:
        raise RuntimeError(f"{label} cache policy differs from engine radix configuration")
    for before_snapshot, after_snapshot in zip(snapshots, snapshots[1:]):
        before_counters = before_snapshot["batch_counters"]
        after_counters = after_snapshot["batch_counters"]
        decreased = [
            name
            for name in BATCH_COUNTER_KEYS
            if after_counters[name] < before_counters[name]
        ]
        if decreased:
            raise RuntimeError(
                f"{label} session counters decreased: {sorted(decreased)}"
            )
        decreased_swa = [
            name
            for name in SWA_COUNTER_FIELDS
            if after_snapshot["swa_activity"][name]
            < before_snapshot["swa_activity"][name]
        ]
        if decreased_swa:
            raise RuntimeError(
                f"{label} SWA counters decreased: {sorted(decreased_swa)}"
            )
        before_frontier = {
            item["domain"]: item["value"]
            for item in before_snapshot["completion_evidence"][
                "completion_high_water"
            ]
        }
        after_frontier = {
            item["domain"]: item["value"]
            for item in after_snapshot["completion_evidence"][
                "completion_high_water"
            ]
        }
        if any(
            after_frontier.get(domain, -1) < counter
            for domain, counter in before_frontier.items()
        ):
            raise RuntimeError(
                f"{label} completion frontier regressed across lifecycle"
            )
    workload = snapshots[2]
    if not _same(
        dict(residency),
        {
            name: workload["manager_stats"][name]
            for name in POST_WORKLOAD_RESIDENCY_KEYS
        },
    ):
        raise RuntimeError(f"{label} residency differs from workload census")
    if snapshots[0]["cache_policy"] == "request_private" and any(residency.values()):
        raise RuntimeError(f"{label} request-private residency is nonzero")
    before = snapshots[1]["batch_counters"]
    after = snapshots[2]["batch_counters"]
    stalled = [name for name in PROGRESS_COUNTER_FIELDS if after.get(name, -1) <= before.get(name, -1)]
    if after.get("event_queries", 0) + after.get("event_waits", 0) <= before.get("event_queries", 0) + before.get("event_waits", 0):
        stalled.append("event_query_or_wait")
    before_frontier = {item["domain"]: item["value"] for item in snapshots[1]["completion_evidence"]["completion_high_water"]}
    after_frontier = {item["domain"]: item["value"] for item in snapshots[2]["completion_evidence"]["completion_high_water"]}
    if not any(value > before_frontier.get(domain, 0) for domain, value in after_frontier.items()):
        stalled.append("completion_frontier")
    if stalled:
        raise RuntimeError(f"{label} measured lifecycle did not advance: {stalled}")
    if has_sliding:
        warmup_swa = snapshots[1]["swa_activity"]
        workload_swa = snapshots[2]["swa_activity"]
        stalled_swa = [
            name
            for name in SWA_COUNTER_FIELDS
            if workload_swa[name] <= warmup_swa[name]
        ]
        if stalled_swa:
            raise RuntimeError(
                f"{label} Sliding workload did not advance native SWA counters: "
                f"{stalled_swa}"
            )
    return manager


def validate_record(value: Any, label: str = "record") -> dict[str, Any]:
    record = _exact(value, RECORD_KEYS, label)
    if record["schema"] != RECORD_SCHEMA:
        raise RuntimeError(f"{label}.schema is unsupported")
    mode = record["mode"]
    if mode not in ("stock", "manager"):
        raise RuntimeError(f"{label}.mode must be 'stock' or 'manager'")
    source = _validate_source(record["source"], mode, f"{label}.source")
    environment = _validate_environment(record["environment"], mode, f"{label}.environment")
    accelerator = _validate_accelerator(
        record["accelerator"], f"{label}.accelerator"
    )
    _text(record["model"], f"{label}.model")
    checkpoint = _validate_checkpoint(record["checkpoint"], f"{label}.checkpoint")
    engine = _validate_engine(record["engine_args"], mode, f"{label}.engine_args")
    if engine["model_path"] != record["model"]:
        raise RuntimeError(f"{label} model path differs from engine argument")
    _validate_server_snapshots(record["server_snapshots"], mode, engine, f"{label}.server_snapshots")
    workload = _validate_workload(record["workload"], engine, f"{label}.workload")
    _validate_sampling(record["sampling_params"], workload, f"{label}.sampling_params")
    outputs = _validate_outputs(record["outputs"], workload, f"{label}.outputs")
    timings = _validate_timings(record["timings"], workload, f"{label}.timings")
    if mode == "stock":
        if record["runtime_manifest"] is not None or record["runtime_binding"] is not None or record["manager"] is not None:
            raise RuntimeError(f"{label} stock record contains manager state")
    else:
        (
            manifest, binding, manager_input_fingerprint, expected_classes,
            expected_cache_policy,
        ) = _validate_runtime_admission(
            record["runtime_manifest"],
            record["runtime_binding"],
            engine,
            label,
        )
        manager = _validate_manager(
            record["manager"],
            manifest,
            binding,
            expected_classes,
            expected_cache_policy,
            manager_input_fingerprint,
            engine,
            workload["warmups"],
            f"{label}.manager",
        )
        retained_provenance = (
            manifest["fingerprint"],
            binding["fingerprint"],
            manager_input_fingerprint,
            manifest["token_manager_plan"]["layout"]["plan_fingerprint"],
        )
        # The producer validates these values before recording its normalized
        # snapshots.  Requiring every retained identity keeps that proof linked.
        if any(not isinstance(value, str) for value in retained_provenance):
            raise RuntimeError(f"{label} manager provenance is malformed")
        if manager["wire_version"] != binding["required_wire_version"]:
            raise RuntimeError(f"{label} manager wire provenance differs")
    return {
        "record": record,
        "mode": mode,
        "source": source,
        "environment": environment,
        "accelerator": accelerator,
        "checkpoint": checkpoint,
        "engine": engine,
        "workload": workload,
        "outputs": outputs,
        "timings": timings,
    }


def _normalized_environment(record: Mapping[str, Any]) -> dict[str, str]:
    environment = {
        name: value
        for name, value in record["environment"].items()
        if not name.startswith("ORBITKV_")
    }
    python_path = environment.get("PYTHONPATH")
    source_python = str(Path(record["source"]["root"]) / "python")
    if not isinstance(python_path, str):
        raise RuntimeError("record environment omits PYTHONPATH")
    entries = python_path.split(os.pathsep)
    if not entries or entries[0] != source_python:
        raise RuntimeError("record PYTHONPATH is not rooted at its SGLang source")
    entries[0] = "<sglang-source-root>/python"
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _normalized_engine(record: Mapping[str, Any]) -> dict[str, Any]:
    engine = dict(record["engine_args"])
    engine.pop("radix_cache_backend", None)
    engine.pop("disable_radix_cache", None)
    return engine


def _normalized_workload(record: Mapping[str, Any]) -> dict[str, Any]:
    workload = dict(record["workload"])
    request_ids = dict(workload["request_ids"])
    request_ids["warmup"] = list(range(len(request_ids["warmup"])))
    request_ids["measured"] = list(range(len(request_ids["measured"])))
    workload["request_ids"] = request_ids
    return workload


def _normalized_output_tokens(record: Mapping[str, Any], partition: str) -> list[list[int]]:
    return [list(item["output_ids"]) for item in record["outputs"][partition]]


def _normalized_source(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record["source"]
    return {"release": source["release"], "revision": source["revision"]}


def verify_pair(
    stock_value: Any, manager_value: Any, *, pair_index: int = 0
) -> dict[str, Any]:
    stock = validate_record(stock_value, f"pair[{pair_index}].stock")
    manager = validate_record(manager_value, f"pair[{pair_index}].manager")
    if stock["mode"] != "stock" or manager["mode"] != "manager":
        raise RuntimeError(f"pair[{pair_index}] requires STOCK then MANAGER records")
    stock_record = stock["record"]
    manager_record = manager["record"]
    comparisons = {
        "pinned source release and revision": (_normalized_source(stock_record), _normalized_source(manager_record)),
        "checkpoint identity": (stock["checkpoint"], manager["checkpoint"]),
        "workload and input IDs": (
            _normalized_workload(stock_record),
            _normalized_workload(manager_record),
        ),
        "sampling parameters": (stock_record["sampling_params"], manager_record["sampling_params"]),
        "engine arguments": (_normalized_engine(stock_record), _normalized_engine(manager_record)),
        "environment outside manager/source-root settings": (_normalized_environment(stock_record), _normalized_environment(manager_record)),
        "accelerator identity": (stock["accelerator"], manager["accelerator"]),
    }
    for description, (left, right) in comparisons.items():
        if not _same(left, right):
            raise RuntimeError(f"pair[{pair_index}] {description} differ")
    if not _same(
        _normalized_output_tokens(stock_record, "warmups"),
        _normalized_output_tokens(manager_record, "warmups"),
    ):
        raise RuntimeError(f"pair[{pair_index}] warmup output token IDs differ")
    stock_iterations = stock["outputs"]["iterations"]
    manager_iterations = manager["outputs"]["iterations"]
    for index, (stock_output, manager_output) in enumerate(
        zip(stock_iterations, manager_iterations, strict=True)
    ):
        if not _same(stock_output["output_ids"], manager_output["output_ids"]):
            raise RuntimeError(f"pair[{pair_index}] output token IDs differ at iteration {index}")
    stock_seconds = [float(value) for value in stock["timings"]["iteration_seconds"]]
    manager_seconds = [float(value) for value in manager["timings"]["iteration_seconds"]]
    latency_ratios = [manager_time / stock_time for stock_time, manager_time in zip(stock_seconds, manager_seconds, strict=True)]
    pair_median = statistics.median(latency_ratios)
    output_throughput_ratio = manager["timings"]["output_tokens_per_second"] / stock["timings"]["output_tokens_per_second"]
    total_throughput_ratio = manager["timings"]["total_tokens_per_second"] / stock["timings"]["total_tokens_per_second"]
    return {
        "pair_index": pair_index,
        "stock": {"path": None, "sha256": None},
        "manager": {"path": None, "sha256": None},
        "iterations": len(latency_ratios),
        "manager_to_stock_latency_ratios": latency_ratios,
        "latency_ratio_median": pair_median,
        "latency_ratio_p95": _percentile(latency_ratios, 0.95),
        "manager_to_stock_output_throughput_ratio": output_throughput_ratio,
        "manager_to_stock_total_throughput_ratio": total_throughput_ratio,
        "observed_direction": (
            "manager_lower_latency"
            if pair_median < 1.0
            else "manager_higher_latency"
            if pair_median > 1.0
            else "equal_latency"
        ),
    }


def verify_pairs(paths: Sequence[tuple[Path, Path]]) -> dict[str, Any]:
    if not paths:
        raise RuntimeError("at least one --pair STOCK MANAGER is required")
    pairs = []
    for index, (stock_path, manager_path) in enumerate(paths):
        stock = _strict_json(stock_path)
        manager = _strict_json(manager_path)
        result = verify_pair(stock, manager, pair_index=index)
        result["stock"] = {"path": str(stock_path), "sha256": _file_sha256(stock_path)}
        result["manager"] = {"path": str(manager_path), "sha256": _file_sha256(manager_path)}
        pairs.append(result)
    all_ratios = [value for pair in pairs for value in pair["manager_to_stock_latency_ratios"]]
    medians = [float(pair["latency_ratio_median"]) for pair in pairs]
    aggregate: dict[str, Any] = {
        "available": len(pairs) >= 3,
        "minimum_pair_count": 3,
        "pair_count": len(pairs),
        "scope": "descriptive_scoped_evidence_not_a_general_performance_claim",
        "all_pairs_same_direction": False,
        "conservative_direction_flag": None,
        "pair_median_latency_ratio_median": None,
        "pair_median_latency_ratio_p95": None,
        "all_iteration_latency_ratio_median": None,
        "all_iteration_latency_ratio_p95": None,
        "pair_output_throughput_ratio_median": None,
        "pair_total_throughput_ratio_median": None,
    }
    if len(pairs) >= 3:
        directions = {pair["observed_direction"] for pair in pairs}
        same = len(directions) == 1
        aggregate.update(
            all_pairs_same_direction=same,
            conservative_direction_flag=(next(iter(directions)) if same else None),
            pair_median_latency_ratio_median=statistics.median(medians),
            pair_median_latency_ratio_p95=_percentile(medians, 0.95),
            all_iteration_latency_ratio_median=statistics.median(all_ratios),
            all_iteration_latency_ratio_p95=_percentile(all_ratios, 0.95),
            pair_output_throughput_ratio_median=statistics.median(
                pair["manager_to_stock_output_throughput_ratio"] for pair in pairs
            ),
            pair_total_throughput_ratio_median=statistics.median(
                pair["manager_to_stock_total_throughput_ratio"] for pair in pairs
            ),
        )
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "claim": "pairwise_correctness_and_descriptive_timing_only",
        "speedup_qualified": False,
        "pair_count": len(pairs),
        "pairs": pairs,
        "aggregate": aggregate,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(f"cannot hash record {path}: {error}") from error
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("STOCK", "MANAGER"),
        type=Path,
        help="one stock and manager Engine E2E record; repeat for 3+ pair aggregation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = verify_pairs([(stock, manager) for stock, manager in args.pair])
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
