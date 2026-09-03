"""Strict validation for generic native-session qualification evidence."""

from __future__ import annotations

import math
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import UUID

from engine_e2e_verifier_contract import (
    layout_fingerprint_from_manager_input,
    manager_input_from_attention_source,
    native_session_contract,
    validate_native_session_profile,
)


NATIVE_RECORD_SCHEMA = "orbitkv.sglang-native-session-single-run.v1"
FULL_SLIDING_TOPOLOGY = "whole_domain_full_sliding_token_kv"
SLIDING_TOPOLOGY = "whole_domain_sliding_token_kv"
NATIVE_PROFILES = frozenset({FULL_SLIDING_TOPOLOGY, SLIDING_TOPOLOGY})

RECORD_KEYS = frozenset({
    "schema", "profile", "mode", "started_at_utc", "command",
    "command_sha256", "environment", "environment_sha256",
    "source_identity", "source_identity_sha256", "runtime_identity",
    "model", "checkpoint", "checkpoint_identity_sha256",
    "runtime_manifest", "runtime_binding", "engine_args",
    "sampling_params", "workload", "timings", "outputs",
    "server_capacity", "manager", "gpu_snapshots", "claims",
})
CHECKPOINT_KEYS = frozenset({"identity", "config", "backend_profile"})
CHECKPOINT_REQUIRED_KEYS = frozenset({
    "architectures", "num_hidden_layers", "vocab_size",
    "max_position_embeddings", "sliding_window", "control_token_ids",
})
CHECKPOINT_OPTIONAL_KEYS = frozenset({"layer_types", "attention_chunk_size"})
BACKEND_PROFILE_KEYS = frozenset({
    "attention_backend", "moe_runner_backend", "moe_a2a_backend",
    "ep_size",
})
MOE_RUNTIME_KEYS = frozenset({"runner", "a2a", "ep_size"})
ENGINE_REQUIRED_KEYS = frozenset({
    "model_path", "load_format", "dtype", "kv_cache_dtype",
    "skip_tokenizer_init", "trust_remote_code", "context_length",
    "page_size", "attention_backend", "disable_hybrid_swa_memory",
    "disable_overlap_schedule", "disable_radix_cache",
    "disable_cuda_graph", "enable_torch_compile",
    "enable_deterministic_inference", "sampling_backend",
    "chunked_prefill_size", "prefill_max_requests",
    "max_prefill_tokens", "enable_dynamic_chunking",
    "enable_mixed_chunk", "max_running_requests", "tp_size",
    "pp_size", "dp_size", "dcp_size", "enable_dp_attention",
    "speculative_algorithm", "disaggregation_mode",
    "enable_hierarchical_cache", "enable_streaming_session",
    "enable_unified_memory", "enable_pdmux", "enable_lmcache",
    "enable_flexkv", "enable_session_radix_cache", "enable_hisparse",
    "enable_page_major_kv_layout", "random_seed", "log_level",
    "max_total_tokens",
})
ENGINE_OPTIONAL_KEYS = frozenset({
    "mem_fraction_static", "swa_full_tokens_ratio",
    "moe_runner_backend", "moe_a2a_backend", "ep_size",
    "radix_cache_backend",
})
MANIFEST_RECORD_KEYS = frozenset({
    "artifact", "document", "manifest_fingerprint",
    "layout_plan_fingerprint", "execution_signature_fingerprint",
    "profile_geometry",
})
MANIFEST_KEYS = frozenset({
    "schema", "version", "fingerprint", "source",
    "token_manager_plan", "attention_state_plan",
    "capability_requirements",
})
BINDING_KEYS = frozenset({
    "schema", "version", "fingerprint", "manifest_fingerprint",
    "target", "admission_profile", "target_contract_fingerprint",
    "required_wire_version", "execution_topology",
    "execution_signature",
})
SIGNATURE_KEYS = frozenset({
    "schema", "version", "fingerprint", "manifest_schema",
    "manifest_version", "manifest_fingerprint", "page_tokens",
    "token_classes", "token_states", "fixed_states",
})
GEOMETRY_KEYS = frozenset({"page_tokens", "classes"})
CLASS_KEYS = frozenset({
    "class_id", "retention", "layers", "window_tokens",
    "minimum_resident_tokens",
})
WORKLOAD_KEYS = frozenset({
    "case", "requests", "prompt_tokens", "decode_tokens",
    "materialized_kv_tokens_per_request", "iterations", "seed",
    "fresh_prompts", "input_token_digest_sha256",
    "input_token_digests_by_iteration_sha256", "profile_geometry",
    "retirement_boundary_crossings_per_request",
})
CAPACITY_KEYS = frozenset({
    "status", "requested_tokens", "available_tokens", "failure",
    "class_capacities",
})
CAPACITY_OBSERVED_KEYS = CAPACITY_KEYS | frozenset({"floor"})
CLASS_CAPACITY_KEYS = frozenset({"full_tokens", "sliding_tokens"})
FLOOR_KEYS = frozenset({
    "configured_max_total_tokens", "expected_full_tokens",
    "expected_sliding_tokens", "full_floor_tokens",
    "sliding_floor_tokens",
})
SNAPSHOT_KEYS = frozenset({
    "stage", "identities", "manager_stats", "arena_stats",
    "batch_counters", "pressure", "runtime_proof",
    "lifecycle_route", "cache_policy", "swa_activity",
    "completion_evidence",
})
IDENTITY_KEYS = frozenset({
    "engine_epoch", "pool_epoch", "pool_id", "class_id",
    "backend_domain", "page_count", "page_tokens",
    "backend_base_index", "first_page_id",
})
ARENA_KEYS = frozenset({
    "engine_epoch", "pool_epoch", "pool_id", "page_count",
    "class_id", "backend_domain", "first_page_id", "free_pages",
    "reserved_pages", "writing_pages", "active_pages",
    "retiring_pages", "quarantined_pages", "exhausted_pages",
    "request_page_refs", "prefix_page_refs", "reader_pins",
})
MANAGER_STATS_KEYS = frozenset({
    "active_requests", "active_snapshots", "active_prefixes",
    "evicted_prefixes", "prepared_steps", "submitted_steps",
    "free_pages", "reserved_pages", "writing_pages", "active_pages",
    "retiring_pages", "quarantined_pages", "exhausted_pages",
    "pending_reclamations", "total_request_page_refs",
    "total_prefix_page_refs", "total_reader_pins",
})
SESSION_COUNTER_KEYS = frozenset({
    "forward_events", "completion_values", "event_queries",
    "event_waits", "fail_stop_count",
})
SWA_COUNTER_KEYS = (
    "swa_retirement_certificates", "swa_pages_reclaimed",
    "swa_wrap_events", "swa_page_reuse_events",
)
SWA_KEYS = frozenset(
    {"status", "applicable", "source", "derived", *SWA_COUNTER_KEYS}
)
COMPLETION_KEYS = frozenset(
    {"event_backend", "pending_events", "completion_high_water"}
)
COMPLETION_POINT_KEYS = frozenset({"domain", "value"})


def validate_json_tree(value: Any, label: str = "value") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        if not -(1 << 63) < value < (1 << 63):
            raise RuntimeError(f"{label} integer is outside the supported range")
        return
    if type(value) is float:
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0
        ):
            raise RuntimeError(
                f"{label} contains a non-finite or negative-zero number"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{label} contains a non-string key")
            validate_json_tree(item, f"{label}.{key}")
        return
    raise RuntimeError(f"{label} is not a JSON value")


def strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        if not isinstance(value, dict):
            raise ValueError("strict JSON value must be an object")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load strict JSON object {path}: {error}") from error
    validate_json_tree(value, str(path))
    return value


def binding_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_gpu_snapshots(
    value: Any, *, allow_partial: bool, label: str, gpu_keys: Iterable[str]
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must contain GPU snapshots")
    expected_stages = (
        "before_engine", "after_load", "after_workload",
        "after_shutdown",
    )
    stages: list[str] = []
    previous_time = -1
    identity = None
    for index, raw in enumerate(value):
        snapshot = _exact(
            raw, {"stage", "time_ns", "gpus"}, f"{label}[{index}]"
        )
        stage = _string(snapshot["stage"], f"{label}[{index}].stage")
        if stage not in expected_stages or stage in stages:
            raise RuntimeError(f"{label} contains an invalid or duplicate stage")
        stages.append(stage)
        time_ns = _integer(
            snapshot["time_ns"], f"{label}[{index}].time_ns", positive=True
        )
        if time_ns <= previous_time:
            raise RuntimeError(f"{label} timestamps are not increasing")
        previous_time = time_ns
        gpus = snapshot["gpus"]
        if not isinstance(gpus, list) or len(gpus) != 1:
            raise RuntimeError(f"{label} requires one GPU")
        rows = []
        for gpu in gpus:
            item = _exact(gpu, gpu_keys, f"{label}.gpu")
            for name in gpu_keys:
                _string(item[name], f"{label}.gpu.{name}")
            rows.append((item["index"], item["name"], item["uuid"]))
        current = tuple(rows)
        if identity is None:
            identity = current
        elif current != identity:
            raise RuntimeError(f"{label} GPU identity changed during the run")
    partial = {
        ("before_engine", "after_shutdown"),
        ("before_engine", "after_load", "after_shutdown"),
    }
    if (allow_partial and tuple(stages) not in partial) or (
        not allow_partial and tuple(stages) != expected_stages
    ):
        raise RuntimeError(f"{label} snapshot stage sequence is incomplete")
    assert identity is not None
    return identity


def _exact(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _object_with_keys(
    value: Any, required: Iterable[str], optional: Iterable[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    required_keys = set(required)
    allowed = required_keys | set(optional)
    actual = set(value)
    if not required_keys <= actual or not actual <= allowed:
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(required_keys - actual)} "
            f"extra={sorted(actual - allowed)}"
        )
    return value


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    if value < (1 if positive else 0) or value > (1 << 63) - 1:
        kind = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{label} must be a {kind} integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result == 0 and math.copysign(1, result) < 0):
        raise RuntimeError(f"{label} must be a finite non-negative-zero number")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"{label} must be a nonempty string")
    return value


def _layer_ids(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item < 0 for item in value)
        or value != sorted(set(value))
    ):
        raise RuntimeError(f"{label} must be increasing unique integers")
    return value


def validate_checkpoint(value: Any, profile: str, label: str) -> Mapping[str, Any]:
    checkpoint = _exact(value, CHECKPOINT_KEYS, label)
    config = _object_with_keys(
        checkpoint["config"], CHECKPOINT_REQUIRED_KEYS, CHECKPOINT_OPTIONAL_KEYS,
        f"{label}.config",
    )
    architectures = config["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise RuntimeError(f"{label}.config.architectures must contain one value")
    architecture = _string(architectures[0], f"{label}.config.architectures[0]")
    for name in (
        "num_hidden_layers", "vocab_size", "max_position_embeddings",
        "sliding_window",
    ):
        _integer(config[name], f"{label}.config.{name}", positive=True)
    if "attention_chunk_size" in config:
        _integer(config["attention_chunk_size"], f"{label}.config.attention_chunk_size", positive=True)
    controls = config["control_token_ids"]
    if not isinstance(controls, dict) or any(
        not isinstance(name, str) or type(item) is not int or item < 0
        for name, item in controls.items()
    ):
        raise RuntimeError(f"{label}.config.control_token_ids is invalid")
    layer_types = config.get("layer_types")
    layer_count = config["num_hidden_layers"]
    if layer_types is not None and (
        not isinstance(layer_types, list)
        or len(layer_types) != layer_count
        or any(item not in {"full_attention", "sliding_attention"} for item in layer_types)
    ):
        raise RuntimeError(f"{label}.config.layer_types is invalid")
    if profile == FULL_SLIDING_TOPOLOGY and (
        layer_types is None
        or "full_attention" not in layer_types
        or "sliding_attention" not in layer_types
    ):
        raise RuntimeError(f"{label} hybrid profile requires explicit Full/Sliding layers")
    if profile == SLIDING_TOPOLOGY and layer_types is not None and (
        layer_types != ["sliding_attention"] * layer_count
    ):
        raise RuntimeError(f"{label} pure Sliding requires every layer to slide")
    backend = _exact(checkpoint["backend_profile"], BACKEND_PROFILE_KEYS, f"{label}.backend_profile")
    if backend["attention_backend"] != "fa3":
        raise RuntimeError(f"{label}.backend_profile must bind FA3")
    moe = {"moe_runner_backend": "triton", "moe_a2a_backend": "none", "ep_size": 1}
    if architecture == "GptOssForCausalLM":
        if any(backend[name] != expected for name, expected in moe.items()):
            raise RuntimeError(f"{label} GPT-OSS MoE backend profile differs")
    elif any(backend[name] is not None for name in moe):
        raise RuntimeError(f"{label} non-GPT-OSS backend profile has MoE values")
    return checkpoint


def profile_artifact_digest(source_identity: Mapping[str, Any]) -> str:
    artifact = source_identity.get("profile_artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("native source identity omits profile_artifact")
    digest = artifact.get("sha256")
    if not isinstance(digest, str):
        raise RuntimeError("native profile artifact digest is malformed")
    return digest


def validate_geometry(value: Any, profile: str, label: str) -> dict[str, Any]:
    geometry = _exact(value, GEOMETRY_KEYS, label)
    page = _integer(geometry["page_tokens"], f"{label}.page_tokens", positive=True)
    if page != 16:
        raise RuntimeError(f"{label}.page_tokens must be 16")
    expected = ("full", "sliding") if profile == FULL_SLIDING_TOPOLOGY else ("sliding",)
    raw_classes = geometry["classes"]
    if not isinstance(raw_classes, list) or len(raw_classes) != len(expected):
        raise RuntimeError(f"{label} class count differs from profile")
    classes = []
    all_layers: list[int] = []
    for class_id, (raw, retention) in enumerate(zip(raw_classes, expected, strict=True)):
        item = _exact(raw, CLASS_KEYS, f"{label}.classes[{class_id}]")
        layers = _layer_ids(item["layers"], f"{label}.classes[{class_id}].layers")
        if any(layer in all_layers for layer in layers):
            raise RuntimeError(f"{label} class layers overlap")
        all_layers.extend(layers)
        if item["class_id"] != class_id or item["retention"] != retention:
            raise RuntimeError(f"{label} class order or retention differs")
        if retention == "full":
            if item["window_tokens"] is not None or item["minimum_resident_tokens"] is not None:
                raise RuntimeError(f"{label} Full class geometry differs")
        else:
            window = _integer(item["window_tokens"], f"{label}.window_tokens", positive=True)
            slots = 1 + (window - 1 + page - 1) // page
            if item["minimum_resident_tokens"] != slots * page:
                raise RuntimeError(f"{label} Sliding resident geometry differs")
        classes.append(dict(item))
    if sorted(all_layers) != list(range(len(all_layers))):
        raise RuntimeError(f"{label} classes do not cover every model layer")
    return {"page_tokens": page, "classes": classes}


def validate_workload(value: Any, profile: str, label: str, sha: Callable[[Any, str], str]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    workload = _exact(value, WORKLOAD_KEYS, label)
    if workload["case"] not in {"roomy", "exact-floor"}:
        raise RuntimeError(f"{label}.case must be roomy or exact-floor")
    for name in ("requests", "prompt_tokens", "decode_tokens", "materialized_kv_tokens_per_request", "iterations"):
        _integer(workload[name], f"{label}.{name}", positive=True)
    _integer(workload["seed"], f"{label}.seed")
    if workload["requests"] != 1 or workload["fresh_prompts"] is not True:
        raise RuntimeError(f"{label} must use one fresh deterministic request")
    sha(workload["input_token_digest_sha256"], f"{label}.input_token_digest_sha256")
    rows = workload["input_token_digests_by_iteration_sha256"]
    if not isinstance(rows, list) or len(rows) != workload["iterations"]:
        raise RuntimeError(f"{label} input digest iteration shape is invalid")
    for row in rows:
        if not isinstance(row, list) or len(row) != 1:
            raise RuntimeError(f"{label} input digest request shape is invalid")
        sha(row[0], f"{label} input digest")
    geometry = validate_geometry(workload["profile_geometry"], profile, f"{label}.profile_geometry")
    materialized = workload["prompt_tokens"] + workload["decode_tokens"] - 1
    sliding = next(item for item in geometry["classes"] if item["retention"] == "sliding")
    period_tokens = sliding["minimum_resident_tokens"]
    crossings = (materialized - 1) // period_tokens
    if (
        workload["materialized_kv_tokens_per_request"] != materialized
        or materialized <= period_tokens
        or workload["retirement_boundary_crossings_per_request"] != crossings
        or crossings <= 0
    ):
        raise RuntimeError(f"{label} derived Sliding geometry differs")
    return workload, geometry


def expected_checkpoint_geometry(checkpoint: Mapping[str, Any], geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = checkpoint["config"]
    types = config.get("layer_types")
    if types is None:
        groups = [("sliding", list(range(config["num_hidden_layers"])))]
    else:
        full = [index for index, kind in enumerate(types) if kind == "full_attention"]
        sliding = [index for index, kind in enumerate(types) if kind == "sliding_attention"]
        groups = ([(("full"), full)] if full else []) + [("sliding", sliding)]
    result = []
    for class_id, (retention, layers) in enumerate(groups):
        result.append({
            "class_id": class_id, "retention": retention, "layers": layers,
            "window_tokens": None if retention == "full" else config["sliding_window"],
            "minimum_resident_tokens": None if retention == "full" else geometry["classes"][-1]["minimum_resident_tokens"],
        })
    return result


def validate_engine(value: Any, mode: str, profile: str, geometry: Mapping[str, Any], checkpoint: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    engine = _object_with_keys(value, ENGINE_REQUIRED_KEYS, ENGINE_OPTIONAL_KEYS, label)
    expected = {
        "load_format": "auto", "dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False, "trust_remote_code": False,
        "page_size": geometry["page_tokens"], "attention_backend": "fa3",
        "disable_hybrid_swa_memory": False, "disable_overlap_schedule": True,
        "disable_radix_cache": profile == SLIDING_TOPOLOGY, "disable_cuda_graph": True,
        "enable_torch_compile": False, "enable_deterministic_inference": True,
        "sampling_backend": "pytorch", "prefill_max_requests": 1,
        "enable_dynamic_chunking": False, "enable_mixed_chunk": False,
        "max_running_requests": 1, "tp_size": 1, "pp_size": 1, "dp_size": 1,
        "dcp_size": 1, "enable_dp_attention": False, "speculative_algorithm": None,
        "disaggregation_mode": "null", "enable_hierarchical_cache": False,
        "enable_streaming_session": False, "enable_unified_memory": False,
        "enable_pdmux": False, "enable_lmcache": False, "enable_flexkv": False,
        "enable_session_radix_cache": False, "enable_hisparse": False,
        "enable_page_major_kv_layout": False, "log_level": "error",
    }
    for name, expected_value in expected.items():
        if not _same(engine[name], expected_value):
            raise RuntimeError(f"{label}.{name} differs from native execution contract")
    for name in ("context_length", "chunked_prefill_size", "max_prefill_tokens", "max_total_tokens"):
        _integer(engine[name], f"{label}.{name}", positive=True)
    _integer(engine["random_seed"], f"{label}.random_seed")
    if engine["chunked_prefill_size"] != engine["max_prefill_tokens"] or engine["chunked_prefill_size"] % geometry["page_tokens"] or engine["max_total_tokens"] % geometry["page_tokens"]:
        raise RuntimeError(f"{label} page geometry differs")
    if "mem_fraction_static" in engine:
        fraction = _number(engine["mem_fraction_static"], f"{label}.mem_fraction_static")
        if not 0 < fraction <= 1:
            raise RuntimeError(f"{label}.mem_fraction_static must be in (0, 1]")
    if profile == FULL_SLIDING_TOPOLOGY:
        ratio = _number(engine.get("swa_full_tokens_ratio"), f"{label}.swa_full_tokens_ratio")
        if not 0 < ratio <= 1:
            raise RuntimeError(f"{label}.swa_full_tokens_ratio must be in (0, 1]")
    elif "swa_full_tokens_ratio" in engine:
        raise RuntimeError(f"{label} pure Sliding must not set swa_full_tokens_ratio")
    moe = {"moe_runner_backend": "triton", "moe_a2a_backend": "none", "ep_size": 1}
    architecture = checkpoint["config"]["architectures"][0]
    present = {name for name in moe if name in engine}
    if architecture == "GptOssForCausalLM":
        if present != set(moe) or any(engine[name] != expected for name, expected in moe.items()):
            raise RuntimeError(f"{label} GPT-OSS MoE engine arguments differ")
    elif present:
        raise RuntimeError(f"{label} non-GPT-OSS engine has MoE arguments")
    for name in moe:
        if engine.get(name) != checkpoint["backend_profile"][name]:
            raise RuntimeError(f"{label} MoE arguments differ from checkpoint profile")
    if mode == "manager":
        if engine.get("radix_cache_backend") != "orbitkv":
            raise RuntimeError(f"{label} manager did not select OrbitKV")
    elif "radix_cache_backend" in engine:
        raise RuntimeError(f"{label} stock selected an implementation backend")
    return engine


def validate_manifest_binding(
    value: Any, binding_value: Any, profile: str, label: str, *,
    exact: Callable[[Any, Iterable[str], str], Mapping[str, Any]],
    fingerprint: Callable[[Any, str], str],
    binding_digest: Callable[[Mapping[str, Any]], str],
    absolute_path: Callable[[Any, str], str],
    sha: Callable[[Any, str], str],
    current_wire: int, target_version: int, target_fingerprint: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    record = _exact(value, MANIFEST_RECORD_KEYS, label)
    artifact = exact(record["artifact"], {"path", "bytes", "sha256"}, f"{label}.artifact")
    absolute_path(artifact["path"], f"{label}.artifact.path")
    _integer(artifact["bytes"], f"{label}.artifact.bytes", positive=True)
    sha(artifact["sha256"], f"{label}.artifact.sha256")
    manifest = _exact(record["document"], MANIFEST_KEYS, f"{label}.document")
    if manifest["schema"] != "orbitkv.runtime-manifest" or manifest["version"] != 3 or fingerprint(manifest["fingerprint"], f"{label}.document.fingerprint") != binding_digest(manifest):
        raise RuntimeError(f"{label}.document is not a canonical RuntimeManifest")
    if record["manifest_fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError(f"{label} manifest fingerprint echo differs")
    manager_input = manager_input_from_attention_source(manifest, f"{label}.document")
    capabilities = {"periodic_addressing", "semantic_retirement", "token_component_geometry", "token_manager"}
    if profile == FULL_SLIDING_TOPOLOGY:
        capabilities.add("append_only_addressing")
    if manifest["capability_requirements"] != sorted(capabilities):
        raise RuntimeError(f"{label} capability requirements differ")
    binding = _exact(binding_value, BINDING_KEYS, f"{label}.binding")
    if binding["schema"] != "orbitkv.runtime-binding" or binding["version"] != 1 or fingerprint(binding["fingerprint"], f"{label}.binding.fingerprint") != binding_digest(binding):
        raise RuntimeError(f"{label}.binding is not canonical")
    if binding["manifest_fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError(f"{label}.binding does not bind the manifest")
    target = _exact(binding["target"], {"id", "contract_version"}, f"{label}.binding.target")
    admission = _exact(binding["admission_profile"], {"id", "version"}, f"{label}.binding.admission_profile")
    if target != {"id": "sglang", "contract_version": target_version} or admission != {"id": "eager-single-device-bf16-nhd", "version": 1}:
        raise RuntimeError(f"{label}.binding target or admission profile differs")
    if binding["target_contract_fingerprint"] != target_fingerprint or binding["required_wire_version"] != current_wire or binding["execution_topology"] != profile:
        raise RuntimeError(f"{label}.binding target, wire, or topology differs")
    signature = _exact(binding["execution_signature"], SIGNATURE_KEYS, f"{label}.binding.execution_signature")
    if signature["schema"] != "orbitkv.execution-signature" or signature["version"] != 1 or signature["manifest_schema"] != manifest["schema"] or signature["manifest_version"] != manifest["version"] or signature["manifest_fingerprint"] != manifest["fingerprint"] or fingerprint(signature["fingerprint"], f"{label}.binding.execution_signature.fingerprint") != binding_digest(signature):
        raise RuntimeError(f"{label}.binding execution signature differs")
    topology, _policy = native_session_contract(signature, label)
    if topology != profile:
        raise RuntimeError(f"{label} profile and signature topology differ")
    classes = validate_native_session_profile(manifest, signature, manager_input, label)
    layout = manifest["token_manager_plan"]["layout"]
    if record["layout_plan_fingerprint"] != layout["plan_fingerprint"] or layout["plan_fingerprint"] != layout_fingerprint_from_manager_input(manager_input, label) or record["execution_signature_fingerprint"] != signature["fingerprint"]:
        raise RuntimeError(f"{label} compiled-plan fingerprint echo differs")
    geometry = signature_geometry(signature, profile, label)
    if not _same(record["profile_geometry"], geometry) or len(classes) != len(geometry["classes"]):
        raise RuntimeError(f"{label}.profile_geometry differs from binding")
    return manifest, binding, geometry


def signature_geometry(signature: Mapping[str, Any], profile: str, label: str) -> dict[str, Any]:
    result = []
    for class_id, (token_class, state) in enumerate(zip(signature["token_classes"], signature["token_states"], strict=True)):
        retention = state["backend"]["retention"]
        result.append({
            "class_id": class_id, "retention": retention, "layers": list(token_class["layers"]),
            "window_tokens": state["backend"]["window_tokens"],
            "minimum_resident_tokens": None if retention == "full" else token_class["minimum_slots_per_request"] * signature["page_tokens"],
        })
    return validate_geometry({"page_tokens": signature["page_tokens"], "classes": result}, profile, label)


def validate_capacity(value: Any, workload: Mapping[str, Any], geometry: Mapping[str, Any], profile: str, engine: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    observed = value.get("status") == "observed"
    capacity = _exact(value, CAPACITY_OBSERVED_KEYS if observed else CAPACITY_KEYS, label)
    requested = _integer(capacity["requested_tokens"], f"{label}.requested_tokens", positive=True)
    if engine["max_total_tokens"] != requested:
        raise RuntimeError(f"{label} engine capacity differs")
    if observed:
        if capacity["available_tokens"] != requested or capacity["failure"] is not None:
            raise RuntimeError(f"{label} observed capacity is inconsistent")
    elif capacity["status"] == "failed":
        if capacity["available_tokens"] is not None:
            _integer(capacity["available_tokens"], f"{label}.available_tokens", positive=True)
        failure = _exact(capacity["failure"], {"type", "message"}, f"{label}.failure")
        _string(failure["type"], f"{label}.failure.type")
        _string(failure["message"], f"{label}.failure.message")
    else:
        raise RuntimeError(f"{label}.status must be observed or failed")
    classes = capacity["class_capacities"]
    if classes is None:
        if observed:
            raise RuntimeError(f"{label} observed capacity omits class capacities")
        return capacity
    classes = _nonnegative_map(classes, CLASS_CAPACITY_KEYS, f"{label}.class_capacities")
    page = geometry["page_tokens"]
    expected_sliding = requested if profile == SLIDING_TOPOLOGY else int(requested * float(engine["swa_full_tokens_ratio"])) // page * page
    if classes["sliding_tokens"] != expected_sliding or classes["sliding_tokens"] <= 0 or classes["full_tokens"] != (requested if profile == FULL_SLIDING_TOPOLOGY else 0):
        raise RuntimeError(f"{label} class capacities differ")
    sliding = next(item for item in geometry["classes"] if item["retention"] == "sliding")
    sliding_floor = sliding["minimum_resident_tokens"] + _ceil_to(
        engine["chunked_prefill_size"], page
    )
    full_floor = (
        max(
            _ceil_to(workload["materialized_kv_tokens_per_request"] + 1, page),
            sliding_floor,
        )
        if profile == FULL_SLIDING_TOPOLOGY
        else 0
    )
    if not observed:
        if workload["case"] != "exact-floor":
            raise RuntimeError(f"{label} failed capacity is not exact-floor")
        if (
            profile == SLIDING_TOPOLOGY and requested != sliding_floor
            or profile == FULL_SLIDING_TOPOLOGY
            and (requested != full_floor or classes["sliding_tokens"] != sliding_floor)
        ):
            raise RuntimeError(f"{label} failed capacity does not use exact class floors")
        return capacity
    floor = _nonnegative_map(capacity["floor"], FLOOR_KEYS, f"{label}.floor")
    expected_floor = {
        "configured_max_total_tokens": requested, "expected_full_tokens": classes["full_tokens"],
        "expected_sliding_tokens": classes["sliding_tokens"], "full_floor_tokens": full_floor,
        "sliding_floor_tokens": sliding_floor,
    }
    if not _same(floor, expected_floor):
        raise RuntimeError(f"{label} class capacity floor differs")
    case = workload["case"]
    if profile == SLIDING_TOPOLOGY:
        valid = requested > sliding_floor if case == "roomy" else requested == sliding_floor
    else:
        valid = (requested >= workload["materialized_kv_tokens_per_request"] and classes["sliding_tokens"] > sliding_floor) if case == "roomy" else (requested == full_floor and classes["sliding_tokens"] == sliding_floor)
    if not valid:
        raise RuntimeError(f"{label} {profile} {case} capacity differs")
    return capacity


def _ceil_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _nonnegative_map(value: Any, keys: Iterable[str], label: str) -> Mapping[str, int]:
    result = _exact(value, keys, label)
    for name in keys:
        _integer(result[name], f"{label}.{name}")
    return result  # type: ignore[return-value]


def _validate_swa(value: Any, label: str) -> Mapping[str, int]:
    activity = _exact(value, SWA_KEYS, label)
    if activity["status"] != "exposed" or activity["applicable"] is not True or activity["source"] != "native_runtime_session" or activity["derived"] is not False:
        raise RuntimeError(f"{label} must be direct applicable native-session evidence")
    for name in SWA_COUNTER_KEYS:
        _integer(activity[name], f"{label}.{name}")
    return activity  # type: ignore[return-value]


def _validate_completion(value: Any, label: str, *, active: bool) -> dict[int, int]:
    evidence = _exact(value, COMPLETION_KEYS, label)
    if evidence["event_backend"] != "cuda_event_current_forward_stream" or evidence["pending_events"] != 0:
        raise RuntimeError(f"{label} completion backend or drain differs")
    frontier = evidence["completion_high_water"]
    if not isinstance(frontier, list):
        raise RuntimeError(f"{label}.completion_high_water must be a list")
    result = {}
    previous = 0
    for index, raw in enumerate(frontier):
        point = _exact(raw, COMPLETION_POINT_KEYS, f"{label}[{index}]")
        domain = _integer(point["domain"], f"{label}[{index}].domain", positive=True)
        counter = _integer(point["value"], f"{label}[{index}].value", positive=True)
        if domain <= previous:
            raise RuntimeError(f"{label} domains are duplicate or unordered")
        previous = domain
        result[domain] = counter
    if active and not result:
        raise RuntimeError(f"{label} completion frontier is empty")
    return result


def _drained(snapshot: Mapping[str, Any]) -> bool:
    fields = ("active_requests", "active_snapshots", "active_prefixes", "prepared_steps", "submitted_steps", "reserved_pages", "writing_pages", "active_pages", "retiring_pages", "quarantined_pages", "pending_reclamations", "total_request_page_refs", "total_prefix_page_refs", "total_reader_pins")
    return all(snapshot["manager_stats"][name] == 0 for name in fields) and all(arena["free_pages"] == arena["page_count"] for arena in snapshot["arena_stats"])


def validate_manager(value: Any, workload: Mapping[str, Any], runtime_proof: Any, profile: str, binding: Mapping[str, Any], label: str, *, current_wire: int) -> None:
    manager = _exact(value, {"wire_version", "snapshots"}, label)
    if manager["wire_version"] != current_wire or manager["wire_version"] != binding["required_wire_version"]:
        raise RuntimeError(f"{label} wire provenance differs")
    snapshots = manager["snapshots"]
    stages = ("after_load", "after_workload", "final")
    if not isinstance(snapshots, list) or len(snapshots) != 3:
        raise RuntimeError(f"{label}.snapshots must contain three stages")
    geometry = workload["profile_geometry"]
    count = len(geometry["classes"])
    policy = "shared_prefix" if profile == FULL_SLIDING_TOPOLOGY else "request_private"
    previous_counters = None
    previous_swa = None
    previous_frontier: dict[int, int] = {}
    identities_once = None
    for index, (raw, stage) in enumerate(zip(snapshots, stages, strict=True)):
        snapshot = _exact(raw, SNAPSHOT_KEYS, f"{label}.snapshots[{index}]")
        if snapshot["stage"] != stage or snapshot["lifecycle_route"] != "native_session" or snapshot["cache_policy"] != policy:
            raise RuntimeError(f"{label} lifecycle/cache policy differs")
        identities, arenas = snapshot["identities"], snapshot["arena_stats"]
        if not isinstance(identities, list) or not isinstance(arenas, list) or len(identities) != count or len(arenas) != count:
            raise RuntimeError(f"{label} arena count differs from profile")
        normalized_arenas = []
        seen_pools: set[tuple[int, int, int]] = set()
        domain_ranges: dict[int, list[tuple[int, int]]] = {}
        for class_id, (identity_value, arena_value) in enumerate(zip(identities, arenas, strict=True)):
            identity = _nonnegative_map(identity_value, IDENTITY_KEYS, f"{label}.identity[{class_id}]")
            arena = _nonnegative_map(arena_value, ARENA_KEYS, f"{label}.arena[{class_id}]")
            if identity["class_id"] != class_id or arena["class_id"] != class_id or identity["page_tokens"] != geometry["page_tokens"]:
                raise RuntimeError(f"{label} arena class/page geometry differs")
            for name in ("engine_epoch", "pool_epoch", "pool_id", "page_count", "class_id", "backend_domain", "first_page_id"):
                if arena[name] != identity[name]:
                    raise RuntimeError(f"{label} arena identity echo differs")
            pool = (
                identity["engine_epoch"], identity["pool_epoch"],
                identity["pool_id"],
            )
            if pool in seen_pools:
                raise RuntimeError(f"{label} arena pool identity is duplicated")
            seen_pools.add(pool)
            start = identity["backend_base_index"]
            end = start + identity["page_count"]
            if end > (1 << 63) - 1:
                raise RuntimeError(f"{label} arena backend range overflows")
            ranges = domain_ranges.setdefault(identity["backend_domain"], [])
            if any(
                start < other_end and other_start < end
                for other_start, other_end in ranges
            ):
                raise RuntimeError(f"{label} arena backend ranges overlap")
            ranges.append((start, end))
            phases = ("free_pages", "reserved_pages", "writing_pages", "active_pages", "retiring_pages", "quarantined_pages", "exhausted_pages")
            if sum(arena[name] for name in phases) != arena["page_count"]:
                raise RuntimeError(f"{label} arena phases do not sum to capacity")
            normalized_arenas.append(arena)
        if identities_once is None:
            identities_once = identities
        elif not _same(identities, identities_once):
            raise RuntimeError(f"{label} arena identity changed")
        stats = _nonnegative_map(snapshot["manager_stats"], MANAGER_STATS_KEYS, f"{label}.manager_stats")
        aggregate = ("free_pages", "reserved_pages", "writing_pages", "active_pages", "retiring_pages", "quarantined_pages", "exhausted_pages")
        if any(stats[name] != sum(arena[name] for arena in normalized_arenas) for name in aggregate) or stats["total_request_page_refs"] != sum(arena["request_page_refs"] for arena in normalized_arenas) or stats["total_prefix_page_refs"] != sum(arena["prefix_page_refs"] for arena in normalized_arenas) or stats["total_reader_pins"] != sum(arena["reader_pins"] for arena in normalized_arenas):
            raise RuntimeError(f"{label} manager/arena aggregate differs")
        counters = _nonnegative_map(snapshot["batch_counters"], SESSION_COUNTER_KEYS, f"{label}.batch_counters")
        if previous_counters is not None and any(counters[name] < previous_counters[name] for name in SESSION_COUNTER_KEYS):
            raise RuntimeError(f"{label} session counters decreased")
        if counters["fail_stop_count"] or counters["forward_events"] != counters["completion_values"]:
            raise RuntimeError(f"{label} runtime-session lifecycle is invalid")
        expected_pressure = {"schema": "orbitkv.runtime-pressure.v1", "enabled": False, "mode": "event_driven_high_water", "sample_count": 0}
        if not _same(snapshot["pressure"], expected_pressure):
            raise RuntimeError(f"{label} native-session pressure must be disabled")
        swa = _validate_swa(snapshot["swa_activity"], f"{label}.swa_activity")
        frontier = _validate_completion(snapshot["completion_evidence"], f"{label}.completion_evidence", active=stage != "after_load")
        domains = {item["backend_domain"] for item in identities}
        if any(domain not in domains for domain in frontier):
            raise RuntimeError(f"{label} completion frontier has an unknown domain")
        if previous_swa is not None and any(swa[name] < previous_swa[name] for name in SWA_COUNTER_KEYS):
            raise RuntimeError(f"{label} SWA counters decreased")
        if any(frontier.get(domain, -1) < counter for domain, counter in previous_frontier.items()):
            raise RuntimeError(f"{label} completion frontier regressed")
        if not _same(snapshot["runtime_proof"], runtime_proof):
            raise RuntimeError(f"{label} runtime proof changed")
        previous_counters, previous_swa, previous_frontier = counters, swa, frontier
    initial, active, final = snapshots
    if workload["case"] == "roomy" and any(
        active["swa_activity"][name] <= initial["swa_activity"][name]
        for name in SWA_COUNTER_KEYS
    ):
        raise RuntimeError(f"{label} Sliding roomy workload did not advance all SWA counters")
    delta = {name: active["batch_counters"][name] - initial["batch_counters"][name] for name in SESSION_COUNTER_KEYS}
    if delta["forward_events"] <= 0 or delta["completion_values"] <= 0 or delta["event_queries"] + delta["event_waits"] <= 0:
        raise RuntimeError(f"{label} runtime-session lifecycle did not advance")
    before_frontier = {item["domain"]: item["value"] for item in initial["completion_evidence"]["completion_high_water"]}
    active_frontier = {item["domain"]: item["value"] for item in active["completion_evidence"]["completion_high_water"]}
    if not any(value > before_frontier.get(domain, 0) for domain, value in active_frontier.items()):
        raise RuntimeError(f"{label} completion frontier did not advance")
    if not _drained(initial) or not _drained(final):
        raise RuntimeError(f"{label} initial/final state did not fully drain")
    if policy == "request_private":
        if not _drained(active):
            raise RuntimeError(f"{label} request-private workload did not drain")
    else:
        forbidden = ("active_requests", "active_snapshots", "prepared_steps", "submitted_steps", "reserved_pages", "writing_pages", "retiring_pages", "quarantined_pages", "pending_reclamations", "total_request_page_refs", "total_reader_pins")
        if any(active["manager_stats"][name] for name in forbidden) or active["manager_stats"]["active_pages"] != active["manager_stats"]["total_prefix_page_refs"]:
            raise RuntimeError(f"{label} shared-prefix workload did not settle")


def validate_runtime_identity(value: Any, mode: str, label: str, keys: Iterable[str]) -> Mapping[str, Any]:
    identity = _object_with_keys(value, keys, {"moe_backend"}, label)
    for name in ("python_executable", "python_version", "platform", "sglang_version", "sglang_package", "sampling_backend"):
        _string(identity[name], f"{label}.{name}")
    try:
        run_id = UUID(_string(identity["run_id"], f"{label}.run_id"))
    except ValueError as error:
        raise RuntimeError(f"{label}.run_id is not a canonical UUIDv4") from error
    if run_id.version != 4 or str(run_id) != identity["run_id"]:
        raise RuntimeError(f"{label}.run_id is not a canonical UUIDv4")
    if mode == "stock" and identity["runtime_proof"] is not None:
        raise RuntimeError(f"{label}.runtime_proof must be null for stock")
    if mode == "manager" and identity["runtime_proof"] is not None and not isinstance(identity["runtime_proof"], dict):
        raise RuntimeError(f"{label}.runtime_proof is malformed")
    expected = {"kv_layout": "nhd", "attention_backend": "fa3", "dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "execution": "eager", "tp_size": 1, "pp_size": 1, "dp_size": 1, "dcp_size": 1, "deterministic_inference": True}
    if any(not _same(identity[name], expected_value) for name, expected_value in expected.items()):
        raise RuntimeError(f"{label} differs from exact execution contract")
    return identity


def validate_moe_runtime_identity(
    identity: Mapping[str, Any], checkpoint: Mapping[str, Any], label: str
) -> None:
    architecture = checkpoint["config"]["architectures"][0]
    if architecture == "GptOssForCausalLM":
        moe = _exact(identity.get("moe_backend"), MOE_RUNTIME_KEYS, f"{label}.moe_backend")
        if moe != {"runner": "triton", "a2a": "none", "ep_size": 1}:
            raise RuntimeError(f"{label}.moe_backend differs for GPT-OSS")
    elif "moe_backend" in identity:
        raise RuntimeError(f"{label}.moe_backend is only valid for GPT-OSS")


def validate_runtime_proof(value: Any, engine: Mapping[str, Any], label: str, exact: Callable[[Any, Iterable[str], str], Mapping[str, Any]]) -> None:
    if value is None:
        return
    proof = exact(value, {"actual_attention_backend", "effective_scheduler"}, label)
    scheduler = exact(proof["effective_scheduler"], {"max_prefill_tokens", "max_running_requests", "effective_max_running_requests_per_dp"}, f"{label}.effective_scheduler")
    expected = {"max_prefill_tokens": engine["max_prefill_tokens"], "max_running_requests": 1, "effective_max_running_requests_per_dp": 1}
    if not _same(scheduler, expected) or not isinstance(proof["actual_attention_backend"], dict):
        raise RuntimeError(f"{label} runtime proof differs")


def validate_record(
    record: Any, label: str, *, exact: Callable, json_tree: Callable, sha: Callable,
    canonical_digest: Callable, binding_digest: Callable, absolute_path: Callable,
    validate_source_binding: Callable, validate_outputs: Callable,
    validate_gpu_snapshots: Callable, validate_claims: Callable,
    runtime_identity_keys: Iterable[str], sampling_keys: Iterable[str],
    timing_keys: Iterable[str], current_wire: int, target_version: int,
    target_fingerprint: str, fingerprint: Callable[[Any, str], str],
) -> dict[str, Any]:
    json_tree(record, label)
    value = exact(record, RECORD_KEYS, label)
    if value["schema"] != NATIVE_RECORD_SCHEMA or value["profile"] not in NATIVE_PROFILES or value["mode"] not in {"stock", "manager"}:
        raise RuntimeError(f"{label} schema, profile, or mode is unsupported")
    profile, mode = value["profile"], value["mode"]
    try:
        started = datetime.fromisoformat(_string(value["started_at_utc"], f"{label}.started_at_utc"))
    except ValueError as error:
        raise RuntimeError(f"{label}.started_at_utc is not ISO-8601") from error
    if started.tzinfo is None:
        raise RuntimeError(f"{label}.started_at_utc lacks a timezone")
    command = value["command"]
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
        raise RuntimeError(f"{label}.command is invalid")
    for payload, digest in (("command", "command_sha256"), ("environment", "environment_sha256"), ("source_identity", "source_identity_sha256"), ("checkpoint", "checkpoint_identity_sha256")):
        if value[digest] != canonical_digest(value[payload]):
            raise RuntimeError(f"{label}.{digest} differs from its payload")
        sha(value[digest], f"{label}.{digest}")
    if not isinstance(value["environment"], dict) or not isinstance(value["source_identity"], dict) or not isinstance(value["engine_args"], dict):
        raise RuntimeError(f"{label} provenance objects are malformed")
    identity = validate_runtime_identity(
        value["runtime_identity"], mode, f"{label}.runtime_identity",
        runtime_identity_keys,
    )
    validate_source_binding(
        value["source_identity"], mode, command=command,
        environment=value["environment"],
        sglang_package=identity["sglang_package"], native=True,
    )
    model = absolute_path(value["model"], f"{label}.model")
    checkpoint = validate_checkpoint(value["checkpoint"], profile, f"{label}.checkpoint")
    validate_moe_runtime_identity(
        identity, checkpoint, f"{label}.runtime_identity"
    )
    workload, geometry = validate_workload(value["workload"], profile, f"{label}.workload", sha)
    if not _same(geometry["classes"], expected_checkpoint_geometry(checkpoint, geometry)):
        raise RuntimeError(f"{label} checkpoint and profile geometry differ")
    engine = validate_engine(value["engine_args"], mode, profile, geometry, checkpoint, f"{label}.engine_args")
    if engine["model_path"] != model or engine["context_length"] > checkpoint["config"]["max_position_embeddings"] or workload["materialized_kv_tokens_per_request"] >= engine["context_length"] or engine["random_seed"] != workload["seed"]:
        raise RuntimeError(f"{label} model, context, or seed differs")
    sampling = exact(value["sampling_params"], sampling_keys, f"{label}.sampling_params")
    if _number(sampling["temperature"], f"{label}.sampling_params.temperature") != 0 or sampling["ignore_eos"] is not True or sampling["max_new_tokens"] != workload["decode_tokens"] or sampling["min_new_tokens"] != workload["decode_tokens"] or sampling["sampling_seed"] != workload["seed"]:
        raise RuntimeError(f"{label} sampling differs from workload")
    capacity = validate_capacity(value["server_capacity"], workload, geometry, profile, engine, f"{label}.server_capacity")
    failed = capacity["status"] == "failed"
    if failed and (mode != "stock" or workload["case"] != "exact-floor"):
        raise RuntimeError(f"{label} contains an inadmissible execution failure")
    timings = exact(value["timings"], timing_keys, f"{label}.timings")
    if _number(timings["load_seconds"], f"{label}.timings.load_seconds") < 0 or _number(timings["total_seconds"], f"{label}.timings.total_seconds") <= 0:
        raise RuntimeError(f"{label} timing values are invalid")
    samples = timings["iteration_seconds"]
    if not isinstance(samples, list) or len(samples) != (0 if failed else workload["iterations"]):
        raise RuntimeError(f"{label}.timings.iteration_seconds shape is invalid")
    for item in samples:
        if _number(item, f"{label}.timings.iteration_seconds") <= 0:
            raise RuntimeError(f"{label} timing sample must be positive")
    validate_outputs(value["outputs"], workload, checkpoint, allow_empty=failed, label=f"{label}.outputs")
    if mode == "stock":
        if value["runtime_manifest"] is not None or value["runtime_binding"] is not None or value["manager"] is not None:
            raise RuntimeError(f"{label} stock record contains manager state")
        if "SGLANG_PLUGINS" in value["environment"] or any(name.startswith("ORBITKV_") for name in value["environment"]):
            raise RuntimeError(f"{label} stock environment selected OrbitKV")
    else:
        manifest, binding, manifest_geometry = validate_manifest_binding(value["runtime_manifest"], value["runtime_binding"], profile, f"{label}.runtime_manifest", exact=exact, fingerprint=fingerprint, binding_digest=binding_digest, absolute_path=absolute_path, sha=sha, current_wire=current_wire, target_version=target_version, target_fingerprint=target_fingerprint)
        artifact = value["runtime_manifest"]["artifact"]
        source_artifact = value["source_identity"]["profile_artifact"]
        if (
            artifact["path"] != source_artifact["path"]
            or artifact["bytes"] != source_artifact["bytes"]
            or artifact["sha256"] != source_artifact["sha256"]
        ):
            raise RuntimeError(
                f"{label} manager manifest and profile artifact differ"
            )
        if not _same(manifest_geometry, geometry):
            raise RuntimeError(f"{label} manifest and workload geometry differ")
        validate_runtime_proof(identity["runtime_proof"], engine, f"{label}.runtime_identity.runtime_proof", exact)
        if "SGLANG_PLUGINS" in value["environment"]:
            raise RuntimeError(f"{label} manager environment selected a plugin")
        validate_manager(value["manager"], workload, identity["runtime_proof"], profile, binding, f"{label}.manager", current_wire=current_wire)
        library = value["source_identity"].get("library")
        if not isinstance(library, dict) or library.get("wire_version") != current_wire or library.get("wire_version") != value["manager"]["wire_version"]:
            raise RuntimeError(f"{label} source identity does not bind manager wire")
    gpu_identity = validate_gpu_snapshots(value["gpu_snapshots"], allow_partial=failed, label=f"{label}.gpu_snapshots")
    validate_claims(value["claims"], f"{label}.claims")
    return {"mode": mode, "case": workload["case"], "failed": failed, "gpu_identity": gpu_identity, "workload": workload, "geometry": geometry, "schema": NATIVE_RECORD_SCHEMA, "profile": profile}


__all__ = [
    "FULL_SLIDING_TOPOLOGY", "NATIVE_PROFILES", "NATIVE_RECORD_SCHEMA",
    "SLIDING_TOPOLOGY", "SWA_COUNTER_KEYS", "binding_digest",
    "profile_artifact_digest", "validate_record",
]
