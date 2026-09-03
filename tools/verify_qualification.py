#!/usr/bin/env python3
"""Verify current SGLang qualification records, pairs, and summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
INTEGRATION_SOURCE = ROOT / "compat/sglang/bridge/src"
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(INTEGRATION_SOURCE))

from orbitkv_sglang.qualification_primitives import (  # noqa: E402
    canonical_json_sha256,
    canonical_relative_path,
    require_exact_keys,
    require_sha256,
    sha256_file,
)
from qualification_source import (  # noqa: E402
    validate_source_binding,
    validate_source_identity,
)
from qualification_gates import (  # noqa: E402
    all_gates_qualified,
    pair_gates,
    summary_gates,
)
import qualification_native  # noqa: E402


RECORD_SCHEMA = "orbitkv.sglang-exact-chunked-single-run.v1"
PAIR_SCHEMA = "orbitkv.sglang-exact-chunked-pair-verification.v1"
SUMMARY_SCHEMA = "orbitkv.sglang-exact-chunked-summary.v1"
NATIVE_RECORD_SCHEMA = qualification_native.NATIVE_RECORD_SCHEMA
NATIVE_PAIR_SCHEMA = (
    "orbitkv.sglang-native-session-pair-verification.v1"
)
NATIVE_SUMMARY_SCHEMA = "orbitkv.sglang-native-session-summary.v1"
EXACT_TOPOLOGY = "whole_domain_chunked_token_kv"
FULL_SLIDING_TOPOLOGY = qualification_native.FULL_SLIDING_TOPOLOGY
SLIDING_TOPOLOGY = qualification_native.SLIDING_TOPOLOGY
NATIVE_PROFILES = qualification_native.NATIVE_PROFILES
CURRENT_WIRE_VERSION = 14
CURRENT_TARGET_CONTRACT_VERSION = 4
CURRENT_TARGET_FINGERPRINT = (
    "sha256:ac915458195e757e477cf04866dae76147cd71a7472661e0d791e9c4474173ba"
)
MAX_INTEGER = (1 << 63) - 1

GATE_KEYS = (
    "correctness_qualified",
    "stream_event_qualified",
    "capacity_qualified",
    "throughput_go",
)
RECORD_KEYS = frozenset({
        "schema", "mode", "started_at_utc", "command", "command_sha256",
        "environment", "environment_sha256", "source_identity",
        "source_identity_sha256", "runtime_identity", "model", "checkpoint",
        "checkpoint_identity_sha256", "runtime_manifest", "runtime_binding",
        "engine_args", "sampling_params", "workload", "timings", "outputs",
        "server_capacity", "manager", "gpu_snapshots", "claims",
})
RUNTIME_IDENTITY_KEYS = frozenset({
        "python_executable", "python_version", "platform", "sglang_version",
        "sglang_package", "kv_layout", "attention_backend", "dtype",
        "kv_cache_dtype", "execution", "tp_size", "pp_size", "dp_size",
        "dcp_size", "deterministic_inference", "sampling_backend",
        "run_id", "runtime_proof",
})
RUNTIME_PROOF_KEYS = frozenset({
    "actual_attention_backend", "effective_scheduler",
})
ACTUAL_BACKEND_KEYS = frozenset({
    "backend_class", "backend_module", "prefill_backend",
    "decode_backend", "has_local_attention", "attention_chunk_size",
    "page_size", "compiled_layer_ids", "use_irope_layer_ids",
})
EFFECTIVE_SCHEDULER_KEYS = frozenset({
    "max_prefill_tokens", "max_running_requests",
    "effective_max_running_requests_per_dp",
})
CHECKPOINT_KEYS = frozenset({"identity", "config"})
CHECKPOINT_CONFIG_KEYS = frozenset({
        "architectures", "num_hidden_layers", "vocab_size",
        "max_position_embeddings", "attention_chunk_size",
        "control_token_ids",
})
MANIFEST_KEYS = frozenset({
        "artifact", "schema", "version", "manifest_fingerprint",
        "retention_program_fingerprint", "layout_plan_fingerprint",
        "execution_signature_fingerprint", "chunk_geometry",
})
ARTIFACT_KEYS = frozenset({"path", "bytes", "sha256"})
GEOMETRY_KEYS = frozenset({"page_tokens", "chunk_tokens", "blocks_per_epoch"})
WORKLOAD_GEOMETRY_KEYS = GEOMETRY_KEYS | frozenset(
    {"chunk_epoch_count_per_request", "epoch_end_crossings_per_request"}
)
WORKLOAD_KEYS = frozenset({
        "case", "requests", "prompt_tokens", "decode_tokens",
        "materialized_kv_tokens_per_request", "iterations", "seed",
        "fresh_prompts", "input_token_digest_sha256",
        "input_token_digests_by_iteration_sha256", "chunk_geometry",
})
SAMPLING_KEYS = frozenset({
        "temperature", "max_new_tokens", "min_new_tokens", "ignore_eos",
        "sampling_seed",
})
TIMING_KEYS = frozenset({"load_seconds", "total_seconds", "iteration_seconds"})
OUTPUT_KEYS = frozenset({"iterations", "aggregate_sha256"})
OUTPUT_ITERATION_KEYS = frozenset({"iteration", "requests"})
OUTPUT_REQUEST_KEYS = frozenset({
        "request_id", "input_sha256", "output_ids", "output_sha256",
        "cached_tokens",
})
CAPACITY_KEYS = frozenset({"status", "requested_tokens", "available_tokens", "failure"})
FAILURE_KEYS = frozenset({"type", "message"})
MANAGER_KEYS = frozenset({"wire_version", "snapshots"})
SNAPSHOT_KEYS = frozenset({
        "stage", "identities", "manager_stats", "arena_stats",
        "batch_counters", "pressure", "runtime_proof",
        "lifecycle_route", "cache_policy",
})
IDENTITY_KEYS = frozenset({
        "engine_epoch", "pool_epoch", "pool_id", "class_id",
        "backend_domain", "page_count", "page_tokens", "backend_base_index",
        "first_page_id",
})
ARENA_KEYS = frozenset({
        "engine_epoch", "pool_epoch", "pool_id", "page_count", "class_id",
        "backend_domain", "first_page_id", "free_pages", "reserved_pages",
        "writing_pages", "active_pages", "retiring_pages",
        "quarantined_pages", "exhausted_pages", "request_page_refs",
        "prefix_page_refs", "reader_pins",
})
MANAGER_STATS_KEYS = frozenset({
        "active_requests", "active_snapshots", "active_prefixes",
        "evicted_prefixes", "prepared_steps", "submitted_steps", "free_pages",
        "reserved_pages", "writing_pages", "active_pages", "retiring_pages",
        "quarantined_pages", "exhausted_pages", "pending_reclamations",
        "total_request_page_refs", "total_prefix_page_refs",
        "total_reader_pins",
})
NATIVE_COUNTER_KEYS = frozenset({
        "request_acquire_batch_calls", "request_fork_batch_calls",
        "token_views_batch_calls", "mark_token_dispositions_batch_calls",
        "prepare_relocation_batch_calls", "submit_relocation_batch_calls",
        "complete_relocation_batch_calls", "abort_relocations_batch_calls",
        "prepare_batch_calls", "submit_batch_calls", "complete_batch_calls",
        "abort_steps_batch_calls", "quarantine_steps_batch_calls",
        "quarantine_submissions_batch_calls", "release_batch_calls",
        "acknowledge_reclamations_batch_calls", "recycle_requests_batch_calls",
        "prefix_lookup_batch_calls", "prefix_attach_batch_calls",
        "prefix_publish_batch_calls", "prefix_publish_release_batch_calls",
        "prefix_evict_batch_calls", "prefix_recycle_batch_calls",
        "buffer_too_small_preflights", "retryable_conflicts", "fail_stops",
        "hot_workspace_allocations", "capacity_memset_bytes",
        "root_entries_crossed", "cold_workspace_allocations",
        "materialized_page_objects",
})
RUNTIME_COUNTER_KEYS = frozenset({
        "forward_events", "completion_values", "event_queries", "event_waits",
        "quarantine_count", "fail_stop_count",
})
BRIDGE_COUNTER_KEYS = frozenset({
        "prefix_matches", "prefix_hits", "prefix_publishes", "prefix_evictions",
        "prefix_evicted_full_tokens", "prefix_evicted_swa_tokens",
        "cow_copy_intents", "cow_move_calls", "cow_copied_tokens",
        "mirror_validation_calls", "mirror_syncs",
        "prefix_global_alias_scans", "token_disposition_batches",
        "token_policy_evictions", "relocation_batches", "relocation_moves",
        "relocation_reclaimed_pages", "relocation_copy_events",
        "relocation_copy_tokens", "fixed_state_prepares", "fixed_state_clears",
        "fixed_state_copies", "fixed_state_events", "fixed_state_retirements",
        "fixed_state_acks",
})
COUNTER_KEYS = NATIVE_COUNTER_KEYS | RUNTIME_COUNTER_KEYS | BRIDGE_COUNTER_KEYS
GPU_SNAPSHOT_KEYS = frozenset({"stage", "time_ns", "gpus"})
GPU_KEYS = frozenset({
        "index", "name", "uuid", "memory.used", "memory.free",
        "utilization.gpu", "temperature.gpu", "power.draw",
})
CLAIM_KEYS = frozenset(GATE_KEYS)
GATE_RESULT_KEYS = frozenset({"qualified", "reasons"})

REFERENCE_KEYS = frozenset({"path", "sha256"})
PAIR_KEYS = frozenset({
        "schema", "case", "contract_sha256", "suite_contract_sha256",
        "records", "thresholds", "throughput_statistics", "gates",
        "overall_qualified",
})
PAIR_RECORD_KEYS = frozenset({"stock", "manager"})
THRESHOLD_KEYS = frozenset({
        "minimum_roomy_epochs", "minimum_samples_per_epoch",
        "maximum_median_regression_fraction",
        "maximum_paired_upper_regression_fraction", "confidence_level",
        "bootstrap_resamples", "bootstrap_seed",
})
THROUGHPUT_STAT_KEYS = frozenset({
        "roomy_epoch_count", "paired_sample_count",
        "paired_median_regression_fraction",
        "paired_bootstrap_upper_regression_fraction",
})
SUMMARY_KEYS = frozenset({
        "schema", "pair_count", "pairs", "suite_contract_sha256",
        "thresholds", "throughput_statistics", "gates",
        "overall_qualified",
})
NATIVE_PAIR_KEYS = PAIR_KEYS | frozenset({"profile"})
NATIVE_SUMMARY_KEYS = SUMMARY_KEYS | frozenset({"profile"})

DEFAULT_THRESHOLDS: dict[str, int | float] = {
    "minimum_roomy_epochs": 3,
    "minimum_samples_per_epoch": 3,
    "maximum_median_regression_fraction": 0.03,
    "maximum_paired_upper_regression_fraction": 0.03,
    "confidence_level": 0.95,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 20260828,
}


def _exact(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    try:
        require_exact_keys(value, expected, label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    assert isinstance(value, dict)
    return value


_json_tree = qualification_native.validate_json_tree


_strict_json = qualification_native.strict_json


def _sha(value: Any, label: str) -> str:
    try:
        return require_sha256(value, label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise RuntimeError(f"{label} is not a canonical SHA-256 fingerprint")
    _sha(value[7:], label)
    return value


def _canonical_digest(value: Any) -> str:
    try:
        return canonical_json_sha256(value)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _binding_digest(value: Mapping[str, Any]) -> str:
    return qualification_native.binding_digest(value)


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if (
        type(value) is not int
        or value < (1 if positive else 0)
        or value > MAX_INTEGER
    ):
        kind = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{label} must be a {kind} integer")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise RuntimeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise RuntimeError(f"{label} must be a finite {qualifier}number")
    if result == 0.0 and math.copysign(1.0, result) < 0:
        raise RuntimeError(f"{label} must not be negative zero")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"{label} must be a nonempty string")
    return value


def _absolute_path(value: Any, label: str) -> str:
    text = _string(value, label)
    if "\\" in text or any(mark in text for mark in ("\r", "\n")):
        raise RuntimeError(f"{label} is not a canonical absolute POSIX path")
    path = Path(text)
    if not path.is_absolute() or str(path) != text or ".." in path.parts:
        raise RuntimeError(f"{label} is not a canonical absolute POSIX path")
    return text


def _validate_claims(value: Any, label: str) -> None:
    claims = _exact(value, CLAIM_KEYS, label)
    for name in GATE_KEYS:
        gate = _exact(claims[name], GATE_RESULT_KEYS, f"{label}.{name}")
        reasons = gate["reasons"]
        if gate["qualified"] is not False:
            raise RuntimeError(
                f"{label}.{name} must remain false in a single-run record"
            )
        if not isinstance(reasons, list) or not reasons or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise RuntimeError(
                f"{label}.{name}.reasons must be nonempty strings"
            )


def _validate_geometry(
    value: Any, label: str, *, workload: bool
) -> dict[str, int]:
    expected = WORKLOAD_GEOMETRY_KEYS if workload else GEOMETRY_KEYS
    geometry = _exact(value, expected, label)
    result = {
        name: _integer(geometry[name], f"{label}.{name}", positive=True)
        for name in GEOMETRY_KEYS
    }
    if (
        result["chunk_tokens"]
        != result["page_tokens"] * result["blocks_per_epoch"]
    ):
        raise RuntimeError(f"{label} chunk geometry is inconsistent")
    if workload:
        result.update(
            chunk_epoch_count_per_request=_integer(
                geometry["chunk_epoch_count_per_request"],
                f"{label}.chunk_epoch_count_per_request",
                positive=True,
            ),
            epoch_end_crossings_per_request=_integer(
                geometry["epoch_end_crossings_per_request"],
                f"{label}.epoch_end_crossings_per_request",
                positive=True,
            ),
        )
    return result


def _validate_checkpoint(value: Any, label: str) -> Mapping[str, Any]:
    checkpoint = _exact(value, CHECKPOINT_KEYS, label)
    _json_tree(checkpoint["identity"], f"{label}.identity")
    config = _exact(
        checkpoint["config"], CHECKPOINT_CONFIG_KEYS, f"{label}.config"
    )
    architectures = config["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise RuntimeError(
            f"{label}.config.architectures must contain one value"
        )
    _string(architectures[0], f"{label}.config.architectures[0]")
    for name in (
        "num_hidden_layers",
        "vocab_size",
        "max_position_embeddings",
        "attention_chunk_size",
    ):
        _integer(config[name], f"{label}.config.{name}", positive=True)
    controls = config["control_token_ids"]
    if not isinstance(controls, dict) or any(
        not isinstance(name, str) or type(item) is not int or item < 0
        for name, item in controls.items()
    ):
        raise RuntimeError(f"{label}.config.control_token_ids is invalid")
    return checkpoint


def _validate_runtime_identity(
    value: Any, mode: str, label: str, *, native: bool = False
) -> Mapping[str, Any]:
    identity = _exact(value, RUNTIME_IDENTITY_KEYS, label)
    for name in (
        "python_executable",
        "python_version",
        "platform",
        "sglang_version",
        "sglang_package",
        "sampling_backend",
    ):
        _string(identity[name], f"{label}.{name}")
    try:
        run_id = UUID(_string(identity["run_id"], f"{label}.run_id"))
    except ValueError as error:
        raise RuntimeError(f"{label}.run_id is not a canonical UUIDv4") from error
    if run_id.version != 4 or str(run_id) != identity["run_id"]:
        raise RuntimeError(f"{label}.run_id is not a canonical UUIDv4")
    if mode == "stock":
        if identity["runtime_proof"] is not None:
            raise RuntimeError(f"{label}.runtime_proof must be null for stock")
    elif native and identity["runtime_proof"] is not None and not isinstance(
        identity["runtime_proof"], dict
    ):
        raise RuntimeError(f"{label}.runtime_proof is malformed")
    elif not native and not isinstance(identity["runtime_proof"], dict):
        raise RuntimeError(f"{label}.runtime_proof is required for manager")
    expected = {
        "kv_layout": "nhd",
        "attention_backend": "fa3",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "execution": "eager",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "deterministic_inference": True,
    }
    for name, expected_value in expected.items():
        if (
            identity[name] != expected_value
            or type(identity[name]) is not type(expected_value)
        ):
            raise RuntimeError(
                f"{label}.{name} differs from the exact execution contract"
            )
    return identity


def _layer_ids(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(item) is not int or item < 0 for item in value)
        or value != sorted(set(value))
    ):
        raise RuntimeError(f"{label} must be strictly increasing unique integers")
    return value


def _validate_runtime_proof(
    identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    workload: Mapping[str, Any],
    engine: Mapping[str, Any],
    binding: Mapping[str, Any],
    label: str,
) -> None:
    proof = _exact(identity["runtime_proof"], RUNTIME_PROOF_KEYS, label)
    backend = _exact(
        proof["actual_attention_backend"],
        ACTUAL_BACKEND_KEYS,
        f"{label}.actual_attention_backend",
    )
    scheduler = _exact(
        proof["effective_scheduler"],
        EFFECTIVE_SCHEDULER_KEYS,
        f"{label}.effective_scheduler",
    )
    geometry = workload["chunk_geometry"]
    compiled = _layer_ids(
        backend["compiled_layer_ids"], f"{label}.compiled_layer_ids"
    )
    irope = _layer_ids(
        backend["use_irope_layer_ids"], f"{label}.use_irope_layer_ids"
    )
    manifest_layers = binding["execution_signature"]["token_classes"][0][
        "layers"
    ]
    expected_layers = list(range(checkpoint["config"]["num_hidden_layers"]))
    expected_backend = {
        "backend_class": "FlashAttentionBackend",
        "backend_module": "sglang.srt.layers.attention.flashattention_backend",
        "prefill_backend": "fa3",
        "decode_backend": "fa3",
        "has_local_attention": True,
        "attention_chunk_size": geometry["chunk_tokens"],
        "page_size": geometry["page_tokens"],
    }
    if any(backend[name] != value for name, value in expected_backend.items()):
        raise RuntimeError(f"{label} actual attention backend differs")
    if compiled != expected_layers or compiled != manifest_layers or irope != compiled:
        raise RuntimeError(f"{label} compiled local-attention layers differ")
    expected_scheduler = {
        "max_prefill_tokens": geometry["chunk_tokens"],
        "max_running_requests": 1,
        "effective_max_running_requests_per_dp": 1,
    }
    if scheduler != expected_scheduler or any(
        engine.get(name) != expected_scheduler[name]
        for name in ("max_prefill_tokens", "max_running_requests")
    ):
        raise RuntimeError(f"{label} effective scheduler differs")


def _validate_manifest(
    value: Any, label: str
) -> tuple[Mapping[str, Any], dict[str, int]]:
    manifest = _exact(value, MANIFEST_KEYS, label)
    artifact = _exact(
        manifest["artifact"], ARTIFACT_KEYS, f"{label}.artifact"
    )
    _absolute_path(artifact["path"], f"{label}.artifact.path")
    _integer(artifact["bytes"], f"{label}.artifact.bytes", positive=True)
    _sha(artifact["sha256"], f"{label}.artifact.sha256")
    if (
        manifest["schema"] != "orbitkv.runtime-manifest"
        or manifest["version"] != 3
    ):
        raise RuntimeError(f"{label} is not the current RuntimeManifest")
    for name in (
        "manifest_fingerprint",
        "retention_program_fingerprint",
        "layout_plan_fingerprint",
        "execution_signature_fingerprint",
    ):
        _fingerprint(manifest[name], f"{label}.{name}")
    return manifest, _validate_geometry(
        manifest["chunk_geometry"],
        f"{label}.chunk_geometry",
        workload=False,
    )


def _validate_runtime_binding(
    value: Any,
    manifest: Mapping[str, Any],
    geometry: Mapping[str, int],
    label: str,
) -> None:
    binding = _exact(
        value,
        {
            "schema",
            "version",
            "fingerprint",
            "manifest_fingerprint",
            "target",
            "admission_profile",
            "target_contract_fingerprint",
            "required_wire_version",
            "execution_topology",
            "execution_signature",
        },
        label,
    )
    if (
        binding["schema"] != "orbitkv.runtime-binding"
        or binding["version"] != 1
    ):
        raise RuntimeError(f"{label} has an unsupported schema or version")
    if (
        _fingerprint(binding["fingerprint"], f"{label}.fingerprint")
        != _binding_digest(binding)
    ):
        raise RuntimeError(f"{label}.fingerprint does not match its payload")
    if binding["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise RuntimeError(f"{label} does not bind the RuntimeManifest")
    target = _exact(
        binding["target"], {"id", "contract_version"}, f"{label}.target"
    )
    profile = _exact(
        binding["admission_profile"],
        {"id", "version"},
        f"{label}.admission_profile",
    )
    if target != {"id": "sglang", "contract_version": CURRENT_TARGET_CONTRACT_VERSION}:
        raise RuntimeError(f"{label} does not target SGLang")
    if profile != {
        "id": "eager-single-device-bf16-nhd",
        "version": 1,
    }:
        raise RuntimeError(f"{label} does not bind the exact execution profile")
    target_fingerprint = _fingerprint(
        binding["target_contract_fingerprint"],
        f"{label}.target_contract_fingerprint",
    )
    if target_fingerprint != CURRENT_TARGET_FINGERPRINT:
        raise RuntimeError(f"{label} does not bind the current runtime target")
    required_wire_version = _integer(
        binding["required_wire_version"],
        f"{label}.required_wire_version",
        positive=True,
    )
    if required_wire_version != CURRENT_WIRE_VERSION:
        raise RuntimeError(
            f"{label} does not require current wire version "
            f"{CURRENT_WIRE_VERSION}"
        )
    if binding["execution_topology"] != EXACT_TOPOLOGY:
        raise RuntimeError(f"{label} does not bind the exact chunked topology")
    signature = _exact(
        binding["execution_signature"],
        {
            "schema",
            "version",
            "fingerprint",
            "manifest_schema",
            "manifest_version",
            "manifest_fingerprint",
            "page_tokens",
            "token_classes",
            "token_states",
            "fixed_states",
        },
        f"{label}.execution_signature",
    )
    if (
        signature["schema"] != "orbitkv.execution-signature"
        or signature["version"] != 1
        or signature["manifest_schema"] != "orbitkv.runtime-manifest"
        or signature["manifest_version"] != 3
        or signature["manifest_fingerprint"]
        != manifest["manifest_fingerprint"]
        or signature["page_tokens"] != geometry["page_tokens"]
        or signature["fixed_states"] != []
    ):
        raise RuntimeError(f"{label}.execution_signature is inconsistent")
    if (
        _fingerprint(
            signature["fingerprint"],
            f"{label}.execution_signature.fingerprint",
        )
        != _binding_digest(signature)
    ):
        raise RuntimeError(
            f"{label}.execution_signature fingerprint is invalid"
        )
    if (
        signature["fingerprint"]
        != manifest["execution_signature_fingerprint"]
    ):
        raise RuntimeError(
            f"{label} execution signature differs from manifest record"
        )
    classes = signature["token_classes"]
    states = signature["token_states"]
    if (
        not isinstance(classes, list)
        or len(classes) != 1
        or not isinstance(states, list)
        or len(states) != 1
    ):
        raise RuntimeError(f"{label} must contain one token class and state")
    token_class = _exact(
        classes[0],
        {
            "name",
            "layers",
            "bytes_per_token_per_layer",
            "address",
            "retirement",
            "minimum_slots_per_request",
        },
        f"{label}.execution_signature.token_classes[0]",
    )
    address = _exact(
        token_class["address"],
        {"kind", "blocks_per_epoch"},
        f"{label}.address",
    )
    retirement = _exact(
        token_class["retirement"],
        {"kind", "blocks_per_epoch"},
        f"{label}.retirement",
    )
    blocks = geometry["blocks_per_epoch"]
    if (
        address
        != {"kind": "resettable_arena", "blocks_per_epoch": blocks}
        or retirement != {"kind": "epoch_end", "blocks_per_epoch": blocks}
        or token_class["minimum_slots_per_request"] != blocks
    ):
        raise RuntimeError(
            f"{label} chunked address/retirement geometry differs"
        )
    state = _exact(
        states[0],
        {"name", "layers", "backend"},
        f"{label}.token_states[0]",
    )
    backend = _exact(
        state["backend"],
        {
            "kind",
            "storage",
            "components",
            "bytes_per_token_per_layer",
            "page_bytes_per_layer",
            "retention",
            "window_tokens",
            "token_relocatable",
        },
        f"{label}.token_states[0].backend",
    )
    if (
        state["name"] != token_class["name"]
        or state["layers"] != token_class["layers"]
        or not isinstance(token_class["layers"], list)
        or token_class["layers"] != list(range(len(token_class["layers"])))
        or type(token_class["bytes_per_token_per_layer"]) is not int
        or token_class["bytes_per_token_per_layer"] <= 0
        or backend["kind"] != "token_slots"
        or backend["storage"] != "token_kv"
        or backend["components"] != []
        or backend["bytes_per_token_per_layer"]
        != token_class["bytes_per_token_per_layer"]
        or backend["page_bytes_per_layer"]
        != token_class["bytes_per_token_per_layer"] * geometry["page_tokens"]
        or backend["retention"] != "chunked"
        or backend["window_tokens"] is not None
        or backend["token_relocatable"] is not True
    ):
        raise RuntimeError(f"{label} token-state contract differs")


def _validate_workload(
    value: Any, label: str
) -> tuple[Mapping[str, Any], dict[str, int]]:
    workload = _exact(value, WORKLOAD_KEYS, label)
    if workload["case"] not in {"roomy", "exact-floor"}:
        raise RuntimeError(f"{label}.case must be roomy or exact-floor")
    for name in (
        "requests",
        "prompt_tokens",
        "decode_tokens",
        "materialized_kv_tokens_per_request",
        "iterations",
    ):
        _integer(workload[name], f"{label}.{name}", positive=True)
    _integer(workload["seed"], f"{label}.seed")
    if workload["requests"] != 1 or workload["fresh_prompts"] is not True:
        raise RuntimeError(
            f"{label} must use one request and fresh deterministic prompts"
        )
    _sha(
        workload["input_token_digest_sha256"],
        f"{label}.input_token_digest_sha256",
    )
    input_digests = workload["input_token_digests_by_iteration_sha256"]
    if (
        not isinstance(input_digests, list)
        or len(input_digests) != workload["iterations"]
    ):
        raise RuntimeError(f"{label} input digest iteration shape is invalid")
    for row in input_digests:
        if not isinstance(row, list) or len(row) != workload["requests"]:
            raise RuntimeError(f"{label} input digest request shape is invalid")
        for digest in row:
            _sha(digest, f"{label} input digest")
    geometry = _validate_geometry(
        workload["chunk_geometry"],
        f"{label}.chunk_geometry",
        workload=True,
    )
    materialized = workload["prompt_tokens"] + workload["decode_tokens"] - 1
    chunk = geometry["chunk_tokens"]
    epochs = (materialized + chunk - 1) // chunk
    crossings = (
        materialized // chunk
        - (workload["prompt_tokens"] - 1) // chunk
    )
    if (
        workload["materialized_kv_tokens_per_request"] != materialized
        or geometry["chunk_epoch_count_per_request"] != epochs
        or geometry["epoch_end_crossings_per_request"] != crossings
        or crossings < 1
    ):
        raise RuntimeError(f"{label} derived chunk geometry differs")
    return workload, geometry


def _validate_outputs(
    value: Any,
    workload: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    allow_empty: bool,
    label: str,
) -> None:
    outputs = _exact(value, OUTPUT_KEYS, label)
    iterations = outputs["iterations"]
    if not isinstance(iterations, list):
        raise RuntimeError(f"{label}.iterations must be a list")
    expected_count = workload["iterations"]
    allowed_counts = {0} if allow_empty else {expected_count}
    if len(iterations) not in allowed_counts:
        raise RuntimeError(f"{label}.iterations cardinality is invalid")
    request_ids: set[str] = set()
    vocab = checkpoint["config"]["vocab_size"]
    input_rows = workload["input_token_digests_by_iteration_sha256"]
    for index, raw_iteration in enumerate(iterations):
        iteration = _exact(
            raw_iteration,
            OUTPUT_ITERATION_KEYS,
            f"{label}.iterations[{index}]",
        )
        if iteration["iteration"] != index:
            raise RuntimeError(f"{label} iteration indexes are not contiguous")
        requests = iteration["requests"]
        if (
            not isinstance(requests, list)
            or len(requests) != workload["requests"]
        ):
            raise RuntimeError(f"{label} request cardinality is invalid")
        for request_index, raw_request in enumerate(requests):
            request = _exact(
                raw_request, OUTPUT_REQUEST_KEYS, f"{label}.request"
            )
            request_id = _string(
                request["request_id"], f"{label}.request_id"
            )
            if request_id in request_ids:
                raise RuntimeError(f"{label} contains a duplicate request_id")
            request_ids.add(request_id)
            _sha(request["input_sha256"], f"{label}.input_sha256")
            if request["input_sha256"] != input_rows[index][request_index]:
                raise RuntimeError(
                    f"{label} request input digest differs from workload"
                )
            ids = request["output_ids"]
            if (
                not isinstance(ids, list)
                or len(ids) != workload["decode_tokens"]
                or any(
                    type(token) is not int or not 0 <= token < vocab
                    for token in ids
                )
            ):
                raise RuntimeError(f"{label} output token vector is invalid")
            if request["output_sha256"] != _canonical_digest(ids):
                raise RuntimeError(f"{label} output digest differs from tokens")
            if request["cached_tokens"] != 0:
                raise RuntimeError(
                    f"{label} fresh request contains cached tokens"
                )
    if outputs["aggregate_sha256"] != _canonical_digest(iterations):
        raise RuntimeError(
            f"{label} aggregate digest differs from outputs"
        )


def _validate_capacity(value: Any, label: str) -> Mapping[str, Any]:
    capacity = _exact(value, CAPACITY_KEYS, label)
    _integer(
        capacity["requested_tokens"],
        f"{label}.requested_tokens",
        positive=True,
    )
    if capacity["status"] == "observed":
        if capacity["failure"] is not None:
            raise RuntimeError(f"{label} observed capacity carries a failure")
        available = _integer(
            capacity["available_tokens"],
            f"{label}.available_tokens",
            positive=True,
        )
        if available != capacity["requested_tokens"]:
            raise RuntimeError(f"{label} observed capacity differs from request")
    elif capacity["status"] == "failed":
        if capacity["available_tokens"] is not None:
            _integer(
                capacity["available_tokens"],
                f"{label}.available_tokens",
                positive=True,
            )
        failure = _exact(
            capacity["failure"], FAILURE_KEYS, f"{label}.failure"
        )
        _string(failure["type"], f"{label}.failure.type")
        _string(failure["message"], f"{label}.failure.message")
    else:
        raise RuntimeError(f"{label}.status must be observed or failed")
    return capacity


def _nonnegative_exact(
    value: Any, keys: Iterable[str], label: str
) -> Mapping[str, int]:
    result = _exact(value, keys, label)
    for name in keys:
        _integer(result[name], f"{label}.{name}")
    return result  # type: ignore[return-value]


def _validate_pressure(value: Any, label: str) -> None:
    disabled = {"schema", "enabled", "mode", "sample_count"}
    enabled = disabled | {
        "scope",
        "last_event",
        "event_counts",
        "active_requests",
        "max_active_requests",
        "global",
        "classes",
    }
    expected = disabled if isinstance(value, dict) and value.get("enabled") is False else enabled
    pressure = _exact(value, expected, label)
    if (
        pressure["schema"] != "orbitkv.runtime-pressure.v1"
        or pressure["mode"] != "event_driven_high_water"
    ):
        raise RuntimeError(f"{label} pressure schema or mode differs")
    if pressure["enabled"] is False:
        if pressure["sample_count"] != 0:
            raise RuntimeError(f"{label} disabled pressure has samples")
    elif pressure["enabled"] is True:
        samples = _integer(
            pressure["sample_count"], f"{label}.sample_count", positive=True
        )
        scope = pressure["scope"]
        counts = pressure["event_counts"]
        global_values = pressure["global"]
        classes = pressure["classes"]
        if (
            not isinstance(scope, dict)
            or not isinstance(counts, dict)
            or not counts
            or any(
                not isinstance(name, str)
                or not name
                or type(count) is not int
                or count <= 0
                for name, count in counts.items()
            )
            or sum(counts.values()) != samples
            or pressure["last_event"] not in counts
            or type(pressure["active_requests"]) is not int
            or pressure["active_requests"] < 0
            or type(pressure["max_active_requests"]) is not int
            or pressure["max_active_requests"] < pressure["active_requests"]
            or not isinstance(global_values, dict)
            or not isinstance(classes, list)
            or len(classes) != 1
            or not isinstance(classes[0], dict)
        ):
            raise RuntimeError(f"{label} enabled pressure census is invalid")
        required_metrics = {
            "resident_data_bytes",
            "semantic_live_bytes",
            "retention_amplification_milli",
            "high_water_resident_data_bytes",
            "high_water_semantic_live_bytes",
            "high_water_retention_amplification_milli",
        }
        for name, metrics in (
            ("global", global_values),
            ("classes[0]", classes[0]),
        ):
            if not required_metrics <= set(metrics):
                raise RuntimeError(
                    f"{label}.{name} omits pressure qualification metrics"
                )
            for key in required_metrics:
                item = metrics[key]
                if item is not None:
                    _integer(item, f"{label}.{name}.{key}")
        for name in ("scope", "event_counts", "global", "classes"):
            _json_tree(pressure[name], f"{label}.{name}")
    else:
        raise RuntimeError(f"{label}.enabled must be boolean")


def _validate_manager(
    value: Any, workload: Mapping[str, Any], runtime_proof: Any, label: str
) -> None:
    manager = _exact(value, MANAGER_KEYS, label)
    if (
        type(manager["wire_version"]) is not int
        or manager["wire_version"] <= 0
    ):
        raise RuntimeError(f"{label} has an invalid wire version")
    if manager["wire_version"] != CURRENT_WIRE_VERSION:
        raise RuntimeError(
            f"{label} does not use current wire version {CURRENT_WIRE_VERSION}"
        )
    snapshots = manager["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != 3:
        raise RuntimeError(f"{label}.snapshots must contain three stages")
    stages = ("after_load", "after_workload", "final")
    previous_counters: Mapping[str, int] | None = None
    identity_value: Any = None
    policy_value: tuple[str, str] | None = None
    for index, (raw, stage) in enumerate(zip(snapshots, stages, strict=True)):
        snapshot = _exact(
            raw, SNAPSHOT_KEYS, f"{label}.snapshots[{index}]"
        )
        if snapshot["stage"] != stage:
            raise RuntimeError(f"{label} snapshot stages differ")
        policy = (snapshot["lifecycle_route"], snapshot["cache_policy"])
        if policy != ("native_session", "request_private"):
            raise RuntimeError(
                f"{label} does not prove the exact Chunked cache policy"
            )
        if policy_value is None:
            policy_value = policy
        elif policy != policy_value:
            raise RuntimeError(f"{label} cache policy changed during the run")
        identities = snapshot["identities"]
        arenas = snapshot["arena_stats"]
        if (
            not isinstance(identities, list)
            or len(identities) != 1
            or not isinstance(arenas, list)
            or len(arenas) != 1
        ):
            raise RuntimeError(
                f"{label} requires exactly one arena identity"
            )
        identity = _nonnegative_exact(
            identities[0], IDENTITY_KEYS, f"{label}.identity"
        )
        arena = _nonnegative_exact(
            arenas[0], ARENA_KEYS, f"{label}.arena"
        )
        stats = _nonnegative_exact(
            snapshot["manager_stats"],
            MANAGER_STATS_KEYS,
            f"{label}.manager_stats",
        )
        prefix_state = {
            name: stats[name]
            for name in (
                "active_prefixes",
                "evicted_prefixes",
                "total_prefix_page_refs",
            )
            if stats[name] != 0
        }
        if prefix_state:
            raise RuntimeError(
                f"{label} request-private Prefix state is nonzero: {prefix_state}"
            )
        counters = _nonnegative_exact(
            snapshot["batch_counters"],
            COUNTER_KEYS,
            f"{label}.batch_counters",
        )
        _validate_pressure(snapshot["pressure"], f"{label}.pressure")
        if snapshot["runtime_proof"] != runtime_proof:
            raise RuntimeError(f"{label} runtime proof changed between snapshots")
        if identity["class_id"] != 0 or arena["class_id"] != 0:
            raise RuntimeError(f"{label} class identity differs")
        for name in (
            "engine_epoch",
            "pool_epoch",
            "pool_id",
            "class_id",
            "backend_domain",
            "page_count",
            "first_page_id",
        ):
            if arena[name] != identity[name]:
                raise RuntimeError(f"{label} arena identity echo differs")
        if (
            identity["page_tokens"]
            != workload["chunk_geometry"]["page_tokens"]
        ):
            raise RuntimeError(f"{label} arena page geometry differs")
        phases = (
            "free_pages",
            "reserved_pages",
            "writing_pages",
            "active_pages",
            "retiring_pages",
            "quarantined_pages",
            "exhausted_pages",
        )
        if sum(arena[name] for name in phases) != arena["page_count"]:
            raise RuntimeError(f"{label} arena phases do not sum to capacity")
        if stats["free_pages"] != arena["free_pages"]:
            raise RuntimeError(
                f"{label} manager and arena free-page counts differ"
            )
        if identity_value is None:
            identity_value = identities
        elif identities != identity_value:
            raise RuntimeError(
                f"{label} arena identity changed during the run"
            )
        if previous_counters is not None and any(
            counters[name] < previous_counters[name] for name in COUNTER_KEYS
        ):
            raise RuntimeError(
                f"{label} counters decreased between snapshots"
            )
        previous_counters = counters

    assert previous_counters is not None
    initial_counters = snapshots[0]["batch_counters"]
    workload_counters = snapshots[1]["batch_counters"]
    workload_batches = workload["iterations"] * workload["decode_tokens"]
    expected_crossings = (
        workload["chunk_geometry"]["epoch_end_crossings_per_request"]
        * workload["requests"]
        * workload["iterations"]
    )
    if (
        workload_counters["prepare_batch_calls"]
        - initial_counters["prepare_batch_calls"]
        != workload_batches
        or workload_counters["submit_batch_calls"]
        - initial_counters["submit_batch_calls"]
        != workload_batches
        or workload_counters["complete_batch_calls"]
        - initial_counters["complete_batch_calls"]
        != workload_batches
        or workload_counters["forward_events"]
        - initial_counters["forward_events"]
        != workload_batches
        or workload_counters["completion_values"]
        - initial_counters["completion_values"]
        != workload_batches
        or workload_counters["acknowledge_reclamations_batch_calls"]
        - initial_counters["acknowledge_reclamations_batch_calls"]
        < expected_crossings
    ):
        raise RuntimeError(
            f"{label} lifecycle counters differ from the exact workload"
        )


def _validate_legacy_record(
    record: Any, label: str = "record"
) -> dict[str, Any]:
    """Validate one legacy exact-Chunked single-run object."""

    _json_tree(record, label)
    value = _exact(record, RECORD_KEYS, label)
    if (
        value["schema"] != RECORD_SCHEMA
        or value["mode"] not in {"stock", "manager"}
    ):
        raise RuntimeError(f"{label} schema or mode is unsupported")
    try:
        started = datetime.fromisoformat(
            _string(value["started_at_utc"], f"{label}.started_at_utc")
        )
    except ValueError as error:
        raise RuntimeError(
            f"{label}.started_at_utc is not ISO-8601"
        ) from error
    if started.tzinfo is None:
        raise RuntimeError(f"{label}.started_at_utc lacks a timezone")
    command = value["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise RuntimeError(f"{label}.command is invalid")
    digest_fields = (
        ("command", "command_sha256"),
        ("environment", "environment_sha256"),
        ("source_identity", "source_identity_sha256"),
        ("checkpoint", "checkpoint_identity_sha256"),
    )
    for payload_name, digest_name in digest_fields:
        if value[digest_name] != _canonical_digest(value[payload_name]):
            raise RuntimeError(
                f"{label}.{digest_name} differs from its payload"
            )
        _sha(value[digest_name], f"{label}.{digest_name}")
    if (
        not isinstance(value["environment"], dict)
        or not isinstance(value["source_identity"], dict)
        or not isinstance(value["engine_args"], dict)
    ):
        raise RuntimeError(f"{label} provenance objects are malformed")
    runtime_identity = _validate_runtime_identity(
        value["runtime_identity"], value["mode"], f"{label}.runtime_identity"
    )
    validate_source_binding(
        value["source_identity"],
        value["mode"],
        command=command,
        environment=value["environment"],
        sglang_package=runtime_identity["sglang_package"],
    )
    _absolute_path(value["model"], f"{label}.model")
    checkpoint = _validate_checkpoint(
        value["checkpoint"], f"{label}.checkpoint"
    )
    sampling = _exact(
        value["sampling_params"],
        SAMPLING_KEYS,
        f"{label}.sampling_params",
    )
    if (
        _number(
            sampling["temperature"],
            f"{label}.sampling_params.temperature",
        )
        != 0
        or sampling["ignore_eos"] is not True
    ):
        raise RuntimeError(
            f"{label} sampling is not deterministic exact decode"
        )
    workload, geometry = _validate_workload(
        value["workload"], f"{label}.workload"
    )
    for name in ("max_new_tokens", "min_new_tokens"):
        if sampling[name] != workload["decode_tokens"]:
            raise RuntimeError(
                f"{label}.sampling_params.{name} differs from workload"
            )
    if sampling["sampling_seed"] != workload["seed"]:
        raise RuntimeError(f"{label} sampling seed differs from workload")
    if (
        checkpoint["config"]["attention_chunk_size"]
        != geometry["chunk_tokens"]
    ):
        raise RuntimeError(
            f"{label} checkpoint chunk size differs from workload"
        )
    engine = value["engine_args"]
    expected_engine = {
        "attention_backend": "fa3",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "page_size": geometry["page_tokens"],
        "chunked_prefill_size": geometry["chunk_tokens"],
        "prefill_max_requests": 1,
        "enable_dynamic_chunking": False,
        "enable_mixed_chunk": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
    }
    if any(
        engine.get(name) != expected
        for name, expected in expected_engine.items()
    ):
        raise RuntimeError(
            f"{label}.engine_args differs from the exact execution contract"
        )
    capacity = _validate_capacity(
        value["server_capacity"], f"{label}.server_capacity"
    )
    if engine.get("max_total_tokens") != capacity["requested_tokens"]:
        raise RuntimeError(
            f"{label} engine capacity differs from its observation"
        )
    if (
        workload["case"] == "roomy"
        and capacity["requested_tokens"]
        < workload["materialized_kv_tokens_per_request"]
    ):
        raise RuntimeError(f"{label} roomy case does not have roomy capacity")
    if (
        workload["case"] == "exact-floor"
        and capacity["requested_tokens"]
        != workload["requests"] * geometry["chunk_tokens"]
    ):
        raise RuntimeError(
            f"{label} exact-floor capacity is not the compiled floor"
        )
    failed = capacity["status"] == "failed"
    if failed and (
        value["mode"] != "stock" or workload["case"] != "exact-floor"
    ):
        raise RuntimeError(
            f"{label} contains an inadmissible execution failure"
        )
    timings = _exact(
        value["timings"], TIMING_KEYS, f"{label}.timings"
    )
    load_seconds = _number(
        timings["load_seconds"], f"{label}.timings.load_seconds"
    )
    if load_seconds < 0:
        raise RuntimeError(f"{label}.timings.load_seconds must be nonnegative")
    _number(
        timings["total_seconds"],
        f"{label}.timings.total_seconds",
        positive=True,
    )
    iteration_seconds = timings["iteration_seconds"]
    allowed_timing_counts = {0} if failed else {workload["iterations"]}
    if (
        not isinstance(iteration_seconds, list)
        or len(iteration_seconds) not in allowed_timing_counts
    ):
        raise RuntimeError(
            f"{label}.timings.iteration_seconds shape is invalid"
        )
    for item in iteration_seconds:
        _number(
            item, f"{label}.timings.iteration_seconds", positive=True
        )
    if failed and iteration_seconds:
        raise RuntimeError(f"{label} failed execution contains timing samples")
    _validate_outputs(
        value["outputs"],
        workload,
        checkpoint,
        allow_empty=failed,
        label=f"{label}.outputs",
    )
    if failed and value["outputs"]["iterations"]:
        raise RuntimeError(f"{label} failed execution contains outputs")
    if value["mode"] == "stock":
        if (
            value["runtime_manifest"] is not None
            or value["runtime_binding"] is not None
            or value["manager"] is not None
        ):
            raise RuntimeError(f"{label} stock record contains manager state")
        if (
            "SGLANG_PLUGINS" in value["environment"]
            or any(
                name.startswith("ORBITKV_")
                for name in value["environment"]
            )
        ):
            raise RuntimeError(f"{label} stock environment selected OrbitKV")
    else:
        manifest, manifest_geometry = _validate_manifest(
            value["runtime_manifest"], f"{label}.runtime_manifest"
        )
        if manifest_geometry != {
            name: geometry[name] for name in GEOMETRY_KEYS
        }:
            raise RuntimeError(
                f"{label} manifest and workload geometry differ"
            )
        _validate_runtime_binding(
            value["runtime_binding"],
            manifest,
            geometry,
            f"{label}.runtime_binding",
        )
        _validate_runtime_proof(
            runtime_identity,
            checkpoint,
            workload,
            engine,
            value["runtime_binding"],
            f"{label}.runtime_identity.runtime_proof",
        )
        if "SGLANG_PLUGINS" in value["environment"]:
            raise RuntimeError(f"{label} manager environment selected a plugin")
        _validate_manager(
            value["manager"],
            workload,
            runtime_identity["runtime_proof"],
            f"{label}.manager",
        )
        library = value["source_identity"].get("library")
        if (
            not isinstance(library, dict)
            or library.get("wire_version") != CURRENT_WIRE_VERSION
            or library.get("wire_version") != value["manager"]["wire_version"]
            or value["runtime_binding"].get("required_wire_version")
            != value["manager"]["wire_version"]
        ):
            raise RuntimeError(
                f"{label} source identity does not bind the manager wire version"
            )
    gpu_identity = qualification_native.validate_gpu_snapshots(
        value["gpu_snapshots"],
        allow_partial=failed,
        label=f"{label}.gpu_snapshots",
        gpu_keys=GPU_KEYS,
    )
    _validate_claims(value["claims"], f"{label}.claims")
    return {
        "mode": value["mode"],
        "case": workload["case"],
        "failed": failed,
        "gpu_identity": gpu_identity,
        "workload": workload,
        "geometry": geometry,
        "schema": RECORD_SCHEMA,
        "profile": EXACT_TOPOLOGY,
    }


def validate_record(record: Any, label: str = "record") -> dict[str, Any]:
    """Validate one supported single-run object and derive its family."""

    _json_tree(record, label)
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} must be an object")
    schema = record.get("schema")
    if schema == RECORD_SCHEMA:
        return _validate_legacy_record(record, label)
    if schema == NATIVE_RECORD_SCHEMA:
        return qualification_native.validate_record(
            record,
            label,
            exact=_exact,
            json_tree=_json_tree,
            sha=_sha,
            canonical_digest=_canonical_digest,
            binding_digest=_binding_digest,
            absolute_path=_absolute_path,
            validate_source_binding=validate_source_binding,
            validate_outputs=_validate_outputs,
            validate_gpu_snapshots=lambda value, **kwargs: qualification_native.validate_gpu_snapshots(
                value, gpu_keys=GPU_KEYS, **kwargs
            ),
            validate_claims=_validate_claims,
            runtime_identity_keys=RUNTIME_IDENTITY_KEYS,
            sampling_keys=SAMPLING_KEYS,
            timing_keys=TIMING_KEYS,
            current_wire=CURRENT_WIRE_VERSION,
            target_version=CURRENT_TARGET_CONTRACT_VERSION,
            target_fingerprint=CURRENT_TARGET_FINGERPRINT,
            fingerprint=_fingerprint,
        )
    raise RuntimeError(f"{label} schema is unsupported")


def _normalized_environment(record: Mapping[str, Any]) -> dict[str, Any]:
    environment = dict(record["environment"])
    for name in tuple(environment):
        if (
            name.startswith("ORBITKV_")
            or name == "PYTHONPATH"
        ):
            environment.pop(name)
    return environment


def _normalized_engine(record: Mapping[str, Any]) -> dict[str, Any]:
    engine = dict(record["engine_args"])
    engine.pop("radix_cache_backend", None)
    return engine


def _normalized_command(record: Mapping[str, Any]) -> list[str]:
    command = list(record["command"])
    implementation_options = {
        "--mode",
        "--runtime-manifest",
        "--manifest",
        "--library",
        "--case",
        "--max-total-tokens",
        "--pressure-telemetry",
        "--sglang-root",
    }
    result: list[str] = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        option = item.split("=", 1)[0]
        if option in implementation_options:
            skip = "=" not in item
            continue
        result.append(item)
    return result


def _suite_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(record["runtime_identity"])
    runtime.pop("run_id", None)
    runtime["sglang_package"] = "python/sglang/__init__.py"
    runtime.pop("runtime_proof", None)
    workload = dict(record["workload"])
    workload.pop("case", None)
    engine = _normalized_engine(record)
    # Capacity is deliberately case-specific and excluded from suite identity.
    engine.pop("max_total_tokens", None)
    result = {
        "command": _normalized_command(record),
        "environment": _normalized_environment(record),
        "source_identity": validate_source_identity(
            record["source_identity"],
            record["mode"],
            native=record["schema"] == NATIVE_RECORD_SCHEMA,
        ),
        "runtime_identity": runtime,
        "model": record["model"],
        "checkpoint": record["checkpoint"],
        "engine_args": engine,
        "sampling_params": record["sampling_params"],
        "workload": workload,
        "gpu_identity": [list(item) for item in validate_record(record)["gpu_identity"]],
    }
    if record["schema"] == NATIVE_RECORD_SCHEMA:
        result["record_schema"] = NATIVE_RECORD_SCHEMA
        result["profile"] = record["profile"]
    return result


def _pair_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    contract = _suite_contract(record)
    contract["case"] = record["workload"]["case"]
    contract["capacity_tokens"] = record["server_capacity"]["requested_tokens"]
    return contract


def _validate_pair_contract(
    stock: Mapping[str, Any], manager: Mapping[str, Any]
) -> tuple[str, str]:
    if stock["mode"] != "stock" or manager["mode"] != "manager":
        raise RuntimeError("pair requires one stock and one manager record")
    if stock["schema"] != manager["schema"] or stock.get(
        "profile", EXACT_TOPOLOGY
    ) != manager.get("profile", EXACT_TOPOLOGY):
        raise RuntimeError("stock/manager qualification schema or profile differs")
    if stock["schema"] == NATIVE_RECORD_SCHEMA and (
        qualification_native.profile_artifact_digest(stock["source_identity"])
        != qualification_native.profile_artifact_digest(
            manager["source_identity"]
        )
    ):
        raise RuntimeError(
            "stock/manager source contract differs: profile artifact digest"
        )
    stock_contract = _pair_contract(stock)
    manager_contract = _pair_contract(manager)
    if stock_contract != manager_contract:
        raise RuntimeError(
            "stock/manager checkpoint, engine, GPU, workload, sampling, "
            "capacity, FA3/BF16/NHD, or source contract differs"
        )
    return _canonical_digest(stock_contract), _canonical_digest(
        _suite_contract(stock)
    )


def _validate_thresholds(value: Any) -> dict[str, int | float]:
    thresholds = _exact(value, THRESHOLD_KEYS, "thresholds")
    normalized: dict[str, int | float] = {
        "minimum_roomy_epochs": _integer(
            thresholds["minimum_roomy_epochs"],
            "thresholds.minimum_roomy_epochs",
            positive=True,
        ),
        "minimum_samples_per_epoch": _integer(
            thresholds["minimum_samples_per_epoch"],
            "thresholds.minimum_samples_per_epoch",
            positive=True,
        ),
        "bootstrap_resamples": _integer(
            thresholds["bootstrap_resamples"],
            "thresholds.bootstrap_resamples",
            positive=True,
        ),
        "bootstrap_seed": _integer(
            thresholds["bootstrap_seed"], "thresholds.bootstrap_seed"
        ),
    }
    for name in (
        "maximum_median_regression_fraction",
        "maximum_paired_upper_regression_fraction",
    ):
        number = _number(thresholds[name], f"thresholds.{name}")
        if not 0 <= number < 1:
            raise RuntimeError(f"thresholds.{name} must be in [0, 1)")
        normalized[name] = number
    confidence = _number(
        thresholds["confidence_level"], "thresholds.confidence_level"
    )
    if not 0 < confidence < 1:
        raise RuntimeError("thresholds.confidence_level must be in (0, 1)")
    normalized["confidence_level"] = confidence
    if normalized != DEFAULT_THRESHOLDS:
        raise RuntimeError("thresholds differ from frozen verifier policy")
    return normalized


def _reference_path(root: Path, value: Any, label: str) -> Path:
    reference = _exact(value, REFERENCE_KEYS, label)
    try:
        relative = canonical_relative_path(reference["path"])
    except ValueError as error:
        raise RuntimeError(f"{label}.path is unsafe: {error}") from error
    digest = _sha(reference["sha256"], f"{label}.sha256")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}.path is not a regular non-symlink file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label}.path escapes its evidence root") from error
    if sha256_file(resolved) != digest:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    return resolved


def _resolve_root(path: Path) -> Path:
    requested = Path(os.path.abspath(path.expanduser()))
    if requested.is_symlink():
        raise RuntimeError("evidence root must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve evidence root: {error}") from error
    if not root.is_dir():
        raise RuntimeError("evidence root is not a directory")
    return root


def _expected_pair(
    root: Path, stored: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(stored, dict):
        raise RuntimeError("pair must be an object")
    pair_schema = stored.get("schema")
    if pair_schema == PAIR_SCHEMA:
        _exact(stored, PAIR_KEYS, "pair")
        native = False
    elif pair_schema == NATIVE_PAIR_SCHEMA:
        _exact(stored, NATIVE_PAIR_KEYS, "pair")
        native = True
    else:
        raise RuntimeError("pair schema is unsupported")
    thresholds = _validate_thresholds(stored["thresholds"])
    records = _exact(stored["records"], PAIR_RECORD_KEYS, "pair.records")
    stock_path = _reference_path(root, records["stock"], "pair.records.stock")
    manager_path = _reference_path(
        root, records["manager"], "pair.records.manager"
    )
    stock = _strict_json(stock_path)
    manager = _strict_json(manager_path)
    stock_meta = validate_record(stock, "stock record")
    manager_meta = validate_record(manager, "manager record")
    if stock_meta["case"] != manager_meta["case"]:
        raise RuntimeError("stock/manager workload cases differ")
    if stock_meta["schema"] != manager_meta["schema"]:
        raise RuntimeError("stock/manager qualification schemas differ")
    expected_record_schema = NATIVE_RECORD_SCHEMA if native else RECORD_SCHEMA
    if stock_meta["schema"] != expected_record_schema:
        raise RuntimeError("pair schema differs from its record family")
    if stock_meta["profile"] != manager_meta["profile"]:
        raise RuntimeError("stock/manager qualification profiles differ")
    if native and stored["profile"] != stock_meta["profile"]:
        raise RuntimeError("pair profile differs from its records")
    contract, suite_contract = _validate_pair_contract(stock, manager)
    if stored["case"] != stock_meta["case"]:
        raise RuntimeError("pair case differs from its records")
    expected = {
        "schema": pair_schema,
        "case": stock_meta["case"],
        "contract_sha256": contract,
        "suite_contract_sha256": suite_contract,
        "records": stored["records"],
        "thresholds": thresholds,
        "throughput_statistics": {
            "roomy_epoch_count": 0,
            "paired_sample_count": 0,
            "paired_median_regression_fraction": None,
            "paired_bootstrap_upper_regression_fraction": None,
        },
        "gates": pair_gates(stock, manager),
        "overall_qualified": False,
    }
    if native:
        expected["profile"] = stock_meta["profile"]
    return expected, stock, manager


def verify_pair(path: Path) -> dict[str, Any]:
    """Verify and rederive one stored pair document."""

    pair_path = Path(os.path.abspath(path.expanduser()))
    if pair_path.is_symlink() or not pair_path.is_file():
        raise RuntimeError("pair path must be a regular non-symlink file")
    root = pair_path.resolve(strict=True).parent
    stored = _strict_json(pair_path)
    expected, _, _ = _expected_pair(root, stored)
    if stored != expected:
        raise RuntimeError("stored pair differs from independently derived pair")
    return expected


def build_pair(
    root: Path,
    stock_path: str,
    manager_path: str,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pair object from two record references without writing it."""

    root = _resolve_root(root)
    threshold_values = _validate_thresholds(
        dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    )
    references = {}
    for mode, raw in (("stock", stock_path), ("manager", manager_path)):
        try:
            relative = canonical_relative_path(raw).as_posix()
        except ValueError as error:
            raise RuntimeError(f"{mode} record path is unsafe: {error}") from error
        record_path = root / relative
        if record_path.is_symlink() or not record_path.is_file():
            raise RuntimeError(f"{mode} record path is not a regular file")
        references[mode] = {
            "path": relative,
            "sha256": sha256_file(record_path),
        }
    record_schemas = []
    record_profiles = []
    record_cases = []
    for mode in ("stock", "manager"):
        record = _strict_json(root / references[mode]["path"])
        metadata = validate_record(record, f"{mode} record")
        record_schemas.append(metadata["schema"])
        record_profiles.append(metadata["profile"])
        record_cases.append(metadata["case"])
    if len(set(record_schemas)) != 1 or len(set(record_profiles)) != 1:
        raise RuntimeError("stock/manager qualification schema or profile differs")
    native = record_schemas[0] == NATIVE_RECORD_SCHEMA
    pair_schema = NATIVE_PAIR_SCHEMA if native else PAIR_SCHEMA
    seed = {
        "schema": pair_schema,
        "case": record_cases[0],
        "contract_sha256": "",
        "suite_contract_sha256": "",
        "records": references,
        "thresholds": threshold_values,
        "throughput_statistics": {
            "roomy_epoch_count": 0,
            "paired_sample_count": 0,
            "paired_median_regression_fraction": None,
            "paired_bootstrap_upper_regression_fraction": None,
        },
        "gates": {},
        "overall_qualified": False,
    }
    if native:
        seed["profile"] = record_profiles[0]
    expected, _, _ = _expected_pair(root, seed)
    return expected


def build_summary(
    root: Path,
    pair_paths: Sequence[str],
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a summary from verified pair references without writing it."""

    root = _resolve_root(root)
    if not pair_paths:
        raise RuntimeError("summary requires at least one pair")
    threshold_values = _validate_thresholds(
        dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    )
    references = []
    pairs = []
    records = []
    seen = set()
    stock_hashes: set[str] = set()
    manager_hashes: set[str] = set()
    record_pairs: set[tuple[str, str]] = set()
    run_ids: set[str] = set()
    run_pairs: set[tuple[str, str]] = set()
    summary_pair_schema: str | None = None
    summary_profile: str | None = None
    for index, raw in enumerate(pair_paths):
        try:
            relative = canonical_relative_path(raw).as_posix()
        except ValueError as error:
            raise RuntimeError(f"pair path is unsafe: {error}") from error
        if relative in seen:
            raise RuntimeError("summary contains a duplicate pair path")
        seen.add(relative)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("summary pair path is not a regular file")
        stored = _strict_json(path)
        expected, stock, manager = _expected_pair(root, stored)
        if stored != expected:
            raise RuntimeError(
                f"stored pair differs from trusted derivation: {relative}"
            )
        if expected["thresholds"] != threshold_values:
            raise RuntimeError("pair and summary throughput thresholds differ")
        if summary_pair_schema is None:
            summary_pair_schema = expected["schema"]
            summary_profile = expected.get("profile")
        elif (
            expected["schema"] != summary_pair_schema
            or expected.get("profile") != summary_profile
        ):
            raise RuntimeError(
                "summary pairs mix qualification schemas or profiles"
            )
        stock_hash = expected["records"]["stock"]["sha256"]
        manager_hash = expected["records"]["manager"]["sha256"]
        identity = (stock_hash, manager_hash)
        stock_run_id = stock["runtime_identity"]["run_id"]
        manager_run_id = manager["runtime_identity"]["run_id"]
        run_identity = (stock_run_id, manager_run_id)
        if (
            stock_hash in stock_hashes
            or manager_hash in manager_hashes
            or identity in record_pairs
            or stock_run_id == manager_run_id
            or stock_run_id in run_ids
            or manager_run_id in run_ids
            or run_identity in run_pairs
        ):
            raise RuntimeError(
                "summary pairs do not contain distinct raw/run/pair epochs"
            )
        stock_hashes.add(stock_hash)
        manager_hashes.add(manager_hash)
        record_pairs.add(identity)
        run_ids.update((stock_run_id, manager_run_id))
        run_pairs.add(run_identity)
        references.append({"path": relative, "sha256": sha256_file(path)})
        pairs.append(expected)
        records.append((stock, manager))
    suite_contracts = {pair["suite_contract_sha256"] for pair in pairs}
    if len(suite_contracts) != 1:
        raise RuntimeError("summary pairs do not share one suite contract")
    statistics_value, gates = summary_gates(
        pairs, records, threshold_values
    )
    native = summary_pair_schema == NATIVE_PAIR_SCHEMA
    result = {
        "schema": NATIVE_SUMMARY_SCHEMA if native else SUMMARY_SCHEMA,
        "pair_count": len(pairs),
        "pairs": references,
        "suite_contract_sha256": next(iter(suite_contracts)),
        "thresholds": threshold_values,
        "throughput_statistics": statistics_value,
        "gates": gates,
        "overall_qualified": all_gates_qualified(gates),
    }
    if native:
        result["profile"] = summary_profile
    return result


def verify_summary(path: Path) -> dict[str, Any]:
    """Verify and rederive one stored multi-epoch summary."""

    summary_path = Path(os.path.abspath(path.expanduser()))
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError("summary path must be a regular non-symlink file")
    root = summary_path.resolve(strict=True).parent
    stored = _strict_json(summary_path)
    if not isinstance(stored, dict):
        raise RuntimeError("summary must be an object")
    if stored.get("schema") == SUMMARY_SCHEMA:
        _exact(stored, SUMMARY_KEYS, "summary")
    elif stored.get("schema") == NATIVE_SUMMARY_SCHEMA:
        _exact(stored, NATIVE_SUMMARY_KEYS, "summary")
        if stored["profile"] not in NATIVE_PROFILES:
            raise RuntimeError("summary profile is unsupported")
    else:
        raise RuntimeError("summary schema is unsupported")
    pair_references = stored["pairs"]
    if not isinstance(pair_references, list):
        raise RuntimeError("summary.pairs must be a list")
    paths = []
    for index, reference in enumerate(pair_references):
        _reference_path(root, reference, f"summary.pairs[{index}]")
        paths.append(reference["path"])
    expected = build_summary(
        root, paths, thresholds=stored["thresholds"]
    )
    if stored != expected:
        raise RuntimeError(
            "stored summary differs from independently derived summary"
        )
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    record = subparsers.add_parser("verify-record")
    record.add_argument("record", type=Path)
    pair = subparsers.add_parser("verify-pair")
    pair.add_argument("pair", type=Path)
    pair.add_argument("--require-qualified", action="store_true")
    make_pair = subparsers.add_parser("build-pair")
    make_pair.add_argument("root", type=Path)
    make_pair.add_argument("stock_record")
    make_pair.add_argument("manager_record")
    summary = subparsers.add_parser("verify-summary")
    summary.add_argument("summary", type=Path)
    summary.add_argument("--require-qualified", action="store_true")
    make_summary = subparsers.add_parser("build-summary")
    make_summary.add_argument("root", type=Path)
    make_summary.add_argument("pairs", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "verify-record":
            record = _strict_json(args.record)
            metadata = validate_record(record)
            result = {
                "schema": metadata["schema"],
                "status": "passed",
                "mode": metadata["mode"],
                "case": metadata["case"],
                "gates": record["claims"],
                "overall_qualified": False,
            }
            if metadata["schema"] == NATIVE_RECORD_SCHEMA:
                result["profile"] = metadata["profile"]
        elif args.action == "verify-pair":
            result = verify_pair(args.pair)
        elif args.action == "build-pair":
            result = build_pair(
                args.root, args.stock_record, args.manager_record
            )
        elif args.action == "verify-summary":
            result = verify_summary(args.summary)
        else:
            result = build_summary(args.root, args.pairs)
        if getattr(args, "require_qualified", False) and not result.get(
            "overall_qualified", False
        ):
            raise RuntimeError("all four independently derived gates are required")
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
