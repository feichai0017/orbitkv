#!/usr/bin/env python3
"""Verify an unsealed H20 naive-vs-relocate diagnostic record matrix.

The input is deliberately narrow: four epochs, B1 and B4, and one JSON
record for each of the naive and relocate modes.  The verifier trusts no
derived pair or summary file and emits a summary recomputed from raw records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = ROOT / "integrations/sglang"
SOURCE_ROOT = INTEGRATION_ROOT / "src"
ADAPTER_ROOT = INTEGRATION_ROOT / "src/orbitkv_sglang"
BENCHMARK_PATH = INTEGRATION_ROOT / "bench_token_relocation.py"
sys.dont_write_bytecode = True
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang import qualification_primitives  # noqa: E402


RECORD_SCHEMA = "orbitkv.sglang-v0517-token-relocation-single-run.v1"
SUMMARY_SCHEMA = (
    "orbitkv.sglang-v0517-token-relocation-diagnostic-summary.v1"
)
MODEL_SLUG = "qwen2.5-0.5b"
EPOCHS = (1, 2, 3, 4)
BATCHES = (1, 4)
MODES = ("naive", "relocate")
GPU_NAME = "NVIDIA H20"
SGLANG_RELEASE = "v0.5.17"
SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
SOURCE_CONTRACT = "pinned_head_plus_canonical_loader_patch"
LOADER_PATH = "python/sglang/srt/plugins/__init__.py"
LOADER_HEAD_GIT_BLOB = "00ae1acd18266765c006d87ba5eec51e9f113d8d"
LOADER_WORKTREE_GIT_BLOB = "7c20ccb51e46942f0bbdfdbcaf88c3148939cb55"
LOADER_WORKTREE_SHA256 = (
    "1fc2e2472e8fd55f564826509b2afa1f8f0d86a4b2ee3a3986c3209e3c09c934"
)
LOADER_PATCH_SHA256 = (
    "6de7acab246b299386b5d6557154a7aa237c8bdb498b8b885bed1c72e849745d"
)
MANAGER_ENTRYPOINT = {
    "name": "orbitkv_manager",
    "value": "orbitkv_sglang.plugin:register",
    "distribution": "orbitkv-sglang",
    "distribution_version": "0.2.0",
}
MODEL_PATH_BASENAME = "qwen2.5-0.5b-instruct"
EXPECTED_CAPACITY_TOKENS = {1: 128, 4: 512}
CHECKPOINT_CONFIG_SHA256 = (
    "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
)
CHECKPOINT_WEIGHT_SHA256 = (
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
)
CHECKPOINT_WEIGHT_BYTES = 988_097_824
DIAGNOSTIC_HARNESS_SHA256 = (
    "7163d7085c78715b77392da9fd5c90a2bc8d3bf14ef0b1b26fec369ce5b6ea5d"
)
DIAGNOSTIC_ADAPTER_IDENTITY_SHA256 = (
    "db5ac3fab68b1e6d61c6962befde570147f855277b92ca37b54962ac365307b8"
)

TRIGGER_TOKENS = 48
DECODE_TOKENS = 41
VICTIM_COUNT = 24
RETAINED_COUNT_AT_TRIGGER = 24
EXPECTED_RECLAMATION_ROUNDS = 2
EXPECTED_ITERATIONS = 5
EXPECTED_VICTIM_POLICY = {
    "trigger_tokens": TRIGGER_TOKENS,
    "retained_per_page": 8,
    "policy_id": 260813263,
    "policy_version": 1,
    "quality_contract": 1,
    "fragmentation_threshold_milli": 500,
    "maximum_source_pages": 3,
    "evacuation_headroom_pages": 2,
}

RECORD_KEYS = frozenset(
    {
        "schema",
        "mode",
        "started_at_utc",
        "command",
        "environment",
        "source_identity",
        "runtime_identity",
        "checkpoint",
        "checkpoint_contract",
        "engine_args",
        "sampling_params",
        "workload",
        "pairing",
        "load_seconds",
        "iteration_seconds",
        "iteration_total_seconds",
        "total_seconds",
        "output_token_digest_sha256",
        "request_output_ids",
        "manager",
        "gpu_snapshots",
    }
)
PAIRING_KEYS = frozenset(
    {"pair_key_sha256", "contract", "only_allowed_difference"}
)
PAIR_CONTRACT_KEYS = frozenset(
    {
        "checkpoint_identity_sha256",
        "engine_args",
        "sampling_params",
        "workload",
        "capacity_tokens",
        "victim_policy",
    }
)
ONLY_ALLOWED_DIFFERENCE = "naive versus byte-exact relocation"

MANAGER_DRAIN_FIELDS = (
    "active_requests",
    "active_snapshots",
    "active_prefixes",
    "evicted_prefixes",
    "prepared_steps",
    "submitted_steps",
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
MANAGER_STATS_FIELDS = MANAGER_DRAIN_FIELDS + ("free_pages",)
IDENTITY_FIELDS = (
    "engine_epoch", "pool_epoch", "pool_id", "class_id",
    "backend_domain", "page_count", "page_tokens",
    "backend_base_index", "first_page_id",
)
ARENA_IDENTITY_FIELDS = (
    "engine_epoch", "pool_epoch", "pool_id", "page_count",
    "class_id", "backend_domain", "first_page_id",
)
ARENA_STATS_FIELDS = ARENA_IDENTITY_FIELDS + (
    "free_pages", "reserved_pages", "writing_pages", "active_pages",
    "retiring_pages", "quarantined_pages", "exhausted_pages",
    "request_page_refs", "prefix_page_refs", "reader_pins",
)
ARENA_DRAIN_FIELDS = (
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
FAILURE_COUNTER_FIELDS = (
    "abort_steps_batch_calls",
    "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls",
    "abort_relocations_batch_calls",
    "retryable_conflicts",
    "fail_stops",
    "quarantine_count",
    "fail_stop_count",
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
)
UNCHANGED_COUNTER_FIELDS = (
    "request_fork_batch_calls",
    "abort_steps_batch_calls",
    "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls",
    "prefix_publish_batch_calls",
    "prefix_evict_batch_calls",
    "prefix_recycle_batch_calls",
    "abort_relocations_batch_calls",
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "retryable_conflicts",
    "fail_stops",
    "quarantine_count",
    "fail_stop_count",
    "prefix_publishes",
    "prefix_evictions",
    "prefix_evicted_full_tokens",
    "prefix_evicted_swa_tokens",
    "prefix_global_alias_scans",
    "cow_copy_intents",
    "cow_move_calls",
    "cow_copied_tokens",
    "fixed_state_prepares",
    "fixed_state_clears",
    "fixed_state_copies",
    "fixed_state_events",
    "fixed_state_retirements",
    "fixed_state_acks",
)
RELOCATION_COUNTER_FIELDS = (
    "relocation_batches",
    "relocation_moves",
    "relocation_reclaimed_pages",
    "relocation_copy_events",
    "relocation_copy_tokens",
    "prepare_relocation_batch_calls",
    "submit_relocation_batch_calls",
    "complete_relocation_batch_calls",
)
BATCH_COUNTER_FIELDS = (
    "request_acquire_batch_calls", "request_fork_batch_calls",
    "prepare_batch_calls", "submit_batch_calls", "complete_batch_calls",
    "abort_steps_batch_calls", "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls", "release_batch_calls",
    "acknowledge_reclamations_batch_calls", "recycle_requests_batch_calls",
    "prefix_lookup_batch_calls", "prefix_attach_batch_calls",
    "prefix_publish_batch_calls", "prefix_publish_release_batch_calls",
    "prefix_evict_batch_calls", "prefix_recycle_batch_calls",
    "token_views_batch_calls", "mark_token_dispositions_batch_calls",
    "prepare_relocation_batch_calls", "submit_relocation_batch_calls",
    "complete_relocation_batch_calls", "abort_relocations_batch_calls",
    "buffer_too_small_preflights", "retryable_conflicts", "fail_stops",
    "hot_workspace_allocations", "capacity_memset_bytes",
    "root_entries_crossed", "cold_workspace_allocations",
    "materialized_page_objects", "forward_events", "completion_values",
    "event_queries", "event_waits", "quarantine_count",
    "fail_stop_count", "prefix_matches", "prefix_hits",
    "prefix_publishes", "prefix_evictions", "prefix_evicted_full_tokens",
    "prefix_evicted_swa_tokens", "prefix_global_alias_scans",
    "cow_copy_intents", "cow_move_calls", "cow_copied_tokens",
    "mirror_validation_calls", "mirror_syncs",
    "token_disposition_batches", "token_policy_evictions",
    "relocation_batches", "relocation_moves",
    "relocation_reclaimed_pages", "relocation_copy_events",
    "relocation_copy_tokens", "fixed_state_prepares",
    "fixed_state_clears", "fixed_state_copies", "fixed_state_events",
    "fixed_state_retirements", "fixed_state_acks",
)
CENSUS_KEYS = frozenset(
    {
        "abi_version", "plan_fingerprint", "state_plan_fingerprint",
        "fixed_state_byte_count", "fixed_state_descriptors",
        "tree_cache_type", "identities", "arena_stats",
        "manager_stats", "swa_activity", "batch_counters",
    }
)
PRESSURE_KEY = "pressure"
PRESSURE_SCHEMA = "orbitkv.runtime-pressure.v1"
PRESSURE_DISABLED = {
    "schema": PRESSURE_SCHEMA,
    "enabled": False,
    "mode": "event_driven_high_water",
    "sample_count": 0,
}
SHA256_LENGTH = 64


def canonical_digest(value: Any) -> str:
    try:
        return qualification_primitives.canonical_json_sha256(value)
    except ValueError as error:
        cause = error.__cause__
        if isinstance(cause, (TypeError, ValueError)):
            raise cause from None
        raise


def _sha256_file(path: Path) -> str:
    return qualification_primitives.sha256_file(path)


def _current_adapter_identity() -> dict[str, Any]:
    paths = [
        INTEGRATION_ROOT / "pyproject.toml",
        INTEGRATION_ROOT / "prepare_pinned_checkout.py",
        INTEGRATION_ROOT / "patches/v0.5.17-orbitkv-fail-closed.patch",
        *sorted(ADAPTER_ROOT.rglob("*.py")),
        *sorted(ADAPTER_ROOT.rglob("*.json")),
    ]
    return {
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in paths
        ]
    }


def _verify_adapter_source_identity(
    source_root: Path, expected: dict[str, Any]
) -> None:
    """Bind an adapter inventory to an explicit checkout/archive root."""

    requested = Path(os.path.abspath(source_root.expanduser()))
    if requested.is_symlink():
        raise RuntimeError("expected adapter source root must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"cannot resolve expected adapter source root {requested}: {error}"
        ) from error
    if (
        not root.is_dir()
        or root.name != "src"
        or root.parent.name != "sglang"
        or root.parent.parent.name != "integrations"
    ):
        raise RuntimeError(
            "expected adapter source root must end in integrations/sglang/src"
        )
    repository_root = root.parents[2]
    files = expected.get("files") if isinstance(expected, dict) else None
    if not isinstance(files, list) or not files:
        raise RuntimeError("expected adapter identity is invalid")
    indexed: dict[str, str] = {}
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
        ):
            raise RuntimeError("expected adapter identity is invalid")
        relative = _safe_relative(item["path"])
        digest = _require_sha256(
            item.get("sha256"), f"adapter source {relative} SHA-256"
        )
        name = relative.as_posix()
        if name in indexed:
            raise RuntimeError(f"duplicate adapter source path: {name}")
        indexed[name] = digest
    expected_paths = {
        "integrations/sglang/pyproject.toml",
        "integrations/sglang/prepare_pinned_checkout.py",
        "integrations/sglang/patches/v0.5.17-orbitkv-fail-closed.patch",
        *(
            path.relative_to(repository_root).as_posix()
            for path in sorted((root / "orbitkv_sglang").rglob("*.py"))
        ),
        *(
            path.relative_to(repository_root).as_posix()
            for path in sorted((root / "orbitkv_sglang").rglob("*.json"))
        ),
    }
    if set(indexed) != expected_paths:
        raise RuntimeError("adapter source closure is incomplete or excessive")
    for name, digest in indexed.items():
        path = repository_root / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(f"adapter source differs from identity: {name}")


def _fresh_input_ids(
    *, requests: int, prompt_tokens: int, vocab_size: int, seed: int,
    iteration: int, token_upper_bound: int,
) -> list[list[int]]:
    domain_size = token_upper_bound - 3
    if domain_size < requests:
        raise RuntimeError("checkpoint vocabulary is too small")
    result = []
    for request in range(requests):
        ordinal = iteration * requests + request
        material = hashlib.shake_256(
            f"orbitkv-fresh-v1:{seed}:{iteration}:{request}".encode("ascii")
        ).digest(prompt_tokens * 4)
        prompt = [
            3 + int.from_bytes(material[offset : offset + 4], "little")
            % domain_size
            for offset in range(0, len(material), 4)
        ]
        prompt[0] = 3 + (seed + ordinal) % domain_size
        result.append(prompt)
    return result


def _input_digest(inputs: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(
        json.dumps(inputs, separators=(",", ":")).encode()
    ).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = qualification_primitives.parse_strict_json_object(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot load strict JSON record {path}: {error}") from error
    except ValueError as error:
        if str(error) == "strict JSON value must be an object":
            raise RuntimeError(f"JSON record is not an object: {path}") from error
        detail = error.__cause__ or error
        message = str(detail)
        message = message.replace(
            "duplicate JSON object key ", "duplicate key ", 1
        ).replace("non-finite JSON number ", "non-finite number ", 1)
        raise RuntimeError(
            f"cannot load strict JSON record {path}: {message}"
        ) from error
    return value


def _safe_relative(value: str) -> PurePosixPath:
    try:
        return qualification_primitives.canonical_relative_path(value)
    except ValueError as error:
        raise RuntimeError(
            f"unsafe or non-canonical archive path: {value!r}"
        ) from error


def _require_exact_keys(value: Any, expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    try:
        qualification_primitives.require_exact_keys(value, expected_set, label)
    except ValueError as error:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(expected_set - actual)} "
            f"extra={sorted(actual - expected_set)}"
        ) from error


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeError(f"{label} must be a finite positive number")
    return float(value)


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RuntimeError(f"{label} must include a UTC offset")
    return parsed


def _same_path(left: Any, right: Any) -> bool:
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    )


def _artifact_identity(value: Any, label: str) -> dict[str, Any]:
    _require_exact_keys(value, {"path", "bytes", "sha256"}, label)
    if not isinstance(value["path"], str) or not value["path"]:
        raise RuntimeError(f"{label} path is invalid")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise RuntimeError(f"{label} byte count is invalid")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"{label} SHA-256 is invalid")
    return value


def _require_sha256(value: Any, label: str) -> str:
    try:
        return qualification_primitives.require_sha256(value, label)
    except ValueError as error:
        raise RuntimeError(f"{label} is not a canonical SHA-256") from error


def _validate_checkpoint(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} checkpoint identity is missing")
    required = {
        "config_sha256", "index_files", "indexed_weight_bytes",
        "indexed_weight_container_overhead_bytes", "indexed_weight_files",
        "indexed_weights_complete", "load_format", "missing_indexed_weights",
        "observed_indexed_weight_bytes", "weight_bytes", "weight_files",
    }
    _require_exact_keys(value, required, f"{label} checkpoint")
    if (
        _require_sha256(
            value["config_sha256"], f"{label} checkpoint config"
        )
        != CHECKPOINT_CONFIG_SHA256
    ):
        raise RuntimeError(f"{label} checkpoint config digest is not Qwen2.5-0.5B")
    if (
        value["index_files"] != []
        or value["indexed_weight_bytes"] is not None
        or value["indexed_weight_container_overhead_bytes"] is not None
        or value["indexed_weight_files"] != []
        or value["indexed_weights_complete"] is not True
        or value["load_format"] != "auto"
        or value["missing_indexed_weights"] != []
        or value["observed_indexed_weight_bytes"] != 0
    ):
        raise RuntimeError(f"{label} checkpoint container identity is invalid")
    weights = value["weight_files"]
    if not isinstance(weights, list) or len(weights) != 1:
        raise RuntimeError(f"{label} checkpoint weight inventory is invalid")
    weight = weights[0]
    _require_exact_keys(weight, {"name", "bytes", "sha256"}, f"{label} weight")
    if (
        weight["name"] != "model.safetensors"
        or type(weight["bytes"]) is not int
        or weight["bytes"] <= 0
        or weight["bytes"] != CHECKPOINT_WEIGHT_BYTES
        or value["weight_bytes"] != CHECKPOINT_WEIGHT_BYTES
    ):
        raise RuntimeError(f"{label} checkpoint weight identity is invalid")
    if (
        _require_sha256(weight["sha256"], f"{label} checkpoint weight")
        != CHECKPOINT_WEIGHT_SHA256
    ):
        raise RuntimeError(f"{label} checkpoint weight digest is not Qwen2.5-0.5B")


def _expected_paths() -> dict[tuple[int, int, str], str]:
    return {
        (epoch, batch, mode): (
            f"epoch-{epoch:03d}/{MODEL_SLUG}-b{batch}-{mode}.json"
        )
        for epoch in EPOCHS
        for batch in BATCHES
        for mode in MODES
    }


def _records_directory(root: Path) -> Path:
    requested = Path(os.path.abspath(root.expanduser()))
    if requested.is_symlink():
        raise RuntimeError("evidence root must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"cannot resolve evidence root {requested}: {error}"
        ) from error
    if not resolved.is_dir():
        raise RuntimeError("evidence root must be a directory")
    records = resolved / "records"
    if records.is_symlink() or not records.is_dir():
        raise RuntimeError("evidence root must contain a regular records directory")

    expected_epochs = {f"epoch-{epoch:03d}" for epoch in EPOCHS}
    actual_epochs = {entry.name for entry in records.iterdir()}
    if actual_epochs != expected_epochs:
        raise RuntimeError(
            "record epoch matrix differs: "
            f"missing={sorted(expected_epochs - actual_epochs)} "
            f"extra={sorted(actual_epochs - expected_epochs)}"
        )
    expected = _expected_paths()
    for epoch in EPOCHS:
        epoch_dir = records / f"epoch-{epoch:03d}"
        if epoch_dir.is_symlink() or not epoch_dir.is_dir():
            raise RuntimeError(f"record epoch is not a regular directory: {epoch_dir}")
        expected_names = {
            Path(relative).name
            for (record_epoch, _, _), relative in expected.items()
            if record_epoch == epoch
        }
        actual_names = {entry.name for entry in epoch_dir.iterdir()}
        if actual_names != expected_names:
            raise RuntimeError(
                f"epoch-{epoch:03d} record matrix differs: "
                f"missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        for name in expected_names:
            path = epoch_dir / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"record is not a regular file: {path}")
    return records


def _validate_workload(
    record: dict[str, Any], batch: int, label: str, *, expected_iterations: int
) -> tuple[int, int]:
    workload = record.get("workload")
    if not isinstance(workload, dict):
        raise RuntimeError(f"{label} workload is missing")
    _require_exact_keys(
        workload,
        {
            "requests", "iterations", "prompt_tokens", "decode_tokens",
            "victim_count_per_request", "retained_count_at_trigger",
            "expected_reclamation_rounds", "seed",
            "input_token_digests_by_iteration_sha256",
            "input_token_digest_sha256",
        },
        f"{label} workload",
    )
    iterations = workload.get("iterations")
    if type(iterations) is not int or iterations != expected_iterations:
        raise RuntimeError(
            f"{label} iterations must be exactly {expected_iterations}"
        )
    expected = {
        "requests": batch,
        "prompt_tokens": TRIGGER_TOKENS,
        "decode_tokens": DECODE_TOKENS,
        "victim_count_per_request": VICTIM_COUNT,
        "retained_count_at_trigger": RETAINED_COUNT_AT_TRIGGER,
        "expected_reclamation_rounds": EXPECTED_RECLAMATION_ROUNDS,
    }
    mismatches = {
        name: {"expected": expected_value, "actual": workload.get(name)}
        for name, expected_value in expected.items()
        if workload.get(name) != expected_value
        or type(workload.get(name)) is not type(expected_value)
    }
    if mismatches:
        raise RuntimeError(f"{label} workload differs: {mismatches}")
    checkpoint_contract = record.get("checkpoint_contract")
    classes = (
        checkpoint_contract.get("classes")
        if isinstance(checkpoint_contract, dict)
        else None
    )
    if not isinstance(classes, list) or not classes:
        raise RuntimeError(f"{label} checkpoint class inventory is invalid")
    expected_contract = {
        "architecture": "Qwen2ForCausalLM",
        "attention_backend": "flashinfer",
        "attention_profile": "full",
        "backend_profile": {"attention_backend": "flashinfer"},
        "classes": [
            {
                "layers": list(range(24)),
                "name": "full",
                "retention": "full",
                "window_tokens": None,
            }
        ],
        "control_token_ids": {},
        "fixed_states": [],
        "max_position_embeddings": 32768,
        "num_hidden_layers": 24,
        "prompt_token_upper_bound": 151936,
        "qualification_scope": "diagnostic_only",
        "sliding_window": None,
        "state_ownership": "request_private",
        "vocab_size": 151936,
        "workload_profile": "fresh_prompt",
    }
    if checkpoint_contract != expected_contract:
        raise RuntimeError(f"{label} is not the fixed Qwen2.5-0.5B contract")
    vocab_size = checkpoint_contract.get("vocab_size")
    sampling = record.get("sampling_params")
    seed = workload.get("seed")
    if type(vocab_size) is not int or vocab_size <= 3 or type(seed) is not int:
        raise RuntimeError(f"{label} deterministic input identity is invalid")
    if not isinstance(sampling, dict) or sampling.get("sampling_seed") != seed:
        raise RuntimeError(f"{label} workload and sampling seeds differ")
    prompts = [
        _fresh_input_ids(
            requests=batch,
            prompt_tokens=TRIGGER_TOKENS,
            vocab_size=vocab_size,
            seed=seed,
            iteration=iteration,
            token_upper_bound=checkpoint_contract["prompt_token_upper_bound"],
        )
        for iteration in range(iterations)
    ]
    expected_digests = [
        [canonical_digest(prompt) for prompt in row] for row in prompts
    ]
    if (
        workload.get("input_token_digests_by_iteration_sha256")
        != expected_digests
        or workload.get("input_token_digest_sha256")
        != _input_digest([prompt for row in prompts for prompt in row])
    ):
        raise RuntimeError(f"{label} input-token digest is invalid")
    return iterations, len(classes)


def _command_arguments(command: Any, label: str) -> dict[str, str]:
    if (
        not isinstance(command, list)
        or len(command) < 3
        or any(not isinstance(value, str) or not value for value in command)
        or Path(command[1]).name != "bench_token_relocation.py"
    ):
        raise RuntimeError(f"{label} command is not the token-relocation runner")
    values: dict[str, str] = {}
    index = 2
    while index < len(command):
        option = command[index]
        if not option.startswith("--") or index + 1 >= len(command):
            raise RuntimeError(f"{label} command arguments are malformed")
        if option in values:
            raise RuntimeError(f"{label} command repeats {option}")
        values[option] = command[index + 1]
        index += 2
    required = {
        "--mode", "--sglang-root", "--model", "--plan", "--library",
        "--requests", "--iterations", "--max-total-tokens",
        "--attention-backend", "--output",
    }
    allowed = required | {
        "--context-length", "--seed", "--mem-fraction-static", "--output"
    }
    if not required.issubset(values):
        raise RuntimeError(
            f"{label} command omits runner arguments: {sorted(required - set(values))}"
        )
    if set(values) - allowed:
        raise RuntimeError(
            f"{label} command has unknown arguments: {sorted(set(values) - allowed)}"
        )
    return values


def _validate_pairing(record: dict[str, Any], mode: str, label: str) -> None:
    pairing = record.get("pairing")
    _require_exact_keys(pairing, PAIRING_KEYS, f"{label} pairing")
    contract = pairing["contract"]
    _require_exact_keys(contract, PAIR_CONTRACT_KEYS, f"{label} pair contract")
    if pairing["pair_key_sha256"] != canonical_digest(contract):
        raise RuntimeError(f"{label} pair key does not bind its contract")
    if pairing["only_allowed_difference"] != ONLY_ALLOWED_DIFFERENCE:
        raise RuntimeError(f"{label} allowed-difference boundary is invalid")
    if contract["checkpoint_identity_sha256"] != canonical_digest(
        record["checkpoint"]
    ):
        raise RuntimeError(f"{label} pair contract does not bind the checkpoint")
    if contract["sampling_params"] != record["sampling_params"]:
        raise RuntimeError(f"{label} pair contract does not bind sampling params")
    if contract["workload"] != record["workload"]:
        raise RuntimeError(f"{label} pair contract does not bind the workload")

    engine = record["engine_args"]
    if not isinstance(engine, dict):
        raise RuntimeError(f"{label} engine args are invalid")
    normalized_engine = dict(engine)
    if normalized_engine.pop("radix_cache_backend", None) != "orbitkv":
        raise RuntimeError(f"{label} is not backed by the OrbitKV manager")
    if contract["engine_args"] != normalized_engine:
        raise RuntimeError(f"{label} pair contract does not bind engine args")
    if contract["capacity_tokens"] != engine.get("max_total_tokens"):
        raise RuntimeError(f"{label} pair contract does not bind capacity")
    capacity = engine.get("max_total_tokens")
    batch = record["workload"]["requests"]
    if (
        type(capacity) is not int
        or capacity != EXPECTED_CAPACITY_TOKENS[batch]
    ):
        raise RuntimeError(f"{label} capacity differs from the fixed B{batch} matrix")
    expected_engine = {
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": 128,
        "page_size": 16,
        "attention_backend": "flashinfer",
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": record["workload"]["requests"]
        * TRIGGER_TOKENS,
        "max_running_requests": record["workload"]["requests"],
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
        "random_seed": record["sampling_params"]["sampling_seed"],
        "log_level": "error",
        "max_total_tokens": engine.get("max_total_tokens"),
        "model_path": engine.get("model_path"),
        "radix_cache_backend": "orbitkv",
    }
    if set(engine) - {"mem_fraction_static"} != set(expected_engine):
        raise RuntimeError(f"{label} engine argument schema is invalid")
    if any(engine.get(name) != value for name, value in expected_engine.items()):
        raise RuntimeError(f"{label} engine contract is invalid")
    if Path(engine["model_path"]).name != MODEL_PATH_BASENAME:
        raise RuntimeError(f"{label} model path is not Qwen2.5-0.5B")
    if "mem_fraction_static" in engine:
        memory_fraction = engine["mem_fraction_static"]
        if (
            isinstance(memory_fraction, bool)
            or not isinstance(memory_fraction, (int, float))
            or not math.isfinite(memory_fraction)
            or not 0 < memory_fraction <= 1
        ):
            raise RuntimeError(f"{label} mem_fraction_static is invalid")
    expected_sampling = {
        "temperature": 0,
        "max_new_tokens": DECODE_TOKENS,
        "min_new_tokens": DECODE_TOKENS,
        "ignore_eos": True,
        "sampling_seed": record["sampling_params"]["sampling_seed"],
    }
    if record["sampling_params"] != expected_sampling:
        raise RuntimeError(f"{label} sampling contract is invalid")

    arguments = _command_arguments(record.get("command"), label)
    command_expected = {
        "--mode": mode,
        "--requests": str(record["workload"]["requests"]),
        "--iterations": str(record["workload"]["iterations"]),
        "--max-total-tokens": str(engine.get("max_total_tokens")),
        "--attention-backend": str(engine.get("attention_backend")),
        "--model": str(engine.get("model_path")),
        "--context-length": str(engine.get("context_length")),
        "--seed": str(record["sampling_params"]["sampling_seed"]),
    }
    command_mismatches = {
        name: {"expected": expected, "actual": arguments.get(name)}
        for name, expected in command_expected.items()
        if arguments.get(
            name, "128" if name == "--context-length" else "20260821"
        ) != expected
    }
    if command_mismatches:
        raise RuntimeError(f"{label} command differs from record: {command_mismatches}")
    if Path(arguments["--output"]).name != Path(label).name:
        raise RuntimeError(f"{label} command output does not match the record filename")
    command_memory = arguments.get("--mem-fraction-static")
    engine_memory = engine.get("mem_fraction_static")
    if (command_memory is None) != (engine_memory is None):
        raise RuntimeError(
            f"{label} command memory fraction differs from engine args"
        )
    if command_memory is not None:
        try:
            parsed_memory = float(command_memory)
        except ValueError as error:
            raise RuntimeError(f"{label} command memory fraction is invalid") from error
        if parsed_memory != engine_memory:
            raise RuntimeError(
                f"{label} command memory fraction differs from engine args"
            )

    environment = record["environment"]
    if not isinstance(environment, dict):
        raise RuntimeError(f"{label} environment is invalid")
    required_environment = {
        "SGLANG_PLUGINS": "orbitkv_manager",
        "SGLANG_USE_HND_KVCACHE": "0",
        "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
        "SGLANG_RADIX_FORCE_MISS": "0",
    }
    if any(
        environment.get(name) != value
        for name, value in required_environment.items()
    ):
        raise RuntimeError(f"{label} manager environment is invalid")
    for name, option in (
        ("ORBITKV_PLAN", "--plan"),
        ("ORBITKV_LIBRARY", "--library"),
        ("ORBITKV_SGLANG_ROOT", "--sglang-root"),
    ):
        if not _same_path(environment.get(name), arguments[option]):
            raise RuntimeError(f"{label} manager environment path is invalid: {name}")
    if (
        not isinstance(environment.get("PATH"), str)
        or not environment["PATH"]
        or not isinstance(environment.get("PYTHONPATH"), str)
        or not environment["PYTHONPATH"]
    ):
        raise RuntimeError(f"{label} process path environment is invalid")
    encoded_policy = environment.get("ORBITKV_TOKEN_RECLAMATION")
    if not isinstance(encoded_policy, str):
        raise RuntimeError(f"{label} token-reclamation policy is missing")
    def unique_policy(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate policy key {key!r}")
            result[key] = value
        return result

    try:
        policy = json.loads(
            encoded_policy,
            object_pairs_hook=unique_policy,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{label} token-reclamation policy is invalid") from error
    if not isinstance(policy, dict) or policy.pop("mode", None) != mode:
        raise RuntimeError(f"{label} token-reclamation mode is invalid")
    if policy != EXPECTED_VICTIM_POLICY:
        raise RuntimeError(
            f"{label} victim policy differs from the diagnostic contract"
        )
    if contract["victim_policy"] != policy:
        raise RuntimeError(f"{label} pair contract does not bind victim policy")


def _validate_outputs(
    record: dict[str, Any], iterations: int, batch: int, label: str
) -> None:
    outputs = record.get("request_output_ids")
    vocab_size = record["checkpoint_contract"]["vocab_size"]
    if not isinstance(outputs, list) or len(outputs) != iterations:
        raise RuntimeError(f"{label} output iteration shape is invalid")
    for row in outputs:
        if not isinstance(row, list) or len(row) != batch:
            raise RuntimeError(f"{label} output batch shape is invalid")
        for output_ids in row:
            if (
                not isinstance(output_ids, list)
                or len(output_ids) != DECODE_TOKENS
                or any(
                    type(token) is not int or not 0 <= token < vocab_size
                    for token in output_ids
                )
            ):
                raise RuntimeError(f"{label} output token vector is invalid")
    if record.get("output_token_digest_sha256") != canonical_digest(outputs):
        raise RuntimeError(f"{label} output-token digest is invalid")


def _validate_timings(
    record: dict[str, Any], iterations: int, label: str
) -> list[float]:
    timings = record.get("iteration_seconds")
    if not isinstance(timings, list) or len(timings) != iterations:
        raise RuntimeError(f"{label} iteration timing shape is invalid")
    normalized = [
        _positive_number(value, f"{label} iteration_seconds")
        for value in timings
    ]
    recorded_total = _positive_number(
        record.get("iteration_total_seconds"),
        f"{label} iteration_total_seconds",
    )
    if not math.isclose(recorded_total, sum(normalized), rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{label} iteration total does not match timings")
    load = _positive_number(record.get("load_seconds"), f"{label} load_seconds")
    total = _positive_number(record.get("total_seconds"), f"{label} total_seconds")
    if total < load + recorded_total:
        raise RuntimeError(f"{label} total duration is shorter than measured work")
    return normalized


def _validate_census(
    record: dict[str, Any], mode: str, batch: int, iterations: int,
    class_count: int, label: str,
) -> str:
    manager = record.get("manager")
    _require_exact_keys(manager, {"after_load", "final_census"}, f"{label} manager")
    after_load = manager["after_load"]
    final = manager["final_census"]
    pressure_contracts: list[str] = []
    for stage, census in (("after-load", after_load), ("final", final)):
        if not isinstance(census, dict):
            raise RuntimeError(f"{label} {stage} manager census is malformed")
        census_keys = set(census)
        if census_keys == set(CENSUS_KEYS):
            pressure_contracts.append("absent")
        elif census_keys == set(CENSUS_KEYS) | {PRESSURE_KEY}:
            pressure = census[PRESSURE_KEY]
            _require_exact_keys(
                pressure, PRESSURE_DISABLED, f"{label} {stage} pressure"
            )
            if any(
                type(pressure[name]) is not type(expected)
                or pressure[name] != expected
                for name, expected in PRESSURE_DISABLED.items()
            ):
                raise RuntimeError(
                    f"{label} {stage} pressure telemetry is not disabled"
                )
            pressure_contracts.append(PRESSURE_SCHEMA)
        else:
            _require_exact_keys(
                census, CENSUS_KEYS, f"{label} {stage} census"
            )
        if not isinstance(census, dict) or census.get("abi_version") != 8:
            raise RuntimeError(f"{label} {stage} census is not ABI8")
        stage_stats = census.get("manager_stats")
        stage_arenas = census.get("arena_stats")
        stage_counters = census.get("batch_counters")
        identities = census.get("identities")
        if (
            not isinstance(stage_stats, dict)
            or not isinstance(stage_arenas, list)
            or len(stage_arenas) != class_count
            or not isinstance(identities, list)
            or len(identities) != class_count
            or not isinstance(stage_counters, dict)
        ):
            raise RuntimeError(f"{label} {stage} manager census is malformed")
        _require_exact_keys(
            stage_stats, MANAGER_STATS_FIELDS, f"{label} {stage} manager_stats"
        )
        _require_exact_keys(
            stage_counters, BATCH_COUNTER_FIELDS, f"{label} {stage} batch_counters"
        )
        for name in BATCH_COUNTER_FIELDS:
            _nonnegative_int(
                stage_counters[name], f"{label} {stage} counter {name}"
            )
        if (
            census.get("state_plan_fingerprint") is not None
            or census.get("fixed_state_byte_count") != 0
            or census.get("fixed_state_descriptors") != []
            or census.get("tree_cache_type")
            != {
                "module": "orbitkv_sglang.plugin.prefix_cache",
                "qualname": "OrbitKvPrefixCache",
            }
        ):
            raise RuntimeError(f"{label} {stage} manager profile is invalid")
        plan_fingerprint = census.get("plan_fingerprint")
        if (
            not isinstance(plan_fingerprint, str)
            or not plan_fingerprint.startswith("sha256:")
        ):
            raise RuntimeError(f"{label} {stage} plan fingerprint is invalid")
        _require_sha256(
            plan_fingerprint.removeprefix("sha256:"),
            f"{label} {stage} plan fingerprint",
        )
        if census.get("swa_activity") != {
            "status": "exposed",
            "applicable": False,
            "swa_retirement_certificates": 0,
            "swa_pages_reclaimed": 0,
            "swa_wrap_events": 0,
        }:
            raise RuntimeError(f"{label} {stage} SWA census is invalid")
        for name in MANAGER_DRAIN_FIELDS:
            if _nonnegative_int(
                stage_stats.get(name), f"{label} {stage} manager_stats.{name}"
            ) != 0:
                raise RuntimeError(f"{label} {stage} manager did not drain: {name}")
        stage_free = _nonnegative_int(
            stage_stats.get("free_pages"),
            f"{label} {stage} manager_stats.free_pages",
        )
        arena_free = 0
        page_ranges = []
        engine_epochs = set()
        for index, (identity, arena) in enumerate(
            zip(identities, stage_arenas, strict=True)
        ):
            if not isinstance(identity, dict) or not isinstance(arena, dict):
                raise RuntimeError(f"{label} {stage} arena[{index}] is invalid")
            _require_exact_keys(
                identity, IDENTITY_FIELDS, f"{label} {stage} identity[{index}]"
            )
            _require_exact_keys(
                arena, ARENA_STATS_FIELDS, f"{label} {stage} arena[{index}]"
            )
            for name in IDENTITY_FIELDS:
                _nonnegative_int(
                    identity.get(name), f"{label} {stage} identity[{index}].{name}"
                )
            if (
                identity["class_id"] != index
                or identity["engine_epoch"] <= 0
                or identity["pool_epoch"] <= 0
                or identity["pool_id"] <= 0
                or identity["backend_domain"] <= 0
                or identity["page_count"] <= 0
                or identity["page_tokens"] != 16
                or identity["first_page_id"] <= 0
            ):
                raise RuntimeError(f"{label} {stage} arena identity is invalid")
            if any(
                arena.get(name) != identity[name]
                for name in ARENA_IDENTITY_FIELDS
            ):
                raise RuntimeError(f"{label} {stage} arena identity echo differs")
            page_range = (
                identity["first_page_id"],
                identity["first_page_id"] + identity["page_count"],
            )
            if any(
                page_range[0] < end and begin < page_range[1]
                for begin, end in page_ranges
            ):
                raise RuntimeError(f"{label} {stage} arena page ranges overlap")
            page_ranges.append(page_range)
            engine_epochs.add(identity["engine_epoch"])
            free = _nonnegative_int(
                arena.get("free_pages"),
                f"{label} {stage} arena[{index}].free_pages",
            )
            pages = _nonnegative_int(
                arena.get("page_count"),
                f"{label} {stage} arena[{index}].page_count",
            )
            if pages <= 0 or free != pages:
                raise RuntimeError(
                    f"{label} {stage} arena[{index}] did not return every page"
                )
            for name in ARENA_DRAIN_FIELDS:
                if _nonnegative_int(
                    arena.get(name), f"{label} {stage} arena[{index}].{name}"
                ) != 0:
                    raise RuntimeError(
                        f"{label} {stage} arena[{index}] did not drain: {name}"
                    )
            arena_free += free
        if len(engine_epochs) != 1:
            raise RuntimeError(f"{label} {stage} arena engine epochs differ")
        if stage_free != arena_free:
            raise RuntimeError(f"{label} {stage} aggregate free-page census differs")
        for name in FAILURE_COUNTER_FIELDS:
            if _nonnegative_int(
                stage_counters.get(name), f"{label} {stage} counter {name}"
            ) != 0:
                raise RuntimeError(
                    f"{label} {stage} failure/fail-stop counter is nonzero: {name}"
                )
        if stage == "after-load":
            dirty_counters = {
                name: value for name, value in stage_counters.items() if value != 0
            }
            if dirty_counters:
                raise RuntimeError(
                    f"{label} after-load counters are nonzero: {dirty_counters}"
                )
    stats = final.get("manager_stats")
    arenas = final.get("arena_stats")
    counters = final.get("batch_counters")
    assert isinstance(stats, dict)
    assert isinstance(arenas, list)
    assert isinstance(counters, dict)
    if after_load["identities"] != final["identities"]:
        raise RuntimeError(f"{label} arena identity changed during the workload")
    if after_load["arena_stats"] != [
        {
            **arena,
            **{name: 0 for name in ARENA_DRAIN_FIELDS},
            "free_pages": arena["page_count"],
        }
        for arena in final["arena_stats"]
    ]:
        raise RuntimeError(f"{label} after-load and final arena geometry differ")
    for name in (
        "plan_fingerprint", "state_plan_fingerprint",
        "fixed_state_byte_count", "fixed_state_descriptors",
        "tree_cache_type",
    ):
        if after_load.get(name) != final.get(name):
            raise RuntimeError(f"{label} manager {name} changed during the workload")
    if pressure_contracts[0] != pressure_contracts[1]:
        raise RuntimeError(
            f"{label} manager pressure schema changed during the workload"
        )
    unexpected_changes = {
        name: {"after_load": after_load["batch_counters"][name],
               "final": counters[name]}
        for name in UNCHANGED_COUNTER_FIELDS
        if counters[name] != after_load["batch_counters"][name]
    }
    if unexpected_changes:
        raise RuntimeError(
            f"{label} unrelated counters changed during relocation: "
            f"{unexpected_changes}"
        )
    batch_call_counts = [
        counters[name]
        for name in (
            "prepare_batch_calls", "submit_batch_calls",
            "complete_batch_calls", "forward_events",
        )
    ]
    if len(set(batch_call_counts)) != 1:
        raise RuntimeError(f"{label} append batch call identities disagree")
    if counters["event_waits"] > counters["forward_events"]:
        raise RuntimeError(f"{label} event wait count exceeds forward events")
    if (
        counters["event_queries"] + counters["event_waits"]
        < counters["forward_events"]
    ):
        raise RuntimeError(f"{label} event observation counts are incomplete")

    scheduler_events = iterations * EXPECTED_RECLAMATION_ROUNDS
    request_rounds = batch * scheduler_events
    expected = {
        "token_disposition_batches": scheduler_events,
        "token_policy_evictions": request_rounds * VICTIM_COUNT * class_count,
        "mark_token_dispositions_batch_calls": (
            scheduler_events if mode == "relocate" else request_rounds
        ),
    }
    append_batches = iterations * DECODE_TOKENS
    expected.update(
        request_acquire_batch_calls=iterations,
        request_fork_batch_calls=0,
        prepare_batch_calls=append_batches,
        submit_batch_calls=append_batches,
        complete_batch_calls=append_batches,
        release_batch_calls=iterations,
        acknowledge_reclamations_batch_calls=(
            iterations + (scheduler_events if mode == "relocate" else 0)
        ),
        recycle_requests_batch_calls=iterations,
        prefix_lookup_batch_calls=0,
        prefix_attach_batch_calls=0,
        prefix_publish_batch_calls=0,
        prefix_publish_release_batch_calls=0,
        prefix_evict_batch_calls=0,
        prefix_recycle_batch_calls=0,
        forward_events=append_batches,
        completion_values=(
            append_batches + (scheduler_events if mode == "relocate" else 0)
        ),
        event_queries=iterations * (DECODE_TOKENS - 1),
        event_waits=iterations,
        prefix_matches=iterations * batch,
        prefix_hits=0,
        prefix_publishes=0,
        prefix_evictions=0,
        prefix_evicted_full_tokens=0,
        prefix_evicted_swa_tokens=0,
        prefix_global_alias_scans=0,
        cow_copy_intents=0,
        cow_move_calls=0,
        cow_copied_tokens=0,
        fixed_state_prepares=0,
        fixed_state_clears=0,
        fixed_state_copies=0,
        fixed_state_events=0,
        fixed_state_retirements=0,
        fixed_state_acks=0,
    )
    if mode == "relocate":
        # Each scheduler event performs one batched pre-view query, one
        # post-view query, and one scalar policy preflight query per request.
        token_view_calls = scheduler_events * (batch + 2) * class_count
        expected.update(
            relocation_batches=scheduler_events,
            relocation_moves=request_rounds * RETAINED_COUNT_AT_TRIGGER,
            relocation_reclaimed_pages=request_rounds,
            relocation_copy_events=scheduler_events,
            relocation_copy_tokens=request_rounds * RETAINED_COUNT_AT_TRIGGER,
            prepare_relocation_batch_calls=scheduler_events,
            submit_relocation_batch_calls=scheduler_events,
            complete_relocation_batch_calls=scheduler_events,
            token_views_batch_calls=token_view_calls,
            mirror_validation_calls=iterations * 2,
            mirror_syncs=iterations,
            buffer_too_small_preflights=token_view_calls + iterations,
            cold_workspace_allocations=(
                token_view_calls + iterations + scheduler_events
            ),
            materialized_page_objects=0,
        )
    else:
        token_view_calls = request_rounds * 3
        expected.update({name: 0 for name in RELOCATION_COUNTER_FIELDS})
        expected.update(
            token_views_batch_calls=token_view_calls,
            mirror_validation_calls=iterations * 2,
            mirror_syncs=iterations,
            buffer_too_small_preflights=token_view_calls + iterations,
            cold_workspace_allocations=token_view_calls + iterations,
            materialized_page_objects=0,
        )
    mismatches = {}
    for name, expected_value in expected.items():
        actual = _nonnegative_int(counters.get(name), f"{label} counter {name}")
        if actual != expected_value:
            mismatches[name] = {"expected": expected_value, "actual": actual}
    if mismatches:
        raise RuntimeError(f"{label} relocation counters differ: {mismatches}")
    return pressure_contracts[0]


def _validate_h20(
    record: dict[str, Any], label: str, started_at: datetime
) -> tuple[str, int, tuple[int, int]]:
    snapshots = record.get("gpu_snapshots")
    expected_labels = (
        "before_engine", "after_load", "after_workload", "after_shutdown"
    )
    if not isinstance(snapshots, list) or len(snapshots) != len(expected_labels):
        raise RuntimeError(f"{label} GPU snapshot matrix is invalid")
    uuid: str | None = None
    previous_time_ns: int | None = None
    for snapshot, expected_label in zip(snapshots, expected_labels, strict=True):
        gpus = snapshot.get("gpus") if isinstance(snapshot, dict) else None
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("label") != expected_label
            or not isinstance(gpus, list)
            or len(gpus) != 1
            or not isinstance(gpus[0], dict)
            or gpus[0].get("name") != GPU_NAME
            or not isinstance(gpus[0].get("uuid"), str)
            or not gpus[0]["uuid"]
        ):
            raise RuntimeError(f"{label} does not contain one consistent H20 snapshot")
        time_ns = snapshot.get("time_ns")
        if type(time_ns) is not int or time_ns <= 0:
            raise RuntimeError(f"{label} GPU snapshot timestamp is invalid")
        if previous_time_ns is not None and time_ns <= previous_time_ns:
            raise RuntimeError(f"{label} GPU snapshots are not chronological")
        previous_time_ns = time_ns
        if uuid is None:
            uuid = gpus[0]["uuid"]
        elif uuid != gpus[0]["uuid"]:
            raise RuntimeError(f"{label} GPU UUID changes between snapshots")
    assert uuid is not None
    before_ns = snapshots[0]["time_ns"]
    after_ns = snapshots[-1]["time_ns"]
    started_ns = int(started_at.timestamp() * 1_000_000_000)
    if not before_ns <= started_ns <= after_ns:
        raise RuntimeError(f"{label} started_at is outside its GPU snapshots")
    observed_seconds = (after_ns - before_ns) / 1_000_000_000
    if observed_seconds > record["total_seconds"]:
        raise RuntimeError(f"{label} GPU interval exceeds total duration")
    return uuid, len(snapshots), (before_ns, after_ns)


def _validate_record(
    record: dict[str, Any], epoch: int, batch: int, mode: str, label: str,
    *, expected_harness_sha256: str, expected_adapter: dict[str, Any],
    expected_adapter_source_root: Path | None = None,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> dict[str, Any]:
    _require_exact_keys(record, RECORD_KEYS, label)
    if record.get("schema") != RECORD_SCHEMA:
        raise RuntimeError(f"{label} record schema is invalid")
    if record.get("mode") != mode:
        raise RuntimeError(f"{label} mode does not match its filename")
    iterations, class_count = _validate_workload(
        record, batch, label, expected_iterations=expected_iterations
    )
    _validate_checkpoint(record.get("checkpoint"), label)
    _validate_pairing(record, mode, label)
    _validate_outputs(record, iterations, batch, label)
    timings = _validate_timings(record, iterations, label)
    pressure_contract = _validate_census(
        record, mode, batch, iterations, class_count, label
    )
    started_at = _timestamp(record.get("started_at_utc"), label)
    gpu_uuid, snapshot_count, observed_interval_ns = _validate_h20(
        record, label, started_at
    )
    source = record.get("source_identity")
    if not isinstance(source, dict) or not source:
        raise RuntimeError(f"{label} source identity is missing")
    source_required = {
        "root", "release", "revision", "python_source_contract",
        "dirty_paths",
        "loader", "plugin_selection", "harness_sha256", "adapter",
        "library", "plan",
    }
    _require_exact_keys(source, source_required, f"{label} source identity")
    if (
        source.get("release") != SGLANG_RELEASE
        or source.get("revision") != SGLANG_REVISION
        or source.get("python_source_contract")
        != SOURCE_CONTRACT
        or source.get("dirty_paths")
        != [LOADER_PATH]
    ):
        raise RuntimeError(f"{label} source contract is invalid")
    revision = source.get("revision")
    harness = source.get("harness_sha256")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(harness, str)
        or len(harness) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in harness)
    ):
        raise RuntimeError(f"{label} source revision or harness digest is invalid")
    if harness != expected_harness_sha256:
        raise RuntimeError(f"{label} relocation harness differs from this checkout")
    loader = source.get("loader")
    expected_loader = {
        "path": LOADER_PATH,
        "head_git_blob": LOADER_HEAD_GIT_BLOB,
        "worktree_git_blob": LOADER_WORKTREE_GIT_BLOB,
        "worktree_sha256": LOADER_WORKTREE_SHA256,
        "patch_sha256": LOADER_PATCH_SHA256,
    }
    if loader != expected_loader:
        raise RuntimeError(f"{label} source loader identity is invalid")
    plugin = source.get("plugin_selection")
    trusted_adapter_source_root = (
        ROOT / "integrations/sglang/src"
        if expected_adapter_source_root is None
        else Path(expected_adapter_source_root)
    ).expanduser().resolve()
    plugin_module = (
        Path(plugin.get("module", "")).resolve()
        if isinstance(plugin, dict)
        else Path("/invalid")
    )
    recorded_adapter_roots = [
        path
        for value in record.get("environment", {}).get("PYTHONPATH", "").split(
            os.pathsep
        )
        if value and (path := Path(value).resolve()).name == "src"
        and (path / "orbitkv_sglang").as_posix()
    ]
    if not recorded_adapter_roots:
        raise RuntimeError(f"{label} PYTHONPATH omits its adapter source root")
    if (
        not isinstance(plugin, dict)
        or set(plugin) != {*MANAGER_ENTRYPOINT.keys(), "module"}
        or any(plugin.get(name) != value for name, value in MANAGER_ENTRYPOINT.items())
        or not isinstance(plugin.get("module"), str)
        or Path(plugin["module"]).name not in {"plugin.py", "__init__.py"}
        or not any(
            plugin_module.is_relative_to(root / "orbitkv_sglang")
            for root in recorded_adapter_roots
        )
    ):
        raise RuntimeError(f"{label} manager plugin identity is invalid")
    adapter = source.get("adapter")
    adapter_files = adapter.get("files") if isinstance(adapter, dict) else None
    if (
        not isinstance(adapter_files, list)
        or not adapter_files
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str)
            or not item["path"].startswith("integrations/sglang/")
            or _require_sha256(
                item["sha256"], f"{label} adapter file SHA-256"
            ) is None
            for item in adapter_files
        )
    ):
        raise RuntimeError(f"{label} adapter identity is invalid")
    if adapter != expected_adapter:
        raise RuntimeError(f"{label} adapter identity differs from this checkout")
    library = _artifact_identity(source.get("library"), f"{label} library")
    plan = _artifact_identity(source.get("plan"), f"{label} plan")
    if (
        Path(library["path"]).name != "liborbitkv_ffi.so"
        or Path(plan["path"]).name
        != "qwen2.5-0.5b-full-page16-bf16.json"
    ):
        raise RuntimeError(f"{label} library or plan filename is invalid")
    arguments = _command_arguments(record.get("command"), label)
    if (
        not _same_path(arguments["--library"], library["path"])
        or not _same_path(arguments["--plan"], plan["path"])
        or not _same_path(arguments["--sglang-root"], source.get("root"))
    ):
        raise RuntimeError(f"{label} command does not bind source artifacts")
    if not any(
        plugin_module.is_relative_to(root / "orbitkv_sglang")
        for root in recorded_adapter_roots
    ):
        raise RuntimeError(f"{label} plugin module is outside PYTHONPATH")
    if not trusted_adapter_source_root.name == "src":
        raise RuntimeError(f"{label} expected adapter source root is invalid")
    runtime = record.get("runtime_identity")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {
            "python", "python_version", "sglang_version", "gpu_profile"
        }
        or runtime.get("sglang_version") != "0.5.17"
        or runtime.get("gpu_profile") != "single H20 eager"
        or not isinstance(runtime.get("python"), str)
        or not runtime["python"]
        or not isinstance(runtime.get("python_version"), str)
        or not runtime["python_version"]
    ):
        raise RuntimeError(f"{label} runtime identity is invalid")
    if record["command"][0] != runtime["python"]:
        raise RuntimeError(f"{label} command Python differs from runtime identity")
    return {
        "epoch": epoch,
        "batch": batch,
        "mode": mode,
        "record": record,
        "started_at": started_at,
        "observed_interval_ns": observed_interval_ns,
        "timings": timings,
        "iterations": iterations,
        "class_count": class_count,
        "gpu_uuid": gpu_uuid,
        "snapshot_count": snapshot_count,
        "source": source,
        "library": library,
        "plan": plan,
        "runtime": runtime,
        "pressure_contract": pressure_contract,
    }


def validate_record(
    record: dict[str, Any], epoch: int, batch: int, mode: str, label: str,
    *, expected_harness_sha256: str, expected_adapter: dict[str, Any],
    expected_adapter_source_root: Path | None = None,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> dict[str, Any]:
    """Validate one record against explicitly supplied source identities."""

    if expected_adapter_source_root is not None:
        _verify_adapter_source_identity(
            Path(expected_adapter_source_root), expected_adapter
        )
    return _validate_record(
        record, epoch, batch, mode, label,
        expected_harness_sha256=expected_harness_sha256,
        expected_adapter=expected_adapter,
        expected_adapter_source_root=expected_adapter_source_root,
        expected_iterations=expected_iterations,
    )


def _equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise RuntimeError(f"{label} differs across the diagnostic matrix")


def _mode_statistics(values: list[float], output_tokens: int) -> dict[str, Any]:
    if len(values) < 2:
        raise RuntimeError("hot timing statistics require at least two samples")
    mean = statistics.fmean(values)
    return {
        "sample_count": len(values),
        "mean_seconds": mean,
        "median_seconds": statistics.median(values),
        "p95_seconds_inclusive": statistics.quantiles(
            values, n=100, method="inclusive"
        )[94],
        "output_tokens_per_second": output_tokens / mean,
    }


def _percent_delta(new: float, baseline: float) -> float:
    return (new / baseline - 1.0) * 100.0


def _build_summary(
    records: dict[tuple[int, int, str], dict[str, Any]],
    execution_orders: list[list[str]],
    identity: dict[str, Any],
    gpu_uuid: str,
    snapshot_count: int,
) -> dict[str, Any]:
    groups = []
    for batch in BATCHES:
        samples = {mode: [] for mode in MODES}
        epochs = []
        iterations = records[(1, batch, "naive")]["iterations"]
        for epoch, order in zip(EPOCHS, execution_orders, strict=True):
            hot = {}
            for mode in MODES:
                values = records[(epoch, batch, mode)]["timings"][1:]
                hot[mode] = values
                samples[mode].extend(values)
            naive_mean = statistics.fmean(hot["naive"])
            relocate_mean = statistics.fmean(hot["relocate"])
            epochs.append(
                {
                    "epoch": epoch,
                    "execution_order": order,
                    "naive_mean_seconds": naive_mean,
                    "relocate_mean_seconds": relocate_mean,
                    "relocate_over_naive_mean_latency_percent": _percent_delta(
                        relocate_mean, naive_mean
                    ),
                }
            )
        output_tokens = batch * DECODE_TOKENS
        naive = _mode_statistics(samples["naive"], output_tokens)
        relocate = _mode_statistics(samples["relocate"], output_tokens)
        groups.append(
            {
                "batch_size": batch,
                "epoch_count": len(EPOCHS),
                "iterations_per_process": iterations,
                "excluded_iteration_indices": [0],
                "hot_iteration_indices": list(range(1, iterations)),
                "hot_sample_count_per_mode": len(samples["naive"]),
                "percentile_method": (
                    "statistics.quantiles(n=100, method=inclusive)"
                ),
                "output_tokens_per_iteration": output_tokens,
                "naive": naive,
                "relocate": relocate,
                "relocate_vs_naive": {
                    "mean_latency_percent": _percent_delta(
                        relocate["mean_seconds"], naive["mean_seconds"]
                    ),
                    "median_latency_percent": _percent_delta(
                        relocate["median_seconds"], naive["median_seconds"]
                    ),
                    "p95_latency_percent": _percent_delta(
                        relocate["p95_seconds_inclusive"],
                        naive["p95_seconds_inclusive"],
                    ),
                    "throughput_percent": _percent_delta(
                        relocate["output_tokens_per_second"],
                        naive["output_tokens_per_second"],
                    ),
                },
                "epochs": epochs,
            }
        )

    naive_first = sum(order[0] == "naive" for order in execution_orders)
    relocate_first = sum(order[0] == "relocate" for order in execution_orders)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "diagnostic_pair_verification_passed",
        "evidence_class": "diagnostic_only",
        "diagnostic_only": True,
        "qualification_claim": "diagnostic_only_not_qualified",
        "sealed": False,
        "source_clean": False,
        "source_dirty": True,
        "hardware_attested": False,
        "qualified": False,
        "performance_go": False,
        "epoch_count": len(EPOCHS),
        "batch_sizes": list(BATCHES),
        "record_count": len(EPOCHS) * len(BATCHES) * len(MODES),
        "pair_count": len(EPOCHS) * len(BATCHES),
        "all_pairs_passed": True,
        "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
        "execution_order": {
            "source": "started_at_utc",
            "alternating": True,
            "naive_first_epochs": naive_first,
            "relocate_first_epochs": relocate_first,
            "epochs": [
                {"epoch": epoch, "order": order}
                for epoch, order in zip(EPOCHS, execution_orders, strict=True)
            ],
        },
        "identity": identity,
        "hardware": {
            "attestation": "recorded_observation_only",
            "observed_name": GPU_NAME,
            "observed_uuid": gpu_uuid,
            "snapshot_count": snapshot_count,
        },
        "groups": groups,
    }


def verify_evidence(
    root: Path, *,
    expected_harness_sha256: str | None = None,
    expected_adapter: dict[str, Any] | None = None,
    expected_adapter_source_root: Path | None = None,
) -> dict[str, Any]:
    records_root = _records_directory(Path(root))
    paths = _expected_paths()
    raw_records = {
        key: _strict_json(records_root / relative)
        for key, relative in paths.items()
    }
    baseline_source = raw_records[(1, 1, "naive")].get("source_identity")
    if not isinstance(baseline_source, dict):
        raise RuntimeError("baseline source identity is missing")
    if expected_harness_sha256 is None:
        expected_harness_sha256 = _require_sha256(
            baseline_source.get("harness_sha256"),
            "historical diagnostic harness SHA-256",
        )
        if expected_harness_sha256 != DIAGNOSTIC_HARNESS_SHA256:
            raise RuntimeError(
                "record relocation harness differs from this checkout "
                "(pinned historical diagnostic identity)"
            )
    else:
        _require_sha256(expected_harness_sha256, "expected harness SHA-256")
    if expected_adapter is None:
        candidate = baseline_source.get("adapter")
        if not isinstance(candidate, dict):
            raise RuntimeError("baseline adapter identity is missing")
        expected_adapter = candidate
        if canonical_digest(expected_adapter) != DIAGNOSTIC_ADAPTER_IDENTITY_SHA256:
            raise RuntimeError(
                "record adapter identity is not the pinned historical diagnostic"
            )
    if expected_adapter_source_root is not None:
        _verify_adapter_source_identity(
            Path(expected_adapter_source_root), expected_adapter
        )
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for key, relative in paths.items():
        epoch, batch, mode = key
        records[key] = _validate_record(
            raw_records[key],
            epoch,
            batch,
            mode,
            relative,
            expected_harness_sha256=expected_harness_sha256,
            expected_adapter=expected_adapter,
            expected_adapter_source_root=expected_adapter_source_root,
        )

    baseline = records[(1, 1, "naive")]
    for item in records.values():
        _equal(item["source"], baseline["source"], "source identity")
        _equal(item["library"], baseline["library"], "library identity")
        _equal(item["plan"], baseline["plan"], "plan identity")
        _equal(
            item["record"]["checkpoint"],
            baseline["record"]["checkpoint"],
            "checkpoint identity",
        )
        _equal(
            item["record"]["checkpoint_contract"],
            baseline["record"]["checkpoint_contract"],
            "checkpoint contract",
        )
        _equal(item["runtime"], baseline["runtime"], "runtime identity")
        _equal(item["class_count"], baseline["class_count"], "class count")
        _equal(
            item["pressure_contract"], baseline["pressure_contract"],
            "pressure census contract",
        )
        _equal(item["gpu_uuid"], baseline["gpu_uuid"], "GPU identity")
        _equal(item["iterations"], baseline["iterations"], "iteration count")
        _equal(
            item["record"]["sampling_params"]["sampling_seed"],
            baseline["record"]["sampling_params"]["sampling_seed"],
            "sampling seed",
        )

    execution_orders: list[list[str]] = []
    previous_order: list[str] | None = None
    previous_epoch_end_ns: int | None = None
    for epoch in EPOCHS:
        epoch_orders = []
        epoch_intervals: list[tuple[int, int, str]] = []
        for batch in BATCHES:
            naive = records[(epoch, batch, "naive")]
            relocate = records[(epoch, batch, "relocate")]
            if naive["record"]["pairing"] != relocate["record"]["pairing"]:
                raise RuntimeError(
                    f"epoch {epoch} B{batch} pair key or contract differs"
                )
            if (
                naive["record"]["request_output_ids"]
                != relocate["record"]["request_output_ids"]
            ):
                raise RuntimeError(f"epoch {epoch} B{batch} output tokens differ")
            naive_counters = naive["record"]["manager"]["final_census"][
                "batch_counters"
            ]
            relocate_counters = relocate["record"]["manager"][
                "final_census"
            ]["batch_counters"]
            for name in ("token_disposition_batches", "token_policy_evictions"):
                if naive_counters[name] != relocate_counters[name]:
                    raise RuntimeError(
                        f"epoch {epoch} B{batch} common counter differs: {name}"
                    )
            if naive["started_at"] == relocate["started_at"]:
                raise RuntimeError(f"epoch {epoch} B{batch} start times are identical")
            order = (
                ["naive", "relocate"]
                if naive["started_at"] < relocate["started_at"]
                else ["relocate", "naive"]
            )
            epoch_orders.append(order)
            epoch_intervals.extend(
                (
                    (
                        naive["observed_interval_ns"][0],
                        naive["observed_interval_ns"][1],
                        f"epoch {epoch} B{batch} naive",
                    ),
                    (
                        relocate["observed_interval_ns"][0],
                        relocate["observed_interval_ns"][1],
                        f"epoch {epoch} B{batch} relocate",
                    ),
                )
            )
        if epoch_orders[0] != epoch_orders[1]:
            raise RuntimeError(f"epoch {epoch} B1/B4 execution orders differ")
        order = epoch_orders[0]
        if previous_order is not None and order == previous_order:
            raise RuntimeError("four-epoch execution order is not alternating")
        ordered_intervals = sorted(epoch_intervals)
        for (_, previous_end, _), (current_start, _, current_label) in zip(
            ordered_intervals, ordered_intervals[1:]
        ):
            if previous_end > current_start:
                raise RuntimeError(f"{current_label} overlaps another record process")
        epoch_start = ordered_intervals[0][0]
        epoch_end = max(end for _, end, _ in ordered_intervals)
        if previous_epoch_end_ns is not None and epoch_start <= previous_epoch_end_ns:
            raise RuntimeError("epoch process intervals are not strictly chronological")
        execution_orders.append(order)
        previous_order = order
        previous_epoch_end_ns = epoch_end

    if sum(order[0] == "naive" for order in execution_orders) != 2:
        raise RuntimeError("four-epoch execution order is not balanced")

    for batch in BATCHES:
        expected_pairing = records[(1, batch, "naive")]["record"]["pairing"]
        expected_outputs = records[(1, batch, "naive")]["record"][
            "request_output_ids"
        ]
        for epoch in EPOCHS:
            _equal(
                records[(epoch, batch, "naive")]["record"]["pairing"],
                expected_pairing,
                f"B{batch} pair identity",
            )
            _equal(
                records[(epoch, batch, "naive")]["record"][
                    "request_output_ids"
                ],
                expected_outputs,
                f"B{batch} deterministic output tokens",
            )

    if (
        records[(1, 1, "naive")]["record"]["workload"][
            "input_token_digest_sha256"
        ]
        == records[(1, 4, "naive")]["record"]["workload"][
            "input_token_digest_sha256"
        ]
    ):
        raise RuntimeError("B1 and B4 input-token identities unexpectedly match")

    identity = {
        "source_identity_sha256": canonical_digest(baseline["source"]),
        "library": baseline["library"],
        "plan": baseline["plan"],
        "checkpoint_identity_sha256": canonical_digest(
            baseline["record"]["checkpoint"]
        ),
        "checkpoint_contract_sha256": canonical_digest(
            baseline["record"]["checkpoint_contract"]
        ),
    }
    all_intervals = sorted(
        (
            item["observed_interval_ns"][0],
            item["observed_interval_ns"][1],
            f"epoch {item['epoch']} B{item['batch']} {item['mode']}",
        )
        for item in records.values()
    )
    for (_, previous_end, previous_label), (current_start, _, current_label) in zip(
        all_intervals, all_intervals[1:]
    ):
        if previous_end > current_start:
            raise RuntimeError(
                f"record processes overlap: {previous_label} and {current_label}"
            )

    return _build_summary(
        records,
        execution_orders,
        identity,
        baseline["gpu_uuid"],
        sum(item["snapshot_count"] for item in records.values()),
    )


def verify_sealed_archive(root: Path) -> dict[str, Any]:
    """Delegate sealed-envelope verification without importing it eagerly."""

    import importlib.util

    path = Path(__file__).with_name("verify_token_relocation_h20_seal.py")
    spec = importlib.util.spec_from_file_location(
        "_orbitkv_token_relocation_seal", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load trusted token-relocation seal verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_sealed_archive(Path(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence_root",
        type=Path,
        help="directory containing records/epoch-001 through epoch-004",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = verify_evidence(args.evidence_root)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
