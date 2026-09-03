from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


PAGE_TOKENS = 16
# Keep this equal to core/ffi/src/manager/mod.rs::MAX_PLAN_JSON_BYTES.
MANAGER_PLAN_JSON_MAX_BYTES = 1024 * 1024
RetentionKind = Literal["full", "sliding", "chunked"]
TokenStorageKind = Literal["token_kv", "latent_kv"]
ReclamationMode = Literal["off", "naive", "relocate"]
FixedStateKind = Literal["mamba", "gdn", "kda", "linear_attention", "convolution"]
ManagerPlanFormat = Literal["kv_plan", "retention_ir"]


@dataclass(frozen=True, slots=True)
class TokenReclamationConfig:
    mode: ReclamationMode = "off"
    trigger_tokens: int = 0
    retained_per_page: int = 0
    policy_id: int = 0
    policy_version: int = 0
    quality_contract: int = 0
    fragmentation_threshold_milli: int = 250
    maximum_source_pages: int = 0
    evacuation_headroom_pages: int = 0


@dataclass(frozen=True, slots=True)
class ClassConfig:
    """One compiler class and its one SGLang physical arena."""

    class_id: int
    pool_id: int
    backend_domain: int
    name: str
    layers: tuple[int, ...]
    retention: RetentionKind
    bytes_per_token_per_layer: int
    window_tokens: int | None
    period_blocks: int | None
    storage: TokenStorageKind = "token_kv"
    components: tuple[tuple[str, int], ...] = ()
    chunk_tokens: int | None = None
    blocks_per_epoch: int | None = None

    @property
    def kernel_window_left(self) -> int | None:
        return None if self.window_tokens is None else self.window_tokens - 1

    @property
    def components_by_name(self) -> dict[str, int]:
        return dict(self.components)

    def minimum_sliding_pool_tokens(
        self, *, maximum_running_requests: int, chunked_prefill_tokens: int
    ) -> int:
        if self.retention != "sliding" or self.period_blocks is None:
            raise ValueError("only a sliding class has a finite resident-pool floor")
        maximum_running_requests = _positive_runtime_int(
            "maximum_running_requests", maximum_running_requests
        )
        chunked_prefill_tokens = _positive_runtime_int(
            "chunked_prefill_tokens", chunked_prefill_tokens
        )
        staging_pages = (
            _ceil_div(chunked_prefill_tokens, PAGE_TOKENS)
            + maximum_running_requests
            - 1
        )
        return (
            self.period_blocks * maximum_running_requests + staging_pages
        ) * PAGE_TOKENS


@dataclass(frozen=True, slots=True)
class FixedStateConfig:
    name: str
    kind: FixedStateKind
    layers: tuple[int, ...]
    state_bytes_per_layer: int
    checkpoint_slots_per_request: int
    kernel_width: int | None = None

    @property
    def byte_count(self) -> int:
        return self.state_bytes_per_layer * len(self.layers)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """One validated runtime-manifest configuration."""

    library_path: Path
    plan_json: bytes
    plan_fingerprint: str
    page_tokens: int
    classes: tuple[ClassConfig, ...]
    token_reclamation: TokenReclamationConfig = TokenReclamationConfig()
    fixed_states: tuple[FixedStateConfig, ...] = ()
    runtime_manifest_path: Path | None = None
    runtime_manifest_fingerprint: str | None = None
    capability_requirements: tuple[str, ...] = ()
    execution_signature: Mapping[str, Any] | None = None
    runtime_binding: Mapping[str, Any] | None = None
    manager_plan_format: ManagerPlanFormat = "kv_plan"

    @property
    def num_hidden_layers(self) -> int:
        layers = {layer for item in self.classes for layer in item.layers}
        layers.update(layer for item in self.fixed_states for layer in item.layers)
        return max(layers, default=-1) + 1

    @property
    def fixed_state_byte_count(self) -> int:
        return sum(item.byte_count for item in self.fixed_states)

    @property
    def classes_by_id(self) -> dict[int, ClassConfig]:
        return {item.class_id: item for item in self.classes}

    @property
    def full_class(self) -> ClassConfig | None:
        return next(
            (item for item in self.classes if item.retention == "full"), None
        )

    @property
    def sliding_class(self) -> ClassConfig | None:
        return next(
            (item for item in self.classes if item.retention == "sliding"), None
        )

    @property
    def chunked_class(self) -> ClassConfig | None:
        return next(
            (item for item in self.classes if item.retention == "chunked"), None
        )

    @property
    def primary_class(self) -> ClassConfig | None:
        return self.full_class or self.sliding_class or self.chunked_class


def load_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load the single canonical runtime-manifest configuration."""

    source = os.environ if environ is None else environ
    from .runtime_manifest import load_runtime_manifest

    return load_runtime_manifest(source)


def _validate_token_reclamation(
    token_reclamation: TokenReclamationConfig,
    retentions: tuple[RetentionKind, ...],
    classes: tuple[ClassConfig, ...],
) -> None:
    if token_reclamation.mode != "off" and retentions not in (
        ("full",),
        ("full", "sliding"),
    ):
        raise ValueError(
            "token reclamation requires Full or ordered Full+SWA classes"
        )
    if (
        token_reclamation.mode != "off"
        and retentions == ("full", "sliding")
        and token_reclamation.trigger_tokens > classes[1].kernel_window_left
    ):
        raise ValueError(
            "token-reclamation trigger exceeds the shared Full/SWA visibility prefix"
        )


def _state_layers(raw: Any, path: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}.layers must be a non-empty list")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in raw
    ):
        raise ValueError(f"{path}.layers must contain nonnegative integers")
    if raw != sorted(set(raw)):
        raise ValueError(f"{path}.layers must be unique and ascending")
    return tuple(raw)


def _token_reclamation_config(
    source: Mapping[str, str],
) -> TokenReclamationConfig:
    encoded = source.get("ORBITKV_TOKEN_RECLAMATION")
    if encoded is None:
        return TokenReclamationConfig()
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid ORBITKV_TOKEN_RECLAMATION JSON: {error}") from error
    value = _mapping(raw, "ORBITKV_TOKEN_RECLAMATION")
    fields = {
        "mode",
        "trigger_tokens",
        "retained_per_page",
        "policy_id",
        "policy_version",
        "quality_contract",
        "fragmentation_threshold_milli",
        "maximum_source_pages",
        "evacuation_headroom_pages",
    }
    _exact_keys(value, "ORBITKV_TOKEN_RECLAMATION", fields)
    mode = _string(value, "mode", "ORBITKV_TOKEN_RECLAMATION")
    if mode not in ("naive", "relocate"):
        raise ValueError(
            "ORBITKV_TOKEN_RECLAMATION.mode must be 'naive' or 'relocate'"
        )
    values = {
        name: _positive_int(value, name, "ORBITKV_TOKEN_RECLAMATION")
        for name in fields - {"mode", "fragmentation_threshold_milli"}
    }
    threshold = value.get("fragmentation_threshold_milli")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 1000:
        raise ValueError(
            "ORBITKV_TOKEN_RECLAMATION.fragmentation_threshold_milli must be in [0, 1000]"
        )
    retained = values["retained_per_page"]
    if retained >= PAGE_TOKENS:
        raise ValueError(
            f"ORBITKV_TOKEN_RECLAMATION.retained_per_page must be below {PAGE_TOKENS}"
        )
    if mode == "relocate":
        trigger = values["trigger_tokens"]
        source_pages = _ceil_div(trigger, PAGE_TOKENS)
        retained_tokens = (
            trigger // PAGE_TOKENS * retained
            + min(trigger % PAGE_TOKENS, retained)
        )
        retained_pages = _ceil_div(retained_tokens, PAGE_TOKENS)
        maximum_source_pages = values["maximum_source_pages"]
        headroom_pages = values["evacuation_headroom_pages"]
        if source_pages > maximum_source_pages:
            raise ValueError(
                "ORBITKV_TOKEN_RECLAMATION trigger requires "
                f"{source_pages} source pages, above maximum_source_pages="
                f"{maximum_source_pages}"
            )
        if retained_pages > headroom_pages:
            raise ValueError(
                "ORBITKV_TOKEN_RECLAMATION retained tokens require "
                f"{retained_pages} evacuation headroom pages, above "
                f"evacuation_headroom_pages={headroom_pages}"
            )
        if source_pages <= retained_pages:
            raise ValueError(
                "ORBITKV_TOKEN_RECLAMATION relocate mode must project a "
                "positive page gain"
            )
        slots = source_pages * PAGE_TOKENS
        expected_fragmentation = (slots - retained_tokens) * 1000 // slots
        if threshold > expected_fragmentation:
            raise ValueError(
                "ORBITKV_TOKEN_RECLAMATION.fragmentation_threshold_milli "
                f"must not exceed expected fragmentation {expected_fragmentation}"
            )
    return TokenReclamationConfig(
        mode=mode,
        fragmentation_threshold_milli=threshold,
        **values,
    )


def _configured_file(environ: Mapping[str, str], name: str) -> Path:
    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"invalid {name} path {path}: {error}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must name a regular file")
    return resolved


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")


def _string(value: Mapping[str, Any], key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return item


def _positive_int(value: Mapping[str, Any], key: str, path: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer")
    return item


def _positive_runtime_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_manager_plan_json(encoded: bytes) -> bytes:
    """Enforce the native manager's inclusive canonical-plan byte limit."""

    if len(encoded) > MANAGER_PLAN_JSON_MAX_BYTES:
        raise ValueError(
            "canonical manager plan JSON exceeds the "
            f"{MANAGER_PLAN_JSON_MAX_BYTES}-byte (1 MiB) native limit"
        )
    return encoded


# runtime_manifest defines these before importing the shared config types and
# helpers above, so both module import orders remain safe.
from .runtime_manifest import (  # noqa: E402
    RUNTIME_MANIFEST_CAPABILITIES as RUNTIME_MANIFEST_CAPABILITIES,
    RUNTIME_MANIFEST_MAX_BYTES as RUNTIME_MANIFEST_MAX_BYTES,
    RUNTIME_MANIFEST_SCHEMA as RUNTIME_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_VERSION as RUNTIME_MANIFEST_VERSION,
)
_EXPLICIT_MANIFEST_EXPORTS = (
    RUNTIME_MANIFEST_CAPABILITIES,
    RUNTIME_MANIFEST_MAX_BYTES,
    RUNTIME_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_VERSION,
)


__all__ = ["RuntimeConfig"]
