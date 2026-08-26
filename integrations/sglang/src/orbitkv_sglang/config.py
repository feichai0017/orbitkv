from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


PAGE_TOKENS = 16
RetentionKind = Literal["full", "sliding"]
TokenStorageKind = Literal["token_kv", "latent_kv"]
ReclamationMode = Literal["off", "naive", "relocate"]
FixedStateKind = Literal["mamba", "gdn", "kda", "linear_attention", "convolution"]


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
class ManagerPlanConfig:
    """The sole breaking ABI8 snapshot/prefix plan; no legacy translation."""

    plan_path: Path
    library_path: Path
    plan_json: bytes
    plan_fingerprint: str
    page_tokens: int
    classes: tuple[ClassConfig, ...]
    token_reclamation: TokenReclamationConfig = TokenReclamationConfig()
    fixed_states: tuple[FixedStateConfig, ...] = ()
    state_plan_path: Path | None = None
    state_plan_fingerprint: str | None = None
    runtime_manifest_path: Path | None = None
    runtime_manifest_fingerprint: str | None = None
    capability_requirements: tuple[str, ...] = ()

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


def load_config(environ: Mapping[str, str] | None = None) -> ManagerPlanConfig:
    """Load a v1 runtime manifest, or the legacy ABI8 plan inputs."""

    source = os.environ if environ is None else environ
    if "ORBITKV_RUNTIME_MANIFEST" in source:
        from .runtime_manifest import load_runtime_manifest

        return load_runtime_manifest(source)
    return _load_legacy_config(source)


def _load_legacy_config(source: Mapping[str, str]) -> ManagerPlanConfig:
    """Preserve the ABI8 ORBITKV_PLAN plus ORBITKV_STATE_PLAN path."""

    plan_path = _configured_file(source, "ORBITKV_PLAN")
    library_path = _configured_file(source, "ORBITKV_LIBRARY")
    try:
        plan_json = plan_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read ORBITKV_PLAN {plan_path}: {error}") from error
    try:
        raw = json.loads(
            plan_json,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid ORBITKV_PLAN JSON: {error}") from error

    root = _mapping(raw, "KvPlanInput")
    _exact_keys(root, "KvPlanInput", {"page_tokens", "classes"})
    page_tokens = _positive_int(root, "page_tokens", "KvPlanInput")
    if page_tokens != PAGE_TOKENS:
        raise ValueError(f"OrbitKV SGLang requires page_tokens={PAGE_TOKENS}")

    raw_classes = root.get("classes")
    if not isinstance(raw_classes, list) or not 1 <= len(raw_classes) <= 2:
        raise ValueError("OrbitKV SGLang requires one or two KV classes")
    classes = tuple(
        _class_config(
            index,
            value,
            page_tokens,
            allow_compiler_name=source.get("ORBITKV_STATE_PLAN") is not None,
        )
        for index, value in enumerate(raw_classes)
    )
    retentions = tuple(item.retention for item in classes)
    if retentions not in (("full",), ("sliding",), ("full", "sliding")):
        raise ValueError(
            "KV classes must be Full, sliding, or ordered Full then sliding"
        )
    storage = tuple(item.storage for item in classes)
    if "latent_kv" in storage and (
        len(classes) != 1 or retentions != ("full",) or storage != ("latent_kv",)
    ):
        raise ValueError(
            "supported SGLang MLA profile requires one Full latent_kv class"
        )

    layers = [layer for item in classes for layer in item.layers]
    if len(set(layers)) != len(layers):
        raise ValueError("KV classes overlap in model-layer ownership")
    if source.get("ORBITKV_STATE_PLAN") is None and sorted(layers) != list(
        range(len(layers))
    ):
        raise ValueError("KV classes must cover every model layer exactly once")

    canonical = _canonical_json(root)
    token_reclamation = _token_reclamation_config(source)
    fixed_states, state_plan_path, state_plan_fingerprint = _fixed_state_config(
        source, page_tokens, classes
    )
    _validate_token_reclamation(token_reclamation, retentions, classes)
    return ManagerPlanConfig(
        plan_path=plan_path,
        library_path=library_path,
        plan_json=canonical,
        plan_fingerprint="sha256:" + hashlib.sha256(canonical).hexdigest(),
        page_tokens=page_tokens,
        classes=classes,
        token_reclamation=token_reclamation,
        fixed_states=fixed_states,
        state_plan_path=state_plan_path,
        state_plan_fingerprint=state_plan_fingerprint,
    )


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


def _fixed_state_config(
    source: Mapping[str, str],
    page_tokens: int,
    classes: tuple[ClassConfig, ...],
) -> tuple[tuple[FixedStateConfig, ...], Path | None, str | None]:
    encoded_path = source.get("ORBITKV_STATE_PLAN")
    if encoded_path is None:
        return (), None, None
    path = _configured_file(source, "ORBITKV_STATE_PLAN")
    try:
        raw = json.loads(
            path.read_bytes(),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid ORBITKV_STATE_PLAN JSON: {error}") from error
    root = _mapping(raw, "AttentionStatePlan")
    _exact_keys(root, "AttentionStatePlan", {"page_tokens", "states"})
    if _positive_int(root, "page_tokens", "AttentionStatePlan") != page_tokens:
        raise ValueError("attention-state and manager plans use different page sizes")
    raw_states = root.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError("AttentionStatePlan.states must be a non-empty list")
    token_states: list[
        tuple[
            str,
            tuple[int, ...],
            str,
            int,
            int | None,
            str,
            tuple[tuple[str, int], ...],
        ]
    ] = []
    fixed: list[FixedStateConfig] = []
    for index, raw_state in enumerate(raw_states):
        state_path = f"AttentionStatePlan.states[{index}]"
        value = _mapping(raw_state, state_path)
        _exact_keys(value, state_path, {"name", "layers", "storage"})
        name = _string(value, "name", state_path)
        layers = _state_layers(value.get("layers"), state_path)
        storage = _mapping(value.get("storage"), f"{state_path}.storage")
        kind = _string(storage, "kind", f"{state_path}.storage")
        if kind in ("token_kv", "latent_kv"):
            expected_fields = (
                {
                    "kind",
                    "key_bytes_per_token_per_layer",
                    "value_bytes_per_token_per_layer",
                    "retention",
                    "window_tokens",
                }
                if kind == "token_kv"
                else {
                    "kind",
                    "latent_bytes_per_token_per_layer",
                    "rope_bytes_per_token_per_layer",
                    "retention",
                    "window_tokens",
                }
            )
            _exact_keys(storage, f"{state_path}.storage", expected_fields)
            first, second = (
                ("key_bytes_per_token_per_layer", "value_bytes_per_token_per_layer")
                if kind == "token_kv"
                else ("latent_bytes_per_token_per_layer", "rope_bytes_per_token_per_layer")
            )
            byte_count = _positive_int(storage, first, state_path) + _positive_int(
                storage, second, state_path
            )
            component_names = ("key", "value") if kind == "token_kv" else (
                "latent",
                "rope",
            )
            components = (
                (component_names[0], _positive_int(storage, first, state_path)),
                (component_names[1], _positive_int(storage, second, state_path)),
            )
            retention = _string(storage, "retention", state_path)
            window = storage.get("window_tokens")
            if retention == "full":
                if window is not None:
                    raise ValueError(
                        f"{state_path}.storage.window_tokens must be null"
                    )
            elif retention == "sliding":
                window = _positive_int(storage, "window_tokens", state_path)
            else:
                raise ValueError(f"{state_path}.storage.retention is unsupported")
            token_states.append(
                (name, layers, retention, byte_count, window, kind, components)
            )
            continue
        if kind == "recurrent":
            _exact_keys(
                storage,
                f"{state_path}.storage",
                {
                    "kind",
                    "family",
                    "state_bytes_per_layer",
                    "checkpoint_slots_per_request",
                },
            )
            family = _string(storage, "family", state_path)
            if family not in ("mamba", "gdn", "kda", "linear_attention"):
                raise ValueError(f"{state_path}.storage.family is unsupported")
            fixed.append(
                FixedStateConfig(
                    name,
                    family,
                    layers,
                    _positive_int(storage, "state_bytes_per_layer", state_path),
                    _positive_int(storage, "checkpoint_slots_per_request", state_path),
                )
            )
            continue
        if kind == "convolution":
            _exact_keys(
                storage,
                f"{state_path}.storage",
                {
                    "kind",
                    "state_bytes_per_layer",
                    "kernel_width",
                    "checkpoint_slots_per_request",
                },
            )
            fixed.append(
                FixedStateConfig(
                    name,
                    kind,
                    layers,
                    _positive_int(storage, "state_bytes_per_layer", state_path),
                    _positive_int(storage, "checkpoint_slots_per_request", state_path),
                    _positive_int(storage, "kernel_width", state_path),
                )
            )
            continue
        raise ValueError(f"{state_path}.storage.kind is unsupported")
    projected = sorted(
        (
            item.name,
            item.layers,
            item.retention,
            item.bytes_per_token_per_layer,
            item.window_tokens,
            item.storage,
            item.components,
        )
        for item in classes
    )
    state_projection = sorted(
        (name, layers, retention, byte_count, window, kind, components)
        for name, layers, retention, byte_count, window, kind, components in token_states
    )
    if state_projection != projected:
        raise ValueError("attention-state token projection differs from ORBITKV_PLAN")
    all_names = [name for name, *_rest in token_states] + [item.name for item in fixed]
    if len(set(all_names)) != len(all_names):
        raise ValueError("attention-state names must be unique")
    if not fixed:
        raise ValueError("ORBITKV_STATE_PLAN contains no fixed-width state")
    if any(item.checkpoint_slots_per_request != 2 for item in fixed):
        raise ValueError("supported fixed-state profile requires two checkpoint slots")
    claimed = [layer for item in (*classes, *fixed) for layer in item.layers]
    if not claimed or sorted(set(claimed)) != list(range(max(claimed) + 1)):
        raise ValueError("attention-state plan must cover every model layer")
    roles: set[tuple[int, str]] = set()
    for item in fixed:
        role = "convolution" if item.kind == "convolution" else "recurrent"
        for layer in item.layers:
            if (layer, role) in roles:
                raise ValueError("attention-state fixed-state roles overlap")
            roles.add((layer, role))
    canonical = _canonical_json(root)
    return (
        tuple(fixed),
        path,
        "sha256:" + hashlib.sha256(canonical).hexdigest(),
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


def _class_config(
    index: int,
    raw: Any,
    page_tokens: int,
    *,
    allow_compiler_name: bool = False,
) -> ClassConfig:
    path = f"KvPlanInput.classes[{index}]"
    value = _mapping(raw, path)
    base_fields = {
        "name",
        "layers",
        "retention",
        "bytes_per_token_per_layer",
        "window_tokens",
    }
    optional_fields = {"storage", "components"}
    retention = _string(value, "retention", path)
    if retention == "full":
        _exact_keys(value, path, base_fields, optional_fields)
        if value.get("window_tokens") is not None:
            raise ValueError(f"{path}.window_tokens must be null for full retention")
        window_tokens = None
        period_blocks = None
    elif retention == "sliding":
        _exact_keys(value, path, base_fields | {"window_tokens"}, optional_fields)
        window_tokens = _positive_int(value, "window_tokens", path)
        period_blocks = 1 + _ceil_div(window_tokens - 1, page_tokens)
    else:
        raise ValueError(f"{path}.retention must be 'full' or 'sliding'")
    name = _string(value, "name", path)
    storage = value.get("storage", "token_kv")
    if storage not in ("token_kv", "latent_kv"):
        raise ValueError(f"{path}.storage must be 'token_kv' or 'latent_kv'")
    expected_name = "full" if retention == "full" else "swa"
    if storage == "token_kv" and name != expected_name and not allow_compiler_name:
        raise ValueError(
            f"{path}.name must be {expected_name!r} for {retention} token_kv"
        )
    components = _token_components(value, path, storage)

    raw_layers = value.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError(f"{path}.layers must be a non-empty list")
    layers: list[int] = []
    for layer_index, layer in enumerate(raw_layers):
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(
                f"{path}.layers[{layer_index}] must be a nonnegative integer"
            )
        layers.append(layer)
    if layers != sorted(set(layers)):
        raise ValueError(f"{path}.layers must be unique and ascending")

    return ClassConfig(
        class_id=index,
        pool_id=index + 1,
        backend_domain=index + 1,
        name=name,
        layers=tuple(layers),
        retention=retention,
        bytes_per_token_per_layer=_positive_int(
            value, "bytes_per_token_per_layer", path
        ),
        window_tokens=window_tokens,
        period_blocks=period_blocks,
        storage=storage,
        components=components,
    )


def _token_components(
    value: Mapping[str, Any], path: str, storage: str
) -> tuple[tuple[str, int], ...]:
    raw = value.get("components", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}.components must be a list")
    if not raw:
        if storage != "token_kv":
            raise ValueError(f"{path}.latent_kv requires latent and rope components")
        return ()
    expected = ("key", "value") if storage == "token_kv" else ("latent", "rope")
    components: list[tuple[str, int]] = []
    for index, item in enumerate(raw):
        component_path = f"{path}.components[{index}]"
        component = _mapping(item, component_path)
        _exact_keys(
            component, component_path, {"name", "bytes_per_token_per_layer"}
        )
        components.append(
            (
                _string(component, "name", component_path),
                _positive_int(component, "bytes_per_token_per_layer", component_path),
            )
        )
    result = tuple(components)
    if tuple(name for name, _bytes in result) != expected:
        raise ValueError(f"{path}.components do not match {storage} storage")
    if sum(byte_count for _name, byte_count in result) != _positive_int(
        value, "bytes_per_token_per_layer", path
    ):
        raise ValueError(f"{path}.component bytes do not match the class width")
    return result


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


# runtime_manifest defines these before importing the shared config types and
# helpers above, so both module import orders remain safe.
from .runtime_manifest import (  # noqa: E402
    RUNTIME_MANIFEST_CAPABILITIES,
    RUNTIME_MANIFEST_MAX_BYTES,
    RUNTIME_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_VERSION,
)


__all__ = [
    "ClassConfig",
    "FixedStateConfig",
    "FixedStateKind",
    "ManagerPlanConfig",
    "PAGE_TOKENS",
    "ReclamationMode",
    "RUNTIME_MANIFEST_CAPABILITIES",
    "RUNTIME_MANIFEST_MAX_BYTES",
    "RUNTIME_MANIFEST_SCHEMA",
    "RUNTIME_MANIFEST_VERSION",
    "RetentionKind",
    "TokenReclamationConfig",
    "load_config",
]
