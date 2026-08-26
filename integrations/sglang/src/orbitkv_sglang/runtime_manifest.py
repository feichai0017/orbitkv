from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from typing import Any, Mapping


RUNTIME_MANIFEST_SCHEMA = "orbitkv.runtime-manifest"
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
RUNTIME_MANIFEST_CAPABILITIES = frozenset(
    {
        "append_only_addressing",
        "convolution_state",
        "fixed_state_checkpoints",
        "periodic_addressing",
        "recurrent_state",
        "semantic_retirement",
        "token_component_geometry",
        "token_manager",
    }
)
_LEGACY_PLAN_ENVIRONMENT = (
    "ORBITKV_PLAN",
    "ORBITKV_STATE_PLAN",
)
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1

# This import intentionally follows the constants above.  It lets this module be
# imported directly while config.py re-exports the public manifest constants
# after defining the shared dataclasses and pure parsing helpers.
from .config import (  # noqa: E402
    PAGE_TOKENS,
    ClassConfig,
    FixedStateConfig,
    ManagerPlanConfig,
    _canonical_json,
    _ceil_div,
    _class_config,
    _configured_file,
    _exact_keys,
    _mapping,
    _object_without_duplicate_keys,
    _positive_int,
    _reject_non_finite_number,
    _state_layers,
    _string,
    _token_reclamation_config,
    _validate_token_reclamation,
)


def load_runtime_manifest(source: Mapping[str, str]) -> ManagerPlanConfig:
    mixed = [name for name in _LEGACY_PLAN_ENVIRONMENT if name in source]
    if mixed:
        raise ValueError(
            "ORBITKV_RUNTIME_MANIFEST cannot be combined with legacy "
            + ", ".join(mixed)
        )

    manifest_path = _configured_file(source, "ORBITKV_RUNTIME_MANIFEST")
    library_path = _configured_file(source, "ORBITKV_LIBRARY")
    try:
        with manifest_path.open("rb") as source_file:
            encoded = source_file.read(RUNTIME_MANIFEST_MAX_BYTES + 1)
    except OSError as error:
        raise ValueError(
            f"cannot read ORBITKV_RUNTIME_MANIFEST {manifest_path}: {error}"
        ) from error
    if len(encoded) > RUNTIME_MANIFEST_MAX_BYTES:
        raise ValueError(
            "ORBITKV_RUNTIME_MANIFEST exceeds the 16777216-byte limit"
        )
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid ORBITKV_RUNTIME_MANIFEST JSON: {error}") from error

    root = _mapping(raw, "RuntimeManifest")
    _exact_keys(
        root,
        "RuntimeManifest",
        {
            "schema",
            "version",
            "fingerprint",
            "token_manager_plan",
            "attention_state_plan",
            "capability_requirements",
        },
    )
    if root.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise ValueError(
            f"RuntimeManifest.schema must be {RUNTIME_MANIFEST_SCHEMA!r}"
        )
    version = root.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != RUNTIME_MANIFEST_VERSION
    ):
        raise ValueError(
            f"RuntimeManifest.version must be {RUNTIME_MANIFEST_VERSION}"
        )
    fingerprint = _sha256_fingerprint(root, "fingerprint", "RuntimeManifest")
    payload = {key: value for key, value in root.items() if key != "fingerprint"}
    try:
        expected_fingerprint = "sha256:" + hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest()
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError(
            f"invalid ORBITKV_RUNTIME_MANIFEST JSON: {error}"
        ) from error
    if fingerprint != expected_fingerprint:
        raise ValueError("RuntimeManifest.fingerprint does not match its payload")

    requirements = _manifest_capability_requirements(
        root.get("capability_requirements")
    )
    state_plan = _compiled_attention_state_plan(root.get("attention_state_plan"))
    token_manager = root.get("token_manager_plan")
    if token_manager is None:
        # Rust permits a fixed-state-only artifact.  ABI8 SGLang still needs a
        # token manager and must reject the artifact before allocating pools.
        derived = _derive_manifest_capabilities((), state_plan[1])
        _require_exact_capabilities(requirements, derived)
        raise ValueError(
            "OrbitKV SGLang requires RuntimeManifest.token_manager_plan"
        )

    manager = _mapping(token_manager, "RuntimeManifest.token_manager_plan")
    _exact_keys(
        manager, "RuntimeManifest.token_manager_plan", {"input", "layout"}
    )
    manager_input = _mapping(
        manager.get("input"), "RuntimeManifest.token_manager_plan.input"
    )
    page_tokens, classes = _manifest_manager_input(manager_input)
    classes = _validate_manifest_layout(
        manager.get("layout"), manager_input, classes
    )
    fixed_states, token_projection = state_plan[1], state_plan[2]
    _validate_manifest_state_projection(
        page_tokens, classes, token_projection, fixed_states
    )
    derived = _derive_manifest_capabilities(classes, fixed_states)
    _require_exact_capabilities(requirements, derived)
    token_reclamation = _token_reclamation_config(source)
    retentions = tuple(item.retention for item in classes)
    _validate_token_reclamation(token_reclamation, retentions, classes)

    canonical_input = _canonical_json(manager_input)
    manager_fingerprint = "sha256:" + hashlib.sha256(canonical_input).hexdigest()
    state_path = manifest_path if fixed_states else None
    state_fingerprint = fingerprint if fixed_states else None
    return ManagerPlanConfig(
        plan_path=manifest_path,
        library_path=library_path,
        plan_json=canonical_input,
        # Keep the token-manager identity stable across legacy and manifest
        # loading.  The envelope fingerprint remains available separately and
        # may change when fixed-state contracts or capabilities change.
        plan_fingerprint=manager_fingerprint,
        page_tokens=page_tokens,
        classes=classes,
        token_reclamation=token_reclamation,
        fixed_states=fixed_states,
        state_plan_path=state_path,
        state_plan_fingerprint=state_fingerprint,
        runtime_manifest_path=manifest_path,
        runtime_manifest_fingerprint=fingerprint,
        capability_requirements=requirements,
    )


def _manifest_manager_input(
    root: Mapping[str, Any],
) -> tuple[int, tuple[ClassConfig, ...]]:
    path = "RuntimeManifest.token_manager_plan.input"
    _exact_keys(root, path, {"page_tokens", "classes"})
    page_tokens = _bounded_positive_int(root, "page_tokens", path, _U64_MAX)
    if page_tokens != PAGE_TOKENS:
        raise ValueError(f"OrbitKV SGLang requires page_tokens={PAGE_TOKENS}")
    raw_classes = root.get("classes")
    if not isinstance(raw_classes, list) or not 1 <= len(raw_classes) <= 2:
        raise ValueError("OrbitKV SGLang requires one or two KV classes")
    for index, raw_class in enumerate(raw_classes):
        value = _mapping(
            raw_class, f"RuntimeManifest.token_manager_plan.input.classes[{index}]"
        )
        if value.get("storage", "token_kv") == "token_kv" and (
            "storage" in value or value.get("components") == []
        ):
            raise ValueError(
                "RuntimeManifest token_kv input must omit default storage and "
                "empty components"
            )
    classes = tuple(
        _class_config(index, value, page_tokens, allow_compiler_name=True)
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
    return page_tokens, classes


def _validate_manifest_layout(
    raw: Any, manager_input: Mapping[str, Any], classes: tuple[ClassConfig, ...]
) -> tuple[ClassConfig, ...]:
    path = "RuntimeManifest.token_manager_plan.layout"
    layout = _mapping(raw, path)
    _exact_keys(
        layout, path, {"schema", "plan_fingerprint", "page_tokens", "classes"}
    )
    if layout.get("schema") != "orbitkv.layout-program.v1":
        raise ValueError(
            f"{path}.schema must be 'orbitkv.layout-program.v1'"
        )
    if _bounded_positive_int(layout, "page_tokens", path, _U64_MAX) != PAGE_TOKENS:
        raise ValueError("runtime manifest layout and manager input use different page sizes")
    plan_fingerprint = _sha256_fingerprint(layout, "plan_fingerprint", path)
    expected_plan_fingerprint = _compiled_plan_fingerprint(manager_input, classes)
    if plan_fingerprint != expected_plan_fingerprint:
        raise ValueError(
            "runtime manifest layout fingerprint differs from the compiled manager input"
        )
    raw_classes = layout.get("classes")
    if not isinstance(raw_classes, list) or len(raw_classes) != len(classes):
        raise ValueError(f"{path}.classes must match the manager input classes")
    admitted_classes: list[ClassConfig] = []
    for index, (raw_class, class_config) in enumerate(
        zip(raw_classes, classes, strict=True)
    ):
        class_path = f"{path}.classes[{index}]"
        value = _mapping(raw_class, class_path)
        _exact_keys(
            value,
            class_path,
            {
                "name",
                "layers",
                "bytes_per_token_per_layer",
                "address",
                "retirement",
                "minimum_slots_per_request",
            },
        )
        if _string(value, "name", class_path) != class_config.name:
            raise ValueError(f"{class_path}.name differs from the manager input")
        if _state_layers(value.get("layers"), class_path) != class_config.layers:
            raise ValueError(f"{class_path}.layers differ from the manager input")
        if (
            _bounded_positive_int(
                value, "bytes_per_token_per_layer", class_path, _U64_MAX
            )
            != class_config.bytes_per_token_per_layer
        ):
            raise ValueError(
                f"{class_path}.bytes_per_token_per_layer differs from the manager input"
            )
        address = _mapping(value.get("address"), f"{class_path}.address")
        retirement = _mapping(
            value.get("retirement"), f"{class_path}.retirement"
        )
        if class_config.retention == "full":
            _exact_keys(address, f"{class_path}.address", {"kind"})
            _exact_keys(retirement, f"{class_path}.retirement", {"kind"})
            _string(address, "kind", f"{class_path}.address")
            _string(retirement, "kind", f"{class_path}.retirement")
            expected_address = {"kind": "append_only"}
            expected_retirement = {"kind": "never"}
            expected_slots = None
            compiled_period_blocks = None
        else:
            _exact_keys(
                address, f"{class_path}.address", {"kind", "period_blocks"}
            )
            _exact_keys(
                retirement,
                f"{class_path}.retirement",
                {"kind", "offset_tokens"},
            )
            _string(address, "kind", f"{class_path}.address")
            compiled_period_blocks = _bounded_positive_int(
                address, "period_blocks", f"{class_path}.address", _U64_MAX
            )
            _string(retirement, "kind", f"{class_path}.retirement")
            _bounded_nonnegative_int(
                retirement,
                "offset_tokens",
                f"{class_path}.retirement",
                _U64_MAX,
            )
            expected_address = {
                "kind": "periodic",
                "period_blocks": class_config.period_blocks,
            }
            expected_retirement = {
                "kind": "block_end_plus",
                "offset_tokens": class_config.window_tokens - 1,
            }
            expected_slots = class_config.period_blocks
        if address != expected_address or retirement != expected_retirement:
            raise ValueError(
                f"{class_path} address or retirement program differs from the manager input"
            )
        minimum_slots = value.get("minimum_slots_per_request")
        if expected_slots is not None:
            minimum_slots = _bounded_positive_int(
                value, "minimum_slots_per_request", class_path, _U64_MAX
            )
        if minimum_slots != expected_slots:
            raise ValueError(
                f"{class_path}.minimum_slots_per_request differs from the manager input"
            )
        # After proving the compiler output is the exact projection of the
        # semantic input, retain the emitted address period as the runtime
        # source of truth.  No independently reconstructed physical value
        # survives manifest admission.
        admitted_classes.append(
            replace(class_config, period_blocks=compiled_period_blocks)
        )
    return tuple(admitted_classes)


def _compiled_plan_fingerprint(
    manager_input: Mapping[str, Any], classes: tuple[ClassConfig, ...]
) -> str:
    digest = hashlib.sha256()

    def update_u64(value: int) -> None:
        digest.update(struct.pack("<Q", value))

    def update_bytes(value: str) -> None:
        encoded = value.encode("utf-8")
        update_u64(len(encoded))
        digest.update(encoded)

    update_u64(PAGE_TOKENS)
    update_u64(len(classes))
    raw_classes = manager_input["classes"]
    for raw, item in zip(raw_classes, classes, strict=True):
        update_bytes(item.name)
        update_u64(len(item.layers))
        for layer in item.layers:
            if layer > _U32_MAX:
                raise ValueError("manager input layer exceeds the Rust u32 range")
            update_u64(layer)
        if item.bytes_per_token_per_layer > _U64_MAX:
            raise ValueError("manager input byte width exceeds the Rust u64 range")
        update_u64(item.bytes_per_token_per_layer)
        update_u64(0 if item.retention == "full" else 1)
        if item.window_tokens is not None and item.window_tokens > _U64_MAX:
            raise ValueError("manager input window exceeds the Rust u64 range")
        update_u64(item.window_tokens or 0)
        components_were_emitted = bool(raw.get("components", []))
        if item.storage != "token_kv" or components_were_emitted:
            update_u64(1)
            update_u64(0 if item.storage == "token_kv" else 1)
            update_u64(len(item.components))
            for name, byte_count in item.components:
                update_bytes(name)
                if byte_count > _U64_MAX:
                    raise ValueError(
                        "manager input component width exceeds the Rust u64 range"
                    )
                update_u64(byte_count)
        if item.period_blocks is not None and item.period_blocks > _U64_MAX:
            raise ValueError("compiled slot count exceeds the Rust u64 range")
        update_u64(item.period_blocks or 0)
    return "sha256:" + digest.hexdigest()


def _compiled_attention_state_plan(
    raw: Any,
) -> tuple[int, tuple[FixedStateConfig, ...], tuple[ClassConfig, ...]]:
    path = "RuntimeManifest.attention_state_plan"
    root = _mapping(raw, path)
    _exact_keys(root, path, {"schema", "page_tokens", "states"})
    if root.get("schema") != "orbitkv.attention-state-plan.v1":
        raise ValueError(
            f"{path}.schema must be 'orbitkv.attention-state-plan.v1'"
        )
    page_tokens = _bounded_positive_int(root, "page_tokens", path, _U64_MAX)
    if page_tokens != PAGE_TOKENS:
        raise ValueError(f"OrbitKV SGLang requires page_tokens={PAGE_TOKENS}")
    raw_states = root.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError(f"{path}.states must be a non-empty list")

    token_states: list[ClassConfig] = []
    fixed_states: list[FixedStateConfig] = []
    names: set[str] = set()
    fixed_roles: set[tuple[int, str]] = set()
    for index, raw_state in enumerate(raw_states):
        state_path = f"{path}.states[{index}]"
        state = _mapping(raw_state, state_path)
        _exact_keys(state, state_path, {"name", "layers", "backend"})
        name = _string(state, "name", state_path)
        if name in names:
            raise ValueError(f"{path} state names must be unique")
        names.add(name)
        layers = _state_layers(state.get("layers"), state_path)
        if any(layer > _U32_MAX for layer in layers):
            raise ValueError(f"{state_path}.layers exceed the Rust u32 range")
        backend_path = f"{state_path}.backend"
        backend = _mapping(state.get("backend"), backend_path)
        kind = _string(backend, "kind", backend_path)
        if kind == "token_slots":
            _exact_keys(
                backend,
                backend_path,
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
            )
            storage = _string(backend, "storage", backend_path)
            if storage not in ("token_kv", "latent_kv"):
                raise ValueError(f"{backend_path}.storage is unsupported")
            components = _compiled_components(
                backend.get("components"), backend_path, storage
            )
            byte_count = _bounded_positive_int(
                backend, "bytes_per_token_per_layer", backend_path, _U64_MAX
            )
            if sum(value for _name, value in components) != byte_count:
                raise ValueError(f"{backend_path}.component bytes do not match its width")
            page_bytes = _bounded_positive_int(
                backend, "page_bytes_per_layer", backend_path, _U64_MAX
            )
            if byte_count > _U64_MAX // page_tokens or page_bytes != byte_count * page_tokens:
                raise ValueError(f"{backend_path}.page_bytes_per_layer is not compiled geometry")
            retention = _string(backend, "retention", backend_path)
            if retention == "full":
                if backend.get("window_tokens") is not None:
                    raise ValueError(f"{backend_path}.window_tokens must be null")
                window_tokens = None
                period_blocks = None
            elif retention == "sliding":
                window_tokens = _bounded_positive_int(
                    backend, "window_tokens", backend_path, _U64_MAX
                )
                period_blocks = 1 + _ceil_div(window_tokens - 1, page_tokens)
            else:
                raise ValueError(f"{backend_path}.retention is unsupported")
            if backend.get("token_relocatable") is not True:
                raise ValueError(f"{backend_path}.token_relocatable must be true")
            token_states.append(
                ClassConfig(
                    class_id=len(token_states),
                    pool_id=len(token_states) + 1,
                    backend_domain=len(token_states) + 1,
                    name=name,
                    layers=layers,
                    retention=retention,
                    bytes_per_token_per_layer=byte_count,
                    window_tokens=window_tokens,
                    period_blocks=period_blocks,
                    storage=storage,
                    components=components,
                )
            )
            continue
        if kind == "recurrent_checkpoints":
            _exact_keys(
                backend,
                backend_path,
                {
                    "kind",
                    "family",
                    "state_bytes_per_layer",
                    "checkpoint_slots_per_request",
                    "checkpoint_bytes_per_request",
                    "token_relocatable",
                },
            )
            family = _string(backend, "family", backend_path)
            if family not in ("mamba", "gdn", "kda", "linear_attention"):
                raise ValueError(f"{backend_path}.family is unsupported")
            kernel_width = None
            fixed_kind = family
            role = "recurrent"
        elif kind == "convolution_ring":
            _exact_keys(
                backend,
                backend_path,
                {
                    "kind",
                    "state_bytes_per_layer",
                    "kernel_width",
                    "checkpoint_slots_per_request",
                    "checkpoint_bytes_per_request",
                    "token_relocatable",
                },
            )
            kernel_width = _bounded_positive_int(
                backend, "kernel_width", backend_path, _U32_MAX
            )
            fixed_kind = "convolution"
            role = "convolution"
        else:
            raise ValueError(f"{backend_path}.kind is unsupported")
        state_bytes = _bounded_positive_int(
            backend, "state_bytes_per_layer", backend_path, _U64_MAX
        )
        slots = _bounded_positive_int(
            backend, "checkpoint_slots_per_request", backend_path, _U32_MAX
        )
        if slots != 2:
            raise ValueError(
                "supported fixed-state profile requires two checkpoint slots"
            )
        checkpoint_bytes = _bounded_positive_int(
            backend, "checkpoint_bytes_per_request", backend_path, _U64_MAX
        )
        if (
            state_bytes > _U64_MAX // len(layers)
            or state_bytes * len(layers) > _U64_MAX // slots
            or checkpoint_bytes != state_bytes * len(layers) * slots
        ):
            raise ValueError(
                f"{backend_path}.checkpoint_bytes_per_request is not compiled geometry"
            )
        if backend.get("token_relocatable") is not False:
            raise ValueError(f"{backend_path}.token_relocatable must be false")
        for layer in layers:
            if (layer, role) in fixed_roles:
                raise ValueError("attention-state fixed-state roles overlap")
            fixed_roles.add((layer, role))
        fixed_states.append(
            FixedStateConfig(
                name=name,
                kind=fixed_kind,
                layers=layers,
                state_bytes_per_layer=state_bytes,
                checkpoint_slots_per_request=slots,
                kernel_width=kernel_width,
            )
        )
    return page_tokens, tuple(fixed_states), tuple(token_states)


def _compiled_components(
    raw: Any, path: str, storage: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{path}.components must contain exactly two components")
    expected = ("key", "value") if storage == "token_kv" else ("latent", "rope")
    result: list[tuple[str, int]] = []
    for index, (item, expected_name) in enumerate(zip(raw, expected, strict=True)):
        item_path = f"{path}.components[{index}]"
        component = _mapping(item, item_path)
        _exact_keys(component, item_path, {"name", "bytes_per_token_per_layer"})
        name = _string(component, "name", item_path)
        if name != expected_name:
            raise ValueError(f"{path}.components do not match {storage} storage")
        result.append(
            (
                name,
                _bounded_positive_int(
                    component, "bytes_per_token_per_layer", item_path, _U64_MAX
                ),
            )
        )
    return tuple(result)


def _validate_manifest_state_projection(
    page_tokens: int,
    classes: tuple[ClassConfig, ...],
    token_states: tuple[ClassConfig, ...],
    fixed_states: tuple[FixedStateConfig, ...],
) -> None:
    if not token_states:
        raise ValueError(
            "RuntimeManifest.token_manager_plan exists without compiled token state"
        )
    if page_tokens != PAGE_TOKENS:
        raise ValueError("runtime manifest state and manager plans use different page sizes")
    project = lambda item: (
        item.name,
        item.layers,
        item.retention,
        item.bytes_per_token_per_layer,
        item.window_tokens,
        item.storage,
        item.components,
    )
    if tuple(map(project, classes)) != tuple(map(project, token_states)):
        raise ValueError(
            "compiled attention-state token projection differs from the manager input"
        )
    claimed = [layer for item in (*classes, *fixed_states) for layer in item.layers]
    if not claimed or sorted(set(claimed)) != list(range(max(claimed) + 1)):
        raise ValueError("runtime manifest must cover every model layer")


def _manifest_capability_requirements(raw: Any) -> tuple[str, ...]:
    path = "RuntimeManifest.capability_requirements"
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"{path} must be a list of strings")
    result = tuple(raw)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{path} must be sorted and unique")
    unknown = set(result) - RUNTIME_MANIFEST_CAPABILITIES
    if unknown:
        raise ValueError(
            f"{path} contains unsupported capabilities: {', '.join(sorted(unknown))}"
        )
    return result


def _derive_manifest_capabilities(
    classes: tuple[ClassConfig, ...], fixed_states: tuple[FixedStateConfig, ...]
) -> tuple[str, ...]:
    capabilities: set[str] = set()
    if classes:
        capabilities.add("token_manager")
        if any(item.components for item in classes):
            capabilities.add("token_component_geometry")
        if any(item.retention == "full" for item in classes):
            capabilities.add("append_only_addressing")
        if any(item.retention == "sliding" for item in classes):
            capabilities.update({"periodic_addressing", "semantic_retirement"})
    if fixed_states:
        capabilities.update(
            {
                "fixed_state_checkpoints",
            }
        )
        if any(item.kind == "convolution" for item in fixed_states):
            capabilities.add("convolution_state")
        if any(item.kind != "convolution" for item in fixed_states):
            capabilities.add("recurrent_state")
    return tuple(sorted(capabilities))


def _require_exact_capabilities(
    declared: tuple[str, ...], derived: tuple[str, ...]
) -> None:
    if declared != derived:
        raise ValueError(
            "RuntimeManifest.capability_requirements differ from the compiled plans"
        )


def _bounded_positive_int(
    value: Mapping[str, Any], key: str, path: str, maximum: int
) -> int:
    item = _positive_int(value, key, path)
    if item > maximum:
        raise ValueError(f"{path}.{key} exceeds the supported integer range")
    return item


def _bounded_nonnegative_int(
    value: Mapping[str, Any], key: str, path: str, maximum: int
) -> int:
    item = value.get(key)
    if (
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item <= maximum
    ):
        raise ValueError(
            f"{path}.{key} must be a nonnegative integer in the supported range"
        )
    return item


def _sha256_fingerprint(
    value: Mapping[str, Any], key: str, path: str
) -> str:
    item = _string(value, key, path)
    if (
        len(item) != len("sha256:") + 64
        or not item.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in item[7:])
    ):
        raise ValueError(f"{path}.{key} must be a lowercase sha256 fingerprint")
    return item


__all__ = [
    "RUNTIME_MANIFEST_CAPABILITIES",
    "RUNTIME_MANIFEST_MAX_BYTES",
    "RUNTIME_MANIFEST_SCHEMA",
    "RUNTIME_MANIFEST_VERSION",
    "load_runtime_manifest",
]
