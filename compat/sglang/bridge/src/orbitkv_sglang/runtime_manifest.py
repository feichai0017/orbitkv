from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Mapping

from ._retention_ir import (
    derive_capability_requirements as _derive_retention_capabilities,
    validate_layout_program as _validate_layout_program,
    validate_retention_program as _validate_retention_program,
)


RUNTIME_MANIFEST_SCHEMA = "orbitkv.runtime-manifest"
RUNTIME_MANIFEST_VERSION = 3
RUNTIME_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
RETENTION_PROGRAM_SCHEMA = "orbitkv.retention-ir.v1"
LAYOUT_PROGRAM_SCHEMA = "orbitkv.layout-program.v1"
RUNTIME_MANIFEST_CAPABILITIES = frozenset(
    {
        "append_only_addressing",
        "block_domain_partitioning",
        "convolution_state",
        "fixed_state_checkpoints",
        "kv_head_partitioning",
        "periodic_addressing",
        "periodic_from_addressing",
        "pinned_addressing",
        "recurrent_state",
        "resettable_arena_addressing",
        "semantic_retirement",
        "token_component_geometry",
        "token_manager",
    }
)
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1

# This import intentionally follows the constants above.  It lets this module be
# imported directly while config.py re-exports the public manifest constants
# after defining the shared dataclasses and pure parsing helpers.
from .config import (  # noqa: E402
    PAGE_TOKENS,
    ClassConfig,
    FixedStateConfig,
    RuntimeConfig,
    _canonical_json,
    _ceil_div,
    _configured_file,
    _exact_keys,
    _mapping,
    _object_without_duplicate_keys,
    _positive_int,
    _reject_non_finite_number,
    _state_layers,
    _string,
    _token_reclamation_config,
    _validate_manager_plan_json,
    _validate_token_reclamation,
)


@dataclass(frozen=True, slots=True)
class _AttentionProjection:
    manager_input: Mapping[str, Any] | None
    classes: tuple[ClassConfig, ...]
    fixed_states: tuple[FixedStateConfig, ...]
    compiled_states: tuple[Mapping[str, Any], ...]


def load_runtime_manifest(source: Mapping[str, str]) -> RuntimeConfig:
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

    root = validate_runtime_manifest(raw)
    return _config_from_manifest(source, manifest_path, library_path, root)


def validate_runtime_manifest(raw: Any) -> dict[str, Any]:
    """Validate and recompile the canonical runtime-manifest envelope."""

    root = _mapping(raw, "RuntimeManifest")
    _exact_keys(
        root,
        "RuntimeManifest",
        {
            "schema",
            "version",
            "fingerprint",
            "source",
            "token_manager_plan",
            "attention_state_plan",
            "capability_requirements",
        },
    )
    if root.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise ValueError(f"RuntimeManifest.schema must be {RUNTIME_MANIFEST_SCHEMA!r}")
    version = root.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != RUNTIME_MANIFEST_VERSION
    ):
        raise ValueError(f"RuntimeManifest.version must be {RUNTIME_MANIFEST_VERSION}")
    fingerprint = _sha256_fingerprint(root, "fingerprint", "RuntimeManifest")
    payload = {key: value for key, value in root.items() if key != "fingerprint"}
    try:
        expected = "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"invalid RuntimeManifest JSON: {error}") from error
    if fingerprint != expected:
        raise ValueError("RuntimeManifest.fingerprint does not match its payload")

    source = _mapping(root.get("source"), "RuntimeManifest.source")
    kind = _string(source, "kind", "RuntimeManifest.source")
    if kind == "attention_state":
        _exact_keys(source, "RuntimeManifest.source", {"kind", "input"})
        projection = _compile_attention_source(source.get("input"))
        expected_attention: Mapping[str, Any] | None = {
            "schema": "orbitkv.attention-state-plan.v1",
            "page_tokens": _mapping(source.get("input"), "RuntimeManifest.source.input").get("page_tokens"),
            "states": list(projection.compiled_states),
        }
        if root.get("attention_state_plan") != expected_attention:
            raise ValueError(
                "RuntimeManifest.attention_state_plan differs from its attention-state source"
            )
        expected_layout = None
        if projection.manager_input is not None:
            expected_layout = _mapping(
                root.get("token_manager_plan"),
                "RuntimeManifest.token_manager_plan",
            )
            _exact_keys(
                expected_layout, "RuntimeManifest.token_manager_plan", {"layout"}
            )
            _validate_manifest_layout(
                expected_layout.get("layout"),
                projection.manager_input,
                projection.classes,
            )
        elif root.get("token_manager_plan") is not None:
            raise ValueError(
                "RuntimeManifest.token_manager_plan must be null for fixed-state-only input"
            )
        derived = _derive_manifest_capabilities(
            projection.classes, projection.fixed_states
        )
    elif kind == "retention_ir":
        _exact_keys(source, "RuntimeManifest.source", {"kind", "program"})
        if root.get("attention_state_plan") is not None:
            raise ValueError(
                "RuntimeManifest.attention_state_plan must be null for Retention IR"
            )
        program = _validate_retention_program(source.get("program"))
        manager = _mapping(
            root.get("token_manager_plan"), "RuntimeManifest.token_manager_plan"
        )
        _exact_keys(manager, "RuntimeManifest.token_manager_plan", {"layout"})
        layout = _validate_layout_program(manager.get("layout"), program)
        derived = _derive_retention_capabilities(layout)
    else:
        raise ValueError("RuntimeManifest.source.kind is unsupported")

    requirements = _manifest_capability_requirements(
        root.get("capability_requirements")
    )
    _require_exact_capabilities(requirements, derived)
    return root


def _config_from_manifest(
    source: Mapping[str, str], manifest_path, library_path, root: Mapping[str, Any]
) -> RuntimeConfig:
    manifest_source = _mapping(root.get("source"), "RuntimeManifest.source")
    kind = manifest_source.get("kind")
    if kind == "attention_state":
        projection = _compile_attention_source(manifest_source.get("input"))
        if projection.manager_input is None:
            raise ValueError(
                "OrbitKV SGLang requires RuntimeManifest.token_manager_plan"
            )
        plan = projection.manager_input
        plan_format = "kv_plan"
        classes = projection.classes
        fixed_states = projection.fixed_states
    else:
        plan = _mapping(manifest_source.get("program"), "RuntimeManifest.source.program")
        plan_format = "retention_ir"
        classes = _extract_runtime_classes(root)
        fixed_states = ()

    encoded_plan = _validate_manager_plan_json(_canonical_json(plan))
    from .runtime_admission import runtime_binding_from_manifest

    runtime_binding = runtime_binding_from_manifest(root)
    requirements = tuple(root["capability_requirements"])
    token_reclamation = _token_reclamation_config(source)
    _validate_token_reclamation(
        token_reclamation, tuple(item.retention for item in classes), classes
    )
    fingerprint = _sha256_fingerprint(root, "fingerprint", "RuntimeManifest")
    return RuntimeConfig(
        library_path=library_path,
        plan_json=encoded_plan,
        plan_fingerprint="sha256:" + hashlib.sha256(encoded_plan).hexdigest(),
        page_tokens=PAGE_TOKENS,
        classes=classes,
        token_reclamation=token_reclamation,
        fixed_states=fixed_states,
        runtime_manifest_path=manifest_path,
        runtime_manifest_fingerprint=fingerprint,
        capability_requirements=requirements,
        execution_signature=runtime_binding["execution_signature"],
        runtime_binding=runtime_binding,
        manager_plan_format=plan_format,
    )


def _compile_attention_source(raw: Any) -> _AttentionProjection:
    path = "RuntimeManifest.source.input"
    source = _mapping(raw, path)
    _exact_keys(source, path, {"page_tokens", "states"})
    page_tokens = _bounded_positive_int(source, "page_tokens", path, _U64_MAX)
    if page_tokens != PAGE_TOKENS:
        raise ValueError(f"OrbitKV SGLang requires page_tokens={PAGE_TOKENS}")
    raw_states = source.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError(f"{path}.states must be a non-empty list")

    names: set[str] = set()
    roles: set[tuple[int, str]] = set()
    classes: list[ClassConfig] = []
    fixed_states: list[FixedStateConfig] = []
    compiled_states: list[Mapping[str, Any]] = []
    manager_classes: list[Mapping[str, Any]] = []
    for index, raw_state in enumerate(raw_states):
        state_path = f"{path}.states[{index}]"
        state = _mapping(raw_state, state_path)
        _exact_keys(state, state_path, {"name", "layers", "storage"})
        name = _string(state, "name", state_path)
        if name in names:
            raise ValueError(f"{path} state names must be unique")
        names.add(name)
        layers = _state_layers(state.get("layers"), state_path)
        if any(layer > _U32_MAX for layer in layers):
            raise ValueError(f"{state_path}.layers exceed the Rust u32 range")
        storage_path = f"{state_path}.storage"
        storage = _mapping(state.get("storage"), storage_path)
        kind = _string(storage, "kind", storage_path)

        if kind in ("token_kv", "latent_kv"):
            fields = (
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
            _exact_keys(storage, storage_path, fields)
            byte_fields = (
                ("key_bytes_per_token_per_layer", "value_bytes_per_token_per_layer")
                if kind == "token_kv"
                else ("latent_bytes_per_token_per_layer", "rope_bytes_per_token_per_layer")
            )
            component_names = ("key", "value") if kind == "token_kv" else ("latent", "rope")
            component_values = tuple(
                _bounded_positive_int(storage, field, storage_path, _U64_MAX)
                for field in byte_fields
            )
            byte_count = component_values[0] + component_values[1]
            if byte_count > _U64_MAX or byte_count > _U64_MAX // page_tokens:
                raise ValueError(f"{storage_path} token geometry overflows")
            retention = _string(storage, "retention", storage_path)
            window = storage.get("window_tokens")
            if retention == "full" and window is None:
                period_blocks = None
            elif retention == "sliding":
                window = _bounded_positive_int(
                    storage, "window_tokens", storage_path, _U64_MAX
                )
                period_blocks = 1 + _ceil_div(window - 1, page_tokens)
            else:
                raise ValueError(f"{storage_path} retention/window contract is invalid")
            for layer in layers:
                if (layer, "token_addressable") in roles:
                    raise ValueError("attention-state token roles overlap")
                roles.add((layer, "token_addressable"))
            components = tuple(zip(component_names, component_values, strict=True))
            class_config = ClassConfig(
                class_id=len(classes),
                pool_id=len(classes) + 1,
                backend_domain=len(classes) + 1,
                name=name,
                layers=layers,
                retention=retention,
                bytes_per_token_per_layer=byte_count,
                window_tokens=window,
                period_blocks=period_blocks,
                storage=kind,
                components=components,
            )
            classes.append(class_config)
            manager_class: dict[str, Any] = {
                "name": name,
                "layers": list(layers),
                "retention": retention,
                "bytes_per_token_per_layer": byte_count,
                "window_tokens": window,
                "components": [
                    {
                        "name": component_name,
                        "bytes_per_token_per_layer": component_bytes,
                    }
                    for component_name, component_bytes in components
                ],
            }
            if kind != "token_kv":
                manager_class["storage"] = kind
            manager_classes.append(manager_class)
            compiled_states.append(
                {
                    "name": name,
                    "layers": list(layers),
                    "backend": {
                        "kind": "token_slots",
                        "storage": kind,
                        "components": manager_class["components"],
                        "bytes_per_token_per_layer": byte_count,
                        "page_bytes_per_layer": byte_count * page_tokens,
                        "retention": retention,
                        "window_tokens": window,
                        "token_relocatable": True,
                    },
                }
            )
            continue

        if kind == "recurrent":
            _exact_keys(
                storage,
                storage_path,
                {
                    "kind",
                    "family",
                    "state_bytes_per_layer",
                    "checkpoint_slots_per_request",
                },
            )
            family = _string(storage, "family", storage_path)
            if family not in ("mamba", "gdn", "kda", "linear_attention"):
                raise ValueError(f"{storage_path}.family is unsupported")
            fixed_kind = family
            role = "recurrent"
            kernel_width = None
            backend_kind = "recurrent_checkpoints"
        elif kind == "convolution":
            _exact_keys(
                storage,
                storage_path,
                {
                    "kind",
                    "state_bytes_per_layer",
                    "kernel_width",
                    "checkpoint_slots_per_request",
                },
            )
            family = None
            fixed_kind = "convolution"
            role = "convolution"
            kernel_width = _bounded_positive_int(
                storage, "kernel_width", storage_path, _U32_MAX
            )
            backend_kind = "convolution_ring"
        else:
            raise ValueError(f"{storage_path}.kind is unsupported")
        state_bytes = _bounded_positive_int(
            storage, "state_bytes_per_layer", storage_path, _U64_MAX
        )
        slots = _bounded_positive_int(
            storage, "checkpoint_slots_per_request", storage_path, _U32_MAX
        )
        if slots < 2:
            raise ValueError(f"{storage_path} needs at least two checkpoint slots")
        if state_bytes > _U64_MAX // len(layers):
            raise ValueError(f"{storage_path} checkpoint geometry overflows")
        checkpoint_bytes = state_bytes * len(layers)
        if checkpoint_bytes > _U64_MAX // slots:
            raise ValueError(f"{storage_path} checkpoint geometry overflows")
        checkpoint_bytes *= slots
        for layer in layers:
            if (layer, role) in roles:
                raise ValueError("attention-state fixed-state roles overlap")
            roles.add((layer, role))
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
        backend: dict[str, Any] = {
            "kind": backend_kind,
            "state_bytes_per_layer": state_bytes,
            "checkpoint_slots_per_request": slots,
            "checkpoint_bytes_per_request": checkpoint_bytes,
            "token_relocatable": False,
        }
        if family is not None:
            backend["family"] = family
        else:
            backend["kernel_width"] = kernel_width
        compiled_states.append(
            {"name": name, "layers": list(layers), "backend": backend}
        )

    manager_input = (
        {"page_tokens": page_tokens, "classes": manager_classes}
        if manager_classes
        else None
    )
    claimed = [layer for item in (*classes, *fixed_states) for layer in item.layers]
    if not claimed or sorted(set(claimed)) != list(range(max(claimed) + 1)):
        raise ValueError("RuntimeManifest attention state must cover every model layer")
    return _AttentionProjection(
        manager_input, tuple(classes), tuple(fixed_states), tuple(compiled_states)
    )


def _extract_runtime_classes(
    root: Mapping[str, Any],
) -> tuple[ClassConfig, ...]:
    """Project validated Retention IR layouts into runtime arena metadata."""

    source = _mapping(root.get("source"), "RuntimeManifest.source")
    program = _mapping(source.get("program"), "RuntimeManifest.source.program")
    manager = _mapping(
        root.get("token_manager_plan"), "RuntimeManifest.token_manager_plan"
    )
    layout = _mapping(
        manager.get("layout"), "RuntimeManifest.token_manager_plan.layout"
    )
    page_tokens = _bounded_positive_int(
        program, "page_tokens", "RuntimeManifest.source.program", _U64_MAX
    )
    if page_tokens != PAGE_TOKENS:
        raise ValueError(f"OrbitKV SGLang requires page_tokens={PAGE_TOKENS}")

    raw_classes = layout.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError(
            "RuntimeManifest.token_manager_plan.layout.classes must be non-empty"
        )
    classes: list[ClassConfig] = []
    for index, raw_class in enumerate(raw_classes):
        path = f"RuntimeManifest.token_manager_plan.layout.classes[{index}]"
        item = _mapping(raw_class, path)
        address = _mapping(item.get("address"), f"{path}.address")
        retirement = _mapping(item.get("retirement"), f"{path}.retirement")
        kind = address.get("kind")
        window_tokens: int | None = None
        period_blocks: int | None = None
        chunk_tokens: int | None = None
        blocks_per_epoch: int | None = None
        if kind in ("append_only", "pinned"):
            retention = "full"
        elif kind in ("periodic", "periodic_from"):
            retention = "sliding"
            period_blocks = _bounded_positive_int(
                address, "period_blocks", f"{path}.address", _U64_MAX
            )
            offset = _bounded_nonnegative_int(
                retirement, "offset_tokens", f"{path}.retirement", _U64_MAX
            )
            window_tokens = offset + 1
        elif kind == "resettable_arena":
            retention = "chunked"
            blocks_per_epoch = _bounded_positive_int(
                address, "blocks_per_epoch", f"{path}.address", _U64_MAX
            )
            if blocks_per_epoch > _U64_MAX // page_tokens:
                raise ValueError(f"{path}.address chunk geometry overflows")
            chunk_tokens = blocks_per_epoch * page_tokens
        else:
            raise ValueError(f"{path}.address.kind is unsupported")
        classes.append(
            ClassConfig(
                class_id=index,
                pool_id=index + 1,
                backend_domain=index + 1,
                name=_string(item, "name", path),
                layers=_state_layers(item.get("layers"), path),
                retention=retention,
                bytes_per_token_per_layer=_bounded_positive_int(
                    item, "bytes_per_token_per_layer", path, _U64_MAX
                ),
                window_tokens=window_tokens,
                period_blocks=period_blocks,
                chunk_tokens=chunk_tokens,
                blocks_per_epoch=blocks_per_epoch,
            )
        )
    return tuple(classes)


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
            _bounded_positive_int(
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
        admitted_classes.append(class_config)
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
