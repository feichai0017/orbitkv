"""Strict RuntimeManifest helpers for untrusted Engine E2E records."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
FULL_TOKEN_KV_TOPOLOGY = "whole_domain_full_token_kv"
FULL_SLIDING_TOKEN_KV_TOPOLOGY = "whole_domain_full_sliding_token_kv"
SLIDING_TOKEN_KV_TOPOLOGY = "whole_domain_sliding_token_kv"
FULL_LATENT_KV_TOPOLOGY = "whole_domain_full_latent_kv"
NATIVE_SESSION_CONTRACTS = {
    (("full", "token_kv"),): (FULL_TOKEN_KV_TOPOLOGY, "shared_prefix"),
    (("full", "latent_kv"),): (FULL_LATENT_KV_TOPOLOGY, "request_private"),
    (("full", "token_kv"), ("sliding", "token_kv")): (
        FULL_SLIDING_TOKEN_KV_TOPOLOGY,
        "shared_prefix",
    ),
    (("sliding", "token_kv"),): (
        SLIDING_TOKEN_KV_TOPOLOGY,
        "request_private",
    ),
}


def exact(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
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


def same(left: Any, right: Any) -> bool:
    """Compare JSON values without treating booleans as integers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            same(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def integer(
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


def bounded_positive_integer(value: Any, label: str, maximum: int) -> int:
    result = integer(value, label, positive=True)
    if result > maximum:
        raise RuntimeError(f"{label} exceeds the supported integer range")
    return result


def attention_state_layers(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must be a nonempty layer list")
    layers = [integer(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if any(item > U32_MAX for item in layers):
        raise RuntimeError(f"{label} exceeds the supported layer range")
    if layers != sorted(set(layers)):
        raise RuntimeError(f"{label} must be sorted and unique")
    return layers


def manager_input_from_attention_source(
    manifest: Mapping[str, Any], label: str
) -> dict[str, Any]:
    source = exact(manifest.get("source"), {"kind", "input"}, f"{label}.source")
    if source["kind"] != "attention_state":
        raise RuntimeError(f"{label}.source.kind must be 'attention_state'")
    attention = exact(
        source["input"], {"page_tokens", "states"}, f"{label}.source.input"
    )
    page_tokens = bounded_positive_integer(
        attention["page_tokens"], f"{label}.source.input.page_tokens", U64_MAX
    )
    if page_tokens != 16:
        raise RuntimeError(f"{label}.source.input.page_tokens must be 16")
    states = attention["states"]
    if not isinstance(states, list) or not states:
        raise RuntimeError(f"{label}.source.input.states must be nonempty")

    names: set[str] = set()
    roles: set[tuple[int, str]] = set()
    claimed_layers: list[int] = []
    classes: list[dict[str, Any]] = []
    for index, raw_state in enumerate(states):
        state_label = f"{label}.source.input.states[{index}]"
        state = exact(raw_state, {"name", "layers", "storage"}, state_label)
        name = state["name"]
        if not isinstance(name, str) or not name or name in names:
            raise RuntimeError(f"{state_label}.name must be nonempty and unique")
        names.add(name)
        layers = attention_state_layers(state["layers"], f"{state_label}.layers")
        claimed_layers.extend(layers)
        storage_label = f"{state_label}.storage"
        storage = state["storage"]
        if not isinstance(storage, dict):
            raise RuntimeError(f"{storage_label} must be an object")
        kind = storage.get("kind")

        if kind in ("token_kv", "latent_kv"):
            if kind == "token_kv":
                byte_fields = (
                    "key_bytes_per_token_per_layer",
                    "value_bytes_per_token_per_layer",
                )
                component_names = ("key", "value")
            else:
                byte_fields = (
                    "latent_bytes_per_token_per_layer",
                    "rope_bytes_per_token_per_layer",
                )
                component_names = ("latent", "rope")
            exact(
                storage,
                {"kind", *byte_fields, "retention", "window_tokens"},
                storage_label,
            )
            component_values = tuple(
                bounded_positive_integer(
                    storage[field], f"{storage_label}.{field}", U64_MAX
                )
                for field in byte_fields
            )
            byte_count = sum(component_values)
            if byte_count > U64_MAX or byte_count > U64_MAX // page_tokens:
                raise RuntimeError(f"{storage_label} token geometry overflows")
            retention = storage["retention"]
            window_tokens = storage["window_tokens"]
            if retention == "full":
                if window_tokens is not None:
                    raise RuntimeError(
                        f"{storage_label} full retention requires a null window"
                    )
            elif retention == "sliding":
                window_tokens = bounded_positive_integer(
                    window_tokens, f"{storage_label}.window_tokens", U64_MAX
                )
            else:
                raise RuntimeError(f"{storage_label}.retention is unsupported")
            if any((layer, "token_addressable") in roles for layer in layers):
                raise RuntimeError(f"{label} token-addressable state layers overlap")
            roles.update((layer, "token_addressable") for layer in layers)
            manager_class = {
                "name": name,
                "layers": list(layers),
                "retention": retention,
                "bytes_per_token_per_layer": byte_count,
                "window_tokens": window_tokens,
                "components": [
                    {
                        "name": component_name,
                        "bytes_per_token_per_layer": component_bytes,
                    }
                    for component_name, component_bytes in zip(
                        component_names, component_values, strict=True
                    )
                ],
            }
            if kind != "token_kv":
                manager_class["storage"] = kind
            classes.append(manager_class)
            continue

        if kind == "recurrent":
            exact(
                storage,
                {
                    "kind",
                    "family",
                    "state_bytes_per_layer",
                    "checkpoint_slots_per_request",
                },
                storage_label,
            )
            if storage["family"] not in ("mamba", "gdn", "kda", "linear_attention"):
                raise RuntimeError(f"{storage_label}.family is unsupported")
            role = "recurrent"
        elif kind == "convolution":
            exact(
                storage,
                {
                    "kind",
                    "state_bytes_per_layer",
                    "kernel_width",
                    "checkpoint_slots_per_request",
                },
                storage_label,
            )
            bounded_positive_integer(
                storage["kernel_width"], f"{storage_label}.kernel_width", U32_MAX
            )
            role = "convolution"
        else:
            raise RuntimeError(f"{storage_label}.kind is unsupported")
        state_bytes = bounded_positive_integer(
            storage["state_bytes_per_layer"], f"{storage_label}.state_bytes_per_layer", U64_MAX
        )
        slots = bounded_positive_integer(
            storage["checkpoint_slots_per_request"],
            f"{storage_label}.checkpoint_slots_per_request",
            U32_MAX,
        )
        if slots < 2:
            raise RuntimeError(f"{storage_label} requires at least two checkpoints")
        if state_bytes > U64_MAX // len(layers):
            raise RuntimeError(f"{storage_label} checkpoint geometry overflows")
        state_bytes *= len(layers)
        if state_bytes > U64_MAX // slots:
            raise RuntimeError(f"{storage_label} checkpoint geometry overflows")
        if any((layer, role) in roles for layer in layers):
            raise RuntimeError(f"{label} fixed-state layers overlap")
        roles.update((layer, role) for layer in layers)

    if sorted(set(claimed_layers)) != list(range(max(claimed_layers) + 1)):
        raise RuntimeError(f"{label} attention states do not cover every model layer")
    if not classes:
        raise RuntimeError(f"{label} attention source has no token manager input")
    return {"page_tokens": page_tokens, "classes": classes}


def manager_input_fingerprint(manifest: Mapping[str, Any], label: str) -> str:
    manager_input = manager_input_from_attention_source(manifest, label)
    try:
        encoded = json.dumps(
            manager_input, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RuntimeError(f"{label} manager input is not canonical JSON") from error
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def layout_fingerprint_from_manager_input(
    manager_input: Mapping[str, Any], label: str
) -> str:
    digest = hashlib.sha256()

    def update_u64(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= U64_MAX:
            raise RuntimeError(f"{label} contains an invalid u64 value")
        digest.update(struct.pack("<Q", value))

    def update_text(value: Any) -> None:
        if not isinstance(value, str):
            raise RuntimeError(f"{label} contains an invalid string")
        encoded = value.encode("utf-8")
        update_u64(len(encoded))
        digest.update(encoded)

    page_tokens = manager_input["page_tokens"]
    classes = manager_input["classes"]
    update_u64(page_tokens)
    update_u64(len(classes))
    for item in classes:
        update_text(item["name"])
        update_u64(len(item["layers"]))
        for layer in item["layers"]:
            update_u64(layer)
        update_u64(item["bytes_per_token_per_layer"])
        retention = item["retention"]
        update_u64(0 if retention == "full" else 1)
        window = item["window_tokens"]
        update_u64(window or 0)
        components = item["components"]
        storage = item.get("storage", "token_kv")
        if storage != "token_kv" or components:
            update_u64(1)
            update_u64(0 if storage == "token_kv" else 1)
            update_u64(len(components))
            for component in components:
                update_text(component["name"])
                update_u64(component["bytes_per_token_per_layer"])
        slots = (
            0
            if retention == "full"
            else 1 + (window - 1 + page_tokens - 1) // page_tokens
        )
        update_u64(slots)
    return "sha256:" + digest.hexdigest()


def _validate_token_components(
    value: Any,
    expected: Sequence[Mapping[str, Any]],
    storage: str,
    label: str,
) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, list)
        or not isinstance(expected, list)
        or len(value) != 2
        or len(expected) != 2
    ):
        raise RuntimeError(f"{label} must contain exactly two components")
    expected_names = ("key", "value") if storage == "token_kv" else ("latent", "rope")
    components = []
    for index, (raw, source) in enumerate(zip(value, expected, strict=True)):
        component = exact(raw, {"name", "bytes_per_token_per_layer"}, f"{label}[{index}]")
        expected_name = expected_names[index]
        width = bounded_positive_integer(
            component["bytes_per_token_per_layer"],
            f"{label}[{index}].bytes_per_token_per_layer",
            U64_MAX,
        )
        if (
            component["name"] != expected_name
            or source.get("name") != expected_name
            or source.get("bytes_per_token_per_layer") != width
        ):
            raise RuntimeError(f"{label} differs from the attention source")
        components.append(component)
    return components


def native_session_contract(
    signature: Mapping[str, Any], label: str
) -> tuple[str, str]:
    states = signature.get("token_states")
    if not isinstance(states, list):
        raise RuntimeError(f"{label} runtime signature token states are malformed")
    shape: list[tuple[Any, Any]] = []
    for index, state in enumerate(states):
        if not isinstance(state, Mapping) or not isinstance(state.get("backend"), Mapping):
            raise RuntimeError(f"{label} runtime token state {index} is malformed")
        backend = state["backend"]
        shape.append((backend.get("retention"), backend.get("storage")))
    contract = NATIVE_SESSION_CONTRACTS.get(tuple(shape))
    if contract is None:
        raise RuntimeError(
            f"{label} runtime signature is not an exact admitted native-session profile"
        )
    return contract


def validate_native_session_profile(
    manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    manager_input: Mapping[str, Any],
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    """Validate the exact Full, Full+Sliding, Sliding, or Full-latent profile."""

    classes = signature["token_classes"]
    states = signature["token_states"]
    fixed = signature["fixed_states"]
    if not isinstance(classes, list) or not isinstance(states, list):
        raise RuntimeError(f"{label} runtime signature token projections are malformed")
    if len(classes) != len(states) or len(classes) not in (1, 2):
        raise RuntimeError(
            f"{label} runtime signature has an unsupported token-class shape"
        )
    if fixed != []:
        raise RuntimeError(f"{label} runtime signature contains fixed state")

    attention_plan = exact(
        manifest["attention_state_plan"],
        {"schema", "page_tokens", "states"},
        f"{label}.runtime_manifest.attention_state_plan",
    )
    if (
        attention_plan["schema"] != "orbitkv.attention-state-plan.v1"
        or attention_plan["page_tokens"] != signature["page_tokens"]
    ):
        raise RuntimeError(f"{label} compiled attention plan identity differs")
    compiled_states = attention_plan["states"]
    layout_wrapper = exact(
        manifest["token_manager_plan"],
        {"layout"},
        f"{label}.runtime_manifest.token_manager_plan",
    )
    layout = exact(
        layout_wrapper["layout"],
        {"schema", "plan_fingerprint", "page_tokens", "classes"},
        f"{label}.runtime_manifest.token_manager_plan.layout",
    )
    source_states = manifest["source"]["input"]["states"]
    if (
        layout["schema"] != "orbitkv.layout-program.v1"
        or layout["page_tokens"] != signature["page_tokens"]
        or not isinstance(compiled_states, list)
        or not isinstance(layout["classes"], list)
        or not isinstance(manager_input.get("classes"), list)
        or len(compiled_states) != len(states)
        or len(layout["classes"]) != len(classes)
        or len(manager_input["classes"]) != len(classes)
        or not isinstance(source_states, list)
        or len(source_states) != len(classes)
    ):
        raise RuntimeError(f"{label} runtime token projections differ")

    native_session_contract(signature, label)
    expected_shape = tuple(
        (state["backend"]["retention"], state["backend"]["storage"])
        for state in states
    )
    names: set[str] = set()
    all_layers: list[int] = []
    normalized: list[Mapping[str, Any]] = []
    for index, (expected_retention, expected_storage) in enumerate(expected_shape):
        item_label = f"{label}.runtime_binding.execution_signature.token_classes[{index}]"
        state_label = f"{label}.runtime_binding.execution_signature.token_states[{index}]"
        token_class = exact(
            classes[index],
            {
                "name", "layers", "bytes_per_token_per_layer", "address",
                "retirement", "minimum_slots_per_request",
            },
            item_label,
        )
        state = exact(states[index], {"name", "layers", "backend"}, state_label)
        compiled = exact(
            compiled_states[index],
            {"name", "layers", "backend"},
            f"{label}.runtime_manifest.attention_state_plan.states[{index}]",
        )
        layout_class = exact(
            layout["classes"][index],
            {
                "name", "layers", "bytes_per_token_per_layer", "address",
                "retirement", "minimum_slots_per_request",
            },
            f"{label}.runtime_manifest.token_manager_plan.layout.classes[{index}]",
        )
        manager_class = manager_input["classes"][index]
        source_state = source_states[index]
        if not isinstance(manager_class, Mapping):
            raise RuntimeError(f"{label} manager-input token class is malformed")
        if not isinstance(source_state, Mapping):
            raise RuntimeError(f"{label} source token state is malformed")

        name = token_class["name"]
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{item_label}.name must be nonempty")
        layers = attention_state_layers(token_class["layers"], f"{item_label}.layers")
        if name in names or any(layer in all_layers for layer in layers):
            raise RuntimeError(f"{label} runtime token classes overlap or reuse a name")
        names.add(name)
        all_layers.extend(layers)
        width = bounded_positive_integer(
            token_class["bytes_per_token_per_layer"],
            f"{item_label}.bytes_per_token_per_layer",
            U64_MAX,
        )
        backend = exact(
            state["backend"],
            {
                "kind", "storage", "components", "bytes_per_token_per_layer",
                "page_bytes_per_layer", "retention", "window_tokens",
                "token_relocatable",
            },
            f"{state_label}.backend",
        )
        exact(
            compiled["backend"],
            set(backend),
            f"{label}.runtime_manifest.attention_state_plan.states[{index}].backend",
        )
        if not same(compiled, state):
            raise RuntimeError(f"{label} runtime token state differs from compiled attention plan")
        if not same(layout_class, token_class):
            raise RuntimeError(f"{label} runtime token class differs from compiled layout")
        backend_width = bounded_positive_integer(
            backend["bytes_per_token_per_layer"],
            f"{state_label}.backend.bytes_per_token_per_layer",
            U64_MAX,
        )
        backend_page_bytes = bounded_positive_integer(
            backend["page_bytes_per_layer"],
            f"{state_label}.backend.page_bytes_per_layer",
            U64_MAX,
        )
        if (
            state["name"] != name
            or not same(state["layers"], layers)
            or manager_class.get("name") != name
            or not same(manager_class.get("layers"), layers)
            or source_state.get("name") != name
            or not same(source_state.get("layers"), layers)
            or manager_class.get("storage", "token_kv") != expected_storage
            or manager_class.get("retention") != expected_retention
            or backend["kind"] != "token_slots"
            or backend["storage"] != expected_storage
            or backend["retention"] != expected_retention
            or backend["token_relocatable"] is not True
            or backend_width != width
            or manager_class.get("bytes_per_token_per_layer") != width
        ):
            raise RuntimeError(f"{label} runtime {expected_retention} token projection differs")
        components = _validate_token_components(
            backend["components"], manager_class.get("components"), expected_storage,
            f"{state_label}.backend.components",
        )
        if sum(component["bytes_per_token_per_layer"] for component in components) != width:
            raise RuntimeError(f"{label} runtime token component geometry differs")
        page_tokens = signature["page_tokens"]
        if width > U64_MAX // page_tokens or backend_page_bytes != width * page_tokens:
            raise RuntimeError(f"{label} runtime token page geometry differs")

        window = backend["window_tokens"]
        if expected_retention == "full":
            if (
                window is not None
                or manager_class.get("window_tokens") is not None
                or not same(token_class["address"], {"kind": "append_only"})
                or not same(token_class["retirement"], {"kind": "never"})
                or token_class["minimum_slots_per_request"] is not None
            ):
                raise RuntimeError(f"{label} runtime Full token_kv geometry differs")
        else:
            window = bounded_positive_integer(
                window, f"{state_label}.backend.window_tokens", U64_MAX
            )
            period = 1 + (window - 1 + page_tokens - 1) // page_tokens
            if (
                manager_class.get("window_tokens") != window
                or not same(
                    token_class["address"],
                    {"kind": "periodic", "period_blocks": period},
                )
                or not same(
                    token_class["retirement"],
                    {"kind": "block_end_plus", "offset_tokens": window - 1},
                )
                or token_class["minimum_slots_per_request"] != period
            ):
                raise RuntimeError(f"{label} runtime Sliding token_kv geometry differs")
        normalized.append(token_class)

    if sorted(all_layers) != list(range(len(all_layers))):
        raise RuntimeError(f"{label} runtime token classes do not cover every model layer")
    return tuple(normalized)
