from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .ffi.library import WIRE_VERSION


RUNTIME_TARGET_SCHEMA = "orbitkv.runtime-target"
RUNTIME_TARGET_VERSION = 1
EXECUTION_SIGNATURE_SCHEMA = "orbitkv.execution-signature"
EXECUTION_SIGNATURE_VERSION = 1
RUNTIME_BINDING_SCHEMA = "orbitkv.runtime-binding"
RUNTIME_BINDING_VERSION = 1
RUNTIME_TARGET_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
CLASSIFIABLE_TOPOLOGIES = (
    "whole_domain_chunked_token_kv",
    "whole_domain_full_latent_kv",
    "whole_domain_full_sliding_token_kv",
    "whole_domain_full_token_kv",
    "whole_domain_full_token_kv_gdn_convolution",
    "whole_domain_full_token_kv_mamba",
    "whole_domain_sliding_token_kv",
)
SUPPORTED_CAPABILITIES = frozenset(
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
_RESOURCE = "resources/runtime_target.json"
_TARGET: Mapping[str, Any] | None = None
_ENGINE_PROFILE_ARTIFACT_MAX_BYTES = 1024 * 1024
_ENGINE_PROFILE_CANDIDATES = (
    # Repository layout: compat/sglang/bridge/src/orbitkv_sglang.
    Path(__file__).resolve().parents[4] / "profile.json",
    # Assembled layout: orbitkv/adapter/src/orbitkv_sglang.
    Path(__file__).resolve().parents[3] / "profile.json",
)
_PRODUCT_PROFILE_KEYS = {
    "execution_topology",
    "ordered_attention_classes",
    "execution",
    "device_scope",
    "plan_format",
    "token_reclamation",
    "fixed_state",
    "cache_policy",
}
_PRODUCT_CACHE_POLICIES = {
    "whole_domain_full_token_kv": "shared_prefix",
    "whole_domain_full_sliding_token_kv": "shared_prefix",
    "whole_domain_sliding_token_kv": "request_private",
    "whole_domain_chunked_token_kv": "request_private",
    "whole_domain_full_latent_kv": "request_private",
}


@dataclass(frozen=True, slots=True)
class ProductTakeoverProfile:
    """One product-level native-session takeover profile."""

    execution_topology: str
    ordered_attention_classes: tuple[str, ...]
    execution: str
    device_scope: str
    plan_format: str
    token_reclamation: str
    fixed_state: str
    cache_policy: str


_PRODUCT_TAKEOVER_PROFILES: tuple[ProductTakeoverProfile, ...] | None = None


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{path} fields differ; missing={missing}, unknown={unknown}"
        )


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _bounded_positive_int(value: Any, path: str, maximum: int) -> int:
    result = _positive_int(value, path)
    if result > maximum:
        raise ValueError(f"{path} exceeds the supported integer range")
    return result


def _nonnegative_int(value: Any, path: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def _exact_version(value: Any, expected: int, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{path} must be {expected}")
    return value


def _stable_id(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not value[0].islower()
        or any(
            not (character.isascii() and (character.islower() or character.isdigit()))
            and character not in ".-_"
            for character in value
        )
    ):
        raise ValueError(f"{path} must be a stable lowercase identifier")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 fingerprint")
    return value


def _sorted_unique_strings(
    value: Any, path: str, known: set[str] | frozenset[str]
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
        or tuple(value) != tuple(sorted(set(value)))
        or any(item not in known for item in value)
    ):
        raise ValueError(f"{path} must be non-empty, sorted, unique, and known")
    return tuple(value)


def _engine_profile_path() -> Path:
    for candidate in _ENGINE_PROFILE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "OrbitKV engine product profile is not installed beside the adapter"
    )


def _product_profile_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _load_product_takeover_profiles(path: Path) -> tuple[ProductTakeoverProfile, ...]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read OrbitKV engine product profile: {error}") from error
    if len(encoded) > _ENGINE_PROFILE_ARTIFACT_MAX_BYTES:
        raise ValueError("OrbitKV engine product profile exceeds its size limit")
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid OrbitKV engine product profile: {error}") from error
    root = _mapping(raw, "EngineProductProfile")
    if root.get("schema") != "orbitkv.engine-product-profile":
        raise ValueError("unsupported OrbitKV engine product profile schema")
    _exact_version(
        root.get("schema_version"), 4, "EngineProductProfile.schema_version"
    )
    if root.get("product_id") != "orbitkv-engine":
        raise ValueError("unexpected OrbitKV engine product identity")

    raw_profiles = root.get("manager_supported_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(
            "EngineProductProfile.manager_supported_profiles must be nonempty"
        )
    profiles: list[ProductTakeoverProfile] = []
    topologies: set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        label = f"EngineProductProfile.manager_supported_profiles[{index}]"
        value = _mapping(raw_profile, label)
        _exact_keys(value, label, _PRODUCT_PROFILE_KEYS)
        topology = _product_profile_text(
            value["execution_topology"], f"{label}.execution_topology"
        )
        if topology not in CLASSIFIABLE_TOPOLOGIES:
            raise ValueError(f"{label}.execution_topology is unknown")
        if topology in topologies:
            raise ValueError(
                "EngineProductProfile.manager_supported_profiles contains "
                f"duplicate topology {topology!r}"
            )
        cache_policy = _product_profile_text(
            value["cache_policy"], f"{label}.cache_policy"
        )
        expected_policy = _PRODUCT_CACHE_POLICIES.get(topology)
        if expected_policy is None or cache_policy != expected_policy:
            raise ValueError(
                f"{label}.cache_policy must be {expected_policy!r} for "
                f"topology {topology!r}"
            )
        classes = value["ordered_attention_classes"]
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(item, str) or not item for item in classes)
            or len(classes) != len(set(classes))
        ):
            raise ValueError(
                f"{label}.ordered_attention_classes must be nonempty and unique"
            )
        profiles.append(
            ProductTakeoverProfile(
                execution_topology=topology,
                ordered_attention_classes=tuple(classes),
                execution=_product_profile_text(value["execution"], f"{label}.execution"),
                device_scope=_product_profile_text(
                    value["device_scope"], f"{label}.device_scope"
                ),
                plan_format=_product_profile_text(
                    value["plan_format"], f"{label}.plan_format"
                ),
                token_reclamation=_product_profile_text(
                    value["token_reclamation"], f"{label}.token_reclamation"
                ),
                fixed_state=_product_profile_text(
                    value["fixed_state"], f"{label}.fixed_state"
                ),
                cache_policy=cache_policy,
            )
        )
        topologies.add(topology)

    unsupported = root.get("manager_unsupported_profiles")
    if (
        not isinstance(unsupported, list)
        or any(not isinstance(item, str) for item in unsupported)
        or tuple(unsupported) != tuple(sorted(set(unsupported)))
    ):
        raise ValueError(
            "EngineProductProfile.manager_unsupported_profiles must be sorted and unique"
        )
    static_topologies = set(load_runtime_target()["supported_topologies"])
    unsupported_set = set(unsupported)
    if topologies & unsupported_set or topologies | unsupported_set != static_topologies:
        raise ValueError(
            "engine product takeover profiles do not partition the runtime target's "
            "static topologies"
        )
    return tuple(profiles)


def load_product_takeover_profiles(
    profile_path: Path | str | None = None,
) -> tuple[ProductTakeoverProfile, ...]:
    """Load the product's native-session takeover set from ``compat/sglang/profile.json``."""

    global _PRODUCT_TAKEOVER_PROFILES
    if profile_path is not None:
        return _load_product_takeover_profiles(Path(profile_path))
    if _PRODUCT_TAKEOVER_PROFILES is None:
        _PRODUCT_TAKEOVER_PROFILES = _load_product_takeover_profiles(
            _engine_profile_path()
        )
    return _PRODUCT_TAKEOVER_PROFILES


def product_takeover_profile(config: Any) -> ProductTakeoverProfile | None:
    """Match a config to the sole product takeover contract.

    This is a side-effect-free selector: incomplete, legacy, unsupported, and
    malformed configs all return ``None``. Product entry points and runtime
    construction use the strict admission helpers below.
    """

    profiles = load_product_takeover_profiles()
    binding = getattr(config, "runtime_binding", None)
    if getattr(config, "runtime_manifest_path", None) is None:
        return None
    topology = (
        binding.get("execution_topology")
        if isinstance(binding, Mapping)
        else None
    )
    classes = tuple(
        f"{getattr(item, 'retention', None)}:{getattr(item, 'storage', None)}"
        for item in getattr(config, "classes", ())
    )
    topology_profile = next(
        (profile for profile in profiles if profile.execution_topology == topology),
        None,
    )
    class_profile = next(
        (profile for profile in profiles if profile.ordered_attention_classes == classes),
        None,
    )
    if topology_profile is None or topology_profile != class_profile:
        return None
    selected = topology_profile
    if getattr(config, "manager_plan_format", None) != selected.plan_format:
        return None
    reclamation = getattr(
        getattr(config, "token_reclamation", None), "mode", None
    )
    if reclamation != selected.token_reclamation:
        return None
    fixed_state = "none" if not tuple(getattr(config, "fixed_states", ())) else "present"
    if fixed_state != selected.fixed_state:
        return None
    return selected


def admit_product_takeover_config(config: Any) -> ProductTakeoverProfile:
    """Admit one manifest-backed config to the product session runtime.

    Runtime-target admission deliberately runs before product-profile
    selection. A missing or stale manifest binding is therefore never
    interpreted as permission to construct a different runtime.
    """

    topology = admit_runtime_config(config)
    declared = next(
        (
            profile
            for profile in load_product_takeover_profiles()
            if profile.execution_topology == topology
        ),
        None,
    )
    if declared is None:
        raise ValueError(
            f"topology {topology!r} is not supported by the OrbitKV Engine "
            "product runtime"
        )
    selected = product_takeover_profile(config)
    if selected != declared:
        raise ValueError(
            f"product native-session takeover topology {topology!r} "
            "does not match its profile"
        )
    return declared


def load_runtime_target() -> Mapping[str, Any]:
    """Load a detached copy of the packaged SGLang runtime target."""

    global _TARGET
    if _TARGET is not None:
        return copy.deepcopy(_TARGET)
    encoded = resources.files("orbitkv_sglang").joinpath(_RESOURCE).read_bytes()
    if len(encoded) > RUNTIME_TARGET_ARTIFACT_MAX_BYTES:
        raise ValueError("OrbitKV runtime target exceeds its size limit")
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid OrbitKV runtime target: {error}") from error
    target = _validate_runtime_target(raw)
    _TARGET = copy.deepcopy(target)
    return copy.deepcopy(target)


def _validate_runtime_target(raw: Any) -> Mapping[str, Any]:
    target = _mapping(raw, "RuntimeTarget")
    _exact_keys(
        target,
        "RuntimeTarget",
        {
            "schema",
            "version",
            "fingerprint",
            "target",
            "admission_profile",
            "page_tokens",
            "supported_manifest_versions",
            "required_wire_version",
            "supported_capabilities",
            "supported_topologies",
        },
    )
    if target["schema"] != RUNTIME_TARGET_SCHEMA:
        raise ValueError("unsupported OrbitKV runtime target schema")
    _exact_version(target["version"], RUNTIME_TARGET_VERSION, "RuntimeTarget.version")
    identity = _mapping(target["target"], "RuntimeTarget.target")
    profile = _mapping(target["admission_profile"], "RuntimeTarget.admission_profile")
    _exact_keys(identity, "RuntimeTarget.target", {"id", "contract_version"})
    _exact_keys(profile, "RuntimeTarget.admission_profile", {"id", "version"})
    _stable_id(identity["id"], "RuntimeTarget.target.id")
    _bounded_positive_int(
        identity["contract_version"], "RuntimeTarget.target.contract_version", _U32_MAX
    )
    _stable_id(profile["id"], "RuntimeTarget.admission_profile.id")
    _bounded_positive_int(
        profile["version"], "RuntimeTarget.admission_profile.version", _U32_MAX
    )
    _bounded_positive_int(target["page_tokens"], "RuntimeTarget.page_tokens", _U64_MAX)
    versions = target["supported_manifest_versions"]
    if (
        not isinstance(versions, list)
        or not versions
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 < item <= _U32_MAX
            for item in versions
        )
        or tuple(versions) != tuple(sorted(set(versions)))
    ):
        raise ValueError(
            "RuntimeTarget.supported_manifest_versions must be non-empty, sorted, and unique"
        )
    _bounded_positive_int(
        target["required_wire_version"], "RuntimeTarget.required_wire_version", _U32_MAX
    )
    _sorted_unique_strings(
        target["supported_capabilities"],
        "RuntimeTarget.supported_capabilities",
        SUPPORTED_CAPABILITIES,
    )
    _sorted_unique_strings(
        target["supported_topologies"],
        "RuntimeTarget.supported_topologies",
        frozenset(CLASSIFIABLE_TOPOLOGIES),
    )
    fingerprint = _sha256(target["fingerprint"], "RuntimeTarget.fingerprint")
    if fingerprint != _fingerprint(target):
        raise ValueError("RuntimeTarget.fingerprint does not match its payload")
    return target


def execution_signature_from_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Derive the complete structural admission signature."""

    from .runtime_manifest import validate_runtime_manifest

    root = validate_runtime_manifest(manifest)
    manager = root.get("token_manager_plan")
    token_classes: list[Mapping[str, Any]] = []
    if manager is not None:
        layout = _mapping(
            _mapping(manager, "RuntimeManifest.token_manager_plan").get("layout"),
            "RuntimeManifest.token_manager_plan.layout",
        )
        token_classes = copy.deepcopy(layout["classes"])

    source = _mapping(root["source"], "RuntimeManifest.source")
    token_states: list[Mapping[str, Any]] = []
    fixed_states: list[Mapping[str, Any]] = []
    if source["kind"] == "attention_state":
        attention = _mapping(
            root["attention_state_plan"], "RuntimeManifest.attention_state_plan"
        )
        page_tokens = attention["page_tokens"]
        for state in attention["states"]:
            value = _mapping(state, "RuntimeManifest.attention_state_plan.state")
            backend = _mapping(
                value.get("backend"),
                "RuntimeManifest.attention_state_plan.state.backend",
            )
            destination = (
                token_states if backend.get("kind") == "token_slots" else fixed_states
            )
            destination.append(copy.deepcopy(value))
    else:
        program = _mapping(source["program"], "RuntimeManifest.source.program")
        page_tokens = program["page_tokens"]
        states = program["states"]
        if len(states) != 1 or len(token_classes) != 1:
            raise ValueError("RuntimeManifest topology is not executable by this runtime target")
        state = states[0]
        item = token_classes[0]
        from ._retention_ir import _infer_retention

        inferred = _infer_retention(state["may_read"])
        address = item["address"]
        retirement = item["retirement"]
        blocks = address.get("blocks_per_epoch")
        if (
            inferred[0] != "chunked"
            or address.get("kind") != "resettable_arena"
            or retirement != {"kind": "epoch_end", "blocks_per_epoch": blocks}
            or blocks != item["minimum_slots_per_request"]
            or inferred[1] != blocks * page_tokens
            or "kv_head_range" in state
            or "kv_head_range" in item
            or "block_domain" in item
            or state["name"] != item["name"]
            or state["layers"] != item["layers"]
            or state["bytes_per_token_per_layer"]
            != item["bytes_per_token_per_layer"]
        ):
            raise ValueError("RuntimeManifest topology is not executable by this runtime target")
        width = state["bytes_per_token_per_layer"]
        token_states.append(
            {
                "name": state["name"],
                "layers": copy.deepcopy(state["layers"]),
                "backend": {
                    "kind": "token_slots",
                    "storage": "token_kv",
                    "components": [],
                    "bytes_per_token_per_layer": width,
                    "page_bytes_per_layer": width * page_tokens,
                    "retention": "chunked",
                    "window_tokens": None,
                    "token_relocatable": True,
                },
            }
        )

    signature: dict[str, Any] = {
        "schema": EXECUTION_SIGNATURE_SCHEMA,
        "version": EXECUTION_SIGNATURE_VERSION,
        "fingerprint": "",
        "manifest_schema": root["schema"],
        "manifest_version": root["version"],
        "manifest_fingerprint": root["fingerprint"],
        "page_tokens": page_tokens,
        "token_classes": token_classes,
        "token_states": token_states,
        "fixed_states": fixed_states,
    }
    signature["fingerprint"] = _fingerprint(signature)
    return _validate_execution_signature(signature)


def _validate_layers(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty layer list")
    layers = [_nonnegative_int(item, path, _U32_MAX) for item in value]
    if layers != sorted(set(layers)):
        raise ValueError(f"{path} must be sorted and unique")
    return layers


def _validate_token_class(value: Any, path: str) -> Mapping[str, Any]:
    item = _mapping(value, path)
    required = {
        "name",
        "layers",
        "bytes_per_token_per_layer",
        "address",
        "retirement",
        "minimum_slots_per_request",
    }
    optional = {"kv_head_range", "block_domain"}
    if not required <= set(item) or set(item) - required - optional:
        raise ValueError(f"{path} has an invalid class wire shape")
    if not isinstance(item["name"], str) or not item["name"]:
        raise ValueError(f"{path}.name must be nonempty")
    _validate_layers(item["layers"], f"{path}.layers")
    _bounded_positive_int(
        item["bytes_per_token_per_layer"],
        f"{path}.bytes_per_token_per_layer",
        _U64_MAX,
    )
    if "kv_head_range" in item:
        head = _mapping(item["kv_head_range"], f"{path}.kv_head_range")
        _exact_keys(head, f"{path}.kv_head_range", {"start", "end_exclusive"})
        start = _nonnegative_int(head["start"], f"{path}.kv_head_range.start", _U32_MAX)
        end = _nonnegative_int(
            head["end_exclusive"], f"{path}.kv_head_range.end_exclusive", _U32_MAX
        )
        if start >= end:
            raise ValueError(f"{path}.kv_head_range must be nonempty")
    if "block_domain" in item:
        domain = _mapping(item["block_domain"], f"{path}.block_domain")
        if set(domain) not in ({"start_block"}, {"start_block", "end_block_exclusive"}):
            raise ValueError(f"{path}.block_domain fields differ")
        start = _nonnegative_int(
            domain["start_block"], f"{path}.block_domain.start_block", _U64_MAX
        )
        if "end_block_exclusive" in domain:
            end = _nonnegative_int(
                domain["end_block_exclusive"],
                f"{path}.block_domain.end_block_exclusive",
                _U64_MAX,
            )
            if start >= end:
                raise ValueError(f"{path}.block_domain must be nonempty")
        elif start == 0:
            raise ValueError(f"{path}.block_domain must be omitted for the whole domain")

    address = _mapping(item["address"], f"{path}.address")
    address_fields = {
        "append_only": {"kind"},
        "pinned": {"kind"},
        "periodic": {"kind", "period_blocks"},
        "periodic_from": {"kind", "period_blocks", "origin_block"},
        "resettable_arena": {"kind", "blocks_per_epoch"},
    }
    kind = address.get("kind")
    if kind not in address_fields:
        raise ValueError(f"{path}.address.kind is unsupported")
    _exact_keys(address, f"{path}.address", address_fields[kind])
    for key in set(address) - {"kind"}:
        if key == "origin_block":
            _nonnegative_int(address[key], f"{path}.address.{key}", _U64_MAX)
        else:
            _bounded_positive_int(address[key], f"{path}.address.{key}", _U64_MAX)

    retirement = _mapping(item["retirement"], f"{path}.retirement")
    retirement_fields = {
        "never": {"kind"},
        "block_end_plus": {"kind", "offset_tokens"},
        "epoch_end": {"kind", "blocks_per_epoch"},
    }
    kind = retirement.get("kind")
    if kind not in retirement_fields:
        raise ValueError(f"{path}.retirement.kind is unsupported")
    _exact_keys(retirement, f"{path}.retirement", retirement_fields[kind])
    for key in set(retirement) - {"kind"}:
        if key == "blocks_per_epoch":
            _bounded_positive_int(retirement[key], f"{path}.retirement.{key}", _U64_MAX)
        else:
            _nonnegative_int(retirement[key], f"{path}.retirement.{key}", _U64_MAX)
    slots = item["minimum_slots_per_request"]
    if slots is not None:
        _bounded_positive_int(slots, f"{path}.minimum_slots_per_request", _U64_MAX)
    return item


def _validate_token_state(value: Any, path: str, page_tokens: int) -> Mapping[str, Any]:
    state = _mapping(value, path)
    _exact_keys(state, path, {"name", "layers", "backend"})
    if not isinstance(state["name"], str) or not state["name"]:
        raise ValueError(f"{path}.name must be nonempty")
    _validate_layers(state["layers"], f"{path}.layers")
    backend = _mapping(state["backend"], f"{path}.backend")
    _exact_keys(
        backend,
        f"{path}.backend",
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
    storage = backend["storage"]
    if backend.get("kind") != "token_slots" or storage not in ("token_kv", "latent_kv"):
        raise ValueError(f"{path}.backend is not a token state")
    width = _bounded_positive_int(
        backend["bytes_per_token_per_layer"],
        f"{path}.backend.bytes_per_token_per_layer",
        _U64_MAX,
    )
    page_bytes = _bounded_positive_int(
        backend["page_bytes_per_layer"],
        f"{path}.backend.page_bytes_per_layer",
        _U64_MAX,
    )
    if width > _U64_MAX // page_tokens or page_bytes != width * page_tokens:
        raise ValueError(f"{path}.backend token geometry differs")
    retention = backend["retention"]
    components = backend["components"]
    if retention == "chunked":
        if storage != "token_kv" or components != [] or backend["window_tokens"] is not None:
            raise ValueError(f"{path}.backend chunked geometry differs")
    else:
        expected = ("key", "value") if storage == "token_kv" else ("latent", "rope")
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError(f"{path}.backend.components must contain two items")
        widths: list[int] = []
        for index, (component, name) in enumerate(zip(components, expected, strict=True)):
            component = _mapping(component, f"{path}.backend.components[{index}]")
            _exact_keys(
                component,
                f"{path}.backend.components[{index}]",
                {"name", "bytes_per_token_per_layer"},
            )
            if component["name"] != name:
                raise ValueError(f"{path}.backend component order differs")
            widths.append(
                _bounded_positive_int(
                    component["bytes_per_token_per_layer"],
                    f"{path}.backend component width",
                    _U64_MAX,
                )
            )
        if sum(widths) != width:
            raise ValueError(f"{path}.backend token geometry differs")
        window = backend["window_tokens"]
        if not (
            (retention == "full" and window is None)
            or (retention == "sliding" and _positive_int(window, f"{path}.backend.window_tokens"))
        ):
            raise ValueError(f"{path}.backend retention is invalid")
    if backend["token_relocatable"] is not True:
        raise ValueError(f"{path}.backend.token_relocatable must be true")
    return state


def _validate_fixed_state(value: Any, path: str) -> Mapping[str, Any]:
    state = _mapping(value, path)
    _exact_keys(state, path, {"name", "layers", "backend"})
    if not isinstance(state["name"], str) or not state["name"]:
        raise ValueError(f"{path}.name must be nonempty")
    layers = _validate_layers(state["layers"], f"{path}.layers")
    backend = _mapping(state["backend"], f"{path}.backend")
    fields = {
        "recurrent_checkpoints": {
            "kind", "family", "state_bytes_per_layer",
            "checkpoint_slots_per_request", "checkpoint_bytes_per_request",
            "token_relocatable",
        },
        "convolution_ring": {
            "kind", "state_bytes_per_layer", "kernel_width",
            "checkpoint_slots_per_request", "checkpoint_bytes_per_request",
            "token_relocatable",
        },
    }
    kind = backend.get("kind")
    if kind not in fields:
        raise ValueError(f"{path}.backend is not a fixed state")
    _exact_keys(backend, f"{path}.backend", fields[kind])
    state_bytes = _bounded_positive_int(
        backend["state_bytes_per_layer"], f"{path}.backend.state_bytes_per_layer", _U64_MAX
    )
    slots = _bounded_positive_int(
        backend["checkpoint_slots_per_request"],
        f"{path}.backend.checkpoint_slots_per_request",
        _U32_MAX,
    )
    checkpoint_bytes = _bounded_positive_int(
        backend["checkpoint_bytes_per_request"],
        f"{path}.backend.checkpoint_bytes_per_request",
        _U64_MAX,
    )
    if (
        slots < 2
        or state_bytes > _U64_MAX // len(layers)
        or state_bytes * len(layers) > _U64_MAX // slots
        or checkpoint_bytes != state_bytes * len(layers) * slots
    ):
        raise ValueError(f"{path}.backend checkpoint geometry differs")
    if backend["token_relocatable"] is not False:
        raise ValueError(f"{path}.backend.token_relocatable must be false")
    if kind == "recurrent_checkpoints" and backend["family"] not in (
        "mamba", "gdn", "kda", "linear_attention"
    ):
        raise ValueError(f"{path}.backend.family is unsupported")
    if kind == "convolution_ring":
        _bounded_positive_int(backend["kernel_width"], f"{path}.backend.kernel_width", _U32_MAX)
    return state


def _whole_domain(item: Mapping[str, Any]) -> bool:
    return "kv_head_range" not in item and "block_domain" not in item


def _canonical_layers(groups: tuple[list[int], ...]) -> bool:
    if any(not group or group != sorted(set(group)) for group in groups):
        return False
    flattened = sorted(layer for group in groups for layer in group)
    return flattened == list(range(len(flattened)))


def _chunked_blocks(item: Mapping[str, Any], path: str) -> int:
    address = item["address"]
    retirement = item["retirement"]
    if not _whole_domain(item) or address.get("kind") != "resettable_arena":
        raise ValueError("chunked execution signature requires the whole token domain")
    blocks = _bounded_positive_int(
        address.get("blocks_per_epoch"), f"{path}.address.blocks_per_epoch", _U64_MAX
    )
    if retirement != {"kind": "epoch_end", "blocks_per_epoch": blocks}:
        raise ValueError("chunked execution signature requires matching chunk geometry")
    return blocks


def _validate_execution_signature(signature: Any) -> Mapping[str, Any]:
    value = _mapping(signature, "ExecutionSignature")
    _exact_keys(
        value,
        "ExecutionSignature",
        {
            "schema", "version", "fingerprint", "manifest_schema",
            "manifest_version", "manifest_fingerprint", "page_tokens",
            "token_classes", "token_states", "fixed_states",
        },
    )
    if value["schema"] != EXECUTION_SIGNATURE_SCHEMA:
        raise ValueError("unsupported OrbitKV execution signature schema")
    _exact_version(value["version"], EXECUTION_SIGNATURE_VERSION, "ExecutionSignature.version")
    if value["manifest_schema"] != "orbitkv.runtime-manifest":
        raise ValueError("ExecutionSignature.manifest_schema is unsupported")
    from .runtime_manifest import RUNTIME_MANIFEST_VERSION

    _exact_version(
        value["manifest_version"], RUNTIME_MANIFEST_VERSION, "ExecutionSignature.manifest_version"
    )
    _sha256(value["manifest_fingerprint"], "ExecutionSignature.manifest_fingerprint")
    page_tokens = _bounded_positive_int(
        value["page_tokens"], "ExecutionSignature.page_tokens", _U64_MAX
    )
    for field in ("token_classes", "token_states", "fixed_states"):
        if not isinstance(value[field], list):
            raise ValueError(f"ExecutionSignature.{field} must be a list")
    classes = [
        _validate_token_class(item, f"ExecutionSignature.token_classes[{index}]")
        for index, item in enumerate(value["token_classes"])
    ]
    tokens = [
        _validate_token_state(item, f"ExecutionSignature.token_states[{index}]", page_tokens)
        for index, item in enumerate(value["token_states"])
    ]
    fixed = [
        _validate_fixed_state(item, f"ExecutionSignature.fixed_states[{index}]")
        for index, item in enumerate(value["fixed_states"])
    ]
    if len(classes) != len(tokens):
        raise ValueError("execution signature token projections differ")
    token_roles: set[int] = set()
    fixed_roles: set[tuple[int, str]] = set()
    for state in tokens:
        for layer in state["layers"]:
            if layer in token_roles:
                raise ValueError("execution signature token state roles overlap")
            token_roles.add(layer)
    for state in fixed:
        role = "convolution" if state["backend"]["kind"] == "convolution_ring" else "recurrent"
        for layer in state["layers"]:
            key = (layer, role)
            if key in fixed_roles:
                raise ValueError("execution signature fixed state roles overlap")
            fixed_roles.add(key)
    for index, (item, state) in enumerate(zip(classes, tokens, strict=True)):
        backend = state["backend"]
        if (
            item["name"] != state["name"]
            or item["layers"] != state["layers"]
            or item["bytes_per_token_per_layer"] != backend["bytes_per_token_per_layer"]
        ):
            raise ValueError(f"execution signature token projection {index} differs")
        retention = backend["retention"]
        if retention == "chunked":
            expected_slots = _chunked_blocks(item, f"ExecutionSignature.token_classes[{index}]")
        elif retention == "full":
            if (
                item["address"] != {"kind": "append_only"}
                or item["retirement"] != {"kind": "never"}
            ):
                raise ValueError(f"execution signature token layout {index} differs")
            expected_slots = None
        else:
            window = backend["window_tokens"]
            period = 1 + (window - 1 + page_tokens - 1) // page_tokens
            if (
                item["address"] != {"kind": "periodic", "period_blocks": period}
                or item["retirement"]
                != {"kind": "block_end_plus", "offset_tokens": window - 1}
            ):
                raise ValueError(f"execution signature token layout {index} differs")
            expected_slots = period
        if item["minimum_slots_per_request"] != expected_slots:
            raise ValueError(f"execution signature token layout {index} differs")
    names = [item["name"] for item in (*tokens, *fixed)]
    if len(names) != len(set(names)):
        raise ValueError("execution signature state names are not unique")
    _sha256(value["fingerprint"], "ExecutionSignature.fingerprint")
    if value["fingerprint"] != _fingerprint(value):
        raise ValueError("ExecutionSignature.fingerprint does not match its payload")
    return value


def classify_execution_signature(signature: Mapping[str, Any]) -> str:
    value = _validate_execution_signature(signature)
    classes = value["token_classes"]
    tokens = value["token_states"]
    fixed = value["fixed_states"]
    page_tokens = value["page_tokens"]

    def full(index: int, storage: str) -> bool:
        item, state = classes[index], tokens[index]
        backend = state["backend"]
        return (
            _whole_domain(item)
            and item["name"] == state["name"]
            and item["layers"] == state["layers"]
            and item["address"] == {"kind": "append_only"}
            and item["retirement"] == {"kind": "never"}
            and item["minimum_slots_per_request"] is None
            and backend["storage"] == storage
            and backend["retention"] == "full"
            and backend["window_tokens"] is None
        )

    def sliding(index: int) -> bool:
        item, state = classes[index], tokens[index]
        backend = state["backend"]
        window = backend.get("window_tokens")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            return False
        period = 1 + (window - 1 + page_tokens - 1) // page_tokens
        return (
            _whole_domain(item)
            and item["name"] == state["name"]
            and item["layers"] == state["layers"]
            and item["address"] == {"kind": "periodic", "period_blocks": period}
            and item["retirement"]
            == {"kind": "block_end_plus", "offset_tokens": window - 1}
            and item["minimum_slots_per_request"] == period
            and backend["storage"] == "token_kv"
            and backend["retention"] == "sliding"
        )

    def chunked(index: int) -> bool:
        item, state = classes[index], tokens[index]
        backend = state["backend"]
        try:
            blocks = _chunked_blocks(item, "ExecutionSignature.token_classes[0]")
        except ValueError:
            return False
        return (
            item["name"] == state["name"]
            and item["layers"] == state["layers"]
            and item["minimum_slots_per_request"] == blocks
            and backend["storage"] == "token_kv"
            and backend["components"] == []
            and backend["retention"] == "chunked"
            and _canonical_layers((state["layers"],))
        )

    if not fixed and len(classes) == len(tokens) == 1 and chunked(0):
        return "whole_domain_chunked_token_kv"
    if not fixed and len(classes) == len(tokens) == 1:
        if full(0, "latent_kv") and _canonical_layers((tokens[0]["layers"],)):
            return "whole_domain_full_latent_kv"
        if full(0, "token_kv") and _canonical_layers((tokens[0]["layers"],)):
            return "whole_domain_full_token_kv"
        if sliding(0) and _canonical_layers((tokens[0]["layers"],)):
            return "whole_domain_sliding_token_kv"
    if (
        not fixed
        and len(classes) == len(tokens) == 2
        and full(0, "token_kv")
        and sliding(1)
        and _canonical_layers((tokens[0]["layers"], tokens[1]["layers"]))
    ):
        return "whole_domain_full_sliding_token_kv"
    if len(classes) == len(tokens) == 1 and full(0, "token_kv"):
        recurrent = [item for item in fixed if item["backend"]["kind"] == "recurrent_checkpoints"]
        convolution = [item for item in fixed if item["backend"]["kind"] == "convolution_ring"]
        if (
            recurrent
            and not convolution
            and all(
                item["backend"]["family"] == "mamba"
                and item["backend"]["checkpoint_slots_per_request"] == 2
                for item in recurrent
            )
            and _canonical_layers(
                tuple([tokens[0]["layers"]] + [item["layers"] for item in recurrent])
            )
        ):
            return "whole_domain_full_token_kv_mamba"
        if (
            len(recurrent) == len(convolution) == 1
            and recurrent[0]["backend"]["family"] == "gdn"
            and recurrent[0]["backend"]["checkpoint_slots_per_request"] == 2
            and convolution[0]["backend"]["checkpoint_slots_per_request"] == 2
            and recurrent[0]["layers"] == convolution[0]["layers"]
            and _canonical_layers((tokens[0]["layers"], recurrent[0]["layers"]))
        ):
            return "whole_domain_full_token_kv_gdn_convolution"
    raise ValueError("RuntimeManifest topology is not executable by this runtime target")


def _validate_target_manifest_version(target: Mapping[str, Any], manifest_version: int) -> None:
    if manifest_version not in target["supported_manifest_versions"]:
        identity = target["target"]
        profile = target["admission_profile"]
        raise ValueError(
            "runtime target "
            f"{identity['id']}@{identity['contract_version']} with admission profile "
            f"{profile['id']}@{profile['version']} does not support manifest version "
            f"{manifest_version}"
        )


def admit_execution_signature(
    signature: Mapping[str, Any],
    target: Mapping[str, Any] | None = None,
) -> str:
    """Admit static topology without importing SGLang, Torch, or hooks."""

    selected = load_runtime_target() if target is None else _validate_runtime_target(target)
    value = _validate_execution_signature(signature)
    _validate_target_manifest_version(selected, value["manifest_version"])
    if value["page_tokens"] != selected["page_tokens"]:
        raise ValueError("RuntimeManifest page_tokens is unsupported by this runtime target")
    topology = classify_execution_signature(value)
    if topology not in selected["supported_topologies"]:
        raise ValueError(f"runtime target does not support topology {topology!r}")
    return topology


def runtime_binding_from_manifest(
    manifest: Mapping[str, Any],
    target: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate and bind one canonical manifest to one exact runtime target."""

    from .runtime_manifest import validate_runtime_manifest

    root = validate_runtime_manifest(manifest)
    selected = load_runtime_target() if target is None else _validate_runtime_target(target)
    _validate_target_manifest_version(selected, root["version"])
    unsupported = set(root["capability_requirements"]) - set(
        selected["supported_capabilities"]
    )
    if unsupported:
        raise ValueError(
            "runtime target does not support manifest capabilities: "
            + ", ".join(sorted(unsupported))
        )
    signature = execution_signature_from_manifest(root)
    topology = admit_execution_signature(signature, selected)
    binding: dict[str, Any] = {
        "schema": RUNTIME_BINDING_SCHEMA,
        "version": RUNTIME_BINDING_VERSION,
        "fingerprint": "",
        "manifest_fingerprint": root["fingerprint"],
        "target": copy.deepcopy(selected["target"]),
        "admission_profile": copy.deepcopy(selected["admission_profile"]),
        "target_contract_fingerprint": selected["fingerprint"],
        "required_wire_version": selected["required_wire_version"],
        "execution_topology": topology,
        "execution_signature": signature,
    }
    binding["fingerprint"] = _fingerprint(binding)
    return binding


def _validate_runtime_binding(
    binding: Any,
    signature: Mapping[str, Any],
    manifest_fingerprint: str,
    target: Mapping[str, Any],
) -> str:
    value = _mapping(binding, "RuntimeBinding")
    _exact_keys(
        value,
        "RuntimeBinding",
        {
            "schema", "version", "fingerprint", "manifest_fingerprint",
            "target", "admission_profile", "target_contract_fingerprint",
            "required_wire_version", "execution_topology", "execution_signature",
        },
    )
    if value["schema"] != RUNTIME_BINDING_SCHEMA:
        raise ValueError("unsupported OrbitKV runtime binding")
    _exact_version(value["version"], RUNTIME_BINDING_VERSION, "RuntimeBinding.version")
    if _sha256(value["fingerprint"], "RuntimeBinding.fingerprint") != _fingerprint(value):
        raise ValueError("RuntimeBinding.fingerprint does not match its payload")
    _bounded_positive_int(
        value["required_wire_version"], "RuntimeBinding.required_wire_version", _U32_MAX
    )
    if (
        value["manifest_fingerprint"] != manifest_fingerprint
        or value["target_contract_fingerprint"] != target["fingerprint"]
        or value["target"] != target["target"]
        or value["admission_profile"] != target["admission_profile"]
        or value["required_wire_version"] != target["required_wire_version"]
        or value["execution_signature"] != signature
    ):
        raise ValueError("runtime binding does not match its manifest or target")
    topology = admit_execution_signature(signature, target)
    if value["execution_topology"] != topology:
        raise ValueError("runtime binding topology differs from its signature")
    return topology


def admit_runtime_config(config: Any) -> str:
    signature = getattr(config, "execution_signature", None)
    binding = getattr(config, "runtime_binding", None)
    manifest_fingerprint = getattr(config, "runtime_manifest_fingerprint", None)
    if signature is None or binding is None or manifest_fingerprint is None:
        raise ValueError(
            "runtime target admission requires ORBITKV_RUNTIME_MANIFEST"
        )
    if signature.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError("execution signature does not match the loaded manifest")
    target = load_runtime_target()
    topology = _validate_runtime_binding(
        binding, signature, manifest_fingerprint, target
    )
    if binding["required_wire_version"] != WIRE_VERSION:
        raise ValueError(
            "runtime binding requires wire version "
            f"{binding['required_wire_version']}, adapter={WIRE_VERSION}"
        )
    return topology


__all__ = [
    "ProductTakeoverProfile",
    "admit_execution_signature",
    "admit_product_takeover_config",
    "admit_runtime_config",
    "classify_execution_signature",
    "execution_signature_from_manifest",
    "load_product_takeover_profiles",
    "load_runtime_target",
    "product_takeover_profile",
    "runtime_binding_from_manifest",
]
