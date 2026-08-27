from __future__ import annotations

import hashlib
import json
import copy
from importlib import resources
from typing import Any, Mapping


TARGET_CONTRACT_SCHEMA = "orbitkv.runtime-target-contract"
TARGET_CONTRACT_VERSION = 1
EXECUTION_SIGNATURE_SCHEMA = "orbitkv.execution-signature"
EXECUTION_SIGNATURE_VERSION = 1
TARGET_BINDING_SCHEMA = "orbitkv.runtime-target-binding"
TARGET_BINDING_VERSION = 1
TARGET_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
SUPPORTED_TOPOLOGIES = (
    "whole_domain_full_latent_kv",
    "whole_domain_full_sliding_token_kv",
    "whole_domain_full_token_kv",
    "whole_domain_full_token_kv_gdn_convolution",
    "whole_domain_full_token_kv_mamba",
    "whole_domain_sliding_token_kv",
)
_RESOURCE = "resources/executor_capabilities.v1.json"
_CONTRACT: Mapping[str, Any] | None = None


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


def _exact_keys(
    value: Mapping[str, Any], path: str, expected: set[str]
) -> None:
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


def load_executor_capabilities() -> Mapping[str, Any]:
    """Load and strictly validate the target contract from package data."""

    global _CONTRACT
    if _CONTRACT is not None:
        return copy.deepcopy(_CONTRACT)
    encoded = resources.files("orbitkv_sglang").joinpath(_RESOURCE).read_bytes()
    if len(encoded) > TARGET_ARTIFACT_MAX_BYTES:
        raise ValueError("OrbitKV executor target contract exceeds its size limit")
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid OrbitKV executor target contract: {error}") from error
    contract = _validate_contract(raw)
    _CONTRACT = copy.deepcopy(contract)
    return copy.deepcopy(contract)


def _validate_contract(raw: Any) -> Mapping[str, Any]:
    contract = _mapping(raw, "RuntimeTargetContractV1")
    _exact_keys(
        contract,
        "RuntimeTargetContractV1",
        {
            "schema",
            "version",
            "fingerprint",
            "target",
            "admission_profile",
            "page_tokens",
            "supported_topologies",
        },
    )
    if contract["schema"] != TARGET_CONTRACT_SCHEMA:
        raise ValueError("unsupported OrbitKV executor target schema")
    _exact_version(
        contract["version"],
        TARGET_CONTRACT_VERSION,
        "RuntimeTargetContractV1.version",
    )
    target = _mapping(contract["target"], "RuntimeTargetContractV1.target")
    profile = _mapping(
        contract["admission_profile"],
        "RuntimeTargetContractV1.admission_profile",
    )
    _exact_keys(target, "RuntimeTargetContractV1.target", {"id", "contract_version"})
    _exact_keys(profile, "RuntimeTargetContractV1.admission_profile", {"id", "version"})
    _stable_id(target["id"], "RuntimeTargetContractV1.target.id")
    _bounded_positive_int(
        target["contract_version"],
        "RuntimeTargetContractV1.target.contract_version",
        _U32_MAX,
    )
    _stable_id(profile["id"], "RuntimeTargetContractV1.admission_profile.id")
    _bounded_positive_int(
        profile["version"],
        "RuntimeTargetContractV1.admission_profile.version",
        _U32_MAX,
    )
    _bounded_positive_int(
        contract["page_tokens"], "RuntimeTargetContractV1.page_tokens", _U64_MAX
    )
    topologies = contract["supported_topologies"]
    if (
        not isinstance(topologies, list)
        or not topologies
        or any(not isinstance(item, str) for item in topologies)
        or tuple(topologies) != tuple(sorted(set(topologies)))
        or any(item not in SUPPORTED_TOPOLOGIES for item in topologies)
    ):
        raise ValueError(
            "RuntimeTargetContractV1.supported_topologies must be sorted, unique, and known"
        )
    fingerprint = _sha256(
        contract["fingerprint"], "RuntimeTargetContractV1.fingerprint"
    )
    if fingerprint != _fingerprint(contract):
        raise ValueError("RuntimeTargetContractV1.fingerprint does not match its payload")
    return contract


def execution_signature_from_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project an already validated v1 manifest without losing topology."""

    manager = _mapping(
        manifest.get("token_manager_plan"), "RuntimeManifest.token_manager_plan"
    )
    layout = _mapping(
        manager.get("layout"), "RuntimeManifest.token_manager_plan.layout"
    )
    attention = _mapping(
        manifest.get("attention_state_plan"),
        "RuntimeManifest.attention_state_plan",
    )
    states = attention.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("RuntimeManifest.attention_state_plan.states must be nonempty")
    token_states = []
    fixed_states = []
    for state in states:
        value = _mapping(state, "RuntimeManifest.attention_state_plan.state")
        backend = _mapping(
            value.get("backend"),
            "RuntimeManifest.attention_state_plan.state.backend",
        )
        destination = (
            token_states if backend.get("kind") == "token_slots" else fixed_states
        )
        destination.append(copy.deepcopy(value))
    signature: dict[str, Any] = {
        "schema": EXECUTION_SIGNATURE_SCHEMA,
        "version": EXECUTION_SIGNATURE_VERSION,
        "fingerprint": "",
        "manifest_schema": manifest.get("schema"),
        "manifest_version": manifest.get("version"),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "page_tokens": attention.get("page_tokens"),
        "token_classes": copy.deepcopy(layout.get("classes")),
        "token_states": token_states,
        "fixed_states": fixed_states,
    }
    signature["fingerprint"] = _fingerprint(signature)
    return signature


def _validate_layers(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty layer list")
    layers = [_nonnegative_int(item, path, _U32_MAX) for item in value]
    if layers != sorted(set(layers)):
        raise ValueError(f"{path} must be sorted and unique")
    return layers


def _validate_token_class(value: Any, path: str) -> Mapping[str, Any]:
    item = _mapping(value, path)
    base = {
        "name", "layers", "bytes_per_token_per_layer", "address",
        "retirement", "minimum_slots_per_request",
    }
    optional = {"kv_head_range", "block_domain"}
    if not base <= set(item) or set(item) - base - optional:
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
    address = _mapping(item["address"], f"{path}.address")
    address_kind = address.get("kind")
    address_fields = {
        "append_only": {"kind"}, "pinned": {"kind"},
        "periodic": {"kind", "period_blocks"},
        "periodic_from": {"kind", "period_blocks", "origin_block"},
        "resettable_arena": {"kind", "blocks_per_epoch"},
    }
    if address_kind not in address_fields:
        raise ValueError(f"{path}.address.kind is unsupported")
    _exact_keys(address, f"{path}.address", address_fields[address_kind])
    for key in set(address) - {"kind"}:
        if key == "origin_block":
            _nonnegative_int(address[key], f"{path}.address.{key}", _U64_MAX)
        else:
            _bounded_positive_int(
                address[key], f"{path}.address.{key}", _U64_MAX
            )
    retirement = _mapping(item["retirement"], f"{path}.retirement")
    retirement_kind = retirement.get("kind")
    retirement_fields = {
        "never": {"kind"},
        "block_end_plus": {"kind", "offset_tokens"},
        "epoch_end": {"kind", "blocks_per_epoch"},
    }
    if retirement_kind not in retirement_fields:
        raise ValueError(f"{path}.retirement.kind is unsupported")
    _exact_keys(retirement, f"{path}.retirement", retirement_fields[retirement_kind])
    for key in set(retirement) - {"kind"}:
        _nonnegative_int(retirement[key], f"{path}.retirement.{key}", _U64_MAX)
    slots = item["minimum_slots_per_request"]
    if slots is not None:
        _bounded_positive_int(
            slots, f"{path}.minimum_slots_per_request", _U64_MAX
        )
    return item


def _validate_state(value: Any, path: str, token: bool, page_tokens: int) -> Mapping[str, Any]:
    state = _mapping(value, path)
    _exact_keys(state, path, {"name", "layers", "backend"})
    if not isinstance(state["name"], str) or not state["name"]:
        raise ValueError(f"{path}.name must be nonempty")
    layers = _validate_layers(state["layers"], f"{path}.layers")
    backend = _mapping(state["backend"], f"{path}.backend")
    kind = backend.get("kind")
    if token:
        _exact_keys(backend, f"{path}.backend", {
            "kind", "storage", "components", "bytes_per_token_per_layer",
            "page_bytes_per_layer", "retention", "window_tokens",
            "token_relocatable",
        })
        if kind != "token_slots" or backend["storage"] not in ("token_kv", "latent_kv"):
            raise ValueError(f"{path}.backend is not a token state")
        expected = ("key", "value") if backend["storage"] == "token_kv" else ("latent", "rope")
        components = backend["components"]
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError(f"{path}.backend.components must contain two items")
        widths = []
        for index, (component, name) in enumerate(zip(components, expected, strict=True)):
            component = _mapping(component, f"{path}.backend.components[{index}]")
            _exact_keys(component, f"{path}.backend.components[{index}]", {"name", "bytes_per_token_per_layer"})
            if component["name"] != name:
                raise ValueError(f"{path}.backend component order differs")
            widths.append(
                _bounded_positive_int(
                    component["bytes_per_token_per_layer"],
                    f"{path}.backend component width",
                    _U64_MAX,
                )
            )
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
        if (
            width != sum(widths)
            or width > _U64_MAX // page_tokens
            or page_bytes != width * page_tokens
        ):
            raise ValueError(f"{path}.backend token geometry differs")
        retention = backend["retention"]
        window = backend["window_tokens"]
        if not ((retention == "full" and window is None) or (retention == "sliding" and _positive_int(window, f"{path}.backend.window_tokens"))):
            raise ValueError(f"{path}.backend retention is invalid")
        if backend["token_relocatable"] is not True:
            raise ValueError(f"{path}.backend.token_relocatable must be true")
        return state
    fixed_fields = {
        "recurrent_checkpoints": {"kind", "family", "state_bytes_per_layer", "checkpoint_slots_per_request", "checkpoint_bytes_per_request", "token_relocatable"},
        "convolution_ring": {"kind", "state_bytes_per_layer", "kernel_width", "checkpoint_slots_per_request", "checkpoint_bytes_per_request", "token_relocatable"},
    }
    if kind not in fixed_fields:
        raise ValueError(f"{path}.backend is not a fixed state")
    _exact_keys(backend, f"{path}.backend", fixed_fields[kind])
    state_bytes = _bounded_positive_int(
        backend["state_bytes_per_layer"],
        f"{path}.backend.state_bytes_per_layer",
        _U64_MAX,
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
    if kind == "recurrent_checkpoints" and backend["family"] not in ("mamba", "gdn", "kda", "linear_attention"):
        raise ValueError(f"{path}.backend.family is unsupported")
    if kind == "convolution_ring":
        _bounded_positive_int(
            backend["kernel_width"], f"{path}.backend.kernel_width", _U32_MAX
        )
    return state


def classify_execution_signature(signature: Mapping[str, Any]) -> str:
    classes = signature["token_classes"]
    tokens = signature["token_states"]
    fixed = signature["fixed_states"]
    page_tokens = signature["page_tokens"]

    def whole(item: Mapping[str, Any]) -> bool:
        return "kv_head_range" not in item and "block_domain" not in item

    def canonical_layers(groups: tuple[list[int], ...]) -> bool:
        if any(not group or group != sorted(set(group)) for group in groups):
            return False
        flattened = sorted(layer for group in groups for layer in group)
        return flattened == list(range(len(flattened)))

    def full(index: int, storage: str) -> bool:
        item, state = classes[index], tokens[index]
        backend = state["backend"]
        return (
            whole(item)
            and item["name"] == state["name"]
            and item["layers"] == state["layers"]
            and item["address"] == {"kind": "append_only"}
            and item["retirement"] == {"kind": "never"}
            and item["minimum_slots_per_request"] is None
            and backend["kind"] == "token_slots"
            and backend["storage"] == storage
            and backend["retention"] == "full"
            and backend["window_tokens"] is None
            and backend["token_relocatable"] is True
        )

    def sliding(index: int) -> bool:
        item, state = classes[index], tokens[index]
        backend = state["backend"]
        window = backend.get("window_tokens")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            return False
        period = 1 + (window - 1 + page_tokens - 1) // page_tokens
        return (
            whole(item)
            and item["name"] == state["name"]
            and item["layers"] == state["layers"]
            and item["address"] == {"kind": "periodic", "period_blocks": period}
            and item["retirement"] == {"kind": "block_end_plus", "offset_tokens": window - 1}
            and item["minimum_slots_per_request"] == period
            and backend["kind"] == "token_slots"
            and backend["storage"] == "token_kv"
            and backend["retention"] == "sliding"
            and backend["token_relocatable"] is True
        )

    if not fixed and len(classes) == len(tokens) == 1:
        if full(0, "latent_kv") and canonical_layers((tokens[0]["layers"],)):
            return "whole_domain_full_latent_kv"
        if full(0, "token_kv") and canonical_layers((tokens[0]["layers"],)):
            return "whole_domain_full_token_kv"
        if sliding(0) and canonical_layers((tokens[0]["layers"],)):
            return "whole_domain_sliding_token_kv"
    if (
        not fixed
        and len(classes) == len(tokens) == 2
        and full(0, "token_kv")
        and sliding(1)
        and canonical_layers((tokens[0]["layers"], tokens[1]["layers"]))
    ):
        return "whole_domain_full_sliding_token_kv"
    if len(classes) == len(tokens) == 1 and full(0, "token_kv"):
        recurrent = [
            item
            for item in fixed
            if item["backend"]["kind"] == "recurrent_checkpoints"
        ]
        convolution = [
            item
            for item in fixed
            if item["backend"]["kind"] == "convolution_ring"
        ]
        if (
            recurrent
            and not convolution
            and all(
                item["backend"]["family"] == "mamba"
                and item["backend"]["checkpoint_slots_per_request"] == 2
                and item["backend"]["token_relocatable"] is False
                for item in recurrent
            )
            and canonical_layers(
                tuple(
                    [tokens[0]["layers"]]
                    + [item["layers"] for item in recurrent]
                )
            )
        ):
            return "whole_domain_full_token_kv_mamba"
        if (
            len(recurrent) == len(convolution) == 1
            and recurrent[0]["backend"]["family"] == "gdn"
            and recurrent[0]["backend"]["checkpoint_slots_per_request"] == 2
            and recurrent[0]["backend"]["token_relocatable"] is False
            and convolution[0]["backend"]["checkpoint_slots_per_request"] == 2
            and convolution[0]["backend"]["token_relocatable"] is False
            and recurrent[0]["layers"] == convolution[0]["layers"]
            and canonical_layers((tokens[0]["layers"], recurrent[0]["layers"]))
        ):
            return "whole_domain_full_token_kv_gdn_convolution"
    raise ValueError("RuntimeManifest topology is not executable by the v1 admission profile")


def _validate_execution_signature(signature: Any) -> Mapping[str, Any]:
    value = _mapping(signature, "ExecutionSignatureV1")
    _exact_keys(value, "ExecutionSignatureV1", {
        "schema", "version", "fingerprint", "manifest_schema",
        "manifest_version", "manifest_fingerprint", "page_tokens",
        "token_classes", "token_states", "fixed_states",
    })
    if value["schema"] != EXECUTION_SIGNATURE_SCHEMA:
        raise ValueError("unsupported OrbitKV execution signature schema")
    _exact_version(value["version"], EXECUTION_SIGNATURE_VERSION, "ExecutionSignatureV1.version")
    if value["manifest_schema"] != "orbitkv.runtime-manifest":
        raise ValueError("ExecutionSignatureV1.manifest_schema is unsupported")
    _exact_version(value["manifest_version"], 1, "ExecutionSignatureV1.manifest_version")
    _sha256(value["manifest_fingerprint"], "ExecutionSignatureV1.manifest_fingerprint")
    page_tokens = _bounded_positive_int(
        value["page_tokens"], "ExecutionSignatureV1.page_tokens", _U64_MAX
    )
    for field in ("token_classes", "token_states", "fixed_states"):
        if not isinstance(value[field], list):
            raise ValueError(f"ExecutionSignatureV1.{field} must be a list")
    classes = [
        _validate_token_class(item, f"ExecutionSignatureV1.token_classes[{index}]")
        for index, item in enumerate(value["token_classes"])
    ]
    tokens = [
        _validate_state(item, f"ExecutionSignatureV1.token_states[{index}]", True, page_tokens)
        for index, item in enumerate(value["token_states"])
    ]
    fixed = [
        _validate_state(item, f"ExecutionSignatureV1.fixed_states[{index}]", False, page_tokens)
        for index, item in enumerate(value["fixed_states"])
    ]
    token_roles: set[int] = set()
    fixed_roles: set[tuple[int, str]] = set()
    for state in tokens:
        for layer in state["layers"]:
            if layer in token_roles:
                raise ValueError("execution signature token state roles overlap")
            token_roles.add(layer)
    for state in fixed:
        role = (
            "convolution"
            if state["backend"]["kind"] == "convolution_ring"
            else "recurrent"
        )
        for layer in state["layers"]:
            key = (layer, role)
            if key in fixed_roles:
                raise ValueError("execution signature fixed state roles overlap")
            fixed_roles.add(key)
    if len(classes) != len(tokens):
        raise ValueError("execution signature token projections differ")
    for index, (item, state) in enumerate(zip(classes, tokens, strict=True)):
        backend = state["backend"]
        if (
            item["name"] != state["name"]
            or item["layers"] != state["layers"]
            or item["bytes_per_token_per_layer"]
            != backend["bytes_per_token_per_layer"]
        ):
            raise ValueError(f"execution signature token projection {index} differs")
        window = backend["window_tokens"]
        if backend["retention"] == "full":
            expected_address = {"kind": "append_only"}
            expected_retirement = {"kind": "never"}
            expected_slots = None
        else:
            period = 1 + (window - 1 + page_tokens - 1) // page_tokens
            expected_address = {"kind": "periodic", "period_blocks": period}
            expected_retirement = {"kind": "block_end_plus", "offset_tokens": window - 1}
            expected_slots = period
        if (
            item["address"] != expected_address
            or item["retirement"] != expected_retirement
            or item["minimum_slots_per_request"] != expected_slots
        ):
            raise ValueError(f"execution signature token layout {index} differs")
    names = [item["name"] for item in (*tokens, *fixed)]
    if len(names) != len(set(names)):
        raise ValueError("execution signature state names are not unique")
    _sha256(value["fingerprint"], "ExecutionSignatureV1.fingerprint")
    if value["fingerprint"] != _fingerprint(value):
        raise ValueError("ExecutionSignatureV1.fingerprint does not match its payload")
    return value


def admit_execution_signature(
    signature: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> str:
    """Admit static topology without importing SGLang, Torch, hooks, or FFI."""

    selected = (
        load_executor_capabilities()
        if contract is None
        else _validate_contract(contract)
    )
    signature = _validate_execution_signature(signature)
    if signature.get("page_tokens") != selected["page_tokens"]:
        raise ValueError("RuntimeManifest page_tokens is unsupported by this executor")
    topology = classify_execution_signature(signature)
    if topology not in selected["supported_topologies"]:
        raise ValueError(f"executor target does not support topology {topology!r}")
    return topology


def runtime_target_binding_from_manifest(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    selected = (
        load_executor_capabilities()
        if contract is None
        else _validate_contract(contract)
    )
    signature = execution_signature_from_manifest(manifest)
    topology = admit_execution_signature(signature, selected)
    binding: dict[str, Any] = {
        "schema": TARGET_BINDING_SCHEMA,
        "version": TARGET_BINDING_VERSION,
        "fingerprint": "",
        "manifest_fingerprint": manifest.get("fingerprint"),
        "target": copy.deepcopy(selected["target"]),
        "admission_profile": copy.deepcopy(selected["admission_profile"]),
        "target_contract_fingerprint": selected["fingerprint"],
        "execution_topology": topology,
        "execution_signature": signature,
    }
    binding["fingerprint"] = _fingerprint(binding)
    return binding


def _validate_runtime_target_binding(
    binding: Any,
    signature: Mapping[str, Any],
    manifest_fingerprint: str,
    contract: Mapping[str, Any],
) -> str:
    value = _mapping(binding, "RuntimeTargetBindingV1")
    _exact_keys(
        value,
        "RuntimeTargetBindingV1",
        {
            "schema",
            "version",
            "fingerprint",
            "manifest_fingerprint",
            "target",
            "admission_profile",
            "target_contract_fingerprint",
            "execution_topology",
            "execution_signature",
        },
    )
    if value["schema"] != TARGET_BINDING_SCHEMA:
        raise ValueError("unsupported OrbitKV runtime target binding")
    _exact_version(value["version"], TARGET_BINDING_VERSION, "RuntimeTargetBindingV1.version")
    if _sha256(value["fingerprint"], "RuntimeTargetBindingV1.fingerprint") != _fingerprint(value):
        raise ValueError("RuntimeTargetBindingV1.fingerprint does not match its payload")
    if (
        value["manifest_fingerprint"] != manifest_fingerprint
        or value["target_contract_fingerprint"] != contract["fingerprint"]
        or value["target"] != contract["target"]
        or value["admission_profile"] != contract["admission_profile"]
        or value["execution_signature"] != signature
    ):
        raise ValueError("runtime target binding does not match its manifest or contract")
    topology = admit_execution_signature(signature, contract)
    if value["execution_topology"] != topology:
        raise ValueError("runtime target binding topology differs from its signature")
    return topology


def admit_runtime_config(config: Any) -> str:
    signature = getattr(config, "execution_signature", None)
    binding = getattr(config, "runtime_target_binding", None)
    manifest_fingerprint = getattr(config, "runtime_manifest_fingerprint", None)
    if signature is None or binding is None or manifest_fingerprint is None:
        raise ValueError(
            "executor target admission requires ORBITKV_RUNTIME_MANIFEST; "
            "legacy plans do not carry a target binding"
        )
    if signature.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError("execution signature does not match the loaded manifest")
    return _validate_runtime_target_binding(
        binding, signature, manifest_fingerprint, load_executor_capabilities()
    )


__all__ = [
    "admit_execution_signature",
    "admit_runtime_config",
    "classify_execution_signature",
    "execution_signature_from_manifest",
    "load_executor_capabilities",
    "runtime_target_binding_from_manifest",
]
