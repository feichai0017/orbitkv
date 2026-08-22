from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


PAGE_TOKENS = 16
RetentionKind = Literal["full", "sliding"]
ReclamationMode = Literal["off", "naive", "relocate"]


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

    @property
    def kernel_window_left(self) -> int | None:
        return None if self.window_tokens is None else self.window_tokens - 1

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
class ManagerPlanConfig:
    """The sole breaking ABI7 snapshot/prefix plan; no legacy translation."""

    plan_path: Path
    library_path: Path
    plan_json: bytes
    plan_fingerprint: str
    page_tokens: int
    classes: tuple[ClassConfig, ...]
    token_reclamation: TokenReclamationConfig = TokenReclamationConfig()

    @property
    def num_hidden_layers(self) -> int:
        return sum(len(item.layers) for item in self.classes)

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
    """Load only the ABI7 Full, sliding, or Full+sliding plan schema."""

    source = os.environ if environ is None else environ
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
        _class_config(index, value, page_tokens)
        for index, value in enumerate(raw_classes)
    )
    retentions = tuple(item.retention for item in classes)
    if retentions not in (("full",), ("sliding",), ("full", "sliding")):
        raise ValueError(
            "KV classes must be Full, sliding, or ordered Full then sliding"
        )

    layers = [layer for item in classes for layer in item.layers]
    if len(set(layers)) != len(layers):
        raise ValueError("KV classes overlap in model-layer ownership")
    if sorted(layers) != list(range(len(layers))):
        raise ValueError("KV classes must cover every model layer exactly once")

    canonical = _canonical_json(root)
    token_reclamation = _token_reclamation_config(source)
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
    return ManagerPlanConfig(
        plan_path=plan_path,
        library_path=library_path,
        plan_json=canonical,
        plan_fingerprint="sha256:" + hashlib.sha256(canonical).hexdigest(),
        page_tokens=page_tokens,
        classes=classes,
        token_reclamation=token_reclamation,
    )


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
    return TokenReclamationConfig(
        mode=mode,
        fragmentation_threshold_milli=threshold,
        **values,
    )


def _class_config(index: int, raw: Any, page_tokens: int) -> ClassConfig:
    path = f"KvPlanInput.classes[{index}]"
    value = _mapping(raw, path)
    base_fields = {
        "name",
        "layers",
        "retention",
        "bytes_per_token_per_layer",
        "window_tokens",
    }
    retention = _string(value, "retention", path)
    if retention == "full":
        _exact_keys(value, path, base_fields)
        if value.get("window_tokens") is not None:
            raise ValueError(f"{path}.window_tokens must be null for full retention")
        expected_name = "full"
        window_tokens = None
        period_blocks = None
    elif retention == "sliding":
        _exact_keys(value, path, base_fields | {"window_tokens"})
        expected_name = "swa"
        window_tokens = _positive_int(value, "window_tokens", path)
        period_blocks = 1 + _ceil_div(window_tokens - 1, page_tokens)
    else:
        raise ValueError(f"{path}.retention must be 'full' or 'sliding'")
    name = _string(value, "name", path)
    if name != expected_name:
        raise ValueError(
            f"{path}.name must be {expected_name!r} for {retention} retention"
        )

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


def _exact_keys(value: Mapping[str, Any], path: str, required: set[str]) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required
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


__all__ = [
    "ClassConfig",
    "ManagerPlanConfig",
    "PAGE_TOKENS",
    "ReclamationMode",
    "RetentionKind",
    "TokenReclamationConfig",
    "load_config",
]
