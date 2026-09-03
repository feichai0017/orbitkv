#!/usr/bin/env python3
"""Read-only validation for the full-source OrbitKV Engine overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "compat/sglang/profile.json"
INTEGRATION_SOURCE = ROOT / "compat/sglang/bridge/src"
DEFAULT_SGLANG_ROOT = ROOT / "compat/sglang/source"

_PROFILE_KEYS = {
    "schema",
    "schema_version",
    "product_id",
    "mode",
    "product_scope",
    "upstream",
    "source_preservation",
    "overlay_targets",
    "capabilities",
    "manager_supported_profiles",
    "manager_unsupported_profiles",
    "manager_takeover_points",
    "lifecycle_notification_points",
    "observability_extensions",
    "compatibility_fixes",
    "license_notice_policy",
}
_UPSTREAM_KEYS = {"source_contract"}
_SOURCE_CONTRACT_KEYS = {"provider"}
_PINNED_CONTRACT_KEYS = {
    "release",
    "revision",
    "patch_path",
    "targets",
    "patch_diff_sha256",
}
_PINNED_TARGET_KEYS = {"path", "base_sha256", "patched_sha256"}
_POINT_KEYS = {"source_path", "symbol"}
_COMPATIBILITY_FIX_KEYS = {"source_path", "symbol", "purpose"}
_SUPPORTED_PROFILE_KEYS = {
    "execution_topology",
    "ordered_attention_classes",
    "cache_policy",
    "execution",
    "device_scope",
    "plan_format",
    "token_reclamation",
    "fixed_state",
}
_LICENSE_NOTICE_KEYS = {
    "upstream_license",
    "license_path",
    "license_action",
    "notice_path",
    "notice_action_when_present",
    "notice_action_when_absent",
}
_REQUIRED_CAPABILITIES = {
    "continuous-batched-text-generation",
    "full-attention",
    "http-generation-api",
    "multi-head-latent-attention",
    "paged-token-kv",
    "prefix-cache",
    "runtime-observability",
    "sliding-window-attention",
}
_CACHE_POLICIES = frozenset({"request_private", "shared_prefix"})
_SUPPORTED_PROFILES = (
    {
        "execution_topology": "whole_domain_full_token_kv",
        "ordered_attention_classes": ["full:token_kv"],
        "cache_policy": "shared_prefix",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "kv_plan",
        "token_reclamation": "off",
        "fixed_state": "none",
    },
    {
        "execution_topology": "whole_domain_full_sliding_token_kv",
        "ordered_attention_classes": [
            "full:token_kv",
            "sliding:token_kv",
        ],
        "cache_policy": "shared_prefix",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "kv_plan",
        "token_reclamation": "off",
        "fixed_state": "none",
    },
    {
        "execution_topology": "whole_domain_sliding_token_kv",
        "ordered_attention_classes": ["sliding:token_kv"],
        "cache_policy": "request_private",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "kv_plan",
        "token_reclamation": "off",
        "fixed_state": "none",
    },
    {
        "execution_topology": "whole_domain_chunked_token_kv",
        "ordered_attention_classes": ["chunked:token_kv"],
        "cache_policy": "request_private",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "retention_ir",
        "token_reclamation": "off",
        "fixed_state": "none",
    },
    {
        "execution_topology": "whole_domain_full_latent_kv",
        "ordered_attention_classes": ["full:latent_kv"],
        "cache_policy": "request_private",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "kv_plan",
        "token_reclamation": "off",
        "fixed_state": "none",
    },
)
_UNSUPPORTED_PROFILES = (
    "whole_domain_full_token_kv_gdn_convolution",
    "whole_domain_full_token_kv_mamba",
)

# source path, public symbol, owner method, preserved native method
_MANAGER_TAKEOVER_CONTRACT = (
    (
        "python/sglang/srt/mem_cache/kv_cache_configurator.py",
        "KVCacheConfigurator.configure",
        "configure",
        "_orbitkv_native_configure",
    ),
    (
        "python/sglang/srt/mem_cache/kv_cache_configurator.py",
        "KVCacheConfigurator._build_token_to_kv_pool_allocator",
        "build_allocator",
        None,
    ),
    (
        "python/sglang/srt/mem_cache/allocation.py",
        "alloc_for_extend",
        "prepare_extend",
        None,
    ),
    (
        "python/sglang/srt/mem_cache/allocation.py",
        "alloc_for_decode",
        "prepare_decode",
        None,
    ),
    (
        "python/sglang/srt/managers/schedule_batch.py",
        "ScheduleBatch.maybe_evict_swa",
        "maybe_evict_swa",
        None,
    ),
    (
        "python/sglang/srt/mem_cache/common.py",
        "release_kv_cache",
        "release_request",
        None,
    ),
)
_LIFECYCLE_NOTIFICATION_CONTRACT = (
    (
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler._prepare_waiting_request_removal",
        "prepare_waiting_request_removal",
        None,
    ),
    (
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler.get_next_batch_to_run",
        "next_batch",
        "_orbitkv_native_get_next_batch_to_run",
    ),
    (
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler.run_batch",
        "run_batch",
        "_orbitkv_native_run_batch",
    ),
)
_OBSERVABILITY_EXTENSION_CONTRACT = (
    (
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler.get_internal_state",
        "internal_state",
        "_orbitkv_native_get_internal_state",
    ),
)
_COMPATIBILITY_FIX_CONTRACT = (
    {
        "source_path": (
            "python/sglang/srt/layers/attention/flashattention_backend.py"
        ),
        "symbol": "make_local_attention_virtual_batches",
        "purpose": "preserve-page-id-addressing-for-local-attention",
    },
)


def _object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise RuntimeError(
            f"{label} has an invalid schema: "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(mark in value for mark in ("\x00", "\r", "\n"))
    ):
        raise RuntimeError(f"{label} must be a nonempty single-line string")
    return value


def _relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in ("", ".", "..") for part in path.parts)
        or any(mark in text for mark in ("*", "?", "[", "]"))
    ):
        raise RuntimeError(f"{label} must be a canonical relative POSIX path")
    return text


def _git_path(value: str, label: str) -> str:
    """Validate a literal Git path without treating legal glob marks specially."""

    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RuntimeError(f"{label} must be a canonical relative POSIX path")
    return value


def _string_set(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must be a nonempty array")
    items = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))
    if list(items) != sorted(set(items)):
        raise RuntimeError(f"{label} must be sorted and unique")
    return items


def _path_set(value: Any, label: str) -> tuple[str, ...]:
    return tuple(
        _relative_path(item, f"{label}[{index}]")
        for index, item in enumerate(_string_set(value, label))
    )


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _symbol_node(source: Path, symbol: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        nodes: Sequence[ast.stmt] = ast.parse(
            source.read_text(encoding="utf-8"), filename=str(source)
        ).body
    except (OSError, SyntaxError, UnicodeError) as error:
        raise RuntimeError(f"cannot parse overlay source {source}: {error}") from error
    parts = symbol.split(".")
    for index, part in enumerate(parts):
        found = [
            node
            for node in nodes
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == part
        ]
        if len(found) != 1:
            return None
        node = found[0]
        final = index == len(parts) - 1
        if (not final and not isinstance(node, ast.ClassDef)) or (
            final and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            return None
        nodes = node.body
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node


def _has_direct_dispatch(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    owner_method: str,
    native_symbol: str | None,
) -> bool:
    imports = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.ImportFrom)
        and item.module == "orbitkv_sglang.engine"
        and any(alias.name == "get_owner" and alias.asname is None for alias in item.names)
    ]
    calls = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == owner_method
        and isinstance(item.func.value, ast.Call)
        and isinstance(item.func.value.func, ast.Name)
        and item.func.value.func.id == "get_owner"
    ]
    if len(imports) != 1 or len(calls) != 1:
        return False
    if native_symbol is None:
        return True
    return any(
        isinstance(item, ast.Attribute) and item.attr == native_symbol
        for item in calls[0].args
    )


def _is_self_attribute(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _validate_waiting_removal_dispatch(source: Path) -> None:
    specifications = (
        ("Scheduler._abort_on_queued_limit", "candidate_req", "idx"),
        ("Scheduler._abort_on_waiting_timeout", "req", "index"),
        ("Scheduler.abort_request", "req", "i"),
    )
    for symbol, request_name, index_name in specifications:
        node = _symbol_node(source, symbol)
        if node is None:
            raise RuntimeError(f"waiting removal symbol is missing: {symbol}")
        cleanup = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and _is_self_attribute(item.func, "_prepare_waiting_request_removal")
            and len(item.args) == 1
            and isinstance(item.args[0], ast.Name)
            and item.args[0].id == request_name
        ]
        removals = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "pop"
            and _is_self_attribute(item.func.value, "waiting_queue")
            and len(item.args) == 1
            and isinstance(item.args[0], ast.Name)
            and item.args[0].id == index_name
        ]
        if len(cleanup) != 1 or len(removals) != 1 or cleanup[0].lineno >= removals[0].lineno:
            raise RuntimeError(
                "waiting request owner cleanup must run exactly once before "
                f"queue removal: {symbol}"
            )
    retry = _symbol_node(source, "Scheduler._retry_waiting_request_removals")
    planner = _symbol_node(source, "Scheduler._orbitkv_native_get_next_batch_to_run")
    if retry is None or planner is None:
        raise RuntimeError("waiting request removal retry path is missing")
    retry_calls = [
        item
        for item in ast.walk(planner)
        if isinstance(item, ast.Call)
        and _is_self_attribute(item.func, "_retry_waiting_request_removals")
    ]
    admission_calls = [
        item
        for item in ast.walk(planner)
        if isinstance(item, ast.Call) and _is_self_attribute(item.func, "get_new_batch_prefill")
    ]
    if (
        len(retry_calls) != 1
        or not admission_calls
        or retry_calls[0].lineno >= min(item.lineno for item in admission_calls)
    ):
        raise RuntimeError("waiting request removal retries must run before admission")
    source_text = source.read_text(encoding="utf-8")
    required_fragments = (
        "if self._prepare_waiting_request_removal(candidate_req):",
        "if not self._prepare_waiting_request_removal(req):",
        "self._defer_waiting_request_removal(candidate_req, output)",
        "self._defer_waiting_request_removal(req, output)",
        "getattr(req, \"_orbitkv_waiting_removal_pending\", False)",
    )
    if any(fragment not in source_text for fragment in required_fragments):
        raise RuntimeError("waiting request removal must gate pop and retain retry state")


def _load_source_contract(provider: str) -> Mapping[str, Any]:
    if provider.count(":") != 1:
        raise RuntimeError("source contract provider must be module:callable")
    module_name, callable_name = provider.split(":", 1)
    source = INTEGRATION_SOURCE.joinpath(*module_name.split(".")).with_suffix(".py")
    try:
        resolved_source = source.resolve(strict=True)
        resolved_integration = INTEGRATION_SOURCE.resolve(strict=True)
        if source.is_symlink() or not resolved_source.is_relative_to(resolved_integration):
            raise RuntimeError("source contract provider escapes integration source")
        spec = importlib.util.spec_from_file_location(
            "_orbitkv_engine_profile_source_contract", resolved_source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("source contract provider cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        callback = getattr(module, callable_name)
        contract = callback()
    except RuntimeError:
        raise
    except (ImportError, AttributeError, OSError, TypeError) as error:
        raise RuntimeError(f"cannot load source contract provider {provider}: {error}") from error
    if not isinstance(contract, dict):
        raise RuntimeError("source contract provider must return an object")
    return contract


def _checkout_file(checkout: Path, relative: str, label: str) -> Path:
    candidate = checkout / relative
    current = checkout
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} must not be a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is missing: {relative}") from error
    if not resolved.is_relative_to(checkout) or not resolved.is_file():
        raise RuntimeError(f"{label} must be a contained regular file: {relative}")
    return resolved


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot verify pinned SGLang checkout{detail}") from error
    return completed.stdout


def _decode_path(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} is not UTF-8") from error


def _head_inventory(checkout: Path) -> dict[str, tuple[str, str]]:
    raw = _git(checkout, "ls-tree", "-rz", "--full-tree", "HEAD")
    inventory: dict[str, tuple[str, str]] = {}
    for raw_record in raw.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, _object_id = metadata.decode("ascii").split(" ")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("cannot parse pinned SGLang HEAD tree") from error
        path = _git_path(_decode_path(raw_path, "tracked path"), "tracked path")
        if path in inventory:
            raise RuntimeError(f"duplicate tracked path in pinned SGLang HEAD: {path}")
        inventory[path] = (mode, object_type)
    if not inventory:
        raise RuntimeError("pinned SGLang HEAD tree is empty")
    return inventory


def _validate_materialized_tree(
    checkout: Path, inventory: Mapping[str, tuple[str, str]]
) -> None:
    for relative, (mode, object_type) in inventory.items():
        path = checkout / relative
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RuntimeError(f"upstream tracked path is not materialized: {relative}") from error
        if object_type == "blob" and mode == "120000":
            valid = stat.S_ISLNK(metadata.st_mode)
        elif object_type == "blob" and mode in {"100644", "100755"}:
            valid = stat.S_ISREG(metadata.st_mode)
            if valid:
                expected_executable = mode == "100755"
                valid = bool(metadata.st_mode & stat.S_IXUSR) == expected_executable
        elif object_type == "commit" and mode == "160000":
            valid = stat.S_ISDIR(metadata.st_mode)
        else:
            raise RuntimeError(
                f"unsupported tracked object in pinned SGLang HEAD: "
                f"{relative} ({mode} {object_type})"
            )
        if not valid:
            raise RuntimeError(
                f"upstream tracked path type or mode changed: {relative}"
            )


def _tracked_changes(checkout: Path) -> tuple[tuple[str, str], ...]:
    fields = _git(
        checkout,
        "diff",
        "--name-status",
        "-z",
        "--no-renames",
        "HEAD",
        "--",
    ).split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("cannot parse pinned SGLang tracked changes")
    changes: list[tuple[str, str]] = []
    for index in range(0, len(fields), 2):
        status_text = fields[index].decode("ascii", errors="replace")
        relative = _git_path(
            _decode_path(fields[index + 1], "changed tracked path"),
            "changed tracked path",
        )
        changes.append((status_text, relative))
    return tuple(changes)


def _validate_full_source_overlay(checkout: Path, overlay_targets: Sequence[str]) -> None:
    try:
        top_level = Path(
            _git(checkout, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        ).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError("cannot resolve pinned SGLang Git root") from error
    if top_level != checkout:
        raise RuntimeError("SGLang source root must be the complete Git checkout root")

    inventory = _head_inventory(checkout)
    _validate_materialized_tree(checkout, inventory)
    changes = _tracked_changes(checkout)
    invalid_kinds = [(kind, path) for kind, path in changes if kind != "M"]
    if invalid_kinds:
        raise RuntimeError(
            "pinned SGLang overlay must not add, delete, rename, or change "
            f"tracked path types: {invalid_kinds}"
        )
    modified = tuple(path for _kind, path in changes)
    if tuple(sorted(modified)) != tuple(overlay_targets):
        raise RuntimeError(
            "modified tracked paths must equal overlay_targets: "
            f"actual={sorted(modified)}, expected={list(overlay_targets)}"
        )
    untracked_raw = _git(
        checkout, "ls-files", "-z", "--others", "--exclude-standard", "--"
    )
    untracked = sorted(
        _decode_path(item, "untracked path")
        for item in untracked_raw.split(b"\0")
        if item
    )
    if untracked:
        raise RuntimeError(f"pinned SGLang overlay has unexpected untracked paths: {untracked}")


def _validate_points(
    value: Any,
    label: str,
    expected_contract: Sequence[tuple[str, str, str, str | None]],
    checkout: Path,
) -> tuple[tuple[str, str, str, str | None], ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must be a nonempty array")
    actual: list[tuple[str, str, str, str | None]] = []
    for index, raw in enumerate(value):
        point = _object(raw, _POINT_KEYS, f"{label}[{index}]")
        source_path = _relative_path(point["source_path"], f"{label}[{index}].source_path")
        symbol = _text(point["symbol"], f"{label}[{index}].symbol")
        matches = [item for item in expected_contract if item[:2] == (source_path, symbol)]
        if len(matches) != 1:
            raise RuntimeError(f"{label} differs from its direct-source contract")
        contract = matches[0]
        source = _checkout_file(checkout, source_path, f"{label} source")
        node = _symbol_node(source, symbol)
        if node is None:
            raise RuntimeError(f"overlay symbol is missing: {source_path}:{symbol}")
        if not _has_direct_dispatch(node, contract[2], contract[3]):
            raise RuntimeError(
                f"overlay symbol does not dispatch through get_owner: {source_path}:{symbol}"
            )
        if contract[3] is not None:
            owner_path, _method = symbol.rsplit(".", 1)
            native_symbol = f"{owner_path}.{contract[3]}"
            if _symbol_node(source, native_symbol) is None:
                raise RuntimeError(
                    f"preserved native implementation is missing: {source_path}:{native_symbol}"
                )
        actual.append(contract)
    if tuple(actual) != tuple(expected_contract):
        raise RuntimeError(f"{label} differs from its direct-source contract")
    return tuple(actual)


def _validate_supported_profiles(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("manager_supported_profiles must be a nonempty array")
    normalized = []
    for index, raw in enumerate(value):
        item = dict(
            _object(
                raw,
                _SUPPORTED_PROFILE_KEYS,
                f"manager_supported_profiles[{index}]",
            )
        )
        classes = item["ordered_attention_classes"]
        if not isinstance(classes, list) or not classes:
            raise RuntimeError(
                f"manager_supported_profiles[{index}].ordered_attention_classes "
                "must be a nonempty array"
            )
        item["ordered_attention_classes"] = [
            _text(entry, f"manager_supported_profiles[{index}].ordered_attention_classes")
            for entry in classes
        ]
        for key in _SUPPORTED_PROFILE_KEYS - {"ordered_attention_classes"}:
            item[key] = _text(item[key], f"manager_supported_profiles[{index}].{key}")
        if item["cache_policy"] not in _CACHE_POLICIES:
            raise RuntimeError(
                f"manager_supported_profiles[{index}].cache_policy is unsupported"
            )
        normalized.append(item)
    if tuple(normalized) != _SUPPORTED_PROFILES:
        raise RuntimeError(
            "manager_supported_profiles differ from the qualified RuntimeSession boundary"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def load_profile(path: Path | str = PROFILE) -> Mapping[str, Any]:
    profile_path = Path(path)
    try:
        value = json.loads(
            profile_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"cannot load engine profile {profile_path}: {error}") from error
    return _object(value, _PROFILE_KEYS, "engine profile")


def validate_profile(value: Any, sglang_root: Path | str) -> dict[str, int | str]:
    profile = _object(value, _PROFILE_KEYS, "engine profile")
    if profile["schema"] != "orbitkv.engine-product-profile":
        raise RuntimeError("engine profile has an unsupported schema")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 4:
        raise RuntimeError("engine profile has an unsupported schema version")
    for key, expected in (
        ("product_id", "orbitkv-engine"),
        ("mode", "pinned-source-overlay"),
        ("product_scope", "inference-engine"),
        ("source_preservation", "all-upstream-tracked-paths"),
    ):
        if profile[key] != expected:
            raise RuntimeError(f"engine profile {key} must be {expected!r}")

    capabilities = set(_string_set(profile["capabilities"], "capabilities"))
    if capabilities != _REQUIRED_CAPABILITIES:
        raise RuntimeError("engine profile capabilities differ from the verified set")
    _validate_supported_profiles(profile["manager_supported_profiles"])
    unsupported = _string_set(
        profile["manager_unsupported_profiles"], "manager_unsupported_profiles"
    )
    if unsupported != _UNSUPPORTED_PROFILES:
        raise RuntimeError(
            "manager_unsupported_profiles differ from the current RuntimeSession boundary"
        )
    supported_topologies = {item["execution_topology"] for item in _SUPPORTED_PROFILES}
    if supported_topologies & set(unsupported):
        raise RuntimeError("supported and unsupported manager profiles overlap")

    upstream = _object(profile["upstream"], _UPSTREAM_KEYS, "upstream")
    source_reference = _object(
        upstream["source_contract"], _SOURCE_CONTRACT_KEYS, "upstream.source_contract"
    )
    provider = _text(source_reference["provider"], "upstream.source_contract.provider")
    if provider != "orbitkv_sglang.pinned:pinned_source_contract":
        raise RuntimeError("engine profile must reference pinned_source_contract")
    contract = _object(
        _load_source_contract(provider), _PINNED_CONTRACT_KEYS, "pinned_source_contract"
    )
    _text(contract["release"], "pinned_source_contract.release")
    revision = _text(contract["revision"], "pinned_source_contract.revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("pinned_source_contract.revision must be a lowercase Git object id")
    patch_path = _relative_path(contract["patch_path"], "pinned_source_contract.patch_path")
    patch_digest = _sha256(
        contract["patch_diff_sha256"], "pinned_source_contract.patch_diff_sha256"
    )
    patch_artifact = _checkout_file(ROOT, patch_path, "source contract patch artifact")
    if hashlib.sha256(patch_artifact.read_bytes()).hexdigest() != patch_digest:
        raise RuntimeError("source contract patch artifact digest differs")

    targets = contract["targets"]
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("pinned_source_contract targets must be nonempty")
    target_contracts: list[Mapping[str, Any]] = []
    target_paths: list[str] = []
    for index, raw_target in enumerate(targets):
        target = _object(raw_target, _PINNED_TARGET_KEYS, f"source contract target {index}")
        relative = _relative_path(target["path"], f"source contract target {index}.path")
        _sha256(target["base_sha256"], f"source contract target {index}.base_sha256")
        _sha256(target["patched_sha256"], f"source contract target {index}.patched_sha256")
        target_contracts.append(target)
        target_paths.append(relative)
    if target_paths != sorted(set(target_paths)):
        raise RuntimeError("source contract patch targets must be sorted and unique")
    overlay_targets = _path_set(profile["overlay_targets"], "overlay_targets")
    if overlay_targets != tuple(target_paths):
        raise RuntimeError("overlay_targets must exactly equal pinned source contract targets")

    try:
        checkout = Path(sglang_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid SGLang source root {sglang_root}: {error}") from error
    if not checkout.is_dir():
        raise RuntimeError("SGLang source root must be a directory")
    actual_revision = _git(checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if actual_revision != revision:
        raise RuntimeError(
            f"SGLang checkout revision {actual_revision!r} differs from pinned_source_contract"
        )
    _validate_full_source_overlay(checkout, overlay_targets)

    categories = (
        ("manager_takeover_points", _MANAGER_TAKEOVER_CONTRACT),
        ("lifecycle_notification_points", _LIFECYCLE_NOTIFICATION_CONTRACT),
        ("observability_extensions", _OBSERVABILITY_EXTENSION_CONTRACT),
    )
    all_points = []
    for label, expected in categories:
        all_points.extend(_validate_points(profile[label], label, expected, checkout))
    identities = [(item[0], item[1]) for item in all_points]
    if len(identities) != len(set(identities)):
        raise RuntimeError("integration point categories overlap")
    if any(path not in overlay_targets for path, _symbol in identities):
        raise RuntimeError("integration point source must be an overlay target")

    fixes = profile["compatibility_fixes"]
    if not isinstance(fixes, list) or not fixes:
        raise RuntimeError("compatibility_fixes must be a nonempty array")
    normalized_fixes = []
    for index, raw in enumerate(fixes):
        fix = dict(_object(raw, _COMPATIBILITY_FIX_KEYS, f"compatibility_fixes[{index}]"))
        fix["source_path"] = _relative_path(
            fix["source_path"], f"compatibility_fixes[{index}].source_path"
        )
        fix["symbol"] = _text(fix["symbol"], f"compatibility_fixes[{index}].symbol")
        fix["purpose"] = _text(fix["purpose"], f"compatibility_fixes[{index}].purpose")
        normalized_fixes.append(fix)
    if tuple(normalized_fixes) != _COMPATIBILITY_FIX_CONTRACT:
        raise RuntimeError("compatibility_fixes differ from the reviewed overlay contract")
    for fix in normalized_fixes:
        if fix["source_path"] not in overlay_targets:
            raise RuntimeError("compatibility fix source must be an overlay target")
        source = _checkout_file(checkout, fix["source_path"], "compatibility fix source")
        if _symbol_node(source, fix["symbol"]) is None:
            raise RuntimeError(
                f"compatibility fix symbol is missing: {fix['source_path']}:{fix['symbol']}"
            )
    classified_sources = {path for path, _symbol in identities} | {
        fix["source_path"] for fix in normalized_fixes
    }
    if classified_sources != set(overlay_targets):
        raise RuntimeError(
            "every overlay target must be classified as an integration point "
            "or compatibility fix"
        )

    _validate_waiting_removal_dispatch(
        _checkout_file(
            checkout,
            "python/sglang/srt/managers/scheduler.py",
            "waiting request removal source",
        )
    )

    for target in target_contracts:
        relative = str(target["path"])
        target_path = _checkout_file(checkout, relative, "source contract patch target")
        digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
        if digest != target["patched_sha256"]:
            raise RuntimeError(f"patched source contract target digest differs: {relative}")

    policy = _object(
        profile["license_notice_policy"], _LICENSE_NOTICE_KEYS, "license_notice_policy"
    )
    required_policy = {
        "upstream_license": "Apache-2.0",
        "license_path": "LICENSE",
        "license_action": "retain-verbatim",
        "notice_path": "NOTICE",
        "notice_action_when_present": "retain-verbatim",
        "notice_action_when_absent": "do-not-synthesize",
    }
    if dict(policy) != required_policy:
        raise RuntimeError("license_notice_policy differs from required policy")
    license_path = _checkout_file(checkout, "LICENSE", "upstream Apache-2.0 LICENSE")
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read upstream LICENSE: {error}") from error
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("upstream LICENSE is not Apache-2.0")
    notice = checkout / "NOTICE"
    if notice.exists() or notice.is_symlink():
        _checkout_file(checkout, "NOTICE", "upstream NOTICE")

    return {
        "product_id": str(profile["product_id"]),
        "capabilities": len(capabilities),
        "overlay_targets": len(overlay_targets),
        "manager_supported_profiles": len(_SUPPORTED_PROFILES),
        "manager_unsupported_profiles": len(unsupported),
        "manager_takeover_points": len(_MANAGER_TAKEOVER_CONTRACT),
        "lifecycle_notification_points": len(_LIFECYCLE_NOTIFICATION_CONTRACT),
        "observability_extensions": len(_OBSERVABILITY_EXTENSION_CONTRACT),
        "compatibility_fixes": len(_COMPATIBILITY_FIX_CONTRACT),
    }


def verify_profile(
    profile_path: Path | str = PROFILE,
    sglang_root: Path | str = DEFAULT_SGLANG_ROOT,
) -> dict[str, int | str]:
    return validate_profile(load_profile(profile_path), sglang_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the full pinned SGLang source plus OrbitKV overlay."
    )
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    arguments = parser.parse_args(argv)
    try:
        summary = verify_profile(arguments.profile, arguments.sglang_root)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    integration_points = (
        int(summary["manager_takeover_points"])
        + int(summary["lifecycle_notification_points"])
        + int(summary["observability_extensions"])
    )
    print(
        "OrbitKV Engine profile passed: "
        f"product={summary['product_id']} "
        f"overlay_targets={summary['overlay_targets']} "
        f"integration_points={integration_points} "
        f"compatibility_fixes={summary['compatibility_fixes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
