"""Frozen source-identity validation for current qualification evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from orbitkv_sglang.qualification_primitives import (
    canonical_json_sha256,
    canonical_relative_path,
    require_exact_keys,
    require_sha256,
)


_PINNED_SOURCE_CONTRACT = {
    "release": "v0.5.17",
    "revision": "29481685462732237d80d86076d6563e1f658102",
    "patch_path": "compat/sglang/overlay/adapter.patch",
    "targets": [
        {
            "path": "python/sglang/srt/layers/attention/flashattention_backend.py",
            "base_sha256": (
                "0ff44cdea99e6dba99b63c7979f79a2764f0b70e26f00a2142b9ef01884cc261"
            ),
            "patched_sha256": (
                "6d1adab7d71fded1e9a5e07e16c91630e7f8d987b5e6e3700ea8b5c43e2685bb"
            ),
        },
        {
            "path": "python/sglang/srt/managers/schedule_batch.py",
            "base_sha256": (
                "7bdd988f3274bb11b3b5fd699b985ae27c28ff9e2e7f601a6320e182efcf1595"
            ),
            "patched_sha256": (
                "67615b423c34bbcce8882ecf0ee849f830f1eeed26d0d817c57fb2cee32464db"
            ),
        },
        {
            "path": "python/sglang/srt/managers/scheduler.py",
            "base_sha256": (
                "1d4a7683e32f0862cf5b03be33e5ec23f9d112e59546d391e28fa222fcd1619b"
            ),
            "patched_sha256": (
                "0a2618321ffa2602fe11620fd9976def84b3d8de76fb74adbe1d2788ec7a0ab7"
            ),
        },
        {
            "path": "python/sglang/srt/mem_cache/allocation.py",
            "base_sha256": (
                "940a05f4031c81794f3a8ac89fce60da1da6eebaf4c3fc25228ced77fa042690"
            ),
            "patched_sha256": (
                "f7210c242b564cfaf62ba94846ca106b1ae00a403b483735a9d35a143620fce2"
            ),
        },
        {
            "path": "python/sglang/srt/mem_cache/common.py",
            "base_sha256": (
                "58fe6b674f784c832be6ba2d41c0d7b1f849416acee103d60af438fc0d39fdee"
            ),
            "patched_sha256": (
                "8ac0df0fa9cf356651a77fa622155ee482538384a18a9da5a63820fb82ec6457"
            ),
        },
        {
            "path": "python/sglang/srt/mem_cache/kv_cache_configurator.py",
            "base_sha256": (
                "a99b6e25efcd7c56a72c215fcd98ba7fe70bd53a37e8e4b8cec6205d3c45bdaf"
            ),
            "patched_sha256": (
                "1f0d502570061ddff2de4dad075e136068734d44b7090127726ccf69e1e56c23"
            ),
        },
    ],
    "patch_diff_sha256": (
        "daa070dc687d3611b845bdeefaaa2da97568f477bc4fd919293cde83b7a94720"
    ),
}
_REVIEWED_PATCH_BYTES = 37199
_SOURCE_KEYS = {
    "contract", "pinned_contract", "direct_source", "sglang_python_sha256",
    "harness", "checkpoint_identity_helper_sha256", "adapter", "build_tool",
    "library",
}
_PROFILE_ARTIFACT_KEYS = {"path", "bytes", "sha256"}
_DIRECT_TAKEOVER_POINTS = (
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure",
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool_allocator",
    "sglang.srt.mem_cache.allocation.alloc_for_extend",
    "sglang.srt.mem_cache.allocation.alloc_for_decode",
    "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
    "sglang.srt.managers.scheduler.Scheduler._prepare_waiting_request_removal",
    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    "sglang.srt.managers.scheduler.Scheduler.run_batch",
    "sglang.srt.managers.scheduler.Scheduler.get_internal_state",
    "sglang.srt.mem_cache.common.release_kv_cache",
)
_OUTER_CONTRACT_KEYS = {
    "root", "release", "revision", "contract", "contract_sha256",
    "reviewed_patch",
}
_PINNED_CONTRACT_KEYS = {
    "release", "revision", "targets", "patch_path", "patch_diff_sha256",
}


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    try:
        require_exact_keys(value, keys, label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    assert isinstance(value, dict)
    return value


def _sha(value: Any, label: str) -> str:
    try:
        return require_sha256(value, label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(
        mark in value for mark in ("\x00", "\r", "\n")
    ):
        raise RuntimeError(f"{label} must be a nonempty single-line string")
    return value


def _absolute(value: Any, label: str) -> str:
    text = _text(value, label)
    path = Path(text)
    if (
        "\\" in text
        or not path.is_absolute()
        or str(path) != text
        or ".." in path.parts
    ):
        raise RuntimeError(f"{label} must be a canonical absolute path")
    return text


def _artifact(value: Any, label: str, *, wire: bool = False) -> None:
    keys = {"path", "bytes", "sha256", "wire_version"} if wire else {
        "path", "sha256",
    }
    item = _exact(value, keys, label)
    _absolute(item["path"], f"{label}.path")
    _sha(item["sha256"], f"{label}.sha256")
    if wire:
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise RuntimeError(f"{label}.bytes must be positive")
        if type(item["wire_version"]) is not int or item["wire_version"] <= 0:
            raise RuntimeError(f"{label} has an invalid wire version")


def _adapter(value: Any) -> None:
    adapter = _exact(value, {"files"}, "source_identity.adapter")
    files = adapter["files"]
    if not isinstance(files, list) or not files:
        raise RuntimeError("source_identity.adapter.files must be nonempty")
    paths = []
    for index, raw in enumerate(files):
        item = _exact(raw, {"path", "sha256"}, f"adapter.files[{index}]")
        try:
            path = canonical_relative_path(item["path"]).as_posix()
        except ValueError as error:
            raise RuntimeError(f"adapter source path is unsafe: {error}") from error
        _sha(item["sha256"], f"adapter.files[{index}].sha256")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise RuntimeError("adapter source paths must be sorted and unique")


def direct_source_identity(mode: str) -> dict[str, object]:
    """Return the canonical direct-source selection for one run mode."""

    if mode == "manager":
        return {
            "kind": "orbitkv_direct_sglang_source",
            "owner": "orbitkv_sglang.engine:get_owner",
            "takeover_points": list(_DIRECT_TAKEOVER_POINTS),
        }
    if mode == "stock":
        return {
            "kind": "stock_sglang_source",
            "owner": None,
            "takeover_points": [],
        }
    raise RuntimeError(f"unknown source verification mode: {mode!r}")


def validate_source_identity(
    value: Any, mode: str, *, native: bool = False
) -> dict[str, Any]:
    """Validate the frozen source contract and return its pair projection."""

    if mode not in {"stock", "manager"}:
        raise RuntimeError(f"unknown source verification mode: {mode!r}")
    expected_keys = (
        _SOURCE_KEYS | {"profile_artifact"} if native else _SOURCE_KEYS
    )
    source = _exact(value, expected_keys, "source_identity")
    outer = _exact(
        source["contract"], _OUTER_CONTRACT_KEYS, "source_identity.contract"
    )
    _absolute(outer["root"], "source_identity.contract.root")
    if (
        outer["release"] != _PINNED_SOURCE_CONTRACT["release"]
        or outer["revision"] != _PINNED_SOURCE_CONTRACT["revision"]
    ):
        raise RuntimeError("source identity release or revision differs")
    contract = _exact(
        outer["contract"], _PINNED_CONTRACT_KEYS, "pinned source contract"
    )
    if contract != _PINNED_SOURCE_CONTRACT:
        raise RuntimeError("pinned source contract differs")
    if source["pinned_contract"] != contract:
        raise RuntimeError("pinned source contract copies differ")
    if outer["contract_sha256"] != canonical_json_sha256(contract):
        raise RuntimeError("pinned source contract digest differs")
    patch = _exact(
        outer["reviewed_patch"],
        {"status", "sha256", "bytes"},
        "source_identity.contract.reviewed_patch",
    )
    if mode == "manager":
        if (
            patch["status"] != "applied"
            or patch["sha256"] != _PINNED_SOURCE_CONTRACT["patch_diff_sha256"]
            or patch["bytes"] != _REVIEWED_PATCH_BYTES
        ):
            raise RuntimeError("manager reviewed patch identity differs")
        _artifact(source["library"], "source_identity.library", wire=True)
    else:
        if patch != {"status": "absent", "sha256": None, "bytes": 0}:
            raise RuntimeError("stock reviewed patch identity differs")
        if source["library"] is not None:
            raise RuntimeError("stock source identity contains a manager library")
    direct_source = _exact(
        source["direct_source"],
        {"kind", "owner", "takeover_points"},
        "source_identity.direct_source",
    )
    if dict(direct_source) != direct_source_identity(mode):
        raise RuntimeError(f"{mode} direct-source identity differs")
    _sha(source["sglang_python_sha256"], "source_identity.sglang_python_sha256")
    _sha(
        source["checkpoint_identity_helper_sha256"],
        "source_identity.checkpoint_identity_helper_sha256",
    )
    _artifact(source["harness"], "source_identity.harness")
    if native:
        profile_artifact = _exact(
            source["profile_artifact"],
            _PROFILE_ARTIFACT_KEYS,
            "source_identity.profile_artifact",
        )
        _absolute(
            profile_artifact["path"],
            "source_identity.profile_artifact.path",
        )
        if type(profile_artifact["bytes"]) is not int or profile_artifact["bytes"] <= 0:
            raise RuntimeError(
                "source_identity.profile_artifact.bytes must be positive"
            )
        _sha(
            profile_artifact["sha256"],
            "source_identity.profile_artifact.sha256",
        )
    _adapter(source["adapter"])
    tool = _exact(
        source["build_tool"], {"path", "version", "sha256"},
        "source_identity.build_tool",
    )
    _absolute(tool["path"], "source_identity.build_tool.path")
    _text(tool["version"], "source_identity.build_tool.version")
    _sha(tool["sha256"], "source_identity.build_tool.sha256")
    normalized = dict(source)
    normalized.pop("library")
    normalized.pop("direct_source")
    if native:
        normalized["profile_artifact"] = {
            "bytes": source["profile_artifact"]["bytes"],
            "sha256": source["profile_artifact"]["sha256"],
        }
    normalized["contract"] = {
        "release": outer["release"],
        "revision": outer["revision"],
        "contract": contract,
        "contract_sha256": outer["contract_sha256"],
    }
    return normalized


def _command_root(command: Any) -> str:
    if not isinstance(command, list) or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise RuntimeError("record.command is invalid")
    roots: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item == "--sglang-root":
            if index + 1 >= len(command):
                raise RuntimeError("command --sglang-root has no value")
            roots.append(command[index + 1])
            index += 2
            continue
        if item.startswith("--sglang-root="):
            roots.append(item.split("=", 1)[1])
        index += 1
    if len(roots) != 1:
        raise RuntimeError("command must contain exactly one --sglang-root")
    return _absolute(roots[0], "command --sglang-root")


def validate_source_binding(
    value: Any,
    mode: str,
    *,
    command: Any,
    environment: Mapping[str, Any],
    sglang_package: Any,
    native: bool = False,
) -> dict[str, Any]:
    """Validate location claims before returning location-free identity."""

    projection = validate_source_identity(value, mode, native=native)
    assert isinstance(value, dict)
    root = value["contract"]["root"]
    expected_package = str(Path(root) / "python/sglang/__init__.py")
    if _command_root(command) != root:
        raise RuntimeError("command --sglang-root differs from source root")
    if _absolute(sglang_package, "runtime_identity.sglang_package") != expected_package:
        raise RuntimeError(
            "runtime SGLang package is not the source checkout package"
        )
    if mode == "manager" and environment.get("ORBITKV_SGLANG_ROOT") != root:
        raise RuntimeError(
            "manager ORBITKV_SGLANG_ROOT differs from source root"
        )
    return projection


__all__ = [
    "direct_source_identity",
    "validate_source_binding",
    "validate_source_identity",
]
