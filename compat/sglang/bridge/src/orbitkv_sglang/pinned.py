from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path


_SUPPORTED_SGLANG_RELEASE = "v0.5.17"
_SUPPORTED_SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
_REVIEWED_PATCH_PATH = "compat/sglang/overlay/adapter.patch"
_PATCHED_SOURCE_INVENTORY: tuple[Mapping[str, str], ...] = (
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
)
_REVIEWED_PATCH_SHA256 = (
    "daa070dc687d3611b845bdeefaaa2da97568f477bc4fd919293cde83b7a94720"
)
__all__ = [
    "apply_reviewed_patch",
    "pinned_source_contract",
    "validate_base_checkout",
    "validate_patched_checkout",
    "validate_patched_source",
]

_ASSEMBLY_MANIFEST = "orbitkv-engine-manifest.json"
_ASSEMBLY_SCHEMA = "orbitkv.engine-source-manifest"
_ASSEMBLY_VERSION = 2
_HEX = frozenset("0123456789abcdef")


def pinned_source_contract() -> dict[str, object]:
    """Return the canonical live contract for the reviewed pinned patch."""

    return {
        "release": _SUPPORTED_SGLANG_RELEASE,
        "revision": _SUPPORTED_SGLANG_REVISION,
        "patch_path": _REVIEWED_PATCH_PATH,
        "targets": [dict(target) for target in _PATCHED_SOURCE_INVENTORY],
        "patch_diff_sha256": _REVIEWED_PATCH_SHA256,
    }


def validate_base_checkout(root: Path | str) -> Path:
    """Require the complete pristine pinned tree the overlay applies to."""

    checkout = _checkout(root)
    _validate_revision(checkout)
    _validate_materialized_source_tree(checkout)
    changes, untracked = _source_tree_state(checkout)
    if changes or untracked:
        raise RuntimeError("pinned SGLang source tree is not pristine")
    _validate_source_inventory(checkout, "base_sha256", "base")
    return checkout


def validate_patched_checkout(root: Path | str) -> Path:
    """Require exactly the reviewed OrbitKV patch and no other edits."""

    checkout = _checkout(root)
    _validate_revision(checkout)
    _validate_materialized_source_tree(checkout)
    changes, untracked = _source_tree_state(checkout)
    expected = tuple(("M", target["path"]) for target in _PATCHED_SOURCE_INVENTORY)
    if changes != expected or untracked:
        raise RuntimeError(
            "pinned SGLang source tree must contain exactly the reviewed "
            "OrbitKV patch"
        )
    _validate_source_inventory(checkout, "patched_sha256", "patched")
    diff = _reviewed_patch_diff(checkout)
    if hashlib.sha256(diff).hexdigest() != _REVIEWED_PATCH_SHA256:
        raise RuntimeError("SGLang source diff is not the reviewed OrbitKV patch")
    return checkout


def validate_patched_source(root: Path | str) -> Path:
    """Validate either the reviewed Git checkout or an assembled product.

    A present assembly manifest is authoritative: malformed assembled products
    never fall back to the Git path.
    """

    checkout = _checkout(root)
    manifest = checkout / _ASSEMBLY_MANIFEST
    if manifest.exists() or manifest.is_symlink():
        _validate_assembled_source(checkout, manifest)
        return checkout
    return validate_patched_checkout(checkout)


def _inventory_digest(entries: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["kind"].encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry["sha256"]))
    return digest.hexdigest()


def _safe_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"assembled source {label} is invalid")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise RuntimeError(f"assembled source {label} is not canonical")
    return path


def _entry_identity(path: Path) -> tuple[str, str]:
    try:
        status = path.lstat()
    except OSError as error:
        raise RuntimeError(f"assembled source file is missing: {path}") from error
    if stat.S_ISLNK(status.st_mode):
        return "symlink", hashlib.sha256(os.readlink(path).encode()).hexdigest()
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"assembled source path has unsupported type: {path}")
    kind = "executable" if status.st_mode & 0o111 else "file"
    return kind, hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_assembled_source(root: Path, manifest_path: Path) -> None:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("assembled source manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("assembled source manifest is invalid") from error
    contract = pinned_source_contract()
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _ASSEMBLY_SCHEMA
        or manifest.get("schema_version") != _ASSEMBLY_VERSION
        or manifest.get("source_revision") != contract["revision"]
        or manifest.get("reviewed_patch_sha256") != contract["patch_diff_sha256"]
    ):
        raise RuntimeError("assembled source contract differs from the pinned product")
    components = manifest.get("components")
    expected_roots = {
        "sglang": ".",
        "orbitkv-product": "orbitkv",
        "orbitkv-adapter": "orbitkv/adapter",
        "orbitkv-manager": "orbitkv/manager",
    }
    if not isinstance(components, dict) or set(components) != set(expected_roots):
        raise RuntimeError("assembled source component set differs")
    actual_paths: set[str] = set()
    aggregate: list[dict[str, str]] = []
    overlay_hashes = {item["path"]: item["patched_sha256"] for item in contract["targets"]}
    for name, component in components.items():
        if not isinstance(component, dict) or component.get("root") != expected_roots[name]:
            raise RuntimeError(f"assembled source component {name} is invalid")
        entries = component.get("files")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"assembled source component {name} inventory is empty")
        normalized: list[dict[str, str]] = []
        previous = ""
        for raw in entries:
            if not isinstance(raw, dict) or set(raw) != {"path", "kind", "sha256"}:
                raise RuntimeError("assembled source inventory entry is invalid")
            relative = _safe_relative(raw["path"], "inventory path").as_posix()
            kind, digest = raw["kind"], raw["sha256"]
            if (
                relative <= previous
                or kind not in ("file", "executable", "symlink")
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in _HEX for char in digest)
                or _entry_identity(root / expected_roots[name] / relative)
                != (kind, digest)
            ):
                raise RuntimeError(f"assembled source inventory differs: {relative}")
            previous = relative
            component_root = expected_roots[name]
            product_path = (
                relative if component_root == "." else f"{component_root}/{relative}"
            )
            if product_path in actual_paths or product_path == _ASSEMBLY_MANIFEST:
                raise RuntimeError("assembled source inventory aliases a path")
            actual_paths.add(product_path)
            normalized.append({"path": relative, "kind": kind, "sha256": digest})
            aggregate.append({"path": product_path, "kind": kind, "sha256": digest})
        if (
            component.get("file_count") != len(normalized)
            or component.get("inventory_sha256") != _inventory_digest(normalized)
        ):
            raise RuntimeError(f"assembled source component {name} summary differs")
        if name == "sglang":
            by_path = {entry["path"]: entry["sha256"] for entry in normalized}
            if any(by_path.get(path) != digest for path, digest in overlay_hashes.items()):
                raise RuntimeError("assembled source reviewed overlay differs")
    aggregate.sort(key=lambda item: item["path"])
    if (
        manifest.get("file_count") != len(aggregate)
        or manifest.get("file_inventory_sha256") != _inventory_digest(aggregate)
    ):
        raise RuntimeError("assembled source aggregate inventory differs")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_dir()
    }
    if observed != actual_paths | {_ASSEMBLY_MANIFEST}:
        raise RuntimeError("assembled source contains missing or extra paths")


def apply_reviewed_patch(root: Path | str, patch_file: Path | str) -> Path:
    """Apply the reviewed OrbitKV patch to a pristine pinned checkout."""

    checkout = validate_base_checkout(root)
    patch = Path(patch_file).expanduser().resolve(strict=True)
    if not patch.is_file():
        raise RuntimeError("OrbitKV patch is not a regular file")
    patch_bytes = patch.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != _REVIEWED_PATCH_SHA256:
        raise RuntimeError("OrbitKV patch artifact has an unexpected hash")
    _git(checkout, "apply", "--check", "--whitespace=error-all", str(patch))
    _git(checkout, "apply", "--whitespace=error-all", str(patch))
    return validate_patched_checkout(checkout)


def _reviewed_patch_diff(checkout: Path) -> bytes:
    return _git(
        checkout,
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--full-index",
        "--binary",
        "--unified=3",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--",
        *(target["path"] for target in _PATCHED_SOURCE_INVENTORY),
    )


def _checkout(root: Path | str) -> Path:
    try:
        checkout = Path(root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid SGLang checkout {root}: {error}") from error
    if not checkout.is_dir() or not (checkout / "python/sglang/__init__.py").is_file():
        raise RuntimeError("path is not an SGLang source checkout")
    return checkout


def _validate_revision(checkout: Path) -> None:
    revision = _git(checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if revision != _SUPPORTED_SGLANG_REVISION:
        raise RuntimeError(
            f"SGLang revision {revision!r} is not pinned revision "
            f"{_SUPPORTED_SGLANG_REVISION}"
        )


def _source_tree_state(
    checkout: Path,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
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
    changes = tuple(
        (
            fields[index].decode("ascii", errors="replace"),
            fields[index + 1].decode("utf-8"),
        )
        for index in range(0, len(fields), 2)
    )
    untracked = tuple(
        item.decode("utf-8")
        for item in _git(
            checkout,
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
            "--",
        ).split(b"\0")
        if item
    )
    return changes, untracked


def _validate_materialized_source_tree(checkout: Path) -> None:
    """Reject sparse, deleted, or type/mode-changed upstream paths."""

    records = _git(
        checkout, "ls-tree", "-rz", "--full-tree", "HEAD"
    ).split(b"\0")
    if not any(records):
        raise RuntimeError("pinned SGLang HEAD tree is empty")
    for record in records:
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, _object_id = metadata.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("cannot parse pinned SGLang HEAD tree") from error
        if (
            not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise RuntimeError("pinned SGLang HEAD contains an invalid path")
        path = checkout / relative
        try:
            entry = path.lstat()
        except OSError as error:
            raise RuntimeError(
                f"pinned SGLang tracked path is not materialized: {relative}"
            ) from error
        if object_type == "blob" and mode == "120000":
            valid = stat.S_ISLNK(entry.st_mode)
        elif object_type == "blob" and mode in {"100644", "100755"}:
            valid = stat.S_ISREG(entry.st_mode)
            if valid:
                valid = bool(entry.st_mode & stat.S_IXUSR) == (mode == "100755")
        elif object_type == "commit" and mode == "160000":
            valid = stat.S_ISDIR(entry.st_mode)
        else:
            raise RuntimeError(
                f"unsupported pinned SGLang tracked object: "
                f"{relative} ({mode} {object_type})"
            )
        if not valid:
            raise RuntimeError(
                f"pinned SGLang tracked path type or mode changed: {relative}"
            )


def _validate_source_inventory(checkout: Path, digest_key: str, state: str) -> None:
    for identity in _PATCHED_SOURCE_INVENTORY:
        relative = identity["path"]
        if _source_sha256(checkout, relative) != identity[digest_key]:
            raise RuntimeError(
                f"{state} SGLang source {relative} has an unexpected hash"
            )


def _source_sha256(checkout: Path, relative: str) -> str:
    source = checkout / relative
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"pinned SGLang source {relative} is missing or is a symlink")
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(f"cannot read the pinned SGLang source {relative}") from error


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot verify or patch pinned SGLang checkout{detail}") from error
    return completed.stdout
