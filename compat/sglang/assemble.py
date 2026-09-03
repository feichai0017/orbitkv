#!/usr/bin/env python3
"""Assemble and verify the complete pinned SGLang source with OrbitKV."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_PROJECT = ROOT / "compat/sglang/bridge"
INTEGRATION_SOURCE = INTEGRATION_PROJECT / "src"
MANAGER_PROJECT = ROOT / "core"
MANIFEST_NAME = "orbitkv-engine-manifest.json"
COMPONENT_ROOT = "orbitkv"
PRODUCT_INCLUDED = ("profile.json",)
ADAPTER_INCLUDED = ("pyproject.toml", "src")
MANAGER_INCLUDED = (
    "Cargo.toml",
    "examples",
    "ffi/Cargo.lock",
    "ffi/Cargo.toml",
    "ffi/include",
    "ffi/src",
    "ffi/tests",
    "fixtures/hybrid-fixed-state-large/PROVENANCE.md",
    "fixtures/hybrid-fixed-state-large/config.json",
    "fixtures/hybrid-fixed-state-small/config.json",
    "src",
    "tests",
)
_REGULAR_MODES = {"100644", "100755"}
_HEX_DIGITS = frozenset("0123456789abcdef")
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)

sys.path.insert(0, str(INTEGRATION_SOURCE))

from orbitkv_sglang.pinned import (  # noqa: E402
    pinned_source_contract,
    validate_base_checkout,
    validate_patched_checkout,
)


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot inspect source checkout{detail}") from error


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RuntimeError(f"{label} is unsafe: {value!r}")
    return value


def _decode_path(value: bytes) -> str:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("tracked SGLang path is not UTF-8") from error
    return _safe_relative(text, "tracked SGLang path")


def _tracked_modes(root: Path) -> dict[str, str]:
    raw = _git(root, "ls-files", "-s", "-z")
    modes: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, _object_id, stage = metadata.decode("ascii").split(" ")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("cannot parse tracked SGLang inventory") from error
        path = _decode_path(raw_path)
        if stage != "0" or mode not in {*_REGULAR_MODES, "120000"}:
            raise RuntimeError(f"unsupported tracked SGLang entry: {path} ({mode})")
        if path in modes:
            raise RuntimeError(f"duplicate tracked SGLang path: {path}")
        modes[path] = mode
    return modes


def _open_regular(root_fd: int, relative: str, label: str) -> int:
    """Open one regular file without following a symlink in any path component."""

    parts = PurePosixPath(_safe_relative(relative, label)).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        result = os.open(parts[-1], _READ_FLAGS, dir_fd=current)
    except OSError as error:
        raise RuntimeError(f"{label} is not a contained regular file: {relative}") from error
    finally:
        os.close(current)
    metadata = os.fstat(result)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(result)
        raise RuntimeError(f"{label} is not a contained regular file: {relative}")
    return result


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular(root_fd: int, relative: str, label: str) -> tuple[bytes, int]:
    fd = _open_regular(root_fd, relative, label)
    try:
        metadata = os.fstat(fd)
        return _read_fd(fd), metadata.st_mode
    finally:
        os.close(fd)


def _symlink_target(root_fd: int, relative: str, label: str) -> str:
    """Read a symlink itself, then require a relative target confined to root."""

    parts = PurePosixPath(_safe_relative(relative, label)).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        probe = os.open(
            parts[-1],
            getattr(os, "O_PATH", os.O_RDONLY)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        try:
            if not stat.S_ISLNK(os.fstat(probe).st_mode):
                raise RuntimeError(f"{label} changed type: {relative}")
            target = os.readlink("", dir_fd=probe)
        finally:
            os.close(probe)
    except OSError as error:
        raise RuntimeError(f"{label} is not a stable symlink: {relative}") from error
    finally:
        os.close(current)
    target_path = PurePosixPath(target)
    if target_path.is_absolute() or not target:
        raise RuntimeError(f"{label} target escapes the source tree: {relative}")
    depth = len(parts) - 1
    for part in target_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                raise RuntimeError(f"{label} target escapes the source tree: {relative}")
        else:
            depth += 1
    return target


def _write_regular(destination: Path, payload: bytes, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _copy_upstream(source: Path, destination: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    modes = _tracked_modes(source)
    paths = tuple(sorted(modes))
    if any(path == COMPONENT_ROOT or path.startswith(f"{COMPONENT_ROOT}/") for path in paths):
        raise RuntimeError("SGLang source collides with the OrbitKV component root")
    if MANIFEST_NAME in paths:
        raise RuntimeError("SGLang source collides with the engine manifest")
    identities: dict[str, str] = {}
    source_fd = os.open(source, _DIRECTORY_FLAGS)
    try:
        for relative in paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = modes[relative]
            if mode == "120000":
                link = _symlink_target(source_fd, relative, "tracked SGLang symlink")
                target.symlink_to(link)
                payload = link.encode("utf-8")
            else:
                payload, actual_mode = _read_regular(
                    source_fd, relative, "tracked SGLang file"
                )
                executable = bool(actual_mode & stat.S_IXUSR)
                if executable != (mode == "100755"):
                    raise RuntimeError(f"tracked SGLang file mode changed: {relative}")
                _write_regular(target, payload, 0o755 if executable else 0o644)
            identities[relative] = hashlib.sha256(payload).hexdigest()
    finally:
        os.close(source_fd)
    return paths, identities


def _component_files(source: Path, included: Sequence[str]) -> tuple[str, ...]:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"component source is not a directory: {source}")
    candidates: list[Path] = []
    for relative in included:
        relative = _safe_relative(relative, "component include path")
        candidate = source / relative
        if candidate.is_symlink():
            raise RuntimeError(f"component source must not be a symlink: {candidate}")
        if candidate.is_file():
            candidates.append(candidate)
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.is_symlink():
                    raise RuntimeError(f"component source must not be a symlink: {path}")
                if path.is_file():
                    candidates.append(path)
        else:
            raise RuntimeError(f"component source is missing: {candidate}")
    files = tuple(
        sorted(
            {
                path.relative_to(source).as_posix()
                for path in candidates
                if "__pycache__" not in path.parts
                and not any(
                    part.endswith((".egg-info", ".dist-info")) for part in path.parts
                )
            }
        )
    )
    if not files:
        raise RuntimeError(f"component source is empty: {source}")
    return files


def _copy_component(
    source: Path, destination: Path, included: Sequence[str]
) -> tuple[str, ...]:
    files = _component_files(source, included)
    source_identity = _source_identity(source)
    source_fd = os.open(source, _DIRECTORY_FLAGS)
    try:
        for relative in files:
            payload, _mode = _read_regular(source_fd, relative, "component source")
            _write_regular(destination / relative, payload, 0o644)
    finally:
        os.close(source_fd)
    if _component_files(source, included) != files or _source_identity(source) != source_identity:
        raise RuntimeError(f"component source changed during assembly: {source}")
    return files


def _copy_manager_component(source: Path, destination: Path) -> tuple[str, ...]:
    """Copy the manager closure and its authoritative workspace lock."""

    files = _copy_component(source, destination, MANAGER_INCLUDED)
    if "Cargo.lock" in files:
        return files
    lock = ROOT / "Cargo.lock"
    if lock.is_symlink() or not lock.is_file():
        raise RuntimeError("workspace Cargo.lock is missing or not a regular file")
    payload = lock.read_bytes()
    _write_regular(destination / "Cargo.lock", payload, 0o644)
    if lock.read_bytes() != payload:
        raise RuntimeError("workspace Cargo.lock changed during assembly")
    return tuple(sorted((*files, "Cargo.lock")))


def _apply_patch(destination: Path, patch_payload: bytes) -> None:
    patch_file = destination.parent / f".{destination.name}.reviewed.patch"
    _write_regular(patch_file, patch_payload, 0o600)
    try:
        for arguments in (
            ("apply", "--check", "--whitespace=error-all", str(patch_file)),
            ("apply", "--whitespace=error-all", str(patch_file)),
        ):
            subprocess.run(
                ["git", *arguments],
                cwd=destination,
                check=True,
                capture_output=True,
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot apply reviewed OrbitKV overlay{detail}") from error
    finally:
        patch_file.unlink(missing_ok=True)


def _reviewed_patch_payload(contract: Mapping[str, Any]) -> bytes:
    raw = _safe_relative(contract.get("patch_path"), "pinned source contract patch_path")
    root_fd = os.open(ROOT, _DIRECTORY_FLAGS)
    try:
        payload, _mode = _read_regular(root_fd, raw, "reviewed OrbitKV patch")
    finally:
        os.close(root_fd)
    expected = contract.get("patch_diff_sha256")
    if not isinstance(expected, str) or hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("reviewed OrbitKV patch digest changed")
    return payload


def _overlay_identities(contract: Mapping[str, Any]) -> dict[str, str]:
    targets = contract.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("pinned source contract targets are invalid")
    identities: dict[str, str] = {}
    for target in targets:
        if not isinstance(target, dict):
            raise RuntimeError("pinned source contract target is invalid")
        relative = _safe_relative(target.get("path"), "pinned source target path")
        expected = target.get("patched_sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not set(expected) <= _HEX_DIGITS
        ):
            raise RuntimeError("pinned source contract target identity is invalid")
        if relative in identities:
            raise RuntimeError("pinned source contract target is duplicated")
        identities[relative] = expected
    if tuple(identities) != tuple(sorted(identities)):
        raise RuntimeError("pinned source contract targets are not sorted")
    return identities


def _validate_overlay_targets(output: Path, contract: Mapping[str, Any]) -> None:
    root = output.resolve(strict=True)
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    try:
        for relative, expected in _overlay_identities(contract).items():
            payload, _mode = _read_regular(root_fd, relative, "assembled overlay target")
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError(f"assembled overlay target differs: {relative}")
    finally:
        os.close(root_fd)


def _tree_changes(checkout: Path) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    fields = _git(
        checkout, "diff", "--name-status", "-z", "--no-renames", "HEAD", "--"
    ).split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("cannot parse SGLang tracked changes")
    changes = tuple(
        (fields[index].decode("ascii"), _decode_path(fields[index + 1]))
        for index in range(0, len(fields), 2)
    )
    untracked = tuple(
        _decode_path(item)
        for item in _git(
            checkout, "ls-files", "-z", "--others", "--exclude-standard", "--"
        ).split(b"\0")
        if item
    )
    return changes, untracked


def _reviewed_diff(checkout: Path, targets: Sequence[str]) -> bytes:
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
        *targets,
    )


def _validate_input_state(
    checkout: Path, contract: Mapping[str, Any], source_state: str
) -> None:
    identities = _overlay_identities(contract)
    changes, untracked = _tree_changes(checkout)
    if untracked:
        raise RuntimeError("SGLang source contains untracked paths")
    if source_state == "base":
        if changes:
            raise RuntimeError("pinned base SGLang checkout has source mutations")
        return
    expected = tuple(("M", path) for path in identities)
    if changes != expected:
        raise RuntimeError("patched SGLang changes do not exactly match overlay targets")
    digest = hashlib.sha256(_reviewed_diff(checkout, tuple(identities))).hexdigest()
    if digest != contract.get("patch_diff_sha256"):
        raise RuntimeError("patched SGLang diff is not the reviewed OrbitKV overlay")
    _validate_overlay_targets(checkout, contract)


def _entry_identity(root: Path, relative: str) -> tuple[str, str]:
    path = root / relative
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"assembled inventory path is missing: {relative}") from error
    if stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        payload = os.readlink(path).encode("utf-8")
    elif stat.S_ISREG(metadata.st_mode):
        kind = "executable" if metadata.st_mode & stat.S_IXUSR else "file"
        payload = path.read_bytes()
    else:
        raise RuntimeError(f"assembled inventory path is invalid: {relative}")
    return kind, hashlib.sha256(payload).hexdigest()


def _inventory(root: Path, paths: Sequence[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for relative in paths:
        kind, digest = _entry_identity(root, relative)
        result.append({"path": relative, "kind": kind, "sha256": digest})
    return result


def _inventory_digest(entries: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8") + b"\0")
        digest.update(entry["kind"].encode("ascii") + b"\0")
        digest.update(bytes.fromhex(entry["sha256"]))
    return digest.hexdigest()


def _source_identity(root: Path) -> dict[str, object]:
    try:
        repository = _git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        if Path(repository).resolve(strict=True) != root.resolve(strict=True):
            return {"git_repository": False, "revision": None, "tree": None}
        revision = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
        tree = _git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    except RuntimeError:
        return {"git_repository": False, "revision": None, "tree": None}
    return {"git_repository": True, "revision": revision, "tree": tree}


def _dirty_identity(root: Path) -> str | None:
    """Bind component bytes to the Git worktree state that supplied them."""

    try:
        top = Path(
            _git(root, "rev-parse", "--show-toplevel")
            .decode("utf-8")
            .strip()
        ).resolve(strict=True)
        relative = root.resolve(strict=True).relative_to(top)
        arguments = ("status", "--porcelain=v1", "-z", "--untracked-files=all")
        status = _git(top, *arguments, "--", relative.as_posix())
    except (RuntimeError, ValueError):
        return None
    return hashlib.sha256(status).hexdigest()


def _orbitkv_source_identity() -> dict[str, object]:
    identity = _source_identity(ROOT)
    identity["worktree_status_sha256"] = _dirty_identity(ROOT)
    return identity


def _component_manifest(
    output: Path,
    relative_root: str,
    paths: Sequence[str],
    source_identity: Mapping[str, object],
) -> dict[str, Any]:
    entries = _inventory(output / relative_root, paths)
    return {
        "root": relative_root,
        "source": dict(source_identity),
        "file_count": len(entries),
        "inventory_sha256": _inventory_digest(entries),
        "files": entries,
    }


def _manifest(
    output: Path,
    contract: Mapping[str, Any],
    source_tree: str,
    upstream_paths: Sequence[str],
    adapter_paths: Sequence[str],
    product_paths: Sequence[str],
    manager_paths: Sequence[str],
    orbitkv_identity: Mapping[str, object],
) -> dict[str, Any]:
    components = {
        "orbitkv-product": _component_manifest(
            output, COMPONENT_ROOT, product_paths, orbitkv_identity
        ),
        "orbitkv-adapter": _component_manifest(
            output, f"{COMPONENT_ROOT}/adapter", adapter_paths, orbitkv_identity
        ),
        "orbitkv-manager": _component_manifest(
            output, f"{COMPONENT_ROOT}/manager", manager_paths, orbitkv_identity
        ),
        "sglang": {
            "root": ".",
            "source": {"git_repository": True, "revision": contract["revision"], "tree": source_tree},
            "file_count": len(upstream_paths),
            "inventory_sha256": _inventory_digest(_inventory(output, upstream_paths)),
            "files": _inventory(output, upstream_paths),
        },
    }
    all_entries = sorted(
        (
            {"path": f"{component['root']}/{entry['path']}".removeprefix("./"), "kind": entry["kind"], "sha256": entry["sha256"]}
            for component in components.values()
            for entry in component["files"]
        ),
        key=lambda entry: entry["path"],
    )
    return {
        "schema": "orbitkv.engine-source-manifest",
        "schema_version": 2,
        "source_revision": contract["revision"],
        "source_tree": source_tree,
        "reviewed_patch_sha256": contract["patch_diff_sha256"],
        "orbitkv_source": dict(orbitkv_identity),
        "file_count": len(all_entries),
        "file_inventory_sha256": _inventory_digest(all_entries),
        "components": components,
    }


def _actual_paths(root: Path) -> set[str]:
    result: set[str] = set()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in tuple(directories):
            path = base / name
            if path.is_symlink():
                result.add(path.relative_to(root).as_posix())
                directories.remove(name)
        result.update((base / name).relative_to(root).as_posix() for name in files)
    return result


def _manifest_error(message: str) -> NoReturn:
    raise RuntimeError(f"invalid assembly manifest: {message}")


def verify_assembly(output: Path | str) -> Path:
    """Verify every assembled file, mode, symlink and component boundary."""

    try:
        root = Path(output).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid assembly root: {output}") from error
    if not root.is_dir():
        raise RuntimeError("assembly root is not a directory")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        _manifest_error("manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid assembly manifest: cannot parse manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "orbitkv.engine-source-manifest" or manifest.get("schema_version") != 2:
        _manifest_error("unsupported schema")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != {"sglang", "orbitkv-product", "orbitkv-adapter", "orbitkv-manager"}:
        _manifest_error("component set differs")
    expected_roots = {
        "sglang": ".",
        "orbitkv-product": COMPONENT_ROOT,
        "orbitkv-adapter": f"{COMPONENT_ROOT}/adapter",
        "orbitkv-manager": f"{COMPONENT_ROOT}/manager",
    }
    orbitkv_source = manifest.get("orbitkv_source")
    if not isinstance(orbitkv_source, dict):
        _manifest_error("OrbitKV source identity is missing")
    expected_paths: set[str] = set()
    aggregate: list[dict[str, str]] = []
    for name, component in components.items():
        if not isinstance(component, dict):
            _manifest_error(f"component {name} is invalid")
        component_root = component.get("root")
        if component_root != expected_roots[name]:
            _manifest_error(f"component {name} root differs")
        source = component.get("source")
        if name == "sglang":
            expected_source = {
                "git_repository": True,
                "revision": manifest.get("source_revision"),
                "tree": manifest.get("source_tree"),
            }
            if source != expected_source:
                _manifest_error("SGLang source identity differs")
        elif source != orbitkv_source:
            _manifest_error(f"component {name} source identity differs")
        if component_root == ".":
            prefix = ""
        else:
            component_root = _safe_relative(component_root, f"component {name} root")
            prefix = f"{component_root}/"
        entries = component.get("files")
        if not isinstance(entries, list) or not entries:
            _manifest_error(f"component {name} inventory is empty")
        normalized: list[dict[str, str]] = []
        previous = ""
        for raw_entry in entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {"path", "kind", "sha256"}:
                _manifest_error(f"component {name} inventory entry is invalid")
            relative = _safe_relative(raw_entry.get("path"), f"component {name} path")
            kind = raw_entry.get("kind")
            digest = raw_entry.get("sha256")
            if (
                relative <= previous
                or kind not in {"file", "executable", "symlink"}
                or not isinstance(digest, str)
                or len(digest) != 64
                or not set(digest) <= _HEX_DIGITS
            ):
                _manifest_error(f"component {name} inventory ordering or identity is invalid")
            previous = relative
            try:
                actual_kind, actual_digest = _entry_identity(
                    root / component_root, relative
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"invalid assembly manifest: component {name} file is invalid: "
                    f"{relative}"
                ) from error
            if (actual_kind, actual_digest) != (kind, digest):
                _manifest_error(f"component {name} file differs: {relative}")
            product_path = prefix + relative
            if product_path in expected_paths or product_path == MANIFEST_NAME:
                _manifest_error(f"duplicate inventory path: {product_path}")
            expected_paths.add(product_path)
            normalized.append({"path": relative, "kind": kind, "sha256": digest})
            aggregate.append({"path": product_path, "kind": kind, "sha256": digest})
        if component.get("file_count") != len(normalized) or component.get("inventory_sha256") != _inventory_digest(normalized):
            _manifest_error(f"component {name} inventory summary differs")
        if name == "orbitkv-product" and tuple(
            (entry["path"], entry["kind"]) for entry in normalized
        ) != tuple((path, "file") for path in PRODUCT_INCLUDED):
            _manifest_error("OrbitKV product inventory differs")
    actual_paths = _actual_paths(root)
    expected_paths.add(MANIFEST_NAME)
    if actual_paths != expected_paths:
        _manifest_error("assembled file set differs from inventory")
    aggregate.sort(key=lambda entry: entry["path"])
    if manifest.get("file_count") != len(aggregate) or manifest.get("file_inventory_sha256") != _inventory_digest(aggregate):
        _manifest_error("aggregate inventory summary differs")
    return root


def _source_tree_id(checkout: Path) -> str:
    return _git(checkout, "rev-parse", "HEAD^{tree}").decode("ascii").strip()


def _verify_product_profile(checkout: Path) -> None:
    module_path = ROOT / "tools/verify_engine_profile.py"
    try:
        subprocess.run(
            [sys.executable, str(module_path), "--sglang-root", str(checkout)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"patched SGLang source fails the engine profile{detail}") from error


def _new_output_path(output: Path | str) -> Path:
    requested = Path(output).expanduser()
    if requested.name in ("", ".", ".."):
        raise RuntimeError("output must name a new directory")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"output parent does not exist: {requested.parent}") from error
    if not parent.is_dir():
        raise RuntimeError("output parent is not a directory")
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"output already exists: {destination}")
    return destination


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish without replacing a concurrently created path."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("atomic no-replace publication is unavailable") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise RuntimeError(f"output already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _cargo_metadata(manager_root: Path) -> None:
    manifests = [manager_root / "Cargo.toml"]
    ffi_manifest = manager_root / "ffi/Cargo.toml"
    if ffi_manifest.is_file() and not ffi_manifest.is_symlink():
        manifests.append(ffi_manifest)
    try:
        for manifest in manifests:
            subprocess.run(
                [
                    "cargo", "metadata", "--no-deps", "--offline", "--locked",
                    "--format-version", "1", "--manifest-path", str(manifest),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"assembled OrbitKV manager metadata is invalid{detail}") from error


def assemble(sglang_root: Path | str, output: Path | str) -> Path:
    """Create one immutable-layout source product from a verified checkout."""

    destination = _new_output_path(output)
    try:
        requested_source = Path(sglang_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid SGLang source root: {sglang_root}") from error
    if destination == requested_source or destination.is_relative_to(requested_source):
        raise RuntimeError("output must be outside the source checkout")
    contract = pinned_source_contract()
    orbitkv_identity = _orbitkv_source_identity()
    try:
        checkout = validate_patched_checkout(requested_source)
        _verify_product_profile(checkout)
        source_state = "patched"
    except RuntimeError as patched_error:
        try:
            checkout = validate_base_checkout(requested_source)
            source_state = "base"
        except RuntimeError as base_error:
            raise RuntimeError(
                "SGLang source is neither the exact pinned base nor reviewed overlay: "
                f"patched={patched_error}; base={base_error}"
            ) from base_error
    if checkout != requested_source:
        raise RuntimeError("SGLang validator returned a different source root")
    _validate_input_state(checkout, contract, source_state)
    source_tree = _source_tree_id(checkout)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.assembling-", dir=destination.parent))
    published = False
    try:
        upstream_paths, source_identities = _copy_upstream(checkout, temporary)
        copied_identities = {
            path: _entry_identity(temporary, path)[1] for path in upstream_paths
        }
        if copied_identities != source_identities or _source_tree_id(checkout) != source_tree:
            raise RuntimeError("SGLang source changed during assembly")
        _validate_input_state(checkout, contract, source_state)
        if source_state == "base":
            _apply_patch(temporary, _reviewed_patch_payload(contract))
        _validate_overlay_targets(temporary, contract)
        adapter_paths = _copy_component(
            INTEGRATION_PROJECT, temporary / COMPONENT_ROOT / "adapter", ADAPTER_INCLUDED
        )
        product_paths = _copy_component(
            ROOT / "compat/sglang", temporary / COMPONENT_ROOT, PRODUCT_INCLUDED
        )
        manager_paths = _copy_manager_component(
            MANAGER_PROJECT, temporary / COMPONENT_ROOT / "manager"
        )
        if _orbitkv_source_identity() != orbitkv_identity:
            raise RuntimeError("OrbitKV source identity changed during assembly")
        _cargo_metadata(temporary / COMPONENT_ROOT / "manager")
        manifest = _manifest(
            temporary, contract, source_tree, upstream_paths, adapter_paths, product_paths,
            manager_paths, orbitkv_identity
        )
        _write_regular(
            temporary / MANIFEST_NAME,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            0o644,
        )
        verify_assembly(temporary)
        _publish_no_replace(temporary, destination)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble or verify the complete pinned SGLang source plus OrbitKV."
    )
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    if raw_arguments and raw_arguments[0].startswith("--"):
        raw_arguments.insert(0, "assemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument(
        "--sglang-root", type=Path, default=ROOT / "compat/sglang/source"
    )
    assemble_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(raw_arguments)
    try:
        if arguments.command == "assemble":
            product = assemble(arguments.sglang_root, arguments.output)
            print(f"OrbitKV Engine source assembled: {product}")
        else:
            product = verify_assembly(arguments.output)
            print(f"OrbitKV Engine source verified: {product}")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
