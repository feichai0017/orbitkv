#!/usr/bin/env python3
"""Offline verifier for sealed ABI8 H20 token-relocation evidence.

Archived Python and ELF objects are data only: this module neither imports the
archived source nor loads the shared library.  Raw-record semantics are
delegated to the trusted sibling verifier supplied by the calling checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from types import ModuleType


_RAW_VERIFIER: ModuleType | None = None


def _load_raw_verifier() -> ModuleType:
    """Load the trusted sibling raw-record verifier without a cycle."""

    global _RAW_VERIFIER
    if _RAW_VERIFIER is None:
        path = Path(__file__).with_name(
            "verify_token_relocation_h20_evidence.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_orbitkv_token_relocation_raw_for_seal", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load trusted raw-record verifier")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RAW_VERIFIER = module
    return _RAW_VERIFIER


SEALED_MANIFEST_SCHEMA = (
    "orbitkv.abi8-h20-token-relocation-sealed-manifest.v1"
)
SEALED_PREFLIGHT_SCHEMA = (
    "orbitkv.abi8-h20-token-relocation-preflight.v1"
)
SEALED_PAIR_SCHEMA = (
    "orbitkv.abi8-h20-token-relocation-pair-verification.v1"
)
SEALED_SUMMARY_SCHEMA = (
    "orbitkv.abi8-h20-token-relocation-multi-epoch-summary.v1"
)
SEALED_VERIFICATION_SCHEMA = (
    "orbitkv.abi8-h20-token-relocation-seal-verification.v1"
)
MODEL_PROVENANCE_SCHEMA = "orbitkv.qwen25-0.5b-model-provenance.v1"
QUALIFICATION_SCOPE = "token_relocation_scoped_correctness_lifecycle"
QUALIFICATION_STATUS = (
    "abi8_sglang_full_token_relocation_correctness_lifecycle_"
    "qualified_performance_pending"
)
QUALIFICATION_CLAIM = "scoped_correctness_and_lifecycle_only"
SEALED_EVIDENCE_CLASS = "sealed_clean_source_scoped_qualification"
PLAN_SHA256 = (
    "6e51963cc09c45a26446621a2d164386ed2e1aed19de585a168db4267ea6cb8a"
)
REQUIREMENTS_INPUT_SHA256 = (
    "472d8f63cad22cd7ac4908059562bebde5e54b8d2432f750640a14d525d2fa97"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
ORBITKV_EDITABLE_TEMPLATE = (
    "-e git+https://github.com/feichai0017/orbitkv.git@{commit}"
    "#egg=orbitkv_sglang&subdirectory=integrations/sglang"
)
SOURCE_REQUIRED_PATHS = frozenset(
    {
        "integrations/sglang/qualify_token_relocation_h20.py",
        "integrations/sglang/qualify_abi8_h20.py",
        "integrations/sglang/bench_token_relocation.py",
        "integrations/sglang/bench_canonical_manager.py",
        "integrations/sglang/checkpoint_identity.py",
        "integrations/sglang/prepare_pinned_checkout.py",
        "integrations/sglang/pyproject.toml",
        "integrations/sglang/patches/v0.5.17-orbitkv-fail-closed.patch",
        "tools/verify_token_relocation_h20_evidence.py",
        "tools/verify_token_relocation_h20_seal.py",
    }
)
COMPONENT_CASES = frozenset(
    {
        "test_opaque_payload_coordinates_are_embedded_without_collisions",
        *(
            "test_stale_member_makes_append_and_relocation_prepare_"
            f"failure_atomic[{batch}]"
            for batch in (1, 4, 32)
        ),
        *(
            "test_real_cuda_opaque_token_relocation_conformance"
            f"[{batch}]"
            for batch in (1, 4, 32)
        ),
    }
)
EXACT_ABI8_SYMBOLS = frozenset(
    {
        "orbitkv_abi_version",
        "orbitkv_manager_abort_relocations_batch",
        "orbitkv_manager_abort_steps_batch",
        "orbitkv_manager_acknowledge_reclamations_batch",
        "orbitkv_manager_arena_identities",
        "orbitkv_manager_arena_stats",
        "orbitkv_manager_complete_batch",
        "orbitkv_manager_complete_relocation_batch",
        "orbitkv_manager_create",
        "orbitkv_manager_destroy",
        "orbitkv_manager_mark_token_dispositions_batch",
        "orbitkv_manager_prefix_attach_batch",
        "orbitkv_manager_prefix_evict_batch",
        "orbitkv_manager_prefix_lookup_batch",
        "orbitkv_manager_prefix_publish_batch",
        "orbitkv_manager_prefix_publish_release_batch",
        "orbitkv_manager_prefix_recycle_batch",
        "orbitkv_manager_prepare_batch",
        "orbitkv_manager_prepare_relocation_batch",
        "orbitkv_manager_quarantine_steps_batch",
        "orbitkv_manager_quarantine_submissions_batch",
        "orbitkv_manager_recycle_requests_batch",
        "orbitkv_manager_release_batch",
        "orbitkv_manager_request_acquire_batch",
        "orbitkv_manager_request_fork_batch",
        "orbitkv_manager_stats",
        "orbitkv_manager_submit_batch",
        "orbitkv_manager_submit_relocation_batch",
        "orbitkv_manager_token_views_batch",
        "orbitkv_state_pool_abort_batch",
        "orbitkv_state_pool_acknowledge_batch",
        "orbitkv_state_pool_complete_batch",
        "orbitkv_state_pool_create",
        "orbitkv_state_pool_current_batch",
        "orbitkv_state_pool_destroy",
        "orbitkv_state_pool_identity",
        "orbitkv_state_pool_prepare_batch",
        "orbitkv_state_pool_retire_owners_batch",
        "orbitkv_state_pool_stats",
        "orbitkv_state_pool_submit_batch",
    }
)
SEALED_MANIFEST_KEYS = frozenset(
    {
        "schema", "qualification_status", "qualification_claim",
        "evidence_class", "sealed", "source_clean", "source_dirty",
        "preflight_bound", "qualified", "hardware_attested",
        "performance_go", "abi_version", "exact_symbol_count",
        "record_schema", "summary_schema", "pair_schema",
        "epoch_count", "batch_sizes", "record_count", "pair_count",
        "scope", "source_commit", "source_inventory_sha256",
        "source_provenance", "sglang_release", "sglang_revision",
        "library_sha256", "model_identity_sha256", "input_hashes",
        "observed_hardware", "artifacts",
    }
)
EXPECTED_SEALED_SCOPE = {
    "cases": [
        {
            "model": "qwen2.5-0.5b", "profile": "full",
            "attention_backend": "flashinfer", "batch_size": batch,
            "modes": ["naive", "relocate"],
        }
        for batch in (1, 4)
    ],
    "execution": {
        "page_tokens": 16, "kv_dtype": "bf16", "kv_layout": "nhd",
        "execution": "eager", "gpu_count": 1,
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "data_parallel_size": 1, "decode_context_parallel_size": 1,
    },
    "excluded": [
        "performance_qualification", "capacity_or_memory_saving_claims",
        "general_speedup_claims", "prefix_or_radix_sharing",
        "hybrid_or_swa_attention", "mla", "cuda_graphs",
        "overlap_scheduling", "speculation", "distributed_execution",
        "multi_gpu", "production_readiness",
    ],
}


def _require_exact_keys(
    value: Any, expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    actual = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or actual != expected_set:
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(expected_set - actual)} "
            f"extra={sorted(actual - expected_set)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a canonical SHA-256")
    return value


def _safe_relative(value: str) -> PurePosixPath:
    if (
        not isinstance(value, str) or not value or "\\" in value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise RuntimeError(f"unsafe or non-canonical archive path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute() or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe or non-canonical archive path: {value!r}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON value is not an object: {path}")
    return value


def _archive_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        for name in list(directory_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative)
            if path.is_symlink():
                raise RuntimeError(f"archive contains a directory symlink: {relative}")
            if not path.is_dir():
                raise RuntimeError(f"archive entry is not a directory: {relative}")
            directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative)
            if path.is_symlink():
                raise RuntimeError(f"archive contains a file symlink: {relative}")
            if not path.is_file():
                raise RuntimeError(f"archive entry is not a regular file: {relative}")
            files.add(relative)
    return files, directories


def _read_sha256sums(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read SHA256SUMS: {error}") from error
    values: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2 or not all(fields):
            raise RuntimeError("SHA256SUMS contains a malformed line")
        digest, name = fields
        _safe_relative(name)
        _require_sha256(digest, f"SHA256SUMS digest for {name}")
        if name == "SHA256SUMS" or name in values:
            raise RuntimeError(f"SHA256SUMS contains an invalid entry: {name}")
        values[name] = digest
    if not values:
        raise RuntimeError("SHA256SUMS is empty")
    return values


def _run(
    arguments: Sequence[str], *, text: bool = True
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            list(arguments), check=True, capture_output=True, text=text,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"command failed: {' '.join(str(value) for value in arguments)}: "
            f"{detail}"
        ) from error


def _expected_sealed_artifacts(source_closure: Iterable[str]) -> set[str]:
    artifacts = {
        "README.md", "preflight.json", "summary.json",
        "component-conformance.xml",
        "qualification/build/liborbitkv_ffi.so",
        "qualification/plans/qwen2.5-0.5b-full-page16-bf16.json",
        "qualification/model/config.json",
        "qualification/model-provenance.json",
        "qualification/requirements.input.lock.txt",
        "qualification/requirements.lock.txt",
        "qualification/source.bundle",
    }
    for epoch in (1, 2, 3, 4):
        for batch in (1, 4):
            slug = f"qwen2.5-0.5b-b{batch}"
            artifacts.add(f"pairs/epoch-{epoch:03d}/{slug}-pair.json")
            for mode in ("naive", "relocate"):
                artifacts.add(f"records/epoch-{epoch:03d}/{slug}-{mode}.json")
                artifacts.add(
                    f"logs/epoch-{epoch:03d}/{slug}-{mode}.stderr.log"
                )
    artifacts.update(f"qualification/source/{path}" for path in source_closure)
    return artifacts


def _validate_sealed_manifest(
    manifest: dict[str, Any], raw: ModuleType | None = None
) -> None:
    raw = _load_raw_verifier() if raw is None else raw
    _require_exact_keys(manifest, SEALED_MANIFEST_KEYS, "manifest")
    fixed = {
        "schema": SEALED_MANIFEST_SCHEMA,
        "qualification_status": QUALIFICATION_STATUS,
        "qualification_claim": QUALIFICATION_CLAIM,
        "evidence_class": SEALED_EVIDENCE_CLASS,
        "sealed": True, "source_clean": True, "source_dirty": False,
        "preflight_bound": True, "qualified": True,
        "hardware_attested": False, "performance_go": False,
        "abi_version": 8, "exact_symbol_count": 40,
        "record_schema": raw.RECORD_SCHEMA,
        "summary_schema": SEALED_SUMMARY_SCHEMA,
        "pair_schema": SEALED_PAIR_SCHEMA, "epoch_count": 4,
        "batch_sizes": [1, 4], "record_count": 16, "pair_count": 8,
        "sglang_release": raw.SGLANG_RELEASE,
        "sglang_revision": raw.SGLANG_REVISION,
    }
    for name, expected in fixed.items():
        if manifest.get(name) != expected or type(manifest.get(name)) is not type(expected):
            raise RuntimeError(f"manifest {name} differs from sealed policy")
    if manifest.get("scope") != EXPECTED_SEALED_SCOPE:
        raise RuntimeError("manifest scope differs from sealed policy")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("manifest source commit is invalid")
    for name in ("source_inventory_sha256", "library_sha256",
                 "model_identity_sha256"):
        _require_sha256(manifest.get(name), f"manifest {name}")
    provenance = manifest.get("source_provenance")
    _require_exact_keys(
        provenance, {"kind", "path", "sha256", "commit", "reference"},
        "manifest source provenance",
    )
    if provenance != {
        "kind": "git_bundle", "path": "qualification/source.bundle",
        "sha256": provenance.get("sha256"), "commit": commit,
        "reference": "HEAD",
    }:
        raise RuntimeError("manifest source provenance is invalid")
    _require_sha256(provenance.get("sha256"), "source bundle SHA-256")
    inputs = manifest.get("input_hashes")
    _require_exact_keys(
        inputs, {"requirements_input_sha256", "requirements_sha256",
                 "plan_sha256", "model"}, "manifest input hashes",
    )
    _require_sha256(inputs.get("requirements_sha256"),
                    "manifest materialized requirements SHA-256")
    if (inputs.get("requirements_input_sha256") != REQUIREMENTS_INPUT_SHA256
            or inputs.get("plan_sha256") != PLAN_SHA256
            or inputs.get("model") != {
                "config_sha256": raw.CHECKPOINT_CONFIG_SHA256,
                "weight_sha256": raw.CHECKPOINT_WEIGHT_SHA256,
                "weight_bytes": raw.CHECKPOINT_WEIGHT_BYTES,
            }):
        raise RuntimeError("manifest input hashes differ from sealed policy")
    hardware = manifest.get("observed_hardware")
    _require_exact_keys(
        hardware, {"name", "uuid", "snapshot_count", "attestation"},
        "manifest observed hardware",
    )
    if (hardware.get("name") != raw.GPU_NAME
            or not isinstance(hardware.get("uuid"), str) or not hardware["uuid"]
            or hardware.get("snapshot_count") != 64
            or hardware.get("attestation") != "recorded_observation_only"):
        raise RuntimeError("manifest observed hardware is invalid")


def _verify_sealed_inventory(
    root: Path, manifest_path: Path, manifest: dict[str, Any],
    source_closure: Iterable[str],
) -> dict[str, str]:
    files, directories = _archive_inventory(root)
    expected_artifacts = _expected_sealed_artifacts(source_closure)
    expected_files = expected_artifacts | {"manifest.json", "SHA256SUMS"}
    if files != expected_files:
        raise RuntimeError(
            "archive file inventory mismatch: "
            f"missing={sorted(expected_files - files)} "
            f"unlisted={sorted(files - expected_files)}"
        )
    expected_directories = {
        parent.as_posix() for name in expected_files
        for parent in PurePosixPath(name).parents if parent.as_posix() != "."
    }
    if directories != expected_directories:
        raise RuntimeError(
            "archive directory inventory mismatch: "
            f"missing={sorted(expected_directories - directories)} "
            f"unlisted={sorted(directories - expected_directories)}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("manifest artifacts must be an object")
    for name, digest in artifacts.items():
        _safe_relative(name)
        _require_sha256(digest, f"manifest artifact digest for {name}")
    if set(artifacts) != expected_artifacts:
        raise RuntimeError("manifest artifact inventory is not exact")
    mismatches = [
        name for name, digest in artifacts.items()
        if _sha256_file(root / name) != digest
    ]
    if mismatches:
        raise RuntimeError(
            "archive artifact SHA-256 mismatch: " + ", ".join(sorted(mismatches))
        )
    expected_sums = dict(artifacts)
    expected_sums["manifest.json"] = _sha256_file(manifest_path)
    if _read_sha256sums(root / "SHA256SUMS") != expected_sums:
        raise RuntimeError("SHA256SUMS inventory differs from manifest artifacts")
    canonical = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(expected_sums.items())
    )
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != canonical:
        raise RuntimeError("SHA256SUMS is not in canonical sorted form")
    return artifacts


def _validate_source_inventory(
    preflight: dict[str, Any], manifest: dict[str, Any],
    raw: ModuleType | None = None,
) -> tuple[list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    raw = _load_raw_verifier() if raw is None else raw
    _require_exact_keys(
        preflight, {"schema", "qualification_scope", "status", "source",
                    "source_closure", "benchmark", "verifier", "library",
                    "build", "sglang", "inputs", "python"}, "preflight",
    )
    if (preflight.get("schema") != SEALED_PREFLIGHT_SCHEMA
            or preflight.get("qualification_scope") != QUALIFICATION_SCOPE
            or preflight.get("status")
            != "host_preflight_passed_gpu_not_initialized"):
        raise RuntimeError("preflight policy is invalid")
    source = preflight.get("source")
    _require_exact_keys(
        source, {"clean", "commit", "tracked_file_count",
                 "inventory_sha256", "inventory"}, "preflight source",
    )
    inventory = source.get("inventory")
    if (source.get("clean") is not True
            or source.get("commit") != manifest["source_commit"]
            or not isinstance(inventory, list) or not inventory):
        raise RuntimeError("preflight clean source identity is invalid")
    normalized: list[dict[str, str]] = []
    indexed: dict[str, str] = {}
    previous: str | None = None
    for item in inventory:
        _require_exact_keys(item, {"path", "sha256"}, "source inventory entry")
        relative = item.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("source inventory path is not a string")
        name = _safe_relative(relative).as_posix()
        digest = _require_sha256(item.get("sha256"),
                                 f"source inventory digest for {name}")
        if name in indexed or (previous is not None and name <= previous):
            raise RuntimeError("source inventory is not unique and sorted")
        indexed[name] = digest
        normalized.append({"path": name, "sha256": digest})
        previous = name
    if (type(source.get("tracked_file_count")) is not int
            or source["tracked_file_count"] != len(normalized)
            or source.get("inventory_sha256") != raw.canonical_digest(normalized)
            or manifest.get("source_inventory_sha256")
            != source.get("inventory_sha256")):
        raise RuntimeError("preflight source inventory identity is invalid")
    closure = preflight.get("source_closure")
    if not isinstance(closure, list) or not closure:
        raise RuntimeError("preflight source closure is missing")
    normalized_closure: list[dict[str, str]] = []
    closure_paths: set[str] = set()
    for item in closure:
        _require_exact_keys(item, {"path", "sha256"}, "source closure entry")
        relative = item.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("source closure path is not a string")
        name = _safe_relative(relative).as_posix()
        digest = _require_sha256(item.get("sha256"), f"source closure {name}")
        if name in closure_paths or indexed.get(name) != digest:
            raise RuntimeError("source closure differs from full source inventory")
        closure_paths.add(name)
        normalized_closure.append({"path": name, "sha256": digest})
    expected_closure = set(SOURCE_REQUIRED_PATHS) | {
        name for name in indexed
        if name.startswith("integrations/sglang/src/orbitkv_sglang/")
        and name.endswith(".py")
    }
    if (closure_paths != expected_closure
            or normalized_closure
            != sorted(normalized_closure, key=lambda item: item["path"])):
        raise RuntimeError("preflight source closure is incomplete or excessive")
    return normalized, indexed, normalized_closure


def _bundle_header(path: Path, expected_commit: str) -> None:
    heads: list[tuple[str, str]] = []
    prerequisites: list[str] = []
    try:
        with path.open("rb") as stream:
            first = stream.readline(4096).rstrip(b"\r\n")
            if first not in (b"# v2 git bundle", b"# v3 git bundle"):
                raise RuntimeError("source bundle has an unsupported header")
            total = len(first)
            while True:
                line = stream.readline(4096)
                total += len(line)
                if not line or total > 1024 * 1024:
                    raise RuntimeError("source bundle header is unterminated")
                stripped = line.rstrip(b"\r\n")
                if not stripped:
                    break
                if stripped.startswith(b"@"):
                    continue
                fields = stripped.split(b" ", 1)
                if len(fields) != 2:
                    raise RuntimeError("source bundle contains an invalid head")
                try:
                    object_id = fields[0].lstrip(b"-").decode("ascii")
                    reference = fields[1].decode("utf-8")
                except UnicodeError as error:
                    raise RuntimeError("source bundle header is invalid") from error
                if COMMIT_PATTERN.fullmatch(object_id) is None:
                    raise RuntimeError("source bundle contains an invalid object id")
                (prerequisites if fields[0].startswith(b"-") else heads).append(
                    object_id if fields[0].startswith(b"-") else (object_id, reference)
                )
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot inspect source bundle: {error}") from error
    if prerequisites:
        raise RuntimeError("source bundle must be self-contained")
    if heads != [(expected_commit, "HEAD")]:
        raise RuntimeError("source bundle must advertise only the preflight HEAD")


def _verify_source_bundle(
    root: Path, inventory: list[dict[str, str]], manifest: dict[str, Any]
) -> None:
    provenance = manifest["source_provenance"]
    bundle = root / provenance["path"]
    if _sha256_file(bundle) != provenance["sha256"]:
        raise RuntimeError("source bundle differs from manifest provenance")
    commit = manifest["source_commit"]
    _bundle_header(bundle, commit)
    with tempfile.TemporaryDirectory(prefix="orbitkv-relocation-source-") as temporary:
        git_dir = Path(temporary) / "repository.git"
        _run(("git", "init", "--bare", "--quiet", str(git_dir)))
        _run(("git", f"--git-dir={git_dir}", "bundle", "verify", str(bundle)))
        _run(("git", f"--git-dir={git_dir}", "fetch", "--quiet",
              "--no-tags", "--no-write-fetch-head", str(bundle),
              f"{commit}:refs/orbitkv/evidence"))
        resolved = _run(("git", f"--git-dir={git_dir}", "rev-parse",
                         "refs/orbitkv/evidence^{commit}")).stdout.strip()
        if resolved != commit:
            raise RuntimeError("source bundle resolved to the wrong commit")
        output = _run(("git", f"--git-dir={git_dir}", "ls-tree", "-rz",
                       "-r", "--full-tree", commit), text=False).stdout
        tree: list[tuple[str, str]] = []
        for entry in output.split(b"\0"):
            if not entry:
                continue
            try:
                metadata, raw_path = entry.split(b"\t", 1)
                mode, object_type, object_id = metadata.decode("ascii").split(" ")
                relative = raw_path.decode("utf-8")
            except (UnicodeError, ValueError) as error:
                raise RuntimeError("source Git tree entry is malformed") from error
            _safe_relative(relative)
            if mode not in {"100644", "100755"} or object_type != "blob":
                raise RuntimeError("source Git tree contains a non-regular entry")
            tree.append((relative, object_id))
        if [path for path, _ in tree] != [item["path"] for item in inventory]:
            raise RuntimeError("preflight inventory is not the complete Git tree")
        for item, (_, object_id) in zip(inventory, tree, strict=True):
            blob = _run(("git", f"--git-dir={git_dir}", "cat-file", "blob",
                         object_id), text=False).stdout
            if hashlib.sha256(blob).hexdigest() != item["sha256"]:
                raise RuntimeError(
                    f"source blob differs from preflight inventory: {item['path']}"
                )


def _artifact_identity(value: Any, label: str) -> dict[str, Any]:
    _require_exact_keys(value, {"path", "bytes", "sha256"}, label)
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise RuntimeError(f"{label} path is invalid")
    if type(value.get("bytes")) is not int or value["bytes"] <= 0:
        raise RuntimeError(f"{label} byte count is invalid")
    _require_sha256(value.get("sha256"), f"{label} SHA-256")
    return value


def _validate_sealed_preflight(
    preflight: dict[str, Any], manifest: dict[str, Any],
    raw: ModuleType | None = None,
) -> tuple[list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    raw = _load_raw_verifier() if raw is None else raw
    inventory, indexed, closure = _validate_source_inventory(
        preflight, manifest, raw
    )
    benchmark = preflight.get("benchmark")
    _require_exact_keys(benchmark, {"path", "sha256", "record_schema"},
                        "preflight benchmark")
    if (not isinstance(benchmark.get("path"), str)
            or Path(benchmark["path"]).name != "bench_token_relocation.py"
            or benchmark.get("record_schema") != raw.RECORD_SCHEMA):
        raise RuntimeError("preflight benchmark identity is invalid")
    _require_sha256(benchmark.get("sha256"), "preflight benchmark SHA-256")
    verifier = preflight.get("verifier")
    _require_exact_keys(verifier, {"path", "sha256"}, "preflight verifier")
    if (not isinstance(verifier.get("path"), str)
            or Path(verifier["path"]).name
            != "verify_token_relocation_h20_evidence.py"):
        raise RuntimeError("preflight verifier identity is invalid")
    _require_sha256(verifier.get("sha256"), "preflight verifier SHA-256")
    library = preflight.get("library")
    _require_exact_keys(
        library, {"path", "sha256", "bytes", "abi_version", "symbols"},
        "preflight library",
    )
    if (not isinstance(library.get("path"), str)
            or Path(library["path"]).name != "liborbitkv_ffi.so"
            or type(library.get("bytes")) is not int or library["bytes"] <= 0
            or library.get("abi_version") != 8
            or library.get("symbols") != sorted(EXACT_ABI8_SYMBOLS)):
        raise RuntimeError("preflight ABI8 library identity is invalid")
    _require_sha256(library.get("sha256"), "preflight library SHA-256")
    build = preflight.get("build")
    _require_exact_keys(build, {"command", "cargo_version", "cargo_target_dir"},
                        "preflight build")
    command = build.get("command")
    if (not isinstance(command, list)
            or not all(isinstance(item, str) and item for item in command)
            or len(command) < 5 or command[1:4] != ["build", "--release", "--locked"]
            or "--manifest-path" not in command
            or not isinstance(build.get("cargo_version"), str)
            or not build["cargo_version"].startswith("cargo ")
            or not isinstance(build.get("cargo_target_dir"), str)
            or not build["cargo_target_dir"]):
        raise RuntimeError("preflight build identity is invalid")
    sglang = preflight.get("sglang")
    _require_exact_keys(
        sglang, {"release", "revision", "manager_root", "manager",
                 "pinned_contract", "manager_entrypoint"}, "preflight SGLang",
    )
    if (sglang.get("release") != raw.SGLANG_RELEASE
            or sglang.get("revision") != raw.SGLANG_REVISION
            or not isinstance(sglang.get("manager_root"), str)
            or not sglang["manager_root"]):
        raise RuntimeError("preflight SGLang release identity is invalid")
    expected_loader = {
        "path": raw.LOADER_PATH, "head_git_blob": raw.LOADER_HEAD_GIT_BLOB,
        "worktree_git_blob": raw.LOADER_WORKTREE_GIT_BLOB,
        "worktree_sha256": raw.LOADER_WORKTREE_SHA256,
        "patch_sha256": raw.LOADER_PATCH_SHA256,
    }
    expected_manager = {
        "root": sglang["manager_root"], "release": raw.SGLANG_RELEASE,
        "revision": raw.SGLANG_REVISION,
        "python_source_contract": raw.SOURCE_CONTRACT,
        "dirty_paths": [raw.LOADER_PATH], "loader": expected_loader,
        "tag": raw.SGLANG_RELEASE,
        "remote": "https://github.com/sgl-project/sglang.git",
    }
    if sglang.get("manager") != expected_manager:
        raise RuntimeError("preflight manager checkout identity is invalid")
    if sglang.get("pinned_contract") != {
        "release": raw.SGLANG_RELEASE, "revision": raw.SGLANG_REVISION,
        "loader_path": raw.LOADER_PATH,
        "base_source_sha256":
            "3a975a73f1a7887e68c81ea7a2530250597ac8ae978efc0b0f70f038a99a3164",
        "patched_source_sha256": raw.LOADER_WORKTREE_SHA256,
        "patch_diff_sha256": raw.LOADER_PATCH_SHA256,
    }:
        raise RuntimeError("preflight pinned SGLang contract is invalid")
    entrypoint = sglang.get("manager_entrypoint")
    if (not isinstance(entrypoint, dict)
            or set(entrypoint) != {*raw.MANAGER_ENTRYPOINT, "module"}
            or any(entrypoint.get(name) != value
                   for name, value in raw.MANAGER_ENTRYPOINT.items())
            or not isinstance(entrypoint.get("module"), str)
            or Path(entrypoint["module"]).name not in {"plugin.py", "__init__.py"}):
        raise RuntimeError("preflight manager entrypoint is invalid")
    inputs = preflight.get("inputs")
    _require_exact_keys(inputs, {"requirements", "model", "plan"},
                        "preflight inputs")
    requirements = _artifact_identity(inputs.get("requirements"),
                                      "preflight requirements")
    plan = _artifact_identity(inputs.get("plan"), "preflight plan")
    if (requirements["sha256"] != REQUIREMENTS_INPUT_SHA256
            or Path(requirements["path"]).name != "requirements-v0.5.17.lock.txt"
            or plan["sha256"] != PLAN_SHA256
            or Path(plan["path"]).name
            != "qwen2.5-0.5b-full-page16-bf16.json"):
        raise RuntimeError("preflight requirements or plan identity is invalid")
    model = inputs.get("model")
    _require_exact_keys(model, {"name", "root", "checkpoint",
                                "identity_sha256"}, "preflight model")
    if (model.get("name") != raw.MODEL_SLUG
            or not isinstance(model.get("root"), str)
            or Path(model["root"]).name != raw.MODEL_PATH_BASENAME):
        raise RuntimeError("preflight model identity is invalid")
    raw._validate_checkpoint(model.get("checkpoint"), "preflight model")
    if (model.get("identity_sha256") != raw.canonical_digest(model["checkpoint"])
            or model["identity_sha256"] != manifest["model_identity_sha256"]):
        raise RuntimeError("preflight model digest is invalid")
    python = preflight.get("python")
    _require_exact_keys(
        python, {"executable", "real_executable", "normalized_freeze_sha256",
                 "active_editable", "locked_editable"}, "preflight Python",
    )
    expected_editable = ORBITKV_EDITABLE_TEMPLATE.format(commit=manifest["source_commit"])
    if (python.get("active_editable") != expected_editable
            or not isinstance(python.get("locked_editable"), str)
            or not python["locked_editable"].startswith("-e git+")
            or "#egg=orbitkv_sglang&subdirectory=integrations/sglang"
            not in python["locked_editable"]
            or not isinstance(python.get("executable"), str)
            or not python["executable"]
            or not isinstance(python.get("real_executable"), str)
            or not python["real_executable"]):
        raise RuntimeError("preflight Python identity is invalid")
    _require_sha256(python.get("normalized_freeze_sha256"),
                    "preflight normalized freeze SHA-256")
    return inventory, indexed, closure


def _adapter_identity_from_closure(
    closure: list[dict[str, str]]
) -> dict[str, Any]:
    indexed = {item["path"]: item["sha256"] for item in closure}
    paths = [
        "integrations/sglang/pyproject.toml",
        "integrations/sglang/prepare_pinned_checkout.py",
        "integrations/sglang/patches/v0.5.17-orbitkv-fail-closed.patch",
        *(
            name for name in sorted(indexed)
            if name.startswith("integrations/sglang/src/orbitkv_sglang/")
            and name.endswith(".py")
        ),
    ]
    if len(paths) != len(set(paths)) or any(path not in indexed for path in paths):
        raise RuntimeError("archived adapter identity is incomplete")
    return {
        "files": [
            {"path": path, "sha256": indexed[path]} for path in paths
        ]
    }


def _verify_source_closure(
    root: Path, indexed: dict[str, str], closure: list[dict[str, str]],
    preflight: dict[str, Any], raw: ModuleType | None = None,
) -> tuple[dict[str, Any], str]:
    raw = _load_raw_verifier() if raw is None else raw
    source_root = root / "qualification/source"
    files, directories = _archive_inventory(source_root)
    expected_files = {item["path"] for item in closure}
    if files != expected_files:
        raise RuntimeError(
            "qualification source closure is incomplete or excessive"
        )
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    if directories != expected_directories:
        raise RuntimeError(
            "qualification source directory closure is invalid"
        )
    for item in closure:
        name = item["path"]
        if indexed.get(name) != item["sha256"] or _sha256_file(
            source_root / name
        ) != item["sha256"]:
            raise RuntimeError(
                f"qualification source differs from preflight: {name}"
            )
    harness = _sha256_file(
        source_root / "integrations/sglang/bench_token_relocation.py"
    )
    if preflight["benchmark"]["sha256"] != harness:
        raise RuntimeError(
            "archived relocation benchmark differs from preflight"
        )
    verifier_digest = _sha256_file(
        source_root / "tools/verify_token_relocation_h20_evidence.py"
    )
    if preflight["verifier"]["sha256"] != verifier_digest:
        raise RuntimeError(
            "archived relocation verifier differs from preflight"
        )
    adapter = _adapter_identity_from_closure(closure)
    raw._verify_adapter_source_identity(
        source_root / "integrations/sglang/src", adapter
    )
    return adapter, harness


def _elf_dynamic_symbols(path: Path) -> list[str]:
    """Parse exported OrbitKV symbols without loading untrusted code."""

    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise RuntimeError("sealed ABI8 library is not an ELF file")
    elf_class, byte_order = data[4], data[5]
    if elf_class not in (1, 2) or byte_order not in (1, 2):
        raise RuntimeError(
            "sealed ABI8 library has an unsupported ELF encoding"
        )
    endian = "<" if byte_order == 1 else ">"
    if elf_class == 2:
        header_format = endian + "HHIQQQIHHHHHH"
        section_format = endian + "IIQQQQIIQQ"
        symbol_format = endian + "IBBHQQ"
        section_offset_index, section_entry_index, section_count_index = 5, 10, 11
    else:
        header_format = endian + "HHIIIIIHHHHHH"
        section_format = endian + "IIIIIIIIII"
        symbol_format = endian + "IIIBBH"
        section_offset_index, section_entry_index, section_count_index = 5, 10, 11
    header_size = struct.calcsize(header_format)
    if len(data) < 16 + header_size:
        raise RuntimeError(
            "sealed ABI8 library has a truncated ELF header"
        )
    header = struct.unpack_from(header_format, data, 16)
    section_offset = header[section_offset_index]
    section_entry_size = header[section_entry_index]
    section_count = header[section_count_index]
    expected_section_size = struct.calcsize(section_format)
    if (
        section_entry_size < expected_section_size
        or section_count <= 0
        or section_offset + section_entry_size * section_count > len(data)
    ):
        raise RuntimeError(
            "sealed ABI8 library has an invalid section table"
        )
    sections = [
        struct.unpack_from(
            section_format, data, section_offset + index * section_entry_size
        )
        for index in range(section_count)
    ]
    symbol_names: set[str] = set()
    for section in sections:
        if section[1] != 11:  # SHT_DYNSYM
            continue
        offset, size, link, entry_size = (
            section[4], section[5], section[6], section[9]
        )
        if link >= len(sections):
            raise RuntimeError(
                "sealed ABI8 library has an invalid string table link"
            )
        strings = sections[link]
        strings_offset, strings_size = strings[4], strings[5]
        expected_symbol_size = struct.calcsize(symbol_format)
        if (
            entry_size < expected_symbol_size
            or offset + size > len(data)
            or strings_offset + strings_size > len(data)
            or size % entry_size
        ):
            raise RuntimeError(
                "sealed ABI8 library has an invalid dynamic symbol table"
            )
        table = data[strings_offset : strings_offset + strings_size]
        for position in range(offset, offset + size, entry_size):
            symbol = struct.unpack_from(symbol_format, data, position)
            name_offset = symbol[0]
            info = symbol[1] if elf_class == 2 else symbol[3]
            section_index = symbol[3] if elf_class == 2 else symbol[5]
            if not name_offset or section_index == 0 or info >> 4 not in (1, 2):
                continue
            if name_offset >= len(table):
                raise RuntimeError(
                    "sealed ABI8 library has an invalid symbol name"
                )
            end = table.find(b"\0", name_offset)
            if end < 0:
                raise RuntimeError(
                    "sealed ABI8 library has an unterminated symbol name"
                )
            try:
                name = table[name_offset:end].decode("ascii")
            except UnicodeDecodeError as error:
                raise RuntimeError(
                    "sealed ABI8 library has a non-ASCII symbol"
                ) from error
            if name.startswith("orbitkv_"):
                symbol_names.add(name)
    return sorted(symbol_names)


def _verify_library(
    root: Path, preflight: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    path = root / "qualification/build/liborbitkv_ffi.so"
    symbols = _elf_dynamic_symbols(path)
    identity = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "abi_version": 8,
        "symbols": symbols,
    }
    recorded = preflight["library"]
    if (
        symbols != sorted(EXACT_ABI8_SYMBOLS)
        or len(symbols) != 40
        or identity["sha256"] != recorded["sha256"]
        or identity["bytes"] != recorded["bytes"]
        or recorded["abi_version"] != 8
        or recorded["symbols"] != symbols
        or manifest["library_sha256"] != identity["sha256"]
    ):
        raise RuntimeError("archived ABI8 library identity is invalid")
    return identity


def _materialize_requirements_lock(source: Path, active_editable: str) -> str:
    try:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            f"cannot read requirements input lock: {error}"
        ) from error
    indexes = [
        index for index, line in enumerate(lines)
        if line.rstrip("\r\n").startswith("-e git+")
        and "#egg=orbitkv_sglang" in line.rstrip("\r\n")
    ]
    if len(indexes) != 1:
        raise RuntimeError(
            "requirements lock must contain exactly one editable orbitkv-sglang"
        )
    original = lines[indexes[0]]
    ending = original[len(original.rstrip("\r\n")) :]
    lines[indexes[0]] = active_editable + ending
    return "".join(lines)


def _verify_inputs(
    root: Path, preflight: dict[str, Any], manifest: dict[str, Any],
    raw: ModuleType | None = None,
) -> None:
    raw = _load_raw_verifier() if raw is None else raw
    inputs = preflight["inputs"]
    input_hashes = manifest["input_hashes"]
    original_lock = root / "qualification/requirements.input.lock.txt"
    materialized_lock = root / "qualification/requirements.lock.txt"
    requirements = inputs["requirements"]
    active_editable = preflight["python"]["active_editable"]
    expected_materialized = _materialize_requirements_lock(
        original_lock, active_editable
    )
    if (
        _sha256_file(original_lock) != REQUIREMENTS_INPUT_SHA256
        or requirements["sha256"] != REQUIREMENTS_INPUT_SHA256
        or original_lock.stat().st_size != requirements["bytes"]
        or materialized_lock.read_text(encoding="utf-8")
        != expected_materialized
        or _sha256_file(materialized_lock)
        != input_hashes["requirements_sha256"]
    ):
        raise RuntimeError("archived requirements locks are inconsistent")
    locked_lines = [
        line for line in original_lock.read_text(encoding="utf-8").splitlines()
        if line.startswith("-e git+") and "#egg=orbitkv_sglang" in line
    ]
    if locked_lines != [preflight["python"]["locked_editable"]]:
        raise RuntimeError(
            "preflight locked editable differs from input lock"
        )

    plan_path = root / (
        "qualification/plans/qwen2.5-0.5b-full-page16-bf16.json"
    )
    plan = inputs["plan"]
    if (
        plan["sha256"] != PLAN_SHA256
        or _sha256_file(plan_path) != PLAN_SHA256
        or plan_path.stat().st_size != plan["bytes"]
    ):
        raise RuntimeError(
            "archived token-manager plan identity is invalid"
        )

    model = inputs["model"]
    provenance_path = root / "qualification/model-provenance.json"
    provenance = _strict_json(provenance_path)
    _require_exact_keys(
        provenance,
        {"schema", "name", "directory_name", "config_sha256",
         "weight_files", "checkpoint_identity_sha256"},
        "model provenance",
    )
    expected_provenance = {
        "schema": MODEL_PROVENANCE_SCHEMA,
        "name": raw.MODEL_SLUG,
        "directory_name": raw.MODEL_PATH_BASENAME,
        "config_sha256": raw.CHECKPOINT_CONFIG_SHA256,
        "weight_files": [
            {
                "name": "model.safetensors",
                "bytes": raw.CHECKPOINT_WEIGHT_BYTES,
                "sha256": raw.CHECKPOINT_WEIGHT_SHA256,
            }
        ],
        "checkpoint_identity_sha256": model["identity_sha256"],
    }
    if provenance != expected_provenance:
        raise RuntimeError(
            "model provenance differs from pinned local checkpoint"
        )
    config_path = root / "qualification/model/config.json"
    if (
        _sha256_file(config_path) != raw.CHECKPOINT_CONFIG_SHA256
        or manifest["model_identity_sha256"] != model["identity_sha256"]
    ):
        raise RuntimeError("archived model config identity is invalid")


def _verify_component_junit(path: Path, hardware: dict[str, Any]) -> None:
    try:
        document = ET.parse(path)
    except (OSError, ET.ParseError) as error:
        raise RuntimeError(f"invalid component JUnit XML: {path}") from error
    document_root = document.getroot()
    suites = (
        [document_root]
        if document_root.tag == "testsuite"
        else document_root.findall("testsuite")
    )
    if len(suites) != 1:
        raise RuntimeError(
            "component JUnit must contain exactly one test suite"
        )
    suite = suites[0]
    expected_counts = {
        "tests": str(len(COMPONENT_CASES)),
        "errors": "0",
        "failures": "0",
        "skipped": "0",
    }
    if any(suite.get(name) != value for name, value in expected_counts.items()):
        raise RuntimeError("component JUnit result counts differ")
    cases = [node.get("name") for node in suite.findall("testcase")]
    if len(cases) != len(set(cases)) or set(cases) != COMPONENT_CASES:
        raise RuntimeError(
            "component JUnit cases differ from qualification set"
        )
    if any(
        case.find("failure") is not None
        or case.find("error") is not None
        or case.find("skipped") is not None
        for case in suite.findall("testcase")
    ):
        raise RuntimeError(
            "component JUnit contains a non-passing case"
        )
    nodes = suite.findall("./properties/property")
    properties = {node.get("name"): node.get("value") for node in nodes}
    if len(properties) != len(nodes):
        raise RuntimeError("component JUnit has duplicate properties")
    expected = {
        "orbitkv.cuda.available": "true",
        "orbitkv.cuda.device_name": hardware["name"],
        "orbitkv.cuda.device_uuid": hardware["uuid"],
    }
    if any(properties.get(name) != value for name, value in expected.items()):
        raise RuntimeError(
            "component JUnit device identity differs"
        )
    for name in ("orbitkv.cuda.runtime_version", "orbitkv.torch.version"):
        if not isinstance(properties.get(name), str) or not properties[name]:
            raise RuntimeError(
                f"component JUnit property is missing: {name}"
            )


def _expected_pair(
    root: Path, epoch: int, batch: int, order: list[str],
    naive: dict[str, Any], relocate: dict[str, Any],
    raw: ModuleType | None = None,
) -> dict[str, Any]:
    raw = _load_raw_verifier() if raw is None else raw
    slug = f"{raw.MODEL_SLUG}-b{batch}"
    naive_name = f"{slug}-naive.json"
    relocate_name = f"{slug}-relocate.json"
    record_root = root / "records" / f"epoch-{epoch:03d}"
    return {
        "schema": SEALED_PAIR_SCHEMA,
        "status": "passed",
        "qualification_claim": QUALIFICATION_CLAIM,
        "preflight_bound": True,
        "hardware_attested": False,
        "qualified": True,
        "performance_go": False,
        "epoch": epoch,
        "batch_size": batch,
        "execution_order": order,
        "naive_record": naive_name,
        "relocate_record": relocate_name,
        "naive_record_sha256": _sha256_file(record_root / naive_name),
        "relocate_record_sha256": _sha256_file(record_root / relocate_name),
        "pair_key_sha256": naive["pairing"]["pair_key_sha256"],
        "output_token_digest_sha256": naive["output_token_digest_sha256"],
        "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }


def _sealed_summary(diagnostic: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(diagnostic))
    result.update(
        schema=SEALED_SUMMARY_SCHEMA,
        status="scoped_correctness_and_lifecycle_qualification_passed",
        evidence_class=SEALED_EVIDENCE_CLASS,
        diagnostic_only=False,
        qualification_status=QUALIFICATION_STATUS,
        qualification_claim=QUALIFICATION_CLAIM,
        sealed=True,
        source_clean=True,
        source_dirty=False,
        preflight_bound=True,
        hardware_attested=False,
        qualified=True,
        performance_go=False,
    )
    return result


def _verify_pairs_and_summary(
    root: Path, diagnostic: dict[str, Any], raw: ModuleType | None = None
) -> dict[str, Any]:
    raw = _load_raw_verifier() if raw is None else raw
    order_by_epoch = {
        item["epoch"]: item["order"]
        for item in diagnostic["execution_order"]["epochs"]
    }
    for epoch in raw.EPOCHS:
        for batch in raw.BATCHES:
            slug = f"{raw.MODEL_SLUG}-b{batch}"
            record_root = root / "records" / f"epoch-{epoch:03d}"
            naive = _strict_json(record_root / f"{slug}-naive.json")
            relocate = _strict_json(record_root / f"{slug}-relocate.json")
            if (
                naive["pairing"] != relocate["pairing"]
                or naive["request_output_ids"]
                != relocate["request_output_ids"]
                or naive["output_token_digest_sha256"]
                != relocate["output_token_digest_sha256"]
            ):
                raise RuntimeError(
                    f"epoch {epoch} B{batch} raw pair semantic identity differs"
                )
            expected = _expected_pair(
                root, epoch, batch, order_by_epoch[epoch], naive, relocate, raw
            )
            stored = _strict_json(
                root / "pairs" / f"epoch-{epoch:03d}/{slug}-pair.json"
            )
            if stored != expected:
                raise RuntimeError(
                    "stored pair differs from raw record derivation: "
                    f"epoch {epoch} B{batch}"
                )
    sealed = _sealed_summary(diagnostic)
    if _strict_json(root / "summary.json") != sealed:
        raise RuntimeError(
            "stored summary differs from raw record derivation"
        )
    return sealed


def verify_sealed_archive(root: Path) -> dict[str, Any]:
    """Verify a portable clean-source ABI8 relocation archive offline."""

    raw = _load_raw_verifier()
    requested = Path(os.path.abspath(Path(root).expanduser()))
    if requested.is_symlink():
        raise RuntimeError("archive directory must not be a symlink")
    try:
        archive = requested.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"cannot resolve evidence archive {requested}: {error}"
        ) from error
    if not archive.is_dir():
        raise RuntimeError("evidence archive path is not a directory")
    manifest_path = archive / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("archive manifest must be a regular file")
    manifest = _strict_json(manifest_path)
    _validate_sealed_manifest(manifest, raw)
    preflight = _strict_json(archive / "preflight.json")
    inventory, indexed, closure = _validate_sealed_preflight(
        preflight, manifest, raw
    )
    _verify_sealed_inventory(
        archive, manifest_path, manifest, (item["path"] for item in closure)
    )
    _verify_source_bundle(archive, inventory, manifest)
    adapter, harness = _verify_source_closure(
        archive, indexed, closure, preflight, raw
    )
    library = _verify_library(archive, preflight, manifest)
    _verify_inputs(archive, preflight, manifest, raw)
    diagnostic = raw.verify_evidence(
        archive,
        expected_harness_sha256=harness,
        expected_adapter=adapter,
        expected_adapter_source_root=(
            archive / "qualification/source/integrations/sglang/src"
        ),
    )
    sealed_summary = _verify_pairs_and_summary(archive, diagnostic, raw)
    hardware = sealed_summary["hardware"]
    if manifest["observed_hardware"] != {
        "name": hardware["observed_name"],
        "uuid": hardware["observed_uuid"],
        "snapshot_count": hardware["snapshot_count"],
        "attestation": hardware["attestation"],
    }:
        raise RuntimeError(
            "manifest hardware identity differs from raw records"
        )
    baseline = _strict_json(
        archive / "records/epoch-001/qwen2.5-0.5b-b1-naive.json"
    )
    bindings = {
        "library SHA-256": (
            baseline["source_identity"]["library"]["sha256"],
            library["sha256"],
        ),
        "library bytes": (
            baseline["source_identity"]["library"]["bytes"],
            library["bytes"],
        ),
        "plan SHA-256": (
            baseline["source_identity"]["plan"]["sha256"], PLAN_SHA256,
        ),
        "plan bytes": (
            baseline["source_identity"]["plan"]["bytes"],
            preflight["inputs"]["plan"]["bytes"],
        ),
        "harness SHA-256": (
            baseline["source_identity"]["harness_sha256"], harness,
        ),
        "adapter identity": (baseline["source_identity"]["adapter"], adapter),
    }
    mismatches = [name for name, (recorded, archived) in bindings.items()
                  if recorded != archived]
    if mismatches:
        raise RuntimeError(
            "raw record artifacts differ from archived qualification inputs: "
            + ", ".join(mismatches)
        )
    _verify_component_junit(
        archive / "component-conformance.xml",
        manifest["observed_hardware"],
    )
    return {
        "schema": SEALED_VERIFICATION_SCHEMA,
        "status": "passed",
        "qualification_status": QUALIFICATION_STATUS,
        "qualification_claim": QUALIFICATION_CLAIM,
        "sealed": True,
        "source_clean": True,
        "preflight_bound": True,
        "hardware_attested": False,
        "qualified": True,
        "performance_go": False,
        "epoch_count": 4,
        "record_count": 16,
        "pair_count": 8,
        "abi_version": 8,
        "exact_symbol_count": len(library["symbols"]),
        "all_pairs_passed": True,
        "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }
