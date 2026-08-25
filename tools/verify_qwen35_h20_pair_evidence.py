#!/usr/bin/env python3
"""Verify the sealed Qwen3.5 ABI8 H20 pair-evidence archive offline.

This verifier treats every file in the archive as untrusted input.  It never
imports the archived Python source or loads the archived shared library.  Pair
semantics and ELF parsing come from the trusted checkout that contains this
script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = ROOT / "integrations/sglang"
SOURCE_ROOT = INTEGRATION_ROOT / "src"
sys.dont_write_bytecode = True
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import qualify_abi8_h20 as qualification  # noqa: E402
from orbitkv_sglang import qualification_primitives  # noqa: E402


ARCHIVE_NAME = (
    "h20-sglang-v0517-abi8-qwen35-fixed-state-"
    "pair-verification-20260823"
)
DEFAULT_ARCHIVE = ROOT / "results" / ARCHIVE_NAME
MANIFEST_SCHEMA = "orbitkv.abi8-h20-qwen35-pair-evidence.v1"
VERIFICATION_SCHEMA = (
    "orbitkv.abi8-h20-qwen35-pair-evidence-verification.v1"
)
PREFLIGHT_SCHEMA = "orbitkv.abi8-h20-preflight.v1"
RECORD_SCHEMA = "orbitkv.sglang-v0517-abi8-single-run.v2"
PAIR_SCHEMA = "orbitkv.abi8-h20-pair-verification.v1"
SUMMARY_SCHEMA = "orbitkv.abi8-h20-multi-epoch-summary.v1"
QUALIFICATION_CLAIM = "pair_verification_only_not_qualified"
SOURCE_COMMIT = "7385ee586974ffd09dffecc415d52098f373e32a"
SOURCE_BASE_COMMIT = "0eb74430b7e3228c45b137a539e6d266b51a5bed"
SGLANG_RELEASE = "v0.5.17"
SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
H20_UUID = "GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3"
MODEL_NAME = "qwen3.5-0.8b"
MODEL_REPOSITORY = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_CONFIG_SHA256 = (
    "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
)
MODEL_INDEX_SHA256 = (
    "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
)
MODEL_WEIGHT_SHARDS_SHA256 = (
    "5d2c4457668ff6a8d4d214f7a1d013888a6ddb82a65f1173221bb61d386fd397"
)
MODEL_CONFIG_GIT_BLOB = "715f0448b9d38103211f0ad88bbb4d6e4f4be8c9"
MODEL_INDEX_GIT_BLOB = "f691cefdb79d73270895ebd6d9594ddcecfc1838"
MODEL_SHARD_GIT_BLOB = "969da4e6aa85b4e224739020212b7cb0d09cee14"
MODEL_SHARD_XET_HASH = (
    "f0140d845aced424f17b1c75ebc5a67ef75fe309c68d2f613acda2eb551db7dd"
)
MODEL_WEIGHT_SHARDS = [
    {
        "filename": "model.safetensors-00001-of-00001.safetensors",
        "sha256": (
            "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
        ),
        "size": 1_746_942_600,
    }
]
TOKEN_PLAN_SHA256 = (
    "e6e41e81e88c03419eafdb533719c101bb4840802fc7427c72d0ba66ef8d778c"
)
STATE_INPUT_SHA256 = (
    "a8cab7f35dd7b839979236253b49a864d0791ac1b57f2a84fc4c2d31076958b5"
)
REQUIREMENTS_SHA256 = (
    "472d8f63cad22cd7ac4908059562bebde5e54b8d2432f750640a14d525d2fa97"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

MANIFEST_KEYS = frozenset(
    {
        "schema",
        "qualification_status",
        "qualification_claim",
        "qualified",
        "hardware_attested",
        "performance_go",
        "abi_version",
        "exact_symbol_count",
        "record_schema",
        "epoch_count",
        "pair_count",
        "scope",
        "source_commit",
        "source_inventory_sha256",
        "source_provenance",
        "observed_hardware",
        "sglang_release",
        "sglang_revision",
        "library_sha256",
        "model_provenance_sha256",
        "input_hashes",
        "artifacts",
    }
)
EXPECTED_SCOPE = {
    "cases": [
        {
            "model": MODEL_NAME,
            "profile": "hybrid_full_gdn",
            "workload_profile": "fresh_prompt",
            "attention_backend": "fa3",
            "batch_size": batch,
        }
        for batch in (1, 4)
    ],
    "execution": {
        "page_tokens": 16,
        "kv_dtype": "bf16",
        "kv_layout": "nhd",
        "execution": "eager",
        "gpu_count": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
    },
    "excluded": [
        "performance_qualification",
        "prefix_or_radix_sharing",
        "fixed_state_copy_or_replacement_trigger",
        "cuda_graphs",
        "overlap_scheduling",
        "speculation",
        "distributed_execution",
        "multi_gpu",
        "other_qwen35_profiles",
        "production_readiness",
    ],
}

SOURCE_CLOSURE_PATHS = frozenset(
    {
        "bench_canonical_manager.py",
        "checkpoint_identity.py",
        "patches/v0.5.17-orbitkv-fail-closed.patch",
        "prepare_pinned_checkout.py",
        "pyproject.toml",
        "qualify_abi8_h20.py",
        "src/orbitkv_sglang/__init__.py",
        "src/orbitkv_sglang/benchmark_profiles.py",
        "src/orbitkv_sglang/config.py",
        "src/orbitkv_sglang/ffi/__init__.py",
        "src/orbitkv_sglang/ffi/layouts.py",
        "src/orbitkv_sglang/ffi/library.py",
        "src/orbitkv_sglang/ffi/manager.py",
        "src/orbitkv_sglang/ffi/state_pool.py",
        "src/orbitkv_sglang/ffi/token_relocation.py",
        "src/orbitkv_sglang/ffi/workspace.py",
        "src/orbitkv_sglang/pinned.py",
        "src/orbitkv_sglang/plugin/__init__.py",
        "src/orbitkv_sglang/plugin/facade.py",
        "src/orbitkv_sglang/plugin/fixed_state.py",
        "src/orbitkv_sglang/plugin/hooks.py",
        "src/orbitkv_sglang/plugin/lowering.py",
        "src/orbitkv_sglang/plugin/mirror_cleanup.py",
        "src/orbitkv_sglang/plugin/prefix_cache.py",
        "src/orbitkv_sglang/plugin/prefix_sanity.py",
        "src/orbitkv_sglang/plugin/relocation.py",
        "src/orbitkv_sglang/plugin/state.py",
        "src/orbitkv_sglang/plugin/validation.py",
        "src/orbitkv_sglang/qualification.py",
        "src/orbitkv_sglang/runtime/__init__.py",
        "src/orbitkv_sglang/runtime/census.py",
        "src/orbitkv_sglang/runtime/completion.py",
        "src/orbitkv_sglang/runtime/identity.py",
        "src/orbitkv_sglang/runtime/identity_index.py",
        "src/orbitkv_sglang/runtime/journal.py",
        "src/orbitkv_sglang/runtime/page_registry.py",
        "src/orbitkv_sglang/runtime/reclamation.py",
        "src/orbitkv_sglang/runtime/records.py",
        "src/orbitkv_sglang/runtime/relocation.py",
        "src/orbitkv_sglang/runtime/snapshot_shadow.py",
        "src/orbitkv_sglang/runtime/state_checkpoint.py",
        "src/orbitkv_sglang/runtime/token_relocation.py",
    }
)


def sha256_file(path: Path) -> str:
    return qualification_primitives.sha256_file(path)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = qualification_primitives.parse_strict_json_object(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot load strict JSON record {path}: {error}") from error
    except ValueError as error:
        if str(error) == "strict JSON value must be an object":
            raise RuntimeError(f"JSON record is not an object: {path}") from error
        detail = error.__cause__ or error
        message = str(detail)
        message = message.replace(
            "duplicate JSON object key ", "duplicate key ", 1
        ).replace("non-finite JSON number ", "non-finite number ", 1)
        raise RuntimeError(
            f"cannot load strict JSON record {path}: {message}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON record is not an object: {path}")
    return value


def _safe_relative(value: str) -> PurePosixPath:
    try:
        return qualification_primitives.canonical_relative_path(value)
    except ValueError as error:
        raise RuntimeError(
            f"unsafe or non-canonical archive path: {value!r}"
        ) from error



def _require_exact_keys(value: Any, expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    try:
        qualification_primitives.require_exact_keys(value, expected_set, label)
    except ValueError as error:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(expected_set - actual)} "
            f"extra={sorted(actual - expected_set)}"
        ) from error


def _require_sha256(value: Any, label: str) -> str:
    try:
        return qualification_primitives.require_sha256(value, label)
    except ValueError as error:
        raise RuntimeError(f"{label} is not a canonical SHA-256") from error


def _require_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise RuntimeError(f"{label} must be exactly {expected}")


def _expected_record_paths() -> set[str]:
    paths: set[str] = set()
    for epoch in range(1, 4):
        prefix = f"records/epoch-{epoch:03d}"
        for batch in (1, 4):
            slug = f"qwen3.5-0.8b-b{batch}"
            paths.update(
                {
                    f"{prefix}/{slug}-stock.json",
                    f"{prefix}/{slug}-stock.stderr.log",
                    f"{prefix}/{slug}-manager.json",
                    f"{prefix}/{slug}-manager.stderr.log",
                    f"{prefix}/{slug}-pair.json",
                }
            )
    return paths


def expected_artifact_paths() -> set[str]:
    return {
        "README.md",
        "preflight.json",
        "summary.json",
        "qualification/build/liborbitkv_ffi.so",
        "qualification/model-provenance.json",
        "qualification/model/config.json",
        "qualification/model/model.safetensors.index.json",
        "qualification/plans/qwen3.5-0.8b-token-manager.json",
        "qualification/plans/qwen3.5-0.8b-attention-state-input.json",
        "qualification/requirements.lock.txt",
        "qualification/source.bundle",
        *(f"qualification/source/{path}" for path in SOURCE_CLOSURE_PATHS),
        *_expected_record_paths(),
    }


def _archive_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def walk_error(error: OSError) -> None:
        raise RuntimeError(f"cannot inspect archive {root}: {error}") from error

    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=walk_error
    ):
        current = Path(current_root)
        for name in directory_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            _safe_relative(relative)
            if candidate.is_symlink():
                raise RuntimeError(f"archive contains a directory symlink: {relative}")
            if not candidate.is_dir():
                raise RuntimeError(f"archive contains a non-directory entry: {relative}")
            directories.add(relative)
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            _safe_relative(relative)
            if candidate.is_symlink():
                raise RuntimeError(f"archive contains a file symlink: {relative}")
            if not candidate.is_file():
                raise RuntimeError(f"archive contains a non-regular file: {relative}")
            files.add(relative)
    return files, directories


def _read_sha256sums(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error
    try:
        return qualification_primitives.parse_sha256sums(text)
    except ValueError as error:
        # Preserve the verifier's established RuntimeError messages while the
        # shared primitive remains the authority for checksum-table parsing.
        values: dict[str, str] = {}
        for line in text.splitlines():
            fields = line.split("  ", 1)
            if len(fields) != 2:
                raise RuntimeError(
                    "SHA256SUMS contains a malformed line"
                ) from error
            digest, name = fields
            _safe_relative(name)
            _require_sha256(digest, f"SHA256SUMS digest for {name}")
            if name == "SHA256SUMS" or name in values:
                raise RuntimeError(
                    f"SHA256SUMS contains an invalid entry: {name}"
                ) from error
            values[name] = digest
        if not values:
            raise RuntimeError("SHA256SUMS is empty") from error
        raise RuntimeError("SHA256SUMS contains a malformed line") from error


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _require_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    fixed = {
        "schema": MANIFEST_SCHEMA,
        "qualification_status": QUALIFICATION_CLAIM,
        "qualification_claim": QUALIFICATION_CLAIM,
        "qualified": False,
        "hardware_attested": False,
        "performance_go": False,
        "record_schema": RECORD_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "sglang_release": SGLANG_RELEASE,
        "sglang_revision": SGLANG_REVISION,
    }
    for name, expected in fixed.items():
        if manifest.get(name) != expected or type(manifest.get(name)) is not type(expected):
            raise RuntimeError(f"manifest {name} differs from the fixed evidence policy")
    _require_int(manifest.get("abi_version"), 8, "manifest abi_version")
    _require_int(
        manifest.get("exact_symbol_count"), 40,
        "manifest exact_symbol_count",
    )
    _require_int(manifest.get("epoch_count"), 3, "manifest epoch_count")
    _require_int(manifest.get("pair_count"), 6, "manifest pair_count")
    if manifest.get("scope") != EXPECTED_SCOPE:
        raise RuntimeError("manifest scope differs from the Qwen3.5 evidence policy")
    if manifest.get("observed_hardware") != {
        "name": "NVIDIA H20",
        "uuid": H20_UUID,
        "snapshot_count": 60,
    }:
        raise RuntimeError("manifest observed H20 evidence is invalid")
    _require_sha256(
        manifest.get("source_inventory_sha256"),
        "manifest source_inventory_sha256",
    )
    _require_sha256(manifest.get("library_sha256"), "manifest library_sha256")
    _require_sha256(
        manifest.get("model_provenance_sha256"),
        "manifest model_provenance_sha256",
    )
    provenance = manifest.get("source_provenance")
    _require_exact_keys(
        provenance,
        {"kind", "path", "sha256", "commit", "prerequisite_commit"},
        "manifest source_provenance",
    )
    expected_provenance = {
        "kind": "git_bundle",
        "path": "qualification/source.bundle",
        "commit": SOURCE_COMMIT,
        "prerequisite_commit": SOURCE_BASE_COMMIT,
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            raise RuntimeError(f"manifest source_provenance {name} is invalid")
    _require_sha256(provenance.get("sha256"), "source bundle SHA-256")
    expected_inputs = {
        "requirements_sha256": REQUIREMENTS_SHA256,
        "plans": {
            "token_manager": TOKEN_PLAN_SHA256,
            "state_input": STATE_INPUT_SHA256,
        },
        "models": {
            MODEL_NAME: {
                "config_sha256": MODEL_CONFIG_SHA256,
                "index_sha256": MODEL_INDEX_SHA256,
                "weight_shards_sha256": MODEL_WEIGHT_SHARDS_SHA256,
            }
        },
    }
    if manifest.get("input_hashes") != expected_inputs:
        raise RuntimeError("manifest Qwen3.5 input hashes are invalid")


def _verify_archive_inventory(
    root: Path, manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, str]:
    files, directories = _archive_inventory(root)
    expected_artifacts = expected_artifact_paths()
    expected_files = expected_artifacts | {"manifest.json", "SHA256SUMS"}
    if files != expected_files:
        raise RuntimeError(
            "archive file inventory mismatch: "
            f"missing={sorted(expected_files - files)} "
            f"unlisted={sorted(files - expected_files)}"
        )
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
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
        raise RuntimeError("manifest artifact inventory is not the exact archive schema")
    mismatches = [
        name for name, digest in artifacts.items()
        if sha256_file(root / Path(name)) != digest
    ]
    if mismatches:
        raise RuntimeError(
            "archive artifact SHA-256 mismatch: " + ", ".join(sorted(mismatches))
        )
    expected_sums = dict(artifacts)
    expected_sums["manifest.json"] = sha256_file(manifest_path)
    actual_sums = _read_sha256sums(root / "SHA256SUMS")
    if actual_sums != expected_sums:
        raise RuntimeError("SHA256SUMS inventory differs from manifest artifacts")
    canonical_text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(expected_sums.items())
    )
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != canonical_text:
        raise RuntimeError("SHA256SUMS is not in canonical sorted form")
    return artifacts


def _validate_preflight(
    preflight: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise RuntimeError("preflight schema is invalid")
    if (preflight.get("qualification_scope") != "qwen35"
            or preflight.get("status")
            != "host_preflight_passed_gpu_not_initialized"):
        raise RuntimeError("preflight does not bind the Qwen3.5 host preflight scope")
    benchmark = preflight.get("benchmark")
    if (not isinstance(benchmark, dict)
            or benchmark.get("record_schema") != RECORD_SCHEMA):
        raise RuntimeError("preflight benchmark schema is invalid")
    _require_sha256(benchmark.get("sha256"), "preflight benchmark SHA-256")
    sglang = preflight.get("sglang")
    if (not isinstance(sglang, dict)
            or sglang.get("release") != SGLANG_RELEASE
            or sglang.get("revision") != SGLANG_REVISION
            or sglang.get("pinned_contract")
            != qualification.benchmark.verify_pinned_module_constants()):
        raise RuntimeError("preflight SGLang identity is invalid")
    source = preflight.get("source")
    _require_exact_keys(
        source,
        {"clean", "commit", "tracked_file_count",
         "inventory_sha256", "inventory"},
        "preflight source",
    )
    inventory = source.get("inventory")
    if (source.get("clean") is not True
            or source.get("commit") != SOURCE_COMMIT
            or not isinstance(inventory, list) or not inventory):
        raise RuntimeError("preflight source identity is invalid")
    indexed: dict[str, str] = {}
    normalized: list[dict[str, str]] = []
    for item in inventory:
        _require_exact_keys(item, {"path", "sha256"}, "source inventory entry")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str):
            raise RuntimeError("source inventory path is not a string")
        _safe_relative(relative)
        _require_sha256(digest, f"source inventory digest for {relative}")
        if relative in indexed:
            raise RuntimeError(f"duplicate source inventory path: {relative}")
        indexed[relative] = digest
        normalized.append({"path": relative, "sha256": digest})
    count = source.get("tracked_file_count")
    if type(count) is not int or count != len(normalized):
        raise RuntimeError("preflight tracked source count is invalid")
    inventory_digest = qualification.canonical_digest(normalized)
    if (source.get("inventory_sha256") != inventory_digest
            or manifest.get("source_inventory_sha256") != inventory_digest
            or manifest.get("source_commit") != source.get("commit")):
        raise RuntimeError("preflight source inventory digest is invalid")
    return normalized, indexed


def _run(
    arguments: Sequence[str], *, cwd: Path | None = None, text: bool = True
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            list(arguments), cwd=cwd, check=True, capture_output=True,
            text=text, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"command failed: {' '.join(arguments)}: {detail}"
        ) from error


def _bundle_header(path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    prerequisites: set[str] = set()
    heads: list[tuple[str, str]] = []
    try:
        with path.open("rb") as stream:
            first = stream.readline(4096).rstrip(b"\r\n")
            if first not in (b"# v2 git bundle", b"# v3 git bundle"):
                raise RuntimeError("source bundle has an unsupported header")
            total = len(first)
            while True:
                line = stream.readline(4096)
                total += len(line)
                if total > 1024 * 1024 or not line:
                    raise RuntimeError("source bundle header is unterminated")
                stripped = line.rstrip(b"\r\n")
                if not stripped:
                    break
                if stripped.startswith(b"@"):
                    continue
                fields = stripped.split(b" ", 1)
                try:
                    object_id = fields[0].lstrip(b"-").decode("ascii")
                except UnicodeDecodeError as error:
                    raise RuntimeError(
                        "source bundle contains an invalid object id"
                    ) from error
                if COMMIT_PATTERN.fullmatch(object_id) is None:
                    raise RuntimeError("source bundle contains an invalid object id")
                if fields[0].startswith(b"-"):
                    if len(fields) != 2 or not fields[1]:
                        raise RuntimeError("source bundle contains an invalid prerequisite")
                    prerequisites.add(object_id)
                else:
                    if len(fields) != 2:
                        raise RuntimeError("source bundle contains an invalid head")
                    try:
                        reference = fields[1].decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise RuntimeError(
                            "source bundle contains an invalid head reference"
                        ) from error
                    heads.append((object_id, reference))
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot inspect source bundle: {error}") from error
    if prerequisites != {SOURCE_BASE_COMMIT}:
        raise RuntimeError("source bundle prerequisite is not the fixed base commit")
    if len(heads) != 1 or heads[0][0] != SOURCE_COMMIT:
        raise RuntimeError(
            "source bundle must advertise only the evidence commit"
        )
    return prerequisites, heads


def _verify_source_bundle(
    root: Path, inventory: list[dict[str, str]], manifest: dict[str, Any],
    *, repository_root: Path = ROOT,
) -> None:
    provenance = manifest["source_provenance"]
    bundle = root / provenance["path"]
    if sha256_file(bundle) != provenance["sha256"]:
        raise RuntimeError("source bundle differs from manifest provenance")
    _bundle_header(bundle)
    with tempfile.TemporaryDirectory(prefix="orbitkv-qwen35-source-") as temporary:
        git_dir = Path(temporary) / "repository.git"
        _run(("git", "init", "--bare", "--quiet", str(git_dir)))
        _run(
            (
                "git", f"--git-dir={git_dir}", "fetch", "--quiet",
                "--no-tags", "--no-write-fetch-head", str(repository_root),
                f"{SOURCE_BASE_COMMIT}:refs/orbitkv/prerequisite",
            )
        )
        _run(("git", f"--git-dir={git_dir}", "bundle", "verify", str(bundle)))
        _run(
            (
                "git", f"--git-dir={git_dir}", "fetch", "--quiet",
                "--no-tags", "--no-write-fetch-head", str(bundle),
                f"{SOURCE_COMMIT}:refs/orbitkv/evidence",
            )
        )
        resolved = _run(
            ("git", f"--git-dir={git_dir}", "rev-parse",
             "refs/orbitkv/evidence^{commit}")
        ).stdout.strip()
        if resolved != SOURCE_COMMIT:
            raise RuntimeError("source bundle resolved to the wrong commit")
        ancestor = subprocess.run(
            (
                "git", f"--git-dir={git_dir}", "merge-base",
                "--is-ancestor", SOURCE_BASE_COMMIT, SOURCE_COMMIT,
            ),
            check=False, capture_output=True, timeout=120,
        )
        if ancestor.returncode != 0:
            raise RuntimeError("evidence commit does not descend from its prerequisite")
        tree_output = _run(
            ("git", f"--git-dir={git_dir}", "ls-tree", "-rz", "-r",
             "--full-tree", SOURCE_COMMIT),
            text=False,
        ).stdout
        tree: list[tuple[str, str]] = []
        for raw in tree_output.split(b"\0"):
            if not raw:
                continue
            try:
                metadata, raw_path = raw.split(b"\t", 1)
                mode, object_type, object_id = metadata.decode("ascii").split(" ")
                relative = raw_path.decode("utf-8")
            except (UnicodeError, ValueError) as error:
                raise RuntimeError("evidence Git tree entry is malformed") from error
            _safe_relative(relative)
            if mode not in {"100644", "100755"} or object_type != "blob":
                raise RuntimeError("evidence Git tree contains a non-regular source entry")
            tree.append((relative, object_id))
        inventory_paths = [item["path"] for item in inventory]
        if [path for path, _ in tree] != inventory_paths:
            raise RuntimeError("preflight inventory is not the complete evidence Git tree")
        for item, (_, object_id) in zip(inventory, tree, strict=True):
            blob = _run(
                ("git", f"--git-dir={git_dir}", "cat-file", "blob", object_id),
                text=False,
            ).stdout
            if hashlib.sha256(blob).hexdigest() != item["sha256"]:
                raise RuntimeError(
                    f"source blob differs from preflight inventory: {item['path']}"
                )


def _verify_source_closure(
    root: Path, preflight: dict[str, Any], indexed: dict[str, str]
) -> tuple[dict[str, Any], str, str]:
    source_root = root / "qualification/source"
    actual_files, actual_directories = _archive_inventory(source_root)
    actual = actual_files
    if actual != SOURCE_CLOSURE_PATHS:
        raise RuntimeError("qualification source closure is incomplete or excessive")
    expected_directories = {
        parent.as_posix()
        for name in SOURCE_CLOSURE_PATHS
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise RuntimeError("qualification source directory closure is invalid")
    for relative in SOURCE_CLOSURE_PATHS:
        repository_path = f"integrations/sglang/{relative}"
        if indexed.get(repository_path) != sha256_file(source_root / relative):
            raise RuntimeError(
                f"qualification source differs from preflight: {repository_path}"
            )
    harness_sha256 = sha256_file(source_root / "bench_canonical_manager.py")
    if preflight.get("benchmark", {}).get("sha256") != harness_sha256:
        raise RuntimeError("archived benchmark differs from preflight")
    adapter = qualification._verify_source_closure(root, preflight)
    checkpoint_sha256 = sha256_file(source_root / "checkpoint_identity.py")
    return adapter, harness_sha256, checkpoint_sha256


def _verify_library(
    root: Path, preflight: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    library = qualification.sealed_library_identity(
        root / "qualification/build/liborbitkv_ffi.so"
    )
    recorded = preflight.get("library")
    if not isinstance(recorded, dict):
        raise RuntimeError("preflight library identity is missing")
    for name in ("sha256", "bytes", "abi_version", "symbols"):
        if library.get(name) != recorded.get(name):
            raise RuntimeError("archived ABI8 library differs from preflight")
    if (manifest.get("library_sha256") != library["sha256"]
            or len(library["symbols"]) != 40
            or library["abi_version"] != 8):
        raise RuntimeError("manifest ABI8 library identity is invalid")
    return library


def _artifact_identity(value: Any, label: str) -> dict[str, Any]:
    _require_exact_keys(value, {"path", "sha256", "bytes"}, label)
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise RuntimeError(f"{label} path is invalid")
    _require_sha256(value.get("sha256"), f"{label} SHA-256")
    if type(value.get("bytes")) is not int or value["bytes"] <= 0:
        raise RuntimeError(f"{label} byte count is invalid")
    return value


def _verify_model_provenance(
    root: Path, preflight: dict[str, Any], manifest: dict[str, Any]
) -> None:
    provenance_path = root / "qualification/model-provenance.json"
    if (sha256_file(provenance_path)
            != manifest.get("model_provenance_sha256")):
        raise RuntimeError("model provenance differs from the manifest identity")
    provenance = _strict_json(provenance_path)
    _require_exact_keys(
        provenance,
        {"schema", "provider", "repo_id", "revision",
         "resolved_ref", "files"},
        "model provenance",
    )
    expected_files = {
        "config.json": {
            "git_blob": MODEL_CONFIG_GIT_BLOB,
            "sha256": MODEL_CONFIG_SHA256,
            "size": 2_907,
        },
        "model.safetensors.index.json": {
            "git_blob": MODEL_INDEX_GIT_BLOB,
            "sha256": MODEL_INDEX_SHA256,
            "size": 50_900,
        },
        "model.safetensors-00001-of-00001.safetensors": {
            "git_blob": MODEL_SHARD_GIT_BLOB,
            "lfs_sha256": MODEL_WEIGHT_SHARDS[0]["sha256"],
            "sha256": MODEL_WEIGHT_SHARDS[0]["sha256"],
            "xet_hash": MODEL_SHARD_XET_HASH,
            "size": MODEL_WEIGHT_SHARDS[0]["size"],
        },
    }
    expected = {
        "schema": "orbitkv.huggingface-model-provenance.v1",
        "provider": "huggingface.co",
        "repo_id": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "resolved_ref": MODEL_REVISION,
        "files": expected_files,
    }
    if provenance != expected:
        raise RuntimeError(
            "model provenance differs from the fixed Hugging Face identity"
        )
    preflight_model = preflight.get("inputs", {}).get("models", {}).get(
        MODEL_NAME
    )
    if not isinstance(preflight_model, dict):
        raise RuntimeError("preflight Qwen3.5 model identity is missing")
    archived_metadata = (
        ("config.json", MODEL_CONFIG_SHA256, 2_907),
        ("model.safetensors.index.json", MODEL_INDEX_SHA256, 50_900),
    )
    for filename, expected_sha256, expected_size in archived_metadata:
        path = root / "qualification/model" / filename
        identity = preflight_model.get(filename)
        data = path.read_bytes()
        git_blob = hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        ).hexdigest()
        if (not isinstance(identity, dict)
                or identity.get("sha256") != expected_sha256
                or identity.get("bytes") != expected_size
                or hashlib.sha256(data).hexdigest() != expected_sha256
                or len(data) != expected_size
                or git_blob != expected_files[filename]["git_blob"]):
            raise RuntimeError(
                f"archived model {filename} differs from provenance or preflight"
            )
    shard = preflight_model.get("weight_shards")
    if (shard != MODEL_WEIGHT_SHARDS
            or qualification.canonical_digest(shard)
            != MODEL_WEIGHT_SHARDS_SHA256):
        raise RuntimeError("preflight model shard differs from provenance")
    lfs_pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{MODEL_WEIGHT_SHARDS[0]['sha256']}\n"
        f"size {MODEL_WEIGHT_SHARDS[0]['size']}\n"
    ).encode("ascii")
    lfs_git_blob = hashlib.sha1(
        b"blob " + str(len(lfs_pointer)).encode("ascii")
        + b"\0" + lfs_pointer
    ).hexdigest()
    if lfs_git_blob != expected_files[MODEL_WEIGHT_SHARDS[0]["filename"]]["git_blob"]:
        raise RuntimeError("model shard LFS pointer Git blob identity is invalid")


def _verify_inputs(
    root: Path, preflight: dict[str, Any], manifest: dict[str, Any]
) -> None:
    inputs = preflight.get("inputs")
    _require_exact_keys(
        inputs, {"requirements", "models", "plans", "state_plans"},
        "preflight inputs",
    )
    plans = inputs.get("plans")
    state_plans = inputs.get("state_plans")
    models = inputs.get("models")
    if (not isinstance(plans, dict) or set(plans) != {MODEL_NAME}
            or not isinstance(state_plans, dict) or set(state_plans) != {MODEL_NAME}
            or not isinstance(models, dict) or set(models) != {MODEL_NAME}):
        raise RuntimeError("preflight Qwen3.5 input matrix is not exact")
    token = _artifact_identity(plans[MODEL_NAME], "preflight token plan")
    state = _artifact_identity(
        state_plans[MODEL_NAME], "preflight attention-state input"
    )
    requirements = _artifact_identity(
        inputs.get("requirements"), "preflight requirements lock"
    )
    paths = (
        (root / "qualification/plans/qwen3.5-0.8b-token-manager.json",
         token, TOKEN_PLAN_SHA256, "token plan"),
        (root / "qualification/plans/qwen3.5-0.8b-attention-state-input.json",
         state, STATE_INPUT_SHA256, "attention-state input"),
        (root / "qualification/requirements.lock.txt", requirements,
         REQUIREMENTS_SHA256, "requirements lock"),
    )
    for path, identity, expected, label in paths:
        if (identity["sha256"] != expected
                or sha256_file(path) != expected
                or path.stat().st_size != identity["bytes"]):
            raise RuntimeError(f"archived {label} identity is invalid")
    model = models[MODEL_NAME]
    _require_exact_keys(
        model,
        {"root", "config.json", "model.safetensors.index.json",
         "weight_shards", "weight_shards_sha256"},
        "preflight Qwen3.5 model",
    )
    config = _artifact_identity(model.get("config.json"), "Qwen3.5 config")
    index = _artifact_identity(
        model.get("model.safetensors.index.json"), "Qwen3.5 index"
    )
    if (config["sha256"] != MODEL_CONFIG_SHA256
            or index["sha256"] != MODEL_INDEX_SHA256
            or model.get("weight_shards") != MODEL_WEIGHT_SHARDS
            or model.get("weight_shards_sha256")
            != qualification.canonical_digest(MODEL_WEIGHT_SHARDS)
            or model.get("weight_shards_sha256")
            != MODEL_WEIGHT_SHARDS_SHA256):
        raise RuntimeError("preflight Qwen3.5 model hashes are invalid")
    if manifest["input_hashes"]["requirements_sha256"] != requirements["sha256"]:
        raise RuntimeError("manifest requirements identity differs from preflight")


def _verify_execution_scope(record: dict[str, Any], label: str) -> str:
    engine = record.get("engine_args")
    runtime = record.get("runtime_identity")
    environment = record.get("environment")
    if not all(isinstance(value, dict) for value in (engine, runtime, environment)):
        raise RuntimeError(f"{label} execution identity is malformed")
    expected_engine = {
        "page_size": 16,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_dp_attention": False,
        "enable_session_radix_cache": False,
        "enable_streaming_session": False,
        "speculative_algorithm": None,
        "trust_remote_code": False,
    }
    if any(engine.get(name) != value for name, value in expected_engine.items()):
        raise RuntimeError(f"{label} differs from the declared execution scope")
    if (runtime.get("kv_layout") != "nhd"
            or runtime.get("execution") != "eager"
            or environment.get("CUDA_VISIBLE_DEVICES") != "0"):
        raise RuntimeError(f"{label} runtime scope is invalid")
    prefix_cache = runtime.get("prefix_cache")
    if (not isinstance(prefix_cache, dict)
            or prefix_cache.get("enabled_readback") is not False):
        raise RuntimeError(f"{label} unexpectedly enabled Prefix/Radix sharing")
    command = record.get("command")
    if (not isinstance(command, list) or any(not isinstance(value, str) for value in command)
            or "--trust-remote-code" in command):
        raise RuntimeError(f"{label} command scope is invalid")
    snapshots = record.get("gpu_snapshots")
    labels = [
        "before_engine", "after_load", "after_workload",
        "after_global_cleanup", "after_shutdown",
    ]
    if not isinstance(snapshots, list) or len(snapshots) != len(labels):
        raise RuntimeError(f"{label} H20 snapshot sequence is invalid")
    previous_time = -1
    for expected_label, snapshot in zip(labels, snapshots, strict=True):
        if (not isinstance(snapshot, dict)
                or snapshot.get("label") != expected_label
                or type(snapshot.get("time_ns")) is not int
                or snapshot["time_ns"] <= previous_time):
            raise RuntimeError(f"{label} H20 snapshot ordering is invalid")
        previous_time = snapshot["time_ns"]
        gpus = snapshot.get("gpus")
        if not isinstance(gpus, list) or len(gpus) != 1:
            raise RuntimeError(f"{label} does not record exactly one GPU")
        gpu = gpus[0]
        if (not isinstance(gpu, dict) or gpu.get("index") != "0"
                or gpu.get("name") != "NVIDIA H20"
                or gpu.get("uuid") != H20_UUID):
            raise RuntimeError(f"{label} GPU identity is not one H20")
    return H20_UUID


def _trusted_qwen35_cases() -> dict[int, Any]:
    expected = {
        1: (1, 528, 1024, 2),
        4: (5, 2112, 4096, 4),
    }
    cases = {case.batch: case for case in qualification.QWEN35_CASES}
    if set(cases) != set(expected):
        raise RuntimeError("trusted qualifier Qwen3.5 case matrix drifted")
    for batch, values in expected.items():
        case = cases[batch]
        observed = (
            case.iterations, case.chunk_tokens, case.capacity_tokens,
            case.request_capacity,
        )
        if (observed != values or case.model != MODEL_NAME
                or case.backend != "fa3"
                or case.profile != "hybrid_full_gdn"
                or case.workload_profile != "fresh_prompt"):
            raise RuntimeError("trusted qualifier Qwen3.5 case contract drifted")
    return cases


def _verify_records(
    root: Path, preflight: dict[str, Any], *, harness_sha256: str,
    adapter_identity: dict[str, Any], checkpoint_helper_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    cases = _trusted_qwen35_cases()
    calculated_pairs: list[dict[str, Any]] = []
    gpu_uuids: set[str] = set()
    snapshot_count = 0
    for epoch in range(1, 4):
        epoch_root = root / f"records/epoch-{epoch:03d}"
        for batch in (1, 4):
            case = cases[batch]
            slug = case.slug
            names = {
                mode: f"{slug}-{mode}.json" for mode in ("stock", "manager")
            }
            stock_path = epoch_root / names["stock"]
            manager_path = epoch_root / names["manager"]
            stock = _strict_json(stock_path)
            manager = _strict_json(manager_path)
            for mode, record in (("stock", stock), ("manager", manager)):
                source = record.get("source_identity")
                if (not isinstance(source, dict)
                        or source.get("checkpoint_identity_helper_sha256")
                        != checkpoint_helper_sha256):
                    raise RuntimeError(
                        f"{mode} record does not bind the archived checkpoint helper"
                    )
                gpu_uuids.add(
                    _verify_execution_scope(record, f"epoch {epoch} {slug} {mode}")
                )
                snapshot_count += len(record["gpu_snapshots"])
            qualification._verify_case_records(stock, manager, case)
            calculated = qualification.verify_pair_files(
                stock_path, manager_path, preflight,
                harness_sha256=harness_sha256,
                adapter_identity=adapter_identity,
            )
            calculated.update(
                epoch=epoch,
                execution_order=list(qualification.execution_order(epoch)),
            )
            pair_path = epoch_root / f"{slug}-pair.json"
            stored = _strict_json(pair_path)
            boundaries = {
                "schema": PAIR_SCHEMA,
                "qualification_claim": QUALIFICATION_CLAIM,
                "preflight_bound": True,
                "hardware_attested": False,
                "qualified": False,
            }
            if any(
                stored.get(name) != value
                or type(stored.get(name)) is not type(value)
                for name, value in boundaries.items()
            ):
                raise RuntimeError(f"stored pair changes its non-qualification boundary: {pair_path}")
            if (stored.get("stock_record") != names["stock"]
                    or stored.get("manager_record") != names["manager"]):
                raise RuntimeError(f"stored pair contains an unsafe record reference: {pair_path}")
            if (qualification._pair_without_record_locations(stored)
                    != qualification._pair_without_record_locations(calculated)):
                raise RuntimeError(f"stored pair differs from trusted derivation: {pair_path}")
            calculated_pairs.append(calculated)
    if len(gpu_uuids) != 1:
        raise RuntimeError("pair evidence was not recorded on one consistent H20")
    if snapshot_count != 60:
        raise RuntimeError("pair evidence does not contain exactly 60 H20 snapshots")
    calculated_summary = qualification.summarize_pairs(calculated_pairs)
    calculated_summary.update(
        qualification_claim=QUALIFICATION_CLAIM,
        preflight_bound=True,
        hardware_attested=False,
        qualified=False,
    )
    stored_summary = _strict_json(root / "summary.json")
    if stored_summary != calculated_summary:
        raise RuntimeError("summary differs from the six independently verified pairs")
    if (stored_summary.get("schema") != SUMMARY_SCHEMA
            or stored_summary.get("pair_count") != 6
            or stored_summary.get("performance_go") is not False
            or stored_summary.get("qualification_claim") != QUALIFICATION_CLAIM
            or stored_summary.get("hardware_attested") is not False
            or stored_summary.get("qualified") is not False):
        raise RuntimeError("summary changes its non-qualification boundary")
    return calculated_pairs, calculated_summary, next(iter(gpu_uuids))


def verify_archive(path: Path = DEFAULT_ARCHIVE) -> dict[str, Any]:
    requested = Path(os.path.abspath(path.expanduser()))
    if requested.is_symlink():
        raise RuntimeError("archive directory must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve evidence archive {requested}: {error}") from error
    if not root.is_dir():
        raise RuntimeError("evidence archive path is not a directory")
    manifest_path = root / "manifest.json"
    manifest = _strict_json(manifest_path)
    _validate_manifest(manifest)
    _verify_archive_inventory(root, manifest_path, manifest)
    preflight = _strict_json(root / "preflight.json")
    inventory, indexed = _validate_preflight(preflight, manifest)
    _verify_source_bundle(root, inventory, manifest)
    adapter, harness_sha256, checkpoint_sha256 = _verify_source_closure(
        root, preflight, indexed
    )
    library = _verify_library(root, preflight, manifest)
    _verify_model_provenance(root, preflight, manifest)
    _verify_inputs(root, preflight, manifest)
    pairs, _, gpu_uuid = _verify_records(
        root, preflight, harness_sha256=harness_sha256,
        adapter_identity=adapter,
        checkpoint_helper_sha256=checkpoint_sha256,
    )
    if len(pairs) != manifest["pair_count"]:
        raise RuntimeError("manifest pair count differs from verified records")
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "archive": ARCHIVE_NAME,
        "source_commit": SOURCE_COMMIT,
        "epoch_count": 3,
        "pair_count": 6,
        "abi_version": library["abi_version"],
        "exact_symbol_count": len(library["symbols"]),
        "gpu": {"name": "NVIDIA H20", "uuid": gpu_uuid},
        "qualification_claim": QUALIFICATION_CLAIM,
        "qualified": False,
        "hardware_attested": False,
        "performance_go": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE,
        help=f"archive directory (default: results/{ARCHIVE_NAME})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = verify_archive(args.archive)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
