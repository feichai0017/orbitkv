from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PUBLISH_GIT = Path("/tmp/orbitkv-publish.git")
ABI8_H20_SCHEMA = "orbitkv.abi8-h20-sealed-manifest.v1"
ABI8_H20_MANIFEST = (
    ROOT / "results/h20-sglang-v0517-abi8-full-hybrid-20260823/manifest.json"
)
QWEN35_H20_PAIR_EVIDENCE_SCHEMA = (
    "orbitkv.abi8-h20-qwen35-pair-evidence.v1"
)
QWEN35_H20_PAIR_EVIDENCE_MANIFEST = (
    ROOT
    / "results/h20-sglang-v0517-abi8-qwen35-fixed-state-"
    "pair-verification-20260823/manifest.json"
)
QWEN35_H20_PAIR_EVIDENCE_VERIFIER = (
    ROOT / "tools/verify_qwen35_h20_pair_evidence.py"
)
QWEN38_H20_DIAGNOSTIC_SCHEMA = (
    "orbitkv.abi8-h20-qwen38-fp8-diagnostic-manifest.v1"
)
QWEN38_H20_DIAGNOSTIC_MANIFEST = (
    ROOT
    / "results/h20-sglang-v0517-abi8-qwen38-fp8-"
    "diagnostic-20260824/manifest.json"
)
QWEN38_H20_DIAGNOSTIC_VERIFIER = (
    QWEN38_H20_DIAGNOSTIC_MANIFEST.parent / "verify.py"
)
QWEN38_H20_DIAGNOSTIC_VERIFIER_SHA256 = (
    "d17adbf52576bd31c9e91eec45cb0c1a4df32163ea9a5d67eddba9d88033b6d5"
)
TOKEN_RELOCATION_H20_DIAGNOSTIC_SCHEMA = (
    "orbitkv.sglang-v0517-token-relocation-diagnostic-manifest.v1"
)
TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST = (
    ROOT
    / "results/h20-sglang-v0517-token-relocation-"
    "diagnostic-20260825/manifest.json"
)
TOKEN_RELOCATION_H20_DIAGNOSTIC_VERIFIER = (
    ROOT / "tools/verify_token_relocation_h20_evidence.py"
)
TOKEN_RELOCATION_H20_DIAGNOSTIC_ARTIFACTS = (
    "README.md",
    "component-conformance.xml",
    "summary.json",
    *(
        f"records/epoch-{epoch:03d}/qwen2.5-0.5b-b{batch}-{mode}.json"
        for epoch in range(1, 5)
        for batch in (1, 4)
        for mode in ("naive", "relocate")
    ),
)
TOKEN_RELOCATION_H20_DIAGNOSTIC_DIRECTORIES = (
    "records",
    *(f"records/epoch-{epoch:03d}" for epoch in range(1, 5)),
)
TOKEN_RELOCATION_COMPONENT_CASES = (
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
)
ABI8_H20_QUALIFICATION_STATUS = (
    "abi8_sglang_full_full_swa_prefix_correctness_qualified_performance_pending"
)
ABI8_H20_CASES = [
    {
        "attention_backend": "fa3",
        "batch_size": 1,
        "model": "qwen2.5-7b",
        "profile": "full",
    },
    {
        "attention_backend": "fa3",
        "batch_size": 1,
        "model": "gpt-oss-20b",
        "profile": "hybrid_full_swa",
    },
    {
        "attention_backend": "fa3",
        "batch_size": 4,
        "model": "qwen2.5-7b",
        "profile": "full",
    },
    {
        "attention_backend": "fa3",
        "batch_size": 4,
        "model": "gpt-oss-20b",
        "profile": "hybrid_full_swa",
    },
]
ABI8_H20_EXCLUSIONS = [
    "token_relocation",
    "mla",
    "fixed_state",
    "overlap_scheduling",
    "cuda_graphs",
    "speculation",
    "distributed_execution",
    "performance_qualification",
]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
ABI8_H20_RUNNER = ROOT / "integrations/sglang/qualify_abi8_h20.py"

_DEFAULT_MANIFESTS = (
    ROOT / "results/h20-owning-vmm-manifest-20260817.json",
    ROOT / "results/owner-ffi-20260817/manifest.json",
    ROOT / "results/h20-generation-vmm-20260817/manifest.json",
    ROOT / "results/retention-ir-20260817/manifest.json",
    ROOT / "results/sink-sliding-20260817/manifest.json",
    ROOT / "results/h20-gpt-oss-20b-real-20260817/manifest.json",
    ROOT / "results/chunked-local-20260817/manifest.json",
    ROOT / "results/lifetime-normalization-20260817/manifest.json",
    ROOT / "results/applicability-h20-20260817/manifest.json",
    ROOT / "results/applicability-h20-20260817/multireq-manifest.json",
    ROOT / "results/applicability-h20-20260817/page16-manifest.json",
    ROOT / "results/applicability-h20-20260817/page16-graph-manifest.json",
    ROOT / "results/h20-capsule-export-20260818/manifest.json",
    ROOT / "results/h20-live-tail-capsule-20260818/manifest.json",
    ROOT / "results/h20-hybrid-capsule-20260818/manifest.json",
    ROOT / "results/h20-runtime-state-plan-20260819/manifest.json",
    ROOT / "results/h20-transactional-binding-20260819/manifest.json",
    ROOT / "results/h20-radix-prefix-20260819/manifest.json",
    ROOT / "results/h20-cuda-event-overlap-20260819/manifest.json",
    ROOT / "results/dense-runtime-20260819/manifest.json",
    ROOT / "results/h20-dense-sglang-20260819/manifest.json",
    ROOT / "results/h20-rust-owned-pages-20260820/manifest.json",
    ROOT / "results/h20-canonical-manager-20260820/manifest.json",
    ROOT / "results/h20-sglang-v0517-full-hybrid-20260821/manifest.json",
    ROOT / "results/h20-sglang-v0517-abi5-full-hybrid-20260821/manifest.json",
    ROOT
    / "results/h20-sglang-v0517-abi5-v5-grouped-release-20260821/manifest.json",
)
DEFAULT_MANIFESTS = _DEFAULT_MANIFESTS + (
    ABI8_H20_MANIFEST,
    QWEN35_H20_PAIR_EVIDENCE_MANIFEST,
    QWEN38_H20_DIAGNOSTIC_MANIFEST,
    TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{path}: duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def git_repository_args() -> list[str]:
    repository_args = ["-C", str(ROOT)]
    if subprocess.run(
        ["git", *repository_args, "rev-parse", "--verify", "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        if not LOCAL_PUBLISH_GIT.exists():
            raise RuntimeError("no usable Git metadata found for historical manifest")
        repository_args = [
            f"--git-dir={LOCAL_PUBLISH_GIT}",
            f"--work-tree={ROOT}",
        ]
    return repository_args


def git_blob(commit: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            *git_repository_args(),
            "show",
            f"{commit}:{relative_path}",
        ]
    )


def optional_git_blob(commit: str, relative_path: str) -> bytes | None:
    result = subprocess.run(
        [
            "git",
            *git_repository_args(),
            "show",
            f"{commit}:{relative_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def checked_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe provenance path: {relative_path}")
    return path


def checked_sealed_relative_path(relative_path: str) -> Path:
    if (
        not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
        or "\n" in relative_path
        or "\r" in relative_path
    ):
        raise RuntimeError(f"unsafe sealed artifact path: {relative_path!r}")
    parts = relative_path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"unsafe sealed artifact path: {relative_path}")
    path = Path(*parts)
    if path.is_absolute() or path.as_posix() != relative_path:
        raise RuntimeError(f"unsafe sealed artifact path: {relative_path}")
    return path


def sealed_regular_files(root: Path) -> set[str]:
    files: set[str] = set()

    def walk_error(error: OSError) -> None:
        raise RuntimeError(f"{root}: cannot inspect sealed file inventory") from error

    for current_root, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        current = Path(current_root)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise RuntimeError(f"{candidate}: symlinks are not allowed in a seal")
        for name in file_names:
            candidate = current / name
            if candidate.is_symlink():
                raise RuntimeError(f"{candidate}: symlinks are not allowed in a seal")
            if not candidate.is_file():
                raise RuntimeError(f"{candidate}: sealed entries must be regular files")
            files.add(candidate.relative_to(root).as_posix())
    return files


def canonical_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(data)


def verify_abi8_source_provenance(
    root: Path, manifest: dict[str, object]
) -> int:
    preflight = load_json(root / "preflight.json")
    if not isinstance(preflight, dict):
        raise RuntimeError(f"{root}: preflight must be a JSON object")
    source = preflight.get("source")
    if not isinstance(source, dict) or source.get("clean") is not True:
        raise RuntimeError(f"{root}: preflight has no clean source identity")
    commit = source.get("commit")
    inventory = source.get("inventory")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or manifest.get("source_commit") != commit
        or not isinstance(inventory, list)
        or not inventory
    ):
        raise RuntimeError(f"{root}: preflight source identity is malformed")
    if (
        type(source.get("tracked_file_count")) is not int
        or source.get("tracked_file_count") != len(inventory)
        or source.get("inventory_sha256") != canonical_digest(inventory)
        or manifest.get("source_inventory_sha256")
        != source.get("inventory_sha256")
    ):
        raise RuntimeError(f"{root}: preflight source inventory identity is invalid")

    paths: list[str] = []
    expected_hashes: dict[str, str] = {}
    for item in inventory:
        if not isinstance(item, dict):
            raise RuntimeError(f"{root}: source inventory entry is malformed")
        relative = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or checked_relative_path(relative).as_posix() != relative
            or relative in expected_hashes
        ):
            raise RuntimeError(f"{root}: source inventory entry is invalid")
        paths.append(relative)
        expected_hashes[relative] = digest

    try:
        tracked = subprocess.check_output(
            [
                "git",
                *git_repository_args(),
                "ls-tree",
                "-r",
                "--name-only",
                "--full-tree",
                commit,
            ],
            stderr=subprocess.PIPE,
            text=True,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{root}: cannot resolve sealed source commit") from error
    if tracked != paths:
        raise RuntimeError(f"{root}: source inventory is not the exact Git tree")
    for relative, expected in expected_hashes.items():
        if sha256(git_blob(commit, relative)) != expected:
            raise RuntimeError(
                f"{root}: {relative}: source blob differs from preflight"
            )
    return len(paths)


def verify_abi8_h20_semantics(
    root: Path, manifest: dict[str, object]
) -> int:
    """Verify provenance and records with trusted code from this checkout."""

    checked = verify_abi8_source_provenance(root, manifest)
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH="",
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(ABI8_H20_RUNNER),
                "verify-seal",
                str(root),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = json.loads(completed.stdout)
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(
            f"{root}: trusted ABI8 semantic verification failed: {detail}"
        ) from error
    expected = {
        "schema": "orbitkv.abi8-h20-seal-verification.v1",
        "status": "passed",
        "epoch_count": 3,
        "pair_count": 12,
        "abi_version": 8,
        "exact_symbol_count": 40,
        "performance_go": False,
    }
    if result != expected:
        raise RuntimeError(f"{root}: trusted semantic verification is incomplete")
    return checked + 1


def verify_abi8_h20_manifest(path: Path, manifest: dict[str, object]) -> int:
    if path.name != "manifest.json":
        raise RuntimeError(f"{path}: sealed manifest must be named manifest.json")
    if path.is_symlink():
        raise RuntimeError(f"{path}: sealed manifest must not be a symlink")
    if manifest.get("schema") != ABI8_H20_SCHEMA:
        raise RuntimeError(f"{path}: unsupported sealed manifest schema")
    expected_values = {
        "abi_version": 8,
        "exact_symbol_count": 40,
        "pair_count": 12,
        "epoch_count": 3,
    }
    for field, expected in expected_values.items():
        value = manifest.get(field)
        if type(value) is not int or value != expected:
            raise RuntimeError(f"{path}: {field} must be exactly {expected}")
    if manifest.get("performance_go") is not False:
        raise RuntimeError(f"{path}: performance_go must be false")
    if manifest.get("qualification_status") != ABI8_H20_QUALIFICATION_STATUS:
        raise RuntimeError(f"{path}: unexpected qualification status")
    expected_scope = {
        "cases": ABI8_H20_CASES,
        "excluded": ABI8_H20_EXCLUSIONS,
    }
    if manifest.get("scope") != expected_scope:
        raise RuntimeError(f"{path}: scope cases or exclusions differ from ABI8 policy")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(f"{path}: artifacts must be a non-empty object")

    root = path.parent
    artifact_paths: dict[str, Path] = {}
    normalized_paths: set[Path] = set()
    for relative_path, expected_digest in artifacts.items():
        if not isinstance(relative_path, str):
            raise RuntimeError(f"{path}: artifact paths must be strings")
        safe_path = checked_sealed_relative_path(relative_path)
        if relative_path in ("manifest.json", "SHA256SUMS"):
            raise RuntimeError(f"{path}: {relative_path} must not be an artifact")
        if safe_path in normalized_paths:
            raise RuntimeError(f"{path}: duplicate artifact path: {relative_path}")
        normalized_paths.add(safe_path)
        if not isinstance(expected_digest, str) or SHA256_PATTERN.fullmatch(
            expected_digest
        ) is None:
            raise RuntimeError(f"{path}: invalid SHA-256 for {relative_path}")

        artifact_path = root / safe_path
        for parent in (artifact_path, *artifact_path.parents):
            if parent == root.parent:
                break
            if parent.is_symlink():
                raise RuntimeError(f"{artifact_path}: symlinks are not allowed")
            if parent == root:
                break
        if not artifact_path.is_file():
            raise RuntimeError(f"{artifact_path}: sealed artifact is missing")
        actual_digest = sha256(artifact_path.read_bytes())
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"{path}: {relative_path}: expected {expected_digest}, got {actual_digest}"
            )
        artifact_paths[relative_path] = artifact_path

    expected_files = set(artifact_paths) | {"manifest.json", "SHA256SUMS"}
    actual_files = sealed_regular_files(root)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unlisted = sorted(actual_files - expected_files)
        raise RuntimeError(
            f"{path}: sealed file inventory mismatch; "
            f"missing={missing}, unlisted={unlisted}"
        )

    sums_path = root / "SHA256SUMS"
    if sums_path.is_symlink() or not sums_path.is_file():
        raise RuntimeError(f"{sums_path}: SHA256SUMS must be a regular file")
    expected_sums = dict(artifacts)
    expected_sums["manifest.json"] = sha256(path.read_bytes())
    expected_sums_text = "".join(
        f"{digest}  {relative_path}\n"
        for relative_path, digest in sorted(expected_sums.items())
    )
    actual_sums_text = sums_path.read_text(encoding="utf-8")
    if actual_sums_text != expected_sums_text:
        raise RuntimeError(
            f"{sums_path}: contents do not exactly match the sealed file inventory"
        )
    return len(expected_sums) + verify_abi8_h20_semantics(root, manifest)


def verify_source_provenance(manifest: dict[str, object]) -> int:
    amendment_path_value = manifest.get("source_provenance_amendment")
    if amendment_path_value is None:
        return 0
    if not isinstance(amendment_path_value, str):
        raise RuntimeError("source_provenance_amendment must be a path")
    amendment_path = ROOT / checked_relative_path(amendment_path_value)
    amendment_bytes = amendment_path.read_bytes()
    expected_amendment_sha = manifest.get("source_provenance_amendment_sha256")
    if not isinstance(expected_amendment_sha, str):
        raise RuntimeError("source provenance amendment hash is missing")
    if sha256(amendment_bytes) != expected_amendment_sha:
        raise RuntimeError(f"{amendment_path}: provenance amendment hash mismatch")
    amendment = json.loads(amendment_bytes)
    if amendment.get("schema") != "orbitkv.source-provenance-amendment.v1":
        raise RuntimeError(f"{amendment_path}: unsupported provenance schema")

    manifest_at_run = amendment.get("manifest_at_run")
    source_patch = amendment.get("source_patch")
    if not isinstance(manifest_at_run, dict) or not isinstance(source_patch, dict):
        raise RuntimeError(f"{amendment_path}: incomplete provenance amendment")

    observed_path = ROOT / checked_relative_path(str(manifest_at_run.get("path")))
    observed_bytes = observed_path.read_bytes()
    if sha256(observed_bytes) != manifest_at_run.get("sha256"):
        raise RuntimeError(f"{observed_path}: observed manifest hash mismatch")
    observed_manifest = json.loads(observed_bytes)
    if observed_manifest.get("sources") != manifest.get("sources"):
        raise RuntimeError(f"{observed_path}: source inventory changed after the run")

    base_commit = source_patch.get("base_commit")
    if (
        not isinstance(base_commit, str)
        or base_commit != observed_manifest.get("base_source_commit")
    ):
        raise RuntimeError(f"{amendment_path}: source patch base mismatch")
    patch_path = ROOT / checked_relative_path(str(source_patch.get("path")))
    patch_bytes = patch_path.read_bytes()
    if sha256(patch_bytes) != source_patch.get("sha256"):
        raise RuntimeError(f"{patch_path}: source patch hash mismatch")

    sources = observed_manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError(f"{observed_path}: source inventory is missing")
    with tempfile.TemporaryDirectory(prefix="orbitkv-provenance-") as temporary:
        reconstructed = Path(temporary)
        for relative_path in sources:
            safe_path = checked_relative_path(relative_path)
            blob = optional_git_blob(base_commit, relative_path)
            if blob is not None:
                destination = reconstructed / safe_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(blob)
        subprocess.run(
            ["git", "apply", "--binary", str(patch_path)],
            cwd=reconstructed,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for relative_path, expected in sources.items():
            actual = sha256((reconstructed / checked_relative_path(relative_path)).read_bytes())
            if actual != expected:
                raise RuntimeError(
                    f"{patch_path}: {relative_path}: expected {expected}, got {actual}"
                )
    return len(sources) + 3


def sealed_source_commit(path: Path, manifest: dict[str, object]) -> tuple[str | None, int]:
    amendment_path = path.parent / "provenance-amendment.json"
    if not amendment_path.is_file():
        return None, 0
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("schema") != "orbitkv.source-tree-provenance-amendment.v1":
        return None, 0
    if amendment.get("observed_base_commit") != manifest.get("base_source_commit"):
        raise RuntimeError(f"{amendment_path}: observed source commit mismatch")
    commit = amendment.get("sealed_source_commit")
    if not isinstance(commit, str):
        raise RuntimeError(f"{amendment_path}: sealed source commit is missing")
    return commit, 1


def historical_unsealed_sources(path: Path, manifest: dict[str, object]) -> tuple[bool, int]:
    amendment_path = path.parent / "provenance-amendment.json"
    if not amendment_path.is_file():
        return False, 0
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("schema") != "orbitkv.historical-unsealed-source-amendment.v1":
        return False, 0
    manifest_record = amendment.get("manifest")
    if not isinstance(manifest_record, dict):
        raise RuntimeError(f"{amendment_path}: manifest identity is missing")
    expected_path = path.relative_to(ROOT).as_posix()
    if manifest_record.get("path") != expected_path:
        raise RuntimeError(f"{amendment_path}: manifest path mismatch")
    if sha256(path.read_bytes()) != manifest_record.get("sha256"):
        raise RuntimeError(f"{amendment_path}: manifest hash mismatch")
    if amendment.get("observed_base_commit") != manifest.get("base_source_commit"):
        raise RuntimeError(f"{amendment_path}: observed source commit mismatch")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError(f"{path}: source inventory is missing")
    canonical_sources = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    if sha256(canonical_sources) != amendment.get("source_inventory_sha256"):
        raise RuntimeError(f"{amendment_path}: source inventory hash mismatch")
    if amendment.get("qualification_status") != "historical_nonqualifying":
        raise RuntimeError(f"{amendment_path}: unsafe qualification status")
    return True, 4


def verify_qwen35_h20_pair_evidence(root: Path) -> int:
    """Run the trusted checkout verifier for a Qwen3.5 evidence archive."""

    spec = importlib.util.spec_from_file_location(
        "_orbitkv_verify_qwen35_h20_pair_evidence",
        QWEN35_H20_PAIR_EVIDENCE_VERIFIER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"{root}: cannot load trusted Qwen3.5 pair-evidence verifier"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verify_archive = getattr(module, "verify_archive", None)
    if not callable(verify_archive):
        raise RuntimeError(
            f"{root}: trusted Qwen3.5 pair-evidence verifier has no API"
        )
    result = verify_archive(root)
    expected = {
        "schema": (
            "orbitkv.abi8-h20-qwen35-pair-evidence-verification.v1"
        ),
        "status": "passed",
        "epoch_count": 3,
        "pair_count": 6,
        "abi_version": 8,
        "exact_symbol_count": 40,
        "qualified": False,
        "hardware_attested": False,
        "performance_go": False,
    }
    if not isinstance(result, dict) or any(
        name not in result
        or type(result[name]) is not type(value)
        or result[name] != value
        for name, value in expected.items()
    ):
        raise RuntimeError(
            f"{root}: trusted Qwen3.5 pair-evidence verification is incomplete"
        )
    return expected["pair_count"]


def verify_qwen38_h20_diagnostic(root: Path) -> int:
    """Run the hash-pinned verifier for the unsealed Qwen3.8 diagnostic."""

    expected_root = QWEN38_H20_DIAGNOSTIC_MANIFEST.parent.resolve(strict=True)
    if root.resolve(strict=True) != expected_root:
        raise RuntimeError(
            f"{root}: Qwen3.8 diagnostic verifier is bound to {expected_root}"
        )
    verifier = QWEN38_H20_DIAGNOSTIC_VERIFIER.resolve(strict=True)
    if sha256(verifier.read_bytes()) != QWEN38_H20_DIAGNOSTIC_VERIFIER_SHA256:
        raise RuntimeError(f"{verifier}: trusted diagnostic verifier hash differs")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(verifier)],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{root}: trusted Qwen3.8 diagnostic verifier failed"
        ) from error
    expected = {
        "schema": "orbitkv.abi8-h20-qwen38-fp8-diagnostic-verification.v1",
        "status": "passed",
        "evidence_class": "diagnostic_only",
        "artifact_count": 43,
        "record_count": 16,
        "stderr_log_count": 16,
        "pair_count": 8,
        "hot_sample_count_per_mode_per_batch": 16,
        "sealed": False,
        "source_clean": False,
        "source_dirty": True,
        "preflight_bound": False,
        "hardware_attested": False,
        "qualified": False,
        "performance_go": False,
    }
    if result != expected:
        raise RuntimeError(
            f"{root}: trusted Qwen3.8 diagnostic verification is incomplete"
        )
    return expected["artifact_count"]


def verify_token_relocation_h20_diagnostic(root: Path) -> int:
    """Verify the unsealed token-relocation diagnostic with trusted code."""

    expected_root = (
        TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST.parent.resolve(strict=True)
    )
    if root.resolve(strict=True) != expected_root:
        raise RuntimeError(
            f"{root}: token-relocation diagnostic verifier is bound to "
            f"{expected_root}"
        )
    path = root / "manifest.json"
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{path}: manifest must be a JSON object")

    expected_values = {
        "schema": TOKEN_RELOCATION_H20_DIAGNOSTIC_SCHEMA,
        "artifact_count": len(TOKEN_RELOCATION_H20_DIAGNOSTIC_ARTIFACTS),
        "directories": list(TOKEN_RELOCATION_H20_DIAGNOSTIC_DIRECTORIES),
        "integrity_scope": (
            "all_payload_artifacts_except_manifest_and_checksum_index"
        ),
        "evidence_class": "diagnostic_only",
        "diagnostic_only": True,
        "qualification_claim": "diagnostic_only_not_qualified",
        "sealed": False,
        "source_clean": False,
        "source_dirty": True,
        "hardware_attested": False,
        "qualified": False,
        "performance_go": False,
        "record_schema": (
            "orbitkv.sglang-v0517-token-relocation-single-run.v1"
        ),
        "summary_schema": (
            "orbitkv.sglang-v0517-token-relocation-diagnostic-summary.v1"
        ),
        "epoch_count": 4,
        "batch_sizes": [1, 4],
        "record_count": 16,
        "pair_count": 8,
    }
    expected_keys = set(expected_values) | {
        "artifacts",
        "identity",
        "observed_hardware",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError(f"{path}: token-relocation manifest keys differ")
    for field, expected in expected_values.items():
        value = manifest.get(field)
        if type(value) is not type(expected) or value != expected:
            raise RuntimeError(
                f"{path}: token-relocation manifest {field} differs"
            )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        TOKEN_RELOCATION_H20_DIAGNOSTIC_ARTIFACTS
    ):
        raise RuntimeError(
            f"{path}: token-relocation artifact inventory differs"
        )
    expected_files = set(TOKEN_RELOCATION_H20_DIAGNOSTIC_ARTIFACTS) | {
        "manifest.json",
        "SHA256SUMS",
    }
    actual_files = sealed_regular_files(root)
    if actual_files != expected_files:
        raise RuntimeError(
            f"{path}: token-relocation file inventory differs; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unlisted={sorted(actual_files - expected_files)}"
        )
    actual_directories = {
        (Path(current_root) / name).relative_to(root).as_posix()
        for current_root, directory_names, _ in os.walk(
            root, followlinks=False
        )
        for name in directory_names
    }
    expected_directories = set(TOKEN_RELOCATION_H20_DIAGNOSTIC_DIRECTORIES)
    if actual_directories != expected_directories:
        raise RuntimeError(
            f"{path}: token-relocation directory inventory differs; "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"unlisted={sorted(actual_directories - expected_directories)}"
        )
    for relative_path, identity in artifacts.items():
        safe_path = checked_sealed_relative_path(relative_path)
        if (
            not isinstance(identity, dict)
            or set(identity) != {"bytes", "sha256"}
            or type(identity.get("bytes")) is not int
            or identity["bytes"] < 0
            or not isinstance(identity.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(identity["sha256"]) is None
        ):
            raise RuntimeError(
                f"{path}: invalid artifact identity for {relative_path}"
            )
        artifact_path = root / safe_path
        if artifact_path.stat().st_size != identity["bytes"]:
            raise RuntimeError(
                f"{path}: artifact byte count differs for {relative_path}"
            )
        if sha256(artifact_path.read_bytes()) != identity["sha256"]:
            raise RuntimeError(
                f"{path}: artifact hash differs for {relative_path}"
            )

    expected_sums = {
        relative_path: identity["sha256"]
        for relative_path, identity in artifacts.items()
    }
    expected_sums["manifest.json"] = sha256(path.read_bytes())
    expected_sums_text = "".join(
        f"{digest}  {relative_path}\n"
        for relative_path, digest in sorted(expected_sums.items())
    )
    sums_path = root / "SHA256SUMS"
    if sums_path.read_text(encoding="utf-8") != expected_sums_text:
        raise RuntimeError(
            f"{sums_path}: contents do not match the diagnostic inventory"
        )

    verifier = TOKEN_RELOCATION_H20_DIAGNOSTIC_VERIFIER.resolve(strict=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(verifier), str(root)],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{root}: trusted token-relocation diagnostic verifier failed"
        ) from error
    if not isinstance(result, dict):
        raise RuntimeError(
            f"{root}: trusted token-relocation verification is incomplete"
        )
    stored_summary = load_json(root / "summary.json")
    if result != stored_summary:
        raise RuntimeError(
            f"{root}: stored token-relocation summary differs from records"
        )
    trusted_values = {
        "schema": expected_values["summary_schema"],
        "status": "diagnostic_pair_verification_passed",
        "evidence_class": "diagnostic_only",
        "diagnostic_only": True,
        "qualification_claim": "diagnostic_only_not_qualified",
        "sealed": False,
        "source_clean": False,
        "source_dirty": True,
        "hardware_attested": False,
        "qualified": False,
        "performance_go": False,
        "epoch_count": 4,
        "batch_sizes": [1, 4],
        "record_count": 16,
        "pair_count": 8,
        "all_pairs_passed": True,
        "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }
    if any(
        name not in result
        or type(result[name]) is not type(expected)
        or result[name] != expected
        for name, expected in trusted_values.items()
    ):
        raise RuntimeError(
            f"{root}: trusted token-relocation verification is incomplete"
        )
    hardware = result.get("hardware")
    if (
        not isinstance(hardware, dict)
        or manifest.get("identity") != result.get("identity")
        or manifest.get("observed_hardware")
        != {
            "name": hardware.get("observed_name"),
            "uuid": hardware.get("observed_uuid"),
            "snapshot_count": hardware.get("snapshot_count"),
            "attestation": hardware.get("attestation"),
        }
    ):
        raise RuntimeError(
            f"{root}: token-relocation manifest identity differs from records"
        )
    verify_token_relocation_component_conformance(
        root / "component-conformance.xml", manifest["observed_hardware"]
    )
    return expected_values["artifact_count"]


def verify_token_relocation_component_conformance(
    path: Path, observed_hardware: dict[str, object]
) -> None:
    """Validate real-CUDA cases and bind their JUnit record to the GPU."""

    try:
        document = ET.parse(path)
    except (OSError, ET.ParseError) as error:
        raise RuntimeError(f"{path}: invalid component-conformance XML") from error
    root = document.getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if len(suites) != 1:
        raise RuntimeError(f"{path}: expected exactly one test suite")
    suite = suites[0]
    expected_counts = {
        "tests": str(len(TOKEN_RELOCATION_COMPONENT_CASES)),
        "errors": "0",
        "failures": "0",
        "skipped": "0",
    }
    if any(suite.get(name) != value for name, value in expected_counts.items()):
        raise RuntimeError(f"{path}: component-conformance counts differ")
    cases = tuple(case.get("name") for case in suite.findall("testcase"))
    if len(cases) != len(set(cases)) or set(cases) != set(
        TOKEN_RELOCATION_COMPONENT_CASES
    ):
        raise RuntimeError(f"{path}: component-conformance cases differ")

    property_nodes = suite.findall("./properties/property")
    properties = {node.get("name"): node.get("value") for node in property_nodes}
    if len(properties) != len(property_nodes):
        raise RuntimeError(f"{path}: duplicate component-conformance property")
    expected_properties = {
        "orbitkv.cuda.available": "true",
        "orbitkv.cuda.device_name": observed_hardware.get("name"),
        "orbitkv.cuda.device_uuid": observed_hardware.get("uuid"),
    }
    if any(properties.get(name) != value for name, value in expected_properties.items()):
        raise RuntimeError(
            f"{path}: component-conformance device identity differs"
        )
    for name in ("orbitkv.cuda.runtime_version", "orbitkv.torch.version"):
        value = properties.get(name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"{path}: component-conformance runtime identity is incomplete"
            )


def verify_manifest(path: Path) -> int:
    path = Path(os.path.abspath(path))
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{path}: manifest must be a JSON object")
    schema = manifest.get("schema")
    if schema == ABI8_H20_SCHEMA:
        return verify_abi8_h20_manifest(path, manifest)
    if schema == QWEN35_H20_PAIR_EVIDENCE_SCHEMA:
        if path.name != "manifest.json":
            raise RuntimeError(
                f"{path}: Qwen3.5 pair-evidence manifest must be named manifest.json"
            )
        return verify_qwen35_h20_pair_evidence(path.parent)
    if schema == QWEN38_H20_DIAGNOSTIC_SCHEMA:
        if path.name != "manifest.json":
            raise RuntimeError(
                f"{path}: Qwen3.8 diagnostic manifest must be named manifest.json"
            )
        return verify_qwen38_h20_diagnostic(path.parent)
    if schema == TOKEN_RELOCATION_H20_DIAGNOSTIC_SCHEMA:
        if path.name != "manifest.json":
            raise RuntimeError(
                f"{path}: token-relocation diagnostic manifest must be named "
                "manifest.json"
            )
        return verify_token_relocation_h20_diagnostic(path.parent)
    if (
        isinstance(schema, str)
        and schema.startswith("orbitkv.abi8-h20-sealed-manifest.")
    ) or "artifacts" in manifest:
        raise RuntimeError(f"{path}: unsupported sealed manifest schema")
    historical_commit = manifest.get("base_source_commit") or manifest.get(
        "source_commit"
    )
    workspace_sections = set(manifest.get("workspace_sections", ()))
    checked = verify_source_provenance(manifest)
    source_commit, amendment_checks = sealed_source_commit(path, manifest)
    checked += amendment_checks
    skip_unsealed_sources, amendment_checks = historical_unsealed_sources(path, manifest)
    checked += amendment_checks
    for section in ("records", "sources", "website"):
        if section == "sources" and skip_unsealed_sources:
            continue
        if section == "sources" and manifest.get("source_provenance_amendment"):
            continue
        for relative_path, expected in manifest.get(section, {}).items():
            if section == "sources" and source_commit is not None:
                data = git_blob(source_commit, relative_path)
            elif historical_commit is not None and section not in workspace_sections:
                data = git_blob(historical_commit, relative_path)
            else:
                data = (ROOT / relative_path).read_bytes()
            actual = sha256(data)
            if actual != expected:
                raise RuntimeError(
                    f"{path}: {relative_path}: expected {expected}, got {actual}"
                )
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="*", type=Path)
    args = parser.parse_args()
    manifests = args.manifests or list(DEFAULT_MANIFESTS)
    checked = sum(verify_manifest(path) for path in manifests)
    print(f"verified {checked} manifest hashes")


if __name__ == "__main__":
    main()
