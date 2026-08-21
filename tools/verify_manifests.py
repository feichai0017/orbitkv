from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PUBLISH_GIT = Path("/tmp/orbitkv-publish.git")
DEFAULT_MANIFESTS = (
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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def verify_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
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
    checked = sum(verify_manifest(path.resolve()) for path in manifests)
    print(f"verified {checked} manifest hashes")


if __name__ == "__main__":
    main()
