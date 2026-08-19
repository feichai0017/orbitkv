from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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


def verify_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    historical_commit = manifest.get("base_source_commit") or manifest.get(
        "source_commit"
    )
    workspace_sections = set(manifest.get("workspace_sections", ()))
    checked = 0
    for section in ("records", "sources", "website"):
        for relative_path, expected in manifest.get(section, {}).items():
            if historical_commit is not None and section not in workspace_sections:
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
