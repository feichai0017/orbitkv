#!/usr/bin/env python3
"""Build the pinned SGLang DeepGEMM wheel with a reversible patch stack."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third-party" / "deepgemm"
PATCHES = ROOT / "third-party" / "patches" / "deepgemm"
EXPECTED_REVISION = "fa3a5ca07d768dd0f9089f70a445208b166c48d1"


def run(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=SOURCE, text=True, check=check)


def output(arguments: list[str]) -> str:
    return subprocess.check_output(arguments, cwd=SOURCE, text=True).strip()


def main() -> None:
    revision = output(["git", "rev-parse", "HEAD"])
    if revision != EXPECTED_REVISION:
        raise SystemExit(f"DeepGEMM patch stack expects {EXPECTED_REVISION}, got {revision}")
    if output(["git", "status", "--porcelain"]):
        raise SystemExit("DeepGEMM submodule must be clean before applying the build patch stack")

    patches = sorted(PATCHES.glob("*.patch"))
    applied: list[Path] = []
    try:
        for patch in patches:
            run(["git", "apply", "--check", str(patch)])
            run(["git", "apply", str(patch)])
            applied.append(patch)
        run(["bash", "build_sgl_deep_gemm.sh"])
    finally:
        for patch in reversed(applied):
            run(["git", "apply", "--reverse", str(patch)], check=False)

    if output(["git", "status", "--porcelain"]):
        raise SystemExit("DeepGEMM submodule is dirty after reversing the build patch stack")
    print(
        json.dumps(
            {
                "revision": revision,
                "patches": [
                    {"path": str(patch.relative_to(ROOT)), "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
                    for patch in patches
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
