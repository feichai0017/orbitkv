#!/usr/bin/env python3
"""Build the pinned SGLang kernel wheel with a reversible Hopper profile."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third-party" / "sglang"
KERNEL = SOURCE / "python" / "sglang" / "kernels" / "aot"
PATCH = ROOT / "third-party" / "patches" / "sglang" / "0001-hopper-only-kernel-build.patch"
REVISION = "095ec6c997bfdd25d3864cb0ce77a6562a934b96"
BUILD = ROOT / ".aletheia" / "build" / "sglang-kernel-hopper"
DIST = KERNEL / "dist"


def check_output(arguments: list[str]) -> str:
    return subprocess.check_output(arguments, cwd=SOURCE, text=True).strip()


def run(arguments: list[str], *, cwd: Path = SOURCE, check: bool = True) -> None:
    subprocess.run(arguments, cwd=cwd, check=check)


def main() -> None:
    if check_output(["git", "rev-parse", "HEAD"]) != REVISION:
        raise SystemExit("SGLang Hopper patch does not match the checked-out revision")
    if check_output(["git", "status", "--porcelain"]):
        raise SystemExit("SGLang submodule must be clean before the build")

    jobs = os.environ.get("ALETHEIA_BUILD_JOBS", "8")
    cmake_args = "-DALETHEIA_HOPPER_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=1"
    environment = os.environ.copy()
    environment.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": jobs,
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "CMAKE_ARGS": cmake_args,
        }
    )
    applied = False
    try:
        run(["git", "apply", "--check", str(PATCH)])
        run(["git", "apply", str(PATCH)])
        applied = True
        BUILD.mkdir(parents=True, exist_ok=True)
        DIST.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                f"-Cbuild-dir={BUILD}",
                str(KERNEL),
                "--outdir",
                str(DIST),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    finally:
        if applied:
            run(["git", "apply", "--reverse", str(PATCH)], check=False)

    if check_output(["git", "status", "--porcelain"]):
        raise SystemExit("SGLang submodule is dirty after reversing the build patch")
    print(
        json.dumps(
            {
                "revision": REVISION,
                "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                "profile": "hopper-only",
                "jobs": int(jobs),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
