#!/usr/bin/env python3
"""Verify and bootstrap the source-pinned inference development stacks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "third-party" / "sources.lock.toml"


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def sources() -> list[dict[str, Any]]:
    with LOCK.open("rb") as stream:
        return tomllib.load(stream)["source"]


def verify() -> None:
    failures: list[str] = []
    report = []
    for source in sources():
        path = ROOT / source["path"]
        if not path.is_dir():
            failures.append(f"{source['name']}: missing {path}")
            continue
        try:
            revision = run(["git", "rev-parse", "HEAD"], cwd=path, capture=True)
            origin = run(["git", "remote", "get-url", "origin"], cwd=path, capture=True)
        except subprocess.CalledProcessError as error:
            failures.append(f"{source['name']}: not a usable Git checkout ({error})")
            continue
        if revision != source["revision"]:
            failures.append(f"{source['name']}: expected {source['revision']}, got {revision}")
        if origin.rstrip("/") != source["repository"].rstrip("/"):
            failures.append(f"{source['name']}: expected origin {source['repository']}, got {origin}")
        missing = [item for item in source["required_paths"] if not (path / item).exists()]
        if missing:
            failures.append(f"{source['name']}: missing required paths {', '.join(missing)}")
        for requirement in source.get("required_symbols", []):
            relative, needle = requirement.split("::", 1)
            candidate = path / relative
            if candidate.is_file() and needle not in candidate.read_text(errors="replace"):
                failures.append(f"{source['name']}: missing symbol {needle!r} in {relative}")
        license_files = list(path.glob("LICENSE*"))
        if not license_files:
            failures.append(f"{source['name']}: no top-level LICENSE file")
        nested = _nested_submodules_initialized(path)
        if nested is False:
            failures.append(f"{source['name']}: nested submodules are not initialized")
        mode = run(["git", "ls-files", "--stage", source["path"]], capture=True).split(maxsplit=1)[0]
        if mode != "160000":
            failures.append(f"{source['name']}: {source['path']} is not recorded as a Git submodule")
        report.append(
            {
                "name": source["name"],
                "path": source["path"],
                "revision": revision,
                "ref": source["ref"],
                "nested_submodules_initialized": nested,
            }
        )
    print(json.dumps({"schema": 1, "sources": report}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("source verification failed:\n- " + "\n- ".join(failures))


def _nested_submodules_initialized(path: Path) -> bool | None:
    if not (path / ".gitmodules").exists():
        return None
    status = run(["git", "submodule", "status", "--recursive"], cwd=path, capture=True)
    return all(not line.startswith("-") for line in status.splitlines())


def commands(stack: str, environment: Path) -> list[tuple[Path, list[str]]]:
    python = environment / "bin" / "python"
    pip = [str(python), "-m", "pip"]
    source_python = _build_environment(environment)
    common = [(ROOT, [sys.executable, "-m", "venv", str(environment)])]
    if stack == "sglang":
        return common + [
            (ROOT, pip + ["install", "--upgrade", "pip", "setuptools", "wheel", "build", "scikit-build-core>=0.10"]),
            (ROOT, pip + ["install", "-e", str(ROOT / "third-party/sglang/python")]),
            (
                ROOT,
                _build_environment(environment) + [str(python), str(ROOT / "tools/build_sglang_kernel.py")],
            ),
            (ROOT, pip + ["install", "--force-reinstall", "--no-deps", _sglang_kernel_wheel()]),
            (ROOT, source_python + [str(python), str(ROOT / "tools/build_deepgemm.py")]),
            (ROOT, pip + ["install", "--force-reinstall", "--no-deps", _deepgemm_wheel()]),
            (ROOT, source_python + pip + ["install", "-e", str(ROOT / "third-party/flashinfer")]),
            (ROOT, pip + ["install", "-e", str(ROOT / "integrations/sglang")]),
            (ROOT, pip + ["install", "-e", str(ROOT / "integrations/providers")]),
        ]
    if stack == "autodeploy":
        return common + [
            (ROOT, pip + ["install", "--upgrade", "pip", "setuptools", "wheel"]),
            (
                ROOT,
                ["env", "TRTLLM_USE_PRECOMPILED=1", *pip, "install", "-e", str(ROOT / "third-party/tensorrt-llm")],
            ),
            (ROOT, pip + ["install", "-e", str(ROOT / "integrations/autodeploy")]),
        ]
    if stack == "tensorrt-source":
        return common + [
            (ROOT, pip + ["install", "--upgrade", "pip", "setuptools", "wheel"]),
            (
                ROOT / "third-party/tensorrt-llm",
                source_python
                + [
                    str(python),
                    "scripts/build_wheel.py",
                    "--cuda_architectures",
                    os.environ.get("ALETHEIA_CUDA_ARCHITECTURES", "90-real"),
                ],
            ),
            (ROOT, pip + ["install", str(ROOT / "third-party/tensorrt-llm/build/tensorrt_llm-*.whl")]),
            (ROOT, pip + ["install", "-e", str(ROOT / "integrations/autodeploy")]),
        ]
    if stack == "kernels":
        return common + [
            (ROOT, pip + ["install", "--upgrade", "pip", "setuptools", "wheel", "build"]),
            (ROOT, pip + ["install", "torch==2.13.0"]),
            (ROOT, source_python + pip + ["install", "-e", str(ROOT / "third-party/flashinfer")]),
            (ROOT, source_python + [str(python), str(ROOT / "tools/build_deepgemm.py")]),
            (ROOT, pip + ["install", "--force-reinstall", "--no-deps", _deepgemm_wheel()]),
            (ROOT, pip + ["install", "-e", str(ROOT / "integrations/providers")]),
        ]
    raise ValueError(stack)


def _deepgemm_wheel() -> str:
    return str(ROOT / "third-party/deepgemm/dist/sgl_deep_gemm-*.whl")


def _sglang_kernel_wheel() -> str:
    return str(ROOT / "third-party/sglang/python/sglang/kernels/aot/dist/sglang_kernel-*.whl")


def _find_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    return next(
        (str(path) for path in sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True) if path.is_file()),
        None,
    )


def _build_environment(environment: Path, extra: dict[str, str] | None = None) -> list[str]:
    paths = [str(environment / "bin")]
    arguments = ["env"]
    cache_root = ROOT / ".aletheia" / "cache"
    arguments.extend(
        [
            f"FLASHINFER_WORKSPACE_BASE={cache_root / 'flashinfer'}",
            f"TORCH_EXTENSIONS_DIR={cache_root / 'torch'}",
            f"DG_JIT_CACHE_DIR={cache_root / 'deepgemm'}",
        ]
    )
    arguments.extend(f"{name}={value}" for name, value in (extra or {}).items())
    nvcc = _find_nvcc()
    if nvcc:
        cuda_home = str(Path(nvcc).resolve().parents[1])
        paths.append(str(Path(cuda_home) / "bin"))
        arguments.append(f"CUDA_HOME={cuda_home}")
    paths.append(os.environ.get("PATH", ""))
    arguments.append(f"PATH={':'.join(paths)}")
    return arguments


def bootstrap(stack: str, environment: Path, execute: bool) -> None:
    plan = commands(stack, environment)
    print(
        json.dumps(
            {
                "stack": stack,
                "environment": str(environment),
                "commands": [{"cwd": str(cwd), "argv": argv} for cwd, argv in plan],
            },
            indent=2,
        )
    )
    if not execute:
        return
    for name in ("flashinfer", "torch", "deepgemm"):
        (ROOT / ".aletheia" / "cache" / name).mkdir(parents=True, exist_ok=True)
    for cwd, command in plan:
        expanded = _expand_wheel(command)
        run(expanded, cwd=cwd)


def doctor() -> None:
    nvcc = _find_nvcc()
    nvidia_smi = shutil.which("nvidia-smi")
    driver_usable = False
    gpu = None
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        driver_usable = result.returncode == 0
        gpu = result.stdout.strip() if driver_usable else None

    torch = None
    try:
        import torch as torch_module

        torch = {
            "version": torch_module.__version__,
            "cuda": torch_module.version.cuda,
            "cuda_available": torch_module.cuda.is_available(),
        }
    except ImportError:
        pass

    def stack(*, needs_nvcc: bool) -> dict[str, Any]:
        missing_build = ["nvcc"] if needs_nvcc and nvcc is None else []
        missing_run = [] if driver_usable else ["usable NVIDIA driver"]
        return {
            "build_ready": not missing_build,
            "run_ready": not missing_run,
            "missing_build": missing_build,
            "missing_run": missing_run,
        }

    stacks = {
        "source-only": {
            "build_ready": True,
            "run_ready": True,
            "missing_build": [],
            "missing_run": [],
        },
        "sglang": stack(needs_nvcc=True),
        "autodeploy": stack(needs_nvcc=False),
        "tensorrt-source": stack(needs_nvcc=True),
        "kernels": stack(needs_nvcc=True),
    }
    print(
        json.dumps(
            {
                "schema": 1,
                "gpu": gpu,
                "driver_usable": driver_usable,
                "nvcc": nvcc,
                "torch": torch,
                "stacks": stacks,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _expand_wheel(command: list[str]) -> list[str]:
    if not command[-1].endswith("*.whl"):
        return command
    wheels = sorted(Path(command[-1]).parent.glob(Path(command[-1]).name))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel matching {command[-1]}, found {len(wheels)}")
    return [*command[:-1], str(wheels[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    subparsers.add_parser("doctor")
    boot = subparsers.add_parser("bootstrap")
    boot.add_argument(
        "--stack",
        choices=["sglang", "autodeploy", "tensorrt-source", "kernels"],
        required=True,
    )
    boot.add_argument("--environment", type=Path)
    boot.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "doctor":
        doctor()
    else:
        environment = args.environment or ROOT / ".venv" / args.stack
        bootstrap(args.stack, environment.resolve(), args.execute)


if __name__ == "__main__":
    main()
