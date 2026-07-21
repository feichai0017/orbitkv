#!/usr/bin/env python3
"""Build the C++ PyTorch dispatcher shim without recompiling CUDA kernels."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

from torch.utils.cpp_extension import load


def main() -> None:
    interpreter_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")
    repository = Path(__file__).resolve().parents[1]
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cuda_include = cuda_home / "include"
    if not cuda_include.is_dir():
        raise FileNotFoundError(f"CUDA headers not found below {cuda_home}")
    build_root = repository / "build"
    native_library = build_root / "libloom_kernels_cuda.so"
    if not native_library.is_file():
        raise FileNotFoundError(
            f"{native_library} is missing; run python/build_native.py first"
        )

    extension_build = build_root / "torch_extension"
    extension_build.mkdir(parents=True, exist_ok=True)
    loaded_path = load(
        name="loom_kernels_torch_ops",
        sources=[str(repository / "python" / "csrc" / "torch_ops.cpp")],
        extra_include_paths=[
            str(repository / "cuda" / "include"),
            str(cuda_include),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[
            f"-L{build_root}",
            "-lloom_kernels_cuda",
            "-Wl,-rpath,'$$ORIGIN'",
            "-Wl,-rpath,'$$ORIGIN/..'",
        ],
        build_directory=str(extension_build),
        is_python_module=False,
        verbose=True,
    )
    source = Path(loaded_path)
    output = build_root / "libloom_kernels_torch.so"
    shutil.copy2(source, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
