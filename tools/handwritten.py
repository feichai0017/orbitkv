"""Handwritten kernels (tools/kernels-src/*.cu): build once, pin by sha256.

A generator writes `**hw("gemm8")` into a step and gets
`{"cubin": "gemm8.cubin", "sha256": "<hash of the current build>"}` — the
display name plus the identity the runtime resolves. The build happens at
most once per process (tools/build_kernels.sh, into target/cubins/).

`prebuilt("mla_decode_h96_p64")` does the same for a checked-in artifact
(tools/kernels-bin/*.cubin, built by a toolchain nvcc is not: see the
README there); build_kernels.sh lands those in target/cubins too.
"""
import functools
import hashlib
import os
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def build_dir() -> pathlib.Path:
    out = REPO / "target" / "cubins"
    subprocess.run([str(REPO / "tools" / "build_kernels.sh"), str(out)], check=True)
    return out


def hw(name: str, **defines) -> dict:
    """`cubin` + `sha256` fields for tools/kernels-src/<name>.cu as built now.
    `defines` (e.g. `HEADS=24`) select a variant of the source, built here
    as `<name>+HEADS=24.cubin` next to the plain build."""
    out = build_dir()
    if not defines:
        cb = out / f"{name}.cubin"
    else:
        key = "+".join(f"{k}={v}" for k, v in sorted(defines.items()))
        cb = out / f"{name}+{key}.cubin"
        src = REPO / "tools" / "kernels-src" / f"{name}.cu"
        if not cb.exists() or cb.stat().st_mtime < src.stat().st_mtime:
            arch = os.environ.get("KERN_SM", "sm_103a")
            flags = [f"-D{k}={v}" for k, v in sorted(defines.items())]
            subprocess.run(["nvcc", "-cubin", f"-arch={arch}", *flags, "-o", str(cb), str(src)], check=True)
    return {"cubin": cb.name, "sha256": hashlib.sha256(cb.read_bytes()).hexdigest()}


def prebuilt(name: str) -> dict:
    """`cubin` + `sha256` fields for the checked-in tools/kernels-bin/<name>.cubin."""
    build_dir()
    cb = REPO / "tools" / "kernels-bin" / f"{name}.cubin"
    return {"cubin": cb.name, "sha256": hashlib.sha256(cb.read_bytes()).hexdigest()}
