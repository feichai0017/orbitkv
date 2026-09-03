from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import RuntimeConfig, load_config
from .ffi.library import LoadedLibrary
from .pinned import validate_patched_source
from .runtime_admission import (
    ProductTakeoverProfile,
    admit_product_takeover_config,
    runtime_binding_from_manifest,
)
from .runtime_manifest import RUNTIME_MANIFEST_MAX_BYTES, validate_runtime_manifest


_COMPILER_ENV = "ORBITKV_BINARY"
_DEFAULT_CHUNKED_PREFILL_TOKENS = 4096
_LOCKED_SERVER_OPTIONS = frozenset(
    {
        "--attention-backend",
        "--chunked-prefill-size",
        "--dcp-size",
        "--decode-attention-backend",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--dp-size",
        "--dtype",
        "--enable-cuda-graph",
        "--enable-dynamic-chunking",
        "--enable-dp-attention",
        "--enable-hierarchical-cache",
        "--enable-lmcache",
        "--enable-overlap-schedule",
        "--kv-cache-dtype",
        "--model",
        "--model-path",
        "--max-prefill-tokens",
        "--max-running-requests",
        "--page-size",
        "--pipeline-parallel-size",
        "--pp-size",
        "--prefill-attention-backend",
        "--prefill-max-requests",
        "--radix-cache-backend",
        "--schedule-policy",
        "--tensor-parallel-size",
        "--tp-size",
    }
)
_RUNTIME_ENVIRONMENT = {
    "SGLANG_USE_HND_KVCACHE": "0",
    "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
    "SGLANG_RADIX_FORCE_MISS": "0",
}


@dataclass(frozen=True, slots=True)
class ServeLaunch:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    cache_policy: str


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbitkv-engine",
        description="Compile and serve the verified OrbitKV Engine product.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="compile a Hugging Face config into a runtime manifest"
    )
    compile_parser.add_argument(
        "--model", required=True, help="model directory or config.json path"
    )
    compile_parser.add_argument("--output", required=True, type=Path)
    compile_parser.add_argument("--page-tokens", type=_positive, default=16)
    compile_parser.add_argument("--kv-dtype-bytes", type=_positive, default=2)

    serve_parser = subparsers.add_parser(
        "serve", help="launch the pinned SGLang server with an admitted manifest"
    )
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--manifest", required=True, type=Path)
    serve_parser.add_argument("--library", required=True, type=Path)
    serve_parser.add_argument("--sglang-root", required=True, type=Path)
    serve_parser.add_argument(
        "--print-command",
        "--dry-run",
        dest="print_command",
        action="store_true",
        help="validate everything and print the launch command without executing it",
    )
    serve_parser.add_argument(
        "server_arguments",
        nargs=argparse.REMAINDER,
        metavar="-- ...",
        help="additional non-conflicting SGLang server arguments",
    )
    return parser


def _model_config(model: str) -> Path:
    candidate = Path(model).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"invalid --model {model!r}: {error}") from error
    if resolved.is_dir():
        resolved = resolved / "config.json"
    if not resolved.is_file():
        raise ValueError("--model must name a config.json file or model directory")
    return resolved


def _compiler_path(environ: Mapping[str, str]) -> Path:
    configured = environ.get(_COMPILER_ENV)
    if configured:
        candidate = Path(configured).expanduser()
    else:
        discovered = shutil.which("orbitkv")
        if discovered is None:
            raise RuntimeError(
                "cannot find the built orbitkv binary; put it on PATH or set "
                f"{_COMPILER_ENV}"
            )
        candidate = Path(discovered)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid {_COMPILER_ENV} binary {candidate}: {error}") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError("the orbitkv compiler must be an executable regular file")
    return resolved


def build_compile_command(
    compiler: Path,
    model_config: Path,
    *,
    page_tokens: int = 16,
    kv_dtype_bytes: int = 2,
) -> tuple[str, ...]:
    return (
        str(compiler),
        "compile-hf-runtime-manifest",
        str(model_config),
        "--page-tokens",
        str(page_tokens),
        "--kv-dtype-bytes",
        str(kv_dtype_bytes),
    )


def build_bind_command(compiler: Path, manifest: Path) -> tuple[str, ...]:
    return (
        str(compiler),
        "bind-runtime-manifest",
        str(manifest),
    )


def _run_compiler(command: Sequence[str], operation: str) -> bytes:
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, timeout=120
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"orbitkv {operation} exited with status {error.returncode}{suffix}"
        ) from error
    return completed.stdout


def _write_atomic(path: Path, payload: bytes) -> None:
    parent = path.expanduser().absolute().parent
    if not parent.is_dir():
        raise ValueError(f"--output parent is not a directory: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path.expanduser().absolute())
    finally:
        if temporary.exists():
            temporary.unlink()


def compile_manifest(
    model: str,
    output: Path,
    *,
    page_tokens: int = 16,
    kv_dtype_bytes: int = 2,
    environ: Mapping[str, str] | None = None,
) -> Path:
    source = os.environ if environ is None else environ
    compiler = _compiler_path(source)
    command = build_compile_command(
        compiler,
        _model_config(model),
        page_tokens=page_tokens,
        kv_dtype_bytes=kv_dtype_bytes,
    )
    encoded_manifest = _run_compiler(command, "manifest compilation")
    if len(encoded_manifest) > RUNTIME_MANIFEST_MAX_BYTES:
        raise RuntimeError("compiled runtime manifest exceeds its size limit")
    try:
        raw = json.loads(encoded_manifest)
        manifest_root = validate_runtime_manifest(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"orbitkv emitted an invalid runtime manifest: {error}") from error
    destination = output.expanduser().absolute()
    if not destination.parent.is_dir():
        raise ValueError(f"--output parent is not a directory: {destination.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.binding-",
        suffix=".json",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_manifest:
            temporary_manifest.write(encoded_manifest)
            temporary_manifest.flush()
            os.fsync(temporary_manifest.fileno())
        encoded_binding = _run_compiler(
            build_bind_command(compiler, temporary), "SGLang runtime binding"
        )
    finally:
        temporary.unlink(missing_ok=True)
    if len(encoded_binding) > RUNTIME_MANIFEST_MAX_BYTES:
        raise RuntimeError("compiled runtime binding exceeds its size limit")
    try:
        binding = json.loads(encoded_binding)
        expected_binding = runtime_binding_from_manifest(manifest_root)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"orbitkv emitted an invalid runtime binding: {error}") from error
    if binding != expected_binding:
        raise RuntimeError(
            "Rust and Python derived different SGLang runtime bindings"
        )
    _write_atomic(destination, encoded_manifest)
    return destination


def _server_extra_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    values = tuple(arguments)
    if values[:1] == ("--",):
        values = values[1:]
    for argument in values:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        if option in _LOCKED_SERVER_OPTIONS:
            raise ValueError(
                f"SGLang option {option} is fixed by the OrbitKV runtime contract"
            )
    return values


def build_serve_command(
    model: str,
    config: RuntimeConfig,
    profile: ProductTakeoverProfile,
    server_arguments: Sequence[str] = (),
    *,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    if profile.cache_policy not in ("shared_prefix", "request_private"):
        raise ValueError(f"unsupported OrbitKV cache policy {profile.cache_policy!r}")
    chunked = config.chunked_class
    chunk_tokens = (
        int(chunked.chunk_tokens)
        if chunked is not None and chunked.chunk_tokens is not None
        else _DEFAULT_CHUNKED_PREFILL_TOKENS
    )
    command = [
        sys.executable if python_executable is None else python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--page-size",
        str(config.page_tokens),
        "--attention-backend",
        "fa3",
        "--schedule-policy",
        "fcfs",
        "--chunked-prefill-size",
        str(chunk_tokens),
        "--max-running-requests",
        "1",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--radix-cache-backend",
        "orbitkv",
        "--tp-size",
        "1",
        "--pp-size",
        "1",
        "--dp-size",
        "1",
        "--dcp-size",
        "1",
    ]
    if chunked is not None:
        command.extend(("--prefill-max-requests", "1"))
        command.extend(("--max-prefill-tokens", str(chunk_tokens)))
    if profile.cache_policy == "request_private":
        command.append("--disable-radix-cache")
    command.extend(_server_extra_arguments(server_arguments))
    return tuple(command)


def prepare_serve_launch(
    *,
    model: str,
    manifest: Path,
    library: Path,
    sglang_root: Path,
    server_arguments: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> ServeLaunch:
    root = validate_patched_source(sglang_root)
    runtime_environment = {
        "ORBITKV_RUNTIME_MANIFEST": str(manifest),
        "ORBITKV_LIBRARY": str(library),
    }
    config = load_config(runtime_environment)
    LoadedLibrary(config.library_path)
    profile = admit_product_takeover_config(config)
    command = build_serve_command(
        model,
        config,
        profile,
        server_arguments,
        python_executable=python_executable,
    )

    base = dict(os.environ if environ is None else environ)
    for name in tuple(base):
        if name.startswith("ORBITKV_"):
            base.pop(name)
    base.update(_RUNTIME_ENVIRONMENT)
    base.update(
        ORBITKV_RUNTIME_MANIFEST=str(config.runtime_manifest_path),
        ORBITKV_LIBRARY=str(config.library_path),
        ORBITKV_SGLANG_ROOT=str(root),
    )
    python_root = str(root / "python")
    prior_python_path = base.get("PYTHONPATH")
    base["PYTHONPATH"] = (
        os.pathsep.join((python_root, prior_python_path))
        if prior_python_path
        else python_root
    )
    return ServeLaunch(command, base, profile.cache_policy)


def format_launch(launch: ServeLaunch) -> str:
    names = (
        "ORBITKV_RUNTIME_MANIFEST",
        "ORBITKV_LIBRARY",
        "ORBITKV_SGLANG_ROOT",
        "PYTHONPATH",
        *_RUNTIME_ENVIRONMENT,
    )
    assignments = tuple(f"{name}={launch.environment[name]}" for name in names)
    return shlex.join((*assignments, *launch.command))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            compile_manifest(
                args.model,
                args.output,
                page_tokens=args.page_tokens,
                kv_dtype_bytes=args.kv_dtype_bytes,
            )
            return 0
        launch = prepare_serve_launch(
            model=args.model,
            manifest=args.manifest,
            library=args.library,
            sglang_root=args.sglang_root,
            server_arguments=args.server_arguments,
        )
        if args.print_command:
            print(format_launch(launch))
            return 0
        os.execvpe(launch.command[0], launch.command, dict(launch.environment))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
