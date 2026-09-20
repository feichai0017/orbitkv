"""Own benchmark service processes and capture the execution environment."""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import signal
import socket
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def server(
    command: list[str],
    env: dict[str, str],
    url: str,
    log: Path,
    health_path: str = "/health",
):
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {log}\n{log.read_text()[-6000:]}")
            try:
                if requests.get(url + health_path, timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError(f"Server startup timed out: {log}")
        yield
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def manifest(args: Namespace, launch, bytes_per_token: int) -> dict:
    packages = [args.engine, "torch", "transformers", "numpy", "prometheus_client"]
    if args.backend in ("lmcache", "flexkv"):
        packages.append(args.backend)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "engine_command": launch.command,
        "manager_command": launch.manager_command,
        "backend_configuration": launch.backend_configuration,
        "library_path": launch.env.get("LD_LIBRARY_PATH", ""),
        "python_path": launch.env["PYTHONPATH"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python": sys.version,
        "kv_bytes_per_token": bytes_per_token,
        "model_revision": (args.model / ".revision").read_text().strip()
        if (args.model / ".revision").exists()
        else None,
        "concurrency": 1,
        "notes": "Serial latency experiment; pressure traffic and startup are excluded from request timings.",
    }
    return manifest
