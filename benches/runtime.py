"""Own benchmark service processes and capture the execution environment."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.metadata
import os
import secrets
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
    """Avoid outgoing TCP ports while GPU initialization delays the listener."""
    low, high = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    for _ in range(128):
        port = 1024 + secrets.randbelow(65536 - 1024)
        if low <= port <= high:
            continue
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                continue
            return sock.getsockname()[1]
    raise RuntimeError("No free benchmark listener port found")


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
        yield process
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def storage_manifest(pid: int, cache_path: Path) -> dict:
    flags = []
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if fd.resolve() != cache_path.resolve():
                continue
            info = Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text()
        except FileNotFoundError:
            continue
        flags.extend(
            int(line.split()[1], 8) for line in info.splitlines() if line.startswith("flags:")
        )
    if not flags or not all(value & os.O_DIRECT for value in flags):
        raise RuntimeError("SSD measurement requires an open O_DIRECT cache file")
    return {
        "path": str(cache_path),
        "capacity_bytes": cache_path.stat().st_size,
        "fd_flags": flags,
        "direct_io": True,
        "payload_retained_after_run": False,
        "mount": subprocess.check_output(
            ["findmnt", "--json", "--target", str(cache_path), "--output", "TARGET,SOURCE,FSTYPE"],
            text=True,
        ),
        "devices": subprocess.check_output(
            ["lsblk", "--json", "--output", "NAME,TYPE,SIZE,ROTA,MODEL"], text=True
        ),
        "notes": "Device inventory does not prove which physical disk backs an overlay mount.",
    }


def process_usage(pid: int) -> dict:
    """Process CPU and I/O counters, excluding child inference processes."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    io = dict(line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
    return {
        "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
        "read_bytes": int(io["read_bytes"]),
        "write_bytes": int(io["write_bytes"]),
    }


def manifest(args: Namespace, launch, bytes_per_token: int) -> dict:
    packages = [args.engine, "torch", "transformers", "numpy", "prometheus_client"]
    if args.backend in ("lmcache", "flexkv"):
        packages.append(args.backend)
    working_set_tokens = (
        sum(args.lengths[index % len(args.lengths)] for index in range(args.working_set))
        if args.workload == "sustained"
        else None
    )
    manager_sha256 = None
    if args.backend == "orbitkv":
        digest = hashlib.sha256()
        with Path(launch.manager_command[0]).open("rb") as binary:
            for chunk in iter(lambda: binary.read(1024**2), b""):
                digest.update(chunk)
        manager_sha256 = digest.hexdigest()
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "engine_command": launch.command,
        "manager_command": launch.manager_command,
        "manager_binary_sha256": manager_sha256,
        "cost_observations": "1" if launch.env.get("ORBITKV_COST_OBSERVATIONS") == "1" else "0",
        "backend_configuration": launch.backend_configuration,
        "library_path": launch.env.get("LD_LIBRARY_PATH", ""),
        "python_path": launch.env["PYTHONPATH"],
        "storage_environment": {
            name: launch.env[name]
            for name in (
                "ORBITKV_NVCOMP_LIBRARY",
                "LD_PRELOAD",
                "CUFILE_ENV_PATH_JSON",
                "CUFILE_ALLOW_COMPAT_MODE",
                "CUFILE_FORCE_COMPAT_MODE",
            )
            if name in launch.env
        },
        "compiler_cache": {
            name: launch.env[name]
            for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
            if name in launch.env
        },
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
        "capacity": {
            "engine_kv_bytes": args.gpu_tokens * bytes_per_token,
            "host_pool_bytes": args.host_gib * 1024**3,
            "ssd_bytes": args.ssd_gib * 1024**3,
            "codec_scratch_budget_bytes_per_worker": args.storage_codec_budget
            if args.backend == "orbitkv"
            else None,
            "working_set_tokens": working_set_tokens,
            "working_set_logical_bytes": working_set_tokens * bytes_per_token
            if working_set_tokens is not None
            else None,
            "notes": "Configured budgets, not measured effective capacity. GPU codec scratch and cuFile staging are additional to engine KV. Encoded-publication counters exclude raw-only slots and do not establish whole-cache savings.",
        },
        "quality_scope": "Synthetic token prompts; exact generated-text comparisons are diagnostic, not task accuracy or general model quality. Greedy decoding is not guaranteed batch invariant.",
        "ttft_scope": "Client time to first nonempty streamed text, including HTTP/scheduling; not engine time to first token ID.",
        "model_revision": (args.model / ".revision").read_text().strip()
        if (args.model / ".revision").exists()
        else None,
        "concurrency": args.concurrencies if args.workload != "serial" else 1,
        "notes": (
            "Sustained closed-loop traffic with at most C client requests in flight. Throughput includes admitted requests' drain time; cache counters cover the whole window and post-request drain. Prepared-prefix output comparisons are diagnostic, not batch-invariant correctness proofs. Memory peaks are sampled lower bounds."
            if args.workload == "sustained"
            else "Closed-loop bursts; tier counters belong to entire batches, not individual requests. Memory peaks are sampled every 25 ms."
            if args.workload == "concurrent"
            else "Serial latency experiment; pressure traffic and startup are excluded from request timings."
        ),
    }
    return manifest
