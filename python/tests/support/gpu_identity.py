"""Keep dedicated GPU leases attached to identity-gate process lifetimes."""

import fcntl
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def identity_gpu_locks(root: Path, devices: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    locks = []
    try:
        for device in sorted(devices):
            lock = (root / f"gpu-{device}.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            quarantine = root / f"gpu-{device}.quarantine.json"
            if quarantine.exists():
                raise RuntimeError(f"GPU {device} is quarantined: {quarantine}")
        yield tuple(lock.fileno() for lock in locks)
    finally:
        for lock in locks:
            lock.close()


def owned_process(process):
    if process is None:
        return None
    result = {"pid": process.pid, "exit_code": process.poll()}
    if result["exit_code"] is None:
        root = Path(f"/proc/{process.pid}")
        try:
            result["start_ticks"] = int((root / "stat").read_text().rsplit(")", 1)[1].split()[19])
            result["command"] = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except FileNotFoundError:
            result["exit_code"] = process.poll()
    return result


def drain_identity_processes(client, server, release: Path, timeout: float = 60):
    """Release the tensor owner before requesting normal Manager drain.

    A timeout leaves the live process and its inherited flock descriptors intact.
    The caller persists quarantine before closing its own descriptors.
    """
    errors = []
    try:
        release.touch()
    except OSError as error:
        errors.append(f"Client release: {error}")
    if client is not None:
        try:
            code = client.wait(timeout=timeout)
            if code != 0:
                errors.append(f"Client normal exit code {code}")
        except subprocess.TimeoutExpired:
            errors.append("Client normal unregister/exit timed out")
    if (client is None or client.poll() is not None) and server.process is not None:
        try:
            code, _ = server.terminate_gracefully(timeout=timeout)
            if code != 0:
                errors.append(f"Manager normal exit code {code}")
        except subprocess.TimeoutExpired:
            errors.append("Manager normal drain timed out")
    processes = [owned_process(process) for process in (client, server.process)]
    remaining = [process for process in processes if process and process["exit_code"] is None]
    return {"errors": errors, "processes": processes, "remaining_processes": remaining}


def require_clean_identity_log(log: str):
    failures = [
        line
        for line in log.splitlines()
        if "ERROR" in line
        or "panicked at" in line
        or "Producer process has been terminated" in line
    ]
    assert not failures, failures


def quarantine_identity_gpus(root: Path, devices: list[str], directory: Path, cleanup: dict):
    if not cleanup["remaining_processes"]:
        return
    for device in devices:
        path = root / f"gpu-{device}.quarantine.json"
        with path.open("x") as stream:
            json.dump({"evidence": str(directory), **cleanup}, stream, indent=2)
            stream.write("\n")
