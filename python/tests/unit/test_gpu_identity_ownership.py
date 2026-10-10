import fcntl
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.support.gpu_identity import (
    drain_identity_processes,
    identity_gpu_locks,
    quarantine_identity_gpus,
    require_clean_identity_log,
)


def test_gpu_lock_survives_controller_exit_until_inheriting_owner_exits(tmp_path):
    with identity_gpu_locks(tmp_path, ["7"]) as descriptors:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            pass_fds=descriptors,
        )
        assert child.stdout.readline() == "ready\n"
    try:
        with (
            (tmp_path / "gpu-7.lock").open("a") as competing,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.communicate("release\n", timeout=5)
    assert child.returncode == 0
    with identity_gpu_locks(tmp_path, ["7"]):
        pass


def test_unknown_client_drain_preserves_owner_and_blocks_next_admission(tmp_path):
    release = tmp_path / "release"
    manager_stops = []
    with identity_gpu_locks(tmp_path, ["7"]) as descriptors:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            pass_fds=descriptors,
        )
        assert child.stdout.readline() == "ready\n"
        try:
            manager = SimpleNamespace(
                process=None, terminate_gracefully=lambda **_: manager_stops.append("SIGTERM")
            )
            cleanup = drain_identity_processes(child, manager, release, timeout=0.02)
            assert cleanup["remaining_processes"][0]["pid"] == child.pid
            assert child.poll() is None
            assert not manager_stops
            quarantine_identity_gpus(tmp_path, ["7"], tmp_path, cleanup)
        finally:
            child.communicate("release\n", timeout=5)
    with pytest.raises(RuntimeError, match="quarantined"), identity_gpu_locks(tmp_path, ["7"]):
        pytest.fail("A quarantined GPU was admitted")


@pytest.mark.parametrize("failure", ["ERROR physical drain failed", "thread panicked at drain"])
def test_zero_exit_does_not_hide_manager_lifecycle_failure(failure):
    with pytest.raises(AssertionError):
        require_clean_identity_log(f"INFO normal work\n{failure}\nINFO shutdown complete")
