"""An isolated etcd process shared by distributed runtime gates."""

import os
import subprocess
import time
from contextlib import contextmanager

import pytest
import requests

from .cache_manager import find_available_port


@contextmanager
def etcd_server(directory):
    binary = os.environ.get("ETCD_BIN")
    if not binary:
        pytest.skip("set ETCD_BIN to run a distributed runtime gate")
    endpoint = f"http://127.0.0.1:{find_available_port()}"
    peer = f"http://127.0.0.1:{find_available_port()}"
    log_path = directory / "etcd.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                binary,
                "--name",
                "test",
                "--data-dir",
                str(directory / "etcd-data"),
                "--listen-client-urls",
                endpoint,
                "--advertise-client-urls",
                endpoint,
                "--listen-peer-urls",
                peer,
                "--initial-advertise-peer-urls",
                peer,
                "--initial-cluster",
                f"test={peer}",
                "--log-level",
                "warn",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    def stop():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    try:
        deadline = time.monotonic() + 15
        while True:
            assert process.poll() is None, log_path.read_text()
            try:
                if requests.get(f"{endpoint}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, "etcd startup timed out"
            time.sleep(0.05)
        yield endpoint, stop
    finally:
        stop()
