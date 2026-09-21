"""Real etcd + two Manager processes: embedded discovery and Mooncake GPU recovery.

Gate for distributed startup, placement, catalog protocol and packaging changes.
Requires ETCD_BIN, built native artifacts, CUDA and MC_FORCE_TCP=1 on one host.
"""

import hashlib
import os
import subprocess
import time
import uuid
from contextlib import ExitStack

import pytest
import requests

from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_embedded_catalog_transfers_between_managers_and_preserves_local_hits(
    tmp_path, monkeypatch
):
    binary = os.environ.get("ETCD_BIN")
    if not binary:
        pytest.skip("set ETCD_BIN to run the distributed process gate")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from orbitkv import QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
    from orbitkv.client.manager import CacheManagerClient

    monkeypatch.setenv("MC_FORCE_TCP", "1")
    endpoint = f"http://127.0.0.1:{find_available_port()}"
    peer = f"http://127.0.0.1:{find_available_port()}"
    with ExitStack() as stack:
        log = stack.enter_context((tmp_path / "etcd.log").open("wb"))
        etcd = subprocess.Popen(
            [
                binary,
                "--name",
                "test",
                "--data-dir",
                str(tmp_path / "etcd-data"),
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

        def stop_etcd():
            if etcd.poll() is None:
                etcd.terminate()
                try:
                    etcd.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    etcd.kill()
                    etcd.wait(timeout=5)

        stack.callback(stop_etcd)
        deadline = time.monotonic() + 15
        while True:
            assert etcd.poll() is None, (tmp_path / "etcd.log").read_text()
            try:
                if requests.get(f"{endpoint}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, "etcd startup timed out"
            time.sleep(0.05)

        managers = []
        clients = []
        tensors = []
        namespace = f"cluster:{uuid.uuid4().hex}"
        pages, block_bytes = 32, 256
        device = resolve_device_id()
        hashes = [hashlib.sha256(f"page-{i}".encode()).digest() for i in range(pages)]
        expected = torch.arange(pages * block_bytes, device="cuda").remainder(251).to(torch.uint8)
        for node in ("source", "consumer"):
            manager = CacheManagerProcess(
                find_available_port(),
                pool_size="64mb",
                http_port=find_available_port(),
                bootstrap_socket=str(tmp_path / f"{node}.sock"),
                extra_args=(
                    "--etcd-endpoints",
                    endpoint,
                    "--node-id",
                    node,
                    "--catalog-nodes",
                    "source,consumer",
                    "--membership-ttl-secs",
                    "12",
                ),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)
            client = CacheManagerClient(manager.bootstrap_socket)
            stack.callback(client.close)
            clients.append(client)
            tensor = expected.clone() if node == "source" else torch.full_like(expected, 253)
            tensors.append(tensor)
            torch.cuda.synchronize()
            client.start_session_watcher(node, namespace, 1, 1)
            ok, message = client.register_context_batch(
                node,
                namespace,
                0,
                0,
                1,
                1,
                device,
                ["kv:0"],
                [serialize_gpu_buffer(tensor)],
                [pages],
                [block_bytes],
                [0],
                [1],
                "direct",
                False,
            )
            assert ok, message

        ok, message = clients[0].save(
            "source", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
        )
        assert ok, message

        def restore(request):
            deadline = time.monotonic() + 30
            while True:
                result = clients[1].query_prefetch("consumer", hashes, request)
                if isinstance(result, QueryReady) and result.num_hit_blocks == pages:
                    break
                if isinstance(result, QueryReady) and result.lease:
                    clients[1].release(result.lease)
                assert time.monotonic() < deadline, [manager.read_logs() for manager in managers]
                time.sleep(0.02)
            operation = clients[1].start_restore(
                "consumer",
                0,
                device,
                [["kv:0"]],
                [(result.lease, [list(range(pages))])],
            )
            status = clients[1].wait_restore(operation, timeout=15)
            assert status.success, status
            torch.cuda.synchronize()
            assert torch.equal(tensors[1], expected)

        restore("remote")
        metrics = fetch_orbitkv_metrics(managers[1].http_port)
        assert metrics["orbitkv_remote_fetch_bytes_total"] >= pages * block_bytes
        stop_etcd()
        time.sleep(7)  # Conservative admission expires within half of the 12-second TTL.
        tensors[1].fill_(253)
        torch.cuda.synchronize()
        restore("after-coordinator-loss")
        assert (
            fetch_orbitkv_metrics(managers[1].http_port)["orbitkv_remote_fetch_bytes_total"]
            == metrics["orbitkv_remote_fetch_bytes_total"]
        )
