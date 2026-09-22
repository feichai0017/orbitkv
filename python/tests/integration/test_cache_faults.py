"""Deterministic faults against the real Manager, CUDA mappings and SSD I/O.

Build orbitkv-server with test-hooks and set ORBITKV_FAULT_TESTS=1 explicitly.
The barriers are absent from normal release binaries.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import CacheManagerProcess, ClientContext, find_available_port
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("ORBITKV_FAULT_TESTS") != "1", reason="requires test-hooks Manager"
    ),
]


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not (result := predicate()):
        assert time.monotonic() < deadline, "fault gate timed out"
        time.sleep(0.01)
    return result


@pytest.fixture
def fault_cache(tmp_path, monkeypatch, request):
    from orbitkv import CacheManagerClient

    monkeypatch.setenv("ORBITKV_TEST_FAULTS", str(tmp_path))
    server = CacheManagerProcess(
        find_available_port(),
        http_port=find_available_port(),
        bootstrap_socket=str(tmp_path / "cache.sock"),
        ssd_cache_path=tmp_path / "ssd",
        channel_service=f"orbitkv/fault/{tmp_path.name}"
        if getattr(request, "param", None)
        else None,
    )
    assert server.start(), server.read_logs()
    client = CacheManagerClient(server.bootstrap_socket, timeout_ms=100)
    ctx = ClientContext(
        client=client,
        instance_id="fault-instance",
        namespace="fault-model",
        device_id=0,
        num_blocks=4,
        num_layers=1,
    )
    ctx.register_kv_caches()
    try:
        yield server, client, ctx, tmp_path
    finally:
        for pause in tmp_path.glob("*.pause"):
            pause.unlink(missing_ok=True)
        client.close()
        server.stop()


def arm(directory: Path, name: str):
    (directory / f"{name}.reached").unlink(missing_ok=True)
    (directory / f"{name}.pause").touch()


def reached(directory, name):
    until(lambda: (directory / f"{name}.reached").exists())


def publish(client, ctx, hashes):
    return client.save(
        ctx.instance_id, 0, 0, 0, [(ctx._layer_names[0], list(range(len(hashes))), hashes)]
    )


def query(client, ctx, hashes, rid):
    from orbitkv import BlockHashes, QueryReady

    batch = BlockHashes(hashes)

    def poll():
        result = client.query_prefetch(ctx.instance_id, batch, rid)
        return result if isinstance(result, QueryReady) else None

    return until(poll)


def drain_ssd(server):
    def drained():
        stats = fetch_orbitkv_metrics(server.http_port)
        return stats.get("orbitkv_ssd_write_bytes_total", 0) > 0 and not any(
            stats.get(name, 0)
            for name in (
                "orbitkv_ssd_write_inflight",
                "orbitkv_ssd_write_queue_pending",
                "orbitkv_inflight_bytes",
            )
        )

    until(drained)
    response = requests.post(f"http://127.0.0.1:{server.http_port}/cache/memory/cleanup", timeout=5)
    response.raise_for_status()


def test_ssd_cancellation_revisions_hold_buffers_until_io_drains(fault_cache):
    from orbitkv import BlockHashes, QueryLoading

    server, client, ctx, directory = fault_cache
    hashes = [bytes([i]) * 32 for i in (1, 2)]
    publish(client, ctx, hashes)
    drain_ssd(server)
    arm(directory, "ssd")
    assert isinstance(
        client.query_prefetch(ctx.instance_id, BlockHashes(hashes), "slow"), QueryLoading
    )
    reached(directory, "ssd")
    reserved = fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"]
    assert reserved > 0
    client.cancel_query(ctx.instance_id, "slow")
    # A late result cannot reappear under a reused request name/revision.
    assert query(client, ctx, [b"different"], "slow").num_hit_blocks == 0
    assert client.health()[0]
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == reserved
    # Start and cancel a second interest while the same submitted read is live.
    assert isinstance(
        client.query_prefetch(ctx.instance_id, BlockHashes(hashes), "second"), QueryLoading
    )
    client.cancel_query(ctx.instance_id, "second")
    (directory / "ssd.pause").unlink()
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )
    ready = query(client, ctx, hashes, "fresh")
    assert ready.num_hit_blocks == 2
    client.release(ready.lease)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )


def test_restore_timeout_and_lost_notification_preserve_destinations(fault_cache):
    import torch

    server, client, ctx, directory = fault_cache
    hashes = [b"restore" * 4]
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    publish(client, ctx, hashes)
    ready = query(client, ctx, hashes, "load")
    arm(directory, "restore")
    arm(directory, "notification")
    handle = client.start_restore(ctx.instance_id, 0, 0, [ctx._layer_names], [(ready.lease, [[2]])])
    reached(directory, "restore")
    with pytest.raises(TimeoutError):
        client.wait_restore(handle, timeout=0.02)
    assert not client.poll_restore(handle).done
    assert query(client, ctx, [b"unrelated"], "independent").num_hit_blocks == 0
    (directory / "restore.pause").unlink()
    assert client.wait_restore(handle, timeout=5).success
    reached(directory, "notification")
    torch.cuda.synchronize()
    assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )


@pytest.mark.parametrize("end", ["complete", "kill", "corrupt-ack"])
def test_publish_stall_keeps_source_owned_without_stalling_queries(fault_cache, end, capfd):
    server, client, ctx, directory = fault_cache
    name = "publish_ack" if end == "corrupt-ack" else "publish"
    arm(directory, name)
    # Always terminate/release the peer before joining a potentially fenced publisher.
    pool = ThreadPoolExecutor(1)
    future = pool.submit(publish, client, ctx, [b"publish"])
    try:
        reached(directory, name)
        time.sleep(0.15)  # Past the client's 100ms ordinary call deadline.
        assert not future.done(), "publisher released GPU sources on an ambiguous timeout/ack"
        assert query(client, ctx, [b"unrelated"], "independent").num_hit_blocks == 0
        expected_log = "holding source pages" if end == "corrupt-ack" else "retaining source pages"
        assert expected_log in capfd.readouterr().err
        if end == "complete":
            (directory / "publish.pause").unlink()
            assert future.result(timeout=5)[0]
        else:
            server.stop()
            with pytest.raises(Exception, match="exited|reconnect"):
                future.result(timeout=5)
    finally:
        (directory / f"{name}.pause").unlink(missing_ok=True)
        if not future.done():
            server.stop()
        pool.shutdown(wait=True)


@pytest.mark.parametrize("fault_cache", [None, "configured-prefix"], indirect=True)
def test_manager_restart_rejects_old_leases_and_restore_handles(fault_cache):
    from orbitkv import CacheManagerClient, OrbitKVError

    server, client, ctx, directory = fault_cache
    publish(client, ctx, [b"restart"])
    ready = query(client, ctx, [b"restart"], "old-lease")
    another = query(client, ctx, [b"restart"], "old-restore")
    arm(directory, "restore")
    handle = client.start_restore(
        ctx.instance_id, 0, 0, [ctx._layer_names], [(another.lease, [[2]])]
    )
    reached(directory, "restore")
    server.stop()
    (directory / "restore.pause").unlink()
    assert server.start(), server.read_logs()
    fresh = CacheManagerClient(server.bootstrap_socket)
    try:
        with pytest.raises(OrbitKVError):
            client.poll_restore(handle)
        with pytest.raises(OrbitKVError, match="reconnect"):
            fresh.poll_restore(handle)
        with pytest.raises(OrbitKVError):
            fresh.release(ready.lease)
        replacement = ClientContext(
            client=fresh,
            instance_id=ctx.instance_id,
            namespace=ctx.namespace,
            device_id=0,
            num_blocks=4,
            num_layers=1,
        )
        replacement.register_kv_caches()
        publish(fresh, replacement, [b"new-engine"])
        new = query(fresh, replacement, [b"new-engine"], "fresh")
        assert new.num_hit_blocks == 1
        fresh.release(new.lease)
        replacement.unregister_context()
    finally:
        fresh.close()
