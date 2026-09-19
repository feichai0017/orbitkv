"""Process-boundary integration for the node-local iceoryx2 client."""

import importlib
import os
import select
import sys
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_python_local_control_lifecycle_reaches_real_sidecar(local_control_server):
    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    service_name = local_control_server.local_control_service
    session_epoch = local_control_server.local_control_session_epoch
    assert service_name is not None
    assert session_epoch is not None
    bootstrap_socket = local_control_server.local_bootstrap_socket
    assert bootstrap_socket is not None

    client = orbitkv_native.LocalControlClient(service_name, session_epoch)
    assert client.service_name == service_name
    assert client.session_epoch == session_epoch
    assert client.ping(value=41, request_id=101) == 42

    stale_client = orbitkv_native.LocalControlClient(service_name, session_epoch - 1)
    with pytest.raises(orbitkv_native.OrbitKVError, match="StaleSession"):
        stale_client.ping(request_id=102)

    client.shutdown(request_id=103)
    assert local_control_server.process is not None
    assert local_control_server.process.wait(timeout=5) == 0, local_control_server.read_logs()


def test_query_bundle_uses_bootstrapped_arena_and_core(
    local_control_server, local_control_client_context
):
    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    torch = pytest.importorskip("torch")
    bootstrap_socket = local_control_server.local_bootstrap_socket
    assert bootstrap_socket is not None
    query_client = orbitkv_native.LocalQueryClient(bootstrap_socket)
    assert query_client.notification_fd >= 0

    with pytest.raises(orbitkv_native.OrbitKVError, match="Invalid"):
        query_client.query_bundle(
            instance_id="missing-instance",
            block_hashes=[],
            req_id="missing-query",
            request_id=201,
        )

    result = query_client.query_bundle(
        instance_id=local_control_client_context.instance_id,
        block_hashes=[],
        req_id="registered-cold-query",
        request_id=202,
    )
    assert isinstance(result, orbitkv_native.QueryReady)
    assert result.num_hit_blocks == 0
    assert result.lease == b""

    block_hashes = [bytes([1]) * 32, bytes([2]) * 32]
    expected = local_control_client_context.get_kv_cache()[:, 0:2].cpu().clone()
    query_client.publish(
        local_control_client_context.instance_id,
        0,
        0,
        0,
        [(local_control_client_context._layer_names[0], [0, 1], block_hashes)],
        request_id=203,
    )
    local_control_client_context.get_kv_cache().zero_()
    torch.cuda.synchronize()

    deadline = time.monotonic() + 5
    while True:
        result = query_client.query_bundle(
            instance_id=local_control_client_context.instance_id,
            block_hashes=block_hashes,
            req_id="registered-warm-query",
            request_id=204,
        )
        if isinstance(result, orbitkv_native.QueryReady) and result.num_hit_blocks == 2:
            break
        assert time.monotonic() < deadline, f"local query never became ready: {result!r}"
        time.sleep(0.05)

    assert result.lease
    operation_id = query_client.restore_submit(
        instance_id=local_control_client_context.instance_id,
        tp_rank=0,
        device_id=0,
        layer_groups=[local_control_client_context._layer_names],
        loads=[(result.lease, [[2, 3]])],
        request_id=205,
    )
    readable, _, _ = select.select([query_client.notification_fd], [], [], 5)
    assert readable == [query_client.notification_fd]
    assert int.from_bytes(os.read(query_client.notification_fd, 8), byteorder=sys.byteorder) >= 1
    state, message = query_client.restore_poll(operation_id, request_id=206)
    assert state == "succeeded", message
    restored = local_control_client_context.get_kv_cache()[:, 2:4].cpu()
    assert restored.equal(expected)
    with pytest.raises(orbitkv_native.OrbitKVError, match="Internal"):
        query_client.restore(
            instance_id=local_control_client_context.instance_id,
            tp_rank=0,
            device_id=0,
            layer_groups=[local_control_client_context._layer_names],
            loads=[(result.lease, [[0, 1]])],
            request_id=207,
        )

    second = query_client.query_bundle(
        instance_id=local_control_client_context.instance_id,
        block_hashes=block_hashes,
        req_id="registered-sync-restore-query",
        request_id=208,
    )
    assert isinstance(second, orbitkv_native.QueryReady)
    local_control_client_context.get_kv_cache()[:, 0:2].zero_()
    torch.cuda.synchronize()
    query_client.restore(
        instance_id=local_control_client_context.instance_id,
        tp_rank=0,
        device_id=0,
        layer_groups=[local_control_client_context._layer_names],
        loads=[(second.lease, [[0, 1]])],
        request_id=209,
    )
    assert local_control_client_context.get_kv_cache()[:, 0:2].cpu().equal(expected)

    third = query_client.query_bundle(
        instance_id=local_control_client_context.instance_id,
        block_hashes=block_hashes,
        req_id="registered-release-query",
        request_id=300,
    )
    assert isinstance(third, orbitkv_native.QueryReady)
    query_client.release(third.lease, request_id=301)
    with pytest.raises(orbitkv_native.OrbitKVError, match="Invalid"):
        query_client.release(third.lease, request_id=302)


def test_local_data_facade_runs_publish_query_restore_release(
    local_control_server, local_control_client_context
):
    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    LocalDataClient = importlib.import_module("orbitkv.client.data_plane").LocalDataClient
    torch = pytest.importorskip("torch")
    bootstrap_socket = local_control_server.local_bootstrap_socket
    assert bootstrap_socket is not None
    client = LocalDataClient(bootstrap_socket)

    block_hash = bytes([9]) * 32
    expected = local_control_client_context.get_kv_cache()[:, 0:1].cpu().clone()
    ok, message = client.save(
        local_control_client_context.instance_id,
        0,
        0,
        0,
        [(local_control_client_context._layer_names[0], [0], [block_hash])],
    )
    assert ok, message
    local_control_client_context.get_kv_cache()[:, 0:1].zero_()
    torch.cuda.synchronize()

    deadline = time.monotonic() + 5
    while True:
        result = client.query_prefetch(
            local_control_client_context.instance_id, [block_hash], "facade-query"
        )
        if isinstance(result, orbitkv_native.QueryReady) and result.num_hit_blocks == 1:
            break
        assert time.monotonic() < deadline, f"local facade query never became ready: {result!r}"
        time.sleep(0.05)

    deadline = time.monotonic() + 5
    restore = client.start_restore(
        local_control_client_context.instance_id,
        0,
        0,
        [local_control_client_context._layer_names],
        [(result.lease, [[0]])],
    )
    while True:
        if client.restore_completions_ready():
            status = client.poll_restore(restore)
            if status.done:
                break
        assert time.monotonic() < deadline, "local facade restore did not complete"
        time.sleep(0.01)

    assert status.success, status.message
    assert local_control_client_context.get_kv_cache()[:, 0:1].cpu().equal(expected)

    release_result = client.query_prefetch(
        local_control_client_context.instance_id, [block_hash], "facade-release"
    )
    assert isinstance(release_result, orbitkv_native.QueryReady)
    client.release(release_result.lease)
