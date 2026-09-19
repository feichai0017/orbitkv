"""Process-boundary integration for the node-local iceoryx2 client."""

import importlib
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
    bootstrap_socket = local_control_server.local_bootstrap_socket
    assert bootstrap_socket is not None
    query_client = orbitkv_native.LocalQueryClient(bootstrap_socket)

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
    ok, message = local_control_client_context.engine_client.save(
        local_control_client_context.instance_id,
        0,
        0,
        0,
        [(local_control_client_context._layer_names[0], [0, 1], block_hashes)],
    )
    assert ok, message

    deadline = time.monotonic() + 5
    while True:
        result = query_client.query_bundle(
            instance_id=local_control_client_context.instance_id,
            block_hashes=block_hashes,
            req_id="registered-warm-query",
            request_id=203,
        )
        if isinstance(result, orbitkv_native.QueryReady) and result.num_hit_blocks == 2:
            break
        assert time.monotonic() < deadline, f"local query never became ready: {result!r}"
        time.sleep(0.05)

    assert result.lease
    query_client.release(result.lease, request_id=204)
    with pytest.raises(orbitkv_native.OrbitKVError, match="Invalid"):
        query_client.release(result.lease, request_id=205)
