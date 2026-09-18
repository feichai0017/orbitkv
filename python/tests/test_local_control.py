"""Process-boundary integration for the node-local iceoryx2 client."""

import importlib

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_python_local_control_lifecycle_reaches_real_sidecar(local_control_server):
    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    service_name = local_control_server.local_control_service
    session_epoch = local_control_server.local_control_session_epoch
    assert service_name is not None
    assert session_epoch is not None

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
