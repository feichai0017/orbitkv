"""Contracts shared by the gRPC and local cache data planes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.client import (  # noqa: E402
    GrpcDataClient,
    LocalDataClient,
    resolve_local_bootstrap_sockets,
)


def test_local_socket_defaults_to_the_sidecars_grpc_port():
    assert resolve_local_bootstrap_sockets(
        enabled=True,
        endpoints=("http://127.0.0.1:50055",),
    ) == ("/tmp/orbitkv-50055.sock",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"enabled": "true", "endpoints": ("http://a:1",)},
            "must be a boolean",
        ),
        (
            {
                "enabled": False,
                "endpoints": ("http://a:1",),
                "bootstrap_socket": "/tmp/a.sock",
            },
            "require orbitkv.local_data=true",
        ),
        (
            {
                "enabled": True,
                "endpoints": ("http://a:1", "http://b:2"),
            },
            "multiple TP shards requires",
        ),
        (
            {
                "enabled": True,
                "endpoints": ("http://a:1", "http://b:2"),
                "shard_bootstrap_sockets": ["/tmp/a.sock"],
            },
            "1 local bootstrap sockets for 2 TP shards",
        ),
    ],
)
def test_local_socket_configuration_rejects_ambiguous_layouts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_local_bootstrap_sockets(**kwargs)


def test_local_data_client_translates_hot_operations(monkeypatch):
    native = MagicMock()
    native.session_epoch = 41
    native.notification_fd = 7
    native.query_bundle.return_value = object()
    native.restore_submit.return_value = 13
    native.restore_poll.side_effect = [("pending", ""), ("succeeded", "")]
    factory = MagicMock(return_value=native)
    monkeypatch.setattr("orbitkv.client.data_plane.LocalQueryClient", factory)

    client = LocalDataClient("/tmp/orbitkv.sock", timeout_ms=123, spin_iterations=8)
    query_result = client.query_prefetch(
        "instance", [b"hash"], "request", wait_for_full_prefix=True, group_id=2
    )
    client.release(b"lease")
    assert client.save("instance", 1, 2, 3, [("layer", [4], [b"hash"])]) == (True, "")
    restore = client.start_restore("instance", 1, 3, [["layer"]], [(b"lease", [[4]])])

    assert query_result is native.query_bundle.return_value
    assert restore.key == "local:41:13"
    assert not client.poll_restore(restore).done
    assert client.poll_restore(restore).success
    factory.assert_called_once_with("/tmp/orbitkv.sock", timeout_ms=123, spin_iterations=8)
    assert [item.kwargs["request_id"] for item in native.method_calls] == [1, 2, 3, 4, 5, 6]
    assert native.method_calls[:4] == [
        call.query_bundle(
            "instance",
            [b"hash"],
            "request",
            wait_for_full_prefix=True,
            group_id=2,
            request_id=1,
        ),
        call.release(b"lease", request_id=2),
        call.publish(
            "instance",
            1,
            2,
            3,
            [("layer", [4], [b"hash"])],
            request_id=3,
        ),
        call.restore_submit(
            "instance",
            1,
            3,
            [["layer"]],
            [(b"lease", [[4]])],
            request_id=4,
        ),
    ]


def test_grpc_data_client_preserves_load_state_completion(monkeypatch):
    state = MagicMock()
    state.shm_name.return_value = "load-state"
    state.is_ready.side_effect = [False, True]
    state.get_state.return_value = -7
    monkeypatch.setattr("orbitkv.client.data_plane.PyLoadState", lambda: state)
    native = MagicMock()
    native.load.return_value = (True, "")
    client = GrpcDataClient(native)

    restore = client.start_restore("instance", 0, 0, [["layer"]], [(b"lease", [[1]])])

    assert restore.key == "load-state"
    assert not client.poll_restore(restore).done
    status = client.poll_restore(restore)
    assert status.done and not status.success and status.message == "load state -7"
    native.load.assert_called_once_with(
        "instance", 0, 0, "load-state", [["layer"]], [(b"lease", [[1]])]
    )


def test_context_defaults_the_data_plane_to_grpc():
    from orbitkv.vllm.common import ConnectorContext

    engine = MagicMock()
    context = ConnectorContext(
        instance_id="instance",
        namespace="namespace",
        block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=engine,
        state_manager=SimpleNamespace(),
    )

    assert context.data_client is not None
    assert context.data_client.transport == "grpc"
