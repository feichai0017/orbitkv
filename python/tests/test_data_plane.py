"""Contracts of the node-local Cache Manager connection."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.client import (  # noqa: E402
    CacheManagerClient,
    resolve_bootstrap_sockets,
)


def test_local_socket_defaults_to_the_cache_manager_addr_port(monkeypatch):
    monkeypatch.setattr("orbitkv.client.data_plane._is_unix_socket", lambda _path: True)
    assert resolve_bootstrap_sockets(
        endpoints=("http://127.0.0.1:50055",),
    ) == ("/tmp/orbitkv-50055.sock",)


def test_local_socket_requires_same_host_socket(monkeypatch):
    monkeypatch.setattr(
        "orbitkv.client.data_plane._is_unix_socket",
        lambda path: path == "/tmp/orbitkv-50055.sock",
    )

    assert resolve_bootstrap_sockets(endpoints=("http://127.0.0.1:50055",)) == (
        "/tmp/orbitkv-50055.sock",
    )
    with pytest.raises(ConnectionError, match="/tmp/orbitkv-50056.sock"):
        resolve_bootstrap_sockets(endpoints=("http://127.0.0.1:50056",))


def test_local_socket_derives_every_tp_shard(monkeypatch):
    monkeypatch.setattr("orbitkv.client.data_plane._is_unix_socket", lambda _path: True)

    assert resolve_bootstrap_sockets(
        endpoints=("http://127.0.0.1:50055", "http://127.0.0.1:50056"),
    ) == ("/tmp/orbitkv-50055.sock", "/tmp/orbitkv-50056.sock")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "endpoints": ("http://127.0.0.1:1", "http://127.0.0.1:2"),
                "shard_bootstrap_sockets": ["/tmp/a.sock"],
            },
            "1 local bootstrap sockets for 2 TP shards",
        ),
        (
            {"endpoints": ("http://remote.invalid:50055",)},
            "node-local Cache Manager",
        ),
    ],
)
def test_local_socket_configuration_rejects_ambiguous_layouts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_bootstrap_sockets(**kwargs)


def test_cache_manager_client_translates_hot_operations(monkeypatch):
    native = MagicMock()
    publisher = MagicMock()
    native.session_epoch = 41
    native.notification_fd = 7
    native.query_bundle.return_value = object()
    native.restore_submit.return_value = 13
    native.restore_poll.side_effect = [("pending", ""), ("succeeded", "")]
    factory = MagicMock(side_effect=[native, publisher])
    monkeypatch.setattr("orbitkv.client.data_plane.ChannelClient", factory)

    client = CacheManagerClient("/tmp/orbitkv.sock", timeout_ms=123, spin_iterations=8)
    query_result = client.query_prefetch(
        "instance", [b"hash"], "request", wait_for_full_prefix=True, group_id=2
    )
    client.release(b"lease")
    assert client.save("instance", 1, 2, 3, [("layer", [4], [b"hash"])]) == (True, "")
    restore = client.start_restore("instance", 1, 3, [["layer"]], [(b"lease", [[4]])])

    assert query_result is native.query_bundle.return_value
    assert restore.key == "manager:41:13"
    assert not client.poll_restore(restore).done
    assert client.poll_restore(restore).success
    assert factory.call_args_list == [
        call("/tmp/orbitkv.sock", timeout_ms=123, spin_iterations=8),
        call("/tmp/orbitkv.sock", timeout_ms=123, spin_iterations=8),
    ]
    assert [item.kwargs["request_id"] for item in native.method_calls] == [1, 2, 4, 5, 6]
    assert native.method_calls[:3] == [
        call.query_bundle(
            "instance",
            [b"hash"],
            "request",
            wait_for_full_prefix=True,
            group_id=2,
            request_id=1,
        ),
        call.release(b"lease", request_id=2),
        call.restore_submit(
            "instance",
            1,
            3,
            [["layer"]],
            [(b"lease", [[4]])],
            request_id=4,
        ),
    ]
    publisher.publish.assert_called_once_with(
        "instance", 1, 2, 3, [("layer", [4], [b"hash"])], request_id=3
    )
    client.close()
    native.close.assert_called_once_with()
    publisher.close.assert_called_once_with()


def test_blocked_publish_does_not_serialize_queries(monkeypatch):
    save_started = threading.Event()
    finish_save = threading.Event()
    query_finished = threading.Event()

    class Session:
        def __init__(self):
            self.descriptor_lock = threading.Lock()

        def publish(self, *_args, **_kwargs):
            with self.descriptor_lock:
                save_started.set()
                assert finish_save.wait(timeout=2)

        def query_bundle(self, *_args, **_kwargs):
            with self.descriptor_lock:
                query_finished.set()

        def close(self):
            pass

    primary, publisher = Session(), Session()
    factory = MagicMock(side_effect=[primary, publisher])
    monkeypatch.setattr("orbitkv.client.data_plane.ChannelClient", factory)
    client = CacheManagerClient("/tmp/orbitkv.sock")

    save_thread = threading.Thread(target=client.save, args=("instance", 0, 0, 0, []))
    query_thread = threading.Thread(
        target=client.query_prefetch, args=("instance", [b"hash"], "request")
    )
    try:
        save_thread.start()
        assert save_started.wait(timeout=1)
        query_thread.start()
        assert query_finished.wait(timeout=1), "query blocked behind an in-flight Publish"
    finally:
        finish_save.set()
        save_thread.join(timeout=2)
        query_thread.join(timeout=2)
        client.close()
    assert not save_thread.is_alive()
    assert not query_thread.is_alive()


def test_context_uses_one_client_for_direct_construction():
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

    assert context.data_client is engine
