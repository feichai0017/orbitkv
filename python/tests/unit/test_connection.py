"""Contracts of the node-local Cache Manager connection."""

from __future__ import annotations

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.client import (  # noqa: E402
    resolve_bootstrap_sockets,
)


def test_local_socket_defaults_to_the_cache_manager_addr_port(monkeypatch):
    monkeypatch.setattr("orbitkv.client.connection._is_unix_socket", lambda _path: True)
    assert resolve_bootstrap_sockets(
        endpoints=("http://127.0.0.1:50055",),
    ) == ("/tmp/orbitkv-50055.sock",)


def test_local_socket_requires_same_host_socket(monkeypatch):
    monkeypatch.setattr(
        "orbitkv.client.connection._is_unix_socket",
        lambda path: path == "/tmp/orbitkv-50055.sock",
    )

    assert resolve_bootstrap_sockets(endpoints=("http://127.0.0.1:50055",)) == (
        "/tmp/orbitkv-50055.sock",
    )
    with pytest.raises(ConnectionError, match="/tmp/orbitkv-50056.sock"):
        resolve_bootstrap_sockets(endpoints=("http://127.0.0.1:50056",))


def test_local_socket_derives_every_tp_shard(monkeypatch):
    monkeypatch.setattr("orbitkv.client.connection._is_unix_socket", lambda _path: True)

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
