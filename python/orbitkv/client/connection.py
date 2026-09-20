"""Connect framework adapters to their node-local Cache Manager."""

from __future__ import annotations

import os
import socket
import stat
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from orbitkv.client.manager import (
    CacheManagerClient,
)


def _int_option(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class CacheConnections:
    """Per-shard clients with one selected endpoint for this adapter."""

    clients: tuple[CacheManagerClient, ...]
    selected_index: int

    def close(self) -> None:
        for client in self.clients:
            client.close()


def connect_cache(
    *,
    endpoints: tuple[str, ...],
    shard_index: int,
    all_shards: bool,
    get_option: Callable[[str, object], object],
) -> CacheConnections:
    """Open cache clients for one or more local TP shards.

    `get_option` reads integration-specific configuration without importing a
    framework into this module. Each required shard needs a same-host socket.
    """
    if not endpoints or not 0 <= shard_index < len(endpoints):
        raise ValueError("cache shard index is outside the configured endpoints")
    bootstrap_socket = get_option("orbitkv.bootstrap_socket", None)
    shard_sockets = get_option("orbitkv.tp_shard_bootstrap_sockets", None)
    selection_endpoints = endpoints
    if not all_shards:
        selection_endpoints = (endpoints[shard_index],)
        if shard_sockets is not None:
            if not isinstance(shard_sockets, (list, tuple)) or len(shard_sockets) != len(endpoints):
                raise ValueError(
                    "orbitkv.tp_shard_bootstrap_sockets must provide one socket per TP shard"
                )
            shard_sockets = (shard_sockets[shard_index],)
        elif len(endpoints) > 1:
            bootstrap_socket = None
    sockets = resolve_bootstrap_sockets(
        endpoints=selection_endpoints,
        bootstrap_socket=bootstrap_socket,
        shard_bootstrap_sockets=shard_sockets,
    )
    selected_index = shard_index if all_shards else 0
    timeout_ms = _int_option(
        get_option("orbitkv.timeout_ms", 5_000),
        "orbitkv.timeout_ms",
        minimum=1,
    )
    spin_iterations = _int_option(
        get_option("orbitkv.spin_iterations", 64),
        "orbitkv.spin_iterations",
        minimum=0,
    )
    selected_sockets = sockets if all_shards else (sockets[shard_index if len(sockets) > 1 else 0],)
    opened = []
    try:
        for socket in selected_sockets:
            opened.append(
                CacheManagerClient(socket, timeout_ms=timeout_ms, spin_iterations=spin_iterations)
            )
    except Exception:
        for client in opened:
            client.close()
        raise
    clients = tuple(opened)
    return CacheConnections(clients=clients, selected_index=selected_index)


def resolve_bootstrap_sockets(
    *,
    endpoints: tuple[str, ...],
    bootstrap_socket: object = None,
    shard_bootstrap_sockets: object = None,
) -> tuple[str, ...]:
    """Resolve the process endpoint for same-host cache clients.

    Every inference shard must connect to a Cache Manager on its own host.
    """
    if not endpoints:
        raise ValueError("cache client requires at least one endpoint")
    if not all(_endpoint_is_local(endpoint) for endpoint in endpoints):
        raise ValueError(
            "inference clients require a node-local Cache Manager; "
            "configure a Cache Manager on each inference host"
        )

    if bootstrap_socket is not None and shard_bootstrap_sockets is not None:
        raise ValueError(
            "configure either orbitkv.bootstrap_socket or "
            "orbitkv.tp_shard_bootstrap_sockets, not both"
        )

    if shard_bootstrap_sockets is not None:
        if not isinstance(shard_bootstrap_sockets, (list, tuple)):
            raise ValueError("orbitkv.tp_shard_bootstrap_sockets must be a list of socket paths")
        sockets = tuple(shard_bootstrap_sockets)
    elif bootstrap_socket is not None:
        if len(endpoints) > 1:
            raise ValueError(
                "orbitkv.bootstrap_socket only supports one TP shard; "
                "use orbitkv.tp_shard_bootstrap_sockets"
            )
        sockets = (bootstrap_socket,)
    else:
        sockets = tuple(_default_bootstrap_socket(endpoint) for endpoint in endpoints)

    if len(sockets) != len(endpoints):
        raise ValueError(
            f"configured {len(sockets)} local bootstrap sockets for {len(endpoints)} TP shards"
        )
    if any(not isinstance(socket, str) or not socket for socket in sockets):
        raise ValueError("local bootstrap socket configuration must contain non-empty strings")
    if len(set(sockets)) != len(sockets):
        raise ValueError("local bootstrap sockets must be distinct for TP shards")
    missing = [socket_path for socket_path in sockets if not _is_unix_socket(socket_path)]
    if missing:
        raise ConnectionError(
            "OrbitKV Cache Manager Unix socket is unavailable: "
            + ", ".join(missing)
            + "; start the node-local Cache Manager"
        )
    return sockets


def _default_bootstrap_socket(endpoint: str) -> str:
    port = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}").port
    if port is None:
        raise ValueError(
            "cannot derive the process socket from the configured endpoint; "
            "set orbitkv.bootstrap_socket"
        )
    return f"/tmp/orbitkv-{port}.sock"


def _is_unix_socket(path: str) -> bool:
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False


def _endpoint_is_local(endpoint: str) -> bool:
    host = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}").hostname
    if host is None:
        return False
    if host in {"localhost", "::1"} or host.startswith("127."):
        return True
    try:
        addresses = socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, socket_type, protocol, _, address in addresses:
        try:
            with socket.socket(family, socket_type, protocol) as probe:
                probe.bind(address)
        except OSError:
            continue
        return True
    return False
