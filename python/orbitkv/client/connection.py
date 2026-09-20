"""Connect framework adapters to their node-local Cache Manager."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from orbitkv.client.data_plane import (
    CacheDataClient,
    CacheLifecycleClient,
    CacheManagerClient,
    resolve_bootstrap_sockets,
)


def _int_option(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class CacheConnections:
    """Per-shard clients with one selected endpoint for this adapter."""

    lifecycle_clients: tuple[CacheLifecycleClient, ...]
    data_clients: tuple[CacheDataClient, ...]
    lifecycle: CacheLifecycleClient
    data: CacheDataClient
    target: str

    def close(self) -> None:
        for client in self.lifecycle_clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def connect_data_client(
    bootstrap_socket: str, *, timeout_ms: int = 5_000, spin_iterations: int = 64
) -> CacheManagerClient:
    """Open the node-local data path used by every inference adapter."""
    return CacheManagerClient(bootstrap_socket, timeout_ms=timeout_ms, spin_iterations=spin_iterations)


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
                connect_data_client(socket, timeout_ms=timeout_ms, spin_iterations=spin_iterations)
            )
    except Exception:
        for client in opened:
            client.close()
        raise
    clients = tuple(opened)
    return CacheConnections(
        lifecycle_clients=clients,
        data_clients=clients,
        lifecycle=clients[selected_index],
        data=clients[selected_index],
        target=selected_sockets[selected_index],
    )
