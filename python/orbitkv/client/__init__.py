"""Framework-neutral access to OrbitKV cache services."""

from __future__ import annotations

from orbitkv import (
    OrbitKVError,
    OrbitKVInternal,
    QueryLoading,
    QueryReady,
)
from orbitkv.client.connection import CacheConnections, connect_cache, connect_data_client
from orbitkv.client.data_plane import (
    CacheDataClient,
    CacheManagerClient,
    RestoreHandle,
    RestoreStatus,
    resolve_bootstrap_sockets,
)

__all__ = [
    "CacheDataClient",
    "CacheConnections",
    "CacheManagerClient",
    "OrbitKVError",
    "OrbitKVInternal",
    "QueryLoading",
    "QueryReady",
    "RestoreHandle",
    "RestoreStatus",
    "resolve_bootstrap_sockets",
    "connect_cache",
    "connect_data_client",
]
