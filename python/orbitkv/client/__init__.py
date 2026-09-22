"""Framework-neutral access to OrbitKV cache services."""

from __future__ import annotations

from orbitkv import (
    CacheManagerClient,
    OrbitKVError,
    OrbitKVInternal,
    QueryLoading,
    QueryReady,
    RestoreHandle,
    RestoreStatus,
)
from orbitkv.client.connection import (
    CacheConnections,
    connect_cache,
    resolve_bootstrap_sockets,
)

__all__ = [
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
]
