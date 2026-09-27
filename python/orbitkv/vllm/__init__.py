"""vLLM adapter; import the connector only when vLLM requests it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orbitkv.vllm.connector import (
        KVConnectorRole,
        OrbitKVConnector,
    )

__all__ = ["OrbitKVConnector", "KVConnectorRole"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from orbitkv.vllm import connector

    return getattr(connector, name)
