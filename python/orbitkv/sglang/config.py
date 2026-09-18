from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrbitKVSGLangConfig:
    """Configuration parsed by the future SGLang HiCache backend."""

    endpoint: str = "unix:///run/orbitkv/orbitkv.sock"
    allocator: str = "shm"
    namespace: str | None = None

    @classmethod
    def from_extra_config(cls, value: dict[str, Any] | None) -> OrbitKVSGLangConfig:
        value = value or {}
        endpoint = str(value.get("endpoint", cls.endpoint))
        allocator = str(value.get("allocator", cls.allocator))
        namespace = value.get("namespace")
        if allocator != "shm":
            raise ValueError(
                "OrbitKV SGLang integration requires allocator='shm' to avoid a second host copy"
            )
        if not endpoint.startswith(("unix://", "http://", "https://")):
            raise ValueError(f"unsupported OrbitKV endpoint: {endpoint!r}")
        return cls(
            endpoint=endpoint,
            allocator=allocator,
            namespace=str(namespace) if namespace is not None else None,
        )
