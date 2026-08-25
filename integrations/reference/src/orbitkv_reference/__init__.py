"""Reference tensor-arena adapter for OrbitKV."""

from .adapter import (
    AdapterError,
    AdapterPoisonedError,
    AdapterPreflightError,
    ArenaBinding,
    ReferencePagedAdapter,
)

__all__ = [
    "AdapterError",
    "AdapterPoisonedError",
    "AdapterPreflightError",
    "ArenaBinding",
    "ReferencePagedAdapter",
]
