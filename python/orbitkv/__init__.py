"""OrbitKV - High-performance key-value storage engine with Python bindings.

This package provides:
1. EngineRpcClient: gRPC client for remote OrbitKV server communication
2. OrbitKVConnector: vLLM KV connector for distributed inference
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

_NATIVE_EXPORTS = {
    "EngineRpcClient",
    "LocalControlClient",
    "OrbitKVError",
    "OrbitKVInternal",
    "PyLoadState",
    "QueryLoading",
    "QueryReady",
}

try:
    from . import orbitkv as _native
except ImportError:
    _native = None

try:
    __version__ = _native.__version__ if _native is not None else version("orbitkv-llm")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(name)
    if _native is None:
        raise ImportError(
            "orbitkv rust extension is not available, check orbitkv-xxx.so file exists"
        ) from None
    return getattr(_native, name)


__all__ = [
    "__version__",
    "EngineRpcClient",
    "LocalControlClient",
    "OrbitKVError",
    "OrbitKVInternal",
    "PyLoadState",
    "QueryLoading",
    "QueryReady",
]
