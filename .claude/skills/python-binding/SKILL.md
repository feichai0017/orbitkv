---
name: python-binding
description: >
  Use when modifying python/src/lib.rs (PyO3 bindings), python/orbitkv/vllm/ (vLLM KV connector),
  python/orbitkv/sglang/ (SGLang adapter), .pyi type stubs, or building with maturin.
---

# Python Development

## Build

```bash
cd python
maturin develop          # Dev build
maturin develop --release  # Release build
```

**Important:** When modifying `python/src/lib.rs` (PyO3 bindings), update the type stub file `python/orbitkv/orbitkv.pyi` to keep type hints in sync.

## Key Files

- `python/src/lib.rs`: PyO3 bindings exposing `OrbitKVEngine` and gRPC client
- `python/orbitkv/orbitkv.pyi`: Type stubs — must stay in sync with `lib.rs`
- `python/orbitkv/vllm/scheduler.py`: vLLM scheduler-side connector
- `python/orbitkv/vllm/worker.py`: vLLM worker-side connector
- `python/orbitkv/connector/__init__.py`: backward-compatible vLLM import alias
- `python/orbitkv/sglang/`: SGLang config and pool contracts; backend pending M1
- `python/orbitkv/client/`: framework-neutral client exports
- `python/orbitkv/ipc_wrapper.py`: CUDA IPC handle wrapper

## vLLM Integration

Configure vLLM to use OrbitKV:

```python
from vllm.distributed.kv_transfer.kv_transfer_agent import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OrbitKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="orbitkv.vllm",
)
```

Connector is split into scheduler-side (`scheduler.py`) and worker-side (`worker.py`).

## SGLang Integration

Only configuration and pool-name mapping are implemented in M0. Do not import
or document an `OrbitKVHiCacheStorage` runtime until M1 adds it. The planned
backend targets SGLang's dynamic `HiCacheStorage` interface and requires the
shared-memory host allocator so the sidecar can map pages without a second host
copy.
