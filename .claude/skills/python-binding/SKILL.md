---
name: python-binding
description: >
  Use when modifying python/src/lib.rs (PyO3 bindings), python/orbitkv/connector/ (vLLM KV connector),
  python/orbitkv/sglang/ (SGLang radix cache), .pyi type stubs, or building with maturin.
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
- `python/orbitkv/connector/scheduler.py`: vLLM scheduler-side connector
- `python/orbitkv/connector/worker.py`: vLLM worker-side connector
- `python/orbitkv/sglang/orbitkv_radix_cache.py`: SGLang radix cache
- `python/orbitkv/ipc_wrapper.py`: CUDA IPC handle wrapper

## vLLM Integration

Configure vLLM to use OrbitKV:

```python
from vllm.distributed.kv_transfer.kv_transfer_agent import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OrbitKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="orbitkv.connector",
)
```

Connector is split into scheduler-side (`scheduler.py`) and worker-side (`worker.py`).

## SGLang Integration

Drop-in replacement for SGLang's `RadixCache` using OrbitKVEngine for distributed KV cache.

```python
from orbitkv.sglang.peagflow_radix_cache import PeagflowRadixCache

kv_cache = PeagflowRadixCache(
    params=cache_params,       # CacheInitParams
    model_config=model_config, # ModelConfig
    tp_size=tp_size,
    rank=tp_rank,
)
```

Behavioral differences vs default RadixCache:
- On prefix miss, queries OrbitKVEngine for remote blocks and loads into local GPU buffers
- On request finish, saves changed blocks back to engine
- All KV operations batched per block and per layer
- Auto-registers CUDA IPC handles on construction, auto-unregisters on shutdown
