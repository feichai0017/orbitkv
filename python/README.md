# OrbitKV Python Package

Framework adapters and Python bindings for the OrbitKV state cache, built with
Rust and PyO3. The vLLM adapter is validated today; the SGLang runtime backend
is tracked as the next milestone.

## Features

- **LocalDataClient**: UDS/iceoryx2 client for the node-local Cache Manager
- **CacheDataClient**: Framework-neutral cache operations for every storage tier
- **OrbitKVConnector**: vLLM KV connector for distributed inference with KV cache transfer
- **SGLang contracts**: Configuration and state-pool mapping without claiming a completed backend

## Installation

The framework release baseline verified on 2026-09-20 is vLLM `0.29.0` and
SGLang `0.5.20`. Keep their GPU dependencies in separate environments. The
SGLang source submodule is pinned to its `v0.5.20` release; its executable
OrbitKV HiCache backend is still in development.

The Cache Manager currently imports PyTorch at startup for CUDA IPC handling.
Run the wheel in an environment with a compatible PyTorch/CUDA runtime; the
base `orbitkv-llm` dependency set does not install PyTorch for you.

```bash
cd python
uv venv ../.venv/vllm-release --python 3.11
uv pip install --python ../.venv/vllm-release/bin/python 'vllm==0.29.0' pytest requests --torch-backend=cu130
uv venv ../.venv/sglang-release --python 3.11
uv pip install --python ../.venv/sglang-release/bin/python 'sglang==0.5.20' --torch-backend=cu130
```

### From Source

```bash
# Install maturin if you haven't already
pip install maturin

# Build and install in development mode
cd python
maturin develop

# Or build a wheel
maturin build --release
```

### From PyPI (coming soon)

```bash
pip install orbitkv
```

## Usage

### Cache Manager client

```python
from orbitkv.client import LocalDataClient

client = LocalDataClient("/tmp/orbitkv-50055.sock")
ok, message = client.health()
client.close()
```

### vLLM KV Connector

```python
from vllm import LLM
from vllm.distributed.kv_transfer.kv_transfer_agent import KVTransferConfig

# Configure vLLM to use OrbitKVConnector
kv_transfer_config = KVTransferConfig(
    kv_connector="OrbitKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="orbitkv.vllm",
)

# Create LLM with KV transfer enabled
llm = LLM(
    model="gpt2",
    kv_transfer_config=kv_transfer_config,
)
```

#### Local data plane

The connector derives the Cache Manager's Unix socket from the configured
endpoint. Scheduler Query/Release and worker Publish/Restore use iceoryx2;
registration, health, session ownership, and cleanup use the bootstrap UDS.
Each inference process requires a Cache Manager on its own host. A missing
socket fails at startup. No extra configuration is needed on one node.

The local path connects to `/tmp/orbitkv-<orbitkv.port>.sock`, matching the
Cache Manager default. Use `orbitkv.local_bootstrap_socket` for a custom single-manager
path. `orbitkv.local_timeout_ms` (default 5000) bounds hot requests and health;
registration and unregister allow at least 120 seconds for CUDA setup/draining.
`orbitkv.local_spin_iterations` defaults to 64. Standalone Cache Managers do
not start gRPC. Client and Cache Manager must use matching
bootstrap protocol versions (currently version 2).

`orbitkv.wait_for_full_prefix` is supported on the local path: pending queries
return `QueryLoading`, and repeated queries with the same instance/request/group
identity retrieve the result. Query arguments must remain unchanged while
pending. Each session permits 128 pending queries with a 60-second lifetime.
Undelivered results release their leases when discarded. Connector shutdown
explicitly closes the UDS session; imported CUDA mappings are released after
queued GPU transfers finish.

#### Connector Modes

`OrbitKVConnector` defaults to `read_write`: it queries OrbitKV for reusable KV
blocks, loads matched blocks into vLLM, and saves newly computed full blocks
back to OrbitKV.

Set `orbitkv.mode` to `save_only` when another vLLM connector is responsible
for reads and OrbitKV should only persist KV blocks for later reuse. This is
intended for `MultiConnector` decode-side setups where an upstream connector
owns the external hit/load path, while OrbitKV records the resulting KV cache.
In `save_only` mode, OrbitKV does not query or load KV blocks.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "connectors": [
        {
          "kv_connector": "<external-read-connector>",
          "kv_role": "kv_both"
        },
        {
          "kv_connector": "OrbitKVConnector",
          "kv_role": "kv_both",
          "kv_connector_module_path": "orbitkv.vllm",
          "kv_connector_extra_config": {
            "orbitkv.mode": "save_only"
          }
        }
      ]
    }
  }'
```

Valid values are `read_write` and `save_only`.

#### TP Shards Across Hosts

CUDA IPC is host-local. When one tensor-parallel replica spans multiple hosts,
run one OrbitKV server on each host and configure the connector with every
server endpoint in global TP-rank order:

```json
{
  "kv_connector": "OrbitKVConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "orbitkv.vllm",
  "kv_connector_extra_config": {
    "orbitkv.tp_shard_endpoints": [
      "http://host-a:50055",
      "http://host-b:50055"
    ]
  }
}
```

For TP8 and two endpoints, global ranks 0-3 register with the first server and
ranks 4-7 register with the second. Each server sees a local TP4 topology and
must manage the four GPUs on its own host. Every vLLM process must receive the
same ordered endpoint list.

The scheduler queries every shard and only reuses the prefix available from all
of them. Each worker loads with the lease issued by its local server. The
connector gives every shard a distinct namespace, so deployments with a
different host split cannot reuse an incompatible cache layout.

TP sharding currently requires equal contiguous shards and TP-only parallelism.
Pipeline, decode-context, and prefill-context parallelism are rejected when
more than one endpoint is configured.

When every TP shard Cache Manager is on the scheduler host, the connector derives
one socket from each endpoint and requires all sockets to exist:

```json
{
  "orbitkv.tp_shard_endpoints": [
    "http://127.0.0.1:50055",
    "http://127.0.0.1:50056"
  ]
}
```

An explicit `orbitkv.tp_shard_bootstrap_sockets` list is only needed for custom
paths. A Unix socket cannot cross a host boundary. Cross-host TP sharding needs
node-local query fan-out and is not supported by this adapter yet.

#### P/D Partial Tail Blocks

vLLM normally exposes hashes only for complete KV blocks. In a P/D deployment,
enable `orbitkv.pd_tail_save` on prefill and `orbitkv.pd_tail_load` on decode
to reuse the final partial prompt block as well. Start both vLLM processes with
the same explicit `PYTHONHASHSEED` and `--prefix-caching-hash-algo xxhash_cbor`.

Prefill: `{"orbitkv.pd_tail_save": true}`

Decode: `{"orbitkv.pd_tail_load": true, "orbitkv.wait_for_full_prefix": true}`

`orbitkv.wait_for_full_prefix` makes decode wait (up to 30s) until the full
prompt prefix is fetchable from a remote node via MetaServer + Mooncake. It only
applies when prefill and decode run separate engines; it does not observe
saves landing in a shared/local engine and has no effect when remote transfer is not
configured.

## Development

See the [examples](../examples/) directory for more usage examples.

## Testing

### Running Unit Tests

The test suite includes integration tests that verify the local client can communicate with a running Cache Manager.

#### Prerequisites

1. **Build the Rust extension**:

   ```bash
   cd python
   maturin develop --release
   ```

2. **Build the server binary**:

   ```bash
   cd ..
   cargo build --release --bin orbitkv-cache-manager
   ```

3. **Ensure CUDA is available** (tests require GPU):
   ```bash
   python -c "import torch; assert torch.cuda.is_available()"
   ```

#### Running Tests

```bash
cd python

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cache_manager_client.py -v

# Run with coverage
pytest tests/ --cov=orbitkv --cov-report=html
```

#### Test Structure

- **`tests/conftest.py`**: Contains pytest fixtures for:

  - `orbitkv_server`: Automatically starts/stops the Cache Manager for integration tests
  - `engine_client`: Creates a local Cache Manager client for the test
  - `client_context`: Provides a `ClientContext` representing a vLLM instance with GPU KV cache tensors
  - `registered_instance`: Provides a registered instance ID for query tests

- **`tests/test_cache_manager_client.py`**: Integration tests for:
  - Server connectivity
  - Query operations with various inputs

#### Test Fixtures

The `ClientContext` class abstracts a vLLM instance and provides:

- `register_kv_caches()`: Register GPU KV cache tensors with the server
- `query(block_hashes)`: Query available blocks
- `unregister_context()`: Unregister context from server

Example test usage:

```python
def test_query(client_context):
    """Test query operation."""
    result = client_context.query([])
    assert result is not None
```

## License

MIT
