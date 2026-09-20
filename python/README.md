# OrbitKV Python Package

Framework adapters and Python bindings for the OrbitKV state cache, built with
Rust and PyO3. The vLLM adapter is validated today; the SGLang runtime backend
is tracked as the next milestone.

## Features

- **EngineRpcClient**: Thin Python client for the local OrbitKV sidecar
- **CacheDataClient**: Common hot-path facade with gRPC and local IPC implementations
- **OrbitKVConnector**: vLLM KV connector for distributed inference with KV cache transfer
- **SGLang contracts**: Configuration and state-pool mapping without claiming a completed backend

## Installation

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

### Sidecar client

```python
from orbitkv.client import EngineRpcClient

client = EngineRpcClient("http://127.0.0.1:50055")
ok, message = client.health()
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

The connector defaults to `orbitkv.local_data="auto"`. It derives the
sidecar's Unix socket from the configured endpoint and uses local IPC for
scheduler Query/Release and worker Publish/Restore whenever every required
socket is available. No extra configuration is needed for the normal
single-node deployment.

Use `true` to require local IPC, or `false` to force the compatibility gRPC
data plane:

```json
{
  "kv_connector": "OrbitKVConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "orbitkv.vllm",
  "kv_connector_extra_config": {
    "orbitkv.local_data": false
  }
}
```

The local path connects to `/tmp/orbitkv-<orbitkv.port>.sock`, matching the
sidecar default. Use `orbitkv.local_bootstrap_socket` for a custom single-sidecar
path. `orbitkv.local_timeout_ms` (default 5000) bounds each local request and
`orbitkv.local_spin_iterations` defaults to 64. Registration, health, session
watching, and unregister remain on gRPC in this mode. The current local
dispatcher is serial, so `orbitkv.wait_for_full_prefix` is rejected together
with local data until QueryBundle has an asynchronous completion protocol; use
the gRPC data path for blocking remote prefetch.

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

When every TP shard sidecar is on the scheduler host, auto mode derives one
socket from each endpoint and uses local IPC only when all sockets exist:

```json
{
  "orbitkv.tp_shard_endpoints": [
    "http://127.0.0.1:50055",
    "http://127.0.0.1:50056"
  ]
}
```

An explicit `orbitkv.tp_shard_bootstrap_sockets` list is only needed for custom
paths. A Unix socket cannot cross a host boundary, so auto mode falls back to
gRPC when one or more shard sockets are not local.

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

The test suite includes integration tests that verify the `EngineRpcClient` can correctly communicate with a running `orbitkv-server` instance.

#### Prerequisites

1. **Build the Rust extension**:

   ```bash
   cd python
   maturin develop --release
   ```

2. **Build the server binary**:

   ```bash
   cd ..
   cargo build --release --bin orbitkv-server
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
pytest tests/test_engine_client.py -v

# Run with coverage
pytest tests/ --cov=orbitkv --cov-report=html
```

#### Test Structure

- **`tests/conftest.py`**: Contains pytest fixtures for:

  - `orbitkv_server`: Automatically starts/stops `orbitkv-server` for integration tests
  - `engine_client`: Creates an `EngineRpcClient` connected to the test server
  - `client_context`: Provides a `ClientContext` representing a vLLM instance with GPU KV cache tensors
  - `registered_instance`: Provides a registered instance ID for query tests

- **`tests/test_engine_client.py`**: Integration tests for:
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
