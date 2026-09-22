# OrbitKV Python Package

Framework adapters and Python bindings for the OrbitKV state cache, built with
Rust and PyO3. Both vLLM and SGLang single-node DRAM cache paths have
GPU-validated adapters for the releases below.

## Features

- **CacheManagerClient**: Native Rust owner of lifecycle, queries, publication and asynchronous restores across storage tiers, exposed directly through PyO3
- **OrbitKVConnector**: vLLM external-cache connector; Cache Manager selects RAM, SSD, or configured remote fetch
- **OrbitKVLinker**: SGLang direct GPU-page cache through the same Cache Manager channel
- **PdConnector**: experimental vLLM P/D handoff through Mooncake; independent of the external-cache connector

Import the native API with `from orbitkv import BlockHashes, CacheManagerClient`.
Build `hashes = BlockHashes(page_hashes)` once for a lookup, then reuse it in
`client.query_prefetch(instance_id, hashes, request_id)` until ready. A slice
such as `hashes[:hit_blocks]` shares the native allocation. `warm_prefix` also
takes a `BlockHashes`; `save` still accepts the per-layer hash lists.
The old Python `client/manager.py` and Python `ChannelClient` API are removed.
`query_prefetch` owns submission/revision/polling internally; `start_restore`
returns a client-bound handle for `poll_restore` or `wait_restore(timeout=...)`.
Native calls release the GIL, including eventfd waits. A wait timeout never
releases GPU destinations or cancels an already submitted copy. Socket selection
stays in `orbitkv.client.connection`; engine page allocation stays in the adapters.

## Installation

The supported engine releases are vLLM `0.29.0` and SGLang `0.5.20`, verified
on 2026-09-20. Their source submodules are pinned to the corresponding release
tags in `third-party/`. Keep the engine GPU dependencies in separate environments
so each installation can be tested independently.

The Cache Manager currently imports PyTorch at startup for CUDA IPC handling.
Run the wheel in an environment with a compatible PyTorch/CUDA runtime; the
base `orbitkv-llm` dependency set does not install PyTorch for you.

The distribution names are `orbitkv-llm` (CUDA 12) and
`orbitkv-llm-cu13` (CUDA 13). The import name is always `orbitkv`. The
`[vllm]` and `[sglang]` extras install the exact engine release we validate;
install one extra per environment. Install a wheel matching the host CUDA
runtime and Python ABI.

The two extras are declared mutually exclusive for `uv` resolution: these
engine releases pin different `numba` versions. Use one engine environment
per installation, including when running the benchmark matrix.

```bash
cd python
uv venv ../.venv/vllm-release --python 3.11
uv pip install --python ../.venv/vllm-release/bin/python 'vllm==0.29.0' pytest requests --torch-backend=cu130
uv venv ../.venv/sglang-release --python 3.11
uv pip install --python ../.venv/sglang-release/bin/python 'sglang==0.5.20' --torch-backend=cu130
```

### From Source

```bash
# Run from the repository root
git submodule update --init --recursive third-party/mooncake
pip install maturin

# Build and install in development mode
cd python
maturin develop

# Or build an installable wheel with the Cache Manager and Mooncake runtime
cd ..
./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
# Install the resulting target/wheels/orbitkv_llm_cu13-*.whl in each engine environment
```

For a CUDA 12 wheel, run `./scripts/build-wheel.sh --release` instead. A
standalone `maturin build` only builds the extension; the script also stages
the service binaries and shared libraries, then checks the completed wheel.

## Usage

For a complete one-manager-per-host deployment, installable wheel commands,
capacity controls, and a warm-hit check for both engines, follow the
[single-node guide](../docs/single-node.md). The examples below describe the
adapter-specific configuration.

### Cache Manager client

```python
from orbitkv.client import CacheManagerClient

client = CacheManagerClient("/tmp/orbitkv-50055.sock")
ok, message = client.health()
client.close()
```

### vLLM KV Connector

```python
from vllm import LLM
from vllm.config import KVTransferConfig

# Configure vLLM to use OrbitKVConnector
kv_transfer_config = KVTransferConfig(
    kv_connector="OrbitKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="orbitkv.vllm",
)

# Create LLM with KV transfer enabled
llm = LLM(
    model="/path/to/immutable-model",
    kv_transfer_config=kv_transfer_config,
)
```

For FullAttention/MLA + aligned Mamba layouts, vLLM uses the shared Rust recovery
contract to join attention pages with exact recurrent/conv checkpoints. SSD
checkpoint queries may defer admission; completed groups stay leased through
the worker handoff. The final-token clamp selects an earlier validated boundary
and skips unused leased pages. See [hybrid recovery](../docs/hybrid-recovery.md)
for the supported layouts, limits and DRAM/SSD validation commands. HBM remains
owned by vLLM; no additional service or deployment flag is required.

### SGLang direct GPU cache

For supported full-attention or hybrid models on SGLang `0.5.20`, use the
OrbitKV RadixCache backend. Install the OrbitKV wheel in the SGLang environment
so its `sglang.srt.plugins` entry point is visible to the scheduler process.
The Cache Manager and SGLang worker must run on the same host; the Unix socket
path must match the Cache Manager's bootstrap socket.

```bash
orbitkv-cache-manager --addr 127.0.0.1:50055 --pool-size 2gb

ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/model \
  --page-size 64 \
  --enable-unified-cache-external-linker \
  --radix-cache-backend orbitkv
```

SGLang owns the HBM page lifecycle. OrbitKV registers the worker's GPU KV
buffers once through CUDA IPC, then queries, saves, and restores page-aligned
blocks through the same local client used by vLLM. iceoryx2 carries cache
commands; GPU data is copied directly between the registered buffers and the
Cache Manager's pinned memory. Single-rank DRAM and SSD recovery across a
SGLang restart are validated while the Cache Manager remains running. The
plugin's admission hook keeps a pending request queued; a subsequent match
reuses its ready lease. A five-second preparation budget cancels waiting
interest and permits recomputation. This is a fixed waiting guard, not a cost
predictor; a submitted GPU restore still requires completion. See the
[SSD measurements](../docs/ssd-performance.md).

Both engines fingerprint local model artifacts and bind their configuration and
registered storage layout to a versioned cache identity. Hub models must use
an immutable commit revision. `ORBITKV_MODEL_FINGERPRINT` can supply a verified
64-digit SHA-256 deployment digest to avoid reading large artifacts at startup;
`ORBITKV_CACHE_SCOPE` optionally isolates tenants or experiments. Dynamic LoRA
is rejected. Restart the engine when weights change; live refits are unsupported.
See [state identity](../docs/state-identity.md) for the exact boundary.

Each tensor-parallel rank registers its local buffers; SGLang
intersects sets of legal boundaries across ranks. This direct path accepts
full-attention MHA/MLA, Full + SWA, and Full + recurrent/conv pools. The compiled
contract requires a complete window or an exact checkpoint at the selected
boundary. DSA, draft, ReplaySSM, int8 checkpoints, SWA request rings and unknown
auxiliary state remain rejected. See [hybrid recovery](../docs/hybrid-recovery.md)
for model evidence, restrictions and reproducible gates. Both `--radix-cache-backend orbitkv` and
`--enable-unified-cache-external-linker`
are required: the second flag makes SGLang schedule device loads and drain
the linker's completion queues. Startup fails if it is omitted. The GPU E2E
has qualified the single-rank path; multi-rank TP recovery remains to be
validated on a matching GPU deployment.

#### Process channel

The connector derives the Cache Manager's Unix socket from the configured
endpoint. Scheduler Query/Release and worker Publish/Restore use iceoryx2;
registration, health, session ownership, and cleanup use the bootstrap UDS.
Each inference process requires a Cache Manager on its own host. A missing
socket fails at startup. No extra configuration is needed on one node.

The process channel connects to `/tmp/orbitkv-<orbitkv.port>.sock`, matching the
Cache Manager default. Use `orbitkv.bootstrap_socket` for a custom single-manager
path. `orbitkv.timeout_ms` (default 5000) bounds hot requests and health;
registration and unregister allow at least 120 seconds for CUDA setup/draining.
`orbitkv.spin_iterations` defaults to 64. Standalone Cache Managers do
not start gRPC. Client and Cache Manager must use matching
bootstrap protocol versions (currently version 2).

`orbitkv.wait_for_full_prefix` is supported on the local path: pending queries
return `QueryLoading`, and repeated queries with the same instance/request/group
identity retrieve the result. Query arguments must remain unchanged while
pending. Superseded queries are explicitly cancelled. Each session permits
128 outstanding queries, with 1024 globally and a 60-second reply lifetime.
Cancellation revokes waiting interest while submitted reads drain safely;
undelivered results release their leases without another readiness poll.
Use the same ABI-3 wheel for the Cache Manager and clients. Connector shutdown
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
vllm serve /path/to/immutable-model \
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

#### TP shards and host boundary

CUDA IPC is host-local. When one tensor-parallel replica spans multiple hosts,
one Cache Manager per host is necessary, but the current vLLM scheduler still
has to query **every** TP shard through a local UDS socket. Thus cross-host TP
sharding is not supported by this adapter yet. Configuring remote HTTP
endpoints does not create an inference-to-Cache-Manager network path.

For multiple Cache Managers on the *same* scheduler host, list their endpoints
in global TP-rank order:

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

For TP8 and two endpoints, global ranks 0-3 register with the first manager and
ranks 4-7 with the second. Both managers and the scheduler must be on the same
host, and every vLLM process must receive the same ordered endpoint list.

The scheduler queries every shard and only reuses the prefix available from all
of them. Each worker loads with the lease issued by its local server. The
connector gives every shard a distinct namespace, so deployments with a
different host split cannot reuse an incompatible cache layout.

TP sharding currently requires equal contiguous shards and TP-only parallelism.
Pipeline, decode-context, and prefill-context parallelism are rejected when
more than one endpoint is configured.

The connector derives one socket from each endpoint and requires all sockets
to exist:

```json
{
  "orbitkv.tp_shard_endpoints": [
    "http://127.0.0.1:50055",
    "http://127.0.0.1:50056"
  ]
}
```

An explicit `orbitkv.tp_shard_bootstrap_sockets` list is needed only for custom
paths. Cross-host TP sharding needs a future node-local query fan-out design.

#### P/D Partial Tail Blocks

vLLM normally exposes hashes only for complete KV blocks. In a P/D deployment,
enable `orbitkv.pd_tail_save` on prefill and `orbitkv.pd_tail_load` on decode
to reuse the final partial prompt block through the **external-cache**
`OrbitKVConnector` path. These options are separate from the direct Mooncake
`PdConnector`. Start both vLLM processes with
the same explicit `PYTHONHASHSEED` and `--prefix-caching-hash-algo xxhash_cbor`.

Prefill: `{"orbitkv.pd_tail_save": true}`

Decode: `{"orbitkv.pd_tail_load": true, "orbitkv.wait_for_full_prefix": true}`

`orbitkv.wait_for_full_prefix` makes decode wait (up to 30s) until the full
prompt prefix is fetchable from a remote node via Catalog + Mooncake. It only
applies when prefill and decode run separate engines; it does not observe
saves landing in a shared/local engine and has no effect when remote transfer is not
configured.

## Development

See the [examples](../examples/) directory for more usage examples.
For P/D transfer, NIXL, and the difference from remote-cache sharing, see
[P/D and NIXL](../docs/pd.md). OrbitKV's `PdConnector` currently targets vLLM
only; SGLang has no OrbitKV P/D adapter.

## Package organization

`orbitkv` is the installed import package; the native extension remains
`orbitkv.orbitkv`. The engine entry points load their own adapters lazily.
Neither engine is required to import the base package or discover its plugins.

| Module | Responsibility |
| --- | --- |
| `identity.py` | Framework-neutral model and computation identity |
| `client/` | Cache Manager connections, CUDA registration, and transfer ownership |
| `vllm/config.py` | Deployment configuration, model namespace, and rank topology |
| `vllm/layout.py` | Cache groups, storage mapping and shared recovery requirements |
| `vllm/metadata.py` | Scheduler/worker transfer intents and completion reports |
| `vllm/scheduler.py`, `worker.py` | Scheduling decisions and GPU page lifetime |
| `vllm/metrics.py` | Connector measurements |
| `vllm/pd/` | Experimental vLLM prefill/decode handoff |
| `sglang/config.py`, `layout.py` | SGLang identity and GPU page-layout validation |
| `sglang/recovery.py` | Absolute recovery evidence, recurrent checkpoints and tree handoff |
| `crates/orbitkv-state`, `src/recovery.rs` | Compiled recovery rules and their PyO3 binding |
| `sglang/linker.py` | RadixCache lookup/offload/restore lifecycle |
| `sglang/plugin.py` | Backend registration and cache construction |

Import the owning module directly. The framework-neutral client is shared by
both adapters; engine-specific metadata stays inside its adapter package.

The published wheel exposes only the `vllm` and `sglang` engine extras.
Contributor dependencies live in the `test`, `dev`, and `bench` dependency
groups in `pyproject.toml`; `dev` includes `test`. Benchmark code, tests, and
results are not installed in the runtime wheel.

## Testing and benchmarks

Correctness tests live in `tests/unit/`, `tests/integration/`, `tests/e2e/`,
and `tests/stress/`. Shared process, path, and import-stub helpers live in
`tests/support/`. Heavy fixtures import torch only when exercised.
See [the test gates](tests/README.md) for the trigger and runtime requirements
of each gate. A source-only unit run needs neither CUDA nor a compiled wheel:

```bash
cd python
uv run --isolated --no-project --with pytest --with numpy --with requests pytest
```

Performance workloads and measurements live at the repository root under
[`benches/`](../benches/README.md), with separate CPU-only harness tests. Run them
using the selected engine environment and preserve the run's manifest with its
measurements.

## License

Apache-2.0
