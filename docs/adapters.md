# Engine adapter configuration

Start with the [single-node quickstart](single-node.md). This reference covers
custom sockets, native client ownership and vLLM connector modes. Model/layout
compatibility is documented in [hybrid recovery](hybrid-recovery.md).

## Native client

```python
from orbitkv import CacheManagerClient

client = CacheManagerClient("/tmp/orbitkv-50055.sock")
ok, message = client.health()
client.close()
```

`CacheManagerClient` is the Rust owner exposed through PyO3. Construct
`BlockHashes(page_hashes)` once per lookup; slices share its native allocation.
`query_prefetch` owns submission, revision and polling. `start_restore` returns
a client-bound handle for `poll_restore` or `wait_restore(timeout=...)`.
Native calls release the GIL. A timeout does not release GPU destinations while
a copy may still be running. See the [type reference](../python/orbitkv/orbitkv.pyi).

## Process channel

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
Use the same wheel version and CUDA variant for the Cache Manager and clients. Connector shutdown
explicitly closes the UDS session; imported CUDA mappings are released after
queued GPU transfers finish.

## Connector Modes

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

## TP shards and host boundary

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
      "http://127.0.0.1:50055",
      "http://127.0.0.1:50056"
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

An explicit `orbitkv.tp_shard_bootstrap_sockets` list is needed only for custom
paths. Cross-host TP sharding needs a future node-local query fan-out design.

## P/D Partial Tail Blocks

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


## Package ownership

The Python adapters own engine callbacks, GPU allocation and layout inspection.
Rust owns shared planning, query state, batching, waiting and transfer lifetimes.
The import package is `orbitkv`; plugins load their engine adapter lazily.

| Module | Responsibility |
| --- | --- |
| `identity.py` | Model and computation identity |
| `client/` | Connection configuration and CUDA registration |
| `vllm/config.py`, `layout.py`, `metadata.py` | Topology, cache groups and transfer intents |
| `vllm/scheduler.py`, `worker.py` | Scheduler/worker handoff and GPU page lifetime |
| `vllm/pd/` | Experimental prefill/decode handoff |
| `sglang/config.py`, `layout.py`, `recovery.py` | Identity, page layout and recovery evidence |
| `sglang/linker.py`, `plugin.py` | RadixCache lifecycle and plugin registration |
| `orbitkv-state`, `orbitkv-channel` | Compiled recovery and shared native client ownership |

Tests live in `python/tests/`; benchmarks live in `benches/`. Neither is included
in the runtime wheel. Contributor dependencies use the `test`, `dev` and `bench`
dependency groups; runtime extras select only one engine per environment.
