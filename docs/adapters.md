# vLLM and SGLang integration

OrbitKV integrates external cache recovery with engine-owned GPU pages and
request lifecycles. Start with the [single-node quickstart](single-node.md).
The [completion plan](completion-plan.md#release-baseline-and-reference-policy)
records release targets and upgrade gates. The package and source pins are
vLLM **0.30.0** and SGLang **0.5.20**. The vLLM upgrade has local A100
DRAM/SSD and eager/graph evidence; independent acceptance and broader deployment
qualification remain open. Historical 0.29.0 model/topology evidence is not
automatically transferred to the new release.

## Ownership contract

| Responsibility | Owner |
| --- | --- |
| Request scheduling, batching, model execution, HBM allocation and native prefix cache | Inference engine |
| State/layout/rank description, lookup and allocation callbacks, consumption dependencies | Engine-specific Python adapter |
| DRAM/SSD/peer replicas, recovery planning, admission, leases and task lifetime | Existing OrbitKV Rust owners |
| Raw DRAM restore | Native executor in the engine process, with Manager-retained sources |
| Publish and SSD/codec execution | Manager physical workers |
| P/D bootstrap and request lifecycle | Currently OrbitKV's vLLM P/D connector or SGLang's native control plane; target native lifecycle for both |
| Remote memory movement | TENT; memory registration and completion do not define cache publication or DecodeReady |

A hit is a recoverable state boundary across every required group, not simply a
stored key. The engine supplies destination pages and their readiness; native work
retains source and destination ownership until drain. Enqueue acknowledgement,
per-layer readiness, terminal completion and source retirement are separate facts.
Model/adapter identity, cache salt, state representation and shard layout must
match. Sharing a Manager does not make engine layouts interchangeable.

## LMCache and FlexKV reference

Use release source, not an unversioned example, when comparing integrations:

| Released reference | Mechanism to learn from | OrbitKV decision |
| --- | --- | --- |
| [vLLM 0.30.0 LMCache MP entry](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py) and [LMCache 0.5.5 implementation](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/integration/vllm/lmcache_mp_connector.py) | Engine connector delegates external-cache behavior to a separately installed package | Keep a small official entry and one maintained adapter; storage and cache scheduling remain in Rust. |
| [SGLang 0.5.20 LMCRadixCache](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py) and [LMCache 0.5.5 example](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/examples/sgl_integration/README.md) | Prefix lookup, load-back, finished-request storage and eviction integrate with native lifecycles; MP connects to a separate daemon | Preserve SGLang tree ownership through the existing UnifiedRadixCache external linker. Do not copy a second Radix tree or LMCache's ZMQ transport into OrbitKV. |
| [LMCache 0.5.5 MP P/D recipe](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/disaggregated_prefill.rst) | Native NIXL handoff and cache reuse compose through MultiConnector, with separate P and D cache servers | Reuse the separation of responsibilities; retain TENT as our payload engine. Verify recipe prerequisites against the selected release. Shared P/D Manager use needs its own contention gate. |

Later SGLang main changes toward LMCacheUnifiedRadixCache are not the 0.5.20
release contract. API similarity and upstream support matrices do not qualify
OrbitKV. Do not inherit LMCache's version compatibility branches or assume its
documented P/D prerequisite patches are in the selected engine release.

The pinned [vLLM 0.30.0 FlexKV entry](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py)
loads its separately installed implementation when the connector is constructed.
The [SGLang 0.5.20 FlexKV integration](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py)
owns task completion and connection lifetime beside the engine cache. OrbitKV
uses these explicit backend boundaries while retaining its existing Rust client,
Unified tree and per-layer GPU completion dependencies.
The separately inspected [FlexKV adapter at `738ddc14`](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/flexkv/integration/vllm/vllm_v1_adapter.py)
is a pinned main reference, not an OrbitKV-supported release. Like LMCache MP,
it separates scheduler callbacks from worker callbacks. OrbitKV follows that
ownership split without inheriting another cache's task engine or compatibility
branches.

## vLLM callbacks and current internal dependencies

`OrbitKVConnector` implements `KVConnectorBase_V1` and `SupportsHMA`.
`SchedulerAdapter` in `scheduler.py` discovers legal external coverage and hands
allocated pages to `WorkerAdapter` in `worker.py`; the worker registers tensors, restores/saves state and reports
completion through the pinned engine contract. `layout.py` and `metadata.py`
describe actual groups and intents rather than a second cache scheduler.

All worker roles implement 0.30.0's `get_transfer_results` directly. P/D failure
and receive completion are drained together into `KVConnectorTransferResults`;
the scheduler consumes native `failed_recving`, including through MultiConnector.
There is no P/D failure-metadata class or duplicate failure queue. Ordinary cache
publication declares `requires_kv_delivery=False`; a reliable P/D producer keeps
the engine's delivery requirement. Divergent hybrid hits remain disabled.

Only the selected role implementation is imported and constructed. Metrics come
from that role; native sessions close if role initialization fails. The connector
inherits unchanged optional callbacks from the pinned engine base. Its required
layer-save callback remains explicit because saves are submitted at step end.
Restore failures propagate with their retained GPU ownership; there is no separate
availability flag or background health-polling thread. Plugin registration errors
remain visible instead of silently selecting a conflicting connector.

Two internal dependencies currently require deliberate replacement:

- `runtime.py` wraps `GPUModelRunner.update_requests`: preempted saves must stop
  before page reuse; restore must follow initialization/COW and precede recurrent
  state preprocessing. Remove it only after the selected engine supplies that order.
- `scheduler.py:bind_gpu_block_pool` disables native multi-group prefix hits to
  prevent a dense group becoming visible before its recurrent state. Replace it
  with correct all-state availability; preserve the fast native HBM path when the
  full state is valid. A capability flag alone does not resolve publication races.

Registration currently uses the external package plugin. Official upstream
registration must replace duplicate registration, not hide name conflicts.
Use the [S5 sequence](completion-plan.md#s5--released-engine-integration-and-upstream-contributions)
for release alignment, lifecycle PRs and removal gates.

## SGLang linker and current internal dependencies

Select `--radix-cache-backend orbitkv` with
`--enable-unified-cache-external-linker`. `plugin.py` constructs a
`UnifiedRadixCache`; `OrbitKVLinker` implements lookup, queued load, layer progress,
offload, completion, reset and close. SGLang retains tree locks and page allocation. The release linker lookup returns
ready boundaries, not a pending ticket; current OrbitKV admission Hooks defer a
pending request while allowing other requests to progress. Do not assume HiCache
scheduler hooks run when OrbitKV rejects the separate hierarchical-cache mode.
The [hybrid recovery contract](hybrid-recovery.md) defines legal state boundaries.

`events.py` owns the CUDA events and producer/consumer counters needed before
graph capture. `linker.py` consumes those events and owns cache queries, loads,
offloads and registration. Plugin discovery imports neither the native extension
nor GPU layout code; graph and admission hooks check the selected backend before
loading OrbitKV resources. Native bindings load on first native API access.

`RecoveryLinkerWrapper` and `RecurrentComponent` currently extend the released
Full/SWA linker with checkpoint-specific behavior. Aborts use the released
`BasePrefixCache.finish(ABORT)` → `UnifiedRadixCache.release_aborted_request` →
`RecoveryLinkerWrapper.release_request` lifecycle. It cancels unconsumed queries
and drains already-published destinations before releasing their tree locks;
there is no additional Scheduler abort Hook.

The ordinary cache profile registers two internal Hooks: graph initialization
and pending-query admission. Enqueue preparation adds one Hook only when
`ORBITKV_PREPARE_REQUESTS=1` or `ORBITKV_QUEUE_WARMUP=1` is set before startup.
The six P/D observation Hooks are registered only with `SGLANG_MOONCAKE_TRANSFER_ENGINE=orbitkv`.
These remaining targets are version-coupled dependencies, not stable public APIs.
Replace them with consumed factory/lifecycle/component contracts and then delete
the duplicate logic. Unknown DSA, draft, auxiliary state and unsupported request
rings remain rejected until they have complete recovery contracts.

## P/D and cache composition

Ordinary remote-cache recovery discovers metadata locally, obtains a source
grant and performs TENT READ into Manager-owned storage before GPU restoration.
Current live P/D performs TENT WRITE into decoder-authorized destinations.
Neither operation direction inherently wins on latency, and neither proves RDMA.

vLLM currently has OrbitKV-owned `PdPrefillConnector` / `PdDecodeConnector`
control logic. The public entry constructs only its selected role. `PrefillWorker`
and `DecodeWorker` each own their engine callbacks and request lifecycle directly;
`PdWorkerBase` owns shared GPU layout registration and TENT construction. There is
no Worker-to-Handler forwarding layer or class-callback mixin. Prefill task
records and inflight accounting live with their executors in `prefill_async.py`;
the decode waiter owns its own executor and generation state. Source completion,
destination grants, cancellation and physical drain retain separate authorities.

The pinned vLLM 0.30.0 Mooncake worker directly constructs the legacy
`mooncake.engine.TransferEngine`; it has no constructor argument or public
factory for OrbitKV TENT. Selecting that connector or replacing its module class
would not preserve the current payload contract. Native lifecycle migration
therefore remains S5.4 work rather than a connector rename.
Its receive-error path also marks a request finished without a remote WRITE
drain acknowledgement. A synchronous TENT wrapper alone cannot make decoder
page reuse safe: cancellation must revoke new writes and retain destinations
until every already-authorized writer has drained. This is a prerequisite for
retiring OrbitKV's generation/grant/cancellation protocol.
The experimental vLLM factory patch now consumes the thin
`orbitkv.vllm.transport.TentTransferEngine` adapter with the native connector and
upstream router. Its separate receive fix retains D pages until all producers
finish; permanent peer loss and shutdown still need S3 evidence. The official
0.30.0 release does not include these patches.

SGLang now registers TENT through an explicit, lazy factory in the patched
0.5.20 profile. The global-class substitution is removed; selecting TENT on an
unpatched release fails startup. Its six private completion Hooks remain
observations, not page-release authority. The upstream deferred-release timeout
still permits release without a full drain acknowledgement. Factory integration
does not close that lifecycle gap. Audit equivalence before deleting existing
owners or claiming full S5.4 acceptance.
Do not allow two connectors to write the same target range or independently
release it. See [current P/D configuration and limits](pd.md).

As checked on 2026-09-30, SGLang's [external-linker construction PR #40595](https://github.com/sgl-project/sglang/pull/40595)
is open, the [Mamba lifecycle proof of concept #40759](https://github.com/sgl-project/sglang/pull/40759)
is closed without merge, and [load-failure lifecycle PR #40896](https://github.com/sgl-project/sglang/pull/40896)
is open. None is a released replacement for the remaining Hooks. Graph events
must exist before capture; constructing a linker later does not establish that
ordering. Track the cutover in S5.3/S5.4 rather than adding version fallbacks.

## Native client

```python
from orbitkv import CacheManagerClient

client = CacheManagerClient("/tmp/orbitkv-50055.sock")
ok, message = client.health()
client.close()
```

`CacheManagerClient` is the Rust owner exposed through PyO3. Construct
`BlockHashes(page_hashes)` once per lookup; slices share its native allocation.
`query_prefetch` owns submission, revision and polling.
`register_context_batch(..., tensors=...)` retains actual tensor/exporter objects
alongside the IPC metadata used by Publish and Manager SSD/codec routes.
`start_restore(..., ready_stream=...)` takes the engine's destination-readiness
stream and returns a client-bound handle for `poll_restore` or
`wait_restore(timeout=...)`. Optional `layer_events` bind retained CUDA events
to registered layers. `wait_restore_enqueued` waits for fresh event records,
allowing per-layer consumer dependencies; it does not acknowledge destination
page reuse. Unencoded DRAM copies execute in the native engine worker; their
final local result follows DMA drain without waiting for Manager source reaping.
vLLM uses layer waits in piecewise mode and coarser entry waits for supported
full-graph layouts; SGLang installs external events before capture.
See [layer consumption](engine-local-restore.md#layer-readiness-and-framework-consumption)
for recurrent, packed-buffer and multi-part limits. Repeated registration of the same binding is rejected, and unregister
or close drains accepted operations. Native calls release the GIL. A timeout
does not release destination page assignments while a copy may still be running. See the [type reference](../python/orbitkv/orbitkv.pyi).

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
bootstrap protocol versions (currently bootstrap 7, channel ABI 11, cache schema 9, and lifecycle 4).
Bootstrap transfers five metadata/notification FDs; GPU registration attaches
the shared payload arena FDs separately. The local executor partitions large
raw plans into at most 1 MiB parts under one whole-operation fence. Operation
metadata is capped at 32 MiB and per-session prepared metadata at 64 MiB;
see [execution scope and qualification gates](engine-local-restore.md).

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
`OrbitKVConnector` path. These options are separate from the direct TENT-backed
`PdPrefillConnector` and `PdDecodeConnector`. Start both vLLM processes with
the same explicit `PYTHONHASHSEED` and `--prefix-caching-hash-algo xxhash_cbor`.

Prefill: `{"orbitkv.pd_tail_save": true}`

Decode: `{"orbitkv.pd_tail_load": true, "orbitkv.wait_for_full_prefix": true}`

`orbitkv.wait_for_full_prefix` makes decode wait (up to 30s) until the full
prompt prefix is fetchable from a remote node via the local global index and Mooncake. It only
applies when prefill and decode run separate engines; it does not observe
saves landing in a shared/local engine and has no effect when remote transfer is not
configured.


## Package responsibilities and qualification

| Module | Responsibility |
| --- | --- |
| `identity.py` | Model artifacts and computation fingerprinting |
| `client/` | Connection configuration and GPU registration handoff |
| `vllm/connector.py`, `scheduler.py`, `worker.py` | vLLM contract and scheduler/worker ownership |
| `vllm/layout.py`, `metadata.py` | Cache groups and transfer intents |
| `vllm/pd/` | Current experimental handoff; native reuse audit required before replacement |
| `sglang/layout.py`, `recovery.py` | Pool geometry and state/checkpoint handoff |
| `sglang/events.py` | Graph-capture events and per-forward restore dependencies |
| `sglang/linker.py`, `plugin.py`, `pd.py` | External linker, registration and current P/D transport adapter |
| `orbitkv-state`, `orbitkv-channel`, core execution owners | Shared semantics, native client and physical lifetime |

Tests live in `python/tests/`; benchmark programs live in `benches/`; neither is
runtime wheel content. Keep source-only unit tests separate from native and engine
gates. Qualify disabled-backend startup, native HBM hits, cold/partial/full misses,
DRAM/SSD/peer restores, request cancellation, preemption, restart and advertised
eager/graph modes. Record actual transfer bytes and resource drain, not output alone.
TP/PP/attention-DP/EP, heterogeneous P/D and containers each need explicit cells.
Latest releases are the upgrade reference; unrun cells remain open.
