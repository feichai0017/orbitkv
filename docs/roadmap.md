# OrbitKV roadmap

The execution order is sustained single-node qualification, independent-replica
cache sharing (DP), P/D plus cache reuse, then replicated catalogs and scale.
Cancellation and transport lifetime safety apply at every stage. Cross-host
TP/PP and KV-aware routing have later gates. See the
[deployment comparison and priority rationale](distributed-comparison.md).
Milestones describe intended gates, not deployed capabilities. The
detailed work queue lives in [TODO.md](../TODO.md).

## Current delivery priorities

The starting point is a tested single-node DRAM/SSD path for both pinned engine
releases, with TP=1 dense full-attention as the shared qualification baseline.
Owned asynchronous queries, byte admission, shared backing reads and GPU
completion lifetimes already exist. Embedded catalog discovery and membership
also exist; real two-host serving, catalog replication and online placement
changes remain unqualified or unimplemented. Automatic queued warming stays
opt-in because the recorded controls do not establish a throughput benefit.

SGLang hybrid pools and vLLM aligned recurrent groups now use the same recovery
validator with absolute token coverage. This closes the duplicated boundary
logic. Hybrid demand now discovers metadata, selects a legal boundary and
reads only its required ranges, then validates actual leases. Deterministic
[process fault gates](fault-qualification.md) cover SSD delay/cancel, lost
notifications, Publish fencing and restart. Deterministic Qwen3-8B TP=1 serving
also covers cancellation with unrelated requests, lost notification and both
process restarts in each engine. Multi-rank/long-running fault stress and
page-generation gates remain distinct qualifications.

The milestone identifiers below name work areas, not a requirement to finish
every optimization before starting the next area. In particular, page-lifetime
and identity fixes apply to every path as it is qualified; they cannot wait for
the later general semantic compiler.

| Priority | Deliverable | Acceptance boundary |
| --- | --- | --- |
| First: reliable ordinary demand | Maintain deterministic cancellation, lost-notification, engine/Manager restart and stuck-Publish gates; extend concurrent model-serving fault stress. Profile the normal DRAM/SSD restore path. | Exact restored bytes and engine output controls; no stale result adoption or page reuse during active DMA; unrelated requests progress; reservations drain after terminal completion or proven revocation. A timeout alone cannot release memory. |
| Implemented, opt-in: bounded preparation | Small arrival-order lookahead, retained leases, bounded reads and stopping controls. | Three matched pairs per engine completed. Keep off by default because SGLang P95 regresses despite a throughput gain; cutoffs also reduce throughput. |
| First distributed serving gate: DP | Qualify two real hosts running independent matching TP=1 replicas, separately for vLLM and SGLang, through the existing embedded catalog and Mooncake TE path. | Positive remote transfer and GPU restore bytes, output controls, source-restart rejection, catalog replay and bounded failure handling. Report discovery, authorization and etcd traffic separately. |
| Then: P/D with cache reuse | Qualify the existing vLLM handoff together with external caching; separately integrate and qualify SGLang's native handoff lifecycle. | A cached P-side prefix still reaches D; completed D-side state can be reused by a later P request. Cancellation and worker restart cannot expose incomplete state. |
| Before production distributed deployment: catalog HA | Add replicated catalog evidence, versioned placement, handoff/repair and operational failure handling. | Three catalog failure domains, partitions, lease expiry, etcd outage and placement changes; bounded replay, source holds and staging. Replicating etcd alone does not replicate the catalog. |
| Later expansion | Remote SSD staging, measured source/cost selection, broader model recovery, copy/compute overlap and optional Dynamo routing. | Each has its own recovery, resource and performance gate; cross-host TP/PP, resharding and cross-engine format conversion are separate capabilities. |

Start the two-host DP harness once the ordinary-demand lifetime gate passes;
local preparation and retention tuning can continue alongside it. A warming
speedup is **not** a prerequisite for DP. Available host capacity determines
when its runtime gate can run: multiple Managers on one machine cannot close
the two-host item. Qualify same-host TP per replica after the initial TP=1 DP
gate, without claiming cross-host tensor parallelism.

### Next reviewable changes

1. **Follow up on the preparation tradeoff.** The opt-in
   [consumer-owned path](request-preparation.md) selects at most four dense
   arrival-order candidates, keeps prepared leases budgeted, and retires stale
   interests without another poll. Bounded batches, best-effort completion and
   a conservative deadline miss are implemented and measured in three matched
   pairs per engine. Keep preparation off: vLLM improves modestly, while SGLang
   trades tail latency for throughput. Isolate SGLang admission effects before
   revising selection; automatic hybrid forecasts and priority prediction remain open.
2. **Reduce measured exposed waits.** The
   [ordinary Qwen3 profile](recovery-performance.md) separates host reads,
   Manager restore and engine completion observation. Investigate read batching
   and vLLM's completion-observation tail with matching traffic before changing
   retention or SSD write admission. Keep deterministic output and ownership gates.
3. **Start real two-host DP qualification.** Ordinary TP=1 demand lifetimes now
   have native and model-serving fault evidence. Use independent same-format
   replicas and Mooncake TE; require positive remote and GPU-copy bytes, source
   incarnation rejection and catalog replay. Local policy tuning can continue
   alongside it. Multi-rank fault soak remains a separate extension.

The [shared-cache driver and restart gate](shared-cache-qualification.md) now pass
for both engines with independent TP=1 replicas over same-host TCP. Rust bounds
source allocation reservations, retains overdue pins and exports remote stage
timing. Transport revocation for
permanently lost requesters remains open; neither a timeout nor an etcd lease
expiry permits memory reuse. Physical two-host/RDMA evidence remains required.

Keep these changes separately reviewable. The existing
[reference-based policy sequence](queued-warming.md#reference-implementations-and-policy-order)
defines the retained-page and stopping contracts. If a preparation policy
increases read amplification without repeatable latency/goodput benefit, leave
it experimental and continue with DP qualification. Retention and SSD write
admission get separate experiments after this comparison.

### Evidence and code ownership

Use Qwen3-8B and the pinned engine releases for continuity. Compare native
engine behavior and ordinary OrbitKV demand before adding compatible LMCache
or FlexKV configurations; record incompatible versions or layouts instead of
silently changing the workload. Keep model revision, HBM/DRAM capacity, storage,
request sequence, concurrency and tracing fixed. Run at least three paired
repetitions with order reversal and report variation, p50/p95/p99 TTFT, TPOT,
throughput, SSD bytes per request, useful/unused prepared bytes and retained
byte-seconds. Do not equate prepared-page reuse with causal latency savings.
The current overlay-filesystem SSD controls do not qualify physical NVMe
performance; publish a separate real-device result before making that claim.

Engines own HBM allocation and scheduling decisions; adapters supply queue and
page-lifetime evidence. The existing Manager endpoint and core query/prefetch/storage code own
operation delivery, external residency, leases, budgets and transfer
dependencies. Catalog/cluster code owns distributed evidence and membership;
Mooncake TE owns remote byte movement. Extend those owners without adding a
second scheduler facade or forwarding-only client. Tests stay under their
package's `tests/`; workloads and results stay under `benches/`. Every delivery
updates its deployment instructions and measured capability claims in the
README and canonical `docs/`, which the website renders directly.

## M0: framework-neutral foundation

Deliver:

- move Rust packages under `crates/` and keep the repository root a virtual
  workspace;
- introduce `orbitkv-state`;
- place the vLLM cache connector and Mooncake P/D adapter under `orbitkv.vllm`;
- establish `orbitkv.sglang` and `orbitkv.client` package boundaries;
- establish the `orbitkv-channel` iceoryx2 control ABI;
- connect lifecycle probes, epoch fencing, and shutdown to the real Cache Manager and
  Python client;
- bootstrap a generation-checked descriptor arena over UDS and execute
  `QueryBundle` through the cache service;
- expose a framework-neutral Rust cache client through PyO3 for query, publish,
  restore, and lease release; keep connection configuration in Python;
- integrate a pinned upstream Mooncake Transfer Engine as the single remote
  movement implementation;
- preserve current vLLM behavior.

Gate:

- Cargo metadata, format, workspace check, and host-safe tests pass;
- default Python unit tests pass;
- the vLLM plugin registers its connectors without loading the engine at
  package import time;
- the SGLang plugin registers the direct GPU-page linker.

## M1: SGLang direct GPU-page linker

Deliver:

- register SGLang GPU KV buffers with the Cache Manager through CUDA IPC;
- switch framework adapters to the available local `QueryBundle`, publish,
  restore, completion, and lease-release APIs;
- support full-attention MHA and MLA layouts;
- reject unsupported draft, DSA, and auxiliary GPU state until complete recovery
  contracts are available;
- expose cold miss, partial prefix, warm hit, cancellation, and restart metrics.

Gate:

- exact generated-text parity against a cold-control namespace;
- a real GPU load after flushing radix cache and after a worker restart;
- SGLang worker restart preserves Cache Manager-resident cache;
- unsupported state fails during initialization rather than reporting a hit.

## M2: common StateBundle query and native local transport

Versioned model/storage keys isolate deployments. SGLang uses compiled
prefix/window/checkpoint rules, and vLLM's aligned recurrent layouts use the
same validator with absolute-span and leased-group evidence; see
[hybrid recovery](hybrid-recovery.md). Page-generation enforcement and wider
recovery coverage remain open. Complete those local semantics before using
cache metadata as evidence for distributed routing.

Deliver:

- extend the implemented model/storage key with absolute token spans and
  request-specific adapter evidence;
- validate token spans, model/format compatibility, and required components at
  each recovery boundary;
- use common bundle validation for vLLM hybrid boundaries;
- use local restore operations and eventfd wakeups for both adapters;
- define framework-neutral region registration and transfer-plan operations;
- add generation validation to every local page reference.

Gate:

- immutable matching deployments hit while weight, tokenizer/processor,
  adapter, dtype/layout/rank, and span changes cannot produce a false hit;
- vLLM and SGLang generate equivalent recovery contracts for a shared test
  model;
- adapter code contains no tier-selection or bundle-completeness policy;
- load/save throughput is not regressed against the M0 baseline.

The [SSD measurements](ssd-performance.md) exposed a missing SGLang readiness
boundary. The pinned-release plugin now supplies a nonblocking admission hook,
and core queries complete and release abandoned results without further polling.
Single-rank DRAM/SSD serving recovery has dedicated GPU gates. Explicit query
operation/revision tickets, retained byte budgets and shared backing reads are
implemented, with [bounded concurrent bursts](concurrent-performance.md) and
[sustained native/DRAM/SSD controls](sustained-performance.md) for both engines.
The vLLM adapter gates new admissions until an admitted restore reaches compute;
this resolves a deferred-queue capacity stall found by the sustained workload.
Bounded [queued-request DRAM warming](queued-warming.md) now uses the existing
query path without retaining warmup leases. Page-lifetime accounting now records
successful H2D, unused releases and completed byte-seconds; hints yield to
foreground ownership and enter the reclaimable class. The
[page-use controls](queued-warming.md#page-use-and-reclamation-controls) still
show no established throughput gain; SGLang releases most warmed pages unused.
The [reference-based policy sequence](queued-warming.md#reference-implementations-and-policy-order)
now has consumer-owned preparation, stopping and drain. Next calibrate expected
use time and restore-versus-recompute decisions without delaying DP qualification.
Measure exposed wait and read amplification separately from retention changes.
Generation-safe layer readiness precedes copy/compute overlap. The
[implementation stages](state-planning.md#implementation-sequence) retain the
larger ownership/fault-qualification requirements and later Dynamo integration.
Multi-rank serving and sustained concurrent goodput remain separate qualification
work. Recoverable distributed-cache work can begin while local optimization
continues.

## M2.5: recoverable multi-node cache

The [distributed cache design](distributed-cache.md) selects etcd for membership
and configuration, an embedded replica catalog, and Mooncake TE for payloads.
D0 inventory recovery and D1 candidate discovery, leased membership, fixed
placement and embedded catalog serving are implemented. Directory replicas and
online migration remain planned. The first serving
gate uses matching dense-attention namespaces and TP=1, testing each engine
separately.

Deliver in order:

- D0 (implemented): versioned DRAM inventories, bounded journals and
  snapshot/delta recovery. Tests cover real directory restart, concurrent
  residency changes, lost replies and history overflow;
- D1 discovery (implemented): bounded positive candidate caching, batched and
  coalesced lookup grouped by catalog host under one deadline, Manager-side planning,
  exact source runtime/residency checks,
  and buffer/hold ownership through asynchronous cancellation;
- D1 completion (implemented): bounded requester records retain release retries
  through control outages, consume late authorization replies after cancellation,
  and keep memory until Mooncake confirms whole-batch release. Requester-crash
  revocation and reconciliation of lost authorization replies remain open;
- D1 membership (implemented): transactional Node ID registration, persistent
  epochs, lease deadlines, bounded snapshots and Watch repair; new remote work
  stops when membership evidence or registration validity is unavailable;
- D1 deployment (implemented): per-shard inventory replay and catalogs embedded
  in Managers; standalone directory binaries and flags removed. Cross-host
  serving and orphaned-transfer revocation qualification remain open;
- D2: versioned rendezvous shard placement, replicated evidence, handoff,
  bounded subscriptions and failure recovery;
- D3: remote SSD staging and calibrated source selection under sender and
  receiver budgets.

The requesting Manager plans transfers. Source Managers validate and pin data;
directory hints cannot authorize reads. Per-block operations do not use etcd.
Retain prior baseline reports and compare metadata traffic and recovery costs.

Gate:

- a directory restart or owner loss cannot cause an incorrect KV hit;
- remote hits recover after inventory replay, without restarting managers;
- cache misses remain bounded when discovery or transfer fails;
- transfer cancellation, requester loss and lease expiry cannot permit memory
  reuse before terminal transport completion or proven revocation;
- multi-node measurements separate discovery RPCs, coordinator activity,
  synchronization traffic, source authorization and payload transfer;
- catalog index memory, replay history, source pins and destination staging
  remain bounded, including during repair and placement changes.

## M3: KV-aware routing and physical planning

Deliver:

- consume vLLM and SGLang KV placement events;
- consume the recovered catalog to track replicas by worker and tier;
- reuse a pinned `dynamo-kv-router` selector and its production service lifecycle
  as the baseline, following the [integration boundary](state-planning.md#reuse-dynamo-for-request-routing);
- feed qualified tier events and request-load lifecycle into that selector;
- evaluate calibrated queue, transfer, recompute, and eviction estimates without
  mixing block scores with milliseconds or counting reuse twice;
- select the worker, then have its Cache Manager revalidate sources and create
  a leased restore plan;
- qualify Mooncake topology-aware slicing, endpoint pooling, and alternate-rail
  retry against OrbitKV transfer plans.

Gate:

- event/hash mapping and load reservation produce the expected upstream
  selections on captured traces;
- joint planning beats load-only and overlap-only baselines on a held-out trace;
- stale events and worker restarts produce bounded fallback/reselection and
  cannot become an incorrect cache hit.

## M4: generation-safe page references

The transfer-lifetime prerequisite is implemented: uncertain restore completion
does not release destinations, and partially submitted GPU work is drained
before a terminal error. Per-page allocator generations are still open. The
pinned engine APIs expose block IDs/indices without allocation generations;
counting transfer requests is not a substitute for observing allocation reuse.

Deliver:

- generation-qualified GPU registrations and external page handles;
- explicit semantic and execution frontiers;
- CUDA/RDMA/SSD completions advance one execution-fence abstraction;
- SGLang and vLLM adapters pass page generations and consume manager-authored
  handles for external replicas; engine HBM allocation remains engine-owned.

Gate:

- generation reuse cannot race an outstanding operation under stress and fault
  injection;
- external page identities come from the Cache Manager; engine-owned HBM page
  IDs are validated at transfer boundaries;
- cache cleanup is safe across cancellation, preemption, and process death.

## M5: semantic state compiler

Deliver:

- a `may_read(query, state)` lifetime IR;
- full, sliding, sink-local, recurrent, and hybrid recovery plans;
- Minimum Persistent State Realization;
- compiled retention, checkpoint, placement, and replication policies.

Gate:

- report Retention Amplification alongside TTFT, TPOT, throughput, and network
  traffic;
- compiled plans reduce physical state without changing exact-model outputs;
- ring/checkpoint/tier choices are derived from the state contract and measured
  cost, not selected by model-name branches.

## M6: backend and ecosystem expansion

Deliver only after M1-M5 gates:

- agentic multi-turn value model;
- multi-DC replica planning;
- signed plan/evidence bundles if deployment requires them.
