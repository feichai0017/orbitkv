# OrbitKV roadmap

The order is local correctness, recoverable distributed cache, then KV-aware
routing. Milestones describe intended gates, not deployed capabilities. The
detailed work queue lives in [TODO.md](../TODO.md).

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
- add a framework-neutral Python cache client for local query, publish,
  restore, and lease release;
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
- reject hybrid, draft, DSA, and auxiliary GPU state until complete recovery
  contracts are available;
- expose cold miss, partial prefix, warm hit, cancellation, and restart metrics.

Gate:

- exact generated-text parity against a cold-control namespace;
- a real GPU load after flushing radix cache and after a worker restart;
- SGLang worker restart preserves Cache Manager-resident cache;
- unsupported state fails during initialization rather than reporting a hit.

## M2: common StateBundle query and native local transport

Versioned model/storage keys now isolate deployments. The component-presence
check is not a safe, model-aware recovery proof. [State identity and recovery](state-identity.md)
defines the migration and gates. Complete those local semantics before using
cache metadata as evidence for distributed routing.

Deliver:

- extend the implemented model/storage key with absolute token spans and
  request-specific adapter evidence;
- validate token spans, model/format compatibility, and required components at
  each recovery boundary;
- move vLLM hybrid reconciliation from the adapter into common bundle logic;
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
implemented, with [bounded concurrent bursts](concurrent-performance.md).
Next add bounded request-driven DRAM warming and measured restore-versus-recompute
decisions. Generation-safe layer readiness precedes copy/compute overlap. The
[implementation stages](state-planning.md#implementation-sequence) retain the
larger ownership/fault-qualification requirements and later Dynamo integration.
Multi-rank serving and sustained concurrent goodput remain separate qualification
work. Recoverable distributed-cache work can begin while local optimization
continues.

## M2.5: recoverable multi-node cache

The [distributed cache design](distributed-cache.md) selects etcd for membership
and configuration, an embedded replica catalog, and Mooncake TE for payloads.
D0 inventory recovery and D1 candidate discovery/planning are implemented
against the current standalone directory.
Optional etcd registration, renewal, cached membership and remote admission are
also implemented. The embedded catalog deployment remains planned. The first serving
gate uses matching dense-attention namespaces and TP=1, testing each engine
separately.

Deliver in order:

- D0 (implemented): versioned DRAM inventories, bounded journals and
  snapshot/delta recovery. Tests cover real directory restart, concurrent
  residency changes, lost replies and history overflow;
- D1 discovery (implemented): bounded positive candidate caching, batched and
  coalesced lookup, Manager-side planning, exact source runtime/residency checks,
  and buffer/hold ownership through asynchronous cancellation;
- D1 membership (implemented): transactional Node ID registration, persistent
  epochs, lease deadlines, bounded snapshots and Watch repair; new remote work
  stops when membership evidence or registration validity is unavailable;
- D1 deployment (remaining): embedded catalog serving and
  qualified peer transfer lifetimes including source timeout/revocation; retire
  the standalone MetaServer deployment after cutover;
- D2: versioned rendezvous shard placement, replicated evidence, handoff,
  bounded subscriptions and failure recovery;
- D3: remote SSD staging and calibrated source selection under sender and
  receiver budgets.

The requesting Manager plans transfers. Source Managers validate and pin data;
directory hints cannot authorize reads. Per-block operations do not use etcd.
Compare against measurements of the current standalone directory before removal.

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
