# OrbitKV architecture

## Mission

OrbitKV is a framework-neutral state cache and physical planner for vLLM and
SGLang. It does not schedule model execution and it is not a second inference
server. Framework adapters expose logical model state and local pages; OrbitKV
owns external replicas, transfer leases, storage tiers, and eventually the
policy that chooses placement, movement, reclamation, routing, or recomputation.

The data plane is derived from PegaFlow 0.24.5. The validated integration today
is vLLM. SGLang support currently consists of adapter contracts and source-pinned
integration targets; the executable HiCache backend remains an M1 deliverable.

## Process topology

The recommended deployment is one OrbitKV sidecar per inference node. Framework
adapters run inside the inference workers. A cluster index/router may run as a
separate service.

```text
                 cluster replica index / router
                 hash · tier · load · topology
                              |
            +-----------------+-----------------+
            |                                   |
   inference node A                    inference node B
 +-------------------+              +-------------------+
 | vLLM or SGLang    |              | vLLM or SGLang    |
 | framework adapter |              | framework adapter |
 +---------+---------+              +---------+---------+
           | local control / registered pages |
 +---------v---------+  Mooncake    +---------v---------+
 | OrbitKV sidecar   |<------------>| OrbitKV sidecar   |
 | pinned DRAM / SSD |              | pinned DRAM / SSD |
 +-------------------+              +-------------------+
```

The lifecycle and compatibility planes still use gRPC. The native local API now
supports query, publish, asynchronous restore completion, and lease release:
iceoryx2 carries fixed descriptors while a Unix socket authenticates the peer,
passes a sealed memfd descriptor arena, and supplies an eventfd for wakeups.
The vLLM adapter exposes this as an explicit `orbitkv.local_data` mode; its
default remains gRPC and the SGLang adapter has not yet been switched. KV bytes
must not travel through either control protocol: vLLM uses
registered CUDA IPC pages, SGLang will use a shared pinned host pool, and remote
transfers use the Mooncake-backed `TransferEngine`. See [transport.md](transport.md) for the
measured decision.

## Layering

```text
vLLM adapter                SGLang adapter
block hashes / CUDA IPC     radix hashes / HiCache pools / shared host pages
                             /
            +---- orbitkv-contract ----+
                 state identity
                 format compatibility
                 local page generations
                 recovery bundles
                           |
                    orbitkv-core
                 cache · leases · tiers
                  /                    \
       orbitkv-local              Mooncake Transfer / SSD
       iceoryx2 + UDS             RDMA · TCP fallback
```

### `orbitkv-contract`

This crate contains no framework or CUDA dependencies. Its first public types
are:

- `StateKey`: content identity, logical token span, component, and byte format;
- `StateFormat`: model/implementation digest, dtype, layout, and parallel shape;
- `StateComponent`: attention KV, MLA, recurrent, convolution, SWA, draft, and
  indexer state;
- `LocalPageRef`: generation-qualified CUDA IPC or shared-host page reference;
- `StateBundle` and `RecoveryContract`: the components needed to claim that a
  logical boundary is restorable.

Physical bytes may be shared across vLLM and SGLang only when their
`StateFormat` values are compatible. Sharing the core and policy never implies
blind cross-framework byte reuse.

### Framework adapters

The adapters translate framework-native state into `orbitkv-contract`:

| Concern | vLLM | SGLang |
| --- | --- | --- |
| Prefix identity | `Request.block_hashes` | Radix page hashes |
| Local GPU pages | vLLM block IDs + CUDA IPC | Radix/HiCache page indices |
| Host pages | OrbitKV-owned pinned blocks today | shared HiCache host pool |
| Hybrid state | KV cache groups and checkpoints | `PoolTransfer` components |
| Lifecycle | KVConnector callbacks | Radix/HiCache events |

Adapters do not decide which component set is a legal recovery point. That
logic belongs in the common recovery contract.

### `orbitkv-core`

The current core provides content-addressed sealed blocks, NUMA-aware pinned
memory, leases, LRU/TinyLFU admission, SSD, remote fetch, and session cleanup.
During M0/M1 it continues accepting the current vLLM-oriented key and page
registration APIs while new APIs are introduced beside them.

### Transfer and backing domains

The native physical domains are:

- framework GPU pages;
- shared or OrbitKV-owned pinned DRAM;
- local SSD;
- remote OrbitKV replicas over Mooncake-selected RDMA or TCP.

Mooncake Transfer Engine is the sole production remote-movement backend. It
contributes Segment/BatchTransfer, multi-NIC topology selection, endpoint
pooling, and rail failover. Mooncake Store Master is not OrbitKV's
semantic authority: bundle completeness, leases, generations, and planning
remain in OrbitKV.

## SGLang integration

### Stage 1: HiCache L3 backend

Implement `orbitkv.sglang.OrbitKVHiCacheStorage` against SGLang's dynamic
`HiCacheStorage` interface. It must support `batch_exists_v2`, `batch_get_v2`,
`batch_set_v2`, and named auxiliary pools. SGLang remains owner of GPU and host
allocation in this stage.

The host pool must use SGLang's shared-memory allocator. The sidecar maps that
same memory; it must not allocate a second DRAM copy.

### Stage 2: Radix lifecycle bridge

Publish prefix materialization, match, release, promotion, demotion, and removal
events from RadixAttention. OrbitKV uses the events to maintain a global replica
index and estimate next touch. It does not maintain a competing radix tree.

### Stage 3: page authority

Radix nodes consume generation-qualified OrbitKV page handles. This is the point
where OrbitKV may truthfully become the sole authority for page identity and
safe reuse.

## Safety invariant

A physical generation may be reused only when both conditions hold:

```text
SemanticDead(page, semantic_frontier)
and
ExecutionComplete(page, execution_frontier)
```

Semantic death proves that no future legal execution can read the state.
Execution completion proves that no submitted CUDA, SSD, or network operation
still references the generation. A lease or refcount supplies execution
evidence; it does not by itself prove semantic death.

## Planning direction

The target optimization problem is Minimum Persistent State Realization: find
the smallest complete `StateBundle` that can resume legal execution, then choose
its physical realization. The planner compares:

```text
queue delay
+ missing-state recomputation
+ restore time by tier and topology
+ transfer queueing
+ destination eviction externality
+ replica failure risk
```

It returns a worker plus a physical plan: source replica, restore or recompute,
target tier, prefetch deadline, eviction set, and replication action. Dynamo's
KV-aware worker scorer is the routing baseline, not the final planner.

## Ownership boundary by milestone

| Milestone | Framework owns | OrbitKV owns |
| --- | --- | --- |
| M0 | local page identity and execution | external replicas and current data plane |
| M1 | GPU + host pages | L3 storage, remote replicas, bundle query |
| M2 | GPU pages | shared host pages, L3, transfers |
| M3 | execution and local page identity | replica catalog, routing, and restore plans |
| M4 | execution and radix topology | page identity, generations, and all placements |
| M5+ | execution | semantic lifetime and compiled physical plans |
