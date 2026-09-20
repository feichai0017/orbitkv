# OrbitKV architecture

## Mission

OrbitKV is a framework-neutral state cache and physical planner for vLLM and
SGLang. It does not schedule model execution and it is not a second inference
server. Framework adapters expose logical model state and local pages; OrbitKV
owns external replicas, transfer leases, storage tiers, and eventually the
policy that chooses placement, movement, reclamation, routing, or recomputation.

The data plane is derived from PegaFlow 0.24.5. The vLLM connector, SGLang
direct GPU linker, and SGLang HiCache L3 backend have passed single-node GPU
recovery tests.

## Process topology

The recommended deployment is one OrbitKV Cache Manager per inference node. Framework
adapters run inside the inference workers. The same cache API is used whether a
hit is in node memory, SSD, or on a peer. A cluster index/router may run as a
separate service after the cache data plane is qualified.

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
 | OrbitKV Cache Manager |<--------->| OrbitKV Cache Manager |
 | pinned DRAM / SSD |              | pinned DRAM / SSD |
 +-------------------+              +-------------------+
```

Standalone deployment has no gRPC listener. Registration, health, sessions, and
cleanup use the authenticated bootstrap UDS. `--metaserver-addr` enables a
peer-only gRPC listener for transfer authorization and lock release. The process IPC supports query, publish, asynchronous
restore completion, and lease release:
iceoryx2 carries fixed descriptors while a Unix socket authenticates the peer,
passes a sealed memfd descriptor arena, and supplies an eventfd for wakeups.
The vLLM adapter requires this path and fails fast if the Cache Manager socket
is missing. Each inference process must reach a Cache Manager on its own host.
Pending queries return `Loading` and continue on Tokio. Publish holds its
iceoryx2 reply until D2H finishes, so the caller does not release source HBM
pages early while the dispatcher remains free. The Python cache client opens a
separate descriptor session for Publish on its first save, so an in-flight
save does not serialize the worker's Query/Restore calls behind that reply.
Instance cleanup serializes
against registration, drains GPU
load/save queues, and only then releases imported CUDA mappings. Superseded
sessions cannot clean up a replacement session. Both vLLM and the SGLang
direct linker register CUDA IPC pages and use iceoryx2 descriptors on the hot
path. The SGLang HiCache L3 compatibility backend still sends bounded host
pages over UDS; shared host page registration is its next transport
optimization. Remote transfers use the Mooncake-backed `TransferEngine`.
See [transport.md](transport.md) for the measured process-transport baseline.

## API and crate boundaries

| Layer | Code | Owns |
| --- | --- | --- |
| Framework adapters | `python/orbitkv/vllm`, `python/orbitkv/sglang` | Framework-specific hashes, layout, and page-lifetime events |
| Cache client | `python/orbitkv/client/data_plane.py`, `connection.py` | Query, publish, restore, release, lifecycle through the node-local connection |
| State contract | `orbitkv-contract` | State identity, format compatibility, bundles, page-reference types |
| Process IPC | `orbitkv-local`, `orbitkv-server/src/endpoint/` | iceoryx2 requests/replies, UDS bootstrap and lifecycle, pending queries, descriptor generation |
| Cache service | `orbitkv-server/src/cache/` | Transport-neutral operations, registration, and session cleanup |
| Cache engine | `orbitkv-core` | Leases, HBM transfer scheduling, pinned DRAM, SSD, local and remote lookup |
| Peer control | `orbitkv-proto`, `orbitkv-core/src/internode/p2p_service.rs` | Network authorization and transfer locks |
| Replica catalog | `orbitkv-metaserver`, `orbitkv-core/src/internode` | Candidate ownership and node liveness; currently a single in-memory service |
| Byte movement | `orbitkv-transfer`, `orbitkv-mooncake-sys` | Mooncake Segment/BatchTransfer over RDMA or TCP |

Transport-specific names belong at physical boundaries. Cache operations and
framework adapters use placement-neutral names and results. Moving a cache hit
from DRAM to SSD or another node should not change `query_prefetch`, `save`,
`start_restore`, or `release` for the caller. The `orbitkv-local` crate name is
kept because it describes one IPC implementation, not a different cache API.

## Layering

```text
vLLM adapter                SGLang adapter
block hashes / CUDA IPC     radix hashes / CUDA IPC or HiCache host pages
                             /
       python/orbitkv/client (cache API)
                    |
    orbitkv-local / iceoryx2 + UDS
                    |
              orbitkv-server/cache/operations
                           |
                    orbitkv-core
                 cache · leases · tiers
                    /           \
                  SSD      Mooncake Transfer
                           |
                    peer DRAM / SSD

     peer control: tonic / gRPC, only with --metaserver-addr

    orbitkv-contract: shared state identity and recovery semantics
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

`StateBundle::has_required_components` currently checks availability by
component kind only. It is not yet a proof of restorable state: token coverage,
model/format compatibility, and the framework's recovery rule must be checked
before a bundle is used to skip prefill or route a request.

Physical bytes may be shared across vLLM and SGLang only when their
`StateFormat` values are compatible. Sharing the core and policy never implies
blind cross-framework byte reuse.

### Framework adapters

The adapters translate framework-native state into `orbitkv-contract`:

| Concern | vLLM | SGLang |
| --- | --- | --- |
| Prefix identity | `Request.block_hashes` | Radix page hashes |
| Local GPU pages | vLLM block IDs + CUDA IPC | Radix page indices + CUDA IPC on the direct path |
| Host pages | OrbitKV-owned pinned blocks today | Direct path uses OrbitKV-owned pinned blocks; HiCache L3 copies SGLang L2 pages into them |
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

### Stage 1: direct GPU linker for full-attention models

`orbitkv.sglang.linker.OrbitKVLinker` is registered through SGLang's plugin
entry point and selected by `--radix-cache-backend orbitkv` together with
`--enable-unified-cache-external-linker`. The latter is required for SGLang's
scheduler to submit GPU restores and drain linker completions. It uses
`UnifiedCacheLinker` callbacks to look up radix page hashes, pin SGLang-owned
GPU slots during asynchronous saves and loads, and transfer bytes through the
same Cache Manager API as vLLM. Each scheduler rank registers its local GPU KV
buffers through CUDA IPC. A model-, rank-, and layout-scoped namespace prevents
incompatible byte reuse. The direct path currently requires a single full-KV
pool; hybrid SWA/Mamba, DSA, draft-model, and auxiliary GPU state need a more
complete recovery contract. SGLang retains authority over HBM allocation and
prefix-tree nodes.

### Stage 1 compatibility: HiCache L3 backend

`orbitkv.sglang.storage.OrbitKVHiCacheStorage` implements SGLang's dynamic
`HiCacheStorage` interface, including `batch_exists_v2`, `batch_get_v2`,
`batch_set_v2`, and named auxiliary pools. SGLang remains owner of HBM and
its L2 host pool. A completed L2 page is copied into Cache Manager-owned
bounded pinned memory, with the core SSD tier available for eviction.
The adapter scopes keys by model, parallel rank, pool, dtype, layout, and page
size. `batch_exists_v2` returns legal `restorable_prefix_pages` for trailing
auxiliary state, and only reports a prefix as a hit when all required pools
are available. The simple path uses per-page UDS operations; shared-region
registration and batched query/transfer are pending performance work.

### Stage 2: Radix lifecycle bridge for routing

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

The descriptor arena validates its slot generation and Cache Manager session epoch,
and vLLM pins save-source blocks until Publish returns. These checks do not
yet validate a framework HBM page's reuse generation. `LocalPageRef` defines
the future contract, but current Publish still carries raw block IDs;
generation enforcement requires page-lifecycle information from the adapter
before a stale ID can be rejected at the Cache Manager boundary.

## Multi-node cache path and deployment

Today, `orbitkv-metaserver` is a separate in-memory gRPC service. A Cache Manager
registers sealed block hashes asynchronously and heartbeats its node session.
After a local miss, it queries the service for candidate owners. A selected
source Cache Manager authorizes and pins its blocks through gRPC, then Mooncake reads
the bytes into the destination's pinned memory. The destination can cache that
replica and restore it to framework HBM through its normal cache API. Network
gRPC carries control metadata and leases; Mooncake carries KV bytes. Mooncake's
P2P handshake supplies transport endpoint metadata, not KV ownership.

The present catalog is soft state, has no replicated persistence, and does not
backfill all resident keys after a metadata-service restart. It is therefore a
single-node failure and remote-hit-rate risk, even though local cache hits can
continue without it. Before declaring distributed cache production-ready, add
resident-inventory replay with a catalog epoch, bounded batched lookup and a
Cache Manager-side candidate cache; verify behavior across service restart, node
failure, and stale transfer capabilities.

The next deployment shape keeps one Cache Manager per inference node and uses a
separately deployed replica directory for discovery. Keep etcd, if adopted, for
small strongly consistent membership/configuration and directory epochs, not
for per-block reads or writes. The block catalog itself can remain a purpose-
built soft-state service with sharded replicas and Cache Manager-local snapshots.
Only a verified remote miss needs a control-plane round trip; the source
Cache Manager remains the authority for an actual transfer. The router can later
consume the same catalog without entering the cache data path. This is a
design target, not current implementation.

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
