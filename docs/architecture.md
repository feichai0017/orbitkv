# OrbitKV architecture

## Mission

OrbitKV is a KV cache for vLLM and SGLang and a proposed framework-neutral
state planner. It does not schedule model execution. Each framework owns its
HBM allocation and active GPU page lifecycle. Its adapter exposes block
identity and registered GPU buffers; OrbitKV currently owns external pinned
DRAM/SSD replicas and transfer leases. Shared recovery semantics and joint
placement/routing policy are future work.

The data plane is derived from PegaFlow 0.24.5. The vLLM connector and SGLang
direct GPU linker have passed single-node GPU recovery tests.

## Process topology

Run one OrbitKV Cache Manager per inference host. Framework adapters run in the
inference processes and use the same cache API for local DRAM, SSD, and remote
fetches. The cache manager decides where to source a hit; the inference engine
still decides when to query and save. Remote fetch is experimental. There is no
OrbitKV KV-aware request router today.

```text
       current multi-node cache (experimental)
   host A                                      host B
   vLLM or SGLang                             vLLM or SGLang
   engine-owned HBM                           engine-owned HBM
        | CUDA IPC + UDS/iceoryx2                   | CUDA IPC + UDS/iceoryx2
   Cache Manager A ---- Mooncake RDMA/TCP ---- Cache Manager B
   pinned DRAM / SSD                         pinned DRAM / SSD
            \                                   /
             \---- MetaServer (in-memory) -----/
                    candidate locations only
```

Single-node deployment consists of one engine and one Cache Manager on the same
host and needs neither MetaServer nor peer gRPC. Current SSD backing is a cache
file truncated on Cache Manager startup, not durable KV storage across manager
restarts. In the current multi-node path,
each manager asynchronously advertises sealed block hashes to the separate
MetaServer, asks it for candidates after a local miss, then authorizes/pins a
source through peer gRPC before Mooncake reads bytes. That directory can lose
remote-hit information on restart; it is not a high-availability deployment.

Standalone deployment has no gRPC listener. Registration, health, sessions, and
cleanup use the authenticated bootstrap UDS. `--metaserver-addr` enables a
peer-only gRPC listener for transfer authorization and lock release. Process
IPC supports query, publish, asynchronous restore completion, and lease
release:
iceoryx2 carries fixed descriptors while a Unix socket authenticates the peer,
passes a sealed memfd descriptor arena, and supplies an eventfd for wakeups.
The vLLM adapter requires this path and fails fast if the Cache Manager socket
is missing. Each inference process must reach a Cache Manager on its own host.
Pending queries return `Loading` and continue on Tokio. The endpoint owns one
session-scoped operation/revision registry bound to instance, request, and group;
the core query future owns its backing
reads and returns a terminal result. Cancelling or disconnecting drops reply
ownership while submitted reads drain, including cache admission and lease
release, without another poll. SGLang's plugin admission hook keeps pending
requests queued until a leased result or bounded fallback is available.
Query reservations use the registered group's padded bytes and remain charged
through preparation, result ownership, and GPU completion. Global and instance
limits bound retained payloads; identical backing reads can be shared while
each request keeps its own ticket and lease. See [query budgets](server.md#query-ownership-budgets).
Publish holds its
iceoryx2 reply until D2H finishes, so the caller does not release source HBM
pages early while the dispatcher remains free. The Python cache client opens a
separate descriptor session for Publish on its first save, so an in-flight
save does not serialize the worker's Query/Restore calls behind that reply.
Instance cleanup serializes against registration, drains GPU load/save queues,
and only then releases imported CUDA mappings. Superseded
sessions cannot clean up a replacement session. Both vLLM and the SGLang
direct linker register CUDA IPC pages and use iceoryx2 descriptors on the hot
path. Remote transfers use the Mooncake-backed `TransferEngine`.
See [transport.md](transport.md) for the measured process-transport baseline.

## API and crate boundaries

| Layer | Code | Owns |
| --- | --- | --- |
| Framework adapters | `python/orbitkv/vllm`, `python/orbitkv/sglang` | Framework-specific hashes, layout, and page-lifetime events |
| Cache client | `python/orbitkv/client/manager.py`, `connection.py` | Query, publish, restore, release, lifecycle through the node-local connection |
| State contract | `orbitkv-state` | State identity, format compatibility, bundles, page-reference types |
| Process IPC | `orbitkv-channel`, `orbitkv-server/src/endpoint/` | iceoryx2 requests/replies, UDS bootstrap and lifecycle, pending queries, descriptor generation |
| Cache service | `orbitkv-server/src/cache/` | Transport-neutral operations, registration, and session cleanup |
| Cache engine | `orbitkv-core` | Leases, HBM transfer scheduling, pinned DRAM, SSD, local and remote lookup |
| Peer control | `orbitkv-proto`, `orbitkv-core/src/internode/p2p_service.rs` | Network authorization and transfer locks |
| Replica catalog | `orbitkv-metaserver`, `orbitkv-core/src/internode` | Candidate ownership and node liveness; currently a single in-memory service |
| Byte movement | `orbitkv-transfer`, `orbitkv-mooncake-sys` | Mooncake Segment/BatchTransfer over RDMA or TCP |

Transport-specific names belong at physical boundaries. Cache operations and
framework adapters use placement-neutral names and results. Moving a cache hit
from DRAM to SSD or another node should not change `query_prefetch`, `save`,
`start_restore`, or `release` for the caller. The process channel implements
the current iceoryx2/UDS connection without defining a separate cache API.

## Layering

```text
vLLM adapter                SGLang adapter
block hashes / CUDA IPC     radix hashes / CUDA IPC
                             /
       python/orbitkv/client (cache API)
                    |
    orbitkv-channel / iceoryx2 + UDS
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

    orbitkv-state: shared state identity and recovery semantics
```

### `orbitkv-state`

This crate contains no framework or CUDA dependencies. Its first public types
are:

- `StateKey`: the materialized model/storage namespace and versioned native prefix/group key;
- `StateDescriptor`: logical token span, component and format evidence for future recovery validation;
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

The adapters resolve a shared versioned identity at startup and translate native
hashes and GPU layouts into the cache API. The manager binds registered storage
geometry and uses `StateKey` across tiers. Full recovery evidence is the next step:

| Concern | vLLM | SGLang |
| --- | --- | --- |
| Prefix identity | `Request.block_hashes` | Radix page hashes |
| Local GPU pages | vLLM block IDs + CUDA IPC | Radix page indices + CUDA IPC on the direct path |
| Host pages | OrbitKV-owned pinned blocks | OrbitKV-owned pinned blocks |
| Hybrid state | KV cache groups and checkpoints | Unsupported until complete recovery contracts are implemented |
| Lifecycle | KVConnector callbacks | Radix-cache events |

Adapters do not decide which component set is a legal recovery point. That
logic belongs in the common recovery contract.

### `orbitkv-core`

The current core provides content-addressed sealed blocks, NUMA-aware pinned
memory, leases, LRU/TinyLFU admission, SSD, remote fetch, and session cleanup.
DRAM, SSD and the directory share `orbitkv-state::StateKey`; registration binds
model identity to stored layout before Query/Publish. Engine page IDs remain raw,
and generation-qualified page types and complete recovery proofs are not yet
enforced. See [state identity](state-identity.md) for fingerprint configuration.

### Transfer and backing domains

The native physical domains are:

- framework GPU pages;
- shared or OrbitKV-owned pinned DRAM;
- local SSD;
- remote OrbitKV replicas over Mooncake-selected RDMA or TCP.

Mooncake Transfer Engine is the sole remote-movement backend in this codebase. It
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

Both DRAM and SSD recovery are GPU-validated at TP=1. SGLang's general plugin
admission hook retains pending requests in the queue and consumes the ready
result on a subsequent match. vLLM reports unresolved lookups through its own
connector scheduler contract. The original SSD readiness failure and successful
follow-up remain in [SSD results](ssd-performance.md); earlier warming and cost
selection are in [state demand and transfer planning](state-planning.md).

### Future: Radix lifecycle bridge for routing

Publish prefix materialization, match, release, promotion, demotion, and removal
events from RadixAttention. OrbitKV uses the events to maintain a global replica
index and estimate next touch. It does not maintain a competing radix tree.

### Future: generation-safe page references

Adapters pass generation-qualified references for engine-owned HBM pages.
OrbitKV validates the registration session and page generation before copying,
while the engine still allocates and reuses its HBM slots. OrbitKV can assign
handles to its own external replicas without taking over the GPU allocator.

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
synchronizes its sealed DRAM inventory asynchronously and heartbeats its node
session. Actual insertions and removals share a monotonic residency sequence;
bounded snapshot pages and ordered deltas reconstruct the directory after
restart or lost history. Incomplete replacement views stay hidden until commit.
After a local miss, it queries the service for candidate owners. A selected
source Cache Manager authorizes and pins its blocks through gRPC, then Mooncake reads
the bytes into the destination's pinned memory. The destination can cache that
replica and restore it to framework HBM through its normal cache API. Network
gRPC carries control metadata and leases; Mooncake carries KV bytes. Mooncake's
P2P handshake supplies transport endpoint metadata, not KV ownership.

The present catalog is soft state and has no replicated persistence. Recovery
from directory restart is implemented and tested over real gRPC, including idle
owners, concurrent eviction and journal overflow. It remains a single service:
remote discovery can be unavailable during failure or reconstruction. D0 covers
DRAM evidence; remote SSD, an embedded candidate index, multi-host failover and
transfer-capability fencing require later qualification. See the
[protocol and limits](../crates/orbitkv-metaserver/README.md).

The agreed target keeps one Cache Manager per host, embeds a sharded replica
catalog in those managers, and uses etcd for membership and versioned placement
configuration. Mooncake TE remains the data plane. Owner inventories publish
ordered changes; snapshots and bounded delta replay repair lost evidence.
Rendezvous hashing assigns logical catalog shards to a small replicated host
set. Each Manager keeps a bounded candidate index so a warm query can avoid
directory RPCs. A miss queries the appropriate shards in batches. The requesting
Manager constructs the fetch plan; the source validates and pins the exact data.

The [distributed cache design](distributed-cache.md) specifies identities,
snapshot cuts, subscriptions, placement transitions, transfer lifetimes and
failure behavior. Metadata replication is asynchronous evidence replication;
it does not imply payload replication or general object-store CAS semantics.
etcd is outside per-block operations. The embedded design is **not implemented
yet**; qualify it against the current standalone directory baseline, then remove
that obsolete deployment at cutover. A later router can consume replica
summaries without entering the transfer path.

```text
host A                                           host B
engine HBM                                      engine HBM
    | UDS + iceoryx2 / CUDA IPC                      | UDS + iceoryx2 / CUDA IPC
Cache Manager A  <---- Mooncake KV bytes ---->  Cache Manager B
  DRAM / SSD · local candidate index              DRAM / SSD · local candidate index
  catalog shards  <---- replicated metadata ---> catalog shards
         \________ etcd membership/placement _________/

catalog summaries ----> future KV-aware router <---- engine load/events
```

## Planning direction

The proposed [state demand and transfer planner](state-planning.md) describes
engine readiness signals, recovery boundaries, prefetch timing, and the local
evaluation sequence. It is a design proposal; current cache hits do not imply
those planning capabilities are implemented.

After the cache and catalog are reliable, the target optimization problem is
Minimum Persistent State Realization: find
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

These terms require calibrated units and critical-path accounting; overlapping
operations cannot simply have their durations added together.

Reuse Dynamo's worker selector for request placement. Dynamo v1.4.2 provides
an independent Rust router crate and a selection service; its runtime is an
optional dependency of the crate. The selected Cache Manager then revalidates
replicas and constructs the leased physical plan: source, restore or recompute
proposal, staging budget, transfer deadline, and completion dependencies. The
engine owns execution admission and HBM allocation. A routing load reservation
does not replace a transfer lease.

The [Dynamo integration boundary](state-planning.md#reuse-dynamo-for-request-routing)
and [implementation stages](state-planning.md#implementation-sequence) specify
the reusable components, pending engine-interface dependency, and acceptance
gates. No router dependency is introduced into the current single-node core.

## Ownership boundary by milestone

| Milestone | Framework owns | OrbitKV owns |
| --- | --- | --- |
| M0 | local page identity and execution | external replicas and current data plane |
| M1 | GPU pages | Pinned DRAM/SSD replicas and direct GPU restore |
| M2 | GPU pages | Common recovery contracts and transfer operations |
| M2.5 | GPU pages and execution | recoverable replica catalog and remote cache fetch |
| M3 | execution and local page identity | KV-aware routing and restore plans |
| M4 | HBM allocation, physical GPU page IDs, and execution | external replica handles, validated GPU references, and transfer fences |
| M5+ | execution | semantic lifetime and compiled physical plans |
