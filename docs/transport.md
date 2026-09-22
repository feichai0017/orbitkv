# OrbitKV transport architecture

This document separates the inference-to-Cache-Manager channel, remote data
movement, and replica discovery so none of them becomes an accidental second
source of KV truth. "Same-host" describes where the client connects; the
Cache Manager decides whether a requested block is resident in RAM, needs SSD
prefetch, or can be fetched from a peer with Mooncake.

## Decision

| Boundary | Control | Payload | Status |
| --- | --- | --- | --- |
| inference process to local Cache Manager | iceoryx2 request/response and UDS lifecycle | registered CUDA IPC pages | integrated |
| local bootstrap and lifecycle | Unix socket with credential and file-descriptor passing | memfd/eventfd handles and registration metadata | implemented; explicit region protocol planned |
| Cache Manager to Cache Manager | Mooncake P2P handshake | Mooncake BatchTransfer over RDMA/TCP | stable Mooncake runtime integrated |
| replica directory | soft-state network API | no KV bytes | embedded fixed shards; replication planned |
| administration | HTTP | no KV bytes | existing |

Both framework adapters require `orbitkv-channel` for hot data operations. Every
inference process must connect to a Cache Manager on its own host; a missing
Unix socket fails fast.
Registration, health, session watching, and unregistration use the
authenticated bootstrap UDS for both adapters.

## Process channel

`orbitkv-channel` uses iceoryx2 `0.10.0`. The workspace minimum Rust version is
therefore `1.89`. Its first ABI is a fixed 64-byte message carrying:

- protocol magic and ABI version;
- command or status code;
- request identity and Cache Manager session epoch;
- offset, length, and generation of a descriptor in a separately registered
  arena;
- two opcode-specific scalar fields.

The initial command vocabulary is `QueryBundle`, `Restore`, `Publish`,
`Release`, and lifecycle probes. Variable-length hashes and page arrays do not
live in the message. KV bytes never live in the message.

One inference process gets one iceoryx2 client endpoint. The Cache Manager exclusively
creates and owns the server endpoint; clients only open it. The endpoint uses
iceoryx2's thread-safe IPC service because the server owns it on a dedicated
control thread. Calls spin only for a bounded number of iterations and then
yield. The server uses a short idle sleep instead of consuming a core. The
measurements below are historical baselines, not a latency guarantee for this
revision.
UDS remains necessary for bootstrap, `SO_PEERCRED`, memfd/eventfd passing, and
process-death detection.

The Cache Manager process endpoint supports `Ping`, `QueryBundle`,
stale-session fencing, and `Shutdown`. `ChannelClient` obtains the service
identity, an exclusive arena slot, a client token, the arena memfd, and a
notification eventfd through a mode-0600 Unix socket. `SO_PEERCRED` restricts the
bootstrap to the Cache Manager's uid. Each request has an odd generation and each
response advances it by one; reconnecting to a reused slot starts beyond the
prior generation, so delayed commands cannot target a new occupant. The memfd
is sealed against growth and shrinking. The eventfd wakes clients when an
asynchronous restore reaches a terminal state; ordinary control responses still
arrive through iceoryx2's request/response channel.

`QueryBundle` has a framework-neutral binary schema for instance identity,
request identity, hashes, group, query mode, hit positions, and the opaque
lease. Iceoryx2 dispatches to the core query function.
`Publish` and `Release` use the same authenticated descriptor session, so GPU
page metadata can be submitted and query leases can complete their lifecycle
without gRPC. Publish now retains its iceoryx2 reply handle while the core save
runs on Tokio. The dispatcher can serve other requests during D2H, but the
caller still waits: success means D2H copies have completed and host
publication has been queued. A later query observes the blocks after the write
pipeline seals them. `Restore` submits
the existing in-process GPU load, returns an operation ID, signals its session's
eventfd at terminal completion, and is consumed through a follow-up poll. Python
exposes both non-blocking `restore_submit`/`restore_poll` plus the notification
fd and a synchronous `restore` convenience wrapper.
Both adapters use these operations through their same-host Cache Manager; KV
payload bytes do not travel through the descriptor arena. The adapter exposes
one cache API:
scheduler Query/Release and worker Publish/Restore use the process channel.
Lifecycle calls use the
persistent bootstrap UDS. Restore completion uses the session eventfd with bounded fallback
polling.

For one Cache Manager, the adapter derives `/tmp/orbitkv-<addr-port>.sock` unless
`orbitkv.bootstrap_socket` is set. A scheduler querying multiple TP shards
uses the socket derived from each shard endpoint; custom paths can be supplied
through `orbitkv.tp_shard_bootstrap_sockets`. In today's centralized vLLM
scheduler topology this requires all configured TP shards to be on the scheduler
host. Cross-host TP sharding needs a future node-local query fan-out path.
`orbitkv.wait_for_full_prefix` is supported locally. A query is polled once on
the dispatcher for resident hits; any pending future continues on Tokio and
returns `Loading`. Channel ABI 5 separates query submission from ticket polling.
The query schema carries an explicit warmup flag: it prepares pages without a
restore lease, skips on warmup-budget pressure and is retired without polling.
See [queued warming](queued-warming.md); the previous ABI is not retained.
An operation has a monotonically increasing ID within its authenticated session,
and a nonzero revision. A newer revision can replace hashes or wait policy while
keeping its instance, request, and group; old polls and cancels cannot consume or
cancel the replacement. Unknown or retired polls never submit new work. The
transport's session epoch rejects messages from an earlier Manager lifetime.
Both native clients and the Manager must be rebuilt together.

The Python Cache Manager client owns tickets for both engine adapters. It submits
once, polls without resending hashes, and retires the ticket after a terminal
result. Operation-capacity pressure explicitly reports unadmitted `Loading`,
which retries with a fresh ticket. A submitted operation can wait for its
[byte budget](server.md#query-ownership-budgets) before touching cache pages.
Outstanding operations are bounded to 128 per session and 1024 globally and
expire after 60 seconds. Cancellation, disconnect,
and expiration revoke result ownership; submitted backing reads drain while
retaining their operation permits. Their completion admits or discards cache
blocks and drops an undelivered lease without another poll. An expired
operation leaves a bounded tombstone so a late poll reports timeout. Delivered
leases retain their byte reservations through GPU completion; session teardown
also releases delivered leases which were never consumed. Publish's reply is sent only after D2H completes, when the
framework may reuse its source pages. This removes the shared dispatcher wait
without making the caller's save completion asynchronous. New measurements of
this revision are still required; the latency table below predates it.
Publish requires a Cache Manager pidfd before submission. Once submitted, the client
waits beyond the ordinary IPC timeout until it receives a reply or the Cache Manager
process exits; ambiguous receive failures also keep its save-source pages pinned
until process death. A live Cache Manager stalled forever will keep the vLLM save
worker waiting. A watchdog/recovery policy is still needed for that availability
case.

Publish batches are split to the slot capacity negotiated at bootstrap. Each
chunk carries the same page range across its layers; the client returns only
after every chunk has completed D2H. This supports long-context, many-layer
models with the default 64 KiB slot without silently skipping oversized saves.

Restore destinations remain owned by the engine until the manager confirms a
terminal result. A lost submission acknowledgement, failed completion poll, or
deadline does not cancel CUDA writes. vLLM stops the engine step in those cases
without reporting reusable blocks; SGLang fails its layer wait and completion
observer without acknowledging the destination pages. A confirmed, drained
failure can still use vLLM's single-cache-group recomputation path.

Both H2D and D2H workers synchronize submitted stream work even when the backend
returns an error partway through a batch. If CUDA cannot establish completion,
the manager terminates instead of publishing a terminal result and recycling
potentially active memory. This is a transfer lifetime fence; allocator-owned
per-page generations and graceful cancellation remain separate work.

The bootstrap protocol is version 2. After FD exchange, its UDS also carries
versioned, epoch-checked lifecycle frames with a 16 MiB metadata limit. These
frames reuse the registration protobuf schema without a gRPC channel or HTTP/2.
Malformed frames close the connection; application errors preserve framing.
Standalone mode starts no gRPC listener. Distributed etcd/placement configuration enables the
peer-only transfer control listener automatically.
Client and Cache Manager need to be upgraded together.

UDS and HTTP cleanup in the Cache Manager share lifecycle serialization. Cleanup
closes GPU queues to new submissions, waits for both streams to drain, then
releases imported mappings. A failed drain retains mappings instead of freeing
memory that may still be in use. Session replacement and disconnect cleanup use
the same instance lock, so stale disconnects cannot remove the new session.
SIGTERM, Ctrl+C, and control-plane shutdown close sessions and drain registered
workers before service exit.

## Measured process-channel baseline

Measurements were collected on one H20 node with two Linux processes and a
64-byte request/response descriptor, before bootstrap version 2. They are
engineering evidence for the transport choice, not measurements of this
revision or end-to-end serving results.

| Path | Mean RTT | p50 | p95 | p99 | Sequential RTT/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| iceoryx2 two-process 64-byte ping | 4.052 us | 3.877 us | 4.585 us | 9.263 us | 246.8k |
| real Python/PyO3 local `QueryBundle` | 106.861 us | 107.404 us | 114.390 us | 120.635 us | 9,358 |
| real Python/PyO3 gRPC `QueryBundle` | 489.575 us | 485.715 us | 547.946 us | 635.653 us | 2,043 |

The real local path is about 4.58x faster than gRPC by both mean RTT and
sequential throughput. Its roughly 107 us RTT is still far above the 4 us
iceoryx2 substrate, so the next local optimization target is descriptor
encode/decode, Python/PyO3 crossings, and the Cache Manager's 50 us idle poll, not a
replacement IPC library. The benchmark is sequential because the scheduler
needs one answer before committing a recovery boundary.

The iceoryx2 result can be reproduced with the two binaries documented in
[`crates/orbitkv-channel/README.md`](../crates/orbitkv-channel/README.md).

## Current local validation

With the pinned vLLM `0.29.0`, a single-node H20 run passed the applicable
pure-attention E2E recovery gates after an inference-process restart, comparing
ordered generated text to vLLM native prefix caching. SGLang `0.5.20` passed
its direct GPU-page restore gate after a radix-cache flush. These are local
correctness checks, not multi-host reliability or throughput measurements.
The SGLang multi-rank and cross-host TP paths still need qualification. See
[Python test gates](../python/tests/README.md) and the [roadmap](roadmap.md).

## Mooncake findings

Mooncake separates three responsibilities:

1. `TransferEngine` exposes Segment, Buffer, and BatchTransfer abstractions.
2. Transfer metadata publishes protocol, topology, registered regions, and
   connection endpoints.
3. Mooncake Store Master owns generic object allocation and the
   `PROCESSING -> COMPLETE` replica lifecycle.

The useful transport mechanisms are:

- topology-aware path selection across CPU, GPU, and multiple NICs;
- request slicing and multi-rail bandwidth aggregation;
- on-demand endpoint creation with a bounded SIEVE endpoint pool;
- two-phase endpoint retirement after outstanding work requests drain;
- retry on alternate local or peer rails;
- SHM, TCP, RDMA, EFA, NVMe-oF, and accelerator-specific backends behind one
  BatchTransfer interface.

`P2PHANDSHAKE` does not mean that Mooncake is control-plane-free. It starts a
custom TCP socket daemon, exchanges JSON metadata and QP/MR information, and
then uses the selected data transport. Larger deployments may instead publish
Segment metadata through etcd, Redis, or HTTP.

OrbitKV pins Mooncake `v0.3.13.post1` at
`719735896c86b56fabec6cf3e825fb2ea640597a` and builds its shared Transfer
Engine through `orbitkv-mooncake-sys`. Native
loading first checks `ORBITKV_MOONCAKE_LIB_DIR`, then the executable or Python
extension directory, then the local `.orbitkv/mooncake/{cuda|cpu}/lib` build cache. Wheels
bundle `libtransfer_engine.so`, `libmooncake_common.so`, and `libasio.so`; system
RDMA/CUDA libraries remain deployment prerequisites.

## What OrbitKV reuses

`orbitkv-transfer::TransferEngine` is a narrow wrapper over the pinned upstream
Mooncake C ABI. There is no runtime backend selector and no OrbitKV-owned verbs
implementation. Authorized plans are lowered directly to Mooncake Segment
addresses, BatchTransfer operations, and notifications.

OrbitKV does not adopt Mooncake Store Master as its semantic authority. Today
the cache engine handles versioned model/storage keys and leases above Mooncake.
The common recovery plan additionally includes:

- validated `StateDescriptor`, `StateBundle`, and `RecoveryContract` evidence;
- replica selection and restore-versus-recompute planning;
- query leases and generation validation;
- semantic and execution frontiers;
- publication only after every required component is complete.

The current Catalog advertises candidate owners, not durable KV data or
permanent raw addresses/rkeys. A selected source authorizes and pins current
bytes before transfer. Fixed shards and owner replay are implemented; directory replication is future work.

## Remote operation choice

- current demand-driven remote cache reuse uses Mooncake READ into destination
  pinned memory before restoring into framework HBM;
- the experimental vLLM P/D connector uses Mooncake WRITE into the decode
  worker's allocated GPU pages and waits for completion notification;
- proactive replica placement, retry across replicas, and bundle-aware
  publication are planning targets, not current guarantees.

The target recovery contract must reject incomplete or incompatible bundles;
the Manager wire protocol does not yet enforce `StateBundle` completeness.
SGLang now validates compiled prefix/window/checkpoint requirements before
restore in its adapter, using the shared Rust contract and leased group positions;
see [hybrid recovery](hybrid-recovery.md). This does not add generation-qualified
page references to the transfer protocol.

## Remaining work

The hot path now uses UDS + iceoryx2 for both adapters. The next transport
work is an explicit GPU-region registration protocol (replacing the Python
CUDA IPC wrapper pickle), page-generation validation, multi-node inventory
replay, and Mooncake RDMA failure/retry qualification. Peer gRPC remains only
for remote transfer authorization and lease release in the current design.
