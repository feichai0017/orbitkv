# OrbitKV transport architecture

This document records the transport decision after validating the current
vLLM path, benchmarking local IPC, and reviewing Mooncake `v0.3.13.post1`. It
separates local control, remote data movement, and replica
discovery so none of them becomes an accidental second source of KV truth.

## Decision

| Boundary | Control | Payload | Status |
| --- | --- | --- | --- |
| inference process to local Cache Manager | iceoryx2 request/response and UDS lifecycle | CUDA IPC or shared host pages | integrated |
| local bootstrap and region registration | Unix socket with credential and file-descriptor passing | memfd handles only | descriptor bootstrap implemented; page-region registration planned |
| Cache Manager to Cache Manager | Mooncake P2P handshake | Mooncake BatchTransfer over RDMA/TCP | stable Mooncake runtime integrated |
| replica directory | soft-state network API | no KV bytes | current MetaServer, redesign planned |
| administration | HTTP | no KV bytes | existing |

The vLLM adapter requires `orbitkv-local` for hot data operations. Every
inference process must connect to a Cache Manager on its own host; a missing
Unix socket fails fast.
Registration, health,
session watching, and unregistration use the authenticated bootstrap UDS. SGLang has
not yet made this cutover.

## Local IPC

`orbitkv-local` uses iceoryx2 `0.10.0`. The workspace minimum Rust version is
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
yield. The lifecycle-only server uses a short idle sleep instead of consuming a
core; the data-command phase must add event-driven wakeup before claiming the
measured busy-poll latency.
UDS remains necessary for bootstrap, `SO_PEERCRED`, memfd/eventfd passing, and
process-death detection.

The Cache Manager process endpoint supports `Ping`, `QueryBundle`,
stale-session fencing, and `Shutdown`. `LocalQueryClient` obtains the service
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
fd and a synchronous `restore` convenience wrapper. The
vLLM uses these operations on a same-host deployment; KV payload bytes
do not travel through the descriptor arena. The adapter exposes one cache API:
scheduler Query/Release and worker Publish/Restore use the local endpoint.
Lifecycle calls use the
persistent bootstrap UDS. Local restore completion uses the session eventfd with bounded fallback
polling.

For one Cache Manager, the adapter derives `/tmp/orbitkv-<addr-port>.sock` unless
`orbitkv.local_bootstrap_socket` is set. A scheduler querying multiple TP shards
uses the socket derived from each shard endpoint; custom paths can be supplied
through `orbitkv.tp_shard_bootstrap_sockets`. In today's centralized vLLM
scheduler topology this requires all configured TP shards to be on the scheduler
host. Cross-host TP sharding needs a future node-local query fan-out path.
`orbitkv.wait_for_full_prefix` is supported locally. A query is polled once on
the dispatcher for resident hits; any pending future continues on Tokio and
returns `Loading`. Polling the same instance/request/group retrieves its result;
changing arguments while pending is rejected. Outstanding queries are bounded
to 128 per session and expire after 60 seconds. Dropping an undelivered result
releases its lease. Publish's reply is sent only after D2H completes, when the
framework may reuse its source pages. This removes the shared dispatcher wait
without making the caller's save completion asynchronous. New measurements of
this revision are still required; the latency table below predates it.
Publish requires a Cache Manager pidfd before submission. Once submitted, the client
waits beyond the ordinary IPC timeout until it receives a reply or the Cache Manager
process exits; ambiguous receive failures also keep its save-source pages pinned
until process death. A live Cache Manager stalled forever will keep the vLLM save
worker waiting. A watchdog/recovery policy is still needed for that availability
case.

The bootstrap protocol is version 2. After FD exchange, its UDS also carries
versioned, epoch-checked lifecycle frames with a 16 MiB metadata limit. These
frames reuse the registration protobuf schema without a gRPC channel or HTTP/2.
Malformed frames close the connection; application errors preserve framing.
Standalone mode starts no gRPC listener. `--metaserver-addr` enables the
peer-only transfer control listener automatically.
Client and Cache Manager need to be upgraded together.

UDS and HTTP cleanup in the Cache Manager share lifecycle serialization. Cleanup
closes GPU queues to new submissions, waits for both streams to drain, then
releases imported mappings. A failed drain retains mappings instead of freeing
memory that may still be in use. Session replacement and disconnect cleanup use
the same instance lock, so stale disconnects cannot remove the new session.
SIGTERM, Ctrl+C, and control-plane shutdown close sessions and drain registered
workers before service exit.

## Measured local-control baseline

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
[`crates/orbitkv-local/README.md`](../crates/orbitkv-local/README.md).

## Historical validation (before inference gRPC removal)

The earlier UDS lifecycle revision passed the Python unit gate, Rust cache/control
unit tests, the real two-process iceoryx2 test, and workspace Clippy.
On one H20, the 36 Rust cache integration tests pass, including all 11 SSD
roundtrips with io_uring enabled. The 8 Python GPU integration cases also pass:
local health/register/publish/query/restore/release with gRPC disabled, retained
gRPC client contracts, and SIGKILL-triggered CUDA IPC cleanup over both UDS and
gRPC. Those gRPC inference tests were removed with the transport. The Cache
Manager and Python extension were built together; test helpers can
pin the Cache Manager artifact through `ORBITKV_CACHE_MANAGER_BINARY`.

On one H20 with vLLM `0.26.0` and `Qwen2.5-0.5B-Instruct`, the revised
local-only E2E passed all six applicable gates; the hybrid-only gate was
skipped. It compares the same ordered prompt plan with native vLLM prefix
caching and requires exact text equality at every step. Native `long_warm`
records a prefix-cache hit, and OrbitKV `long_warm` loads saved KV after the
vLLM process restarts. The Cache Manager ran without gRPC. A prior cold-prefill
baseline had diverged at `long_warm`; the matched native-prefix baseline
removed that comparison error. This validates the pure-attention local path,
not hybrid models or multi-GPU deployments.
The same six applicable gates pass with the official vLLM `0.29.0` release,
PyTorch `2.13.0+cu130`, and the same model on the H20. That run saved 74
blocks / 14.5 MB and hit 40 blocks / 7.9 MB, again with no gRPC listener.
With `Qwen3.5-0.8B` on the same vLLM release, all seven gates pass, including
the same-process HMA restore gate and exact agreement with the native
prefix-cache control across the 12-step plan. It saved 25 blocks / 167.1 MB
and hit 14 blocks / 234.0 MB. The HMA scheduler boundary-state hand-off is
present in this release; vLLM `0.26.0` lacks it and the adapter rejects that
hybrid configuration at startup.

## vLLM path evidence

Before the UDS lifecycle migration, the compatibility gRPC path and the opt-in
local data path were exercised with vLLM `0.26.0`, PyTorch `2.11.0`, CUDA 13,
`Qwen2.5-0.5B-Instruct`, and a separate OrbitKV Cache Manager. The local run selected
`transport=local` in both vLLM processes and:

- registered 24 attention layers through CUDA IPC;
- saved 74 blocks / 14.5 MB;
- hit 40 blocks and loaded 7.9 MB after a vLLM process restart;
- exercised exact and partial prefixes;
- reported no connector/server data-path failures and no KV load failures.

The former full-prefill versus warm-prefix comparison differed in one strict
text assertion. An independent native prefix-cache control reproduced the
same divergence at the same output position. The revised matched-prefix
comparison passes on this setup, as recorded above.

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

OrbitKV does not adopt Mooncake Store Master as its semantic authority. The
following remain above every mover:

- `StateKey`, `StateBundle`, and `RecoveryContract`;
- replica selection and restore-versus-recompute planning;
- query leases and generation validation;
- semantic and execution frontiers;
- publication only after every required component is complete.

The global directory stores candidate node/tier/session information. It does
not store permanent raw addresses or rkeys. A selected source issues a short
lived transfer capability after validating a lease.

## Remote operation choice

- demand-driven remote cache reuse uses Mooncake READ; the consumer controls
  destination allocation and can retry another replica;
- P/D transfer and proactive replication use Mooncake WRITE; the destination first
  reserves pages and publishes them only after completion;
- final WRITE completion uses a Mooncake notification after all batches complete;
- failures invalidate only the physical plan and fall back to another replica
  or recomputation.

No remote backend may claim a bundle is available before the bundle recovery
contract is complete.

## Migration sequence

1. Land the versioned `orbitkv-local` ABI, Cache Manager lifecycle endpoint,
   Python client, and cross-process tests. (complete)
2. Add UDS bootstrap and a shared descriptor arena. (complete for control
   descriptors; framework-owned page registration remains)
3. Move `QueryBundle`, `Publish`, `Release`, and `Restore` to iceoryx2; remove
   inference gRPC and per-load shared-memory status objects. (complete for vLLM)
4. Keep the pinned stable Mooncake runtime as the remote transfer backend.
5. Qualify Mooncake RDMA/GPUDirect against the transfer-plan and P/D gates.
6. Keep peer control RPCs for transfer authorization and leases.
