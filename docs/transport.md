# OrbitKV transport architecture

This document records the transport decision after validating the current
vLLM path, benchmarking local IPC, and reviewing Mooncake at commit
`ffe01351`. It separates local control, remote data movement, and replica
discovery so none of them becomes an accidental second source of KV truth.

## Decision

| Boundary | Control | Payload | Status |
| --- | --- | --- | --- |
| inference process to local sidecar | iceoryx2 request/response | CUDA IPC or shared host pages | lifecycle and QueryBundle integrated |
| local bootstrap and region registration | Unix socket with credential and file-descriptor passing | memfd handles only | descriptor bootstrap implemented; page-region registration planned |
| sidecar to sidecar | Mooncake P2P handshake | Mooncake BatchTransfer over RDMA/TCP | upstream Mooncake provider integrated |
| replica directory | soft-state network API | no KV bytes | current MetaServer, redesign planned |
| administration | HTTP or compatibility gRPC | no KV bytes | existing |

The existing gRPC data path remains the default compatibility and correctness
baseline. The vLLM adapter can opt into `orbitkv-local` for its hot data
operations while keeping registration, health, session watching, and
unregistration on gRPC. SGLang has not yet made this cutover.

## Local IPC

`orbitkv-local` uses iceoryx2 `0.10.0`. The workspace minimum Rust version is
therefore `1.89`. Its first ABI is a fixed 64-byte message carrying:

- protocol magic and ABI version;
- command or status code;
- request identity and sidecar session epoch;
- offset, length, and generation of a descriptor in a separately registered
  arena;
- two opcode-specific scalar fields.

The initial command vocabulary is `QueryBundle`, `Restore`, `Publish`,
`Release`, and lifecycle probes. Variable-length hashes and page arrays do not
live in the message. KV bytes never live in the message.

One inference process gets one iceoryx2 client endpoint. The sidecar exclusively
creates and owns the server endpoint; clients only open it. The endpoint uses
iceoryx2's thread-safe IPC service because the server owns it on a dedicated
control thread. Calls spin only for a bounded number of iterations and then
yield. The lifecycle-only server uses a short idle sleep instead of consuming a
core; the data-command phase must add event-driven wakeup before claiming the
measured busy-poll latency.
UDS remains necessary for bootstrap, `SO_PEERCRED`, memfd/eventfd passing, and
process-death detection.

The real `orbitkv-server` endpoint now supports `Ping`, `QueryBundle`,
stale-session fencing, and `Shutdown`. `LocalQueryClient` obtains the service
identity, an exclusive arena slot, a client token, the arena memfd, and a
notification eventfd through a mode-0600 Unix socket. `SO_PEERCRED` restricts the
bootstrap to the sidecar's uid. Each request has an odd generation and each
response advances it by one; reconnecting to a reused slot starts beyond the
prior generation, so delayed commands cannot target a new occupant. The memfd
is sealed against growth and shrinking. The eventfd wakes clients when an
asynchronous restore reaches a terminal state; ordinary control responses still
arrive through iceoryx2's request/response channel.

`QueryBundle` has a framework-neutral binary schema for instance identity,
request identity, hashes, group, query mode, hit positions, and the opaque
lease. Both gRPC and iceoryx2 dispatch through the same core query function.
`Publish` and `Release` use the same authenticated descriptor session, so local
GPU page metadata can be submitted and query leases can complete their
lifecycle without gRPC. Publish retains the existing asynchronous core save
semantics: success means the validated GPU copy job was accepted, while a later
query observes it after the write pipeline seals the blocks. `Restore` submits
the existing in-process GPU load, returns an operation ID, signals its session's
eventfd at terminal completion, and is consumed through a follow-up poll. Python
exposes both non-blocking `restore_submit`/`restore_poll` plus the notification
fd and a synchronous `restore` convenience wrapper. The
vLLM selects these operations with `orbitkv.local_data=true`; KV payload bytes
do not travel through the descriptor arena. The adapter wraps both transports
behind one data-plane interface: scheduler Query/Release and worker
Publish/Restore use the selected transport, while lifecycle calls stay on
gRPC. Local restore completion uses the session eventfd with bounded fallback
polling, replacing `PyLoadState` only on the opt-in path.

For one sidecar, the adapter derives `/tmp/orbitkv-<grpc-port>.sock` unless
`orbitkv.local_bootstrap_socket` is set. A scheduler querying multiple TP shards
must receive one locally reachable socket per shard through
`orbitkv.tp_shard_bootstrap_sockets`; using one socket for multiple sidecars is
rejected. In today's centralized vLLM scheduler topology this makes the full
local path a same-host feature. Cross-host TP shards keep the gRPC Query/Release
path until query fan-out is delegated to node-local agents.
`orbitkv.wait_for_full_prefix` is also rejected in local mode because the
current local dispatcher is serial and a blocking remote fetch would stall
unrelated Publish/Restore calls. Supporting that combination requires an
asynchronous QueryBundle operation, analogous to Restore.

## Measured local-control baseline

Measurements were collected on one H20 node with two Linux processes and a
64-byte request/response descriptor. They are engineering evidence for the
transport choice, not end-to-end serving results.

| Path | Mean RTT | p50 | p95 | p99 | Sequential RTT/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| iceoryx2 two-process 64-byte ping | 4.052 us | 3.877 us | 4.585 us | 9.263 us | 246.8k |
| real Python/PyO3 local `QueryBundle` | 106.861 us | 107.404 us | 114.390 us | 120.635 us | 9,358 |
| real Python/PyO3 gRPC `QueryBundle` | 489.575 us | 485.715 us | 547.946 us | 635.653 us | 2,043 |

The real local path is about 4.58x faster than gRPC by both mean RTT and
sequential throughput. Its roughly 107 us RTT is still far above the 4 us
iceoryx2 substrate, so the next local optimization target is descriptor
encode/decode, Python/PyO3 crossings, and the sidecar's 50 us idle poll, not a
replacement IPC library. The benchmark is sequential because the scheduler
needs one answer before committing a recovery boundary.

The iceoryx2 result can be reproduced with the two binaries documented in
[`crates/orbitkv-local/README.md`](../crates/orbitkv-local/README.md).

## vLLM path evidence

Both the compatibility gRPC path and the opt-in local data path have been
exercised with vLLM `0.26.0`, PyTorch `2.11.0`, CUDA 13,
`Qwen2.5-0.5B-Instruct`, and a separate OrbitKV sidecar. The local run selected
`transport=local` in both vLLM processes and:

- registered 24 attention layers through CUDA IPC;
- saved 74 blocks / 14.5 MB;
- hit 40 blocks and loaded 7.9 MB after a vLLM process restart;
- exercised exact and partial prefixes;
- reported no connector/server data-path failures and no KV load failures.

One strict text-equality assertion still differs between full prefill and warm
prefix reuse. An independent vLLM native prefix-cache control reproduced the
same divergence at the same output position, while a no-prefix-cache control
was stable. The local run therefore passes transport selection, cache activity,
and failure-counter gates, but not the repository's strict full-text gate. This
is evidence of a vLLM execution-path numerical difference, not evidence that
OrbitKV corrupted the transferred bytes. Future correctness qualification must
compare equivalent prefix-reuse paths and add logits or top-1-margin checks
where exact text is unstable.

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

OrbitKV pins Mooncake at `ffe013517eaafa8f33e5e0ee034fd6b8f5561e92` and
builds its shared Transfer Engine through `orbitkv-mooncake-provider`. Native
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

1. Keep gRPC as a tested compatibility transport.
2. Land the versioned `orbitkv-local` ABI, sidecar lifecycle endpoint, Python
   client, and cross-process tests.
3. Add UDS bootstrap and a shared descriptor arena. (complete for control
   descriptors; framework-owned page registration remains)
4. Move `QueryBundle` to iceoryx2. (complete; vLLM opt-in, SGLang pending)
5. Move `Publish` and `Release`. (complete; vLLM opt-in, SGLang pending)
6. Move `Restore` and remove per-load shared-memory status objects. (complete
   for the opt-in vLLM local path; gRPC compatibility and SGLang pending)
7. Keep the pinned Mooncake provider as the only remote transfer backend.
8. Qualify Mooncake RDMA/GPUDirect against the transfer-plan and P/D gates.
9. Keep network control RPCs for authorization and leases; the custom verbs
   handshake RPC has been removed.
