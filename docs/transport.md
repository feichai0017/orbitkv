# OrbitKV transport architecture

This document records the transport decision after validating the current
vLLM path, benchmarking local IPC, and reviewing Mooncake at commit
`ffe01351`. It separates local control, remote data movement, and replica
discovery so none of them becomes an accidental second source of KV truth.

## Decision

| Boundary | Control | Payload | Status |
| --- | --- | --- | --- |
| inference process to local sidecar | iceoryx2 request/response | CUDA IPC or shared host pages | lifecycle endpoint integrated; data commands planned |
| local bootstrap and region registration | Unix socket with credential and file-descriptor passing | memfd handles only | planned |
| sidecar to sidecar | backend-specific session control | RDMA READ/WRITE | native RDMA exists; Mooncake adapter planned |
| replica directory | soft-state network API | no KV bytes | current MetaServer, redesign planned |
| administration | HTTP or compatibility gRPC | no KV bytes | existing |

The existing gRPC connector remains the compatibility and correctness baseline
until the same vLLM and SGLang gates pass over `orbitkv-local`.

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

The real `orbitkv-server` lifecycle endpoint and Python `LocalControlClient` now
support `Ping`, stale-session fencing, and `Shutdown`. `QueryBundle`, `Restore`,
`Publish`, and `Release` return `Invalid` rather than silently falling back or
claiming an incomplete implementation. The existing vLLM gRPC data path is
unchanged.

## Measured local-control baseline

Measurements were collected on one H20 node with two Linux processes and a
64-byte request/response descriptor. They are engineering evidence for the
transport choice, not end-to-end serving results.

| Path | Mean RTT | p50 | p99 | Sequential RTT/s |
| --- | ---: | ---: | ---: | ---: |
| iceoryx2 0.10, dedicated cores, busy poll | 1.51 us | 1.26 us | 1.47 us | 664k |
| iceoryx2 0.10, cooperative yield polling | 1.97 us | 1.72 us | 2.21 us | 506k |
| Unix stream, fixed 64-byte echo | 5.97 us | 5.68 us | 11.66 us | 168k |
| Tonic TCP, native Rust client, release sidecar | 79.69 us | 75.52 us | 103.43 us | 12.5k |
| Python to PyO3 to Tonic TCP, release sidecar | 92.75 us | 88.35 us | 114.86 us | 10.8k |

The benchmark is deliberately sequential because the scheduler needs one
answer before committing a recovery boundary. Infinite busy polling is not a
supported production mode: pinning both peers to one CPU demonstrated
starvation.

The iceoryx2 result can be reproduced with the two binaries documented in
[`crates/orbitkv-local/README.md`](../crates/orbitkv-local/README.md).

## vLLM path evidence

The existing gRPC path was exercised with vLLM `0.26.0`, PyTorch `2.11.0`
CUDA 13, `Qwen2.5-0.5B-Instruct`, and a separate OrbitKV sidecar. The run:

- registered 24 attention layers through CUDA IPC;
- saved 74 blocks / 14.5 MB;
- hit 40 blocks and loaded 7.9 MB after a vLLM process restart;
- exercised exact and partial prefixes;
- reported no connector/server RPC failures and no KV load failures.

One strict text-equality assertion differed between full prefill and warm
prefix reuse. An independent vLLM native prefix-cache control reproduced the
same divergence at the same output position, while a no-prefix-cache control
was stable. This is evidence of a vLLM execution-path numerical difference,
not evidence that OrbitKV corrupted the transferred bytes. Future correctness
qualification must compare equivalent prefix-reuse paths and add logits or
top-1-margin checks where exact text is unstable.

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

## What OrbitKV reuses

`orbitkv-transfer::RemoteMover` is the data-movement boundary. The existing
Rust verbs engine implements it as `NativeRdma`. A future optional
`MooncakeMover` will wrap Mooncake's C ABI/Rust binding and use its Segment
offsets, BatchTransfer, multi-rail selection, and failover.

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

- demand-driven remote cache reuse uses RDMA READ; the consumer controls
  destination allocation and can retry another replica;
- P/D transfer and proactive replication use RDMA WRITE; the destination first
  reserves pages and publishes them only after completion;
- final WRITE completion uses WRITE_WITH_IMM or an explicit notification;
- failures invalidate only the physical plan and fall back to another replica
  or recomputation.

No remote backend may claim a bundle is available before the bundle recovery
contract is complete.

## Migration sequence

1. Keep gRPC as a tested compatibility transport.
2. Land the versioned `orbitkv-local` ABI, sidecar lifecycle endpoint, Python
   client, and cross-process tests.
3. Add UDS bootstrap and a shared descriptor arena.
4. Move `QueryBundle` and completions to iceoryx2.
5. Move `Restore`, `Publish`, and `Release`; remove per-load shared-memory
   status objects.
6. Implement `MooncakeMover` behind an optional build/runtime feature.
7. Qualify native RDMA and Mooncake against the same transfer plan tests.
8. Remove cross-node gRPC data-path RPCs only after equivalent lease, fencing,
   retry, and observability gates pass.
