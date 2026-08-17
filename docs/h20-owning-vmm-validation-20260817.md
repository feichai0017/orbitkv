# H20 Owning Manager and CUDA VMM Validation — 2026-08-17

## Scope

This record qualifies two independent OrbitKV milestones:

1. a Rust owning control plane that authorizes SGLang SWA page reclamation
   through two-phase retirement certificates;
2. an isolated NVIDIA CUDA VMM physical-slot primitive that remaps fresh
   physical backing at a stable virtual address.

The VMM primitive is not yet used as SGLang tensor storage. The owning SGLang
adapter still executes physical frees through SGLang's existing paged
allocator.

The owner now accepts both the legacy Full/SWA syntax and a declarative
`orbitkv.retention-ir.v1` relation. The H20 owner path was smoke-tested with
the declarative relation `query_position - key_position < 1024`; it inferred
the same 1,024-token window, 65-cell periodic layout, plan fingerprint, and
SGLang policy as the legacy frontend.

## Owning protocol

The owning adapter executes:

```text
Rust Retention Plan
    -> page-aligned retirement certificate
    -> SGLang physical free group
    -> atomic Rust certificate commit
    -> committed semantic frontier advances
```

Each certificate contains:

- a stable plan fingerprint;
- request and state-class identity;
- exact token range;
- sliding-window semantic proof;
- non-overlap scheduler execution barrier.

The adapter fails closed:

- a plan/frontier mismatch aborts reclamation;
- unsupported radix, overlap, and speculative paths are rejected;
- a physical-free exception does not commit the certificate;
- batch commit preflights every certificate before mutating Rust state.

## H20 owner result

The workload is the same dummy-weight 62-layer hybrid fixture used by the
fixed-budget admission experiment:

- 10 Full layers and 52 SWA layers;
- eight requests;
- 6,000 prompt tokens plus 32 decode tokens each;
- 4.608 GiB reported KV budget;
- page size 16 and SWA window 1,024;
- radix cache, speculative decoding, overlap scheduling, and CUDA Graph
  disabled.

Three fresh-process Policy/Owner pairs produced:

| Pair | Policy | Owner | Owner / Policy |
| ---: | ---: | ---: | ---: |
| 0 | 6.5094 s | 6.4361 s | 0.9887x |
| 1 | 6.2953 s | 6.3524 s | 1.0091x |
| 2 | 6.3406 s | 6.3800 s | 1.0062x |

Median Owner/Policy ratio: **1.0062x**, or **+0.62%**.

Both modes reported:

- Full token capacity: 49,888;
- SWA token capacity: 36,848;
- no request retractions;
- output digest:
  `76cf04ef40736dcfe5761952139ce6c1e73c99b41b1a0a7cf05ccead8be3333d`.

A traced H20 smoke run emitted and committed a certificate for `[0, 1024)`;
the certificate fingerprint matched the loaded Rust policy.

This shows that ownership and proof-carrying reclamation can preserve the
capacity benefit without a visible large control-plane regression. Three pairs
are not enough to claim a tight universal overhead bound.

## In-process owner ABI

The first prototype used a persistent JSONL subprocess. The current adapter
uses `orbitkv-ffi`, a versioned C ABI loaded in the SGLang scheduler process.
The ABI exposes fixed-layout functions for:

```text
create
plan_chunk_reclamation
commit_reclamations
release_request
stats
destroy
```

`OrbitKvCertificateV1` carries the certificate id, token range, semantic
proof fields, execution epoch, and binary SHA-256 plan fingerprint without
JSON serialization. Panics are contained at the ABI boundary.

A release-build host transport benchmark ran five trials of 5,000
`plan + commit` cycles:

| Transport | Median |
| --- | ---: |
| JSONL sidecar | 31.03 µs |
| In-process FFI | 7.29 µs |

Median control-plane speedup: **4.26x**.

Six fresh-process H20 pairs alternated transport execution order:

```text
FFI / sidecar median ratio: 1.0030x
FFI / sidecar mean ratio:   1.0008x
```

All pairs reported Full capacity 49,888, SWA capacity 36,848, zero
retractions, and the same output digest. This establishes that FFI removes
most transport overhead without changing serving semantics; it does not prove
a measurable end-to-end serving speedup for this GPU-dominated workload.

## CUDA VMM result

The `orbitkv-cuda` crate uses NVIDIA CUDA Driver API VMM operations:

```text
cuMemAddressReserve
cuMemCreate
cuMemMap
cuMemSetAccess
cuMemUnmap
cuMemRelease
cuMemAddressFree
```

On the NVIDIA H20:

| Property | Value |
| --- | ---: |
| Compute capability | 9.0 |
| VMM supported | yes |
| Minimum granularity | 2 MiB |
| Recommended granularity | 2 MiB |
| Requested slot | 1 MiB |
| Reserved/mapped slot | 2 MiB |
| Fresh backing remaps | 64 |
| Stable VA checks | passed |
| Data-pattern checks | passed |
| Physical backing replacement | passed |
| GPU memory before / after | 0 MiB / 0 MiB |

The 2 MiB minimum granularity is operationally important. OrbitKV's physical
optimizer now chooses VMM only when:

```text
stable virtual address is required
and
rounded VMM bytes / logical bytes <= configured threshold
```

Small regions fall back to the paged backend instead of paying large rounding
amplification.

## Generation-aware VMM lifecycle

The pinned SGLang revision already contains a post-capture CUDA VMM backing
arena. That implementation reserves stable VA and monotonically commits the
final KV span after graph capture. OrbitKV does not claim stable VA reservation
itself as a new contribution.

OrbitKV's distinct physical contract is:

```text
compiled logical cell + deterministic cycle
    -> generation-checked physical binding
    -> CUDA event execution completion
    -> semantic retirement certificate
    -> generation-matched VMM unmap receipt
    -> manager commit
    -> next generation may reuse the stable VA
```

An H20 lifecycle test used a two-cell `Sliding(W=2)` address program for 64
cycles. It verified:

- 64 CUDA submission events completed;
- 64 device-memory patterns were read back correctly;
- 63 stale-generation handles were rejected after slot reuse;
- 65 physical reclamation receipts were committed, including final request
  release;
- temporal cycle and physical generation progressed consistently;
- each physical slot retained one stable virtual address across generations;
- manager residency and pending events returned to zero;
- GPU memory before and after remained 0 MiB.

This is a closed-loop core-manager/CUDA-backend qualification. It is not yet a
replacement for SGLang's real KV tensor storage.

## Claim boundary

This evidence supports:

- a real Rust-owned reclamation protocol on the tested SGLang path;
- fail-closed two-phase certificate commit;
- low measured prototype control-plane overhead for the tested workload;
- functioning H20 CUDA VMM reserve/map/remap/unmap/release lifecycle;
- stable virtual addresses across fresh physical backing generations;
- generation-aware VMM reclaim/remap driven by manager certificates and CUDA
  event completion;
- cost-aware rejection of VMM for inefficient small regions.

It does not yet support:

- radix/prefix cache, overlap, speculative, cancellation, or CUDA Graph
  qualification for the owning SGLang adapter;
- VMM-backed SGLang KV tensors;
- released-checkpoint qualification of the in-process Rust FFI path;
- multi-GPU multicast or fabric-memory claims;
- released-checkpoint model-quality claims.

Status update on 2026-08-17: the first released-checkpoint qualification was
completed on `openai/gpt-oss-20b`; see
`docs/h20-gpt-oss-20b-real-validation-20260817.md`. VMM-backed SGLang KV
tensors and model-quality evaluation remain outside the qualified boundary.

## Next gate

Connect `CudaVmmSlot` generations to `BlockHandle` and expose a graph-stable
KV descriptor backed by VMM regions. Then enable CUDA Graph and overlap one at
a time while preserving:

```text
PhysicalReuse
    => SemanticDead
    and ExecutionComplete
    and CertificateCommitted
```
