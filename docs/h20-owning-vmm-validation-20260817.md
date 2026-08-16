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

## Owning protocol

The prototype executes:

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

## Claim boundary

This evidence supports:

- a real Rust-owned reclamation protocol on the tested SGLang path;
- fail-closed two-phase certificate commit;
- low measured prototype control-plane overhead for the tested workload;
- functioning H20 CUDA VMM reserve/map/remap/unmap/release lifecycle;
- stable virtual addresses across fresh physical backing generations;
- cost-aware rejection of VMM for inefficient small regions.

It does not yet support:

- radix/prefix cache, overlap, speculative, cancellation, or CUDA Graph
  qualification for the owning SGLang adapter;
- VMM-backed SGLang KV tensors;
- a production in-process Rust FFI path;
- multi-GPU multicast or fabric-memory claims;
- released-checkpoint model-quality claims.

## Next gate

Replace JSONL sidecar calls with an in-process Rust ABI, connect
`CudaVmmSlot` generations to `BlockHandle`, and expose a graph-stable KV
descriptor backed by VMM regions. Then enable CUDA Graph and overlap one at a
time while preserving:

```text
PhysicalReuse
    => SemanticDead
    and ExecutionComplete
    and CertificateCommitted
```
