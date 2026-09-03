# Capability Matrix

This matrix separates implemented source, host verification, device execution,
measured benefit, and production readiness for the current
`core + executor + server` architecture. Historical results qualify only their
recorded source closure.

## Evidence levels

| Level | Meaning |
| --- | --- |
| L1 Compiler | Declarative semantics compile into deterministic plans |
| L2 Host | Rust ownership, lifecycle, lowering, and failure invariants pass host tests |
| L3 Device | The current source executes correct kernels with real stream/event evidence |
| L4 Engine | A released model passes matched end-to-end correctness and lifecycle tests |
| L5 Benefit | Matched measurements pass predefined latency, throughput, memory, or capacity gates |
| L6 Production | Soak, cancellation, pressure, concurrency, observability, and release gates pass |

## Product surfaces

| Surface | Current status | Boundary |
| --- | --- | --- |
| Attention-state compiler | L1 + L2 | Compiles typed attention state or Retention IR into a fingerprinted `RuntimeManifest` |
| KV manager | L2 | Owns page allocation, generations, snapshots, Prefix/COW, token placement, retirement, ACK, and reuse |
| RuntimeSession | L2 | Presents transactional engine operations without exposing manager capabilities |
| Executor plan | L2 | Compiles a manifest directly into Full, Sliding, Full+Sliding, or exact Chunked attention classes |
| Luminal paged-attention boundary | Source implemented; device qualification pending | Accepts OrbitKV-authored page geometry and CSR metadata; never allocates or recycles pages |
| Rust server boundary | L2 contract | Async local `Engine` accepts logical batch intent and streams output events without physical state |
| OpenAI-compatible API | Not implemented | HTTP/SSE/WebSocket, tokenizer, scheduler, and sampling integration remain roadmap work |
| Complete model executor | Not implemented in the parent composition crate | The fork contains inference building blocks; the full OrbitKV lifecycle is not yet wired through a released model graph |

## Attention-state coverage

| State shape | Compiler and manager | Executor lowering | Real-device engine status |
| --- | --- | --- | --- |
| Full token KV | Host-tested, including shared Prefix and COW | Implemented | Current architecture unqualified |
| Sliding token KV | Host-tested periodic placement, retirement, ACK, and reuse | Implemented | Current architecture unqualified |
| Full + Sliding | Host-tested class-separated lifecycle and joint Prefix/COW | Implemented | Current architecture unqualified |
| Exact Chunked token KV | Host-tested resettable epoch lifecycle | Implemented | Current architecture unqualified |
| Full latent KV | Host-tested component-aware core lifecycle | Rejected until a matching Luminal kernel contract exists | Unqualified |
| Recurrent checkpoints | Host-tested independent pool | Not integrated into one model transaction | Unqualified |
| Convolution state | Host-tested independent pool | Not integrated into one model transaction | Unqualified |
| Per-head or region-partitioned layouts | Compiler primitives exist | Not generally admitted by the current executor plan | Unqualified |

## Token-level lifecycle

Token placement and disposition are core manager state, not an optional server
feature. Reclamation behavior follows compiled semantics:

- Full state remains live unless the request, Prefix, or explicit disposition
  proves otherwise. Relocation is policy-gated because it can add copy cost
  without reducing semantic state.
- Sliding state retires pages as the visibility frontier advances and reuses a
  generation only after executor completion and exact acknowledgement.
- Full + Sliding keeps independent class frontiers; one class cannot justify
  reclaiming another.
- Chunked state retires at proved epoch boundaries.
- Latent and fixed state do not relocate until their component-specific device
  copy and publication contracts are validated.

## Removed surfaces

The active product intentionally has no compatibility tree, Python runtime, C
ABI, packaged engine target, numbered wire contract, generic adapter framework,
or second page allocator. These are breaking removals, not deprecated aliases.
Historical files under `results/**` may preserve such identities as provenance.

## Current claim boundary

The current architecture is host-verified. It has no same-source L3/L4 closure
and therefore no current speedup, capacity, memory-saving, production, or
complete-replacement claim. A future benefit statement must compare the same
model, weights, dtype, kernels, batching policy, request trace, device budget,
and output semantics, and must report both successful and failed gates.
