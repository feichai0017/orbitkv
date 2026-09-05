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
| Physical-residence ablation | L2 + narrow L3 mechanism check | `Compiled` and explicit `RequestLifetime` policies preserve attention semantics while changing only addressing and physical retirement; conservative mode is diagnostic and rejects incompatible lifecycle surfaces |
| RuntimeSession | L2 | Presents transactional engine operations without exposing manager capabilities |
| External KV tier transactions | L2 host | Export/restore run through an object-safe async transport contract; a real-byte host adapter verifies compact partial tails, per-page checksums, deletion, cross-session restore, and unobserved/ambiguous fault mapping; Mooncake/NIXL and hybrid restore remain open |
| Executor plan | L2 | Compiles a manifest directly into Full, Sliding, Full+Sliding, or exact Chunked attention classes |
| Luminal paged-attention boundary | L3 | Accepts OrbitKV-authored page geometry and CSR metadata; real-device block-page and packed-page decode pass; Luminal never allocates or recycles pages |
| Token relocation executor | L3 | Lowers manager-authored moves to per-layer K/V byte ranges, performs stream-ordered D2D copies, and exposes success evidence only after a CUDA event |
| Bucketed model runtime | L4 correctness | One symbolic graph is searched once into decode/prefill executables; query, batch, and per-class context dimensions have bounded capacities, dynamic inputs are preallocated, and one stable K/V arena per class survives prefill plus repeated decode |
| Fixed-signature decode CUDA Graph | L3 + narrow L4 correctness; narrow matched benefit | Flattened replay failed at +24.9%; selected child-graph composition passed correctness and reduced matched batch-one fixed-step wall time by 8.3% over 20 iterations and 5.8% over 100 iterations; throughput remains unqualified |
| On-device greedy sampling | L3 + narrow L4 parity | Fused dynamic-row argmax runs in the decoder graph; default execution reads one token ID per query row, and released-checkpoint outputs match host argmax across prefill and decode |
| Rust server boundary | L2 contract | Async local `Engine` accepts logical batch/sampling intent, streams output events, and exposes cancellation without physical state |
| vLLM frontend adapter | L2 protocol tests | Optional pinned Rust frontend dependency; tokenized Add/Abort, request-ID mapping, terminal token translation, and unsupported-field rejection pass host tests |
| OpenAI-compatible API | L2 HTTP protocol closure | A real HTTP completion smoke passes through tokenizer, Add bridge, a local test `Engine`, event translation, detokenization, and OpenAI JSON; model execution is not part of that smoke |
| Complete model executor | Narrow L4 correctness closure | A configuration-driven Full token-KV checkpoint completes prefill and repeated decode; multi-class Full+Sliding graph construction and a synthetic-policy H20 plumbing smoke pass, but no released hybrid-architecture checkpoint is qualified; scheduler and serving integration remain open |

## Decoder operator and model boundary

| Capability | Current status |
| --- | --- |
| Dense decoder blocks | BF16 embedding, linear projections, residuals, RMSNorm, RoPE, SwiGLU, optional QKV bias and QK norm |
| Attention | MHA/GQA paged attention; query-head count must divide by KV-head count; head dimension 64, 128, or 256 |
| KV execution | Manager-authored CSR page views, stable persistent arena, scatter writes, Prefix/COW lowering, stream-ordered token relocation |
| Output | Tied or untied LM head; fused on-device greedy argmax by default; full logits only through an explicit diagnostic path |
| Checkpoint family | Configuration-driven dense decoder with the expected tensor layout; one released full-attention checkpoint has real-device correctness evidence |
| Not yet executable as complete models | MoE, MLA/latent KV, recurrent or convolution state, quantized weights, multimodal encoders, speculative decoding, and device-qualified multi-class hybrid checkpoints |

Core support for a retention policy means its lifecycle can be compiled and
host-tested. It does not by itself imply that all model operators or the
corresponding device kernel path exist.

The current two-candidate device smoke selected materialized KV updates with a
graph-visible D2D epilogue into the stable arena, not all-layer
`ScatterNoCopy`. That still removes runtime-to-runtime cache transfer and keeps
addresses stable, but it is not evidence of zero-copy KV writes.

## Attention-state coverage

| State shape | Compiler and manager | Executor lowering | Real-device engine status |
| --- | --- | --- | --- |
| Full token KV | Host-tested, including shared Prefix, COW, disposition, and relocation | Implemented, including CUDA relocation | Minimal released-checkpoint prefill/decode and packed relocation/decode pass |
| Sliding token KV | Host-tested periodic placement, retirement, ACK, and reuse; same-semantics request-lifetime baseline | Implemented; CSR geometry matches across residence policies | Synthetic-policy H20 boundary-crossing output parity; no released Sliding model qualification |
| Full + Sliding | Host-tested class-separated lifecycle and joint Prefix/COW | Manifest-driven layer binding and independent per-class inputs/arenas; synthetic-policy H20 plumbing passes | Released hybrid-model execution unqualified |
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
ABI, packaged engine target, numbered wire contract, general plugin framework,
or second page allocator. It has one narrow external byte-transport contract;
this does not grant adapters KV lifecycle authority. These are breaking removals,
not deprecated aliases.
Historical files under `results/**` may preserve such identities as provenance.

## Current claim boundary

The current architecture has same-source L3 device correctness for paged
attention and token relocation, plus a narrow L4 released-checkpoint correctness
closure for Full token KV. Full+Sliding has a synthetic-policy device plumbing
smoke. A same-graph boundary-crossing prefill+decode check produced identical
output while reducing the Sliding class from 6 pages / 589,824 bytes to 5 pages
/ 491,520 bytes. Sliding, Full+Sliding, and exact Chunked still lack independent
released-model qualification. No repeated matched L5 experiment has completed,
so there is no current throughput, production-capacity, production, or
complete-replacement claim. A future benefit statement must
compare the same model, weights, dtype, kernels, batching policy, request trace,
device budget, and output semantics, and must report both successful and failed
gates.
