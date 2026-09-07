# Capability Matrix

This matrix separates implemented source, host verification, device execution,
measured benefit, and production readiness for the current
`orbitkv + orbitkv-executor + orbitkv-server + orbitkv-engine` architecture.
Historical results qualify only their recorded source closure.

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
| Physical-residence ablation | Narrow L5 same-executor closure | Released-hybrid paired runs preserve 256 output tokens while compiled residence lowers live payload by 27.8%, extends the fixed-budget boundary by 32 tokens, and slightly reduces total test-path time; serving throughput remains open |
| RuntimeSession | L2 | Presents transactional engine operations without exposing manager capabilities |
| External KV tier transactions | L2 host | Export/restore run through an object-safe async transport contract; a real-byte host adapter verifies compact partial tails, per-page checksums, deletion, cross-session restore, and unobserved/ambiguous fault mapping; Mooncake/NIXL and hybrid restore remain open |
| Executor plan | L2 | Compiles a manifest directly into Full, Sliding, Full+Sliding, or exact Chunked attention classes |
| Luminal paged-attention boundary | L3 | Accepts OrbitKV-authored page geometry and CSR metadata; real-device block-page and packed-page decode pass; Luminal never allocates or recycles pages |
| Token relocation executor | L3 | Lowers manager-authored moves to per-layer K/V byte ranges, performs stream-ordered D2D copies, and exposes success evidence only after a CUDA event |
| Bucketed model runtime | L4 correctness | One symbolic graph is searched once into decode/prefill executables; query, batch, and per-class context dimensions have bounded capacities, dynamic inputs are preallocated, and one stable K/V arena per class survives prefill plus repeated decode |
| Decoder schedule artifact | L3 + narrow L4 correctness | Persists selected decode/prefill schedules with paged-attention custom ops; strict manifest/model/arena/bucket identity, LLIR fingerprints, and required persistent-state aliases fail closed |
| Fixed-signature decode CUDA Graph | L3 + narrow L4 correctness; narrow matched benefit | Capture accepts one-token-per-request decode batches. Batch-one child-graph replay reduced matched fixed-step wall time by 5.8-8.3%; exact-signature automatic C2 recapture reduced throughput by 13.7% and is not used by serving |
| Compiler-constrained persistent state | Reference-gated measured improvement | A 16-candidate search selected 36/36 in-place K/V tensors in both buckets; four C2 epochs improved throughput 14.5%, TTFT 32.2%, TPOT 10.2%, and E2E 12.8% versus the prior OrbitKV artifact; random-trace digests differ |
| Joint compiler facts | L2 + compile-path integration | `orbitkv` derives backend-neutral storage, retention, address, retirement, and legal-layout facts; the executor binds stable arenas, injects deterministic facts into every Luminal bucket, binds paged-attention nodes to class IDs, and fingerprints the contract in schedule identity. No cost-driven rewrite is enabled yet |
| On-device greedy sampling | L3 + narrow L4 parity | Fused dynamic-row argmax runs in the decoder graph; default execution reads one token ID per query row, and released-checkpoint outputs match host argmax across prefill and decode |
| Rust server boundary | L2 contract | Async local `Engine` accepts logical batch/sampling intent, streams output events, and exposes cancellation without physical state |
| Single-process model engine | Narrow L4 correctness + load closure | One dedicated thread owns bounded admission/output queues, an active set, `RuntimeSession`, and `CompiledDecoder`; released hybrid tests cover B=2 mixed scheduling and direct B=1/B=8 logit parity. Fresh-prompt/greedy only |
| vLLM frontend adapter | L2 protocol + narrow L4 integration | Pinned Rust request/tokenizer/chat/SSE crates; tokenized Add/Abort, request-ID mapping, unsupported-field rejection, and dropped-stream auto-abort pass through the real model engine |
| OpenAI-compatible API | Narrow L4 correctness + load closure | `orbitkv-serve` passes real-checkpoint non-streaming, ordered SSE, cancellation, shutdown, final drain, and a fixed C1/C2/C4/C8 load trace with complete outputs; fairness, soak, capacity limit, and comparative benefit remain open |
| Complete model executor | Narrow L4 correctness closure | Configuration-driven Full and released Full+Sliding dense checkpoints complete on H20; the hybrid closure includes independent reference-token parity, native-window retirement/reuse, cancellation, final drain, B=8 logit isolation, bounded continuous batching, and the real HTTP path |

## Decoder operator and model boundary

| Capability | Current status |
| --- | --- |
| Dense decoder blocks | BF16 embedding, linear projections, pre-norm or sandwich-norm residuals, direct or unit-offset RMSNorm weights, global/local RoPE, SwiGLU or GeGLU, optional QKV bias and QK norm |
| Attention | MHA/GQA paged attention; query-head count must divide by KV-head count; head dimension 64, 128, 256, or 512 when the compiled FlashInfer specialization exists |
| KV execution | Manager-authored CSR page views, stable persistent arena, scatter writes, Prefix/COW lowering, stream-ordered token relocation |
| Output | Tied or untied LM head; fused on-device greedy argmax by default; full logits only through an explicit diagnostic path |
| Checkpoint family | Configuration-driven dense decoder with fail-closed config and safetensors-header validation; released Full and Full+Sliding checkpoints have real-device correctness evidence |
| Not yet executable as complete models | MoE, MLA/latent KV, recurrent or convolution state, quantized weights, multimodal encoders, speculative decoding, and tensor/pipeline parallel models |

Core support for a retention policy means its lifecycle can be compiled and
host-tested. It does not by itself imply that all model operators or the
corresponding device kernel path exist.

Persistent K/V state now uses a required-alias contract. Luminal rejects a
candidate or stored artifact unless every K/V output resolves to the same
registered input arena in every bucket. The qualified artifact reports 36/36
in-place tensors and zero copy-back bytes for both decode/prefill buckets.

Validated state-layout facts now enter the same e-graph as the decoder. This is
the first half of joint compilation, not yet a performance feature: Luminal has
no packed-layout rewrite or exported cost profile, and the manager's relocation
decision still uses its conservative static policy.

## Attention-state coverage

| State shape | Compiler and manager | Executor lowering | Real-device engine status |
| --- | --- | --- | --- |
| Full token KV | Host-tested, including shared Prefix, COW, disposition, and relocation | Implemented, including CUDA relocation | Minimal released-checkpoint prefill/decode and packed relocation/decode pass |
| Sliding token KV | Host-tested periodic placement, retirement, ACK, and reuse; same-semantics request-lifetime baseline | Implemented; CSR geometry matches across residence policies | Native Sliding layers cross their 512-token window in the released hybrid H20 closure |
| Full + Sliding | Host-tested class-separated lifecycle and joint Prefix/COW | Manifest-driven layer binding and independent per-class inputs/arenas | Released 3-Full/15-Sliding checkpoint passes reference parity, retirement/reuse, cancellation, and final drain on H20 |
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
attention and token relocation, plus narrow L4 released-checkpoint correctness
closures for Full and Full+Sliding token KV. The Full+Sliding run crosses the
native Sliding boundary and matches independently generated greedy tokens while
qualifying lifecycle reuse and cancellation. A ten-pair release-mode
same-executor ablation produced identical 256-token outputs, reduced physical
resident payload from 14,155,776 to 10,223,616 bytes, and increased the
fixed-budget boundary from 528 to 560; Retention Amplification fell from 1.387
to 1.002. Median batch-one test-path time improved
by 0.77%, with a paired mean improvement of 11.633 ms and 95% confidence interval
5.697-17.570 ms. This qualifies a narrow lifecycle-management benefit, not
serving advantage. The single-process HTTP path now completes a fixed load
through C8, but the measured increase from 184.24 to 518.88 output token/s costs
substantially higher TTFT and TPOT. A subsequent artifact-fixed four-epoch C2
comparison initially reached 0.534x tuned SGLang throughput. Compiler-constrained
search improves the same engine by 14.5% and raises the current ratio to 0.598x,
with 1.59x TPOT and 4.79x TTFT. The configured K/V tensor payload is 40.4%
smaller than SGLang's Gemma3 fallback, but this still does not offset the
executor gap. Exact Chunked still lacks independent
released-model qualification. There is no production, broad-model, or
complete-replacement claim. A future serving statement must
compare the same model, weights, dtype, kernels, batching policy, request trace,
device budget, and output semantics, and must report both successful and failed
gates.
