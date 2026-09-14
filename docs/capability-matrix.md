# Capability Matrix

This matrix separates implemented source, host verification, device execution,
measured benefit, and production readiness for the current
`orbitkv + orbitkv-executor + orbitkv-engine` architecture.
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
| KV manager | L2 | Owns page allocation, generations, snapshots, Prefix/COW, compiled retirement, ACK, and reuse |
| Physical-residence ablation | Narrow L5 same-executor closure | Released-hybrid paired runs preserve 256 output tokens while compiled residence lowers live payload by 27.8%, extends the fixed-budget boundary by 32 tokens, and slightly reduces total test-path time; serving throughput remains open |
| RuntimeSession | L2 | Presents transactional engine operations without exposing manager capabilities |
| External KV tier transactions | L2 host | Export/restore run through an object-safe async transport contract; a real-byte host adapter verifies compact partial tails, per-page checksums, deletion, cross-session restore, and unobserved/ambiguous fault mapping; Mooncake/NIXL and hybrid restore remain open |
| Executor plan | L2 | Compiles and jointly validates per-layer token-KV or recurrent-plus-convolution ownership; the production decoder dispatches layers from this topology |
| Luminal attention/view boundary | L3 | Logical attention and OrbitKV-authored paged KV have separate contracts; capability rules admit FlashInfer CUDA-core/tensor-core algorithms and optional SM90 FlashAttention-3. Provider-owned scratch and metadata conversion do not transfer page allocation or recycling authority from OrbitKV |
| Bucketed model runtime | L4 correctness for token KV; narrow L3 for fixed-state operators | Token-KV and stateful graphs are searched once into decode/prefill executables. Dynamic inputs are preallocated, request segmentation is shared across attention and recurrent operators, and every state class has a stable arena |
| Decoder schedule artifact | L3 + narrow L4 correctness | Schema 6 persists selected decode/prefill schedules and required generated CUDA module images; strict manifest/model/arena/bucket identity, LLIR fingerprints, required persistent-state aliases, and RMSNorm layout legality fail closed; older formats require regeneration |
| Fixed-signature decode CUDA Graph | L3 + narrow L4 correctness; historical narrow matched benefit | Capture accepts one-token-per-request decode batches. In its recorded source closure, batch-one child-graph replay reduced matched fixed-step wall time by 5.8-8.3%; exact-signature automatic C2 recapture reduced throughput by 13.7% and is not used by serving |
| Compiler-constrained persistent state | Reference-gated measured improvement | A 16-candidate search selected 36/36 in-place K/V tensors in both buckets; four C2 epochs improved throughput 14.5%, TTFT 32.2%, TPOT 10.2%, and E2E 12.8% versus the prior OrbitKV artifact; random-trace digests differ |
| Joint compiler facts | L2 + compile-path integration | `orbitkv` derives backend-neutral storage, retention, address, and retirement facts; the executor binds stable arenas, injects deterministic facts into every Luminal bucket, binds paged-attention nodes to class IDs, and fingerprints the contract in schedule identity. FlashInfer and FlashAttention consume that contract through egglog provider rewrites; physical-layout competition remains open |
| Workload tuning | L2 + bounded L4 model integration | Artifact-bound batch/query/context representatives, feasible joint buckets, private-page profiling metadata, cooperative exploration budgets and configurable CUDA Graph finalists; B4/B8 aligned and ragged reference/replay/drain pass on one seven-bucket artifact. Shared-Prefix physical-layout tuning remains open |
| Shared FP8 preparation | L3 + bounded L4 model parity | Opt-in egglog producer sharing and prequantized DeepGEMM consumers preserve the combined alternatives. Independent packed-byte/output checks and eight-step B1 plus B4/B8 aligned/ragged reference/replay/drain pass on H20. Captured scratch owners survive dynamic growth and resident graph reuse. The isolated two-consumer gain is not a whole-model serving claim; see [FP8 region tuning](fp8-region-tuning.md) |
| On-device greedy sampling | L3 + narrow L4 parity | Fused dynamic-row argmax runs in the decoder graph; default execution reads one token ID per query row, and released-checkpoint outputs match host argmax across prefill and decode |
| Rust server boundary | L2 contract | Async local `Engine` accepts logical batch/sampling intent, streams output events, and exposes cancellation without physical state |
| Single-process model engine | Narrow L4 correctness + load closure | One dedicated thread owns bounded admission/output queues, an active set, `RuntimeSession`, and `CompiledDecoder`; released hybrid tests cover B=2 mixed scheduling and direct B=1/B=8 logit parity. Fresh-prompt/greedy only |
| vLLM frontend adapter | L2 protocol + narrow L4 integration | Pinned Rust request/tokenizer/chat/SSE crates; tokenized Add/Abort, request-ID mapping, unsupported-field rejection, and dropped-stream auto-abort pass through the real model engine |
| OpenAI-compatible API | Narrow L4 correctness + load closure | `orbitkv-serve` passes real-checkpoint non-streaming, ordered SSE, cancellation, shutdown, final drain, and a fixed C1/C2/C4/C8 load trace with complete outputs; fairness, soak, capacity limit, and comparative benefit remain open |
| Complete model executor | Narrow L4 correctness closure | Configuration-driven Full and released Full+Sliding dense checkpoints complete on H20; the 27B hybrid path also passes a fresh 16-candidate prefill plus seven-step teacher-forced logit gate after excluding unproven 3-D fused-RMSNorm layouts. Lifecycle closure includes native-window retirement/reuse, cancellation, final drain, B=8 logit isolation, bounded continuous batching, and the real HTTP path |

## Decoder operator and model boundary

| Capability | Current status |
| --- | --- |
| Checkpoint import | Explicit `qwen2`, `mistral`, `gemma3_text` and Qwen3.5 text/envelope importers; unknown or contradictory architecture metadata is rejected. [Import contracts](checkpoint-import.md) are distinct from device/model qualification |
| Dense decoder blocks | BF16 embedding, linear projections, pre-norm or sandwich-norm residuals, direct or unit-offset RMSNorm weights, global/local RoPE, SwiGLU or GeGLU, optional QKV bias, QK norm, and per-head attention output gates |
| Attention | MHA/GQA paged attention; query-head count must be divisible by KV-head count; default scale is `head_dim^-0.5`; head dimension 64, 128, 256, or 512 when the compiled FlashInfer specialization exists |
| KV execution | Manager-authored CSR page views, stable persistent arena, scatter writes, and Prefix/COW lowering |
| Output | Tied or untied LM head; fused on-device greedy argmax by default; full logits only through an explicit diagnostic path |
| Checkpoint family | Configuration-driven dense decoder plus nested hybrid text-config parsing with fail-closed capability gates; released Full and Full+Sliding checkpoints have real-device correctness evidence |
| Primary target boundary | The Qwen3.8-27B-FP8 checkpoint compiles to 16 Full plus 48 recurrent/convolution layers. Projection weights and 128x128 inverse-scale tensors enter provider-neutral block-scaled linear nodes. Luminal generates four DeepGEMM variants per node for per-bucket search. A recorded schema-5 16-candidate artifact selects 32/32 in-place token-KV updates in both buckets and passes four-token prefill, seven teacher-forced decode steps, joint token/fixed-state evidence, release, and final drain on H20. That independent Transformers 5.12.1 reference run observed maximum absolute logit error 0.7461; the later seven-bucket artifact passes B4/B8 aligned and ragged cases with a maximum of 0.90625 under the unchanged 1.0 gate. Required in-place fixed-state writes and equal-valued loop-input equivalence remove the former multi-gigabyte state copy path. The recorded cross-engine C1 diagnostic reaches 0.620x SGLang / 0.539x vLLM throughput, with differing output digests at a known near tie |
| Not yet end-to-end supported | Qwen3.8-27B-FP8 still lacks soak, serving-scale and long-context qualification; its vision encoder and MTP path are not admitted. MoE, sparse/latent attention, speculative decoding, and tensor/pipeline parallel models remain unsupported |

The [schema 7 boundary qualification](../results/semantic-boundaries-20260914/README.md)
records the preceding fork reduction, backend-free semantic construction, final
27B replay/reference and HTTP lifecycle gates. Import no longer hardcodes CUDA
head sizes; the attention row above describes the previously qualified provider
geometries, not a parser restriction. Current schema 9 separates
logical attention/KV views and records explicit provider algorithms; see
[attention providers](attention-providers.md).

Core support for a retention policy means its lifecycle can be compiled and
host-tested. It does not by itself imply that all model operators or the
corresponding device kernel path exist.

Persistent K/V state now uses a required-alias contract. Luminal rejects a
candidate or stored artifact unless every K/V output resolves to the same
registered input arena in every bucket. The qualified artifact reports 36/36
in-place tensors and zero copy-back bytes for both decode/prefill buckets.

Validated state facts enter the same e-graph as the decoder. Logical attention and the
paged KV view are provider-neutral. FlashInfer algorithms and optional FlashAttention-3 enter through guarded egglog
provider rewrites. This establishes multiple executable candidates, with exact
ABI/target constraints, while joint physical-layout competition remains open.

## Attention-state coverage

| State shape | Compiler and manager | Executor lowering | Real-device engine status |
| --- | --- | --- | --- |
| Full token KV | Host-tested, including shared Prefix and COW | Implemented with stable arenas and paged attention | Released dense prefill/decode passes |
| Sliding token KV | Host-tested periodic placement, retirement, ACK, and reuse; same-semantics request-lifetime baseline | Implemented; CSR geometry matches across residence policies | Native Sliding layers cross their 512-token window in the released hybrid H20 closure |
| Full + Sliding | Host-tested class-separated lifecycle and joint Prefix/COW | Manifest-driven layer binding and independent per-class inputs/arenas | Released 3-Full/15-Sliding checkpoint passes reference parity, retirement/reuse, cancellation, and final drain on H20 |
| Exact Chunked token KV | Host-tested resettable epoch lifecycle | Implemented | Current architecture unqualified |
| Full latent KV | Host-tested component-aware core lifecycle | Rejected until a matching Luminal kernel contract exists | Unqualified |
| Recurrent checkpoints | Host-tested pool and joint RuntimeSession transaction | Geometry/compiler facts plus atomic prepare, submit, completion, abort, release, and reuse with token KV | Ragged packed operator and stable-arena transitions pass on H20; released model unqualified |
| Convolution state | Host-tested pool and joint RuntimeSession transaction | Minimal-history geometry plus the same atomic token-KV/fixed-state lifecycle | Ragged packed operator passes on H20; released model unqualified |
| Per-head or region-partitioned layouts | Compiler primitives exist | Not generally admitted by the current executor plan | Unqualified |

## Compiled lifecycle

Reclamation behavior follows compiled semantics:

- Full state remains live until request or Prefix ownership ends.
- Sliding state retires pages as the visibility frontier advances and reuses a
  generation only after executor completion and exact acknowledgement.
- Full + Sliding keeps independent class frontiers; one class cannot justify
  reclaiming another.
- Chunked state retires at proved epoch boundaries.
- Latent and fixed state require their component-specific device copy and
  publication contracts before end-to-end admission.

## Removed surfaces

The active product intentionally has no compatibility tree, Python runtime, C
ABI, packaged engine target, numbered wire contract, general plugin framework,
second page allocator, or live-token relocation/compaction state machine. It has
one narrow external byte-transport contract; this does not grant adapters KV
lifecycle authority. These are breaking removals, not deprecated aliases.
Historical files under `results/**` may preserve such identities as provenance.

## Current claim boundary

The current architecture has same-source L3 device correctness for paged
attention, plus narrow L4 released-checkpoint correctness
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
