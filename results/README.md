# Validation Records

The [Capability Matrix](../docs/capability-matrix.md) is the normative support
boundary. This directory is append-only evidence: every record qualifies only
the exact source, ABI, engine, hardware, commands, and outputs bound by its
manifest.

The live tree is ABI6. Its Rust core, exact 23-symbol C wire, Python runtime,
and SGLang Prefix adapter are host L2. Its H20 Prefix qualification is
pending. Every record below predates ABI6 and therefore remains historical.

## Latest engine record

| Record | Scope |
| --- | --- |
| `h20-sglang-v0517-abi5-v5-grouped-release-20260821` | Frozen `9233c06d…` ABI5-v5 on official SGLang v0.5.17 and one H20; scoped L4 correctness for Qwen Full and GPT-OSS Full+SWA B1/B4; grouped release; same-cap memory reduction 0%; `performance_go=false` |

The ABI5-v5 record contains four PRIMARY manager records and their stock
references. All eight JSON records pass independent verification, all 84
request traces match stock token-for-token, and every Full/SWA arena drains.
Each B4 manager record releases 20 requests using five release/recycle
transactions.

The qualified profile is page16 BF16 NHD, eager ChunkCache, TP/PP/DP/DCP = 1,
Qwen2.5-7B Full/FlashInfer, and GPT-OSS-20B ordered Full+SWA128/FA3 with
SGLang's built-in Triton MoE. Radix/Prefix, overlap, Graph, speculation,
disaggregation, streaming, hierarchical cache, and remote cache were disabled.

The compared manager and stock processes reserve equal KV tensor capacity, so
same-capacity intrinsic memory reduction is **0%**. The single epoch reports
B4 steady manager overhead of +4.1932% for Qwen and -5.2048% for GPT-OSS;
Qwen B1 is +5.0009%. With no repeated-epoch statistics,
`performance_go=false`; the GPT result is not a general speedup claim.

Nothing in this record qualifies ABI6 Prefix/COW, Python, SGLang integration,
relocation, Graph, or distributed execution.

## Earlier records

| Record | Historical boundary |
| --- | --- |
| `h20-sglang-v0517-abi5-full-hybrid-20260821` | Frozen ABI5-v4 official-release Full/Hybrid epoch |
| `h20-sglang-v0517-full-hybrid-20260821` | Frozen ABI4 official-release Full/Hybrid epoch |
| `h20-canonical-manager-20260820` | ABI3/development-pin Mistral pure-SWA lifecycle and memory accounting |
| `h20-rust-owned-pages-20260820` | Rust-selected physical SWA pages through SGLang allocation kernels |
| `h20-dense-sglang-20260819` | Pure-SWA 128-token Dense page binding through SGLang |
| `dense-runtime-20260819` | Fixed-capacity ownership reference and differential benchmark |
| `h20-cuda-event-overlap-20260819` | Historical request-scoped CUDA-event frontier experiment |
| `h20-radix-prefix-20260819` | Historical component-aware Full+SWA Prefix experiment |
| `h20-transactional-binding-20260819` | Historical prepare/load/commit hydration transaction |
| `h20-runtime-state-plan-20260819` | Historical shared runtime-artifact experiment |
| `h20-hybrid-capsule-20260818` | Historical GPT-OSS Full+SWA continuation experiment |
| `h20-live-tail-capsule-20260818` | Historical pure-SWA live-tail continuation experiment |
| `h20-capsule-export-20260818` | Historical checkpoint KV export/host restore experiment |
| `applicability-h20-20260817` | Historical Qwen Full, Mistral bounded, and GPT-OSS Hybrid geometry |
| `h20-gpt-oss-20b-real-20260817` | Historical real-checkpoint systems experiment |
| `lifetime-normalization-20260817` | Per-head window and retention-amplification analysis |
| `chunked-local-20260817` | Same-chunk to resettable-arena compiler proof |
| `sink-sliding-20260817` | Sink plus local lifetime partitioning |
| `retention-ir-20260817` | Declarative retention IR and legacy equivalence |
| `h20-generation-vmm-20260817` | Historical generation-aware CUDA VMM lifecycle |
| `owner-ffi-20260817` | Historical in-process Owner ABI |

Other directories are earlier calibration records retained for auditability.

## Interpretation rules

- Historical manifests and raw outputs are never rewritten to claim a later
  source tree. A provenance correction is a separate append-only amendment.
- A breaking ABI cannot inherit a prior ABI's L2 or L4 result.
- A host microbenchmark is not model throughput, and a single process epoch is
  not a performance GO.
- Smaller admitted capacity is not KV compression. Same-capacity comparisons
  must account for actual tensor-arena bytes, padding, and temporary headroom.
- Token compaction means exact K/V relocation. It is not quantization or a
  numerical compression claim.
- Compiler/reference proofs are not GPU or engine qualification.
- An isolated VMM record does not prove that VMM backs live SGLang KV tensors.

`h20-canonical-manager-20260820/provenance-amendment.json` records that its old
workspace snapshot was not originally sealed; its original manifest remains
unchanged. The newer ABI5-v5 record carries the exact sealed source closure
used by its run.
