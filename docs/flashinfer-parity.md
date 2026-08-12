# FlashInfer parity matrix

Oxide Infer uses this pinned comparison baseline:

| Item | Reference |
| --- | --- |
| Release | [FlashInfer v0.6.17](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17) |
| Source | [`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`](https://github.com/flashinfer-ai/flashinfer/commit/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7) |

Benchmark records verify the installed wheel version. They record a provider
commit only when the installed artifact proves its source revision.

Parity means matching an admitted operator contract. It does not mean matching
Python wrappers, file structure, or symbol count. Shape, dtype, layout,
masking, numerical behavior, workspace, aliasing, stream, and Graph semantics
must agree before two paths form a matched comparison.

Oxide Infer does not claim complete domain-level parity.

## States

| State | Meaning |
| --- | --- |
| `partial device correct` | A narrower Oxide Infer contract passed its declared device correctness gate |
| `requalification` | Oxide Infer changed the contract after its last device record |
| `planned` | The roadmap names the domain, but no permanent provider is admitted |
| `unscoped` | Oxide Infer has not admitted a contract for the upstream domain |

## Domain coverage

| Domain | Representative upstream surface | Oxide Infer state |
| --- | --- | --- |
| Dense decode attention | `single_decode_with_kv_cache`, paged batch decode, XQA | `partial device correct`: current R1 covers declared BF16 NHD single-decode and NHD/HND paged-decode runner cases; XQA and broader contracts remain open |
| Prefill attention | Single, ragged batch, and paged batch prefill | `partial device correct`: current R1 covers declared BF16 NHD D128 ragged and paged runner cases |
| Paged KV append | Standard and MLA paged append, index and position generation | `partial device correct`: current R1 covers the declared fused standard-RoPE BF16 exclusive-target-page cases; MLA remains open |
| Attention state and cascade | State merge and cascade wrappers | `planned` at state-merge level |
| Mixed-batch attention | Batch attention and attention sinks | `unscoped` |
| MLA attention | Paged MLA decode and prefill | `unscoped` |
| Sparse, MSA, and POD attention | Sparse, multiple-sequence, and combined prefill/decode wrappers | `unscoped` |
| Dense GEMM | BF16, FP8, FP4, and tiny GEMM | `partial device correct`: current R1 covers one contiguous BF16 cuBLASLt contract; FP8, FP4, and broader GEMM remain open |
| Grouped GEMM | BF16, FP8, and FP4 grouped matrix work | `planned` through vendor providers |
| Normalization | RMSNorm, add RMSNorm, LayerNorm, and fused QK norm | `partial device correct`: current R1 covers declared contiguous RMSNorm F32, FP16, and BF16 cases |
| RoPE | Standard, Llama 3.1, and fused KV variants | `partial device correct`: current R1 covers declared standard BF16 D128 NeoX and fused paged-append cases |
| Sampling and speculation | Sampling, logits processors, and speculative verification | `planned` |
| MoE | Routing and fused expert execution | `planned` |
| Quantization | Packbits, FP4, FP8, and KV formats | `planned` |
| Communication | AllReduce and all-to-all variants | `planned` after a measured distributed workload |
| Activation and MLP tail | SiLU-multiply and GELU variants | `unscoped` |
| GDN, KDA, Mamba, and SSM | Recurrent and state-update operators | `unscoped` |
| Supporting backends | cuDNN attention, CuTe DSL, and green contexts | `unscoped`. Oxide native kernels use cuda-oxide |

The upstream links remain in the pinned
[FlashInfer API index](https://docs.flashinfer.ai/).

## Admitted attention matrix

| Oxide Infer contract | CUDA implementation | Historical device matrix | Historical Graph evidence |
| --- | --- | --- | --- |
| Single decode | Direct online softmax | MHA, MQA, and GQA | None |
| Single decode split-K | Explicit partitions, F32 partial workspace, and eight-warp merge | MQA and GQA | None |
| Paged batch decode | Direct MHA. Eight-warp token-parallel MQA and GQA | MHA, MQA, GQA, mixed lengths, page order, and read-only page reuse | None |
| Ragged causal prefill | Direct, eight-warp, sixteen-warp, and tiled GQA4 split-four | Direct short MHA, MQA, and GQA. Long MQA uses sixteen warps. Current stages also cover tiled GQA4 | Tiled long GQA4 only |
| Paged causal prefill | Direct, eight-warp, sixteen-warp, and tiled GQA4 split-four | Short MHA, MQA, and GQA plus long MQA and tiled GQA4 | Tiled long GQA4 plus invalid-page rejection |
| Standard RoPE | D128 NeoX split-half with explicit I32 positions | Positions through 32,767 in the recorded fixture | None |
| Fused RoPE plus paged append | One through 64 explicit tokens with per-page reference counts | Requalification | Requalification |

All attention rows fix BF16, NHD layout, head dimension 128, full attention
unless the row says causal, and F32 softmax state. Paged rows fix page size 16.
No row covers sliding windows, soft caps, custom masks, FP8 KV, or MLA.

The phase 2 R1 record supersedes the pre-rename device status only for the
exact current-source cases emitted by its permanent runners. Historical
performance rows remain historical, and unlisted algorithms or Graph paths
remain open.

## Dispatch limits

Paged decode chooses direct only when query-head count equals KV-head count.
It chooses eight-warp token parallelism for MQA and GQA. The policy does not
use KV length.

Ragged prefill uses average KV length across the batch:

- below 64 tokens: direct.
- at least 64 tokens with one KV head: sixteen warps.
- other long shapes: eight warps.
- GQA group size four with average KV length at least 256: tiled split-four.

This policy does not use a length histogram or request grouping. The tiled
Graph record does not qualify the other ragged algorithms.

Paged prefill uses explicit caller selection. The current long MQA fixture uses
sixteen warps; the long GQA4 fixture uses tiled split-four with a caller-owned
F32 workspace. The tiled Graph record does not qualify mutable metadata, graph
updates, concurrent replay, or other algorithms.

## Paged KV ownership difference

FlashInfer parity at the tensor level does not define the engine's KV ownership
policy. Oxide Infer makes write ownership explicit.

Paged decode and prefill may read shared physical pages. Fused append accepts
an authoritative reference-count snapshot and writes only to pages whose count
is one. The engine or pager must make the target private and remap the request
before enqueue.

The old append records have these limits:

- They use the earlier 2026-08-06 contract.
- Some fixtures reuse a physical page at different write offsets.
- They do not qualify the new rule.

See the [evidence index](results/README.md) for the historical files.

## Evidence interpretation

The single-decode, paged-decode, ragged-prefill, paged-prefill, and standalone
RoPE records cover only their named shapes and timed regions. Stable results
retain both provider orders and raw samples. The record excludes a ranking
when its order variance exceeds the acceptance limit.

The project keeps four boundaries separate:

| Boundary | Required proof |
| --- | --- |
| Host | Contract validation and CPU or independent reference |
| Device | Correctness, edge cases, declared target, and sanitizer tools |
| Graph | Capture and replay under one declared binding policy |
| Performance | Matched providers, timed region, raw samples, and order variance |

The [current matched attention record](results/h20-flashinfer-v0.6.17-attention-eager-performance-7f3d08e-20260812.json)
covers 14 paged-decode, ragged-prefill, and paged-prefill shapes against the
pinned FlashInfer release. It retains both provider orders and 2,800 raw
latency samples. Oxide has lower combined median eager latency in eight stable
shapes and FlashInfer in six.

The [current paged-GQA4 record](results/h20-flashinfer-v0.6.17-paged-prefill-current-gqa4-eager-performance-02faf27-20260812.json)
supersedes only the long paged-GQA4 row. It keeps provider and source
progression in separate two-order cohorts and binds raw samples, exact runner
and artifact hashes, correctness and Graph gates, and all four sanitizer tools.
The earlier [paged optimization record](results/h20-flashinfer-v0.6.17-paged-prefill-tiled-gqa4-eager-performance-49290b5-20260812.json)
remains its immutable precursor.

The [optimized ragged-GQA4 record](results/h20-flashinfer-v0.6.17-ragged-prefill-dual-tile-gqa4-eager-performance-f9b95b0-20260812.json)
supersedes only the long ragged-GQA4 row. It binds the dual-tile source and
parent, both provider orders, raw samples, exact runner hashes, correctness and
Graph gates, and all four sanitizer tools.

The older [token-parallel correctness record](results/h20-bf16-paged-prefill-token-parallel-correctness-20260807.json)
and [long-context performance record](results/h20-flashinfer-v0.6.16.post1-paged-prefill-long-eager-performance-20260807.json)
remain historical source `8478ee9` evidence. No eager record qualifies
token-parallel Graph execution.

Engine and serving parity remain open. No existing record proves continuous
batching, end-to-end model speed, TTFT, TPOT, throughput, or memory savings.

## Advancing the pin

Release candidates, nightly builds, and rolling documentation do not change
the baseline. To advance the pin:

1. Record the new release and source commit.
2. Diff the operator contracts used by admitted rows.
3. Update fixtures and independent references.
4. Rerun affected correctness, sanitizer, Graph, and matched performance
   gates.
5. Preserve old records as historical evidence.
