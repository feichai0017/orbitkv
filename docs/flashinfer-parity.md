# FlashInfer parity matrix

Loom Infer uses this pinned comparison baseline:

| Item | Reference |
| --- | --- |
| Release | [FlashInfer v0.6.16.post1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.16.post1) |
| Source | [`5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57`](https://github.com/flashinfer-ai/flashinfer/commit/5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57) |

Benchmark records verify the installed wheel version. They record a provider
commit only when the installed artifact proves its source revision.

Parity means matching an admitted operator contract. It does not mean matching
Python wrappers, file structure, or symbol count. Shape, dtype, layout,
masking, numerical behavior, workspace, aliasing, stream, and Graph semantics
must agree before two paths form a matched comparison.

Loom does not claim complete domain-level parity.

## States

| State | Meaning |
| --- | --- |
| `partial device correct` | A narrower Loom contract passed its declared H20 correctness gate |
| `requalification` | Loom changed the contract after its last H20 record |
| `planned` | The roadmap names the domain, but no permanent provider is admitted |
| `unscoped` | Loom has not admitted a contract for the upstream domain |

## Domain coverage

| Domain | Representative upstream surface | Loom state |
| --- | --- | --- |
| Dense decode attention | `single_decode_with_kv_cache`, paged batch decode, XQA | `requalification`: BF16 NHD D128 providers exist, but their records predate the DeviceRegion launch path |
| Prefill attention | Single, ragged batch, and paged batch prefill | `requalification`: BF16 NHD D128 providers exist, but their records predate the DeviceRegion launch path |
| Paged KV append | Standard and MLA paged append, index and position generation | `requalification`: fused standard-RoPE BF16 append now requires exclusive target pages |
| Attention state and cascade | State merge and cascade wrappers | `planned` at state-merge level |
| Mixed-batch attention | Batch attention and attention sinks | `unscoped` |
| MLA attention | Paged MLA decode and prefill | `unscoped` |
| Sparse, MSA, and POD attention | Sparse, multiple-sequence, and combined prefill/decode wrappers | `unscoped` |
| Dense GEMM | BF16, FP8, FP4, and tiny GEMM | `requalification`: one contiguous BF16 cuBLASLt plan requires a current-source H20 record |
| Grouped GEMM | BF16, FP8, and FP4 grouped matrix work | `planned` through vendor providers |
| Normalization | RMSNorm, add RMSNorm, LayerNorm, and fused QK norm | `requalification`: contiguous RMSNorm F32, FP16, and BF16 providers require current-source H20 records |
| RoPE | Standard, Llama 3.1, and fused KV variants | `requalification`: standalone BF16 D128 NeoX passed the local H20 gate, but no current-source record is published |
| Sampling and speculation | Sampling, logits processors, and speculative verification | `planned` |
| MoE | Routing and fused expert execution | `planned` |
| Quantization | Packbits, FP4, FP8, and KV formats | `planned` |
| Communication | AllReduce and all-to-all variants | `planned` after a measured distributed workload |
| Activation and MLP tail | SiLU-multiply and GELU variants | `unscoped` |
| GDN, KDA, Mamba, and SSM | Recurrent and state-update operators | `unscoped` |
| Supporting backends | cuDNN attention, CuTe DSL, and green contexts | `unscoped`. Loom custom kernels use cuda-oxide |

The upstream links remain in the pinned
[FlashInfer API index](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/index.rst).

## Admitted attention matrix

| Loom contract | CUDA implementation | Historical H20 matrix | Historical Graph evidence |
| --- | --- | --- | --- |
| Single decode | Direct online softmax | MHA, MQA, and GQA | None |
| Single decode split-K | Explicit partitions, F32 partial workspace, and eight-warp merge | MQA and GQA | None |
| Paged batch decode | Direct MHA. Eight-warp token-parallel MQA and GQA | MHA, MQA, GQA, mixed lengths, page order, and read-only page reuse | None |
| Ragged causal prefill | Direct, eight-warp, sixteen-warp, and tiled GQA4 split-eight | Direct short MHA, MQA, and GQA. Long MQA uses sixteen warps. Published stages also cover eight-warp and tiled GQA4 paths | Tiled long GQA4 only |
| Paged causal prefill | Direct online softmax | MHA, MQA, GQA, mixed query and KV lengths, page order, and read-only page reuse | One direct GQA4 fixture |
| Standard RoPE | D128 NeoX split-half with explicit I32 positions | Positions through 32,767 in the recorded fixture | None |
| Fused RoPE plus paged append | One through 64 explicit tokens with per-page reference counts | Requalification | Requalification |

All attention rows fix BF16, NHD layout, head dimension 128, full attention
unless the row says causal, and F32 softmax state. Paged rows fix page size 16.
No row covers sliding windows, soft caps, custom masks, FP8 KV, or MLA.

Every matrix entry predates the DeviceRegion submission path. Current-source
device and Graph records remain open.

## Dispatch limits

Paged decode chooses direct only when query-head count equals KV-head count.
It chooses eight-warp token parallelism for MQA and GQA. The policy does not
use KV length.

Ragged prefill uses average KV length across the batch:

- below 64 tokens: direct.
- at least 64 tokens with one KV head: sixteen warps.
- other long shapes: eight warps.
- GQA group size four with average KV length at least 256: tiled split-eight.

This policy does not use a length histogram or request grouping. The tiled
Graph record does not qualify the other ragged algorithms.

Paged prefill has one direct provider. Its GQA4 Graph record does not qualify
MHA, MQA, mutable metadata, graph updates, or optimized long-context paths.

## Paged KV ownership difference

FlashInfer parity at the tensor level does not define the engine's KV ownership
policy. Loom now makes write ownership explicit.

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
| Device | H20 correctness, edge cases, and declared sanitizer tools |
| Graph | Capture and replay under one declared binding policy |
| Performance | Matched providers, timed region, raw samples, and order variance |

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
