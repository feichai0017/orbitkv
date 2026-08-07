# FlashInfer parity matrix

Loom Infer targets the inference-operator surface of
[FlashInfer v0.6.16.post1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.16.post1),
pinned to source commit
[`5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57`](https://github.com/flashinfer-ai/flashinfer/commit/5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57).

Release candidates, nightly builds, and rolling online documentation do not
change this acceptance baseline. They enter an upstream watch list before the
project deliberately advances the pin.

Loom measures parity by operator contract, not file or symbol count. Each
admitted row records shapes, dtypes, layouts, architecture, and execution
semantics.

Each row also records its oracle, H20 correctness, matched performance, CUDA
Graph, and engine integration. A domain remains incomplete until every admitted
contract passes its declared gates.

## States

| State | Meaning |
| --- | --- |
| `partial device correct` | A narrower Loom contract passed its declared device correctness gate |
| `planned` | The roadmap names the domain, but no permanent provider is admitted |
| `unscoped` | The pinned upstream surface is recorded, but Loom has not admitted a contract |

No complete domain-level parity is currently claimed.

## Operator domains

| Domain | Representative v0.6.16.post1 surface | Loom state |
| --- | --- | --- |
| Dense decode attention | [`single_decode_with_kv_cache`, `BatchDecodeWithPagedKVCacheWrapper`, `xqa`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/attention.rst) | `partial device correct`; BF16 NHD D128 has single-request direct/split-K and page-size-16 batch-decode GPU paths |
| Prefill and append attention | [`single_prefill_with_kv_cache`, paged and ragged batch prefill](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/attention.rst) | `partial device correct`; BF16 NHD D128 has ragged and page-size-16 paged batch causal prefill GPU paths |
| Mixed-batch attention | [`BatchAttention`, attention-sink wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/attention.rst) | `unscoped` |
| MLA attention | [`BatchMLAPagedAttentionWrapper`, XQA and TRT-LLM MLA decode](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/attention.rst) | `unscoped` |
| Attention state and cascade | [`merge_state`, `merge_states`, `MultiLevelCascadeAttentionWrapper`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/cascade.rst) | `planned` at state-merge level |
| Sparse and MSA attention | [`BlockSparseAttentionWrapper`, variable block sparse, MSA](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/sparse.rst) | `unscoped` |
| POD attention | [`PODWithPagedKVCacheWrapper`, batch POD](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/pod.rst) | `unscoped` |
| Paged KV operations | [`append_paged_kv_cache`, MLA append, index/position generation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/page.rst) | `partial device correct`; BF16 NHD D128, page-size-16 fused standard-RoPE append supports explicit 1..=64 tokens |
| Dense and quantized GEMM | [`mm_bf16`, `mm_fp8`, `mm_fp4`, `tinygemm_bf16`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/gemm.rst) | `partial device correct`; one fixed contiguous BF16 cuBLASLt plan only |
| Grouped GEMM | [`grouped_mm_bf16`, `grouped_mm_fp8`, `grouped_mm_fp4`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/grouped_mm.rst) | `planned` through vendor providers |
| Fused MoE | [routing and fused MoE providers](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/fused_moe.rst) | `planned`; none implemented |
| Sampling and speculation | [logits/probability sampling and chain speculative sampling](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/sampling.rst) | `planned`; none implemented |
| Logits processing | [`LogitsPipe`, temperature, Top-K, Top-P, Min-P, sample](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/logits_processor.rst) | `planned`; no pipeline API |
| Standalone Top-K | [`top_k` and page-table/ragged transforms](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/topk.rst) | `planned`; none implemented |
| Normalization | [RMSNorm, fused add RMSNorm, LayerNorm, fused QK RMSNorm/RoPE](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/norm.rst) | `partial device correct`; contiguous RMSNorm F32/FP16/BF16 only |
| RoPE | [standard and Llama 3.1 RoPE, fused FP8 KV append](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/rope.rst) | `partial device correct`; BF16 D128 NeoX split-half has standalone and explicit 1..=64-token paged-KV append paths |
| Activation and gated MLP tail | [`silu_and_mul`, GELU tanh/exact variants](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/activation.rst) | `unscoped` |
| Quantization | [`packbits`, FP4, NVFP4 KV, MXFP8](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/quantization.rst) | `planned`; none implemented |
| Communication | [AllReduce fusion, quantized AllReduce, MoE and decode A2A](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/comm.rst) | `planned` only after a measured distributed workload |
| GDN prefill/decode | [chunk GDN](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/gdn_prefill.rst), [recurrent GDN decode](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/gdn_decode.rst) | `unscoped` |
| KDA decode | [`recurrent_kda`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/kda_decode.rst) | `unscoped` |
| Mamba and SSM | [`selective_state_update`, checkpointing SSU](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/mamba.rst) | `unscoped` |
| mHC | [post and pre-fusion operators](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/mhc.rst) | `unscoped` |
| MLA packing | [`concat_mla_k`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/concat_ops.rst) | `unscoped` |

## Supporting surfaces

| Surface | Representative v0.6.16.post1 surface | Loom state |
| --- | --- | --- |
| cuDNN attention backend | [cuDNN batch decode and prefill](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/cudnn.rst) | `unscoped`; current vendor work is cuBLASLt GEMM only |
| CuTe DSL backend | [CuTe DSL operators and wrappers](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/cute_dsl.rst) | Outside the custom-kernel source boundary; Loom kernels use cuda-oxide |
| CUDA green contexts | [device green-context partitioning](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/green_ctx.rst) | `unscoped` |
| Testing and benchmarks | [FLOP, bandwidth, GPU-time, and Graph helpers](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post1/docs/api/testing.rst) | Infrastructure partial; matched eager, CUPTI kernel-duration, and fixed-address Graph records exist, while hardware counters and broader Graph coverage remain open |

## First attention sequence

The first admitted attention contract is a BF16 SM90 single-request decode
slice: NHD layout, head dimension 128, MHA/GQA, full window, no positional
encoding or soft cap, F32 score/softmax accumulation, BF16 output, and F32
log2-LSE. It covers head mapping, online softmax, numerical stability, and
non-power-of-two KV lengths.

The
[H20 result](results/h20-bf16-single-decode-correctness-20260803.json) is a
Loom GPU versus CPU-oracle gate. It contains no matched FlashInfer run.

The first [matched eager-provider result](results/h20-flashinfer-v0.6.16.post1-eager-performance-20260805.json)
uses identical BF16 operand bit patterns and both provider orders. It measures
the pre-split-K Loom source: FlashInfer has 6.29x lower median latency at GQA KV
length 127 and 80.08x lower median latency at GQA KV length 4096 under that
declared metric. Loom has lower median latency for the KV-length-1 MHA case, but
short eager paths show larger cross-process order variance and are not an
isolated kernel-duration claim.

The current [matched split-K result](results/h20-flashinfer-v0.6.16.post1-split-k-eager-performance-20260805.json)
retains the same semantic shapes and fixtures while recording both providers'
execution metadata. Split-K lowers Loom median latency by 3.79x at GQA KV
length 127 and 26.79x at KV length 4096. FlashInfer remains 1.69x and 3.00x
lower-latency under the declared eager metric.

The current [parallel-merge result](results/h20-flashinfer-v0.6.16.post1-parallel-merge-eager-performance-20260805.json)
uses an eight-warp block-local merge and retuned partition counts. Relative to
the direct baseline, Loom reaches 5.39x and 38.19x lower latency at GQA KV
lengths 127 and 4096. FlashInfer remains 1.17x and 2.09x lower-latency.

The second contract adds BF16 paged batch decode with NHD pages, head dimension
128, page size 16, MHA/MQA/GQA, one query token per request, full window, and no
positional encoding or soft cap. Its `i32` page table uses `indptr`, physical
page indices, and per-request last-page lengths compatible with the pinned
FlashInfer wrapper.

The [paged H20 result](results/h20-bf16-paged-batch-decode-correctness-20260806.json)
covers mixed request lengths, arbitrary physical page order and reuse, exact
metadata spans, and a device-side invalid-page guard. All valid cases produce
bit-exact BF16 output against the CPU oracle, maximum log2-LSE error is
`4.768371582e-7`, and four Compute Sanitizer tools report no errors.

The first [matched paged eager result](results/h20-flashinfer-v0.6.16.post1-paged-batch-decode-eager-performance-20260806.json)
uses identical BF16 page-pool bits, `i32` page tables, preallocated buffers,
CUDA events, and both provider orders. Loom has 4.21x lower combined median
latency for batch-1 MHA at KV length 1. FlashInfer has 1.62x lower combined
median latency for the mixed-length batch-3 MQA case.

The current [token-parallel result](results/h20-flashinfer-v0.6.16.post1-paged-token-parallel-eager-performance-20260806.json)
preserves the same fixtures and measurement. Eight warps partition each
MQA/GQA request-head KV range and merge stable F32 state inside one block.
Loom MQA and GQA eager latency falls by 3.78x and 3.32x relative to the direct
record. Loom is now 4.41x lower-latency for MHA and 2.35x lower-latency for
MQA than FlashInfer.

The batch-4 GQA eager ranking remains excluded: FlashInfer's provider-order
delta is 60.62%. CUPTI records Loom kernel medians of `2.176`, `4.864`, and
`4.928` microseconds for MHA, MQA, and GQA, versus direct medians of `2.176`,
`20.928`, and `18.304` microseconds.

The third contract adds BF16 ragged batch prefill over contiguous NHD storage.
Query and KV rows use separate `i32` `indptr` arrays, head dimension is 128,
and the causal mask is bottom-right aligned. The first contract requires every
request to satisfy `1 <= qo_len <= kv_len`; it does not include empty requests,
RoPE, sliding windows, soft caps, or custom masks.

The current [ragged prefill H20 result](results/h20-bf16-ragged-prefill-tiled-split-k-correctness-20260806.json)
covers direct, eight-warp, sixteen-warp, and tiled eight-partition MHA, MQA,
and GQA execution, equal and mixed query/KV lengths, exact metadata spans,
explicit tiled workspace, and a device-side nonmonotonic-indptr guard. Maximum
BF16 output error is `4.8828125e-4`, maximum log2-LSE error is
`2.861022949e-6`, and four Compute Sanitizer tools report no errors.

The [matched ragged eager result](results/h20-flashinfer-v0.6.16.post1-ragged-prefill-cp-async-eager-performance-20260806.json)
retains 600 samples and both provider orders. Unrolled 16-byte `cp.async` K/V
staging lowers Loom long-GQA latency to `48.232` microseconds, `1.148x` below
the previous tiled split-K result and `7.729x` below direct. FlashInfer remains
`2.206x` lower-latency on stable long GQA. Short-MHA and mixed-MQA rankings are
excluded because FlashInfer's provider-order median deltas are `10.643%` and
`14.097%`.

The [ragged fixed-address Graph result](results/h20-bf16-ragged-prefill-cuda-graph-correctness-20260806.json)
captures the tiled partial and merge kernels and replays them twice after
external provider, plan, and read-buffer owners are dropped. Output and
log2-LSE preserve the standalone digests, and four Compute Sanitizer tools
report no errors or leaks.

The [matched ragged Graph performance result](results/h20-flashinfer-v0.6.16.post1-ragged-prefill-graph-performance-20260806.json)
records one fixed-address replay and one completion event per CUDA-event
sample. Loom and FlashInfer combined medians are `50.480` and `32.640`
microseconds, so FlashInfer is `1.547x` lower-latency on this shape. The
provider-order deltas are `0.127%` and `0.344%`. This single-replay Graph metric
is separate from eager provider and isolated-kernel measurements. Engine
invocation and serving results remain open before a continuous-batching parity
claim.

The fourth attention contract adds BF16 paged causal prefill with ragged query
rows and NHD page-size-16 KV storage. It uses query `indptr`, page `indptr`,
physical page indices, and last-page lengths compatible with the pinned
FlashInfer paged wrapper. The causal mask remains bottom-right aligned.

The [paged prefill H20 result](results/h20-bf16-paged-prefill-correctness-20260807.json)
covers direct MHA, MQA, and GQA execution, equal and mixed query/KV lengths,
partial pages, physical-page reordering and reuse, short metadata,
duplicate-binding preflight, and an invalid-page device guard. Maximum BF16
output error is `1.220703125e-4`, maximum log2-LSE error is
`9.536743164e-7`, and four Compute Sanitizer tools report no errors.

The [matched paged-prefill eager result](results/h20-flashinfer-v0.6.16.post1-paged-prefill-eager-performance-20260807.json)
retains 600 samples and both provider orders. Loom is `3.264x`
lower-latency on short MHA and `1.288x` lower-latency on GQA4. FlashInfer is
`1.081x` lower-latency on mixed MQA. Every provider-order delta is below
`2.1%`; no universal winner is claimed. Graph replay and optimized
long-context paths remain open.

The first standalone RoPE contract adds BF16 NHD D128 Q/K tensors with
explicit I32 position IDs, full-dimension NeoX split-half rotation, scale one,
and theta 10,000. The [H20 correctness result](results/h20-bf16-rope-pos-ids-correctness-20260806.json)
covers positions through 32,767 and all four Compute Sanitizer tools.

The [matched RoPE eager result](results/h20-flashinfer-v0.6.16.post1-bf16-rope-pos-ids-eager-performance-20260806.json)
uses positions matching two ragged prefill suffixes. Loom and FlashInfer
combined medians are `3.997` and `5.077` microseconds, so Loom is `1.270x`
lower-latency. Provider-order deltas are `0.048%` and `2.872%`. Both providers
pass independent standard RoPE references within the shared BF16 error limit;
their full-math and fast-math output bits are not equal.

The first fused append contract adds one BF16 Q/K/V token per request, derives
each position from the request's extended page table, rotates Q/K in NeoX
split-half style, and writes rotated K plus unmodified V to the final physical
NHD slot. The [H20 correctness result](results/h20-bf16-rope-paged-kv-append-correctness-20260806.json)
is bit-exact with the Loom CPU reference for full Q and K/V page pools, covers
duplicate-slot and invalid-page guards, and passes all four Compute Sanitizer
tools.

The [matched fused append eager result](results/h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-eager-performance-20260806.json)
compares Loom's one kernel with FlashInfer's standard RoPE plus paged append
composition. Fixed-affinity combined medians are `3.989` and `11.735`
microseconds, making Loom `2.942x` lower-latency on the admitted batch-4
Q16/K4 D128 page-size-16 case. Provider-order deltas are `0.128%` and
`3.159%`. That immutable first record is limited to one token per request.

The explicit [multi-token correctness result](results/h20-bf16-rope-paged-kv-append-tokens-correctness-20260806.json)
extends the same NHD D128/page-size-16 boundary to 1 through 64 tokens with
caller-supplied batch indices and positions. The six-token case covers each
request's final two tokens in shuffled order; the 64-token case exercises both
validation warps. Short metadata fails before submission, four invalid
device-resident metadata classes preserve output sentinels, and all four
Compute Sanitizer tools pass.

The [matched explicit multi-token eager result](results/h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-tokens-eager-performance-20260806.json)
records Loom and FlashInfer combined medians of `5.510` and `11.732`
microseconds, making Loom `2.129x` lower-latency on the admitted six-token
suffix case. Provider-order deltas are `2.689%` and `4.164%`. Both providers
meet independent references within the shared BF16 limit, but Q/K output and
reference bits are not claimed equal. More than 64 tokens, MLA, FP8, other
RoPE/layout variants, engine, and serving boundaries remain open.

The [explicit append Graph correctness result](results/h20-bf16-rope-paged-kv-append-tokens-cuda-graph-correctness-20260806.json)
captures one checked Loom command and replays it after external resource owners
are dropped. The [matched Graph result](results/h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-tokens-graph-performance-20260806.json)
records one-node Loom and two-node FlashInfer combined medians of `8.288` and
`13.728` microseconds, making Loom `1.656x` lower-latency under the declared
single-replay completion-event metric. Provider-order deltas are `2.330%` and
`0.350%`; all 400 samples are retained. Graph updates, engine, and serving
boundaries remain open.
