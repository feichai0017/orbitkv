# Operator catalog

This catalog describes the current Rust API, CUDA provider, and published
evidence. It does not claim complete FlashInfer or FlashAttention parity.
See the [FlashInfer parity matrix](flashinfer-parity.md) for the broader target.

## States

| State | Meaning |
| --- | --- |
| `contract` | A public Rust specification and CPU reference exist |
| `device correct` | The CUDA provider passed its declared H20 correctness gate |
| `requalification` | The source contract changed after the last H20 record |
| `planned` | The roadmap names the work, but no provider is admitted |

Each state applies only to the contract shown in the same row. Graph,
performance, engine, and serving evidence remain separate.

## Implemented surface

| Operator | Admitted contract and CUDA path | Published evidence | Open boundary |
| --- | --- | --- | --- |
| RMSNorm | Contiguous F32, FP16, and BF16. Scalar and packed paths | Historical H20 correctness, sanitizer, and one Graph record | Current-source H20 requalification, matched performance, and external-region qualification |
| Dense GEMM | Contiguous BF16 `D=A*W^T` with F32 accumulation. One fixed cuBLASLt algorithm | Historical H20 correctness, sanitizer, and one Graph record | Current-source H20 requalification, more shapes, and engine invocation |
| Single decode | BF16 NHD D128 full attention. Direct MHA, MQA, and GQA | Historical H20 correctness and matched eager records | Current-source H20 requalification, CUDA Graph, and engine invocation |
| Single decode split-K | Explicit partitions and caller-owned F32 workspace. The historical H20 matrix covers MQA and GQA | Historical H20 correctness, matched eager, and CUPTI records | Current-source H20 requalification, MHA evidence, and automatic policy |
| Paged batch decode | BF16 NHD or HND D128, page size 16. MHA uses direct. MQA and GQA use eight-warp block-local merge. A device validator returns typed metadata errors | Historical NHD H20 correctness and matched eager records | Publish current-source NHD/HND status, correctness, sanitizer, Graph, and performance records |
| Ragged causal prefill | BF16 NHD D128 with bottom-right causal masking. Direct, eight-warp, sixteen-warp, and tiled GQA4 providers exist | Historical H20 correctness, matched eager, and tiled long-GQA4 Graph records | Current-source H20 requalification, per-request dispatch, and engine invocation |
| Paged causal prefill | BF16 NHD D128, page size 16, bottom-right causal masking. The caller selects direct, eight-warp, or sixteen-warp execution. A device validator returns typed metadata errors | Historical H20 correctness, matched eager, and one direct-GQA4 Graph record | Publish current-source status, correctness, sanitizer, Graph, and performance records |
| Standard RoPE | BF16 NHD D128 NeoX split-half with explicit I32 positions | Historical H20 correctness, sanitizer, and matched eager records | Publish the current-source H20 rerun, then cover more RoPE variants |
| Fused RoPE plus paged KV append | BF16 NHD D128, page size 16, one through 64 explicit tokens. A validator emits a cache-bound compact map; every target page must have reference count one | `requalification` | Publish current-source correctness, sanitizer, Graph, and performance records under the ownership and typed-status contract |

## Attention dispatch

Plan creation fixes the selected algorithm.

| Operator | Selection rule | Graph evidence |
| --- | --- | --- |
| Paged decode | MHA selects direct. MQA and GQA select eight-warp token parallelism | None |
| Ragged prefill | Average KV length below 64 selects direct. Long MQA selects sixteen warps. Other long shapes select eight warps. GQA group size four with average KV length at least 256 selects tiled split-eight | Tiled long GQA4 only |
| Paged prefill | The caller selects direct, eight-warp, or sixteen-warp execution. Contract checks reject unsupported combinations | One historical fixed-address direct GQA4 fixture |

Ragged dispatch uses the batch average KV length. It does not group requests by
length or use a per-shape tuning database.

The 2026-08-07 paged-prefill token-parallel correctness and matched performance
records apply to source `8478ee9`. They do not qualify the merged DeviceRegion
submission path.

## KV append ownership

Paged attention may read shared physical pages. Fused append has a stricter
write contract:

- The caller supplies one reference count for every physical page.
- The host contract and device guard require count one for each target page.
- The engine or KV pager makes a shared tail private before enqueue.
- The caller updates the page table and counts before append.
- The caller keeps that metadata snapshot stable through completion.

The operator does not allocate pages, copy shared data, or remap requests.

The 2026-08-06 fused-append records predate this ownership contract. They
remain immutable historical measurements and do not qualify the current
source. See the [evidence index](results/README.md).

## Dynamic metadata

Host-resident metadata can return a `ContractError` before submission.
Paged decode, paged prefill, and fused append also validate device-resident metadata on the CUDA stream.
They return typed completion failures with the checked bindings.
Semantic rejection does not poison the queue or Graph.

## Engine interop

The source accepts leased external regions and an engine-owned CUDA stream for direct single decode and NHD or HND paged decode.
The queue supports bounded detached completions and returns stream-ordered engine authority after Loom queues the post-event wait.
The H20 gate uses a simulated engine.

An experimental paired-repository POC routes Mistral.rs decode attention through Loom.
Historical records ran Mistral.rs sources `9f6acf2a` and `805dc8f1` against Loom
`d27b6e5`. They show provider hits and matching selected token strings for one Qwen request.
They also show no adapter-issued device-to-device copy and typed recovery after one rejected metadata command.

The POC does not qualify production safety, general model coverage, full-model zero-copy execution, or performance.
Mistral.rs source `84602212` replaces the process-global runtime with model-owned state.
Its H20 record covers one Qwen request, typed rejection and reuse, and two concurrent drain callers.
That qualification is limited to one model, one H20, and one ordinary stream.
See the [Mistral.rs integration boundary](integrations/mistralrs.md).

## Target surface

| Family | Planned work |
| --- | --- |
| Attention | Sliding window, mixed-batch attention, MLA, and broader head dimensions |
| KV cache | Gather, scatter, compaction, remapping, FP8, and INT8 storage |
| Decode tail | Logits processing, penalties, Top-K, Top-P, Min-P, logprobs, and deterministic sampling |
| Speculation | Greedy, stochastic, and tree verification |
| MoE | Routing, permutation, grouped-GEMM inputs, and weighted combine |
| Quantization | Scale, pack, unpack, dequantize, and layout conversion |
| Matrix work | Qualified dense, quantized, and grouped GEMM providers |
| Communication | Collectives for measured tensor-parallel and expert-parallel workloads |

## Admission

Every new contract records its call site, tensors, numerical limit, baseline,
hardware, metric, and stop condition. Unsupported contracts return an error.
The project records each evidence level independently.
