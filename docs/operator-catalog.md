# Operator catalog

This catalog projects the current Rust API, CUDA providers, and recorded
evidence. Source decides what exists. Evidence decides what passed. The
roadmap does not change either state.

## State model

| State | Meaning |
| --- | --- |
| `current` | The product source implements the stated contract. |
| `experimental` | Source exists, but promotion gates remain open. |
| `requalification` | A source, contract, provider, or ownership change occurred after the last record. |
| `planned` | The roadmap admits the work. No public module or provider exists. |

Graph, performance, engine, and serving evidence remain separate from these
states.

## Framework lifecycle

Every admitted operator converges on one lifecycle:

```text
Spec -> Provider -> Algorithm -> Plan -> Operands -> CommandScope -> Completion
```

The source still uses `*Args` for some invocation types. Framework migration
renames them to `*Operands` without forwarding aliases. Target names in this
document do not claim that migration is complete.

## Family summary

| Family | State | Admitted or next contract |
| --- | --- | --- |
| `attention` | Current | Single decode, paged decode, ragged prefill, and paged prefill |
| `gemm` | Current and experimental | cuBLASLt BF16 dense is current. Native SM90a M=1 GEMV is experimental. |
| `kv_cache` | Requalification | Fused RoPE plus paged append with exclusive target pages |
| `normalization` | Requalification | F32, FP16, and BF16 RMSNorm |
| `position` | Requalification | BF16 D128 NeoX RoPE with explicit I32 positions |
| `activation` | Planned | SwiGLU or another engine-observed gated activation |
| `sampling` | Planned | Logits processing, Top-K, Top-P, Min-P, logprobs, and deterministic RNG |
| `speculation` | Planned | Draft verification and token compaction |
| `quantization` | Planned | Scale, packing, conversion, and dequantization |
| `moe` | Planned | Routing, permutation, grouped-GEMM inputs, and combine |
| `communication` | Planned | Measured tensor-parallel or expert-parallel collectives |

The project does not create an empty namespace for a planned family.

## Namespace migration

| Implemented domain | Current public area | Final family |
| --- | --- | --- |
| Decode and prefill | `attention` | `attention` |
| Fused paged append | `attention::paged_append` plus CUDA `rope` | `kv_cache::paged_append` |
| Dense BF16 GEMM | `gemm` | `gemm`, then `gemm::dense` when another GEMM contract exists |
| RMSNorm | `rms_norm` | `normalization::rms_norm` |
| Standard RoPE | `rope` | `position::rope` |

The migration moves source directly. It adds no compatibility module.

## Current and experimental operators

| Operator | Contract and algorithm | Source state | Evidence boundary |
| --- | --- | --- | --- |
| RMSNorm | Contiguous F32, FP16, and BF16. Scalar and packed algorithms. | Current source under `rms_norm`; family migration pending | Current R1 device correctness, lifecycle, and sanitizer for the permanent runner; no standalone Graph. |
| Dense BF16 GEMM, vendor | Contiguous `D=A*W^T`, BF16 storage, F32 accumulation, explicit `CublasLtHeuristic` | Current | Current R1 device correctness, RMSNorm-to-GEMM Graph, and sanitizer. |
| Dense BF16 GEMM, native | Same `Spec`, explicit `OxideSm90SimtGemvM1N16K64`, zero workspace | Experimental, performance-stopped | Current R1 device correctness, five Graph shapes, and sanitizer. The R2 matched comparison passed both baselines on only one of five shapes; SASS and engine gates were not run after the stop. |
| Single decode | BF16 NHD D128, direct MHA, MQA, and GQA | Current | Current R1 device correctness, lifecycle, and sanitizer. No Graph or current performance claim. |
| Single decode split-K | Explicit partitions and caller-owned F32 workspace | Current | Current R1 runner covers declared MQA and GQA shapes plus sanitizer. MHA, Graph, and current performance remain open. |
| Paged batch decode | BF16 NHD or HND D128, page size 16. Direct MHA and eight-warp MQA or GQA. | Current | Current R1 device correctness, rejection Graph, simulated-engine boundary, and sanitizer. Current R2 matched eager performance covers six shapes; valid-output Graph remains open. |
| Ragged causal prefill | BF16 NHD D128, bottom-right causal mask. Direct, eight-warp, sixteen-warp, and tiled GQA4. | Current | Current R1 device correctness, tiled GQA4 Graph, and sanitizer. Current R2 matched eager performance covers three shapes; the long GQA4 row has source-bound optimization evidence. |
| Paged causal prefill | BF16 NHD D128, page size 16. Caller selects direct, eight-warp, sixteen-warp, or tiled GQA4 with F32 workspace. | Current | Current device correctness, long tiled valid-output Graph, rejection Graph, sanitizer, and matched eager performance. |
| Standard RoPE | BF16 NHD D128, NeoX split-half, explicit I32 positions | Current source under `rope`; family migration pending | Current R1 device correctness and sanitizer; no standalone RoPE Graph. |
| Fused RoPE plus paged append | BF16 NHD D128, page size 16, one through 64 tokens, exclusive target pages | Current | Current R1 device correctness, six-token valid-output Graph, rejection Graph, and sanitizer. |

## Attention plan policy

Plan creation fixes one algorithm.

| Operator | Current selection | Current Graph boundary |
| --- | --- | --- |
| Paged decode | MHA selects direct. MQA and GQA select eight-warp token parallelism. | Current rejection-only invalid-page Graph; no valid-output Graph |
| Ragged prefill | Average KV length below 64 selects direct. Long MQA selects sixteen warps. Other long cases select eight warps. Long GQA4 can select tiled split-four. | Current tiled long-GQA4 valid-output Graph |
| Paged prefill | Caller selects direct, eight-warp, sixteen-warp, or tiled GQA4. Tiled GQA4 requires group size four and explicit workspace. | Current tiled long-GQA4 valid-output Graph and invalid-page rejection Graph |

Ragged selection uses batch-average KV length. It has no request grouping or
persistent tuning database. Enqueue does not change the chosen algorithm.

## Dense GEMM providers

Both providers use one `Bf16DenseGemmSpec`, plan type, operands type, command
path, completion, and Graph path.

| Provider | Algorithm | State | Role |
| --- | --- | --- | --- |
| `CublasLt` | `CublasLtHeuristic` | Current | General vendor BF16 dense baseline |
| `Oxide` | `OxideSm90SimtGemvM1N16K64` | Experimental, performance-stopped | Native cuda-oxide M=1 algorithm retained for evidence; not selected for production |

`GemmPlanner` accepts explicit provider selection. Unsupported native shapes
return a planning error. Enqueue does not switch to cuBLASLt.

The native algorithm admits this exact contract:

- `M=1`
- `N % 16 = 0`
- `K % 64 = 0`
- contiguous row-major BF16 `D=A*W^T`
- no post-operation
- four-byte alignment
- zero workspace
- H20 with an `sm_90a` artifact

One untimed Qwen2.5-1.5B census recorded 1,184 matching calls across five
logical shapes. They represent 87.574% of calls and 16.708% of FLOPs in that
single-request workload. This census is workload evidence, not performance
evidence.

The [experimental contract](development/sm90-simt-gemv-m1.md) defines both
baselines, promotion gates, and stop conditions.

## Workspace ownership

Each plan declares exact workspace bytes and alignment. The caller allocates
and binds that workspace through operands. A provider cannot allocate hidden
workspace during enqueue.

Split-K attention owns caller-visible F32 partial state. Token-parallel paged
attention uses block-local state and needs no caller workspace. The native M=1
GEMV plan declares zero workspace.

## Page-table ownership

Paged attention and paged append receive page tables as operands. The engine
or KV pager owns allocation, sharing, copy-on-write, eviction, and remapping.

Paged attention accepts shared read-only physical pages. Paged append requires
reference count one for every target page. The caller keeps the validated page
table and reference-count snapshot stable through completion.

The append operator does not allocate pages, copy shared tails, or remap
requests.

## Dynamic metadata

Paged decode, paged prefill, and fused append validate device metadata on the
CUDA stream. A semantic rejection returns a typed completion error, preserves
outputs, and returns checked bindings. It does not poison the queue or Graph.

## Architecture support

| Architecture | State | Admitted boundary |
| --- | --- | --- |
| `sm_90a` | Current first target | Architecture-specific native artifacts and provider gates; current evidence was recorded on H20 |
| `sm_100a` | Planned | Separate Blackwell algorithms and evidence |
| `sm_120` | Planned | Separate consumer Blackwell algorithms and evidence |

No architecture target inherits qualification from another target. TMA,
WGMMA, and tcgen05 matrix operations require named algorithms and independent
evidence.

## Engine interop

The source accepts leased external regions and an engine-owned CUDA stream for
direct single decode and NHD or HND paged decode. A simulated-engine device gate
covers bounded in-flight work, typed rejection, stream order, and lease
retention.

Historical Mistral.rs records show one Qwen decode path, provider hits, matching
selected token strings, and no adapter-issued device copy. Those source pairs
use the former project name. They do not qualify renamed source, general model
coverage, production recovery, or performance.

## Planned contracts

| Family | Admission input | First required proof |
| --- | --- | --- |
| Attention | Measured engine shape and mask contract | Host reference and one named CUDA algorithm |
| KV cache | Pager ownership protocol | Copy-on-write and metadata-lifetime proof before device code |
| GEMM | Workload census and both baselines | One exact dense, grouped, or quantized contract |
| Normalization | Model call site absent from current RMSNorm | Numerical contract and independent reference |
| Position | Model layout or dimension demand | Position semantics and reference vectors |
| Activation | Measured unfused engine call | Standalone reference before any fusion experiment |
| Sampling | Exact distribution and RNG state contract | Deterministic replay and statistical test plan |
| Speculation | Engine draft/target state machine | Accepted-token and RNG commit semantics |
| Quantization | Model format and quality budget | Scale ownership, packing format, and error bound |
| MoE | Engine routing trace | Stable routing, permutation, and combine contract |
| Communication | Measured distributed workload | Collective ordering, failure, topology, and baseline contract |

## Admission rule

Every new contract records its call site, tensors, numerical limit, provider,
algorithm, hardware, metric, and stop condition. Unsupported combinations
return errors. A planned row does not authorize an empty API or source module.
