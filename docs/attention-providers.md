# Attention contracts and provider admission

The model graph describes scaled dot-product attention and an explicit KV view.
CUDA providers supply legal implementations through egglog. FlashInfer exposes
CUDA-core decode and tensor-core attention as separate algorithms. The optional
FlashAttention provider uses pinned upstream FlashAttention-3 CUDA kernels.
The former handwritten native attention implementation and experimental flag
have been removed.

## Implemented boundary

| Owner | Contract |
| --- | --- |
| `orbitkv-ops` `AttentionSpec` | Query/KV head counts, independent QK and value dimensions, dtype, explicit scale and logical visibility |
| `orbitkv-ops` `AttentionInputs` | Packed queries, query segmentation and a typed KV view |
| `orbitkv-ops` `KvView::Paged` | Separate K/V allocations, page size, NHD/HND element order, page traversal metadata and external state class |
| CUDA `AttentionProviderCapabilities` | Admitted target families and complete dtype/dimension/layout/phase/page/algorithm combinations |
| Provider `HostOp` | Native preparation, launch ABI, captured resources, workspace ownership and output layout |
| OrbitKV | Page identities, CSR contents, visibility, state updates, generations, completion, publication and retirement |

`attention-op` carries mathematical semantics; `paged-kv-view` carries storage
facts. `persistent-state-attention-op` preserves the connection to the manager's
state class. Capability records generate `cuda-attention-provider` eligibility
facts through egglog rules. Separate provider rewrites add equivalent executable
nodes. There is no Rust pass that recognizes a model or rewrites its LLIR.

The graph API rejects invalid scales, mismatched head geometry, cross-graph
tensors, noncontiguous views, inconsistent metadata shapes and statically known
partial/mismatched pages before inserting an operation. Query and context sizes
come from tensors instead of a second, potentially inconsistent geometry record.
The state owner still validates runtime metadata and allocation authority;
provider preparation checks live buffers and resources.

Query-token count, context-page count and request count survive provider lowering
as expressions. Retained buckets evaluate those expressions with their own
allocation dimensions; they must not derive request geometry from CSR lengths
left installed by another profiling bucket. FlashInfer still accounts for the
physical K/V buffer capacities, and execution requires both CSR pointers to have
exactly `requests + 1` entries and last-page lengths to have `requests` entries.
The structural path without explicit CSR can describe one-query-per-request
decode or single-request prefill; packed multi-request prefill needs explicit
segmentation.

The semantic vocabulary includes unmasked attention and unequal QK/V dimensions,
and the view vocabulary describes both NHD and HND order. Current CUDA adapters
admit only causal/sliding, equal-dimension, NHD combinations. Defining the other
contracts does not make them executable. Contiguous/ragged and latent KV views
are not implemented variants. The output remains `[heads, query_tokens, value_dim]`;
this refactor does not add a layout conversion.

## Current CUDA implementations

| Implementation | Role and admitted scope |
| --- | --- |
| FlashInfer CUDA-core decode | One query per request; F32 at head dimensions 64/128/256, F16/BF16 at 64/128/256/512; SM8+ admission |
| FlashInfer tensor-core attention | F16/BF16 at 64/128/256/512, packed prefill and one-query decode; SM8+ admission |
| FlashAttention-3 | Optional SM90 F16/BF16, equal head dimensions 64/128/256, NHD paged K/V, causal/sliding decode and packed prefill |
| cuBLASLt | General/batched matrix products and supported scale, bias and activation epilogues; additional restricted tensor-scaled FP8 equivalents |
| DeepGEMM | Current SM90 block-scaled FP8 dense-linear contract and tile variants; optional shared activation preparation |
| Generated CUDA | Primitive and specialized operations, state updates and legal fused regions |

All attention candidates consume the same seven logical inputs and produce
`[heads, query_tokens, dimension]`. Capability admission is separate from local
source/toolkit availability and actual device qualification.

FlashInfer's tensor-core paged-prefill kernel also supports decode with one
query per request, as in its upstream `use_tensor_cores` path. The serialized
algorithm, plan cache and capture identity distinguish it from CUDA-core decode.
A request's phase establishes legality; it does not silently change the selected
algorithm at launch time.

### FlashAttention adapter

The provider uses [FlashAttention-3's native forward launch templates](https://github.com/Dao-AILab/flash-attention/blob/8d3a3b80d4758ebde5a867c50d24d4351443cf2b/hopper/flash_fwd_launch_template.h),
with pinned CUTLASS headers, through a small C ABI. PyTorch and the vLLM/SGLang
Python backend classes are not runtime dependencies. `jit.rs` resolves and hashes
the source and compiles the adapter; `plan.rs` owns pointer-free scratch planning
and captured allocations; `wrapper.cu` prepares metadata and launches the
upstream scheduler/kernel; `paged_attention.egg` adds the equivalent candidate.

OrbitKV's compact CSR page rows become a provider-owned rectangular block table
and actual KV lengths on the GPU. Conversion, scheduler preparation and attention
launches all participate in measured execution and CUDA Graph replay. K/V payloads
retain their existing allocations. The current safe table width is the total
number of compact page references, so metadata storage is O(requests × references).
The resource plan charges this storage, scheduler arrays and LSE; retained graphs
own their scratch even after the operation prepares another shape.

The initial algorithm uses non-TMA paged loads, packed GQA and one unsplit KV
partition. Upstream tile selection remains intact. Split-KV/TMA algorithms, FA2,
FA4, quantized KV and latent attention require separate admitted adapters; this
integration does not claim their coverage or a whole-model speedup.

Prepare sources explicitly before compilation:

```sh
cargo run -p orbitkv-cuda --bin providers -- fetch flashattention
```

Alternatively set `ORBITKV_FLASHATTENTION_DIR` to a complete checkout with its
pinned CUTLASS dependency. Model compilation performs no network fetch.
`ORBITKV_CACHE_DIR` controls the shared source/library cache root. The exact
provider and CUTLASS commits live in `crates/orbitkv-cuda/providers.lock.json`. Source,
wrapper and ABI content determine provider identity; compiler/flags determine
additional native-library cache identity, following the common provider policy.

Decoder artifacts retain request-count expressions in the FlashInfer operator
ABI and bind the provider lock. Artifacts from before the backend reorganization
require fresh search. Historical qualification directories remain immutable.
External `.so` libraries are cached separately from generated CUDA module images.

## Qualification

Independent CPU softmax references live under CUDA `tests/unit/providers/attention`.
They exercise each admitted algorithm directly, including non-power-of-two GQA,
ragged queries, permuted pages and sliding visibility. Separate tests cover
semantic extraction, saved-schedule replay, changing CSR contents and captured
allocation lifetime. See the current [provider qualification](validation/provider-kernels-20260914/README.md)
for actual device/model gates and limits; the preceding [attention contract result](validation/attention-contract-20260914/README.md)
documents the earlier implementation and schema.

## Reference engine designs

The following upstream source was reviewed on 2026-09-14. These are architectural
references, not claims that every listed combination works on every platform.

| Engine | Implementation structure | Representative integrations |
| --- | --- | --- |
| PegaInfer | Model-owned Rust execution uses shared native kernels/FFI; CUDA is compiled by its build pipeline, with Triton AOT and TileLang generation for selected paths | cuBLAS/cuBLASLt, FlashInfer, DeepGEMM, FlashMLA; DeepEP serves expert communication. See its [kernel manifest](https://github.com/pegainfer-project/pegainfer/blob/main/pegainfer-kernels/Cargo.toml), [build pipeline](https://github.com/pegainfer-project/pegainfer/blob/main/pegainfer-kernels/build.rs) and [submodules](https://github.com/pegainfer-project/pegainfer/blob/main/.gitmodules). |
| vLLM | An attention backend exposes implementation, metadata builder, cache layout and capability validation; compiled native implementations coexist with Triton paths | FlashAttention, FlashInfer, FlashMLA and Triton attention in the [registry](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py); [DeepGEMM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py) and [CUTLASS](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py) also provide quantized linear implementations. |
| SGLang | Its attention interface separates decode/extend execution and metadata preparation outside/inside graph capture, including hybrid wrappers | FlashInfer, FlashAttention, FlashMLA and Triton appear in the [attention registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py). Its [FP8 implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py) also integrates DeepGEMM, CUTLASS and Marlin. See the [attention base interface](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py) for metadata ownership. |

vLLM's [backend interface](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py)
checks more than head size: dtype, cache encoding, block geometry, visibility,
target and feature combinations affect admission. This motivates our explicit
capability boundary. PegaInfer's model-owned scheduling is a different design
choice from OrbitKV's shared semantic compiler; adopting its native launch/AOT
techniques does not require copying its per-model execution organization.

Our additional requirement is that legal alternatives enter one compiler search
space and are measured on compatible deployment paths. A backend registry alone
does not provide joint layout search or synthesize fused attention internals.

## Candidate families and FlashMLA sequencing

Candidate membership follows semantics and storage ABI, not a library name:

| Computation and view | Possible candidates | Current integration |
| --- | --- | --- |
| Dense causal/sliding attention with separate paged K/V | FlashInfer, a compatible FlashAttention adapter, direct CUDA/Triton/TileLang implementations | FlashInfer algorithms and FlashAttention-3 |
| Dense attention over contiguous/ragged K/V | Compatible FlashAttention/FlashInfer kernels and generated tiled algorithms | View/adapters remain future work |
| Latent/MQA realization of MLA | Compatible FlashMLA, FlashInfer MLA or another independently qualified implementation | No latent execution contract/provider yet |
| Indexed sparse attention | Kernels accepting the exact index, cache-encoding and visibility contracts | Not admitted |
| Recurrent/linear attention such as GDN | Scan/recurrent algorithms and convolution/norm/gate regions | Separate state-transition operations, not softmax-attention equivalents |

FlashMLA now includes several families: dense MLA decode, sparse prefill/decode,
and SM100 dense MHA prefill. Its documented requirements include CUDA 12.8+
(12.9+ for SM100); device and geometry support vary by kernel. Consequently,
the SM100 MHA implementation does not supply an H20 GQA replacement. See the
upstream [support matrix and APIs](https://github.com/deepseek-ai/FlashMLA#requirements).

Integrate the first FlashMLA implementation when these prerequisites are ready:

1. A normalized MLA calculation and explicit latent/positional cache components,
   with matching state byte geometry and read/write semantics.
2. A pinned native adapter for a supported kernel, including its block/index
   metadata, scheduler preparation, workspace, output/LSE convention and target.
3. Independent numeric and next-state tests, followed by strict artifact replay,
   mixed phase transitions and final state drain.

An isolated MLA operator witness can establish these contracts before an entire
MoE or multi-device decoder is ready. Dense prefill may use an expanded form and
decode a latent form, provided their equivalence and shared persistent ABI or
explicit conversion are validated. The current 27B GQA/GDN workload does not
justify adding a latent-attention dependency by itself.

After this boundary, use current workload traces to prioritize quantized
projection preparation, GDN/generated regions and compile/search cost. New
attention families and megakernels need their own measured motivation.
