# Components and external projects

OrbitKV is one Rust inference product assembled from seven owned crates and a
small number of explicit external boundaries. An imported component never gains
KV lifecycle authority.

## Owned components

| Component | Responsibility | Does not own |
| --- | --- | --- |
| `orbitkv` core | Compile attention-state semantics; own request/snapshot identity, page generations, Prefix/COW, retirement, publication, and reuse | Kernels, HTTP, network byte movement |
| `orbitkv-executor` | Lower manager plans, bind persistent tensor arenas, execute OrbitKV compiler graphs, move local or external bytes, and produce completion evidence | Page allocation, semantic liveness, final publication |
| `orbitkv-engine` | Logical request/event contracts, optional tokenization/HTTP adaptation, admission, batching, backpressure, cancellation, release, and execution coordination | A second cache index, allocator, or kernel runtime; protocol/frontend modules cannot name physical pages or device buffers |

The compiler subsystem is also owned: `orbitkv-compiler` provides graph and
search infrastructure, `orbitkv-ops` portable operation contracts,
`orbitkv-cuda` device implementations and execution, and `orbitkv-tracing`
diagnostics. These crates derive from Luminal and retain its licenses; see
[source ancestry](compiler-maintenance.md).

The model coordinator is implemented and real-device qualified for bounded
continuous batching. It compiles once at startup, merges fresh requests into
decode-first token-budgeted dispatches, streams through bounded event queues,
suppresses stop tokens, cancels at token boundaries, and drains manager state.
The `orbitkv-serve` binary now wires that coordinator into the optional vLLM
Rust HTTP frontend. A released-checkpoint H20 load qualification reaches eight
concurrent requests with complete fixed-length outputs; the remaining serving
gaps are fairness, long soak, the actual capacity failure point, and a matched
reference-engine comparison.

## External projects

The checked-in OrbitKV compiler/CUDA module boundaries, implemented fusion levels
and GPU search objective are detailed in [OrbitKV compiler design](compiler.md).

| Project | Relationship | Reused or planned surface | Excluded surface |
| --- | --- | --- | --- |
| vLLM | Embedded frontend, benchmark client and kernel/policy reference | Rust OpenAI/tokenizer/chat/SSE, request identity, stream-drop auto-abort; `vllm bench serve`; selected kernel and offload techniques below | vLLM scheduler or KV block manager in the OrbitKV process |
| SGLang | Kernel and scheduling reference | Packed recurrence, normalization/gating, expert dispatch, fixed-buffer weight prefetch | Python model runtime, a second state pool or radix/page manager |
| PegaInfer | Design and measurement reference only | Small Rust server boundary, hybrid full/linear-attention operator structure, matched vLLM-client workflow | Source copying, model-name dispatch, contiguous model-owned KV cache |
| Dynamo | Distributed-system reference and optional outer control plane | KV-aware routing ideas, event schemas, telemetry, service discovery where justified | `kvbm-logical`, `KvBlockManager`, lifecycle pins, or any second page manager |
| Mooncake | Planned external storage transport | Registered memory, Store objects, placement/lease observations, RDMA/TCP transfer completion | Local page allocation, generation, retirement, or publication |
| NIXL | Planned external data plane | Registered regions, asynchronous transfer descriptors, topology-aware movement | Dynamo KVBM ownership semantics |

The PegaInfer source reviewed at commit `14338f65` does not depend on Dynamo. Its
Rust crate directly uses Axum, Tokio, Hugging Face tokenizers, safetensors,
cudarc, and its own CUDA kernels. Its vLLM relationship is primarily the use of
`vllm bench serve` as a common client. OrbitKV should therefore learn hybrid
execution and benchmark discipline from PegaInfer, while studying Dynamo
directly for distributed routing and NIXL integration.

The Dynamo source reviewed at commit `7ee72a3a` exposes `dynamo-runtime`,
`dynamo-llm`, `kvbm-logical`, and physical/NIXL layers. `dynamo-llm` enables its
block manager by default, while `kvbm-logical` defines its own manager IDs,
lifecycle pins, pools, registry, eviction, and event pipeline. Those middle
layers conflict with OrbitKV authority and are intentionally not dependencies.

## Kernel and scheduling reuse

The following inventory was inspected on 16 September 2026. SGLang sources are
pinned to `095ec6c997bfdd25d3864cb0ce77a6562a934b96`; the installed vLLM 0.29.0
corresponds to source revision `98dff2a81d747d1dba01a47f939f48c3526d4206`. The inventory distinguishes
implemented source ports from migration candidates; it does not add providers
or qualified model support. Start with Qwen numerical correctness and
useful work elimination, then shared MoE/offload capabilities.

| Surface | Concrete upstream reference | Integration and acceptance |
| --- | --- | --- |
| Packed recurrent update | [SGLang FLA packed decode](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/kernels/ops/attention/fla/fused_recurrent.py) | Reuse direct state-pool indexing, QKV unpacking and update/readout tiling through a CUDA port or qualified Triton artifact. Preserve OrbitKV state layout, explicit slot indices, alias/effect proof and completion; eliminate gather/scatter only after those contracts exist. |
| Causal convolution | [SGLang causal-conv CUDA](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/kernels/aot/csrc/mamba/causal_conv1d.cu) | Borrow indexed state updates and launch layouts. Extract a native ABI if copying kernels; ATen/c10 wrappers are not a Rust-runtime dependency. Compare against the existing convolution candidate. |
| Activation and normalization | [Activation CUDA](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/kernels/jit/csrc/elementwise/activation.cuh), [fused add/RMSNorm CUDA](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/kernels/jit/csrc/elementwise/fused_add_rmsnorm.cuh) | Vectorized SiLU × up is ported as described below; add/RMSNorm remains a candidate. `kRoundActivation` and `kCastXBeforeOutMul` select different numerical contracts, not free tuning knobs; match the actual graph's casts. Remove TVM FFI wrapper dependencies. |
| MoE dispatch and expert GEMM | [SGLang align CUDA](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu), [vLLM DeepGEMM experts](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py) | Reuse token/expert sorting, padding and combine dataflow. Extend the existing DeepGEMM adapter for grouped GEMM; validate routing weights, permutation, scale layout, empty experts and padded rows independently. |
| Weight prefetch | [vLLM fixed-buffer offloader](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/offloader/prefetch.py), [SGLang offloader](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/srt/utils/offloader.py) | Reimplement bounded stable buffers, copy-stream dependencies and completion-safe slot reuse in the executor. Start with a static residency policy; Torch hooks and module patching do not transfer. |
| Admission and prefill | [vLLM scheduler](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/sched/scheduler.py) | Apply measured token-budget, chunked-prefill and pressure policies to the existing engine. Decode-first continuous batching already exists; test fairness/cancellation/pressure without importing another scheduler or allocator. |
| Residual mixing | [SGLang mHC](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/kernels/ops/layernorm/mhc.py) | A TileLang artifact or compatible library entry can implement admitted multi-output semantics. Confirm each checkpoint's mHC variant and H20 target before choosing an implementation. |

The pinned DeepGEMM source has grouped FP8 GEMM APIs, but not the
`fp8_mega_moe`/`mega_moe_pre_dispatch_sm90` APIs used by SGLang's newer
[SM90 Mega-MoE wrapper](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/srt/layers/moe/mega_moe_sm90.py).
Extend the pinned grouped path first. A provider upgrade requires source/dependency
pins and numerical, resource and full-model qualification; a Python import is
not evidence that the current provider implements it.

The inspected trees contain DeepSeek V4/mHC/sparse-MLA components, but no identified
V4.1 CSA2/Engram or `glm5_next` model path. FP8 sparse MLA is not the new FP4-KV
contract. These components need checkpoint-specific semantic verification before
they can establish GLM-5.3-Flash or DeepSeek-V4.1-Flash execution.
The newer official FlashMLA source has a separate
[integration assessment](attention-providers.md#next-backend-integrations): it
includes V4.1 kernels, but its V4.1 decode path requires SM100.

Keep the original license and authorship for each copied file, not just the
engine's top-level license: FLA-derived code retains MIT notices and
causal-conv1d has BSD-3-Clause provenance; engine-owned code is generally
Apache-2.0. Record exact source and wrapper identity in the existing provider
inventory when a native integration is added. Source ports live with their owning
kernel and provenance; Triton/TileLang are not yet integrated providers.

### Ported vector activation

[`KernelSiluMul`](../crates/orbitkv-cuda/src/kernel/silu_mul.rs) adapts SGLang's
16-byte vector loads/stores, flattened work assignment and intermediate BF16
activation rounding. It accepts separate gate/up inputs, as produced by the
current MLP. Egglog admits it only for the matching F32 primitive chain, BF16
activation materialization and contiguous two-dimensional BF16 inputs with a
width divisible by eight. Symbolic row counts require a positive lower bound
and a proved dense physical span; external pointers
without vector alignment use scalar memory accesses inside the same kernel.

SGLang's `expf` formulation is replaced with the graph's existing F32
`exp2`/reciprocal/multiply sequence. The intermediate BF16 cast is retained;
neither mathematical SiLU equivalence nor an upstream option permits changing
rounding. This candidate does not match the separate all-BF16 SwiGLU contract.
Selection remains an egglog alternative, with no checkpoint-name dispatch or
Rust graph rewrite. The source's [NOTICE](../crates/orbitkv-cuda/src/kernel/silu_mul/NOTICE)
records the revision, copied structure and adaptations, with the upstream
Apache-2.0 license alongside it. This source port alone establishes no complete
model correctness or serving-performance improvement.

H20 checks cover all 65,280 finite BF16 gate values against the original GPU
graph, width 17,408 with 1/8/32 rows, and unaligned external buffers. Frozen
Qwen layer-0 activation/product and residual/MLP probes are bitwise equal for
prefill and decode at C1/C8. Four of those eight selected programs use the new
kernel; the other programs retain existing alternatives. These probes isolate
one layer with identical reference inputs and do not close full-model gates.

## Recurrent kernel algorithm reference

The register-resident delta scan follows the value-axis tiling and retained-state
approach in [vLLM's FLA-derived recurrent implementation](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py).
That source credits Songlin Yang and Yu Zhang and retains the original MIT notice
within vLLM's Apache-2.0 source. OrbitKV's CUDA implementation does not copy the
Triton kernel: it preserves its existing `[K,V]` state layout, F32 operation order,
packed output and manager-owned lifecycle. Egglog admits the alternative only for
positive key widths at most 128; the original scan remains available. Direct
state-slot writes and the upstream `[V,K]` layout are not part of this candidate.

[`KernelDeltaGather`](../crates/orbitkv-cuda/src/kernel/sequence_state/delta_gather.rs)
extends that register implementation with direct indexed reads from the state
arena. Egglog absorbs an F32 `Gather` feeding the scan, retaining both index and
data view addressing. The shared CUDA template changes only the initial-state
load; recurrence arithmetic, packed token/state output and explicit Scatter
commit remain the same. The candidate supports the existing ragged scan,
including empty requests and repeated read indices.

When no other consumer needs the selected-state tensor, extraction eliminates
its Gather launch and F32 intermediate allocation. The index tensor remains
available to the commit. This is not compact slot-descriptor lowering or an
in-place recurrent update. The arena is a visible read dependency, and existing
alias validation still rejects unordered old-state readers combined with a
mutating commit. No new state planner or executor binding API is introduced.

## Dependency rule

Every future integration must fit one of two directions around the sole
authority:

```text
external routing/events/telemetry
                |
                v
        OrbitKV RuntimeSession
                |
       +--------+---------+
       |                  |
       v                  v
OrbitKV compiler execution   ExternalKvTransport
                          |
                    Mooncake / NIXL
```

No adapter may mint a `PageLease`, advance semantic death, infer completion
from enqueue, or recycle a generation.
