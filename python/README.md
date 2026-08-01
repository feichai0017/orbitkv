# Loom Kernels · Python

Native current-stream PyTorch operators and narrow, opt-in vLLM 0.24/0.25
integration for [Loom Kernels](https://github.com/feichai0017/loom-kernels).

[Project README](../README.md) · [Integration guide](../docs/guides/vllm-ir-provider.md) · [Operator catalog](../docs/operator-catalog.md)

> [!IMPORTANT]
> The bridge-ABI-11 native wheel is H20-qualified but is not published to a
> package index. It includes all nineteen semantic operators, including
> SiLU-and-Mul-to-dynamic-INT8,
> direct plus persistent-vLLM explicit-state categorical sampling and the
> optional-residual RMSNorm-to-FP8/INT8 schemas. A
> source-only wheel is intentionally unsupported:
> `pip wheel ./python` fails unless `build_wheel.py` has staged both native
> libraries and their manifest.
>
> RMSNorm-to-dynamic-INT8 remains an explicit experimental engine route: wheel
> qualification proves distribution and framework compatibility, not exact
> model output, default admission, or a stable performance win.
>
> Current source is bridge ABI12 with twenty-one operators and no ABI11
> compatibility shim. It adds `moe_permute` and `moe_combine` around vendor
> grouped GEMM. H20 source matrices, direct movement gates, and an explicit
> vLLM 0.25.1 Cutlass engine-admission gate pass; an ABI12 clean-wheel matrix
> and production-workload MoE performance gate remain open. The qualified
> distributable artifact is still the ABI11 predecessor above.

## Qualified artifact

The current matrix row is:

| Axis | Qualified value |
| --- | --- |
| Artifact | `py3-none-linux_x86_64` |
| CUDA build | toolkit 13.1, `sm_90` |
| PyTorch runtime | `>=2.10,<2.12` through a 2.10 Stable ABI target |
| Python runtime tested on H20 | 3.11 |
| vLLM extra | `>=0.24,<0.26` |
| Native payload | `libloom_cuda_bridge.so`, `libloom_kernels_torch.so` |

The qualified artifact's build tag encodes bridge ABI 11:
`11cu131torch210sm90`. The exact H20 artifact, binary audit, and three
repository-free clean-install gates are recorded in the
[native-wheel evidence](../docs/results/h20-native-wheel-clean-install-abi11-20260801.json).
The same wheel passes 342 tests with each supported vLLM minor and 231
applicable tests in the vLLM-free PyTorch 2.10 environment. It includes static
FP8 E4M3 KV quantize-on-write, sparse token penalties, sampled-token plus
top-k logprobs, exact top-k/top-p paths, fused logits preprocessing, and
deterministic categorical sampling with persistent request-owned state, plus
optional-residual RMSNorm-to-FP8/INT8. It is bound to source revision
`afc54c46e3607d0d09f2860e0805f02dead88915` and adds the exact
SiLU-and-Mul-to-INT8 API plus explicit compiler route.

The preceding ABI7 refresh from revision
`f98a9311c8b204c02fa77da10a768c54de3d08db` packages the final FP8 KV adapter
and passes the complete 286-test vLLM 0.24 H20 suite plus 22 focused adapter
tests from a fresh environment. It is retained as historical evidence. See the
[refresh evidence](../docs/results/h20-native-wheel-clean-install-abi7-refresh-20260727.json).

The older `10cu131torch210sm90` ABI-10, `9cu131torch210sm90` ABI-9,
`8cu131torch210sm90` ABI-8,
`6cu131torch210sm90` ABI-6,
`5cu131torch210sm90` ABI-5, `4cu131torch210sm90` ABI-4,
`2cu131torch210sm90` ABI-2, and `1cu131torch210sm90` ABI-1 wheels remain
historical evidence only.
`build_wheel.py` uses the ABI-specific tag so incompatible bridge signatures
cannot overwrite or masquerade as one another. No native Python wheel has been
published.

## Install a built wheel

The wheel has a hard PyTorch dependency because its dispatcher is not useful
without PyTorch. vLLM and tests remain explicit extras:

```bash
python3 -m venv .venv-loom
.venv-loom/bin/pip install \
  'dist/loom_kernels-1.0.0a1-11cu131torch210sm90-py3-none-linux_x86_64.whl[test]'

# Add the supported vLLM integration when needed.
.venv-loom/bin/pip install \
  'dist/loom_kernels-1.0.0a1-11cu131torch210sm90-py3-none-linux_x86_64.whl[vllm,test]' \
  'vllm>=0.24,<0.26'
```

No repository checkout, `PYTHONPATH`, `LD_LIBRARY_PATH`, or external library
override is used at runtime. The installed package reads
`loom_kernels/lib/native.json`, validates the PyTorch range and bridge ABI,
verifies both library hashes, and loads only its packaged dispatcher.

```python
import loom_kernels

print(loom_kernels.native_build_info())
```

## Build the matrix wheel

Use a clean Linux x86_64 checkout with Cargo, CUDA, ELF inspection tools, and a
CUDA-enabled PyTorch build:

```bash
python3 -m venv .venv-wheel
.venv-wheel/bin/pip install \
  'setuptools>=80,<82' 'wheel>=0.45' build 'torch>=2.10,<2.12'

CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  .venv-wheel/bin/python python/build_wheel.py \
  --cuda-home /usr/local/cuda-13.1 \
  --archs 90 \
  --wheel-dir dist
```

`build_wheel.py` is the only binary-wheel entrypoint. It builds the Rust CUDA
bridge, builds the boxed LibTorch Stable ABI dispatcher, rejects ATen/c10 C++
and raw CUDA-launch dependencies, verifies `$ORIGIN` loading, writes the
revision/toolkit/SM/runtime manifest, and checks the final archive contains
exactly the two Loom `.so` files. The current checkout emits an ABI12-tagged
artifact that has not completed clean-install qualification. The exact
`afc54c4` ABI11 predecessor passes the repository-free matrix, has SHA256
`20402f02c44f17646c45b71ae279c702748458ad5de5b970570f0c3ce314f3c6`, and
remains unpublished.

## Source development

Editable source work remains available without creating a distributable
source wheel:

```bash
python3 -m venv .venv-dev
.venv-dev/bin/pip install -e 'python[test]'

CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  .venv-dev/bin/python python/build_native.py
CUDA_HOME=/usr/local/cuda-13.1 \
  .venv-dev/bin/python python/build_torch_extension.py
```

Source checkouts discover the paired libraries only under repository
`build/`. Installed wheels discover them only under `loom_kernels/lib/`.
Every operator, including padded logits and strided paged-cache views, enters
checked borrowed Rust dispatch. There is no ctypes, ATen dispatcher twin, or
direct raw-CUDA framework path. Both source libraries must be rebuilt together:
the current ABI12 dispatcher rejects an ABI11 bridge instead of retaining a
compatibility shim.

## Direct PyTorch use

```python
import torch

from loom_kernels import (
    apply_token_penalties_,
    categorical_sample,
    greedy_sample_logprobs,
    greedy_speculative_verify,
    logits_preprocess_,
    min_p_filter_,
    moe_combine,
    moe_permute,
    rms_norm_dynamic_fp8,
    rms_norm_dynamic_int8,
    rope_paged_kv_write_,
    selected_token_logprobs,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_int8,
    token_penalties_workspace_capacity,
    top_k_filter_,
    top_p_renorm_,
    topk_sampled_logprobs,
)

fp8_output, block_scales = silu_and_mul_dynamic_fp8(
    gate_and_up_bf16,
    group_size=128,
)
int8_output, token_scales = rms_norm_dynamic_int8(
    hidden_bf16,
    norm_weight_bf16,
    epsilon=1.0e-6,
    residual=residual_bf16,
)
mlp_int8_output, mlp_token_scales = silu_and_mul_dynamic_int8(
    gate_and_up_bf16,
)

permuted, expert_offsets, inverse, assignment_ids = moe_permute(
    hidden_bf16,
    topk_expert_ids_i32,
    num_experts=64,
)
expert_outputs = engine_grouped_gemm(permuted, expert_offsets)
moe_output = moe_combine(
    expert_outputs,
    routing_weights_f32,
    inverse,
    expert_offsets,
)

token_ids, logprobs, ranks = greedy_sample_logprobs(logits)
logits_preprocess_(
    sampling_logits_f32,
    temperatures_f32,
    blocked_token_mask,
    bias_row_ids_i32,
    bias_token_ids_i32,
    bias_values_f32,
    suppressed_row_ids_i32,
    suppressed_token_ids_i32,
)
logprobs, ranks = selected_token_logprobs(logits, sampled_ids_i64)
topk_ids, topk_logprobs, sampled_ranks = topk_sampled_logprobs(
    logits, sampled_ids_i64, top_k=20
)
top_k_filter_(sampling_logits_f32, per_row_top_k_i32)
sampling_probabilities_f32 = top_p_renorm_(
    sampling_logits_f32, per_row_top_p_f32
)
rng_state_i64 = torch.stack(
    (per_row_seed_i64, per_row_counter_i64),
    dim=1,
)
sampled_ids_i64 = categorical_sample(
    sampling_probabilities_f32,
    rng_state_i64,
)
verified_ids, accepted_lengths, emitted_lengths = greedy_speculative_verify(
    flattened_draft_ids_i32,
    flattened_target_argmax_ids_i64,
    bonus_ids_i32,
    inclusive_cumulative_draft_lengths_i32,
    max_draft_tokens,
)
min_p_filter_(sampling_logits_f32, min_p_f32)
penalty_workspace = torch.empty(
    (
        sampling_logits_f32.shape[0],
        token_penalties_workspace_capacity(
            prompt_token_ids_i64.shape[1],
            output_token_ids_i64.shape[1],
        ),
    ),
    device=sampling_logits_f32.device,
    dtype=torch.int64,
)
apply_token_penalties_(
    sampling_logits_f32,
    prompt_token_ids_i64,
    output_token_ids_i64,
    presence_penalties_f32,
    frequency_penalties_f32,
    repetition_penalties_f32,
    penalty_workspace,
)

# Native caches ignore the scale values. FP8 uint8 caches use either one
# calibrated F32 scale or one scale per KV head.
cache_scales = torch.ones(1, device=query.device, dtype=torch.float32)
rope_paged_kv_write_(
    query,
    key,
    value,
    positions_i64,
    cos_sin_cache,
    packed_kv_cache,
    cache_scales,
    cache_scales,
    slot_mapping_i64,
    is_neox=True,
)
```

All CUDA calls use PyTorch's current stream. Out variants accept caller-owned
buffers for capture-safe reuse. Public APIs are inference-only and reject
tensors that require gradients.

`categorical_sample` requires normalized contiguous F32 probabilities and
non-negative contiguous int64 `[rows, 2]` state. It mutates only the counter
column, exactly once per successful row. Keep that tensor alive across decode
steps and CUDA Graph replays. Its Philox/CDF stream is Loom-owned and does not
reproduce vLLM's native token for the same integer seed.

`rms_norm_dynamic_fp8` and its out variant accept an optional mutable
same-shape residual. The residual is updated with the storage-dtype-rounded
`input + residual` sum, while the FP8 result and F32 per-row scales match
vLLM's dynamic per-token fusion. A scale upper bound is intentionally not
supported.

`rms_norm_dynamic_int8` and its out variant use the same optional-residual
shape contract, return signed INT8 plus one F32 `absmax / 127` scale per row,
and preserve the native W8A8 rounding boundary. They are included in the
qualified ABI11 wheel, while the vLLM compiler route remains disabled by
default because its separate quality and stable-performance gates are open.

`silu_and_mul_dynamic_int8` and its out variant accept contiguous FP16/BF16
split-half gate/up input, return signed INT8 plus one F32 scale per flattened
row, and preserve the vLLM compiled-native rounding boundary: F32 SiLU and
multiplication followed by one storage-dtype product rounding before dynamic
INT8 quantization. The H20 source and real W8A8 engine gates are exact, but
compiled CUDA Graph ratios are below parity and engine latency is not
order-stable. The ABI11 route is therefore explicit-only with no speedup or
default claim; its repository-free wheel matrix is qualified separately. See
the [admission evidence](../docs/results/h20-vllm-silu-int8-admission-20260801.json)
and [wheel evidence](../docs/results/h20-native-wheel-clean-install-abi11-20260801.json).

## Exported operator families

| Family | Python entry points |
| --- | --- |
| Normalization | `rms_norm`, `rms_norm_out`, `add_rms_norm_`, `rms_norm_dynamic_fp8`, `rms_norm_dynamic_fp8_out`, `rms_norm_dynamic_int8`, `rms_norm_dynamic_int8_out` |
| Activation | `silu_and_mul`, `silu_and_mul_out`, `silu_and_mul_dynamic_fp8`, `silu_and_mul_dynamic_fp8_out`, `silu_and_mul_dynamic_int8`, `silu_and_mul_dynamic_int8_out` |
| Position and KV | `rope_paged_kv_write_` for native or static FP8 E4M3 paged caches |
| Decode tail | `logits_preprocess_`, `greedy_sample_logprobs`, `selected_token_logprobs`, `top_k_filter_`, `top_p_renorm_`, `topk_sampled_logprobs`, `min_p_filter_`, `apply_token_penalties_` |
| Speculative decode | `greedy_speculative_verify` |
| MoE movement | `moe_permute`, `moe_combine` |
| Attention | `paged_decode_attention`, `paged_decode_attention_out` |

MoE permutation accepts F32/FP16/BF16/FP8 E4M3FN activations, while weighted
combine accepts F32/FP16/BF16 expert outputs, int32 top-k IDs and optional
expert maps, and F32 routing weights. It emits vLLM-compatible grouped-GEMM
offset/inverse metadata while leaving grouped GEMM to the engine. Allocating
Python entry points and standard caller-owned `torch.ops.loom_kernels.*.out`
overloads share one implementation. Direct H20 evidence and an explicit
vLLM/Cutlass engine-admission gate are qualified; the synthetic checkpoint does
not establish production-model speedup. See the
[MoE movement design](../docs/design/moe-movement.md).

The base paged-decode API accepts one contiguous `[B, Hq, D]` query,
dense-inner NHD paged K/V views, and contiguous int32 block tables and sequence
lengths. It directly accepts K/V views from vLLM's
`[blocks, 2, block, Hkv, D]` storage.

`rope_paged_kv_write_` accepts F32/FP16/BF16 sources and one packed
`[blocks, 2, block, Hkv, D]` cache allocation with either the matching source
dtype or `torch.uint8` FP8 E4M3 storage. The single mutable cache tensor is the
real vLLM allocation and remains functionalization-safe across PyTorch
2.10/2.11; separate K/V mutable-view arguments are not supported. K/V scales
are contiguous CUDA F32 tensors with one element or one element per KV head.
Dynamic per-token-head scales, E5M2, INT8, and NVFP4 are not silently coerced
into this contract.

## vLLM opt-ins

| Route | Enable |
| --- | --- |
| Add+RMSNorm IR provider | `ir_op_priority={"fused_add_rms_norm": ["loom_cuda"]}` |
| Standalone SiLU-and-Mul | `LOOM_KERNELS_ENABLE_SILU_AND_MUL=1` |
| SiLU-and-Mul→block FP8 | `LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8=1` |
| Experimental SiLU-and-Mul→dynamic INT8 | `LOOM_KERNELS_ENABLE_SILU_AND_MUL_INT8=1` |
| Optional-residual RMSNorm→dynamic FP8 | `LOOM_KERNELS_ENABLE_RMS_NORM_FP8=1` |
| Experimental optional-residual RMSNorm→dynamic INT8 | `LOOM_KERNELS_ENABLE_RMS_NORM_INT8=1` |
| RoPE+paged-KV compiler pass | `configure_vllm_rope_paged_kv(...)` |
| Short paged decode | `LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION=1` |
| MoE permutation/combine around vendor GEMM | `LOOM_KERNELS_ENABLE_MOE_MOVEMENT=1` |
| Mixed-sampling logits preprocessing | `register_vllm_logits_preprocess()` |
| Explicit-seed categorical sampling | `register_vllm_categorical_sample()` |
| Greedy sampled logprob | `register_vllm_greedy_sample_logprobs()` |
| Greedy speculative verify | `register_vllm_greedy_speculative_verify()` |
| Selected-token logprob | `register_vllm_selected_token_logprobs()` |
| Top-k sampled logprobs | `register_vllm_topk_sampled_logprobs()` |
| Sparse token penalties | `register_vllm_token_penalties()` |
| Min-P processor | `LOOM_KERNELS_ENABLE_MIN_P=1` |

Every route checks its exact dtype, shape, layout, and semantic contract.
Shape-gated routes run the original vLLM path instead of copying, casting, or
reshaping into eligibility.

The MoE opt-in replaces only vLLM's production `moe_permute` and
`moe_unpermute` wrappers for admitted contracts. It reuses vLLM caller-owned
scratch/output tensors, preserves per-token FP8 scale reordering, and patches
already-imported Cutlass/Humming consumers without changing their grouped-GEMM
functions or weights. Unsupported contracts use the original vLLM wrapper
before admission; an admitted Loom launch is fail-closed.

Categorical registration is instead an engine-lifetime semantic choice. Every
random request must have an explicit non-negative signed-int64 seed and
speculative decoding must be disabled; unsupported admission fails early
instead of switching an active request between RNG streams. State lives on
`CachedRequestState` and a contiguous active-batch tensor, survives
remove/condense/resume/swap, and is not rebuilt in Python each decode step.
Loom's seed-to-token stream intentionally differs from native vLLM.

The [compatibility matrix](../docs/compatibility.md) records the qualified
PyTorch/vLLM versions and binary distribution boundary. Build details and
validation commands live in the
[vLLM provider guide](../docs/guides/vllm-ir-provider.md).
