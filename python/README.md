# Loom Kernels · Python adapters

Current-stream PyTorch operators and narrow, opt-in vLLM 0.24/0.25 integration
for [Loom Kernels](https://github.com/feichai0017/loom-kernels).

[Project README](../README.md) · [Integration guide](../docs/guides/vllm-ir-provider.md) · [Operator catalog](../docs/operator-catalog.md)

> [!NOTE]
> The source wheel contains adapters, not prebuilt CUDA binaries. Build the
> native library and LibTorch Stable ABI dispatcher from a repository checkout.

## Install

Choose only the framework dependencies used by the consumer:

```bash
pip install -e 'python[torch,test]'
pip install -e 'python[vllm,test]'
```

## Build the native bridge

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  python python/build_native.py

CUDA_HOME=/usr/local/cuda \
  python python/build_torch_extension.py
```

Repository checkouts discover the dispatcher under `build/`. Packaged or
externally managed deployments can set its absolute path explicitly:

```bash
export LOOM_KERNELS_TORCH_LIBRARY=/path/to/libloom_kernels_torch.so
```

`build_native.py` builds the only native backend library,
`libloom_cuda_bridge.so`. Keep it next to `libloom_kernels_torch.so` (or in its
parent directory) so the dispatcher's relative runtime search path can load
it. `build_torch_extension.py` builds the sole boxed LibTorch Stable ABI
dispatcher with a PyTorch 2.10 target. Every operator, including padded logits
and strided paged-cache views, enters checked borrowed Rust dispatch. There is
no ctypes, ATen dispatcher twin, or direct raw-CUDA framework path.

The exact H20 dispatcher binary is qualified on PyTorch 2.10 and 2.11.
Automated native binary wheels are not published yet.

## Direct PyTorch use

```python
from loom_kernels import (
    greedy_sample_logprobs,
    min_p_filter_,
    selected_token_logprobs,
    silu_and_mul_dynamic_fp8,
)

fp8_output, block_scales = silu_and_mul_dynamic_fp8(
    gate_and_up_bf16,
    group_size=128,
)

token_ids, logprobs, ranks = greedy_sample_logprobs(logits)
logprobs, ranks = selected_token_logprobs(logits, sampled_ids_i64)
min_p_filter_(sampling_logits_f32, min_p_f32)
```

All CUDA calls use PyTorch's current stream. Out variants accept caller-owned
buffers for capture-safe reuse. Public APIs are inference-only and reject
tensors that require gradients.

## Exported operator families

| Family | Python entry points |
| --- | --- |
| Normalization | `rms_norm`, `rms_norm_out`, `add_rms_norm_`, `rms_norm_dynamic_fp8`, `rms_norm_dynamic_fp8_out` |
| Activation | `silu_and_mul`, `silu_and_mul_out`, `silu_and_mul_dynamic_fp8`, `silu_and_mul_dynamic_fp8_out` |
| Position and KV | `rope_paged_kv_write_` |
| Decode tail | `greedy_sample_logprobs`, `selected_token_logprobs`, `min_p_filter_` |
| Attention | `paged_decode_attention`, `paged_decode_attention_out` |

The base paged-decode API accepts one contiguous `[B, Hq, D]` query,
dense-inner NHD paged K/V views, and contiguous int32 block tables and sequence
lengths. It directly accepts K/V views from vLLM's
`[blocks, 2, block, Hkv, D]` storage.

## vLLM opt-ins

| Route | Enable |
| --- | --- |
| Add+RMSNorm IR provider | `ir_op_priority={"fused_add_rms_norm": ["loom_cuda"]}` |
| Standalone SiLU-and-Mul | `LOOM_KERNELS_ENABLE_SILU_AND_MUL=1` |
| SiLU-and-Mul→block FP8 | `LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8=1` |
| RoPE+paged-KV compiler pass | `configure_vllm_rope_paged_kv(...)` |
| Short paged decode | `LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION=1` |
| Greedy sampled logprob | `register_vllm_greedy_sample_logprobs()` |
| Selected-token logprob | `register_vllm_selected_token_logprobs()` |
| Min-P processor | `LOOM_KERNELS_ENABLE_MIN_P=1` |

Every route checks its exact dtype, shape, layout, and semantic contract. An
unsupported request runs the original vLLM path instead of being copied,
cast, or reshaped into eligibility.

The [compatibility matrix](../docs/compatibility.md) records the qualified
PyTorch/vLLM versions and binary distribution boundary. Build details and
validation commands live in the
[vLLM provider guide](../docs/guides/vllm-ir-provider.md).
