"""Bindings for Loom's single PyTorch -> C++ -> Rust -> CUDA path."""

from __future__ import annotations

import torch

from ._torch_extension import load_torch_extension


load_torch_extension()

_rms_norm = torch.ops.loom_kernels.rms_norm.default
_add_rms_norm_mut = torch.ops.loom_kernels.add_rms_norm_mut.default
_rms_norm_dynamic_per_token_fp8 = (
    torch.ops.loom_kernels.rms_norm_dynamic_per_token_fp8.default
)
_silu_and_mul = torch.ops.loom_kernels.silu_and_mul.default
_silu_and_mul_dynamic_fp8 = (
    torch.ops.loom_kernels.silu_and_mul_dynamic_fp8.default
)
_categorical_sample = torch.ops.loom_kernels.categorical_sample.default
_greedy_sample_logprobs = torch.ops.loom_kernels.greedy_sample_logprobs.default
_selected_token_logprobs = (
    torch.ops.loom_kernels.selected_token_logprobs.default
)
_top_k_filter = torch.ops.loom_kernels.top_k_filter_.default
_top_p_renorm = torch.ops.loom_kernels.top_p_renorm_.default
_topk_sampled_logprobs = torch.ops.loom_kernels.topk_sampled_logprobs.default
_apply_token_penalties = torch.ops.loom_kernels.apply_token_penalties_.default
_greedy_speculative_verify = (
    torch.ops.loom_kernels.greedy_speculative_verify.default
)
_logits_preprocess = torch.ops.loom_kernels.logits_preprocess_.default
_min_p_filter = torch.ops.loom_kernels.min_p_filter_.default
_paged_decode_attention = torch.ops.loom_kernels.paged_decode_attention.default
_rope_paged_kv_write = torch.ops.loom_kernels.rope_paged_kv_write_.default
_bridge_abi_version = torch.ops.loom_kernels.bridge_abi_version.default
_bridge_launch_count = torch.ops.loom_kernels.bridge_launch_count.default
_reset_bridge_launch_count = (
    torch.ops.loom_kernels.reset_bridge_launch_count.default
)
