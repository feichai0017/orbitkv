from __future__ import annotations

from types import MappingProxyType


GDN_FIXED_STATE_BACKEND_PROFILE = MappingProxyType(
    {
        "attention_backend": "fa3",
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "linear_attn_prefill_backend": "triton",
        "mamba_ssm_dtype": "float32",
        "mamba_radix_cache_strategy": "no_buffer",
    }
)


__all__ = ["GDN_FIXED_STATE_BACKEND_PROFILE"]
