"""vLLM integration facade for Loom Kernels."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import torch_extension_available
from . import activation as _activation
from . import attention as _attention
from . import logits as _logits
from . import rope_kv as _rope_kv
from . import sampling as _sampling
from . import speculative as _speculative
from ._runtime import (
    DEFAULT_PROVIDER,
    SUPPORTED_VLLM_SERIES,
    installed_vllm_version,
    supports_installed_vllm,
)
from .activation import (
    ACT_QUANT_OVERRIDE_ENV,
    ACT_QUANT_OVERRIDE_KEY,
    SILU_OVERRIDE_ENV,
    SILU_OVERRIDE_KEY,
    register_vllm_silu_and_mul,
    register_vllm_silu_and_mul_dynamic_fp8,
)
from .attention import (
    PAGED_DECODE_FAST_PATH_MAX_BATCH,
    PAGED_DECODE_FAST_PATH_MAX_CONTEXT,
    PAGED_DECODE_OVERRIDE_ENV,
    PAGED_DECODE_OVERRIDE_KEY,
    register_vllm_paged_decode_attention,
    supports_vllm_paged_decode_shape,
)
from .logits import (
    MIN_P_FAST_PATH_MIN_ROWS,
    MIN_P_FAST_PATH_MIN_VOCAB_SIZE,
    MIN_P_OVERRIDE_ENV,
    MIN_P_OVERRIDE_KEY,
    register_vllm_min_p,
)
from .rope_kv import (
    ROPE_PAGED_KV_OVERRIDE_KEY,
    configure_vllm_rope_paged_kv,
    register_vllm_rope_paged_kv,
)
from .sampling import (
    GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY,
    SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY,
    register_vllm_greedy_sample_logprobs,
    register_vllm_selected_token_logprobs,
)
from .speculative import (
    GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY,
    register_vllm_greedy_speculative_verify,
)


def register_vllm_ir(provider: str = DEFAULT_PROVIDER) -> str | None:
    """Register Loom as an in-place fused_add_rms_norm IR provider."""
    if not supports_installed_vllm() or not torch_extension_available():
        return None

    from vllm import ir
    import vllm.ir.ops.layernorm  # noqa: F401 - registers the IR operation

    from ..torch_ops import supports_add_rms_norm

    if _activation._silu_override_requested():
        register_vllm_silu_and_mul()
    if _activation._act_quant_override_requested():
        register_vllm_silu_and_mul_dynamic_fp8()
    if _logits._min_p_override_requested():
        register_vllm_min_p()
    if _attention._paged_decode_override_requested():
        register_vllm_paged_decode_attention()

    operation = ir.ops.fused_add_rms_norm
    implementations = getattr(operation, "impls", {})
    if provider in implementations:
        return provider

    def implementation(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight is None or variance_size is not None:
            raise ValueError("unsupported Loom Add+RMSNorm contract reached dispatch")
        torch.ops.loom_kernels.add_rms_norm_mut.default(
            x, x_residual, weight, epsilon
        )
        return x, x_residual

    def supports(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> bool:
        return supports_add_rms_norm(
            x, x_residual, weight, epsilon, variance_size
        )

    decorator = operation.register_impl(
        provider,
        supported=torch_extension_available(),
        supports_args=supports,
        inplace=True,
    )
    decorator(implementation)
    return provider


def provider_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": DEFAULT_PROVIDER,
        "vllm_version": installed_vllm_version(),
        "vllm_supported": supports_installed_vllm(),
        "supported_vllm_series": [
            f"{major}.{minor}" for major, minor in SUPPORTED_VLLM_SERIES
        ],
        "extension_available": torch_extension_available(),
        "operator": "fused_add_rms_norm",
        "inplace": True,
    }
    for domain in (
        _activation,
        _attention,
        _logits,
        _rope_kv,
        _sampling,
        _speculative,
    ):
        metadata.update(domain._metadata())
    return metadata


__all__ = [
    "ACT_QUANT_OVERRIDE_ENV",
    "ACT_QUANT_OVERRIDE_KEY",
    "DEFAULT_PROVIDER",
    "GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY",
    "GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY",
    "MIN_P_FAST_PATH_MIN_ROWS",
    "MIN_P_FAST_PATH_MIN_VOCAB_SIZE",
    "MIN_P_OVERRIDE_ENV",
    "MIN_P_OVERRIDE_KEY",
    "PAGED_DECODE_FAST_PATH_MAX_BATCH",
    "PAGED_DECODE_FAST_PATH_MAX_CONTEXT",
    "PAGED_DECODE_OVERRIDE_ENV",
    "PAGED_DECODE_OVERRIDE_KEY",
    "ROPE_PAGED_KV_OVERRIDE_KEY",
    "SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY",
    "SILU_OVERRIDE_ENV",
    "SILU_OVERRIDE_KEY",
    "SUPPORTED_VLLM_SERIES",
    "configure_vllm_rope_paged_kv",
    "installed_vllm_version",
    "provider_metadata",
    "register_vllm_ir",
    "register_vllm_min_p",
    "register_vllm_paged_decode_attention",
    "register_vllm_greedy_sample_logprobs",
    "register_vllm_greedy_speculative_verify",
    "register_vllm_rope_paged_kv",
    "register_vllm_selected_token_logprobs",
    "register_vllm_silu_and_mul",
    "register_vllm_silu_and_mul_dynamic_fp8",
    "supports_installed_vllm",
    "supports_vllm_paged_decode_shape",
]
