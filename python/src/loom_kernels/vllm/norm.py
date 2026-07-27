"""vLLM normalization-quantization fusion registrations."""

from __future__ import annotations

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import _env_enabled, supports_installed_vllm

RMS_NORM_FP8_OVERRIDE_KEY = "rms_norm_dynamic_fp8"
RMS_NORM_FP8_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_RMS_NORM_FP8"

_RMS_NORM_FP8_OVERRIDE_REGISTERED = False


def _rms_norm_fp8_override_requested() -> bool:
    return _env_enabled(RMS_NORM_FP8_OVERRIDE_ENV)


def register_vllm_rms_norm_dynamic_fp8() -> str | None:
    """Route vLLM's per-token FP8 RMSNorm fusions to Loom."""
    global _RMS_NORM_FP8_OVERRIDE_REGISTERED
    if _RMS_NORM_FP8_OVERRIDE_REGISTERED:
        return RMS_NORM_FP8_OVERRIDE_KEY
    if not torch_extension_available():
        return None
    if not supports_installed_vllm():
        return None

    from vllm.compilation.passes.fusion.rms_quant_fusion import (
        FUSED_OPS,
        FusedRMSQuantKey,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    load_torch_extension()
    implementation = (
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_fp8.default
    )
    FUSED_OPS[
        FusedRMSQuantKey(kFp8DynamicTokenSym, fused_add=False)
    ] = implementation
    FUSED_OPS[
        FusedRMSQuantKey(kFp8DynamicTokenSym, fused_add=True)
    ] = implementation
    _RMS_NORM_FP8_OVERRIDE_REGISTERED = True
    return RMS_NORM_FP8_OVERRIDE_KEY


def _metadata() -> dict[str, object]:
    return {
        "rms_norm_fp8_override_requested": (
            _rms_norm_fp8_override_requested()
        ),
        "rms_norm_fp8_override": _RMS_NORM_FP8_OVERRIDE_REGISTERED,
    }
