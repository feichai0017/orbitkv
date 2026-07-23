"""vLLM activation and activation-quantization registrations."""

from __future__ import annotations

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import _env_enabled, supports_installed_vllm

SILU_OVERRIDE_KEY = "SiluAndMul"
SILU_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_SILU_AND_MUL"
ACT_QUANT_OVERRIDE_KEY = "silu_and_mul_dynamic_fp8"
ACT_QUANT_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8"

_SILU_OVERRIDE_CLASS: type | None = None
_ACT_QUANT_OVERRIDE_REGISTERED = False


def _silu_override_requested() -> bool:
    return _env_enabled(SILU_OVERRIDE_ENV)


def _act_quant_override_requested() -> bool:
    return _env_enabled(ACT_QUANT_OVERRIDE_ENV)


def register_vllm_silu_and_mul() -> str | None:
    """Override vLLM's standard SwiGLU layer with the Loom CUDA operator."""
    global _SILU_OVERRIDE_CLASS
    if _SILU_OVERRIDE_CLASS is not None:
        return SILU_OVERRIDE_KEY
    if not torch_extension_available():
        return None
    if not supports_installed_vllm():
        return None

    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.activation import SiluAndMul

    load_torch_extension()
    implementation = torch.ops.loom_kernels.silu_and_mul.default

    @CustomOp.register_oot(name=SILU_OVERRIDE_KEY)
    class LoomSiluAndMul(SiluAndMul):
        def __init__(self, *, compile_native: bool = True):
            # vLLM may globally disable CustomOp kernels while compiling its
            # native fallback.  An out-of-tree replacement must opt back in,
            # otherwise the registered class exists but never reaches Loom.
            del compile_native
            CustomOp.__init__(self, enforce_enable=True, compile_native=False)

        def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
            width = x.shape[-1] // 2
            output = torch.empty(
                (*x.shape[:-1], width), dtype=x.dtype, device=x.device
            )
            implementation(x, output)
            return output

    _SILU_OVERRIDE_CLASS = LoomSiluAndMul
    return SILU_OVERRIDE_KEY


def register_vllm_silu_and_mul_dynamic_fp8() -> str | None:
    """Route vLLM's 64/128-element activation-quant fusions to Loom."""
    global _ACT_QUANT_OVERRIDE_REGISTERED
    if _ACT_QUANT_OVERRIDE_REGISTERED:
        return ACT_QUANT_OVERRIDE_KEY
    if not torch_extension_available():
        return None
    if not supports_installed_vllm():
        return None

    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    load_torch_extension()
    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    FUSED_OPS[kFp8Dynamic64Sym] = implementation
    FUSED_OPS[kFp8Dynamic128Sym] = implementation
    _ACT_QUANT_OVERRIDE_REGISTERED = True
    return ACT_QUANT_OVERRIDE_KEY


def _metadata() -> dict[str, object]:
    return {
        "silu_and_mul_override_requested": _silu_override_requested(),
        "silu_and_mul_override": _SILU_OVERRIDE_CLASS is not None,
        "silu_and_mul_fp8_override_requested": _act_quant_override_requested(),
        "silu_and_mul_fp8_override": _ACT_QUANT_OVERRIDE_REGISTERED,
    }
