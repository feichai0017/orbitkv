"""INT8 extension to vLLM's activation-plus-quantization compiler pass."""

from __future__ import annotations

from typing import Any

import torch


_INT8_PATTERN_INSTALLED = False


class SiluMulDynamicInt8QuantPattern:
    """Match split-half SwiGLU followed by dynamic per-token INT8."""

    def __init__(self) -> None:
        from vllm.compilation.passes.fusion.matcher_utils import (
            MatcherSiluAndMul,
        )

        self.silu_and_mul_matcher = MatcherSiluAndMul()

        def pattern(
            input_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            activated = self.silu_and_mul_matcher(input_tensor)
            return self._quantize(activated)

        def replacement(
            input_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            from vllm.compilation.passes.fusion import (
                rms_quant_fusion as fusion,
            )

            width = input_tensor.shape[-1] // 2
            output_shape = input_tensor.shape[:-1] + (width,)
            result = torch.empty(
                output_shape,
                device=input_tensor.device,
                dtype=torch.int8,
            )
            scale = torch.empty(
                (input_tensor.numel() // input_tensor.shape[-1], 1),
                device=input_tensor.device,
                dtype=torch.float32,
            )
            at = fusion.auto_functionalized(
                torch.ops.loom_kernels.silu_and_mul_dynamic_per_token_int8.default,
                result=result,
                input=input_tensor,
                scale=scale,
            )
            return at[1], at[2]

        self.pattern = pattern
        self.replacement = replacement

    @staticmethod
    def _quantize(
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.compilation.passes.fusion import rms_quant_fusion as fusion

        result = torch.empty_like(input_tensor, dtype=torch.int8)
        scale = torch.empty(
            (input_tensor.numel() // input_tensor.shape[-1], 1),
            device=input_tensor.device,
            dtype=torch.float32,
        )
        at = fusion.auto_functionalized(
            torch.ops._C.dynamic_scaled_int8_quant.default,
            result=result,
            input=input_tensor,
            scale=scale,
            azp=None,
        )
        return at[1], at[2]

    def get_inputs(self) -> list[torch.Tensor]:
        return self.silu_and_mul_matcher.inputs()


def install_silu_and_mul_dynamic_int8_pattern() -> None:
    """Extend vLLM's activation-quant pass with the INT8 fusion pattern."""
    global _INT8_PATTERN_INSTALLED
    if _INT8_PATTERN_INSTALLED:
        return

    from vllm.compilation.passes.fusion import act_quant_fusion as fusion
    from vllm.compilation.passes.fusion import rms_quant_fusion

    original_pass = fusion.ActivationQuantFusionPass
    if getattr(original_pass, "_loom_supports_dynamic_int8", False):
        _INT8_PATTERN_INSTALLED = True
        return

    original_init = original_pass.__init__

    @rms_quant_fusion.enable_fake_mode
    def loom_init(self: Any, config: Any) -> None:
        original_init(self, config)
        self.register(SiluMulDynamicInt8QuantPattern())

    original_pass.__init__ = loom_init
    original_pass._loom_supports_dynamic_int8 = True
    _INT8_PATTERN_INSTALLED = True


def silu_and_mul_dynamic_int8_pattern_installed() -> bool:
    return _INT8_PATTERN_INSTALLED
