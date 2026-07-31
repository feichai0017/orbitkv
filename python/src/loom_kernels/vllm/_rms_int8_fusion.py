"""INT8 extensions to vLLM's RMSNorm plus quantization compiler pass."""

from __future__ import annotations

from typing import Any

import torch


_INT8_PATTERNS_INSTALLED = False


class RMSNormDynamicInt8QuantPattern:
    """Match RMSNorm followed by symmetric dynamic per-token INT8."""

    def __init__(self, epsilon: float) -> None:
        from vllm.config import get_current_vllm_config

        self.epsilon = epsilon
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None

        # Keep pattern and replacement as closures. FX otherwise treats the
        # `self` argument of a bound method as an additional trace input.
        def pattern(
            input_tensor: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            import vllm.ir.ops

            normalized = vllm.ir.ops.rms_norm(
                input_tensor, weight, self.epsilon
            )
            return self._quantize(normalized)

        def replacement(
            input_tensor: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            from vllm.compilation.passes.fusion import (
                rms_quant_fusion as fusion,
            )

            input_tensor = input_tensor.to(dtype=self.model_dtype)
            result = torch.empty_like(input_tensor, dtype=torch.int8)
            scale = torch.empty(
                (input_tensor.numel() // input_tensor.shape[-1], 1),
                device=input_tensor.device,
                dtype=torch.float32,
            )
            at = fusion.auto_functionalized(
                torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8.default,
                result=result,
                input=input_tensor,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                residual=None,
            )
            return at[1], at[2]

        self.pattern = pattern
        self.replacement = replacement

    @staticmethod
    def _quantize(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        from vllm.compilation.passes.fusion import rms_quant_fusion as fusion

        return [fusion.empty_bf16(5, 16), fusion.empty_bf16(16)]

    def register(self, matcher_pass: Any) -> None:
        from vllm.compilation.passes.fusion import rms_quant_fusion as fusion

        fusion.pm.register_replacement(
            self.pattern,
            self.replacement,
            self.get_inputs(),
            fusion.pm.fwd_only,
            matcher_pass,
            extra_check=fusion._rms_input_weight_dtype_match,
        )


class FusedAddRMSNormDynamicInt8QuantPattern(
    RMSNormDynamicInt8QuantPattern
):
    """Match residual Add+RMSNorm followed by dynamic per-token INT8."""

    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon)

        def pattern(
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            import vllm.ir.ops

            normalized, updated_residual = (
                vllm.ir.ops.fused_add_rms_norm(
                    input_tensor, residual, weight, self.epsilon
                )
            )
            result, scale = self._quantize(normalized)
            return result, updated_residual, scale

        def replacement(
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            from vllm.compilation.passes.fusion import (
                rms_quant_fusion as fusion,
            )

            input_tensor = input_tensor.to(dtype=self.model_dtype)
            result = torch.empty_like(input_tensor, dtype=torch.int8)
            scale = torch.empty(
                (input_tensor.numel() // input_tensor.shape[-1], 1),
                device=input_tensor.device,
                dtype=torch.float32,
            )
            at = fusion.auto_functionalized(
                torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8.default,
                result=result,
                input=input_tensor,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                residual=residual,
            )
            return at[1], at[3], at[2]

        self.pattern = pattern
        self.replacement = replacement

    def get_inputs(self) -> list[torch.Tensor]:
        from vllm.compilation.passes.fusion import rms_quant_fusion as fusion

        return [
            fusion.empty_bf16(5, 16),
            fusion.empty_bf16(16),
            fusion.empty_bf16(5, 16),
        ]


def install_rms_norm_dynamic_int8_patterns() -> None:
    """Extend vLLM's existing RMSNorm fusion pass with INT8 patterns."""
    global _INT8_PATTERNS_INSTALLED
    if _INT8_PATTERNS_INSTALLED:
        return

    from vllm.compilation.passes.fusion import rms_quant_fusion as fusion

    original_pass = fusion.RMSNormQuantFusionPass
    if getattr(original_pass, "_loom_supports_dynamic_int8", False):
        _INT8_PATTERNS_INSTALLED = True
        return

    original_init = original_pass.__init__
    original_uuid = original_pass.uuid

    @fusion.enable_fake_mode
    def loom_init(self: Any, config: Any) -> None:
        original_init(self, config)
        for epsilon in (1.0e-5, 1.0e-6):
            FusedAddRMSNormDynamicInt8QuantPattern(epsilon).register(
                self.patterns
            )
            RMSNormDynamicInt8QuantPattern(epsilon).register(self.patterns)

    def loom_uuid(self: Any) -> str:
        return original_uuid(self) + self.hash_source(
            RMSNormDynamicInt8QuantPattern,
            FusedAddRMSNormDynamicInt8QuantPattern,
        )

    original_pass.__init__ = loom_init
    original_pass.uuid = loom_uuid
    original_pass._loom_supports_dynamic_int8 = True
    _INT8_PATTERNS_INSTALLED = True


def rms_norm_dynamic_int8_patterns_installed() -> bool:
    return _INT8_PATTERNS_INSTALLED
