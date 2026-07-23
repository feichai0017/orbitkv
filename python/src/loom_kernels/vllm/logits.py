"""vLLM logits-processing registrations."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import torch_extension_available
from ._runtime import _env_enabled, supports_installed_vllm

MIN_P_OVERRIDE_KEY = "min_p_filter"
MIN_P_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_MIN_P"
MIN_P_FAST_PATH_MIN_ROWS = 32
MIN_P_FAST_PATH_MIN_VOCAB_SIZE = 65536

_MIN_P_REGISTERED = False
_MIN_P_ORIGINAL_APPLY: Any | None = None


def _min_p_override_requested() -> bool:
    return _env_enabled(MIN_P_OVERRIDE_ENV)


def register_vllm_min_p() -> str | None:
    """Replace vLLM 0.24/0.25 allocating min-p with Loom's in-place kernel."""
    global _MIN_P_ORIGINAL_APPLY
    global _MIN_P_REGISTERED
    if _MIN_P_REGISTERED:
        return MIN_P_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    from ..torch_ops import supports_min_p_filter

    if not supports_installed_vllm():
        return None

    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    implementation = torch.ops.loom_kernels.min_p_filter_.default
    original_apply = MinPLogitsProcessor.apply

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits
        if (
            not supports_min_p_filter(logits, self.min_p)
            or logits.shape[0] < MIN_P_FAST_PATH_MIN_ROWS
            or logits.shape[1] < MIN_P_FAST_PATH_MIN_VOCAB_SIZE
        ):
            return original_apply(self, logits)
        implementation(logits, self.min_p)
        return logits

    apply.__module__ = __name__
    _MIN_P_ORIGINAL_APPLY = original_apply
    MinPLogitsProcessor.apply = apply
    _MIN_P_REGISTERED = True
    return MIN_P_OVERRIDE_KEY
def _metadata() -> dict[str, object]:
    return {
        "min_p_override_requested": _min_p_override_requested(),
        "min_p_override": _MIN_P_REGISTERED,
        "min_p_fast_path_min_rows": MIN_P_FAST_PATH_MIN_ROWS,
        "min_p_fast_path_min_vocab_size": MIN_P_FAST_PATH_MIN_VOCAB_SIZE,
    }
