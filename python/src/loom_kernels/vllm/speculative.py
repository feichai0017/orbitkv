"""vLLM deterministic speculative-verification registration."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import supports_installed_vllm

GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY = "greedy_speculative_verify"

_GREEDY_SPECULATIVE_VERIFY_REGISTERED = False
_GREEDY_SPECULATIVE_VERIFY_ORIGINAL: Any | None = None
_GREEDY_SPECULATIVE_VERIFY_FIRST_CONTRACT: dict[str, Any] | None = None
_GREEDY_SPECULATIVE_VERIFY_FIRST_REJECTION: dict[str, Any] | None = None


def register_vllm_greedy_speculative_verify() -> str | None:
    """Route vLLM's deterministic greedy draft verification through Loom.

    vLLM remains responsible for draft generation, target-logit processing,
    target argmax, bonus-token sampling, and every stochastic rejection path.
    Loom replaces only the standard all-greedy ragged verify/compact kernel.
    """
    global _GREEDY_SPECULATIVE_VERIFY_ORIGINAL
    global _GREEDY_SPECULATIVE_VERIFY_REGISTERED
    if _GREEDY_SPECULATIVE_VERIFY_REGISTERED:
        return GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY
    if not torch_extension_available() or not supports_installed_vllm():
        return None

    from ..torch_ops import (
        greedy_speculative_verify,
        supports_greedy_speculative_verify,
    )
    from vllm.v1.sample import rejection_sampler

    load_torch_extension()
    original = rejection_sampler.rejection_sample

    def verify(
        draft_token_ids: torch.Tensor,
        num_draft_tokens: list[int],
        max_spec_len: int,
        cu_num_draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        sampling_metadata: Any,
        synthetic_mode: bool = False,
        synthetic_conditional_rates: torch.Tensor | None = None,
        use_fp64_gumbel: bool = False,
    ) -> torch.Tensor:
        global _GREEDY_SPECULATIVE_VERIFY_FIRST_CONTRACT
        global _GREEDY_SPECULATIVE_VERIFY_FIRST_REJECTION
        basic_contract = bool(
            sampling_metadata.all_greedy
            and not synthetic_mode
            and isinstance(max_spec_len, int)
            and max_spec_len > 0
            and len(num_draft_tokens) == cu_num_draft_tokens.numel()
            and target_logits.device.type == "cuda"
            and target_logits.dim() == 2
            and target_logits.shape[0] == draft_token_ids.numel()
        )
        if basic_contract:
            target_token_ids = target_logits.argmax(dim=-1)
            use_fast_path = supports_greedy_speculative_verify(
                draft_token_ids,
                target_token_ids,
                bonus_token_ids,
                cu_num_draft_tokens,
                max_spec_len,
            )
        else:
            target_token_ids = None
            use_fast_path = False

        if not use_fast_path:
            if _GREEDY_SPECULATIVE_VERIFY_FIRST_REJECTION is None:
                _GREEDY_SPECULATIVE_VERIFY_FIRST_REJECTION = {
                    "all_greedy": sampling_metadata.all_greedy,
                    "synthetic_mode": synthetic_mode,
                    "draft_shape": list(draft_token_ids.shape),
                    "draft_dtype": str(draft_token_ids.dtype),
                    "target_logits_shape": list(target_logits.shape),
                    "bonus_shape": list(bonus_token_ids.shape),
                    "cumulative_shape": list(cu_num_draft_tokens.shape),
                    "max_spec_len": max_spec_len,
                }
            fallback = _GREEDY_SPECULATIVE_VERIFY_ORIGINAL
            if fallback is None:
                raise RuntimeError(
                    "vLLM greedy speculative fallback is unavailable"
                )
            return fallback(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
                synthetic_mode,
                synthetic_conditional_rates,
                use_fp64_gumbel,
            )

        assert target_token_ids is not None
        if _GREEDY_SPECULATIVE_VERIFY_FIRST_CONTRACT is None:
            _GREEDY_SPECULATIVE_VERIFY_FIRST_CONTRACT = {
                "requests": len(num_draft_tokens),
                "draft_tokens": draft_token_ids.numel(),
                "max_spec_len": max_spec_len,
                "draft_dtype": str(draft_token_ids.dtype),
                "target_dtype": str(target_token_ids.dtype),
                "bonus_shape": list(bonus_token_ids.shape),
                "cumulative_dtype": str(cu_num_draft_tokens.dtype),
            }
        output_token_ids, _, _ = greedy_speculative_verify(
            draft_token_ids,
            target_token_ids,
            bonus_token_ids,
            cu_num_draft_tokens,
            max_spec_len,
        )
        return output_token_ids

    verify.__module__ = __name__
    _GREEDY_SPECULATIVE_VERIFY_ORIGINAL = original
    rejection_sampler.rejection_sample = verify
    _GREEDY_SPECULATIVE_VERIFY_REGISTERED = True
    return GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY
def _metadata() -> dict[str, object]:
    return {
        "greedy_speculative_verify_override": _GREEDY_SPECULATIVE_VERIFY_REGISTERED,
        "greedy_speculative_verify_first_contract": _GREEDY_SPECULATIVE_VERIFY_FIRST_CONTRACT,
        "greedy_speculative_verify_first_rejection": _GREEDY_SPECULATIVE_VERIFY_FIRST_REJECTION,
    }
