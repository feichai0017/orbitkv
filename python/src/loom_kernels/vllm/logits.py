"""vLLM logits-processing registrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .._torch_extension import torch_extension_available
from ._runtime import _env_enabled, supports_installed_vllm

MIN_P_OVERRIDE_KEY = "min_p_filter"
MIN_P_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_MIN_P"
MIN_P_FAST_PATH_MIN_ROWS = 32
MIN_P_FAST_PATH_MIN_VOCAB_SIZE = 65536

LOGITS_PREPROCESS_OVERRIDE_KEY = "logits_preprocess"
LOGITS_PREPROCESS_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_LOGITS_PREPROCESS"
_LOGITS_PREPROCESS_STATE_ATTRIBUTE = "_loom_logits_preprocess_state"

_MIN_P_REGISTERED = False
_MIN_P_ORIGINAL_APPLY: Any | None = None
_LOGITS_PREPROCESS_REGISTERED = False
_LOGITS_PREPROCESS_ORIGINAL_APPLY: Any | None = None
_LOGITS_PREPROCESS_ORIGINAL_SAMPLE: Any | None = None
_LOGITS_PREPROCESS_FIRST_CONTRACT: dict[str, Any] | None = None
_LOGITS_PREPROCESS_FIRST_REJECTION: dict[str, Any] | None = None
_LOGITS_PREPROCESS_OBSERVED = {
    "accepted_contracts": 0,
    "blocked_mask": False,
    "maximum_bias_count": 0,
    "maximum_suppression_count": 0,
    "bad_words": False,
    "min_tokens": False,
}


@dataclass(frozen=True)
class _LogitsPreprocessState:
    blocked_mask: torch.Tensor | None
    bias_row_ids: torch.Tensor | None
    bias_token_ids: torch.Tensor | None
    bias_values: torch.Tensor | None
    suppressed_row_ids: torch.Tensor | None
    suppressed_token_ids: torch.Tensor | None


def _min_p_override_requested() -> bool:
    return _env_enabled(MIN_P_OVERRIDE_ENV)


def _logits_preprocess_override_requested() -> bool:
    return _env_enabled(LOGITS_PREPROCESS_OVERRIDE_ENV)


def _active_bad_word_targets(
    bad_words_token_ids: dict[int, list[list[int]]],
    output_token_ids: list[list[int]],
    rows: int,
    vocab_size: int,
) -> tuple[list[int], list[int]] | None:
    targets: set[tuple[int, int]] = set()
    for row, bad_words in bad_words_token_ids.items():
        if row < 0 or row >= rows or row >= len(output_token_ids):
            return None
        past_tokens = output_token_ids[row]
        for bad_word in bad_words:
            if not bad_word:
                return None
            if len(bad_word) > len(past_tokens) + 1:
                continue
            prefix_length = len(bad_word) - 1
            expected_prefix = bad_word[:prefix_length]
            actual_prefix = (
                past_tokens[-prefix_length:] if prefix_length > 0 else []
            )
            if actual_prefix == expected_prefix:
                token_id = bad_word[-1]
                if token_id < 0 or token_id >= vocab_size:
                    return None
                targets.add((row, token_id))
    ordered = sorted(targets)
    return (
        [row for row, _ in ordered],
        [token_id for _, token_id in ordered],
    )


def _upload_sparse_targets(
    rows: list[int],
    token_ids: list[int],
    device: torch.device,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not rows:
        return None
    host_rows = torch.tensor(
        rows, dtype=torch.int32, device="cpu", pin_memory=pin_memory
    )
    host_token_ids = torch.tensor(
        token_ids, dtype=torch.int32, device="cpu", pin_memory=pin_memory
    )
    return (
        host_rows.to(device, non_blocking=True),
        host_token_ids.to(device, non_blocking=True),
    )


def register_vllm_logits_preprocess() -> str | None:
    """Fuse vLLM mixed-sampling logits preprocessing into one CUDA pass."""
    global _LOGITS_PREPROCESS_ORIGINAL_APPLY
    global _LOGITS_PREPROCESS_ORIGINAL_SAMPLE
    global _LOGITS_PREPROCESS_REGISTERED
    if _LOGITS_PREPROCESS_REGISTERED:
        return LOGITS_PREPROCESS_OVERRIDE_KEY
    if not torch_extension_available() or not supports_installed_vllm():
        return None

    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.sampler import Sampler

    from ..torch_ops import supports_logits_preprocess

    implementation = torch.ops.loom_kernels.logits_preprocess_.default
    original_apply = Sampler.apply_logits_processors
    original_sample = Sampler.sample

    def reject(
        reason: str,
        logits: torch.Tensor,
        sampling_metadata: Any,
    ) -> None:
        global _LOGITS_PREPROCESS_FIRST_REJECTION
        if _LOGITS_PREPROCESS_FIRST_REJECTION is None:
            _LOGITS_PREPROCESS_FIRST_REJECTION = {
                "reason": reason,
                "rows": int(logits.shape[0]) if logits.dim() == 2 else None,
                "vocab_size": (
                    int(logits.shape[1]) if logits.dim() == 2 else None
                ),
                "all_greedy": bool(sampling_metadata.all_greedy),
                "all_random": bool(sampling_metadata.all_random),
            }

    def apply_logits_processors(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        global _LOGITS_PREPROCESS_FIRST_CONTRACT
        if hasattr(sampling_metadata, _LOGITS_PREPROCESS_STATE_ATTRIBUTE):
            delattr(sampling_metadata, _LOGITS_PREPROCESS_STATE_ATTRIBUTE)
        if sampling_metadata.all_greedy or sampling_metadata.all_random:
            reject(
                "only mixed greedy/random batches are admitted",
                logits,
                sampling_metadata,
            )
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )
        if sampling_metadata.temperature is None:
            reject(
                "mixed sampling has no temperature tensor",
                logits,
                sampling_metadata,
            )
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )
        if not sampling_metadata.no_penalties:
            reject("token penalties are active", logits, sampling_metadata)
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )
        holder = sampling_metadata.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            reject(
                "thinking-budget processing is active",
                logits,
                sampling_metadata,
            )
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )

        bias_processor: Any | None = None
        min_tokens_processor: Any | None = None
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, LogitBiasLogitsProcessor):
                if bias_processor is not None:
                    reject(
                        "multiple logit-bias processors are active",
                        logits,
                        sampling_metadata,
                    )
                    return original_apply(
                        self, logits, sampling_metadata, predict_bonus_token
                    )
                bias_processor = processor
            elif isinstance(processor, MinTokensLogitsProcessor):
                if min_tokens_processor is not None:
                    reject(
                        "multiple min-tokens processors are active",
                        logits,
                        sampling_metadata,
                    )
                    return original_apply(
                        self, logits, sampling_metadata, predict_bonus_token
                    )
                min_tokens_processor = processor
            else:
                reject(
                    "a custom non-argmax-invariant processor is active",
                    logits,
                    sampling_metadata,
                )
                return original_apply(
                    self, logits, sampling_metadata, predict_bonus_token
                )

        bias_row_ids: torch.Tensor | None = None
        bias_token_ids: torch.Tensor | None = None
        bias_values: torch.Tensor | None = None
        if bias_processor is not None and bias_processor.biases:
            bias_row_ids, bias_token_ids = bias_processor.logits_slice
            bias_values = bias_processor.bias_tensor

        min_tokens_targets: tuple[torch.Tensor, torch.Tensor] | None = None
        if min_tokens_processor is not None and min_tokens_processor.min_toks:
            min_tokens_targets = min_tokens_processor.logits_slice

        output_token_ids = sampling_metadata.output_token_ids
        bad_words = sampling_metadata.bad_words_token_ids
        if predict_bonus_token and bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids, sampling_metadata.spec_token_ids
            )
        bad_word_targets = _active_bad_word_targets(
            bad_words,
            output_token_ids,
            int(logits.shape[0]),
            int(logits.shape[1]),
        )
        if bad_word_targets is None:
            reject(
                "bad-word metadata is outside the logits shape",
                logits,
                sampling_metadata,
            )
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )
        uploaded_bad_words = _upload_sparse_targets(
            bad_word_targets[0],
            bad_word_targets[1],
            logits.device,
            self.pin_memory,
        )
        if min_tokens_targets is not None and uploaded_bad_words is not None:
            reject(
                "min-tokens and active bad-word suppression overlap",
                logits,
                sampling_metadata,
            )
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )
        suppression = min_tokens_targets or uploaded_bad_words
        suppressed_row_ids = suppression[0] if suppression is not None else None
        suppressed_token_ids = (
            suppression[1] if suppression is not None else None
        )
        blocked_mask = sampling_metadata.allowed_token_ids_mask

        if not supports_logits_preprocess(
            logits,
            sampling_metadata.temperature,
            blocked_mask,
            bias_row_ids,
            bias_token_ids,
            bias_values,
            suppressed_row_ids,
            suppressed_token_ids,
        ):
            reject(
                "tensor metadata misses the fused CUDA contract",
                logits,
                sampling_metadata,
            )
            return original_apply(
                self, logits, sampling_metadata, predict_bonus_token
            )

        state = _LogitsPreprocessState(
            blocked_mask,
            bias_row_ids,
            bias_token_ids,
            bias_values,
            suppressed_row_ids,
            suppressed_token_ids,
        )
        setattr(sampling_metadata, _LOGITS_PREPROCESS_STATE_ATTRIBUTE, state)
        contract = {
            "rows": int(logits.shape[0]),
            "vocab_size": int(logits.shape[1]),
            "mixed_sampling": True,
            "blocked_mask": blocked_mask is not None,
            "bias_count": (
                0 if bias_row_ids is None else int(bias_row_ids.numel())
            ),
            "suppression_count": (
                0
                if suppressed_row_ids is None
                else int(suppressed_row_ids.numel())
            ),
            "bad_words": uploaded_bad_words is not None,
            "min_tokens": min_tokens_targets is not None,
        }
        if _LOGITS_PREPROCESS_FIRST_CONTRACT is None:
            _LOGITS_PREPROCESS_FIRST_CONTRACT = contract
        _LOGITS_PREPROCESS_OBSERVED["accepted_contracts"] += 1
        _LOGITS_PREPROCESS_OBSERVED["blocked_mask"] |= contract[
            "blocked_mask"
        ]
        _LOGITS_PREPROCESS_OBSERVED["maximum_bias_count"] = max(
            _LOGITS_PREPROCESS_OBSERVED["maximum_bias_count"],
            contract["bias_count"],
        )
        _LOGITS_PREPROCESS_OBSERVED["maximum_suppression_count"] = max(
            _LOGITS_PREPROCESS_OBSERVED["maximum_suppression_count"],
            contract["suppression_count"],
        )
        _LOGITS_PREPROCESS_OBSERVED["bad_words"] |= contract["bad_words"]
        _LOGITS_PREPROCESS_OBSERVED["min_tokens"] |= contract["min_tokens"]
        return logits

    def sample(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        logprobs_mode_override: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = getattr(
            sampling_metadata, _LOGITS_PREPROCESS_STATE_ATTRIBUTE, None
        )
        if state is None:
            return original_sample(
                self,
                logits,
                sampling_metadata,
                logprobs_mode_override,
            )
        delattr(sampling_metadata, _LOGITS_PREPROCESS_STATE_ATTRIBUTE)
        implementation(
            logits,
            sampling_metadata.temperature,
            state.blocked_mask,
            state.bias_row_ids,
            state.bias_token_ids,
            state.bias_values,
            state.suppressed_row_ids,
            state.suppressed_token_ids,
        )

        greedy_sampled = self.greedy_sample(logits)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
        sampled = torch.where(
            sampling_metadata.temperature < 1.0e-5,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        return sampled, processed_logprobs

    apply_logits_processors.__module__ = __name__
    sample.__module__ = __name__
    _LOGITS_PREPROCESS_ORIGINAL_APPLY = original_apply
    _LOGITS_PREPROCESS_ORIGINAL_SAMPLE = original_sample
    Sampler.apply_logits_processors = apply_logits_processors
    Sampler.sample = sample
    _LOGITS_PREPROCESS_REGISTERED = True
    return LOGITS_PREPROCESS_OVERRIDE_KEY


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
        "logits_preprocess_override_requested": (
            _logits_preprocess_override_requested()
        ),
        "logits_preprocess_override": _LOGITS_PREPROCESS_REGISTERED,
        "logits_preprocess_first_contract": (
            _LOGITS_PREPROCESS_FIRST_CONTRACT
        ),
        "logits_preprocess_observed": dict(_LOGITS_PREPROCESS_OBSERVED),
        "logits_preprocess_first_rejection": (
            _LOGITS_PREPROCESS_FIRST_REJECTION
        ),
        "logits_preprocess_admission": "mixed greedy/random batches only",
        "min_p_override_requested": _min_p_override_requested(),
        "min_p_override": _MIN_P_REGISTERED,
        "min_p_fast_path_min_rows": MIN_P_FAST_PATH_MIN_ROWS,
        "min_p_fast_path_min_vocab_size": MIN_P_FAST_PATH_MIN_VOCAB_SIZE,
    }
