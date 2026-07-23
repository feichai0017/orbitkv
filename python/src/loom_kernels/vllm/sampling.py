"""vLLM greedy and selected-token sampling-tail registrations."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import supports_installed_vllm

GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY = "greedy_sample_logprobs"
SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY = "selected_token_logprobs"

_GREEDY_SAMPLE_LOGPROBS_REGISTERED = False
_GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH: Any | None = None
_GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None
_SELECTED_TOKEN_LOGPROBS_REGISTERED = False
_SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None


def register_vllm_greedy_sample_logprobs() -> str | None:
    """Install the deterministic vLLM 0.24/0.25 greedy+logprob fast path.

    The override is deliberately narrow: all requests must be greedy, request
    only the sampled token's raw logprob (`max_num_logprobs == 0`), and have no
    logits mutation from masks, bad words, penalties, or processors. Every
    other sampler contract executes vLLM's original implementation.
    """
    global _GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD
    global _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH
    global _GREEDY_SAMPLE_LOGPROBS_REGISTERED
    if _GREEDY_SAMPLE_LOGPROBS_REGISTERED:
        return GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    if not supports_installed_vllm():
        return None

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.sampler import Sampler

    load_torch_extension()
    implementation = torch.ops.loom_kernels.greedy_sample_logprobs.default
    original_forward = Sampler.forward

    def non_argmax_processors_are_inactive(processors: list[Any]) -> bool:
        for processor in processors:
            if isinstance(processor, MinTokensLogitsProcessor):
                if not processor.min_toks:
                    continue
            elif isinstance(processor, LogitBiasLogitsProcessor):
                if not processor.biases:
                    continue
            elif isinstance(processor, AdapterLogitsProcessor):
                if not processor.req_info:
                    continue
            return False
        return True

    def can_use_fast_path(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool,
        logprobs_mode_override: Any,
    ) -> bool:
        logprobs_mode = logprobs_mode_override or sampler.logprobs_mode
        holder = sampling_metadata.thinking_budget_state_holder
        thinking_active = holder is not None and holder.has_tracked_requests()
        return bool(
            logprobs_mode == "raw_logprobs"
            and sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs == 0
            and not sampling_metadata.logprob_token_ids
            and sampling_metadata.no_penalties
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and non_argmax_processors_are_inactive(
                sampling_metadata.logitsprocs.non_argmax_invariant
            )
            and not thinking_active
            and not predict_bonus_token
            and logits.device.type == "cuda"
            and logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and logits.dim() == 2
            and logits.shape[0] > 0
            and logits.shape[1] > 0
            and logits.shape[1] <= 0x7FFF_FFFF
            and logits.stride(1) == 1
            and logits.stride(0) >= logits.shape[1]
            and not logits.requires_grad
        )

    def forward(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        global _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT
        global _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION
        use_fast_path = can_use_fast_path(
            sampler,
            logits,
            sampling_metadata,
            predict_bonus_token,
            logprobs_mode_override,
        )
        if not use_fast_path:
            if (
                _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION is None
                and (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.all_greedy
                )
            ):
                holder = sampling_metadata.thinking_budget_state_holder
                _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION = {
                    "shape": list(logits.shape),
                    "stride": list(logits.stride()),
                    "dtype": str(logits.dtype),
                    "logprobs_mode": (
                        logprobs_mode_override or sampler.logprobs_mode
                    ),
                    "max_num_logprobs": sampling_metadata.max_num_logprobs,
                    "has_logprob_token_ids": bool(
                        sampling_metadata.logprob_token_ids
                    ),
                    "all_greedy": sampling_metadata.all_greedy,
                    "no_penalties": sampling_metadata.no_penalties,
                    "has_allowed_mask": (
                        sampling_metadata.allowed_token_ids_mask is not None
                    ),
                    "has_bad_words": bool(sampling_metadata.bad_words_token_ids),
                    "non_argmax_processors": len(
                        sampling_metadata.logitsprocs.non_argmax_invariant
                    ),
                    "thinking_active": (
                        holder is not None and holder.has_tracked_requests()
                    ),
                    "predict_bonus_token": predict_bonus_token,
                    "is_contiguous": logits.is_contiguous(),
                    "requires_grad": logits.requires_grad,
                }
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        if _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT is None:
            _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT = {
                "shape": list(logits.shape),
                "stride": list(logits.stride()),
                "dtype": str(logits.dtype),
                "max_num_logprobs": sampling_metadata.max_num_logprobs,
                "all_greedy": sampling_metadata.all_greedy,
            }
        token_ids, logprobs, ranks = implementation(logits)
        token_ids = token_ids.unsqueeze(-1)
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=token_ids,
            logprobs=logprobs.unsqueeze(-1),
            selected_token_ranks=ranks,
        )
        return SamplerOutput(
            sampled_token_ids=token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    _GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD = original_forward
    _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH = can_use_fast_path
    Sampler.forward = forward
    _GREEDY_SAMPLE_LOGPROBS_REGISTERED = True
    return GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY


def register_vllm_selected_token_logprobs() -> str | None:
    """Avoid full-vocabulary raw log-softmax after vLLM 0.24/0.25 sampling.

    vLLM remains responsible for masks, processors, penalties, temperature,
    top-k/top-p, RNG, and token selection. For BF16/FP16 logits requesting
    only the sampled token's raw logprob (`max_num_logprobs == 0`), Loom scans
    the preserved raw logits once after sampling and returns just that token's
    normalized logprob and tie-aware rank. Other contracts execute vLLM's
    original implementation. All-greedy batches retain Loom's narrower fused
    argmax+logprob path.
    """
    global _SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD
    global _SELECTED_TOKEN_LOGPROBS_REGISTERED
    if _SELECTED_TOKEN_LOGPROBS_REGISTERED:
        return SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY
    if register_vllm_greedy_sample_logprobs() is None:
        return None

    if not supports_installed_vllm():
        return None

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    load_torch_extension()
    implementation = torch.ops.loom_kernels.selected_token_logprobs.default
    original_forward = Sampler.forward

    def can_use_fast_path(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        logprobs_mode_override: Any,
    ) -> bool:
        logprobs_mode = logprobs_mode_override or sampler.logprobs_mode
        topk_topp_mode = getattr(
            sampler.topk_topp_sampler, "logprobs_mode", sampler.logprobs_mode
        )
        return bool(
            sampler.logprobs_mode == "raw_logprobs"
            and topk_topp_mode == "raw_logprobs"
            and logprobs_mode == "raw_logprobs"
            and sampling_metadata.max_num_logprobs == 0
            and not sampling_metadata.logprob_token_ids
            and logits.device.type == "cuda"
            and logits.dtype in (torch.float16, torch.bfloat16)
            and logits.dim() == 2
            and logits.shape[0] > 0
            and logits.shape[1] > 0
            and logits.shape[1] <= 0x7FFF_FFFF
            and logits.stride(1) == 1
            and logits.stride(0) >= logits.shape[1]
            and not logits.requires_grad
        )

    def forward(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        global _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT
        global _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION
        if not can_use_fast_path(
            sampler, logits, sampling_metadata, logprobs_mode_override
        ):
            if (
                _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION is None
                and sampling_metadata.max_num_logprobs is not None
                and not sampling_metadata.all_greedy
            ):
                _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION = {
                    "shape": list(logits.shape),
                    "stride": list(logits.stride()),
                    "dtype": str(logits.dtype),
                    "sampler_logprobs_mode": sampler.logprobs_mode,
                    "logprobs_mode": (
                        logprobs_mode_override or sampler.logprobs_mode
                    ),
                    "max_num_logprobs": sampling_metadata.max_num_logprobs,
                    "has_logprob_token_ids": bool(
                        sampling_metadata.logprob_token_ids
                    ),
                    "all_greedy": sampling_metadata.all_greedy,
                    "requires_grad": logits.requires_grad,
                }
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        if (
            sampling_metadata.all_greedy
            and _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH is not None
            and _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )
        ):
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        if _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT is None:
            _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT = {
                "shape": list(logits.shape),
                "stride": list(logits.stride()),
                "dtype": str(logits.dtype),
                "max_num_logprobs": sampling_metadata.max_num_logprobs,
                "all_random": sampling_metadata.all_random,
                "has_top_k": sampling_metadata.top_k is not None,
                "has_top_p": sampling_metadata.top_p is not None,
                "no_penalties": sampling_metadata.no_penalties,
                "predict_bonus_token": predict_bonus_token,
            }

        raw_logits = logits
        sampling_logits = logits.to(torch.float32)
        sampling_logits = sampler.apply_logits_processors(
            sampling_logits, sampling_metadata, predict_bonus_token
        )
        sampled, processed_logprobs = sampler.sample(
            sampling_logits, sampling_metadata
        )
        if processed_logprobs is not None:
            raise RuntimeError(
                "vLLM returned processed logprobs under Loom's raw-logprob "
                "selected-token contract"
            )
        sampled = sampled.long().contiguous()
        logprobs, ranks = implementation(raw_logits, sampled)
        sampled = sampled.to(torch.int32)
        sampled_column = sampled.unsqueeze(-1)
        return SamplerOutput(
            sampled_token_ids=sampled_column,
            logprobs_tensors=LogprobsTensors(
                logprob_token_ids=sampled_column,
                logprobs=logprobs.unsqueeze(-1),
                selected_token_ranks=ranks,
            ),
        )

    _SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD = original_forward
    Sampler.forward = forward
    _SELECTED_TOKEN_LOGPROBS_REGISTERED = True
    return SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY

def _metadata() -> dict[str, object]:
    return {
        "greedy_sample_logprobs_override": _GREEDY_SAMPLE_LOGPROBS_REGISTERED,
        "greedy_sample_logprobs_first_contract": _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT,
        "greedy_sample_logprobs_first_rejection": _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION,
        "selected_token_logprobs_override": _SELECTED_TOKEN_LOGPROBS_REGISTERED,
        "selected_token_logprobs_first_contract": _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT,
        "selected_token_logprobs_first_rejection": _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION,
    }
