"""vLLM sampling-tail registrations."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import supports_installed_vllm

GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY = "greedy_sample_logprobs"
SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY = "selected_token_logprobs"
TOKEN_PENALTIES_OVERRIDE_KEY = "token_penalties"
TOP_K_FILTER_OVERRIDE_KEY = "top_k_filter"
TOP_K_FILTER_MAX_ROWS = 7
TOPK_SAMPLED_LOGPROBS_OVERRIDE_KEY = "topk_sampled_logprobs"
TOPK_SAMPLED_LOGPROBS_MAX_ROWS = 32

_GREEDY_SAMPLE_LOGPROBS_REGISTERED = False
_GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH: Any | None = None
_GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None
_SELECTED_TOKEN_LOGPROBS_REGISTERED = False
_SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None
_TOKEN_PENALTIES_REGISTERED = False
_TOKEN_PENALTIES_ORIGINAL_APPLY: Any | None = None
_TOKEN_PENALTIES_FIRST_CONTRACT: dict[str, Any] | None = None
_TOKEN_PENALTIES_FIRST_REJECTION: dict[str, Any] | None = None
_TOKEN_PENALTIES_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
_TOP_K_FILTER_REGISTERED = False
_TOP_K_FILTER_ORIGINAL_APPLY: Any | None = None
_TOPK_SAMPLED_LOGPROBS_REGISTERED = False
_TOPK_SAMPLED_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_TOPK_SAMPLED_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_TOPK_SAMPLED_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None


def _token_penalties_workspace(
    logits: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(logits.device).cuda_stream)
    key = (device_index, stream_id)
    workspace = _TOKEN_PENALTIES_WORKSPACES.get(key)
    if (
        workspace is None
        or workspace.shape[0] < rows
        or workspace.shape[1] < capacity
    ):
        workspace = torch.empty(
            (
                max(rows, 0 if workspace is None else workspace.shape[0]),
                max(capacity, 0 if workspace is None else workspace.shape[1]),
            ),
            dtype=torch.int64,
            device=logits.device,
        )
        _TOKEN_PENALTIES_WORKSPACES[key] = workspace
    return workspace[:rows, :capacity]


def register_vllm_top_k_filter() -> str | None:
    """Replace vLLM's native top-k-only sort with Loom's exact filter.

    This registration deliberately leaves FlashInfer, top-p, softmax, random
    sampling, per-request generators, and processed-logit return semantics
    under vLLM's ownership. It changes only the top-k-only filtering step used
    by the PyTorch-native fallback.
    """
    global _TOP_K_FILTER_ORIGINAL_APPLY
    global _TOP_K_FILTER_REGISTERED
    if _TOP_K_FILTER_REGISTERED:
        return TOP_K_FILTER_OVERRIDE_KEY
    if not torch_extension_available() or not supports_installed_vllm():
        return None

    from vllm.v1.sample.ops import topk_topp_sampler

    from ..torch_ops import supports_top_k_filter

    implementation = torch.ops.loom_kernels.top_k_filter_.default
    original_apply = topk_topp_sampler.apply_top_k_top_p

    def apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            p is None
            and k is not None
            and logits.shape[0] <= TOP_K_FILTER_MAX_ROWS
            and supports_top_k_filter(logits, k)
        ):
            implementation(logits, k)
            return logits
        return original_apply(logits, k, p)

    _TOP_K_FILTER_ORIGINAL_APPLY = original_apply
    topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p
    _TOP_K_FILTER_REGISTERED = True
    return TOP_K_FILTER_OVERRIDE_KEY


def register_vllm_token_penalties() -> str | None:
    """Replace full-vocabulary vLLM penalty temporaries with sparse history."""
    global _TOKEN_PENALTIES_ORIGINAL_APPLY
    global _TOKEN_PENALTIES_REGISTERED
    if _TOKEN_PENALTIES_REGISTERED:
        return TOKEN_PENALTIES_OVERRIDE_KEY
    if not torch_extension_available() or not supports_installed_vllm():
        return None

    from vllm.v1.sample.ops.penalties import _convert_to_tensors
    from vllm.v1.sample.sampler import Sampler

    from ..torch_ops import (
        supports_apply_token_penalties,
        token_penalties_workspace_capacity,
    )

    implementation = torch.ops.loom_kernels.apply_token_penalties_.default
    original_apply = Sampler.apply_penalties

    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: Any,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        global _TOKEN_PENALTIES_FIRST_CONTRACT
        global _TOKEN_PENALTIES_FIRST_REJECTION
        if sampling_metadata.no_penalties:
            return logits
        prompt_token_ids = sampling_metadata.prompt_token_ids
        output_width = max(
            1,
            max((len(tokens) for tokens in output_token_ids), default=0),
        )
        basic_contract = bool(
            prompt_token_ids is not None
            and logits.device.type == "cuda"
            and logits.dtype == torch.float32
            and logits.dim() == 2
            and logits.shape[0] > 0
            and logits.shape[1] > 0
            and prompt_token_ids.device == logits.device
            and prompt_token_ids.dtype == torch.int64
            and prompt_token_ids.dim() == 2
            and prompt_token_ids.shape[0] == logits.shape[0]
            and prompt_token_ids.shape[1] > 0
        )
        if not basic_contract:
            if _TOKEN_PENALTIES_FIRST_REJECTION is None:
                _TOKEN_PENALTIES_FIRST_REJECTION = {
                    "reason": "unsupported logits or prompt tensor contract",
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                }
            return original_apply(logits, sampling_metadata, output_token_ids)

        capacity = token_penalties_workspace_capacity(
            prompt_token_ids.shape[1], output_width
        )
        # Hashing more slots than the vocabulary ceases to be a sparse path.
        if capacity > logits.shape[1]:
            if _TOKEN_PENALTIES_FIRST_REJECTION is None:
                _TOKEN_PENALTIES_FIRST_REJECTION = {
                    "reason": "history workspace is not sparse versus vocabulary",
                    "shape": list(logits.shape),
                    "prompt_tokens": prompt_token_ids.shape[1],
                    "output_tokens": output_width,
                    "workspace_capacity": capacity,
                }
            return original_apply(logits, sampling_metadata, output_token_ids)

        if all(not tokens for tokens in output_token_ids):
            output_tokens_t = torch.full(
                (logits.shape[0], 1),
                logits.shape[1],
                dtype=torch.int64,
                device=logits.device,
            )
        else:
            output_tokens_t = _convert_to_tensors(
                output_token_ids, logits.shape[1], logits.device
            )
        workspace = _token_penalties_workspace(logits, capacity)
        supported = supports_apply_token_penalties(
            logits,
            prompt_token_ids,
            output_tokens_t,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            workspace,
        )
        if not supported:
            if _TOKEN_PENALTIES_FIRST_REJECTION is None:
                _TOKEN_PENALTIES_FIRST_REJECTION = {
                    "reason": "expanded tensor contract is unsupported",
                    "shape": list(logits.shape),
                    "prompt_tokens": prompt_token_ids.shape[1],
                    "output_tokens": output_tokens_t.shape[1],
                    "workspace_capacity": capacity,
                }
            return original_apply(logits, sampling_metadata, output_token_ids)

        if _TOKEN_PENALTIES_FIRST_CONTRACT is None:
            _TOKEN_PENALTIES_FIRST_CONTRACT = {
                "shape": list(logits.shape),
                "prompt_tokens": prompt_token_ids.shape[1],
                "output_tokens": output_tokens_t.shape[1],
                "workspace_capacity": capacity,
                "workspace_bytes": workspace.numel() * workspace.element_size(),
            }
        implementation(
            logits,
            prompt_token_ids,
            output_tokens_t,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            workspace,
        )
        return logits

    apply_penalties.__module__ = __name__
    _TOKEN_PENALTIES_ORIGINAL_APPLY = original_apply
    Sampler.apply_penalties = staticmethod(apply_penalties)
    _TOKEN_PENALTIES_REGISTERED = True
    return TOKEN_PENALTIES_OVERRIDE_KEY


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


def register_vllm_topk_sampled_logprobs() -> str | None:
    """Remove vLLM's full-vocabulary raw-logprob tensor around top-k.

    vLLM retains ownership of processors, penalties, temperature, top-k/top-p
    sampling, RNG, the selected token, and `torch.topk`'s observable tie order.
    Loom replaces full-vocabulary log-softmax and sampled-token ranking with
    its selected-token reduction, then normalizes vLLM's small top-k values.
    This path is admitted only for one through 32 requested raw logprobs and
    no separately requested token IDs.
    """
    global _TOPK_SAMPLED_LOGPROBS_ORIGINAL_FORWARD
    global _TOPK_SAMPLED_LOGPROBS_REGISTERED
    if _TOPK_SAMPLED_LOGPROBS_REGISTERED:
        return TOPK_SAMPLED_LOGPROBS_OVERRIDE_KEY
    if register_vllm_selected_token_logprobs() is None:
        return None
    if not supports_installed_vllm():
        return None

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    load_torch_extension()
    selected_implementation = (
        torch.ops.loom_kernels.selected_token_logprobs.default
    )
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
        num_logprobs = sampling_metadata.max_num_logprobs
        maximum = min(logits.shape[1], 32) if logits.dim() == 2 else 0
        return bool(
            sampler.logprobs_mode == "raw_logprobs"
            and topk_topp_mode == "raw_logprobs"
            and logprobs_mode == "raw_logprobs"
            and isinstance(num_logprobs, int)
            and not isinstance(num_logprobs, bool)
            and 1 <= num_logprobs <= maximum
            and logits.shape[0] <= TOPK_SAMPLED_LOGPROBS_MAX_ROWS
            and not sampling_metadata.logprob_token_ids
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
        global _TOPK_SAMPLED_LOGPROBS_FIRST_CONTRACT
        global _TOPK_SAMPLED_LOGPROBS_FIRST_REJECTION
        if not can_use_fast_path(
            sampler, logits, sampling_metadata, logprobs_mode_override
        ):
            if (
                _TOPK_SAMPLED_LOGPROBS_FIRST_REJECTION is None
                and sampling_metadata.max_num_logprobs is not None
                and sampling_metadata.max_num_logprobs != 0
            ):
                _TOPK_SAMPLED_LOGPROBS_FIRST_REJECTION = {
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
                    "requires_grad": logits.requires_grad,
                }
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        num_logprobs = sampling_metadata.max_num_logprobs
        assert isinstance(num_logprobs, int)
        if _TOPK_SAMPLED_LOGPROBS_FIRST_CONTRACT is None:
            _TOPK_SAMPLED_LOGPROBS_FIRST_CONTRACT = {
                "shape": list(logits.shape),
                "stride": list(logits.stride()),
                "dtype": str(logits.dtype),
                "max_num_logprobs": num_logprobs,
                "all_greedy": sampling_metadata.all_greedy,
                "all_random": sampling_metadata.all_random,
                "has_top_k": sampling_metadata.top_k is not None,
                "has_top_p": sampling_metadata.top_p is not None,
                "no_penalties": sampling_metadata.no_penalties,
                "predict_bonus_token": predict_bonus_token,
                "topk_order": "vLLM torch.topk",
                "loom_kernel": "selected_token_logprobs",
            }

        raw_logits = logits
        sampling_logits = logits.to(torch.float32)
        topk_values, topk_token_ids = torch.topk(
            sampling_logits, num_logprobs, dim=-1
        )
        sampling_logits = sampler.apply_logits_processors(
            sampling_logits, sampling_metadata, predict_bonus_token
        )
        sampled, processed_logprobs = sampler.sample(
            sampling_logits, sampling_metadata
        )
        if processed_logprobs is not None:
            raise RuntimeError(
                "vLLM returned processed logprobs under Loom's raw-logprob "
                "top-k contract"
            )
        sampled = sampled.long().contiguous()
        sampled_logprobs, ranks = selected_implementation(
            raw_logits, sampled
        )
        sampled_raw_logits = (
            raw_logits.gather(-1, sampled.unsqueeze(-1))
            .to(torch.float32)
            .squeeze(-1)
        )
        log_normalizer = sampled_raw_logits - sampled_logprobs
        topk_logprobs = topk_values - log_normalizer.unsqueeze(-1)
        sampled_column = sampled.to(torch.int32).unsqueeze(-1)
        token_ids = torch.cat(
            (sampled_column, topk_token_ids.to(torch.int32)), dim=-1
        )
        logprobs = torch.cat(
            (sampled_logprobs.unsqueeze(-1), topk_logprobs), dim=-1
        )
        return SamplerOutput(
            sampled_token_ids=sampled_column,
            logprobs_tensors=LogprobsTensors(
                logprob_token_ids=token_ids,
                logprobs=logprobs,
                selected_token_ranks=ranks,
            ),
        )

    _TOPK_SAMPLED_LOGPROBS_ORIGINAL_FORWARD = original_forward
    Sampler.forward = forward
    _TOPK_SAMPLED_LOGPROBS_REGISTERED = True
    return TOPK_SAMPLED_LOGPROBS_OVERRIDE_KEY


def _metadata() -> dict[str, object]:
    return {
        "greedy_sample_logprobs_override": _GREEDY_SAMPLE_LOGPROBS_REGISTERED,
        "greedy_sample_logprobs_first_contract": _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT,
        "greedy_sample_logprobs_first_rejection": _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION,
        "selected_token_logprobs_override": _SELECTED_TOKEN_LOGPROBS_REGISTERED,
        "selected_token_logprobs_first_contract": _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT,
        "selected_token_logprobs_first_rejection": _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION,
        "token_penalties_override": _TOKEN_PENALTIES_REGISTERED,
        "token_penalties_first_contract": _TOKEN_PENALTIES_FIRST_CONTRACT,
        "token_penalties_first_rejection": _TOKEN_PENALTIES_FIRST_REJECTION,
        "top_k_filter_override": _TOP_K_FILTER_REGISTERED,
        "topk_sampled_logprobs_override": _TOPK_SAMPLED_LOGPROBS_REGISTERED,
        "topk_sampled_logprobs_first_contract": _TOPK_SAMPLED_LOGPROBS_FIRST_CONTRACT,
        "topk_sampled_logprobs_first_rejection": _TOPK_SAMPLED_LOGPROBS_FIRST_REJECTION,
    }
