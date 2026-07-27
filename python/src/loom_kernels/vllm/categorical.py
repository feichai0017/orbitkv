"""Persistent request-state integration for vLLM categorical sampling."""

from __future__ import annotations

import contextvars
import inspect
from typing import Any

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import supports_installed_vllm

CATEGORICAL_SAMPLE_OVERRIDE_KEY = "categorical_sample"

_BATCH_STATE_ATTRIBUTE = "_loom_categorical_batch_state"
_BATCH_REQUESTS_ATTRIBUTE = "_loom_categorical_batch_requests"
_METADATA_STATE_ATTRIBUTE = "_loom_categorical_rng_state"
_REQUEST_SEED_ATTRIBUTE = "_loom_categorical_seed"
_REQUEST_STATE_ATTRIBUTE = "_loom_categorical_rng_state"

_CATEGORICAL_SAMPLE_REGISTERED = False
_CATEGORICAL_SAMPLE_ORIGINAL_INPUT_BATCH_INIT: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_ADD_REQUEST: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_REMOVE_REQUEST: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_SWAP_STATES: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_CONDENSE: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_MAKE_METADATA: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_SAMPLE: Any | None = None
_CATEGORICAL_SAMPLE_ORIGINAL_RANDOM_SAMPLE: Any | None = None
_CATEGORICAL_SAMPLE_FIRST_CONTRACT: dict[str, Any] | None = None
_CATEGORICAL_SAMPLE_FIRST_REJECTION: dict[str, Any] | None = None

_ACTIVE_CATEGORICAL_STATE: contextvars.ContextVar[torch.Tensor | None] = (
    contextvars.ContextVar(
        "loom_vllm_categorical_state",
        default=None,
    )
)


def _attach_batch_state_to_metadata(batch: Any, metadata: Any) -> Any:
    state = getattr(batch, _BATCH_STATE_ATTRIBUTE, None)
    if state is None:
        setattr(metadata, _METADATA_STATE_ATTRIBUTE, None)
    else:
        setattr(
            metadata,
            _METADATA_STATE_ATTRIBUTE,
            state[: batch.num_reqs],
        )
    return metadata


def _validate_seed(seed: Any) -> int:
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or seed > 0x7FFF_FFFF_FFFF_FFFF
    ):
        raise ValueError(
            "Loom categorical sampling requires every random vLLM request "
            "to provide a non-negative signed-int64 seed"
        )
    return seed


def _validate_persistent_state(
    state: Any,
    device: torch.device,
) -> torch.Tensor:
    if not (
        isinstance(state, torch.Tensor)
        and state.device == device
        and state.dtype == torch.int64
        and state.shape == (2,)
        and state.is_contiguous()
        and not state.requires_grad
    ):
        raise RuntimeError(
            "Loom found an invalid persistent categorical RNG state on a "
            "resumed vLLM request"
        )
    return state


def register_vllm_categorical_sample() -> str | None:
    """Own vLLM's explicitly seeded categorical RNG for an engine lifetime.

    Registration must happen before constructing the vLLM engine. The adapter
    deliberately enforces one service contract: every random request has an
    explicit signed-int64 seed, and speculative decoding is disabled. Greedy
    requests may share the batch. Once registered, unsupported random requests
    fail at admission instead of silently switching an in-flight request
    between vLLM and Loom RNG streams.
    """

    global _CATEGORICAL_SAMPLE_FIRST_CONTRACT
    global _CATEGORICAL_SAMPLE_FIRST_REJECTION
    global _CATEGORICAL_SAMPLE_ORIGINAL_ADD_REQUEST
    global _CATEGORICAL_SAMPLE_ORIGINAL_CONDENSE
    global _CATEGORICAL_SAMPLE_ORIGINAL_INPUT_BATCH_INIT
    global _CATEGORICAL_SAMPLE_ORIGINAL_MAKE_METADATA
    global _CATEGORICAL_SAMPLE_ORIGINAL_RANDOM_SAMPLE
    global _CATEGORICAL_SAMPLE_ORIGINAL_REMOVE_REQUEST
    global _CATEGORICAL_SAMPLE_ORIGINAL_SAMPLE
    global _CATEGORICAL_SAMPLE_ORIGINAL_SWAP_STATES
    global _CATEGORICAL_SAMPLE_REGISTERED

    if _CATEGORICAL_SAMPLE_REGISTERED:
        return CATEGORICAL_SAMPLE_OVERRIDE_KEY
    if not torch_extension_available() or not supports_installed_vllm():
        return None

    from vllm.sampling_params import SamplingType
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu_input_batch import InputBatch

    from ..torch_ops import supports_categorical_sample

    load_torch_extension()
    implementation = torch.ops.loom_kernels.categorical_sample.default

    original_input_batch_init = InputBatch.__init__
    original_add_request = InputBatch.add_request
    original_remove_request = InputBatch.remove_request
    original_swap_states = InputBatch.swap_states
    original_condense = InputBatch.condense
    original_make_metadata = InputBatch._make_sampling_metadata
    original_sample = Sampler.sample
    original_random_sample = topk_topp_sampler.random_sample
    input_batch_signature = inspect.signature(original_input_batch_init)

    def input_batch_init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = input_batch_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        num_spec_tokens = int(bound.arguments["num_spec_tokens"])
        is_pooling_model = bool(bound.arguments["is_pooling_model"])
        if num_spec_tokens != 0 and not is_pooling_model:
            raise RuntimeError(
                "Loom categorical sampling does not support speculative "
                "decoding; construct a non-speculative vLLM engine"
            )

        original_input_batch_init(self, *args, **kwargs)
        if is_pooling_model:
            setattr(self, _BATCH_STATE_ATTRIBUTE, None)
            setattr(self, _BATCH_REQUESTS_ATTRIBUTE, None)
        else:
            setattr(
                self,
                _BATCH_STATE_ATTRIBUTE,
                torch.zeros(
                    (self.max_num_reqs, 2),
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
            setattr(
                self,
                _BATCH_REQUESTS_ATTRIBUTE,
                [None] * self.max_num_reqs,
            )
        _attach_batch_state_to_metadata(self, self.sampling_metadata)

    def add_request(self: Any, request: Any) -> int:
        batch_state = getattr(self, _BATCH_STATE_ATTRIBUTE, None)
        if batch_state is None:
            return original_add_request(self, request)

        sampling_params = request.sampling_params
        persistent_state: torch.Tensor | None = None
        seed: int | None = None
        if sampling_params is not None:
            sampling_type = sampling_params.sampling_type
            if sampling_type != SamplingType.GREEDY:
                if (
                    sampling_type != SamplingType.RANDOM_SEED
                    or request.generator is None
                ):
                    raise ValueError(
                        "Loom categorical sampling owns the engine RNG and "
                        "requires an explicit seed on every random request"
                    )
                seed = _validate_seed(sampling_params.seed)
                persistent_state = getattr(
                    request,
                    _REQUEST_STATE_ATTRIBUTE,
                    None,
                )
                if persistent_state is None:
                    persistent_state = torch.tensor(
                        (seed, 0),
                        dtype=torch.int64,
                        device=self.device,
                    )
                else:
                    persistent_state = _validate_persistent_state(
                        persistent_state,
                        batch_state.device,
                    )
                    previous_seed = getattr(
                        request,
                        _REQUEST_SEED_ATTRIBUTE,
                        seed,
                    )
                    if previous_seed != seed:
                        raise ValueError(
                            "Loom cannot change the seed of an in-flight "
                            "vLLM request"
                        )

        req_index = original_add_request(self, request)
        requests = getattr(self, _BATCH_REQUESTS_ATTRIBUTE)
        requests[req_index] = request
        if persistent_state is None:
            batch_state[req_index].zero_()
        else:
            setattr(request, _REQUEST_SEED_ATTRIBUTE, seed)
            setattr(request, _REQUEST_STATE_ATTRIBUTE, persistent_state)
            batch_state[req_index].copy_(persistent_state)
        return req_index

    def remove_request(self: Any, req_id: str) -> int | None:
        batch_state = getattr(self, _BATCH_STATE_ATTRIBUTE, None)
        requests = getattr(self, _BATCH_REQUESTS_ATTRIBUTE, None)
        req_index = self.req_id_to_index.get(req_id)
        if batch_state is not None and requests is not None and req_index is not None:
            request = requests[req_index]
            if request is not None:
                persistent_state = getattr(
                    request,
                    _REQUEST_STATE_ATTRIBUTE,
                    None,
                )
                if persistent_state is not None:
                    persistent_state.copy_(batch_state[req_index])

        removed_index = original_remove_request(self, req_id)
        if requests is not None and removed_index is not None:
            requests[removed_index] = None
        return removed_index

    def swap_states(self: Any, i1: int, i2: int) -> None:
        original_swap_states(self, i1, i2)
        batch_state = getattr(self, _BATCH_STATE_ATTRIBUTE, None)
        requests = getattr(self, _BATCH_REQUESTS_ATTRIBUTE, None)
        if batch_state is None or requests is None:
            return
        temporary = batch_state[i1].clone()
        batch_state[i1].copy_(batch_state[i2])
        batch_state[i2].copy_(temporary)
        requests[i1], requests[i2] = requests[i2], requests[i1]

    def condense(self: Any) -> None:
        batch_state = getattr(self, _BATCH_STATE_ATTRIBUTE, None)
        requests = getattr(self, _BATCH_REQUESTS_ATTRIBUTE, None)
        moved_before = len(self.batch_update_builder.moved)
        original_condense(self)
        if batch_state is None or requests is None:
            return
        for source, destination, _direction in self.batch_update_builder.moved[
            moved_before:
        ]:
            batch_state[destination].copy_(batch_state[source])
            requests[destination] = requests[source]
            requests[source] = None
        if self.num_reqs == 0:
            requests[:] = [None] * len(requests)

    def make_sampling_metadata(self: Any) -> Any:
        metadata = original_make_metadata(self)
        return _attach_batch_state_to_metadata(self, metadata)

    def sample(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        logprobs_mode_override: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = getattr(sampling_metadata, _METADATA_STATE_ATTRIBUTE, None)
        if state is None:
            return original_sample(
                self,
                logits,
                sampling_metadata,
                logprobs_mode_override,
            )
        if state.shape[0] != logits.shape[0]:
            raise RuntimeError(
                "Loom categorical RNG state is stale relative to the vLLM "
                "sampling batch"
            )
        token = _ACTIVE_CATEGORICAL_STATE.set(state)
        try:
            return original_sample(
                self,
                logits,
                sampling_metadata,
                logprobs_mode_override,
            )
        finally:
            _ACTIVE_CATEGORICAL_STATE.reset(token)

    def random_sample(
        probabilities: torch.Tensor,
        generators: dict[int, torch.Generator],
        use_fp64_gumbel: bool = False,
    ) -> torch.Tensor:
        global _CATEGORICAL_SAMPLE_FIRST_CONTRACT
        global _CATEGORICAL_SAMPLE_FIRST_REJECTION

        state = _ACTIVE_CATEGORICAL_STATE.get()
        if state is None:
            return original_random_sample(
                probabilities,
                generators,
                use_fp64_gumbel,
            )

        if not generators:
            if _CATEGORICAL_SAMPLE_FIRST_REJECTION is None:
                _CATEGORICAL_SAMPLE_FIRST_REJECTION = {
                    "reason": "random row without explicit request generator",
                    "shape": list(probabilities.shape),
                }
            raise RuntimeError(
                "Loom categorical sampling reached a random vLLM batch "
                "without explicit per-request generators"
            )
        if not supports_categorical_sample(probabilities, state):
            if _CATEGORICAL_SAMPLE_FIRST_REJECTION is None:
                _CATEGORICAL_SAMPLE_FIRST_REJECTION = {
                    "reason": "unsupported probability or state tensor",
                    "shape": list(probabilities.shape),
                    "dtype": str(probabilities.dtype),
                    "contiguous": probabilities.is_contiguous(),
                    "state_shape": list(state.shape),
                }
            raise RuntimeError(
                "vLLM produced a categorical probability tensor outside "
                "Loom's registered engine contract"
            )

        if _CATEGORICAL_SAMPLE_FIRST_CONTRACT is None:
            _CATEGORICAL_SAMPLE_FIRST_CONTRACT = {
                "shape": list(probabilities.shape),
                "dtype": str(probabilities.dtype),
                "seeded_rows": len(generators),
                "persistent_state": True,
                "use_fp64_gumbel": use_fp64_gumbel,
            }
        return implementation(probabilities, state)

    input_batch_init.__module__ = __name__
    add_request.__module__ = __name__
    remove_request.__module__ = __name__
    swap_states.__module__ = __name__
    condense.__module__ = __name__
    make_sampling_metadata.__module__ = __name__
    sample.__module__ = __name__
    random_sample.__module__ = __name__

    _CATEGORICAL_SAMPLE_ORIGINAL_INPUT_BATCH_INIT = original_input_batch_init
    _CATEGORICAL_SAMPLE_ORIGINAL_ADD_REQUEST = original_add_request
    _CATEGORICAL_SAMPLE_ORIGINAL_REMOVE_REQUEST = original_remove_request
    _CATEGORICAL_SAMPLE_ORIGINAL_SWAP_STATES = original_swap_states
    _CATEGORICAL_SAMPLE_ORIGINAL_CONDENSE = original_condense
    _CATEGORICAL_SAMPLE_ORIGINAL_MAKE_METADATA = original_make_metadata
    _CATEGORICAL_SAMPLE_ORIGINAL_SAMPLE = original_sample
    _CATEGORICAL_SAMPLE_ORIGINAL_RANDOM_SAMPLE = original_random_sample

    InputBatch.__init__ = input_batch_init
    InputBatch.add_request = add_request
    InputBatch.remove_request = remove_request
    InputBatch.swap_states = swap_states
    InputBatch.condense = condense
    InputBatch._make_sampling_metadata = make_sampling_metadata
    Sampler.sample = sample
    topk_topp_sampler.random_sample = random_sample

    _CATEGORICAL_SAMPLE_REGISTERED = True
    return CATEGORICAL_SAMPLE_OVERRIDE_KEY


def _metadata() -> dict[str, object]:
    return {
        "categorical_sample_override": _CATEGORICAL_SAMPLE_REGISTERED,
        "categorical_sample_first_contract": _CATEGORICAL_SAMPLE_FIRST_CONTRACT,
        "categorical_sample_first_rejection": _CATEGORICAL_SAMPLE_FIRST_REJECTION,
        "categorical_sample_policy": (
            "all random requests explicitly seeded; non-speculative engine"
        ),
    }
