from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from loom_kernels.vllm import (
    ACT_INT8_OVERRIDE_ENV,
    ACT_INT8_OVERRIDE_KEY,
    ACT_QUANT_OVERRIDE_ENV,
    ACT_QUANT_OVERRIDE_KEY,
    CATEGORICAL_SAMPLE_OVERRIDE_KEY,
    DEFAULT_PROVIDER,
    GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY,
    GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY,
    LOGITS_PREPROCESS_OVERRIDE_KEY,
    MIN_P_OVERRIDE_ENV,
    MIN_P_OVERRIDE_KEY,
    PAGED_DECODE_OVERRIDE_ENV,
    PAGED_DECODE_OVERRIDE_KEY,
    RMS_NORM_FP8_OVERRIDE_ENV,
    RMS_NORM_FP8_OVERRIDE_KEY,
    RMS_NORM_INT8_OVERRIDE_ENV,
    RMS_NORM_INT8_OVERRIDE_KEY,
    ROPE_PAGED_KV_OVERRIDE_KEY,
    SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY,
    SILU_OVERRIDE_ENV,
    SILU_OVERRIDE_KEY,
    SUPPORTED_VLLM_SERIES,
    TOKEN_PENALTIES_OVERRIDE_KEY,
    TOP_K_FILTER_MAX_ROWS,
    TOP_K_FILTER_OVERRIDE_KEY,
    TOP_P_RENORM_MAX_ROWS,
    TOP_P_RENORM_MIN_ROWS,
    TOP_P_RENORM_MIN_VOCAB_SIZE,
    TOP_P_RENORM_OVERRIDE_KEY,
    TOPK_SAMPLED_LOGPROBS_OVERRIDE_KEY,
    configure_vllm_rope_paged_kv,
    installed_vllm_version,
    provider_metadata,
    register_vllm_categorical_sample,
    register_vllm_ir,
    register_vllm_logits_preprocess,
    register_vllm_min_p,
    register_vllm_paged_decode_attention,
    register_vllm_greedy_sample_logprobs,
    register_vllm_greedy_speculative_verify,
    register_vllm_rope_paged_kv,
    register_vllm_rms_norm_dynamic_fp8,
    register_vllm_rms_norm_dynamic_int8,
    register_vllm_selected_token_logprobs,
    register_vllm_silu_and_mul,
    register_vllm_silu_and_mul_dynamic_fp8,
    register_vllm_silu_and_mul_dynamic_int8,
    register_vllm_token_penalties,
    register_vllm_top_k_filter,
    register_vllm_top_p_renorm,
    register_vllm_topk_sampled_logprobs,
    supports_installed_vllm,
    supports_vllm_paged_decode_shape,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_logits_preprocess_fuses_mixed_sampling_and_preserves_fallbacks():
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.sampler import Sampler

    import loom_kernels.vllm.logits as logits_integration
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    class CaptureSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, logits, generators, top_k, top_p):
            self.seen = logits.clone()
            return logits.argmax(dim=-1), logits.clone()

    assert (
        register_vllm_logits_preprocess()
        == LOGITS_PREPROCESS_OVERRIDE_KEY
    )
    original_apply = logits_integration._LOGITS_PREPROCESS_ORIGINAL_APPLY
    original_sample = logits_integration._LOGITS_PREPROCESS_ORIGINAL_SAMPLE
    assert original_apply is not None
    assert original_sample is not None

    rows, vocab_size = 3, 257
    temperatures = torch.tensor([0.0, 0.7, 1.2], device="cuda")
    all_random_temperatures = torch.tensor(
        [0.6, 0.7, 1.2], device="cuda"
    )
    blocked_mask = torch.zeros(
        (rows, vocab_size), device="cuda", dtype=torch.bool
    )
    blocked_mask[:, 13] = True
    bias = LogitBiasLogitsProcessor.__new__(LogitBiasLogitsProcessor)
    bias.biases = {0: {5: 0.5}, 2: {11: -0.25}}
    bias.logits_slice = (
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([5, 11], device="cuda", dtype=torch.int32),
    )
    bias.bias_tensor = torch.tensor([0.5, -0.25], device="cuda")
    min_tokens = MinTokensLogitsProcessor.__new__(MinTokensLogitsProcessor)
    min_tokens.min_toks = {1: (4, [], {7, 9})}
    min_tokens.logits_slice = (
        torch.tensor([1, 1], device="cuda", dtype=torch.int32),
        torch.tensor([7, 9], device="cuda", dtype=torch.int32),
    )
    min_tokens.neg_inf_tensor = torch.tensor(-float("inf"), device="cuda")

    def metadata(*, all_random: bool = False):
        return SimpleNamespace(
            temperature=(
                all_random_temperatures if all_random else temperatures
            ),
            all_greedy=False,
            all_random=all_random,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(rows, device="cuda"),
            presence_penalties=torch.zeros(rows, device="cuda"),
            repetition_penalties=torch.ones(rows, device="cuda"),
            output_token_ids=[[], [], []],
            allowed_token_ids_mask=blocked_mask,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors([min_tokens, bias]),
            logprob_token_ids=None,
            spec_token_ids=None,
            thinking_budget_state_holder=None,
        )

    torch.manual_seed(503)
    source = torch.randn((rows, vocab_size), device="cuda")
    expected_metadata = metadata()
    expected_sampler = Sampler(logprobs_mode="processed_logits")
    expected_sampler.topk_topp_sampler = CaptureSampler()
    expected_logits = original_apply(
        expected_sampler,
        source.clone(),
        expected_metadata,
        False,
    )
    expected_ids, expected_processed = original_sample(
        expected_sampler,
        expected_logits,
        expected_metadata,
    )

    actual_metadata = metadata()
    actual_sampler = Sampler(logprobs_mode="processed_logits")
    actual_sampler.topk_topp_sampler = CaptureSampler()
    actual_logits = source.clone()
    reset_launch_count(Operator.LOGITS_PREPROCESS)
    deferred = actual_sampler.apply_logits_processors(
        actual_logits, actual_metadata, False
    )
    assert deferred is actual_logits
    assert torch.equal(actual_logits, source)
    actual_ids, actual_processed = actual_sampler.sample(
        actual_logits, actual_metadata
    )
    torch.cuda.synchronize()

    assert launch_count(Operator.LOGITS_PREPROCESS) == 1
    assert torch.equal(actual_ids, expected_ids)
    assert actual_processed is not None
    assert expected_processed is not None
    torch.testing.assert_close(
        actual_processed, expected_processed, rtol=1.0e-6, atol=1.0e-6
    )
    contract = provider_metadata()["logits_preprocess_first_contract"]
    assert contract is not None
    assert contract["mixed_sampling"] is True
    assert contract["bias_count"] == 2
    assert contract["suppression_count"] == 2
    observed = provider_metadata()["logits_preprocess_observed"]
    assert observed["accepted_contracts"] >= 1
    assert observed["blocked_mask"] is True
    assert observed["maximum_bias_count"] >= 2
    assert observed["maximum_suppression_count"] >= 2
    assert observed["min_tokens"] is True

    fallback_metadata = metadata(all_random=True)
    fallback_logits = source.clone()
    reset_launch_count(Operator.LOGITS_PREPROCESS)
    fallback_sampler = Sampler()
    fallback_sampler.topk_topp_sampler = CaptureSampler()
    fallback_sampler.apply_logits_processors(
        fallback_logits, fallback_metadata, False
    )
    fallback_sampler.sample(fallback_logits, fallback_metadata)
    torch.cuda.synchronize()
    assert launch_count(Operator.LOGITS_PREPROCESS) == 0
    assert provider_metadata()["logits_preprocess_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_logits_preprocess_matches_active_bad_word_suppression():
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.sampler import Sampler

    import loom_kernels.vllm.logits as logits_integration
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_logits_preprocess()
        == LOGITS_PREPROCESS_OVERRIDE_KEY
    )
    original_apply = logits_integration._LOGITS_PREPROCESS_ORIGINAL_APPLY
    assert original_apply is not None

    rows, vocab_size = 2, 127
    source = torch.randn((rows, vocab_size), device="cuda")

    def metadata():
        return SimpleNamespace(
            temperature=torch.tensor([0.0, 0.8], device="cuda"),
            all_greedy=False,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(rows, device="cuda"),
            presence_penalties=torch.zeros(rows, device="cuda"),
            repetition_penalties=torch.ones(rows, device="cuda"),
            output_token_ids=[[4, 10], [3]],
            allowed_token_ids_mask=None,
            bad_words_token_ids={0: [[10, 17], [99]], 1: [[8, 23]]},
            logitsprocs=LogitsProcessors(),
            logprob_token_ids=None,
            spec_token_ids=None,
            thinking_budget_state_holder=None,
        )

    expected_metadata = metadata()
    expected_sampler = Sampler()
    expected = original_apply(
        expected_sampler, source.clone(), expected_metadata, False
    )
    expected.div_(
        torch.where(
            expected_metadata.temperature < 1.0e-5,
            1.0,
            expected_metadata.temperature,
        ).unsqueeze(1)
    )

    actual_metadata = metadata()
    actual_sampler = Sampler()
    actual = source.clone()
    reset_launch_count(Operator.LOGITS_PREPROCESS)
    actual_sampler.apply_logits_processors(
        actual, actual_metadata, False
    )
    state = getattr(
        actual_metadata,
        logits_integration._LOGITS_PREPROCESS_STATE_ATTRIBUTE,
    )
    torch.ops.loom_kernels.logits_preprocess_(
        actual,
        actual_metadata.temperature,
        state.blocked_mask,
        state.bias_row_ids,
        state.bias_token_ids,
        state.bias_values,
        state.suppressed_row_ids,
        state.suppressed_token_ids,
    )
    delattr(
        actual_metadata,
        logits_integration._LOGITS_PREPROCESS_STATE_ATTRIBUTE,
    )
    torch.cuda.synchronize()

    assert launch_count(Operator.LOGITS_PREPROCESS) == 1
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    assert torch.isneginf(actual[0, 17])
    assert torch.isneginf(actual[0, 99])
    assert provider_metadata()["logits_preprocess_observed"][
        "bad_words"
    ] is True


def test_installed_vllm_series_is_supported():
    assert SUPPORTED_VLLM_SERIES == ((0, 24), (0, 25))
    assert supports_installed_vllm()
    assert installed_vllm_version() is not None
    assert provider_metadata()["vllm_supported"] is True


def test_unqualified_vllm_series_is_rejected(monkeypatch):
    import loom_kernels.vllm as integration
    import loom_kernels.vllm._runtime as runtime

    monkeypatch.setattr(
        runtime, "installed_vllm_version", lambda: "0.26.0"
    )
    assert integration.supports_installed_vllm() is False
    assert integration.register_vllm_ir() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_greedy_sample_logprobs_fast_path_matches_sampler_semantics():
    from vllm.v1.sample.sampler import Sampler

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_greedy_sample_logprobs()
        == GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY
    )
    logits = torch.randn((5, 4096), device="cuda", dtype=torch.float32)
    metadata = SimpleNamespace(
        all_greedy=True,
        max_num_logprobs=0,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
        thinking_budget_state_holder=None,
    )
    reset_launch_count(Operator.GREEDY_SAMPLE_LOGPROBS)
    output = Sampler().forward(logits, metadata)
    expected_ids = logits.argmax(-1).to(torch.int32)
    expected_logprobs = logits.log_softmax(-1).gather(
        -1, expected_ids.long().unsqueeze(-1)
    )
    torch.cuda.synchronize()

    assert torch.equal(output.sampled_token_ids[:, 0], expected_ids)
    assert output.logprobs_tensors is not None
    torch.testing.assert_close(
        output.logprobs_tensors.logprobs,
        expected_logprobs,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    expected_ranks = (
        logits
        >= logits.gather(-1, expected_ids.long().unsqueeze(-1))
    ).sum(dim=-1)
    assert torch.equal(
        output.logprobs_tensors.selected_token_ranks,
        expected_ranks,
    )
    assert launch_count(Operator.GREEDY_SAMPLE_LOGPROBS) == 1
    assert provider_metadata()["greedy_sample_logprobs_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_greedy_speculative_verify_matches_rejection_semantics(
    monkeypatch,
):
    from vllm.v1.sample import rejection_sampler

    import loom_kernels.vllm.speculative as speculative_integration
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_greedy_speculative_verify()
        == GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY
    )
    draft = torch.tensor(
        [10, 11, 12, 20, 21, 22, 23],
        dtype=torch.int32,
        device="cuda",
    )
    target_ids = torch.tensor(
        [10, 99, 12, 20, 21, 22, 23],
        dtype=torch.int64,
        device="cuda",
    )
    target_logits = torch.full(
        (7, 128), -100.0, dtype=torch.float32, device="cuda"
    )
    target_logits.scatter_(1, target_ids.unsqueeze(1), 100.0)
    bonus = torch.tensor(
        [[100], [101], [102]], dtype=torch.int32, device="cuda"
    )
    cumulative = torch.tensor([3, 3, 7], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(all_greedy=True)

    reset_launch_count(Operator.GREEDY_SPECULATIVE_VERIFY)
    output = rejection_sampler.rejection_sample(
        draft,
        [3, 0, 4],
        4,
        cumulative,
        None,
        target_logits,
        bonus,
        metadata,
    )
    torch.cuda.synchronize()

    assert output.tolist() == [
        [10, 99, -1, -1, -1],
        [101, -1, -1, -1, -1],
        [20, 21, 22, 23, 102],
    ]
    assert launch_count(Operator.GREEDY_SPECULATIVE_VERIFY) == 1
    assert provider_metadata()["greedy_speculative_verify_override"] is True

    sentinel = torch.empty(0, dtype=torch.int32, device="cuda")
    fallback_calls = 0

    def fallback(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return sentinel

    monkeypatch.setattr(
        speculative_integration,
        "_GREEDY_SPECULATIVE_VERIFY_ORIGINAL",
        fallback,
    )
    metadata.all_greedy = False
    reset_launch_count(Operator.GREEDY_SPECULATIVE_VERIFY)
    fallback_output = rejection_sampler.rejection_sample(
        draft,
        [3, 0, 4],
        4,
        cumulative,
        None,
        target_logits,
        bonus,
        metadata,
    )

    assert fallback_output is sentinel
    assert fallback_calls == 1
    assert launch_count(Operator.GREEDY_SPECULATIVE_VERIFY) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_selected_token_fast_path_preserves_engine_selection(monkeypatch):
    from vllm.v1.sample.sampler import Sampler

    assert (
        register_vllm_selected_token_logprobs()
        == SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY
    )
    logits = torch.randn((5, 4096), device="cuda", dtype=torch.bfloat16)
    sampled = torch.tensor([0, 17, 2048, 4095, 7], device="cuda")
    metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        max_num_logprobs=0,
        logprob_token_ids=None,
        top_k=torch.full((5,), 50, device="cuda", dtype=torch.int32),
        top_p=torch.full((5,), 0.9, device="cuda"),
        no_penalties=False,
    )
    sampler = Sampler()
    observed = {}

    def apply_processors(sampling_logits, received_metadata, predict_bonus_token):
        observed["input_dtype"] = sampling_logits.dtype
        observed["metadata"] = received_metadata
        observed["predict_bonus_token"] = predict_bonus_token
        sampling_logits.add_(1.0)
        return sampling_logits

    def sample(sampling_logits, received_metadata):
        observed["sample_logits_dtype"] = sampling_logits.dtype
        observed["sample_metadata"] = received_metadata
        return sampled, None

    monkeypatch.setattr(sampler, "apply_logits_processors", apply_processors)
    monkeypatch.setattr(sampler, "sample", sample)
    output = sampler.forward(logits, metadata, predict_bonus_token=True)
    expected_logprobs = logits.float().log_softmax(-1).gather(
        -1, sampled.unsqueeze(-1)
    )
    selected = logits.float().gather(-1, sampled.unsqueeze(-1))
    expected_ranks = (logits.float() >= selected).sum(dim=-1)
    torch.cuda.synchronize()

    assert observed == {
        "input_dtype": torch.float32,
        "metadata": metadata,
        "predict_bonus_token": True,
        "sample_logits_dtype": torch.float32,
        "sample_metadata": metadata,
    }
    assert torch.equal(output.sampled_token_ids[:, 0], sampled.to(torch.int32))
    assert output.logprobs_tensors is not None
    torch.testing.assert_close(
        output.logprobs_tensors.logprobs,
        expected_logprobs,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert torch.equal(
        output.logprobs_tensors.selected_token_ranks, expected_ranks
    )
    assert provider_metadata()["selected_token_logprobs_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_selected_token_path_handles_processed_greedy_batches(monkeypatch):
    from vllm.v1.sample.sampler import Sampler

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    register_vllm_selected_token_logprobs()
    reset_launch_count(Operator.GREEDY_SAMPLE_LOGPROBS)
    reset_launch_count(Operator.SELECTED_TOKEN_LOGPROBS)
    logits = torch.randn((3, 1024), device="cuda", dtype=torch.float16)
    sampled = torch.tensor([7, 511, 1023], device="cuda")
    metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        max_num_logprobs=0,
        logprob_token_ids=None,
        no_penalties=False,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
        thinking_budget_state_holder=None,
        top_k=None,
        top_p=None,
    )
    sampler = Sampler()
    monkeypatch.setattr(
        sampler,
        "apply_logits_processors",
        lambda sampling_logits, _metadata, _predict_bonus: sampling_logits,
    )
    monkeypatch.setattr(
        sampler,
        "sample",
        lambda _sampling_logits, _metadata: (sampled, None),
    )

    output = sampler.forward(logits, metadata)
    torch.cuda.synchronize()

    assert torch.equal(output.sampled_token_ids[:, 0], sampled.to(torch.int32))
    assert launch_count(Operator.GREEDY_SAMPLE_LOGPROBS) == 0
    assert launch_count(Operator.SELECTED_TOKEN_LOGPROBS) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_topk_sampled_logprobs_preserves_engine_selection(monkeypatch):
    from vllm.v1.sample.sampler import Sampler

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_topk_sampled_logprobs()
        == TOPK_SAMPLED_LOGPROBS_OVERRIDE_KEY
    )
    rows, vocab_size, top_k = 5, 4096, 20
    torch.manual_seed(193)
    logits = torch.randn(
        (rows, vocab_size), device="cuda", dtype=torch.float32
    )
    sampled = torch.tensor([0, 17, 2048, 4095, 7], device="cuda")
    metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        max_num_logprobs=top_k,
        logprob_token_ids=None,
        top_k=torch.full((rows,), 50, device="cuda", dtype=torch.int32),
        top_p=torch.full((rows,), 0.9, device="cuda"),
        no_penalties=False,
    )
    sampler = Sampler()
    observed = {}

    def apply_processors(sampling_logits, received_metadata, predict_bonus_token):
        observed["input_dtype"] = sampling_logits.dtype
        observed["metadata"] = received_metadata
        observed["predict_bonus_token"] = predict_bonus_token
        return sampling_logits

    def sample(sampling_logits, received_metadata):
        observed["sample_logits_dtype"] = sampling_logits.dtype
        observed["sample_metadata"] = received_metadata
        return sampled, None

    monkeypatch.setattr(sampler, "apply_logits_processors", apply_processors)
    monkeypatch.setattr(sampler, "sample", sample)
    reset_launch_count(Operator.SELECTED_TOKEN_LOGPROBS)
    reset_launch_count(Operator.TOPK_SAMPLED_LOGPROBS)
    output = sampler.forward(logits, metadata, predict_bonus_token=True)
    raw_logprobs = logits.log_softmax(-1)
    top_logprobs, top_token_ids = torch.topk(raw_logprobs, top_k, dim=-1)
    sampled_logprobs = raw_logprobs.gather(-1, sampled.unsqueeze(-1))
    sampled_logits = logits.gather(-1, sampled.unsqueeze(-1))
    expected_ranks = (logits >= sampled_logits).sum(dim=-1)
    expected_ids = torch.cat(
        (sampled.unsqueeze(-1), top_token_ids), dim=-1
    ).to(torch.int32)
    expected_logprobs = torch.cat(
        (sampled_logprobs, top_logprobs), dim=-1
    )
    torch.cuda.synchronize()

    assert observed == {
        "input_dtype": torch.float32,
        "metadata": metadata,
        "predict_bonus_token": True,
        "sample_logits_dtype": torch.float32,
        "sample_metadata": metadata,
    }
    assert torch.equal(output.sampled_token_ids[:, 0], sampled.to(torch.int32))
    assert output.logprobs_tensors is not None
    assert torch.equal(
        output.logprobs_tensors.logprob_token_ids, expected_ids
    )
    torch.testing.assert_close(
        output.logprobs_tensors.logprobs,
        expected_logprobs,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert torch.equal(
        output.logprobs_tensors.selected_token_ranks, expected_ranks
    )
    assert launch_count(Operator.SELECTED_TOKEN_LOGPROBS) == 1
    assert launch_count(Operator.TOPK_SAMPLED_LOGPROBS) == 0
    assert provider_metadata()["topk_sampled_logprobs_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_top_k_filter_preserves_native_rng_and_top_p_fallback():
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

    import loom_kernels.vllm.sampling as sampling_integration
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert register_vllm_top_k_filter() == TOP_K_FILTER_OVERRIDE_KEY
    original_apply = sampling_integration._TOP_K_FILTER_ORIGINAL_APPLY
    assert original_apply is not None

    rows, vocab_size = 5, 4096
    torch.manual_seed(401)
    source = torch.randn((rows, vocab_size), device="cuda")
    source[0, :4] = torch.tensor([5.0, 4.0, 4.0, 1.0], device="cuda")
    top_ks = torch.tensor(
        [2, 17, 64, 1024, vocab_size],
        device="cuda",
        dtype=torch.int32,
    )

    def generators() -> dict[int, torch.Generator]:
        return {
            row: torch.Generator(device="cuda").manual_seed(1000 + row)
            for row in range(rows)
        }

    expected_processed = original_apply(
        source.clone(), top_ks.clone(), None
    )
    expected = topk_topp_sampler.random_sample(
        expected_processed.softmax(dim=-1, dtype=torch.float32),
        generators(),
    )
    sampler = TopKTopPSampler(logprobs_mode="processed_logits")
    reset_launch_count(Operator.TOP_K_FILTER)
    actual, actual_processed = sampler.forward_native(
        source.clone(), generators(), top_ks, None
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert actual_processed is not None
    assert torch.equal(actual_processed, expected_processed)
    assert torch.isfinite(actual_processed[0]).sum().item() == 3
    assert launch_count(Operator.TOP_K_FILTER) == 1

    top_p = torch.full((rows,), 0.9, device="cuda")
    expected_top_p = original_apply(source.clone(), top_ks.clone(), top_p)
    reset_launch_count(Operator.TOP_K_FILTER)
    actual_top_p = topk_topp_sampler.apply_top_k_top_p(
        source.clone(), top_ks, top_p
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_top_p, expected_top_p)
    assert launch_count(Operator.TOP_K_FILTER) == 0
    assert provider_metadata()["top_k_filter_override"] is True

    triton_source = torch.randn((8, vocab_size), device="cuda")
    triton_top_ks = torch.full(
        (8,), 50, device="cuda", dtype=torch.int32
    )
    expected_triton = original_apply(
        triton_source.clone(), triton_top_ks.clone(), None
    )
    reset_launch_count(Operator.TOP_K_FILTER)
    actual_triton = topk_topp_sampler.apply_top_k_top_p(
        triton_source.clone(), triton_top_ks, None
    )
    torch.cuda.synchronize()
    assert TOP_K_FILTER_MAX_ROWS == 7
    assert torch.equal(actual_triton, expected_triton)
    assert launch_count(Operator.TOP_K_FILTER) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_top_p_renorm_preserves_native_rng_and_fallbacks():
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

    import loom_kernels.vllm.sampling as sampling_integration
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert register_vllm_top_p_renorm() == TOP_P_RENORM_OVERRIDE_KEY
    original_forward = sampling_integration._TOP_P_RENORM_ORIGINAL_FORWARD
    assert original_forward is not None

    rows, vocab_size = 5, TOP_P_RENORM_MIN_VOCAB_SIZE
    torch.manual_seed(431)
    source = torch.randn((rows, vocab_size), device="cuda")
    source[0, :8] = 0.0
    top_ps = torch.tensor(
        [0.1, 0.5, 0.75, 0.9, 1.0],
        device="cuda",
        dtype=torch.float32,
    )

    def generators() -> dict[int, torch.Generator]:
        return {
            row: torch.Generator(device="cuda").manual_seed(3000 + row)
            for row in range(rows)
        }

    expected_sampler = TopKTopPSampler(logprobs_mode="processed_logits")
    expected, expected_processed = original_forward(
        expected_sampler,
        source.clone(),
        generators(),
        None,
        top_ps,
    )
    actual_sampler = TopKTopPSampler(logprobs_mode="processed_logits")
    reset_launch_count(Operator.TOP_P_RENORM)
    actual, actual_processed = actual_sampler.forward(
        source.clone(),
        generators(),
        None,
        top_ps,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert actual_processed is not None
    assert expected_processed is not None
    assert torch.equal(
        torch.isneginf(actual_processed),
        torch.isneginf(expected_processed),
    )
    assert launch_count(Operator.TOP_P_RENORM) == 1

    expected_logprob_sampler = TopKTopPSampler(
        logprobs_mode="processed_logprobs"
    )
    expected_logprob_ids, expected_logprobs = original_forward(
        expected_logprob_sampler,
        source.clone(),
        generators(),
        None,
        top_ps,
    )
    actual_logprob_sampler = TopKTopPSampler(
        logprobs_mode="processed_logprobs"
    )
    reset_launch_count(Operator.TOP_P_RENORM)
    actual_logprob_ids, actual_logprobs = actual_logprob_sampler.forward(
        source.clone(),
        generators(),
        None,
        top_ps,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_logprob_ids, expected_logprob_ids)
    assert actual_logprobs is not None
    assert expected_logprobs is not None
    torch.testing.assert_close(
        actual_logprobs,
        expected_logprobs,
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    assert launch_count(Operator.TOP_P_RENORM) == 1

    top_ks = torch.full(
        (rows,), 64, device="cuda", dtype=torch.int32
    )
    expected_joint, expected_joint_processed = original_forward(
        expected_sampler,
        source.clone(),
        generators(),
        top_ks,
        top_ps,
    )
    reset_launch_count(Operator.TOP_P_RENORM)
    actual_joint, actual_joint_processed = actual_sampler.forward_native(
        source.clone(),
        generators(),
        top_ks,
        top_ps,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_joint, expected_joint)
    assert torch.equal(actual_joint_processed, expected_joint_processed)
    assert launch_count(Operator.TOP_P_RENORM) == 0

    larger_source = torch.randn((8, vocab_size), device="cuda")
    larger_top_ps = torch.full((8,), 0.9, device="cuda")
    reset_launch_count(Operator.TOP_P_RENORM)
    larger_sampler = TopKTopPSampler()
    larger_sampler.forward_native(
        larger_source,
        {},
        None,
        larger_top_ps,
    )
    torch.cuda.synchronize()
    assert TOP_P_RENORM_MAX_ROWS == 7
    assert launch_count(Operator.TOP_P_RENORM) == 0

    for fallback_source in (
        torch.randn(
            (TOP_P_RENORM_MIN_ROWS - 1, vocab_size), device="cuda"
        ),
        torch.randn(
            (TOP_P_RENORM_MIN_ROWS, TOP_P_RENORM_MIN_VOCAB_SIZE - 1),
            device="cuda",
        ),
    ):
        fallback_top_ps = torch.full(
            (fallback_source.shape[0],), 0.9, device="cuda"
        )
        reset_launch_count(Operator.TOP_P_RENORM)
        TopKTopPSampler().forward_native(
            fallback_source,
            {},
            None,
            fallback_top_ps,
        )
        torch.cuda.synchronize()
        assert launch_count(Operator.TOP_P_RENORM) == 0
    assert provider_metadata()["top_p_renorm_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_token_penalties_use_sparse_workspace():
    from vllm.v1.sample.sampler import Sampler

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_token_penalties() == TOKEN_PENALTIES_OVERRIDE_KEY
    )
    rows, vocab_size = 4, 4096
    logits = torch.randn((rows, vocab_size), device="cuda")
    original = logits.clone()
    prompt = torch.randint(0, vocab_size, (rows, 65), device="cuda")
    prompt[0, :3] = torch.tensor([17, 17, vocab_size], device="cuda")
    output = [
        [17, 17, 31],
        [],
        [5, 9, 5, 9],
        [7],
    ]
    presence = torch.tensor([0.4, 0.0, -0.2, 0.1], device="cuda")
    frequency = torch.tensor([0.2, 0.0, 0.3, -0.1], device="cuda")
    repetition = torch.tensor([1.1, 0.9, 1.2, 1.0], device="cuda")
    metadata = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=prompt,
        presence_penalties=presence,
        frequency_penalties=frequency,
        repetition_penalties=repetition,
    )

    expected = original.clone()
    for row in range(rows):
        prompt_ids = {
            int(token)
            for token in prompt[row].tolist()
            if 0 <= int(token) < vocab_size
        }
        counts: dict[int, int] = {}
        for token in output[row]:
            counts[token] = counts.get(token, 0) + 1
        for token in prompt_ids | counts.keys():
            value = expected[row, token]
            expected[row, token] = (
                value / repetition[row]
                if value > 0
                else value * repetition[row]
            )
            if token in counts:
                expected[row, token] -= frequency[row] * counts[token]
                expected[row, token] -= presence[row]

    reset_launch_count(Operator.TOKEN_PENALTIES)
    returned = Sampler.apply_penalties(logits, metadata, output)
    torch.cuda.synchronize()

    assert returned is logits
    assert launch_count(Operator.TOKEN_PENALTIES) == 1
    torch.testing.assert_close(logits, expected, rtol=1.0e-6, atol=1.0e-6)
    contract = provider_metadata()["token_penalties_first_contract"]
    assert contract is not None
    assert contract["workspace_capacity"] == 256
    assert contract["workspace_bytes"] == rows * 256 * 8


def test_configures_vllm_rope_paged_kv_fusion():
    from vllm.compilation.passes.fusion import rope_kvcache_fusion
    from vllm.config import CompilationConfig
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    assert register_vllm_rope_paged_kv() == ROPE_PAGED_KV_OVERRIDE_KEY
    config = configure_vllm_rope_paged_kv(max_token_num=128)

    assert isinstance(config, CompilationConfig)
    assert config.pass_config.fuse_rope_kvcache is True
    assert config.pass_config.rope_kvcache_fusion_max_token_num == 128
    assert config.splitting_ops == []
    assert "+rotary_embedding" in config.custom_ops
    assert "+quant_fp8" in config.custom_ops
    assert FlashAttentionImpl.fused_rope_kvcache_supported.__module__ == (
        "loom_kernels.vllm.rope_kv"
    )
    assert FlashInferImpl.fused_rope_kvcache_supported.__module__ == (
        "loom_kernels.vllm.rope_kv"
    )
    for cache_dtype in ("auto", "fp8", "fp8_e4m3", torch.bfloat16):
        attention = SimpleNamespace(
            attn_type=AttentionType.DECODER,
            kv_cache_dtype=cache_dtype,
            kv_sharing_target_layer_name=None,
        )
        assert FlashAttentionImpl.fused_rope_kvcache_supported(attention)
        assert FlashInferImpl.fused_rope_kvcache_supported(attention)
    for cache_dtype in (
        "fp8_e5m2",
        "fp8_per_token_head",
        "int8",
        "nvfp4",
    ):
        attention = SimpleNamespace(
            attn_type=AttentionType.DECODER,
            kv_cache_dtype=cache_dtype,
            kv_sharing_target_layer_name=None,
        )
        assert not FlashAttentionImpl.fused_rope_kvcache_supported(attention)
        assert not FlashInferImpl.fused_rope_kvcache_supported(attention)
    shared_attention = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="fp8",
        kv_sharing_target_layer_name="model.layers.0.self_attn",
    )
    assert not FlashAttentionImpl.fused_rope_kvcache_supported(shared_attention)
    encoder_attention = SimpleNamespace(
        attn_type=AttentionType.ENCODER,
        kv_cache_dtype="fp8",
        kv_sharing_target_layer_name=None,
    )
    assert not FlashAttentionImpl.fused_rope_kvcache_supported(encoder_attention)
    assert getattr(
        rope_kvcache_fusion.RopeStaticQQuantKVCachePattern,
        "_loom_selects_query_scale_layout",
        False,
    )
    pattern_type = rope_kvcache_fusion.RopeStaticQQuantKVCachePattern
    pattern_layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        num_heads=4,
        num_kv_heads=2,
        head_size=8,
        head_size_v=8,
        _q_scale=torch.ones(1),
    )
    pattern = object.__new__(pattern_type)
    pattern._loom_layer = pattern_layer
    pattern.num_kv_heads = pattern_layer.num_kv_heads
    assert pattern._loom_query_scale_layout == "tensor"
    pattern_layer._q_scale = torch.ones(2)
    assert pattern._loom_query_scale_layout == "per_kv_head"
    pattern_layer._q_scale = torch.ones(3)
    with pytest.raises(RuntimeError, match="unsupported attention Q scale count"):
        pattern._loom_query_scale_layout
    assert provider_metadata()["rope_paged_kv_override"] is True
    assert provider_metadata()["rope_paged_kv_per_head_pattern"] is True


def test_vllm_paged_decode_shape_gate_is_conservative():
    qualified = {
        "dtype": torch.bfloat16,
        "batch": 32,
        "query_heads": 32,
        "kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_sequence_length": 32,
    }
    assert supports_vllm_paged_decode_shape(**qualified)
    for field, rejected in (
        ("dtype", torch.float32),
        ("batch", 129),
        ("query_heads", 64),
        ("kv_heads", 4),
        ("head_size", 64),
        ("block_size", 8),
        ("max_sequence_length", 64),
    ):
        candidate = {**qualified, field: rejected}
        assert not supports_vllm_paged_decode_shape(**candidate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_paged_decode_fast_path_matches_flash_attention():
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    batch = 8
    context = 32
    block_size = 16
    max_blocks = context // block_size
    num_blocks = batch * max_blocks
    query = torch.randn((batch, 32, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((batch, 8, 128), device="cuda", dtype=query.dtype)
    value = torch.empty_like(key)
    kv_cache = torch.randn(
        (num_blocks, 2, block_size, 8, 128),
        device="cuda",
        dtype=query.dtype,
    )
    block_table = torch.randperm(num_blocks, device="cuda", dtype=torch.int64)
    block_table = block_table.reshape(batch, max_blocks).to(torch.int32)
    seq_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        max_seq_len=context,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(batch, device="cuda", dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    attention = FlashAttentionImpl(
        num_heads=32,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    expected = torch.empty((batch, 32, 128), device="cuda", dtype=query.dtype)
    attention.forward(
        layer, query, key, value, kv_cache, metadata, expected
    )
    # Real FA3 decode metadata carries an AOT scheduler tensor and represents
    # the inactive DCP context length as zero. Neither changes attention
    # semantics, so Loom must not reject the otherwise qualified path.
    metadata.max_dcp_context_kv_len = 0
    metadata.scheduler_metadata = torch.zeros(
        (1,), device="cuda", dtype=torch.int32
    )

    assert (
        register_vllm_paged_decode_attention()
        == PAGED_DECODE_OVERRIDE_KEY
    )
    reset_launch_count(Operator.PAGED_DECODE_ATTENTION)
    actual = torch.empty_like(expected)
    returned = attention.forward(
        layer, query, key, value, kv_cache, metadata, actual
    )
    torch.cuda.synchronize()

    assert returned is actual
    assert launch_count(Operator.PAGED_DECODE_ATTENTION) == 1
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
    assert provider_metadata()["paged_decode_override"] is True


def test_paged_decode_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(PAGED_DECODE_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["paged_decode_override_requested"] is False
    monkeypatch.setenv(PAGED_DECODE_OVERRIDE_ENV, "true")
    assert provider_metadata()["paged_decode_override_requested"] is True


def test_registers_inplace_fused_add_rms_norm_provider():
    from vllm import ir

    assert register_vllm_ir() == DEFAULT_PROVIDER
    assert DEFAULT_PROVIDER in ir.ops.fused_add_rms_norm.impls
    implementation = ir.ops.fused_add_rms_norm.impls[DEFAULT_PROVIDER]
    assert implementation.inplace is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_min_p_processor_uses_loom_without_probability_tensor():
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert register_vllm_min_p() == MIN_P_OVERRIDE_KEY
    reset_launch_count(Operator.MIN_P_FILTER)
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = 31
    processor.min_p = torch.linspace(0.0, 0.8, 32, device="cuda").unsqueeze(1)
    logits = torch.randn((32, 151936), device="cuda", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    expected = logits.clone().masked_fill_(
        probabilities
        < probabilities.amax(dim=-1, keepdim=True) * processor.min_p,
        -float("inf"),
    )

    returned = processor.apply(logits)
    torch.cuda.synchronize()

    assert returned is logits
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))
    assert launch_count(Operator.MIN_P_FILTER) == 1
    assert provider_metadata()["min_p_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_min_p_processor_falls_back_below_measured_fast_path():
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert register_vllm_min_p() == MIN_P_OVERRIDE_KEY
    reset_launch_count(Operator.MIN_P_FILTER)
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = 2
    processor.min_p = torch.tensor([[0.0], [0.2], [0.8]], device="cuda")
    logits = torch.randn((3, 4096), device="cuda", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    expected = logits.clone().masked_fill_(
        probabilities
        < probabilities.amax(dim=-1, keepdim=True) * processor.min_p,
        -float("inf"),
    )

    returned = processor.apply(logits)
    torch.cuda.synchronize()

    assert returned is logits
    assert torch.equal(logits, expected)
    assert launch_count(Operator.MIN_P_FILTER) == 0


def test_min_p_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(MIN_P_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["min_p_override_requested"] is False
    monkeypatch.setenv(MIN_P_OVERRIDE_ENV, "yes")
    assert provider_metadata()["min_p_override_requested"] is True


def test_registers_vllm_silu_and_mul_override():
    from vllm.model_executor.custom_op import op_registry_oot

    assert register_vllm_silu_and_mul() == SILU_OVERRIDE_KEY
    assert SILU_OVERRIDE_KEY in op_registry_oot


def test_silu_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(SILU_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_override_requested"] is False
    monkeypatch.setenv(SILU_OVERRIDE_ENV, "true")
    assert provider_metadata()["silu_and_mul_override_requested"] is True


def test_registers_vllm_silu_and_mul_dynamic_fp8_fusion():
    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    assert (
        register_vllm_silu_and_mul_dynamic_fp8() == ACT_QUANT_OVERRIDE_KEY
    )
    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    assert FUSED_OPS[kFp8Dynamic64Sym] == implementation
    assert FUSED_OPS[kFp8Dynamic128Sym] == implementation


def test_act_quant_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(ACT_QUANT_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_fp8_override_requested"] is False
    monkeypatch.setenv(ACT_QUANT_OVERRIDE_ENV, "on")
    assert provider_metadata()["silu_and_mul_fp8_override_requested"] is True


def test_registers_vllm_silu_and_mul_dynamic_int8_pattern():
    from vllm.compilation.passes.fusion import act_quant_fusion

    assert (
        register_vllm_silu_and_mul_dynamic_int8()
        == ACT_INT8_OVERRIDE_KEY
    )
    assert getattr(
        act_quant_fusion.ActivationQuantFusionPass,
        "_loom_supports_dynamic_int8",
        False,
    )


def test_act_int8_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(ACT_INT8_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_int8_override_requested"] is False
    monkeypatch.setenv(ACT_INT8_OVERRIDE_ENV, "on")
    assert provider_metadata()["silu_and_mul_int8_override_requested"] is True


def test_registers_vllm_rms_norm_dynamic_fp8_fusions():
    from vllm.compilation.passes.fusion.rms_quant_fusion import (
        FUSED_OPS,
        FusedRMSQuantKey,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    assert (
        register_vllm_rms_norm_dynamic_fp8()
        == RMS_NORM_FP8_OVERRIDE_KEY
    )
    implementation = (
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_fp8.default
    )
    assert (
        FUSED_OPS[
            FusedRMSQuantKey(kFp8DynamicTokenSym, fused_add=False)
        ]
        == implementation
    )
    assert (
        FUSED_OPS[
            FusedRMSQuantKey(kFp8DynamicTokenSym, fused_add=True)
        ]
        == implementation
    )


def test_rms_norm_fp8_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(RMS_NORM_FP8_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["rms_norm_fp8_override_requested"] is False
    monkeypatch.setenv(RMS_NORM_FP8_OVERRIDE_ENV, "on")
    assert provider_metadata()["rms_norm_fp8_override_requested"] is True


def test_registers_vllm_rms_norm_dynamic_int8_patterns():
    from vllm.compilation.passes.fusion import rms_quant_fusion

    assert (
        register_vllm_rms_norm_dynamic_int8()
        == RMS_NORM_INT8_OVERRIDE_KEY
    )
    assert getattr(
        rms_quant_fusion.RMSNormQuantFusionPass,
        "_loom_supports_dynamic_int8",
        False,
    )


def test_rms_norm_int8_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(RMS_NORM_INT8_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["rms_norm_int8_override_requested"] is False
    monkeypatch.setenv(RMS_NORM_INT8_OVERRIDE_ENV, "on")
    assert provider_metadata()["rms_norm_int8_override_requested"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "pattern_class_name",
    [
        "RMSNormDynamicInt8QuantPattern",
        "FusedAddRMSNormDynamicInt8QuantPattern",
    ],
)
def test_vllm_rms_norm_dynamic_int8_pattern_rewrites_to_loom(
    pattern_class_name,
):
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
        enable_fake_mode,
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    from loom_kernels.vllm import _rms_int8_fusion

    config = VllmConfig()
    with set_current_vllm_config(config):
        register_vllm_rms_norm_dynamic_int8()
        pattern_class = getattr(_rms_int8_fusion, pattern_class_name)
        pattern = pattern_class(1.0e-5)
        fusion_pass = VllmFusionPatternMatcherPass(
            config, "loom_rms_int8_quant_test"
        )
        fusion_pass.register(pattern)

        @enable_fake_mode
        def trace_official_pattern():
            return fusion_pass._trace_fn(pattern.pattern, pattern.get_inputs())

        reference_graph_module = trace_official_pattern()
        graph_module = trace_official_pattern()
        fusion_pass(graph_module.graph)

    loom_operator = (
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8.default
    )
    loom_target_present = any(
        node.op == "call_function"
        and node.args
        and node.args[0] == loom_operator
        for node in graph_module.graph.nodes
    )
    assert fusion_pass.matched_count == 1
    assert loom_target_present

    inputs = [
        torch.randn_like(input_tensor) for input_tensor in pattern.get_inputs()
    ]
    expected = reference_graph_module(
        *[input_tensor.clone() for input_tensor in inputs]
    )
    actual = graph_module(*[input_tensor.clone() for input_tensor in inputs])
    torch.cuda.synchronize()

    integer_delta = (
        actual[0].to(torch.int16) - expected[0].to(torch.int16)
    ).abs()
    assert integer_delta.max().item() <= 1
    assert torch.count_nonzero(integer_delta).item() <= actual[0].shape[0]
    scale_index = 2 if len(actual) == 3 else 1
    torch.testing.assert_close(
        actual[scale_index],
        expected[scale_index],
        rtol=2.0e-6,
        atol=1.0e-8,
    )
    if len(actual) == 3:
        assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_activation_quant_pattern_rewrites_to_loom():
    from vllm.compilation.passes.fusion.act_quant_fusion import (
        SiluMulBlockQuantPattern,
    )
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
        enable_fake_mode,
    )
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
    )

    config = VllmConfig()
    with set_current_vllm_config(config):
        register_vllm_silu_and_mul_dynamic_fp8()
        pattern = SiluMulBlockQuantPattern(kFp8Dynamic128Sym)
        fusion_pass = VllmFusionPatternMatcherPass(
            config, "loom_activation_quant_test"
        )
        fusion_pass.register(pattern)

        @enable_fake_mode
        def trace_official_pattern():
            return fusion_pass._trace_fn(pattern.pattern, pattern.get_inputs())

        graph_module = trace_official_pattern()
        fusion_pass(graph_module.graph)

    loom_operator = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    loom_target_present = any(
        node.op == "call_function"
        and node.args
        and node.args[0] == loom_operator
        for node in graph_module.graph.nodes
    )
    assert fusion_pass.matched_count == 1
    assert loom_target_present


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_activation_int8_pattern_rewrites_to_loom():
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
        enable_fake_mode,
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    from loom_kernels.vllm._silu_int8_fusion import (
        SiluMulDynamicInt8QuantPattern,
    )

    config = VllmConfig()
    with set_current_vllm_config(config):
        register_vllm_silu_and_mul_dynamic_int8()
        pattern = SiluMulDynamicInt8QuantPattern()
        fusion_pass = VllmFusionPatternMatcherPass(
            config, "loom_activation_int8_quant_test"
        )
        fusion_pass.register(pattern)

        @enable_fake_mode
        def trace_official_pattern():
            return fusion_pass._trace_fn(pattern.pattern, pattern.get_inputs())

        reference_graph_module = trace_official_pattern()
        graph_module = trace_official_pattern()
        fusion_pass(graph_module.graph)
        graph_module.recompile()

    loom_operator = (
        torch.ops.loom_kernels.silu_and_mul_dynamic_per_token_int8.default
    )
    loom_target_present = any(
        node.op == "call_function"
        and node.args
        and node.args[0] == loom_operator
        for node in graph_module.graph.nodes
    )
    native_quant_target = torch.ops._C.dynamic_scaled_int8_quant.default
    native_quant_present = any(
        node.op == "call_function"
        and (
            node.target == native_quant_target
            or (node.args and node.args[0] == native_quant_target)
        )
        for node in graph_module.graph.nodes
    )
    assert fusion_pass.matched_count == 1
    assert loom_target_present
    assert not native_quant_present, graph_module.code

    torch.manual_seed(83)
    inputs = [
        torch.randn_like(value, dtype=torch.bfloat16)
        for value in pattern.get_inputs()
    ]
    expected_output, expected_scales = torch.compile(
        reference_graph_module, fullgraph=True
    )(*inputs)
    actual_output, actual_scales = graph_module(*inputs)
    torch.cuda.synchronize()

    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_scales, expected_scales), {
        "maximum_absolute_delta": float(
            (actual_scales - expected_scales).abs().max().item()
        ),
        "actual_bits": actual_scales.view(torch.int32).cpu().tolist(),
        "expected_bits": expected_scales.view(torch.int32).cpu().tolist(),
    }


@pytest.mark.parametrize(
    "pattern_name",
    [
        "RMSNormDynamicQuantPattern",
        "FusedAddRMSNormDynamicQuantPattern",
    ],
)
def test_vllm_rms_norm_dynamic_fp8_patterns_capture_loom(pattern_name):
    from vllm.compilation.passes.fusion import rms_quant_fusion
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        register_vllm_rms_norm_dynamic_fp8()
        pattern_class = getattr(rms_quant_fusion, pattern_name)
        pattern = pattern_class(1.0e-5, torch.float8_e4m3fn)

    loom_operator = (
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_fp8.default
    )
    assert pattern.FUSED_OP == loom_operator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_silu_layer_dispatches_to_loom():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul

    register_vllm_silu_and_mul()
    with set_current_vllm_config(VllmConfig()):
        activation = SiluAndMul()
    assert type(activation).__name__ == "LoomSiluAndMul"

    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(expected, input_tensor)
    actual = activation(input_tensor)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_ir_dispatches_to_loom_provider():
    from vllm import ir
    from vllm.platforms import current_platform

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    input_tensor = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    expected_residual = (input_tensor.float() + residual.float()).to(torch.bfloat16)

    reset_launch_count(Operator.ADD_RMS_NORM)
    with operation.set_priority([DEFAULT_PROVIDER, "native"]):
        assert (
            operation.dispatch(input_tensor, residual, weight, 1.0e-5).provider
            == DEFAULT_PROVIDER
        )
        output, residual_output = operation.maybe_inplace(
            input_tensor, residual, weight, 1.0e-5
        )
    torch.cuda.synchronize()

    assert output is input_tensor
    assert residual_output is residual
    assert launch_count(Operator.ADD_RMS_NORM) == 1
    torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 4096), (8, 4096), (128, 4096), (8, 8192)])
def test_loom_is_bitwise_equal_to_vllm_cuda_provider(shape):
    from vllm import ir
    from vllm.platforms import current_platform

    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    if "vllm_c" not in operation.impls or not operation.impls["vllm_c"].supported:
        pytest.skip("vLLM CUDA provider is unavailable")

    torch.manual_seed(20260721)
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    outputs = {}
    for provider in (DEFAULT_PROVIDER, "vllm_c"):
        provider_input = input_tensor.clone()
        provider_residual = residual.clone()
        with operation.set_priority([provider, "native"]):
            outputs[provider] = operation.maybe_inplace(
                provider_input, provider_residual, weight, 1.0e-5
            )
    torch.cuda.synchronize()

    loom_output, loom_residual = outputs[DEFAULT_PROVIDER]
    vllm_output, vllm_residual = outputs["vllm_c"]
    assert torch.equal(loom_output, vllm_output)
    assert torch.equal(loom_residual, vllm_residual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_categorical_state_survives_request_lifecycle():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_categorical_sample()
        == CATEGORICAL_SAMPLE_OVERRIDE_KEY
    )

    def request(req_id: str, seed: int | None) -> CachedRequestState:
        sampling_params = SamplingParams(temperature=1.0, seed=seed)
        generator = (
            None
            if seed is None
            else torch.Generator(device="cuda").manual_seed(seed)
        )
        return CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=sampling_params,
            generator=generator,
            block_ids=([0],),
            num_computed_tokens=0,
            output_token_ids=[],
        )

    def make_batch(*, num_spec_tokens: int = 0) -> InputBatch:
        return InputBatch(
            max_num_reqs=4,
            max_model_len=32,
            max_num_batched_tokens=32,
            device=torch.device("cuda"),
            vocab_size=257,
            block_sizes=[16],
            kernel_block_sizes=[16],
            max_num_blocks_per_req=[2],
            num_spec_tokens=num_spec_tokens,
        )

    with pytest.raises(RuntimeError, match="non-speculative"):
        make_batch(num_spec_tokens=1)

    batch = make_batch()
    with pytest.raises(ValueError, match="explicit seed"):
        batch.add_request(request("unseeded", None))

    requests = [request(f"req-{index}", 10 + index) for index in range(3)]
    for item in requests:
        batch.add_request(item)
    batch.refresh_metadata()
    sampler = Sampler()

    def draw() -> list[int]:
        logits = torch.zeros(
            (batch.num_reqs, batch.vocab_size),
            dtype=torch.float32,
            device="cuda",
        )
        reset_launch_count(Operator.CATEGORICAL_SAMPLE)
        token_ids, processed = sampler.sample(logits, batch.sampling_metadata)
        torch.cuda.synchronize()
        assert processed is None
        assert launch_count(Operator.CATEGORICAL_SAMPLE) == 1
        return token_ids.tolist()

    assert len(draw()) == 3
    assert batch.sampling_metadata._loom_categorical_rng_state.tolist() == [
        [10, 1],
        [11, 1],
        [12, 1],
    ]

    batch.remove_request("req-1")
    batch.condense()
    batch.refresh_metadata()
    assert batch.req_id_to_index == {"req-0": 0, "req-2": 1}
    assert len(draw()) == 2
    assert requests[1]._loom_categorical_rng_state.tolist() == [11, 1]

    batch.add_request(requests[1])
    batch.refresh_metadata()
    assert len(draw()) == 3
    assert batch.sampling_metadata._loom_categorical_rng_state.tolist() == [
        [10, 3],
        [12, 3],
        [11, 2],
    ]

    batch.swap_states(0, 2)
    batch.refresh_metadata()
    assert batch.req_id_to_index == {
        "req-0": 2,
        "req-1": 0,
        "req-2": 1,
    }
    assert len(draw()) == 3

    for req_id in list(batch.req_id_to_index):
        batch.remove_request(req_id)
    batch.condense()
    batch.refresh_metadata()
    torch.cuda.synchronize()
    assert [
        item._loom_categorical_rng_state.tolist() for item in requests
    ] == [
        [10, 4],
        [11, 3],
        [12, 4],
    ]

    metadata = provider_metadata()
    assert metadata["categorical_sample_override"] is True
    assert metadata["categorical_sample_first_contract"] == {
        "shape": [3, 257],
        "dtype": "torch.float32",
        "seeded_rows": 3,
        "persistent_state": True,
        "use_fp64_gumbel": False,
    }
