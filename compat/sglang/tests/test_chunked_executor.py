from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import orbitkv_sglang.bridge.facade as facade
import orbitkv_sglang.bridge.lowering as lowering
import orbitkv_sglang.bridge.state as state
import orbitkv_sglang.bridge.validation as validation
from orbitkv_sglang.config import (
    ClassConfig,
    RuntimeConfig,
    TokenReclamationConfig,
)
from orbitkv_sglang.runtime import CacheSharingPolicy


PAGE_TOKENS = 16
CHUNK_TOKENS = 32


def _class(*, layers=(0,)) -> ClassConfig:
    return ClassConfig(
        class_id=0,
        pool_id=1,
        backend_domain=7,
        name="chunked_attention",
        layers=tuple(layers),
        retention="chunked",
        bytes_per_token_per_layer=128,
        window_tokens=None,
        period_blocks=None,
        chunk_tokens=CHUNK_TOKENS,
        blocks_per_epoch=CHUNK_TOKENS // PAGE_TOKENS,
    )


def _config(
    *, layers=(0,), reclamation=TokenReclamationConfig()
) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:chunked-bridge-test",
        page_tokens=PAGE_TOKENS,
        classes=(_class(layers=layers),),
        token_reclamation=reclamation,
        manager_plan_format="retention_ir",
    )


def _multi_layer_config() -> RuntimeConfig:
    return _config(layers=(0, 1))


def _nonchunked_config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:nonchunked-bridge-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=7,
                name="full_attention",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
        ),
    )


def _runtime_backend_proof() -> state.RuntimeAttentionBackendProof:
    return state.RuntimeAttentionBackendProof(
        backend_class="FlashAttentionBackend",
        backend_module="sglang.srt.layers.attention.flashattention_backend",
        prefill_backend="fa3",
        decode_backend="fa3",
        has_local_attention=True,
        attention_chunk_size=CHUNK_TOKENS,
        page_size=PAGE_TOKENS,
        compiled_layer_ids=(0,),
        use_irope_layer_ids=(0,),
    )


class _InitializationRuntime:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _assert_initialization_unpublished() -> None:
    assert state._LIMITS is None
    assert state._RUNTIME is None
    assert state._RUNTIME_BACKEND_PROOF is None
    assert state._ALLOCATOR is None
    assert state._MIRROR_CLEANUP is None
    assert state._FIXED_STATE is None
    assert state._FIXED_STATE_ALLOCATOR_RESTORE is None
    assert state._INITIALIZATION_TOKEN is None


def _install(*, maximum_requests=4, reclamation=TokenReclamationConfig()):
    state._install_test_state(
        config=_config(reclamation=reclamation),
        limits=state.RuntimeLimits(maximum_requests, 64, 256),
    )


def _builder(*, pages=8, hybrid=False):
    return facade._build_token_to_kv_pool_allocator(
        SimpleNamespace(
            page_size=PAGE_TOKENS,
            device="cuda:0",
            kv_cache_dtype=torch.bfloat16,
            is_hybrid_swa=hybrid,
            is_draft_worker=False,
            use_mla_backend=False,
        ),
        sizes=SimpleNamespace(max_total_num_tokens=pages * PAGE_TOKENS),
        token_to_kv_pool=SimpleNamespace(),
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )


def test_chunked_builder_uses_generic_paged_facade_and_one_arena(monkeypatch):
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    _install()
    registrations = []

    class Runtime:
        page_tokens = PAGE_TOKENS
        failure_reason = None

        def bind_prefix_eviction_cleanup(self, coordinator):
            self.coordinator = coordinator

        def capacity_arena_stats(self):
            return (SimpleNamespace(class_id=0, free_pages=8),)

        def fail_stop(self, reason):
            self.failure_reason = reason

    runtime = Runtime()

    def install(values):
        registrations.extend(values)
        state._RUNTIME = runtime
        return runtime

    monkeypatch.setattr(
        state,
        "_admit_product_runtime_profile",
        lambda: SimpleNamespace(cache_policy="request_private"),
    )
    monkeypatch.setattr(facade, "_new_runtime", install)
    allocator = _builder()

    assert isinstance(allocator, PagedTokenToKVPoolAllocator)
    assert allocator.size == 8 * PAGE_TOKENS
    assert len(registrations) == 1
    assert (
        registrations[0].class_id,
        registrations[0].pool_id,
        registrations[0].backend_domain,
        registrations[0].page_count,
    ) == (0, 1, 7, 8)
    assert allocator.available_size() == 8 * PAGE_TOKENS


@pytest.mark.parametrize(
    ("pages", "hybrid", "message"),
    ((7, False, "epoch floor"), (8, True, "hybrid storage")),
)
def test_chunked_builder_rejects_capacity_or_storage_drift(
    monkeypatch, pages, hybrid, message
):
    _install()
    monkeypatch.setattr(
        state,
        "_admit_product_runtime_profile",
        lambda: SimpleNamespace(cache_policy="request_private"),
    )
    monkeypatch.setattr(
        facade,
        "_new_runtime",
        lambda _registrations: pytest.fail("runtime construction preceded rejection"),
    )

    with pytest.raises(RuntimeError, match=message):
        _builder(pages=pages, hybrid=hybrid)


def _model(
    *, chunk_tokens=CHUNK_TOKENS, hybrid=False, is_local_attention_model=True
):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["RenamedLocalModel"]),
        hf_text_config=SimpleNamespace(num_hidden_layers=1, num_key_value_heads=2),
        head_dim=16,
        v_head_dim=16,
        swa_head_dim=8,
        swa_v_head_dim=8,
        is_hybrid_swa=hybrid,
        full_attention_layer_ids=[],
        swa_attention_layer_ids=[],
        sliding_window_size=None,
        disable_hybrid_swa_memory=False,
        is_deepseek_v4_arch=False,
        is_hybrid_swa_compress=False,
        attention_chunk_size=chunk_tokens,
        is_local_attention_model=is_local_attention_model,
        has_attention_sinks=False,
    )


def _geometry_configurator(model=None, *, backend="fa3", layers=((0, True),)):
    from sglang.srt.layers.radix_attention import RadixAttention

    attention_layers = []
    for layer_id, use_irope in layers:
        layer = object.__new__(RadixAttention)
        object.__setattr__(layer, "layer_id", layer_id)
        object.__setattr__(layer, "use_irope", use_irope)
        attention_layers.append(layer)
    loaded_model = SimpleNamespace(
        named_modules=lambda: enumerate(attention_layers)
    )
    return SimpleNamespace(
        model_config=_model() if model is None else model,
        model=loaded_model,
        kv_cache_dtype=torch.bfloat16,
        use_mla_backend=False,
        server_args=SimpleNamespace(
            get_attention_backends=lambda: (backend, backend)
        ),
    )


def test_chunked_geometry_accepts_structural_full_layer_local_contract():
    state._install_test_state(config=_config())

    validation._validate_checkpoint_geometry(_geometry_configurator())
    validation._validate_attention_backend_contract(_geometry_configurator())


@pytest.mark.parametrize("chunk_tokens", (None, True, CHUNK_TOKENS + PAGE_TOKENS))
def test_chunked_geometry_rejects_mismatched_chunk_or_backend(chunk_tokens):
    state._install_test_state(config=_config())

    with pytest.raises(RuntimeError, match="chunk size differs"):
        validation._validate_checkpoint_geometry(
            _geometry_configurator(_model(chunk_tokens=chunk_tokens))
        )
    with pytest.raises(RuntimeError, match="requires SGLang FA3"):
        validation._validate_attention_backend_contract(
            _geometry_configurator(backend="flashinfer")
        )


@pytest.mark.parametrize(
    ("model", "layers", "message"),
    (
        (
            _model(is_local_attention_model=False),
            ((0, True),),
            "is_local_attention_model=True",
        ),
        (
            _model(is_local_attention_model=None),
            ((0, True),),
            "is_local_attention_model=True",
        ),
        (_model(), (), "post-model-load/pre-KV-pool hook"),
        (_model(), ((1, True),), "do not exactly cover"),
        (_model(), ((0, True), (0, True)), "do not exactly cover"),
        (_model(), ((True, True),), "integer layer_id"),
        (_model(), ((0, False),), "use_irope=True"),
        (_model(), ((0, None),), "boolean use_irope"),
        (_model(), ((0, 1),), "boolean use_irope"),
    ),
)
def test_chunked_geometry_fails_closed_without_exact_runtime_layer_proof(
    model, layers, message
):
    state._install_test_state(config=_config())

    with pytest.raises(RuntimeError, match=message):
        validation._validate_checkpoint_geometry(
            _geometry_configurator(model, layers=layers)
        )


def test_chunked_runtime_layer_proof_is_order_independent():
    state._install_test_state(config=_multi_layer_config())
    model = _model()
    model.hf_text_config.num_hidden_layers = 2

    validation._validate_checkpoint_geometry(
        _geometry_configurator(model, layers=((1, True), (0, True)))
    )


def test_chunked_geometry_fails_closed_when_loaded_model_is_unavailable():
    state._install_test_state(config=_config())
    configurator = _geometry_configurator()
    del configurator.model

    with pytest.raises(RuntimeError, match="post-model-load/pre-KV-pool hook"):
        validation._validate_checkpoint_geometry(configurator)


def test_chunked_geometry_fails_closed_when_local_attention_marker_is_missing():
    state._install_test_state(config=_config())
    model = _model()
    del model.is_local_attention_model

    with pytest.raises(RuntimeError, match="is_local_attention_model=True"):
        validation._validate_checkpoint_geometry(_geometry_configurator(model))


def _scheduler_configurator(**overrides):
    values = {
        "chunked_prefill_size": CHUNK_TOKENS,
        "prefill_max_requests": 1,
        "enable_dynamic_chunking": False,
        "enable_mixed_chunk": False,
        "max_prefill_tokens": CHUNK_TOKENS,
        "cuda_graph_config": SimpleNamespace(),
    }
    values.update(overrides)
    return SimpleNamespace(server_args=SimpleNamespace(**values))


@pytest.mark.parametrize(
    "max_prefill_tokens", (CHUNK_TOKENS, CHUNK_TOKENS * 4)
)
def test_chunked_scheduler_accepts_one_exact_contiguous_prefill_chunk(
    max_prefill_tokens,
):
    state._install_test_state(config=_config())

    validation._validate_chunked_scheduler_contract(
        _scheduler_configurator(max_prefill_tokens=max_prefill_tokens)
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"chunked_prefill_size": None}, "chunked_prefill_size=32"),
        ({"chunked_prefill_size": True}, "chunked_prefill_size=32"),
        ({"chunked_prefill_size": CHUNK_TOKENS * 2}, "chunked_prefill_size=32"),
        ({"prefill_max_requests": None}, "prefill_max_requests=1"),
        ({"prefill_max_requests": True}, "prefill_max_requests=1"),
        ({"prefill_max_requests": 2}, "prefill_max_requests=1"),
        ({"enable_dynamic_chunking": None}, "enable_dynamic_chunking=false"),
        ({"enable_dynamic_chunking": True}, "enable_dynamic_chunking=false"),
        ({"enable_mixed_chunk": None}, "enable_mixed_chunk=false"),
        ({"enable_mixed_chunk": True}, "enable_mixed_chunk=false"),
        ({"max_prefill_tokens": None}, "max_prefill_tokens>=32"),
        ({"max_prefill_tokens": True}, "max_prefill_tokens>=32"),
        ({"max_prefill_tokens": CHUNK_TOKENS - 1}, "max_prefill_tokens>=32"),
    ),
)
def test_chunked_scheduler_rejects_unsafe_prefill_shapes(overrides, message):
    state._install_test_state(config=_config())

    with pytest.raises(RuntimeError, match=message):
        validation._validate_chunked_scheduler_contract(
            _scheduler_configurator(**overrides)
        )


@pytest.mark.parametrize(
    "field",
    (
        "chunked_prefill_size",
        "prefill_max_requests",
        "enable_dynamic_chunking",
        "enable_mixed_chunk",
        "max_prefill_tokens",
    ),
)
def test_chunked_scheduler_rejects_missing_fields(field):
    state._install_test_state(config=_config())
    configurator = _scheduler_configurator()
    delattr(configurator.server_args, field)

    with pytest.raises(RuntimeError, match=field):
        validation._validate_chunked_scheduler_contract(configurator)


def test_configurator_rejects_unsafe_chunk_scheduler_before_pool_construction(
    monkeypatch,
):
    state._install_test_state(config=_config())
    for name in (
        "_validate_gdn_fixed_state_backend_contract",
        "_validate_radix_cache_contract",
        "_validate_attention_backend_contract",
        "_validate_token_reclamation_backend_contract",
    ):
        monkeypatch.setattr(validation, name, lambda _configurator: None)
    calls = []

    with pytest.raises(RuntimeError, match="prefill_max_requests=1"):
        validation._validate_configurator(
            lambda *_args, **_kwargs: calls.append(True),
            _scheduler_configurator(prefill_max_requests=2),
        )

    assert calls == []


def test_chunked_configurator_holds_initialization_transaction_for_backend(
    monkeypatch,
):
    state._install_test_state(config=_config())
    runtime = _InitializationRuntime()
    result = object()

    def configure_once(*_args, **_kwargs):
        state._RUNTIME = runtime
        state._ALLOCATOR = object()
        return result

    monkeypatch.setattr(validation, "_validate_configurator_once", configure_once)

    assert validation._validate_configurator(lambda: None, object()) is result
    token = state._pending_initialization_token()
    assert state._RUNTIME is runtime
    assert state._RUNTIME_BACKEND_PROOF is None
    with pytest.raises(RuntimeError, match="already in progress"):
        state._begin_initialization()

    state._rollback_initialization(RuntimeError("test cleanup"), token=token)
    state._finish_initialization(token)
    assert runtime.close_count == 1
    _assert_initialization_unpublished()


def test_model_runner_backend_hook_commits_runtime_proof(monkeypatch):
    state._install_test_state(config=_config())
    state._begin_initialization()
    runtime = _InitializationRuntime()
    allocator = object()
    state._RUNTIME = runtime
    state._ALLOCATOR = allocator
    proof = _runtime_backend_proof()
    calls = []
    result = object()
    runner = object()

    def original(received_runner, *args, **kwargs):
        calls.append((received_runner, args, kwargs))
        return result

    monkeypatch.setattr(
        validation, "_validate_loaded_runtime_backend", lambda value: proof
    )

    assert (
        validation._validate_runtime_attention_backend(
            original, runner, "arg", keyword="value"
        )
        is result
    )
    assert calls == [(runner, ("arg",), {"keyword": "value"})]
    assert state._RUNTIME is runtime
    assert state._ALLOCATOR is allocator
    assert state._RUNTIME_BACKEND_PROOF is proof
    assert state._runtime_backend_proof() == {
        "backend_class": "FlashAttentionBackend",
        "backend_module": "sglang.srt.layers.attention.flashattention_backend",
        "prefill_backend": "fa3",
        "decode_backend": "fa3",
        "has_local_attention": True,
        "attention_chunk_size": CHUNK_TOKENS,
        "page_size": PAGE_TOKENS,
        "compiled_layer_ids": [0],
        "use_irope_layer_ids": [0],
    }
    assert state._INITIALIZATION_TOKEN is None
    assert runtime.close_count == 0

    state._rollback_initialization(RuntimeError("test cleanup"))
    assert runtime.close_count == 1
    _assert_initialization_unpublished()


@pytest.mark.parametrize("failure_stage", ("original", "proof"))
def test_model_runner_backend_hook_rolls_back_failure_and_allows_retry(
    monkeypatch, failure_stage
):
    state._install_test_state(config=_config())
    token = state._begin_initialization()
    runtime = _InitializationRuntime()
    state._LIMITS = state.RuntimeLimits(1, CHUNK_TOKENS, 256)
    state._RUNTIME = runtime
    state._ALLOCATOR = object()
    error = RuntimeError(f"injected {failure_stage} failure")
    proof_calls = []

    def original(*_args, **_kwargs):
        if failure_stage == "original":
            raise error
        return object()

    def validate_backend(_runner):
        proof_calls.append(True)
        if failure_stage == "proof":
            raise error
        return _runtime_backend_proof()

    monkeypatch.setattr(
        validation, "_validate_loaded_runtime_backend", validate_backend
    )

    with pytest.raises(RuntimeError) as captured:
        validation._validate_runtime_attention_backend(original, object())

    assert captured.value is error
    assert proof_calls == ([] if failure_stage == "original" else [True])
    assert runtime.close_count == 1
    _assert_initialization_unpublished()

    retry_token = state._begin_initialization()
    assert retry_token is not token
    state._finish_initialization(retry_token)


def test_nonchunked_model_runner_backend_hook_is_exact_passthrough(monkeypatch):
    state._install_test_state(config=_nonchunked_config())
    calls = []
    result = object()
    runner = object()

    def original(received_runner, *args, **kwargs):
        calls.append((received_runner, args, kwargs))
        return result

    monkeypatch.setattr(
        state,
        "_pending_initialization_token",
        lambda: pytest.fail("nonchunked backend hook requested a transaction"),
    )
    monkeypatch.setattr(
        validation,
        "_validate_loaded_runtime_backend",
        lambda _runner: pytest.fail("nonchunked backend hook validated a proof"),
    )

    assert (
        validation._validate_runtime_attention_backend(
            original, runner, "arg", keyword="value"
        )
        is result
    )
    assert calls == [(runner, ("arg",), {"keyword": "value"})]
    assert state._RUNTIME_BACKEND_PROOF is None
    assert state._runtime_backend_proof() is None


def test_chunked_forces_private_radix_and_disables_token_reclamation(
    monkeypatch,
):
    state._install_test_state(config=_config())
    monkeypatch.setattr(
        state,
        "_cache_sharing_policy",
        lambda: CacheSharingPolicy.REQUEST_PRIVATE,
    )
    assert state._requires_disabled_radix_cache()

    state._install_test_state(
        config=_config(reclamation=TokenReclamationConfig(mode="naive"))
    )
    with pytest.raises(RuntimeError, match="reclamation.*off"):
        validation._validate_token_reclamation_backend_contract(
            _geometry_configurator()
        )


def _extend_batch(previous: int, target: int):
    row = torch.full((2, 128), 777, dtype=torch.int32)
    req = SimpleNamespace(prefix_indices=torch.arange(previous, dtype=torch.int64))
    return SimpleNamespace(
        reqs=[req],
        prefix_lens=[previous],
        extend_lens=[target - previous],
        extend_num_tokens=target - previous,
        seq_lens_cpu=[target],
        seq_lens=torch.tensor([target], dtype=torch.int64),
        device=torch.device("cpu"),
        req_to_token_pool=SimpleNamespace(
            req_to_token=row, max_context_len=int(row.shape[1])
        ),
    )


def test_chunked_extend_cross_boundary_fails_before_any_mutation(monkeypatch):
    state._install_test_state(config=_config())
    batch = _extend_batch(CHUNK_TOKENS - 1, CHUNK_TOKENS + 1)
    before = batch.req_to_token_pool.req_to_token.clone()
    with pytest.raises(RuntimeError, match="crosses.*chunk boundary"):
        validation._preflight_extend_batch(batch)

    assert torch.equal(batch.req_to_token_pool.req_to_token, before)
    assert not hasattr(batch.reqs[0], "req_pool_idx")


def test_primary_locations_and_new_epoch_write_stay_absolute():
    state._install_test_state(config=_config())
    locations = torch.tensor([81], dtype=torch.int64)
    assert lowering._primary_locations({0: locations}) is locations

    row = torch.zeros((1, 96), dtype=torch.int32)
    row[0, CHUNK_TOKENS] = locations.to(torch.int32)[0]
    assert int(row[0, CHUNK_TOKENS]) == 81
    assert not torch.count_nonzero(row[0, :CHUNK_TOKENS])
