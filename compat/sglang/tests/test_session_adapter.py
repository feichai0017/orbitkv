from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.facade as facade  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
import orbitkv_sglang.bridge.validation as validation  # noqa: E402
from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402


def _class(class_id: int, retention: str) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 1,
        name="swa" if retention == "sliding" else "full",
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=32 if retention == "sliding" else None,
        period_blocks=3 if retention == "sliding" else None,
    )


def _config(retentions: tuple[str, ...]) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:test-session-adapter",
        page_tokens=16,
        classes=tuple(_class(index, value) for index, value in enumerate(retentions)),
    )


class _FakeSessionRuntime:
    def __init__(self, config: RuntimeConfig, registrations: tuple[object, ...]) -> None:
        self.page_tokens = config.page_tokens
        self.registrations = registrations
        self.failure_reason = None
        self.free_pages = {
            registration.class_id: registration.page_count for registration in registrations
        }
        self.mirror_cleanup = None

    def capacity_arena_stats(self):
        return tuple(
            SimpleNamespace(class_id=registration.class_id, free_pages=self.free_pages[registration.class_id])
            for registration in self.registrations
        )

    def bind_mirror_cleanup(self, callback):
        self.mirror_cleanup = callback

    def fail_stop(self, reason):
        self.failure_reason = str(reason)

    def close(self) -> None:
        return None


class _FakeKvPool:
    def __init__(self) -> None:
        self.mapping = None

    def register_mapping(self, mapping) -> None:
        self.mapping = mapping


def _install_allocator_runtime(
    monkeypatch: pytest.MonkeyPatch,
    config: RuntimeConfig,
    cache_policy: str,
    *,
    limits: object | None = None,
) -> list[_FakeSessionRuntime]:
    if limits is None:
        limits = state.RuntimeLimits(
            maximum_running_requests=4,
            chunked_prefill_tokens=64,
            maximum_context_tokens=256,
        )
    profile = SimpleNamespace(cache_policy=cache_policy)
    state._install_test_state(config=config, limits=limits, runtime=None)
    monkeypatch.setattr(state, "_admit_product_runtime_profile", lambda: profile)
    monkeypatch.setattr(state, "_uses_runtime_session", lambda: True)
    runtimes: list[_FakeSessionRuntime] = []

    def new_runtime(registrations):
        runtime = _FakeSessionRuntime(config, tuple(registrations))
        state._RUNTIME = runtime
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(facade, "_new_runtime", new_runtime)
    return runtimes


def _configurator(*, hybrid: bool):
    return SimpleNamespace(
        page_size=16,
        device="cuda:0",
        kv_cache_dtype=torch.bfloat16,
        is_hybrid_swa=hybrid,
        is_draft_worker=False,
        use_mla_backend=False,
    )


@pytest.fixture
def cpu_mapping_tensors(monkeypatch: pytest.MonkeyPatch):
    original_zeros = torch.zeros
    original_arange = torch.arange
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: original_zeros(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *args, **kwargs: original_arange(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )


def test_full_builder_installs_paged_facade_and_one_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    config = _config(("full",))
    runtimes = _install_allocator_runtime(monkeypatch, config, "shared_prefix")

    allocator = facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=False),
        sizes=SimpleNamespace(max_total_num_tokens=128),
        token_to_kv_pool=_FakeKvPool(),
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )

    (runtime,) = runtimes
    assert isinstance(allocator, PagedTokenToKVPoolAllocator)
    assert tuple(item.page_count for item in runtime.registrations) == (8,)
    assert allocator.available_size() == 128
    runtime.free_pages[0] = 2
    assert allocator.available_size() == 32
    with pytest.raises(RuntimeError, match="native KV authority"):
        allocator.alloc(16)
    with pytest.raises(RuntimeError, match="native KV authority"):
        allocator.free(object())


def test_hybrid_builder_installs_swa_facade_and_tracks_two_arenas(
    monkeypatch: pytest.MonkeyPatch,
    cpu_mapping_tensors,
) -> None:
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    config = _config(("full", "sliding"))
    pool = _FakeKvPool()
    runtimes = _install_allocator_runtime(monkeypatch, config, "shared_prefix")

    allocator = facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=True),
        sizes=SimpleNamespace(
            max_total_num_tokens=64,
            full_max_total_num_tokens=128,
            swa_max_total_num_tokens=320,
        ),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )

    (runtime,) = runtimes
    assert isinstance(allocator, SWATokenToKVPoolAllocator)
    assert tuple(item.page_count for item in runtime.registrations) == (8, 20)
    assert pool.mapping is allocator.full_to_swa_index_mapping
    assert int(pool.mapping[-1]) == -1
    runtime.free_pages.update({0: 3, 1: 1})
    assert allocator.full_available_size() == 48
    assert allocator.swa_available_size() == 16
    assert allocator.available_size() == 16
    assert allocator.new_pages_available(3, 1) is True
    assert allocator.new_pages_available(4, 1) is False


def test_pure_sliding_builder_uses_identity_lut(
    monkeypatch: pytest.MonkeyPatch,
    cpu_mapping_tensors,
) -> None:
    from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator

    config = _config(("sliding",))
    pool = _FakeKvPool()
    runtimes = _install_allocator_runtime(monkeypatch, config, "request_private")

    allocator = facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=True),
        sizes=SimpleNamespace(
            max_total_num_tokens=64,
            full_max_total_num_tokens=0,
            swa_max_total_num_tokens=320,
        ),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )

    (runtime,) = runtimes
    assert isinstance(allocator, PureSWATokenToKVPoolAllocator)
    assert tuple(item.page_count for item in runtime.registrations) == (20,)
    assert torch.equal(pool.mapping[:320], torch.arange(320))
    assert int(pool.mapping[-1]) == -1
    assert allocator.swa_available_size() == 320


def test_hybrid_builder_accepts_exact_sliding_floor_and_rejects_below_it(
    monkeypatch: pytest.MonkeyPatch,
    cpu_mapping_tensors,
) -> None:
    base = _config(("full", "sliding"))
    config = replace(
        base,
        classes=(
            base.classes[0],
            replace(base.classes[1], window_tokens=128, period_blocks=19),
        ),
    )
    limits = state.RuntimeLimits(
        maximum_running_requests=1,
        chunked_prefill_tokens=256,
        maximum_context_tokens=1024,
    )
    runtimes = _install_allocator_runtime(
        monkeypatch, config, "shared_prefix", limits=limits
    )

    facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=True),
        sizes=SimpleNamespace(
            max_total_num_tokens=512,
            full_max_total_num_tokens=512,
            swa_max_total_num_tokens=560,
        ),
        token_to_kv_pool=_FakeKvPool(),
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )
    assert tuple(item.page_count for item in runtimes[0].registrations) == (32, 35)

    state._install_test_state(config=config, limits=limits)
    with pytest.raises(RuntimeError, match=r"capacity=544 minimum=560"):
        facade._build_token_to_kv_pool_allocator(
            _configurator(hybrid=True),
            sizes=SimpleNamespace(
                max_total_num_tokens=512,
                full_max_total_num_tokens=512,
                swa_max_total_num_tokens=544,
            ),
            token_to_kv_pool=_FakeKvPool(),
            is_dsv4_model=False,
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=None,
        )


@pytest.mark.parametrize(
    ("rid", "expected"),
    (
        ("request-1", ("str", "request-1")),
        (b"request-1", ("bytes", b"request-1")),
        (7, ("int", 7)),
    ),
)
def test_request_key_uses_stable_typed_rid(rid, expected) -> None:
    assert state._request_key(SimpleNamespace(rid=rid)) == expected


@pytest.mark.parametrize("rid", (None, "", b"", -1, True, object()))
def test_request_key_rejects_unstable_or_empty_identity(rid) -> None:
    with pytest.raises(RuntimeError, match="rid"):
        state._request_key(SimpleNamespace(rid=rid))


def test_geometry_gates_are_semantic_not_architecture_allowlists() -> None:
    for retentions, architecture, layers in (
        (("full",), "Qwen2ForCausalLM", ((0,), ())),
        (("full", "sliding"), "GptOssForCausalLM", ((0,), (1,))),
        (("full", "sliding"), "Olmo3ForCausalLM", ((0,), (1,))),
    ):
        config = _config(retentions)
        state._install_test_state(config=config)
        model = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            hf_text_config=SimpleNamespace(num_hidden_layers=len(retentions), num_key_value_heads=2),
            head_dim=16,
            v_head_dim=16,
            swa_head_dim=16,
            swa_v_head_dim=16,
            is_hybrid_swa=len(retentions) == 2,
            full_attention_layer_ids=list(layers[0]),
            swa_attention_layer_ids=list(layers[1]),
            sliding_window_size=32,
            disable_hybrid_swa_memory=False,
            is_deepseek_v4_arch=False,
            is_hybrid_swa_compress=False,
            attention_chunk_size=None,
            has_attention_sinks=architecture == "GptOssForCausalLM",
        )
        validation._validate_checkpoint_geometry(
            SimpleNamespace(
                model_config=model,
                kv_cache_dtype=torch.bfloat16,
                use_mla_backend=False,
            )
        )


@pytest.mark.parametrize("backend", ("flashinfer", "fa3"))
def test_pinned_server_args_reports_explicit_uniform_backend_pair(backend: str) -> None:
    from sglang.srt.server_args import ServerArgs

    server = SimpleNamespace(
        attention_backend=backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )
    assert ServerArgs.get_attention_backends(server) == (backend, backend)


def test_attention_backend_contract_accepts_token_kv_capability_backend() -> None:
    state._install_test_state(config=_config(("full",)))
    configurator = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["RenamedArchitecture"]),
            has_attention_sinks=False,
        ),
        use_mla_backend=False,
        server_args=SimpleNamespace(
            get_attention_backends=lambda: ("fa3", "fa3")
        ),
    )
    assert validation._validate_attention_backend_contract(configurator) is None


@pytest.mark.parametrize(
    "backends", (("fa3", "flashinfer"), ("triton", "triton"))
)
def test_attention_backend_contract_rejects_nonuniform_or_unsupported_token_backend(
    backends: tuple[str, str],
) -> None:
    state._install_test_state(config=_config(("full",)))
    configurator = SimpleNamespace(
        model_config=SimpleNamespace(has_attention_sinks=False),
        use_mla_backend=False,
        server_args=SimpleNamespace(get_attention_backends=lambda: backends),
    )
    with pytest.raises(RuntimeError, match="token-KV capability requires"):
        validation._validate_attention_backend_contract(configurator)


def test_attention_sink_capability_requires_fa3_without_architecture_identity() -> None:
    state._install_test_state(config=_config(("full",)))
    configurator = SimpleNamespace(
        model_config=SimpleNamespace(has_attention_sinks=True),
        use_mla_backend=False,
        server_args=SimpleNamespace(
            get_attention_backends=lambda: ("fa3", "fa3")
        ),
    )
    validation._validate_attention_backend_contract(configurator)
    configurator.server_args.get_attention_backends = lambda: (
        "flashinfer",
        "flashinfer",
    )
    with pytest.raises(
        RuntimeError, match="attention sinks capability requires SGLang FA3"
    ):
        validation._validate_attention_backend_contract(configurator)
