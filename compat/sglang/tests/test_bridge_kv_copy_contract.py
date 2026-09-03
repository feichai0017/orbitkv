from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.facade as facade  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402


def _set_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(
        facade,
        "_config",
        lambda: SimpleNamespace(
            token_reclamation=SimpleNamespace(mode=mode),
        ),
    )


def _mha_pool(*, native: bool):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = object.__new__(MHATokenToKVPool)
    pool.use_native_move_kv_cache = native
    return pool


@pytest.mark.parametrize("mode", ("off", "naive"))
def test_non_relocation_modes_do_not_inspect_the_pool(monkeypatch, mode):
    class UninspectablePool:
        def __getattribute__(self, _name):
            raise AssertionError("non-relocation mode inspected the KV pool")

    _set_mode(monkeypatch, mode)
    facade._ensure_relocation_kv_copy_contract(UninspectablePool())


def test_mha_native_copy_enablement_is_instance_local_and_idempotent(monkeypatch):
    _set_mode(monkeypatch, "relocate")
    pool = _mha_pool(native=False)
    pool._init_kv_copy_and_warmup = lambda: (_ for _ in ()).throw(
        AssertionError("private warmup must not run")
    )
    native_env_before = os.environ.get("SGLANG_NATIVE_MOVE_KV_CACHE")

    facade._ensure_relocation_kv_copy_contract(pool)
    facade._ensure_relocation_kv_copy_contract(pool)

    assert pool.use_native_move_kv_cache is True
    assert os.environ.get("SGLANG_NATIVE_MOVE_KV_CACHE") == native_env_before


def test_swa_wrapper_enables_both_mha_subpools(monkeypatch):
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    _set_mode(monkeypatch, "relocate")
    wrapper = object.__new__(SWAKVPool)
    wrapper.full_kv_pool = _mha_pool(native=False)
    wrapper.swa_kv_pool = _mha_pool(native=False)

    facade._ensure_relocation_kv_copy_contract(wrapper)

    assert wrapper.full_kv_pool.use_native_move_kv_cache is True
    assert wrapper.swa_kv_pool.use_native_move_kv_cache is True


def test_hybrid_linear_wrapper_enables_its_full_mha_pool(monkeypatch):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    _set_mode(monkeypatch, "relocate")
    wrapper = object.__new__(HybridLinearKVPool)
    wrapper.full_kv_pool = _mha_pool(native=False)

    facade._ensure_relocation_kv_copy_contract(wrapper)

    assert wrapper.full_kv_pool.use_native_move_kv_cache is True


def test_mla_requires_callable_move_without_adding_an_mha_flag(monkeypatch):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    _set_mode(monkeypatch, "relocate")
    pool = object.__new__(MLATokenToKVPool)

    facade._ensure_relocation_kv_copy_contract(pool)

    assert callable(pool.move_kv_cache)
    assert not hasattr(pool, "use_native_move_kv_cache")

    pool.move_kv_cache = None
    with pytest.raises(RuntimeError, match="MLA KV pool relocation contract"):
        facade._ensure_relocation_kv_copy_contract(pool)


def test_unknown_pool_is_rejected_even_when_it_exposes_move(monkeypatch):
    _set_mode(monkeypatch, "relocate")
    pool = SimpleNamespace(
        use_native_move_kv_cache=False,
        move_kv_cache=lambda *_args: None,
    )

    with pytest.raises(RuntimeError, match="does not support SGLang"):
        facade._ensure_relocation_kv_copy_contract(pool)

    assert pool.use_native_move_kv_cache is False


def test_mha_method_presence_without_native_flag_is_rejected(monkeypatch):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    _set_mode(monkeypatch, "relocate")
    pool = object.__new__(MHATokenToKVPool)
    assert callable(pool.move_kv_cache)

    with pytest.raises(RuntimeError, match="MHA KV pool relocation contract"):
        facade._ensure_relocation_kv_copy_contract(pool)


def test_builder_rejects_copy_contract_before_runtime_or_allocator_publication(
    monkeypatch,
):
    config = SimpleNamespace(
        page_tokens=16,
        token_reclamation=SimpleNamespace(mode="relocate"),
        classes=(
            SimpleNamespace(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                retention="full",
                storage="token_kv",
            ),
        ),
    )
    created = []
    runtime_publications = []

    class LocalAllocator:
        def __init__(self, *_args, **_kwargs):
            created.append(self)

    monkeypatch.setattr(facade, "_config", lambda: config)
    monkeypatch.setattr(
        facade, "_facade_types", lambda: (LocalAllocator,) * 3
    )
    monkeypatch.setattr(
        facade, "_new_runtime", lambda _arenas: runtime_publications.append(True)
    )
    monkeypatch.setattr(state, "_admit_product_runtime_profile", lambda: object())
    monkeypatch.setattr(state, "_ALLOCATOR", None)
    monkeypatch.setattr(state, "_RUNTIME", None)

    with pytest.raises(RuntimeError, match="does not support SGLang"):
        facade._build_token_to_kv_pool_allocator(
            SimpleNamespace(
                page_size=16,
                device="cuda:0",
                kv_cache_dtype=object(),
                is_hybrid_swa=False,
                use_mla_backend=False,
                is_draft_worker=False,
            ),
            sizes=SimpleNamespace(max_total_num_tokens=128),
            token_to_kv_pool=SimpleNamespace(move_kv_cache=lambda *_args: None),
            is_dsv4_model=False,
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=None,
        )

    assert len(created) == 1
    assert runtime_publications == []
    assert state._RUNTIME is None
    assert state._ALLOCATOR is None
