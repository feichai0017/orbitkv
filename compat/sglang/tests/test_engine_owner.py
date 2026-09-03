from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.engine as engine  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
import orbitkv_sglang.runtime_admission as runtime_admission  # noqa: E402
import orbitkv_sglang.session_runtime as session_runtime  # noqa: E402
from orbitkv_sglang.config import (  # noqa: E402
    ClassConfig,
    RuntimeConfig,
    TokenReclamationConfig,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaRegistration,
    CacheSharingPolicy,
    ManagerCreateSettings,
    SessionCreateSettings,
)


def _class(class_id, retention, *, storage="token_kv"):
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 1,
        name=retention,
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=128 if retention == "sliding" else None,
        period_blocks=9 if retention == "sliding" else None,
        storage=storage,
        chunk_tokens=128 if retention == "chunked" else None,
        blocks_per_epoch=8 if retention == "chunked" else None,
    )


def _config(*classes, token_reclamation=TokenReclamationConfig()):
    selected = classes or (_class(0, "full"),)
    shape = tuple((item.retention, item.storage) for item in selected)
    topologies = {
        (("full", "token_kv"),): "whole_domain_full_token_kv",
        (
            ("full", "token_kv"),
            ("sliding", "token_kv"),
        ): "whole_domain_full_sliding_token_kv",
        (("sliding", "token_kv"),): "whole_domain_sliding_token_kv",
        (("chunked", "token_kv"),): "whole_domain_chunked_token_kv",
        (("full", "latent_kv"),): "whole_domain_full_latent_kv",
    }
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:engine-owner-test",
        page_tokens=16,
        classes=selected,
        token_reclamation=token_reclamation,
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding={"execution_topology": topologies[shape]},
        manager_plan_format=(
            "retention_ir" if shape == (("chunked", "token_kv"),) else "kv_plan"
        ),
    )


@pytest.fixture(autouse=True)
def _reset_owner_state(monkeypatch):
    previous_owner = engine._OWNER
    previous_state = (
        state._CONFIG,
        state._PRODUCT_PROFILE,
        state._LIMITS,
        state._RUNTIME,
    )
    engine._OWNER = None
    state._CONFIG = None
    state._PRODUCT_PROFILE = None
    state._LIMITS = None
    state._RUNTIME = None
    monkeypatch.setattr(engine, "_validate_source_checkout", lambda: None)

    def admit_synthetic_config(config):
        binding = config.runtime_binding
        if config.runtime_manifest_path is None or not isinstance(binding, dict):
            raise ValueError(
                "runtime target admission requires ORBITKV_RUNTIME_MANIFEST"
            )
        return binding["execution_topology"]

    monkeypatch.setattr(
        runtime_admission, "admit_runtime_config", admit_synthetic_config
    )
    try:
        yield
    finally:
        engine._OWNER = previous_owner
        (
            state._CONFIG,
            state._PRODUCT_PROFILE,
            state._LIMITS,
            state._RUNTIME,
        ) = previous_state


def _create(monkeypatch, config=None):
    loaded = _config() if config is None else config
    registrations = []
    monkeypatch.setattr(engine, "load_config", lambda environ=None: loaded)
    monkeypatch.setattr(engine, "_register_radix_factory", registrations.append)
    owner = engine.OrbitKvLifecycleOwner.create({"test": "environment"})
    return owner, loaded, registrations


def test_create_publishes_one_canonical_owner_and_factory(monkeypatch):
    owner, config, factories = _create(monkeypatch)

    assert engine.get_owner() is owner
    assert owner.config is config
    assert state._CONFIG is config
    assert owner.product_profile is state._PRODUCT_PROFILE
    assert owner.product_profile.execution_topology == "whole_domain_full_token_kv"
    assert owner.product_profile.cache_policy == "shared_prefix"
    assert factories == [owner.radix_factory]
    with pytest.raises(RuntimeError, match="more than once"):
        engine.OrbitKvLifecycleOwner.create()
    with pytest.raises(TypeError, match="must be created"):
        engine.OrbitKvLifecycleOwner(object(), config, owner.product_profile)


def test_create_rejects_foreign_config_and_rolls_back_registry_failure(monkeypatch):
    state._CONFIG = _config()
    monkeypatch.setattr(
        engine,
        "load_config",
        lambda _environ=None: pytest.fail("foreign ownership must fail first"),
    )
    with pytest.raises(RuntimeError, match="another integration"):
        engine.get_owner()
    assert engine._OWNER is None

    state._CONFIG = None
    config = _config()
    monkeypatch.setattr(engine, "load_config", lambda _environ=None: config)
    monkeypatch.setattr(
        engine,
        "_register_radix_factory",
        lambda _factory: (_ for _ in ()).throw(RuntimeError("collision")),
    )
    with pytest.raises(RuntimeError, match="collision"):
        engine.OrbitKvLifecycleOwner.create()
    assert engine._OWNER is None
    assert state._CONFIG is None
    assert state._PRODUCT_PROFILE is None


def test_create_validates_patched_source_before_loading_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        engine,
        "_validate_source_checkout",
        lambda: (_ for _ in ()).throw(RuntimeError("source mismatch")),
    )
    monkeypatch.setattr(
        engine,
        "load_config",
        lambda _environ=None: calls.append("load"),
    )

    with pytest.raises(RuntimeError, match="source mismatch"):
        engine.OrbitKvLifecycleOwner.create()

    assert calls == []
    assert engine._OWNER is None
    assert state._CONFIG is None


def test_create_rejects_missing_manifest_before_registration(monkeypatch):
    registered = []
    config = _config()
    config = type(config)(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field != "runtime_manifest_path"
        },
        runtime_manifest_path=None,
    )
    monkeypatch.setattr(engine, "load_config", lambda _environ=None: config)
    monkeypatch.setattr(engine, "_register_radix_factory", registered.append)

    with pytest.raises(RuntimeError, match="requires ORBITKV_RUNTIME_MANIFEST"):
        engine.OrbitKvLifecycleOwner.create({})

    assert registered == []
    assert engine._OWNER is None
    assert state._CONFIG is None


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            _config(
                token_reclamation=TokenReclamationConfig(
                    mode="naive", trigger_tokens=16, retained_per_page=1
                )
            ),
            "does not match its profile",
        ),
    ),
)
def test_create_rejects_out_of_scope_plans(monkeypatch, config, message):
    registered = []
    monkeypatch.setattr(engine, "load_config", lambda _environ=None: config)
    monkeypatch.setattr(engine, "_register_radix_factory", registered.append)
    with pytest.raises(RuntimeError, match=message):
        engine.OrbitKvLifecycleOwner.create()
    assert registered == []
    assert engine._OWNER is None
    assert state._CONFIG is None
    assert state._PRODUCT_PROFILE is None


def test_ordered_full_swa_scope_is_admitted(monkeypatch):
    config = _config(_class(0, "full"), _class(1, "sliding"))
    owner, loaded, factories = _create(monkeypatch, config)
    assert owner.config is loaded is config
    assert factories == [owner.radix_factory]


def test_pure_sliding_scope_is_admitted(monkeypatch):
    config = _config(_class(0, "sliding"))
    owner, loaded, factories = _create(monkeypatch, config)
    assert owner.config is loaded is config
    assert factories == [owner.radix_factory]


def test_chunked_retention_ir_scope_is_admitted(monkeypatch):
    config = _config(_class(0, "chunked"))
    owner, loaded, factories = _create(monkeypatch, config)

    assert owner.config is loaded is config
    assert state._uses_runtime_session()
    assert state._requires_disabled_radix_cache()
    assert state._cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE
    assert factories == [owner.radix_factory]


def test_full_latent_scope_is_admitted_as_request_private_native_session(
    monkeypatch,
):
    config = _config(_class(0, "full", storage="latent_kv"))
    owner, loaded, factories = _create(monkeypatch, config)

    assert owner.config is loaded is config
    assert owner.product_profile is state._PRODUCT_PROFILE
    assert owner.product_profile.execution_topology == "whole_domain_full_latent_kv"
    assert owner.product_profile.ordered_attention_classes == ("full:latent_kv",)
    assert owner.product_profile.cache_policy == "request_private"
    assert state._uses_runtime_session()
    assert state._requires_disabled_radix_cache()
    assert state._cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE
    assert factories == [owner.radix_factory]


@pytest.mark.parametrize(
    ("config", "expected_policy"),
    (
        (_config(), CacheSharingPolicy.SHARED_PREFIX),
        (
            _config(_class(0, "full", storage="latent_kv")),
            CacheSharingPolicy.REQUEST_PRIVATE,
        ),
    ),
)
def test_owner_profile_drives_explicit_native_session_create_settings(
    monkeypatch, config, expected_policy
):
    created = []

    class SessionClient:
        @staticmethod
        def create(loaded, settings, registrations):
            raw = SimpleNamespace(
                cache_sharing_policy=settings.cache_sharing_policy,
                close=lambda: None,
            )
            created.append((loaded, settings, tuple(registrations), raw))
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            self.mirror_cleanup = mirror_cleanup
            self.cache_sharing_policy = session.cache_sharing_policy

        def close(self):
            self.session.close()

    owner, loaded, _factories = _create(monkeypatch, config)
    state._LIMITS = state.RuntimeLimits(2, 32, 128)
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)
    registration = ArenaRegistration(0, 1, 1, 4, 8)

    runtime = state._new_runtime((registration,))

    assert runtime is state._RUNTIME
    assert owner.product_profile is state._PRODUCT_PROFILE
    assert created[0][0] is loaded is config
    settings = created[0][1]
    assert type(settings) is SessionCreateSettings
    assert type(settings.manager) is ManagerCreateSettings
    assert settings.manager == ManagerCreateSettings(2, 2, 4, 4, 32)
    assert settings.cache_sharing_policy is expected_policy
    assert created[0][2] == (registration,)


def test_direct_methods_dispatch_and_preserve_identity(monkeypatch):
    owner, _loaded, _factories = _create(monkeypatch)
    calls = []
    values = [object() for _ in range(12)]
    native_configure, native_next, native_run, native_state = values[:4]
    configurator, scheduler, batch, req, tree_cache = values[4:9]
    sizes, token_pool, req_pool = values[9:]

    def record(name, result=None):
        def implementation(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return implementation

    configured, extend, decode, selected, forwarded, reported = (
        object() for _ in range(6)
    )
    allocator = SimpleNamespace()
    monkeypatch.setattr(engine, "_validate_configurator", record("configure", configured))
    monkeypatch.setattr(
        engine, "_build_token_to_kv_pool_allocator", record("allocator", allocator)
    )
    monkeypatch.setattr(engine, "_alloc_for_extend", record("extend", extend))
    monkeypatch.setattr(engine, "_alloc_for_decode", record("decode", decode))
    monkeypatch.setattr(engine, "_manager_maybe_evict_swa", record("evict"))
    monkeypatch.setattr(engine, "_get_next_batch_to_run", record("next", selected))
    monkeypatch.setattr(engine, "_run_scheduled_batch", record("run", forwarded))
    monkeypatch.setattr(engine, "_get_internal_state", record("state", reported))
    monkeypatch.setattr(
        engine,
        "_prepare_waiting_request_removal",
        record("waiting-removal", True),
    )
    monkeypatch.setattr(engine, "_release_kv_cache", record("release"))

    assert owner.configure(native_configure, configurator, flag=True) is configured
    assert owner.build_allocator(
        configurator,
        sizes=sizes,
        token_to_kv_pool=token_pool,
        is_dsv4_model=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=None,
    ) is allocator
    assert allocator._orbitkv_lifecycle_owner is owner
    assert owner.prepare_extend(batch) is extend
    assert owner.prepare_decode(batch, 1) is decode
    assert owner.maybe_evict_swa(batch) is None
    assert owner.next_batch(native_next, scheduler, flag=False) is selected
    assert owner.run_batch(native_run, scheduler, batch, mode="eager") is forwarded
    assert owner.internal_state(native_state, scheduler) is reported
    assert owner.prepare_waiting_request_removal(req, tree_cache) is True
    assert owner.release_request(req, tree_cache, False) is None

    assert [item[0] for item in calls] == [
        "configure", "allocator", "extend", "decode", "evict",
        "next", "run", "state", "waiting-removal", "release",
    ]
    assert calls[0][1] == (native_configure, configurator)
    assert calls[5][1] == (native_next, scheduler)
    assert calls[6][1] == (native_run, scheduler, batch)
    assert calls[8][1] == (req, tree_cache)
    assert calls[9][1] == (req, tree_cache, False)


def test_radix_factory_binds_owner_and_rejects_foreign_cache(monkeypatch):
    owner, _loaded, _factories = _create(monkeypatch)
    context, cache = object(), SimpleNamespace()
    monkeypatch.setattr(engine, "_build_prefix_cache", lambda value: cache)
    assert owner.radix_factory(context) is cache
    assert cache._orbitkv_lifecycle_owner is owner
    cache._orbitkv_lifecycle_owner = object()
    with pytest.raises(RuntimeError, match="foreign radix cache"):
        owner.radix_factory(context)


def test_stale_owner_cannot_dispatch(monkeypatch):
    owner, config, _factories = _create(monkeypatch)
    monkeypatch.setattr(
        engine,
        "_alloc_for_extend",
        lambda _batch: pytest.fail("stale owner reached implementation"),
    )
    state._CONFIG = _config()
    with pytest.raises(RuntimeError, match="foreign or stale owner"):
        owner.prepare_extend(object())
    assert owner.config is config


def test_owner_module_has_no_hook_registry_dependency():
    source = inspect.getsource(engine).lower()
    assert "hook_registry" not in source
    assert "apply_hooks" not in source
    assert "plugin.hooks" not in source
    assert "def alloc_for_" not in source
    assert "around_run_batch" not in source


def test_reporting_import_has_no_legacy_hook_module_or_registry_dependency():
    assert not (SOURCE_ROOT / "orbitkv_sglang/plugin").exists()
    script = """
import importlib.util
import sys

import orbitkv_sglang.engine
import orbitkv_sglang.observability
assert 'orbitkv_sglang.bridge.hooks' not in sys.modules
assert 'sglang.srt.plugins.hook_registry' not in sys.modules
assert importlib.util.find_spec('orbitkv_sglang.bridge.hooks') is None
assert importlib.util.find_spec('orbitkv_sglang.plugin') is None
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
