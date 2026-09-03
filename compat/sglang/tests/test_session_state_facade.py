from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import inspect
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from orbitkv_sglang import runtime_admission, session_runtime
from orbitkv_sglang.config import (
    ClassConfig,
    FixedStateConfig,
    RuntimeConfig,
    TokenReclamationConfig,
)
from orbitkv_sglang.bridge import facade, state
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    ManagerCreateSettings,
    SessionCreateSettings,
)


def _full_config(**changes: Any) -> RuntimeConfig:
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:session-state-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
                storage="token_kv",
            ),
        ),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )
    return replace(config, **changes)


def _latent_config(**changes: Any) -> RuntimeConfig:
    full = _full_config().classes[0]
    config = replace(
        _full_config(),
        classes=(
            replace(
                full,
                name="latent_mla",
                storage="latent_kv",
                components=(("latent", 96), ("rope", 32)),
            ),
        ),
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_latent_kv"}
        ),
    )
    return replace(config, **changes)


def _hybrid_config(**changes: Any) -> RuntimeConfig:
    full = _full_config().classes[0]
    config = replace(
        _full_config(),
        classes=(
            full,
            replace(
                full,
                class_id=1,
                pool_id=2,
                backend_domain=2,
                name="sliding",
                layers=(1,),
                retention="sliding",
                window_tokens=32,
                period_blocks=2,
            ),
        ),
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_sliding_token_kv"}
        ),
    )
    return replace(config, **changes)


def _pure_sliding_config(**changes: Any) -> RuntimeConfig:
    full = _full_config().classes[0]
    config = replace(
        _full_config(),
        classes=(
            replace(
                full,
                name="sliding",
                retention="sliding",
                window_tokens=32,
                period_blocks=3,
            ),
        ),
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_sliding_token_kv"}
        ),
    )
    return replace(config, **changes)


def _chunked_config(**changes: Any) -> RuntimeConfig:
    full = _full_config().classes[0]
    config = replace(
        _full_config(),
        classes=(
            replace(
                full,
                name="chunked",
                retention="chunked",
                chunk_tokens=32,
                blocks_per_epoch=2,
            ),
        ),
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_chunked_token_kv"}
        ),
        manager_plan_format="retention_ir",
    )
    return replace(config, **changes)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    config: RuntimeConfig,
    *,
    runtime: Any = None,
) -> None:
    monkeypatch.setattr(state, "_CONFIG", config)
    monkeypatch.setattr(state, "_PRODUCT_PROFILE", None)
    monkeypatch.setattr(state, "_LIMITS", state.RuntimeLimits(8, 64, 128))
    monkeypatch.setattr(state, "_RUNTIME", runtime)
    monkeypatch.setattr(state, "_RUNTIME_BACKEND_PROOF", None)
    monkeypatch.setattr(state, "_ALLOCATOR", None)
    monkeypatch.setattr(state, "_MIRROR_CLEANUP", None)
    monkeypatch.setattr(state, "_FIXED_STATE", None)
    monkeypatch.setattr(state, "_FIXED_STATE_ALLOCATOR_RESTORE", None)

    def admit_synthetic_config(candidate: RuntimeConfig) -> str:
        binding = candidate.runtime_binding
        if candidate.runtime_manifest_path is None or not isinstance(
            binding, Mapping
        ):
            raise ValueError(
                "runtime target admission requires ORBITKV_RUNTIME_MANIFEST"
            )
        topology = binding.get("execution_topology")
        if not isinstance(topology, str):
            raise ValueError("runtime binding has no execution topology")
        return topology

    monkeypatch.setattr(
        runtime_admission, "admit_runtime_config", admit_synthetic_config
    )


@pytest.mark.parametrize(
    "change",
    (
        {"runtime_manifest_path": None},
        {"runtime_binding": None},
        pytest.param(
            {
                "runtime_binding": MappingProxyType(
                    {
                        "execution_topology": (
                            "whole_domain_full_sliding_token_kv"
                        )
                    }
                )
            },
            id="hybrid-topology-with-full-only-shape",
        ),
        {"manager_plan_format": "retention_ir"},
        {
            "classes": (
                replace(_full_config().classes[0], retention="sliding"),
            )
        },
        {
            "classes": (
                replace(_full_config().classes[0], storage="latent_kv"),
            )
        },
        pytest.param(
            {
                "classes": (
                    _full_config().classes[0],
                    replace(
                        _full_config().classes[0],
                        class_id=1,
                        pool_id=2,
                        backend_domain=2,
                        name="second",
                    ),
                )
            },
            id="two-full-classes-with-full-only-topology",
        ),
        {"token_reclamation": TokenReclamationConfig(mode="naive")},
        {
            "fixed_states": (
                FixedStateConfig("state", "mamba", (1,), 64, 2),
            )
        },
    ),
)
def test_runtime_session_selector_is_exact_and_manifest_backed(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    _install(monkeypatch, _full_config(**change))

    assert not state._uses_runtime_session()


def test_full_token_kv_runtime_session_allows_shared_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _full_config())

    assert state._uses_runtime_session()
    assert state._cache_sharing_policy() is CacheSharingPolicy.SHARED_PREFIX
    assert not state._requires_disabled_radix_cache()


def test_full_sliding_token_kv_runtime_session_allows_shared_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _hybrid_config())

    assert state._uses_runtime_session()
    assert state._cache_sharing_policy() is CacheSharingPolicy.SHARED_PREFIX
    assert not state._requires_disabled_radix_cache()


def test_pure_sliding_token_kv_runtime_session_is_request_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _pure_sliding_config())

    assert state._uses_runtime_session()
    assert state._cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE
    assert state._requires_disabled_radix_cache()


def test_full_latent_kv_runtime_session_is_request_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _latent_config())

    assert state._uses_runtime_session()
    assert state._cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE
    assert state._requires_disabled_radix_cache()


def test_chunked_token_kv_runtime_session_is_request_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _chunked_config())

    assert state._uses_runtime_session()
    assert state._cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE
    assert state._requires_disabled_radix_cache()


@pytest.mark.parametrize(
    "changes",
    (
        pytest.param({"manager_plan_format": "kv_plan"}, id="wrong-plan-format"),
        pytest.param(
            {
                "runtime_binding": MappingProxyType(
                    {"execution_topology": "whole_domain_full_token_kv"}
                )
            },
            id="wrong-topology",
        ),
        pytest.param(
            {"token_reclamation": TokenReclamationConfig(mode="naive")},
            id="token-reclamation",
        ),
        pytest.param(
            {
                "fixed_states": (
                    FixedStateConfig("state", "mamba", (1,), 64, 2),
                )
            },
            id="fixed-state",
        ),
    ),
)
def test_chunked_runtime_session_selector_rejects_inexact_profiles(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    _install(monkeypatch, _chunked_config(**changes))

    assert not state._uses_runtime_session()


@pytest.mark.parametrize(
    "changes",
    (
        pytest.param({"runtime_manifest_path": None}, id="no-manifest"),
        pytest.param(
            {
                "runtime_binding": MappingProxyType(
                    {"execution_topology": "whole_domain_full_token_kv"}
                )
            },
            id="token-kv-topology",
        ),
        pytest.param(
            {
                "classes": (
                    replace(_latent_config().classes[0], storage="token_kv"),
                )
            },
            id="token-kv-storage",
        ),
        pytest.param(
            {"manager_plan_format": "retention_ir"}, id="wrong-plan-format"
        ),
        pytest.param(
            {"token_reclamation": TokenReclamationConfig(mode="naive")},
            id="token-reclamation",
        ),
        pytest.param(
            {
                "fixed_states": (
                    FixedStateConfig("state", "mamba", (1,), 64, 2),
                )
            },
            id="fixed-state",
        ),
    ),
)
def test_full_latent_runtime_session_selector_rejects_inexact_profiles(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    _install(monkeypatch, _latent_config(**changes))

    assert not state._uses_runtime_session()


@pytest.mark.parametrize(
    "changes",
    (
        pytest.param({"runtime_manifest_path": None}, id="no-manifest"),
        pytest.param(
            {
                "runtime_binding": MappingProxyType(
                    {"execution_topology": "whole_domain_full_token_kv"}
                )
            },
            id="full-only-topology",
        ),
        pytest.param(
            {"manager_plan_format": "retention_ir"}, id="retention-ir-plan"
        ),
        pytest.param(
            {"classes": tuple(reversed(_hybrid_config().classes))},
            id="sliding-before-full",
        ),
        pytest.param(
            {
                "classes": (
                    _hybrid_config().classes[0],
                    replace(_hybrid_config().classes[1], storage="latent_kv"),
                )
            },
            id="non-token-storage",
        ),
        pytest.param(
            {"token_reclamation": TokenReclamationConfig(mode="naive")},
            id="token-reclamation",
        ),
        pytest.param(
            {"token_reclamation": TokenReclamationConfig(mode="relocate")},
            id="token-relocation",
        ),
        pytest.param(
            {
                "fixed_states": (
                    FixedStateConfig("state", "mamba", (2,), 64, 2),
                )
            },
            id="fixed-state",
        ),
    ),
)
def test_full_sliding_runtime_session_selector_rejects_inexact_profiles(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    _install(monkeypatch, _hybrid_config(**changes))

    assert not state._uses_runtime_session()
@pytest.mark.parametrize(
    "changes",
    (
        pytest.param(
            {
                "fixed_states": (
                    FixedStateConfig("state", "mamba", (1,), 64, 2),
                )
            },
            id="fixed-state",
        ),
        pytest.param(
            {"token_reclamation": TokenReclamationConfig(mode="naive")},
            id="token-reclamation",
        ),
    ),
)
def test_non_product_profiles_have_no_cache_sharing_policy(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    _install(monkeypatch, _full_config(**changes))

    assert not state._uses_runtime_session()
    with pytest.raises(
        RuntimeError, match="cache policy is unavailable or unsupported"
    ):
        state._requires_disabled_radix_cache()


class _RawSession:
    def __init__(
        self,
        cache_sharing_policy: CacheSharingPolicy = CacheSharingPolicy.SHARED_PREFIX,
    ) -> None:
        self.close_count = 0
        self.cache_sharing_policy = cache_sharing_policy

    def close(self) -> None:
        self.close_count += 1


def test_full_sliding_runtime_creates_session_runtime_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()
    creates = []

    class SessionClient:
        @staticmethod
        def create(config, settings, registrations):
            creates.append((config, settings, tuple(registrations)))
            raw.cache_sharing_policy = settings.cache_sharing_policy
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            self.mirror_cleanup = mirror_cleanup
            self.cache_sharing_policy = session.cache_sharing_policy

        def close(self):
            self.session.close()

    _install(monkeypatch, _hybrid_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)
    registrations = (
        ArenaRegistration(0, 1, 1, 8, 3),
        ArenaRegistration(1, 2, 2, 4, 131),
    )

    runtime = state._new_runtime(registrations)

    assert runtime is state._RUNTIME
    assert runtime.session is raw
    assert runtime.mirror_cleanup is session_runtime.unbound_mirror_cleanup
    assert creates[0][0] is state._CONFIG
    settings = creates[0][1]
    assert settings.manager.maximum_requests == 8
    assert settings.manager.maximum_operations == 8
    assert settings.manager.maximum_prefixes == 12
    assert settings.manager.maximum_reclamations == 12
    assert settings.manager.maximum_step_tokens == 64
    assert settings.cache_sharing_policy is CacheSharingPolicy.SHARED_PREFIX
    assert creates[0][2] == registrations
    assert raw.close_count == 0

    state._close_owned_runtime(runtime)

    assert raw.close_count == 1
    assert state._RUNTIME is None


def test_full_sliding_runtime_session_creation_failure_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSessionClient:
        @staticmethod
        def create(*_args: Any) -> Any:
            raise RuntimeError("injected session creation failure")

    _install(monkeypatch, _hybrid_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", BrokenSessionClient)

    with pytest.raises(RuntimeError, match="injected session creation failure"):
        state._new_runtime(
            (
                ArenaRegistration(0, 1, 1, 8),
                ArenaRegistration(1, 2, 2, 4),
            )
        )

    assert state._RUNTIME is None


def test_pure_sliding_runtime_creates_session_without_legacy_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()

    class SessionClient:
        @staticmethod
        def create(config, settings, registrations):
            assert config is state._CONFIG
            assert tuple(registrations) == (ArenaRegistration(0, 1, 1, 4),)
            raw.cache_sharing_policy = settings.cache_sharing_policy
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            self.mirror_cleanup = mirror_cleanup
            self.cache_sharing_policy = session.cache_sharing_policy

        def close(self):
            self.session.close()

    _install(monkeypatch, _pure_sliding_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)

    runtime = state._new_runtime((ArenaRegistration(0, 1, 1, 4),))

    assert runtime is state._RUNTIME
    assert runtime.session is raw
    assert runtime.mirror_cleanup is session_runtime.unbound_mirror_cleanup
    assert runtime.cache_sharing_policy is CacheSharingPolicy.REQUEST_PRIVATE


def test_full_latent_runtime_creates_request_private_native_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()
    creates = []

    class SessionClient:
        @staticmethod
        def create(config, settings, registrations):
            creates.append((config, settings, tuple(registrations)))
            raw.cache_sharing_policy = settings.cache_sharing_policy
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            self.mirror_cleanup = mirror_cleanup
            self.cache_sharing_policy = session.cache_sharing_policy

        def close(self):
            self.session.close()

    _install(monkeypatch, _latent_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)
    registration = ArenaRegistration(0, 1, 1, 8)

    runtime = state._new_runtime((registration,))

    assert runtime is state._RUNTIME
    assert runtime.session is raw
    assert runtime.mirror_cleanup is session_runtime.unbound_mirror_cleanup
    assert runtime.cache_sharing_policy is CacheSharingPolicy.REQUEST_PRIVATE
    assert creates[0][0] is state._CONFIG
    settings = creates[0][1]
    assert type(settings) is SessionCreateSettings
    assert type(settings.manager) is ManagerCreateSettings
    assert settings.manager.maximum_requests == 8
    assert settings.manager.maximum_operations == 8
    assert settings.manager.maximum_prefixes == 8
    assert settings.manager.maximum_reclamations == 8
    assert settings.manager.maximum_step_tokens == 64
    assert settings.cache_sharing_policy is CacheSharingPolicy.REQUEST_PRIVATE
    assert creates[0][2] == (registration,)


def test_chunked_runtime_creates_session_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()
    creates = []

    class SessionClient:
        @staticmethod
        def create(config, settings, registrations):
            creates.append((config, settings, tuple(registrations)))
            raw.cache_sharing_policy = settings.cache_sharing_policy
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            self.mirror_cleanup = mirror_cleanup
            self.cache_sharing_policy = session.cache_sharing_policy

        def close(self):
            self.session.close()

    _install(monkeypatch, _chunked_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)
    registration = ArenaRegistration(0, 1, 1, 8, 3)

    runtime = state._new_runtime((registration,))

    assert runtime is state._RUNTIME
    assert runtime.session is raw
    assert runtime.mirror_cleanup is session_runtime.unbound_mirror_cleanup
    assert creates[0][0] is state._CONFIG
    assert creates[0][0].manager_plan_format == "retention_ir"
    assert (
        creates[0][1].cache_sharing_policy
        is CacheSharingPolicy.REQUEST_PRIVATE
    )
    assert creates[0][2] == (registration,)
    assert raw.close_count == 0


def test_chunked_session_creation_failure_never_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSessionClient:
        @staticmethod
        def create(*_args: Any) -> Any:
            raise RuntimeError("injected chunked session creation failure")

    _install(monkeypatch, _chunked_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", BrokenSessionClient)

    with pytest.raises(
        RuntimeError, match="injected chunked session creation failure"
    ):
        state._new_runtime((ArenaRegistration(0, 1, 1, 8),))

    assert state._RUNTIME is None


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"runtime_binding": None},
            "requires ORBITKV_RUNTIME_MANIFEST",
        ),
        ({"manager_plan_format": "retention_ir"}, "product native-session"),
        (
            {"token_reclamation": TokenReclamationConfig(mode="naive")},
            "product native-session",
        ),
        (
            {
                "fixed_states": (
                    FixedStateConfig("state", "mamba", (1,), 64, 2),
                )
            },
            "product native-session",
        ),
    ),
)
def test_declared_takeover_topology_never_falls_back_for_profile_mismatch(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any], message: str
) -> None:
    _install(monkeypatch, _full_config(**changes))
    monkeypatch.setattr(
        state.CtypesRuntimeSession,
        "create",
        lambda *_args: pytest.fail("runtime was created before admission"),
    )

    with pytest.raises(ValueError, match=message):
        state._new_runtime((ArenaRegistration(0, 1, 1, 8),))

    assert state._RUNTIME is None


def test_new_runtime_transfers_the_session_handle_to_session_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()
    creates = []
    cleanup = lambda _updates, _retirements: True

    class SessionClient:
        @staticmethod
        def create(config, settings, registrations):
            creates.append((config, settings, tuple(registrations)))
            raw.cache_sharing_policy = settings.cache_sharing_policy
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            self.cleanup = mirror_cleanup
            self.close_count = 0
            self.cache_sharing_policy = session.cache_sharing_policy

        def close(self):
            self.close_count += 1
            self.session.close()

    _install(monkeypatch, _full_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)
    registration = ArenaRegistration(0, 1, 1, 8, 3)

    runtime = state._new_runtime(
        (registration,), mirror_cleanup=cleanup
    )

    assert runtime is state._RUNTIME
    assert runtime.session is raw
    assert runtime.cleanup is cleanup
    assert raw.close_count == 0
    assert creates[0][0] is state._CONFIG
    settings = creates[0][1]
    assert settings.manager.maximum_requests == 8
    assert settings.manager.maximum_operations == 8
    assert settings.manager.maximum_prefixes == 8
    assert settings.manager.maximum_reclamations == 8
    assert settings.manager.maximum_step_tokens == 64
    assert settings.cache_sharing_policy is CacheSharingPolicy.SHARED_PREFIX
    assert creates[0][2] == (registration,)

    state._close_owned_runtime(runtime)

    assert runtime.close_count == 1
    assert raw.close_count == 1
    assert state._RUNTIME is None


def test_new_runtime_closes_raw_session_when_wrapper_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()

    class SessionClient:
        @staticmethod
        def create(*_args: Any) -> _RawSession:
            return raw

    class BrokenRuntime:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("injected wrapper failure")

    _install(monkeypatch, _full_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", BrokenRuntime)

    with pytest.raises(RuntimeError, match="injected wrapper failure"):
        state._new_runtime((ArenaRegistration(0, 1, 1, 8),))

    assert raw.close_count == 1
    assert state._RUNTIME is None


def test_new_runtime_uses_unbound_cleanup_marker_for_two_phase_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RawSession()
    observed = []

    class SessionClient:
        @staticmethod
        def create(*_args: Any) -> _RawSession:
            return raw

    class Runtime:
        def __init__(self, session, *, mirror_cleanup):
            self.session = session
            observed.append(mirror_cleanup)
            self.cache_sharing_policy = session.cache_sharing_policy

        def bind_mirror_cleanup(self, cleanup):
            observed.append(cleanup)

        def close(self):
            self.session.close()

    _install(monkeypatch, _full_config())
    monkeypatch.setattr(state, "CtypesRuntimeSession", SessionClient)
    monkeypatch.setattr(session_runtime, "SessionRuntime", Runtime)

    state._new_runtime((ArenaRegistration(0, 1, 1, 8),))
    assert observed == [session_runtime.unbound_mirror_cleanup]

    cleanup = lambda _updates, _retirements: True
    state._bind_session_cleanup(cleanup)
    assert observed == [session_runtime.unbound_mirror_cleanup, cleanup]


def test_session_allocator_availability_reads_arena_stats_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        page_tokens = 16
        failure_reason = None

        def __init__(self):
            self.calls = []

        def capacity_arena_stats(self):
            self.calls.extend(("poll", "arena_stats"))
            return (SimpleNamespace(class_id=0, free_pages=3),)

        def census(self):
            pytest.fail("session availability must not use manager census")

        def fail_stop(self, reason):
            self.failure_reason = reason

    runtime = Runtime()
    _install(monkeypatch, _full_config(), runtime=runtime)

    assert state._arena_available_tokens(0) == 48
    assert state._arena_available_tokens(0) == 48
    assert runtime.calls == [
        "poll",
        "arena_stats",
        "poll",
        "arena_stats",
    ]


def test_missing_manifest_fails_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_config(), runtime_manifest_path=None)
    _install(monkeypatch, config)
    monkeypatch.setattr(
        state.CtypesRuntimeSession,
        "create",
        lambda *_args: pytest.fail("runtime was created before admission"),
    )

    with pytest.raises(ValueError, match="requires ORBITKV_RUNTIME_MANIFEST"):
        state._new_runtime((ArenaRegistration(0, 1, 1, 8),))

    assert state._RUNTIME is None
    assert not state._uses_runtime_session()


def test_product_state_has_no_legacy_runtime_factory() -> None:
    source = inspect.getsource(state)

    assert "CtypesManagerFactory" not in source
    assert "CanonicalRuntime" not in source
    assert "_FACTORY" not in source


def test_allocator_build_uses_session_mirror_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Allocator:
        def __init__(self, *_args: Any, class_id: int, **_kwargs: Any):
            self.class_id = class_id

    class Runtime:
        def bind_prefix_eviction_cleanup(self, _coordinator):
            pytest.fail("session runtime received legacy prefix authority")

    runtime = Runtime()
    coordinator = object()
    _install(monkeypatch, _full_config())
    monkeypatch.setattr(facade, "_facade_types", lambda: (Allocator, object, object))
    monkeypatch.setattr(facade, "_ensure_relocation_kv_copy_contract", lambda _pool: None)

    def new_runtime(_registrations):
        state._RUNTIME = runtime
        return runtime

    monkeypatch.setattr(facade, "_new_runtime", new_runtime)
    monkeypatch.setattr(
        facade, "_mirror_cleanup_coordinator", lambda *_args: coordinator
    )
    token_pool = object()

    allocator = facade._build_token_to_kv_pool_allocator(
        SimpleNamespace(
            page_size=16,
            device="cuda:0",
            kv_cache_dtype=object(),
            use_mla_backend=False,
            is_hybrid_swa=False,
            is_draft_worker=False,
        ),
        sizes=SimpleNamespace(max_total_num_tokens=128),
        token_to_kv_pool=token_pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )

    assert allocator is state._ALLOCATOR
    assert allocator.class_id == 0
