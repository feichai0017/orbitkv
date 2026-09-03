from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang.ffi import session as session_ffi
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.runtime import (
    CacheSharingPolicy,
    ManagerError,
    SessionCreateSettings,
)
from test_session_ffi import (
    REGISTRATIONS,
    SETTINGS,
    FakeSessionLibrary,
    _config,
    _open_session_with_settings,
)


@pytest.mark.parametrize(
    "policy",
    (CacheSharingPolicy.REQUEST_PRIVATE, CacheSharingPolicy.SHARED_PREFIX),
)
def test_session_create_encodes_and_exposes_explicit_cache_sharing_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: CacheSharingPolicy,
) -> None:
    fake = FakeSessionLibrary()
    settings = SessionCreateSettings(SETTINGS.manager, policy)

    session = _open_session_with_settings(tmp_path, monkeypatch, fake, settings)

    assert fake.created is not None
    assert fake.created["cache_sharing_policy"] == int(policy)
    assert fake.created["reserved"] == 0
    assert session.cache_sharing_policy is policy
    session.close()


@pytest.mark.parametrize(
    "settings",
    (
        SETTINGS.manager,
        SessionCreateSettings(SETTINGS.manager, 1),
        SessionCreateSettings(SETTINGS.manager, True),
        SessionCreateSettings(SETTINGS.manager, "shared_prefix"),
        SessionCreateSettings(object(), CacheSharingPolicy.SHARED_PREFIX),
    ),
)
def test_session_create_rejects_noncanonical_settings_before_loading_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Any,
) -> None:
    loaded = False

    def unexpected_load(_path: Path) -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError("native library was loaded")

    monkeypatch.setattr(session_ffi, "LoadedLibrary", unexpected_load)

    with pytest.raises(ManagerError, match="create settings|manager settings|policy"):
        CtypesRuntimeSession.create(
            _config(tmp_path / "not-loaded.so"), settings, REGISTRATIONS
        )

    assert not loaded


def test_session_create_settings_are_frozen_and_require_explicit_policy() -> None:
    with pytest.raises(TypeError):
        SessionCreateSettings(SETTINGS.manager)
    with pytest.raises(FrozenInstanceError):
        SETTINGS.cache_sharing_policy = CacheSharingPolicy.REQUEST_PRIVATE
