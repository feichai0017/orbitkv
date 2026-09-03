from __future__ import annotations

from pathlib import Path

import pytest

from orbitkv_sglang.ffi import session as session_ffi
from orbitkv_sglang.ffi.session_types import (
    EnginePrefixId,
    EnginePrefixPublishItem,
    EnginePrefixPublishReleasePlan,
    EnginePublishedPrefixRelease,
    EngineReleaseDisposition,
    EngineReleaseEvidence,
    EngineReleaseId,
    EngineReleaseOutcome,
    EngineReleasePlan,
    EngineReleasedRequest,
    EngineRequestId,
)
from orbitkv_sglang.runtime import PrefixSemanticKey
from test_session_ffi import (
    CONTROL_SETTINGS,
    FakeSessionLibrary,
    _open_session_with_settings,
    _raw_detached,
)


def test_prefix_publish_release_registers_pending_release_and_reuses_confirm_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_session_with_settings(
        tmp_path, monkeypatch, fake, CONTROL_SETTINGS
    )

    plan = session.prefix_publish_release_batch(
        (
            EnginePrefixPublishItem(
                EngineRequestId(101),
                PrefixSemanticKey(b"a" * 32, b"b" * 32, 16),
            ),
            EnginePrefixPublishItem(
                EngineRequestId(202),
                PrefixSemanticKey(b"c" * 32, b"d" * 32, 32),
            ),
        )
    )
    assert plan == EnginePrefixPublishReleasePlan(
        EngineReleaseId(77, 4),
        (
            EnginePublishedPrefixRelease(
                EnginePrefixId(77, 30),
                PrefixSemanticKey(b"a" * 32, b"b" * 32, 16),
                1,
                EngineReleasedRequest(
                    EngineRequestId(101),
                    (session_ffi._detached(_raw_detached(10)),),
                ),
            ),
            EnginePublishedPrefixRelease(
                EnginePrefixId(77, 31),
                PrefixSemanticKey(b"c" * 32, b"d" * 32, 32),
                2,
                EngineReleasedRequest(
                    EngineRequestId(202),
                    (session_ffi._detached(_raw_detached(11)),),
                ),
            ),
        ),
    )
    assert plan.release_id in session._pending_releases
    assert session._pending_releases[plan.release_id] == EngineReleasePlan(
        plan.release_id,
        tuple(item.release for item in plan.outputs),
        (),
    )

    outcome = session.confirm_release(
        EngineReleaseEvidence(plan.release_id, True, ())
    )
    assert outcome == EngineReleaseOutcome(
        plan.release_id, EngineReleaseDisposition.COMPLETED
    )
    assert plan.release_id not in session._pending_releases
    assert fake.invocations[-1] == "confirm_release"
    session.close()
