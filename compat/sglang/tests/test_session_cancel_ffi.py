from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang.ffi import layouts as L
from orbitkv_sglang.ffi.library import (
    STATUS_MANAGER_ERROR,
    STATUS_OK,
    STATUS_RETRYABLE_CONFLICT,
)
from orbitkv_sglang.ffi.session_types import (
    EngineControlId,
    EngineMaterializationPlan,
    EnginePendingAttachCancel,
    EnginePendingAttachCancelDisposition,
    EnginePendingAttachCancelOutcome,
    EnginePrefixAttachItem,
    EnginePrefixId,
    EngineRequestId,
)
from orbitkv_sglang.runtime import (
    FailStopped,
    ManagerError,
    PrefixSemanticKey,
    RetryableConflict,
)
from test_session_ffi import (
    CONTROL_SETTINGS,
    FakeSessionLibrary,
    _open_session_with_settings,
    _store,
)


class FakeSessionCancelLibrary(FakeSessionLibrary):
    def orbitkv_session_cancel_pending_attach(
        self,
        _handle: Any,
        expected: Any,
        out_outcome: Any,
        error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("cancel_pending_attach")
        if self.fault == "pending_attach_cancel_raise":
            raise RuntimeError("lost pending attach cancel return")
        if self.fault == "pending_attach_cancel_retryable":
            return STATUS_RETRYABLE_CONFLICT
        return _write_pending_attach_outcome(
            expected,
            out_outcome,
            error,
            EnginePendingAttachCancelDisposition.RECYCLE_PENDING,
        )

    def orbitkv_session_finalize_pending_attach_cancel(
        self,
        _handle: Any,
        expected: Any,
        out_outcome: Any,
        error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("finalize_pending_attach_cancel")
        if self.fault == "pending_attach_finalize_raise":
            raise RuntimeError("lost pending attach finalize return")
        if self.fault == "pending_attach_finalize_retryable":
            return STATUS_RETRYABLE_CONFLICT
        return _write_pending_attach_outcome(
            expected,
            out_outcome,
            error,
            EnginePendingAttachCancelDisposition.FINALIZED,
        )


def _write_pending_attach_outcome(
    expected: Any,
    out_outcome: Any,
    error: Any,
    disposition: EnginePendingAttachCancelDisposition,
) -> int:
    control_id = (
        int(expected.control_id.session_epoch),
        int(expected.control_id.sequence),
    )
    request_id = int(expected.request_id)
    prefix_id = (
        int(expected.prefix_id.session_epoch),
        int(expected.prefix_id.sequence),
    )
    view_version = int(expected.view_version)
    boundary = int(expected.boundary)
    resident_count = int(expected.resident_count)
    if control_id == (77, 41):
        canonical = (202, (77, 11), 9, 32, 2)
        if (request_id, prefix_id, view_version, boundary, resident_count) != canonical:
            error.value = b"pending attach mismatch"
            return STATUS_MANAGER_ERROR
        request_id, prefix_id, view_version, boundary, resident_count = canonical

    _store(
        out_outcome,
        L.SessionPendingAttachCancelOutcomeLayout,
        L.SessionPendingAttachCancelOutcomeLayout(
            L.SessionControlIdLayout(*control_id),
            request_id,
            L.SessionPrefixIdLayout(*prefix_id),
            view_version,
            boundary,
            resident_count,
            int(disposition),
        ),
    )
    return STATUS_OK


def _open_cancel_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeSessionCancelLibrary,
):
    return _open_session_with_settings(
        tmp_path, monkeypatch, fake, CONTROL_SETTINGS
    )


def _prepare_pending_attach_cancel(session: Any) -> EnginePendingAttachCancel:
    key = PrefixSemanticKey(b"a" * 32, b"b" * 32, 16)
    control_id = session.prepare_prefix_attach(
        (
            EnginePrefixAttachItem(
                EngineRequestId(202), EnginePrefixId(77, 11), key, 2
            ),
        )
    )
    session.commit_control(control_id)
    plan = session.read_control_plan(control_id)
    assert isinstance(plan, EngineMaterializationPlan)
    return EnginePendingAttachCancel(
        control_id,
        plan.requests[0].request_id,
        EnginePrefixId(77, 11),
        plan.requests[0].view_version,
        plan.requests[0].boundary,
        plan.requests[0].resident_count,
    )


def test_pending_attach_cancel_and_finalize_replay_exact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionCancelLibrary()
    session = _open_cancel_session(tmp_path, monkeypatch, fake)
    expected = _prepare_pending_attach_cancel(session)

    first = session.cancel_pending_attach(expected)
    assert first == EnginePendingAttachCancelOutcome(
        expected.control_id,
        expected.request_id,
        expected.prefix_id,
        expected.view_version,
        expected.boundary,
        expected.resident_count,
        EnginePendingAttachCancelDisposition.RECYCLE_PENDING,
    )
    assert fake.invocations[-1] == "cancel_pending_attach"
    assert expected.control_id not in session._pending_controls

    before = tuple(fake.invocations)
    replay = session.cancel_pending_attach(expected)
    assert replay == first
    assert tuple(fake.invocations) == before

    finalized = session.finalize_pending_attach_cancel(expected)
    assert finalized == EnginePendingAttachCancelOutcome(
        expected.control_id,
        expected.request_id,
        expected.prefix_id,
        expected.view_version,
        expected.boundary,
        expected.resident_count,
        EnginePendingAttachCancelDisposition.FINALIZED,
    )
    assert fake.invocations[-1] == "finalize_pending_attach_cancel"
    assert expected.control_id not in session._pending_attach_cancels
    assert session._finalized_attach_cancels[expected.control_id] == finalized

    before = tuple(fake.invocations)
    assert session.finalize_pending_attach_cancel(expected) == finalized
    assert tuple(fake.invocations) == before

    before = tuple(fake.invocations)
    assert session.cancel_pending_attach(expected) == finalized
    assert tuple(fake.invocations) == before
    session.close()


def test_pending_attach_cancel_mismatch_and_retryable_failure_preserve_exact_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionCancelLibrary()
    session = _open_cancel_session(tmp_path, monkeypatch, fake)
    expected = _prepare_pending_attach_cancel(session)
    mismatched = EnginePendingAttachCancel(
        expected.control_id,
        EngineRequestId(int(expected.request_id) + 1),
        expected.prefix_id,
        expected.view_version,
        expected.boundary,
        expected.resident_count,
    )

    with pytest.raises(ManagerError, match="pending attach mismatch"):
        session.cancel_pending_attach(mismatched)
    assert expected.control_id in session._pending_controls
    assert expected.control_id not in session._pending_attach_cancels

    first = session.cancel_pending_attach(expected)
    assert first.disposition is EnginePendingAttachCancelDisposition.RECYCLE_PENDING

    before = tuple(fake.invocations)
    with pytest.raises(ManagerError, match="differs from replay"):
        session.finalize_pending_attach_cancel(mismatched)
    assert tuple(fake.invocations) == before
    session.close()


def test_pending_attach_finalize_replays_are_bounded_by_operation_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionCancelLibrary()
    session = _open_cancel_session(tmp_path, monkeypatch, fake)
    assert (
        session._finalized_attach_cancel_limit
        == CONTROL_SETTINGS.manager.maximum_operations
    )

    def make_expected(control_sequence: int, request_id: int) -> EnginePendingAttachCancel:
        return EnginePendingAttachCancel(
            EngineControlId(77, control_sequence),
            EngineRequestId(request_id),
            EnginePrefixId(77, 11),
            9,
            32,
            2,
        )

    first = make_expected(41, 202)
    second = make_expected(42, 203)
    third = make_expected(43, 204)
    for expected in (first, second, third):
        session._remember_finalized_attach_cancel(
            EnginePendingAttachCancelOutcome(
                expected.control_id,
                expected.request_id,
                expected.prefix_id,
                expected.view_version,
                expected.boundary,
                expected.resident_count,
                EnginePendingAttachCancelDisposition.FINALIZED,
            )
        )
    assert tuple(session._finalized_attach_cancels) == (
        second.control_id,
        third.control_id,
    )
    session.close()


def test_pending_attach_retryable_and_unknown_failures_preserve_or_poison_as_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retryable = FakeSessionCancelLibrary("pending_attach_cancel_retryable")
    session = _open_cancel_session(tmp_path, monkeypatch, retryable)
    expected = _prepare_pending_attach_cancel(session)
    with pytest.raises(RetryableConflict, match="retryable conflict"):
        session.cancel_pending_attach(expected)
    assert expected.control_id in session._pending_controls
    assert expected.control_id not in session._pending_attach_cancels
    assert not session._poisoned
    session.close()

    unknown = FakeSessionCancelLibrary("pending_attach_finalize_raise")
    session = _open_cancel_session(tmp_path, monkeypatch, unknown)
    expected = _prepare_pending_attach_cancel(session)
    session.cancel_pending_attach(expected)
    with pytest.raises(FailStopped, match="poisoned"):
        session.finalize_pending_attach_cancel(expected)
    before = tuple(unknown.invocations)
    with pytest.raises(FailStopped, match="poisoned"):
        session.acquire_requests((303,))
    assert tuple(unknown.invocations) == before
    assert expected.control_id in session._pending_attach_cancels
    assert expected.control_id not in session._finalized_attach_cancels
    session.close()
