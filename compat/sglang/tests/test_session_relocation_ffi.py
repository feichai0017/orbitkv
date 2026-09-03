from __future__ import annotations

import ctypes
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest

import orbitkv_sglang.ffi as public_ffi
from orbitkv_sglang.ffi import layouts as L
from orbitkv_sglang.ffi import session_types as session_dtos
from orbitkv_sglang.ffi.library import (
    EXACT_SYMBOL_ALLOWLIST,
    FUNCTION_SPECS,
    STATUS_BUFFER_TOO_SMALL,
    STATUS_FAIL_STOPPED,
    STATUS_OK,
    WIRE_VERSION,
)
from orbitkv_sglang.ffi.session_types import (
    EngineAppendIntent,
    EngineCompletionEvidence,
    EnginePrepareRelocationItem,
    EngineRelocationAbortEvidence,
    EngineRelocationCopyEvidence,
    EngineRelocationExecutionEvidence,
    EngineRelocationId,
    EngineRelocationPublicationEvidence,
    EngineRelocationRequestEvidence,
    EngineRequestId,
    EngineTokenDispositionBatchItem,
    EngineTokenDispositionUpdate,
    EngineTokenViewQuery,
    retirement_evidence,
)
from orbitkv_sglang.runtime import (
    FailStopped,
    ManagerError,
    RelocationPolicy,
    TokenDisposition,
    TokenDispositionKind,
)
from test_session_ffi import (
    FakeSessionLibrary,
    SESSION_SYMBOL_ARITIES,
    _assert_no_internal_leases,
    _count,
    _open_session,
    _native_config,
    _native_execution_evidence,
    _raw_page,
    _raw_retirement,
    _store,
)
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    ManagerCreateSettings,
    SessionCreateSettings,
)
from ffi_test_support import ffi_library


__all__ = ["ffi_library"]


RELOCATION_LAYOUTS = {
    L.SessionRelocationIdLayout: (16, ("session_epoch", "sequence")),
    L.SessionTokenViewQueryLayout: (
        24, ("request_id", "expected_boundary", "class_id", "reserved16", "reserved32")
    ),
    L.SessionTokenViewLayout: (
        40, ("request_id", "view_version", "placement_offset", "placement_count", "page_tokens", "class_id", "reserved16", "reserved32")
    ),
    L.SessionTokenDispositionBatchItemLayout: (
        16, ("request_id", "update_offset", "update_count")
    ),
    L.SessionPrepareRelocationItemLayout: (
        32, ("request_id", "policy", "class_id", "reserved16", "reserved32")
    ),
    L.SessionRelocationPlanLayout: (
        64, ("request_id", "base_view_version", "target_view_version", "source_offset", "source_count", "destination_offset", "destination_count", "move_offset", "move_count", "projected_reclaimed_pages", "fragmentation_milli", "class_id", "reserved32")
    ),
    L.SessionRelocationRequestEvidenceLayout: (
        16, ("request_id", "copy_offset", "copy_count")
    ),
    L.SessionRelocationCopyEvidenceLayout: (
        112, ("token_id", "source", "destination", "observed", "copied", "reserved16", "reserved32")
    ),
    L.SessionRelocationAbortEvidenceLayout: (
        16, ("request_id", "backend_unobserved", "reserved")
    ),
    L.SessionRelocationRequestPublicationLayout: (
        32, ("request_id", "view_version", "boundary", "resident_count", "reserved")
    ),
    L.SessionRelocationPublicationEvidenceLayout: (
        24, ("relocation_id", "mirror_cleanup_confirmed", "reserved")
    ),
}

RELOCATION_SYMBOL_ARITIES = {
    "orbitkv_session_token_views_batch": 11,
    "orbitkv_session_mark_token_dispositions_batch": 10,
    "orbitkv_session_prepare_relocation_batch": 18,
    "orbitkv_session_abort_prepared_relocation": 6,
    "orbitkv_session_quarantine_relocation": 4,
    "orbitkv_session_submit_relocation": 8,
    "orbitkv_session_complete_relocation": 11,
    "orbitkv_session_confirm_relocation_publication": 6,
}

RELOCATION_DTO_NAMES = (
    "EngineRelocationId", "EngineTokenViewQuery", "EngineTokenView",
    "EngineTokenDispositionUpdate", "EngineTokenDispositionBatchItem",
    "EnginePrepareRelocationItem", "EngineRelocationPlan",
    "EnginePreparedRelocation", "EngineRelocationCopyEvidence",
    "EngineRelocationRequestEvidence", "EngineRelocationExecutionEvidence",
    "EngineRelocationAbortEvidence", "EngineRelocationTicket",
    "EngineRelocationRequestPublication", "EngineRelocationPublication",
    "EngineRelocationPublicationEvidence",
)


def test_session_relocation_wire_contract_is_frozen() -> None:
    assert WIRE_VERSION == 14
    assert len(FUNCTION_SPECS) == 47
    assert len(EXACT_SYMBOL_ALLOWLIST) == 48
    assert not any(
        name.startswith("orbitkv_manager_")
        for name in EXACT_SYMBOL_ALLOWLIST
    )
    for layout, (size, names) in RELOCATION_LAYOUTS.items():
        assert L.FROZEN_LAYOUTS[layout] == (size, 8)
        assert ctypes.sizeof(layout) == size
        assert ctypes.alignment(layout) == 8
        assert tuple(name for name, _ctype in layout._fields_) == names
    for name, arity in RELOCATION_SYMBOL_ARITIES.items():
        assert name in EXACT_SYMBOL_ALLOWLIST
        assert len(FUNCTION_SPECS[name]) == arity
    assert {
        name for name in FUNCTION_SPECS if name.startswith("orbitkv_session_")
    } == set(SESSION_SYMBOL_ARITIES) | set(RELOCATION_SYMBOL_ARITIES)


def test_session_relocation_dtos_are_opaque_and_frozen() -> None:
    for name in RELOCATION_DTO_NAMES:
        dto = getattr(session_dtos, name)
        assert getattr(public_ffi, name) is dto
        assert dto.__dataclass_params__.frozen
        assert "__slots__" in dto.__dict__
        annotations = " ".join(str(field.type) for field in fields(dto))
        assert "RequestLease" not in annotations
        assert "SnapshotLease" not in annotations
        assert "RelocationLease" not in annotations
    relocation_id = EngineRelocationId(77, 1)
    assert relocation_id != EngineRelocationId(78, 1)
    with pytest.raises(FrozenInstanceError):
        relocation_id.sequence = 2  # type: ignore[misc]


def _raw_location(
    page_id: int, backend_index: int, offset: int
) -> L.TokenLocationLayout:
    return L.TokenLocationLayout(
        _raw_page(page_id, 11), backend_index, offset, 0
    )


class FakeSessionRelocationLibrary(FakeSessionLibrary):
    def __init__(self, fault: str | None = None) -> None:
        super().__init__(fault)
        self.token_view_calls = 0
        self.disposition_input: tuple[Any, ...] = ()
        self.relocation_input: tuple[Any, ...] = ()
        self.relocation_evidence: tuple[Any, ...] = ()
        self.relocation_confirmation: tuple[Any, ...] = ()

    def orbitkv_session_token_views_batch(
        self,
        _handle: Any,
        queries: Any,
        query_count: int,
        views: Any,
        view_capacity: int,
        out_view_count: Any,
        placements: Any,
        placement_capacity: int,
        out_placement_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("token_views")
        self.token_view_calls += 1
        expected = sum(
            int(queries[index].expected_boundary)
            for index in range(query_count)
        )
        _count(out_view_count, query_count)
        _count(
            out_placement_count,
            expected + (1 if self.fault == "token_view_preflight_count" else 0),
        )
        if view_capacity < query_count or placement_capacity < expected:
            return STATUS_BUFFER_TOO_SMALL
        cursor = 0
        for index in range(query_count):
            query = queries[index]
            boundary = int(query.expected_boundary)
            views[index] = L.SessionTokenViewLayout(
                int(query.request_id),
                2,
                cursor,
                boundary,
                16,
                int(query.class_id),
                1 if self.fault == "token_view_reserved" else 0,
                0,
            )
            for token_id in range(boundary):
                retained = token_id == 0
                disposition = L.TokenDispositionLayout(
                    0 if retained else 7,
                    0 if retained else 1,
                    0,
                    int(
                        TokenDispositionKind.RETAINED
                        if retained
                        else TokenDispositionKind.SEMANTICALLY_DEAD
                    ),
                    0,
                    0,
                )
                placements[cursor + token_id] = L.TokenPlacementLayout(
                    token_id,
                    disposition,
                    _raw_location(10, 1000, token_id),
                    1,
                    0,
                )
            cursor += boundary
        return STATUS_OK

    def orbitkv_session_mark_token_dispositions_batch(
        self,
        _handle: Any,
        items: Any,
        item_count: int,
        updates: Any,
        update_count: int,
        outputs: Any,
        output_capacity: int,
        out_output_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("mark_token_dispositions")
        assert output_capacity >= item_count
        self.disposition_input = tuple(
            (
                int(updates[index].class_id),
                int(updates[index].token_id),
                int(updates[index].disposition.kind),
            )
            for index in range(update_count)
        )
        for index in range(item_count):
            outputs[index] = L.SessionRequestViewLayout(
                int(items[index].request_id), 2, 2, 1, 0
            )
        _count(out_output_count, item_count)
        return STATUS_OK

    def orbitkv_session_prepare_relocation_batch(
        self,
        _handle: Any,
        items: Any,
        item_count: int,
        out_id: Any,
        plans: Any,
        plan_capacity: int,
        out_plan_count: Any,
        sources: Any,
        source_capacity: int,
        out_source_count: Any,
        destinations: Any,
        destination_capacity: int,
        out_destination_count: Any,
        moves: Any,
        move_capacity: int,
        out_move_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prepare_relocation")
        assert item_count == 1
        assert plan_capacity >= 1
        assert source_capacity >= 1 and destination_capacity >= 1
        assert move_capacity >= 1
        item = items[0]
        self.relocation_input = (
            int(item.request_id),
            int(item.class_id),
            int(item.policy.full_evacuation),
        )
        _store(
            out_id,
            L.SessionRelocationIdLayout,
            L.SessionRelocationIdLayout(77, 51),
        )
        source = _raw_location(10, 1000, 0)
        destination = _raw_location(11, 1001, 0)
        sources[0] = source.page
        destinations[0] = destination.page
        moves[0] = L.TokenMoveLayout(0, source, destination)
        plans[0] = L.SessionRelocationPlanLayout(
            int(item.request_id),
            2,
            3,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            500,
            int(item.class_id),
            1 if self.fault == "relocation_plan_reserved" else 0,
        )
        _count(out_plan_count, 1)
        _count(out_source_count, 1)
        _count(out_destination_count, 1)
        _count(out_move_count, 1)
        return STATUS_OK

    def orbitkv_session_abort_prepared_relocation(
        self,
        _handle: Any,
        relocation_id: Any,
        evidence: Any,
        evidence_count: int,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("abort_relocation")
        assert (int(relocation_id.session_epoch), int(relocation_id.sequence)) == (
            77,
            51,
        )
        assert evidence_count == 1
        assert (int(evidence[0].request_id), int(evidence[0].backend_unobserved)) == (
            101,
            1,
        )
        return STATUS_OK

    def orbitkv_session_quarantine_relocation(
        self,
        _handle: Any,
        _relocation_id: Any,
        error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("quarantine_relocation")
        if self.fault == "quarantine_fail_stopped":
            error.value = b"relocation quarantined"
            return STATUS_FAIL_STOPPED
        return STATUS_OK

    def orbitkv_session_submit_relocation(
        self,
        _handle: Any,
        relocation_id: Any,
        requests: Any,
        request_count: int,
        copies: Any,
        copy_count: int,
        error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("submit_relocation")
        self.relocation_evidence = (
            (int(relocation_id.session_epoch), int(relocation_id.sequence)),
            tuple(
                (
                    int(requests[index].request_id),
                    int(requests[index].copy_offset),
                    int(requests[index].copy_count),
                )
                for index in range(request_count)
            ),
            tuple(
                (
                    int(copies[index].token_id),
                    int(copies[index].observed),
                    int(copies[index].copied),
                )
                for index in range(copy_count)
            ),
        )
        if self.fault == "submit_fail_stopped":
            error.value = b"relocation batch quarantined"
            return STATUS_FAIL_STOPPED
        return STATUS_OK

    def orbitkv_session_complete_relocation(
        self,
        _handle: Any,
        relocation_id: Any,
        completion: Any,
        publications: Any,
        publication_capacity: int,
        out_publication_count: Any,
        retirements: Any,
        retirement_capacity: int,
        out_retirement_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("complete_relocation")
        assert (int(relocation_id.session_epoch), int(relocation_id.sequence)) == (
            77,
            51,
        )
        assert (
            int(completion.completion_domain),
            int(completion.completion_value),
            int(completion.confirmed),
        ) == (9, 90, 1)
        assert publication_capacity >= 1 and retirement_capacity >= 1
        publications[0] = L.SessionRelocationRequestPublicationLayout(
            101, 3, 2, 1, 0
        )
        retirements[0] = _raw_retirement(0)
        _count(out_publication_count, 1)
        _count(out_retirement_count, 1)
        return STATUS_OK

    def orbitkv_session_confirm_relocation_publication(
        self,
        _handle: Any,
        evidence: Any,
        retirements: Any,
        retirement_count: int,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("confirm_relocation")
        self.relocation_confirmation = (
            (
                int(evidence.relocation_id.session_epoch),
                int(evidence.relocation_id.sequence),
                int(evidence.mirror_cleanup_confirmed),
            ),
            tuple(
                (
                    int(retirements[index].page.page_id),
                    int(retirements[index].acknowledged),
                    int(retirements[index].backend_index),
                )
                for index in range(retirement_count)
            ),
        )
        return STATUS_OK


def _prepare(session: Any) -> Any:
    return session.prepare_relocation_batch(
        (
            EnginePrepareRelocationItem(
                EngineRequestId(101),
                0,
                RelocationPolicy(1, 1, 250, True),
            ),
        )
    )


def _execution(prepared: Any) -> EngineRelocationExecutionEvidence:
    plan = prepared.plans[0]
    return EngineRelocationExecutionEvidence(
        prepared.relocation_id,
        (
            EngineRelocationRequestEvidence(
                plan.request_id,
                tuple(
                    EngineRelocationCopyEvidence(
                        move.token_id,
                        move.source,
                        move.destination,
                        True,
                        True,
                    )
                    for move in plan.moves
                ),
            ),
        ),
    )


def test_session_relocation_opaque_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionRelocationLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)

    views = session.token_views_batch(
        (EngineTokenViewQuery(EngineRequestId(101), 0, 2),)
    )
    assert fake.token_view_calls == 2
    assert views[0].request_id == EngineRequestId(101)
    assert views[0].view_version == 2
    assert len(views[0].placements) == 2
    assert views[0].placements[1].disposition.kind is (
        TokenDispositionKind.SEMANTICALLY_DEAD
    )

    updated = session.mark_token_dispositions_batch(
        (
            EngineTokenDispositionBatchItem(
                EngineRequestId(101),
                (
                    EngineTokenDispositionUpdate(
                        0,
                        1,
                        TokenDisposition(
                            TokenDispositionKind.SEMANTICALLY_DEAD, 7, 1, 0
                        ),
                    ),
                ),
            ),
        )
    )
    assert updated[0].request_id == EngineRequestId(101)
    assert fake.disposition_input == ((0, 1, 1),)

    prepared = _prepare(session)
    assert prepared.relocation_id == EngineRelocationId(77, 51)
    assert fake.relocation_input == (101, 0, 1)
    ticket = session.submit_relocation(_execution(prepared))
    assert ticket.relocation_id == prepared.relocation_id
    assert fake.relocation_evidence[0] == (77, 51)

    publication = session.complete_relocation(
        ticket.relocation_id, EngineCompletionEvidence(9, 90, True)
    )
    assert publication.requests[0].request_id == EngineRequestId(101)
    assert publication.requests[0].view_version == 3
    assert len(publication.retirements) == 1
    session.confirm_relocation_publication(
        EngineRelocationPublicationEvidence(
            publication.relocation_id,
            True,
            retirement_evidence(publication.retirements),
        )
    )
    assert fake.relocation_confirmation == ((77, 51, 1), ((30, 1, 3000),))
    assert session._pending_relocations == {}
    for value in (views, updated, prepared, ticket, publication):
        _assert_no_internal_leases(value)
    session.close()


def test_native_session_token_view_and_disposition_round_trip(
    ffi_library: Path,
) -> None:
    config = _native_config(ffi_library)
    settings = SessionCreateSettings(
        ManagerCreateSettings(1, 1, 1, 8, 64),
        CacheSharingPolicy.REQUEST_PRIVATE,
    )
    registrations = (ArenaRegistration(0, 21, 10, 8, 1000),)
    with CtypesRuntimeSession.create(
        config, settings, registrations
    ) as session:
        request_id = EngineRequestId(101)
        session.acquire_requests((request_id,))
        append = session.prepare_append((EngineAppendIntent(request_id, 48),))
        session.submit_execution(
            _native_execution_evidence(append, session.arenas)
        )
        publication = session.complete_execution(
            append.batch_id, EngineCompletionEvidence(7, 1, True)
        )
        session.confirm_publication(
            session_dtos.EnginePublicationEvidence(
                publication.publication_id,
                True,
                retirement_evidence(publication.retirements),
            )
        )

        view = session.token_views_batch(
            (EngineTokenViewQuery(request_id, 0, 48),)
        )[0]
        assert len(view.placements) == 48
        assert all(
            item.disposition.kind is TokenDispositionKind.RETAINED
            for item in view.placements
        )
        updated = session.mark_token_dispositions_batch(
            (
                EngineTokenDispositionBatchItem(
                    request_id,
                    tuple(
                        EngineTokenDispositionUpdate(
                            0,
                            token_id,
                            TokenDisposition(
                                TokenDispositionKind.POLICY_EVICTED,
                                17,
                                1,
                                99,
                            ),
                        )
                        for token_id in range(8, 16)
                    ),
                ),
            )
        )
        assert updated[0].view_version == view.view_version + 1
        marked = session.token_views_batch(
            (EngineTokenViewQuery(request_id, 0, 48),)
        )[0]
        assert all(
            marked.placements[index].disposition.kind
            is TokenDispositionKind.POLICY_EVICTED
            for index in range(8, 16)
        )


def test_native_session_opaque_relocation_lifecycle(ffi_library: Path) -> None:
    config = _native_config(ffi_library)
    settings = SessionCreateSettings(
        ManagerCreateSettings(1, 4, 1, 16, 64),
        CacheSharingPolicy.REQUEST_PRIVATE,
    )
    registrations = (ArenaRegistration(0, 21, 10, 16, 1000),)
    with CtypesRuntimeSession.create(
        config, settings, registrations
    ) as session:
        request_id = EngineRequestId(101)
        session.acquire_requests((request_id,))
        append = session.prepare_append((EngineAppendIntent(request_id, 48),))
        session.submit_execution(
            _native_execution_evidence(append, session.arenas)
        )
        append_publication = session.complete_execution(
            append.batch_id, EngineCompletionEvidence(7, 1, True)
        )
        session.confirm_publication(
            session_dtos.EnginePublicationEvidence(
                append_publication.publication_id, True, ()
            )
        )
        session.mark_token_dispositions_batch(
            (
                EngineTokenDispositionBatchItem(
                    request_id,
                    tuple(
                        EngineTokenDispositionUpdate(
                            0,
                            token_id,
                            TokenDisposition(
                                TokenDispositionKind.POLICY_EVICTED,
                                17,
                                1,
                                99,
                            ),
                        )
                        for token_id in (
                            *range(8, 16),
                            *range(24, 32),
                            *range(40, 48),
                        )
                    ),
                ),
            )
        )
        prepared = session.prepare_relocation_batch(
            (
                EnginePrepareRelocationItem(
                    request_id, 0, RelocationPolicy(8, 2, 250, True)
                ),
            )
        )
        assert prepared.relocation_id.session_epoch == session.arenas[0].engine_epoch
        assert prepared.plans[0].request_id == request_id
        assert len(prepared.plans[0].moves) == 24
        session.submit_relocation(_execution(prepared))
        publication = session.complete_relocation(
            prepared.relocation_id, EngineCompletionEvidence(7, 2, True)
        )
        assert publication.requests[0].view_version == (
            prepared.plans[0].target_view_version
        )
        session.confirm_relocation_publication(
            EngineRelocationPublicationEvidence(
                publication.relocation_id,
                True,
                retirement_evidence(publication.retirements),
            )
        )
        compacted = session.token_views_batch(
            (EngineTokenViewQuery(request_id, 0, 48),)
        )[0]
        assert all(
            placement.location is None
            for placement in compacted.placements
            if placement.disposition.kind
            is TokenDispositionKind.POLICY_EVICTED
        )


def test_abort_is_exact_and_returns_relocation_to_local_ready_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionRelocationLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)
    prepared = _prepare(session)
    before = tuple(fake.invocations)

    with pytest.raises(ManagerError, match="ordering differs from plan"):
        session.abort_prepared_relocation(
            prepared.relocation_id,
            (EngineRelocationAbortEvidence(EngineRequestId(202), True),),
        )
    assert tuple(fake.invocations) == before

    session.abort_prepared_relocation(
        prepared.relocation_id,
        (EngineRelocationAbortEvidence(EngineRequestId(101), True),),
    )
    assert prepared.relocation_id not in session._pending_relocations
    session.close()


@pytest.mark.parametrize(
    "fault", (None, "quarantine_fail_stopped")
)
def test_relocation_quarantine_is_sticky_for_success_and_fail_stopped_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    fake = FakeSessionRelocationLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)
    prepared = _prepare(session)
    fake.fault = fault

    with pytest.raises(FailStopped, match="poisoned"):
        session.quarantine_relocation(prepared.relocation_id)
    assert prepared.relocation_id not in session._pending_relocations
    before = tuple(fake.invocations)
    with pytest.raises(FailStopped, match="poisoned"):
        session.acquire_requests((202,))
    assert tuple(fake.invocations) == before
    session.close()


def test_batch_quarantined_submit_fail_stops_and_discards_local_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionRelocationLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)
    prepared = _prepare(session)
    fake.fault = "submit_fail_stopped"

    with pytest.raises(FailStopped, match="poisoned"):
        session.submit_relocation(_execution(prepared))
    assert prepared.relocation_id not in session._pending_relocations
    before = tuple(fake.invocations)
    with pytest.raises(FailStopped, match="poisoned"):
        session.token_views_batch(
            (EngineTokenViewQuery(EngineRequestId(101), 0, 2),)
        )
    assert tuple(fake.invocations) == before
    session.close()


@pytest.mark.parametrize(
    "fault", ("token_view_preflight_count", "token_view_reserved")
)
def test_hostile_token_view_bounds_or_output_fail_stop_stickily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    fake = FakeSessionRelocationLibrary(fault)
    session = _open_session(tmp_path, monkeypatch, fake)

    with pytest.raises(FailStopped, match="poisoned"):
        session.token_views_batch(
            (EngineTokenViewQuery(EngineRequestId(101), 0, 2),)
        )
    before = tuple(fake.invocations)
    with pytest.raises(FailStopped, match="poisoned"):
        session.acquire_requests((202,))
    assert tuple(fake.invocations) == before
    session.close()
