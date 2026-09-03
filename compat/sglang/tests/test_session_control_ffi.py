from __future__ import annotations

from pathlib import Path

import pytest

from orbitkv_sglang.ffi.session_types import (
    EngineAppendIntent,
    EngineCompletionEvidence,
    EngineControlDisposition,
    EngineControlEvidence,
    EngineControlId,
    EngineControlKind,
    EngineControlOutcome,
    EngineControlPlanInfo,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixAttachItem,
    EnginePrefixEvictionPlan,
    EnginePrefixId,
    EnginePrefixLookup,
    EnginePrefixPublishItem,
    EnginePublishedPrefix,
    EnginePublicationEvidence,
    EngineReleaseEvidence,
    EngineRequestForkItem,
    EngineRequestId,
    EngineRetirement,
    EngineRetirementEvidence, retirement_evidence,
)
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    ManagerCreateSettings,
    ManagerError,
    PrefixSemanticKey,
    SessionCreateSettings,
)
from test_session_ffi import (
    CONTROL_SETTINGS,
    FakeSessionLibrary,
    _dto_page,
    _native_config,
    _native_execution_evidence,
    _open_session_with_settings,
    ffi_library,
)


__all__ = ["ffi_library"]


def _open_control_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeSessionLibrary
):
    return _open_session_with_settings(
        tmp_path, monkeypatch, fake, CONTROL_SETTINGS
    )


@pytest.mark.parametrize(
    "operation",
    (
        "prefix_lookup_batch",
        "prefix_publish_batch",
        "prefix_publish_release_batch",
        "prepare_prefix_attach",
        "prepare_request_fork",
        "prepare_prefix_evict",
        "abort_control",
        "commit_control",
        "read_control_plan",
        "cancel_pending_attach",
        "finalize_pending_attach_cancel",
        "confirm_control",
        "quarantine_control",
    ),
)
def test_request_private_session_rejects_every_sharing_control_before_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fake = FakeSessionLibrary()
    settings = SessionCreateSettings(
        CONTROL_SETTINGS.manager,
        CacheSharingPolicy.REQUEST_PRIVATE,
    )
    session = _open_session_with_settings(
        tmp_path, monkeypatch, fake, settings
    )
    before = tuple(fake.invocations)

    with pytest.raises(ManagerError, match="request-private cache sharing policy"):
        getattr(session, operation)(object())

    assert tuple(fake.invocations) == before
    assert session._pending_controls == {}
    assert session._pending_attach_cancels == {}
    session.close()


def test_fake_session_control_materialization_and_eviction_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_control_session(tmp_path, monkeypatch, fake)
    assert session.prefix_capacity == 4
    assert session.control_batch_capacity == 2
    assert session.prefix_eviction_batch_capacity == 2

    lookup = session.prefix_lookup_batch(
        (
            PrefixSemanticKey(b"a" * 32, b"b" * 32, 16),
            PrefixSemanticKey(b"c" * 32, b"d" * 32, 32),
        )
    )
    assert lookup[0] == EnginePrefixLookup(
        lookup[0].key, EnginePrefixId(77, 11), 2
    )
    assert lookup[1] == EnginePrefixLookup(lookup[1].key, None, 0)

    published = session.prefix_publish_batch(
        (EnginePrefixPublishItem(EngineRequestId(101), lookup[0].key),)
    )
    assert published == (
        EnginePublishedPrefix(EnginePrefixId(77, 20), lookup[0].key, 1),
    )

    attach = session.prepare_prefix_attach(
        (
            EnginePrefixAttachItem(
                EngineRequestId(202),
                EnginePrefixId(77, 11),
                lookup[0].key,
                2,
            ),
        )
    )
    assert attach == EngineControlId(77, 41)
    attach_info = session.commit_control(attach)
    assert attach_info == EngineControlPlanInfo(
        attach, EngineControlKind.MATERIALIZATION, 1, 2, 0, 0
    )
    materialized = session.read_control_plan(attach)
    assert materialized == EngineMaterializationPlan(
        attach,
        (
            EngineMaterializedRequest(
                EngineRequestId(202),
                9,
                32,
                2,
                materialized.requests[0].pages,
            ),
        ),
    )
    materialized_again = session.read_control_plan(attach)
    assert materialized_again is materialized
    materialized_outcome = session.confirm_control(
        EngineControlEvidence(
            attach,
            True,
            (),
        )
    )
    assert materialized_outcome == EngineControlOutcome(
        attach, EngineControlDisposition.MATERIALIZED
    )

    evict = session.prepare_prefix_evict((EnginePrefixId(77, 11),))
    assert evict == EngineControlId(77, 43)
    evict_info = session.commit_control(evict)
    assert evict_info == EngineControlPlanInfo(
        evict, EngineControlKind.PREFIX_EVICTION, 0, 0, 1, 1
    )
    eviction = session.read_control_plan(evict)
    assert eviction == EnginePrefixEvictionPlan(
        evict,
        (EnginePrefixId(77, 11),),
        (
            EngineRetirement(
                _dto_page(30, 11), 0, 3, 0, 3000, 0, 16, 9, 90
            ),
        ),
    )
    evicted_outcome = session.confirm_control(
        EngineControlEvidence(
            evict, True, retirement_evidence(eviction.retirements)
        )
    )
    assert evicted_outcome == EngineControlOutcome(
        evict, EngineControlDisposition.EVICTED
    )
    session.close()


def test_prefix_evict_rejects_items_beyond_control_capacity_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_control_session(tmp_path, monkeypatch, fake)
    control_id = session.prepare_prefix_evict(
        (EnginePrefixId(77, 11), EnginePrefixId(77, 12))
    )
    session.abort_control(control_id)
    before = tuple(fake.invocations)

    with pytest.raises(ManagerError, match="configured batch bound"):
        session.prepare_prefix_evict(
            tuple(EnginePrefixId(77, value) for value in range(11, 14))
        )

    assert tuple(fake.invocations) == before
    session.close()


def test_prefix_evict_capacity_does_not_inherit_request_batch_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    settings = SessionCreateSettings(
        ManagerCreateSettings(1, 3, 4, 6, 32),
        CacheSharingPolicy.SHARED_PREFIX,
    )
    session = _open_session_with_settings(
        tmp_path, monkeypatch, fake, settings
    )

    assert session.control_batch_capacity == 3
    assert session.prefix_eviction_batch_capacity == 3
    control_id = session.prepare_prefix_evict(
        tuple(EnginePrefixId(77, value) for value in range(11, 14))
    )
    assert control_id == EngineControlId(77, 43)
    session.abort_control(control_id)
    session.close()


def test_native_prefix_evict_capacity_one_drains_four_prefixes(
    ffi_library: Path,
) -> None:
    config = _native_config(ffi_library)
    settings = SessionCreateSettings(
        ManagerCreateSettings(1, 1, 4, 4, 16),
        CacheSharingPolicy.SHARED_PREFIX,
    )
    registrations = (ArenaRegistration(0, 21, 10, 4, 1000),)

    with CtypesRuntimeSession.create(
        config, settings, registrations
    ) as session:
        prefixes = []
        for offset in range(4):
            request_id = EngineRequestId(101 + offset)
            session.acquire_requests((request_id,))
            append = session.prepare_append(
                (EngineAppendIntent(request_id, 16),)
            )
            session.submit_execution(
                _native_execution_evidence(append, session.arenas)
            )
            publication = session.complete_execution(
                append.batch_id,
                EngineCompletionEvidence(7, offset + 1, True),
            )
            session.confirm_publication(
                EnginePublicationEvidence(
                    publication.publication_id, True, ()
                )
            )
            semantic = PrefixSemanticKey(
                b"n" * 32, bytes([offset + 1]) * 32, 16
            )
            released = session.prefix_publish_release_batch(
                (EnginePrefixPublishItem(request_id, semantic),)
            )
            prefixes.append(released.outputs[0].prefix_id)
            session.confirm_release(
                EngineReleaseEvidence(released.release_id, True, ())
            )

        assert session.control_batch_capacity == 1
        assert session.prefix_eviction_batch_capacity == 1
        assert session.stats().active_prefixes == 4
        with pytest.raises(ManagerError, match="configured batch bound"):
            session.prepare_prefix_evict(tuple(prefixes))

        for prefix_id in prefixes:
            control_id = session.prepare_prefix_evict((prefix_id,))
            session.commit_control(control_id)
            plan = session.read_control_plan(control_id)
            assert plan.prefix_ids == (prefix_id,)
            session.confirm_control(
                EngineControlEvidence(
                    control_id, True, retirement_evidence(plan.retirements)
                )
            )

        stats = session.stats()
        assert stats.active_requests == 0
        assert stats.active_prefixes == 0
        assert stats.evicted_prefixes == 0
        assert stats.pending_reclamations == 0
        assert stats.total_request_page_refs == 0
        assert stats.total_prefix_page_refs == 0
        assert stats.free_pages == 4


def test_control_quarantine_is_local_and_non_poisoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_control_session(tmp_path, monkeypatch, fake)
    control_id = session.prepare_request_fork(
        (EngineRequestForkItem(EngineRequestId(101), EngineRequestId(202)),)
    )
    plan_info = session.commit_control(control_id)
    assert plan_info.kind is EngineControlKind.MATERIALIZATION
    session.quarantine_control(control_id)
    assert control_id not in session._pending_controls
    assert fake.quarantined_controls[-1] == (77, 42)
    views = session.acquire_requests((101,))
    assert views[0].request_id == EngineRequestId(101)
    session.close()


def test_confirm_control_rejects_missing_or_mismatched_receipts_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_control_session(tmp_path, monkeypatch, fake)
    control_id = session.prepare_prefix_evict((EnginePrefixId(77, 11),))
    session.commit_control(control_id)
    plan = session.read_control_plan(control_id)
    assert isinstance(plan, EnginePrefixEvictionPlan)

    before = tuple(fake.invocations)
    with pytest.raises(ManagerError, match="cardinality differs from plan"):
        session.confirm_control(EngineControlEvidence(control_id, True, ()))
    assert tuple(fake.invocations) == before

    exact = retirement_evidence(plan.retirements)
    mismatched = EngineRetirementEvidence(
        exact[0].page,
        exact[0].backend_domain,
        exact[0].acknowledged,
        exact[0].backend_index + 1,
    )
    with pytest.raises(ManagerError, match="differs from plan"):
        session.confirm_control(
            EngineControlEvidence(control_id, True, (mismatched,))
        )
    assert tuple(fake.invocations) == before
    session.close()
