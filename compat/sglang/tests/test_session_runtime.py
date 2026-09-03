from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineBatchId,
    EngineBatchPlan,
    EngineBatchPublication,
    EngineBatchTicket,
    EnginePrefixId,
    EnginePrefixPublishItem,
    EnginePrefixPublishReleasePlan,
    EnginePublishedPrefixRelease,
    EnginePublicationId,
    EngineReleaseDisposition,
    EngineReleaseId,
    EngineReleaseOutcome,
    EngineReleasePlan,
    EngineReleasedRequest,
    EngineRequestId,
    EngineRequestView,
    EngineRetirement,
    EngineStepExecutionEvidence,
    EngineStepPlan,
    EngineStepPublication,
    ExecutionEvidence,
    retirement_evidence,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    CacheSharingPolicy,
    DETACHED_PREFIX_TRANSFER,
    DetachedBinding,
    FailStopped,
    ManagerError,
    ManagerStats,
    PageLease,
    PrefixSemanticKey,
)
from orbitkv_sglang.session_runtime import (  # noqa: E402
    ReleaseAckRetryPending,
    ReleaseRecyclePending,
    SessionMirrorUpdate,
    SessionRuntime,
    collective_mirror_cleanup,
    unbound_mirror_cleanup,
)


def _page(page_id: int, generation: int = 1) -> PageLease:
    return PageLease(7, 11, generation, page_id, 3)


def _detached(end: int = 16) -> DetachedBinding:
    return DetachedBinding(
        _page(5),
        PageLease(0, 0, 0, 0, 0),
        0,
        4,
        0,
        0,
        end,
        0,
        17,
        1,
        3,
        0,
    )


def _prefix_detached(end: int = 16) -> DetachedBinding:
    return replace(_detached(end), reason=DETACHED_PREFIX_TRANSFER)


def _retirement(end: int = 16) -> EngineRetirement:
    return EngineRetirement(_page(5), 0, 17, 0, 4, 0, end, 9, 1)


class FakeSession:
    def __init__(self, trace: list[Any] | None = None) -> None:
        self.arenas = (ArenaIdentity(7, 11, 3, 0, 17, 32, 16, 0, 1),)
        self.cache_sharing_policy = CacheSharingPolicy.SHARED_PREFIX
        self.prefix_capacity = 8
        self.control_batch_capacity = 8
        self.prefix_eviction_batch_capacity = 8
        self.trace = [] if trace is None else trace
        self.acquire_error: BaseException | None = None
        self.complete_error: BaseException | None = None
        self.confirm_publication_error: BaseException | None = None
        self.confirm_release_error: BaseException | None = None
        self.prefix_publish_release_error: BaseException | None = None
        self.prefix_publish_release_result: Any | None = None
        self.publication_detached: tuple[DetachedBinding, ...] = ()
        self.publication_retirements: tuple[EngineRetirement, ...] = ()
        self.release_detached: tuple[DetachedBinding, ...] = ()
        self.release_retirements: tuple[EngineRetirement, ...] = ()
        self.acquired: list[tuple[EngineRequestId, ...]] = []
        self.completions = []
        self.publication_confirmations = []
        self.release_confirmations = []
        self.release_outcome: Any = EngineReleaseDisposition.COMPLETED
        self.release_outcomes: list[Any] = []
        self.prefix_publish_release_calls: list[
            tuple[EnginePrefixPublishItem, ...]
        ] = []
        self._batch_requests: dict[EngineBatchId, tuple[EngineRequestId, ...]] = {}
        self._batch_boundaries: dict[
            EngineBatchId, tuple[int, ...]
        ] = {}
        self._batch_sequence = 1
        self._publication_sequence = 1
        self._release_sequence = 1
        self._prefix_sequence = 1
        self.close_calls = 0
        self.aborted = []
        self.quarantined_prepared = []
        self.quarantined_submitted = []
        self.arena_stats_value: tuple[Any, ...] = ()

    def acquire_requests(
        self, request_ids: tuple[EngineRequestId, ...]
    ) -> tuple[EngineRequestView, ...]:
        values = tuple(request_ids)
        self.trace.append(("acquire", values))
        self.acquired.append(values)
        if self.acquire_error is not None:
            error, self.acquire_error = self.acquire_error, None
            raise error
        return tuple(EngineRequestView(item, 1, 0, 0) for item in values)

    def prepare_append(self, intents: tuple[Any, ...]) -> EngineBatchPlan:
        values = tuple(intents)
        self.trace.append(
            ("prepare", tuple((item.request_id, item.target_boundary) for item in values))
        )
        batch_id = EngineBatchId(7, self._batch_sequence)
        self._batch_sequence += 1
        steps = tuple(
            EngineStepPlan(
                item.request_id,
                1,
                2,
                0,
                item.target_boundary,
                (),
                (),
                (),
                (),
            )
            for item in values
        )
        self._batch_boundaries[batch_id] = tuple(
            item.target_boundary for item in values
        )
        return EngineBatchPlan(batch_id, steps)

    def submit_execution(self, evidence: ExecutionEvidence) -> EngineBatchTicket:
        requests = tuple(item.request_id for item in evidence.steps)
        self.trace.append(("submit", evidence.batch_id, requests))
        ticket = EngineBatchTicket(evidence.batch_id, requests)
        self._batch_requests[evidence.batch_id] = requests
        return ticket

    def abort_prepared(self, batch_id, evidence) -> None:
        self.trace.append(("abort", batch_id))
        self.aborted.append((batch_id, tuple(evidence)))

    def quarantine_prepared(self, batch_id) -> None:
        self.trace.append(("quarantine-prepared", batch_id))
        self.quarantined_prepared.append(batch_id)

    def quarantine_submitted(self, batch_id) -> None:
        self.trace.append(("quarantine-submitted", batch_id))
        self.quarantined_submitted.append(batch_id)

    def complete_execution(
        self, batch_id: EngineBatchId, evidence: Any
    ) -> EngineBatchPublication:
        self.trace.append(
            (
                "complete",
                batch_id,
                evidence.completion_domain,
                evidence.completion_value,
                evidence.confirmed,
            )
        )
        self.completions.append(evidence)
        if self.complete_error is not None:
            raise self.complete_error
        request_ids = self._batch_requests[batch_id]
        boundaries = self._batch_boundaries[batch_id]
        publication_id = EnginePublicationId(7, self._publication_sequence)
        self._publication_sequence += 1
        return EngineBatchPublication(
            publication_id,
            batch_id,
            tuple(
                EngineStepPublication(
                    request_id,
                    2,
                    boundary,
                    (boundary + 15) // 16,
                    self.publication_detached,
                )
                for request_id, boundary in zip(
                    request_ids, boundaries, strict=True
                )
            ),
            self.publication_retirements,
        )

    def confirm_publication(self, evidence: Any) -> None:
        self.trace.append(("confirm-publication", evidence.publication_id))
        self.publication_confirmations.append(evidence)
        if self.confirm_publication_error is not None:
            raise self.confirm_publication_error

    def prepare_release(
        self, request_ids: tuple[EngineRequestId, ...]
    ) -> EngineReleasePlan:
        values = tuple(request_ids)
        self.trace.append(("prepare-release", values))
        release_id = EngineReleaseId(7, self._release_sequence)
        self._release_sequence += 1
        return EngineReleasePlan(
            release_id,
            tuple(
                EngineReleasedRequest(item, self.release_detached)
                for item in values
            ),
            self.release_retirements,
        )

    def prefix_publish_release_batch(
        self, items: tuple[EnginePrefixPublishItem, ...]
    ) -> EnginePrefixPublishReleasePlan:
        values = tuple(items)
        self.trace.append(("prefix-publish-release", values))
        self.prefix_publish_release_calls.append(values)
        if self.prefix_publish_release_error is not None:
            raise self.prefix_publish_release_error
        if self.prefix_publish_release_result is not None:
            return self.prefix_publish_release_result
        release_id = EngineReleaseId(7, self._release_sequence)
        self._release_sequence += 1
        detached = self.release_detached
        outputs = tuple(
            EnginePublishedPrefixRelease(
                EnginePrefixId(7, self._prefix_sequence + index),
                item.key,
                len(detached),
                EngineReleasedRequest(item.request_id, detached),
            )
            for index, item in enumerate(values)
        )
        self._prefix_sequence += len(values)
        return EnginePrefixPublishReleasePlan(release_id, outputs)

    def confirm_release(self, evidence: Any) -> EngineReleaseOutcome:
        self.trace.append(("confirm-release", evidence.release_id))
        self.release_confirmations.append(evidence)
        if self.confirm_release_error is not None:
            raise self.confirm_release_error
        configured = (
            self.release_outcomes.pop(0)
            if self.release_outcomes
            else self.release_outcome
        )
        if isinstance(configured, EngineReleaseDisposition):
            return EngineReleaseOutcome(evidence.release_id, configured)
        return configured

    def stats(self) -> ManagerStats:
        self.trace.append("stats")
        return ManagerStats(*(0 for _ in range(17)))

    def arena_stats(self) -> tuple[Any, ...]:
        self.trace.append("arena-stats")
        return self.arena_stats_value

    def close(self) -> None:
        self.trace.append("close")
        self.close_calls += 1


class Event:
    def __init__(
        self,
        trace: list[Any],
        *,
        ready: bool = False,
        query_error: BaseException | None = None,
        wait_error: BaseException | None = None,
    ) -> None:
        self.trace = trace
        self.ready = ready
        self.query_error = query_error
        self.wait_error = wait_error

    def query(self) -> bool:
        self.trace.append("query")
        if self.query_error is not None:
            raise self.query_error
        return self.ready

    def synchronize(self) -> None:
        self.trace.append("synchronize")
        if self.wait_error is not None:
            raise self.wait_error
        self.ready = True


class FatalSignal(BaseException):
    pass


def _cleanup(
    trace: list[Any], *, confirmed: bool = True
):
    def callback(updates, retirements):
        trace.append(("cleanup", updates, retirements))
        return confirmed

    return callback


def _submit(runtime: SessionRuntime, *items: tuple[Any, int]):
    plan = runtime.prepare(items)
    evidence = ExecutionEvidence(
        plan.batch_id,
        tuple(
            EngineStepExecutionEvidence(step.request_id, (), ())
            for step in plan.steps
        ),
    )
    return runtime.submit(evidence)


def _release(runtime: SessionRuntime, *keys: Any) -> None:
    plan = runtime.prepare_release(keys)
    runtime.confirm_release(plan)


def _acquire_bound(
    runtime: SessionRuntime, assignments: tuple[tuple[Any, int], ...]
) -> tuple[EngineRequestView, ...]:
    """Create native identities before claiming their engine mirror rows."""

    values = tuple(assignments)
    views = runtime.acquire_unbound(tuple(key for key, _row in values))
    runtime.bind_request_rows(values)
    return views


def _semantic(tag: int, boundary: int) -> PrefixSemanticKey:
    return PrefixSemanticKey(bytes([tag]) * 32, bytes([tag + 1]) * 32, boundary)


@pytest.mark.parametrize("policy", (None, 1, True, "shared_prefix"))
def test_runtime_rejects_missing_or_noncanonical_cache_sharing_policy(
    policy: Any,
) -> None:
    session = FakeSession()
    if policy is None:
        del session.cache_sharing_policy
    else:
        session.cache_sharing_policy = policy

    with pytest.raises(ManagerError, match="cache sharing policy"):
        SessionRuntime(session)


@pytest.mark.parametrize(
    "policy",
    (CacheSharingPolicy.REQUEST_PRIVATE, CacheSharingPolicy.SHARED_PREFIX),
)
def test_runtime_exposes_native_cache_sharing_policy(
    policy: CacheSharingPolicy,
) -> None:
    session = FakeSession()
    session.cache_sharing_policy = policy

    runtime = SessionRuntime(session)

    assert runtime.cache_sharing_policy is policy
    runtime.close()


def test_two_phase_cleanup_binding_blocks_row_binding_and_is_one_shot() -> None:
    session = FakeSession()
    runtime = SessionRuntime(
        session, mirror_cleanup=unbound_mirror_cleanup
    )

    view = runtime.acquire_unbound((("str", "a"),))[0]
    with pytest.raises(ManagerError, match="must be bound before row binding"):
        runtime.bind_request_rows(((("str", "a"), 1),))
    assert session.acquired == [(EngineRequestId(1),)]
    assert runtime.binding_for(("str", "a")).request_row is None

    callback = _cleanup(session.trace)
    runtime.bind_mirror_cleanup(callback)
    assert view.request_id == 1
    runtime.bind_request_rows(((("str", "a"), 1),))
    assert runtime.view_for(("str", "a")) is view
    with pytest.raises(ManagerError, match="already bound"):
        runtime.bind_mirror_cleanup(callback)

    _release(runtime, ("str", "a"))
    runtime.close()


def test_request_identity_and_row_mirrors_are_monotonic_until_confirm() -> None:
    session = FakeSession()
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )
    first = ("str", "first")
    second = ("str", "second")

    assert _acquire_bound(runtime, ((first, 1),))[0].request_id == 1
    binding = runtime.binding_for(first)
    assert (binding.key, binding.request_id, binding.request_row) == (first, 1, 1)
    assert runtime.views_for((first,)) == (
        EngineRequestView(EngineRequestId(1), 1, 0, 0),
    )
    plan = runtime.prepare_release((first,))
    runtime.acquire_unbound((second,))
    with pytest.raises(ManagerError, match="already owned"):
        runtime.bind_request_rows(((second, 1),))
    assert runtime.binding_for(first) is binding
    assert runtime.binding_for(second).request_row is None

    runtime.confirm_release(plan)
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(first)
    runtime.bind_request_rows(((second, 1),))
    assert runtime.view_for(second).request_id == 2

    _release(runtime, second)
    runtime.close()
    assert session.acquired == [(EngineRequestId(1),), (EngineRequestId(2),)]


def test_safe_acquire_error_burns_request_id_without_fail_stop() -> None:
    session = FakeSession()
    session.acquire_error = ManagerError("capacity")
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )

    with pytest.raises(ManagerError, match="capacity"):
        _acquire_bound(runtime, ((("str", "first"), 1),))
    assert runtime.failure_reason is None
    view = _acquire_bound(runtime, ((("str", "second"), 1),))[0]
    assert view.request_id == 2

    _release(runtime, ("str", "second"))
    runtime.close()


def test_acquire_unbound_installs_identity_and_view_without_row() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)
    first, second = ("str", "first"), ("str", "second")

    views = runtime.acquire_unbound((first, second))

    assert tuple(view.request_id for view in views) == (
        EngineRequestId(1),
        EngineRequestId(2),
    )
    assert runtime.binding_for(first).request_row is None
    assert runtime.binding_for(second).request_row is None
    assert runtime.view_for(first) is views[0]
    assert runtime.views_for((second, first)) == (views[1], views[0])
    assert runtime._row_owners == {}

    runtime.fail_stop("test teardown")


def test_acquire_unbound_rejects_duplicate_and_live_keys_before_native() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)
    key = ("str", "request")

    with pytest.raises(ManagerError, match="duplicate request keys"):
        runtime.acquire_unbound((key, key))
    assert session.acquired == []

    runtime.acquire_unbound((key,))
    with pytest.raises(ManagerError, match="already acquired"):
        runtime.acquire_unbound((key,))
    assert session.acquired == [(EngineRequestId(1),)]

    _release(runtime, key)
    runtime.close()


def test_acquire_unbound_rejects_nonempty_native_initial_view() -> None:
    class InvalidInitialSession(FakeSession):
        def acquire_requests(self, request_ids):
            values = super().acquire_requests(request_ids)
            return tuple(
                EngineRequestView(item.request_id, 1, 1, 1)
                for item in values
            )

    session = InvalidInitialSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)

    with pytest.raises(FailStopped, match="initial state"):
        runtime.acquire_unbound((("str", "request"),))

    assert runtime.failure_reason is not None
    assert session.close_calls == 1
    assert runtime._bindings == {}


def test_bind_request_rows_is_collective_and_once_only() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    first, second = ("str", "first"), ("str", "second")
    runtime.acquire_unbound((first, second))

    runtime.bind_request_rows(((first, 3), (second, 4)))

    assert runtime.binding_for(first).request_row == 3
    assert runtime.binding_for(second).request_row == 4
    assert runtime._row_owners == {3: first, 4: second}
    with pytest.raises(ManagerError, match="already bound"):
        runtime.bind_request_rows(((first, 3),))
    assert runtime.binding_for(first).request_row == 3
    assert runtime.binding_for(second).request_row == 4

    _release(runtime, first, second)
    runtime.close()


def test_bind_request_rows_validation_never_partially_mutates() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    first, second, owner = (
        ("str", "first"),
        ("str", "second"),
        ("str", "owner"),
    )
    runtime.acquire_unbound((first, second, owner))
    runtime.bind_request_rows(((owner, 9),))

    invalid_batches = (
        (((first, 1), (second, 0)), "positive integer"),
        (((first, 1), (second, 1)), "aliases"),
        (((first, 1), (first, 2)), "duplicate request keys"),
        (((first, 1), (second, 9)), "already owned"),
    )
    for assignments, message in invalid_batches:
        with pytest.raises(ManagerError, match=message):
            runtime.bind_request_rows(assignments)
        assert runtime.binding_for(first).request_row is None
        assert runtime.binding_for(second).request_row is None
        assert runtime.binding_for(owner).request_row == 9
        assert runtime._row_owners == {9: owner}

    runtime.bind_request_rows(((first, 1), (second, 2)))
    _release(runtime, first, second, owner)
    runtime.close()


def test_explicit_acquire_then_bind_installs_identity_and_row() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = ("str", "request")

    view = runtime.acquire_unbound((key,))[0]
    assert runtime.binding_for(key).request_row is None
    runtime.bind_request_rows(((key, 7),))

    assert view.request_id == EngineRequestId(1)
    assert runtime.binding_for(key).request_row == 7
    assert runtime.view_for(key) is view
    assert runtime._row_owners == {7: key}
    assert session.acquired == [(EngineRequestId(1),)]

    _release(runtime, key)
    runtime.close()


def test_prepare_rejects_unbound_request_before_native_prepare() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = ("str", "request")
    runtime.acquire_unbound((key,))

    with pytest.raises(ManagerError, match="requires a bound ReqToToken row"):
        runtime.prepare(((key, 16),))

    assert not any(
        isinstance(item, tuple) and item[0] == "prepare"
        for item in session.trace
    )
    assert runtime.failure_reason is None
    runtime.bind_request_rows(((key, 1),))
    _release(runtime, key)
    runtime.close()


def test_native_execution_ticket_cannot_name_unbound_request() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)
    key = ("str", "request")
    view = runtime.acquire_unbound((key,))[0]
    evidence = ExecutionEvidence(
        EngineBatchId(7, 99),
        (EngineStepExecutionEvidence(view.request_id, (), ()),),
    )

    with pytest.raises(FailStopped, match="unbound request"):
        runtime.submit(evidence)

    assert runtime.failure_reason is not None
    assert session.close_calls == 1


def test_unbound_release_needs_no_cleanup_callback_or_mirror_update() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)
    key = ("str", "request")
    runtime.acquire_unbound((key,))

    plan = runtime.prepare_release((key,))
    runtime.confirm_release(plan)

    assert [
        item if isinstance(item, str) else item[0] for item in session.trace
    ] == ["acquire", "prepare-release", "confirm-release"]
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(key)
    runtime.close()


def test_unbound_release_prevents_late_row_binding() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = ("str", "request")
    runtime.acquire_unbound((key,))
    plan = runtime.prepare_release((key,))

    with pytest.raises(ManagerError, match="pending release group"):
        runtime.bind_request_rows(((key, 1),))

    assert runtime.binding_for(key).request_row is None
    runtime.confirm_release(plan)
    runtime.close()


@pytest.mark.parametrize("physical_state", ("detached", "retirement"))
def test_all_unbound_release_with_physical_state_fail_stops(
    physical_state: str,
) -> None:
    session = FakeSession()
    if physical_state == "detached":
        session.release_detached = (_detached(),)
    else:
        session.release_retirements = (_retirement(),)
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)
    key = ("str", "request")
    runtime.acquire_unbound((key,))
    plan = runtime.prepare_release((key,))

    with pytest.raises(FailStopped):
        runtime.confirm_release(plan)

    assert runtime.failure_reason is not None
    assert session.release_confirmations == []
    assert session.close_calls == 1


def test_native_acquire_local_install_failure_fail_stops() -> None:
    class HostileKey:
        armed = False

        def __hash__(self) -> int:
            if self.armed:
                raise RuntimeError("hostile post-native hash")
            return 17

    key = HostileKey()

    class ArmingSession(FakeSession):
        def acquire_requests(self, request_ids):
            views = super().acquire_requests(request_ids)
            key.armed = True
            return views

    session = ArmingSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)

    with pytest.raises(FailStopped, match="installed locally"):
        runtime.acquire_unbound((key,))

    assert session.acquired == [(EngineRequestId(1),)]
    assert session.close_calls == 1
    assert runtime.failure_reason is not None
    assert runtime._bindings == {}
    assert runtime._views == {}
    assert runtime._request_keys == {}


def test_clean_close_rejects_unbound_live_request() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=unbound_mirror_cleanup)
    key = ("str", "request")
    runtime.acquire_unbound((key,))

    with pytest.raises(ManagerError, match="live requests"):
        runtime.close()

    assert session.close_calls == 0
    plan = runtime.prepare_release((key,))
    runtime.confirm_release(plan)
    runtime.close()
    assert session.close_calls == 1


def test_poll_queries_before_native_completion_then_cleans_and_confirms() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.publication_detached = (_detached(),)
    session.publication_retirements = (_retirement(),)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 3),))
    ticket = _submit(runtime, (key, 16))
    event = Event(trace)
    runtime.register_event(ticket, event, 9)

    assert runtime.poll() == ()
    assert runtime.completion_evidence() == {
        "event_backend": "cuda_event_current_forward_stream",
        "pending_events": 1,
        "completion_high_water": [],
    }
    assert session.completions == []
    event.ready = True
    publication = runtime.poll()[0]
    assert runtime.view_for(key) == EngineRequestView(
        EngineRequestId(1), 2, 16, 1
    )

    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names[-4:] == [
        "query",
        "complete",
        "cleanup",
        "confirm-publication",
    ]
    cleanup_call = trace[-2]
    update = cleanup_call[1][0]
    assert update == SessionMirrorUpdate(
        key, EngineRequestId(1), 3, 16, (_detached(),), False
    )
    assert cleanup_call[2] == (_retirement(),)
    confirmation = session.publication_confirmations[0]
    assert confirmation.publication_id == publication.publication_id
    assert confirmation.mirror_cleanup_confirmed is True
    assert tuple(item.page for item in confirmation.reclamation_receipts) == (
        _page(5),
    )
    assert runtime.completion_evidence() == {
        "event_backend": "cuda_event_current_forward_stream",
        "pending_events": 0,
        "completion_high_water": [{"domain": 9, "value": 1}],
    }

    _release(runtime, key)
    runtime.close()


def test_failure_wrappers_delegate_without_copying_plan_phase() -> None:
    session = FakeSession()
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    plan = runtime.prepare(((key, 1),))
    runtime.abort_prepared(plan)
    assert session.aborted[0][0] == plan.batch_id
    assert tuple(item.request_id for item in session.aborted[0][1]) == (
        EngineRequestId(1),
    )
    assert all(item.backend_unobserved is True for item in session.aborted[0][1])

    plan = runtime.prepare(((key, 1),))
    with pytest.raises(FailStopped, match="prepared execution was quarantined"):
        runtime.quarantine_prepared(plan.batch_id)
    assert session.quarantined_prepared == [plan.batch_id]
    assert session.close_calls == 1


def test_submitted_quarantine_validates_issued_ticket_then_fail_stops() -> None:
    session = FakeSession()
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 1))

    with pytest.raises(ManagerError, match="issued batch ticket"):
        runtime.quarantine_submitted(
            EngineBatchTicket(ticket.batch_id, ticket.requests)
        )
    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        runtime.quarantine_submitted(ticket)
    assert session.quarantined_submitted == [ticket.batch_id]
    assert session.close_calls == 1


def test_poll_queries_every_event_before_completing_ready_groups() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    first, second = ("str", "first"), ("str", "second")
    _acquire_bound(runtime, ((first, 1), (second, 2)))
    first_ticket = _submit(runtime, (first, 1))
    second_ticket = _submit(runtime, (second, 1))
    runtime.register_event(first_ticket, Event(trace, ready=True), 4)
    runtime.register_event(second_ticket, Event(trace, ready=True), 5)

    runtime.poll()
    names = [item if isinstance(item, str) else item[0] for item in trace]
    first_complete = names.index("complete")
    assert names[first_complete - 2 : first_complete] == ["query", "query"]
    assert [item.completion_value for item in session.completions] == [1, 2]
    assert runtime.completion_evidence() == {
        "event_backend": "cuda_event_current_forward_stream",
        "pending_events": 0,
        "completion_high_water": [
            {"domain": 4, "value": 1},
            {"domain": 5, "value": 2},
        ],
    }

    _release(runtime, first, second)
    runtime.close()


def test_prepare_release_waits_and_publishes_before_native_release() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.release_detached = (_detached(13),)
    session.release_retirements = (_retirement(13),)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 2),))
    ticket = _submit(runtime, (key, 13))
    runtime.register_event(ticket, Event(trace), 8)

    plan = runtime.prepare_release((key,))
    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names[-5:] == [
        "synchronize",
        "complete",
        "cleanup",
        "confirm-publication",
        "prepare-release",
    ]
    runtime.confirm_release(plan)
    release_cleanup = trace[-2]
    update = release_cleanup[1][0]
    assert (update.key, update.request_row, update.boundary, update.releasing) == (
        key,
        2,
        13,
        True,
    )
    assert trace[-1][0] == "confirm-release"
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(key)
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.view_for(key)
    runtime.close()


def test_prefix_publish_release_waits_registers_and_reuses_confirm_release() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.release_detached = (_prefix_detached(),)
    session.release_outcomes = [
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.COMPLETED,
    ]
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    semantic = _semantic(31, 16)
    _acquire_bound(runtime, ((key, 2),))
    ticket = _submit(runtime, (key, 16))
    runtime.register_event(ticket, Event(trace), 8)

    published = runtime.prepare_prefix_publish_release(((key, semantic),))

    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names[-5:] == [
        "synchronize",
        "complete",
        "cleanup",
        "confirm-publication",
        "prefix-publish-release",
    ]
    assert published.outputs == (
        EnginePublishedPrefixRelease(
            EnginePrefixId(7, 1),
            semantic,
            1,
            EngineReleasedRequest(EngineRequestId(1), (_prefix_detached(),)),
        ),
    )
    assert session.prefix_publish_release_calls == [
        (EnginePrefixPublishItem(EngineRequestId(1), semantic),)
    ]
    assert runtime._prefixes[semantic].prefix_id == published.outputs[0].prefix_id
    release = runtime.pending_release((key,))
    assert release == EngineReleasePlan(
        published.release_id, (published.outputs[0].release,), ()
    )
    assert release is runtime._releases[published.release_id].plan

    assert release is not None
    with pytest.raises(ReleaseRecyclePending) as pending:
        runtime.confirm_release(release)
    assert pending.value.plan is release
    assert runtime.pending_release((key,)) is release
    assert semantic in runtime._prefixes
    assert len(
        [
            item
            for item in trace
            if isinstance(item, tuple) and item[0] == "cleanup"
        ]
    ) == 2

    runtime.confirm_release(release)
    assert semantic in runtime._prefixes
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(key)
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.pending_release((key,))
    runtime.fail_stop("teardown resident prefix")


def test_ordinary_release_is_not_pending_until_cleanup_ack_commits() -> None:
    session = FakeSession()
    session.release_outcomes = [
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.COMPLETED,
    ]
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = "request"
    _acquire_bound(runtime, ((key, 1),))

    release = runtime.prepare_release((key,))
    assert runtime.pending_release((key,)) is None
    with pytest.raises(ReleaseRecyclePending):
        runtime.confirm_release(release)
    assert runtime.pending_release((key,)) is release

    runtime.confirm_release(release)
    runtime.close()


@pytest.mark.parametrize(
    ("items", "message"),
    (
        ((), "must be nonempty"),
        ((("request", object()),), "PrefixSemanticKey"),
    ),
)
def test_prefix_publish_release_preflight_rejection_is_atomic(
    items: tuple[Any, ...], message: str
) -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = "request"
    _acquire_bound(runtime, ((key, 1),))

    with pytest.raises(ManagerError, match=message):
        runtime.prepare_prefix_publish_release(items)

    assert session.prefix_publish_release_calls == []
    assert runtime._prefixes == {}
    assert runtime.pending_release((key,)) is None
    _release(runtime, key)
    runtime.close()


def test_prefix_publish_release_rejects_duplicate_keys_and_semantics_precommit(
) -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    first, second = "first", "second"
    _acquire_bound(runtime, ((first, 1), (second, 2)))
    one = _semantic(41, 0)
    two = _semantic(42, 0)

    with pytest.raises(ManagerError, match="duplicate request keys"):
        runtime.prepare_prefix_publish_release(((first, one), (first, two)))
    with pytest.raises(ManagerError, match="duplicate semantic keys"):
        runtime.prepare_prefix_publish_release(((first, one), (second, one)))

    assert session.prefix_publish_release_calls == []
    assert runtime._prefixes == {}
    _release(runtime, first, second)
    runtime.close()


def test_prefix_publish_release_rejects_pending_control_and_release_precommit() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    first, second = "first", "second"
    _acquire_bound(runtime, ((first, 1), (second, 2)))
    runtime._request_occupancy[first] = object()

    with pytest.raises(ManagerError, match="pending control group|busy"):
        runtime.prepare_prefix_publish_release(((first, _semantic(43, 0)),))
    runtime._request_occupancy.clear()
    release = runtime.prepare_release((second,))
    with pytest.raises(ManagerError, match="pending release group|busy"):
        runtime.prepare_prefix_publish_release(((second, _semantic(44, 0)),))

    assert session.prefix_publish_release_calls == []
    runtime.confirm_release(release)
    _release(runtime, first)
    runtime.close()


def test_prefix_publish_release_manager_error_is_precommit_and_retryable() -> None:
    session = FakeSession()
    session.prefix_publish_release_error = ManagerError("duplicate prefix")
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = "request"
    semantic = _semantic(51, 0)
    _acquire_bound(runtime, ((key, 1),))

    with pytest.raises(ManagerError, match="duplicate prefix"):
        runtime.prepare_prefix_publish_release(((key, semantic),))

    assert runtime.failure_reason is None
    assert runtime._prefixes == {}
    assert runtime.pending_release((key,)) is None
    session.prefix_publish_release_error = None
    published = runtime.prepare_prefix_publish_release(((key, semantic),))
    release = runtime.pending_release((key,))
    assert release is not None
    assert release.release_id == published.release_id
    runtime.confirm_release(release)
    runtime.fail_stop("teardown resident prefix")


def test_prefix_publish_release_rejects_existing_semantic_before_native() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    first, second = "first", "second"
    semantic = _semantic(53, 0)
    _acquire_bound(runtime, ((first, 1), (second, 2)))
    published = runtime.prepare_prefix_publish_release(((first, semantic),))

    with pytest.raises(ManagerError, match="already published"):
        runtime.prepare_prefix_publish_release(((second, semantic),))

    assert len(session.prefix_publish_release_calls) == 1
    release = runtime.pending_release((first,))
    assert release is not None and release.release_id == published.release_id
    runtime.confirm_release(release)
    _release(runtime, second)
    runtime.fail_stop("teardown resident prefix")


def test_prefix_publish_release_rejects_boundary_mismatch_before_native() -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = "request"
    _acquire_bound(runtime, ((key, 1),))

    with pytest.raises(ManagerError, match="boundary differs"):
        runtime.prepare_prefix_publish_release(((key, _semantic(54, 16)),))

    assert session.prefix_publish_release_calls == []
    assert runtime.failure_reason is None
    _release(runtime, key)
    runtime.close()


@pytest.mark.parametrize(
    "view",
    (
        EngineRequestView(EngineRequestId(1), True, 0, 0),
        EngineRequestView(EngineRequestId(1), -1, 0, 0),
        EngineRequestView(EngineRequestId(1), 1 << 64, 0, 0),
        EngineRequestView(EngineRequestId(1), 1, 0, 1 << 32),
    ),
)
def test_prefix_publish_release_rejects_malformed_view_before_native(
    view: EngineRequestView,
) -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    key = "request"
    _acquire_bound(runtime, ((key, 1),))
    runtime._views[key] = view

    with pytest.raises(FailStopped, match="view mirror is invalid"):
        runtime.prepare_prefix_publish_release(((key, _semantic(55, 0)),))

    assert session.prefix_publish_release_calls == []
    assert session.close_calls == 1


@pytest.mark.parametrize("mutation", ("quarantine", "row-owner"))
def test_prefix_publish_release_postcommit_liveness_change_fail_stops(
    mutation: str,
) -> None:
    session = FakeSession()
    runtime: SessionRuntime
    key = "request"
    semantic = _semantic(56, 0)

    def mutate(items):
        values = tuple(items)
        if mutation == "quarantine":
            runtime._quarantined_requests.add(key)
        else:
            runtime._row_owners[1] = key
        return EnginePrefixPublishReleasePlan(
            EngineReleaseId(7, 1),
            (
                EnginePublishedPrefixRelease(
                    EnginePrefixId(7, 1),
                    semantic,
                    0,
                    EngineReleasedRequest(values[0].request_id, ()),
                ),
            ),
        )

    session.prefix_publish_release_batch = mutate
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    if mutation == "quarantine":
        _acquire_bound(runtime, ((key, 1),))
    else:
        runtime.acquire_unbound((key,))

    with pytest.raises(FailStopped, match="participants changed"):
        runtime.prepare_prefix_publish_release(((key, semantic),))

    assert runtime.failure_reason is not None
    assert session.close_calls == 1
    assert runtime._prefixes == {}


def test_prefix_publish_release_does_not_install_prefix_before_native_returns() -> None:
    session = FakeSession()
    key = "request"
    semantic = _semantic(52, 0)
    runtime: SessionRuntime

    def observed(items):
        assert runtime._prefixes == {}
        values = tuple(items)
        return EnginePrefixPublishReleasePlan(
            EngineReleaseId(7, 1),
            (
                EnginePublishedPrefixRelease(
                    EnginePrefixId(7, 1),
                    semantic,
                    0,
                    EngineReleasedRequest(values[0].request_id, ()),
                ),
            ),
        )

    session.prefix_publish_release_batch = observed
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    _acquire_bound(runtime, ((key, 1),))

    published = runtime.prepare_prefix_publish_release(((key, semantic),))

    assert runtime._prefixes[semantic].prefix_id == published.outputs[0].prefix_id
    release = runtime.pending_release((key,))
    assert release is not None
    runtime.confirm_release(release)
    runtime.fail_stop("teardown resident prefix")


@pytest.mark.parametrize(
    "corruption",
    (
        "type",
        "outputs-container",
        "cardinality",
        "release-epoch-type",
        "request",
        "request-type",
        "key",
        "resident-count",
        "detached-page",
        "duplicate-prefix",
        "prefix-epoch-type",
    ),
)
def test_prefix_publish_release_malformed_postcommit_output_fail_stops(
    corruption: str,
) -> None:
    session = FakeSession()
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(session.trace))
    first, second = "first", "second"
    semantics = (_semantic(61, 16), _semantic(71, 16))
    _acquire_bound(runtime, ((first, 1), (second, 2)))
    runtime._views[first] = EngineRequestView(EngineRequestId(1), 2, 16, 1)
    runtime._views[second] = EngineRequestView(EngineRequestId(2), 2, 16, 1)
    valid = EnginePrefixPublishReleasePlan(
        EngineReleaseId(7, 1),
        (
            EnginePublishedPrefixRelease(
                EnginePrefixId(7, 1),
                semantics[0],
                1,
                EngineReleasedRequest(EngineRequestId(1), (_prefix_detached(),)),
            ),
            EnginePublishedPrefixRelease(
                EnginePrefixId(7, 2),
                semantics[1],
                1,
                EngineReleasedRequest(EngineRequestId(2), (_prefix_detached(),)),
            ),
        ),
    )
    if corruption == "type":
        session.prefix_publish_release_result = object()
    elif corruption == "outputs-container":
        session.prefix_publish_release_result = EnginePrefixPublishReleasePlan(
            valid.release_id, list(valid.outputs)
        )
    elif corruption == "cardinality":
        session.prefix_publish_release_result = replace(
            valid, outputs=valid.outputs[:1]
        )
    elif corruption == "release-epoch-type":
        session.prefix_publish_release_result = replace(
            valid, release_id=EngineReleaseId(True, 1)
        )
    elif corruption == "request":
        changed = replace(
            valid.outputs[0].release, request_id=EngineRequestId(99)
        )
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                replace(valid.outputs[0], release=changed),
                valid.outputs[1],
            ),
        )
    elif corruption == "request-type":
        changed = replace(valid.outputs[0].release, request_id=True)
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                replace(valid.outputs[0], release=changed),
                valid.outputs[1],
            ),
        )
    elif corruption == "key":
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                replace(valid.outputs[0], key=semantics[1]),
                valid.outputs[1],
            ),
        )
    elif corruption == "resident-count":
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                replace(valid.outputs[0], resident_count=2),
                valid.outputs[1],
            ),
        )
    elif corruption == "detached-page":
        changed = replace(
            valid.outputs[0].release,
            detached=(replace(_prefix_detached(), old=_page(99)),),
        )
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                replace(valid.outputs[0], release=changed),
                valid.outputs[1],
            ),
        )
    elif corruption == "duplicate-prefix":
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                valid.outputs[0],
                replace(valid.outputs[1], prefix_id=EnginePrefixId(7, 1)),
            ),
        )
    else:
        session.prefix_publish_release_result = replace(
            valid,
            outputs=(
                replace(
                    valid.outputs[0],
                    prefix_id=EnginePrefixId(True, 1),
                ),
                valid.outputs[1],
            ),
        )

    with pytest.raises(FailStopped, match="malformed postcommit output"):
        runtime.prepare_prefix_publish_release(
            ((first, semantics[0]), (second, semantics[1]))
        )

    assert runtime.failure_reason is not None
    assert session.close_calls == 1
    assert runtime._prefixes == {}
    assert runtime._releases == {}


def test_prefix_publish_release_public_dtos_have_no_manager_leases() -> None:
    forbidden = {
        "request",
        "snapshot",
        "prefix",
        "reclamation",
        "step",
        "submission",
    }
    for dto in (
        EnginePrefixPublishReleasePlan,
        EnginePublishedPrefixRelease,
    ):
        names = {field.name for field in fields(dto)}
        assert names.isdisjoint(forbidden)


@pytest.mark.parametrize("atomic", (False, True), ids=("ordinary", "atomic"))
def test_pre_ack_manager_error_retries_full_evidence_without_recleanup(
    atomic: bool,
) -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    if atomic:
        session.release_detached = (_prefix_detached(),)
    else:
        session.release_retirements = (_retirement(),)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 2),))
    if atomic:
        ticket = _submit(runtime, (key, 16))
        runtime.register_event(ticket, Event(trace, ready=True), 1)
        runtime.poll()
        semantic = _semantic(63, 16)
        published = runtime.prepare_prefix_publish_release(((key, semantic),))
        plan = runtime.pending_release((key,))
        assert plan is not None
        assert plan.release_id == published.release_id
    else:
        plan = runtime.prepare_release((key,))
        assert runtime.pending_release((key,)) is None

    session.confirm_release_error = ManagerError("ACK precommit rejection")
    with pytest.raises(
        ReleaseAckRetryPending, match="native ACK must be retried"
    ) as pending:
        runtime.confirm_release(plan)

    group = runtime._releases[plan.release_id]
    assert pending.value.plan is plan
    assert isinstance(pending.value.__cause__, ManagerError)
    assert str(pending.value.__cause__) == "ACK precommit rejection"
    assert group.plan is plan
    assert group.cleanup_confirmed is True
    assert group.ack_committed is False
    assert runtime.pending_release((key,)) is plan
    assert runtime.binding_for(key).request_row == 2
    assert runtime.failure_reason is None
    release_cleanup_calls = [
        item
        for item in trace
        if isinstance(item, tuple) and item[0] == "cleanup"
        and any(update.releasing for update in item[1])
    ]
    assert len(release_cleanup_calls) == 1
    assert len(session.release_confirmations) == 1
    first = session.release_confirmations[0]
    assert first.mirror_cleanup_confirmed is True
    assert first.reclamation_receipts == retirement_evidence(plan.retirements)
    assert first.acknowledged_retry is False

    session.confirm_release_error = None
    runtime.confirm_release(plan)

    assert len(
        [
            item
            for item in trace
            if isinstance(item, tuple) and item[0] == "cleanup"
            and any(update.releasing for update in item[1])
        ]
    ) == 1
    assert len(session.release_confirmations) == 2
    assert session.release_confirmations[1] == first
    assert plan.release_id not in runtime._releases
    if atomic:
        assert semantic in runtime._prefixes
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(key)
    runtime.close()


def test_release_pending_outcome_retries_once_with_id_only_evidence() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.release_retirements = (_retirement(),)
    session.release_outcomes = [
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.COMPLETED,
    ]
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 2),))
    plan = runtime.prepare_release((key,))

    runtime.confirm_release(plan)

    cleanup_calls = [
        item
        for item in trace
        if isinstance(item, tuple) and item[0] == "cleanup"
    ]
    assert len(cleanup_calls) == 1
    assert [
        item if isinstance(item, str) else item[0] for item in trace[-3:]
    ] == ["cleanup", "confirm-release", "confirm-release"]
    assert len(session.release_confirmations) == 2
    initial, retry = session.release_confirmations
    assert initial.release_id == plan.release_id
    assert initial.mirror_cleanup_confirmed is True
    assert initial.reclamation_receipts == retirement_evidence(plan.retirements)
    assert initial.acknowledged_retry is False
    assert retry.release_id == plan.release_id
    assert retry.mirror_cleanup_confirmed is False
    assert retry.reclamation_receipts == ()
    assert retry.acknowledged_retry is True
    assert runtime.failure_reason is None
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(key)
    runtime.close()


def test_post_ack_manager_error_retains_id_only_retry_state() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.release_retirements = (_retirement(),)
    session.release_outcomes = [EngineReleaseDisposition.RECYCLE_PENDING]
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 2),))
    plan = runtime.prepare_release((key,))
    original_confirm = session.confirm_release
    calls = 0

    def confirm(evidence: Any) -> EngineReleaseOutcome:
        nonlocal calls
        calls += 1
        if calls == 2:
            session.trace.append(("confirm-release", evidence.release_id))
            session.release_confirmations.append(evidence)
            raise ManagerError("recycle retry rejected")
        return original_confirm(evidence)

    session.confirm_release = confirm
    with pytest.raises(ReleaseRecyclePending) as pending:
        runtime.confirm_release(plan)

    assert pending.value.plan is plan
    group = runtime._releases[plan.release_id]
    assert group.cleanup_confirmed is True
    assert group.ack_committed is True
    assert runtime.failure_reason is None
    assert len(session.release_confirmations) == 2
    initial, retry = session.release_confirmations
    assert initial.acknowledged_retry is False
    assert retry.mirror_cleanup_confirmed is False
    assert retry.reclamation_receipts == ()
    assert retry.acknowledged_retry is True

    session.confirm_release = original_confirm
    runtime.confirm_release(plan)
    assert session.release_confirmations[-1].acknowledged_retry is True
    assert len(
        [item for item in trace if isinstance(item, tuple) and item[0] == "cleanup"]
    ) == 1
    runtime.close()


def test_two_pending_outcomes_retain_identity_until_same_plan_retry() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.release_retirements = (_retirement(),)
    session.release_outcomes = [
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.RECYCLE_PENDING,
        EngineReleaseDisposition.COMPLETED,
    ]
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    binding = _acquire_bound(runtime, ((key, 9),))[0]
    plan = runtime.prepare_release((key,))

    with pytest.raises(ReleaseRecyclePending) as pending:
        runtime.confirm_release(plan)

    group = runtime._releases[plan.release_id]
    assert group.cleanup_confirmed is True
    assert group.ack_committed is True
    assert pending.value.plan is plan
    assert pending.value.release_id == plan.release_id
    assert runtime.pending_release((key,)) is plan
    assert runtime.view_for(key) is binding
    assert runtime.binding_for(key).request_row == 9
    replacement = ("str", "replacement")
    runtime.acquire_unbound((replacement,))
    with pytest.raises(ManagerError, match="ReqToToken row is already owned"):
        runtime.bind_request_rows(((replacement, 9),))
    assert runtime.binding_for(replacement).request_row is None
    assert runtime.failure_reason is None
    assert session.close_calls == 0
    assert len(
        [
            item
            for item in trace
            if isinstance(item, tuple) and item[0] == "cleanup"
        ]
    ) == 1
    assert len(session.release_confirmations) == 2

    runtime.confirm_release(plan)

    assert len(session.release_confirmations) == 3
    initial, immediate_retry, later_retry = session.release_confirmations
    assert initial.acknowledged_retry is False
    for retry in (immediate_retry, later_retry):
        assert retry.release_id == plan.release_id
        assert retry.mirror_cleanup_confirmed is False
        assert retry.reclamation_receipts == ()
        assert retry.acknowledged_retry is True
    assert len(
        [
            item
            for item in trace
            if isinstance(item, tuple) and item[0] == "cleanup"
        ]
    ) == 1
    with pytest.raises(ManagerError, match="unknown session request key"):
        runtime.binding_for(key)
    runtime.bind_request_rows(((replacement, 9),))
    _release(runtime, replacement)
    runtime.close()


@pytest.mark.parametrize(
    "outcome_kind",
    ("untyped", "wrong-release-id", "invalid-disposition"),
)
def test_malformed_release_outcome_fail_stops(outcome_kind: str) -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    plan = runtime.prepare_release((key,))
    if outcome_kind == "untyped":
        session.release_outcomes = [object()]
    elif outcome_kind == "wrong-release-id":
        session.release_outcomes = [
            EngineReleaseOutcome(
                EngineReleaseId(
                    plan.release_id.session_epoch, plan.release_id.sequence + 1
                ),
                EngineReleaseDisposition.COMPLETED,
            )
        ]
    else:
        invalid_disposition: Any = 99
        session.release_outcomes = [
            EngineReleaseOutcome(plan.release_id, invalid_disposition)
        ]

    with pytest.raises(FailStopped, match="malformed outcome"):
        runtime.confirm_release(plan)

    assert runtime.failure_reason is not None
    assert "malformed outcome" in runtime.failure_reason
    assert session.close_calls == 1
    assert len(session.release_confirmations) == 1
    assert len(
        [
            item
            for item in trace
            if isinstance(item, tuple) and item[0] == "cleanup"
        ]
    ) == 1
    with pytest.raises(FailStopped, match="malformed outcome"):
        runtime.binding_for(key)


@pytest.mark.parametrize(
    ("stage", "message"),
    (("query", "query failed"), ("wait", "wait failed")),
)
def test_event_uncertainty_fail_stops_and_closes(stage: str, message: str) -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 1))
    event = Event(
        trace,
        query_error=RuntimeError(message) if stage == "query" else None,
        wait_error=RuntimeError(message) if stage == "wait" else None,
    )
    runtime.register_event(ticket, event, 1)

    operation = runtime.poll if stage == "query" else lambda: runtime.wait_requests((key,))
    with pytest.raises(FailStopped, match=message):
        operation()
    assert session.close_calls == 1
    assert runtime.failure_reason is not None
    assert runtime.completion_evidence() == {
        "event_backend": "cuda_event_current_forward_stream",
        "pending_events": 0,
        "completion_high_water": [],
    }
    with pytest.raises(FailStopped, match=message):
        runtime.poll()


def test_native_completion_uncertainty_fail_stops_and_closes() -> None:
    session = FakeSession()
    session.complete_error = FailStopped("native poisoned")
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 1))
    runtime.register_event(ticket, Event(session.trace, ready=True), 1)

    with pytest.raises(FailStopped, match="native poisoned"):
        runtime.poll()
    assert session.close_calls == 1
    assert session.publication_confirmations == []
    assert runtime.failure_reason is not None


def test_cleanup_must_explicitly_confirm_before_native_confirmation() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(trace, confirmed=False)
    )
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 1))
    runtime.register_event(ticket, Event(trace, ready=True), 1)

    with pytest.raises(FailStopped, match="not explicitly confirmed"):
        runtime.poll()
    assert session.publication_confirmations == []
    assert session.close_calls == 1


def test_confirmed_view_is_not_advanced_before_cleanup_and_native_confirm() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    key = ("str", "request")
    observed = []
    runtime: SessionRuntime

    def cleanup(_updates, _retirements):
        observed.append(runtime.view_for(key))
        return True

    runtime = SessionRuntime(session, mirror_cleanup=cleanup)
    acquired = _acquire_bound(runtime, ((key, 1),))[0]
    ticket = _submit(runtime, (key, 16))
    runtime.register_event(ticket, Event(trace, ready=True), 1)
    runtime.poll()

    assert observed == [acquired]
    assert runtime.view_for(key) == EngineRequestView(
        EngineRequestId(1), 2, 16, 1
    )
    _release(runtime, key)
    runtime.close()


def test_release_confirmation_uncertainty_keeps_identity_until_fail_stop() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    session.confirm_release_error = FailStopped("lost release return")
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    plan = runtime.prepare_release((key,))

    with pytest.raises(FailStopped, match="lost release return"):
        runtime.confirm_release(plan)
    assert session.close_calls == 1
    assert runtime.failure_reason is not None
    # The coordinator never makes the row reusable after an unknown confirm.
    assert runtime._next_request_id == 2


def test_collective_cleanup_adapter_preserves_four_phase_order() -> None:
    trace: list[Any] = []

    class Coordinator:
        def preflight(self, items, retirements):
            trace.append(("preflight", items, retirements))
            return object()

        def commit(self, plan):
            trace.append(("commit", plan))

        def synchronize(self, plan):
            trace.append(("synchronize", plan))

        def finalize(self, plan):
            trace.append(("finalize", plan))

    update = SessionMirrorUpdate(
        ("str", "request"), 1, 2, 16, (_detached(),), True
    )
    cleanup = collective_mirror_cleanup(
        Coordinator(), lambda item: (item.key, item.request_row)
    )

    assert cleanup((update,), (_retirement(),)) is True
    assert [item[0] for item in trace] == [
        "preflight",
        "commit",
        "synchronize",
        "finalize",
    ]
    mirror_item = trace[0][1][0]
    assert mirror_item.context == (("str", "request"), 2)
    assert mirror_item.detached == (_detached(),)
    assert mirror_item.releasing is True
    assert mirror_item.boundary == 16
    assert trace[0][2] == (_retirement(),)


def test_empty_publication_skips_cleanup_phases_and_still_allows_ack() -> None:
    coordinator_calls: list[str] = []

    class Coordinator:
        def preflight(self, _items, _retirements):
            coordinator_calls.append("preflight")
            return object()

        def commit(self, _plan):
            coordinator_calls.append("commit")

        def synchronize(self, _plan):
            coordinator_calls.append("synchronize")

        def finalize(self, _plan):
            coordinator_calls.append("finalize")

    trace: list[Any] = []
    session = FakeSession(trace)
    cleanup = collective_mirror_cleanup(
        Coordinator(), lambda item: (item.key, item.request_row)
    )
    runtime = SessionRuntime(session, mirror_cleanup=cleanup)
    key = ("str", "empty-publication")
    _acquire_bound(runtime, ((key, 2),))
    ticket = _submit(runtime, (key, 16))
    runtime.register_event(ticket, Event(trace, ready=True), 9)

    publication = runtime.poll()[0]

    assert coordinator_calls == []
    assert len(session.publication_confirmations) == 1
    confirmation = session.publication_confirmations[0]
    assert confirmation.publication_id == publication.publication_id
    assert confirmation.mirror_cleanup_confirmed is True
    assert confirmation.reclamation_receipts == ()
    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names[-3:] == ["query", "complete", "confirm-publication"]

    _release(runtime, key)
    runtime.close()


def test_collective_cleanup_keeps_empty_request_release() -> None:
    trace: list[Any] = []

    class Coordinator:
        def preflight(self, items, retirements):
            trace.append(("preflight", items, retirements))
            return object()

        def commit(self, plan):
            trace.append(("commit", plan))

        def synchronize(self, plan):
            trace.append(("synchronize", plan))

        def finalize(self, plan):
            trace.append(("finalize", plan))

    update = SessionMirrorUpdate(
        ("str", "empty"), 1, 2, 0, (), True
    )
    cleanup = collective_mirror_cleanup(
        Coordinator(), lambda item: (item.key, item.request_row)
    )

    assert cleanup((update,), ()) is True
    assert [item[0] for item in trace] == [
        "preflight",
        "commit",
        "synchronize",
        "finalize",
    ]
    mirror_item = trace[0][1][0]
    assert mirror_item.releasing is True
    assert mirror_item.detached == ()
    assert mirror_item.boundary == 0


@pytest.mark.parametrize("stage", ("query", "cleanup"))
def test_base_exception_uncertainty_also_fail_stops(stage: str) -> None:
    session = FakeSession()

    def fatal_cleanup(_updates, _retirements):
        raise FatalSignal("fatal cleanup")

    runtime = SessionRuntime(
        session,
        mirror_cleanup=(
            fatal_cleanup if stage == "cleanup" else _cleanup(session.trace)
        ),
    )
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 1))
    event = Event(session.trace, ready=True)
    if stage == "query":
        event.query_error = FatalSignal("fatal query")
    runtime.register_event(ticket, event, 1)

    with pytest.raises(FailStopped, match=f"fatal {stage}"):
        runtime.poll()
    assert session.close_calls == 1
    assert runtime.failure_reason is not None


def test_read_only_surface_forwards_census_and_reports_local_events() -> None:
    session = FakeSession()
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )

    assert runtime.arenas == session.arenas
    assert runtime.arenas_by_class == {0: session.arenas[0]}
    assert runtime.engine_epoch == 7
    assert runtime.page_tokens == 16
    stats, arena_stats = runtime.census()
    assert stats.active_requests == 0
    assert arena_stats == ()
    assert runtime.performance_counters() == {
        "forward_events": 0,
        "completion_values": 0,
        "event_queries": 0,
        "event_waits": 0,
        "fail_stop_count": 0,
    }
    runtime.close()
    runtime.close()
    assert session.close_calls == 1


def test_removed_runtime_aliases_are_not_public_surface() -> None:
    runtime = SessionRuntime(
        FakeSession(), mirror_cleanup=lambda _updates, _retirements: True
    )

    for name in (
        "acquire",
        "create",
        "failed",
        "closed",
        "pending_events",
        "completion_high_water",
    ):
        assert not hasattr(runtime, name)
    assert "materialization_bound" not in SessionRuntime.__dict__
    assert runtime.materialization_bound is False
    runtime.close()


def test_clean_close_rejects_live_requests_but_fail_stop_forces_close() -> None:
    session = FakeSession()
    runtime = SessionRuntime(
        session, mirror_cleanup=_cleanup(session.trace)
    )
    _acquire_bound(runtime, ((("str", "request"), 1),))

    with pytest.raises(ManagerError, match="live requests"):
        runtime.close()
    assert session.close_calls == 0
    runtime.fail_stop("owner shutdown became uncertain")
    assert session.close_calls == 1
    assert runtime.failure_reason == "owner shutdown became uncertain"
    assert runtime.performance_counters()["fail_stop_count"] == 1


def test_coordinator_does_not_duplicate_native_lease_or_phase_state() -> None:
    runtime = SessionRuntime(
        FakeSession(), mirror_cleanup=lambda _u, _r: True
    )
    assert not {
        "_prepared",
        "_submitted",
        "_pending_publications",
        "_snapshot_leases",
        "_step_leases",
        "_submission_leases",
        "_page_shadow",
    }.intersection(vars(runtime))
    runtime.close()
