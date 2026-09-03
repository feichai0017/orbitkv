from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineControlDisposition,
    EngineControlId,
    EngineControlKind,
    EngineControlOutcome,
    EngineControlPlanInfo,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixEvictionPlan,
    EnginePrefixId,
    EnginePrefixLookup,
    EnginePublishedPrefix,
    EngineRequestId,
    EngineRequestView,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    CacheSharingPolicy,
    FailStopped,
    ManagerError,
    PageLease,
    PrefixSemanticKey,
    SnapshotPage,
)
from orbitkv_sglang.session_runtime import (  # noqa: E402
    SessionMaterializationUpdate,
    SessionRuntime,
    unbound_materialization,
)


def _semantic(boundary: int) -> PrefixSemanticKey:
    seed = bytes([boundary % 251 + 1])
    return PrefixSemanticKey(seed * 32, bytes([boundary % 241 + 3]) * 32, boundary)


def _snapshot_page(
    page_id: int, *, logical_ordinal: int = 0, backend_index: int = 100
) -> SnapshotPage:
    return SnapshotPage(
        PageLease(7, 11, 1, page_id, 3),
        logical_ordinal,
        0,
        0,
        backend_index + page_id,
        0,
        17,
        16,
        logical_ordinal * 16,
        16,
    )


class FakeControlSession:
    def __init__(self) -> None:
        self.arenas = (ArenaIdentity(7, 11, 3, 0, 17, 32, 16, 0, 1),)
        self.cache_sharing_policy = CacheSharingPolicy.SHARED_PREFIX
        self.prefix_capacity = 8
        self.control_batch_capacity = 8
        self.prefix_eviction_batch_capacity = 8
        self.trace: list[Any] = []
        self.close_calls = 0
        self.control_sequence = 1
        self.acquire_sequence = 1
        self.prefix_sequence = 20
        self.controls: dict[EngineControlId, dict[str, Any]] = {}
        self.confirm_control_results: list[Any] = []
        self.quarantined_controls: list[EngineControlId] = []

    def close(self) -> None:
        self.trace.append("close")
        self.close_calls += 1

    def stats(self):
        raise AssertionError("unused")

    def arena_stats(self):
        return ()

    def acquire_requests(self, request_ids):
        values = tuple(request_ids)
        self.trace.append(("acquire", values))
        return tuple(EngineRequestView(item, 1, 0, 0) for item in values)

    def prefix_lookup_batch(self, keys):
        values = tuple(keys)
        self.trace.append(("prefix_lookup", values))
        return tuple(
            EnginePrefixLookup(value, EnginePrefixId(7, index + 1), value.boundary // 16)
            for index, value in enumerate(values)
        )

    def prefix_publish_batch(self, items):
        values = tuple(items)
        self.trace.append(("prefix_publish", values))
        outputs = []
        for item in values:
            outputs.append(
                EnginePublishedPrefix(
                    EnginePrefixId(7, self.prefix_sequence), item.key, item.key.boundary // 16
                )
            )
            self.prefix_sequence += 1
        return tuple(outputs)

    def prepare_prefix_attach(self, items):
        values = tuple(items)
        self.trace.append(("prepare_prefix_attach", values))
        control_id = EngineControlId(7, self.control_sequence)
        self.control_sequence += 1
        pages = (_snapshot_page(30), _snapshot_page(31, logical_ordinal=1))
        target = values[0].target_request_id
        self.controls[control_id] = {
            "kind": "materialization",
            "targets": tuple(item.target_request_id for item in values),
            "plan_info": EngineControlPlanInfo(
                control_id,
                EngineControlKind.MATERIALIZATION,
                len(values),
                len(pages),
                0,
                0,
            ),
            "plan": EngineMaterializationPlan(
                control_id,
                (
                    EngineMaterializedRequest(
                        target,
                        9,
                        32,
                        2,
                        pages,
                    ),
                ),
            ),
        }
        return control_id

    def prepare_request_fork(self, items):
        values = tuple(items)
        self.trace.append(("prepare_request_fork", values))
        control_id = EngineControlId(7, self.control_sequence)
        self.control_sequence += 1
        source, target = values[0].source_request_id, values[0].target_request_id
        pages = (_snapshot_page(40), _snapshot_page(40, logical_ordinal=1))
        self.controls[control_id] = {
            "kind": "materialization",
            "targets": (target,),
            "plan_info": EngineControlPlanInfo(
                control_id, EngineControlKind.MATERIALIZATION, 1, 2, 0, 0
            ),
            "plan": EngineMaterializationPlan(
                control_id,
                (
                    EngineMaterializedRequest(
                        target,
                        7,
                        32,
                        2,
                        pages,
                    ),
                ),
            ),
            "source": source,
        }
        return control_id

    def prepare_prefix_evict(self, prefix_ids):
        values = tuple(prefix_ids)
        self.trace.append(("prepare_prefix_evict", values))
        control_id = EngineControlId(7, self.control_sequence)
        self.control_sequence += 1
        self.controls[control_id] = {
            "kind": "eviction",
            "plan_info": EngineControlPlanInfo(
                control_id,
                EngineControlKind.PREFIX_EVICTION,
                0,
                0,
                len(values),
                0,
            ),
            "plan": EnginePrefixEvictionPlan(control_id, values, ()),
        }
        return control_id

    def abort_control(self, control_id):
        self.trace.append(("abort_control", control_id))
        self.controls.pop(control_id, None)

    def commit_control(self, control_id):
        self.trace.append(("commit_control", control_id))
        return self.controls[control_id]["plan_info"]

    def read_control_plan(self, control_id):
        self.trace.append(("read_control_plan", control_id))
        return self.controls[control_id]["plan"]

    def confirm_control(self, evidence):
        self.trace.append(("confirm_control", evidence))
        if self.confirm_control_results:
            result = self.confirm_control_results.pop(0)
        else:
            kind = self.controls[evidence.control_id]["kind"]
            disposition = (
                EngineControlDisposition.MATERIALIZED
                if kind == "materialization"
                else EngineControlDisposition.EVICTED
            )
            result = EngineControlOutcome(evidence.control_id, disposition)
        if isinstance(result, BaseException):
            raise result
        self.controls.pop(evidence.control_id, None)
        return result

    def quarantine_control(self, control_id):
        self.trace.append(("quarantine_control", control_id))
        self.quarantined_controls.append(control_id)
        self.controls.pop(control_id, None)


def _runtime(
    session: FakeControlSession,
    *,
    materialization=None,
):
    return SessionRuntime(
        session,
        mirror_cleanup=lambda _updates, _retirements: True,
        materialization=(
            unbound_materialization if materialization is None else materialization
        ),
    )


@pytest.mark.parametrize(
    "operation",
    (
        lambda runtime: runtime.prefix_lookup(()),
        lambda runtime: runtime.prefix_publish(()),
        lambda runtime: runtime.prepare_prefix_publish_release(()),
        lambda runtime: runtime.prepare_prefix_attach(()),
        lambda runtime: runtime.prepare_request_fork(()),
        lambda runtime: runtime.prepare_prefix_evict(()),
        lambda runtime: runtime.abort_control(object()),
        lambda runtime: runtime.commit_control(object()),
        lambda runtime: runtime.read_control(object()),
        lambda runtime: runtime.cancel_pending_attach(object(), object()),
        lambda runtime: runtime.finalize_pending_attach_cancel(object(), object()),
        lambda runtime: runtime.confirm_control(object()),
        lambda runtime: runtime.quarantine_control(object()),
    ),
)
def test_request_private_runtime_rejects_every_sharing_control_before_state(
    operation: Any,
) -> None:
    session = FakeControlSession()
    session.cache_sharing_policy = CacheSharingPolicy.REQUEST_PRIVATE
    runtime = _runtime(session)

    with pytest.raises(ManagerError, match="request-private cache sharing policy"):
        operation(runtime)

    assert session.trace == []
    assert runtime._controls == {}
    assert runtime._pending_attach_cancels == {}
    runtime.close()


@pytest.mark.parametrize("capacity", (None, True, 0, -1, 9, "1"))
def test_runtime_rejects_invalid_prefix_eviction_capacity(capacity: Any) -> None:
    session = FakeControlSession()
    if capacity is None:
        del session.prefix_eviction_batch_capacity
    else:
        session.prefix_eviction_batch_capacity = capacity

    with pytest.raises(ManagerError, match="capacity"):
        _runtime(session)


def _acquire_bound(
    runtime: SessionRuntime, assignments: tuple[tuple[Any, int], ...]
) -> tuple[EngineRequestView, ...]:
    values = tuple(assignments)
    views = runtime.acquire_unbound(tuple(key for key, _row in values))
    runtime.bind_request_rows(values)
    return views


def test_materialization_confirm_updates_views_only_after_native_confirm() -> None:
    session = FakeControlSession()
    observed: list[Any] = []
    key = ("req", "target")
    runtime: SessionRuntime

    def materialize(updates):
        observed.append((runtime.view_for(key), updates))
        return True

    runtime = _runtime(session, materialization=materialize)
    original = _acquire_bound(runtime, ((key, 3),))[0]
    control_id = runtime.prepare_prefix_attach(((key, runtime.prefix_lookup((_semantic(32),))[0]),))
    runtime.commit_control(control_id)
    plan = runtime.read_control(control_id)

    assert runtime.view_for(key) is original
    outcome = runtime.confirm_control(control_id)

    assert outcome == EngineControlOutcome(control_id, EngineControlDisposition.MATERIALIZED)
    assert observed[0][0] is original
    update = observed[0][1][0]
    assert update == SessionMaterializationUpdate(
        key,
        EngineRequestId(1),
        3,
        9,
        32,
        2,
        plan.requests[0].pages,
    )
    assert runtime.view_for(key).boundary == 32
    assert runtime.view_for(key).view_version == 9
    runtime.fail_stop("teardown")


def test_prefix_attach_target_binds_after_commit_and_read() -> None:
    session = FakeControlSession()
    materialized_calls: list[tuple[SessionMaterializationUpdate, ...]] = []

    def materialize(updates):
        materialized_calls.append(tuple(updates))
        return True

    runtime = _runtime(session, materialization=materialize)
    target = ("req", "attached")
    runtime.acquire_unbound((target,))

    attach_control = runtime.prepare_prefix_attach(
        ((target, runtime.prefix_lookup((_semantic(32),))[0]),)
    )
    assert runtime.binding_for(target).request_row is None
    attach_info = runtime.commit_control(attach_control)
    assert runtime.binding_for(target).request_row is None
    attach_plan = runtime.read_control(attach_control)
    assert attach_info.kind is EngineControlKind.MATERIALIZATION
    assert attach_plan.control_id == attach_control
    assert runtime.binding_for(target).request_row is None

    runtime.bind_request_rows(((target, 3),))
    outcome = runtime.confirm_control(attach_control)

    assert outcome == EngineControlOutcome(
        attach_control, EngineControlDisposition.MATERIALIZED
    )
    assert len(materialized_calls) == 1
    assert materialized_calls[0][0].key == target
    assert materialized_calls[0][0].request_row == 3
    assert runtime.view_for(target).boundary == 32
    runtime.fail_stop("teardown")


def test_request_fork_target_binds_after_commit_and_read() -> None:
    session = FakeControlSession()
    materialized_calls: list[tuple[SessionMaterializationUpdate, ...]] = []

    def materialize(updates):
        materialized_calls.append(tuple(updates))
        return True

    runtime = _runtime(session, materialization=materialize)
    source = ("req", "source")
    target = ("req", "fork-target")
    _acquire_bound(runtime, ((source, 1),))
    runtime.acquire_unbound((target,))

    fork_control = runtime.prepare_request_fork(((source, target),))
    assert runtime.binding_for(target).request_row is None
    fork_info = runtime.commit_control(fork_control)
    assert runtime.binding_for(target).request_row is None
    fork_plan = runtime.read_control(fork_control)
    assert fork_info.kind is EngineControlKind.MATERIALIZATION
    assert fork_plan.control_id == fork_control
    assert runtime.binding_for(target).request_row is None

    runtime.bind_request_rows(((target, 2),))
    outcome = runtime.confirm_control(fork_control)

    assert outcome == EngineControlOutcome(
        fork_control, EngineControlDisposition.MATERIALIZED
    )
    assert len(materialized_calls) == 1
    assert materialized_calls[0][0].key == target
    assert materialized_calls[0][0].request_row == 2
    assert runtime.view_for(target).boundary == 32
    runtime.fail_stop("teardown")


def test_materialization_confirm_requires_bound_target_row() -> None:
    session = FakeControlSession()
    materialized_calls: list[tuple[SessionMaterializationUpdate, ...]] = []

    def materialize(updates):
        materialized_calls.append(tuple(updates))
        return True

    runtime = _runtime(session, materialization=materialize)
    target = ("req", "attached")
    runtime.acquire_unbound((target,))
    control_id = runtime.prepare_prefix_attach(
        ((target, runtime.prefix_lookup((_semantic(32),))[0]),)
    )
    runtime.commit_control(control_id)
    runtime.read_control(control_id)

    with pytest.raises(FailStopped, match="bound target ReqToToken row"):
        runtime.confirm_control(control_id)
    assert session.quarantined_controls == [control_id]
    assert target in runtime._quarantined_requests
    assert materialized_calls == []


def test_materialization_confirm_preserves_mirror_done_for_retryable_manager_error() -> None:
    session = FakeControlSession()
    calls: list[tuple[SessionMaterializationUpdate, ...]] = []

    def materialize(updates):
        calls.append(tuple(updates))
        return True

    runtime = _runtime(session, materialization=materialize)
    source = ("req", "source")
    target = ("req", "target")
    _acquire_bound(runtime, ((source, 1), (target, 2)))
    control_id = runtime.prepare_request_fork(((source, target),))
    runtime.commit_control(control_id)
    runtime.read_control(control_id)
    session.confirm_control_results = [
        ManagerError("native retry"),
        EngineControlOutcome(control_id, EngineControlDisposition.MATERIALIZED),
    ]

    with pytest.raises(ManagerError, match="native retry"):
        runtime.confirm_control(control_id)
    assert len(calls) == 1
    evidence = [
        item[1]
        for item in session.trace
        if isinstance(item, tuple) and item[0] == "confirm_control"
    ][0]
    assert evidence.mirror_cleanup_confirmed is True

    outcome = runtime.confirm_control(control_id)
    assert outcome.disposition is EngineControlDisposition.MATERIALIZED
    assert len(calls) == 1
    assert runtime.view_for(target).boundary == 32
    runtime.fail_stop("teardown")


def test_materialization_callback_failure_quarantines_and_fail_stops() -> None:
    session = FakeControlSession()
    key = ("req", "target")

    def materialize(_updates):
        raise RuntimeError("mirror blew up")

    runtime = _runtime(session, materialization=materialize)
    _acquire_bound(runtime, ((key, 3),))
    control_id = runtime.prepare_prefix_attach(((key, runtime.prefix_lookup((_semantic(32),))[0]),))
    runtime.commit_control(control_id)
    runtime.read_control(control_id)

    with pytest.raises(FailStopped, match="mirror blew up"):
        runtime.confirm_control(control_id)
    assert session.quarantined_controls == [control_id]
    assert key in runtime._quarantined_requests
    assert session.close_calls == 1


def test_hostile_materialization_outcome_quarantines_and_fail_stops() -> None:
    session = FakeControlSession()
    key = ("req", "target")
    runtime = _runtime(session, materialization=lambda _updates: True)
    _acquire_bound(runtime, ((key, 3),))
    control_id = runtime.prepare_prefix_attach(((key, runtime.prefix_lookup((_semantic(32),))[0]),))
    runtime.commit_control(control_id)
    runtime.read_control(control_id)
    session.confirm_control_results = [
        EngineControlOutcome(control_id, EngineControlDisposition.EVICTED)
    ]

    with pytest.raises(FailStopped, match="hostile outcome"):
        runtime.confirm_control(control_id)
    assert session.quarantined_controls == [control_id]
    assert key in runtime._quarantined_requests


def test_fork_source_and_target_operations_remain_control_blocked() -> None:
    session = FakeControlSession()
    runtime = _runtime(session, materialization=lambda _updates: True)
    source = ("req", "source")
    target = ("req", "target")
    runtime.acquire_unbound((source,))
    _acquire_bound(runtime, ((target, 2),))
    control_id = runtime.prepare_request_fork(((source, target),))
    runtime.commit_control(control_id)
    runtime.read_control(control_id)

    with pytest.raises(ManagerError, match="pending control group"):
        runtime.bind_request_rows(((source, 9),))
    with pytest.raises(ManagerError, match="pending control group"):
        runtime.prepare(((target, 16),))
    with pytest.raises(ManagerError, match="pending control group"):
        runtime.prepare_release((target,))


def test_quarantined_materialization_target_rejects_row_binding() -> None:
    session = FakeControlSession()
    runtime = _runtime(session, materialization=lambda _updates: True)
    target = ("req", "target")
    runtime.acquire_unbound((target,))
    control_id = runtime.prepare_prefix_attach(
        ((target, runtime.prefix_lookup((_semantic(32),))[0]),)
    )
    runtime.commit_control(control_id)
    runtime.read_control(control_id)
    runtime.quarantine_control(control_id)

    with pytest.raises(ManagerError, match="quarantined"):
        runtime.bind_request_rows(((target, 2),))
    assert runtime.binding_for(target).request_row is None
    assert runtime._row_owners == {}


@pytest.mark.parametrize(
    "corrupt",
    (
        "wrong-kind",
        "wrong-owner",
        "missing-owner",
        "ambiguous-group",
        "duplicate-target",
        "unrelated-occupancy",
    ),
)
def test_unsafe_control_occupancy_rejects_target_row_binding(
    corrupt: str,
) -> None:
    session = FakeControlSession()
    runtime = _runtime(session, materialization=lambda _updates: True)
    target = ("req", "target")
    runtime.acquire_unbound((target,))
    control_id = runtime.prepare_prefix_attach(
        ((target, runtime.prefix_lookup((_semantic(32),))[0]),)
    )
    runtime.commit_control(control_id)
    runtime.read_control(control_id)
    group = runtime._controls[control_id]

    if corrupt == "wrong-kind":
        assert group.plan_info is not None
        group.plan_info = replace(
            group.plan_info, kind=EngineControlKind.PREFIX_EVICTION
        )
    elif corrupt == "wrong-owner":
        runtime._request_occupancy[target] = EngineControlId(7, 900)
    elif corrupt == "missing-owner":
        del runtime._request_occupancy[target]
    elif corrupt == "ambiguous-group":
        other_id = EngineControlId(7, 901)
        runtime._controls[other_id] = replace(group, control_id=other_id)
    elif corrupt == "duplicate-target":
        group.target_keys = (target, target)
    else:
        runtime._request_occupancy[("req", "unrelated")] = EngineControlId(
            7, 902
        )

    with pytest.raises(ManagerError, match="unsafe|inconsistent"):
        runtime.bind_request_rows(((target, 2),))
    assert runtime.binding_for(target).request_row is None
    assert runtime._row_owners == {}


def test_row_binding_batch_is_atomic_across_target_and_unsafe_source() -> None:
    session = FakeControlSession()
    runtime = _runtime(session, materialization=lambda _updates: True)
    source = ("req", "source")
    target = ("req", "target")
    runtime.acquire_unbound((source, target))
    control_id = runtime.prepare_request_fork(((source, target),))
    runtime.commit_control(control_id)
    runtime.read_control(control_id)

    with pytest.raises(ManagerError, match="unsafe pending control group"):
        runtime.bind_request_rows(((target, 2), (source, 3)))
    assert runtime.binding_for(target).request_row is None
    assert runtime.binding_for(source).request_row is None
    assert runtime._row_owners == {}


def test_shared_duplicate_pages_are_accepted_for_materialization() -> None:
    session = FakeControlSession()
    runtime = _runtime(session, materialization=lambda _updates: True)
    source = ("req", "source")
    target = ("req", "target")
    _acquire_bound(runtime, ((source, 1), (target, 2)))
    control_id = runtime.prepare_request_fork(((source, target),))
    runtime.commit_control(control_id)
    plan = runtime.read_control(control_id)

    assert plan.requests[0].pages[0].page == plan.requests[0].pages[1].page
    runtime.confirm_control(control_id)
    assert runtime.view_for(target).boundary == 32
    runtime.fail_stop("teardown")
