from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.facade as facade  # noqa: E402
import orbitkv_sglang.bridge.lowering as lowering  # noqa: E402
import orbitkv_sglang.bridge.session_lowering as session_lowering  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
from orbitkv_sglang.ffi.session_types import EngineReleaseId  # noqa: E402
from orbitkv_sglang.session_runtime import (  # noqa: E402
    ReleaseAckRetryPending,
    ReleaseRecyclePending,
    ReleaseRetryPending,
)

from test_session_lowering import _install  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def test_public_release_entrypoint_dispatches_to_session_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime_value, batch, _events = _install(monkeypatch, previous=4)
    calls = []
    monkeypatch.setattr(
        session_lowering,
        "release_kv_cache",
        lambda req, cache, *, is_insert: calls.append(
            (req, cache, is_insert)
        ),
    )

    lowering._release_kv_cache(
        batch.reqs[0], batch.tree_cache, is_insert=False
    )

    assert calls == [(batch.reqs[0], batch.tree_cache, False)]


@pytest.mark.parametrize(
    "pending_error",
    (
        lambda plan: ReleaseRecyclePending(plan),
        lambda plan: ReleaseAckRetryPending(plan),
    ),
)
def test_single_release_pending_is_retained_and_scheduler_retries_first(
    monkeypatch: pytest.MonkeyPatch,
    pending_error: Any,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    allocator = state._ALLOCATOR
    candidate = object()
    plan = SimpleNamespace(release_id=EngineReleaseId(7, 91))
    calls = []

    monkeypatch.setattr(
        session_lowering.session_cache,
        "release_candidate",
        lambda req, cache, *, is_insert: candidate,
    )

    def flush(candidates: Any) -> None:
        values = tuple(candidates)
        calls.append(values)
        if len(calls) == 1:
            raise pending_error(plan)

    monkeypatch.setattr(
        session_lowering.session_cache, "flush_release_group", flush
    )

    with pytest.raises(ReleaseRetryPending):
        session_lowering.release_kv_cache(
            batch.reqs[0], batch.tree_cache, is_insert=False
        )

    assert runtime.failure_reason is None
    assert allocator._orbitkv_free_group_state == "retry_pending"
    assert allocator.free_group == (candidate,)
    assert allocator.is_not_in_free_group is True
    with pytest.raises(RuntimeError, match="recycle is pending"):
        session_lowering.release_kv_cache(
            batch.reqs[0], batch.tree_cache, is_insert=False
        )
    assert runtime.failure_reason is None

    result = session_lowering.get_next_batch_to_run(
        lambda owner: (events.append("next"), owner)[1], object()
    )

    assert result is not None
    assert calls == [(candidate,), (candidate,)]
    assert allocator._orbitkv_free_group_state == "idle"
    assert allocator.free_group == []
    assert events[-4:] == ["turn-enter", "poll", "next", "turn-exit"]


def test_pre_ack_retry_crosses_scheduler_without_republish_or_recleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_session_runtime import (
        Event,
        FakeSession,
        _cleanup,
        _prefix_detached,
        _semantic,
        _submit,
    )
    from orbitkv_sglang.runtime import ManagerError
    from orbitkv_sglang.session_runtime import SessionRuntime

    trace: list[Any] = []
    native = FakeSession(trace)
    native.release_detached = (_prefix_detached(),)
    runtime = SessionRuntime(native, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    runtime.acquire_unbound((key,))
    runtime.bind_request_rows(((key, 2),))
    ticket = _submit(runtime, (key, 16))
    runtime.register_event(ticket, Event(trace, ready=True), 1)
    runtime.poll()

    allocator = facade._NativeAuthorityForbidden()
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    candidate = object()
    state._RUNTIME = runtime
    state._ALLOCATOR = allocator
    plans = []

    def flush(values: Any) -> None:
        assert tuple(values) == (candidate,)
        plan = runtime.pending_release((key,))
        if plan is None:
            published = runtime.prepare_prefix_publish_release(
                ((key, _semantic(64, 16)),)
            )
            plan = runtime.pending_release((key,))
            assert plan is not None and plan.release_id == published.release_id
            plans.append(plan)
        runtime.confirm_release(plan)

    monkeypatch.setattr(
        session_lowering.session_cache,
        "flush_release_group",
        flush,
    )
    native.confirm_release_error = ManagerError("ACK precommit rejection")

    with pytest.raises(ReleaseAckRetryPending) as pending:
        session_lowering.flush_release_group((candidate,))
    plan = plans[0]
    assert pending.value.plan is plan
    assert allocator.free_group == (candidate,)
    assert allocator._orbitkv_free_group_state == "retry_pending"
    assert runtime._releases[plan.release_id].ack_committed is False

    native.confirm_release_error = None
    result = session_lowering.get_next_batch_to_run(
        lambda owner: owner, object()
    )

    assert result is not None
    assert len(native.prefix_publish_release_calls) == 1
    release_cleanups = [
        item
        for item in trace
        if isinstance(item, tuple)
        and item[0] == "cleanup"
        and any(update.releasing for update in item[1])
    ]
    assert len(release_cleanups) == 1
    assert len(native.release_confirmations) == 2
    assert native.release_confirmations[1] == native.release_confirmations[0]
    assert native.release_confirmations[1].acknowledged_retry is False
    assert allocator.free_group == []
    assert allocator._orbitkv_free_group_state == "idle"
    assert runtime.failure_reason is None
    runtime.fail_stop("teardown resident prefix")


def test_scheduler_keeps_still_pending_release_without_poisoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _batch, events = _install(monkeypatch, previous=4)
    allocator = state._ALLOCATOR
    candidate = object()
    plan = SimpleNamespace(release_id=EngineReleaseId(7, 92))
    allocator._orbitkv_free_group_state = "retry_pending"
    allocator.free_group = (candidate,)
    allocator.is_not_in_free_group = True
    monkeypatch.setattr(
        session_lowering.session_cache,
        "flush_release_group",
        lambda values: (_ for _ in ()).throw(ReleaseRecyclePending(plan)),
    )

    with pytest.raises(ReleaseRecyclePending):
        session_lowering.get_next_batch_to_run(
            lambda _owner: events.append("unexpected-next"), object()
        )

    assert runtime.failure_reason is None
    assert allocator._orbitkv_free_group_state == "retry_pending"
    assert allocator.free_group == (candidate,)
    assert "poll" not in events
    assert "unexpected-next" not in events


def test_new_request_rollback_propagates_release_pending_without_clearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=4)
    allocator = state._ALLOCATOR
    req = batch.reqs[0]
    batch.tree_cache.root_node = object()
    req.last_node = batch.tree_cache.root_node
    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
    for marker in (
        "_orbitkv_prefix_node",
        "_orbitkv_prefix_semantic",
        "_orbitkv_provisional_prefix_lock",
        "_orbitkv_prefix_lock_held",
    ):
        if hasattr(req, marker):
            delattr(req, marker)
    key = req._orbitkv_request_key
    entry = batch.tree_cache._session_requests[key]
    request_id = req._orbitkv_engine_request_id
    plan = SimpleNamespace(
        release_id=EngineReleaseId(7, 94),
        releases=(SimpleNamespace(request_id=request_id),),
    )
    confirmations = 0
    monkeypatch.setattr(
        runtime, "prepare_release", lambda keys: plan, raising=False
    )
    monkeypatch.setattr(
        runtime, "pending_release", lambda keys: plan, raising=False
    )

    def confirm_release(selected: Any) -> None:
        nonlocal confirmations
        confirmations += 1
        assert selected is plan
        if confirmations < 3:
            raise ReleaseRecyclePending(selected)
        del runtime.bindings[key]
        del runtime.views[key]

    monkeypatch.setattr(
        runtime,
        "confirm_release",
        confirm_release,
        raising=False,
    )

    with pytest.raises(ReleaseRecyclePending) as pending:
        session_lowering._release_new_requests(batch, (req,))

    assert pending.value.plan is plan
    assert runtime.failure_reason is None
    assert batch.tree_cache._session_requests[key] is entry
    assert runtime.binding_for(key).request_row == 1
    assert req.req_pool_idx == 1
    assert req.kv is not None
    assert req._orbitkv_request_key == key
    assert batch.req_to_token_pool.freed == []
    assert allocator._orbitkv_free_group_state == "retry_pending"
    assert allocator.free_group == []
    assert allocator._orbitkv_pending_release_work.plan is plan
    with pytest.raises(RuntimeError, match="recycle is pending"):
        facade._NativeAuthorityForbidden.free_group_begin(allocator)
    assert runtime.failure_reason is None

    with pytest.raises(ReleaseRecyclePending):
        facade._NativeAuthorityForbidden.free_group_end(allocator)
    retained = allocator._orbitkv_pending_release_work
    assert retained.plan is plan
    assert allocator._orbitkv_free_group_state == "retry_pending"
    assert runtime.failure_reason is None

    facade._NativeAuthorityForbidden.free_group_end(allocator)

    assert confirmations == 3
    assert runtime.bindings == {}
    assert batch.tree_cache._session_requests == {}
    assert batch.req_to_token_pool.freed == [req.rid]
    assert req.req_pool_idx is None
    assert req.kv is None
    assert not hasattr(req, "_orbitkv_request_key")
    assert not hasattr(req, "_orbitkv_engine_request_id")
    assert allocator._orbitkv_pending_release_work is None
    assert allocator._orbitkv_free_group_state == "idle"


def test_free_group_retains_exact_pending_candidates_for_explicit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _batch, _events = _install(monkeypatch, previous=4)
    allocator = facade._NativeAuthorityForbidden()
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    candidate = object()
    plan = SimpleNamespace(release_id=EngineReleaseId(7, 93))
    attempts = 0

    def flush(values: Any) -> None:
        nonlocal attempts
        attempts += 1
        assert tuple(values) == (candidate,)
        if attempts < 3:
            raise ReleaseRecyclePending(plan)

    monkeypatch.setattr(lowering, "_flush_release_group", flush)
    allocator.free_group_begin()
    allocator.free_group.append(candidate)

    with pytest.raises(ReleaseRecyclePending):
        allocator.free_group_end()
    retained = allocator.free_group
    assert retained == (candidate,)
    assert allocator._orbitkv_free_group_state == "retry_pending"
    assert allocator.is_not_in_free_group is True
    assert runtime.failure_reason is None
    with pytest.raises(RuntimeError, match="recycle is pending"):
        allocator.free_group_begin()
    assert runtime.failure_reason is None

    with pytest.raises(ReleaseRecyclePending):
        allocator.free_group_end()
    assert allocator.free_group is retained
    assert allocator._orbitkv_free_group_state == "retry_pending"

    allocator.free_group_end()
    assert attempts == 3
    assert allocator.free_group == []
    assert allocator._orbitkv_free_group_state == "idle"
    assert allocator.is_not_in_free_group is True
    assert runtime.failure_reason is None
