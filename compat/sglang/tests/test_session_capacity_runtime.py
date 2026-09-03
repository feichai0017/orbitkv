from __future__ import annotations

from typing import Any

import pytest

import orbitkv_sglang.bridge.lowering as lowering
from orbitkv_sglang.runtime import ArenaStats, FailStopped
from orbitkv_sglang.session_runtime import SessionRuntime
from test_session_lowering import _install
from test_session_runtime import (
    Event,
    FakeSession,
    _acquire_bound,
    _cleanup,
    _release,
    _submit,
)


def test_session_next_batch_polls_and_fail_stops_callback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _batch_value, events = _install(monkeypatch, previous=0)
    scheduler = object()
    assert lowering._get_next_batch_to_run(
        lambda owner: (events.append("next"), owner)[1], scheduler
    ) is scheduler
    assert events[-4:] == ["turn-enter", "poll", "next", "turn-exit"]

    with pytest.raises(FailStopped, match="pre-forward"):
        lowering._get_next_batch_to_run(
            lambda _owner: (_ for _ in ()).throw(RuntimeError("boom")),
            scheduler,
        )
    assert "boom" in runtime.failure_reason


def test_scheduler_turn_polls_once_and_caches_one_native_arena_sample() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 16))
    event = Event(trace)
    runtime.register_event(ticket, event, 7)

    with runtime.scheduler_turn():
        assert runtime.poll() == ()
        assert runtime.arena_stats() == ()
        assert runtime.arena_stats() == ()

    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names.count("query") == 1
    assert names.count("arena-stats") == 1
    assert session.completions == []

    event.ready = True
    with runtime.scheduler_turn():
        assert runtime.poll() == ()
        assert runtime.arena_stats() == ()

    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names.count("query") == 2
    assert names.count("arena-stats") == 2
    assert len(session.completions) == 1

    _release(runtime, key)
    runtime.close()


def test_scheduler_turn_reuses_poll_and_stats_for_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitkv_sglang.bridge.state as state

    trace: list[Any] = []
    session = FakeSession(trace)
    session.arena_stats_value = (
        ArenaStats(7, 11, 3, 32, 0, 17, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))
    ticket = _submit(runtime, (key, 16))
    event = Event(trace)
    runtime.register_event(ticket, event, 7)
    monkeypatch.setattr(state, "_RUNTIME", runtime)

    with runtime.scheduler_turn():
        assert state._arena_available_tokens(0) == 48
        assert state._arena_available_tokens(0) == 48

    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names.count("query") == 1
    assert names.count("arena-stats") == 1

    assert state._arena_available_tokens(0) == 48
    assert state._arena_available_tokens(0) == 48
    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names.count("query") == 3
    assert names.count("arena-stats") == 3

    event.ready = True
    with runtime.scheduler_turn():
        assert state._arena_available_tokens(0) == 48
    assert len(session.completions) == 1

    _release(runtime, key)
    runtime.close()


def test_scheduler_turn_capacity_mutation_refreshes_arena_stats() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))
    key = ("str", "request")
    _acquire_bound(runtime, ((key, 1),))

    with runtime.scheduler_turn():
        assert runtime.arena_stats() == ()
        plan = runtime.prepare(((key, 16),))
        assert runtime.arena_stats() == ()
        assert runtime.arena_stats() == ()

    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names.count("arena-stats") == 2

    runtime.abort_prepared(plan)
    _release(runtime, key)
    runtime.close()


def test_arena_stats_are_not_cached_outside_scheduler_turn() -> None:
    trace: list[Any] = []
    session = FakeSession(trace)
    runtime = SessionRuntime(session, mirror_cleanup=_cleanup(trace))

    assert runtime.arena_stats() == ()
    assert runtime.arena_stats() == ()

    assert trace == ["arena-stats", "arena-stats"]
    runtime.close()
