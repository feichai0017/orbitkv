from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.execution_plan import (
    BindingResult,
    StepExecutionResult,
    confirm_execution,
    expected_bindings,
    lower_batch_plan,
)
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.ffi.session_types import (
    EngineBatchId,
    EngineBatchPlan,
    EngineBatchPublication,
    EnginePublicationId,
    EngineRequestId,
    EngineRetirement,
    EngineStepPlan,
    EngineStepPublication,
)
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    ArenaIdentity,
    CacheSharingPolicy,
    ClassLowering,
    PageLease,
    ManagerCreateSettings,
    ManagerError,
    SessionCreateSettings,
    TailAction,
    WriteIntent,
    TAIL_NONE,
)
from orbitkv_sglang.session_activity import SessionActivityTracker
from orbitkv_sglang.session_runtime import SessionRuntime
from ffi_test_support import ffi_library


__all__ = ["ffi_library"]


def _class(
    class_id: int, retention: str, period_blocks: int | None = None
) -> Any:
    return SimpleNamespace(
        class_id=class_id,
        retention=retention,
        period_blocks=period_blocks,
    )


def _arena(
    class_id: int, *, pool_epoch: int, pool_id: int, first_page_id: int
) -> ArenaIdentity:
    return ArenaIdentity(
        7, pool_epoch, pool_id, class_id, 20 + class_id, 3, 16, 0,
        first_page_id,
    )


def _page(arena: ArenaIdentity, page_id: int, generation: int) -> PageLease:
    return PageLease(
        arena.engine_epoch,
        arena.pool_epoch,
        generation,
        page_id,
        arena.pool_id,
    )


def _plan(
    sequence: int,
    request_ids: int | tuple[int, ...],
    reservations: tuple[tuple[ArenaIdentity, int, int], ...],
) -> EngineBatchPlan:
    writes = tuple(
        WriteIntent(generation, page_id)
        for _arena_value, page_id, generation in reservations
    )
    lowerings = tuple(
        ClassLowering(
            arena.class_id, 0, index, 1, 0, 0, index, 1, 0, 0, 16
        )
        for index, (arena, _page_id, _generation) in enumerate(reservations)
    )
    tails = tuple(
        TailAction(
            arena.class_id, TAIL_NONE, 0, 0,
            PageLease(0, 0, 0, 0, 0), PageLease(0, 0, 0, 0, 0),
        )
        for arena, _page_id, _generation in reservations
    )
    ids = (request_ids,) if isinstance(request_ids, int) else request_ids
    return EngineBatchPlan(
        EngineBatchId(7, sequence),
        tuple(
            EngineStepPlan(
                EngineRequestId(request_id), 1, 2, 0, 16,
                lowerings, tails, (), writes,
            )
            for request_id in ids
        ),
    )


def _publication(
    sequence: int,
    request_ids: int | tuple[int, ...],
    boundary: int,
    retirements: tuple[EngineRetirement, ...] = (),
) -> EngineBatchPublication:
    ids = (request_ids,) if isinstance(request_ids, int) else request_ids
    return EngineBatchPublication(
        EnginePublicationId(7, sequence),
        EngineBatchId(7, sequence),
        tuple(
            EngineStepPublication(
                EngineRequestId(request_id), sequence + 1, boundary, 2, ()
            )
            for request_id in ids
        ),
        retirements,
    )


def _retirement(
    arena: ArenaIdentity, page_id: int, generation: int
) -> EngineRetirement:
    return EngineRetirement(
        _page(arena, page_id, generation), arena.class_id,
        arena.backend_domain, 0, page_id - arena.first_page_id, 0, 16, 9, 1,
    )


def test_sliding_activity_accumulates_across_requests_and_ack_gates_reuse() -> None:
    arena = _arena(0, pool_epoch=11, pool_id=3, first_page_id=1)
    classes = (_class(0, "sliding", 3),)
    tracker = SessionActivityTracker((arena,), 16)

    # Any first-seen generation is a baseline, not a reuse event.
    initial = _plan(1, 11, ((arena, 1, 7),))
    tracker.observe_plan(initial)
    first = tracker.activity(classes)
    assert first.page_reuse_events == 0

    retirement = _retirement(arena, 1, 7)
    tracker.observe_publication(
        _publication(1, 11, 49, (retirement,)),
        ((EngineRequestId(11), 0, 49),),
    )
    before_ack = tracker.activity(classes)
    assert before_ack.retirement_certificates == 1
    assert before_ack.pages_reclaimed == 0
    assert before_ack.wrap_events == 1

    tracker.acknowledge_publication((retirement,))
    reuse = _plan(2, 12, ((arena, 1, 8),))
    tracker.observe_plan(reuse)
    second = tracker.activity(classes)
    assert second.pages_reclaimed == 1
    assert second.page_reuse_events == 1

    # A different request and a multi-cycle boundary jump accumulate globally.
    tracker.observe_publication(
        _publication(4, (12, 13), 145),
        (
            (EngineRequestId(12), 0, 145),
            (EngineRequestId(13), 48, 145),
        ),
    )
    assert tracker.activity(classes).wrap_events == 7
    tracker.forget_requests(
        (EngineRequestId(11), EngineRequestId(12), EngineRequestId(13))
    )
    tracker.observe_publication(
        _publication(5, 11, 49),
        ((EngineRequestId(11), 0, 49),),
    )
    assert tracker.activity(classes).wrap_events == 8


def test_full_activity_is_not_applicable_and_never_counts_swa_events() -> None:
    arena = _arena(0, pool_epoch=12, pool_id=4, first_page_id=10)
    classes = (_class(0, "full"),)
    tracker = SessionActivityTracker((arena,), 16)
    initial = _plan(1, 21, ((arena, 10, 1),))
    tracker.observe_plan(initial)
    retirement = _retirement(arena, 10, 1)
    tracker.observe_publication(
        _publication(1, 21, 96, (retirement,)),
        ((EngineRequestId(21), 0, 96),),
    )
    tracker.acknowledge_publication((retirement,))
    reuse = _plan(2, 22, ((arena, 10, 2),))
    tracker.observe_plan(reuse)

    activity = tracker.activity(classes)
    assert activity.applicable is False
    assert activity.retirement_certificates == 0
    assert activity.pages_reclaimed == 0
    assert activity.wrap_events == 0
    assert activity.page_reuse_events == 0


def test_generation_increase_without_ack_does_not_count_reuse() -> None:
    arena = _arena(0, pool_epoch=13, pool_id=5, first_page_id=20)
    classes = (_class(0, "sliding", 3),)
    tracker = SessionActivityTracker((arena,), 16)
    tracker.observe_plan(_plan(1, 31, ((arena, 20, 1),)))
    tracker.observe_plan(_plan(2, 31, ((arena, 20, 2),)))
    assert tracker.activity(classes).page_reuse_events == 0


def test_release_or_control_ack_enables_reuse_without_retention_counts() -> None:
    arena = _arena(0, pool_epoch=14, pool_id=6, first_page_id=30)
    classes = (_class(0, "sliding", 3),)
    tracker = SessionActivityTracker((arena,), 16)
    tracker.observe_plan(_plan(1, 41, ((arena, 30, 1),)))
    tracker.acknowledge_reuse_candidates((_retirement(arena, 30, 1),))
    tracker.observe_plan(_plan(2, 42, ((arena, 30, 2),)))

    activity = tracker.activity(classes)
    assert activity.retirement_certificates == 0
    assert activity.pages_reclaimed == 0
    assert activity.page_reuse_events == 1


def _sliding_config(library: Path) -> RuntimeConfig:
    class_config = ClassConfig(
        class_id=0, pool_id=1, backend_domain=1, name="sliding",
        layers=(0,), retention="sliding", bytes_per_token_per_layer=128,
        window_tokens=18, period_blocks=3,
    )
    plan = {
        "page_tokens": 16,
        "classes": [{
            "name": "sliding", "layers": [0],
            "retention": "sliding",
            "bytes_per_token_per_layer": 128, "window_tokens": 18,
        }],
    }
    return RuntimeConfig(
        library_path=library, plan_json=json.dumps(plan).encode(),
        plan_fingerprint="sha256:session-swa-activity", page_tokens=16,
        classes=(class_config,),
    )


class _ReadyEvent:
    @staticmethod
    def query() -> bool:
        return True

    @staticmethod
    def synchronize() -> None:
        return None


def _execute(
    runtime: SessionRuntime, config: RuntimeConfig, key: Any, target: int
) -> Any:
    plan = runtime.prepare(((key, target),))
    request_id = runtime.binding_for(key).request_id
    lowered = lower_batch_plan(
        plan, config, runtime.arenas, {request_id: runtime.view_for(key)}
    )
    bindings = expected_bindings(plan, lowered, runtime.arenas)[0]
    result = StepExecutionResult(
        request_id,
        tuple(
            BindingResult(
                item.page, item.backend_domain, item.backend_index, True, True
            )
            for item in bindings
        ),
        tuple(
            intent
            for class_spec in lowered.steps[0].class_specs
            for intent in class_spec.copy_intents
        ),
        True,
    )
    ticket = runtime.submit(
        confirm_execution(plan, lowered, runtime.arenas, (result,))
    )
    runtime.register_event(ticket, _ReadyEvent(), 9)
    return runtime.poll()[0]


def test_real_session_sliding_activity_tracks_ack_then_generation_reuse(
    ffi_library: Path,
) -> None:
    config = _sliding_config(ffi_library)
    native = CtypesRuntimeSession.create(
        config,
        SessionCreateSettings(
            ManagerCreateSettings(2, 2, 1, 3, 64),
            CacheSharingPolicy.REQUEST_PRIVATE,
        ),
        (ArenaRegistration(0, 1, 1, 3, 0),),
    )
    runtime = SessionRuntime(
        native
    )
    first, second = "first", "second"
    pre_ack_activity = []
    check_pre_ack = False

    def cleanup(_updates: Any, retirements: tuple[EngineRetirement, ...]) -> bool:
        if check_pre_ack and retirements:
            pre_ack_activity.append(runtime.swa_activity(config.classes))
            with pytest.raises(ManagerError, match="capacity"):
                runtime.prepare(((second, 16),))
        return True

    runtime.bind_mirror_cleanup(cleanup)
    drained = False
    try:
        runtime.acquire_unbound((first, second))
        runtime.bind_request_rows(((first, 1), (second, 2)))
        _execute(runtime, config, first, 18)
        check_pre_ack = True
        wrapped = _execute(runtime, config, first, 35)
        check_pre_ack = False
        assert len(wrapped.retirements) == 1
        retirement = wrapped.retirements[0]
        assert len(pre_ack_activity) == 1
        assert pre_ack_activity[0].retirement_certificates == 1
        assert pre_ack_activity[0].pages_reclaimed == 0
        assert pre_ack_activity[0].page_reuse_events == 0
        activity = runtime.swa_activity(config.classes)
        assert (
            activity.retirement_certificates, activity.pages_reclaimed,
            activity.wrap_events, activity.page_reuse_events,
        ) == (1, 1, 0, 0)

        reuse = runtime.prepare(((second, 16),))
        write = reuse.steps[0].write_intents[0]
        assert write.page_id == retirement.page.page_id
        assert write.page_generation == retirement.page.generation + 1
        activity = runtime.swa_activity(config.classes)
        assert activity.page_reuse_events == 1
        runtime.abort_prepared(reuse)

        crossed = _execute(runtime, config, first, 49)
        assert len(crossed.retirements) == 1
        activity = runtime.swa_activity(config.classes)
        assert (
            activity.retirement_certificates, activity.pages_reclaimed,
            activity.wrap_events, activity.page_reuse_events,
        ) == (2, 2, 1, 1)

        for key in (first, second):
            release = runtime.prepare_release((key,))
            runtime.confirm_release(release)
        runtime.close()
        drained = True
    finally:
        if not drained and runtime.failure_reason is None:
            runtime.fail_stop("test cleanup after incomplete SWA lifecycle")
