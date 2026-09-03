from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.execution_plan import (
    BatchPlan,
    BindingResult,
    ClassSpec,
    StepExecutionResult,
    StepPlan,
    confirm_execution,
    expected_bindings,
    lower_batch_plan,
)
from orbitkv_sglang.ffi.session_types import (
    EngineBatchId,
    EngineBatchPlan,
    EngineRequestId,
    EngineRequestView,
    EngineStepPlan,
)
from orbitkv_sglang.runtime import (
    CLASS_LOWERING_PACKED,
    TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
    ArenaIdentity,
    ClassLowering,
    CopyIntent,
    ManagerError,
    PageLease,
    TailAction,
    WriteIntent,
)
from orbitkv_sglang.runtime.snapshot_shadow import (
    CLASS_LOWERING_EPOCH_START,
    CLASS_LOWERING_RESETTABLE,
)


PAGE_TOKENS = 16
_ZERO_PAGE = PageLease(0, 0, 0, 0, 0)
_LEASE_NAMES = (
    "RequestLease",
    "SnapshotLease",
    "StepLease",
    "SubmissionLease",
)


def _class(
    class_id: int, pool_id: int, backend_domain: int, retention: str
) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=pool_id,
        backend_domain=backend_domain,
        name=f"class-{class_id}",
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=18 if retention == "sliding" else None,
        period_blocks=2 if retention == "sliding" else None,
    )


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:execution-plan-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            _class(0, 11, 3, "full"),
            _class(1, 22, 4, "sliding"),
        ),
    )


def _arenas() -> tuple[ArenaIdentity, ...]:
    return (
        ArenaIdentity(77, 901, 11, 0, 3, 8, PAGE_TOKENS, 100, 10),
        ArenaIdentity(77, 902, 22, 1, 4, 8, PAGE_TOKENS, 200, 20),
    )


def _page(arena: ArenaIdentity, page_id: int, generation: int) -> PageLease:
    return PageLease(
        arena.engine_epoch, arena.pool_epoch, generation, page_id, arena.pool_id
    )


def _batch() -> EngineBatchPlan:
    full, sliding = _arenas()
    full_source = _page(full, 10, 7)
    full_destination = _page(full, 11, 8)
    sliding_destination = _page(sliding, 20, 4)
    full_copy = CopyIntent(
        0, 3, 9, 0, 0, full_source, full_destination, 100, 101, 0
    )
    return EngineBatchPlan(
        EngineBatchId(77, 5),
        (
            EngineStepPlan(
                EngineRequestId(101),
                12,
                13,
                25,
                42,
                (
                    ClassLowering(0, 0, 0, 1, 0, 1, 0, 1, 0, 25, 42),
                    ClassLowering(1, 0, 1, 1, 1, 0, 1, 1, 0, 25, 42),
                ),
                (
                    TailAction(
                        0, 2, 9, 1, full_source, full_destination, 0
                    ),
                    TailAction(1, TAIL_FRESH, 0, 1, _ZERO_PAGE, sliding_destination, 0),
                ),
                (full_copy,),
                (
                    WriteIntent(20, 12, 0),
                    WriteIntent(30, 21, 0),
                ),
            ),
        ),
    )


def _replace_step(batch: EngineBatchPlan, **changes: Any) -> EngineBatchPlan:
    return replace(batch, steps=(replace(batch.steps[0], **changes),))


def _views(
    batch: EngineBatchPlan,
) -> dict[EngineRequestId, EngineRequestView]:
    return {
        step.request_id: EngineRequestView(
            step.request_id, step.base_view_version, step.previous_boundary, 0
        )
        for step in batch.steps
    }


def _chunked_config() -> RuntimeConfig:
    return replace(
        _config(),
        classes=(
            replace(
                _config().classes[0],
                retention="chunked",
                chunk_tokens=32,
                blocks_per_epoch=2,
            ),
        ),
    )


def _chunked_batch(
    previous: int, target: int, flags: int
) -> EngineBatchPlan:
    arena = _arenas()[0]
    first_new = (previous + PAGE_TOKENS - 1) // PAGE_TOKENS
    new_end = (target + PAGE_TOKENS - 1) // PAGE_TOKENS
    writes = tuple(
        WriteIntent(10 + offset, arena.first_page_id + offset, 0)
        for offset, _ordinal in enumerate(range(first_new, new_end))
    )
    return EngineBatchPlan(
        EngineBatchId(77, 9),
        (
            EngineStepPlan(
                EngineRequestId(303),
                4,
                5,
                previous,
                target,
                (
                    ClassLowering(
                        0, flags, 0, 1, 0, 0, 0, len(writes),
                        0, previous, target
                    ),
                ),
                (TailAction(0, 0, 0, 0, _ZERO_PAGE, _ZERO_PAGE, 0),),
                (),
                writes,
            ),
        ),
    )


def _assert_no_control_leases(value: Any) -> None:
    if type(value).__name__ in _LEASE_NAMES:
        pytest.fail(f"lightweight plan leaked {type(value).__name__}")
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_no_control_leases(getattr(value, field.name))
    elif isinstance(value, (tuple, list, dict)):
        values = value.values() if isinstance(value, dict) else value
        for item in values:
            _assert_no_control_leases(item)


def test_lower_batch_plan_builds_helper_compatible_physical_shape() -> None:
    batch = _batch()
    lowered = lower_batch_plan(batch, _config(), _arenas(), _views(batch))

    assert isinstance(lowered, BatchPlan)
    assert lowered.batch_id == EngineBatchId(77, 5)
    assert lowered.arenas == _arenas()
    assert len(lowered.steps) == 1
    step = lowered.steps[0]
    assert isinstance(step, StepPlan)
    assert (step.request_id, step.previous_boundary, step.target_boundary) == (
        EngineRequestId(101),
        25,
        42,
    )
    assert tuple(step.by_class) == (0, 1)
    assert all(isinstance(item, ClassSpec) for item in step.class_specs)

    full = step.by_class[0]
    assert full.pool_id == 11
    assert full.last_location == 2 * PAGE_TOKENS + 9 - 1
    assert full.exact_new_pages == (3,)
    assert full.tail_action.kind == TAIL_COPY_ON_WRITE
    assert full.copy_intents == _batch().steps[0].copy_intents
    assert (full.previous_layout_boundary, full.target_layout_boundary) == (25, 42)

    sliding = step.by_class[1]
    assert sliding.pool_id == 22
    assert sliding.last_location == PAGE_TOKENS + 9 - 1
    assert sliding.exact_new_pages == (2,)
    assert sliding.copy_intents == ()
    _assert_no_control_leases(lowered)


def test_confirm_execution_requires_observed_results_and_is_exact() -> None:
    batch = _batch()
    arenas = _arenas()
    lowered = lower_batch_plan(batch, _config(), arenas, _views(batch))
    bindings = expected_bindings(batch, lowered, arenas)
    result = StepExecutionResult(
        batch.steps[0].request_id,
        tuple(
            BindingResult(
                item.page, item.backend_domain, item.backend_index, True, True
            )
            for item in bindings[0]
        ),
        batch.steps[0].copy_intents,
        True,
    )
    evidence = confirm_execution(batch, lowered, arenas, (result,))

    assert evidence.batch_id == batch.batch_id
    assert len(evidence.steps) == 1
    step = evidence.steps[0]
    assert step.request_id == EngineRequestId(101)
    assert tuple(item.page.page_id for item in step.bind_receipts) == (
        11, 12, 20, 21
    )
    assert tuple(item.backend_domain for item in step.bind_receipts) == (
        3, 3, 4, 4
    )
    assert tuple(item.backend_index for item in step.bind_receipts) == (
        101, 102, 200, 201
    )
    assert all(item.mapped and item.writable for item in step.bind_receipts)

    assert len(step.copy_receipts) == 1
    copy = step.copy_receipts[0]
    intent = batch.steps[0].copy_intents[0]
    assert (
        copy.class_id,
        copy.backend_domain,
        copy.token_count,
        copy.source_token_offset,
        copy.destination_token_offset,
        copy.source,
        copy.destination,
        copy.source_backend_index,
        copy.destination_backend_index,
    ) == (
        intent.class_id,
        intent.backend_domain,
        intent.token_count,
        intent.source_token_offset,
        intent.destination_token_offset,
        intent.source,
        intent.destination,
        intent.source_backend_index,
        intent.destination_backend_index,
    )
    assert copy.observed and copy.copied and copy.ordered_before_writes


@pytest.mark.parametrize(
    "change",
    (
        {"view_version": 11},
        {"boundary": 24},
        {"request_id": EngineRequestId(999)},
    ),
)
def test_lower_batch_plan_rejects_stale_request_mirror(
    change: dict[str, Any]
) -> None:
    batch = _batch()
    current = _views(batch)[EngineRequestId(101)]
    views = {EngineRequestId(101): replace(current, **change)}
    with pytest.raises(ManagerError, match="current request mirror"):
        lower_batch_plan(batch, _config(), _arenas(), views)


def test_initial_request_view_rejects_packed_lowering() -> None:
    batch = _batch()
    step = batch.steps[0]
    initial = replace(
        step,
        base_view_version=1,
        target_view_version=2,
        previous_boundary=0,
        target_boundary=17,
        class_lowerings=(
            ClassLowering(0, CLASS_LOWERING_PACKED, 0, 1, 0, 0, 0, 2, 0, 0, 17),
            ClassLowering(1, 0, 1, 1, 0, 0, 2, 2, 0, 0, 17),
        ),
        tail_actions=(
            TailAction(0, 0, 0, 0, _ZERO_PAGE, _ZERO_PAGE, 0),
            TailAction(1, 0, 0, 0, _ZERO_PAGE, _ZERO_PAGE, 0),
        ),
        copy_intents=(),
    )
    initial_batch = replace(batch, steps=(initial,))
    views = {
        EngineRequestId(101): EngineRequestView(
            EngineRequestId(101), 1, 0, 0
        )
    }
    with pytest.raises(ManagerError, match="initial request view"):
        lower_batch_plan(initial_batch, _config(), _arenas(), views)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"bindings": ()}, "binding results"),
        ({"completed_copies": ()}, "completed copy results"),
        ({"copies_ordered_before_writes": False}, "before writes"),
    ),
)
def test_confirm_execution_rejects_missing_or_unordered_observations(
    change: dict[str, Any], message: str
) -> None:
    batch = _batch()
    arenas = _arenas()
    lowered = lower_batch_plan(batch, _config(), arenas, _views(batch))
    bindings = tuple(
        BindingResult(
            item.page, item.backend_domain, item.backend_index, True, True
        )
        for item in expected_bindings(batch, lowered, arenas)[0]
    )
    result = StepExecutionResult(
        batch.steps[0].request_id,
        bindings,
        batch.steps[0].copy_intents,
        True,
    )
    with pytest.raises(ManagerError, match=message):
        confirm_execution(batch, lowered, arenas, (replace(result, **change),))


@pytest.mark.parametrize(
    "change",
    (
        {"backend_domain": 99},
        {"backend_index": 999},
        {"mapped": False},
        {"writable": False},
    ),
)
def test_confirm_execution_rejects_inexact_binding_observation(
    change: dict[str, Any]
) -> None:
    batch = _batch()
    arenas = _arenas()
    lowered = lower_batch_plan(batch, _config(), arenas, _views(batch))
    observed = tuple(
        BindingResult(
            item.page, item.backend_domain, item.backend_index, True, True
        )
        for item in expected_bindings(batch, lowered, arenas)[0]
    )
    result = StepExecutionResult(
        batch.steps[0].request_id,
        (replace(observed[0], **change), *observed[1:]),
        batch.steps[0].copy_intents,
        True,
    )
    with pytest.raises(ManagerError, match="binding results"):
        confirm_execution(batch, lowered, arenas, (result,))


def test_confirm_execution_rejects_a_different_lowered_plan() -> None:
    batch = _batch()
    arenas = _arenas()
    lowered = lower_batch_plan(batch, _config(), arenas, _views(batch))
    bindings = tuple(
        BindingResult(
            item.page, item.backend_domain, item.backend_index, True, True
        )
        for item in expected_bindings(batch, lowered, arenas)[0]
    )
    result = StepExecutionResult(
        batch.steps[0].request_id, bindings, batch.steps[0].copy_intents, True
    )
    tampered = replace(
        lowered,
        steps=(replace(lowered.steps[0], target_boundary=41),),
    )
    with pytest.raises(ManagerError, match="lowered plan changed"):
        confirm_execution(batch, tampered, arenas, (result,))

    tampered_spec = replace(
        lowered,
        steps=(
            replace(
                lowered.steps[0],
                class_specs=(
                    replace(
                        lowered.steps[0].class_specs[0],
                        exact_new_pages=(99,),
                    ),
                    lowered.steps[0].class_specs[1],
                ),
            ),
        ),
    )
    with pytest.raises(ManagerError, match="class specs changed"):
        confirm_execution(batch, tampered_spec, arenas, (result,))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda batch: _replace_step(
                batch, class_lowerings=tuple(reversed(batch.steps[0].class_lowerings))
            ),
            "compiled order",
        ),
        (
            lambda batch: _replace_step(
                batch,
                class_lowerings=(
                    replace(batch.steps[0].class_lowerings[0], write_offset=1),
                    batch.steps[0].class_lowerings[1],
                ),
            ),
            "class write span",
        ),
        (
            lambda batch: _replace_step(
                batch,
                class_lowerings=(
                    replace(batch.steps[0].class_lowerings[0], flags=8),
                    batch.steps[0].class_lowerings[1],
                ),
            ),
            "flags",
        ),
        (
            lambda batch: _replace_step(
                batch,
                class_lowerings=(
                    replace(
                        batch.steps[0].class_lowerings[0],
                        target_layout_boundary=41,
                    ),
                    batch.steps[0].class_lowerings[1],
                ),
            ),
            "append delta",
        ),
        (
            lambda batch: _replace_step(
                batch,
                tail_actions=(
                    replace(batch.steps[0].tail_actions[0], class_id=1),
                    batch.steps[0].tail_actions[1],
                ),
            ),
            "tail action",
        ),
        (
            lambda batch: _replace_step(
                batch,
                write_intents=(
                    replace(batch.steps[0].write_intents[0], page_generation=0),
                    *batch.steps[0].write_intents[1:],
                ),
            ),
            "generation",
        ),
        (
            lambda batch: _replace_step(
                batch,
                write_intents=(
                    replace(batch.steps[0].write_intents[0], page_id=20),
                    *batch.steps[0].write_intents[1:],
                ),
            ),
            "outside its class arena",
        ),
        (
            lambda batch: _replace_step(
                batch,
                copy_intents=(
                    replace(
                        batch.steps[0].copy_intents[0],
                        destination_backend_index=102,
                    ),
                ),
            ),
            "exact tail echo",
        ),
    ),
)
def test_lower_batch_plan_rejects_tampered_step(
    mutate: Any, message: str
) -> None:
    batch = mutate(_batch())
    with pytest.raises(ManagerError, match=message):
        lower_batch_plan(batch, _config(), _arenas(), _views(batch))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda arenas: (replace(arenas[0], pool_id=99), arenas[1]),
            "differs from its plan class",
        ),
        (
            lambda arenas: (arenas[0], replace(arenas[1], class_id=0)),
            "class-id ordered",
        ),
        (
            lambda arenas: (arenas[0], replace(arenas[1], engine_epoch=78)),
            "engine epochs",
        ),
        (
            lambda arenas: (arenas[0], replace(arenas[1], first_page_id=15)),
            "page-id ranges overlap",
        ),
    ),
)
def test_lower_batch_plan_rejects_tampered_arena(
    mutate: Any, message: str
) -> None:
    batch = _batch()
    with pytest.raises(ManagerError, match=message):
        lower_batch_plan(batch, _config(), mutate(_arenas()), _views(batch))


def test_lower_batch_plan_rejects_overlapping_backend_ranges() -> None:
    config = _config()
    classes = (config.classes[0], replace(config.classes[1], backend_domain=3))
    arenas = (
        _arenas()[0],
        replace(_arenas()[1], backend_domain=3, backend_base_index=105),
    )
    with pytest.raises(ManagerError, match="backend arena ranges overlap"):
        batch = _batch()
        lower_batch_plan(
            batch, replace(config, classes=classes), arenas, _views(batch)
        )


def test_lower_batch_plan_rejects_foreign_batch_identity_and_duplicate_requests() -> None:
    foreign = replace(_batch(), batch_id=EngineBatchId(78, 5))
    with pytest.raises(ManagerError, match="batch identity"):
        lower_batch_plan(foreign, _config(), _arenas(), _views(foreign))

    batch = _batch()
    duplicate = replace(batch, steps=(batch.steps[0], batch.steps[0]))
    with pytest.raises(ManagerError, match="duplicate request"):
        lower_batch_plan(duplicate, _config(), _arenas(), _views(duplicate))


def test_packed_class_boundaries_remain_available_to_lowering_helpers() -> None:
    batch = _batch()
    packed_lowering = replace(
        batch.steps[0].class_lowerings[0],
        flags=CLASS_LOWERING_PACKED,
        previous_layout_boundary=9,
        target_layout_boundary=26,
    )
    source = batch.steps[0].tail_actions[0].source
    destination = batch.steps[0].tail_actions[0].destination
    packed = _replace_step(
        batch,
        class_lowerings=(packed_lowering, batch.steps[0].class_lowerings[1]),
        tail_actions=(
            replace(
                batch.steps[0].tail_actions[0],
                valid_token_count=9,
                logical_ordinal=0,
            ),
            batch.steps[0].tail_actions[1],
        ),
        copy_intents=(
            replace(
                batch.steps[0].copy_intents[0],
                token_count=9,
                source=source,
                destination=destination,
            ),
        ),
    )

    spec = lower_batch_plan(
        packed, _config(), _arenas(), _views(packed)
    ).steps[0].by_class[0]
    assert (spec.previous_layout_boundary, spec.target_layout_boundary) == (9, 26)


@pytest.mark.parametrize(
    ("previous", "target", "flags"),
    (
        (0, 16, CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START),
        (16, 32, CLASS_LOWERING_RESETTABLE),
        (32, 48, CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START),
    ),
)
def test_chunked_flags_follow_resettable_epoch_geometry(
    previous: int, target: int, flags: int
) -> None:
    batch = _chunked_batch(previous, target, flags)
    lowered = lower_batch_plan(
        batch,
        _chunked_config(),
        (_arenas()[0],),
        _views(batch),
    )
    spec = lowered.steps[0].by_class[0]
    assert (spec.previous_layout_boundary, spec.target_layout_boundary) == (
        previous, target
    )
    assert spec.tail_action.kind == 0


@pytest.mark.parametrize(
    ("batch", "message"),
    (
        (
            _chunked_batch(0, 16, CLASS_LOWERING_RESETTABLE),
            "resettable lowering geometry",
        ),
        (
            _chunked_batch(16, 33, CLASS_LOWERING_RESETTABLE),
            "resettable lowering geometry",
        ),
        (
            _chunked_batch(32, 48, CLASS_LOWERING_RESETTABLE),
            "resettable lowering geometry",
        ),
    ),
)
def test_chunked_lowering_rejects_wrong_flags_or_cross_epoch(
    batch: EngineBatchPlan, message: str
) -> None:
    with pytest.raises(ManagerError, match=message):
        lower_batch_plan(
            batch, _chunked_config(), (_arenas()[0],), _views(batch)
        )


def test_batch_rejects_destination_alias_across_steps() -> None:
    batch = _batch()
    duplicate = replace(
        batch,
        steps=(
            batch.steps[0],
            replace(
                batch.steps[0],
                request_id=EngineRequestId(202),
                base_view_version=20,
                target_view_version=21,
            ),
        ),
    )
    with pytest.raises(ManagerError, match="aliases a destination page across steps"):
        lower_batch_plan(duplicate, _config(), _arenas(), _views(duplicate))
