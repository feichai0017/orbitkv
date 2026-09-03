from __future__ import annotations

import ctypes
from typing import Any, Sequence, TypeVar

from orbitkv_sglang.runtime import ClassLowering, ManagerError, WriteIntent

from .session_types import (
    EngineBatchId,
    EngineControlId,
    EnginePrefixId,
    EnginePublicationId,
    EngineReleaseId,
    EngineRequestId,
    EngineStepPlan,
    _copy_intent,
    _session_id_c,
    _tail,
)


_SessionIdT = TypeVar(
    "_SessionIdT",
    EngineBatchId,
    EnginePublicationId,
    EngineReleaseId,
    EnginePrefixId,
    EngineControlId,
)


def decode_operation_id(
    value: Any,
    target_type: type[_SessionIdT],
    layout: Any,
    label: str,
) -> _SessionIdT:
    result = target_type(int(value.session_epoch), int(value.sequence))
    _session_id_c(result, target_type, layout, label)
    return result


def require_batch_count(
    values: Sequence[Any], label: str, capacity: int
) -> int:
    count = len(values)
    if not 0 < count <= capacity:
        raise ManagerError(
            f"{label} cardinality exceeds its configured batch bound"
        )
    return count


def canonical_span_end(
    offset: int, count: int, cursor: int, total: int, label: str
) -> int:
    end = offset + count
    if offset != cursor or end < offset or end > total:
        raise ManagerError(f"{label} spans are not canonical")
    return end


def bounded_output_counts(
    counts: Sequence[ctypes.c_uint32],
    capacities: Sequence[int],
    operation: str,
) -> tuple[int, ...]:
    values = tuple(int(item.value) for item in counts)
    if len(values) != len(capacities) or any(
        value > capacity
        for value, capacity in zip(values, capacities, strict=True)
    ):
        raise ManagerError(
            f"{operation} output counts exceed fixed workspace bounds"
        )
    return values


def decode_write_intent(value: Any) -> WriteIntent:
    if int(value.reserved) != 0:
        raise ManagerError("write intent reserved field is nonzero")
    return WriteIntent(int(value.page_generation), int(value.page_id), 0)


def decode_prepared_steps(
    request_ids: tuple[EngineRequestId, ...],
    count: int,
    prepared_steps: Any,
    class_lowering_buffer: Any,
    tail_actions: Any,
    copy_intents: Any,
    write_intents: Any,
    class_total: int,
    tail_total: int,
    copy_total: int,
    write_total: int,
) -> tuple[EngineStepPlan, ...]:
    cursors = [0, 0, 0, 0]
    result = []
    for index in range(count):
        raw_step = prepared_steps[index]
        if int(raw_step.request_id) != request_ids[index]:
            raise ManagerError("session prepare request ordering changed")
        ends = [
            canonical_span_end(
                int(raw_step.class_offset),
                int(raw_step.class_count),
                cursors[0],
                class_total,
                "session prepare class",
            ),
            canonical_span_end(
                int(raw_step.tail_offset),
                int(raw_step.tail_count),
                cursors[1],
                tail_total,
                "session prepare tail",
            ),
            canonical_span_end(
                int(raw_step.copy_offset),
                int(raw_step.copy_count),
                cursors[2],
                copy_total,
                "session prepare copy",
            ),
            canonical_span_end(
                int(raw_step.write_offset),
                int(raw_step.write_count),
                cursors[3],
                write_total,
                "session prepare write",
            ),
        ]
        class_lowerings = []
        local_tail = local_copy = local_write = 0
        for position in range(cursors[0], ends[0]):
            raw = class_lowering_buffer[position]
            if int(raw.reserved) != 0:
                raise ManagerError(
                    "class lowering reserved field is nonzero"
                )
            next_tail = canonical_span_end(
                int(raw.tail_offset),
                int(raw.tail_count),
                cursors[1] + local_tail,
                ends[1],
                "session class tail",
            )
            next_copy = canonical_span_end(
                int(raw.copy_offset),
                int(raw.copy_count),
                cursors[2] + local_copy,
                ends[2],
                "session class copy",
            )
            next_write = canonical_span_end(
                int(raw.write_offset),
                int(raw.write_count),
                cursors[3] + local_write,
                ends[3],
                "session class write",
            )
            class_lowerings.append(
                ClassLowering(
                    int(raw.class_id),
                    int(raw.flags),
                    local_tail,
                    int(raw.tail_count),
                    local_copy,
                    int(raw.copy_count),
                    local_write,
                    int(raw.write_count),
                    0,
                    int(raw.previous_layout_boundary),
                    int(raw.target_layout_boundary),
                )
            )
            local_tail = next_tail - cursors[1]
            local_copy = next_copy - cursors[2]
            local_write = next_write - cursors[3]
        if (local_tail, local_copy, local_write) != (
            ends[1] - cursors[1],
            ends[2] - cursors[2],
            ends[3] - cursors[3],
        ):
            raise ManagerError(
                "session class spans do not cover step outputs"
            )
        result.append(
            EngineStepPlan(
                EngineRequestId(int(raw_step.request_id)),
                int(raw_step.base_view_version),
                int(raw_step.target_view_version),
                int(raw_step.previous_boundary),
                int(raw_step.target_boundary),
                tuple(class_lowerings),
                tuple(
                    _tail(tail_actions[position])
                    for position in range(cursors[1], ends[1])
                ),
                tuple(
                    _copy_intent(copy_intents[position])
                    for position in range(cursors[2], ends[2])
                ),
                tuple(
                    decode_write_intent(write_intents[position])
                    for position in range(cursors[3], ends[3])
                ),
            )
        )
        cursors = ends
    if tuple(cursors) != (
        class_total,
        tail_total,
        copy_total,
        write_total,
    ):
        raise ManagerError(
            "session prepare flat outputs are not fully partitioned"
        )
    return tuple(result)


__all__ = [
    "bounded_output_counts",
    "canonical_span_end",
    "decode_operation_id",
    "decode_prepared_steps",
    "decode_write_intent",
    "require_batch_count",
]
