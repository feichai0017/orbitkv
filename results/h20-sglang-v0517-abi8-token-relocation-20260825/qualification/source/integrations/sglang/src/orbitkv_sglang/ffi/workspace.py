from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import layouts as L


UINT32_MAX = (1 << 32) - 1


def checked_product(name: str, *values: int) -> int:
    result = 1
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} factor must be a nonnegative integer")
        result *= value
        if result > UINT32_MAX:
            raise ValueError(f"{name} exceeds uint32_t")
    return result


def array(layout: Any, capacity: int) -> Any:
    if capacity <= 0:
        return None
    return (layout * capacity)()


@dataclass(frozen=True, slots=True)
class HotBounds:
    batch: int
    classes: int
    pages_per_step_class: int
    physical_pages: int
    class_outputs: int
    copy_outputs: int
    write_outputs: int
    bind_outputs: int
    completion_detached: int
    completion_retirements: int

    @classmethod
    def compile(
        cls,
        *,
        maximum_batch: int,
        class_count: int,
        maximum_step_tokens: int,
        page_tokens: int,
        physical_pages: int,
    ) -> HotBounds:
        if page_tokens <= 0 or maximum_step_tokens <= 0:
            raise ValueError("page and step token bounds must be positive")
        pages_per_step_class = (maximum_step_tokens + page_tokens - 1) // page_tokens
        class_outputs = checked_product("class output bound", maximum_batch, class_count)
        write_outputs = min(
            physical_pages,
            checked_product(
                "write output bound", maximum_batch, class_count, pages_per_step_class
            ),
        )
        copy_outputs = min(physical_pages, class_outputs)
        bind_outputs = min(
            physical_pages,
            checked_product(
                "bind output bound",
                maximum_batch,
                class_count,
                pages_per_step_class + 1,
            ),
        )
        completion_detached = checked_product(
            "completion detached bound",
            maximum_batch,
            class_count,
            pages_per_step_class + 2,
        )
        return cls(
            batch=maximum_batch,
            classes=class_count,
            pages_per_step_class=pages_per_step_class,
            physical_pages=physical_pages,
            class_outputs=class_outputs,
            copy_outputs=copy_outputs,
            write_outputs=write_outputs,
            bind_outputs=bind_outputs,
            completion_detached=completion_detached,
            completion_retirements=min(physical_pages, completion_detached),
        )


class HotWorkspace:
    """Reusable O(B*C*K) storage for prepare/submit/complete hot calls."""

    def __init__(self, bounds: HotBounds):
        self.bounds = bounds
        self.request_views = array(L.RequestViewLayout, bounds.batch)
        self.prepare_items = array(L.PrepareItemLayout, bounds.batch)
        self.prepared = array(L.PreparedItemLayout, bounds.batch)
        self.class_lowerings = array(L.ClassLoweringLayout, bounds.class_outputs)
        self.tail_actions = array(L.TailActionLayout, bounds.class_outputs)
        self.copy_intents = array(L.CopyIntentLayout, bounds.copy_outputs)
        self.write_intents = array(L.WriteIntentLayout, bounds.write_outputs)
        self.submit_items = array(L.SubmitItemLayout, bounds.batch)
        self.bind_receipts = array(L.BindReceiptLayout, bounds.bind_outputs)
        self.copy_receipts = array(L.CopyReceiptLayout, bounds.copy_outputs)
        self.submitted = array(L.SubmittedItemLayout, bounds.batch)
        self.complete_items = array(L.CompleteItemLayout, bounds.batch)
        self.completed = array(L.CompletedItemLayout, bounds.batch)
        self.detached = array(L.DetachedBindingLayout, bounds.completion_detached)
        self.retirements = array(
            L.ReclamationCertificateLayout, bounds.completion_retirements
        )


@dataclass(slots=True)
class ColdOutput:
    items: Any
    pages_or_detached: Any
    retirements: Any = None


def cold_materialization(item_layout: Any, item_count: int, page_count: int) -> ColdOutput:
    return ColdOutput(
        array(item_layout, item_count),
        array(L.SnapshotPageLayout, page_count),
    )


def cold_reclamation(
    item_layout: Any,
    item_count: int,
    detached_count: int,
    retirement_count: int,
) -> ColdOutput:
    return ColdOutput(
        array(item_layout, item_count),
        array(L.DetachedBindingLayout, detached_count),
        array(L.ReclamationCertificateLayout, retirement_count),
    )


__all__ = [
    "ColdOutput",
    "HotBounds",
    "HotWorkspace",
    "array",
    "checked_product",
    "cold_materialization",
    "cold_reclamation",
]
