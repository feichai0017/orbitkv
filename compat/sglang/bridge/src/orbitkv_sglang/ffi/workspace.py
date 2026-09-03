from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    class_outputs: int
    copy_outputs: int
    write_outputs: int
    bind_outputs: int
    completion_detached: int

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
            class_outputs=class_outputs,
            copy_outputs=copy_outputs,
            write_outputs=write_outputs,
            bind_outputs=bind_outputs,
            completion_detached=completion_detached,
        )


__all__ = [
    "HotBounds",
    "array",
    "checked_product",
]
