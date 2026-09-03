"""Validation and rollback helpers for SGLang request-table rows."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Callable, Sequence

from ..runtime import FailStopped
from .private_prefix import (
    PRIVATE_PREFIX_MARKER,
    SHARED_PREFIX_MARKERS,
    validate_private_prefix,
)


def validate_private_resident_count(
    config: Any, view: Any, boundary: int
) -> None:
    """Validate the exact page census for pure Sliding or Chunked state."""

    classes = tuple(config.classes)
    if len(classes) != 1 or classes[0].retention not in ("sliding", "chunked"):
        return
    class_config = classes[0]
    page_tokens = int(config.page_tokens)
    end = (boundary + page_tokens - 1) // page_tokens
    if class_config.retention == "chunked":
        chunk_tokens = int(class_config.chunk_tokens)
        begin = boundary // chunk_tokens * int(class_config.blocks_per_epoch)
    else:
        retained_begin = max(0, boundary - (int(class_config.window_tokens) - 1))
        begin = retained_begin // page_tokens
    expected = end - begin
    if (
        isinstance(view.resident_count, bool)
        or not isinstance(view.resident_count, Integral)
        or int(view.resident_count) != expected
    ):
        raise RuntimeError(
            "request-private continuation resident count differs from retention"
        )


def validate_allocated_rows(
    batch: Any, req_pool_indices: Any, *, integer_vector: Callable[..., Any]
) -> tuple[int, ...]:
    """Return exact allocated row identities after collective validation."""

    values = integer_vector(
        "allocated req_pool_indices", req_pool_indices, len(batch.reqs)
    )
    row_capacity = int(batch.req_to_token_pool.req_to_token.shape[0])
    if len(set(values)) != len(values):
        raise RuntimeError("SGLang allocated duplicate request-pool rows")
    if any(value <= 0 or value >= row_capacity for value in values):
        raise RuntimeError("SGLang allocated an out-of-range request-pool row")
    if any(
        req.req_pool_idx is None or int(req.req_pool_idx) != value
        for req, value in zip(batch.reqs, values, strict=True)
    ):
        raise RuntimeError(
            "SGLang allocated request-pool identity is inconsistent"
        )
    return values


def validate_private_admission(
    cache: Any,
    req: Any,
    key: Any,
    boundary: int,
    *,
    pending_attach: Any | None,
    runtime: Any,
    config: Any,
) -> bool:
    """Validate one exact request-private continuation or empty admission."""

    if pending_attach is not None or any(
        hasattr(req, name) for name in SHARED_PREFIX_MARKERS
    ):
        raise RuntimeError(
            "request-private admission retained shared Prefix authority"
        )
    request_id = getattr(req, "_orbitkv_engine_request_id", None)
    raw_row = getattr(req, "req_pool_idx", None)
    prefix = getattr(req, "prefix_indices", None)
    kv = getattr(req, "kv", None)
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    validate_private_resident_count(config, view, int(boundary))
    try:
        prefix_count = len(prefix)
    except BaseException as error:
        raise RuntimeError(
            "request-private admission has an unreadable prefix mirror"
        ) from error
    if (
        getattr(req, "_orbitkv_request_key", None) != key
        or isinstance(request_id, bool)
        or not isinstance(request_id, Integral)
        or int(request_id) <= 0
        or binding.request_id != request_id
        or view.request_id != request_id
    ):
        raise RuntimeError(
            "request-private admission differs from runtime identity"
        )
    if raw_row is None:
        table = cache.req_to_token_pool.req_to_token
        if (
            boundary != 0
            or int(view.boundary) != 0
            or int(view.resident_count) != 0
            or binding.request_row is not None
            or prefix_count != 0
            or kv is not None
            or hasattr(req, "_orbitkv_request_lease")
            or hasattr(req, PRIVATE_PREFIX_MARKER)
            or type(getattr(req, "cache_protected_len", None)) is not int
            or req.cache_protected_len != 0
        ):
            raise RuntimeError(
                "rowless request-private admission is not exactly empty"
            )
        validate_private_prefix(req, table[0], key, request_id, 0)
        return False
    table = cache.req_to_token_pool.req_to_token
    kv_boundary = getattr(kv, "kv_allocated_len", None)
    if (
        isinstance(raw_row, bool)
        or not isinstance(raw_row, Integral)
        or not 0 < int(raw_row) < int(table.shape[0])
        or binding.request_row != int(raw_row)
        or int(view.boundary) != boundary
        or prefix_count != boundary
        or isinstance(kv_boundary, bool)
        or not isinstance(kv_boundary, Integral)
        or int(kv_boundary) != boundary
    ):
        raise RuntimeError(
            "request-private continuation differs from runtime row authority"
        )
    validate_private_prefix(
        req, table[int(raw_row)], key, request_id, boundary
    )
    return False


def request_row_tensors(
    req_pool_values: Sequence[int],
    device: Any,
    execution_context: Any | None = None,
) -> tuple[Any, Any]:
    """Build the CPU row authority and its device mirror."""

    import torch

    cpu = torch.tensor(tuple(req_pool_values), dtype=torch.int64)
    if execution_context is not None:
        execution_context.validate_current()
    return cpu, cpu.to(device, non_blocking=True)


def free_new_rows(
    batch: Any,
    new_req_slots: Sequence[bool],
    *,
    runtime: Any,
) -> None:
    """Return only newly allocated rows after exact ownership checks."""

    try:
        values = tuple(
            req
            for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
            if is_new and req.req_pool_idx is not None
        )
        existing_rows = {
            int(req.req_pool_idx)
            for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
            if not is_new and req.req_pool_idx is not None
        }
        rows = tuple(getattr(req, "req_pool_idx", None) for req in values)
        capacity = int(batch.req_to_token_pool.req_to_token.shape[0])
        if (
            any(
                isinstance(row, bool)
                or not isinstance(row, Integral)
                or not 0 < int(row) < capacity
                for row in rows
            )
            or len({int(row) for row in rows}) != len(rows)
            or any(int(row) in existing_rows for row in rows)
        ):
            raise RuntimeError(
                "allocated session request rows cannot be rolled back exactly"
            )
        for req in values:
            batch.req_to_token_pool.free(req)
    except BaseException as error:
        runtime.fail_stop(f"session request-row rollback became uncertain: {error}")
        raise FailStopped(
            runtime.failure_reason or "session request-row rollback failed"
        ) from error


__all__ = [
    "free_new_rows",
    "request_row_tensors",
    "validate_private_resident_count",
    "validate_allocated_rows",
    "validate_private_admission",
]
