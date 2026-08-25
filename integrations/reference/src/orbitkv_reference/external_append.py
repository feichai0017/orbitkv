"""Private canonicalization helpers for engine-owned append tickets."""

from __future__ import annotations

from collections.abc import Sequence

from orbitkv_runtime import (
    BackendTokenAddress,
    DataPlaneOperation,
    ExternalTokenWrite,
    OperationContext,
    PageLease,
    RequestLease,
    StepLease,
)


def snapshot_external_write(write: ExternalTokenWrite) -> ExternalTokenWrite:
    """Copy one public authorization into non-aliased value state."""

    if not isinstance(write, ExternalTokenWrite):
        raise TypeError("external write must be an ExternalTokenWrite")
    context = write.context
    if (
        not isinstance(context, OperationContext)
        or context.operation is not DataPlaneOperation.APPEND
        or not isinstance(context.request, RequestLease)
        or not isinstance(context.transaction, StepLease)
    ):
        raise TypeError("external write has an invalid append context")
    destination = write.destination
    if not isinstance(destination, BackendTokenAddress) or not isinstance(
        destination.page, PageLease
    ):
        raise TypeError("external write has an invalid destination")

    request = context.request
    transaction = context.transaction
    page = destination.page
    return ExternalTokenWrite(
        OperationContext(
            RequestLease(request.engine_epoch, request.slot, request.generation),
            StepLease(
                transaction.engine_epoch,
                transaction.slot,
                transaction.generation,
            ),
            context.operation,
        ),
        write.token_id,
        BackendTokenAddress(
            PageLease(
                page.engine_epoch,
                page.pool_epoch,
                page.generation,
                page.page_id,
                page.pool_id,
            ),
            destination.class_id,
            destination.backend_domain,
            destination.backend_index,
            destination.token_offset,
        ),
        write.byte_count,
    )


def snapshot_external_writes(
    writes: Sequence[ExternalTokenWrite],
) -> tuple[ExternalTokenWrite, ...]:
    return tuple(snapshot_external_write(item) for item in writes)
