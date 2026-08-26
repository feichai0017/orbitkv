"""Pure value and execution-context validation for external KV writes.

The external-write state machine lives in :mod:`external_append`.  This module
contains only stateless canonicalization helpers and the public errors they
raise, keeping validation reusable without splitting lifecycle ownership.
"""

from __future__ import annotations

from numbers import Integral

from orbitkv_runtime import (
    ArenaRegistration,
    BackendPageAddress,
    BackendTokenAddress,
    DataPlaneOperation,
    ExternalTokenWrite,
    OperationContext,
    PageLease,
    RequestLease,
    StepLease,
)

from ..runtime import FailStopped, ManagerError


_U64_LIMIT = 1 << 64
_MISSING = object()


class ExternalAppendError(ManagerError):
    """The adapter rejected an operation before accepting uncertainty."""


class ExternalAppendPoisonedError(FailStopped):
    """The adapter cannot continue after an uncertain external mutation."""


def canonical_integer(
    name: str, value: object, *, positive: bool = False
) -> int:
    """Return an exact unsigned 64-bit value, rejecting bool coercion."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ExternalAppendError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ExternalAppendError(f"{name} must be {qualifier}")
    if result >= _U64_LIMIT:
        raise ExternalAppendError(f"{name} exceeds uint64")
    return result


def copy_registration(value: object) -> ArenaRegistration:
    """Deeply reconstruct an arena registration as immutable value state."""

    if not isinstance(value, ArenaRegistration):
        raise ExternalAppendError(
            "structured arena registration must be an ArenaRegistration"
        )
    try:
        return ArenaRegistration(
            value.engine_epoch,
            value.pool_epoch,
            value.pool_id,
            value.class_id,
            value.backend_domain,
            value.page_count,
            value.page_tokens,
            value.token_bytes,
            value.backend_base_index,
            value.first_page_id,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ExternalAppendError(
            "structured arena registration is invalid"
        ) from error


def copy_write(write: ExternalTokenWrite) -> ExternalTokenWrite:
    """Copy a public write into non-aliased, validated value state."""

    if not isinstance(write, ExternalTokenWrite):
        raise ExternalAppendError(
            "external append batch must contain ExternalTokenWrite values"
        )
    try:
        context = write.context
        destination = write.destination
        request = context.request
        transaction = context.transaction
        page = destination.page
        if (
            not isinstance(context, OperationContext)
            or context.operation is not DataPlaneOperation.APPEND
            or not isinstance(request, RequestLease)
            or not isinstance(transaction, StepLease)
            or not isinstance(destination, BackendTokenAddress)
            or not isinstance(page, PageLease)
        ):
            raise TypeError("external write has an invalid append identity")
        return ExternalTokenWrite(
            OperationContext(
                RequestLease(
                    request.engine_epoch, request.slot, request.generation
                ),
                StepLease(
                    transaction.engine_epoch,
                    transaction.slot,
                    transaction.generation,
                ),
                DataPlaneOperation.APPEND,
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
    except (AttributeError, TypeError, ValueError) as error:
        raise ExternalAppendError(
            "external append write changed after construction"
        ) from error


def copy_page(address: BackendPageAddress) -> BackendPageAddress:
    """Copy an exact generation-bearing page address."""

    if not isinstance(address, BackendPageAddress):
        raise ExternalAppendError(
            "last-use entries must be BackendPageAddress values"
        )
    try:
        page = address.page
        if not isinstance(page, PageLease):
            raise TypeError("page lease has an invalid type")
        return BackendPageAddress(
            PageLease(
                page.engine_epoch,
                page.pool_epoch,
                page.generation,
                page.page_id,
                page.pool_id,
            ),
            address.class_id,
            address.backend_domain,
            address.backend_index,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ExternalAppendError("last-use page is invalid") from error


def device_key(value: object) -> tuple[str, int | None]:
    """Normalize a supported device into a comparison-safe identity."""

    encoded = str(value).lower()
    if encoded == "cpu" or encoded.startswith("cpu:"):
        return "cpu", None
    if encoded == "cuda":
        return "cuda", None
    if encoded.startswith("cuda:"):
        try:
            index = int(encoded.split(":", 1)[1])
        except ValueError as error:
            raise ExternalAppendError("CUDA device is invalid") from error
        if index < 0:
            raise ExternalAppendError("CUDA device is invalid")
        return "cuda", index
    raise ExternalAppendError(
        "structured arenas require a CPU or CUDA device"
    )


def compatible_device(
    left: tuple[str, int | None], right: tuple[str, int | None]
) -> bool:
    """Compatibility helper for comparing normalized device identities."""

    return left == right


def stream_key(stream: object) -> tuple[object, ...] | None:
    """Return a stable CUDA-stream identity when the wrapper exposes one."""

    handle = getattr(stream, "cuda_stream", _MISSING)
    if handle is _MISSING:
        handle = getattr(stream, "stream_id", _MISSING)
    if handle is _MISSING:
        return None
    try:
        handle = int(handle)
    except (TypeError, ValueError):
        return None
    return handle, str(getattr(stream, "device", ""))


def same_stream(actual: object | None, expected: object | None) -> bool:
    """Compare stream identity without trusting wrapper equality."""

    if actual is expected:
        return True
    if actual is None or expected is None:
        return False
    left = stream_key(actual)
    right = stream_key(expected)
    return left is not None and left == right


__all__ = [
    "ExternalAppendError",
    "ExternalAppendPoisonedError",
    "canonical_integer",
    "compatible_device",
    "copy_page",
    "copy_registration",
    "copy_write",
    "device_key",
    "same_stream",
    "stream_key",
]
