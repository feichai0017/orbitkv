"""Batch-scoped CUDA execution identity for runtime-session forwards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ffi.session_types import EngineBatchTicket


_CONTEXT = "_orbitkv_forward_execution_context"
_MISSING = object()


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


@dataclass(slots=True)
class ForwardExecutionContext:
    """Identity shared by one lowering, forward, and completion event."""

    device_module: Any
    device: Any
    stream: Any
    _ticket: EngineBatchTicket | None = field(default=None, init=False)
    _batch: Any = field(default=None, init=False, repr=False)

    @property
    def ticket(self) -> EngineBatchTicket | None:
        return self._ticket

    def bind_ticket(self, ticket: EngineBatchTicket) -> None:
        if not isinstance(ticket, EngineBatchTicket):
            raise RuntimeError("forward execution context received an invalid ticket")
        if self._ticket is not None:
            raise RuntimeError("forward execution context ticket was bound twice")
        self._ticket = ticket

    def require_ticket(self, ticket: Any) -> EngineBatchTicket:
        if self._ticket is None:
            raise RuntimeError("forward execution context has no batch ticket")
        if ticket is not self._ticket:
            raise RuntimeError(
                "forward execution context does not match the batch ticket"
            )
        return self._ticket

    def validate_current(
        self, *, device_module: Any | None = None, device: Any = _MISSING
    ) -> Any:
        module = self.device_module if device_module is None else device_module
        selected_device = self.device if device is _MISSING else device
        current_stream = getattr(module, "current_stream", None)
        if not callable(current_stream):
            raise RuntimeError(
                "forward execution device module has no current_stream"
            )
        current = current_stream(selected_device)
        if not same_stream(current, self.stream):
            raise RuntimeError("forward execution stream identity drifted")
        return current


def capture_forward_context(batch: Any) -> ForwardExecutionContext:
    """Capture and attach the sole execution context for one batch."""

    if getattr(batch, _CONTEXT, _MISSING) is not _MISSING:
        raise RuntimeError("batch already has a forward execution context")
    import torch

    device = batch.device
    device_module = torch.get_device_module(device)
    current_stream = getattr(device_module, "current_stream", None)
    if not callable(current_stream):
        raise RuntimeError(
            "forward execution device module has no current_stream"
        )
    stream = current_stream(device)
    if stream is None:
        raise RuntimeError("forward execution has no current stream")
    context = ForwardExecutionContext(device_module, device, stream)
    context._batch = batch
    setattr(batch, _CONTEXT, context)
    if getattr(batch, _CONTEXT, _MISSING) is not context:
        raise RuntimeError("batch did not retain its forward execution context")
    return context


def require_forward_context(batch: Any) -> ForwardExecutionContext:
    context = getattr(batch, _CONTEXT, _MISSING)
    if (
        type(context) is not ForwardExecutionContext
        or context._batch is not batch
    ):
        raise RuntimeError("batch has no valid forward execution context")
    return context


def clear_forward_context(
    batch: Any, context: ForwardExecutionContext
) -> None:
    if getattr(batch, _CONTEXT, _MISSING) is not context:
        raise RuntimeError(
            "batch forward execution context changed before cleanup"
        )
    delattr(batch, _CONTEXT)
    if getattr(batch, _CONTEXT, _MISSING) is not _MISSING:
        raise RuntimeError("batch forward execution context cleanup failed")
    context._batch = None


__all__ = [
    "ForwardExecutionContext",
    "capture_forward_context",
    "clear_forward_context",
    "require_forward_context",
    "same_stream",
    "stream_key",
]
