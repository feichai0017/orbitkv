"""Native RuntimeSession forward and CUDA completion lifecycle."""

from __future__ import annotations

from typing import Any, NoReturn

from ..runtime import FailStopped
from . import state as _state


_SESSION_TICKET = "_orbitkv_session_ticket"
_MISSING = object()


def completion_domain_for_device(device: Any, gpu_id: Any = None) -> int:
    index = getattr(device, "index", None)
    if callable(index):
        index = None
    encoded = str(device)
    if index is None and ":" in encoded:
        suffix = encoded.rsplit(":", 1)[-1]
        index = int(suffix) if suffix.isdigit() else 0
    if index is None and not isinstance(gpu_id, bool) and isinstance(gpu_id, int):
        index = gpu_id
    if index is None:
        index = 0
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise RuntimeError("SGLang completion device index is invalid")
    domain = index + 1
    if domain >= 1 << 64:
        raise RuntimeError("SGLang completion domain exceeds uint64")
    return domain


def _fail_runtime_session_batch(
    runtime: Any, ticket: Any | None, reason: str, error: BaseException
) -> NoReturn:
    quarantine_error: BaseException | None = None
    if ticket is not None and runtime.failure_reason is None:
        try:
            runtime.quarantine_submitted(ticket)
        except FailStopped:
            pass
        except BaseException as caught:
            quarantine_error = caught
    if runtime.failure_reason is None:
        if quarantine_error is not None:
            reason += (
                "; submitted quarantine became uncertain: "
                f"{type(quarantine_error).__name__}: {quarantine_error}"
            )
        runtime.fail_stop(reason)
    raise FailStopped(runtime.failure_reason or reason) from error


def run_scheduled_batch(
    original_fn: Any, scheduler: Any, batch: Any, *args: Any, **kwargs: Any
) -> Any:
    """Fence one eager SGLang forward with session-owned completion."""

    from ..ffi.session_types import EngineBatchTicket
    from .execution_context import clear_forward_context, require_forward_context
    from .validation import _validate_batch

    runtime = _state._runtime()
    submitted_ticket: EngineBatchTicket | None = None
    try:
        raw_ticket = getattr(batch, _SESSION_TICKET, _MISSING)
        if isinstance(raw_ticket, EngineBatchTicket):
            submitted_ticket = raw_ticket
        context = require_forward_context(batch)
        if isinstance(context.ticket, EngineBatchTicket):
            submitted_ticket = context.ticket
        ticket = context.require_ticket(raw_ticket)
        submitted_ticket = ticket
        _validate_batch(batch)
        runtime.poll()
        completion_domain = completion_domain_for_device(
            scheduler.device,
            getattr(getattr(scheduler, "ps", None), "gpu_id", None),
        )
        context.validate_current(
            device_module=scheduler.device_module, device=scheduler.device
        )
    except BaseException as error:
        _fail_runtime_session_batch(
            runtime, submitted_ticket, f"SGLang session pre-forward failed: {error}", error
        )

    try:
        result = original_fn(scheduler, batch, *args, **kwargs)
    except BaseException as error:
        _fail_runtime_session_batch(
            runtime,
            submitted_ticket,
            f"SGLang session model forward failed: {error}",
            error,
        )

    try:
        context.validate_current(
            device_module=scheduler.device_module, device=scheduler.device
        )
        event = scheduler.device_module.Event()
        event.record(stream=context.stream)
        runtime.register_event(submitted_ticket, event, completion_domain)
    except BaseException as error:
        _fail_runtime_session_batch(
            runtime,
            submitted_ticket,
            f"SGLang session CUDA completion became uncertain: {error}",
            error,
        )

    try:
        if getattr(batch, _SESSION_TICKET, _MISSING) is not submitted_ticket:
            raise RuntimeError("SGLang session ticket changed before cleanup")
        delattr(batch, _SESSION_TICKET)
        if getattr(batch, _SESSION_TICKET, _MISSING) is not _MISSING:
            raise RuntimeError("SGLang session ticket cleanup did not remove the ticket")
        clear_forward_context(batch, context)
    except BaseException as error:
        _fail_runtime_session_batch(
            runtime,
            submitted_ticket,
            f"SGLang session batch cleanup became uncertain: {error}",
            error,
        )
    return result


__all__ = ["completion_domain_for_device", "run_scheduled_batch"]
