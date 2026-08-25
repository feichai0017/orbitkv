"""SGLang lifecycle coordination for the structured external-write SPI."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..runtime import BatchRecord, FailStopped, LoweringPlan
from . import state as _state


_TICKET = "_orbitkv_external_ticket"
_MANIFEST = "_orbitkv_external_manifest"


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


def prepare_external_writes(
    batch: Any,
    batch_record: BatchRecord,
    plans: Sequence[LoweringPlan],
    locations: Mapping[int, Any],
) -> None:
    """Reserve exact structured destinations before candidate mutation."""

    if _state._DATA_PLANE is None:
        return
    from .structured_arena import build_external_append_manifest

    manifest = build_external_append_manifest(
        batch_record,
        plans,
        locations,
        _state._runtime(),
        _state._config(),
        _state._STRUCTURED_ARENAS,
    )
    ticket = _state._data_plane().prepare_external_append(
        manifest.writes,
        completion_domain=_state._data_plane().completion_domain,
    )
    try:
        setattr(batch, _TICKET, ticket)
        setattr(batch, _MANIFEST, manifest)
    except Exception as error:
        _state._data_plane().poison(
            f"SGLang lost an issued external-write ticket: {error}"
        )
        raise


def poison_external_writes(batch: Any, reason: str) -> None:
    """Poison only after this batch crossed the ticket-issuance boundary."""

    if _state._DATA_PLANE is not None and getattr(batch, _TICKET, None) is not None:
        _state._DATA_PLANE.poison(reason)


def validate_external_launch(
    batch: Any, scheduler: Any, completion_domain: int
) -> None:
    if _state._DATA_PLANE is None:
        return
    ticket = getattr(batch, _TICKET, None)
    manifest = getattr(batch, _MANIFEST, None)
    if ticket is None or manifest is None:
        raise RuntimeError(
            "OrbitKV forward has no complete structured external-write ticket"
        )
    config = _state._config()
    primary = config.full_class or config.sliding_class
    primary_locations = getattr(batch, "out_cache_loc", None)
    if primary is None or primary_locations is None:
        raise RuntimeError("SGLang forward lost its primary KV write locations")
    locations: dict[int, Any] = {primary.class_id: primary_locations}
    if config.full_class is not None and config.sliding_class is not None:
        mapping = getattr(
            _state._ALLOCATOR, "full_to_swa_index_mapping", None
        )
        if mapping is None:
            raise RuntimeError("SGLang forward lost its Full-to-SWA LUT")
        locations[config.sliding_class.class_id] = mapping[primary_locations]
    stream = scheduler.device_module.current_stream(scheduler.device)
    _state._data_plane().validate_launch(
        ticket,
        manifest.expected_locations_by_class,
        locations,
        current_stream=stream,
        completion_domain=completion_domain,
    )


def register_forward_completion(
    batch: Any,
    batch_record: BatchRecord,
    runtime: Any,
    scheduler: Any,
    completion_domain: int,
) -> tuple[object, Any | None]:
    """Register legacy or structured completion and return its raw event."""

    if _state._DATA_PLANE is None:
        stream = scheduler.device_module.current_stream(scheduler.device)
        event = scheduler.device_module.Event()
        event.record(stream=stream)
        runtime.register_event(batch_record, event, completion_domain)
        return event, None

    ticket = getattr(batch, _TICKET, None)
    manifest = getattr(batch, _MANIFEST, None)
    if ticket is None or manifest is None:
        raise RuntimeError(
            "OrbitKV structured completion lost its external-write ticket"
        )
    _state._data_plane().record_external_data_ready(ticket)
    last_use = _state._data_plane().record_last_use(
        manifest.last_use_pages, completion_domain=completion_domain
    )
    event = _state._data_plane().event_for(last_use)
    if event is None:
        raise RuntimeError("SGLang CUDA last-use fence has no raw event")
    runtime.register_external_event(
        batch_record,
        last_use,
        event,
        adapter=_state._data_plane(),
    )
    return event, last_use


def clear_external_batch_state(batch: Any) -> None:
    for name in (_TICKET, _MANIFEST):
        if hasattr(batch, name):
            delattr(batch, name)


def run_scheduled_batch(
    original_fn: Any, scheduler: Any, batch: Any, *args: Any, **kwargs: Any
) -> Any:
    """Bracket one eager SGLang forward with exact completion evidence."""

    from .validation import _validate_batch

    runtime = _state._runtime()
    batch_record = getattr(batch, "_orbitkv_batch", None)
    state_records = (
        ()
        if _state._FIXED_STATE is None
        else tuple(getattr(batch, "_orbitkv_state_records", ()))
    )
    completion_domain = (
        _state._data_plane().completion_domain
        if _state._DATA_PLANE is not None
        else completion_domain_for_device(
            scheduler.device,
            getattr(getattr(scheduler, "ps", None), "gpu_id", None),
        )
    )
    try:
        _validate_batch(batch)
        runtime.poll()
        if batch.reqs and not isinstance(batch_record, BatchRecord):
            raise RuntimeError("OrbitKV forward has no submitted manager step")
        expected_keys = tuple(_state._request_key(req) for req in batch.reqs)
        if (
            not isinstance(batch_record, BatchRecord)
            or batch_record.keys != expected_keys
        ):
            raise RuntimeError(
                "OrbitKV forward records do not match batch request order"
            )
        validate_external_launch(batch, scheduler, completion_domain)
        runtime.mark_forward(batch_record)
    except Exception as error:
        poison_external_writes(
            batch, f"forward failed after external authorization: {error}"
        )
        if isinstance(batch_record, BatchRecord):
            runtime.forward_failed(batch_record, error)
        if _state._FIXED_STATE is not None:
            _state._FIXED_STATE.pre_forward_failed(error)
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            runtime.failure_reason or "pre-forward manager state became uncertain"
        ) from error

    try:
        result = original_fn(scheduler, batch, *args, **kwargs)
        if _state._FIXED_STATE is not None:
            state_records = _state._FIXED_STATE.records_for_schedule_batch(batch)
    except Exception as error:
        poison_external_writes(
            batch, f"model forward failed after external authorization: {error}"
        )
        runtime.forward_failed(batch_record, error)
        if _state._FIXED_STATE is not None:
            _state._FIXED_STATE.forward_failed(state_records, error)
        raise FailStopped(runtime.failure_reason or "forward failed") from error

    try:
        event, last_use = register_forward_completion(
            batch, batch_record, runtime, scheduler, completion_domain
        )
        if _state._FIXED_STATE is not None:
            if last_use is None:
                _state._FIXED_STATE.register_event(
                    expected_keys,
                    state_records,
                    event,
                    completion_domain,
                    scheduler.device_module,
                    scheduler.device,
                )
            else:
                _state._FIXED_STATE.register_external_event(
                    expected_keys,
                    state_records,
                    event,
                    last_use,
                    scheduler.device_module,
                    scheduler.device,
                    adapter=_state._data_plane(),
                )
        batch._orbitkv_batch = None
        clear_external_batch_state(batch)
        if hasattr(batch, "_orbitkv_state_records"):
            batch._orbitkv_state_records = ()
    except Exception as error:
        poison_external_writes(
            batch, f"external completion registration failed: {error}"
        )
        runtime.event_registration_failed(batch_record, error)
        if _state._FIXED_STATE is not None:
            _state._FIXED_STATE.event_registration_failed(state_records, error)
        raise FailStopped(
            runtime.failure_reason or "event registration failed"
        ) from error
    return result


__all__ = [
    "clear_external_batch_state",
    "completion_domain_for_device",
    "poison_external_writes",
    "prepare_external_writes",
    "register_forward_completion",
    "run_scheduled_batch",
    "validate_external_launch",
]
