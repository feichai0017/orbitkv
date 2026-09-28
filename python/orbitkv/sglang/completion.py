"""Decode-owned SGLang P/D completion observations."""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DecodeCompletionEvidence:
    source_endpoint: str
    transfer_generation: int
    logical_bytes: int
    fragment_count: int
    queue_depth: int
    queue_parallelism: int
    tent_inflight_bytes: int
    tent_bandwidth_bytes_per_second: int
    started_ns: int


@dataclass(frozen=True, slots=True)
class _CompletionReporter:
    client: Any
    instance_id: str
    device_id: int


_LOCK = threading.Lock()
_REPORTER: _CompletionReporter | None = None
_TRANSFER_GENERATIONS = itertools.count(1)


def register_completion_reporter(client: Any, instance_id: str, device_id: int) -> None:
    global _REPORTER
    if not instance_id or device_id < 0:
        raise ValueError("SGLang completion reporter requires a registered instance and device")
    with _LOCK:
        if _REPORTER is not None and _REPORTER.client is not client:
            raise RuntimeError("one SGLang process cannot register multiple completion reporters")
        _REPORTER = _CompletionReporter(client, instance_id, device_id)


def unregister_completion_reporter(client: Any) -> None:
    global _REPORTER
    with _LOCK:
        if _REPORTER is not None and _REPORTER.client is client:
            _REPORTER = None


def capture_decode_pages(
    result: Any,
    receiver: Any,
    kv_indices: Any,
    aux_index: int | None = None,
    state_indices: list[Any] | None = None,
    decode_prefix_len: int | None = None,
    device_kv_indices: Any | None = None,
) -> Any:
    del decode_prefix_len
    if _reporter() is None:
        return result
    indices = device_kv_indices if device_kv_indices is not None else kv_indices
    kv_count = len(indices)
    kv_item_lens = tuple(int(value) for value in receiver.kv_mgr.kv_args.kv_item_lens)
    logical_bytes = kv_count * sum(kv_item_lens)
    fragments = kv_count * len(kv_item_lens)
    if aux_index is not None:
        aux_item_lens = tuple(int(value) for value in receiver.kv_mgr.kv_args.aux_item_lens)
        logical_bytes += sum(aux_item_lens)
        fragments += len(aux_item_lens)
    for group_indices, item_lens in zip(
        state_indices or (),
        receiver.kv_mgr.kv_args.state_item_lens,
        strict=False,
    ):
        count = len(group_indices)
        logical_bytes += count * sum(int(value) for value in item_lens)
        fragments += count * len(item_lens)
    receiver._orbitkv_completion_evidence = _DecodeCompletionEvidence(
        source_endpoint=str(receiver.bootstrap_addr),
        transfer_generation=0,
        logical_bytes=logical_bytes,
        fragment_count=fragments,
        queue_depth=0,
        queue_parallelism=1,
        tent_inflight_bytes=0,
        tent_bandwidth_bytes_per_second=0,
        started_ns=0,
    )
    return result


def capture_handoff_admission(result: Any, queue: Any, decode_req: Any) -> Any:
    receiver = decode_req.kv_receiver
    evidence = getattr(receiver, "_orbitkv_completion_evidence", None)
    if evidence is None:
        return result
    inflight_bytes, bandwidth = _tent_pressure(receiver.kv_mgr.engine)
    receiver._orbitkv_completion_evidence = replace(
        evidence,
        transfer_generation=next(_TRANSFER_GENERATIONS),
        queue_depth=max(1, len(queue.queue)),
        tent_inflight_bytes=inflight_bytes,
        tent_bandwidth_bytes_per_second=bandwidth,
        started_ns=time.monotonic_ns(),
    )
    return result


def observe_decode_ready(original_fn: Any, queue: Any, decode_req: Any) -> Any:
    receiver = decode_req.kv_receiver
    try:
        result = original_fn(queue, decode_req)
    except BaseException:
        _report(receiver, "failed")
        raise
    outcome = "failed" if getattr(decode_req.req, "to_finish", None) is not None else "completed"
    _report(receiver, outcome)
    return result


def mark_decode_abort(result: Any, receiver: Any) -> Any:
    receiver._orbitkv_completion_outcome = "cancelled"
    return result


def observe_decode_failure(original_fn: Any, receiver: Any) -> Any:
    try:
        return original_fn(receiver)
    except BaseException:
        outcome = getattr(receiver, "_orbitkv_completion_outcome", "failed")
        if outcome != "cancelled":
            _report(receiver, outcome)
        raise


def observe_deferred_release(original_fn: Any, queue: Any, decode_req: Any, index: int) -> Any:
    receiver = decode_req.kv_receiver
    result = original_fn(queue, decode_req, index)
    _report(receiver, "cancelled")
    return result


def _reporter() -> _CompletionReporter | None:
    with _LOCK:
        return _REPORTER


def _report(receiver: Any, outcome: str) -> None:
    reporter = _reporter()
    evidence = getattr(receiver, "_orbitkv_completion_evidence", None)
    if reporter is None or evidence is None or evidence.transfer_generation == 0:
        return
    with _LOCK:
        if getattr(receiver, "_orbitkv_completion_reported", False):
            return
        receiver._orbitkv_completion_reported = True
    elapsed_ns = max(1, time.monotonic_ns() - evidence.started_ns)
    try:
        reporter.client.observe_prefill_to_decode_completion(
            reporter.instance_id,
            reporter.device_id,
            evidence.source_endpoint,
            evidence.transfer_generation,
            evidence.logical_bytes,
            evidence.logical_bytes if outcome == "completed" else 0,
            evidence.fragment_count,
            elapsed_ns,
            evidence.logical_bytes,
            evidence.queue_depth,
            evidence.queue_parallelism,
            evidence.tent_inflight_bytes,
            evidence.tent_bandwidth_bytes_per_second,
            admitted=True,
            outcome=outcome,
            representation="raw",
        )
    except Exception:
        logger.exception("Disabling SGLang P/D completion observations after report failure")
        unregister_completion_reporter(reporter.client)


def _tent_pressure(engine: Any) -> tuple[int, int]:
    try:
        stats = engine.nic_load_stats()
    except Exception:
        logger.exception("Could not read SGLang TENT NIC pressure")
        return 0, 0
    return (
        sum(max(0, int(stat[1])) for stat in stats),
        sum(max(0, int(float(stat[2]))) for stat in stats),
    )


__all__ = [
    "capture_decode_pages",
    "capture_handoff_admission",
    "mark_decode_abort",
    "observe_decode_failure",
    "observe_decode_ready",
    "observe_deferred_release",
    "register_completion_reporter",
    "unregister_completion_reporter",
]
