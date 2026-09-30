"""Prefill layer writes and completion notification, with physical-drain accounting."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from orbitkv.logging_utils import get_connector_logger
from orbitkv.vllm.pd.layout import LayerBlockSlices
from orbitkv.vllm.pd.mooncake import MooncakePort

logger = get_connector_logger()
T = TypeVar("T")


@dataclass(frozen=True)
class _LayerPushTask:
    transfer: MooncakePort
    req_id: str
    layer_idx: int
    block_slices: list[LayerBlockSlices]
    request_generation: int
    event: Any = None


@dataclass(frozen=True)
class _PushFinalizeTask:
    transfer: MooncakePort
    req_ids: tuple[str, ...]
    target_request_id: str
    num_blocks: int
    chunk_count: int
    first_save_ts_ns: int | None
    schedule_queued_ts_ns: int
    transfer_bytes: int


class _InflightTaskRunner(Generic[T]):
    """Task pool that tracks inflight count, errors, and supports ``wait_all``.

    Shared by the prefill push sender and finalizer. Uses a ``Condition`` so
    callers can block until all (or a subset of) submitted tasks drain.
    Task exceptions are captured and reported after the requested work drains;
    an error does not establish that another task has stopped accessing pages.

    Subclasses implement ``_run(task)`` (the work body) and may override
    ``_on_submit_locked`` / ``_on_finish_locked`` to maintain per-request state.
    ``_set_inflight_metric_locked`` is called whenever the inflight count
    changes so subclasses can publish their own gauge.
    """

    def __init__(
        self,
        thread_name_prefix: str,
        metrics: Any | None = None,
        max_workers: int = 16,
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix=thread_name_prefix
        )
        self._condition = threading.Condition()
        self._inflight = 0
        self._error: BaseException | None = None
        self._closed = False
        self._metrics = metrics

    def _submit(self, task: T, closed_message: str) -> bool:
        """Register and schedule a task.

        Returns False if the subclass's ``_on_submit_locked`` rejected the task
        (e.g. dedup); raises if the pool is closed or holds a pending error.
        """
        with self._condition:
            if self._closed:
                raise RuntimeError(closed_message)
            if self._error is not None:
                raise self._error
            if not self._on_submit_locked(task):
                return False
            self._inflight += 1
            self._set_inflight_metric_locked()
        try:
            self._executor.submit(self._execute, task)
        except BaseException:
            with self._condition:
                self._finish_locked(task)
            raise
        return True

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0:
                self._condition.wait()
            self._raise_pending_error_locked()

    def is_idle(self) -> bool:
        with self._condition:
            return self._inflight == 0 and self._error is None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, task: T) -> None:
        try:
            self._run(task)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            self._on_error(task, exc)
        finally:
            with self._condition:
                self._finish_locked(task)

    def _finish_locked(self, task: T) -> None:
        self._inflight -= 1
        self._on_finish_locked(task)
        self._set_inflight_metric_locked()
        self._condition.notify_all()

    def _raise_pending_error_locked(self) -> None:
        if self._error is not None:
            error = self._error
            self._error = None
            raise error

    def _run(self, task: T) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _on_submit_locked(self, task: T) -> bool:
        """Update per-request state under the condition lock. Return False to
        skip scheduling (e.g. duplicate task)."""
        return True

    def _on_finish_locked(self, task: T) -> None:
        """Update per-request state when a task completes (lock held)."""

    def _on_error(self, task: T, exc: BaseException) -> None:
        """Side effects on task failure (e.g. metrics), outside the lock."""

    def _set_inflight_metric_locked(self) -> None:
        """Publish the inflight gauge. Default no-op."""


def _elapsed_ms(start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if start_ts_ns is None or end_ts_ns is None:
        return 0.0
    return (end_ts_ns - start_ts_ns) / 1_000_000


def _gbps(bytes_total: int, start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if bytes_total <= 0 or start_ts_ns is None or end_ts_ns is None or end_ts_ns <= start_ts_ns:
        return 0.0
    return bytes_total * 8 / ((end_ts_ns - start_ts_ns) / 1_000_000_000) / 1e9


def _pct(value: float, total: float) -> float:
    if value <= 0 or total <= 0:
        return 0.0
    return value / total * 100


def _transfer_link_gbps(transfer: MooncakePort | None) -> float:
    if transfer is None:
        return 0.0
    try:
        return transfer.aggregated_link_speed() / 1e9
    except Exception:
        logger.exception("[PdConnector] failed to read Mooncake link speed")
        return 0.0


def _transfer_write_stats(
    transfer: MooncakePort,
    req_ids: tuple[str, ...],
    *,
    fallback_bytes: int,
    fallback_start_ts_ns: int | None,
    fallback_end_ts_ns: int | None,
) -> dict[str, float | int | bool]:
    stats_by_req = []
    for req_id in req_ids:
        try:
            stats_by_req.append(transfer.write_stats(req_id))
        except AttributeError:
            continue
        except Exception:
            logger.exception("[PdConnector] failed to read Mooncake write stats req=%s", req_id)
    if not stats_by_req:
        return {
            "submitted": 0,
            "completed": 0,
            "errors": 0,
            "bytes": fallback_bytes,
            "has_submit": False,
            "has_complete": False,
            "submit_span_ms": 0.0,
            "xfer_window_ms": 0.0,
            "completion_tail_ms": 0.0,
            "window_gbps": 0.0,
            "gbps": _gbps(fallback_bytes, fallback_start_ts_ns, fallback_end_ts_ns),
        }

    bytes_total = sum(int(stats.get("bytes", 0)) for stats in stats_by_req)
    xfer_window_ms = max(float(stats.get("xfer_window_ms", 0.0)) for stats in stats_by_req)
    window_gbps = bytes_total * 8 / (xfer_window_ms / 1000.0) / 1e9 if xfer_window_ms > 0 else 0.0
    write_latency_sum_ms = sum(
        float(stats.get("write_latency_sum_ms", 0.0)) for stats in stats_by_req
    )
    active_gbps = (
        bytes_total * 8 / (write_latency_sum_ms / 1000.0) / 1e9
        if write_latency_sum_ms > 0
        else window_gbps
    )
    return {
        "submitted": sum(int(stats.get("submitted", 0)) for stats in stats_by_req),
        "completed": sum(int(stats.get("completed", 0)) for stats in stats_by_req),
        "errors": sum(int(stats.get("errors", 0)) for stats in stats_by_req),
        "bytes": bytes_total,
        "has_submit": any(bool(stats.get("has_submit", False)) for stats in stats_by_req),
        "has_complete": any(bool(stats.get("has_complete", False)) for stats in stats_by_req),
        "submit_span_ms": max(float(stats.get("submit_span_ms", 0.0)) for stats in stats_by_req),
        "first_complete_ms": max(
            float(stats.get("first_complete_ms", 0.0)) for stats in stats_by_req
        ),
        "xfer_window_ms": xfer_window_ms,
        "completion_tail_ms": max(
            float(stats.get("completion_tail_ms", 0.0)) for stats in stats_by_req
        ),
        "write_latency_sum_ms": write_latency_sum_ms,
        "write_latency_max_ms": max(
            float(stats.get("write_latency_max_ms", 0.0)) for stats in stats_by_req
        ),
        "window_gbps": window_gbps,
        "gbps": active_gbps,
    }


# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------


class _AsyncLayerPushSender(_InflightTaskRunner["_LayerPushTask"]):
    def __init__(self, metrics: Any | None = None, max_workers: int = 16) -> None:
        super().__init__("pd-transfer-push", metrics=metrics, max_workers=max_workers)
        self._inflight_by_req: dict[str, int] = {}
        self._cancelled: set[str] = set()

    def submit(self, task: _LayerPushTask) -> None:
        self._submit(task, "PdConnector Mooncake push sender is closed")

    def wait_req(self, req_id: str) -> None:
        with self._condition:
            while self._inflight_by_req.get(req_id, 0) > 0:
                self._condition.wait()
            self._raise_pending_error_locked()

    def cancel(self, req_id: str) -> None:
        self.cancel_many((req_id,))

    def cancel_many(self, req_ids: tuple[str, ...]) -> None:
        with self._condition:
            self._cancelled.update(
                req_id for req_id in req_ids if self._inflight_by_req.get(req_id, 0) > 0
            )
            self._condition.notify_all()

    def _on_submit_locked(self, task: _LayerPushTask) -> bool:
        self._inflight_by_req[task.req_id] = self._inflight_by_req.get(task.req_id, 0) + 1
        return True

    def _run(self, task: _LayerPushTask) -> None:
        with self._condition:
            cancelled = task.req_id in self._cancelled
        if not cancelled:
            _run_layer_push(task)

    def _on_finish_locked(self, task: _LayerPushTask) -> None:
        remaining = self._inflight_by_req.get(task.req_id, 0) - 1
        if remaining > 0:
            self._inflight_by_req[task.req_id] = remaining
        else:
            self._inflight_by_req.pop(task.req_id, None)
            self._cancelled.discard(task.req_id)

    def _set_inflight_metric_locked(self) -> None:
        if self._metrics is not None:
            self._metrics.set_prefill_inflight_push_tasks(self._inflight)


def _run_layer_push(task: _LayerPushTask) -> None:
    if task.event is not None:
        task.event.synchronize()
    task.transfer.push_layer(
        task.req_id,
        task.layer_idx,
        task.block_slices,
        request_generation=task.request_generation,
    )


class _AsyncPushFinalizer(_InflightTaskRunner["_PushFinalizeTask"]):
    def __init__(
        self,
        push_sender: _AsyncLayerPushSender,
        metrics: Any | None = None,
        max_workers: int = 16,
    ) -> None:
        super().__init__("pd-transfer-finalize", metrics=metrics, max_workers=max_workers)
        self._push_sender = push_sender
        self._submitted: set[tuple[str, ...]] = set()
        self._cancelled: set[str] = set()

    def submit(self, task: _PushFinalizeTask) -> None:
        self._submit(task, "PdConnector Mooncake push finalizer is closed")

    def cancel_many(self, req_ids: tuple[str, ...]) -> None:
        with self._condition:
            submitted_req_ids = {
                submitted_req_id for submitted in self._submitted for submitted_req_id in submitted
            }
            self._cancelled.update(req_id for req_id in req_ids if req_id in submitted_req_ids)
            self._condition.notify_all()

    def _on_submit_locked(self, task: _PushFinalizeTask) -> bool:
        if task.req_ids in self._submitted:
            return False
        self._submitted.add(task.req_ids)
        return True

    def _run(self, task: _PushFinalizeTask) -> None:
        completed = False
        for req_id in task.req_ids:
            if self._is_cancelled(req_id):
                continue
            self._push_sender.wait_req(req_id)
            if self._is_cancelled(req_id):
                continue
            task.transfer.push_done(req_id)
            completed = True
        done_ts_ns = time.time_ns()
        write_stats = _transfer_write_stats(
            task.transfer,
            task.req_ids,
            fallback_bytes=task.transfer_bytes,
            fallback_start_ts_ns=task.first_save_ts_ns,
            fallback_end_ts_ns=done_ts_ns,
        )
        if completed and self._metrics is not None:
            push_gbps = float(write_stats.get("gbps", 0.0))
            self._metrics.record_prefill_push(
                duration_s=(done_ts_ns - task.schedule_queued_ts_ns) / 1_000_000_000,
                first_save_to_done_s=(
                    (done_ts_ns - task.first_save_ts_ns) / 1_000_000_000
                    if task.first_save_ts_ns is not None
                    else None
                ),
                blocks=task.num_blocks,
                bytes_total=task.transfer_bytes,
                gbps=push_gbps,
                success=True,
            )
        push_gbps = float(write_stats.get("gbps", 0.0))
        link_gbps = _transfer_link_gbps(task.transfer)
        logger.info(
            "[PdConnector] P Mooncake done reqs=%s target_req=%s chunks=%d blocks=%d "
            "transfer_bytes=%d save_to_done_ms=%.3f schedule_to_done_ms=%.3f "
            "submit_span_ms=%.3f xfer_window_ms=%.3f completion_tail_ms=%.3f "
            "write_latency_sum_ms=%.3f write_latency_max_ms=%.3f "
            "writes=%d/%d errors=%d gbps=%.2f window_gbps=%.2f "
            "link_gbps=%.2f link_util_pct=%.2f ts_ns=%d",
            list(task.req_ids),
            task.target_request_id,
            task.chunk_count,
            task.num_blocks,
            task.transfer_bytes,
            _elapsed_ms(task.first_save_ts_ns, done_ts_ns),
            _elapsed_ms(task.schedule_queued_ts_ns, done_ts_ns),
            float(write_stats.get("submit_span_ms", 0.0)),
            float(write_stats.get("xfer_window_ms", 0.0)),
            float(write_stats.get("completion_tail_ms", 0.0)),
            float(write_stats.get("write_latency_sum_ms", 0.0)),
            float(write_stats.get("write_latency_max_ms", 0.0)),
            int(write_stats.get("completed", 0)),
            int(write_stats.get("submitted", 0)),
            int(write_stats.get("errors", 0)),
            push_gbps,
            float(write_stats.get("window_gbps", 0.0)),
            link_gbps,
            _pct(push_gbps, link_gbps),
            done_ts_ns,
        )

    def _on_error(self, task: _PushFinalizeTask, exc: BaseException) -> None:
        logger.exception(
            "[PdConnector] P finalize failed reqs=%s target_req=%s chunks=%d blocks=%d bytes=%d",
            list(task.req_ids),
            task.target_request_id,
            task.chunk_count,
            task.num_blocks,
            task.transfer_bytes,
        )
        if self._metrics is not None:
            self._metrics.record_prefill_push(
                duration_s=(time.time_ns() - task.schedule_queued_ts_ns) / 1_000_000_000,
                first_save_to_done_s=None,
                blocks=task.num_blocks,
                bytes_total=task.transfer_bytes,
                gbps=None,
                success=False,
            )

    def _is_cancelled(self, req_id: str) -> bool:
        with self._condition:
            return req_id in self._cancelled

    def _on_finish_locked(self, task: _PushFinalizeTask) -> None:
        self._submitted.discard(task.req_ids)
        for req_id in task.req_ids:
            self._cancelled.discard(req_id)

    def _set_inflight_metric_locked(self) -> None:
        if self._metrics is not None:
            self._metrics.set_prefill_inflight_finalize_tasks(self._inflight)
