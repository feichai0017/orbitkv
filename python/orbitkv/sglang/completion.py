"""Consume SGLang's public, decode-owned P/D completion observations."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.disaggregation.lifecycle import PDTransferEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CompletionReporter:
    client: Any
    instance_id: str
    device_id: int


_LOCK = threading.Lock()
_REPORTER: _CompletionReporter | None = None


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


def observe_pd_transfer(event: PDTransferEvent) -> None:
    """Record terminal observations; quarantine is not a transfer completion."""
    if event.phase not in {"completed", "failed", "cancelled"}:
        return
    with _LOCK:
        reporter = _REPORTER
    if reporter is None:
        return
    try:
        reporter.client.observe_prefill_to_decode_completion(
            reporter.instance_id,
            reporter.device_id,
            event.source_endpoint,
            int(event.transfer_id[:16], 16),
            event.logical_bytes,
            event.logical_bytes if event.phase == "completed" else 0,
            event.fragment_count,
            event.elapsed_ns,
            event.logical_bytes,
            event.queue_depth,
            1,
            0,
            0,
            admitted=True,
            outcome=event.phase,
            representation="raw",
        )
    except Exception:
        logger.exception("Disabling SGLang P/D completion observations after report failure")
        unregister_completion_reporter(reporter.client)
