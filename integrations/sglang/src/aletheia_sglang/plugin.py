"""Observe the real SGLang execution path for AletheiaRT.

SGLang imports this module through the ``sglang.srt.plugins`` entry-point.
Registration is intentionally cheap: no CUDA module, model, or kernel provider
is imported here. A JSONL trace is written only when ``ALETHEIA_TRACE``
is set.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

_TRACE_ENV = "ALETHEIA_TRACE"
_write_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def _string(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _integers(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    # Only callers' explicitly CPU-resident mirrors reach here. Never call
    # .cpu() in this hook: that would synchronize the serving hot path.
    tolist = getattr(value, "tolist", None)
    if tolist is None:
        return []
    result = tolist()
    if isinstance(result, list):
        return [int(item) for item in result]
    return [int(result)]


def _phase(value: Any) -> str | None:
    text = (_string(value) or "").lower()
    if "verify" in text:
        return "verify"
    if "decode" in text:
        return "decode"
    if "extend" in text or "prefill" in text:
        return "prefill"
    return None


def _range(values: list[int]) -> dict[str, int] | None:
    return {"min": min(values), "max": max(values)} if values else None


def _batch_facts(batch: Any, forward_batch: Any) -> dict[str, Any]:
    source = batch if batch is not None else forward_batch
    if source is None:
        return {"batch_size": None, "phase": None, "workload": None}

    requests = getattr(source, "reqs", None)
    batch_size = len(requests) if requests is not None else getattr(source, "batch_size", None)
    mode = getattr(source, "forward_mode", None)
    if mode is None and forward_batch is not None:
        mode = getattr(forward_batch, "forward_mode", None)
    phase = _phase(mode)
    seq_lens = _integers(getattr(source, "seq_lens_cpu", None))
    raw_extend_lens = getattr(source, "extend_lens", None)
    if raw_extend_lens is None:
        raw_extend_lens = getattr(source, "extend_seq_lens_cpu", None)
    extend_lens = _integers(raw_extend_lens)
    if phase == "decode":
        rows = [1] * int(batch_size or 0)
    elif phase == "prefill":
        rows = extend_lens
    else:
        rows = []

    contexts = []
    if seq_lens and rows and len(seq_lens) == len(rows):
        contexts = [max(length - added, 0) for length, added in zip(seq_lens, rows)]
    elif seq_lens:
        contexts = seq_lens

    workload = None
    if phase is not None and batch_size and rows and contexts:
        workload = {
            "phase": phase,
            "batch": int(batch_size),
            "rows_per_sequence": _range(rows),
            "context_tokens": _range(contexts),
        }
    return {"batch_size": batch_size, "phase": phase, "workload": workload}


def _append(event: dict[str, Any]) -> None:
    target = os.environ.get(_TRACE_ENV)
    if not target:
        return
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with _write_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except OSError:
        # Observability must never turn a successful inference into a failure.
        _logger.exception("Could not append AletheiaRT SGLang trace")


def _around_forward_batch_generation(
    original: Callable[..., Any],
    worker: Any,
    batch: Any = None,
    forward_batch: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    started = time.perf_counter_ns()
    outcome = "ok"
    try:
        result = original(worker, batch, forward_batch, *args, **kwargs)
        return result
    except BaseException:
        outcome = "error"
        raise
    finally:
        event = {
            "schema_version": 1,
            "event": "sglang.forward_batch_generation",
            "timestamp_micros": time.time_ns() // 1_000,
            "host_elapsed_micros": (time.perf_counter_ns() - started) // 1_000,
            "outcome": outcome,
            "worker_type": type(worker).__qualname__,
            **_batch_facts(batch, forward_batch),
        }
        _append(event)


def register() -> None:
    """Register the trace hook with the source-pinned SGLang runtime."""

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
        _around_forward_batch_generation,
        HookType.AROUND,
    )
