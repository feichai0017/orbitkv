from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable


_EVENTS: queue.Queue[dict[str, Any] | None] = queue.Queue()
_WRITER: threading.Thread | None = None


def _trace_path() -> Path:
    return Path(os.environ.get("ORBITKV_TRACE_PATH", "/tmp/orbitkv-sglang.jsonl"))


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _available(allocator: Any, method: str) -> int | None:
    callback = getattr(allocator, method, None)
    return None if callback is None else _integer(callback())


def _tensor_numel(value: Any) -> int | None:
    callback = getattr(value, "numel", None)
    return None if callback is None else _integer(callback())


def _writer_main() -> None:
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        while True:
            event = _EVENTS.get()
            if event is None:
                stream.flush()
                _EVENTS.task_done()
                return
            stream.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
            if event.get("event") == "plugin_loaded":
                stream.flush()
            _EVENTS.task_done()


def _stop_writer() -> None:
    global _WRITER

    if _WRITER is None:
        return
    _EVENTS.put(None)
    _EVENTS.join()
    _WRITER.join(timeout=5)
    _WRITER = None


def _emit(event: dict[str, Any]) -> None:
    _EVENTS.put(
        {
            "schema": "orbitkv.sglang-shadow.v1",
            "timestamp_ns": time.time_ns(),
            **event,
        }
    )


def _allocator_event(
    original_fn: Callable,
    allocator: Any,
    *args: Any,
    operation: str,
    **kwargs: Any,
):
    before_full = _available(allocator, "full_available_size")
    before_swa = _available(allocator, "swa_available_size")
    before_total = _available(allocator, "available_size")
    result = original_fn(allocator, *args, **kwargs)
    _emit(
        {
            "event": operation,
            "allocator_type": type(allocator).__name__,
            "page_size": _integer(getattr(allocator, "page_size", None)),
            "size_full": _integer(getattr(allocator, "size_full", None)),
            "size_swa": _integer(getattr(allocator, "size_swa", None)),
            "requested_tokens": (
                _integer(args[0]) if operation == "alloc" and args else None
            ),
            "input_tokens": _tensor_numel(args[0]) if args else None,
            "output_tokens": _tensor_numel(result),
            "full_available_before": before_full,
            "full_available_after": _available(allocator, "full_available_size"),
            "swa_available_before": before_swa,
            "swa_available_after": _available(allocator, "swa_available_size"),
            "available_before": before_total,
            "available_after": _available(allocator, "available_size"),
        }
    )
    return result


def register() -> None:
    global _WRITER

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if _WRITER is None:
        _WRITER = threading.Thread(
            target=_writer_main,
            name="orbitkv-sglang-shadow",
            daemon=True,
        )
        _WRITER.start()
        atexit.register(_stop_writer)

    target = "sglang.srt.mem_cache.allocator.swa.SWATokenToKVPoolAllocator"
    for method in ("alloc", "alloc_extend", "alloc_decode", "free", "free_swa"):
        operation = method

        def around(original_fn, allocator, *args, _operation=operation, **kwargs):
            return _allocator_event(
                original_fn,
                allocator,
                *args,
                operation=_operation,
                **kwargs,
            )

        HookRegistry.register(
            f"{target}.{method}",
            around,
            HookType.AROUND,
        )

    _emit(
        {
            "event": "plugin_loaded",
            "sglang_expected_revision": os.environ.get(
                "ORBITKV_SGLANG_REVISION",
                "095ec6c997bfdd25d3864cb0ce77a6562a934b96",
            ),
        }
    )
