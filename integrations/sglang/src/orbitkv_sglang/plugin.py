from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


_EVENTS: queue.Queue[dict[str, Any] | None] = queue.Queue()
_WRITER: threading.Thread | None = None
_POLICY: dict[str, Any] | None = None
_OWNER: "OwnerClient | None" = None


class OwnerClient:
    def __init__(self, orbitkv_bin: str, plan_path: str):
        self._process = subprocess.Popen(
            [orbitkv_bin, "serve-sglang-owner", plan_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("OrbitKV owner did not expose command pipes")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._lock = threading.Lock()

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                stderr = (
                    self._process.stderr.read()
                    if self._process.stderr is not None
                    else ""
                )
                raise RuntimeError(
                    f"OrbitKV owner exited with {self._process.returncode}: {stderr}"
                )
            self._stdin.write(
                json.dumps(command, separators=(",", ":"), sort_keys=True)
            )
            self._stdin.write("\n")
            self._stdin.flush()
            line = self._stdout.readline()
            if not line:
                stderr = (
                    self._process.stderr.read()
                    if self._process.stderr is not None
                    else ""
                )
                raise RuntimeError(f"OrbitKV owner closed its response stream: {stderr}")
            response = json.loads(line)
            if response.get("status") == "error":
                raise RuntimeError(
                    f"OrbitKV owner rejected command "
                    f"[{response.get('code')}]: {response.get('message')}"
                )
            return response

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        self._stdin.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)


def _owner_enabled() -> bool:
    return os.environ.get("ORBITKV_SGLANG_OWNING", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _trace_path() -> Path:
    return Path(os.environ.get("ORBITKV_TRACE_PATH", "/tmp/orbitkv-sglang.jsonl"))


def _trace_allocations_enabled() -> bool:
    return os.environ.get("ORBITKV_TRACE_ALLOCATIONS", "1").lower() not in (
        "0",
        "false",
        "no",
    )


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
            batch = [event]
            while True:
                try:
                    batch.append(_EVENTS.get_nowait())
                except queue.Empty:
                    break

            should_stop = False
            for event in batch:
                if event is None:
                    should_stop = True
                else:
                    stream.write(
                        json.dumps(event, separators=(",", ":"), sort_keys=True)
                    )
                    stream.write("\n")
                _EVENTS.task_done()
            if batch:
                stream.flush()
            if should_stop:
                return


def _stop_writer() -> None:
    global _WRITER

    if _WRITER is None:
        return
    _EVENTS.put(None)
    _EVENTS.join()
    _WRITER.join(timeout=5)
    _WRITER = None


def _stop_owner() -> None:
    global _OWNER

    if _OWNER is None:
        return
    _OWNER.close()
    _OWNER = None


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


def _load_policy() -> dict[str, Any] | None:
    plan_path = os.environ.get("ORBITKV_SGLANG_POLICY")
    if not plan_path:
        return None
    orbitkv_bin = os.environ.get("ORBITKV_BIN", "orbitkv")
    command = [orbitkv_bin, "emit-sglang-policy", plan_path]
    if eviction_interval := os.environ.get("ORBITKV_SGLANG_EVICTION_INTERVAL"):
        command.extend(["--eviction-interval", eviction_interval])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    policy = json.loads(completed.stdout)
    if policy.get("schema") != "orbitkv.sglang-policy.v1":
        raise ValueError(f"unsupported OrbitKV SGLang policy: {policy!r}")
    return policy


def _require_owner() -> OwnerClient:
    if _OWNER is None:
        raise RuntimeError("OrbitKV owning mode is not initialized")
    return _OWNER


def _own_swa_reclamation(
    _original_fn: Callable,
    batch: Any,
    req: Any,
    pre_len: int,
):
    if batch.enable_overlap:
        raise RuntimeError(
            "OrbitKV owning mode currently requires disable_overlap_schedule"
        )
    if not batch.tree_cache.is_chunk_cache():
        raise RuntimeError(
            "OrbitKV owning mode currently requires disable_radix_cache"
        )
    if not batch.spec_algorithm.is_none():
        raise RuntimeError(
            "OrbitKV owning mode currently requires speculative decoding disabled"
        )
    if req.kv is None:
        return None

    owner = _require_owner()
    response = owner.command(
        {
            "op": "plan_reclamation",
            "request_id": str(req.rid),
            "observed_evicted_seqlen": int(req.kv.swa_evicted_seqlen),
            "semantic_frontier": int(pre_len),
            "execution_epoch": int(
                max(
                    getattr(req, "decode_batch_idx", 0),
                    getattr(req, "extend_batch_idx", 0),
                )
            ),
            "cache_kind": "chunk",
        }
    )
    certificate = response.get("certificate")
    if certificate is None:
        return None
    if certificate.get("schema") != "orbitkv.sglang-retirement-certificate.v1":
        raise RuntimeError(f"unsupported OrbitKV certificate: {certificate!r}")
    if certificate.get("plan_fingerprint") != _POLICY.get("plan_fingerprint"):
        raise RuntimeError(
            "OrbitKV certificate fingerprint does not match the loaded policy"
        )
    if certificate.get("page_tokens") != batch.tree_cache.page_size:
        raise RuntimeError(
            "OrbitKV certificate page size does not match SGLang's physical pool"
        )
    token_start = int(certificate["token_start"])
    token_end = int(certificate["token_end_exclusive"])
    if token_start != int(req.kv.swa_evicted_seqlen):
        raise RuntimeError(
            "OrbitKV certificate does not begin at SGLang's committed SWA frontier"
        )
    if token_end <= token_start:
        raise RuntimeError("OrbitKV certificate contains an empty retirement range")

    _emit_owner_certificate(certificate)
    free_slots = batch.req_to_token_pool.req_to_token[
        req.req_pool_idx, token_start:token_end
    ]
    batch.token_to_kv_pool_allocator.free_swa(free_slots)
    pending = getattr(batch, "_orbitkv_pending_certificates", None)
    if pending is None:
        raise RuntimeError(
            "OrbitKV certificate was generated outside a managed free group"
        )
    pending.append((req, certificate))
    return None


def _commit_swa_reclamations(
    original_fn: Callable,
    batch: Any,
    *args: Any,
    **kwargs: Any,
):
    if getattr(batch, "_orbitkv_pending_certificates", None) is not None:
        raise RuntimeError("nested OrbitKV reclamation group")
    batch._orbitkv_pending_certificates = []
    try:
        result = original_fn(batch, *args, **kwargs)
        pending = batch._orbitkv_pending_certificates
        if pending:
            certificates = [certificate for _, certificate in pending]
            response = _require_owner().command(
                {
                    "op": "commit_reclamations",
                    "certificate_ids": [
                        int(certificate["certificate_id"])
                        for certificate in certificates
                    ],
                }
            )
            committed = response.get("certificate_ids")
            expected = [
                int(certificate["certificate_id"]) for certificate in certificates
            ]
            if committed != expected:
                raise RuntimeError(
                    "OrbitKV owner returned a mismatched batch commit response"
                )
            for req, certificate in pending:
                req.kv.swa_evicted_seqlen = int(
                    certificate["token_end_exclusive"]
                )
            if _trace_allocations_enabled():
                for _, certificate in pending:
                    _emit(
                        {
                            "event": "retirement_committed",
                            "request_id": certificate["request_id"],
                            "certificate": certificate,
                        }
                    )
        return result
    finally:
        batch._orbitkv_pending_certificates = None


def _release_owned_request(
    original_fn: Callable,
    req: Any,
    *args: Any,
    **kwargs: Any,
):
    result = original_fn(req, *args, **kwargs)
    if req is not None:
        _require_owner().command(
            {
                "op": "release_request",
                "request_id": str(req.rid),
            }
        )
    return result


def _emit_owner_certificate(certificate: dict[str, Any]) -> None:
    if _trace_allocations_enabled():
        _emit(
            {
                "event": "retirement_certificate",
                "request_id": certificate["request_id"],
                "certificate": certificate,
            }
        )


def register() -> None:
    global _OWNER, _POLICY, _WRITER

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _POLICY = _load_policy()
    if _POLICY is not None:
        os.environ["SGLANG_SWA_EVICTION_INTERVAL"] = str(
            _POLICY["swa_eviction_interval_tokens"]
        )

    if _owner_enabled():
        if _POLICY is None:
            raise RuntimeError(
                "ORBITKV_SGLANG_OWNING requires ORBITKV_SGLANG_POLICY"
            )
        if _POLICY["page_tokens"] <= 1:
            raise RuntimeError("OrbitKV owning mode requires a paged SGLang pool")
        _OWNER = OwnerClient(
            os.environ.get("ORBITKV_BIN", "orbitkv"),
            os.environ["ORBITKV_SGLANG_POLICY"],
        )
        atexit.register(_stop_owner)
        HookRegistry.register(
            "sglang.srt.managers.schedule_batch.ScheduleBatch._evict_swa",
            _own_swa_reclamation,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
            _commit_swa_reclamations,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.mem_cache.common.release_kv_cache",
            _release_owned_request,
            HookType.AROUND,
        )

    trace_allocations = _trace_allocations_enabled()
    if trace_allocations and _WRITER is None:
        _WRITER = threading.Thread(
            target=_writer_main,
            name="orbitkv-sglang-shadow",
            daemon=True,
        )
        _WRITER.start()
        atexit.register(_stop_writer)

    if trace_allocations:
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
                "policy": _POLICY,
            }
        )
