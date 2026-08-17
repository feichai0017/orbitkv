from __future__ import annotations

import atexit
import ctypes
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
_PHYSICAL_PLAN: dict[str, Any] | None = None
_OWNER: "OwnerClient | None" = None


class OwnerClient:
    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SidecarOwnerClient(OwnerClient):
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
        if self._process.poll() is None:
            self._stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._stdout.close()
        if self._process.stderr is not None:
            self._process.stderr.close()


class _FfiCertificate(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("certificate_id", ctypes.c_uint64),
        ("page_tokens", ctypes.c_uint64),
        ("token_start", ctypes.c_uint64),
        ("token_end_exclusive", ctypes.c_uint64),
        ("semantic_frontier", ctypes.c_uint64),
        ("window_tokens", ctypes.c_uint64),
        ("maximum_reclaimable_end", ctypes.c_uint64),
        ("execution_epoch", ctypes.c_uint64),
        ("plan_fingerprint", ctypes.c_uint8 * 32),
    ]


class _FfiOwnerStats(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("tracked_requests", ctypes.c_uint64),
        ("pending_certificates", ctypes.c_uint64),
        ("committed_reclamations", ctypes.c_uint64),
        ("committed_tokens", ctypes.c_uint64),
        ("plan_fingerprint", ctypes.c_uint8 * 32),
    ]


class FfiOwnerClient(OwnerClient):
    ABI_VERSION = 1
    STATUS_OK = 0
    STATUS_NO_CERTIFICATE = 1

    def __init__(self, library_path: str, plan_path: str, policy: dict[str, Any]):
        self._library = ctypes.CDLL(library_path)
        self._configure_signatures()
        if self._library.orbitkv_owner_abi_version() != self.ABI_VERSION:
            raise RuntimeError("unsupported OrbitKV owner ABI version")
        self._error = ctypes.create_string_buffer(1024)
        self._handle = ctypes.c_void_p()
        plan = Path(plan_path).read_bytes()
        plan_buffer = (ctypes.c_uint8 * len(plan)).from_buffer_copy(plan)
        status = self._library.orbitkv_owner_create(
            plan_buffer,
            len(plan),
            ctypes.byref(self._handle),
            self._error,
            len(self._error),
        )
        self._check(status, "create owner")
        if not self._handle.value:
            raise RuntimeError("OrbitKV FFI returned a null owner")
        bounded = policy.get("bounded_classes", [])
        if len(bounded) != 1:
            self.close()
            raise RuntimeError(
                "OrbitKV FFI owner requires exactly one bounded SGLang class"
            )
        self._class_name = str(bounded[0]["name"])
        self._policy_fingerprint = str(policy["plan_fingerprint"])
        self._lock = threading.Lock()

    def _configure_signatures(self) -> None:
        library = self._library
        library.orbitkv_owner_abi_version.argtypes = []
        library.orbitkv_owner_abi_version.restype = ctypes.c_uint32
        library.orbitkv_owner_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.orbitkv_owner_create.restype = ctypes.c_int32
        library.orbitkv_owner_plan_chunk_reclamation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(_FfiCertificate),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.orbitkv_owner_plan_chunk_reclamation.restype = ctypes.c_int32
        library.orbitkv_owner_commit_reclamations.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.orbitkv_owner_commit_reclamations.restype = ctypes.c_int32
        library.orbitkv_owner_release_request.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.orbitkv_owner_release_request.restype = ctypes.c_int32
        library.orbitkv_owner_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_FfiOwnerStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.orbitkv_owner_stats.restype = ctypes.c_int32
        library.orbitkv_owner_destroy.argtypes = [ctypes.c_void_p]
        library.orbitkv_owner_destroy.restype = None

    def _check(self, status: int, operation: str) -> None:
        if status == self.STATUS_OK:
            return
        message = self._error.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OrbitKV FFI {operation} failed with status {status}: {message}"
        )

    @staticmethod
    def _request_buffer(request_id: str):
        encoded = request_id.encode("utf-8")
        return encoded, (ctypes.c_uint8 * len(encoded)).from_buffer_copy(encoded)

    @staticmethod
    def _fingerprint(value) -> str:
        return f"sha256:{bytes(value).hex()}"

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("OrbitKV FFI owner is closed")
            operation = command["op"]
            if operation == "plan_reclamation":
                return self._plan(command)
            if operation == "commit_reclamations":
                return self._commit(command)
            if operation == "release_request":
                return self._release(command)
            if operation == "stats":
                return self._stats()
            raise RuntimeError(f"unsupported OrbitKV FFI command: {operation}")

    def _plan(self, command: dict[str, Any]) -> dict[str, Any]:
        if command.get("cache_kind") != "chunk":
            raise RuntimeError("OrbitKV FFI owner only supports chunk cache")
        encoded, request = self._request_buffer(str(command["request_id"]))
        certificate = _FfiCertificate()
        status = self._library.orbitkv_owner_plan_chunk_reclamation(
            self._handle,
            request,
            len(encoded),
            int(command["observed_evicted_seqlen"]),
            int(command["semantic_frontier"]),
            int(command["execution_epoch"]),
            ctypes.byref(certificate),
            self._error,
            len(self._error),
        )
        if status == self.STATUS_NO_CERTIFICATE:
            return {"status": "reclamation", "certificate": None}
        self._check(status, "plan reclamation")
        fingerprint = self._fingerprint(certificate.plan_fingerprint)
        if fingerprint != self._policy_fingerprint:
            raise RuntimeError(
                "OrbitKV FFI certificate fingerprint does not match policy"
            )
        return {
            "status": "reclamation",
            "certificate": {
                "schema": "orbitkv.sglang-retirement-certificate.v1",
                "plan_fingerprint": fingerprint,
                "certificate_id": certificate.certificate_id,
                "request_id": str(command["request_id"]),
                "class_name": self._class_name,
                "page_tokens": certificate.page_tokens,
                "token_start": certificate.token_start,
                "token_end_exclusive": certificate.token_end_exclusive,
                "semantic_proof": {
                    "kind": "sliding_window",
                    "semantic_frontier": certificate.semantic_frontier,
                    "window_tokens": certificate.window_tokens,
                    "maximum_reclaimable_end": (
                        certificate.maximum_reclaimable_end
                    ),
                },
                "execution_proof": {
                    "kind": "non_overlap_scheduler_barrier",
                    "execution_epoch": certificate.execution_epoch,
                },
            },
        }

    def _commit(self, command: dict[str, Any]) -> dict[str, Any]:
        certificate_ids = [int(value) for value in command["certificate_ids"]]
        values = (ctypes.c_uint64 * len(certificate_ids))(*certificate_ids)
        status = self._library.orbitkv_owner_commit_reclamations(
            self._handle,
            values if certificate_ids else None,
            len(certificate_ids),
            self._error,
            len(self._error),
        )
        self._check(status, "commit reclamations")
        return {"status": "committed", "certificate_ids": certificate_ids}

    def _release(self, command: dict[str, Any]) -> dict[str, Any]:
        request_id = str(command["request_id"])
        encoded, request = self._request_buffer(request_id)
        status = self._library.orbitkv_owner_release_request(
            self._handle,
            request,
            len(encoded),
            self._error,
            len(self._error),
        )
        self._check(status, "release request")
        return {"status": "released", "request_id": request_id}

    def _stats(self) -> dict[str, Any]:
        stats = _FfiOwnerStats()
        status = self._library.orbitkv_owner_stats(
            self._handle,
            ctypes.byref(stats),
            self._error,
            len(self._error),
        )
        self._check(status, "read stats")
        return {
            "status": "stats",
            "stats": {
                "plan_fingerprint": self._fingerprint(stats.plan_fingerprint),
                "tracked_requests": stats.tracked_requests,
                "pending_certificates": stats.pending_certificates,
                "committed_reclamations": stats.committed_reclamations,
                "committed_tokens": stats.committed_tokens,
            },
        }

    def close(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            if getattr(self, "_handle", None) is None or not self._handle.value:
                return
            self._library.orbitkv_owner_destroy(self._handle)
            self._handle = ctypes.c_void_p()


def _owner_transport() -> str:
    return os.environ.get("ORBITKV_OWNER_TRANSPORT", "ffi").lower()


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
    global _PHYSICAL_PLAN

    physical_plan_path = os.environ.get("ORBITKV_SGLANG_PHYSICAL_PLAN")
    if physical_plan_path:
        artifact = json.loads(Path(physical_plan_path).read_text(encoding="utf-8"))
        if artifact.get("schema") != "orbitkv.hf-physical-compilation.v1":
            raise ValueError(f"unsupported OrbitKV physical artifact: {artifact!r}")
        physical_plan = artifact.get("physical_plan")
        if (
            not isinstance(physical_plan, dict)
            or physical_plan.get("schema") != "orbitkv.sglang-physical-plan.v1"
        ):
            raise ValueError(
                f"unsupported OrbitKV SGLang physical plan: {physical_plan!r}"
            )
        selected = physical_plan.get("selected")
        policy = selected.get("policy") if isinstance(selected, dict) else None
        if (
            not isinstance(policy, dict)
            or policy.get("schema") != "orbitkv.sglang-policy.v1"
        ):
            raise ValueError(
                f"OrbitKV physical plan does not contain a selected policy: {selected!r}"
            )
        if not str(physical_plan.get("physical_plan_fingerprint", "")).startswith(
            "sha256:"
        ):
            raise ValueError("OrbitKV physical plan is missing its fingerprint")
        if physical_plan.get("plan_fingerprint") != policy.get("plan_fingerprint"):
            raise ValueError(
                "OrbitKV physical plan semantic fingerprint does not match policy"
            )
        selected_interval = int(
            physical_plan["selected_eviction_interval_tokens"]
        )
        if selected_interval != int(policy["swa_eviction_interval_tokens"]):
            raise ValueError(
                "OrbitKV physical plan interval does not match selected policy"
            )
        if legacy_interval := os.environ.get(
            "ORBITKV_SGLANG_EVICTION_INTERVAL"
        ):
            if int(legacy_interval) != selected_interval:
                raise ValueError(
                    "legacy OrbitKV eviction interval conflicts with physical plan"
                )
        _PHYSICAL_PLAN = physical_plan
        return policy

    _PHYSICAL_PLAN = None
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


def _validate_physical_contract(batch: Any) -> None:
    if _PHYSICAL_PLAN is None:
        return
    contract = _PHYSICAL_PLAN["contract"]
    if (
        contract.get("require_overlap_schedule_disabled")
        and batch.enable_overlap
    ):
        raise RuntimeError(
            "OrbitKV physical plan requires disable_overlap_schedule"
        )
    if (
        contract.get("require_radix_cache_disabled")
        and not batch.tree_cache.is_chunk_cache()
    ):
        raise RuntimeError("OrbitKV physical plan requires disable_radix_cache")
    if (
        contract.get("require_speculative_decoding_disabled")
        and not batch.spec_algorithm.is_none()
    ):
        raise RuntimeError(
            "OrbitKV physical plan requires speculative decoding disabled"
        )
    if contract.get("cache_kind") != "swa_chunk_cache":
        raise RuntimeError("OrbitKV physical plan uses an unsupported cache kind")
    if int(_POLICY["page_tokens"]) != int(batch.tree_cache.page_size):
        raise RuntimeError(
            "OrbitKV physical plan page size does not match SGLang"
        )
    selected_cost = _PHYSICAL_PLAN["selected"]["cost"]
    allocator = batch.token_to_kv_pool_allocator
    actual_full = getattr(allocator, "size_full", None)
    actual_swa = getattr(allocator, "size_swa", None)
    if actual_full is not None and int(actual_full) != int(
        selected_cost["full_token_capacity"]
    ):
        raise RuntimeError(
            "OrbitKV physical plan Full capacity does not match SGLang"
        )
    if actual_swa is not None and int(actual_swa) != int(
        selected_cost["physical_swa_token_slots"]
    ):
        raise RuntimeError(
            "OrbitKV physical plan SWA capacity does not match SGLang"
        )


def _run_with_physical_contract(
    original_fn: Callable,
    batch: Any,
    *args: Any,
    **kwargs: Any,
):
    _validate_physical_contract(batch)
    return original_fn(batch, *args, **kwargs)


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
    _validate_physical_contract(batch)
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
        transport = _owner_transport()
        if transport == "ffi":
            library_path = os.environ.get("ORBITKV_OWNER_LIB")
            if not library_path:
                raise RuntimeError(
                    "ORBITKV_OWNER_LIB is required for FFI owning mode"
                )
            _OWNER = FfiOwnerClient(
                library_path,
                os.environ["ORBITKV_SGLANG_POLICY"],
                _POLICY,
            )
        elif transport == "sidecar":
            _OWNER = SidecarOwnerClient(
                os.environ.get("ORBITKV_BIN", "orbitkv"),
                os.environ["ORBITKV_SGLANG_POLICY"],
            )
        else:
            raise RuntimeError(
                f"unsupported ORBITKV_OWNER_TRANSPORT: {transport}"
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
    elif _PHYSICAL_PLAN is not None:
        HookRegistry.register(
            "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
            _run_with_physical_contract,
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
