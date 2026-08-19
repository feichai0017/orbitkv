from __future__ import annotations

import atexit
import base64
import ctypes
import hashlib
import json
import os
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

_EVENTS: queue.Queue[dict[str, Any] | None] = queue.Queue()
_WRITER: threading.Thread | None = None
_POLICY: dict[str, Any] | None = None
_PHYSICAL_PLAN: dict[str, Any] | None = None
_STATE_PLAN: dict[str, Any] | None = None
_RUNTIME_STATE_PLAN: dict[str, Any] | None = None
_UNIFORM_SWA_CONTRACT: dict[str, Any] | None = None
_STATE_PLAN_MODE: str | None = None
_OWNER: "OwnerClient | None" = None
_BINDINGS: "SidecarOwnerClient | None" = None
_CAPSULES: "CapsuleClient | None" = None
_CAPSULES_LOCK = threading.Lock()
_BINDINGS_LOCK = threading.Lock()


class CapsuleClient:
    def __init__(self, orbitkv_bin: str, root: str):
        self._process = subprocess.Popen(
            [orbitkv_bin, "serve-capsules", root],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("OrbitKV Capsule store did not expose command pipes")
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
                    f"OrbitKV Capsule store exited with "
                    f"{self._process.returncode}: {stderr}"
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
                raise RuntimeError(
                    f"OrbitKV Capsule store closed its response stream: {stderr}"
                )
            response = json.loads(line)
            if response.get("status") == "error":
                raise RuntimeError(
                    f"OrbitKV Capsule store rejected command "
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
                "binding": {
                    "pending_bindings": 0,
                    "committed_bindings": 0,
                    "aborted_bindings": 0,
                },
            },
        }

    def close(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            if getattr(self, "_handle", None) is None or not self._handle.value:
                return
            self._library.orbitkv_owner_destroy(self._handle)
            self._handle = ctypes.c_void_p()


def _owner_transport() -> str:
    if _RUNTIME_STATE_PLAN is not None:
        transport = _RUNTIME_STATE_PLAN["execution"].get("owner_transport")
        if transport is None:
            raise RuntimeError("OrbitKV runtime StatePlan has no owner transport")
        return str(transport)
    return os.environ.get("ORBITKV_OWNER_TRANSPORT", "ffi").lower()


def _owner_enabled() -> bool:
    if _RUNTIME_STATE_PLAN is not None:
        return _RUNTIME_STATE_PLAN["execution"]["mode"] == "owner"
    return os.environ.get("ORBITKV_SGLANG_OWNING", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _capsules_enabled() -> bool:
    if _RUNTIME_STATE_PLAN is not None:
        return bool(_RUNTIME_STATE_PLAN["capsule"]["enabled"])
    return "ORBITKV_CAPSULE_STORE" in os.environ


def _capsule_identity() -> dict[str, Any]:
    encoded = os.environ.get("ORBITKV_CAPSULE_IDENTITY")
    if not encoded:
        raise RuntimeError("ORBITKV_CAPSULE_IDENTITY is required")
    identity = json.loads(encoded)
    required = {
        "namespace",
        "model_fingerprint",
        "tokenizer_fingerprint",
        "adapter_fingerprint",
        "state_plan_fingerprint",
    }
    if set(identity) != required:
        raise RuntimeError("OrbitKV Capsule identity fields are incomplete")
    namespace = identity["namespace"]
    if isinstance(namespace, str):
        identity["namespace"] = list(base64.b64decode(namespace, validate=True))
    for field in (
        "model_fingerprint",
        "tokenizer_fingerprint",
        "adapter_fingerprint",
        "state_plan_fingerprint",
    ):
        value = identity[field]
        if isinstance(value, str):
            value = value.removeprefix("sha256:")
            decoded = bytes.fromhex(value)
            if len(decoded) != 32:
                raise RuntimeError(f"OrbitKV Capsule {field} must be SHA-256")
            identity[field] = list(decoded)
    if _RUNTIME_STATE_PLAN is not None:
        expected = _RUNTIME_STATE_PLAN["artifact_fingerprint"]
        expected_bytes = bytes.fromhex(str(expected).removeprefix("sha256:"))
        if bytes(identity["state_plan_fingerprint"]) != expected_bytes:
            raise RuntimeError(
                "OrbitKV Capsule identity does not match the runtime StatePlan"
            )
    return identity


def _capsule_chunk_tokens() -> int:
    chunk_tokens = int(
        _RUNTIME_STATE_PLAN["capsule"]["chunk_tokens"]
        if _RUNTIME_STATE_PLAN is not None
        else os.environ.get("ORBITKV_CAPSULE_CHUNK_TOKENS", "256")
    )
    if chunk_tokens <= 0:
        raise RuntimeError("ORBITKV_CAPSULE_CHUNK_TOKENS must be positive")
    return chunk_tokens


def _bounded_window_tokens() -> int:
    if _POLICY is None:
        raise RuntimeError("OrbitKV Capsule hydration requires an SGLang policy")
    bounded = _POLICY.get("bounded_classes", ())
    if len(bounded) != 1:
        raise RuntimeError(
            "OrbitKV Capsule hydration requires exactly one bounded KV class"
        )
    window_tokens = int(bounded[0].get("window_tokens", 0))
    if window_tokens <= 0:
        raise RuntimeError("OrbitKV Capsule policy has an invalid window")
    return window_tokens


def _pure_swa_capsule() -> bool:
    return (
        _POLICY is not None
        and _POLICY.get("unbounded_classes") == []
        and len(_POLICY.get("bounded_classes", ())) == 1
    )


def _hybrid_swa_capsule() -> bool:
    return (
        _POLICY is not None
        and len(_POLICY.get("unbounded_classes", ())) == 1
        and len(_POLICY.get("bounded_classes", ())) == 1
    )


def _encode_capsule_payload(value: Any) -> bytes:
    from .capsule_wire import encode_cpu_tensors

    return encode_cpu_tensors(value)


def _decode_capsule_payload(payload: bytes) -> Any:
    from .capsule_wire import decode_cpu_tensors

    return decode_cpu_tensors(payload)


def _capsule_payload_limit() -> int:
    maximum_payload = int(
        _RUNTIME_STATE_PLAN["capsule"]["maximum_payload_bytes"]
        if _RUNTIME_STATE_PLAN is not None
        else os.environ.get(
            "ORBITKV_CAPSULE_MAX_PAYLOAD_BYTES", str(64 * 1024 * 1024)
        )
    )
    if maximum_payload <= 0:
        raise RuntimeError("ORBITKV_CAPSULE_MAX_PAYLOAD_BYTES must be positive")
    return maximum_payload


def _capsule_live_start(prefix_tokens: int) -> int:
    if not (_pure_swa_capsule() or _hybrid_swa_capsule()):
        return 0
    page_tokens = int(_POLICY["page_tokens"])
    live_start = max(0, prefix_tokens - _bounded_window_tokens())
    return live_start // page_tokens * page_tokens


def _hybrid_capsule_cpu_copy(
    allocator: Any,
    indices: Any,
    live_start: int,
) -> dict[str, Any]:
    import torch

    kvcache = getattr(allocator, "_kvcache", None)
    full_pool = getattr(kvcache, "full_kv_pool", None)
    swa_pool = getattr(kvcache, "swa_kv_pool", None)
    mapping = getattr(kvcache, "full_to_swa_index_mapping", None)
    if full_pool is None or swa_pool is None or mapping is None:
        raise RuntimeError("OrbitKV Hybrid Capsule requires separate Full and SWA pools")
    full_cpu = full_pool.get_cpu_copy(indices)
    tail_indices = indices[live_start:]
    swa_indices = mapping[tail_indices]
    if swa_indices.numel() != tail_indices.numel() or not bool(
        torch.all(swa_indices > 0).item()
    ):
        raise RuntimeError("OrbitKV Hybrid Capsule live SWA tail is incomplete")
    swa_cpu = swa_pool.get_cpu_copy(swa_indices)
    swa_mask = torch.zeros((len(indices),), dtype=torch.bool)
    swa_mask[live_start:] = True
    return {"full": full_cpu, "swa": swa_cpu, "swa_mask": swa_mask}


def _encode_capsule_components(
    kv_cache_cpu: Any,
    prefix_tokens: int,
    live_start: int,
) -> tuple[bytes, list[dict[str, Any]]]:
    if _hybrid_swa_capsule():
        if not isinstance(kv_cache_cpu, dict) or set(kv_cache_cpu) != {
            "full",
            "swa",
            "swa_mask",
        }:
            raise RuntimeError("OrbitKV Hybrid Capsule CPU state is malformed")
        full_payload = _encode_capsule_payload(kv_cache_cpu["full"])
        swa_payload = _encode_capsule_payload(
            {
                "swa": kv_cache_cpu["swa"],
                "swa_mask": kv_cache_cpu["swa_mask"],
            }
        )
        return (
            full_payload + swa_payload,
            [
                {
                    "state_class": "full-kv",
                    "length_bytes": len(full_payload),
                    "token_start": 0,
                    "token_end_exclusive": prefix_tokens,
                },
                {
                    "state_class": "swa-kv",
                    "length_bytes": len(swa_payload),
                    "token_start": live_start,
                    "token_end_exclusive": prefix_tokens,
                },
            ],
        )
    payload = _encode_capsule_payload(kv_cache_cpu)
    state_class = "swa-kv" if _pure_swa_capsule() else "sglang-kv"
    return (
        payload,
        [
            {
                "state_class": state_class,
                "length_bytes": len(payload),
                "token_start": live_start,
                "token_end_exclusive": prefix_tokens,
            }
        ],
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


def _stop_bindings() -> None:
    global _BINDINGS

    if _BINDINGS is None:
        return
    _BINDINGS.close()
    _BINDINGS = None


def _stop_capsules() -> None:
    global _CAPSULES

    if _CAPSULES is None:
        return
    _CAPSULES.close()
    _CAPSULES = None


def _require_capsules() -> CapsuleClient:
    global _CAPSULES

    with _CAPSULES_LOCK:
        if _CAPSULES is None:
            _CAPSULES = CapsuleClient(
                os.environ.get("ORBITKV_BIN", "orbitkv"),
                os.environ["ORBITKV_CAPSULE_STORE"],
            )
        return _CAPSULES


def _binding_plan_path() -> str:
    if _RUNTIME_STATE_PLAN is not None:
        return os.environ["ORBITKV_RUNTIME_STATE_PLAN"]
    path = os.environ.get("ORBITKV_SGLANG_POLICY")
    if not path:
        raise RuntimeError("OrbitKV binding runtime requires a semantic plan")
    return path


def _require_bindings() -> SidecarOwnerClient:
    global _BINDINGS

    with _BINDINGS_LOCK:
        if _BINDINGS is None:
            _BINDINGS = SidecarOwnerClient(
                os.environ.get("ORBITKV_BIN", "orbitkv"),
                _binding_plan_path(),
            )
        return _BINDINGS


def _export_capsule_before_release(
    req: Any,
    tree_cache: Any,
    *,
    is_insert: bool,
) -> dict[str, Any] | None:
    if (
        not _capsules_enabled()
        or req is None
        or req.req_pool_idx is None
        or req.kv is None
    ):
        return None
    if not is_insert or not req.finished():
        return None
    if not tree_cache.is_chunk_cache():
        raise RuntimeError("OrbitKV Capsule export currently requires ChunkCache")
    committed = int(req.effective_kv_committed_len())
    chunk_tokens = _capsule_chunk_tokens()
    aligned = (committed // chunk_tokens) * chunk_tokens
    if aligned == 0:
        return None
    hydrated_prefix = int(getattr(req, "_orbitkv_capsule_prefix_tokens", 0))
    if hydrated_prefix and aligned <= hydrated_prefix:
        return None
    fill_ids = [
        int(token)
        for token in list(req.origin_input_ids) + list(req.output_ids)
    ][:aligned]
    if len(fill_ids) != aligned:
        raise RuntimeError("OrbitKV Capsule token identity is shorter than committed KV")
    existing = _require_capsules().command(
        {
            "op": "restore",
            "identity": _capsule_identity(),
            "chunk_tokens": chunk_tokens,
            "token_ids": fill_ids,
        }
    )
    if existing.get("status") == "restored":
        manifest = existing.get("manifest")
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "orbitkv.continuation-capsule.v1"
        ):
            raise RuntimeError("OrbitKV Capsule catalog returned an invalid manifest")
        existing_boundary = int(manifest.get("prefix_token_count", 0))
        if existing_boundary > aligned:
            raise RuntimeError("OrbitKV Capsule catalog exceeded the query boundary")
        if existing_boundary == aligned:
            if _trace_allocations_enabled():
                _emit(
                    {
                        "event": "capsule_reused",
                        "request_id": str(req.rid),
                        "prefix_token_count": aligned,
                    }
                )
            return existing
    elif existing.get("status") != "miss":
        raise RuntimeError("OrbitKV Capsule catalog returned an invalid status")
    live_start = _capsule_live_start(aligned)
    full_indices = tree_cache.req_to_token_pool.req_to_token[
        req.req_pool_idx, :aligned
    ]
    kvcache = getattr(tree_cache.token_to_kv_pool_allocator, "_kvcache", None)
    if (
        _pure_swa_capsule()
        and int(req.kv.swa_evicted_seqlen) > live_start
        and getattr(kvcache, "full_kv_pool", None) is None
    ):
        raise RuntimeError(
            "OrbitKV Capsule export cannot read evicted pure-SWA slots"
        )
    if _hybrid_swa_capsule():
        kv_cache_cpu = _hybrid_capsule_cpu_copy(
            tree_cache.token_to_kv_pool_allocator,
            full_indices,
            live_start,
        )
    else:
        indices = (
            full_indices[live_start:] if _pure_swa_capsule() else full_indices
        )
        kv_cache_cpu = tree_cache.token_to_kv_pool_allocator.get_cpu_copy(
            indices, mamba_indices=req.mamba_pool_idx
        )
    payload, components = _encode_capsule_components(
        kv_cache_cpu,
        aligned,
        live_start,
    )
    maximum_payload = _capsule_payload_limit()
    if len(payload) > maximum_payload:
        raise RuntimeError(
            f"OrbitKV Capsule payload exceeds configured limit: "
            f"{len(payload)} > {maximum_payload}"
        )
    root = Path(os.environ["ORBITKV_CAPSULE_STORE"])
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    payload_path = staging / f"{req.rid}.{time.time_ns()}.payload"
    payload_path.write_bytes(payload)
    try:
        response = _require_capsules().command(
            {
                "op": "publish",
                "identity": _capsule_identity(),
                "chunk_tokens": chunk_tokens,
                "token_ids": fill_ids,
                "live_token_count": (
                    aligned if _hybrid_swa_capsule() else aligned - live_start
                ),
                "payload_path": str(payload_path),
                "components": components,
                "created_unix_ms": time.time_ns() // 1_000_000,
            }
        )
    finally:
        payload_path.unlink(missing_ok=True)
    if _trace_allocations_enabled():
        _emit(
            {
                "event": "capsule_published",
                "request_id": str(req.rid),
                "prefix_token_count": int(response["prefix_token_count"]),
                "payload_bytes": int(response["payload_bytes"]),
                "created": bool(response["created"]),
            }
        )
    return response


def _read_capsule_payload(response: dict[str, Any]) -> tuple[Any, int, int]:
    manifest = response.get("manifest")
    if (
        response.get("status") != "restored"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != "orbitkv.continuation-capsule.v1"
    ):
        raise RuntimeError("OrbitKV Capsule restore returned an invalid manifest")
    prefix_tokens = int(manifest.get("prefix_token_count", 0))
    live_tokens = int(manifest.get("live_token_count", 0))
    payload_bytes = int(manifest.get("payload_bytes", 0))
    if prefix_tokens <= 0 or live_tokens <= 0 or live_tokens > prefix_tokens:
        raise RuntimeError("OrbitKV SGLang Capsule has an invalid live-state range")
    if live_tokens != prefix_tokens and not _pure_swa_capsule():
        raise RuntimeError("OrbitKV partial Capsule requires a pure-SWA state plan")
    if live_tokens % int(_POLICY["page_tokens"]) != 0:
        raise RuntimeError("OrbitKV Capsule live state is not page aligned")
    if payload_bytes <= 0 or payload_bytes > _capsule_payload_limit():
        raise RuntimeError("OrbitKV Capsule restore payload size is invalid")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise RuntimeError("OrbitKV Capsule SGLang components are invalid")
    payload_path = Path(str(response.get("payload_path", "")))
    if not payload_path.is_file() or payload_path.stat().st_size != payload_bytes:
        raise RuntimeError("OrbitKV Capsule payload object is missing or truncated")
    with payload_path.open("rb") as stream:
        payload = bytearray(stream.read())
    expected_digest = bytes(manifest.get("payload_digest", ()))
    if (
        len(expected_digest) != hashlib.sha256().digest_size
        or hashlib.sha256(payload).digest() != expected_digest
    ):
        raise RuntimeError("OrbitKV Capsule payload checksum does not match")
    decoded = _decode_capsule_components(
        payload,
        components,
        prefix_tokens,
        live_tokens,
    )
    return decoded, prefix_tokens, live_tokens


def _decode_capsule_components(
    payload: bytearray,
    components: list[dict[str, Any]],
    prefix_tokens: int,
    live_tokens: int,
) -> Any:
    expected_offset = 0
    decoded: dict[str, Any] = {}
    ranges: dict[str, tuple[int | None, int | None]] = {}
    for component in components:
        state_class = component.get("state_class")
        if not isinstance(state_class, str) or not state_class:
            raise RuntimeError("OrbitKV Capsule component name is invalid")
        offset = int(component.get("offset_bytes", -1))
        length = int(component.get("length_bytes", -1))
        if offset != expected_offset or length <= 0:
            raise RuntimeError("OrbitKV Capsule component coverage is invalid")
        end = offset + length
        if end > len(payload):
            raise RuntimeError("OrbitKV Capsule component exceeds its payload")
        checksum = bytes(component.get("checksum", ()))
        component_payload = memoryview(payload)[offset:end]
        if (
            len(checksum) != hashlib.sha256().digest_size
            or hashlib.sha256(component_payload).digest() != checksum
        ):
            raise RuntimeError("OrbitKV Capsule component checksum does not match")
        token_start = component.get("token_start")
        token_end = component.get("token_end_exclusive")
        if (token_start is None) != (token_end is None):
            raise RuntimeError("OrbitKV Capsule component token range is incomplete")
        if token_start is not None:
            token_start = int(token_start)
            token_end = int(token_end)
            if not 0 <= token_start < token_end <= prefix_tokens:
                raise RuntimeError("OrbitKV Capsule component token range is invalid")
        if state_class in decoded:
            raise RuntimeError("OrbitKV Capsule component name is duplicated")
        decoded[state_class] = _decode_capsule_payload(component_payload)
        ranges[state_class] = (token_start, token_end)
        expected_offset = end
    if expected_offset != len(payload):
        raise RuntimeError("OrbitKV Capsule components do not cover the payload")

    if set(decoded) == {"full-kv", "swa-kv"}:
        if not _hybrid_swa_capsule() or live_tokens != prefix_tokens:
            raise RuntimeError("OrbitKV Hybrid Capsule does not match its state plan")
        expected_live_start = _capsule_live_start(prefix_tokens)
        if ranges["full-kv"] != (0, prefix_tokens) or ranges["swa-kv"] != (
            expected_live_start,
            prefix_tokens,
        ):
            raise RuntimeError("OrbitKV Hybrid Capsule component ranges do not match")
        swa_component = decoded["swa-kv"]
        if (
            not isinstance(swa_component, dict)
            or set(swa_component) != {"swa", "swa_mask"}
        ):
            raise RuntimeError("OrbitKV Hybrid SWA component is malformed")
        swa_mask = swa_component["swa_mask"]
        if (
            _tensor_numel(swa_mask) != prefix_tokens
            or bool(swa_mask[:expected_live_start].any().item())
            or not bool(swa_mask[expected_live_start:].all().item())
        ):
            raise RuntimeError("OrbitKV Hybrid SWA component mask is invalid")
        return {
            "full": decoded["full-kv"],
            "swa": swa_component["swa"],
            "swa_mask": swa_mask,
        }

    if len(decoded) != 1:
        raise RuntimeError("OrbitKV Capsule component set is unsupported")
    state_class, value = next(iter(decoded.items()))
    if state_class not in ("sglang-kv", "swa-kv"):
        raise RuntimeError("OrbitKV Capsule state class is unsupported")
    token_range = ranges[state_class]
    if token_range != (None, None):
        expected_start = prefix_tokens - live_tokens
        if token_range != (expected_start, prefix_tokens):
            raise RuntimeError("OrbitKV Capsule token range does not match live state")
    return value


def _allocate_capsule_slots(allocator: Any, prefix_tokens: int):
    import torch

    page_tokens = int(allocator.page_size)
    if prefix_tokens <= 0 or prefix_tokens % page_tokens != 0:
        raise RuntimeError("OrbitKV Capsule prefix is not allocator-page aligned")
    if page_tokens == 1:
        return allocator.alloc(prefix_tokens)
    device = allocator.device
    prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)
    prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
    seq_lens = torch.tensor([prefix_tokens], dtype=torch.int64, device=device)
    seq_lens_cpu = torch.tensor([prefix_tokens], dtype=torch.int64)
    last_loc = torch.full((1,), -1, dtype=torch.int64, device=device)
    if _pure_swa_capsule():
        return allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            prefix_tokens,
        )
    allocate_tail = getattr(allocator, "alloc_extend_swa_tail", None)
    kvcache = getattr(allocator, "_kvcache", None)
    if allocate_tail is not None and getattr(kvcache, "full_kv_pool", None) is not None:
        return allocate_tail(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            prefix_tokens,
            min(prefix_tokens, _bounded_window_tokens()),
        )
    return allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        prefix_tokens,
    )


def _expected_binding_components(
    prefix_tokens: int,
    live_tokens: int,
) -> list[dict[str, Any]]:
    if _POLICY is None:
        raise RuntimeError("OrbitKV binding requires a loaded policy")
    bounded = _POLICY.get("bounded_classes", ())
    unbounded = _POLICY.get("unbounded_classes", ())
    if len(bounded) != 1:
        raise RuntimeError("OrbitKV binding requires one bounded class")
    components = []
    if unbounded:
        if len(unbounded) != 1 or live_tokens != prefix_tokens:
            raise RuntimeError("OrbitKV Hybrid binding class geometry is unsupported")
        components.append(
            {
                "state_class": str(unbounded[0]),
                "token_start": 0,
                "token_end_exclusive": prefix_tokens,
                "physical_tokens": prefix_tokens,
            }
        )
    local_start = _capsule_live_start(prefix_tokens)
    components.append(
        {
            "state_class": str(bounded[0]["name"]),
            "token_start": local_start,
            "token_end_exclusive": prefix_tokens,
            "physical_tokens": prefix_tokens - local_start,
        }
    )
    return sorted(components, key=lambda component: component["state_class"])


def _prepare_capsule_binding(
    req: Any,
    prefix_tokens: int,
    live_tokens: int,
) -> dict[str, Any]:
    response = _require_bindings().command(
        {
            "op": "prepare_binding",
            "request_id": str(req.rid),
            "prefix_tokens": prefix_tokens,
        }
    )
    if response.get("status") != "binding_prepared":
        raise RuntimeError("OrbitKV binding runtime returned an invalid prepare response")
    intent = response.get("intent")
    if (
        not isinstance(intent, dict)
        or intent.get("schema") != "orbitkv.state-binding-intent.v1"
        or intent.get("plan_fingerprint") != _POLICY.get("plan_fingerprint")
        or intent.get("request_id") != str(req.rid)
        or sorted(
            intent.get("components", ()),
            key=lambda component: component.get("state_class", ""),
        )
        != _expected_binding_components(prefix_tokens, live_tokens)
    ):
        raise RuntimeError("OrbitKV binding intent does not match Capsule semantics")
    if _trace_allocations_enabled():
        _emit(
            {
                "event": "binding_prepared",
                "request_id": str(req.rid),
                "binding_id": int(intent["binding_id"]),
                "components": intent["components"],
            }
        )
    return intent


def _binding_receipt(
    intent: dict[str, Any],
    indices: Any,
) -> dict[str, Any]:
    data_pointer = getattr(indices, "data_ptr", lambda: id(indices))()
    return {
        "schema": "orbitkv.physical-state-binding-receipt.v1",
        "plan_fingerprint": intent["plan_fingerprint"],
        "binding_id": int(intent["binding_id"]),
        "backend_transaction_id": (
            f"sglang:{intent['request_id']}:{intent['binding_id']}:{data_pointer}"
        ),
        "components": [
            {
                **component,
                "physical_binding_id": (
                    f"{component['state_class']}:{data_pointer}:"
                    f"{component['token_start']}:{component['token_end_exclusive']}"
                ),
                "payload_ready": True,
            }
            for component in intent["components"]
        ],
    }


def _abort_capsule_binding(intent: dict[str, Any]) -> None:
    response = _require_bindings().command(
        {
            "op": "abort_binding",
            "binding_id": int(intent["binding_id"]),
        }
    )
    if (
        response.get("status") != "binding_aborted"
        or int(response.get("binding_id", -1)) != int(intent["binding_id"])
    ):
        raise RuntimeError("OrbitKV binding runtime returned an invalid abort response")
    if _trace_allocations_enabled():
        _emit(
            {
                "event": "binding_aborted",
                "request_id": intent["request_id"],
                "binding_id": int(intent["binding_id"]),
            }
        )


def _commit_capsule_binding(intent: dict[str, Any], indices: Any) -> None:
    response = _require_bindings().command(
        {
            "op": "commit_binding",
            "receipt": _binding_receipt(intent, indices),
        }
    )
    if (
        response.get("status") != "binding_committed"
        or int(response.get("binding_id", -1)) != int(intent["binding_id"])
    ):
        raise RuntimeError("OrbitKV binding runtime returned an invalid commit response")
    if _trace_allocations_enabled():
        _emit(
            {
                "event": "binding_committed",
                "request_id": intent["request_id"],
                "binding_id": int(intent["binding_id"]),
            }
        )


def _try_hydrate_capsule(req: Any, tree_cache: Any):
    if not _capsules_enabled() or getattr(req, "_orbitkv_capsule_miss", False):
        return None
    if not tree_cache.is_chunk_cache():
        raise RuntimeError("OrbitKV Capsule hydration currently requires ChunkCache")
    if (
        req.req_pool_idx is not None
        or req.kv is not None
        or len(req.prefix_indices) != 0
        or getattr(req, "is_retracted", False)
    ):
        return None
    input_tokens = [int(token) for token in req.full_untruncated_fill_ids]
    maximum_prefix = int(req._compute_max_prefix_len(len(input_tokens)))
    if maximum_prefix <= 0:
        req._orbitkv_capsule_miss = True
        return None
    started_ns = time.perf_counter_ns()
    response = _require_capsules().command(
        {
            "op": "restore",
            "identity": _capsule_identity(),
            "chunk_tokens": _capsule_chunk_tokens(),
            "token_ids": input_tokens[:maximum_prefix],
        }
    )
    lookup_done_ns = time.perf_counter_ns()
    if response.get("status") == "miss":
        req._orbitkv_capsule_miss = True
        if _trace_allocations_enabled():
            _emit(
                {
                    "event": "capsule_miss",
                    "request_id": str(req.rid),
                    "query_token_count": maximum_prefix,
                }
            )
        return None
    kv_cache_cpu, prefix_tokens, live_tokens = _read_capsule_payload(response)
    payload_done_ns = time.perf_counter_ns()
    if prefix_tokens > maximum_prefix:
        raise RuntimeError("OrbitKV Capsule prefix exceeds SGLang match boundary")
    intent = _prepare_capsule_binding(req, prefix_tokens, live_tokens)
    allocator = tree_cache.token_to_kv_pool_allocator
    indices = _allocate_capsule_slots(allocator, live_tokens)
    allocation_done_ns = time.perf_counter_ns()
    if indices is None:
        _abort_capsule_binding(intent)
        if _trace_allocations_enabled():
            _emit(
                {
                    "event": "capsule_deferred",
                    "request_id": str(req.rid),
                    "prefix_token_count": prefix_tokens,
                    "live_token_count": live_tokens,
                    "reason": "kv_capacity",
                }
            )
        return None
    try:
        allocator.load_cpu_copy(
            kv_cache_cpu,
            indices,
            mamba_indices=req.mamba_pool_idx,
        )
    except Exception:
        allocator.free(indices)
        _abort_capsule_binding(intent)
        raise
    load_done_ns = time.perf_counter_ns()
    import torch

    dead_prefix_tokens = prefix_tokens - live_tokens
    if dead_prefix_tokens:
        prefix_indices = torch.zeros(
            (prefix_tokens,), dtype=torch.int64, device=indices.device
        )
        prefix_indices[dead_prefix_tokens:] = indices.to(dtype=torch.int64)
        req.prefix_indices = prefix_indices
    else:
        req.prefix_indices = indices.to(dtype=torch.int64, copy=True)
    req.cache_protected_len = dead_prefix_tokens
    req._orbitkv_capsule_prefix_tokens = prefix_tokens
    req._orbitkv_capsule_live_tokens = live_tokens
    req._orbitkv_capsule_hydration_ns = {
        "lookup": lookup_done_ns - started_ns,
        "payload_decode": payload_done_ns - lookup_done_ns,
        "allocation": allocation_done_ns - payload_done_ns,
        "load": load_done_ns - allocation_done_ns,
        "total": load_done_ns - started_ns,
    }
    return {"indices": indices, "intent": intent}


def _rollback_capsule_hydration(
    req: Any,
    allocator: Any,
    transaction: dict[str, Any],
) -> None:
    indices = transaction["indices"]
    allocator.free(indices)
    _abort_capsule_binding(transaction["intent"])
    req.prefix_indices = req.prefix_indices.new_empty((0,))
    req._orbitkv_capsule_prefix_tokens = 0
    req._orbitkv_capsule_live_tokens = 0
    req._orbitkv_capsule_hydration_ns = None


def _hydrate_capsule_for_admission(
    original_fn: Callable,
    adder: Any,
    req: Any,
    *args: Any,
    **kwargs: Any,
):
    tree_cache = adder.tree_cache
    transaction = _try_hydrate_capsule(req, tree_cache)
    if transaction is None:
        return original_fn(adder, req, *args, **kwargs)
    before = len(adder.can_run_list)
    try:
        result = original_fn(adder, req, *args, **kwargs)
    except Exception:
        _rollback_capsule_hydration(
            req, tree_cache.token_to_kv_pool_allocator, transaction
        )
        raise
    admitted = len(adder.can_run_list) > before and adder.can_run_list[-1] is req
    if not admitted:
        _rollback_capsule_hydration(
            req, tree_cache.token_to_kv_pool_allocator, transaction
        )
        return result
    try:
        _commit_capsule_binding(transaction["intent"], transaction["indices"])
    except Exception:
        if adder.can_run_list and adder.can_run_list[-1] is req:
            adder.can_run_list.pop()
        _rollback_capsule_hydration(
            req, tree_cache.token_to_kv_pool_allocator, transaction
        )
        raise
    if _trace_allocations_enabled():
        _emit(
            {
                "event": "capsule_hydrated",
                "request_id": str(req.rid),
                "prefix_token_count": int(req._orbitkv_capsule_prefix_tokens),
                "live_token_count": int(req._orbitkv_capsule_live_tokens),
                "duration_ns": req._orbitkv_capsule_hydration_ns,
            }
        )
    return result


def _admit_uniform_swa_capsule_request(
    original_fn: Callable,
    adder: Any,
    req: Any,
    *args: Any,
    **kwargs: Any,
):
    def admit(current_adder: Any, current_req: Any, *inner_args: Any, **inner_kwargs: Any):
        return _admit_uniform_swa_request(
            original_fn,
            current_adder,
            current_req,
            *inner_args,
            **inner_kwargs,
        )

    return _hydrate_capsule_for_admission(
        admit,
        adder,
        req,
        *args,
        **kwargs,
    )


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
        return _load_physical_plan_value(artifact)

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


def _load_physical_plan_value(artifact: dict[str, Any]) -> dict[str, Any]:
    global _PHYSICAL_PLAN

    if artifact:
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
    raise ValueError("OrbitKV physical artifact is missing")


def _load_runtime_state_plan() -> dict[str, Any] | None:
    path = os.environ.get("ORBITKV_RUNTIME_STATE_PLAN")
    if not path:
        return None
    conflicts = [
        name
        for name in (
            "ORBITKV_SGLANG_POLICY",
            "ORBITKV_SGLANG_PHYSICAL_PLAN",
            "ORBITKV_SGLANG_STATE_PLAN",
            "ORBITKV_SGLANG_STATE_PLAN_MODE",
            "ORBITKV_SGLANG_EVICTION_INTERVAL",
            "ORBITKV_SGLANG_OWNING",
            "ORBITKV_OWNER_TRANSPORT",
            "ORBITKV_CAPSULE_CHUNK_TOKENS",
            "ORBITKV_CAPSULE_MAX_PAYLOAD_BYTES",
        )
        if name in os.environ
    ]
    if conflicts:
        raise RuntimeError(
            "ORBITKV_RUNTIME_STATE_PLAN conflicts with legacy runtime settings: "
            + ", ".join(conflicts)
        )
    completed = subprocess.run(
        [
            os.environ.get("ORBITKV_BIN", "orbitkv"),
            "validate-runtime-state-plan",
            str(Path(path).resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)
    if artifact.get("schema") != "orbitkv.runtime-state-plan.v1":
        raise RuntimeError("OrbitKV runtime StatePlan schema is unsupported")
    return artifact


def _load_state_plan() -> dict[str, Any] | None:
    global _STATE_PLAN_MODE, _UNIFORM_SWA_CONTRACT

    path = os.environ.get("ORBITKV_SGLANG_STATE_PLAN")
    if not path:
        _STATE_PLAN_MODE = None
        _UNIFORM_SWA_CONTRACT = None
        return None
    mode = os.environ.get("ORBITKV_SGLANG_STATE_PLAN_MODE", "execute")
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return _load_state_plan_value(artifact, mode)


def _load_state_plan_value(
    artifact: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    global _STATE_PLAN_MODE, _UNIFORM_SWA_CONTRACT

    if mode not in ("execute", "kernel_reference"):
        raise ValueError(f"unsupported OrbitKV state-plan mode: {mode!r}")
    if artifact.get("schema") != "orbitkv.hf-state-plan.v4":
        raise ValueError(f"unsupported OrbitKV state plan: {artifact!r}")
    lowering = artifact.get("sglang_lowering")
    if (
        not isinstance(lowering, dict)
        or lowering.get("status") != "enabled"
        or lowering.get("kind") != "uniform_swa"
    ):
        raise ValueError("OrbitKV state plan has no enabled uniform-SWA lowering")
    contract = lowering.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("schema")
        != "orbitkv.sglang-uniform-swa-contract.v4"
    ):
        raise ValueError("OrbitKV state plan has an invalid uniform-SWA contract")
    layout = artifact.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("OrbitKV state plan is missing its layout")
    if contract.get("plan_fingerprint") != layout.get("plan_fingerprint"):
        raise ValueError("OrbitKV state-plan fingerprint does not match its layout")
    expected_contract_fingerprint = _uniform_swa_contract_fingerprint(contract)
    if contract.get("contract_fingerprint") != expected_contract_fingerprint:
        raise ValueError("OrbitKV uniform-SWA contract fingerprint does not match")
    if int(contract.get("page_tokens", 0)) not in (1, 16):
        raise ValueError(
            "OrbitKV uniform-SWA SGLang lowering requires page_tokens 1 or 16"
        )
    for field in (
        "maximum_running_requests",
        "chunked_prefill_tokens",
        "eviction_interval_tokens",
        "decode_headroom_tokens",
        "per_request_resident_tokens",
        "global_staging_tokens",
        "minimum_pool_tokens",
        "maximum_context_tokens",
        "logical_index_tokens",
    ):
        if int(contract.get(field, 0)) <= 0:
            raise ValueError(
                f"OrbitKV uniform-SWA contract has invalid {field}"
            )
    expected_per_request = (
        int(contract["window_tokens"])
        + int(contract["eviction_interval_tokens"])
        + int(contract["page_tokens"])
        + int(contract["decode_headroom_tokens"])
    )
    if int(contract["per_request_resident_tokens"]) != expected_per_request:
        raise ValueError("OrbitKV uniform-SWA per-request budget does not match")
    expected_staging = int(contract["chunked_prefill_tokens"]) + int(
        contract["page_tokens"]
    )
    if int(contract["global_staging_tokens"]) != expected_staging:
        raise ValueError("OrbitKV uniform-SWA staging budget does not match")
    expected_minimum = (
        expected_per_request * int(contract["maximum_running_requests"])
        + expected_staging
    )
    if int(contract["minimum_pool_tokens"]) != expected_minimum:
        raise ValueError("OrbitKV uniform-SWA minimum pool does not match")
    if int(contract["kernel_window_left"]) != int(contract["window_tokens"]) - 1:
        raise ValueError("OrbitKV uniform-SWA kernel window does not match")
    backend = contract.get("physical_backend")
    if backend == "direct_periodic":
        expected_logical = expected_minimum
    elif backend == "paged_periodic":
        page = int(contract["page_tokens"])
        context = int(contract["maximum_context_tokens"])
        aligned_context = (context + page - 1) // page * page
        expected_logical = aligned_context * int(
            contract["maximum_running_requests"]
        )
    else:
        raise ValueError("OrbitKV uniform-SWA physical backend is unsupported")
    if int(contract["logical_index_tokens"]) != expected_logical:
        raise ValueError("OrbitKV uniform-SWA logical index budget does not match")
    if contract.get("scheduler_admission") != "pure_swa_live_state":
        raise ValueError("OrbitKV uniform-SWA scheduler admission is unsupported")
    graph_mode = contract.get("cuda_graph_mode")
    capture_sizes = contract.get("decode_cuda_graph_batch_sizes")
    if graph_mode == "disabled":
        if capture_sizes != []:
            raise ValueError("disabled CUDA Graph contract has capture sizes")
    elif graph_mode == "decode":
        expected_sizes = list(
            range(1, int(contract["maximum_running_requests"]) + 1)
        )
        if contract["physical_backend"] != "paged_periodic":
            raise ValueError("decode CUDA Graph requires paged periodic")
        if capture_sizes != expected_sizes:
            raise ValueError("decode CUDA Graph capture sizes do not match")
    else:
        raise ValueError("OrbitKV uniform-SWA CUDA Graph mode is unsupported")
    os.environ["SGLANG_SWA_EVICTION_INTERVAL"] = str(
        contract["eviction_interval_tokens"]
    )
    _STATE_PLAN_MODE = mode
    _UNIFORM_SWA_CONTRACT = contract
    return artifact


def _uniform_swa_contract_fingerprint(contract: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in (
        contract["plan_fingerprint"],
        contract["config_sha256"],
        contract["architecture"],
        contract["cuda_graph_mode"],
    ):
        encoded = value.encode()
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    for value in (
        contract["num_hidden_layers"],
        contract["num_key_value_heads"],
        contract["head_dim"],
        contract["window_tokens"],
        contract["page_tokens"],
        contract["maximum_context_tokens"],
        contract["maximum_running_requests"],
        contract["chunked_prefill_tokens"],
        contract["eviction_interval_tokens"],
        contract["decode_headroom_tokens"],
    ):
        digest.update(struct.pack("<Q", int(value)))
    for batch_size in contract["decode_cuda_graph_batch_sizes"]:
        digest.update(struct.pack("<Q", int(batch_size)))
    return f"sha256:{digest.hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _build_paged_periodic_allocator():
    from sglang.srt.mem_cache.allocator.swa import (
        PureSWATokenToKVPoolAllocator as OriginalPureSwaAllocator,
    )
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    class OrbitKvPagedPeriodicAllocator(OriginalPureSwaAllocator):
        def __init__(
            self,
            size_swa,
            page_size,
            dtype,
            device,
            kvcache,
            need_sort,
        ):
            contract = _UNIFORM_SWA_CONTRACT
            if contract is None or contract["physical_backend"] != "paged_periodic":
                raise RuntimeError(
                    "OrbitKV paged-periodic allocator requires its compiled contract"
                )
            if int(page_size) != int(contract["page_tokens"]) or int(page_size) <= 1:
                raise RuntimeError(
                    "OrbitKV paged-periodic allocator page size does not match"
                )
            if not isinstance(kvcache, BaseSWAKVPool):
                raise RuntimeError(
                    "OrbitKV paged-periodic allocator requires an SWA KV pool"
                )
            logical_index_tokens = int(contract["logical_index_tokens"])
            if logical_index_tokens % int(page_size) != 0:
                raise RuntimeError(
                    "OrbitKV paged-periodic logical index space is not page aligned"
                )
            if int(size_swa) < int(contract["minimum_pool_tokens"]):
                raise RuntimeError(
                    "OrbitKV paged-periodic physical pool is below the compiled minimum"
                )
            super(OriginalPureSwaAllocator, self).__init__(
                logical_index_tokens,
                int(size_swa),
                int(page_size),
                dtype,
                device,
                kvcache,
                need_sort,
            )
            self.logical_index_tokens = logical_index_tokens
            self.physical_swa_tokens = int(size_swa)

        def translate_loc_from_full_to_swa(self, kv_indices):
            return super(
                OriginalPureSwaAllocator, self
            ).translate_loc_from_full_to_swa(kv_indices)

        def new_pages_available(self, num_full_pages, num_swa_pages):
            return super(
                OriginalPureSwaAllocator, self
            ).new_pages_available(num_full_pages, num_swa_pages)

        def alloc_extend(self, *args, **kwargs):
            return super(OriginalPureSwaAllocator, self).alloc_extend(
                *args, **kwargs
            )

        def alloc_decode(self, *args, **kwargs):
            return super(OriginalPureSwaAllocator, self).alloc_decode(
                *args, **kwargs
            )

        def free(self, free_index):
            return super(OriginalPureSwaAllocator, self).free(free_index)

        def free_swa(self, free_index):
            return super(OriginalPureSwaAllocator, self).free_swa(free_index)

        def free_group_begin(self):
            return super(
                OriginalPureSwaAllocator, self
            ).free_group_begin()

        def free_group_end(self):
            return super(OriginalPureSwaAllocator, self).free_group_end()

        def clear(self):
            return super(OriginalPureSwaAllocator, self).clear()

        def resize(self, config):
            return super(OriginalPureSwaAllocator, self).resize(config)

    OrbitKvPagedPeriodicAllocator.__name__ = (
        "OrbitKvPagedPeriodicAllocator"
    )
    return OrbitKvPagedPeriodicAllocator


def _finish_paged_periodic_request(
    original_fn: Callable,
    cache: Any,
    req: Any,
    *args: Any,
    **kwargs: Any,
):
    contract = _UNIFORM_SWA_CONTRACT
    if contract is None or contract["physical_backend"] != "paged_periodic":
        return original_fn(cache, req, *args, **kwargs)
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    return ChunkCache.cache_finished_req(cache, req, *args, **kwargs)


def _activate_uniform_swa_model_config(
    original_fn: Callable,
    model_config: Any,
    *args: Any,
    **kwargs: Any,
):
    result = original_fn(model_config, *args, **kwargs)
    contract = _UNIFORM_SWA_CONTRACT
    if contract is None:
        return result
    architecture = model_config.hf_config.architectures[0]
    if architecture != contract["architecture"]:
        raise RuntimeError(
            "OrbitKV uniform-SWA architecture does not match SGLang model"
        )
    config_path = Path(model_config.model_path) / "config.json"
    if not config_path.is_file():
        raise RuntimeError("OrbitKV uniform-SWA model config.json is missing")
    if _sha256_file(config_path) != contract["config_sha256"]:
        raise RuntimeError(
            "OrbitKV uniform-SWA config hash does not match SGLang model"
        )
    num_layers = int(model_config.hf_text_config.num_hidden_layers)
    if num_layers != int(contract["num_hidden_layers"]):
        raise RuntimeError("OrbitKV uniform-SWA layer count does not match SGLang")
    if int(model_config.sliding_window_size) != int(contract["window_tokens"]):
        raise RuntimeError("OrbitKV uniform-SWA window does not match SGLang")
    model_config.sliding_window_size = int(contract["kernel_window_left"])
    if _STATE_PLAN_MODE == "kernel_reference":
        return result
    model_config.is_hybrid_swa = True
    model_config.is_deepseek_v4_arch = False
    model_config.swa_attention_layer_ids = list(range(num_layers))
    model_config.full_attention_layer_ids = []
    return result


def _activate_uniform_swa_kernel(
    original_fn: Callable,
    attention: Any,
    config: Any,
    *args: Any,
    **kwargs: Any,
):
    result = original_fn(attention, config, *args, **kwargs)
    contract = _UNIFORM_SWA_CONTRACT
    if contract is None:
        return result
    architectures = getattr(config, "architectures", None) or []
    if contract["architecture"] not in architectures:
        raise RuntimeError(
            "OrbitKV uniform-SWA kernel config does not match the compiled architecture"
        )
    if int(attention.total_num_kv_heads) != int(contract["num_key_value_heads"]):
        raise RuntimeError("OrbitKV uniform-SWA KV head count does not match kernel")
    if int(attention.head_dim) != int(contract["head_dim"]):
        raise RuntimeError("OrbitKV uniform-SWA head dimension does not match kernel")
    attention.attn.sliding_window_size = int(contract["kernel_window_left"])
    return result


def _validate_uniform_swa_runtime(
    original_fn: Callable,
    configurator: Any,
    *args: Any,
    **kwargs: Any,
):
    contract = _UNIFORM_SWA_CONTRACT
    if contract is None:
        return original_fn(configurator, *args, **kwargs)
    server_args = configurator.server_args
    graph_mode = contract["cuda_graph_mode"]
    expected_disabled = {
        "radix_cache",
        "overlap_schedule",
        "speculative_decoding",
        "disaggregation",
        (
            "cuda_graph"
            if graph_mode == "disabled"
            else "prefill_cuda_graph"
        ),
    }
    if set(contract.get("required_disabled_features", [])) != expected_disabled:
        raise RuntimeError(
            "OrbitKV uniform-SWA disabled-feature contract is unsupported"
        )
    required = {
        "disable_radix_cache": bool(server_args.disable_radix_cache),
        "disable_overlap_schedule": bool(server_args.disable_overlap_schedule),
        "disaggregation_disabled": server_args.disaggregation_mode == "null",
        "speculative_decoding_disabled": configurator.spec_algorithm.is_none(),
        "page_tokens": int(configurator.page_size)
        == int(contract["page_tokens"]),
        "chunked_prefill_tokens": int(server_args.chunked_prefill_size)
        == int(contract["chunked_prefill_tokens"]),
    }
    required["maximum_running_requests"] = int(
        server_args.max_running_requests
    ) <= int(contract["maximum_running_requests"])
    graph_config = server_args.cuda_graph_config
    if graph_mode == "disabled":
        required["cuda_graph_disabled"] = (
            graph_config.decode.backend == "disabled"
            and graph_config.prefill.backend == "disabled"
        )
    else:
        required["decode_cuda_graph"] = graph_config.decode.backend == "full"
        required["prefill_cuda_graph_disabled"] = (
            graph_config.prefill.backend == "disabled"
        )
        required["decode_cuda_graph_batch_sizes"] = list(
            graph_config.decode.bs
        ) == list(contract["decode_cuda_graph_batch_sizes"])
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "OrbitKV uniform-SWA runtime contract failed: " + ", ".join(failed)
        )
    result = original_fn(configurator, *args, **kwargs)
    allocator = result.token_to_kv_pool_allocator
    if _STATE_PLAN_MODE == "kernel_reference":
        if type(allocator).__name__ == "PureSWATokenToKVPoolAllocator":
            raise RuntimeError(
                "OrbitKV kernel reference must retain the ordinary KV allocator"
            )
        return result
    backend = contract["physical_backend"]
    expected_allocator = (
        "PureSWATokenToKVPoolAllocator"
        if backend == "direct_periodic"
        else "OrbitKvPagedPeriodicAllocator"
    )
    if type(allocator).__name__ != expected_allocator:
        raise RuntimeError(
            "OrbitKV uniform-SWA plan did not produce its compiled allocator"
        )
    if int(allocator.page_size) != int(contract["page_tokens"]):
        raise RuntimeError("OrbitKV uniform-SWA allocator page size mismatch")
    if int(allocator.size_swa) < int(contract["minimum_pool_tokens"]):
        raise RuntimeError(
            "OrbitKV uniform-SWA pool is smaller than the compiled minimum"
        )
    if backend == "paged_periodic" and int(allocator.size_full) != int(
        contract["logical_index_tokens"]
    ):
        raise RuntimeError(
            "OrbitKV paged-periodic logical index capacity does not match"
        )
    return result


def _record_decode_graph_replay(
    original_fn: Callable,
    runner: Any,
    forward_batch: Any,
    *args: Any,
    **kwargs: Any,
):
    _emit(
        {
            "event": "decode_graph_replay",
            "batch_size": int(forward_batch.batch_size),
            "forward_mode": str(forward_batch.forward_mode),
        }
    )
    return original_fn(runner, forward_batch, *args, **kwargs)


def _resolve_uniform_swa_window(
    original_fn: Callable,
    model: Any,
    model_config: Any,
):
    result = original_fn(model, model_config)
    if _UNIFORM_SWA_CONTRACT is None:
        return result
    return int(_UNIFORM_SWA_CONTRACT["kernel_window_left"])


def _expand_uniform_swa_worker_info(
    original_fn: Callable,
    worker: Any,
    *args: Any,
    **kwargs: Any,
):
    result = original_fn(worker, *args, **kwargs)
    if _UNIFORM_SWA_CONTRACT is None or _STATE_PLAN_MODE != "execute":
        return result
    values = list(result)
    max_request_len = int(worker.model_config.context_len) - 1
    values[4] = max_request_len
    values[5] = max_request_len - 5
    return tuple(values)


def _init_uniform_swa_max_new_tokens(
    original_fn: Callable,
    scheduler: Any,
    req: Any,
):
    if _UNIFORM_SWA_CONTRACT is None or _STATE_PLAN_MODE != "execute":
        return original_fn(scheduler, req)
    requested = req.sampling_params.max_new_tokens
    requested_minimum = req.sampling_params.min_new_tokens
    result = original_fn(scheduler, req)
    if requested is None:
        requested = 1 << 30
    if (
        scheduler.max_new_tokens_limit is not None
        and scheduler.max_new_tokens_limit > 0
    ):
        requested = min(requested, scheduler.max_new_tokens_limit)
    req.sampling_params.max_new_tokens = max(
        0,
        min(
            requested,
            scheduler.max_req_len - len(req.origin_input_ids) - 1,
        ),
    )
    req.sampling_params.min_new_tokens = min(
        requested_minimum,
        req.sampling_params.max_new_tokens,
    )
    return result


def _admit_uniform_swa_request(
    original_fn: Callable,
    adder: Any,
    req: Any,
    *args: Any,
    **kwargs: Any,
):
    if _UNIFORM_SWA_CONTRACT is None or _STATE_PLAN_MODE != "execute":
        return original_fn(adder, req, *args, **kwargs)
    if not adder.is_all_swa:
        raise RuntimeError("OrbitKV uniform-SWA admission requires PureSWA")
    if int(adder.max_running_requests or 0) > int(
        _UNIFORM_SWA_CONTRACT["maximum_running_requests"]
    ):
        raise RuntimeError(
            "OrbitKV uniform-SWA admission exceeds compiled request concurrency"
        )
    candidate_tokens = (
        len(req.full_untruncated_fill_ids)
        - len(req.prefix_indices)
        + int(req.sampling_params.max_new_tokens)
        + int(adder.page_size)
    )
    boost = max(0, candidate_tokens - int(adder.rem_total_tokens) + 1)
    adder.rem_total_token_offset -= boost
    try:
        return original_fn(adder, req, *args, **kwargs)
    finally:
        adder.rem_total_token_offset += boost


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
    tree_cache: Any,
    *args: Any,
    **kwargs: Any,
):
    is_insert = bool(kwargs.get("is_insert", args[0] if args else True))
    _export_capsule_before_release(req, tree_cache, is_insert=is_insert)
    result = original_fn(req, tree_cache, *args, **kwargs)
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
    global _CAPSULES, _OWNER, _PHYSICAL_PLAN, _POLICY, _RUNTIME_STATE_PLAN
    global _STATE_PLAN, _STATE_PLAN_MODE, _UNIFORM_SWA_CONTRACT, _WRITER

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _RUNTIME_STATE_PLAN = _load_runtime_state_plan()
    if _RUNTIME_STATE_PLAN is None:
        _STATE_PLAN = _load_state_plan()
        _POLICY = _load_policy()
        owner_plan_path = os.environ.get("ORBITKV_SGLANG_POLICY")
    else:
        embedded_uniform = _RUNTIME_STATE_PLAN.get("uniform_state_plan")
        uniform_mode = _RUNTIME_STATE_PLAN["execution"].get(
            "uniform_state_plan_mode"
        )
        if embedded_uniform is None:
            _STATE_PLAN = None
            _STATE_PLAN_MODE = None
            _UNIFORM_SWA_CONTRACT = None
        else:
            _STATE_PLAN = _load_state_plan_value(
                embedded_uniform,
                str(uniform_mode or "execute"),
            )
        embedded_physical = _RUNTIME_STATE_PLAN.get("physical_plan")
        if embedded_physical is None:
            _PHYSICAL_PLAN = None
            _POLICY = _RUNTIME_STATE_PLAN["sglang_policy"]
        else:
            _POLICY = _load_physical_plan_value(embedded_physical)
        if (
            _POLICY.get("schema") != "orbitkv.sglang-policy.v1"
            or _POLICY.get("plan_fingerprint")
            != _RUNTIME_STATE_PLAN.get("plan_fingerprint")
        ):
            raise RuntimeError("OrbitKV runtime StatePlan policy is invalid")
        owner_plan_path = os.environ["ORBITKV_RUNTIME_STATE_PLAN"]
    if _POLICY is not None:
        if (
            _UNIFORM_SWA_CONTRACT is not None
            and _POLICY.get("plan_fingerprint")
            != _UNIFORM_SWA_CONTRACT.get("plan_fingerprint")
        ):
            raise RuntimeError(
                "OrbitKV SGLang policy and state-plan fingerprints differ"
            )
        os.environ["SGLANG_SWA_EVICTION_INTERVAL"] = str(
            _POLICY["swa_eviction_interval_tokens"]
        )

    if _capsules_enabled():
        if not _owner_enabled():
            raise RuntimeError(
                "OrbitKV Capsule export currently requires owning mode"
            )
        if "ORBITKV_CAPSULE_STORE" not in os.environ:
            raise RuntimeError(
                "ORBITKV_CAPSULE_STORE is required by the runtime StatePlan"
            )
        _capsule_identity()
        atexit.register(_stop_capsules)
        atexit.register(_stop_bindings)

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
                owner_plan_path,
                _POLICY,
            )
        elif transport == "sidecar":
            _OWNER = SidecarOwnerClient(
                os.environ.get("ORBITKV_BIN", "orbitkv"),
                owner_plan_path,
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
        if _capsules_enabled() and _UNIFORM_SWA_CONTRACT is None:
            HookRegistry.register(
                "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
                _hydrate_capsule_for_admission,
                HookType.AROUND,
            )
    elif _PHYSICAL_PLAN is not None:
        HookRegistry.register(
            "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
            _run_with_physical_contract,
            HookType.AROUND,
        )

    if _UNIFORM_SWA_CONTRACT is not None:
        if _UNIFORM_SWA_CONTRACT["physical_backend"] == "paged_periodic":
            HookRegistry.register(
                "sglang.srt.mem_cache.allocator.swa.PureSWATokenToKVPoolAllocator",
                _build_paged_periodic_allocator(),
                HookType.REPLACE,
            )
            HookRegistry.register(
                "sglang.srt.mem_cache.chunk_cache.PureSWAChunkCache.cache_finished_req",
                _finish_paged_periodic_request,
                HookType.AROUND,
            )
        HookRegistry.register(
            "sglang.srt.configs.model_config.ModelConfig._derive_hybrid_model",
            _activate_uniform_swa_model_config,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.models.llama.LlamaAttention.__init__",
            _activate_uniform_swa_kernel,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.model_executor.model_runner_components.load_model_utils.resolve_sliding_window_size",
            _resolve_uniform_swa_window,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure",
            _validate_uniform_swa_runtime,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.managers.tp_worker.TpModelWorker.get_worker_info",
            _expand_uniform_swa_worker_info,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.managers.scheduler.Scheduler.init_req_max_new_tokens",
            _init_uniform_swa_max_new_tokens,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
            (
                _admit_uniform_swa_capsule_request
                if _capsules_enabled()
                else _admit_uniform_swa_request
            ),
            HookType.AROUND,
        )
        if _UNIFORM_SWA_CONTRACT["cuda_graph_mode"] == "decode":
            HookRegistry.register(
                "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute",
                _record_decode_graph_replay,
                HookType.AROUND,
            )

    trace_allocations = _trace_allocations_enabled()
    trace_graph_replays = (
        _UNIFORM_SWA_CONTRACT is not None
        and _UNIFORM_SWA_CONTRACT["cuda_graph_mode"] == "decode"
    )
    if (trace_allocations or trace_graph_replays) and _WRITER is None:
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
                "runtime_state_plan_fingerprint": (
                    _RUNTIME_STATE_PLAN.get("artifact_fingerprint")
                    if _RUNTIME_STATE_PLAN is not None
                    else None
                ),
                "policy": _POLICY,
            }
        )
