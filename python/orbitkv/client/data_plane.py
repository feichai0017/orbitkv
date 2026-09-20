"""Framework-neutral cache data-plane clients.

The vLLM and SGLang adapters should depend on this narrow surface instead of
depending directly on either the compatibility gRPC client or the local IPC
protocol. Lifecycle operations such as registration, health, and session
watching intentionally remain outside this module.
"""

from __future__ import annotations

import os
import select
import socket
import stat
import threading
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from orbitkv import EngineRpcClient, LocalQueryClient, PyLoadState, QueryLoading, QueryReady


@dataclass(frozen=True, slots=True)
class RestoreStatus:
    """One non-blocking observation of a submitted restore."""

    done: bool
    success: bool
    message: str = ""


class RestoreHandle(Protocol):
    """Opaque restore identity owned by one data-plane client."""

    @property
    def key(self) -> str: ...


class CacheDataClient(Protocol):
    """Hot cache operations shared by framework adapters."""

    @property
    def transport(self) -> str: ...

    def query_prefetch(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
        wait_for_full_prefix: bool = False,
        group_id: int = 0,
    ) -> QueryLoading | QueryReady: ...

    def release(self, lease: bytes) -> None: ...

    def save(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]: ...

    def start_restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups: list[list[str]],
        loads: list[tuple[bytes, list[list[int | None]]]],
    ) -> RestoreHandle: ...

    def restore_completions_ready(self) -> bool: ...

    def poll_restore(self, handle: RestoreHandle) -> RestoreStatus: ...


@dataclass(frozen=True, slots=True)
class _GrpcRestoreHandle:
    state: PyLoadState
    shm_name: str

    @property
    def key(self) -> str:
        return self.shm_name


class GrpcDataClient:
    """Compatibility data plane backed by the existing gRPC client."""

    def __init__(self, client: EngineRpcClient):
        self._client = client

    @property
    def transport(self) -> str:
        return "grpc"

    def query_prefetch(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
        wait_for_full_prefix: bool = False,
        group_id: int = 0,
    ) -> QueryLoading | QueryReady:
        if group_id:
            return self._client.query_prefetch(
                instance_id,
                block_hashes,
                req_id=req_id,
                wait_for_full_prefix=wait_for_full_prefix,
                group_id=group_id,
            )
        return self._client.query_prefetch(
            instance_id,
            block_hashes,
            req_id=req_id,
            wait_for_full_prefix=wait_for_full_prefix,
        )

    def release(self, lease: bytes) -> None:
        self._client.release(lease)

    def save(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]:
        return self._client.save(instance_id, tp_rank, pp_rank, device_id, saves)

    def start_restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups: list[list[str]],
        loads: list[tuple[bytes, list[list[int | None]]]],
    ) -> RestoreHandle:
        state = PyLoadState()
        shm_name = state.shm_name()
        ok, message = self._client.load(
            instance_id,
            tp_rank,
            device_id,
            shm_name,
            layer_groups,
            loads,
        )
        if not ok:
            raise RuntimeError(message or "gRPC restore submission failed")
        return _GrpcRestoreHandle(state=state, shm_name=shm_name)

    def restore_completions_ready(self) -> bool:
        return True

    def poll_restore(self, handle: RestoreHandle) -> RestoreStatus:
        if not isinstance(handle, _GrpcRestoreHandle):
            raise TypeError("restore handle does not belong to the gRPC data client")
        if not handle.state.is_ready():
            return RestoreStatus(done=False, success=False)
        state = handle.state.get_state()
        return RestoreStatus(
            done=True,
            success=state >= 0,
            message="" if state >= 0 else f"load state {state}",
        )


@dataclass(frozen=True, slots=True)
class _LocalRestoreHandle:
    operation_id: int
    session_epoch: int

    @property
    def key(self) -> str:
        return f"local:{self.session_epoch}:{self.operation_id}"


class LocalDataClient:
    """Local data plane backed by UDS bootstrap and iceoryx2 commands."""

    _FALLBACK_POLL_SECONDS = 0.05

    def __init__(
        self,
        bootstrap_socket: str,
        *,
        timeout_ms: int = 5_000,
        spin_iterations: int = 64,
    ):
        self._bootstrap_socket = bootstrap_socket
        self._client = LocalQueryClient(
            bootstrap_socket,
            timeout_ms=timeout_ms,
            spin_iterations=spin_iterations,
        )
        self._request_lock = threading.Lock()
        self._next_request_id = 1
        self._last_completion_poll = time.monotonic()

    @property
    def transport(self) -> str:
        return "local"

    @property
    def bootstrap_socket(self) -> str:
        return self._bootstrap_socket

    def query_prefetch(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
        wait_for_full_prefix: bool = False,
        group_id: int = 0,
    ) -> QueryLoading | QueryReady:
        return self._client.query_bundle(
            instance_id,
            block_hashes,
            req_id,
            wait_for_full_prefix=wait_for_full_prefix,
            group_id=group_id,
            request_id=self._request_id(),
        )

    def release(self, lease: bytes) -> None:
        self._client.release(lease, request_id=self._request_id())

    def save(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]:
        self._client.publish(
            instance_id,
            tp_rank,
            pp_rank,
            device_id,
            saves,
            request_id=self._request_id(),
        )
        return True, ""

    def start_restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups: list[list[str]],
        loads: list[tuple[bytes, list[list[int | None]]]],
    ) -> RestoreHandle:
        operation_id = self._client.restore_submit(
            instance_id,
            tp_rank,
            device_id,
            layer_groups,
            loads,
            request_id=self._request_id(),
        )
        return _LocalRestoreHandle(
            operation_id=operation_id,
            session_epoch=self._client.session_epoch,
        )

    def restore_completions_ready(self) -> bool:
        now = time.monotonic()
        if now - self._last_completion_poll >= self._FALLBACK_POLL_SECONDS:
            self._last_completion_poll = now
            return True

        fd = self._client.notification_fd
        readable, _, _ = select.select((fd,), (), (), 0)
        if not readable:
            return False
        try:
            os.read(fd, 8)
        except BlockingIOError:
            return False
        self._last_completion_poll = now
        return True

    def poll_restore(self, handle: RestoreHandle) -> RestoreStatus:
        if not isinstance(handle, _LocalRestoreHandle):
            raise TypeError("restore handle does not belong to the local data client")
        if handle.session_epoch != self._client.session_epoch:
            raise RuntimeError("restore handle belongs to a stale local sidecar session")
        state, message = self._client.restore_poll(
            handle.operation_id,
            request_id=self._request_id(),
        )
        if state == "pending":
            return RestoreStatus(done=False, success=False)
        if state == "succeeded":
            return RestoreStatus(done=True, success=True)
        if state == "failed":
            return RestoreStatus(done=True, success=False, message=message)
        raise RuntimeError(f"unknown local restore state {state!r}")

    def _request_id(self) -> int:
        with self._request_lock:
            request_id = self._next_request_id
            if request_id == (1 << 64) - 1:
                raise OverflowError("local data request ids exhausted")
            self._next_request_id += 1
        return request_id


def resolve_local_bootstrap_sockets(
    *,
    enabled: object,
    endpoints: tuple[str, ...],
    bootstrap_socket: object = None,
    shard_bootstrap_sockets: object = None,
) -> tuple[str, ...] | None:
    """Resolve the local data plane, using it automatically when available.

    ``enabled="auto"`` selects local IPC only when every derived bootstrap
    path is a live Unix socket. Explicit ``True`` remains fail-fast, while
    explicit ``False`` forces the compatibility gRPC data plane.
    """
    if enabled == "auto":
        automatic = True
    elif isinstance(enabled, bool):
        automatic = False
    else:
        raise ValueError("orbitkv.local_data must be true, false, or 'auto'")

    if enabled is False:
        if bootstrap_socket is not None or shard_bootstrap_sockets is not None:
            raise ValueError("local bootstrap sockets require orbitkv.local_data=true or 'auto'")
        return None
    if not endpoints:
        raise ValueError("local data requires at least one gRPC endpoint")

    if bootstrap_socket is not None and shard_bootstrap_sockets is not None:
        raise ValueError(
            "configure either orbitkv.local_bootstrap_socket or "
            "orbitkv.tp_shard_bootstrap_sockets, not both"
        )

    if shard_bootstrap_sockets is not None:
        if not isinstance(shard_bootstrap_sockets, (list, tuple)):
            raise ValueError("orbitkv.tp_shard_bootstrap_sockets must be a list of socket paths")
        sockets = tuple(shard_bootstrap_sockets)
    elif bootstrap_socket is not None:
        if len(endpoints) > 1:
            raise ValueError(
                "orbitkv.local_bootstrap_socket only supports one TP shard; "
                "use orbitkv.tp_shard_bootstrap_sockets"
            )
        sockets = (bootstrap_socket,)
    else:
        sockets = tuple(_default_bootstrap_socket(endpoint) for endpoint in endpoints)

    if len(sockets) != len(endpoints):
        raise ValueError(
            f"configured {len(sockets)} local bootstrap sockets for {len(endpoints)} TP shards"
        )
    if any(not isinstance(socket, str) or not socket for socket in sockets):
        raise ValueError("local bootstrap socket configuration must contain non-empty strings")
    if automatic:
        if len(set(sockets)) != len(sockets) or not all(
            _endpoint_is_local(endpoint) and _is_unix_socket(socket_path)
            for endpoint, socket_path in zip(endpoints, sockets, strict=True)
        ):
            return None
    elif len(set(sockets)) != len(sockets):
        raise ValueError("orbitkv.tp_shard_bootstrap_sockets must not contain duplicates")
    return sockets


def _default_bootstrap_socket(endpoint: str) -> str:
    port = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}").port
    if port is None:
        raise ValueError(
            "cannot derive the local bootstrap socket from the gRPC endpoint; "
            "set orbitkv.local_bootstrap_socket"
        )
    return f"/tmp/orbitkv-{port}.sock"


def _is_unix_socket(path: str) -> bool:
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False


def _endpoint_is_local(endpoint: str) -> bool:
    host = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}").hostname
    if host is None:
        return False
    if host in {"localhost", "::1"} or host.startswith("127."):
        return True
    try:
        addresses = socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, socket_type, protocol, _, address in addresses:
        try:
            with socket.socket(family, socket_type, protocol) as probe:
                probe.bind(address)
        except OSError:
            continue
        return True
    return False


__all__ = [
    "CacheDataClient",
    "GrpcDataClient",
    "LocalDataClient",
    "RestoreHandle",
    "RestoreStatus",
    "resolve_local_bootstrap_sockets",
]
