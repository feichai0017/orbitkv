"""Framework-neutral cache operations.

The connection module opens a process connection. A cache hit can come from
DRAM, SSD, or a peer node without changing this interface. The process-local
transport uses UDS for lifecycle and iceoryx2 descriptors for hot commands.
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

from orbitkv import LocalQueryClient, QueryLoading, QueryReady


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


class CacheLifecycleClient(Protocol):
    """Lifecycle surface of the local Cache Manager connection."""

    def health(self) -> tuple[bool, str]: ...

    def put_host_page(self, namespace: str, key: bytes, data: bytes) -> None: ...

    def get_host_page(self, namespace: str, key: bytes) -> bytes | None: ...

    def has_host_page(self, namespace: str, key: bytes) -> bool: ...

    def register_context_batch(
        self,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        pp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        layer_names: list[str],
        wrapper_bytes_list: list[bytes],
        num_blocks_list: list[int],
        bytes_per_block_list: list[int],
        kv_stride_bytes_list: list[int],
        segments_list: list[int],
        transfer_backend: str,
        page_first: bool,
        layer_group_ids: list[int] | None = None,
    ) -> tuple[bool, str]: ...

    def unregister_context(self, instance_id: str) -> tuple[bool, str]: ...

    def start_session_watcher(
        self, instance_id: str, namespace: str, tp_size: int, world_size: int
    ) -> None: ...


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
        self._client_options = (timeout_ms, spin_iterations)
        self._publish_client: LocalQueryClient | None = None
        self._publish_lock = threading.Lock()
        self._closed = False
        self._request_lock = threading.Lock()
        self._next_request_id = 1
        self._last_completion_poll = time.monotonic()

    @property
    def transport(self) -> str:
        return "local"

    @property
    def bootstrap_socket(self) -> str:
        return self._bootstrap_socket

    def close(self) -> None:
        with self._publish_lock:
            self._closed = True
            self._client.close()
            if self._publish_client is not None:
                self._publish_client.close()

    def health(self) -> tuple[bool, str]:
        return self._client.health()

    def put_host_page(self, namespace: str, key: bytes, data: bytes) -> None:
        self._client.put_host_page(namespace, key, data)

    def get_host_page(self, namespace: str, key: bytes) -> bytes | None:
        return self._client.get_host_page(namespace, key)

    def has_host_page(self, namespace: str, key: bytes) -> bool:
        return self._client.has_host_page(namespace, key)

    def register_context_batch(self, *args, **kwargs) -> tuple[bool, str]:
        return self._client.register_context_batch(*args, **kwargs)

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        return self._client.unregister_context(instance_id)

    def start_session_watcher(
        self, instance_id: str, namespace: str, tp_size: int, world_size: int
    ) -> None:
        self._client.start_session_watcher(instance_id, namespace, tp_size, world_size)

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
        # Publish retains its descriptor until D2H finishes. Give it a
        # separate session so one long save cannot serialize later restores
        # behind the query/restore session's descriptor lock.
        self._publisher().publish(
            instance_id,
            tp_rank,
            pp_rank,
            device_id,
            saves,
            request_id=self._request_id(),
        )
        return True, ""

    def _publisher(self) -> LocalQueryClient:
        with self._publish_lock:
            if self._closed:
                raise RuntimeError("local data client is closed")
            if self._publish_client is None:
                timeout_ms, spin_iterations = self._client_options
                self._publish_client = LocalQueryClient(
                    self._bootstrap_socket,
                    timeout_ms=timeout_ms,
                    spin_iterations=spin_iterations,
                )
            return self._publish_client

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
            raise RuntimeError("restore handle belongs to a stale Cache Manager session")
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
    endpoints: tuple[str, ...],
    bootstrap_socket: object = None,
    shard_bootstrap_sockets: object = None,
) -> tuple[str, ...]:
    """Resolve the process endpoint for same-host cache clients.

    Every inference shard must connect to a Cache Manager on its own host.
    """
    if not endpoints:
        raise ValueError("cache client requires at least one endpoint")
    if not all(_endpoint_is_local(endpoint) for endpoint in endpoints):
        raise ValueError(
            "inference clients require a node-local Cache Manager; "
            "configure a Cache Manager on each inference host"
        )

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
    if len(set(sockets)) != len(sockets):
        raise ValueError("local bootstrap sockets must be distinct for TP shards")
    missing = [socket_path for socket_path in sockets if not _is_unix_socket(socket_path)]
    if missing:
        raise ConnectionError(
            "OrbitKV Cache Manager Unix socket is unavailable: "
            + ", ".join(missing)
            + "; start the node-local Cache Manager"
        )
    return sockets


def _default_bootstrap_socket(endpoint: str) -> str:
    port = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}").port
    if port is None:
        raise ValueError(
            "cannot derive the process socket from the configured endpoint; "
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
    "LocalDataClient",
    "RestoreHandle",
    "RestoreStatus",
    "resolve_local_bootstrap_sockets",
]
