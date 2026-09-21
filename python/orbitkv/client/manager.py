"""Framework-neutral cache operations.

The connection module opens a process connection. A cache hit can come from
DRAM, SSD, or a peer node without changing this interface. The process-local
transport uses UDS for lifecycle and iceoryx2 descriptors for hot commands.
"""

from __future__ import annotations

import os
import select
import threading
import time
from dataclasses import dataclass

from orbitkv import ChannelClient, QueryLoading, QueryReady


@dataclass(frozen=True, slots=True)
class RestoreStatus:
    """One non-blocking observation of a submitted restore."""

    done: bool
    success: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class RestoreHandle:
    operation_id: int
    session_epoch: int

    @property
    def key(self) -> str:
        return f"manager:{self.session_epoch}:{self.operation_id}"


@dataclass(slots=True)
class _Query:
    operation_id: int
    revision: int
    hashes: tuple[bytes, ...]
    wait_for_full_prefix: bool


class CacheManagerClient:
    """Cache operations through the same-host Cache Manager process channel."""

    _FALLBACK_POLL_SECONDS = 0.05
    _MAX_WARMUPS = 16
    _WARMUP_SECONDS = 5.0

    def __init__(
        self,
        bootstrap_socket: str,
        *,
        timeout_ms: int = 5_000,
        spin_iterations: int = 64,
    ):
        self._bootstrap_socket = bootstrap_socket
        self._client = ChannelClient(
            bootstrap_socket,
            timeout_ms=timeout_ms,
            spin_iterations=spin_iterations,
        )
        self._client_options = (timeout_ms, spin_iterations)
        self._publish_client: ChannelClient | None = None
        self._publish_lock = threading.Lock()
        self._closed = False
        self._request_lock = threading.Lock()
        self._next_request_id = 1
        self._query_lock = threading.Lock()
        self._next_operation_id = 1
        self._queries: dict[tuple[str, str, int], _Query] = {}
        self._warmups: dict[tuple[str, str, int], tuple[int, float]] = {}
        self._last_completion_poll = time.monotonic()

    @property
    def transport(self) -> str:
        return "iceoryx2"

    @property
    def bootstrap_socket(self) -> str:
        return self._bootstrap_socket

    def close(self) -> None:
        with self._publish_lock:
            self._closed = True
            self._client.close()
            if self._publish_client is not None:
                self._publish_client.close()
        with self._query_lock:
            self._queries.clear()
            self._warmups.clear()

    def health(self) -> tuple[bool, str]:
        return self._client.health()

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
        key = (instance_id, req_id, group_id)
        hashes = tuple(block_hashes)
        with self._query_lock:
            warmup = self._warmups.pop(key, None)
            if warmup is not None:
                self._client.cancel_query(warmup[0], 1, request_id=self._request_id())
            query = self._queries.get(key)
            changed = query is not None and (
                query.hashes != hashes or query.wait_for_full_prefix != wait_for_full_prefix
            )
            submit = query is None or changed
            if query is None:
                if self._next_operation_id >= 1 << 64:
                    raise OverflowError("Cache Manager query ids exhausted")
                query = _Query(self._next_operation_id, 1, hashes, wait_for_full_prefix)
                self._next_operation_id += 1
                self._queries[key] = query
            elif changed:
                if query.revision == (1 << 64) - 1:
                    raise OverflowError("Cache Manager query revisions exhausted")
                query.revision += 1
                query.hashes = hashes
                query.wait_for_full_prefix = wait_for_full_prefix
            if submit:
                result = self._client.query_submit(
                    instance_id,
                    block_hashes,
                    req_id,
                    query.operation_id,
                    query.revision,
                    wait_for_full_prefix=wait_for_full_prefix,
                    group_id=group_id,
                    request_id=self._request_id(),
                )
            else:
                result = self._client.query_poll(
                    query.operation_id, query.revision, request_id=self._request_id()
                )
            if not isinstance(result, QueryLoading) or not result.admitted:
                self._queries.pop(key)
            return result

    def warm_prefix(self, instance_id: str, block_hashes: list[bytes], req_id: str) -> bool:
        """Try preparing queued demand. Completion retains no lease or reservation.

        Admission revalidates the prefix with a new query. Cancellation withdraws
        interest while submitted reads drain under their original byte budget.
        """
        if not block_hashes or os.environ.get("ORBITKV_QUEUE_WARMUP", "1") == "0":
            return False
        key = (instance_id, req_id, 0)
        with self._query_lock:
            now = time.monotonic()
            for expired, (operation, submitted) in list(self._warmups.items()):
                if now - submitted >= self._WARMUP_SECONDS:
                    self._client.cancel_query(operation, 1, request_id=self._request_id())
                    del self._warmups[expired]
            if (
                key in self._warmups
                or key in self._queries
                or len(self._warmups) >= self._MAX_WARMUPS
            ):
                return False
            if self._next_operation_id >= 1 << 64:
                raise OverflowError("Cache Manager query ids exhausted")
            operation = self._next_operation_id
            self._next_operation_id += 1
            result = self._client.query_submit(
                instance_id,
                block_hashes,
                req_id,
                operation,
                1,
                warmup=True,
                request_id=self._request_id(),
            )
            if isinstance(result, QueryLoading):
                if result.admitted:
                    self._warmups[key] = (operation, now)
                return result.admitted
            if not isinstance(result, QueryReady) or result.num_hit_blocks or result.lease:
                raise RuntimeError("warmup must not return a restore lease or hit promise")
            return True

    def release(self, lease: bytes) -> None:
        self._client.release(lease, request_id=self._request_id())

    def cancel_query(self, instance_id: str, req_id: str, group_id: int = 0) -> None:
        with self._query_lock:
            key = (instance_id, req_id, group_id)
            warmup = self._warmups.pop(key, None)
            if warmup is not None:
                self._client.cancel_query(warmup[0], 1, request_id=self._request_id())
            query = self._queries.pop(key, None)
            if query is not None:
                self._client.cancel_query(
                    query.operation_id, query.revision, request_id=self._request_id()
                )

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

    def _publisher(self) -> ChannelClient:
        with self._publish_lock:
            if self._closed:
                raise RuntimeError("Cache Manager client is closed")
            if self._publish_client is None:
                timeout_ms, spin_iterations = self._client_options
                self._publish_client = ChannelClient(
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
        return RestoreHandle(
            operation_id=operation_id,
            session_epoch=self._client.session_epoch,
        )

    def restore_completions_ready(self, *, timeout: float = 0.0) -> bool:
        now = time.monotonic()
        fallback_remaining = self._FALLBACK_POLL_SECONDS - (now - self._last_completion_poll)
        if fallback_remaining <= 0:
            self._last_completion_poll = now
            return True

        fd = self._client.notification_fd
        readable, _, _ = select.select((fd,), (), (), min(timeout, fallback_remaining))
        now = time.monotonic()
        if not readable:
            if now - self._last_completion_poll >= self._FALLBACK_POLL_SECONDS:
                self._last_completion_poll = now
                return True
            return False
        try:
            os.read(fd, 8)
        except BlockingIOError:
            return False
        self._last_completion_poll = now
        return True

    def wait_restore(self, handle: RestoreHandle, *, timeout: float) -> RestoreStatus:
        """Wait for terminal state; a deadline never releases the destination pages."""
        deadline = time.monotonic() + timeout
        while True:
            status = self.poll_restore(handle)
            if status.done:
                return status
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OrbitKV GPU restore timed out")
            self.restore_completions_ready(timeout=remaining)

    def poll_restore(self, handle: RestoreHandle) -> RestoreStatus:
        if not isinstance(handle, RestoreHandle):
            raise TypeError("restore handle does not belong to this Cache Manager client")
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
        raise RuntimeError(f"unknown restore state {state!r}")

    def _request_id(self) -> int:
        with self._request_lock:
            request_id = self._next_request_id
            if request_id == (1 << 64) - 1:
                raise OverflowError("Cache Manager request ids exhausted")
            self._next_request_id += 1
        return request_id


__all__ = [
    "CacheManagerClient",
    "RestoreHandle",
    "RestoreStatus",
]
