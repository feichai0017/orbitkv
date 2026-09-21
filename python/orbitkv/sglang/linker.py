"""SGLang GPU-page integration with the node-local Cache Manager."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

from orbitkv.client import CacheManagerClient
from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
from orbitkv.logging_utils import trace_transfer

from .config import derive_namespace
from .layout import GpuLayout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Lookup:
    keys: tuple[str, ...]
    lease: bytes
    hit_pages: int


@dataclass(frozen=True)
class _Load:
    rid: str
    lease: bytes
    targets: tuple[int | None, ...]


class _LayerDoneCounter:
    """SGLang waits on this counter before consuming restored GPU KV."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self._futures: dict[int, list[Future[None]]] = {}
        self.request_ids: dict[int, list[str]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self._futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self._futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
            if threshold == 0:
                for rid in self.request_ids.get(index, ()):
                    trace_transfer("first_use", rid, engine="sglang")
        finally:
            if threshold == self.num_layers - 1:
                self._futures.pop(index, None)
                self.request_ids.pop(index, None)

    def complete(self, index: int, error: Exception | None = None) -> None:
        for future in self._futures[index]:
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self._futures.clear()
        self.request_ids.clear()


class OrbitKVLinker(UnifiedCacheLinker):
    """Store aligned SGLang GPU pages using OrbitKV's CUDA IPC data path.

    The SGLang radix tree owns GPU slots. Its offload/load callbacks hold those
    slots until the completion queues acknowledge the Cache Manager transfer.
    Each scheduler rank has one short-lived GPU registration and one stable,
    model-scoped namespace that other replicas of the same rank can reuse.
    """

    _RESTORE_WINDOW = 8
    _QUERY_WAIT_SECONDS = 5.0

    def __init__(self, server_args: Any, params: Any, *, components: set[ComponentType]):
        if components != {ComponentType.FULL}:
            raise ValueError("OrbitKV direct GPU linker currently supports full-attention KV only")
        self.layout = GpuLayout.from_pool(params, components)
        self.page_size = self.layout.page_size
        self.namespace = derive_namespace(server_args, params, self.layout)
        self.instance_id = f"sglang-{uuid.uuid4().hex}"
        self.device_id = resolve_device_id()
        endpoint = os.environ.get("ORBITKV_SGLANG_ENDPOINT", "unix:///run/orbitkv/orbitkv.sock")
        if not endpoint.startswith("unix://"):
            raise ValueError("ORBITKV_SGLANG_ENDPOINT must be a unix:// socket")
        self.client = CacheManagerClient(endpoint.removeprefix("unix://"))
        try:
            self.client.start_session_watcher(self.instance_id, self.namespace, 1, 1)
            wrappers = [serialize_gpu_buffer(tensor) for tensor in self.layout.pool.kv_buffer]
            ok, message = self.client.register_context_batch(
                self.instance_id,
                self.namespace,
                0,
                0,
                1,
                1,
                self.device_id,
                self.layout.layer_names,
                wrappers,
                self.layout.num_blocks,
                self.layout.block_bytes,
                [0] * len(wrappers),
                [1] * len(wrappers),
                "direct",
                False,
            )
            if not ok:
                raise RuntimeError(f"OrbitKV GPU registration failed: {message}")
        except Exception:
            self.client.close()
            raise

        self.layer_done_counter = _LayerDoneCounter(self.layout.pool_group.num_layers)
        self._lookups: dict[str, _Lookup] = {}
        self._pending_queries: dict[str, tuple[tuple[str, ...], float]] = {}
        self._expired_queries: set[str] = set()
        self._queued_loads: dict[str, _Load] = {}
        self._load_queue: queue.Queue[tuple[int, list[_Load], torch.cuda.Event] | None] = (
            queue.Queue()
        )
        self._offload_queue: queue.Queue[
            tuple[list[tuple[str, list[int], list[bytes]]], torch.cuda.Event] | None
        ] = queue.Queue()
        self._completed_loads: queue.Queue[list[str]] = queue.Queue()
        self._load_error: Exception | None = None
        self._completed_offloads: queue.Queue[bool] = queue.Queue()
        self._load_thread = threading.Thread(
            target=self._load_worker, daemon=True, name="orbitkv-sglang-load"
        )
        self._offload_thread = threading.Thread(
            target=self._offload_worker, daemon=True, name="orbitkv-sglang-offload"
        )
        self._load_thread.start()
        self._offload_thread.start()
        logger.info(
            "OrbitKV direct GPU linker registered %s buffers, %s pages on device %s",
            len(self.layout.layer_names),
            self.layout.num_blocks[0],
            self.device_id,
        )

    @staticmethod
    def _hashes(keys: list[str] | tuple[str, ...]) -> list[bytes]:
        return [hashlib.sha256(key.encode()).digest() for key in keys]

    def _release_lookup(self, rid: str) -> None:
        lookup = self._lookups.pop(rid, None)
        if lookup is not None and lookup.lease:
            self.client.release(lookup.lease)

    def cancel_query(self, rid: str) -> None:
        self._pending_queries.pop(rid, None)
        self.client.cancel_query(self.instance_id, rid)
        self._release_lookup(rid)

    def expire_query(self, rid: str) -> None:
        self.cancel_query(rid)
        self._expired_queries.add(rid)

    def query_state(self, rid: str) -> int:
        pending = self._pending_queries.get(rid)
        if rid in self._expired_queries or (
            pending is not None and time.monotonic() - pending[1] >= self._QUERY_WAIT_SECONDS
        ):
            return 2
        return int(rid in self._pending_queries)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("OrbitKV direct linker expected one KV lookup")
        keys = tuple(transfers[0].keys or ())
        if not keys or rid in self._expired_queries:
            return []
        lookup = self._lookups.get(rid)
        if lookup is not None:
            if lookup.keys == keys:
                return list(range(1, lookup.hit_pages + 1))
            self._release_lookup(rid)
        pending = self._pending_queries.get(rid)
        if pending is not None and pending[0] != keys:
            self.cancel_query(rid)
            pending = None
        started = pending[1] if pending is not None else time.monotonic()
        if time.monotonic() - started >= self._QUERY_WAIT_SECONDS:
            logger.warning("Cache query wait expired; recomputing request %s", rid)
            self.expire_query(rid)
            return []
        from orbitkv import QueryReady

        result = self.client.query_prefetch(
            self.instance_id, self._hashes(keys), rid, wait_for_full_prefix=False
        )
        if not isinstance(result, QueryReady):
            self._pending_queries[rid] = (keys, started)
            return []
        self._pending_queries.pop(rid, None)
        hit_pages = result.num_hit_blocks
        if hit_pages and not result.lease:
            raise RuntimeError("OrbitKV reported GPU page hits without a restore lease")
        self._lookups[rid] = _Lookup(keys, result.lease, hit_pages)
        return list(range(1, hit_pages + 1))

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._check_load_failure()
        if rid in self._queued_loads:
            raise ValueError(f"duplicate OrbitKV load for {rid}")
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            self._release_lookup(rid)
            return False
        transfer = transfers[0]
        keys = tuple(transfer.keys or ())
        if not keys or transfer.device_indices is None:
            self._release_lookup(rid)
            return False
        lookup = self._lookups.pop(rid, None)
        if lookup is None:
            raise RuntimeError(f"SGLang load for {rid} has no OrbitKV lookup lease")
        try:
            block_ids = self.layout.block_ids(transfer.device_indices, len(keys))
            positions = {key: index for index, key in enumerate(lookup.keys[: lookup.hit_pages])}
            if len(positions) != lookup.hit_pages:
                raise ValueError("SGLang lookup contains duplicate page hashes")
            targets: list[int | None] = [None] * lookup.hit_pages
            for key, block_id in zip(keys, block_ids, strict=True):
                targets[positions[key]] = block_id
            self._queued_loads[rid] = _Load(rid, lookup.lease, tuple(targets))
        except Exception:
            self.client.release(lookup.lease)
            raise
        return True

    def start_layer_wise_loading(self) -> int:
        self._check_load_failure()
        if not self._queued_loads:
            return -1
        pending = list(self._queued_loads.values())
        self._queued_loads.clear()
        index = self.layer_done_counter.update_producer()
        self.layer_done_counter.request_ids[index] = [load.rid for load in pending]
        ready = torch.cuda.Event()
        ready.record()
        self._load_queue.put((index, pending, ready))
        return index

    def _load_worker(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                index, pending, ready = task
                submitted = 0
                try:
                    self._check_load_failure()
                    ready.synchronize()
                    for offset in range(0, len(pending), self._RESTORE_WINDOW):
                        restores = []
                        for load in pending[offset : offset + self._RESTORE_WINDOW]:
                            # A lost submission acknowledgement still leaves GPU ownership
                            # unresolved. Only leases never attempted can be released.
                            submitted += 1
                            trace_transfer("restore_submit", load.rid, engine="sglang")
                            restores.append(
                                self.client.start_restore(
                                    self.instance_id,
                                    0,
                                    self.device_id,
                                    [self.layout.layer_names],
                                    [(load.lease, [list(load.targets)])],
                                )
                            )
                        for load, restore in zip(
                            pending[offset : offset + self._RESTORE_WINDOW], restores, strict=True
                        ):
                            status = self.client.wait_restore(restore, timeout=120)
                            if not status.success:
                                raise RuntimeError(status.message)
                            trace_transfer("gpu_ready", load.rid, engine="sglang", success=True)
                    self.layer_done_counter.complete(index)
                    self._completed_loads.put([load.rid for load in pending])
                except Exception as error:
                    self._load_error = error
                    logger.exception("OrbitKV SGLang GPU restore failed")
                    for load in pending[submitted:]:
                        try:
                            self.client.release(load.lease)
                        except Exception:
                            logger.warning(
                                "Could not release an unsubmitted SGLang restore lease",
                                exc_info=True,
                            )
                    self.layer_done_counter.complete(index, error)
            finally:
                self._load_queue.task_done()

    def cancel_queued_load(self, rid: str) -> bool:
        self.cancel_query(rid)
        self._expired_queries.discard(rid)
        load = self._queued_loads.pop(rid, None)
        if load is None:
            return False
        self.client.release(load.lease)
        return True

    def num_completed_loads(self) -> int:
        self._check_load_failure()
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        self._check_load_failure()
        return self._completed_loads.get_nowait()

    def _check_load_failure(self) -> None:
        if self._load_error is not None:
            raise RuntimeError(
                "OrbitKV restore failed; GPU pages remain held until transfer teardown"
            ) from self._load_error

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            return False
        transfer = transfers[0]
        keys = list(transfer.keys or ())
        if not keys or transfer.device_indices is None:
            return False
        block_ids = self.layout.block_ids(transfer.device_indices, len(keys))
        hashes = self._hashes(keys)
        saves = [(name, block_ids, hashes) for name in self.layout.layer_names]
        ready = torch.cuda.Event()
        ready.record()
        self._offload_queue.put((saves, ready))
        return True

    def _offload_worker(self) -> None:
        while True:
            task = self._offload_queue.get()
            try:
                if task is None:
                    return
                saves, ready = task
                success = False
                try:
                    ready.synchronize()
                    success, message = self.client.save(
                        self.instance_id, 0, 0, self.device_id, saves
                    )
                    if not success:
                        logger.error("OrbitKV SGLang GPU offload failed: %s", message)
                except Exception:
                    logger.exception("OrbitKV SGLang GPU offload failed")
                self._completed_offloads.put(success)
            finally:
                self._offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self._completed_offloads.qsize()

    def pop_completed_offload(self) -> bool:
        return self._completed_offloads.get_nowait()

    def reset(self) -> None:
        self._load_queue.join()
        self._offload_queue.join()
        for rid in list(self._pending_queries):
            self.cancel_query(rid)
        self._expired_queries.clear()
        for rid in list(self._lookups):
            self._release_lookup(rid)
        for rid in list(self._queued_loads):
            self.cancel_queued_load(rid)
        while not self._completed_loads.empty():
            self._completed_loads.get_nowait()
        while not self._completed_offloads.empty():
            self._completed_offloads.get_nowait()
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self._load_queue.put(None)
        self._offload_queue.put(None)
        self._load_thread.join()
        self._offload_thread.join()
        try:
            self.client.unregister_context(self.instance_id)
        except Exception:
            logger.warning("Could not unregister SGLang GPU context", exc_info=True)
        self.client.close()
