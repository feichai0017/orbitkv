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
from importlib.metadata import version
from typing import Any

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from orbitkv.client import CacheManagerClient
from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
from orbitkv.identity import model_identity, state_namespace

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
        finally:
            if threshold == self.num_layers - 1:
                self._futures.pop(index, None)

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


class OrbitKVLinker(UnifiedCacheLinker):
    """Store aligned SGLang GPU pages using OrbitKV's CUDA IPC data path.

    The SGLang radix tree owns GPU slots. Its offload/load callbacks hold those
    slots until the completion queues acknowledge the Cache Manager transfer.
    Each scheduler rank has one short-lived GPU registration and one stable,
    model-scoped namespace that other replicas of the same rank can reuse.
    """

    def __init__(self, server_args: Any, params: Any, *, components: set[ComponentType]):
        if components != {ComponentType.FULL}:
            raise ValueError("OrbitKV direct GPU linker currently supports full-attention KV only")
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        if type(kvcache) in (MHATokenToKVPool, MLATokenToKVPool):
            # SGLang's stock direct-linker assembler does not implement its
            # plain KV strategy. Both ordinary layouts expose page-aligned
            # rows, so they can use the same DevicePoolEntry contract.
            buffers = (
                [list(kvcache.k_buffer), list(kvcache.v_buffer)]
                if type(kvcache) is MHATokenToKVPool
                else [list(kvcache.kv_buffer)]
            )
            entry = DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=kvcache,
                components=buffers,
                layer_mapping={i: i for i in range(kvcache.layer_num)},
                page_size=self.page_size,
                rows_are_pages=bool(getattr(kvcache, "use_hnd", False)),
            )
            self.pool_group = DevicePoolGroup([entry], kvcache.layer_num, self.page_size)
        else:
            self.pool_group = resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=self.page_size,
                params=params,
                components=components,
            )
        if set(self.pool_group.entry_map) != {PoolName.KV}:
            raise ValueError(
                "OrbitKV direct GPU linker requires one KV pool; auxiliary GPU pools are unsupported"
            )
        self.pool = self.pool_group.entry_map[PoolName.KV]
        self._row_span = self.pool._row_span
        if self._row_span not in (1, self.page_size):
            raise ValueError(f"unsupported SGLang GPU page row span: {self._row_span}")

        self._layer_names = [f"kv:{i}" for i in range(len(self.pool.kv_buffer))]
        if not self._layer_names:
            raise ValueError("SGLang KV pool has no GPU buffers")
        self._num_blocks = []
        self._block_bytes = []
        for tensor in self.pool.kv_buffer:
            if tensor.device.type != "cuda" or not tensor.is_contiguous():
                raise ValueError("OrbitKV direct GPU linker requires contiguous CUDA KV buffers")
            if tensor.shape[0] % self._row_span:
                raise ValueError("SGLang GPU KV buffer is not page aligned")
            self._num_blocks.append(tensor.shape[0] // self._row_span)
            self._block_bytes.append(tensor.stride(0) * tensor.element_size() * self._row_span)
        if len(set(self._num_blocks)) != 1:
            raise ValueError("SGLang GPU KV buffers have different page counts")

        if server_args.enable_lora:
            raise ValueError(
                "OrbitKV requires immutable adapter identities; dynamic LoRA is unsupported"
            )
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        tp_rank = parallel.tp_rank
        tp_size = parallel.tp_size
        computation = {
            "weight_version": getattr(server_args, "weight_version", None),
            "quantization": getattr(server_args, "quantization", None),
            "model_overrides": getattr(server_args, "json_model_override_args", None),
            "dtype": server_args.dtype,
            "attention_backend": server_args.attention_backend,
            "prefill_attention_backend": server_args.prefill_attention_backend,
            "decode_attention_backend": server_args.decode_attention_backend,
        }
        representation = {
            "kv_cache_dtype": getattr(server_args, "kv_cache_dtype", None),
            "tp": [tp_rank, tp_size],
            "pp": [params.pp_rank, params.pp_size],
            "cp": [params.attn_cp_rank, params.attn_cp_size],
            "page_size": self.page_size,
            "buffers": [
                {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape[1:]),
                    "stride": list(tensor.stride()[1:]),
                    "block_bytes": block_bytes,
                }
                for tensor, block_bytes in zip(self.pool.kv_buffer, self._block_bytes, strict=True)
            ],
        }
        self.namespace = state_namespace(
            engine="sglang",
            engine_version=version("sglang"),
            model=model_identity(
                server_args.model_path,
                revision=server_args.revision,
                tokenizer=server_args.tokenizer_path,
                tokenizer_revision=server_args.revision,
            ),
            computation=computation,
            representation=representation,
        )
        self.instance_id = f"sglang-{uuid.uuid4().hex}"
        self.device_id = resolve_device_id()
        endpoint = os.environ.get("ORBITKV_SGLANG_ENDPOINT", "unix:///run/orbitkv/orbitkv.sock")
        if not endpoint.startswith("unix://"):
            raise ValueError("ORBITKV_SGLANG_ENDPOINT must be a unix:// socket")
        self.client = CacheManagerClient(endpoint.removeprefix("unix://"))
        try:
            self.client.start_session_watcher(self.instance_id, self.namespace, 1, 1)
            wrappers = [serialize_gpu_buffer(tensor) for tensor in self.pool.kv_buffer]
            ok, message = self.client.register_context_batch(
                self.instance_id,
                self.namespace,
                0,
                0,
                1,
                1,
                self.device_id,
                self._layer_names,
                wrappers,
                self._num_blocks,
                self._block_bytes,
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

        self.layer_done_counter = _LayerDoneCounter(self.pool_group.num_layers)
        self._lookups: dict[str, _Lookup] = {}
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
            len(self._layer_names),
            self._num_blocks[0],
            self.device_id,
        )

    @staticmethod
    def _hashes(keys: list[str] | tuple[str, ...]) -> list[bytes]:
        return [hashlib.sha256(key.encode()).digest() for key in keys]

    def _block_ids(self, indices: torch.Tensor, expected_pages: int) -> list[int]:
        if indices.numel() != expected_pages * self.page_size:
            raise ValueError("SGLang GPU indices do not cover exactly the requested pages")
        rows = self.pool.prepare_locations(indices)
        block_ids = [row // self._row_span for row in rows]
        if any(block_id >= self._num_blocks[0] for block_id in block_ids):
            raise ValueError("SGLang GPU page index is outside its registered buffers")
        return block_ids

    def _release_lookup(self, rid: str) -> None:
        lookup = self._lookups.pop(rid, None)
        if lookup is not None:
            self.client.release(lookup.lease)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        self._release_lookup(rid)
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("OrbitKV direct linker expected one KV lookup")
        keys = tuple(transfers[0].keys or ())
        if not keys:
            return []
        from orbitkv import QueryReady

        result = self.client.query_prefetch(
            self.instance_id, self._hashes(keys), rid, wait_for_full_prefix=False
        )
        if not isinstance(result, QueryReady):
            return []
        hit_pages = result.num_hit_blocks
        if hit_pages:
            if not result.lease:
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
            block_ids = self._block_ids(transfer.device_indices, len(keys))
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
                    for load in pending:
                        submitted += 1
                        restore = self.client.start_restore(
                            self.instance_id,
                            0,
                            self.device_id,
                            [self._layer_names],
                            [(load.lease, [list(load.targets)])],
                        )
                        deadline = time.monotonic() + 120
                        while True:
                            status = self.client.poll_restore(restore)
                            if status.done:
                                if not status.success:
                                    raise RuntimeError(status.message)
                                break
                            if time.monotonic() >= deadline:
                                raise TimeoutError("OrbitKV SGLang GPU restore timed out")
                            time.sleep(0.01)
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
        self._release_lookup(rid)
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
        block_ids = self._block_ids(transfer.device_indices, len(keys))
        hashes = self._hashes(keys)
        saves = [(name, block_ids, hashes) for name in self._layer_names]
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


def create_cache(ctx: Any) -> UnifiedRadixCache:
    """Factory selected by SGLang's ``--radix-cache-backend orbitkv``."""
    if ctx.disable_radix_cache:
        raise ValueError("OrbitKV direct GPU linker requires RadixCache")
    if ctx.enable_hierarchical_cache:
        raise ValueError("OrbitKV direct GPU linker does not support hierarchical cache")
    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm:
        raise ValueError("OrbitKV direct GPU linker currently supports full-attention KV only")
    if (
        ctx.is_dsa
        or ctx.params.is_eagle
        or ctx.params.mtp_draft_device_pools
        or ctx.params.component_registry_override
        or hasattr(ctx.params.req_to_token_pool, "req_to_c128_sidecar")
    ):
        raise ValueError(
            "OrbitKV direct GPU linker cannot restore DSA, draft, or auxiliary GPU state; "
            "select a backend with a complete recovery contract for that model"
        )
    from sglang.srt.runtime_context import get_disagg, get_memory

    if not get_memory().enable_unified_cache_external_linker:
        raise ValueError(
            "OrbitKV direct GPU linker requires --enable-unified-cache-external-linker "
            "so SGLang schedules GPU KV restores"
        )
    if get_disagg().disaggregation_decode_retraction_backup == "host_pool":
        raise ValueError("OrbitKV direct GPU linker does not support host-pool retraction")

    # SGLang's built-in unified-cache factory hardcodes Mooncake/Mori when the
    # external-linker flag is set. Construct its public RadixCache component
    # directly and attach OrbitKV through the public linker interface.
    ctx.params.tree_components = (ComponentType.FULL,)
    cache = UnifiedRadixCache(ctx.params)
    linker = OrbitKVLinker(ctx.server_args, ctx.params, components=set(cache.components))
    try:
        cache.init_cache_linker(linker)
    except Exception:
        linker.close()
        raise
    counter = linker.layer_done_counter
    kvcache = ctx.params.token_to_kv_pool_allocator.get_kvcache()
    kvcache.register_layer_transfer_counter(counter)
    ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
    return cache
