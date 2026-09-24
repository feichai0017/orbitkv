"""
Worker-side connector logic.
"""

import queue
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from orbitkv import RestoreHandle
from orbitkv.client.gpu import serialize_gpu_buffer
from orbitkv.logging_utils import TRANSFER_TRACING, get_connector_logger, trace_transfer
from orbitkv.vllm.config import ConnectorContext, parse_env_int
from orbitkv.vllm.layout import CacheGroupLayout
from orbitkv.vllm.metadata import (
    OrbitKVConnectorMetadata,
    OrbitKVWorkerMetadata,
    SaveIntent,
)
from orbitkv.vllm.metrics import OrbitKVConnectorStats

logger = get_connector_logger()


if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext


_CROSS_LAYER_KEY = "ALL_LAYERS"

_LOAD_TIMEOUT_FLOOR_SECONDS = 30
_LOAD_TIMEOUT_RAW = parse_env_int("ORBITKV_LOAD_TIMEOUT_SECONDS", 120)
if _LOAD_TIMEOUT_RAW < _LOAD_TIMEOUT_FLOOR_SECONDS:
    logger.warning(
        "[OrbitKVConnector] ORBITKV_LOAD_TIMEOUT_SECONDS=%d clamped to %d "
        "(minimum guard against load-path deadlock)",
        _LOAD_TIMEOUT_RAW,
        _LOAD_TIMEOUT_FLOOR_SECONDS,
    )
    _LOAD_TIMEOUT_RAW = _LOAD_TIMEOUT_FLOOR_SECONDS


@dataclass
class SaveTask:
    metadata: OrbitKVConnectorMetadata
    request_ids: list[str]
    ready: torch.cuda.Event
    # HMA boundary-state jobs carried by this task; reported back to the
    # scheduler through OrbitKVWorkerMetadata once the batch is done.
    boundary_job_ids: list[int] = field(default_factory=list)


_KVCacheLayout = Literal["KV-first", "blocks-first"]


@dataclass(frozen=True)
class _KVCacheRegistrationInfo:
    layout: _KVCacheLayout
    num_blocks: int
    bytes_per_block: int
    kv_stride_bytes: int
    segments: int
    physical_blocks_per_logical_block: int


def _infer_kv_cache_registration(
    kv_cache: torch.Tensor,
    logical_block_size: int,
    *,
    is_mla: bool = False,
    is_recurrent_state: bool = False,
) -> _KVCacheRegistrationInfo:
    """Infer the OrbitKV registration from a vLLM KV cache tensor.

    vLLM may split one scheduler/manager block into multiple kernel KV rows
    when the attention backend only supports a smaller kernel block size. For
    example, FlashMLA supports 64-token kernel blocks while the manager can use
    128-token blocks. OrbitKV stores hashes at scheduler-block granularity, so
    each registered OrbitKV block must cover all physical rows for that logical
    block.
    """
    shape = tuple(kv_cache.shape)
    stride = tuple(kv_cache.stride())
    element_size = kv_cache.element_size()

    if logical_block_size <= 0:
        raise ValueError(f"logical block size must be > 0, got {logical_block_size}")

    if is_recurrent_state or not is_mla:
        if not is_recurrent_state and len(shape) >= 2 and shape[0] == 2:
            layout = "KV-first"
            num_blocks = shape[1]
            bytes_per_block = stride[1] * element_size
            kv_stride_bytes = stride[0] * element_size
            segments = 2
        else:
            layout = "blocks-first"
            num_blocks = shape[0]
            bytes_per_block = stride[0] * element_size
            kv_stride_bytes = 0
            segments = 1

        if num_blocks <= 0:
            raise ValueError(f"physical block count must be > 0, got {num_blocks}")
        if bytes_per_block == 0:
            raise ValueError(f"Invalid bytes_per_block: shape={shape} stride={stride}")

        return _KVCacheRegistrationInfo(
            layout=layout,
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            kv_stride_bytes=kv_stride_bytes,
            segments=segments,
            physical_blocks_per_logical_block=1,
        )

    layout = "blocks-first"
    physical_num_blocks = shape[0]
    if len(shape) == 4:
        # vLLM's standardized per-layer view is ``[B, H, N, C]``: kernel
        # blocks, head slots, tokens (states) per kernel block, content.
        physical_block_size = shape[2]
    elif len(shape) >= 2:
        # Legacy MLA cache: ``[num_blocks, block_size, head_dim]``.
        physical_block_size = shape[1]
    else:
        physical_block_size = logical_block_size
    physical_bytes_per_block = stride[0] * element_size
    kv_stride_bytes = 0
    segments = 1

    if physical_num_blocks <= 0:
        raise ValueError(f"physical block count must be > 0, got {physical_num_blocks}")
    if physical_block_size <= 0:
        raise ValueError(f"physical block size must be > 0, got {physical_block_size}")
    if logical_block_size % physical_block_size != 0:
        raise ValueError(
            "logical block size must be a multiple of physical block size "
            f"(logical={logical_block_size}, physical={physical_block_size})"
        )

    physical_blocks_per_logical_block = logical_block_size // physical_block_size
    if physical_num_blocks % physical_blocks_per_logical_block != 0:
        raise ValueError(
            "physical block count must be divisible by physical/logical split ratio "
            f"(physical_blocks={physical_num_blocks}, ratio={physical_blocks_per_logical_block})"
        )

    bytes_per_block = physical_bytes_per_block * physical_blocks_per_logical_block
    if bytes_per_block == 0:
        raise ValueError(f"Invalid bytes_per_block: shape={shape} stride={stride}")

    return _KVCacheRegistrationInfo(
        layout=layout,
        num_blocks=physical_num_blocks // physical_blocks_per_logical_block,
        bytes_per_block=bytes_per_block,
        kv_stride_bytes=kv_stride_bytes,
        segments=segments,
        physical_blocks_per_logical_block=physical_blocks_per_logical_block,
    )


def _registration_tensor(kv_cache) -> torch.Tensor:
    if not isinstance(kv_cache, (tuple, list)):
        return kv_cache

    states = tuple(kv_cache)
    if not states or not all(isinstance(state, torch.Tensor) for state in states):
        raise TypeError("KV cache must be a tensor or a non-empty sequence of state tensors")

    # The page is registered from the state that starts it; the others must
    # live inside that page (same allocation, same block stride).
    first = min(states, key=lambda state: state.data_ptr())
    storage_ptr = first.untyped_storage().data_ptr()
    num_blocks = first.shape[0]
    page_bytes = first.stride(0) * first.element_size()
    page_start = first.data_ptr()
    for state in states:
        if state.untyped_storage().data_ptr() != storage_ptr:
            raise RuntimeError("recurrent-state tensors must share one CUDA allocation")
        if state.shape[0] != num_blocks:
            raise RuntimeError("recurrent-state tensors must have the same block count")
        if state.stride(0) * state.element_size() != page_bytes:
            raise RuntimeError("recurrent-state tensors must have one common page stride")
        if state.data_ptr() - page_start >= page_bytes:
            raise RuntimeError("recurrent-state tensors must share one page per block")

    return first


class WorkerConnector:
    """Holds worker-only state and behaviors."""

    # Maximum time to wait for an in-flight load to reach terminal state before
    # giving up and reporting it as a load error to vLLM. Load is pure H2D once
    # prefetch has completed, so 120s is generous. Overridable via env var, but
    # values below _LOAD_TIMEOUT_FLOOR_SECONDS are clamped at module import time
    # to prevent production misconfiguration from dropping every in-flight load.
    LOAD_TIMEOUT_SECONDS: int = _LOAD_TIMEOUT_RAW

    def __init__(
        self,
        context: ConnectorContext,
        vllm_config=None,
        kv_cache_config=None,
    ):
        self._ctx = context
        self._client = context.client
        self._kv_cache_config = kv_cache_config
        self._cache_groups = CacheGroupLayout.from_config(kv_cache_config)
        self._layer_to_group = self._cache_groups.layer_to_group()
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        self._use_mla_layer_split_registration = context.is_mla and bool(
            additional_config.get("mla_layer_split_kv_cache", False)
        )

        self._save_queue = queue.Queue()
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="OrbitKVSaveWorker"
        )
        self._save_thread.start()

        self._req_pending_save_tasks: dict[str, int] = {}
        self._completed_saves: set[str] = set()
        self._save_completion_lock = threading.Lock()
        self._save_completion_events: dict[str, threading.Event] = {}
        # Boundary-state jobs finished since the last worker-meta report.
        self._completed_boundary_jobs: list[int] = []
        self._current_metadata: OrbitKVConnectorMetadata | None = None

        self._pending_loads: dict[str, RestoreHandle] = {}
        self._pending_load_reqs: dict[str, set[str]] = {}
        self._pending_load_meta: dict[
            str, tuple[float, int, list[int]]
        ] = {}  # shm_name -> (start_time, num_blocks, block_ids)
        self._load_completion_lock = threading.Lock()

        # Failure surface for vLLM's get_block_ids_with_load_errors / get_finished.
        # Populated when start_load_kv fails synchronously or when an in-flight
        # load times out waiting for the server. Drained once per get_finished
        # and get_block_ids_with_load_errors call.
        self._failed_load_block_ids: set[int] = set()

        self._registered_layers: list[str] = []
        # Page-first storage: all layers of a block in one host page, one slot
        # per tp_rank. Saves distribute by block stripe instead of by layer.
        self._page_first: bool = False
        self._torch_device: torch.device | None = None

        self._cross_layer_mode = False
        self._cross_layer_key = _CROSS_LAYER_KEY

        self._finished_requests: set[str] = set()

        # Stats collection
        self._stats = OrbitKVConnectorStats()
        self._stats_lock = threading.Lock()

    def shutdown(self) -> None:
        self._save_queue.put(None)
        self._save_thread.join()
        self.unregister_context()

    def unregister_context(self) -> None:
        if not self._registered_layers:
            return

        if self._ctx.local_physical_tp_rank == 0:
            ok, message = self._ctx.client.unregister_context(self._ctx.instance_id)
            if not ok:
                logger.warning("[OrbitKVConnector] Unregister context failed: %s", message)

        self._registered_layers.clear()

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        """Register exactly the KV caches vLLM built on this device.

        The engine derives the instance-wide layer-id space once every worker
        has registered, so optional speculative MTP layers, external drafters,
        and hybrid attention layouts need no connector-side layer accounting.
        """
        assert self._ctx.device_id is not None, (
            "CUDA device id is unknown; cannot register KV caches"
        )

        if self._use_mla_layer_split_registration:
            kv_cache_tensors = getattr(self._kv_cache_config, "kv_cache_tensors", None)
            if not kv_cache_tensors:
                raise RuntimeError(
                    "Layer-split KV cache registration requires kv_cache_config.kv_cache_tensors"
                )

            layer_names = [
                layer_name
                for kv_cache_tensor in kv_cache_tensors
                for layer_name in (getattr(kv_cache_tensor, "shared_by", None) or ())
            ]
            if not layer_names:
                raise RuntimeError("Layer-split KV cache registration selected no layers")

            missing_layer_names = [
                layer_name for layer_name in layer_names if layer_name not in kv_caches
            ]
            if missing_layer_names:
                raise RuntimeError(
                    "Layer-split KV cache registration is missing layers: "
                    f"{missing_layer_names[:8]}"
                )

            kv_caches = {layer_name: kv_caches[layer_name] for layer_name in layer_names}
        if not kv_caches:
            raise RuntimeError("No KV cache layers were selected for registration")

        self._page_first = self._use_page_first()
        first_tensor = _registration_tensor(next(iter(kv_caches.values())))
        self._torch_device = first_tensor.device

        if self._cache_groups.group_count > 1:
            unmapped_layers = [name for name in kv_caches if name not in self._layer_to_group]
            if unmapped_layers:
                raise RuntimeError(
                    f"HMA registration contains layers outside cache groups: {unmapped_layers[:8]}"
                )

        layout = "unknown"

        layer_names = []
        buffer_registrations = []
        layer_num_blocks = []
        layer_bytes_per_block = []
        layer_kv_stride_bytes = []
        layer_segments = []
        layer_formats = []
        split_layer_count = 0
        split_blocks_per_logical = 1
        split_logical_blocks = 0
        logged_kinds: set[bool] = set()

        for layer_name, kv_cache in kv_caches.items():
            is_recurrent_state = (
                layer_name in self._cache_groups.recurrent_layer_names
                or isinstance(kv_cache, (tuple, list))
            )
            registration_tensor = _registration_tensor(kv_cache)

            if is_recurrent_state not in logged_kinds:
                logged_kinds.add(is_recurrent_state)
                logger.info(
                    "[OrbitKVConnector] %s layer %s: shape=%s stride=%s dtype=%s storage_offset=%d",
                    "recurrent-state" if is_recurrent_state else "attention",
                    layer_name,
                    tuple(registration_tensor.shape),
                    tuple(registration_tensor.stride()),
                    registration_tensor.dtype,
                    registration_tensor.storage_offset(),
                )

            wrapper_bytes = serialize_gpu_buffer(registration_tensor)

            registration = _infer_kv_cache_registration(
                registration_tensor,
                self._ctx.block_size,
                is_mla=self._ctx.is_mla,
                is_recurrent_state=is_recurrent_state,
            )
            layout = registration.layout

            layer_names.append(layer_name)
            buffer_registrations.append(wrapper_bytes)
            layer_num_blocks.append(registration.num_blocks)
            layer_bytes_per_block.append(registration.bytes_per_block)
            layer_kv_stride_bytes.append(registration.kv_stride_bytes)
            layer_segments.append(registration.segments)
            layer_formats.append(
                "exact"
                if is_recurrent_state
                else {"torch.bfloat16": "bf16", "torch.float16": "fp16"}.get(
                    str(registration_tensor.dtype), "exact"
                )
            )

            if registration.physical_blocks_per_logical_block > 1:
                split_layer_count += 1
                split_blocks_per_logical = registration.physical_blocks_per_logical_block
                split_logical_blocks = registration.num_blocks
                logger.debug(
                    "[OrbitKVConnector] Registered %s with virtual block split: "
                    "logical_block_size=%d physical_blocks_per_logical=%d logical_blocks=%d",
                    layer_name,
                    self._ctx.block_size,
                    registration.physical_blocks_per_logical_block,
                    registration.num_blocks,
                )

        layer_group_ids = [
            self._cache_groups.storage_group_ids[self._layer_to_group.get(name, 0)]
            for name in layer_names
        ]

        ok, message = self._ctx.client.register_context_batch(
            self._ctx.instance_id,
            self._ctx.namespace,
            self._ctx.effective_tp_rank,
            self._ctx.pp_rank,
            self._ctx.effective_tp_size,
            self._ctx.effective_world_size,
            self._ctx.device_id,
            layer_names,
            buffer_registrations,
            layer_num_blocks,
            layer_bytes_per_block,
            layer_kv_stride_bytes,
            layer_segments,
            self._ctx.transfer_backend,
            self._page_first,
            layer_group_ids=layer_group_ids,
            layer_formats=layer_formats,
        )

        if not ok:
            if "OrbitKV version mismatch" in message:
                raise RuntimeError(f"Register context failed: {message}")
            raise RuntimeError(f"Register context batch failed for layers {layer_names}: {message}")

        self._registered_layers = layer_names

        if split_layer_count:
            logger.info(
                "[OrbitKVConnector] Registered %d/%d KV cache layers with virtual "
                "block split: logical_block_size=%d physical_blocks_per_logical=%d "
                "logical_blocks=%d",
                split_layer_count,
                len(kv_caches),
                self._ctx.block_size,
                split_blocks_per_logical,
                split_logical_blocks,
            )

        logger.debug(
            "[OrbitKVConnector] Registered %d KV cache layers (%s layout) instance=%s",
            len(kv_caches),
            layout,
            self._ctx.instance_id,
        )

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend) -> None:
        self._cross_layer_mode = True
        if self._ctx.pp_size > 1:
            self._cross_layer_key = f"{_CROSS_LAYER_KEY}_pp{self._ctx.pp_rank}"
        else:
            self._cross_layer_key = _CROSS_LAYER_KEY
        self.register_kv_caches({self._cross_layer_key: kv_cache})

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] | None = None
        finished_recving: set[str] | None = None

        with self._save_completion_lock:
            self._finished_requests.update(
                finished_req_ids.intersection(self._req_pending_save_tasks)
            )
            done_saves = self._completed_saves & self._finished_requests
            done_saves.update(self._completed_saves & finished_req_ids)

            if done_saves:
                self._completed_saves -= done_saves
                self._finished_requests -= done_saves
                finished_sending = done_saves

        hma_load_failure: str | None = None
        with self._load_completion_lock:
            completed_reqs: set[str] = set()
            completed_restore_keys: list[str] = []
            load_stats_to_record: list[tuple[float, int, bool]] = []
            now = time.perf_counter()

            should_poll_restores = False
            if self._pending_load_reqs:
                try:
                    should_poll_restores = self._client.restore_completions_ready()
                except Exception as error:
                    self._ctx.state_manager.mark_unavailable(
                        f"restore notification check exception: {error}"
                    )
                    raise RuntimeError(
                        "OrbitKV lost restore completion visibility; GPU pages remain held"
                    ) from error
            for restore_key, req_ids in self._pending_load_reqs.items():
                sample_req_id = next(iter(req_ids))
                restore = self._pending_loads.get(sample_req_id)
                if restore is None:
                    continue

                meta = self._pending_load_meta.get(restore_key)
                status = None
                if should_poll_restores:
                    try:
                        status = self._client.poll_restore(restore)
                    except Exception as error:
                        self._ctx.state_manager.mark_unavailable(
                            f"restore completion poll exception: {error}"
                        )
                        raise RuntimeError(
                            "OrbitKV lost restore completion visibility; GPU pages remain held"
                        ) from error
                ready = status is not None and status.done
                timed_out = (
                    not ready and meta is not None and (now - meta[0]) > self.LOAD_TIMEOUT_SECONDS
                )

                if ready:
                    assert status is not None
                    success = status.success
                    if not success:
                        logger.error(
                            "[OrbitKVConnector] async_load_failed: reqs=%s error=%s",
                            req_ids,
                            status.message,
                        )
                        if self._cache_groups.group_count > 1:
                            hma_load_failure = (
                                f"async load failed for requests {sorted(req_ids)}: "
                                f"{status.message}"
                            )
                        elif meta is not None:
                            self._failed_load_block_ids.update(meta[2])
                    else:
                        for req_id in req_ids:
                            trace_transfer("gpu_ready", req_id, engine="vllm", success=True)
                        logger.debug(
                            "[OrbitKVConnector] async_load_completed: reqs=%s",
                            req_ids,
                        )

                    if meta is not None:
                        start_time, num_blocks, _ = meta
                        duration = now - start_time
                        load_stats_to_record.append((duration, num_blocks, success))

                    completed_reqs.update(req_ids)
                    completed_restore_keys.append(restore_key)
                elif timed_out:
                    assert meta is not None
                    self._ctx.state_manager.mark_unavailable("restore completion timeout")
                    # A deadline does not cancel DMA in the Cache Manager.
                    # Fail the engine step without acknowledging or recycling
                    # any destination; instance teardown drains the GPU worker.
                    raise RuntimeError(
                        f"OrbitKV restore timed out for {sorted(req_ids)}; "
                        "GPU pages remain held until transfer teardown"
                    )

            for restore_key in completed_restore_keys:
                restore_req_ids = self._pending_load_reqs.pop(restore_key, set())
                self._pending_load_meta.pop(restore_key, None)
                for req_id in restore_req_ids:
                    self._pending_loads.pop(req_id, None)

            if completed_reqs:
                finished_recving = completed_reqs

        if load_stats_to_record:
            with self._stats_lock:
                for duration, num_blocks, success in load_stats_to_record:
                    self._stats.record_load(duration, num_blocks, success)

        if hma_load_failure is not None:
            self._ctx.state_manager.mark_unavailable(hma_load_failure)
            raise RuntimeError(
                f"OrbitKV HMA load failed; vLLM cannot recover failed "
                f"loads for multiple cache groups: {hma_load_failure}"
            )

        if finished_sending:
            logger.debug(
                "[OrbitKVConnector] async_save_completed: reqs=%s",
                finished_sending,
            )
        if finished_recving:
            logger.debug(
                "[OrbitKVConnector] finished loading KV for requests: %s",
                finished_recving,
            )
        return (finished_sending, finished_recving)

    def start_load_kv(
        self,
        metadata: OrbitKVConnectorMetadata,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        self._current_metadata = metadata

        if not metadata.load_intents:
            return

        total_requests = len(metadata.load_intents)
        load_start = time.perf_counter()

        all_block_ids: list[int] = []
        loads: list[tuple[bytes, list[list[int | None]]]] = []
        request_ids: list[str] = []

        for req_id, load_intent in metadata.load_intents.items():
            if len(load_intent.leases) != self._ctx.tp_shard_count:
                raise RuntimeError(
                    f"load intent has {len(load_intent.leases)} TP shard leases; "
                    f"expected {self._ctx.tp_shard_count}"
                )
            block_ids_by_group = [list(group) for group in load_intent.block_ids_by_group]
            hold = load_intent.recovery_hold
            auxiliary_groups = [
                index for index, group in enumerate(self._cache_groups.storage_group_ids) if group
            ]
            if hold is not None:
                # Only full-attention layers consume the prefix lease.
                for group_index in auxiliary_groups:
                    block_ids_by_group[group_index] = [None] * len(block_ids_by_group[group_index])
            for block_ids in block_ids_by_group:
                all_block_ids.extend(block_id for block_id in block_ids if block_id is not None)
            loads.append((load_intent.leases[self._ctx.tp_shard_index], block_ids_by_group))
            if hold is not None:
                shard = self._ctx.tp_shard_index
                for slot, group_index in enumerate(auxiliary_groups):
                    positions = hold.hit_positions[slot][shard]
                    if hold.last_position not in positions:
                        raise RuntimeError(
                            f"req {req_id}: state group {group_index} shard {shard} "
                            f"lease does not reach query position {hold.last_position}"
                        )
                    destinations = load_intent.block_ids_by_group[group_index]
                    if any(position >= len(destinations) for position in positions):
                        raise RuntimeError(
                            f"req {req_id}: state lease exceeds its GPU destinations"
                        )
                    vectors: list[list[int | None]] = [
                        [None] * len(positions) for _ in range(self._cache_groups.group_count)
                    ]
                    selected = [destinations[position] for position in positions]
                    if any(block_id is None or block_id == 0 for block_id in selected):
                        raise RuntimeError(f"req {req_id}: state lease has no live GPU destination")
                    vectors[group_index] = selected
                    loads.append((hold.leases[slot][shard], vectors))
                    all_block_ids.extend(selected)
            request_ids.append(req_id)

        if not all_block_ids:
            return

        if self._cross_layer_mode:
            layer_groups = [[self._cross_layer_key]]
        else:
            assert self._registered_layers, (
                "KV caches must be registered before submitting load intents"
            )
            layer_groups = [[] for _ in range(self._cache_groups.group_count)]
            for layer_name in self._registered_layers:
                group_index = self._layer_to_group.get(layer_name, 0)
                layer_groups[group_index].append(layer_name)

        if not any(layer_groups):
            return

        try:
            if TRANSFER_TRACING:
                for req_id in request_ids:
                    trace_transfer("restore_submit", req_id, engine="vllm")
            restore = self._client.start_restore(
                self._ctx.instance_id,
                self._ctx.effective_tp_rank,
                self._ctx.device_id,
                layer_groups,
                loads,
            )
            if TRANSFER_TRACING:
                for req_id in request_ids:
                    trace_transfer("restore_link", req_id, engine="vllm", restore_key=restore.key)
        except Exception as error:
            self._ctx.state_manager.mark_unavailable(f"restore submit exception: {error}")
            # A lost acknowledgement can hide an accepted transfer. Releasing
            # its lease or asking vLLM to recompute would race that GPU write.
            raise RuntimeError(
                "OrbitKV restore submission did not establish completion; "
                "GPU pages remain held until transfer teardown"
            ) from error

        num_layers = sum(len(group) for group in layer_groups)
        num_blocks = len(all_block_ids)
        restore_key = restore.key

        schedule_end = time.perf_counter()
        schedule_time_us = (schedule_end - load_start) * 1e6

        with self._load_completion_lock:
            for req_id in request_ids:
                self._pending_loads[req_id] = restore
            self._pending_load_reqs[restore_key] = set(request_ids)
            self._pending_load_meta[restore_key] = (
                load_start,
                num_blocks,
                all_block_ids,
            )

        logger.debug(
            "[OrbitKVConnector] started async load: %d blocks across %d layers for %d reqs, "
            "schedule %.0f us, restore=%s transport=%s",
            num_blocks,
            num_layers,
            total_requests,
            schedule_time_us,
            restore_key,
            self._client.transport,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs whose load failed since the last call, then clear.

        vLLM calls this each forward pass and re-schedules reported blocks for
        local recomputation. Only terminal Cache Manager failures establish
        that DMA has stopped; timeouts and lost acknowledgements are fatal.
        """
        with self._load_completion_lock:
            failed = self._failed_load_block_ids
            self._failed_load_block_ids = set()
        return failed

    def save_kv_layer(
        self,
        metadata: OrbitKVConnectorMetadata,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # Save is metadata-driven and submitted from wait_for_save() outside
        # layer callbacks so CUDA graph replay cannot suppress it.
        pass

    def wait_for_save(self) -> None:
        metadata = self._current_metadata
        self._current_metadata = None
        if metadata is None:
            return
        if not metadata.save_intents and not metadata.boundary_save_intents:
            return
        if metadata.boundary_save_intents and self._cache_groups.group_count <= 1:
            raise RuntimeError("boundary-state save intents are only valid for HMA")

        # This callback runs after the forward launch (including graph replay
        # and boundary-state copies). Fence that producer stream here: recording
        # from the save thread would miss the producer, while a device-wide wait
        # would also block on unrelated work submitted by later steps.
        ready = torch.cuda.Event(blocking=True)
        ready.record(torch.cuda.current_stream(self._torch_device))

        # Both kinds of save read blocks that stay allocated until this worker
        # reports completion: request blocks are held by request_finished /
        # handle_preemptions, boundary-state blocks are pinned by the
        # scheduler until the job id comes back in OrbitKVWorkerMetadata. So
        # every save can run asynchronously behind the forward pass.
        if metadata.save_intents:
            self._save_queue.put(self._make_save_task(metadata.save_intents, ready))
        if metadata.boundary_save_intents:
            self._save_queue.put(
                SaveTask(
                    metadata=OrbitKVConnectorMetadata(
                        save_intents={
                            f"boundary:{job_id}": intent
                            for job_id, intent in metadata.boundary_save_intents.items()
                        }
                    ),
                    request_ids=[],
                    ready=ready,
                    boundary_job_ids=list(metadata.boundary_save_intents),
                )
            )

    def build_connector_worker_meta(self) -> OrbitKVWorkerMetadata | None:
        with self._save_completion_lock:
            if not self._completed_boundary_jobs:
                return None
            completed = self._completed_boundary_jobs
            self._completed_boundary_jobs = []
        return OrbitKVWorkerMetadata(
            completed_boundary_jobs=dict.fromkeys(completed, 1),
        )

    def _make_save_task(
        self, save_intents: dict[str, SaveIntent], ready: torch.cuda.Event
    ) -> SaveTask:
        request_ids = list(save_intents)

        with self._save_completion_lock:
            for req_id in request_ids:
                pending_tasks = self._req_pending_save_tasks.get(req_id, 0)
                if pending_tasks == 0:
                    self._completed_saves.discard(req_id)
                    self._save_completion_events[req_id] = threading.Event()
                self._req_pending_save_tasks[req_id] = pending_tasks + 1

        return SaveTask(
            metadata=OrbitKVConnectorMetadata(save_intents=save_intents),
            request_ids=request_ids,
            ready=ready,
        )

    def _save_worker(self) -> None:
        logger.debug("[OrbitKVConnector] Save worker thread started")

        while True:
            task = self._save_queue.get()
            if task is None:
                self._save_queue.task_done()
                break

            batch: list[SaveTask] = [task]
            while True:
                try:
                    t = self._save_queue.get_nowait()
                    if t is None:
                        self._run_save_batch(batch)
                        for _ in batch:
                            self._save_queue.task_done()
                        self._save_queue.task_done()
                        logger.debug("[OrbitKVConnector] Save worker thread stopped")
                        return
                    batch.append(t)
                except queue.Empty:
                    break

            self._run_save_batch(batch)
            for _ in batch:
                self._save_queue.task_done()

        logger.debug("[OrbitKVConnector] Save worker thread stopped")

    def _run_save_batch(self, batch: list[SaveTask]) -> None:
        # The save thread is the only consumer of the queue and the only
        # path that reports save completion to the scheduler. If it died on
        # an unexpected error every later request would stay held (blocks
        # never freed) and the engine would bleed KV cache until restart.
        try:
            self._process_save_batch(batch)
        except Exception:
            logger.exception(
                "[OrbitKVConnector] save batch failed for %d task(s); blocks released without save",
                len(batch),
            )

    def _process_save_batch(self, batch: list[SaveTask]) -> None:
        all_request_ids = [req_id for task in batch for req_id in task.request_ids]
        all_boundary_job_ids = [job_id for task in batch for job_id in task.boundary_job_ids]
        try:
            self._save_batch(batch)
        finally:
            # Always complete the save lifecycle, even if save failed: the
            # scheduler releases held/pinned blocks on completion, not success.
            self._complete_save_requests(all_request_ids)
            if all_boundary_job_ids:
                with self._save_completion_lock:
                    self._completed_boundary_jobs.extend(all_boundary_job_ids)

    def _save_batch(self, batch: list[SaveTask]) -> None:
        saves_by_layer: dict[str, tuple[list[int], list[bytes]]] = {}

        for task in batch:
            for req_id, save_intent in task.metadata.save_intents.items():
                try:
                    layer_saves = tuple(self._layer_saves(save_intent))
                except Exception:
                    # A malformed intent is a scheduler-side bug; drop this
                    # request's save rather than the whole batch (or thread).
                    logger.exception(
                        "[OrbitKVConnector] req=%s malformed save intent skipped", req_id
                    )
                    continue
                for layer_name, block_ids, block_hashes in layer_saves:
                    if layer_name not in saves_by_layer:
                        saves_by_layer[layer_name] = ([], [])
                    saves_by_layer[layer_name][0].extend(block_ids)
                    saves_by_layer[layer_name][1].extend(block_hashes)

        if not saves_by_layer:
            return

        # Pages remain pinned until the native Publish completes its D2H copy.
        # Each queued task owns the event recorded by its producing step.
        for task in batch:
            task.ready.synchronize()

        saves_list = [(name, ids, hashes) for name, (ids, hashes) in saves_by_layer.items()]
        total_blocks = sum(len(ids) for _, ids, _ in saves_list)

        save_start = time.perf_counter()
        success = False

        try:
            ok, message = self._client.save(
                self._ctx.instance_id,
                self._ctx.effective_tp_rank,
                self._ctx.pp_rank,
                self._ctx.device_id,
                saves_list,
            )

            if not ok:
                logger.error(
                    "[OrbitKVConnector] Save batch failed: %s (continuing without save)",
                    message,
                )
            else:
                success = True
                logger.debug(
                    "[OrbitKVConnector] Batch saved %d layers, %d total blocks",
                    len(saves_list),
                    total_blocks,
                )
        except Exception as e:
            logger.error(
                "[OrbitKVConnector] Save data-plane exception: %s (continuing without save)",
                e,
            )

        save_duration = time.perf_counter() - save_start

        with self._stats_lock:
            self._stats.record_save(save_duration, total_blocks, success)

    def _layer_saves(
        self, save_intent: SaveIntent
    ) -> Iterator[tuple[str, tuple[int, ...], tuple[bytes, ...]]]:
        """Yield `(layer, block_ids, block_hashes)` this rank writes for one intent."""
        if not any(save_intent.block_ids_by_group):
            return

        if self._cross_layer_mode:
            target_layers = (self._cross_layer_key,)
        else:
            assert self._registered_layers, (
                "KV caches must be registered before submitting save intents"
            )
            target_layers = tuple(self._registered_layers)

        for layer_name in target_layers:
            group_index = self._layer_to_group.get(layer_name, 0)
            try:
                block_ids = save_intent.block_ids_by_group[group_index]
            except IndexError as exc:
                raise RuntimeError(
                    f"save intent is missing cache group {group_index} for {layer_name}"
                ) from exc
            block_hashes = save_intent.block_hashes
            if len(block_ids) != len(block_hashes):
                raise RuntimeError(
                    f"save block/hash count mismatch for {layer_name}: "
                    f"blocks={len(block_ids)} hashes={len(block_hashes)}"
                )
            non_null = tuple(
                (block_id, block_hash)
                for block_id, block_hash in zip(block_ids, block_hashes, strict=True)
                if block_id != 0
            )
            block_ids = tuple(block_id for block_id, _ in non_null)
            block_hashes = tuple(block_hash for _, block_hash in non_null)
            if self._page_first and not self._use_mla_layer_split_registration:
                # Full-replica (one shard): every rank holds all layers,
                # so spread the whole-page writes across ranks by block
                # stripe. Layer-split ranks are each the sole writer of
                # their shard and keep the full block set (no striping).
                block_ids, block_hashes = self._block_shard(block_ids, block_hashes)
            if block_ids:
                yield layer_name, block_ids, block_hashes

    def _complete_save_requests(self, request_ids: list[str]) -> None:
        completed_reqs: list[str] = []

        with self._save_completion_lock:
            for req_id in request_ids:
                pending_tasks = self._req_pending_save_tasks.get(req_id)
                if pending_tasks is None:
                    continue
                if pending_tasks > 1:
                    self._req_pending_save_tasks[req_id] = pending_tasks - 1
                    continue

                del self._req_pending_save_tasks[req_id]
                self._completed_saves.add(req_id)
                completed_reqs.append(req_id)
                event = self._save_completion_events.pop(req_id, None)
                if event:
                    event.set()

        self._handle_save_completion(completed_reqs)

    def _handle_save_completion(
        self, request_ids: Iterable[str], reason: str | None = None
    ) -> None:
        req_list = list(request_ids)
        if not req_list:
            return

        suffix = "" if not reason else f" ({reason})"
        for req_id in req_list:
            logger.debug(
                "[OrbitKVConnector] Request %s save completed%s",
                req_id,
                suffix,
            )

    def _use_page_first(self) -> bool:
        """Whether this instance stores blocks page-first.

        Page-first requires one worker to write every layer in its page shard.
        PP workers hold only part of a sealed shard, DCP workers hold different
        token slices, and HMA groups can save different block sets, so none can
        safely use one common block set across every layer in a page.
        """
        return (
            self._ctx.is_mla
            and self._cache_groups.group_count == 1
            and self._ctx.dcp_world_size == 1
            and self._ctx.pp_size == 1
        )

    def _block_shard(
        self,
        block_ids: Iterable[int],
        block_hashes: Iterable[bytes],
    ) -> tuple[list[int], list[bytes]]:
        """`(block_ids, hashes)` this rank saves under page-first: a block stripe.

        A page needs all layers, so page-first distributes save work by block
        rather than by layer: rank r saves physical blocks where
        `block_id % tp_size == r`, writing all their layers. The stripes are
        disjoint and complete, so every block's page is written exactly once.
        With tp_size == 1 this is the whole set.
        """
        tp_size = self._ctx.local_physical_tp_size
        if tp_size <= 1:
            return list(block_ids), list(block_hashes)
        tp_rank = self._ctx.local_physical_tp_rank
        ids: list[int] = []
        hashes: list[bytes] = []
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
            if block_id % tp_size == tp_rank:
                ids.append(block_id)
                hashes.append(block_hash)
        return ids, hashes

    def handle_preemptions(self, preempted_req_ids: set[str] | None) -> None:
        """Wait for preempted requests' saves to complete before blocks are reused.

        Called by vLLM BEFORE preempted blocks are overwritten. This prevents
        data corruption when async saves are still reading from blocks that
        will be reassigned to new requests.
        """
        if not preempted_req_ids:
            return

        events_to_wait: list[tuple[str, threading.Event]] = []
        with self._save_completion_lock:
            for req_id in preempted_req_ids:
                event = self._save_completion_events.get(req_id)
                if event:
                    events_to_wait.append((req_id, event))

        if events_to_wait:
            logger.debug(
                "[OrbitKVConnector] preemption: waiting for %d requests' saves: %s",
                len(events_to_wait),
                [req_id for req_id, _ in events_to_wait],
            )
            for req_id, event in events_to_wait:
                event.wait()
                logger.debug("[OrbitKVConnector] preemption: req=%s save completed", req_id)
        else:
            logger.debug(
                "[OrbitKVConnector] preemption: %d requests (no pending saves)",
                len(preempted_req_ids),
            )

    def get_stats(self) -> OrbitKVConnectorStats | None:
        """Get and reset worker stats for the current interval."""
        with self._stats_lock:
            with self._save_completion_lock:
                self._stats.data["pending_save_requests"] = len(self._req_pending_save_tasks)

            if self._stats.is_empty():
                return None
            return self._stats.clone_and_reset()


__all__ = ["WorkerConnector"]
