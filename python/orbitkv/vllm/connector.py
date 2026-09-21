"""
Facade for the OrbitKV vLLM connector, split into scheduler/worker implementations.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_rank

from orbitkv.client.connection import connect_cache
from orbitkv.client.gpu import resolve_device_id as _resolve_device_id
from orbitkv.logging_utils import get_connector_logger
from orbitkv.vllm.config import (
    ConnectorContext,
    OrbitKVConnectorMode,
    TpShardTopology,
    derive_namespace,
    detect_mla,
    resolve_instance_id,
    resolve_transfer_backend,
)
from orbitkv.vllm.layout import CacheGroupLayout
from orbitkv.vllm.metadata import OrbitKVConnectorMetadata
from orbitkv.vllm.metrics import OrbitKVConnectorStats, OrbitKVPromMetrics
from orbitkv.vllm.scheduler import SchedulerConnector
from orbitkv.vllm.state_manager import ServiceStateManager
from orbitkv.vllm.worker import WorkerConnector

logger = get_connector_logger()


class OrbitKVConnector(KVConnectorBase_V1, SupportsHMA):
    """v1 KV connector for OrbitKV with separated scheduler/worker logic."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)

        instance_id = resolve_instance_id(vllm_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        world_size = vllm_config.parallel_config.world_size
        is_mla = detect_mla(vllm_config)
        cache_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if cache_groups and CacheGroupLayout.from_config(kv_cache_config).has_recurrent_state:
            from vllm.v1.core.sched.output import SchedulerOutput

            if "kv_connector_block_state" not in getattr(
                SchedulerOutput, "__dataclass_fields__", {}
            ):
                raise RuntimeError(
                    "OrbitKV HMA requires vLLM SchedulerOutput.kv_connector_block_state "
                    "to hand off pinned Mamba boundary states; this vLLM build does not "
                    "provide it. Use a build with the boundary-state hand-off before "
                    "enabling OrbitKV for hybrid models."
                )
        collapse_mla_tp = is_mla and len(cache_groups) <= 1
        dcp_world_size = (
            getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1
        )
        pcp_world_size = (
            getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1) or 1
        )
        effective_tp_size = max(1, dcp_world_size) if collapse_mla_tp else tp_size

        if dcp_world_size > 1 and not is_mla:
            logger.warning(
                "[OrbitKVConnector] DCP with non-MLA model detected "
                "(dcp_world_size=%d). effective_tp_rank will use tp_rank only; "
                "KV data may collide across DCP ranks if workers share "
                "the same tp_rank.",
                dcp_world_size,
            )

        block_size = vllm_config.cache_config.block_size
        hash_block_size: int | None = None
        if kv_cache_config is not None:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
            if (
                scheduler_block_size != block_size * dcp_world_size
                or scheduler_block_size % hash_block_size != 0
            ):
                raise ValueError(
                    f"vLLM scheduler block {scheduler_block_size} / hash block "
                    f"{hash_block_size} do not fit the connector block {block_size}"
                )

        cross_layer_blocks = os.environ.get("ORBITKV_CROSS_LAYER_BLOCKS", "1") == "1"
        base_namespace = derive_namespace(
            vllm_config,
            effective_tp_size,
            dcp_world_size,
            pcp_world_size,
            cross_layer_blocks=cross_layer_blocks,
            hash_block_size=hash_block_size,
        )

        tp_rank: int | None = None
        device_id: int | None = None
        dcp_rank: int = 0
        pp_rank: int = 0
        pp_size: int = getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) or 1
        if role == KVConnectorRole.WORKER:
            tp_rank = get_tensor_model_parallel_rank()
            pp_group = get_pp_group()
            pp_rank = pp_group.rank_in_group
            if dcp_world_size > 1:
                from vllm.distributed.parallel_state import (
                    get_decode_context_model_parallel_rank,
                )

                dcp_rank = get_decode_context_model_parallel_rank()
            if torch.cuda.is_available():
                device_id = _resolve_device_id()

        assert vllm_config.kv_transfer_config is not None
        server_host = os.environ.get(
            "ORBITKV_HOST"
        ) or vllm_config.kv_transfer_config.get_from_extra_config(
            "orbitkv.host", "http://127.0.0.1"
        )
        server_port = os.environ.get(
            "ORBITKV_PORT"
        ) or vllm_config.kv_transfer_config.get_from_extra_config("orbitkv.port", 50055)
        mode = OrbitKVConnectorMode.from_config(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "orbitkv.mode", OrbitKVConnectorMode.READ_WRITE.value
            )
        )
        transfer_backend = resolve_transfer_backend(
            is_mla,
            vllm_config.kv_transfer_config.get_from_extra_config("orbitkv.transfer_backend", None),
        )
        wait_for_full_prefix = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "orbitkv.wait_for_full_prefix", False
            )
        )
        default_endpoint = f"{server_host}:{server_port}"
        tp_shards = TpShardTopology.from_config(
            default_endpoint=default_endpoint,
            configured_endpoints=vllm_config.kv_transfer_config.get_from_extra_config(
                "orbitkv.tp_shard_endpoints", None
            ),
            global_tp_size=tp_size,
            global_world_size=world_size,
        )
        if tp_shards.shard_count > 1 and (
            world_size != tp_size or dcp_world_size != 1 or pcp_world_size != 1
        ):
            raise ValueError(
                "orbitkv.tp_shard_endpoints currently supports TP-only parallelism; "
                f"got tp_size={tp_size}, world_size={world_size}, "
                f"dcp_world_size={dcp_world_size}, pcp_world_size={pcp_world_size}"
            )
        shard_index = tp_shards.shard_index(tp_rank) if tp_rank is not None else 0
        namespace = tp_shards.namespace(base_namespace, shard_index)
        self._engine_endpoint = tp_shards.endpoints[shard_index]
        self._connections = connect_cache(
            endpoints=tp_shards.endpoints,
            shard_index=shard_index,
            all_shards=role == KVConnectorRole.SCHEDULER,
            get_option=vllm_config.kv_transfer_config.get_from_extra_config,
        )
        clients = self._connections.clients
        client = clients[self._connections.selected_index]
        logger.debug("[OrbitKVConnector] Connected to Cache Manager at %s", client.bootstrap_socket)
        logger.info(
            "[OrbitKVConnector] Cache Manager channel: transport=%s target=%s",
            client.transport,
            client.bootstrap_socket,
        )

        self._state_manager = ServiceStateManager(client)

        self._ctx = ConnectorContext(
            instance_id=instance_id,
            namespace=namespace,
            block_size=block_size,
            tp_size=tp_size,
            world_size=world_size,
            tp_rank=tp_rank,
            device_id=device_id,
            client=client,
            state_manager=self._state_manager,
            is_mla=is_mla,
            collapse_mla_tp=collapse_mla_tp,
            transfer_backend=transfer_backend,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            dcp_rank=dcp_rank,
            pp_rank=pp_rank,
            pp_size=pp_size,
            mode=mode,
            wait_for_full_prefix=wait_for_full_prefix,
            tp_shards=tp_shards,
            hash_block_size=hash_block_size,
        )

        # MLA attention backends expose no num-layers stride dimension, so vLLM
        # cannot build a cross-layer (uniform) KV cache for MLA. Requesting it
        # is silently ignored upstream and falls back to per-layer — surface
        # that instead of pretending the request was honored.
        env_cross_layer = os.environ.get("ORBITKV_CROSS_LAYER_BLOCKS", "1") == "1"
        self._prefer_cross_layer = env_cross_layer and not is_mla and len(cache_groups) <= 1
        if is_mla and env_cross_layer:
            logger.warning(
                "[OrbitKVConnector] ORBITKV_CROSS_LAYER_BLOCKS=1 is ignored for MLA "
                "models: cross-layer KV cache is unsupported by MLA attention backends; "
                "using per-layer registration."
            )

        self._scheduler: SchedulerConnector | None = None
        self._worker: WorkerConnector | None = None
        if role == KVConnectorRole.SCHEDULER:
            pd_tail_save = bool(
                vllm_config.kv_transfer_config.get_from_extra_config("orbitkv.pd_tail_save", False)
            )
            pd_tail_load = bool(
                vllm_config.kv_transfer_config.get_from_extra_config("orbitkv.pd_tail_load", False)
            )
            self._scheduler = SchedulerConnector(
                self._ctx,
                clients=clients,
                pd_tail_save=pd_tail_save,
                pd_tail_load=pd_tail_load,
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Open the liveness stream from the scheduler process only. One
            # stream per vllm replica is enough — if any tp worker crashes,
            # the scheduler dies too, closing this stream and triggering
            # server-side cleanup of the instance's CUDA IPC mappings.
            for index, client in enumerate(clients):
                client.start_session_watcher(
                    instance_id,
                    tp_shards.namespace(base_namespace, index),
                    self._ctx.effective_tp_size,
                    self._ctx.effective_world_size,
                )
        else:
            self._worker = WorkerConnector(
                self._ctx,
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )

        logger.debug(
            "[OrbitKVConnector] Initialized role=%s instance_id=%s device=%s "
            "tp_rank=%s tp_size=%d pp_rank=%d pp_size=%d world_size=%d namespace=%s "
            "is_mla=%s collapse_mla_tp=%s transfer_backend=%s dcp_world_size=%d "
            "pcp_world_size=%d dcp_rank=%d tp_shard=%d/%d "
            "mode=%s wait_for_full_prefix=%s data_transport=%s",
            role.name,
            instance_id,
            device_id if device_id is not None else "cpu",
            tp_rank if tp_rank is not None else "N/A",
            tp_size,
            pp_rank,
            pp_size,
            world_size,
            namespace,
            is_mla,
            collapse_mla_tp,
            transfer_backend,
            dcp_world_size,
            pcp_world_size,
            dcp_rank,
            shard_index,
            tp_shards.shard_count,
            mode.value,
            wait_for_full_prefix,
            client.transport,
        )

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        if not self._worker:
            return
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        self._worker.start_load_kv(metadata, forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._worker:
            return
        self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        # Save is submitted from wait_for_save() using scheduler metadata.
        # Layer callbacks are intentionally ignored so CUDA graph replay
        # cannot suppress save submission.
        pass

    def wait_for_save(self) -> None:
        if not self._worker:
            return
        self._worker.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if not self._worker:
            return (None, None)
        return self._worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self):
        if not self._worker:
            return None
        return self._worker.build_connector_worker_meta()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if not self._worker:
            return
        self._worker.register_kv_caches(kv_caches)

    def unregister_context(self) -> None:
        if self._worker:
            self._worker.unregister_context()

    def handle_preemptions(self, preempted) -> None:
        if not self._worker:
            return
        # Compat: old vLLM passes set[str], new vLLM passes KVConnectorMetadata
        if isinstance(preempted, set):
            preempted_req_ids = preempted
        else:
            preempted_req_ids = getattr(preempted, "preempted_req_ids", None) or set()
        self._worker.handle_preemptions(preempted_req_ids)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def on_new_request(self, request) -> None:
        if self._scheduler:
            self._scheduler.on_new_request(request)

    def update_connector_output(self, connector_output) -> None:
        if self._scheduler:
            self._scheduler.update_connector_output(connector_output)

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        if self._scheduler:
            self._scheduler.bind_gpu_block_pool(gpu_block_pool)

    def request_finished(
        self,
        request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._scheduler:
            return self._scheduler.request_finished(request, (block_ids,))
        return (False, None)

    def request_finished_all_groups(
        self,
        request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._scheduler:
            return self._scheduler.request_finished(request, block_ids)
        return (False, None)

    def take_events(self) -> Iterable:
        return ()

    def has_pending_push_work(self) -> bool:
        if not self._scheduler:
            return False
        return self._scheduler.has_pending_push_work()

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._scheduler:
            return (0, False)
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ) -> None:
        if self._scheduler:
            self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output) -> OrbitKVConnectorMetadata:
        if not self._scheduler:
            return OrbitKVConnectorMetadata()
        return self._scheduler.build_connector_meta(scheduler_output)

    # ==============================
    # Defaults and shutdown
    # ==============================
    def get_block_ids_with_load_errors(self) -> set[int]:
        if not self._worker:
            return set()
        return self._worker.get_block_ids_with_load_errors()

    def get_kv_connector_stats(self) -> OrbitKVConnectorStats | None:
        stats: OrbitKVConnectorStats | None = None

        # Collect scheduler-side stats
        if self._scheduler:
            stats = self._scheduler.get_stats()

        # Collect worker-side stats
        if self._worker:
            worker_stats = self._worker.get_stats()
            if worker_stats is not None:
                stats = worker_stats if stats is None else stats.aggregate(worker_stats)

        return stats

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> OrbitKVConnectorStats | None:
        if data is None:
            return None
        return OrbitKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ) -> OrbitKVPromMetrics:
        return OrbitKVPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    def get_handshake_metadata(self):
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return self._prefer_cross_layer

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        if not self._worker:
            return
        self._worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def set_host_xfer_buffer_ops(self, copy_operation):
        return

    def get_finished_count(self) -> int | None:
        return None

    def shutdown(self):
        try:
            if self._scheduler:
                self._scheduler.shutdown()
            if self._worker:
                self._worker.shutdown()
        finally:
            if self._state_manager:
                self._state_manager.shutdown()
            self._connections.close()


class NoopKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Connector-path baseline for tests."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._is_mla = detect_mla(vllm_config)
        self._cache_group_count = len(tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ()))

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # MLA cannot use cross-layer KV (no num-layers stride); be honest.
        return (
            not self._is_mla
            and self._cache_group_count <= 1
            and os.environ.get("ORBITKV_CROSS_LAYER_BLOCKS", "1") == "1"
        )

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        return

    def build_connector_meta(self, scheduler_output) -> OrbitKVConnectorMetadata:
        return OrbitKVConnectorMetadata()

    def request_finished_all_groups(
        self,
        request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return (False, None)


__all__ = ["OrbitKVConnector", "NoopKVConnector", "KVConnectorRole"]
