"""Experimental P/D Mooncake-push vLLM connectors."""

from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)

from orbitkv.vllm.pd.layout import model_uses_mla
from orbitkv.vllm.pd.metadata import PdConnectorMetadata
from orbitkv.vllm.pd.metrics import PdKVConnectorStats, PdMetricsTracker, PdPromMetrics


class _PdSplitConnector(KVConnectorBase_V1, SupportsHMA):
    """Public vLLM callbacks shared by Prefill and Decode."""

    _is_prefill = False

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        if model_uses_mla(vllm_config):
            parallel_config = getattr(vllm_config, "parallel_config", None)
            dcp_world_size = int(getattr(parallel_config, "decode_context_parallel_size", 1) or 1)
            pcp_world_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
            assert dcp_world_size == 1, (
                "PdConnector MLA first version requires decode_context_parallel_size == 1"
            )
            assert pcp_world_size == 1, (
                "PdConnector MLA first version requires prefill_context_parallel_size == 1"
            )
        self._scheduler: Any | None = None
        self._worker: Any | None = None
        self._metrics = PdMetricsTracker()
        if role == KVConnectorRole.SCHEDULER:
            from .scheduler import (
                PdDecodeSchedulerConnector,
                PdPrefillSchedulerConnector,
            )

            scheduler_cls = (
                PdPrefillSchedulerConnector if self._is_prefill else PdDecodeSchedulerConnector
            )
            self._scheduler = scheduler_cls(vllm_config)
        elif role == KVConnectorRole.WORKER:
            if self._is_prefill:
                from .prefill_worker import PrefillWorker as worker_cls
            else:
                from .decode_worker import DecodeWorker as worker_cls

            self._worker = worker_cls(
                vllm_config,
                kv_cache_config=kv_cache_config,
                metrics=self._metrics,
            )
        else:
            raise ValueError(f"unsupported KV connector role: {role}")

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
        if model_uses_mla(vllm_config):
            return None
        return "HND"

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return True

    def get_kv_connector_stats(self) -> PdKVConnectorStats | None:
        return self._metrics.get_stats()

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> PdKVConnectorStats | None:
        if data is None:
            return None
        return PdKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ) -> PdPromMetrics:
        return PdPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    # -- worker-side common callbacks --------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self._worker is None:
            return None, None
        return self._worker.get_finished(finished_req_ids)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
        if self._scheduler is not None:
            self._scheduler.shutdown()

    # -- scheduler-side common callbacks -----------------------------------

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> PdConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any) -> None:
        if self._scheduler is not None:
            self._scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: Any,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)


class PdDecodeConnector(_PdSplitConnector):
    """Decode-side vLLM connector for P/D Mooncake push."""

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        if self._worker is None:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PdConnectorMetadata)
        self._worker.start_load_kv(metadata, forward_context, **kwargs)

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return False

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker is not None:
            self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self._worker is None:
            return set()
        return self._worker.get_block_ids_with_load_errors()

    def build_connector_worker_meta(self) -> Any | None:
        if self._worker is None:
            return None
        return self._worker.build_connector_worker_meta()


class PdPrefillConnector(_PdSplitConnector):
    """Prefill-side vLLM connector for P/D Mooncake push."""

    _is_prefill = True

    def bind_connector_metadata(self, connector_metadata: PdConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, PdConnectorMetadata)
        if self._worker is not None:
            self._worker.prepare_pushes(connector_metadata)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        # V2 may call this after forward; send ownership must already be bound.
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if self._worker is not None:
            self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        if self._worker is not None:
            self._worker.wait_for_save()


__all__ = ["PdDecodeConnector", "PdPrefillConnector"]
