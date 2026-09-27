"""Worker-side logic for the experimental P/D connectors."""

from __future__ import annotations

import time
from typing import Any

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from orbitkv.logging_utils import get_connector_logger
from orbitkv.vllm.pd.config import extra_config_value
from orbitkv.vllm.pd.decode_worker import DecodeHandler
from orbitkv.vllm.pd.layout import KvCacheLayout, layout_from_tensor
from orbitkv.vllm.pd.metadata import (
    RELEASE_CONSUMER_ABORT,
    RELEASE_PRODUCER_PREEMPTED,
    BlockIds,
    LayerRemoteLayout,
    PdConnectorMetadata,
    PdWorkerMetadata,
)
from orbitkv.vllm.pd.metrics import PdKVConnectorStats, PdMetricsTracker
from orbitkv.vllm.pd.mooncake import MooncakePort, build_mooncake_port
from orbitkv.vllm.pd.prefill_worker import PrefillHandler

logger = get_connector_logger()


class PdWorkerBase:
    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any = None,
        transfer: MooncakePort | None = None,
        metrics: PdMetricsTracker | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.metrics = metrics or PdMetricsTracker()
        self.use_mla = model_uses_mla(vllm_config)
        self.logical_block_size = _logical_block_size(vllm_config)
        self._layer_specs = _layer_specs_from_config(kv_cache_config)
        self._layer_group_indices = _layer_group_indices_from_config(kv_cache_config)
        self.transfer = transfer
        self._transfer_is_injected = transfer is not None
        self.engine_id = getattr(vllm_config.kv_transfer_config, "engine_id", None) or ""
        self.tp_rank, self.tp_size = _tensor_parallel_identity(vllm_config)
        logger.info(
            "[PdConnector] worker initialized engine=%s tp_rank=%d tp_size=%d",
            self.engine_id,
            self.tp_rank,
            self.tp_size,
        )
        self.layouts: dict[str, KvCacheLayout] = {}
        self.layer_names: list[str] = []
        self._registered_layers: dict[str, LayerRemoteLayout] = {}
        self.device_id: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.device_id = _infer_cuda_device(kv_caches)
        expected_num_blocks = _expected_num_blocks(self.kv_cache_config)
        self.layouts = {
            layer_name: layout_from_tensor(
                layer_name,
                tensor,
                layer_spec=self._layer_spec(layer_name),
                logical_block_size=self.logical_block_size,
                expected_num_blocks=expected_num_blocks,
            )
            for layer_name, tensor in kv_caches.items()
        }
        num_blocks_by_layer = {name: layout.num_blocks for name, layout in self.layouts.items()}
        assert len(set(num_blocks_by_layer.values())) == 1, (
            "PdConnector requires all KV cache tensors to share num_blocks; "
            f"num_blocks_by_layer={num_blocks_by_layer}"
        )
        self.layer_names = list(kv_caches.keys())
        if not self._transfer_is_injected:
            self.transfer = build_mooncake_port(
                self.vllm_config,
                self.device_id,
                tp_rank=self.tp_rank,
            )
        assert self.transfer is not None
        registered_layers = self.transfer.register_local_layers(
            tuple(
                self.layouts[layer_name].remote_layout(layer_idx)
                for layer_idx, layer_name in enumerate(self.layer_names)
            )
        )
        self._registered_layers = {layer.layer_name: layer for layer in registered_layers}
        logger.info(
            "[PdConnector] registered %d KV cache layers",
            len(self.layouts),
        )

    def _layer_spec(self, layer_name: str) -> Any | None:
        layer_spec = self._layer_specs.get(layer_name)
        assert layer_spec is not None or not self.use_mla, (
            f"PdConnector MLA requires KVCacheSpec for layer={layer_name}; "
            "pass kv_cache_config into the connector"
        )
        return layer_spec

    def get_stats(self) -> PdKVConnectorStats:
        return self.metrics.get_stats()

    def _layer_idx(self, layer_name: str) -> int:
        try:
            return self.layer_names.index(layer_name)
        except ValueError as exc:
            raise AssertionError(f"unknown layer {layer_name}") from exc

    def group_idx_for_layer(self, layer_name: str) -> int:
        if not self._layer_group_indices:
            return 0
        try:
            return self._layer_group_indices[layer_name]
        except KeyError as exc:
            raise AssertionError(f"unknown KV cache group for layer {layer_name}") from exc

    def block_ids_for_layer(self, block_ids: BlockIds, layer_name: str) -> set[int]:
        group_idx = self.group_idx_for_layer(layer_name)
        if group_idx >= len(block_ids):
            return set()
        return set(block_ids[group_idx])


class PdDecodeWorkerConnector(PdWorkerBase):
    """Decode-side owner of receive, page-grant and completion state."""

    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any = None,
        transfer: MooncakePort | None = None,
        prefill_sender: Any | None = None,
        metrics: PdMetricsTracker | None = None,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, transfer, metrics)
        self._failed_load_block_ids: set[int] = set()
        observation_socket = extra_config_value(
            vllm_config,
            "orbitkv.pd.completion_observation_socket",
        )
        self._completion_instance_id = str(
            extra_config_value(
                vllm_config,
                "orbitkv.pd.completion_observation_instance_id",
                "",
            )
            or ""
        )
        self._completion_client: Any | None = None
        if observation_socket is not None:
            if not isinstance(observation_socket, str) or not observation_socket:
                raise ValueError(
                    "orbitkv.pd.completion_observation_socket must be a non-empty path"
                )
            if not self._completion_instance_id:
                raise ValueError(
                    "orbitkv.pd.completion_observation_instance_id is required when "
                    "completion observation is enabled"
                )
            from orbitkv import CacheManagerClient

            self._completion_client = CacheManagerClient(observation_socket)
        self._decode = DecodeHandler(
            self,
            prefill_sender=prefill_sender,
            completion_callback=(
                self._observe_handoff_completion if self._completion_client is not None else None
            ),
        )

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        super().register_kv_caches(kv_caches)
        if not self._transfer_is_injected:
            self._decode.init_transfer_waiter()
        self._decode.gather_peer_info()
        logger.info(
            "[PdConnector] decode gathered %d peer ranks",
            len(self._decode._peers.layouts),
        )

    def start_load_kv(
        self,
        metadata: PdConnectorMetadata,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug(
            "[PdConnector] decode start_load_kv metadata=%s wait_reqs=%s release=%s known_wait=%s",
            metadata,
            sorted(metadata.reqs_to_wait),
            sorted(metadata.reqs_to_release),
            sorted(self._decode.wait_reqs),
        )
        if not metadata.reqs_to_wait and not metadata.reqs_to_release and self._decode.is_idle():
            return
        assert self.transfer is not None, "PdConnector Mooncake port is not initialized"
        self._decode.process_wait_reqs(metadata.reqs_to_wait)
        for req_id in metadata.reqs_to_release:
            reason = metadata.release_reasons.get(req_id, RELEASE_CONSUMER_ABORT)
            logger.debug("[PdConnector] decode release req=%s reason=%s", req_id, reason)
            if reason == RELEASE_CONSUMER_ABORT:
                self._decode.release(req_id)
            else:
                self.transfer.close_request(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert layer_name in self.layouts, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )

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

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if not finished_req_ids and self._decode.is_idle():
            return None, None
        assert self.transfer is not None, "PdConnector Mooncake port is not initialized"
        finished_recving = self.transfer.pop_finished_recving()
        finished_recving.update(self._decode.pop_finished_transfer_waits())
        finished_recving.update(self._decode.pop_finished_aborted_recving())
        finished_recving.update(self._decode.pop_failed_recving())
        if finished_recving:
            logger.info(
                "[PdConnector] D worker finished_recving reqs=%s count=%d "
                "remaining_wait_before=%d ts_ns=%d",
                sorted(finished_recving),
                len(finished_recving),
                len(self._decode.wait_reqs),
                time.time_ns(),
            )
        self._decode.finish_recving(finished_recving)
        return None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        self._failed_load_block_ids.update(self._decode.pop_failed_block_ids())
        failed = self._failed_load_block_ids
        self._failed_load_block_ids = set()
        return failed

    def build_connector_worker_meta(self) -> PdWorkerMetadata | None:
        failed_recving = self._decode.pop_failed_recving_for_meta()
        if not failed_recving:
            return None
        return PdWorkerMetadata(failed_recving=failed_recving)

    def shutdown(self) -> None:
        self._decode.shutdown()
        if self._completion_client is not None:
            self._completion_client.close()
            self._completion_client = None

    def _observe_handoff_completion(self, evidence: Any, outcome: str) -> None:
        client = self._completion_client
        if client is None:
            return
        if self.device_id is None:
            logger.warning("[PdConnector] cannot report completion before KV cache registration")
            return
        elapsed_ns = max(1, time.time_ns() - evidence.queued_ts_ns)
        try:
            client.observe_prefill_to_decode_completion(
                self._completion_instance_id,
                self.device_id,
                evidence.source_endpoint,
                evidence.notification_generation,
                evidence.logical_bytes,
                evidence.wire_bytes if outcome == "completed" else 0,
                evidence.fragment_count,
                elapsed_ns,
                admitted=True,
                outcome=outcome,
                representation="raw",
            )
        except Exception:
            logger.exception(
                "[PdConnector] disabling completion observations after Cache Manager failure"
            )
            client.close()
            self._completion_client = None
            self._decode.disable_completion_observations()


class PdPrefillWorkerConnector(PdWorkerBase):
    """Prefill-side owner of push, completion and source-page state."""

    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any = None,
        transfer: MooncakePort | None = None,
        metrics: PdMetricsTracker | None = None,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, transfer, metrics)
        self._prefill = PrefillHandler(self)

    def start_load_kv(
        self,
        metadata: PdConnectorMetadata,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug(
            "[PdConnector] prefill start_load_kv metadata=%s push_reqs=%s release=%s known_push=%s",
            metadata,
            sorted(metadata.reqs_to_push),
            sorted(metadata.reqs_to_release),
            sorted(self._prefill.push_reqs),
        )
        if (
            not metadata.reqs_to_push
            and not metadata.reqs_to_release
            and not metadata.preempted_req_ids
            and not self._prefill.has_state()
        ):
            return
        assert self.transfer is not None, "PdConnector Mooncake port is not initialized"
        self._prefill.process_push_reqs(metadata.reqs_to_push)
        for req_id in metadata.preempted_req_ids:
            logger.debug("[PdConnector] prefill preempt req=%s", req_id)
            for push_req_id in self._prefill.release(req_id, RELEASE_PRODUCER_PREEMPTED):
                self.transfer.close_request(push_req_id)
        for req_id in metadata.reqs_to_release:
            reason = metadata.release_reasons.get(req_id, RELEASE_CONSUMER_ABORT)
            logger.debug("[PdConnector] prefill release req=%s reason=%s", req_id, reason)
            released_push_req_ids = self._prefill.release(req_id, reason)
            if released_push_req_ids:
                for push_req_id in released_push_req_ids:
                    self.transfer.close_request(push_req_id)
            elif reason != RELEASE_CONSUMER_ABORT:
                self.transfer.close_request(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        self._prefill.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self._prefill.wait_for_save()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def build_connector_worker_meta(self) -> PdWorkerMetadata | None:
        return None

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if not finished_req_ids and not self._prefill.has_state():
            return None, None
        releasable_sending = self._prefill.get_finished_sending(finished_req_ids)
        return releasable_sending or None, None

    def shutdown(self) -> None:
        self._prefill.shutdown()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _infer_cuda_device(kv_caches: dict[str, Any]) -> int | None:
    for tensor in kv_caches.values():
        device = getattr(tensor, "device", None)
        index = getattr(device, "index", None)
        if index is not None:
            return int(index)
    return None


def _tensor_parallel_identity(vllm_config: Any) -> tuple[int, int]:
    try:
        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        return (
            int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0),
            int(getattr(parallel_config, "tensor_parallel_size", 1) or 1),
        )


def model_uses_mla(vllm_config: Any) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if bool(getattr(model_config, "use_mla", False)):
        return True
    hf_config = getattr(model_config, "hf_text_config", None)
    return getattr(hf_config, "kv_lora_rank", None) is not None


def _logical_block_size(vllm_config: Any) -> int:
    cache_config = getattr(vllm_config, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 0) or 0)
    if block_size > 0:
        return block_size
    return 16


def _layer_specs_from_config(kv_cache_config: Any) -> dict[str, Any]:
    if kv_cache_config is None:
        return {}
    return {
        layer_name: group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
    }


def _layer_group_indices_from_config(kv_cache_config: Any) -> dict[str, int]:
    if kv_cache_config is None:
        return {}
    return {
        layer_name: group_idx
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }


def _expected_num_blocks(kv_cache_config: Any) -> int | None:
    if kv_cache_config is None:
        return None
    num_blocks = getattr(kv_cache_config, "num_blocks", None)
    return int(num_blocks) if num_blocks is not None else None
