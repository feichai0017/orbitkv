"""Worker-side logic for the experimental P/D connectors."""

from __future__ import annotations

from typing import Any

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from orbitkv.logging_utils import get_connector_logger
from orbitkv.vllm.pd.layout import KvCacheLayout, model_uses_mla
from orbitkv.vllm.pd.metadata import (
    BlockIds,
    LayerRemoteLayout,
)
from orbitkv.vllm.pd.metrics import PdMetricsTracker
from orbitkv.vllm.pd.mooncake import MooncakePort, build_mooncake_port

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
            layer_name: KvCacheLayout.from_tensor(
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
