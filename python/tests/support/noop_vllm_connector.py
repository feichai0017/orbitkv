"""Test-only vLLM connector used to preserve the connector allocation path."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from orbitkv.vllm.config import detect_mla


class NoopKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Keep vLLM's connector/HMA allocation path without external I/O."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._is_mla = detect_mla(vllm_config)
        self._cache_group_count = len(tuple(kv_cache_config.kv_cache_groups))

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return (
            not self._is_mla
            and self._cache_group_count <= 1
            and os.environ.get("ORBITKV_CROSS_LAYER_BLOCKS", "1") == "1"
        )

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        return None

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        return KVConnectorMetadata()

    def request_finished_all_groups(
        self,
        request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None
