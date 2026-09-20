"""Validate SGLang GPU pools and map token indices to registered pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType


@dataclass(frozen=True)
class GpuLayout:
    page_size: int
    pool_group: DevicePoolGroup
    pool: DevicePoolEntry
    row_span: int
    layer_names: list[str]
    num_blocks: list[int]
    block_bytes: list[int]

    @classmethod
    def from_pool(cls, params: Any, components: set[ComponentType]) -> GpuLayout:
        page_size = params.page_size
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
                page_size=page_size,
                rows_are_pages=bool(getattr(kvcache, "use_hnd", False)),
            )
            pool_group = DevicePoolGroup([entry], kvcache.layer_num, page_size)
        else:
            pool_group = resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=page_size,
                params=params,
                components=components,
            )
        if set(pool_group.entry_map) != {PoolName.KV}:
            raise ValueError(
                "OrbitKV direct GPU linker requires one KV pool; auxiliary GPU pools are unsupported"
            )
        pool = pool_group.entry_map[PoolName.KV]
        row_span = pool._row_span
        if row_span not in (1, page_size):
            raise ValueError(f"unsupported SGLang GPU page row span: {row_span}")

        layer_names = [f"kv:{i}" for i in range(len(pool.kv_buffer))]
        if not layer_names:
            raise ValueError("SGLang KV pool has no GPU buffers")
        num_blocks = []
        block_bytes = []
        for tensor in pool.kv_buffer:
            if tensor.device.type != "cuda" or not tensor.is_contiguous():
                raise ValueError("OrbitKV direct GPU linker requires contiguous CUDA KV buffers")
            if tensor.shape[0] % row_span:
                raise ValueError("SGLang GPU KV buffer is not page aligned")
            num_blocks.append(tensor.shape[0] // row_span)
            block_bytes.append(tensor.stride(0) * tensor.element_size() * row_span)
        if len(set(num_blocks)) != 1:
            raise ValueError("SGLang GPU KV buffers have different page counts")

        return cls(page_size, pool_group, pool, row_span, layer_names, num_blocks, block_bytes)

    def block_ids(self, indices: torch.Tensor, expected_pages: int) -> list[int]:
        if indices.numel() != expected_pages * self.page_size:
            raise ValueError("SGLang GPU indices do not cover exactly the requested pages")
        rows = self.pool.prepare_locations(indices)
        block_ids = [row // self.row_span for row in rows]
        if any(block_id >= self.num_blocks[0] for block_id in block_ids):
            raise ValueError("SGLang GPU page index is outside its registered buffers")
        return block_ids
