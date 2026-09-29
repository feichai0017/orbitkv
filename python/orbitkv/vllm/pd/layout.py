"""Validated vLLM 0.29 cache views and P/D byte ranges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbitkv.vllm.pd.metadata import LayerRemoteLayout, TransferRegionLayout

BlockIdSelection = set[int] | tuple[int, ...] | None


@dataclass(frozen=True)
class BlockRegionSlice:
    block_id: int
    src_offset_bytes: int
    bytes: int

    def __post_init__(self) -> None:
        assert self.block_id >= 0
        assert self.src_offset_bytes >= 0
        assert self.bytes > 0


@dataclass(frozen=True)
class LayerBlockSlices:
    regions: tuple[BlockRegionSlice, ...]

    def __post_init__(self) -> None:
        assert self.regions


@dataclass(frozen=True)
class KvCacheLayout:
    """One raw [blocks, heads, states, content] view, with dense HNC pages.

    Content includes both K and V for attention. MLA/indexer/state pages use
    the same raw view contract. Page padding and interleaved layers are gaps,
    not payload; only the validated content bytes are transferred.
    """

    layer_name: str
    shape: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]
    element_size: int
    base_addr: int
    block_size: int

    @classmethod
    def from_tensor(
        cls,
        layer_name: str,
        tensor: Any,
        *,
        logical_block_size: int,
        layer_spec: Any | None = None,
        expected_num_blocks: int | None = None,
    ) -> KvCacheLayout:
        shape = tuple(int(dim) for dim in tensor.shape)
        strides = tuple(int(stride) for stride in tensor.stride())
        detail = f"layer={layer_name} shape={shape} strides={strides}"
        assert len(shape) == len(strides) == 4, f"PdConnector requires 4D BHNC cache; {detail}"
        assert logical_block_size > 0 and all(dim > 0 for dim in shape), detail
        assert all(stride > 0 for stride in strides), detail
        element_size = int(tensor.element_size())
        assert element_size > 0, detail
        blocks, heads, states, content = shape
        for dim, stride, dense_stride in zip(
            shape[1:], strides[1:], (states * content, content, 1), strict=True
        ):
            assert dim == 1 or stride == dense_stride, (
                f"PdConnector requires dense HNC pages; {detail}"
            )
        block_bytes = heads * states * content * element_size
        block_stride = strides[0] * element_size
        assert block_stride >= block_bytes, f"PdConnector overlapping pages; {detail}"
        if expected_num_blocks is not None:
            assert blocks == expected_num_blocks, (
                "PdConnector does not support physical/logical block split or unequal "
                f"num_blocks; expected={expected_num_blocks} {detail}"
            )
        if layer_spec is not None:
            specs = getattr(layer_spec, "kv_cache_specs", None)
            if isinstance(specs, dict):
                assert layer_name in specs, f"Missing layer KVCacheSpec: {layer_name}"
                layer_spec = specs[layer_name]
            assert int(layer_spec.block_size) == logical_block_size, (
                f"PdConnector does not support physical/logical block split; {detail}"
            )
            expected_tail_bytes = (
                int(layer_spec.num_heads),
                int(layer_spec.num_states),
                int(layer_spec.state_content_size_bytes),
            )
            assert (heads, states, content * element_size) == expected_tail_bytes, (
                f"PdConnector view does not match KVCacheSpec; "
                f"expected HNC bytes={expected_tail_bytes} {detail}"
            )
            page_size = int(layer_spec.page_size_bytes)
            assert block_bytes <= page_size <= block_stride, (
                f"PdConnector KVCacheSpec page size={page_size} exceeds page stride; {detail}"
            )
        base_addr = int(tensor.data_ptr())
        storage = tensor.untyped_storage()
        storage_base = int(storage.data_ptr())
        storage_end = storage_base + int(storage.nbytes())
        span = (blocks - 1) * block_stride + block_bytes
        assert 0 < storage_base <= base_addr and base_addr + span <= storage_end <= 2**64, (
            f"PdConnector view exceeds backing storage; {detail}"
        )
        return cls(
            layer_name=layer_name,
            shape=shape,
            strides=strides,
            element_size=element_size,
            base_addr=base_addr,
            block_size=logical_block_size,
        )

    @property
    def num_blocks(self) -> int:
        return self.shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self.shape[1]

    @property
    def head_bytes(self) -> int:
        return self.shape[2] * self.shape[3] * self.element_size

    @property
    def block_bytes(self) -> int:
        return self.num_kv_heads * self.head_bytes

    @property
    def block_stride(self) -> int:
        return self.strides[0] * self.element_size

    def block_slices(
        self, block_id: int, start_head: int = 0, end_head: int | None = None
    ) -> LayerBlockSlices:
        assert 0 <= block_id < self.num_blocks, (
            f"block_id {block_id} out of range for layer={self.layer_name} "
            f"num_blocks={self.num_blocks}"
        )
        if end_head is None:
            end_head = self.num_kv_heads
        assert 0 <= start_head < end_head <= self.num_kv_heads, (
            f"invalid KV head range [{start_head}, {end_head}) for layer={self.layer_name}"
        )
        return LayerBlockSlices(
            regions=(
                BlockRegionSlice(
                    block_id=block_id,
                    src_offset_bytes=block_id * self.block_stride + start_head * self.head_bytes,
                    bytes=(end_head - start_head) * self.head_bytes,
                ),
            ),
        )

    def remote_layout(
        self, layer_idx: int, block_ids: BlockIdSelection = None
    ) -> LayerRemoteLayout:
        if block_ids is None:
            ordered = tuple(range(self.num_blocks))
        elif isinstance(block_ids, tuple):
            ordered = block_ids
        else:
            ordered = tuple(sorted(block_ids))
        assert len(set(ordered)) == len(ordered) and all(
            0 <= block_id < self.num_blocks for block_id in ordered
        ), f"invalid remote block selection for layer={self.layer_name}"
        return LayerRemoteLayout(
            layer_name=self.layer_name,
            layer_idx=layer_idx,
            block_ids=ordered,
            regions=(
                TransferRegionLayout(
                    region_idx=0,
                    base_addr=self.base_addr,
                    block_len=self.block_bytes,
                    block_stride=self.block_stride
                    if self.block_stride != self.block_bytes
                    else None,
                ),
            ),
        )


def block_slices_bytes(block_slices: list[LayerBlockSlices]) -> int:
    return sum(region.bytes for block in block_slices for region in block.regions)
