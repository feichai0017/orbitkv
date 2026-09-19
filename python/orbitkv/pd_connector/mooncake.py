"""P/D transfer port backed by the upstream Mooncake Transfer Engine."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from orbitkv.logging_utils import get_connector_logger
from orbitkv.pd_connector.layout import (
    BlockRegionSlice,
    LayerBlockSlices,
    block_slices_bytes,
)
from orbitkv.pd_connector.metadata import (
    LayerRemoteLayout,
    PdHandshake,
    layer_layout_from_dict,
)

logger = get_connector_logger()
_MISSING = object()


class MooncakePort(Protocol):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]: ...

    def open_request(self, req_id: str, handshake: PdHandshake) -> None: ...

    def endpoint(self) -> str: ...

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def wait_for_pushes(self, req_id: str) -> None: ...

    def push_done(self, req_id: str) -> None: ...

    def write_stats(self, req_id: str) -> dict[str, Any]: ...

    def fail_request(self, req_id: str) -> None: ...

    def abort_request(self, req_id: str) -> None: ...

    def aggregated_link_speed(self) -> int: ...

    def wait_done(self, req_id: str) -> None: ...

    def pop_finished_sending(self) -> set[str]: ...

    def pop_finished_recving(self) -> set[str]: ...

    def close_request(self, req_id: str) -> None: ...


class MockMooncakePort:
    """A test double that records transfer calls without loading Mooncake."""

    def __init__(self) -> None:
        self.local_layers: tuple[LayerRemoteLayout, ...] = ()
        self.registered: set[str] = set()
        self.peer_handshakes: dict[str, PdHandshake | None] = {}
        self.pushed_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = {}
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        self.local_layers = layers
        return layers

    def open_request(self, req_id: str, handshake: PdHandshake) -> None:
        self.registered.add(req_id)
        self.peer_handshakes[req_id] = handshake

    def endpoint(self) -> str:
        return "127.0.0.1:15290"

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.pushed_layers.setdefault(req_id, [])
        self.pushed_layers[req_id].append((layer_idx, blocks))

    def wait_for_pushes(self, req_id: str) -> None:
        return None

    def push_done(self, req_id: str) -> None:
        self._finished_sending.add(req_id)

    def write_stats(self, req_id: str) -> dict[str, Any]:
        bytes_total = sum(
            block_slices_bytes(blocks) for _, blocks in self.pushed_layers.get(req_id, [])
        )
        return {
            "submitted": len(self.pushed_layers.get(req_id, [])),
            "completed": len(self.pushed_layers.get(req_id, [])),
            "errors": 0,
            "bytes": bytes_total,
            "has_submit": bytes_total > 0,
            "has_complete": bytes_total > 0,
        }

    def fail_request(self, req_id: str) -> None:
        return None

    def abort_request(self, req_id: str) -> None:
        self._finished_recving.add(req_id)

    def aggregated_link_speed(self) -> int:
        return 400_000_000_000

    def wait_done(self, req_id: str) -> None:
        return None

    def pop_finished_sending(self) -> set[str]:
        finished = self._finished_sending
        self._finished_sending = set()
        return finished

    def pop_finished_recving(self) -> set[str]:
        finished = self._finished_recving
        self._finished_recving = set()
        return finished

    def close_request(self, req_id: str) -> None:
        self.registered.discard(req_id)
        self.peer_handshakes.pop(req_id, None)
        self.pushed_layers.pop(req_id, None)
        self._finished_sending.discard(req_id)
        self._finished_recving.discard(req_id)


def _block_slice_to_native(block: BlockRegionSlice) -> dict[str, int]:
    return {
        "block_id": block.block_id,
        "src_offset_bytes": block.src_offset_bytes,
        "bytes": block.bytes,
    }


def _layer_blocks_to_native(blocks: list[LayerBlockSlices]) -> list[dict[str, Any]]:
    return [
        {
            "regions": [
                {"region_idx": region_idx, **_block_slice_to_native(region)}
                for region_idx, region in enumerate(block.regions)
            ],
        }
        for block in _coalesce_contiguous_blocks(blocks)
    ]


def _coalesce_contiguous_blocks(blocks: list[LayerBlockSlices]) -> list[LayerBlockSlices]:
    if len(blocks) < 2:
        return blocks

    coalesced: list[LayerBlockSlices] = []
    current = blocks[0]
    for block in blocks[1:]:
        if _can_extend_block_range(current, block):
            current = LayerBlockSlices(
                regions=tuple(
                    BlockRegionSlice(
                        block_id=current_region.block_id,
                        src_offset_bytes=current_region.src_offset_bytes,
                        bytes=current_region.bytes + block_region.bytes,
                    )
                    for current_region, block_region in zip(
                        current.regions,
                        block.regions,
                        strict=True,
                    )
                ),
            )
            continue
        coalesced.append(current)
        current = block
    coalesced.append(current)
    return coalesced


def _can_extend_block_range(prev: LayerBlockSlices, nxt: LayerBlockSlices) -> bool:
    if len(prev.regions) != len(nxt.regions):
        return False
    for prev_region, next_region in zip(prev.regions, nxt.regions, strict=True):
        if prev_region.bytes % next_region.bytes != 0:
            return False
        block_count = prev_region.bytes // next_region.bytes
        if prev_region.block_id + block_count != next_region.block_id:
            return False
        if prev_region.src_offset_bytes + prev_region.bytes != next_region.src_offset_bytes:
            return False
    return True


def _layer_from_native(layer: LayerRemoteLayout | dict[str, Any]) -> LayerRemoteLayout:
    if isinstance(layer, LayerRemoteLayout):
        return layer
    return layer_layout_from_dict(layer)


class RealMooncakePort:
    """P/D layout adapter over the upstream Mooncake Transfer Engine."""

    def __init__(self, engine: Any, *, nic_count: int = 1) -> None:
        self.engine = engine
        self.nic_count = nic_count
        self.local_layers: dict[int, LayerRemoteLayout] = {}
        self.peer_handshakes: dict[str, PdHandshake] = {}
        self._request_generations: dict[str, int] = {}
        self._next_request_generation = 0
        self._stats: dict[str, dict[str, Any]] = {}
        self._notification_counts: dict[tuple[str, str], int] = {}
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._lock = threading.RLock()

    def endpoint(self) -> str:
        return str(self.engine.endpoint)

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        start = time.perf_counter()
        regions = []
        for layer in layers:
            max_block_id = max(layer.block_ids)
            for region in layer.regions:
                stride = region.block_stride or region.block_len
                regions.append(
                    {
                        "addr": region.base_addr,
                        "len": max_block_id * stride + region.block_len,
                        "location": "*",
                    }
                )
            self.local_layers[layer.layer_idx] = layer
        self.engine.register_memory(regions)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[PdConnector] Mooncake register_local_layers endpoint=%s layers=%d blocks_per_layer=%s regions_per_layer=%s native_ms=%.3f",
            self.endpoint(),
            len(layers),
            [len(layer.block_ids) for layer in layers],
            [len(layer.regions) for layer in layers],
            elapsed_ms,
        )
        return layers

    def open_request(self, req_id: str, handshake: PdHandshake) -> None:
        start = time.perf_counter()
        if not handshake.transfer_endpoint:
            raise ValueError("Mooncake handshake is missing transfer_endpoint")
        with self._lock:
            self._next_request_generation += 1
            self.peer_handshakes[req_id] = handshake
            self._request_generations[req_id] = self._next_request_generation
        elapsed_ms = (time.perf_counter() - start) * 1000
        blocks_per_layer = len(handshake.layers[0].block_ids) if handshake.layers else 0
        logger.info(
            "[PdConnector] Mooncake open_request req=%s remote_req=%s endpoint=%s tp_rank=%d/%d layers=%d blocks_per_layer=%d block_size=%d native_ms=%.3f",
            req_id,
            handshake.request_id,
            handshake.transfer_endpoint,
            handshake.tp_rank,
            handshake.tp_size,
            len(handshake.layers),
            blocks_per_layer,
            handshake.block_size,
            elapsed_ms,
        )

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        start = time.perf_counter()
        with self._lock:
            local = self.local_layers.get(layer_idx)
            handshake = self.peer_handshakes.get(req_id)
        if local is None:
            raise RuntimeError(f"local layer {layer_idx} is not registered")
        if handshake is None:
            raise RuntimeError(f"remote request {req_id} is not registered")
        remote = handshake.layers_by_idx.get(layer_idx)
        if remote is None:
            raise RuntimeError(f"remote layer {layer_idx} for request {req_id} is not registered")

        native_blocks = _layer_blocks_to_native(blocks)
        slices: list[tuple[int, int, int]] = []
        allowed_blocks = set(remote.block_ids)
        local_base = min(region.base_addr for region in local.regions)
        for block in native_blocks:
            for region_idx, block_region in enumerate(block["regions"]):
                remote_region = remote.regions[region_idx]
                block_id = int(block_region["block_id"])
                source_offset = int(block_region["src_offset_bytes"])
                bytes_len = int(block_region["bytes"])
                stride = remote_region.block_stride or remote_region.block_len
                if bytes_len % remote_region.block_len:
                    raise ValueError(
                        f"block slice bytes {bytes_len} must be a multiple of "
                        f"remote block length {remote_region.block_len}"
                    )
                block_count = bytes_len // remote_region.block_len
                for offset in range(block_count):
                    remote_block_id = block_id + offset
                    if remote_block_id not in allowed_blocks:
                        raise RuntimeError(f"remote block {remote_block_id} is not authorized")
                    slices.append(
                        (
                            local_base + source_offset + offset * remote_region.block_len,
                            remote_region.base_addr + remote_block_id * stride,
                            remote_region.block_len,
                        )
                    )
        bytes_total = sum(length for _, _, length in slices)
        self.engine.write(handshake.transfer_endpoint, slices, timeout_s=30.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        with self._lock:
            stats = self._stats.setdefault(
                req_id,
                {"submitted": 0, "completed": 0, "errors": 0, "bytes": 0},
            )
            stats["submitted"] += 1
            stats["completed"] += 1
            stats["bytes"] += bytes_total
            stats["xfer_window_ms"] = stats.get("xfer_window_ms", 0.0) + elapsed_ms
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[PdConnector] Mooncake push_layer req=%s layer=%d input_blocks=%d coalesced_blocks=%d regions=%d bytes=%d native_ms=%.3f",
                req_id,
                layer_idx,
                len(blocks),
                len(native_blocks),
                sum(len(block["regions"]) for block in native_blocks),
                block_slices_bytes(blocks),
                elapsed_ms,
            )

    def wait_for_pushes(self, req_id: str) -> None:
        return None

    def push_done(self, req_id: str) -> None:
        self._send_status(req_id, "done")
        with self._lock:
            self._finished_sending.add(req_id)

    def write_stats(self, req_id: str) -> dict[str, Any]:
        with self._lock:
            stats = dict(self._stats.get(req_id, {}))
        stats.setdefault("submitted", 0)
        stats.setdefault("completed", 0)
        stats.setdefault("errors", 0)
        stats.setdefault("bytes", 0)
        stats["has_submit"] = stats["submitted"] > 0
        stats["has_complete"] = stats["completed"] > 0
        return stats

    def fail_request(self, req_id: str) -> None:
        self._send_status(req_id, "failed")

    def abort_request(self, req_id: str) -> None:
        self._send_status(req_id, "aborted")

    def aggregated_link_speed(self) -> int:
        return 0

    def wait_done(self, req_id: str) -> None:
        with self._lock:
            handshake = self.peer_handshakes.get(req_id)
            generation = self._request_generations.get(req_id)
        if handshake is None:
            raise RuntimeError(f"remote request {req_id} is not registered")
        deadline = time.monotonic() + 30.0
        while True:
            self._poll_notifications()
            with self._lock:
                if self._request_generations.get(req_id) != generation:
                    return
                failed = self._notification_counts.get((handshake.request_id, "failed"), 0)
                aborted = self._notification_counts.get((handshake.request_id, "aborted"), 0)
                done = self._notification_counts.get((handshake.request_id, "done"), 0)
                if failed:
                    raise RuntimeError(f"Mooncake transfer failed for request {req_id}")
                if aborted:
                    self._finished_recving.add(req_id)
                    return
                if done >= handshake.expected_notify_count:
                    self._finished_recving.add(req_id)
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Mooncake notification timed out for request {req_id}")
            time.sleep(0.00005)

    def pop_finished_sending(self) -> set[str]:
        with self._lock:
            finished = self._finished_sending
            self._finished_sending = set()
            return finished

    def pop_finished_recving(self) -> set[str]:
        with self._lock:
            finished = self._finished_recving
            self._finished_recving = set()
            return finished

    def close_request(self, req_id: str) -> None:
        with self._lock:
            handshake = self.peer_handshakes.pop(req_id, None)
            self._request_generations.pop(req_id, None)
            self._stats.pop(req_id, None)
            self._finished_sending.discard(req_id)
            self._finished_recving.discard(req_id)
            if handshake is not None:
                for status in ("done", "failed", "aborted"):
                    self._notification_counts.pop((handshake.request_id, status), None)

    def _send_status(self, req_id: str, status: str) -> None:
        with self._lock:
            handshake = self.peer_handshakes.get(req_id)
        if handshake is None:
            raise RuntimeError(f"remote request {req_id} is not registered")
        self.engine.send_notification(handshake.transfer_endpoint, handshake.request_id, status)

    def _poll_notifications(self) -> None:
        notifications = self.engine.take_notifications()
        if not notifications:
            return
        with self._lock:
            for name, status in notifications:
                key = (str(name), str(status))
                self._notification_counts[key] = self._notification_counts.get(key, 0) + 1


def build_mooncake_port(
    vllm_config: Any,
    cuda_device: int | None,
    *,
    tp_rank: int | None = None,
) -> MooncakePort:
    config = getattr(vllm_config, "kv_transfer_config", None)
    enabled = _extra(config, "orbitkv.pd.mooncake.enabled", _MISSING)
    if enabled is not _MISSING and not _as_bool(enabled):
        raise RuntimeError("PdConnector requires Mooncake; orbitkv.pd.mooncake.enabled=false")

    try:
        from orbitkv.orbitkv import MooncakeTransferEngine
    except ImportError as exc:
        raise RuntimeError("PdConnector requires the OrbitKV native extension") from exc
    except AttributeError as exc:
        raise RuntimeError("orbitkv.orbitkv does not expose MooncakeTransferEngine") from exc

    resolved_cuda_device = int(cuda_device or 0)
    resolved_tp_rank = _tp_rank(vllm_config) if tp_rank is None else int(tp_rank)
    rank_config = _rank_mooncake_config(config, resolved_tp_rank, cuda_device=resolved_cuda_device)
    bind_host = str(
        _extra(
            config,
            "orbitkv.pd.mooncake.bind_host",
            os.getenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "127.0.0.1"),
        )
    )
    nics = [rank_config.nic] if rank_config.nic else []
    engine = MooncakeTransferEngine(bind_host=bind_host, nics=nics)
    logger.info(
        "[PdConnector] Mooncake enabled tp_rank=%d cuda=%d nics=%s endpoint=%s",
        rank_config.tp_rank,
        resolved_cuda_device,
        nics,
        engine.endpoint,
    )
    return RealMooncakePort(engine, nic_count=len(nics))


def _extra(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get_from_extra_config", None)
    if getter is not None:
        return getter(key, default)
    extra_config = getattr(config, "extra_config", None)
    if isinstance(extra_config, dict):
        return extra_config.get(key, default)
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class _RankMooncakeConfig:
    tp_rank: int
    nic: str | None


def _tp_rank(vllm_config: Any) -> int:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0)


def _rank_mooncake_config(
    config: Any,
    tp_rank: int,
    *,
    cuda_device: int | None = None,
) -> _RankMooncakeConfig:
    for legacy_key in ("orbitkv.pd.rdma.domains", "orbitkv.pd.rdma.rank_map"):
        if _extra(config, legacy_key, _MISSING) is not _MISSING:
            raise RuntimeError(
                f"{legacy_key} was removed; configure orbitkv.pd.mooncake.rank_map instead"
            )
    rank_map = _extra(config, "orbitkv.pd.mooncake.rank_map", _MISSING)
    if rank_map is _MISSING:
        return _RankMooncakeConfig(tp_rank=tp_rank, nic=None)
    if not isinstance(rank_map, dict):
        raise RuntimeError("orbitkv.pd.mooncake.rank_map must be an object")
    rank_entry = rank_map.get(str(tp_rank))
    selected_rank = tp_rank
    if (
        cuda_device is not None
        and tp_rank == 0
        and cuda_device != 0
        and str(cuda_device) in rank_map
    ):
        rank_entry = rank_map[str(cuda_device)]
        selected_rank = cuda_device
    if not isinstance(rank_entry, dict):
        known = ", ".join(sorted(str(rank) for rank in rank_map))
        raise RuntimeError(
            f"PdConnector Mooncake rank_map missing tp_rank={tp_rank}; configured ranks=[{known}]"
        )
    nic = str(rank_entry.get("nic") or "") or None
    return _RankMooncakeConfig(
        tp_rank=selected_rank,
        nic=nic,
    )
