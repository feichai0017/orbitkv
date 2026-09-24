from __future__ import annotations

# ruff: noqa: E402,F401
import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (  # noqa: E402
    PromMetric,
)

import orbitkv.orbitkv as native  # noqa: E402
import orbitkv.vllm.pd.decode_worker as decode_worker_mod  # noqa: E402
import orbitkv.vllm.pd.prefill as prefill_mod  # noqa: E402
import orbitkv.vllm.pd.prefill_worker as prefill_worker_mod  # noqa: E402
import orbitkv.vllm.pd.worker as worker_mod  # noqa: E402
from orbitkv.vllm.pd import (  # noqa: E402
    PdConnector,
    PdDecodeConnector,
    PdPrefillConnector,
)
from orbitkv.vllm.pd.kv_params import parse_consumer  # noqa: E402
from orbitkv.vllm.pd.layout import (  # noqa: E402
    BlockRegionSlice,
    FlashAttnHndLayout,
    LayerBlockSlices,
    block_slices_bytes,
    unique_blocks_from_slot_mapping,
)
from orbitkv.vllm.pd.metadata import (  # noqa: E402
    RELEASE_CONSUMER_ABORT,
    RELEASE_PRODUCER_ABORT,
    RELEASE_PRODUCER_FINISHED,
    RELEASE_PRODUCER_PREEMPTED,
    LayerRemoteLayout,
    PdConnectorMetadata,
    PdHandshake,
    PdWorkerMetadata,
    PushReqMeta,
    TransferRegionLayout,
    WaitReqMeta,
    handshake_from_dict,
    handshake_to_compact_dict,
    handshake_to_dict,
    handshakes_from_dicts,
)
from orbitkv.vllm.pd.mooncake import (  # noqa: E402
    RealMooncakePort,
    _layer_blocks_to_native,
)
from orbitkv.vllm.pd.prefill import (  # noqa: E402
    AsyncPrefillSender,
    PrefillHttpTask,
)
from orbitkv.vllm.pd.proxy import (  # noqa: E402
    PdEndpoint,
    ProxyConfig,
    RoundRobinPairRouter,
    build_pd_proxy_request,
    build_router,
    iter_http_stream_bytes,
    render_proxy_metrics,
)
from orbitkv.vllm.pd.scheduler import (  # noqa: E402
    PdDecodeSchedulerConnector,
    PdPrefillSchedulerConnector,
)
from orbitkv.vllm.pd.worker import (  # noqa: E402
    PdDecodeWorkerConnector,
    PdPrefillWorkerConnector,
)


class MockMooncakePort:
    """A test double that records transfer calls without loading Mooncake."""

    def __init__(self) -> None:
        self.local_layers: tuple[LayerRemoteLayout, ...] = ()
        self.registered: set[str] = set()
        self.peer_handshakes: dict[str, PdHandshake | None] = {}
        self.pushed_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = {}
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._request_generations: dict[str, int] = {}
        self._next_request_generation = 0

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        self.local_layers = layers
        return layers

    def open_request(self, req_id: str, handshake: PdHandshake) -> int:
        self._next_request_generation += 1
        self._request_generations[req_id] = self._next_request_generation
        self.registered.add(req_id)
        self.peer_handshakes[req_id] = handshake
        return self._next_request_generation

    def endpoint(self) -> str:
        return "127.0.0.1:15290"

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
        *,
        request_generation: int,
    ) -> None:
        if self._request_generations.get(req_id) != request_generation:
            raise RuntimeError(f"stale Mooncake push generation for request {req_id}")
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
        self._request_generations.pop(req_id, None)
        self.pushed_layers.pop(req_id, None)
        self._finished_sending.discard(req_id)
        self._finished_recving.discard(req_id)


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        ptr: int = 0x1000,
        element_size: int = 2,
        device_index: int | None = None,
    ) -> None:
        self.shape = shape
        self._stride = stride
        self._ptr = ptr
        self._element_size = element_size
        self.device = SimpleNamespace(index=device_index) if device_index is not None else None

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def data_ptr(self) -> int:
        return self._ptr

    def element_size(self) -> int:
        return self._element_size


class FakeSlotMapping(list[int]):
    def __init__(self, values: list[int]) -> None:
        super().__init__(values)
        self.cpu_calls = 0

    def detach(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self

    def tolist(self):
        return list(self)


class FakePrefillSender:
    def __init__(self) -> None:
        self.tasks = []
        self.cancelled = []

    def submit(self, task) -> None:
        self.tasks.append(task)

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class FakeMooncakeTransferEngine:
    def __init__(self) -> None:
        self.endpoint = "127.0.0.1:15290"
        self.registered_regions = []
        self.writes = []
        self.notifications = []

    def register_memory(self, regions):
        self.registered_regions.extend(regions)

    def write(
        self,
        remote_endpoint,
        slices,
        timeout_s=30.0,
        notify_name=None,
        notify_message=None,
    ):
        self.writes.append((remote_endpoint, slices, timeout_s))
        if notify_name is not None:
            self.notifications.append((notify_name, notify_message))
        return sum(length for _, _, length in slices)

    def send_notification(self, remote_endpoint, name, message):
        self.notifications.append((name, message))

    def take_notifications(self):
        notifications = self.notifications
        self.notifications = []
        return notifications

    def complete(self, request_id: str, status: str = "done") -> None:
        self.notifications.append((request_id, status))


class FakeMooncakeTransferEngineCtor(FakeMooncakeTransferEngine):
    last_kwargs = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).last_kwargs = kwargs


def drain_pd_pushes(worker: PdDecodeWorkerConnector | PdPrefillWorkerConnector) -> None:
    worker._push_sender.wait_all()
    worker._push_finalizer.wait_all()


def pushed_layers_by_idx(
    transfer: MockMooncakePort,
    req_id: str,
) -> dict[int, list[LayerBlockSlices]]:
    return dict(transfer.pushed_layers[req_id])


DUMMY_HANDSHAKE = PdHandshake(
    request_id="",
    engine_id="",
    transfer_endpoint="placeholder:1",
    tp_rank=0,
    tp_size=1,
    block_size=16,
    layers=(),
)


def hnd_remote_layer(
    *,
    layer_name: str = "layer.0",
    layer_idx: int = 0,
    block_ids: tuple[int, ...] = (0,),
    k_base: int = 0x1000,
    v_base: int = 0x8000,
    block_len: int = 0x400,
) -> LayerRemoteLayout:
    return LayerRemoteLayout(
        layer_name=layer_name,
        layer_idx=layer_idx,
        block_ids=block_ids,
        regions=(
            TransferRegionLayout(region_idx=0, base_addr=k_base, block_len=block_len),
            TransferRegionLayout(region_idx=1, base_addr=v_base, block_len=block_len),
        ),
    )


def decode_handshakes(tp_size: int, *, block_size: int = 16) -> tuple[PdHandshake, ...]:
    return tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            transfer_endpoint=f"127.0.0.1:{15290 + rank}",
            tp_rank=rank,
            tp_size=tp_size,
            block_size=block_size,
            layers=(),
        )
        for rank in range(tp_size)
    )


def fake_mla_config(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    block_size: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="pd"),
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=tp_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def fake_mtp_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="pd"),
        model_config=SimpleNamespace(
            use_mla=False,
            hf_text_config=SimpleNamespace(num_nextn_predict_layers=1),
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def fake_kv_cache_config(
    *,
    num_blocks: int,
    specs: dict[str, SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=tuple(specs),
                kv_cache_spec=SimpleNamespace(kv_cache_specs=specs),
            )
        ],
    )


def fake_mtp_kv_cache_config(*, num_blocks: int = 8) -> SimpleNamespace:
    base_spec = SimpleNamespace(block_size=16, page_size_bytes=2 * 16 * 4 * 32 * 2)
    draft_spec = SimpleNamespace(block_size=16, page_size_bytes=2 * 16 * 4 * 32 * 2)
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=("model.layers.0.self_attn",),
                kv_cache_spec=SimpleNamespace(
                    kv_cache_specs={"model.layers.0.self_attn": base_spec}
                ),
                is_eagle_group=False,
            ),
            SimpleNamespace(
                layer_names=("model.layers.27.self_attn",),
                kv_cache_spec=SimpleNamespace(
                    kv_cache_specs={"model.layers.27.self_attn": draft_spec}
                ),
                is_eagle_group=True,
            ),
        ],
    )


__all__ = [name for name in globals() if not name.startswith("__") and name != "teardown_module"]
