from __future__ import annotations

# ruff: noqa: E402,F401
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (  # noqa: E402
    PromMetric,
)

import orbitkv.orbitkv as native  # noqa: E402
import orbitkv.pd_connector.decode_worker as decode_worker_mod  # noqa: E402
import orbitkv.pd_connector.prefill as prefill_mod  # noqa: E402
import orbitkv.pd_connector.prefill_worker as prefill_worker_mod  # noqa: E402
import orbitkv.pd_connector.worker as worker_mod  # noqa: E402
from orbitkv.pd_connector import (  # noqa: E402
    PdConnector,
    PdDecodeConnector,
    PdPrefillConnector,
)
from orbitkv.pd_connector.kv_params import parse_consumer  # noqa: E402
from orbitkv.pd_connector.layout import (  # noqa: E402
    BlockRegionSlice,
    FlashAttnHndLayout,
    LayerBlockSlices,
    unique_blocks_from_slot_mapping,
)
from orbitkv.pd_connector.metadata import (  # noqa: E402
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
from orbitkv.pd_connector.mooncake import (  # noqa: E402
    MockMooncakePort,
    RealMooncakePort,
    _layer_blocks_to_native,
)
from orbitkv.pd_connector.prefill import (  # noqa: E402
    AsyncPrefillSender,
    PrefillHttpTask,
)
from orbitkv.pd_connector.proxy import (  # noqa: E402
    PdEndpoint,
    ProxyConfig,
    RoundRobinPairRouter,
    build_pd_proxy_request,
    build_router,
    iter_http_stream_bytes,
    render_proxy_metrics,
)
from orbitkv.pd_connector.scheduler import (  # noqa: E402
    PdDecodeSchedulerConnector,
    PdPrefillSchedulerConnector,
)
from orbitkv.pd_connector.worker import (  # noqa: E402
    PdDecodeWorkerConnector,
    PdPrefillWorkerConnector,
)


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
