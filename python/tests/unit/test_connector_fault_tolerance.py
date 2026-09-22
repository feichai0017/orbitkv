"""GPU destinations stay held until a restore has a confirmed terminal result."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402

from orbitkv import RestoreStatus  # noqa: E402
from orbitkv.vllm.config import ConnectorContext  # noqa: E402
from orbitkv.vllm.metadata import LoadIntent, OrbitKVConnectorMetadata  # noqa: E402
from orbitkv.vllm.worker import WorkerConnector  # noqa: E402


class FakeEngineClient:
    """Minimal Cache Manager client for restore and lifecycle tests.

    Only implements what WorkerConnector touches in the load path. Save path is
    not exercised here since these tests are focused on load fault tolerance.
    """

    transport = "local"

    def __init__(self) -> None:
        self.fail_load_with_exception: Exception | None = None
        self.load_calls: list[tuple] = []
        self.register_response: tuple[bool, str] = (True, "ok")
        self.register_exception: Exception | None = None
        self.register_calls: list[tuple] = []
        self.register_kwargs: list[dict] = []
        self.unregister_calls: list[str] = []
        self.release_calls: list[bytes] = []

    def start_restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups,
        loads,
    ) -> SimpleNamespace:
        block_ids = [block_id for _, groups in loads for ids in groups for block_id in ids]
        self.load_calls.append(
            (
                instance_id,
                tp_rank,
                device_id,
                None,
                [list(group) for group in layer_groups],
                list(block_ids),
            )
        )
        if self.fail_load_with_exception is not None:
            raise self.fail_load_with_exception
        return SimpleNamespace(key=f"restore-{len(self.load_calls)}")

    def restore_completions_ready(self) -> bool:
        return False

    def poll_restore(self, _handle) -> RestoreStatus:
        return RestoreStatus(done=False, success=False)

    def register_context_batch(self, *args, **kwargs) -> tuple[bool, str]:
        self.register_calls.append(args)
        self.register_kwargs.append(kwargs)
        if self.register_exception is not None:
            raise self.register_exception
        return self.register_response

    def health(self) -> tuple[bool, str]:
        return (True, "ok")

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        self.unregister_calls.append(instance_id)
        return (True, "ok")

    def release(self, lease: bytes) -> None:
        self.release_calls.append(lease)


def _make_worker(
    pp_rank: int = 0,
    pp_size: int = 1,
    kv_cache_config=None,
    vllm_config=None,
    **ctx_kwargs,
) -> tuple[WorkerConnector, FakeEngineClient, MagicMock]:
    client = FakeEngineClient()
    state_manager = MagicMock()
    state_manager.is_available.return_value = True
    defaults = {
        "instance_id": "test_instance",
        "namespace": "ns",
        "block_size": 16,
        "tp_size": 1,
        "world_size": 1,
        "tp_rank": 0,
        "device_id": 0,
        "client": client,
        "state_manager": state_manager,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
    }
    defaults.update(ctx_kwargs)
    ctx = ConnectorContext(**defaults)
    worker = WorkerConnector(
        ctx,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
    )
    # cross-layer mode skips forward_context layer enumeration so we can drive
    # start_load_kv with a stub forward_context.
    worker._cross_layer_mode = True
    worker._cross_layer_key = "ALL_LAYERS"
    return worker, client, state_manager


def _stub_forward_context() -> MagicMock:
    ctx = MagicMock()
    ctx.no_compile_layers = {}
    return ctx


def _single_attention_cache_group(*layer_names: str) -> MagicMock:
    spec = object.__new__(FullAttentionSpec)
    object.__setattr__(spec, "block_size", 16)
    return MagicMock(layer_names=layer_names, kv_cache_spec=spec)


def _load_metadata(req_id: str, block_ids: tuple[int, ...]) -> OrbitKVConnectorMetadata:
    return OrbitKVConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids_by_group=(block_ids,),
                leases=(f"lease-{req_id}".encode(),),
                num_tokens=len(block_ids) * 16,
            )
        }
    )


def _configure_hma_worker(worker: WorkerConnector) -> None:
    worker._cross_layer_mode = False
    worker._cache_groups = MagicMock(group_count=2, has_recurrent_state=True)
    worker._registered_layers = ["attention", "recurrent"]
    worker._layer_to_group = {"attention": 0, "recurrent": 1}


def _hma_load_metadata(req_id: str) -> OrbitKVConnectorMetadata:
    return OrbitKVConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids_by_group=((11,), (21,)),
                leases=(f"lease-{req_id}".encode(),),
                num_tokens=16,
            )
        }
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "error", [ConnectionError("lost acknowledgement"), RuntimeError("rejected")]
)
def test_restore_submission_failure_does_not_release_destinations(hybrid, error):
    worker, client, state_mgr = _make_worker()
    if hybrid:
        _configure_hma_worker(worker)
        metadata = _hma_load_metadata("submit")
    else:
        metadata = _load_metadata("submit", (1, 2))
    client.fail_load_with_exception = error

    try:
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            worker.start_load_kv(metadata, _stub_forward_context())
        assert worker.get_block_ids_with_load_errors() == set()
        assert client.release_calls == []
        assert worker.get_finished(set())[1] is None
        assert state_mgr.mark_unavailable.called
    finally:
        worker.shutdown()


def test_hma_load_distinguishes_block_zero_from_absent_recurrent_target():
    worker, client, _state_mgr = _make_worker()
    _configure_hma_worker(worker)
    metadata = OrbitKVConnectorMetadata(
        load_intents={
            "hma-sparse": LoadIntent(
                block_ids_by_group=((0, 12), (None, 21)),
                leases=(b"lease-hma-sparse",),
                num_tokens=32,
            )
        }
    )

    worker.start_load_kv(metadata, _stub_forward_context())

    assert client.load_calls[0][5] == [0, 12, None, 21]
    worker.shutdown()


@pytest.mark.parametrize("hybrid", [False, True])
def test_restore_timeout_keeps_pages_pending(monkeypatch, hybrid):
    worker, _client, state_mgr = _make_worker()
    if hybrid:
        _configure_hma_worker(worker)
        metadata = _hma_load_metadata("timeout")
    else:
        metadata = _load_metadata("timeout", (5, 6, 7))
    clock = {"now": 10_000.0}
    monkeypatch.setattr("orbitkv.vllm.worker.time.perf_counter", lambda: clock["now"])
    try:
        worker.start_load_kv(metadata, _stub_forward_context())
        clock["now"] += worker.LOAD_TIMEOUT_SECONDS - 1
        assert worker.get_finished(set())[1] is None
        assert not state_mgr.mark_unavailable.called
        clock["now"] += 2
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            worker.get_finished(set())
        assert worker.get_block_ids_with_load_errors() == set()
        assert "timeout" in worker._pending_loads
        assert worker._pending_load_reqs
        assert worker._pending_load_meta
        assert state_mgr.mark_unavailable.called
    finally:
        worker.shutdown()


@pytest.mark.parametrize("failure", ["notification", "poll"])
def test_lost_completion_visibility_never_acknowledges_pages(failure):
    worker, client, state_mgr = _make_worker()
    try:
        worker.start_load_kv(_load_metadata("pending", (8, 9)), _stub_forward_context())
        if failure == "notification":
            client.restore_completions_ready = MagicMock(side_effect=OSError("closed fd"))
        else:
            client.restore_completions_ready = lambda: True
            client.poll_restore = MagicMock(side_effect=ConnectionError("disconnected"))
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            worker.get_finished(set())
        assert worker.get_block_ids_with_load_errors() == set()
        assert "pending" in worker._pending_loads
        assert client.release_calls == []
        assert state_mgr.mark_unavailable.called
    finally:
        worker.shutdown()


@pytest.mark.parametrize("hybrid", [False, True])
def test_only_confirmed_failure_permits_recovery(hybrid):
    worker, client, _ = _make_worker()
    if hybrid:
        _configure_hma_worker(worker)
        metadata = _hma_load_metadata("failed")
    else:
        metadata = _load_metadata("failed", (5, 6))
    client.restore_completions_ready = lambda: True
    client.poll_restore = lambda _: RestoreStatus(done=True, success=False, message="drained")
    try:
        worker.start_load_kv(metadata, _stub_forward_context())
        if hybrid:
            with pytest.raises(RuntimeError, match="cannot recover failed loads"):
                worker.get_finished(set())
        else:
            assert worker.get_finished(set())[1] == {"failed"}
            assert worker.get_block_ids_with_load_errors() == {5, 6}
            assert worker.get_block_ids_with_load_errors() == set()
        assert worker._pending_loads == {}
    finally:
        worker.shutdown()


def test_load_uses_registered_layer_names_before_forward_context_names():
    """Load must use the same layer names registered with the server."""
    worker, client, _ = _make_worker()
    worker._cross_layer_mode = False
    worker._registered_layers = ["registered.layer.0", "registered.layer.1"]

    forward_context = MagicMock()
    forward_layer = MagicMock()
    forward_layer.kv_cache = object()
    forward_context.no_compile_layers = {"model.layers.0.attn": forward_layer}

    worker.start_load_kv(_load_metadata("req_registered_layers", (1, 2)), forward_context)

    assert len(client.load_calls) == 1
    assert client.load_calls[0][4] == [["registered.layer.0", "registered.layer.1"]]

    worker.shutdown()


def test_worker_consumes_restore_completion():
    data_client = MagicMock(transport="iceoryx2")
    restore = SimpleNamespace(key="local:41:9")
    data_client.start_restore.return_value = restore
    data_client.restore_completions_ready.return_value = True
    data_client.poll_restore.return_value = RestoreStatus(done=True, success=True)
    worker, _unused_client, _state_manager = _make_worker(client=data_client)

    worker.start_load_kv(_load_metadata("local-restore", (3, 4)), _stub_forward_context())
    _, finished_recving = worker.get_finished(set())

    assert finished_recving == {"local-restore"}
    data_client.start_restore.assert_called_once_with(
        "test_instance",
        0,
        0,
        [["ALL_LAYERS"]],
        [(b"lease-local-restore", [[3, 4]])],
    )
    data_client.poll_restore.assert_called_once_with(restore)
    worker.shutdown()


class FakeTensor:
    shape = (1, 16)
    dtype = "float16"

    def storage_offset(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return "cuda:0"

    def stride(self) -> tuple[int, int]:
        return (16, 1)

    def element_size(self) -> int:
        return 2


class FakeCudaIPCWrapper:
    def __init__(self, _tensor) -> None:
        pass


def test_register_version_mismatch_raises_startup_error(monkeypatch):
    worker, client, _ = _make_worker()
    client.register_response = (
        False,
        "OrbitKV version mismatch: client=0.22.4 server=0.22.5",
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="OrbitKV version mismatch") as exc_info:
        worker.register_kv_caches({"layer.0": FakeTensor()})

    assert "client=0.22.4" in str(exc_info.value)
    assert "server=0.22.5" in str(exc_info.value)
    assert "for layer.0" not in str(exc_info.value)
    assert len(client.register_calls) == 1
    assert client.unregister_calls == []

    worker.shutdown()
    assert client.unregister_calls == []


def test_register_non_version_failure_reports_batch_layers(monkeypatch):
    worker, client, _ = _make_worker()
    client.register_response = (False, "invalid tensor metadata")

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="invalid tensor metadata") as exc_info:
        worker.register_kv_caches(
            {
                "layer.0": FakeTensor(),
                "layer.1": FakeTensor(),
            }
        )

    message = str(exc_info.value)
    assert "Register context batch failed for layers ['layer.0', 'layer.1']" in message
    assert "for layer.1" not in message
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.0", "layer.1"]

    worker.shutdown()


def test_register_kv_caches_ignores_shared_by_without_layer_split_opt_in(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        _single_attention_cache_group("layer.0", "layer.1", "layer.2")
    ]
    kv_cache_config.kv_cache_tensors = [
        MagicMock(shared_by=("layer.1",)),
    ]
    worker, client, _ = _make_worker(
        kv_cache_config=kv_cache_config,
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_kv_caches(
        {
            "layer.0": FakeTensor(),
            "layer.1": FakeTensor(),
            "layer.2": FakeTensor(),
        }
    )

    assert worker._registered_layers == ["layer.0", "layer.1", "layer.2"]
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.0", "layer.1", "layer.2"]

    worker.shutdown()


def test_register_kv_caches_uses_layer_split_shared_by_plan(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        _single_attention_cache_group("layer.0", "layer.1", "layer.2")
    ]
    kv_cache_config.kv_cache_tensors = [
        MagicMock(shared_by=("layer.1",)),
        MagicMock(shared_by=()),
        MagicMock(shared_by=("layer.0",)),
    ]
    worker, client, _ = _make_worker(
        kv_cache_config=kv_cache_config,
        vllm_config=MagicMock(additional_config={"mla_layer_split_kv_cache": True}),
        is_mla=True,
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_kv_caches(
        {
            "layer.0": FakeTensor(),
            "layer.1": FakeTensor(),
            "layer.2": FakeTensor(),
        }
    )

    assert worker._registered_layers == ["layer.1", "layer.0"]
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.1", "layer.0"]

    worker.shutdown()


def test_register_kv_caches_requires_shared_by_layers(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [_single_attention_cache_group("layer.0", "layer.1")]
    kv_cache_config.kv_cache_tensors = [MagicMock(shared_by=("layer.1",))]
    worker, _, _ = _make_worker(
        kv_cache_config=kv_cache_config,
        vllm_config=MagicMock(additional_config={"mla_layer_split_kv_cache": True}),
        is_mla=True,
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="missing layers"):
        worker.register_kv_caches({"layer.0": FakeTensor()})

    worker.shutdown()


def test_cross_layer_registration_uses_pp_suffixed_name(monkeypatch):
    worker, client, _ = _make_worker(pp_rank=1, pp_size=4)

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_cross_layers_kv_cache(FakeTensor(), attn_backend=object())

    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["ALL_LAYERS_pp1"]

    worker.shutdown()


def test_register_version_mismatch_rpc_error_stops_startup(monkeypatch):
    worker, client, _ = _make_worker()
    client.register_exception = RuntimeError(
        "register_context_batch RPC failed: status: FailedPrecondition, "
        'message: "OrbitKV version mismatch: client=0.22.4 server=0.22.5"'
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="OrbitKV version mismatch") as exc_info:
        worker.register_kv_caches({"layer.0": FakeTensor()})

    assert "FailedPrecondition" in str(exc_info.value)
    assert "client=0.22.4" in str(exc_info.value)
    assert "server=0.22.5" in str(exc_info.value)
    assert len(client.register_calls) == 1

    worker.shutdown()
