"""Contracts for splitting one vLLM TP replica across OrbitKV servers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

from orbitkv.orbitkv import QueryLoading, QueryReady  # noqa: E402
from orbitkv.vllm import OrbitKVConnector  # noqa: E402
from orbitkv.vllm.common import (  # noqa: E402
    ConnectorContext,
    LoadIntent,
    OrbitKVConnectorMetadata,
    TpShardTopology,
)
from orbitkv.vllm.scheduler import SchedulerConnector  # noqa: E402
from orbitkv.vllm.worker import WorkerConnector  # noqa: E402


@pytest.fixture(autouse=True)
def _available_local_sockets(monkeypatch):
    monkeypatch.setattr("orbitkv.client.data_plane._is_unix_socket", lambda _path: True)


def _topology() -> TpShardTopology:
    return TpShardTopology.from_config(
        default_endpoint="http://unused:50055",
        configured_endpoints=["http://node-a:50055", "http://node-b:50055"],
        global_tp_size=8,
        global_world_size=8,
    )


def _context(**kwargs) -> ConnectorContext:
    defaults = {
        "instance_id": "instance",
        "namespace": "namespace:tp-shard-0-of-2",
        "block_size": 16,
        "tp_size": 8,
        "world_size": 8,
        "tp_rank": 0,
        "device_id": 0,
        "engine_client": MagicMock(),
        "state_manager": MagicMock(),
        "tp_shards": _topology(),
    }
    defaults.update(kwargs)
    return ConnectorContext(**defaults)  # type: ignore[arg-type]


def _vllm_config(*, extra_overrides=None, **parallel_overrides):
    extra_config = {
        "orbitkv.tp_shard_endpoints": [
            "http://127.0.0.1:50055",
            "http://127.0.0.1:50056",
        ]
    }
    extra_config.update(extra_overrides or {})
    kv_transfer_config = SimpleNamespace(
        engine_id="instance",
        get_from_extra_config=lambda key, default: extra_config.get(key, default),
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="model",
            dtype="bfloat16",
            hf_text_config=SimpleNamespace(kv_lora_rank=None),
            get_total_num_kv_heads=lambda: 8,
            get_head_size=lambda: 128,
            get_total_num_hidden_layers=lambda: 32,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            **{
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "world_size": 8,
                "decode_context_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                **parallel_overrides,
            }
        ),
        kv_transfer_config=kv_transfer_config,
        additional_config={},
    )


def test_topology_maps_contiguous_global_tp_ranks_to_local_servers():
    topology = _topology()

    assert topology.local_tp_size == 4
    assert topology.local_world_size == 4
    assert topology.shard_index(3) == 0
    assert topology.shard_index(4) == 1
    assert topology.local_tp_rank(4) == 0
    assert topology.local_tp_rank(7) == 3
    assert topology.namespace("base", 1) == "base:tp-shard-1-of-2"


@pytest.mark.parametrize(
    ("endpoints", "tp_size", "world_size", "message"),
    [
        ([], 8, 8, "non-empty strings"),
        (["http://a", "http://a"], 8, 8, "duplicates"),
        (["http://a", "http://b", "http://c"], 8, 8, "tensor_parallel_size"),
        (["http://a", "http://b"], 8, 9, "world_size"),
    ],
)
def test_topology_rejects_ambiguous_shard_layouts(endpoints, tp_size, world_size, message):
    with pytest.raises(ValueError, match=message):
        TpShardTopology.from_config("http://unused", endpoints, tp_size, world_size)


def test_context_exposes_node_local_server_topology_for_hma():
    context = _context(tp_rank=5, is_mla=True, collapse_mla_tp=False)

    assert context.tp_shard_index == 1
    assert context.effective_tp_rank == 1
    assert context.effective_tp_size == 4
    assert context.effective_world_size == 4


def test_worker_connector_routes_global_tp_rank_to_its_local_manager(monkeypatch):
    client = MagicMock(transport="local")
    monkeypatch.setattr("orbitkv.vllm.get_tensor_model_parallel_rank", lambda: 5)
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", client_factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())

    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.WORKER)
    try:
        assert connector._engine_endpoint == "http://127.0.0.1:50056"
        assert connector._ctx.namespace.endswith(":tp-shard-1-of-2")
        assert connector._ctx.effective_tp_rank == 1
        assert connector._ctx.effective_tp_size == 4
        assert connector._ctx.effective_world_size == 4
    finally:
        connector.shutdown()

    client_factory.assert_called_once_with(
        "/tmp/orbitkv-50056.sock", timeout_ms=5_000, spin_iterations=64
    )


def test_scheduler_opens_a_local_topology_session_on_every_manager(monkeypatch):
    first = MagicMock(transport="local")
    second = MagicMock(transport="local")
    monkeypatch.setattr(
        "orbitkv.client.connection.LocalDataClient", MagicMock(side_effect=[first, second])
    )
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())

    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    try:
        first.start_session_watcher.assert_called_once()
        assert first.start_session_watcher.call_args.args[1].endswith(":tp-shard-0-of-2")
        assert first.start_session_watcher.call_args.args[2:] == (4, 4)
        second.start_session_watcher.assert_called_once()
        assert second.start_session_watcher.call_args.args[1].endswith(":tp-shard-1-of-2")
        assert second.start_session_watcher.call_args.args[2:] == (4, 4)
    finally:
        connector.shutdown()


def test_scheduler_maps_each_tp_shard_to_its_local_socket(monkeypatch):
    clients = [MagicMock(transport="local"), MagicMock(transport="local")]
    factory = MagicMock(side_effect=clients)
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_bootstrap_sockets": [
                "/run/orbitkv/a.sock",
                "/run/orbitkv/b.sock",
            ],
        }
    )

    connector = OrbitKVConnector(config, KVConnectorRole.SCHEDULER)
    try:
        assert connector._ctx.data_client is clients[0]
        assert connector._scheduler._tp_shard_client._clients == tuple(clients)
        assert connector._lifecycle_clients == tuple(clients)
        for client in clients:
            client.start_session_watcher.assert_called_once()
    finally:
        connector.shutdown()

    assert factory.call_args_list == [
        call("/run/orbitkv/a.sock", timeout_ms=5_000, spin_iterations=64),
        call("/run/orbitkv/b.sock", timeout_ms=5_000, spin_iterations=64),
    ]


def test_scheduler_derives_distinct_local_sockets(monkeypatch):
    clients = [MagicMock(transport="local"), MagicMock(transport="local")]
    factory = MagicMock(side_effect=clients)
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())

    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    try:
        assert connector._ctx.data_client is clients[0]
    finally:
        connector.shutdown()
    assert factory.call_args_list == [
        call("/tmp/orbitkv-50055.sock", timeout_ms=5_000, spin_iterations=64),
        call("/tmp/orbitkv-50056.sock", timeout_ms=5_000, spin_iterations=64),
    ]


def test_scheduler_rejects_remote_inference_shard(monkeypatch):
    monkeypatch.setattr("orbitkv.client.data_plane._endpoint_is_local", lambda _endpoint: False)
    local_factory = MagicMock()
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", local_factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())

    with pytest.raises(ValueError, match="node-local Cache Manager"):
        OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    local_factory.assert_not_called()


def test_scheduler_fails_if_local_bootstrap_fails(monkeypatch):
    monkeypatch.setattr(
        "orbitkv.client.connection.LocalDataClient",
        MagicMock(side_effect=RuntimeError("stale socket")),
    )
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    with pytest.raises(RuntimeError, match="stale socket"):
        OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)


def test_worker_uses_only_its_tp_shard_socket(monkeypatch):
    client = MagicMock(transport="local")
    monkeypatch.setattr("orbitkv.vllm.get_tensor_model_parallel_rank", lambda: 5)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_bootstrap_sockets": ["/run/orbitkv/a.sock", "/run/orbitkv/b.sock"],
        }
    )
    connector = OrbitKVConnector(config, KVConnectorRole.WORKER)
    try:
        assert connector._ctx.data_client is client
        assert connector._worker._data_client is client
    finally:
        connector.shutdown()
    factory.assert_called_once_with("/run/orbitkv/b.sock", timeout_ms=5_000, spin_iterations=64)


def test_worker_uses_its_local_shard_when_other_shards_are_remote(monkeypatch):
    client = MagicMock(transport="local")
    monkeypatch.setattr("orbitkv.vllm.get_tensor_model_parallel_rank", lambda: 5)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    monkeypatch.setattr(
        "orbitkv.client.data_plane._endpoint_is_local",
        lambda endpoint: endpoint == "http://node-b:50055",
    )
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_endpoints": ["http://node-a:50055", "http://node-b:50055"]
        }
    )
    connector = OrbitKVConnector(config, KVConnectorRole.WORKER)
    try:
        assert connector._ctx.data_client is client
    finally:
        connector.shutdown()
    factory.assert_called_once_with("/tmp/orbitkv-50055.sock", timeout_ms=5_000, spin_iterations=64)


def test_worker_uses_selected_socket_when_other_shards_are_remote(monkeypatch):
    client = MagicMock(transport="local")
    monkeypatch.setattr("orbitkv.vllm.get_tensor_model_parallel_rank", lambda: 5)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", factory)
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    monkeypatch.setattr(
        "orbitkv.client.data_plane._endpoint_is_local",
        lambda endpoint: endpoint == "http://node-b:50055",
    )
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_endpoints": ["http://node-a:50055", "http://node-b:50055"],
            "orbitkv.tp_shard_bootstrap_sockets": ["/run/a.sock", "/run/b.sock"],
        }
    )
    connector = OrbitKVConnector(config, KVConnectorRole.WORKER)
    try:
        assert connector._ctx.data_client is client
    finally:
        connector.shutdown()
    factory.assert_called_once_with("/run/b.sock", timeout_ms=5_000, spin_iterations=64)


def test_rejects_legacy_grpc_opt_out(monkeypatch):
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    config = _vllm_config(extra_overrides={"orbitkv.local_data": False})
    with pytest.raises(ValueError, match="orbitkv.local_data=false is removed"):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER)


def test_full_prefix_prefetch_uses_local_client(monkeypatch):
    client = MagicMock(transport="local")
    monkeypatch.setattr("orbitkv.client.connection.LocalDataClient", MagicMock(return_value=client))
    monkeypatch.setattr("orbitkv.vllm.ServiceStateManager", MagicMock())
    config = _vllm_config(extra_overrides={"orbitkv.wait_for_full_prefix": True})
    connector = OrbitKVConnector(config, KVConnectorRole.SCHEDULER)
    try:
        assert connector._ctx.wait_for_full_prefix
        assert connector._ctx.engine_client is client
    finally:
        connector.shutdown()
    assert client.close.called


@pytest.mark.parametrize(
    "parallel_overrides",
    [
        {"pipeline_parallel_size": 2, "world_size": 16},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
    ],
)
def test_connector_rejects_non_tp_parallelism_across_server_shards(monkeypatch, parallel_overrides):
    with pytest.raises(ValueError, match="TP-only parallelism"):
        OrbitKVConnector(_vllm_config(**parallel_overrides), KVConnectorRole.SCHEDULER)


def test_pure_mla_collapses_storage_tp_but_stripes_saves_within_each_node():
    context = _context(tp_rank=5, is_mla=True, collapse_mla_tp=True)

    assert context.effective_tp_rank == 0
    assert context.effective_tp_size == 1
    assert context.effective_world_size == 4
    assert context.local_physical_tp_rank == 1
    assert context.local_physical_tp_size == 4


def test_scheduler_uses_common_prefix_and_exact_per_shard_leases():
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.side_effect = [
        QueryReady(3, b"first-long"),
        QueryReady(2, b"first-exact"),
    ]
    second.query_prefetch.return_value = QueryReady(2, b"second-exact")
    scheduler = SchedulerConnector(_context(), data_clients=(first, second))
    hashes = [b"h0", b"h1", b"h2"]

    ready = scheduler._count_available_block_prefix(hashes, "request")

    assert ready is not None
    assert ready.num_hit_blocks == 2
    assert ready.leases == (b"first-exact", b"second-exact")
    assert first.query_prefetch.call_args_list == [
        call(
            "instance",
            hashes,
            req_id="request",
            wait_for_full_prefix=False,
        ),
        call(
            "instance",
            hashes[:2],
            req_id="request:tp-common-2",
            wait_for_full_prefix=False,
        ),
    ]
    first.release.assert_called_once_with(b"first-long")
    second.release.assert_not_called()


def test_scheduler_releases_ready_shards_when_another_shard_is_loading():
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.return_value = QueryReady(2, b"first")
    second.query_prefetch.return_value = QueryLoading()
    scheduler = SchedulerConnector(_context(), data_clients=(first, second))

    assert scheduler._count_available_block_prefix([b"h0", b"h1"], "request") is None
    first.release.assert_called_once_with(b"first")


def test_scheduler_discards_drifted_prefetch_before_querying_new_hashes():
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.side_effect = [
        QueryLoading(),
        QueryReady(4, b"first-old"),
        QueryLoading(),
    ]
    second.query_prefetch.return_value = QueryReady(4, b"second-old")
    scheduler = SchedulerConnector(_context(), data_clients=(first, second))
    request = SimpleNamespace(
        request_id="request",
        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
        num_tokens=64,
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert scheduler.get_num_new_matched_tokens(request, 32) == (None, False)
    assert scheduler.get_num_new_matched_tokens(request, 32) == (None, False)

    original_hashes = request.block_hashes
    current_hashes = request.block_hashes[2:]
    assert first.query_prefetch.call_args_list == [
        call(
            "instance",
            original_hashes,
            req_id="request",
            wait_for_full_prefix=False,
        ),
        call(
            "instance",
            original_hashes,
            req_id="request",
            wait_for_full_prefix=False,
        ),
        call(
            "instance",
            current_hashes,
            req_id="request",
            wait_for_full_prefix=False,
        ),
    ]
    first.release.assert_called_once_with(b"first-old")
    second.release.assert_called_once_with(b"second-old")


@pytest.mark.parametrize(
    "invalid_ready",
    [
        QueryReady(3, b"too-many"),
        QueryReady(1, b""),
    ],
)
def test_scheduler_rejects_invalid_shard_query_results_without_leaking_lease(invalid_ready):
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.return_value = QueryReady(2, b"first")
    second.query_prefetch.return_value = invalid_ready
    scheduler = SchedulerConnector(_context(), data_clients=(first, second))

    with pytest.raises(RuntimeError, match="TP shard 1"):
        scheduler._count_available_block_prefix([b"h0", b"h1"], "request")

    first.release.assert_called_once_with(b"first")
    if invalid_ready.lease:
        second.release.assert_called_once_with(invalid_ready.lease)


def test_worker_selects_the_lease_for_its_local_server():
    engine_client = MagicMock()
    engine_client.start_restore.return_value = SimpleNamespace(key="restore-1")
    context = _context(
        tp_rank=5,
        device_id=1,
        namespace="namespace:tp-shard-1-of-2",
        engine_client=engine_client,
    )
    worker = WorkerConnector(context)
    worker._registered_layers = ["layer"]
    metadata = OrbitKVConnectorMetadata(
        load_intents={
            "request": LoadIntent(
                block_ids_by_group=((7,),),
                leases=(b"node-a", b"node-b"),
                num_tokens=16,
            )
        }
    )

    try:
        worker.start_load_kv(metadata, SimpleNamespace(no_compile_layers={}))
    finally:
        worker._registered_layers = []
        worker.shutdown()

    loads = engine_client.start_restore.call_args.args[4]
    assert loads == [(b"node-b", [[7]])]


def test_each_tp_shard_has_a_local_unregister_leader():
    for tp_rank in range(8):
        engine_client = MagicMock()
        engine_client.unregister_context.return_value = (True, "")
        worker = WorkerConnector(_context(tp_rank=tp_rank, engine_client=engine_client))
        worker._registered_layers = ["layer"]

        worker.unregister_context()
        worker.shutdown()

        if tp_rank in (0, 4):
            engine_client.unregister_context.assert_called_once_with("instance")
        else:
            engine_client.unregister_context.assert_not_called()
