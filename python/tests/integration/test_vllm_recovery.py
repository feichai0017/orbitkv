"""Native recovery validation and the vLLM scheduler/worker CUDA handoff."""

import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

pytestmark = pytest.mark.integration


def config():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    return SimpleNamespace(
        kv_cache_groups=(
            SimpleNamespace(
                layer_names=("model.layers.1.mamba",),
                kv_cache_spec=MambaSpec(
                    block_size=16,
                    shapes=((64,), (64,)),
                    dtypes=(torch.float32, torch.float32),
                    mamba_cache_mode="align",
                ),
            ),
            SimpleNamespace(
                layer_names=("model.layers.0.attn",),
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=32,
                    dtype=torch.float32,
                ),
            ),
        )
    )


def context(client, *, instance_id="instance", namespace="model/layout"):
    from orbitkv.vllm.config import ConnectorContext

    return ConnectorContext(
        instance_id=instance_id,
        namespace=namespace,
        block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        client=client,
        state_manager=MagicMock(),
    )


@pytest.mark.parametrize(
    ("attention", "positions", "tokens", "expected"),
    [
        (4, [1, 3], 128, 32),
        (4, [1, 3], 129, 64),
        (4, [], 129, 0),
        (4, [3], 128, 0),
        (2, [1], 129, 32),
    ],
    ids=[
        "clamp-to-earlier-checkpoint",
        "last-checkpoint",
        "missing-checkpoint",
        "clamp-has-no-checkpoint",
        "partial-attention",
    ],
)
def test_native_contract_gates_vllm_hits(attention, positions, tokens, expected):
    cache_config = config()
    from orbitkv import QueryCandidates, QueryReady
    from orbitkv.vllm.scheduler import SchedulerConnector

    client = MagicMock()
    client.query_candidates.side_effect = [
        QueryCandidates(list(range(attention))),
        QueryCandidates(positions),
    ]
    client.read_recovery.side_effect = (
        lambda *args: QueryReady((args[6] - args[5]) // 16, b"attention")
        if args[-1] == 0
        else QueryReady(1, b"state", [(args[6] - args[5]) // 16 - 1])
    )
    scheduler = SchedulerConnector(context(client), kv_cache_config=cache_config)
    try:
        req = SimpleNamespace(
            request_id="r",
            num_tokens=tokens,
            shared_prefix_boundary=0,
            block_hashes=[bytes([index]) for index in range(tokens // 16)],
        )
        assert scheduler.get_num_new_matched_tokens(req, 64) == (expected, expected > 0)
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("positions", [[1, 1], [3, 1]])
def test_native_contract_rejects_duplicate_or_unordered_evidence(positions):
    cache_config = config()
    from orbitkv import QueryCandidates
    from orbitkv.vllm.scheduler import SchedulerConnector

    client = MagicMock()
    client.query_candidates.side_effect = [
        QueryCandidates([0, 1, 2, 3]),
        QueryCandidates(positions),
    ]
    scheduler = SchedulerConnector(context(client), kv_cache_config=cache_config)
    req = SimpleNamespace(
        request_id="r",
        num_tokens=129,
        shared_prefix_boundary=0,
        block_hashes=[bytes([index]) for index in range(8)],
    )
    try:
        with pytest.raises(ValueError, match="page ends"):
            scheduler.get_num_new_matched_tokens(req, 64)
        client.read_recovery.assert_not_called()
        client.release.assert_not_called()
        assert not scheduler._pending_query_probes
    finally:
        scheduler.shutdown()


@pytest.mark.gpu
@pytest.mark.parametrize("channel_server", ["dram", "ssd"], indirect=True)
def test_complete_checkpoint_restores_through_vllm_worker(channel_server):
    torch = pytest.importorskip("torch")
    cache_config = config()
    from orbitkv import CacheManagerClient
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata
    from orbitkv.vllm.scheduler import SchedulerConnector
    from orbitkv.vllm.worker import WorkerConnector
    from tests.support.metrics import fetch_orbitkv_metrics

    client = CacheManagerClient(channel_server.bootstrap_socket)
    identity = f"recovery-{uuid.uuid4().hex}"
    ctx = context(client, instance_id=identity, namespace=identity)
    client.start_session_watcher(identity, identity, 1, 1)
    scheduler = SchedulerConnector(ctx, kv_cache_config=cache_config)
    worker = WorkerConnector(ctx, kv_cache_config=cache_config)
    kv = torch.arange(16 * 2 * 16 * 32, device="cuda", dtype=torch.float32).reshape(
        16, 2, 16, 1, 32
    )
    state = torch.arange(16 * 128, device="cuda", dtype=torch.float32).reshape(16, 128) + 10000
    expected_kv, expected_state = kv.clone(), state.clone()
    hashes = [index.to_bytes(32, "little") for index in range(8)]
    try:
        worker.register_kv_caches(
            {
                "model.layers.0.attn": kv,
                "model.layers.1.mamba": (state[:, :64], state[:, 64:]),
            }
        )
        req = SimpleNamespace(
            request_id="cold",
            num_tokens=128,
            block_hashes=hashes,
            shared_prefix_boundary=0,
        )
        deadline = time.monotonic() + 15
        while (result := scheduler.get_num_new_matched_tokens(req, 64))[0] is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result == (0, False)
        cold_metrics = fetch_orbitkv_metrics(channel_server.http_port)
        assert cold_metrics["orbitkv_cache_candidate_misses_total"] == 8
        assert cold_metrics["orbitkv_hll_total_requests"] > 0
        assert cold_metrics.get("orbitkv_ssd_prefetch_bytes_total", 0) == 0
        assert cold_metrics.get("orbitkv_query_reserved_bytes", 0) == 0
        torch.cuda.synchronize()
        ok, message = client.save(
            identity,
            0,
            0,
            0,
            [
                ("model.layers.0.attn", [1, 2, 3, 4], hashes[4:]),
                ("model.layers.1.mamba", [1, 3], [hashes[5], hashes[7]]),
            ],
        )
        assert ok, message
        if channel_server.ssd_cache_path is not None:
            deadline = time.monotonic() + 15
            while True:
                metrics = fetch_orbitkv_metrics(channel_server.http_port)
                if metrics.get("orbitkv_ssd_write_bytes_total", 0) and not any(
                    metrics.get(name, 0)
                    for name in (
                        "orbitkv_ssd_write_inflight",
                        "orbitkv_ssd_write_queue_pending",
                        "orbitkv_inflight_bytes",
                    )
                ):
                    break
                assert time.monotonic() < deadline, metrics
                time.sleep(0.01)
            response = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup", timeout=10
            )
            response.raise_for_status()
            assert response.json()["evicted_blocks"] == 6

        req = SimpleNamespace(
            request_id="restore",
            num_tokens=128,
            block_hashes=hashes,
            shared_prefix_boundary=0,
        )
        deadline = time.monotonic() + 15
        while (result := scheduler.get_num_new_matched_tokens(req, 64))[0] is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result == (32, True)
        probe = scheduler._pending_query_probes["restore"]
        assert probe.selected_boundary == 96
        assert probe.leased_blocks == 2
        assert probe.recurrent_hold.checkpoint == 1

        blocks = SimpleNamespace(
            get_block_ids=lambda: ([0, 1, 2, 3, 8, 9, 10, 11],) * 2,
            blocks=[
                [SimpleNamespace(block_hash=b"hbm" if index < 4 else None) for index in range(8)]
                for _ in range(2)
            ],
        )
        scheduler.update_state_after_alloc(req, blocks, 32)
        kv[8:12].fill_(-1)
        state[8:12].fill_(-1)
        torch.cuda.synchronize()
        worker.start_load_kv(
            OrbitKVConnectorMetadata(load_intents=scheduler._pending_load_intents),
            SimpleNamespace(no_compile_layers={}),
        )
        while worker.get_finished(set())[1] != {"restore"}:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        torch.cuda.synchronize()
        assert torch.equal(kv[8:10], expected_kv[1:3])
        assert torch.equal(state[9], expected_state[1])  # Both conv and temporal tensors.
        assert (kv[10:12] == -1).all()  # Extra leased pages have no destination.
        assert (state[[8, 10, 11]] == -1).all()
        assert not worker._pending_loads
        metrics = fetch_orbitkv_metrics(channel_server.http_port)
        assert metrics["orbitkv_cache_candidate_hits_total"] == 6
        assert metrics["orbitkv_cache_candidate_misses_total"] == 10
        # One cold and one warm discovery, regardless of HLL window count.
        assert (
            metrics["orbitkv_hll_total_requests"] == 2 * cold_metrics["orbitkv_hll_total_requests"]
        )
        if channel_server.ssd_cache_path is not None:
            page_bytes = kv[0].numel() * kv.element_size()
            state_bytes = state[0].numel() * state.element_size()
            assert (
                metrics.get("orbitkv_ssd_prefetch_bytes_total", 0) == 2 * page_bytes + state_bytes
            )
            assert metrics.get("orbitkv_load_bytes_total", 0) == 2 * page_bytes + state_bytes
            assert metrics.get("orbitkv_query_reserved_bytes", 0) == 0
    finally:
        scheduler.shutdown()
        worker.shutdown()
        client.close()
