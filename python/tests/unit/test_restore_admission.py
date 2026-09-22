"""A deferred lookup cannot strand a restore behind an allocation failure."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv import BlockHashes  # noqa: E402
from orbitkv.vllm.config import ConnectorContext  # noqa: E402
from orbitkv.vllm.scheduler import SchedulerConnector  # noqa: E402
from orbitkv.vllm.tp_shards import ShardedQueryReady  # noqa: E402


def request(req_id, tokens=32):
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=tokens,
        num_prompt_tokens=tokens,
        block_hashes=[f"{req_id}-{i}".encode() for i in range(tokens // 16)],
    )


def test_enqueue_warms_only_legal_missing_prefix_without_creating_a_load(monkeypatch):
    monkeypatch.setenv("ORBITKV_QUEUE_WARMUP", "1")
    client = MagicMock()
    scheduler = SchedulerConnector(
        ConnectorContext(
            instance_id="warm",
            namespace="warm",
            block_size=16,
            tp_size=1,
            world_size=1,
            tp_rank=0,
            device_id=0,
            client=client,
            state_manager=MagicMock(),
        )
    )
    pool = MagicMock()
    pool.get_cached_block.side_effect = [[object()], None]
    scheduler.bind_gpu_block_pool(pool)
    req = request("queued", 64)
    scheduler.on_new_request(req)
    client.warm_prefix.assert_called_once_with(
        "warm", BlockHashes([b"queued-1", b"queued-2", b"queued-3"]), "queued"
    )
    client.query_prefetch.assert_not_called()
    assert not scheduler._pending_load_intents
    assert not scheduler._pending_query_probes
    assert not scheduler._restores_awaiting_compute
    assert scheduler.request_finished(req, ()) == (False, None)
    client.cancel_query.assert_called_once_with("warm", "queued")
    assert not scheduler._queued_at
    pool.reset_mock()
    client.reset_mock()
    monkeypatch.delenv("ORBITKV_QUEUE_WARMUP")
    scheduler.on_new_request(req)
    pool.get_cached_block.assert_not_called()
    client.warm_prefix.assert_not_called()

    monkeypatch.setenv("ORBITKV_QUEUE_WARMUP", "1")
    pool.get_cached_block.side_effect = None
    pool.get_cached_block.return_value = None
    client.warm_prefix.side_effect = RuntimeError("manager unavailable")
    scheduler.on_new_request(req)
    client.warm_prefix.assert_called_once()
    assert req.request_id in scheduler._queued_at
    assert not scheduler._pending_load_intents


def step(tokens=0):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="restore", block_ids=([1, 2],), num_computed_tokens=31)
        ]
        if tokens
        else [],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"restore": tokens} if tokens else {},
        preempted_req_ids=set(),
    )


@pytest.fixture
def restoring():
    scheduler = SchedulerConnector(
        ConnectorContext(
            instance_id="test",
            namespace="test",
            block_size=16,
            tp_size=1,
            world_size=1,
            tp_rank=0,
            device_id=0,
            client=MagicMock(),
            state_manager=MagicMock(),
        )
    )
    scheduler._tp_shard_client.query = MagicMock(return_value=ShardedQueryReady(2, (b"hold",)))
    restoring = request("restore")
    assert scheduler.get_num_new_matched_tokens(restoring, 0) == (31, True)
    blocks = SimpleNamespace(
        blocks=([SimpleNamespace(block_hash=None) for _ in range(2)],),
        get_block_ids=lambda: ([1, 2],),
    )
    scheduler.update_state_after_alloc(restoring, blocks, 31)
    assert "restore" in scheduler.build_connector_meta(step()).load_intents
    yield scheduler, restoring
    scheduler.shutdown()


@pytest.mark.parametrize("kind", ["hit", "miss", "short"])
def test_lookup_waits_for_compute_but_prefetch_can_complete(restoring, kind):
    scheduler, _ = restoring
    waiting = request("waiting", 7 if kind == "short" else 32)
    ready = ShardedQueryReady(2, (b"next",)) if kind == "hit" else ShardedQueryReady(0, (b"",))
    scheduler._tp_shard_client.query.return_value = ready

    assert scheduler.get_num_new_matched_tokens(waiting, 0) == (None, False)
    calls = scheduler._tp_shard_client.query.call_count
    assert calls == (1 if kind == "short" else 2)

    scheduler.update_connector_output(
        SimpleNamespace(finished_sending=set(), finished_recving={"restore"})
    )
    scheduler.build_connector_meta(step())
    assert scheduler.get_num_new_matched_tokens(waiting, 0) == (None, False)
    assert scheduler._tp_shard_client.query.call_count == calls

    scheduler.build_connector_meta(step(tokens=1))
    expected = (31, True) if kind == "hit" else (0, False)
    assert scheduler.get_num_new_matched_tokens(waiting, 0) == expected
    assert scheduler._tp_shard_client.query.call_count == calls


@pytest.mark.parametrize("pending_save", [False, True])
def test_aborted_restore_unblocks_admission_without_releasing_save_hold(restoring, pending_save):
    scheduler, restored = restoring
    if pending_save:
        scheduler._pending_saves.add(restored.request_id)
    assert scheduler.request_finished(restored, ([1, 2],)) == (pending_save, None)
    assert scheduler.get_num_new_matched_tokens(request("short", 7), 0) == (0, False)
    assert (restored.request_id in scheduler._held_requests) == pending_save
