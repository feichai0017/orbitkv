"""vLLM evidence translation and ownership around the native recovery validator.

The validator is mocked here; its rules are exercised by Rust and native
integration gates, without requiring a compiled extension for default tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.orbitkv import BlockHashes, QueryCandidates, QueryLoading, QueryReady  # noqa: E402
from orbitkv.vllm.config import ConnectorContext, TpShardTopology  # noqa: E402
from orbitkv.vllm.scheduler import SchedulerConnector  # noqa: E402

from .test_cache_group_layout import _config, _full_attention, _group, _mamba  # noqa: E402


@pytest.fixture
def hybrid(monkeypatch):
    validator = MagicMock()
    factory = MagicMock(return_value=validator)
    monkeypatch.setattr("orbitkv.RecoveryContract", factory)
    schedulers = []

    def make(*, shards=1, groups=1):
        current = tuple(MagicMock() for _ in range(shards))
        for client in current:
            client.query_candidates.side_effect = lambda *args, **kw: QueryCandidates(
                list(range(4)) if kw["group_id"] == 0 else [1, 3]
            )
        validator.select_boundary.return_value = 96
        topology = TpShardTopology.from_config(
            default_endpoint="http://127.0.0.1:50055",
            configured_endpoints=[f"http://127.0.0.1:{50055 + index}" for index in range(shards)],
            global_tp_size=shards,
            global_world_size=shards,
        )
        context = ConnectorContext(
            instance_id="instance",
            namespace="model/layout",
            block_size=16,
            tp_size=shards,
            world_size=shards,
            tp_rank=0,
            device_id=0,
            client=current[0],
            state_manager=MagicMock(),
            tp_shards=topology,
        )
        config = _config(
            _group("state.0", _mamba()),
            _group("attention", _full_attention()),
            *(_group(f"state.{index}", _mamba()) for index in range(1, groups)),
        )
        scheduler = SchedulerConnector(context, clients=current, kv_cache_config=config)
        validator.required_ranges.return_value = [
            (0, 64, 96),
            *((index + 1, 80, 96) for index in range(groups)),
        ]
        schedulers.append(scheduler)
        factory.assert_called_with(
            "model/layout",
            16,
            ((0, "attention", 0), *((index + 1, "recurrent", 0) for index in range(groups))),
        )
        return scheduler, current, validator

    yield make
    for scheduler in schedulers:
        scheduler.shutdown()


def request(tokens=161):
    return SimpleNamespace(
        request_id="r",
        num_tokens=tokens,
        shared_prefix_boundary=0,
        block_hashes=[bytes([index]) for index in range(tokens // 16)],
    )


def allocations():
    return SimpleNamespace(
        get_block_ids=lambda: (list(range(20, 30)), list(range(40, 50))),
        blocks=[
            [SimpleNamespace(block_hash=b"local" if index < 4 else None) for index in range(10)]
            for _ in range(2)
        ],
    )


def test_engine_limit_selects_earlier_checkpoint_before_payload_reads(hybrid):
    scheduler, (client,), validator = hybrid()
    validator.select_boundary.return_value = 96
    client.read_recovery.side_effect = [
        QueryReady(2, b"attention", [0, 1]),
        QueryReady(1, b"state", [1]),
    ]
    req = request(tokens=128)
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    validator.select_boundary.assert_called_once_with(
        "model/layout", 64, 128, [[(0, (0, 1, 2, 3)), (1, (1, 3))]], 127
    )
    assert all(entry.args[5:7] == (64, 96) for entry in client.read_recovery.call_args_list)
    assert all(
        entry.args[1] == BlockHashes(req.block_hashes[4:])
        for entry in client.read_recovery.call_args_list
    )
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    assert client.read_recovery.call_count == 2
    client.query_prefetch.assert_not_called()
    scheduler.update_state_after_alloc(req, allocations(), 32)
    validator.required_ranges.assert_called_once_with("model/layout", 64, 96)
    intent = scheduler._pending_load_intents["r"]
    assert intent.block_ids_by_group == ((None, 25), (44, 45))
    assert intent.recurrent_hold.checkpoint == 1
    assert intent.recurrent_hold.hit_positions == (((1,),),)
    assert intent.recurrent_hold.leases == ((b"state",),)
    assert intent.leases == (b"attention",)
    scheduler._cleanup_request("r")
    client.release.assert_not_called()


@pytest.mark.parametrize("tokens", [31, 48])
def test_allocation_cannot_change_the_leased_checkpoint(hybrid, tokens):
    scheduler, (client,), _ = hybrid()
    client.read_recovery.side_effect = [
        QueryReady(2, b"attention", [0, 1]),
        QueryReady(1, b"state", [1]),
    ]
    req = request()
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    with pytest.raises(
        RuntimeError, match="allocation changed the recovery boundary|load block mismatch"
    ):
        scheduler.update_state_after_alloc(req, allocations(), tokens)
    assert client.release.call_args_list == [call(b"attention"), call(b"state")]
    assert not scheduler._pending_load_intents


def test_ready_groups_remain_owned_while_another_checkpoint_loads(hybrid):
    scheduler, (client,), _ = hybrid(groups=2)
    client.read_recovery.side_effect = [
        QueryReady(2, b"attention", [0, 1]),
        QueryReady(1, b"state-1", [1]),
        QueryLoading(),
        QueryReady(1, b"state-2", [1]),
    ]
    req = request()
    assert scheduler.get_num_new_matched_tokens(req, 64) == (None, False)
    assert scheduler._prefetch_tracker.pending_prefetches == 1
    client.release.assert_not_called()
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    assert scheduler._prefetch_tracker.pending_prefetches == 0
    assert [entry.args[-1] for entry in client.read_recovery.call_args_list] == [0, 1, 2, 2]
    scheduler._cleanup_request("r")
    assert client.release.call_args_list == [call(b"attention"), call(b"state-1"), call(b"state-2")]


@pytest.mark.parametrize("retire", ["cancel", "shutdown", "drift", "expiry"])
def test_retirement_releases_ready_groups_and_cancels_pending_group(hybrid, retire):
    scheduler, (client,), _ = hybrid(groups=2)
    client.read_recovery.side_effect = [
        QueryReady(2, b"attention", [0, 1]),
        QueryReady(1, b"state-1", [1]),
        QueryLoading(),
        QueryReady(0, b""),
    ]
    req = request()
    assert scheduler.get_num_new_matched_tokens(req, 64) == (None, False)
    if retire == "cancel":
        scheduler._cleanup_request("r")
    elif retire == "shutdown":
        scheduler.shutdown()
    elif retire == "drift":
        assert scheduler.get_num_new_matched_tokens(req, 80) == (0, False)
    else:
        scheduler._pending_query_probes["r"].started_at -= 6
        assert scheduler.get_num_new_matched_tokens(req, 64) == (0, False)
    assert not scheduler._pending_query_probes
    assert scheduler._prefetch_tracker.pending_prefetches == 0
    assert client.release.call_args_list == [call(b"attention"), call(b"state-1")]
    assert call("instance", "r", group_id=2) in client.cancel_query.call_args_list


def test_disjoint_rank_candidates_never_read_payloads(hybrid):
    scheduler, clients, validator = hybrid(shards=2)
    for client, positions in zip(clients, ([1, 3], [0, 2]), strict=True):
        client.query_candidates.side_effect = [
            QueryCandidates([0, 1, 2, 3]),
            QueryCandidates(positions),
        ]
    validator.select_boundary.return_value = None
    assert scheduler.get_num_new_matched_tokens(request(), 64) == (0, False)
    validator.select_boundary.assert_called_once_with(
        "model/layout",
        64,
        160,
        [
            [(0, (0, 1, 2, 3)), (1, (1, 3))],
            [(0, (0, 1, 2, 3)), (1, (0, 2))],
        ],
        160,
    )
    for client in clients:
        client.read_recovery.assert_not_called()
        client.release.assert_not_called()


def test_invalid_candidate_evidence_retires_query_before_read(hybrid):
    scheduler, (client,), validator = hybrid()
    validator.select_boundary.side_effect = ValueError("invalid coverage")
    with pytest.raises(ValueError, match="invalid coverage"):
        scheduler.get_num_new_matched_tokens(request(), 64)
    client.read_recovery.assert_not_called()
    assert not scheduler._pending_query_probes


def test_eviction_between_discovery_and_read_falls_back_and_releases_attention(hybrid):
    scheduler, (client,), _ = hybrid()
    client.read_recovery.side_effect = [QueryReady(2, b"attention", [0, 1]), QueryReady(0, b"")]
    assert scheduler.get_num_new_matched_tokens(request(), 64) == (0, False)
    client.release.assert_called_once_with(b"attention")
    assert not scheduler._pending_query_probes


def test_pending_checkpoint_shard_does_not_leak_other_shard_result(hybrid):
    scheduler, (first, second), _ = hybrid(shards=2)
    first.read_recovery.side_effect = [
        QueryReady(2, b"attention-0", [0, 1]),
        QueryReady(1, b"state-0", [1]),
    ]
    second.read_recovery.side_effect = [QueryReady(2, b"attention-1", [0, 1]), QueryLoading()]
    assert scheduler.get_num_new_matched_tokens(request(), 64) == (None, False)
    first.release.assert_called_once_with(b"state-0")
    second.release.assert_not_called()
    scheduler.shutdown()
    assert first.release.call_args_list == [call(b"state-0"), call(b"attention-0")]
    second.release.assert_called_once_with(b"attention-1")
    second.cancel_query.assert_any_call("instance", "r", group_id=1)


def test_changed_engine_limit_cannot_reuse_a_different_checkpoint(hybrid):
    scheduler, (client,), validator = hybrid()
    validator.select_boundary.return_value = 128
    client.read_recovery.side_effect = [
        QueryReady(4, b"attention", [0, 1, 2, 3]),
        QueryReady(1, b"state", [3]),
    ]
    req = request(tokens=129)
    assert scheduler.get_num_new_matched_tokens(req, 64) == (64, True)
    # Same hash batch, changed engine budget: checkpoint 128 cannot prove 96.
    req.num_tokens = 128
    assert scheduler.get_num_new_matched_tokens(req, 64) == (0, False)
    assert client.read_recovery.call_count == 2
    assert client.release.call_args_list == [call(b"attention"), call(b"state")]
