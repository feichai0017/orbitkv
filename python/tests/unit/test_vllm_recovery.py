"""vLLM evidence translation and ownership around the native recovery validator.

The validator is mocked here; its rules are exercised by Rust and native
integration gates, without requiring a compiled extension for default tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.orbitkv import QueryLoading, QueryReady  # noqa: E402
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


def test_absolute_evidence_clamp_and_handoff_keep_the_original_leases(hybrid):
    scheduler, (client,), validator = hybrid()
    client.query_prefetch.side_effect = [
        QueryReady(4, b"attention"),
        QueryReady(2, b"state", [1, 3]),
    ]
    validator.restorable_boundaries.return_value = [96, 128]
    req = request(tokens=128)

    # The engine owns 64 tokens; its final-token clamp excludes boundary 128.
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    validator.restorable_boundaries.assert_called_once_with(
        "model/layout", 64, 128, [(0, [80, 96, 112, 128]), (1, [96, 128])]
    )
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    assert client.query_prefetch.call_count == 2
    scheduler.update_state_after_alloc(req, allocations(), 32)
    intent = scheduler._pending_load_intents["r"]
    assert intent.block_ids_by_group == ((None, 25, None, None), (44, 45, None, None))
    assert intent.recurrent_hold.checkpoint == 1
    assert intent.recurrent_hold.leases == ((b"state",),)
    assert intent.leases == (b"attention",)
    scheduler._cleanup_request("r")
    client.release.assert_not_called()  # Worker/transfer now owns these leases.


@pytest.mark.parametrize("tokens", [31, 48])
def test_allocation_cannot_change_the_validated_checkpoint(hybrid, tokens):
    scheduler, (client,), validator = hybrid()
    client.query_prefetch.side_effect = [
        QueryReady(4, b"attention"),
        QueryReady(1, b"state", [1]),
    ]
    validator.restorable_boundaries.return_value = [96]
    req = request()
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    with pytest.raises(RuntimeError, match="allocation changed the recovery boundary"):
        scheduler.update_state_after_alloc(req, allocations(), tokens)
    assert client.release.call_args_list == [call(b"attention"), call(b"state")]
    assert not scheduler._pending_load_intents


def test_ready_groups_remain_owned_while_another_checkpoint_loads(hybrid):
    scheduler, (client,), validator = hybrid(groups=2)
    client.query_prefetch.side_effect = [
        QueryReady(4, b"attention"),
        QueryReady(2, b"state-1", [1, 3]),
        QueryLoading(),
        QueryReady(1, b"state-2", [1]),
    ]
    validator.restorable_boundaries.return_value = [96]
    req = request()

    assert scheduler.get_num_new_matched_tokens(req, 64) == (None, False)
    assert scheduler._prefetch_tracker.pending_prefetches == 1
    client.release.assert_not_called()
    assert scheduler.get_num_new_matched_tokens(req, 64) == (32, True)
    assert scheduler._prefetch_tracker.pending_prefetches == 0
    assert [entry.kwargs.get("group_id", 0) for entry in client.query_prefetch.call_args_list] == [
        0,
        1,
        2,
        2,
    ]
    # Checkpoint queries stop at the available attention prefix, not the whole prompt.
    assert client.query_prefetch.call_args_list[1].args[1] == req.block_hashes[4:8]
    scheduler._cleanup_request("r")
    assert client.release.call_args_list == [call(b"attention"), call(b"state-1"), call(b"state-2")]


@pytest.mark.parametrize("retire", ["cancel", "shutdown", "drift", "expiry"])
def test_retirement_releases_ready_groups_and_cancels_pending_group(hybrid, retire):
    scheduler, (client,), _ = hybrid(groups=2)
    client.query_prefetch.side_effect = [
        QueryReady(4, b"attention"),
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


def test_rank_validation_intersects_boundaries_instead_of_minimizing_maxima(hybrid):
    scheduler, clients, validator = hybrid(shards=2)
    for client, positions in zip(clients, ([1, 3], [0, 2]), strict=True):
        client.query_prefetch.side_effect = [
            QueryReady(4, b"attention"),
            QueryReady(2, b"state", positions),
        ]
    validator.restorable_boundaries.side_effect = [[96, 128], [80, 112]]
    assert scheduler.get_num_new_matched_tokens(request(), 64) == (0, False)
    assert validator.restorable_boundaries.call_count == 2
    for client in clients:
        assert client.release.call_args_list == [call(b"attention"), call(b"state")]


@pytest.mark.parametrize("failure", ["validator", "position", "count", "missing-lease"])
def test_invalid_evidence_retires_all_acquired_leases(hybrid, failure):
    scheduler, (client,), validator = hybrid()
    membership = {
        "validator": QueryReady(2, b"state", [1, 1]),
        "position": QueryReady(1, b"state", [4]),
        "count": QueryReady(2, b"state", [1]),
        "missing-lease": QueryReady(1, b"", [1]),
    }[failure]
    client.query_prefetch.side_effect = [QueryReady(4, b"attention"), membership]
    validator.restorable_boundaries.side_effect = ValueError("invalid coverage")
    with pytest.raises((RuntimeError, ValueError)):
        scheduler.get_num_new_matched_tokens(request(), 64)
    released = [args.args[0] for args in client.release.call_args_list]
    assert sorted(released) == sorted([b"attention"] + ([b"state"] if membership.lease else []))
    assert not scheduler._pending_query_probes


def test_pending_checkpoint_shard_does_not_leak_other_shard_result(hybrid):
    scheduler, (first, second), _ = hybrid(shards=2)
    first.query_prefetch.side_effect = [
        QueryReady(4, b"attention-0"),
        QueryReady(1, b"state-0", [1]),
    ]
    second.query_prefetch.side_effect = [QueryReady(4, b"attention-1"), QueryLoading()]
    assert scheduler.get_num_new_matched_tokens(request(), 64) == (None, False)
    first.release.assert_called_once_with(b"state-0")
    second.release.assert_not_called()
    scheduler.shutdown()
    assert first.release.call_args_list == [call(b"state-0"), call(b"attention-0")]
    second.release.assert_called_once_with(b"attention-1")
    second.cancel_query.assert_any_call("instance", "r", group_id=1)
