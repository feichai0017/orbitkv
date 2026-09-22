"""Exercise the pinned SGLang admission hook with controlled backing completion."""

import threading
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def linker():
    pytest.importorskip("sglang")
    from orbitkv.sglang.linker import OrbitKVLinker

    result = object.__new__(OrbitKVLinker)
    from orbitkv import RecoveryContract

    result.instance_id = "admission"
    result.namespace = "test-model-state"
    result.layout = SimpleNamespace(
        pools={"kv": SimpleNamespace(group_id=0, kind="attention", window=0)}
    )
    result.recovery = RecoveryContract(result.namespace, 64, [(0, "attention", 0)])
    result._origins = dict.fromkeys(["slow", "req", "shared", "queued", "other"], 0)
    result._load_boundaries = {}
    result.page_size = 64
    result.client = MagicMock()
    result._lookups = {}
    result._pending_queries = {}
    result._expired_queries = set()
    result._queued_loads = {}
    return result


def request(rid, token_count=256):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(rid, "", array("q", [1] * token_count), SamplingParams())
    req._refresh_fill_ids()
    return req


def transfer(keys):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    return [SimpleNamespace(name=PoolName.KV, keys=keys)]


@pytest.mark.parametrize("preparation", [False, True], ids=["warming", "owned"])
def test_enqueue_uses_the_same_salted_storage_keys_without_triggering_a_load(
    linker, monkeypatch, preparation
):
    from orbitkv import BlockHashes

    setting = "ORBITKV_PREPARE_REQUESTS" if preparation else "ORBITKV_QUEUE_WARMUP"
    monkeypatch.setenv(setting, "1")
    import torch
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinkerWrapper
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    from orbitkv.sglang.admission import abort_request, enqueue_request

    req = request("queued", 257)
    req.extra_key, req.cache_salt = "tenant", "salt"
    req.return_logprob, req.logprob_start_len = True, 192
    key = RadixKey(req.origin_input_ids, extra_key="tenant", cache_salt="salt", limit=192)
    hashes = get_storage_hash_str(key, page_size=64)
    cache = SimpleNamespace(page_size=64, get_last_hash_value=lambda _: hashes[0])
    wrapper = object.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache, wrapper.cache_linker = cache, linker
    cache.linker = wrapper
    cache.match_prefix = MagicMock(
        return_value=SimpleNamespace(
            device_indices=torch.arange(64),
            last_device_node=object(),
        )
    )
    scheduler = SimpleNamespace(tree_cache=cache, waiting_queue=[])

    def accepted(scheduler, req):
        scheduler.waiting_queue.append(req)

    enqueue_request(accepted, scheduler, req)
    batch = BlockHashes(linker._hashes(hashes[1:]))
    if preparation:
        linker.client.prepare_recovery.assert_called_once_with(
            "admission", batch, req.rid, linker.recovery, linker.namespace, 64, 192, 0
        )
    else:
        linker.client.warm_prefix.assert_called_once_with("admission", batch, req.rid)
    submit = linker.client.prepare_recovery if preparation else linker.client.warm_prefix
    assert cache.match_prefix.call_args.args[0].req is None
    assert not linker._lookups and not linker._queued_loads
    abort_request(MagicMock(), scheduler, req)
    linker.client.cancel_query.assert_called_once_with("admission", req.rid, group_id=0)

    linker.client.reset_mock()
    scheduler.waiting_queue.clear()
    enqueue_request(MagicMock(), scheduler, req)
    submit.assert_not_called()

    submit.side_effect = RuntimeError("manager unavailable")
    enqueue_request(accepted, scheduler, req)
    assert scheduler.waiting_queue[-1] is req
    submit.assert_called_once()
    linker.client.reset_mock()
    cache.match_prefix.reset_mock()
    monkeypatch.delenv(setting)
    enqueue_request(accepted, scheduler, req)
    cache.match_prefix.assert_not_called()
    submit.assert_not_called()
    if preparation:
        monkeypatch.setenv(setting, "1")
        scheduler.waiting_queue[:] = [object()] * 4
        enqueue_request(accepted, scheduler, req)
        submit.assert_not_called()


def test_first_use_is_observed_once_even_when_the_first_layer_wait_repeats(monkeypatch):
    from orbitkv.sglang.linker import _LayerDoneCounter

    trace = MagicMock()
    monkeypatch.setattr("orbitkv.sglang.linker.trace_transfer", trace)
    counter = _LayerDoneCounter(2)
    index = counter.update_producer()
    counter.request_ids[index] = ["restored"]
    counter.complete(index)
    counter.set_consumer(index)
    counter.wait_until(0)
    counter.wait_until(0)
    counter.wait_until(1)
    trace.assert_called_once_with("first_use", "restored", engine="sglang")


def test_graph_consumer_waits_without_any_python_layer_access():
    from orbitkv.sglang.linker import _LayerDoneCounter

    counter = _LayerDoneCounter(2)
    index = counter.update_producer()
    entered, executing = threading.Event(), threading.Event()

    def replay():
        entered.set()
        counter.set_consumer(index)
        executing.set()

    thread = threading.Thread(target=replay, daemon=True)
    thread.start()
    assert entered.wait(timeout=1)
    try:
        assert not executing.wait(timeout=0.05)
    finally:
        counter.complete(index)
        thread.join(timeout=1)
    assert executing.is_set()
    assert index not in counter._futures
    assert not counter.request_ids and not counter._futures


def test_pending_query_defers_only_its_request_and_preserves_ready_lease(linker):
    from sglang.srt.managers.schedule_policy import AddReqResult

    from orbitkv import QueryLoading, QueryReady
    from orbitkv.sglang.admission import admit_request

    linker.client.query_prefetch.side_effect = [QueryLoading(), QueryReady(2, b"lease")]
    original = MagicMock(return_value=AddReqResult.CONTINUE)
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker), _all_reduce_attn_groups=MagicMock()
    )
    adder = SimpleNamespace(tree_cache=cache)
    req = request("slow")
    keys = transfer(["a", "b"])
    assert linker.lookup(req.rid, keys) == []
    assert admit_request(original, adder, req) == AddReqResult.CONTINUE
    original.assert_not_called()
    unrelated = request("other")
    admit_request(original, adder, unrelated)
    original.assert_called_once_with(adder, unrelated)

    assert linker.lookup(req.rid, keys) == [1, 2]
    assert linker.lookup(req.rid, keys) == [1, 2]
    assert linker.client.query_prefetch.call_count == 2
    linker.client.release.assert_not_called()
    admit_request(original, adder, req)
    assert original.call_count == 2
    linker.cancel_queued_load(req.rid)
    linker.client.release.assert_called_once_with(b"lease")
    assert not linker._lookups


@pytest.mark.parametrize("kind", ["recurrent", "window"])
def test_hybrid_discovery_preserves_earlier_boundaries_before_reading(linker, kind):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    from orbitkv import BlockHashes, QueryCandidates, QueryLoading, QueryReady, RecoveryContract

    auxiliary = PoolName.MAMBA if kind == "recurrent" else PoolName.SWA
    linker.layout.pools[auxiliary] = SimpleNamespace(
        group_id=1, kind=kind, window=128 if kind == "window" else 0
    )
    linker.recovery = RecoveryContract(
        linker.namespace, 64, [(0, "attention", 0), (1, kind, 128 if kind == "window" else 0)]
    )
    linker._origins["req"] = 64
    keys = ["a", "b", "c", "d"]
    transfers = [SimpleNamespace(name=name, keys=keys) for name in linker.layout.pools]
    linker.client.query_candidates.side_effect = [
        QueryLoading(),
        QueryCandidates([0, 1]),
        QueryCandidates([0, 1, 2]),
    ]
    assert linker.lookup("req", transfers) == []
    assert linker.lookup("req", transfers) == [1, 2]
    assert linker._lookups["req"].boundaries == (128, 192)
    linker.client.read_recovery.assert_not_called()
    linker.client.query_prefetch.assert_not_called()
    assert all(
        entry.args[1] == BlockHashes(linker._hashes(keys))
        for entry in linker.client.query_candidates.call_args_list
    )
    positions = [1] if kind == "recurrent" else [0, 1]
    linker.client.read_recovery.side_effect = [
        QueryReady(2, b"attention", [0, 1]),
        QueryLoading(),
        QueryReady(len(positions), b"state", positions),
    ]
    assert linker.prepare_recovery("req", 192) == 0
    linker.client.release.assert_not_called()
    assert linker.prepare_recovery("req", 192) == 1
    assert [entry.args[-1] for entry in linker.client.read_recovery.call_args_list] == [0, 1, 1]
    linker.cancel_query("req")
    assert sorted(entry.args[0] for entry in linker.client.release.call_args_list) == [
        b"attention",
        b"state",
    ]


def test_invalid_hybrid_candidates_never_materialize_state(linker):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    from orbitkv import QueryCandidates, RecoveryContract

    linker.layout.pools[PoolName.MAMBA] = SimpleNamespace(group_id=1, kind="recurrent", window=0)
    linker.recovery = RecoveryContract(
        linker.namespace, 64, [(0, "attention", 0), (1, "recurrent", 0)]
    )
    linker.client.query_candidates.side_effect = [QueryCandidates([0]), QueryCandidates([2])]
    with pytest.raises(ValueError, match="page ends"):
        linker.lookup(
            "req", [SimpleNamespace(name=name, keys=["a", "b"]) for name in linker.layout.pools]
        )
    assert not linker._lookups and not linker._pending_queries
    linker.client.read_recovery.assert_not_called()
    linker.client.release.assert_not_called()


def test_changed_keys_cancel_old_query_and_deadline_stops_restarting_io(linker):
    from orbitkv import QueryLoading

    linker.client.query_prefetch.return_value = QueryLoading()
    assert linker.lookup("req", transfer(["old"])) == []
    assert linker.lookup("req", transfer(["new"])) == []
    linker.client.cancel_query.assert_called_once_with("admission", "req", group_id=0)
    linker._QUERY_WAIT_SECONDS = 0
    assert linker.lookup("req", transfer(["new"])) == []
    assert linker.query_state("req") == 2
    assert linker.lookup("req", transfer(["new"])) == []
    assert linker.client.query_prefetch.call_count == 2
    assert linker.client.cancel_query.call_count == 2
    linker.cancel_queued_load("req")
    assert linker.query_state("req") == 0


@pytest.mark.parametrize(
    "tokens,prefix,logprob_start,pending,wait",
    [
        (256, 128, -1, True, True),
        (256, 192, -1, True, False),
        (257, 192, -1, True, True),
        (256, 128, 128, True, False),
        (256, 192, -1, False, False),
    ],
)
def test_resident_prefix_retires_unused_external_query(
    linker, tokens, prefix, logprob_start, pending, wait
):
    from orbitkv import QueryLoading, QueryReady
    from orbitkv.sglang.admission import admit_request

    req = request("shared", tokens)
    req.prefix_indices = list(range(prefix))
    req.return_logprob = logprob_start >= 0
    req.logprob_start_len = logprob_start
    linker.client.query_prefetch.return_value = (
        QueryLoading() if pending else QueryReady(3, b"lease")
    )
    linker.lookup(req.rid, transfer(["a", "b", "c"]))
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker), _all_reduce_attn_groups=MagicMock()
    )
    original = MagicMock()
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    assert original.call_count == int(not wait)
    assert linker.query_state(req.rid) == int(wait)
    if not wait:
        assert req.rid not in linker._lookups
        if pending:
            linker.client.cancel_query.assert_called_once_with("admission", req.rid, group_id=0)
        else:
            linker.client.release.assert_called_once_with(b"lease")


def test_admission_expires_pending_work_without_another_lookup(linker):
    from orbitkv import QueryLoading
    from orbitkv.sglang.admission import admit_request

    req = request("unpolled")
    linker._origins[req.rid] = 0
    linker.client.query_prefetch.return_value = QueryLoading()
    linker.lookup(req.rid, transfer(["a", "b", "c"]))
    linker._QUERY_WAIT_SECONDS = 0
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker), _all_reduce_attn_groups=MagicMock()
    )
    original = MagicMock()
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    original.assert_called_once()
    linker.client.cancel_query.assert_called_once_with("admission", req.rid, group_id=0)
    assert linker.query_state(req.rid) == 2
    assert linker.lookup(req.rid, transfer(["a", "b", "c"])) == []
    linker.client.query_prefetch.assert_called_once()


@pytest.mark.parametrize("peer_state", [1, 2])
def test_attention_ranks_share_wait_and_expiration_decisions(linker, peer_state):
    from orbitkv import QueryReady
    from orbitkv.sglang.admission import admit_request

    linker.client.query_prefetch.return_value = QueryReady(1, b"lease")
    linker.lookup("req", transfer(["key"]))
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker),
        _all_reduce_attn_groups=lambda state, op: state.fill_(peer_state),
    )
    original = MagicMock()
    req = request("req")
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    if peer_state == 1:
        original.assert_not_called()
        linker.client.release.assert_not_called()
    else:
        original.assert_called_once()
        linker.client.release.assert_called_once_with(b"lease")
        assert linker.query_state("req") == 2
