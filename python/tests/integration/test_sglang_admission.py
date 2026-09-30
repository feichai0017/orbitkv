"""Exercise the pinned SGLang admission hook with controlled backing completion."""

import threading
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def linker(monkeypatch):
    pytest.importorskip("sglang")
    from orbitkv.sglang.linker import OrbitKVLinker

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_memory",
        lambda: SimpleNamespace(radix_cache_backend="orbitkv"),
    )
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
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    from orbitkv.sglang.admission import enqueue_request

    req = request("queued", 257)
    req.extra_key, req.cache_salt = "tenant", "salt"
    req.return_logprob, req.logprob_start_len = True, 192
    key = RadixKey(req.origin_input_ids, extra_key="tenant", cache_salt="salt", limit=192)
    hashes = get_storage_hash_str(key, page_size=64)
    cache = SimpleNamespace(page_size=64, get_last_hash_value=lambda _: hashes[0])
    wrapper = object.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache, wrapper.cache_linker = cache, linker
    wrapper.restore_from_store = True
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


@pytest.mark.parametrize("outcome", ["SUCCESS", "ABORT"])
def test_native_cache_finish_cancels_aborted_query_without_scheduler_hook(linker, outcome):
    from sglang.srt.mem_cache.base_prefix_cache import (
        CacheRequestHandle,
        CacheRequestOutcome,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    from orbitkv.sglang.recovery import RecoveryLinkerWrapper

    cache = object.__new__(UnifiedRadixCache)
    wrapper = object.__new__(RecoveryLinkerWrapper)
    wrapper.cache, wrapper.cache_linker = cache, linker
    wrapper.hit_markers = {"queued": object()}
    cache.linker = wrapper
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache.storage_prefetch_retries = MagicMock()
    cache.buffer_pipeline = None
    cache.discard_storage_prefetch_accounting = MagicMock()
    cache.ongoing_prefetch = {}
    cache.finish(CacheRequestHandle("queued", 0), CacheRequestOutcome[outcome])
    if outcome == "ABORT":
        linker.client.cancel_query.assert_called_once_with("admission", "queued", group_id=0)
        assert "queued" not in linker._origins
        assert not wrapper.hit_markers
    else:
        linker.client.cancel_query.assert_not_called()
        assert "queued" in linker._origins


@pytest.fixture
def layer_counter(monkeypatch):
    from orbitkv.sglang.events import _LayerDoneCounter

    monkeypatch.setattr("torch.cuda.Event", MagicMock())
    stream = SimpleNamespace(cuda_stream=17, wait_event=MagicMock())
    monkeypatch.setattr("torch.cuda.current_stream", lambda: stream)
    layout = SimpleNamespace(
        num_layers=2,
        pools={
            "kv": SimpleNamespace(
                layer_names=["k:0", "k:1", "v:0", "v:1"],
                entry=SimpleNamespace(layer_mapping={0: 0, 1: 1}),
            )
        },
    )
    return _LayerDoneCounter(layout), stream


def test_first_use_is_observed_once_even_when_the_first_layer_wait_repeats(
    layer_counter, monkeypatch
):
    counter, stream = layer_counter
    trace = MagicMock()
    monkeypatch.setattr("orbitkv.sglang.events.trace_transfer", trace)
    index = counter.update_producer()
    counter.request_ids[index] = ["restored"]
    counter.publish_events(index)
    counter.set_consumer(index)
    counter.wait_until(0)
    counter.wait_until(0)
    counter.wait_until(1)
    assert stream.wait_event.call_count == 6
    trace.assert_called_once_with("first_use", "restored", engine="sglang")


def test_graph_capture_keeps_dependencies_and_forward_waits_for_new_records(layer_counter):
    counter, stream = layer_counter
    # Registration stays component-major; Restore follows the consumer's layer order.
    assert counter.layer_groups == [["k:0", "v:0", "k:1", "v:1"]]
    assert counter.layout.pools["kv"].layer_names == ["k:0", "k:1", "v:0", "v:1"]
    # Capture precedes the first Restore; both components must still emit a wait.
    counter.wait_until(0)
    assert stream.wait_event.call_count == 2
    for _ in range(2):
        index = counter.update_producer()
        activation = counter._activations[index]
        entered, executing = threading.Event(), threading.Event()

        def replay(index=index, entered=entered, executing=executing):
            entered.set()
            counter.set_consumer(index)
            executing.set()

        thread = threading.Thread(target=replay, daemon=True)
        thread.start()
        assert entered.wait(timeout=1)
        try:
            assert activation.result(timeout=1) == 17
            assert not executing.wait(timeout=0.05)
        finally:
            counter.publish_events(index)
            thread.join(timeout=1)
        assert executing.is_set()
        assert not counter._activations and not counter._staged
        assert not counter.request_ids


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

    from orbitkv import (
        BlockHashes,
        QueryCandidates,
        QueryLoading,
        QueryReady,
        RecoveryContract,
    )

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


@pytest.mark.parametrize("resident_tokens", [0, 64])
def test_decode_only_promises_resident_pages_and_never_prepares_external_loads(
    linker, monkeypatch, resident_tokens
):
    import torch
    from sglang.srt.disaggregation.decode_hicache_mixin import (
        DecodeHiCachePreallocMixin,
    )
    from sglang.srt.mem_cache.base_prefix_cache import MatchResult
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from orbitkv.sglang.admission import enqueue_request
    from orbitkv.sglang.recovery import RecoveryLinkerWrapper

    wrapper = object.__new__(RecoveryLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.restore_from_store = False
    wrapper.hit_markers = {}
    req = request("req")
    resident = MatchResult(torch.arange(resident_tokens), 7, 7, 7)
    matched = wrapper.match(RadixKey(req.origin_input_ids), req, resident)
    assert matched is resident
    prealloc = SimpleNamespace(scheduler=SimpleNamespace(enable_decode_hicache=False))
    promised = DecodeHiCachePreallocMixin._build_decode_prefix_match(prealloc, req, matched)
    assert promised.decode_prefix_len == resident_tokens
    assert promised.last_device_node == 7
    assert not promised.needs_local_restore
    assert not wrapper.hit_markers and not linker._lookups

    monkeypatch.setenv("ORBITKV_PREPARE_REQUESTS", "1")
    scheduler = SimpleNamespace(tree_cache=SimpleNamespace(linker=wrapper), waiting_queue=[])

    def accepted(scheduler, req):
        scheduler.waiting_queue.append(req)

    enqueue_request(accepted, scheduler, req)
    assert scheduler.waiting_queue == [req]
    assert linker.client.mock_calls == []


@pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
def test_factory_selects_restore_owner_from_pinned_runtime_role(monkeypatch, mode):
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    from orbitkv.sglang.plugin import create_cache

    monkeypatch.setenv("ORBITKV_SGLANG_TENT", "1")
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode=mode,
            disaggregation_transfer_backend="mooncake",
            disaggregation_decode_retraction_backup="none",
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_memory",
        lambda: SimpleNamespace(enable_unified_cache_external_linker=True),
    )
    cache = MagicMock()
    cache.tree_components = (ComponentType.FULL,)
    cache.components = {ComponentType.FULL: MagicMock()}
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache", lambda _: cache
    )
    linker = MagicMock()
    monkeypatch.setattr("orbitkv.sglang.linker.OrbitKVLinker", lambda *args, **kwargs: linker)
    params = SimpleNamespace(
        is_eagle=False,
        mtp_draft_device_pools=None,
        component_registry_override=None,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=MagicMock(),
    )
    ctx = SimpleNamespace(
        disable_radix_cache=False,
        enable_hierarchical_cache=False,
        is_dsa=False,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        params=params,
        server_args=object(),
        tp_worker=MagicMock(),
    )
    assert create_cache(ctx) is cache
    assert cache.linker.restore_from_store == (mode != "decode")
    assert cache.linker.cache_linker is linker
    assert cache.write_through_threshold == 1
