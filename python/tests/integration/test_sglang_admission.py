"""Exercise the pinned SGLang admission hook with controlled backing completion."""

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
    result.instance_id = "admission"
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


def test_changed_keys_cancel_old_query_and_deadline_stops_restarting_io(linker):
    from orbitkv import QueryLoading

    linker.client.query_prefetch.return_value = QueryLoading()
    assert linker.lookup("req", transfer(["old"])) == []
    assert linker.lookup("req", transfer(["new"])) == []
    linker.client.cancel_query.assert_called_once_with("admission", "req")
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
            linker.client.cancel_query.assert_called_once_with("admission", req.rid)
        else:
            linker.client.release.assert_called_once_with(b"lease")


def test_admission_expires_pending_work_without_another_lookup(linker):
    from orbitkv import QueryLoading
    from orbitkv.sglang.admission import admit_request

    req = request("unpolled")
    linker.client.query_prefetch.return_value = QueryLoading()
    linker.lookup(req.rid, transfer(["a", "b", "c"]))
    linker._QUERY_WAIT_SECONDS = 0
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker), _all_reduce_attn_groups=MagicMock()
    )
    original = MagicMock()
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    original.assert_called_once()
    linker.client.cancel_query.assert_called_once_with("admission", req.rid)
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
