from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineControlId,
    EngineControlKind,
    EngineControlPlanInfo,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixLookup,
    EnginePrefixId,
    EnginePublishedPrefix,
    EnginePublishedPrefixRelease,
    EngineReleaseId,
    EngineReleasePlan,
    EngineReleasedRequest,
    EngineRequestId,
)
from orbitkv_sglang.bridge import (  # noqa: E402
    lowering, prefix_cache, private_prefix, session_cache,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    DETACHED_PREFIX_TRANSFER,
    PageLease,
    PrefixSemanticKey,
    SnapshotPage,
)
from orbitkv_sglang.session_runtime import (  # noqa: E402
    ReleaseRecyclePending,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402

from test_session_cache import PAGE_TOKENS, _cache, _request
from test_session_lowering import _install as _lowering_install


def _tokens(boundary: int) -> tuple[int, ...]:
    return tuple(range(1, boundary + 1))


def _shared_req(rid: str, row: int, boundary: int) -> Any:
    req = _request(rid, row, boundary)
    req.origin_input_ids = _tokens(boundary)
    req.output_ids = ()
    return req


def _semantic(cache: Any, boundary: int) -> PrefixSemanticKey:
    return cache._semantic(_tokens(boundary))


def _install_shared_prefix(
    cache: Any,
    req: Any,
    *,
    row: int,
    boundary: int,
    prefix_id: EnginePrefixId,
) -> tuple[Any, PrefixSemanticKey]:
    semantic = _semantic(cache, boundary)
    node = type(cache.root_node)(
        boundary,
        tuple(_tokens(boundary)[boundary - cache.page_size : boundary]),
        semantic.digest,
        prefix_id,
        boundary // cache.page_size,
        0,
        cache.root_node,
    )
    cache.root_node.children[semantic.digest] = node
    cache._nodes[(boundary, semantic.digest)] = node
    cache.inc_lock_ref(node)
    indices = cache.req_to_token_pool.req_to_token[row, :boundary].to(
        dtype=torch.int64, copy=True
    )
    private_prefix.install_shared_prefix_metadata(
        req,
        key=("str", req.rid),
        request_id=req._orbitkv_engine_request_id,
        prefix_id=prefix_id,
        node=node,
        semantic=semantic,
        indices=indices,
        boundary=boundary,
        provisional=False,
    )
    session_cache.register_active_shared_prefix(
        cache,
        req=req,
        prefix_id=prefix_id,
        semantic=semantic,
        node=node,
        boundary=boundary,
    )
    return node, semantic


def _prefix_transfer_release(runtime: Any, key: Any) -> EngineReleasePlan:
    base = runtime.prepare_release((key,))
    released = base.releases[0]
    detached = tuple(
        item.__class__(
            item.old,
            item.replacement,
            item.logical_ordinal,
            item.old_backend_index,
            item.replacement_backend_index,
            item.token_begin,
            item.token_end_exclusive,
            item.class_id,
            item.backend_domain,
            item.action,
            DETACHED_PREFIX_TRANSFER,
        )
        for item in released.detached
    )
    runtime.pending_plan = None
    runtime.release_cleanup_confirmed = False
    return EngineReleasePlan(
        EngineReleaseId(base.release_id.session_epoch, base.release_id.sequence + 100),
        (EngineReleasedRequest(released.request_id, detached),),
        (),
    )


def test_session_miss_registers_pending_identity_before_returning_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, trace = _cache(monkeypatch)
    req = _request("miss-shared", None, None)

    def acquire_unbound(keys: Any) -> tuple[Any, ...]:
        values = tuple(keys)
        trace.append(("acquire-unbound", values))
        runtime.add_unbound(req, 101)
        return (runtime.view_for(values[0]),)

    runtime.acquire_unbound = acquire_unbound

    result = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=(1, 2, 3, 4)), req=req)
    )

    assert result.device_indices.numel() == 0
    assert trace[-1] == ("acquire-unbound", (("str", "miss-shared"),))
    assert cache._session_pending_requests[("str", "miss-shared")].req is req
    assert getattr(req, "_orbitkv_engine_request_id") == EngineRequestId(101)


def test_match_hit_flows_to_lowering_with_empty_view_until_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _lowering_install(monkeypatch, previous=0)
    cache = batch.tree_cache
    req = batch.reqs[0]
    tokens = _tokens(PAGE_TOKENS)
    req.origin_input_ids = tokens + (99,)
    req.output_ids = ()
    cache._namespace = b"n" * 32
    cache.device = torch.device("cpu")
    cache._empty = torch.empty((0,), dtype=torch.int64)
    cache._nodes = {}
    cache._clock = 0
    cache._full_total_tokens = 0
    cache._full_evictable_tokens = 0
    cache._full_protected_tokens = 0
    cache._swa_total_tokens = 0
    cache._swa_evictable_tokens = 0
    cache._swa_protected_tokens = 0
    root = prefix_cache._PrefixNode(0, (), b"", None, 0, 0, None, lock_ref=1)
    cache.root_node = root
    semantic = cache._semantic(tokens)
    prefix_id = EnginePrefixId(7, 901)
    node = prefix_cache._PrefixNode(
        PAGE_TOKENS, tokens, semantic.digest, prefix_id, 1, 0, root
    )
    root.children[semantic.digest] = node
    cache._nodes[(PAGE_TOKENS, semantic.digest)] = node
    page = SnapshotPage(
        PageLease(7, 11, 1, 2, 3), 0, 0, 0, 1, 0, 1,
        PAGE_TOKENS, 0, PAGE_TOKENS,
    )
    control_id = EngineControlId(7, 902)

    runtime.prefix_lookup = lambda values: (
        EnginePrefixLookup(semantic, prefix_id, 1),
    )
    runtime.prepare_prefix_attach = lambda values: control_id
    runtime.commit_control = lambda value: EngineControlPlanInfo(
        value, EngineControlKind.MATERIALIZATION, 1, 1, 0, 0
    )

    def read_control(value: EngineControlId) -> EngineMaterializationPlan:
        plan = EngineMaterializationPlan(
            value,
            (EngineMaterializedRequest(
                req._orbitkv_engine_request_id, 2, PAGE_TOKENS, 1, (page,)
            ),),
        )
        runtime.pending_controls[value] = (req._orbitkv_request_key, plan)
        return plan

    runtime.read_control = read_control
    cache._materialize_prefix_pages = lambda _pages, boundary, _resident: torch.arange(
        2 * PAGE_TOKENS, 2 * PAGE_TOKENS + boundary, dtype=torch.int64
    )

    match = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=tokens), req=req)
    )
    key = req._orbitkv_request_key
    assert match.device_indices.tolist() == list(range(32, 48))
    assert runtime.view_for(key).boundary == 0
    assert cache._session_pending_shared_prefix[key].req is req
    assert req._orbitkv_provisional_prefix_lock is True
    # SGLang adds the official run lock after selecting this request.
    cache.inc_lock_ref(node)
    batch.prefix_lens = [PAGE_TOKENS]
    batch.extend_lens = [1]
    batch.extend_num_tokens = 1
    batch.seq_lens_cpu = torch.tensor([PAGE_TOKENS + 1], dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.clone()
    monkeypatch.setattr(
        lowering, "_lower_all_extend",
        lambda *_args: {0: torch.tensor([32], dtype=torch.int64)},
    )

    lowering._alloc_for_extend(batch)

    assert runtime.view_for(key).boundary == PAGE_TOKENS
    assert cache._session_pending_shared_prefix == {}
    assert cache._session_active_shared_prefix[key].req is req
    assert cache._session_pending_requests == {}
    assert cache._session_requests[key].req is req
    assert req._orbitkv_provisional_prefix_lock is False
    assert req._orbitkv_prefix_lock_held is True
    assert node.lock_ref == 1
    assert events.index(("bind-rows", ((key, 1),))) < events.index(
        ("confirm-control", control_id)
    )


def test_shared_unfinished_waits_and_publishes_with_session_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, trace = _cache(monkeypatch)
    req = _request("unfinished-shared", None, None)
    req.origin_input_ids = _tokens(16)
    runtime.add_unbound(req, 111)
    pending = cache._register_pending_session_request(req)
    req.req_pool_idx = 2
    runtime.bind_request_rows(((pending.key, 2),))
    cache._promote_pending_session_requests((req,))
    req.kv = SimpleNamespace(kv_allocated_len=16)
    runtime.views[pending.key] = runtime.views[pending.key].__class__(
        pending.request_id, 2, 16, 1
    )
    pool.req_to_token[2, :16] = torch.arange(32, 48, dtype=pool.req_to_token.dtype)

    boundary = 16
    semantic = _semantic(cache, boundary)
    prefix_id = EnginePrefixId(7, 31)

    def prefix_publish(items: Any) -> tuple[Any, ...]:
        values = tuple(items)
        trace.append(("prefix-publish", values))
        return (EnginePublishedPrefix(prefix_id, semantic, 1),)

    runtime.prefix_publish = prefix_publish

    cache.cache_unfinished_req(req)

    assert trace[-2:] == [
        ("wait", (("str", "unfinished-shared"),)),
        ("prefix-publish", ((("str", "unfinished-shared"), semantic),)),
    ]
    assert getattr(req, "_orbitkv_engine_prefix_id") == prefix_id
    assert getattr(req, "_orbitkv_prefix_lock_held") is True
    assert cache._session_active_shared_prefix[("str", "unfinished-shared")].prefix_id == prefix_id


def test_shared_release_uses_publish_release_then_confirms_pending_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, trace = _cache(monkeypatch)
    req = _shared_req("shared-release", 1, 32)
    runtime.add(req, 121, 1, 32)
    key = ("str", req.rid)
    cache._session_requests[key] = session_cache._RequestEntry(
        req, key, req._orbitkv_engine_request_id, 1
    )
    pool.req_to_token[1, :32] = torch.arange(16, 48, dtype=pool.req_to_token.dtype)
    prefix_id = EnginePrefixId(7, 41)
    node, _semantic16 = _install_shared_prefix(
        cache, req, row=1, boundary=16, prefix_id=prefix_id
    )
    semantic = _semantic(cache, 32)

    release = _prefix_transfer_release(runtime, ("str", "shared-release"))

    def prepare_prefix_publish_release(items: Any) -> Any:
        values = tuple(items)
        trace.append(("publish-release", values))
        runtime.pending_plan = release
        runtime.release_cleanup_confirmed = False
        return SimpleNamespace(
            release_id=release.release_id,
            outputs=(
                EnginePublishedPrefixRelease(
                    EnginePrefixId(7, 42),
                    semantic,
                    2,
                    release.releases[0],
                ),
            ),
        )

    runtime.prepare_prefix_publish_release = prepare_prefix_publish_release
    req.origin_input_ids = _tokens(32)

    candidate = session_cache.release_candidate(req, cache, is_insert=True)
    session_cache.flush_release_group((candidate,))

    assert any(
        isinstance(item, tuple)
        and item[0] == "publish-release"
        and item[1] == ((("str", "shared-release"), semantic),)
        for item in trace
    )
    assert "confirm-release" in trace
    assert req.req_pool_idx is None
    assert req.kv is None
    assert not hasattr(req, "_orbitkv_engine_prefix_id")
    assert cache._session_active_shared_prefix == {}
    assert cache._session_requests == {}
    assert node.lock_ref == 0


def test_shared_release_duplicate_publication_falls_back_to_prepare_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, trace = _cache(monkeypatch)
    req = _shared_req("shared-fallback", 1, 16)
    runtime.add(req, 131, 1, 16)
    key = ("str", req.rid)
    cache._session_requests[key] = session_cache._RequestEntry(
        req, key, req._orbitkv_engine_request_id, 1
    )
    pool.req_to_token[1, :32] = torch.arange(16, 48, dtype=pool.req_to_token.dtype)
    prefix_id = EnginePrefixId(7, 51)
    node, semantic = _install_shared_prefix(
        cache, req, row=1, boundary=16, prefix_id=prefix_id
    )
    duplicate_digest = _semantic(cache, 32).digest
    published = type(cache.root_node)(
        32,
        tuple(_tokens(32)[32 - cache.page_size : 32]),
        duplicate_digest,
        EnginePrefixId(7, 52),
        2,
        0,
        cache.root_node,
    )
    cache.root_node.children[duplicate_digest] = published
    cache._nodes[(32, duplicate_digest)] = published
    req.origin_input_ids = _tokens(32)
    req.kv.kv_allocated_len = 32
    runtime.views[("str", "shared-fallback")] = runtime.views[("str", "shared-fallback")].__class__(
        runtime.views[("str", "shared-fallback")].request_id,
        2,
        32,
        2,
    )
    req.prefix_indices = pool.req_to_token[1, :16].to(dtype=torch.int64, copy=True)
    req.cache_protected_len = 16
    req.last_node = node
    req.last_host_node = node
    req.best_match_node = node

    candidate = session_cache.release_candidate(req, cache, is_insert=True)
    session_cache.flush_release_group((candidate,))

    assert any(
        isinstance(item, tuple) and item[0] == "prepare-release" for item in trace
    )
    assert not any(
        isinstance(item, tuple) and item[0] == "publish-release" for item in trace
    )


def test_shared_release_recycle_retry_preserves_identity_until_same_group_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, trace = _cache(monkeypatch)
    req = _shared_req("shared-retry", 1, 16)
    runtime.add(req, 141, 1, 16)
    key = ("str", req.rid)
    cache._session_requests[key] = session_cache._RequestEntry(
        req, key, req._orbitkv_engine_request_id, 1
    )
    pool.req_to_token[1, :16] = torch.arange(16, 32, dtype=pool.req_to_token.dtype)
    prefix_id = EnginePrefixId(7, 61)
    _install_shared_prefix(cache, req, row=1, boundary=16, prefix_id=prefix_id)
    req.origin_input_ids = _tokens(16)
    runtime.recycle_pending = 1

    release = _prefix_transfer_release(runtime, ("str", "shared-retry"))
    runtime.pending_plan = release

    def prepare_prefix_publish_release(items: Any) -> Any:
        values = tuple(items)
        trace.append(("publish-release", values))
        return SimpleNamespace(
            release_id=release.release_id,
            outputs=(
                EnginePublishedPrefixRelease(
                    EnginePrefixId(7, 62),
                    _semantic(cache, 16),
                    1,
                    release.releases[0],
                ),
            ),
        )

    runtime.prepare_prefix_publish_release = prepare_prefix_publish_release

    candidate = session_cache.release_candidate(req, cache, is_insert=True)
    with pytest.raises(ReleaseRecyclePending):
        session_cache.flush_release_group((candidate,))

    assert req.req_pool_idx == 1
    assert req.kv.kv_allocated_len == 16
    assert cache._session_requests[("str", "shared-retry")].req is req

    session_cache.flush_release_group((candidate,))

    assert trace.count("recycle-pending") == 1
    assert trace.count("confirm-release") == 1
    assert req.req_pool_idx is None
    assert cache._session_requests == {}


def test_shared_reset_requires_empty_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, _trace = _cache(monkeypatch)
    req = _request("shared-reset", None, None)
    req.origin_input_ids = _tokens(16)
    runtime.add_unbound(req, 151)
    pending = cache._register_pending_session_request(req)
    req.req_pool_idx = 1
    runtime.bind_request_rows(((pending.key, 1),))
    cache._promote_pending_session_requests((req,))
    req.kv = SimpleNamespace(kv_allocated_len=16)
    runtime.views[pending.key] = runtime.views[pending.key].__class__(
        pending.request_id, 2, 16, 1
    )
    pool.req_to_token[1, :16] = torch.arange(16, 32, dtype=pool.req_to_token.dtype)
    _install_shared_prefix(cache, req, row=1, boundary=16, prefix_id=EnginePrefixId(7, 71))

    with pytest.raises(RuntimeError, match="live session requests"):
        cache.reset()
