from __future__ import annotations

import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.plugin.prefix_cache as prefix_cache  # noqa: E402
import orbitkv_sglang.plugin.state as state  # noqa: E402
import orbitkv_sglang.ffi.manager as ffi_manager  # noqa: E402
from orbitkv_sglang.config import ClassConfig, ManagerPlanConfig  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    AttachedPrefix,
    ArenaStats,
    EvictedPrefix,
    FailStopped,
    ManagerError,
    MaterializedRequestView,
    ManagerStats,
    PageLease,
    PrefixEvictionBatch,
    PrefixLease,
    PrefixLookupHint,
    PrefixSemanticKey,
    PublishedPrefix,
    RequestLease,
    RequestView,
    RetryableConflict,
    SnapshotLease,
    SnapshotPage,
)
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    BasePrefixCache,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402


def _class(
    class_id: int, retention: str, *, window_tokens: int = 32
) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 1,
        name="full" if retention == "full" else "swa",
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=None if retention == "full" else window_tokens,
        period_blocks=None if retention == "full" else 3,
    )


def _config(*retentions: str, window_tokens: int = 32) -> ManagerPlanConfig:
    return ManagerPlanConfig(
        plan_path=Path("plan.json"),
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:prefix-test",
        page_tokens=16,
        classes=tuple(
            _class(index, value, window_tokens=window_tokens)
            for index, value in enumerate(retentions)
        ),
    )


class _ReqPool:
    def __init__(self) -> None:
        self.req_to_token = torch.zeros((16, 256), dtype=torch.int32)
        self.max_context_len = 256
        self.device = torch.device("cpu")


class _Allocator:
    def __init__(self, hybrid: bool) -> None:
        self.full_to_swa_index_mapping = (
            torch.zeros((1024,), dtype=torch.int64) if hybrid else None
        )


class _PrefixRuntime:
    def __init__(self, config: ManagerPlanConfig, page_capacity: int = 64) -> None:
        self.config = config
        self.failure_reason = None
        self.records = {}
        self.request_rows = {}
        self.materialized = {}
        self.prefix_keys = {}
        self.calls = []
        self.next_request = 0
        self.next_prefix = 10
        self.recycle_conflicts = 0
        self.evict_conflict_calls = set()
        self.evict_call_count = 0
        self.zero_retirement_leases = set()
        self.arenas_by_class = {
            item.class_id: ArenaIdentity(
                engine_epoch=1,
                pool_epoch=2 + item.class_id,
                pool_id=item.pool_id,
                class_id=item.class_id,
                backend_domain=item.backend_domain,
                page_count=page_capacity,
                page_tokens=16,
                backend_base_index=0,
                first_page_id=1 + item.class_id * page_capacity,
            )
            for item in config.classes
        }

    def has_request(self, key):
        return key in self.records

    def record_for(self, key):
        return self.records[key]

    def bind_request_rows(self, assignments):
        for key, row, install_row in assignments:
            if install_row or self.request_rows.get(key) != row:
                self.fail_stop("ReqToToken row ownership changed")
                raise FailStopped(self.failure_reason)

    def request_acquire_batch(self, keys):
        outputs = []
        for key in keys:
            lease = RequestLease(1, self.next_request, 1)
            self.next_request += 1
            self.records[key] = SimpleNamespace(lease=lease, boundary=0)
            outputs.append(RequestView(lease, SnapshotLease(1, lease.slot, 1), 0, 0, 0))
        self.calls.append(("acquire", len(outputs)))
        return tuple(outputs)

    def prefix_lookup_batch(self, keys):
        self.calls.append(("lookup", len(keys)))
        return tuple(
            PrefixLookupHint(
                key,
                self.materialized[key][0] if key in self.materialized else None,
                (
                    self.materialized[key][1].view.resident_count
                    if key in self.materialized
                    else 0
                ),
            )
            for key in keys
        )

    def prefix_attach_batch(self, items):
        outputs = []
        for key, hint in items:
            prefix, template = self.materialized[hint.key]
            record = self.records[key]
            view = RequestView(
                record.lease,
                template.view.snapshot,
                template.view.view_version,
                template.view.boundary,
                template.view.resident_count,
            )
            target = MaterializedRequestView(view, template.pages)
            record.boundary = view.boundary
            outputs.append(AttachedPrefix(prefix, target))
        self.calls.append(("attach", len(outputs)))
        return tuple(outputs)

    def wait_batch(self, keys):
        self.calls.append(("wait", len(keys)))

    def prefix_publish_batch(self, items):
        outputs = []
        for key, semantic in items:
            prefix = PrefixLease(1, self.next_prefix, 1)
            self.next_prefix += 1
            outputs.append(PublishedPrefix(prefix, semantic, semantic.boundary // 16))
            self.prefix_keys[prefix] = semantic
        self.calls.append(("publish", len(outputs)))
        return tuple(outputs)

    def prefix_evict_batch(self, leases):
        self.evict_call_count += 1
        if self.evict_call_count in self.evict_conflict_calls:
            self.calls.append(("evict-conflict", len(leases)))
            raise RetryableConflict("injected prefix eviction conflict")
        self.calls.append(("evict", len(leases)))
        return PrefixEvictionBatch(
            tuple(EvictedPrefix(lease, self.prefix_keys[lease]) for lease in leases),
            tuple(
                SimpleNamespace(class_id=self.config.full_class.class_id)
                for lease in leases
                if lease not in self.zero_retirement_leases
            ),
        )

    def prefix_recycle_batch(self, leases):
        if self.recycle_conflicts:
            self.recycle_conflicts -= 1
            self.calls.append(("recycle-conflict", len(leases)))
            raise RetryableConflict("injected prefix recycle conflict")
        self.calls.append(("recycle", len(leases)))
        for lease in leases:
            self.prefix_keys.pop(lease, None)

    def release_batch(self, keys):
        self.calls.append(("release", len(keys)))
        for key in keys:
            self.records.pop(key)

    def fail_stop(self, reason):
        self.failure_reason = reason

    def poll(self):
        return None

    def stats(self):
        active_requests = len(self.records)
        active_prefixes = len(self.prefix_keys)
        active_pages = active_requests + active_prefixes
        return ManagerStats(
            active_requests=active_requests,
            active_snapshots=active_requests,
            active_prefixes=active_prefixes,
            evicted_prefixes=0,
            prepared_steps=0,
            submitted_steps=0,
            free_pages=128 - active_pages,
            reserved_pages=0,
            writing_pages=0,
            active_pages=active_pages,
            retiring_pages=0,
            quarantined_pages=0,
            exhausted_pages=0,
            pending_reclamations=0,
            total_request_page_refs=active_requests,
            total_prefix_page_refs=active_prefixes,
            total_reader_pins=0,
        )

    def arena_stats(self):
        stats = self.stats()
        return tuple(
            ArenaStats(
                engine_epoch=arena.engine_epoch,
                pool_epoch=arena.pool_epoch,
                pool_id=arena.pool_id,
                page_count=arena.page_count,
                class_id=arena.class_id,
                backend_domain=arena.backend_domain,
                first_page_id=arena.first_page_id,
                free_pages=arena.page_count - stats.active_pages,
                reserved_pages=0,
                writing_pages=0,
                active_pages=stats.active_pages,
                retiring_pages=0,
                quarantined_pages=0,
                exhausted_pages=0,
                request_page_refs=stats.total_request_page_refs,
                prefix_page_refs=stats.total_prefix_page_refs,
                reader_pins=0,
            )
            for arena in self.arenas_by_class.values()
        )

    def census(self):
        return self.stats(), self.arena_stats()

    def close(self):
        if self.failure_reason is None and (self.prefix_keys or self.records):
            raise ManagerError("cannot destroy a manager with live ownership")
        self.calls.append(("close", 1))


def _cache(
    *retentions: str, window_tokens: int = 32, page_capacity: int = 64
):
    config = _config(*retentions, window_tokens=window_tokens)
    runtime = _PrefixRuntime(config, page_capacity)
    allocator = _Allocator(len(retentions) == 2)
    req_pool = _ReqPool()
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 256),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=16,
        sliding_window_size=window_tokens if len(retentions) == 2 else None,
    )
    return prefix_cache.OrbitKvPrefixCache(params), runtime, allocator, req_pool


def _tokens(count: int) -> tuple[int, ...]:
    return tuple(range(count))


def _assert_size_census(cache: prefix_cache.OrbitKvPrefixCache) -> None:
    nodes = tuple(cache._nodes.values())
    full_evictable = sum(
        cache._full_edge_tokens(node) for node in nodes if node.lock_ref == 0
    )
    full_protected = sum(
        cache._full_edge_tokens(node) for node in nodes if node.lock_ref > 0
    )
    swa_evictable = sum(
        cache.page_size
        for node in nodes
        if node.swa_ref_count > 0 and node.lock_ref == 0
    )
    swa_protected = sum(
        cache.page_size
        for node in nodes
        if node.swa_ref_count > 0 and node.lock_ref > 0
    )
    assert cache.total_size() == full_evictable + full_protected
    assert cache.full_evictable_size() == full_evictable
    assert cache.full_protected_size() == full_protected
    assert cache.swa_evictable_size() == swa_evictable
    assert cache.swa_protected_size() == swa_protected


def _materialized(
    runtime: _PrefixRuntime, boundary: int, *, hybrid: bool
) -> MaterializedRequestView:
    request = RequestLease(1, 99, 1)
    pages = []
    for ordinal in range(boundary // 16):
        pages.append(
            SnapshotPage(
                PageLease(1, 2, 1, 1 + ordinal, 1),
                ordinal,
                0,
                0,
                ordinal,
                0,
                1,
                16,
                0,
                16,
            )
        )
    if hybrid:
        for ordinal in range(boundary // 16):
            pages.append(
                SnapshotPage(
                    PageLease(1, 3, 1, 65 + ordinal, 2),
                    ordinal,
                    0,
                    0,
                    4 + ordinal,
                    1,
                    2,
                    16,
                    0,
                    16,
                )
            )
    return MaterializedRequestView(
        RequestView(
            request,
            SnapshotLease(1, 99, 1),
            1,
            boundary,
            len(pages),
        ),
        tuple(pages),
    )


def _publish(
    cache: prefix_cache.OrbitKvPrefixCache,
    runtime: _PrefixRuntime,
    tokens: tuple[int, ...],
    slot: int,
    *,
    resident_count: int | None = None,
):
    semantic = cache._semantic(tokens)
    lease = PrefixLease(1, slot, 1)
    publication = PublishedPrefix(
        lease,
        semantic,
        resident_count if resident_count is not None else len(tokens) // 16,
    )
    node = cache._record_publication(publication, tokens)
    runtime.prefix_keys[lease] = semantic
    return semantic, lease, node


def _naive_swa_plan_tokens(cache, plan):
    nodes_by_prefix = {
        node.prefix: node
        for node in cache._nodes.values()
        if node.prefix is not None
    }
    remaining = {node: node.swa_ref_count for node in cache._nodes.values()}
    result = []
    for item in plan:
        node = nodes_by_prefix[item.prefix]
        current = node
        released = 0
        pages = node.resident_count - node.boundary // cache.page_size
        for _ in range(pages):
            remaining[current] -= 1
            assert remaining[current] >= 0
            if remaining[current] == 0:
                released += cache.page_size
            current = current.parent
            assert current is not None
        result.append(released)
    return tuple(result)


def _request(rid: str, tokens: tuple[int, ...]):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(tokens),
        output_ids=[],
        extra_key=None,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        last_node=None,
    )


def test_prefix_cache_is_official_backend_with_exact_namespace_and_linear_endpoints():
    cache, _runtime, _allocator, _pool = _cache("full")
    assert isinstance(cache, BasePrefixCache)
    assert len(cache._namespace) == 32
    assert set(prefix_cache._PrefixNode.__slots__) == {
        "boundary",
        "edge",
        "digest",
        "prefix",
        "resident_count",
        "swa_ref_count",
        "parent",
        "children",
        "lock_ref",
        "last_access",
        "evicted",
        "backuped",
    }
    assert cache.supports_fast_match_prefix() is False
    endpoints = cache._semantic_endpoints(_tokens(8192))
    assert len(endpoints) == 512
    assert endpoints[-1].boundary == 8192


def test_prefix_cache_abstract_method_call_signatures_match_official_v0517():
    names = (
        "reset",
        "match_prefix",
        "cache_finished_req",
        "cache_unfinished_req",
        "evict",
        "inc_lock_ref",
        "dec_lock_ref",
    )
    for name in names:
        official = inspect.signature(getattr(BasePrefixCache, name))
        implementation = inspect.signature(
            getattr(prefix_cache.OrbitKvPrefixCache, name)
        )
        assert [
            (item.name, item.kind, item.default)
            for item in implementation.parameters.values()
        ] == [
            (item.name, item.kind, item.default)
            for item in official.parameters.values()
        ]


@pytest.mark.parametrize("field", ("namespace", "digest"))
@pytest.mark.parametrize("length", (0, 1, 31, 33, 64))
def test_prefix_wire_key_rejects_non_exact_digest_width_before_manager_call(
    field, length
):
    values = {"namespace": b"n" * 32, "digest": b"d" * 32}
    values[field] = b"x" * length
    with pytest.raises(ManagerError, match="exactly 32 bytes"):
        ffi_manager._key_to_c(PrefixSemanticKey(**values, boundary=16))

    converted = ffi_manager._key_to_c(
        PrefixSemanticKey(b"n" * 32, b"d" * 32, 16)
    )
    assert bytes(converted.namespace_bytes) == b"n" * 32
    assert bytes(converted.digest) == b"d" * 32


def test_prefix_miss_does_not_acquire_and_hit_attaches_one_deepest_endpoint():
    cache, runtime, _allocator, _pool = _cache("full")
    tokens = _tokens(32)
    miss = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=tokens), req=_request("miss", tokens))
    )
    assert len(miss.device_indices) == 0
    assert runtime.calls == []

    semantic, lease, node = _publish(cache, runtime, tokens, 1)
    runtime.materialized[semantic] = (lease, _materialized(runtime, 32, hybrid=False))
    req = _request("hit", tokens)
    hit = cache.match_prefix(MatchPrefixParams(RadixKey(token_ids=tokens), req=req))
    assert torch.equal(hit.device_indices, torch.arange(16, 48))
    assert runtime.calls[-3:] == [("lookup", 1), ("acquire", 1), ("attach", 1)]
    assert node.lock_ref == 1
    assert req._orbitkv_provisional_prefix_lock is True
    assert cache.get_prefix_hash_values(node)[-1] == semantic.digest.hex()


def test_repeated_attached_match_rejects_lost_lock_before_manager_lookup():
    cache, runtime, _allocator, _pool = _cache("full")
    tokens = _tokens(32)
    semantic, lease, node = _publish(cache, runtime, tokens, 31)
    runtime.materialized[semantic] = (lease, _materialized(runtime, 32, hybrid=False))
    req = _request("hostile-repeat", tokens)
    first = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=tokens), req=req)
    )
    req.prefix_indices = first.device_indices
    req.last_node = node
    delattr(req, "_orbitkv_provisional_prefix_lock")
    before = tuple(runtime.calls)

    with pytest.raises(RuntimeError, match="prefix-lock state changed"):
        cache.match_prefix(MatchPrefixParams(RadixKey(token_ids=tokens), req=req))

    assert tuple(runtime.calls) == before
    assert node.lock_ref == 1


def test_repeated_attached_match_keeps_shorter_live_prefix_identity():
    cache, runtime, _allocator, _pool = _cache("full")
    short = _tokens(16)
    long = _tokens(32)
    short_semantic, short_lease, short_node = _publish(
        cache, runtime, short, 32
    )
    long_semantic, long_lease, _long_node = _publish(cache, runtime, long, 33)
    runtime.materialized[short_semantic] = (
        short_lease,
        _materialized(runtime, 16, hybrid=False),
    )
    runtime.materialized[long_semantic] = (
        long_lease,
        _materialized(runtime, 32, hybrid=False),
    )
    req = _request("short-live", short)
    first = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=short), req=req)
    )
    req.prefix_indices = first.device_indices
    req.last_node = short_node

    repeated = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=long), req=req)
    )

    assert len(repeated.device_indices) == 16
    assert repeated.last_device_node is short_node
    assert runtime.calls[-1] == ("lookup", 1)


def test_hybrid_attach_only_validates_manager_mapping_and_bitflip_fail_stops():
    cache, runtime, allocator, _pool = _cache("full", "sliding")
    tokens = _tokens(32)
    semantic, lease, _node = _publish(
        cache, runtime, tokens, 2, resident_count=4
    )
    materialized = _materialized(runtime, 32, hybrid=True)
    runtime.materialized[semantic] = (lease, materialized)
    full_locations = torch.arange(16, 48)
    expected_swa = torch.arange(80, 112)
    allocator.full_to_swa_index_mapping[full_locations] = expected_swa
    before = allocator.full_to_swa_index_mapping.clone()
    hit = cache.match_prefix(
        MatchPrefixParams(RadixKey(token_ids=tokens), req=_request("ok", tokens))
    )
    assert len(hit.device_indices) == 32
    assert torch.equal(allocator.full_to_swa_index_mapping, before)

    cache2, runtime2, allocator2, _pool2 = _cache("full", "sliding")
    semantic2, lease2, _node2 = _publish(
        cache2, runtime2, tokens, 3, resident_count=4
    )
    runtime2.materialized[semantic2] = (lease2, _materialized(runtime2, 32, hybrid=True))
    allocator2.full_to_swa_index_mapping[full_locations] = expected_swa
    allocator2.full_to_swa_index_mapping[23] = 999
    corrupted = allocator2.full_to_swa_index_mapping.clone()
    with pytest.raises(FailStopped, match="materialization"):
        cache2.match_prefix(
            MatchPrefixParams(
                RadixKey(token_ids=tokens), req=_request("corrupt", tokens)
            )
        )
    assert runtime2.failure_reason is not None
    assert torch.equal(allocator2.full_to_swa_index_mapping, corrupted)


def test_structural_radix_is_publication_order_independent_and_locks_ancestors():
    cache, runtime, _allocator, _pool = _cache("full")
    long_tokens = _tokens(64)
    short_tokens = long_tokens[:32]
    _long_key, long_lease, long_node = _publish(cache, runtime, long_tokens, 4)
    _short_key, short_lease, short_node = _publish(cache, runtime, short_tokens, 5)
    assert cache.total_size() == 64
    assert long_node.parent is not None
    assert short_node in (long_node.parent, long_node.parent.parent)
    cache.inc_lock_ref(long_node)
    assert short_node.lock_ref == 1
    assert cache._eviction_plan(16, 0) == ()
    cache.dec_lock_ref(long_node)
    assert cache._eviction_plan(32, 0) == (long_lease,)
    assert cache._eviction_plan(48, 0) == (long_lease, short_lease)

    reverse, _runtime2, _allocator2, _pool2 = _cache("full")
    _publish(reverse, _runtime2, short_tokens, 6)
    _publish(reverse, _runtime2, long_tokens, 7)
    assert reverse.total_size() == cache.total_size()


def test_hybrid_resident_census_drives_exact_swa_size_and_lock_protection():
    cache, runtime, _allocator, _pool = _cache("full", "sliding")
    _semantic, _lease, node = _publish(
        cache, runtime, _tokens(64), 8, resident_count=6
    )
    assert cache.full_evictable_size() == 64
    assert cache.swa_evictable_size() == 32
    _assert_size_census(cache)
    cache.inc_lock_ref(node)
    assert cache.swa_evictable_size() == 0
    assert cache.swa_protected_size() == 32
    _assert_size_census(cache)
    cache.dec_lock_ref(node)
    assert cache.swa_evictable_size() == 32
    _assert_size_census(cache)


def test_size_getters_do_not_iterate_prefix_nodes():
    cache, runtime, _allocator, _pool = _cache("full", "sliding")
    _publish(cache, runtime, _tokens(64), 37, resident_count=6)
    expected = (
        cache.total_size(),
        cache.full_evictable_size(),
        cache.full_protected_size(),
        cache.swa_evictable_size(),
        cache.swa_protected_size(),
    )

    class NoNodeIteration(dict):
        def values(self):
            raise AssertionError("hot size getter iterated radix nodes")

    cache._nodes = NoNodeIteration(cache._nodes)
    assert (
        cache.total_size(),
        cache.full_evictable_size(),
        cache.full_protected_size(),
        cache.swa_evictable_size(),
        cache.swa_protected_size(),
    ) == expected


def test_official_prefill_adder_reads_compiled_sliding_window(monkeypatch):
    import sglang.srt.managers.schedule_policy as schedule_policy

    PrefillAdder = schedule_policy.PrefillAdder
    SWATokenToKVPoolAllocator = schedule_policy.SWATokenToKVPoolAllocator
    monkeypatch.setattr(
        schedule_policy, "is_dsa_prefill_cp_in_seq_split", lambda: False
    )
    monkeypatch.setattr(
        schedule_policy, "is_prefill_context_parallel_enabled", lambda: False
    )

    cache, _runtime, _allocator, _pool = _cache("full", "sliding")
    official_allocator = object.__new__(SWATokenToKVPoolAllocator)
    adder = PrefillAdder(
        page_size=16,
        tree_cache=cache,
        token_to_kv_pool_allocator=official_allocator,
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=256,
        rem_chunk_tokens=64,
    )

    assert cache.sliding_window_size == 32
    assert adder.is_hybrid_swa is True
    assert adder._swa_budget_for_req(48, 8) == 64


def test_prefix_evict_batches_topological_chain_and_counts_actual_certificates():
    cache, runtime, _allocator, _pool = _cache("full")
    for endpoint in (16, 32, 48, 64):
        _publish(cache, runtime, _tokens(endpoint), 20 + endpoint)
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    result = cache.evict(EvictParams(num_tokens=64))
    assert result.num_tokens_evicted == 64
    assert [call for call in runtime.calls if call[0] == "evict"] == [("evict", 4)]
    assert [call for call in runtime.calls if call[0] == "recycle"] == [
        ("recycle", 4)
    ]
    assert cache.total_size() == 0


def test_prefix_evict_continues_stored_plan_after_shared_zero_cert(monkeypatch):
    cache, runtime, _allocator, _pool = _cache("full")
    _short_semantic, _short_lease, _short_node = _publish(
        cache, runtime, _tokens(16), 35
    )
    _long_semantic, long_lease, _long_node = _publish(
        cache, runtime, _tokens(32), 36
    )
    runtime.zero_retirement_leases.add(long_lease)
    builds = 0
    original = prefix_cache.OrbitKvPrefixCache._complete_eviction_plan

    def counted_plan(self):
        nonlocal builds
        builds += 1
        return original(self)

    monkeypatch.setattr(
        prefix_cache.OrbitKvPrefixCache,
        "_complete_eviction_plan",
        counted_plan,
    )
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    result = cache.evict(EvictParams(num_tokens=16))

    assert result.num_tokens_evicted == 16
    assert builds == 1
    assert [call for call in runtime.calls if call[0] == "evict"] == [
        ("evict", 1),
        ("evict", 1),
    ]
    assert [call for call in runtime.calls if call[0] == "recycle"] == [
        ("recycle", 1),
        ("recycle", 1),
    ]


def test_complete_eviction_plan_visits_each_comb_edge_once(monkeypatch):
    cache, runtime, _allocator, _pool = _cache("full")
    trunk = _tokens(64)
    for branch in range(64):
        edge = tuple(10_000 + branch * 16 + offset for offset in range(16))
        _publish(cache, runtime, trunk + edge, 100 + branch)
    visits = 0
    original = prefix_cache.OrbitKvPrefixCache._full_edge_tokens

    def counted_edge(node):
        nonlocal visits
        visits += 1
        return original(node)

    monkeypatch.setattr(
        prefix_cache.OrbitKvPrefixCache,
        "_full_edge_tokens",
        staticmethod(counted_edge),
    )

    plan = cache._complete_eviction_plan()

    assert len(plan) == 64
    assert visits <= len(cache._nodes)


def test_hybrid_eviction_plan_and_accept_large_window_work_is_near_linear(
    monkeypatch,
):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    measurements = []
    accept_measurements = []
    for pages in (64, 128):
        cache, runtime, _allocator, _pool = _cache(
            "full",
            "sliding",
            window_tokens=pages * 16,
            page_capacity=pages,
        )
        for depth in range(1, pages + 1):
            _publish(
                cache,
                runtime,
                _tokens(depth * 16),
                depth,
                resident_count=depth * 2,
            )
        work = {}

        plan = cache._complete_eviction_plan(_work=work)

        assert len(plan) == pages
        assert tuple(item.swa_tokens for item in plan) == (16,) * pages
        node_count = len(cache._nodes)
        bound = 12 * (node_count + len(plan))
        assert work["swa_path_steps"] <= bound
        measurements.append(work["swa_path_steps"])
        accept_work = {}
        original_accept = cache._accept_evictions

        def counted_accept(outputs, requested, *, _original=original_accept):
            return _original(outputs, requested, _work=accept_work)

        monkeypatch.setattr(cache, "_accept_evictions", counted_accept)
        result = cache.evict(EvictParams(num_tokens=pages * 16))
        assert result.num_tokens_evicted == pages * 16
        assert accept_work["swa_path_steps"] <= 4 * (node_count + len(plan))
        accept_measurements.append(accept_work["swa_path_steps"])
    assert measurements[1] <= measurements[0] * 3
    assert accept_measurements[1] <= accept_measurements[0] * 3


def test_hybrid_eviction_plan_locked_reference_matches_exact_naive_oracle():
    cache, runtime, _allocator, _pool = _cache(
        "full", "sliding", window_tokens=32
    )
    trunk = _tokens(32)
    left = trunk + tuple(range(10_000, 10_016))
    right = trunk + tuple(range(20_000, 20_016))
    _left_key, left_lease, left_node = _publish(
        cache, runtime, left, 1_201, resident_count=5
    )
    _right_key, right_lease, _right_node = _publish(
        cache, runtime, right, 1_202, resident_count=5
    )
    cache.inc_lock_ref(left_node)

    plan = cache._complete_eviction_plan()

    assert tuple(item.prefix for item in plan) == (right_lease,)
    assert left_lease not in tuple(item.prefix for item in plan)
    assert tuple(item.swa_tokens for item in plan) == _naive_swa_plan_tokens(
        cache, plan
    )
    assert plan[0].swa_tokens == 16


def test_zero_retirement_chain_defers_one_linear_local_accept(monkeypatch):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    for pages in (64, 128, 256):
        cache, runtime, _allocator, _pool = _cache(
            "full",
            "sliding",
            window_tokens=pages * 16,
            page_capacity=pages,
        )
        leases = []
        for depth in range(1, pages + 1):
            _semantic, lease, _node = _publish(
                cache,
                runtime,
                _tokens(depth * 16),
                depth,
                resident_count=depth * 2,
            )
            leases.append(lease)
        runtime.zero_retirement_leases.update(leases)
        map_calls = node_visits = accept_steps = 0
        original_map = cache._published_parent_map
        original_accept = cache._accept_evictions

        def counted_map():
            nonlocal map_calls, node_visits
            map_calls += 1
            node_visits += len(cache._nodes) + 1
            return original_map()

        def counted_accept(outputs, requested):
            nonlocal accept_steps
            work = {}
            result = original_accept(outputs, requested, _work=work)
            accept_steps += work["swa_path_steps"]
            return result

        monkeypatch.setattr(cache, "_published_parent_map", counted_map)
        monkeypatch.setattr(cache, "_accept_evictions", counted_accept)

        result = cache.evict(EvictParams(num_tokens=16))

        native_batches = [count for name, count in runtime.calls if name == "evict"]
        assert result.num_tokens_evicted == 0
        assert result.swa_num_tokens_evicted == 0
        assert sum(native_batches) == pages
        assert native_batches == [1] * pages
        assert map_calls == 2
        assert node_visits == 2 * (pages + 1)
        assert accept_steps <= 4 * (pages + 1)
        assert cache.total_size() == 0
        assert cache.swa_evictable_size() == 0
        assert not cache._nodes
        assert not runtime.prefix_keys


def test_exact_native_probes_stop_at_first_capacity_certificate():
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    for pages in (64, 128, 256):
        cache, runtime, _allocator, _pool = _cache(
            "full",
            "sliding",
            window_tokens=pages * 16,
            page_capacity=pages,
        )
        for depth in range(1, pages + 1):
            _publish(
                cache,
                runtime,
                _tokens(depth * 16),
                depth,
                resident_count=depth * 2,
            )
        plan = cache._complete_eviction_plan()
        first_capacity = pages // 2
        runtime.zero_retirement_leases.update(
            item.prefix for item in plan[: first_capacity - 1]
        )

        result = cache.evict(EvictParams(num_tokens=16))

        native_batches = [count for name, count in runtime.calls if name == "evict"]
        assert result.num_tokens_evicted == 16
        assert native_batches == [1] * first_capacity
        assert len(runtime.prefix_keys) == pages - first_capacity
        assert cache.total_size() == (pages - first_capacity) * 16
        assert len(cache._nodes) == pages - first_capacity


def test_late_probe_conflict_installs_prior_commit_then_fail_stops():
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    cache, runtime, _allocator, _pool = _cache("full")
    _short_key, short_lease, _short_node = _publish(
        cache, runtime, _tokens(16), 1
    )
    _long_key, long_lease, _long_node = _publish(
        cache, runtime, _tokens(32), 2
    )
    runtime.zero_retirement_leases.add(long_lease)
    runtime.evict_conflict_calls.add(2)

    with pytest.raises(FailStopped, match="earlier commit"):
        cache.evict(EvictParams(num_tokens=16))

    assert runtime.failure_reason == "prefix eviction failed after an earlier commit"
    assert runtime.calls[-3:] == [
        ("evict", 1),
        ("recycle", 1),
        ("evict-conflict", 1),
    ]
    assert long_lease not in runtime.prefix_keys
    assert short_lease in runtime.prefix_keys
    assert cache.total_size() == 16
    assert tuple(
        node.prefix for node in cache._nodes.values() if node.prefix is not None
    ) == (short_lease,)


def test_prefix_recycle_conflict_retries_before_local_tree_removal():
    cache, runtime, _allocator, _pool = _cache("full")
    _publish(cache, runtime, _tokens(16), 91)
    runtime.recycle_conflicts = 1
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    result = cache.evict(EvictParams(num_tokens=16))

    assert result.num_tokens_evicted == 16
    assert runtime.calls[-3:] == [
        ("evict", 1),
        ("recycle-conflict", 1),
        ("recycle", 1),
    ]
    assert cache.total_size() == 0


def test_unfinished_publish_adopts_official_lock_and_complete_hash_chain():
    cache, runtime, _allocator, pool = _cache("full")
    tokens = _tokens(32)
    req = _request("unfinished", tokens)
    req.req_pool_idx = 1
    req.kv = SimpleNamespace(kv_allocated_len=32)
    req.last_node = cache.root_node
    key = ("str", "unfinished")
    lease = RequestLease(1, 7, 1)
    runtime.records[key] = SimpleNamespace(lease=lease, boundary=32)
    runtime.request_rows[key] = 1
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease
    pool.req_to_token[1, :32] = torch.arange(16, 48, dtype=torch.int32)
    cache.cache_unfinished_req(req)
    assert req._orbitkv_prefix_lock_held is True
    assert req.last_node.lock_ref == 1
    assert len(cache.get_prefix_hash_values(req.last_node)) == 2
    assert torch.equal(req.prefix_indices, torch.arange(16, 48))


def test_unfinished_post_publish_adoption_fault_fail_stops_runtime(monkeypatch):
    cache, runtime, _allocator, pool = _cache("full")
    tokens = _tokens(16)
    req = _request("unfinished-fault", tokens)
    req.req_pool_idx = 1
    req.kv = SimpleNamespace(kv_allocated_len=16)
    req.last_node = cache.root_node
    key = ("str", "unfinished-fault")
    lease = RequestLease(1, 12, 1)
    runtime.records[key] = SimpleNamespace(lease=lease, boundary=16)
    runtime.request_rows[key] = 1
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease
    pool.req_to_token[1, :16] = torch.arange(16, 32, dtype=torch.int32)
    monkeypatch.setattr(
        prefix_cache.OrbitKvPrefixCache,
        "_adopt_published_node",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("adoption fault")),
    )

    with pytest.raises(FailStopped, match="unfinished prefix adoption failed"):
        cache.cache_unfinished_req(req)

    assert ("publish", 1) in runtime.calls
    assert len(runtime.prefix_keys) == 1
    assert runtime.failure_reason == "unfinished prefix adoption failed: adoption fault"


def test_unfinished_malformed_old_lock_rejects_before_native_publish():
    cache, runtime, _allocator, _pool = _cache("full")
    _semantic, _prefix, node = _publish(cache, runtime, _tokens(16), 34)
    req = _request("unfinished-hostile", _tokens(32))
    req.req_pool_idx = 1
    req.kv = SimpleNamespace(kv_allocated_len=32)
    req.last_node = None
    key = ("str", "unfinished-hostile")
    lease = RequestLease(1, 13, 1)
    runtime.records[key] = SimpleNamespace(lease=lease, boundary=32)
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease
    req._orbitkv_prefix_node = node
    req._orbitkv_prefix_lock_held = True

    with pytest.raises(RuntimeError, match="differs from its held lock"):
        cache.cache_unfinished_req(req)

    assert not any(call[0] == "publish" for call in runtime.calls)
    assert node.lock_ref == 0


def test_unfinished_foreign_row_rejects_before_native_publish():
    cache, runtime, _allocator, _pool = _cache("full")
    req = _request("unfinished-row", _tokens(16))
    req.req_pool_idx = 2
    req.kv = SimpleNamespace(kv_allocated_len=16)
    req.last_node = cache.root_node
    key = ("str", "unfinished-row")
    lease = RequestLease(1, 14, 1)
    runtime.records[key] = SimpleNamespace(lease=lease, boundary=16)
    runtime.request_rows[key] = 1
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease

    with pytest.raises(FailStopped, match="ReqToToken row ownership changed"):
        cache.cache_unfinished_req(req)

    assert not any(call[0] == "publish" for call in runtime.calls)
    assert runtime.records[key].lease == lease


def test_cache_init_rejects_official_options_it_does_not_implement():
    config = _config("full")
    runtime = _PrefixRuntime(config)
    allocator = _Allocator(False)
    pool = _ReqPool()
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 256),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator
    base = dict(
        disable=False,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        page_size=16,
    )
    with pytest.raises(RuntimeError, match="LRU"):
        prefix_cache.OrbitKvPrefixCache(
            CacheInitParams(**base, eviction_policy="fifo")
        )
    with pytest.raises(RuntimeError, match="TTL"):
        prefix_cache.OrbitKvPrefixCache(
            CacheInitParams(**base, cache_ttl_seconds=1.0)
        )
    with pytest.raises(RuntimeError, match="KV cache events"):
        prefix_cache.OrbitKvPrefixCache(
            CacheInitParams(**base, enable_kv_cache_events=True)
        )


def test_live_graceful_shutdown_closes_manager_even_when_reset_is_unsafe():
    cache, runtime, _allocator, _pool = _cache("full")
    _semantic, _lease, node = _publish(cache, runtime, _tokens(16), 92)
    cache.inc_lock_ref(node)

    with pytest.raises(RuntimeError, match="prefixes are protected"):
        cache.release_host_resources()

    assert runtime.failure_reason == "shutdown encountered live OrbitKV prefix ownership"
    assert runtime.calls[-1] == ("close", 1)
    cache.release_host_resources()
    assert runtime.calls.count(("close", 1)) == 1


def test_live_prefix_miss_shutdown_poison_closes_without_tree_nodes():
    cache, runtime, _allocator, _pool = _cache("full")
    key = ("str", "live-miss")
    runtime.records[key] = SimpleNamespace(
        lease=RequestLease(1, 9, 1), boundary=0
    )

    cache.release_host_resources()

    assert runtime.failure_reason == (
        "shutdown encountered live OrbitKV request ownership"
    )
    assert cache._nodes == {}
    assert runtime.calls[-1] == ("close", 1)
