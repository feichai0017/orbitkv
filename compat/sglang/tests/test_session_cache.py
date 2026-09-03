from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402
from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineControlDisposition,
    EngineControlId,
    EngineControlKind,
    EngineControlOutcome,
    EngineControlPlanInfo,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixEvictionPlan,
    EnginePrefixId,
    EnginePublishedPrefix,
    EngineReleaseId,
    EngineReleasePlan,
    EngineReleasedRequest,
    EngineRequestId,
    EngineRequestView,
    EngineRetirement,
)
from orbitkv_sglang.bridge import (  # noqa: E402
    prefix_cache,
    private_prefix,
    session_cache,
    session_lowering,
    state,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    DETACHED_CLEAR,
    DETACHED_REQUEST_RELEASE,
    ArenaIdentity,
    ArenaStats,
    DetachedBinding,
    ManagerStats,
    PageLease,
)
from orbitkv_sglang.session_runtime import (  # noqa: E402
    ReleaseRecyclePending,
    SessionMaterializationUpdate,
    SessionMirrorUpdate,
    SessionRequestBinding,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402


PAGE_TOKENS = 16


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:session-cache-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=3,
                backend_domain=17,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
                storage="token_kv",
            ),
        ),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )


class _Pool:
    def __init__(self, trace: list[Any]) -> None:
        self.req_to_token = torch.zeros((8, 64), dtype=torch.int32)
        self.max_context_len = 64
        self.device = torch.device("cpu")
        self.trace = trace
        self.free_slots = list(range(1, 8))

    def free(self, req: Any) -> None:
        assert not torch.count_nonzero(
            self.req_to_token[int(req.req_pool_idx)]
        )
        row = int(req.req_pool_idx)
        assert row not in self.free_slots
        self.trace.append(("free", req.rid))
        self.free_slots.append(row)
        req.req_pool_idx = None


class _Runtime:
    def __init__(self, pool: _Pool, trace: list[Any]) -> None:
        self.pool = pool
        self.trace = trace
        self.failure_reason: str | None = None
        self.cleanup = None
        self.materialization = None
        self.bindings: dict[Any, SessionRequestBinding] = {}
        self.views: dict[Any, EngineRequestView] = {}
        self.pending_plan: EngineReleasePlan | None = None
        self.pending_control = None
        self.control_sequence = 1
        self.block_pending_release = False
        self.release_cleanup_confirmed = False
        self.recycle_pending = 0
        self.closed = False
        self.prefix_capacity = 4
        self.control_batch_capacity = 1
        self.prefix_eviction_batch_capacity = 1
        self.arenas_by_class = {
            0: ArenaIdentity(7, 11, 3, 0, 17, 32, PAGE_TOKENS, 0, 1)
        }

    def bind_mirror_cleanup(self, callback: Any) -> None:
        if self.cleanup is not None:
            raise RuntimeError("cleanup already bound")
        self.trace.append("bind")
        self.cleanup = callback

    def bind_materialization(self, callback: Any) -> None:
        if self.materialization is not None:
            raise RuntimeError("materialization already bound")
        self.trace.append("bind-materialization")
        self.materialization = callback

    def add(self, req: Any, request_id: int, row: int, boundary: int) -> None:
        key = ("str", req.rid)
        engine_id = EngineRequestId(request_id)
        self.bindings[key] = SessionRequestBinding(key, engine_id, row)
        self.views[key] = EngineRequestView(
            engine_id, 1, boundary, (boundary + PAGE_TOKENS - 1) // PAGE_TOKENS
        )
        req._orbitkv_request_key = key
        req._orbitkv_engine_request_id = engine_id

    def add_unbound(self, req: Any, request_id: int) -> None:
        key = ("str", req.rid)
        engine_id = EngineRequestId(request_id)
        self.bindings[key] = SessionRequestBinding(key, engine_id, None)
        self.views[key] = EngineRequestView(engine_id, 1, 0, 0)
        req._orbitkv_request_key = key
        req._orbitkv_engine_request_id = engine_id

    def bind_request_rows(self, assignments: Any) -> None:
        values = tuple(assignments)
        self.trace.append(("bind-rows", values))
        for key, row in values:
            binding = self.bindings[key]
            assert binding.request_row is None
            self.bindings[key] = SessionRequestBinding(
                key, binding.request_id, int(row)
            )

    def binding_for(self, key: Any) -> SessionRequestBinding:
        return self.bindings[key]

    def view_for(self, key: Any) -> EngineRequestView:
        return self.views[key]

    def wait_requests(self, keys: Any) -> None:
        self.trace.append(("wait", tuple(keys)))

    def prepare_release(self, keys: Any) -> EngineReleasePlan:
        values = tuple(keys)
        if self.block_pending_release:
            raise RuntimeError("pending control group")
        if self.pending_plan is not None:
            return self.pending_plan
        self.trace.append(("prepare-release", values))
        releases = []
        retirements = []
        for key in values:
            binding = self.bindings[key]
            boundary = int(self.views[key].boundary)
            detached = []
            for ordinal, begin in enumerate(range(0, boundary, PAGE_TOKENS)):
                end = min(begin + PAGE_TOKENS, boundary)
                backend_index = (int(binding.request_row) - 1) * 4 + ordinal
                page = PageLease(7, 11, 1, backend_index + 1, 3)
                detached.append(
                    DetachedBinding(
                        page,
                        PageLease(0, 0, 0, 0, 0),
                        ordinal,
                        backend_index,
                        0,
                        begin,
                        end,
                        0,
                        17,
                        DETACHED_CLEAR,
                        DETACHED_REQUEST_RELEASE,
                    )
                )
                retirements.append(
                    EngineRetirement(
                        page,
                        0,
                        17,
                        ordinal,
                        backend_index,
                        begin,
                        end,
                        9,
                        1,
                    )
                )
            releases.append(
                EngineReleasedRequest(binding.request_id, tuple(detached))
            )
        self.pending_plan = EngineReleasePlan(
            EngineReleaseId(7, 1), tuple(releases), tuple(retirements)
        )
        return self.pending_plan

    def pending_release(self, keys: Any) -> EngineReleasePlan | None:
        values = tuple(keys)
        if self.pending_plan is None:
            return None
        assert values == tuple(self.bindings)
        return self.pending_plan

    def confirm_release(self, plan: EngineReleasePlan) -> None:
        assert plan is self.pending_plan
        keys = tuple(self.bindings)
        if not self.release_cleanup_confirmed:
            updates = tuple(
                SessionMirrorUpdate(
                    key,
                    released.request_id,
                    int(self.bindings[key].request_row),
                    int(self.views[key].boundary),
                    released.detached,
                    True,
                )
                for key, released in zip(keys, plan.releases, strict=True)
                if self.bindings[key].request_row is not None
            )
            if updates:
                self.trace.append("cleanup")
                assert self.cleanup is not None
                assert self.cleanup(updates, plan.retirements) is True
            self.release_cleanup_confirmed = True
        if self.recycle_pending:
            self.recycle_pending -= 1
            self.trace.append("recycle-pending")
            raise ReleaseRecyclePending(plan)
        self.trace.append("confirm-release")
        for key in keys:
            del self.bindings[key]
            del self.views[key]
        self.pending_plan = None
        self.release_cleanup_confirmed = False

    def confirm_control(self, control_id: EngineControlId) -> EngineControlOutcome:
        if isinstance(self.pending_control, EnginePrefixEvictionPlan):
            assert self.pending_control.control_id == control_id
            self.trace.append(("confirm-evict", control_id))
            self.pending_control = None
            return EngineControlOutcome(
                control_id, EngineControlDisposition.EVICTED
            )
        assert self.pending_control is not None
        assert self.pending_control["control_id"] == control_id
        updates = (
            SessionMaterializationUpdate(
                self.pending_control["key"],
                self.pending_control["request_id"],
                self.pending_control["row"],
                self.pending_control["view_version"],
                self.pending_control["boundary"],
                self.pending_control["resident_count"],
                self.pending_control["pages"],
            ),
        )
        self.trace.append(("confirm-control", control_id))
        assert self.materialization is not None
        assert self.materialization(updates) is True
        self.views[self.pending_control["key"]] = EngineRequestView(
            self.pending_control["request_id"],
            self.pending_control["view_version"],
            self.pending_control["boundary"],
            self.pending_control["resident_count"],
        )
        self.pending_control = None
        return EngineControlOutcome(control_id, EngineControlDisposition.MATERIALIZED)

    def prepare_prefix_evict(self, prefix_ids: Any) -> EngineControlId:
        values = tuple(prefix_ids)
        if len(values) > self.prefix_eviction_batch_capacity:
            raise AssertionError("prefix eviction exceeded control capacity")
        control_id = EngineControlId(7, self.control_sequence)
        self.control_sequence += 1
        self.trace.append(("prepare-evict", values))
        self.pending_control = EnginePrefixEvictionPlan(
            control_id,
            values,
            tuple(
                EngineRetirement(
                    PageLease(7, 11, 1, item.sequence, 3),
                    0, 17, item.sequence - 1, item.sequence - 1,
                    (item.sequence - 1) * PAGE_TOKENS,
                    item.sequence * PAGE_TOKENS, 9, item.sequence,
                )
                for item in values
            ),
        )
        return control_id

    def commit_control(self, control_id: EngineControlId) -> EngineControlPlanInfo:
        assert isinstance(self.pending_control, EnginePrefixEvictionPlan)
        assert self.pending_control.control_id == control_id
        self.trace.append(("commit-evict", control_id))
        return EngineControlPlanInfo(
            control_id, EngineControlKind.PREFIX_EVICTION,
            0, 0, len(self.pending_control.prefix_ids),
            len(self.pending_control.retirements),
        )

    def read_control(self, control_id: EngineControlId) -> EnginePrefixEvictionPlan:
        assert isinstance(self.pending_control, EnginePrefixEvictionPlan)
        assert self.pending_control.control_id == control_id
        self.trace.append(("read-evict", control_id))
        return self.pending_control

    def fail_stop(self, reason: str) -> None:
        if self.failure_reason is None:
            self.failure_reason = reason
        self.trace.append(("fail-stop", reason))

    def poll(self) -> tuple[Any, ...]:
        self.trace.append("poll")
        return ()

    def stats(self) -> ManagerStats:
        self.trace.append("stats")
        return ManagerStats(*(0 for _ in range(17)))

    def arena_stats(self) -> tuple[ArenaStats, ...]:
        self.trace.append("arena-stats")
        arena = self.arenas_by_class[0]
        return (
            ArenaStats(
                arena.engine_epoch,
                arena.pool_epoch,
                arena.pool_id,
                arena.page_count,
                arena.class_id,
                arena.backend_domain,
                arena.first_page_id,
                arena.page_count,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        )

    def census(self) -> tuple[ManagerStats, tuple[ArenaStats, ...]]:
        raise AssertionError("session shutdown must not use census")

    def close(self) -> None:
        self.trace.append("close")
        self.closed = True


def _cache(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, _Runtime, _Pool, list[Any]]:
    del monkeypatch
    trace: list[Any] = []
    pool = _Pool(trace)
    allocator = SimpleNamespace()
    runtime = _Runtime(pool, trace)
    pool.free_slots = []
    state._install_test_state(
        config=_config(),
        limits=state.RuntimeLimits(4, 32, 64),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator
    cache = prefix_cache._build_prefix_cache(
        SimpleNamespace(
            disable_radix_cache=False,
            is_hybrid_ssm=False,
            enable_hierarchical_cache=False,
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=PAGE_TOKENS,
            ),
        )
    )
    return cache, runtime, pool, trace


def _request(rid: str, row: int | None, boundary: int | None) -> Any:
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=row,
        kv=(
            None
            if boundary is None
            else SimpleNamespace(kv_allocated_len=boundary)
        ),
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        cache_protected_len=0,
        origin_input_ids=(),
        output_ids=(),
    )


def _published_node(cache: Any, ordinal: int) -> Any:
    boundary = (ordinal + 1) * PAGE_TOKENS
    tokens = tuple(range(boundary))
    semantic = cache._semantic(tokens)
    publication = EnginePublishedPrefix(
        EnginePrefixId(7, ordinal + 1),
        semantic,
        boundary // PAGE_TOKENS,
    )
    return cache._record_publication(publication, tokens)


def test_reset_batches_all_prefixes_by_control_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, trace = _cache(monkeypatch)
    for ordinal in range(4):
        _published_node(cache, ordinal)

    cache.reset()

    batches = [
        tuple(item.sequence for item in event[1])
        for event in trace
        if isinstance(event, tuple) and event[0] == "prepare-evict"
    ]
    assert batches == [(4,), (3,), (2,), (1,)]
    assert cache._nodes == {}
    assert runtime.pending_control is None


def test_pressure_evict_batches_plan_by_control_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    cache, runtime, _pool, trace = _cache(monkeypatch)
    runtime.prefix_eviction_batch_capacity = 2
    for ordinal in range(4):
        _published_node(cache, ordinal)

    result = cache.evict(EvictParams(num_tokens=4 * PAGE_TOKENS))

    batches = [
        tuple(item.sequence for item in event[1])
        for event in trace
        if isinstance(event, tuple) and event[0] == "prepare-evict"
    ]
    assert batches == [(4, 3), (2, 1)]
    assert result.num_tokens_evicted == 4 * PAGE_TOKENS
    assert cache._nodes == {}


def test_build_binds_runtime_session_context_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, trace = _cache(monkeypatch)

    assert trace == ["bind", "bind-materialization"]
    assert runtime.cleanup is not None
    assert runtime.materialization is not None
    assert cache._session_requests == {}
    assert cache._session_pending_requests == {}
    assert cache._session_pending_shared_prefix == {}
    assert cache._session_active_shared_prefix == {}
    with pytest.raises(RuntimeError, match="bound twice"):
        session_cache.bind_session_cache(cache)


def test_pending_request_registers_exact_unbound_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, _trace = _cache(monkeypatch)
    req = _request("waiting", None, None)
    runtime.add_unbound(req, 31)

    entry = cache._register_pending_session_request(req)

    assert entry.req is req
    assert entry.key == ("str", "waiting")
    assert entry.request_id == EngineRequestId(31)
    assert cache._session_pending_requests == {entry.key: entry}
    assert cache._session_requests == {}
    with pytest.raises(RuntimeError, match="registered twice"):
        cache._register_pending_session_request(req)


def test_pending_request_promotes_only_after_exact_runtime_row_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, trace = _cache(monkeypatch)
    req = _request("promote", None, None)
    runtime.add_unbound(req, 33)
    pending = cache._register_pending_session_request(req)
    req.req_pool_idx = 3

    with pytest.raises(RuntimeError, match="cannot be promoted"):
        cache._promote_pending_session_requests((req,))

    runtime.bind_request_rows(((pending.key, 3),))
    (promoted,) = cache._promote_pending_session_requests((req,))

    assert promoted.key == pending.key
    assert promoted.request_id == pending.request_id
    assert promoted.row == 3
    assert cache._session_pending_requests == {}
    assert cache._session_requests == {pending.key: promoted}
    assert trace[-1] == ("bind-rows", ((pending.key, 3),))


def test_pending_request_promotion_confirms_matching_shared_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, trace = _cache(monkeypatch)
    req = _request("shared-promote", None, None)
    runtime.add_unbound(req, 45)
    pending = cache._register_pending_session_request(req)
    key = pending.key
    req.req_pool_idx = 3
    row = pool.req_to_token[3]
    row[:16] = torch.arange(16, 32, dtype=row.dtype)
    node = SimpleNamespace(
        boundary=16, digest=b"x" * 32, resident_count=1, lock_ref=2
    )
    semantic = SimpleNamespace(boundary=16, digest=b"x" * 32)
    prefix_id = EnginePrefixId(7, 9)
    control_id = EngineControlId(7, 11)
    page = object()
    plan = EngineMaterializationPlan(
        control_id,
        (
            EngineMaterializedRequest(
                pending.request_id,
                3,
                16,
                1,
                (page,),
            ),
        ),
    )
    runtime.pending_control = {
        "control_id": control_id,
        "key": key,
        "request_id": pending.request_id,
        "row": 3,
        "view_version": 3,
        "boundary": 16,
        "resident_count": 1,
        "pages": (page,),
    }
    private_prefix.install_shared_prefix_metadata(
        req,
        key=key,
        request_id=pending.request_id,
        prefix_id=prefix_id,
        node=node,
        semantic=semantic,
        indices=row[:16].to(dtype=torch.int64, copy=True),
        boundary=16,
        provisional=True,
    )
    cache._materialize_prefix_pages = (
        lambda pages, boundary, resident_count: row[:boundary].to(
            dtype=torch.int64, copy=True
        )
    )
    cache.dec_lock_ref = lambda value: setattr(
        value, "lock_ref", value.lock_ref - 1
    )
    session_cache.register_pending_shared_prefix(
        cache,
        req=req,
        prefix_id=prefix_id,
        semantic=semantic,
        node=node,
        boundary=16,
        control_id=control_id,
        plan=plan,
    )

    runtime.bind_request_rows(((key, 3),))
    session_cache.confirm_pending_shared_prefix(cache, req)
    (promoted,) = cache._promote_pending_session_requests((req,))

    assert promoted.row == 3
    assert trace[-2:] == [("bind-rows", ((key, 3),)), ("confirm-control", control_id)]
    assert cache._session_pending_shared_prefix == {}
    assert key in cache._session_active_shared_prefix
    assert getattr(req, "_orbitkv_prefix_lock_held") is True
    assert getattr(req, "_orbitkv_provisional_prefix_lock") is False
    assert node.lock_ref == 1


def test_pending_request_cancel_clears_pending_shared_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, pool, trace = _cache(monkeypatch)
    req = _request("cancel", None, None)
    runtime.add_unbound(req, 36)
    entry = cache._register_pending_session_request(req)
    node = SimpleNamespace(
        boundary=16, digest=b"y" * 32, resident_count=1, lock_ref=1
    )
    semantic = SimpleNamespace(boundary=16, digest=b"y" * 32)
    prefix_id = EnginePrefixId(7, 12)
    private_prefix.install_shared_prefix_metadata(
        req,
        key=entry.key,
        request_id=entry.request_id,
        prefix_id=prefix_id,
        node=node,
        semantic=semantic,
        indices=torch.arange(16, dtype=torch.int64, device=pool.device),
        boundary=16,
        provisional=True,
    )
    control_id = EngineControlId(7, 12)
    plan = EngineMaterializationPlan(
        control_id,
        (
            EngineMaterializedRequest(
                entry.request_id, 2, 16, 1, (object(),)
            ),
        ),
    )
    session_cache.register_pending_shared_prefix(
        cache, req=req, prefix_id=prefix_id, semantic=semantic, node=node,
        boundary=16, control_id=control_id, plan=plan,
    )
    runtime.block_pending_release = True

    with pytest.raises(RuntimeError, match="pending control group"):
        cache._cancel_pending_session_request(req)

    assert cache._session_pending_requests[entry.key] is entry
    assert cache._session_pending_shared_prefix[entry.key].plan is plan
    assert getattr(req, "_orbitkv_provisional_prefix_lock") is True


def test_cleanup_context_rejects_changed_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, _trace = _cache(monkeypatch)
    req = _request("identity", None, None)
    runtime.add_unbound(req, 71)
    pending = cache._register_pending_session_request(req)
    req.req_pool_idx = 4
    runtime.bind_request_rows(((pending.key, 4),))
    cache._promote_pending_session_requests((req,))
    req.kv = SimpleNamespace(kv_allocated_len=0)
    req._orbitkv_engine_request_id = EngineRequestId(72)

    with pytest.raises(RuntimeError, match="changed or was not registered"):
        cache._session_cleanup_context(
            SessionMirrorUpdate(
                ("str", "identity"), EngineRequestId(71), 4, 0, (), True
            )
        )


def test_production_lowering_preflights_registry_before_native_acquire() -> None:
    source = inspect.getsource(session_lowering._prepare_lowering)

    assert source.index("pending_registry = getattr(") < source.index(
        "runtime.bind_request_rows(new_assignments)"
    )


def test_clean_session_cache_shutdown_uses_stats_and_arena_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, runtime, _pool, trace = _cache(monkeypatch)

    cache.release_host_resources()
    cache.release_host_resources()

    assert runtime.closed is True
    assert trace == [
        "bind",
        "bind-materialization",
        "poll",
        "stats",
        "poll",
        "stats",
        "arena-stats",
        "close",
    ]
    assert cache._released is True
    assert cache._session_requests == {}
    assert cache._session_pending_requests == {}
