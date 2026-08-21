from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.plugin.facade as facade  # noqa: E402
import orbitkv_sglang.plugin.lowering as lowering  # noqa: E402
import orbitkv_sglang.plugin.prefix_cache as prefix_cache  # noqa: E402
import orbitkv_sglang.plugin.state as state  # noqa: E402
from orbitkv_sglang.config import ClassConfig, ManagerPlanConfig  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    ArenaRegistration,
    ArenaStats,
    FailStopped,
    ManagerStats,
    RequestLease,
    RetryableConflict,
)


def _config() -> ManagerPlanConfig:
    return ManagerPlanConfig(
        plan_path=Path("plan.json"),
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:lifecycle-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
        ),
    )


class _RowPool:
    def __init__(self):
        self.req_to_token = torch.zeros((8, 128), dtype=torch.int32)
        self.freed = []

    def free(self, req):
        self.freed.append(req.rid)
        req.req_pool_idx = None


class _LockTree:
    def __init__(self):
        self.dec_calls = 0

    def inc_lock_ref(self, node):
        node.lock_ref += 1

    def dec_lock_ref(self, node):
        if node.lock_ref <= 0:
            raise RuntimeError("lock underflow")
        node.lock_ref -= 1
        self.dec_calls += 1

    def _preflight_release_node(self, req, *, provisional):
        marker = (
            "_orbitkv_provisional_prefix_lock"
            if provisional
            else "_orbitkv_prefix_lock_held"
        )
        node = getattr(req, "_orbitkv_prefix_node", None)
        if getattr(req, marker, False) is not True or req.last_node is not node:
            raise RuntimeError("invalid test prefix lock")
        if node is None or node.lock_ref <= 0:
            raise RuntimeError("invalid test prefix node")
        return node

    def _commit_release_node(self, req, node, *, provisional):
        assert self._preflight_release_node(req, provisional=provisional) is node
        self.dec_lock_ref(node)
        marker = (
            "_orbitkv_provisional_prefix_lock"
            if provisional
            else "_orbitkv_prefix_lock_held"
        )
        delattr(req, marker)


class _AdmissionRuntime:
    def __init__(self, key, lease):
        self.failure_reason = None
        self.records = {key: SimpleNamespace(lease=lease, boundary=16)}
        self.rows = {}
        self.events = []
        self.release_conflicts = 0

    def has_request(self, key):
        return key in self.records

    def record_for(self, key):
        return self.records[key]

    def wait_batch(self, _keys):
        return None

    def bind_request_rows(self, assignments):
        for key, row, install_row in assignments:
            self.events.append(("bind_row", key, row, install_row))
            if install_row:
                if key not in self.records or key in self.rows:
                    raise RuntimeError("invalid attached row install")
                self.rows[key] = row
            elif key not in self.records or self.rows.get(key) != row:
                raise RuntimeError("invalid attached row validation")

    def rollback_request_rows(self, assignments):
        for key, row in assignments:
            self.events.append(("rollback_row", key, row))
            if self.rows.get(key) != row:
                raise RuntimeError("row rollback identity changed")
            del self.rows[key]

    def release_batch(self, keys):
        self.events.append(("manager_release", tuple(keys)))
        if self.release_conflicts:
            self.release_conflicts -= 1
            raise RetryableConflict("injected release conflict")
        for key in keys:
            del self.records[key]

    def fail_stop(self, reason):
        self.failure_reason = reason


def _install_runtime(runtime):
    state._install_test_state(
        config=_config(),
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    state._ALLOCATOR = object()


class _DestroyManager:
    def __init__(self):
        self.destroy_count = 0

    def destroy(self):
        self.destroy_count += 1


class _DestroyFactory:
    def __init__(self):
        self.manager = _DestroyManager()

    def create(self, _config, _settings, _registrations):
        return self.manager


def test_new_runtime_destroys_factory_handle_when_constructor_raises(monkeypatch):
    factory = _DestroyFactory()
    state._install_test_state(
        config=_config(),
        limits=state.RuntimeLimits(8, 64, 128),
        factory=factory,
    )
    monkeypatch.setattr(
        state,
        "CanonicalRuntime",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("constructor fault")),
    )

    with pytest.raises(RuntimeError, match="constructor fault"):
        state._new_runtime((ArenaRegistration(0, 1, 1, 8),))

    assert factory.manager.destroy_count == 1
    assert state._RUNTIME is None


def test_new_runtime_destroys_factory_handle_on_arena_identity_mismatch(
    monkeypatch,
):
    factory = _DestroyFactory()
    state._install_test_state(
        config=_config(),
        limits=state.RuntimeLimits(8, 64, 128),
        factory=factory,
    )
    bad_identity = ArenaIdentity(1, 2, 99, 0, 1, 8, 16, 0, 1)

    class Runtime:
        arenas = (bad_identity,)
        failure_reason = None

        def fail_stop(self, reason):
            self.failure_reason = reason

    monkeypatch.setattr(state, "CanonicalRuntime", lambda *_args: Runtime())

    with pytest.raises(FailStopped, match="differs from SGLang"):
        state._new_runtime((ArenaRegistration(0, 1, 1, 8),))

    assert factory.manager.destroy_count == 1
    assert state._RUNTIME is None


def test_attached_request_precommit_retry_rolls_back_row_and_official_lock(
    monkeypatch,
):
    import sglang.srt.mem_cache.allocation as allocation

    key = ("str", "attached")
    lease = RequestLease(1, 3, 1)
    runtime = _AdmissionRuntime(key, lease)
    _install_runtime(runtime)
    node = SimpleNamespace(lock_ref=2)
    tree = _LockTree()
    pool = _RowPool()
    req = SimpleNamespace(
        rid="attached",
        req_pool_idx=None,
        prefix_indices=torch.arange(16, dtype=torch.int64),
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
        _orbitkv_prefix_node=node,
        _orbitkv_provisional_prefix_lock=True,
    )
    batch = SimpleNamespace(
        reqs=[req],
        tree_cache=tree,
        req_to_token_pool=pool,
        maybe_evict_swa=lambda: None,
        device=torch.device("cpu"),
        extend_num_tokens=1,
    )
    monkeypatch.setattr(lowering, "_validate_batch", lambda _batch: None)
    monkeypatch.setattr(
        lowering, "_preflight_extend_batch", lambda _batch: ([16], [1], [17])
    )
    monkeypatch.setattr(lowering, "_ensure_prepare_capacity", lambda *_args: None)

    def allocate_row(_pool, reqs, _tree):
        reqs[0].req_pool_idx = 1
        return [1]

    monkeypatch.setattr(allocation, "alloc_req_slots", allocate_row)
    monkeypatch.setattr(
        lowering,
        "_prepare_batch",
        lambda *_args: (_ for _ in ()).throw(RetryableConflict("prepare conflict")),
    )

    with pytest.raises(RetryableConflict, match="prepare conflict"):
        lowering._alloc_for_extend(batch)

    assert runtime.records[key].lease == lease
    assert runtime.rows == {}
    assert req.req_pool_idx is None
    assert node.lock_ref == 1
    assert req._orbitkv_provisional_prefix_lock is True
    assert not hasattr(req, "_orbitkv_prefix_lock_held")
    assert runtime.events == [
        ("bind_row", key, 1, True),
        ("rollback_row", key, 1),
    ]

    tree.inc_lock_ref(node)
    req.req_pool_idx = 1
    runtime.bind_request_rows(((key, 1, True),))
    lowering._promote_prefix_locks(batch)
    assert node.lock_ref == 1
    assert req._orbitkv_prefix_lock_held is True
    assert not hasattr(req, "_orbitkv_provisional_prefix_lock")


def test_waiting_cancel_commits_manager_release_before_dropping_provisional_lock():
    key = ("str", "waiting")
    lease = RequestLease(1, 4, 1)
    runtime = _AdmissionRuntime(key, lease)
    runtime.release_conflicts = 1
    _install_runtime(runtime)
    node = SimpleNamespace(lock_ref=1)
    events = runtime.events

    class Tree(_LockTree):
        def _commit_release_node(self, req, node, *, provisional):
            events.append("drop_provisional")
            super()._commit_release_node(req, node, provisional=provisional)

    tree = Tree()
    req = SimpleNamespace(
        rid="waiting",
        req_pool_idx=None,
        kv=None,
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
        _orbitkv_prefix_node=node,
        _orbitkv_provisional_prefix_lock=True,
        last_node=node,
    )

    with pytest.raises(RetryableConflict):
        lowering._release_kv_cache(req, tree)
    assert node.lock_ref == 1
    assert runtime.has_request(key)
    assert req._orbitkv_provisional_prefix_lock is True
    assert "drop_provisional" not in events

    lowering._release_kv_cache(req, tree)
    assert events[-2:] == [
        ("manager_release", (key,)),
        "drop_provisional",
    ]
    assert node.lock_ref == 0
    assert not runtime.has_request(key)
    assert not hasattr(req, "_orbitkv_request_lease")


def test_waiting_cancel_rejects_lost_node_before_manager_release():
    key = ("str", "waiting-hostile")
    lease = RequestLease(1, 6, 1)
    runtime = _AdmissionRuntime(key, lease)
    _install_runtime(runtime)
    node = SimpleNamespace(lock_ref=1)
    req = SimpleNamespace(
        rid="waiting-hostile",
        req_pool_idx=None,
        kv=None,
        last_node=None,
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
        _orbitkv_prefix_node=node,
        _orbitkv_provisional_prefix_lock=True,
    )

    with pytest.raises(RuntimeError, match="invalid test prefix lock"):
        lowering._release_kv_cache(req, _LockTree())

    assert runtime.has_request(key)
    assert runtime.events == []
    assert node.lock_ref == 1


def test_finished_release_rejects_lost_held_node_before_manager_release():
    key = ("str", "finished-hostile")
    lease = RequestLease(1, 7, 1)
    runtime = _AdmissionRuntime(key, lease)
    _install_runtime(runtime)
    allocator = object()
    state._ALLOCATOR = allocator
    tree = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree.root_node = object()
    tree.token_to_kv_pool_allocator = allocator
    tree.req_to_token_pool = _RowPool()
    node = SimpleNamespace(lock_ref=1)
    req = SimpleNamespace(
        rid="finished-hostile",
        req_pool_idx=1,
        kv=SimpleNamespace(kv_allocated_len=16),
        prefix_indices=torch.arange(16, dtype=torch.int64),
        last_node=None,
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
        _orbitkv_prefix_node=node,
        _orbitkv_prefix_lock_held=True,
        effective_kv_committed_len=lambda: 16,
    )

    with pytest.raises(RuntimeError, match="differs from its held lock"):
        lowering._release_kv_cache(req, tree, is_insert=False)

    assert runtime.has_request(key)
    assert runtime.events == []
    assert tree.req_to_token_pool.freed == []


def test_finished_pending_event_waits_before_boundary_validation_and_release():
    key = ("str", "finished")
    lease = RequestLease(1, 5, 1)
    allocator = SimpleNamespace(_orbitkv_free_group_state="idle")
    pool = _RowPool()
    events = []

    class Runtime(_AdmissionRuntime):
        def wait_batch(self, keys):
            events.append(("wait", tuple(keys)))
            self.records[key].boundary = 9

        def release_batch(self, keys):
            events.append(("release", tuple(keys)))
            super().release_batch(keys)

        def unbind_request_rows(self, assignments):
            events.append(("unbind", tuple(assignments)))

    runtime = Runtime(key, lease)
    runtime.rows[key] = 1
    _install_runtime(runtime)
    state._ALLOCATOR = allocator
    tree = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree.disable_finished_insert = True
    tree.req_to_token_pool = pool
    tree.token_to_kv_pool_allocator = allocator
    tree.root_node = object()
    req = SimpleNamespace(
        rid="finished",
        req_pool_idx=1,
        kv=SimpleNamespace(kv_allocated_len=9),
        prefix_indices=torch.arange(8, dtype=torch.int64),
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
        effective_kv_committed_len=lambda: 9,
    )

    lowering._release_kv_cache(req, tree, is_insert=False)

    assert events == [
        ("wait", (key,)),
        ("release", (key,)),
        ("unbind", ((key, 1),)),
    ]
    assert runtime.events == [
        ("bind_row", key, 1, False),
        ("manager_release", (key,)),
    ]
    assert req.req_pool_idx is None
    assert req.kv is None
    assert not hasattr(req, "_orbitkv_request_lease")


def test_finished_callback_identity_mutation_is_rejected_before_native_release():
    key = ("str", "callback-hostile")
    lease = RequestLease(1, 8, 1)
    allocator = SimpleNamespace(_orbitkv_free_group_state="idle")
    pool = _RowPool()

    class Runtime(_AdmissionRuntime):
        def wait_batch(self, _keys):
            self.events.append("wait")

    runtime = Runtime(key, lease)
    runtime.rows[key] = 1
    _install_runtime(runtime)
    state._ALLOCATOR = allocator
    tree = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree.disable_finished_insert = True
    tree.req_to_token_pool = pool
    tree.token_to_kv_pool_allocator = allocator
    tree.root_node = object()
    req = SimpleNamespace(
        rid="callback-hostile",
        req_pool_idx=1,
        kv=SimpleNamespace(kv_allocated_len=16),
        prefix_indices=torch.arange(16, dtype=torch.int64),
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
    )

    def hostile_boundary():
        req._orbitkv_request_key = ("str", "mutated")
        return 16

    req.effective_kv_committed_len = hostile_boundary

    with pytest.raises(FailStopped, match="release group became uncertain"):
        lowering._release_kv_cache(req, tree, is_insert=False)

    assert runtime.has_request(key)
    assert runtime.events == ["wait"]
    assert pool.freed == []


def test_finished_foreign_row_is_rejected_before_native_release():
    key = ("str", "foreign-row")
    lease = RequestLease(1, 9, 1)
    runtime = _AdmissionRuntime(key, lease)
    runtime.rows[key] = 1
    _install_runtime(runtime)
    allocator = SimpleNamespace(_orbitkv_free_group_state="idle")
    state._ALLOCATOR = allocator
    pool = _RowPool()
    tree = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree.disable_finished_insert = True
    tree.req_to_token_pool = pool
    tree.token_to_kv_pool_allocator = allocator
    tree.root_node = object()
    req = SimpleNamespace(
        rid="foreign-row",
        req_pool_idx=2,
        kv=SimpleNamespace(kv_allocated_len=16),
        prefix_indices=torch.arange(16, dtype=torch.int64),
        _orbitkv_request_key=key,
        _orbitkv_request_lease=lease,
        effective_kv_committed_len=lambda: 16,
    )

    with pytest.raises(FailStopped, match="attached row validation"):
        lowering._release_kv_cache(req, tree, is_insert=False)

    assert runtime.has_request(key)
    assert not any(event[0] == "manager_release" for event in runtime.events)
    assert pool.freed == []


def _manager_stats(active_requests: int) -> ManagerStats:
    return ManagerStats(
        active_requests=active_requests,
        active_snapshots=0,
        active_prefixes=0,
        evicted_prefixes=0,
        prepared_steps=0,
        submitted_steps=0,
        free_pages=8 - active_requests,
        reserved_pages=0,
        writing_pages=0,
        active_pages=active_requests,
        retiring_pages=0,
        quarantined_pages=0,
        exhausted_pages=0,
        pending_reclamations=0,
        total_request_page_refs=active_requests,
        total_prefix_page_refs=0,
        total_reader_pins=0,
    )


def _arena_stats(active_requests: int) -> ArenaStats:
    return ArenaStats(
        engine_epoch=1,
        pool_epoch=2,
        pool_id=1,
        page_count=8,
        class_id=0,
        backend_domain=1,
        first_page_id=1,
        free_pages=8 - active_requests,
        reserved_pages=0,
        writing_pages=0,
        active_pages=active_requests,
        retiring_pages=0,
        quarantined_pages=0,
        exhausted_pages=0,
        request_page_refs=active_requests,
        prefix_page_refs=0,
        reader_pins=0,
    )


def _pressure_config(hybrid: bool) -> ManagerPlanConfig:
    classes = [_config().classes[0]]
    if hybrid:
        classes.append(
            ClassConfig(
                class_id=1,
                pool_id=2,
                backend_domain=2,
                name="sliding",
                layers=(1,),
                retention="sliding",
                bytes_per_token_per_layer=128,
                window_tokens=32,
                period_blocks=3,
            )
        )
    base = _config()
    return ManagerPlanConfig(
        plan_path=base.plan_path,
        library_path=base.library_path,
        plan_json=base.plan_json,
        plan_fingerprint=base.plan_fingerprint,
        page_tokens=base.page_tokens,
        classes=tuple(classes),
    )


def test_hybrid_combined_availability_uses_one_runtime_census():
    config = _pressure_config(True)

    class Runtime:
        page_tokens = 16
        failure_reason = None

        def __init__(self):
            self.census_calls = 0

        def poll(self):
            return None

        def census(self):
            self.census_calls += 1
            arenas = tuple(
                ArenaStats(
                    1,
                    2 + item.class_id,
                    item.pool_id,
                    8,
                    item.class_id,
                    item.backend_domain,
                    1 + item.class_id * 8,
                    3 - item.class_id,
                    0,
                    5 + item.class_id,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                for item in config.classes
            )
            return _manager_stats(0), arenas

        def fail_stop(self, reason):
            self.failure_reason = reason

    class Pool:
        def register_mapping(self, mapping):
            self.mapping = mapping

    runtime = Runtime()
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    hybrid_type = facade._facade_types()[1]
    allocator = hybrid_type(
        128,
        128,
        16,
        torch.float16,
        torch.device("cpu"),
        Pool(),
        False,
        full_class_id=0,
        swa_class_id=1,
    )

    assert allocator.available_size() == 32
    assert runtime.census_calls == 1
    runtime.census_calls = 0
    assert allocator.new_pages_available(3, 2)
    assert runtime.census_calls == 1


@pytest.mark.parametrize("hybrid", (False, True))
def test_prepare_pressure_evicts_exact_full_and_swa_deficits_before_prepare(hybrid):
    from sglang.srt.mem_cache.base_prefix_cache import EvictResult

    config = _pressure_config(hybrid)
    free = {item.class_id: 0 for item in config.classes}
    events = []

    class Runtime:
        failure_reason = None

        def poll(self):
            events.append("poll")

        def stats(self):
            events.append("stats")
            return _manager_stats(0)

        def arena_stats(self):
            return tuple(
                ArenaStats(
                    engine_epoch=1,
                    pool_epoch=2 + item.class_id,
                    pool_id=item.pool_id,
                    page_count=8,
                    class_id=item.class_id,
                    backend_domain=item.backend_domain,
                    first_page_id=1 + item.class_id * 8,
                    free_pages=free[item.class_id],
                    reserved_pages=0,
                    writing_pages=0,
                    active_pages=8 - free[item.class_id],
                    retiring_pages=0,
                    quarantined_pages=0,
                    exhausted_pages=0,
                    request_page_refs=0,
                    prefix_page_refs=8 - free[item.class_id],
                    reader_pins=0,
                )
                for item in config.classes
            )

        def census(self):
            return self.stats(), self.arena_stats()

        def fail_stop(self, reason):
            self.failure_reason = reason

    class Tree:
        def evict(self, params):
            events.append(("evict", params.num_tokens, params.swa_num_tokens))
            for class_id in free:
                free[class_id] = 2
            return EvictResult(
                num_tokens_evicted=32,
                swa_num_tokens_evicted=32 if hybrid else 0,
            )

    runtime = Runtime()
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    lowering._ensure_prepare_capacity(
        SimpleNamespace(tree_cache=Tree()), [0], [32]
    )

    assert ("evict", 32, 32 if hybrid else 0) in events
    assert events.index(("evict", 32, 32 if hybrid else 0)) < len(events) - 1
    assert all(value == 2 for value in free.values())


def test_prepare_pressure_shortfall_fails_before_any_manager_prepare():
    from sglang.srt.mem_cache.base_prefix_cache import EvictResult

    config = _pressure_config(True)
    events = []

    class Runtime:
        failure_reason = None

        def poll(self):
            return None

        def stats(self):
            return _manager_stats(0)

        def arena_stats(self):
            return tuple(
                ArenaStats(
                    engine_epoch=1,
                    pool_epoch=2 + item.class_id,
                    pool_id=item.pool_id,
                    page_count=8,
                    class_id=item.class_id,
                    backend_domain=item.backend_domain,
                    first_page_id=1 + item.class_id * 8,
                    free_pages=0,
                    reserved_pages=0,
                    writing_pages=0,
                    active_pages=8,
                    retiring_pages=0,
                    quarantined_pages=0,
                    exhausted_pages=0,
                    request_page_refs=0,
                    prefix_page_refs=8,
                    reader_pins=0,
                )
                for item in config.classes
            )

        def census(self):
            return self.stats(), self.arena_stats()

        def prepare_batch(self, _items):
            events.append("prepare")

        def fail_stop(self, reason):
            self.failure_reason = reason

    class Tree:
        def evict(self, _params):
            events.append("evict")
            return EvictResult()

    runtime = Runtime()
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    with pytest.raises(RuntimeError, match="insufficient evictable"):
        lowering._ensure_prepare_capacity(
            SimpleNamespace(tree_cache=Tree()), [0], [32]
        )

    assert events == ["evict"]


@pytest.mark.parametrize("active_requests", (0, 1))
def test_allocator_clear_is_quiescent_validation_only(active_requests):
    class Runtime:
        def __init__(self):
            self.failure_reason = None

        def poll(self):
            return None

        def stats(self):
            return _manager_stats(active_requests)

        def arena_stats(self):
            return (_arena_stats(active_requests),)

        def census(self):
            return self.stats(), self.arena_stats()

        def fail_stop(self, reason):
            self.failure_reason = reason

    runtime = Runtime()
    _install_runtime(runtime)
    authority = facade._NativeAuthorityForbidden()

    if active_requests:
        with pytest.raises(FailStopped, match="non-quiescent"):
            authority.clear()
        assert runtime.failure_reason is not None
    else:
        authority.clear()
        assert runtime.failure_reason is None
