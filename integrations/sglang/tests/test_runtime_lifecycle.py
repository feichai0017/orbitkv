from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Sequence

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]

from orbitkv_sglang.config import load_config
from orbitkv_sglang.ffi import CtypesManagerFactory
from orbitkv_sglang.ffi.manager import CtypesManager
from orbitkv_sglang.runtime import (
    DETACHED_CLEAR,
    DETACHED_REPLACE,
    ArenaRegistration,
    ArenaIdentity,
    ArenaStats,
    CanonicalRuntime,
    CompletionBatch,
    EvictedPrefix,
    FailStopped,
    ManagerCreateSettings,
    ManagerError,
    ManagerStats,
    MirrorCleanupBinding,
    MirrorCleanupItem,
    PageLease,
    PageShadow,
    PrefixEvictionBatch,
    PrefixLease,
    PrefixSemanticKey,
    ReclamationCertificate,
    ReclamationLease,
    ReleaseBatchItem,
    RequestLease,
    RetryableConflict,
    SnapshotLease,
    TAIL_FRESH,
    reclamation_receipts,
)
from orbitkv_sglang.runtime.completion import completion_cursor_delta


@pytest.fixture(scope="session")
def ffi_library() -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(REPOSITORY_ROOT / "crates/orbitkv-ffi/Cargo.toml"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return REPOSITORY_ROOT / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"


def _runtime(
    tmp_path: Path,
    library: Path,
    *,
    hybrid: bool = True,
    window_tokens: int = 18,
    requests: int = 16,
) -> tuple[Any, CtypesManager, CanonicalRuntime]:
    classes = [
        {
            "name": "full",
            "layers": [0],
            "retention": "full",
            "bytes_per_token_per_layer": 128,
            "window_tokens": None,
        }
    ]
    if hybrid:
        classes.append(
            {
                "name": "swa",
                "layers": [1],
                "retention": "sliding",
                "bytes_per_token_per_layer": 128,
                "window_tokens": window_tokens,
            }
        )
    plan = tmp_path / f"plan-{hybrid}-{window_tokens}.json"
    plan.write_text(json.dumps({"page_tokens": 16, "classes": classes}))
    config = load_config(
        {"ORBITKV_PLAN": str(plan), "ORBITKV_LIBRARY": str(library)}
    )
    arenas = tuple(
        ArenaRegistration(
            item.class_id, item.pool_id, item.backend_domain, 64, 0
        )
        for item in config.classes
    )
    manager = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(
            requests,
            4,
            requests,
            64 * len(arenas),
            64,
        ),
        arenas,
    )
    assert isinstance(manager, CtypesManager)
    return config, manager, CanonicalRuntime(config, manager)


class ReadyEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class NoIterationPages(dict[tuple[int, int], PageShadow]):
    @staticmethod
    def _forbid(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("steady completion iterated the resident cursor")

    __iter__ = _forbid
    copy = _forbid
    items = _forbid
    keys = _forbid
    values = _forbid


class NoRequestCensus(dict[Any, Any]):
    def values(self) -> Any:
        raise AssertionError("steady prepare scanned every live request")


def _step_batch(
    runtime: CanonicalRuntime,
    values: Sequence[tuple[Any, int]],
    *,
    domain: int = 1,
) -> Any:
    batch, plans = runtime.prepare_batch(tuple(values))
    assert len(plans) == len(values)
    runtime.mark_lowered(batch)
    submitted = runtime.submit_batch(batch)
    assert len(submitted) == len(values)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), domain)
    runtime.poll()
    return batch


def test_prepare_uses_incremental_identity_indexes_not_live_request_scans(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, _manager_value, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    view = runtime.request_acquire_batch(("request",))[0]
    assert runtime._request_leases == {view.request}
    assert runtime._snapshot_leases == {view.snapshot}
    runtime._requests = NoRequestCensus(runtime._requests)

    batch = _step_batch(runtime, (("request", 18),))
    record = runtime._requests["request"]
    assert view.snapshot not in runtime._snapshot_leases
    assert runtime._snapshot_leases == {record.head}
    assert batch.records[0].prepared.step not in runtime._step_leases

    runtime.release_batch(("request",))
    assert runtime._request_leases == set()
    assert runtime._snapshot_leases == set()
    assert runtime._step_leases == set()
    runtime.close()


def test_identity_index_corruption_fails_before_native_prepare(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    view = runtime.request_acquire_batch(("request",))[0]
    runtime._snapshot_leases.remove(view.snapshot)
    calls: list[int] = []
    original = manager.prepare_batch

    def traced(_self: CtypesManager, items: Sequence[Any]) -> Any:
        calls.append(len(items))
        return original(items)

    manager.prepare_batch = MethodType(traced, manager)
    with pytest.raises(FailStopped, match="identity index"):
        runtime.prepare_batch((("request", 1),))
    assert calls == []
    runtime.close()


def test_close_rejects_a_stale_identity_index(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, _manager_value, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    stale = SnapshotLease(runtime.engine_epoch, 999, 1)
    runtime._snapshot_leases.add(stale)
    with pytest.raises(ManagerError, match="live requests"):
        runtime.close()
    runtime._snapshot_leases.remove(stale)
    runtime.close()


def _finish_failed_runtime(manager: CtypesManager, runtime: CanonicalRuntime) -> None:
    records = tuple(runtime._requests.values())
    if records:
        output = manager.release_batch(
            tuple(ReleaseBatchItem(item.lease, item.head) for item in records)
        )
        if output.retirements:
            manager.acknowledge_reclamations_batch(
                reclamation_receipts(output.retirements)
            )
        manager.recycle_requests_batch(tuple(item.lease for item in records))
    manager.destroy()


def _abort_native_prepares(manager: CtypesManager, prepared: Sequence[Any]) -> None:
    if prepared:
        manager.abort_steps_batch(
            tuple(
                SimpleNamespace(step=item.step, backend_unobserved=1, reserved=0)
                for item in prepared
            )
        )


def _release_native_requests(
    manager: CtypesManager, identities: Sequence[tuple[Any, Any]]
) -> None:
    if not identities:
        return
    output = manager.release_batch(
        tuple(ReleaseBatchItem(request, snapshot) for request, snapshot in identities)
    )
    if output.retirements:
        manager.acknowledge_reclamations_batch(
            reclamation_receipts(output.retirements)
        )
    manager.recycle_requests_batch(tuple(request for request, _snapshot in identities))


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("hybrid", [False, True])
def test_real_b2_b4_runtime_lifecycle_is_collective_and_reference_exact(
    tmp_path: Path, ffi_library: Path, batch_size: int, hybrid: bool
) -> None:
    config, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=hybrid
    )
    keys = tuple(f"r{index}" for index in range(batch_size))
    _step_batch(runtime, tuple((key, 18) for key in keys))
    assert all(runtime.record_for(key).boundary == 18 for key in keys)
    _step_batch(runtime, tuple((key, 48) for key in keys))
    assert all(runtime.record_for(key).boundary == 48 for key in keys)
    assert manager.performance_counters["complete_batch_calls"] == 2
    runtime.release_batch(keys)
    stats = runtime.stats()
    assert stats.free_pages == 64 * len(config.classes)
    assert stats.active_requests == stats.active_snapshots == 0
    assert stats.pending_reclamations == 0
    assert stats.total_request_page_refs == stats.total_prefix_page_refs == 0
    runtime.close()


def _completion_projection(
    runtime: CanonicalRuntime, record: Any, pending: Any, completion: Any
) -> dict[tuple[int, int], PageShadow]:
    delta = completion_cursor_delta(
        record.cursor,
        pending,
        completion,
        runtime.arenas_by_class,
        runtime.config.classes,
        runtime.page_tokens,
        runtime._zero_page(),
    )
    projection = dict(record.cursor.pages)
    delta.apply(projection)
    return projection


def test_fail_stopped_close_destroys_a_live_native_handle(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    runtime.request_acquire_batch(("live",))
    assert runtime.stats().active_requests == 1
    runtime.fail_stop("simulated shutdown with live authority")
    runtime.close()
    assert not manager._handle or not manager._handle.value
    runtime.close()


def test_steady_completion_uses_only_cursor_deltas(tmp_path: Path, ffi_library: Path) -> None:
    _config, _manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    for boundary in range(64, 513, 64):
        _step_batch(runtime, (("request", boundary),))
    record = runtime.record_for("request")
    guarded = NoIterationPages(record.cursor.pages)
    record.cursor.pages = guarded

    _step_batch(runtime, (("request", 513),))

    assert record.boundary == 513
    record.cursor.pages = {
        key: dict.__getitem__(guarded, key) for key in dict.__iter__(guarded)
    }
    runtime.release_batch(("request",))
    runtime.close()


def test_abort_batch_discards_all_reserved_candidate_shadows(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library)
    batch, _plans = runtime.prepare_batch((('a', 18), ('b', 18)))
    assert runtime._candidate_pages
    runtime.abort_unobserved(batch)
    assert not runtime._candidate_pages
    assert all(runtime.record_for(key).pending is None for key in ('a', 'b'))
    runtime.release_batch(('a', 'b'))
    runtime.close()


@pytest.mark.parametrize("collision", ["batch", "live", "pending"])
def test_hostile_acquire_snapshot_identity_is_not_journaled(
    tmp_path: Path, ffi_library: Path, collision: str
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    pending_outputs: tuple[Any, ...] = ()
    occupied_snapshot = None
    if collision in ("live", "pending"):
        runtime.request_acquire_batch(("owner",))
        occupied_snapshot = runtime.record_for("owner").head
    if collision == "pending":
        pending_batch, _plans = runtime.prepare_batch((("owner", 16),))
        pending_outputs = (pending_batch.records[0].prepared,)
        occupied_snapshot = pending_outputs[0].target_snapshot

    baseline_keys = set(runtime._requests)
    baseline_candidates = dict(runtime._candidate_pages)
    submit_calls = manager.performance_counters["submit_batch_calls"]
    original_acquire = manager.request_acquire_batch
    captured: list[Any] = []

    def corrupt(_self: CtypesManager, count: int) -> tuple[Any, ...]:
        views = list(original_acquire(count))
        captured.extend(views)
        if collision == "batch":
            views[1] = replace(views[1], snapshot=views[0].snapshot)
        else:
            views[0] = replace(views[0], snapshot=occupied_snapshot)
        return tuple(views)

    manager.request_acquire_batch = MethodType(corrupt, manager)
    keys = ("a", "b") if collision == "batch" else ("victim",)
    try:
        with pytest.raises(FailStopped, match="invalid acquired views"):
            runtime.request_acquire_batch(keys)
        assert set(runtime._requests) == baseline_keys
        assert runtime._candidate_pages == baseline_candidates
        assert manager.performance_counters["submit_batch_calls"] == submit_calls
    finally:
        manager.request_acquire_batch = original_acquire
        _abort_native_prepares(manager, pending_outputs)
        existing = tuple(
            (record.lease, record.head) for record in runtime._requests.values()
        )
        acquired = tuple((view.request, view.snapshot) for view in captured)
        _release_native_requests(manager, existing + acquired)
        runtime._requests.clear()
        runtime._candidate_pages.clear()
        runtime.close()


@pytest.mark.parametrize(
    "collision", ["batch-target", "live-target", "pending-target", "batch-step"]
)
def test_hostile_prepare_control_identity_is_not_journaled_or_submitted(
    tmp_path: Path, ffi_library: Path, collision: str
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    batched = collision in ("batch-target", "batch-step")
    victim_keys = ("a", "b") if batched else ("victim",)
    setup_keys = victim_keys + (() if batched else ("owner",))
    runtime.request_acquire_batch(setup_keys)

    pending_outputs: tuple[Any, ...] = ()
    occupied_snapshot = None
    if collision == "live-target":
        occupied_snapshot = runtime.record_for("owner").head
    elif collision == "pending-target":
        pending_batch, _plans = runtime.prepare_batch((("owner", 16),))
        pending_outputs = (pending_batch.records[0].prepared,)
        occupied_snapshot = pending_outputs[0].target_snapshot

    baseline_pending = {
        key: record.pending for key, record in runtime._requests.items()
    }
    baseline_candidates = dict(runtime._candidate_pages)
    submit_calls = manager.performance_counters["submit_batch_calls"]
    original_prepare = manager.prepare_batch
    original_quarantine = manager.quarantine_steps_batch
    captured: list[Any] = []
    quarantine_calls: list[tuple[Any, ...]] = []

    def corrupt(_self: CtypesManager, items: Sequence[Any]) -> tuple[Any, ...]:
        outputs = list(original_prepare(items))
        captured.extend(outputs)
        if collision == "batch-target":
            outputs[1] = replace(
                outputs[1], target_snapshot=outputs[0].target_snapshot
            )
        elif collision in ("live-target", "pending-target"):
            outputs[0] = replace(outputs[0], target_snapshot=occupied_snapshot)
        else:
            outputs[1] = replace(outputs[1], step=outputs[0].step)
        return tuple(outputs)

    def record_quarantine(_self: CtypesManager, steps: Sequence[Any]) -> None:
        quarantine_calls.append(tuple(steps))

    manager.prepare_batch = MethodType(corrupt, manager)
    manager.quarantine_steps_batch = MethodType(record_quarantine, manager)
    try:
        with pytest.raises(FailStopped, match="invalid prepare"):
            runtime.prepare_batch(tuple((key, 16) for key in victim_keys))
        assert {
            key: record.pending for key, record in runtime._requests.items()
        } == baseline_pending
        assert runtime._candidate_pages == baseline_candidates
        assert manager.performance_counters["submit_batch_calls"] == submit_calls
        assert len(quarantine_calls) == 1
    finally:
        manager.prepare_batch = original_prepare
        manager.quarantine_steps_batch = original_quarantine
        _abort_native_prepares(manager, pending_outputs + tuple(captured))
        _release_native_requests(
            manager,
            tuple((record.lease, record.head) for record in runtime._requests.values()),
        )
        runtime._requests.clear()
        runtime._candidate_pages.clear()
        runtime.close()


def test_runtime_b4_fork_joint_cow_and_prefix_attach_evict(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library)
    _step_batch(runtime, (('source', 18),))
    runtime.request_acquire_batch(('t0', 't1', 't2', 't3'))
    forked = runtime.request_fork_batch(
        tuple(('source', f't{index}') for index in range(4))
    )
    assert len(forked) == 4
    runtime.release_batch(('source',))
    batch, plans = runtime.prepare_batch((('t0', 19),))
    assert [item.tail_action.kind for item in plans[0].class_specs] == [2, 2]
    assert sum(len(item.copy_intents) for item in plans[0].class_specs) == 2
    runtime.mark_lowered(batch)
    assert len(batch.records[0].copy_receipts) == 2
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    runtime.poll()

    _step_batch(runtime, (('prefix-source', 32),))
    key = PrefixSemanticKey(b'n' * 32, b'd' * 32, 32)
    published = runtime.prefix_publish_batch((('prefix-source', key),))[0]
    hint = runtime.prefix_lookup_batch((key,))[0]
    runtime.request_acquire_batch(('attached',))
    attached = runtime.prefix_attach_batch((('attached', hint),))[0]
    assert attached.target.view.boundary == 32
    assert len(attached.target.pages) == published.resident_count

    runtime.release_batch(
        ('t0', 't1', 't2', 't3', 'prefix-source', 'attached')
    )
    runtime.bind_prefix_eviction_cleanup(OrderedMirror([]))
    evicted = runtime.prefix_evict_batch((published.prefix,))
    assert len(evicted.retirements) == published.resident_count
    runtime.prefix_recycle_batch((published.prefix,))
    assert runtime.stats().free_pages == 128
    runtime.close()


def test_fork_after_multiple_appends_advances_the_empty_target_once(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, _manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("source", 18),))
    _step_batch(runtime, (("source", 48),))
    runtime.request_acquire_batch(("target",))
    source_before = runtime.record_for("source").cursor
    target_before = runtime.record_for("target").cursor
    assert source_before.view_version == 2
    assert target_before.view_version == 0

    output = runtime.request_fork_batch((("source", "target"),))[0]

    target_after = runtime.record_for("target").cursor
    assert output.target.view.view_version == target_before.view_version + 1
    assert target_after.view_version == target_before.view_version + 1
    assert target_after.boundary == source_before.boundary
    assert {
        key: (page.page, page.backend_index)
        for key, page in target_after.pages.items()
    } == {
        key: (page.page, page.backend_index)
        for key, page in source_before.pages.items()
    }
    runtime.release_batch(("source", "target"))
    runtime.close()


def test_joint_cow_can_retire_an_unpublished_swa_destination(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, _manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=True, window_tokens=18
    )
    _step_batch(runtime, (("source", 18),))
    runtime.request_acquire_batch(("target",))
    runtime.request_fork_batch((("source", "target"),))

    _step_batch(runtime, (("target", 80),))

    assert runtime.record_for("target").boundary == 80
    runtime.release_batch(("source", "target"))
    assert runtime.stats().free_pages == 128
    runtime.close()


def test_completion_rejects_a_same_count_wrong_retention_ordinal(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=True, window_tokens=18
    )
    _step_batch(runtime, (("source", 48),))
    runtime.request_acquire_batch(("target",))
    runtime.request_fork_batch((("source", "target"),))
    forged = runtime._requests["target"].cursor.pages[(1, 2)]
    original_complete = manager.complete_batch
    original_ack = manager.acknowledge_reclamations_batch
    captured: list[CompletionBatch] = []
    ack_calls: list[int] = []

    def corrupt(_self: CtypesManager, receipt: Any, submissions: Any) -> CompletionBatch:
        output = original_complete(receipt, submissions)
        captured.append(output)
        completion = output.completions[0]
        detached = list(completion.detached)
        index = next(
            i
            for i, item in enumerate(detached)
            if item.action == DETACHED_CLEAR and item.class_id == 1
        )
        detached[index] = replace(
            detached[index],
            old=forged.page,
            logical_ordinal=forged.logical_ordinal,
            old_backend_index=forged.backend_index,
            token_begin=32,
            token_end_exclusive=48,
        )
        return replace(
            output,
            completions=(replace(completion, detached=tuple(detached)),),
        )

    def traced_ack(_self: CtypesManager, receipts: Any) -> None:
        ack_calls.append(len(receipts))
        original_ack(receipts)

    manager.complete_batch = MethodType(corrupt, manager)
    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    batch, _plans = runtime.prepare_batch((("target", 64),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    with pytest.raises(FailStopped, match="completion"):
        runtime.poll()
    assert ack_calls == []

    manager.complete_batch = original_complete
    manager.acknowledge_reclamations_batch = original_ack
    source = runtime._requests["source"]
    target = runtime._requests["target"]
    release = manager.release_batch(
        (
            ReleaseBatchItem(source.lease, source.head),
            ReleaseBatchItem(
                target.lease, captured[0].completions[0].published_snapshot
            ),
        )
    )
    original_ack(reclamation_receipts(release.retirements))
    manager.recycle_requests_batch((source.lease, target.lease))
    manager.destroy()


@pytest.mark.parametrize("class_id", [0, 1])
def test_completion_rejects_a_short_cow_detach_span(
    tmp_path: Path, ffi_library: Path, class_id: int
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("source", 18),))
    runtime.request_acquire_batch(("target",))
    runtime.request_fork_batch((("source", "target"),))
    original_complete = manager.complete_batch
    original_ack = manager.acknowledge_reclamations_batch
    captured: list[CompletionBatch] = []
    ack_calls: list[int] = []

    def corrupt(_self: CtypesManager, receipt: Any, submissions: Any) -> CompletionBatch:
        output = original_complete(receipt, submissions)
        captured.append(output)
        completion = output.completions[0]
        detached = tuple(
            replace(item, token_end_exclusive=item.token_end_exclusive - 1)
            if item.action == DETACHED_REPLACE and item.class_id == class_id
            else item
            for item in completion.detached
        )
        return replace(
            output, completions=(replace(completion, detached=detached),)
        )

    def traced_ack(_self: CtypesManager, receipts: Any) -> None:
        ack_calls.append(len(receipts))
        original_ack(receipts)

    manager.complete_batch = MethodType(corrupt, manager)
    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    batch, _plans = runtime.prepare_batch((("target", 19),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    with pytest.raises(FailStopped, match="completion"):
        runtime.poll()
    assert ack_calls == []

    manager.complete_batch = original_complete
    manager.acknowledge_reclamations_batch = original_ack
    source = runtime._requests["source"]
    target = runtime._requests["target"]
    release = manager.release_batch(
        (
            ReleaseBatchItem(source.lease, source.head),
            ReleaseBatchItem(
                target.lease, captured[0].completions[0].published_snapshot
            ),
        )
    )
    original_ack(reclamation_receipts(release.retirements))
    manager.recycle_requests_batch((source.lease, target.lease))
    manager.destroy()


def test_prefix_attach_after_multiple_appends_preserves_materialized_identity(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, _manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("source", 18),))
    _step_batch(runtime, (("source", 48),))
    semantic = PrefixSemanticKey(b"a" * 32, b"b" * 32, 48)
    published = runtime.prefix_publish_batch((("source", semantic),))[0]
    hint = runtime.prefix_lookup_batch((semantic,))[0]
    runtime.request_acquire_batch(("target",))

    attached = runtime.prefix_attach_batch((("target", hint),))[0]

    assert attached.prefix == published.prefix
    assert attached.target.view.view_version == 1
    assert runtime.record_for("target").boundary == 48
    runtime.release_batch(("source", "target"))
    runtime.bind_prefix_eviction_cleanup(OrderedMirror([]))
    runtime.prefix_evict_batch((published.prefix,))
    runtime.prefix_recycle_batch((published.prefix,))
    runtime.close()


@pytest.mark.parametrize("fault", ["snapshot", "version", "logical-placement"])
def test_forged_fork_target_identity_fail_stops(
    tmp_path: Path, ffi_library: Path, fault: str
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    _step_batch(runtime, (("source", 32),))
    runtime.request_acquire_batch(("target",))
    original_fork = manager.request_fork_batch
    captured: list[Any] = []

    def corrupt(_self: CtypesManager, items: Sequence[Any]) -> tuple[Any, ...]:
        outputs = tuple(original_fork(items))
        captured.extend(outputs)
        output = outputs[0]
        materialized = output.target
        view = materialized.view
        pages = materialized.pages
        if fault == "snapshot":
            view = replace(view, snapshot=runtime.record_for("source").head)
        elif fault == "version":
            view = replace(view, view_version=view.view_version + 1)
        else:
            pages = (
                replace(pages[0], logical_ordinal=pages[0].logical_ordinal + 1),
            ) + pages[1:]
        return (replace(output, target=replace(materialized, view=view, pages=pages)),)

    manager.request_fork_batch = MethodType(corrupt, manager)
    with pytest.raises(FailStopped, match="invalid fork output"):
        runtime.request_fork_batch((("source", "target"),))
    assert len(captured) == 1
    assert runtime.failure_reason is not None

    manager.request_fork_batch = original_fork
    source = runtime._requests["source"]
    target = runtime._requests["target"]
    release = manager.release_batch(
        (
            ReleaseBatchItem(source.lease, source.head),
            ReleaseBatchItem(target.lease, captured[0].target.view.snapshot),
        )
    )
    if release.retirements:
        manager.acknowledge_reclamations_batch(
            reclamation_receipts(release.retirements)
        )
    manager.recycle_requests_batch((source.lease, target.lease))
    runtime.close()


def test_prefix_attached_request_can_install_and_rollback_its_first_row(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, _manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    _step_batch(runtime, (("source", 32),))
    semantic = PrefixSemanticKey(b"n" * 32, b"r" * 32, 32)
    published = runtime.prefix_publish_batch((("source", semantic),))[0]
    hint = runtime.prefix_lookup_batch((semantic,))[0]
    runtime.request_acquire_batch(("attached",))
    runtime.prefix_attach_batch((("attached", hint),))

    runtime.bind_request_rows((("attached", 7, True),))
    assert runtime._request_rows["attached"] == 7
    runtime.rollback_request_rows((("attached", 7),))
    assert "attached" not in runtime._request_rows
    runtime.bind_request_rows((("attached", 7, True),))

    mirror = OrderedMirror([])
    runtime.bind_reclamation_cleanup(
        "attached", MirrorCleanupBinding(mirror, object())
    )
    runtime.release_batch(("attached",))
    runtime.unbind_request_rows((("attached", 7),))
    runtime.release_batch(("source",))
    runtime.bind_prefix_eviction_cleanup(mirror)
    runtime.prefix_evict_batch((published.prefix,))
    runtime.prefix_recycle_batch((published.prefix,))
    runtime.close()


class OrderedMirror:
    def __init__(self, log: list[str]):
        self.log = log
        self.preflight_detached = 0
        self.preflight_certificates = 0

    def preflight(
        self,
        items: Sequence[MirrorCleanupItem],
        retirements: Sequence[ReclamationCertificate],
    ) -> object:
        self.log.append("preflight")
        self.preflight_detached = sum(len(item.detached) for item in items)
        self.preflight_certificates = len(retirements)
        return object()

    def commit(self, _plan: object) -> None:
        self.log.append("commit")

    def synchronize(self, _plan: object) -> None:
        self.log.append("sync")

    def finalize(self, _plan: object) -> None:
        self.log.append("finalize")


def test_detach_and_global_certificates_are_consumed_before_one_ack(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library)
    log: list[str] = []
    original_ack = manager.acknowledge_reclamations_batch

    def traced_ack(_self: CtypesManager, receipts: Any) -> None:
        log.append("ack")
        original_ack(receipts)

    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    first, _plans = runtime.prepare_batch((('a', 18), ('b', 18)))
    coordinator = OrderedMirror(log)
    for key in ('a', 'b'):
        runtime.bind_reclamation_cleanup(
            key, MirrorCleanupBinding(coordinator, key)
        )
    runtime.mark_lowered(first)
    runtime.submit_batch(first)
    runtime.mark_forward(first)
    runtime.register_event(first, ReadyEvent(), 1)
    runtime.poll()
    log.clear()
    _step_batch(runtime, (('a', 48), ('b', 48)))
    assert coordinator.preflight_detached == 2
    assert coordinator.preflight_certificates == 2
    assert log == ['preflight', 'commit', 'sync', 'finalize', 'ack']
    log.clear()
    runtime.release_batch(('a', 'b'))
    assert log == ['preflight', 'commit', 'sync', 'finalize', 'ack']
    runtime.close()


def test_retryable_prefix_conflict_does_not_fail_stop_runtime(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, _manager, runtime = _runtime(tmp_path, ffi_library)
    _step_batch(runtime, (('source', 32),))
    key = PrefixSemanticKey(b'x' * 32, b'y' * 32, 32)
    published = runtime.prefix_publish_batch((('source', key),))[0]
    with pytest.raises(RetryableConflict):
        runtime.prefix_publish_batch((('source', key),))
    assert runtime.failure_reason is None
    runtime.release_batch(('source',))
    runtime.bind_prefix_eviction_cleanup(OrderedMirror([]))
    runtime.prefix_evict_batch((published.prefix,))
    runtime.prefix_recycle_batch((published.prefix,))
    runtime.close()


@pytest.mark.parametrize("fault", ["raise-after-inner", "identity-corruption"])
def test_native_prefix_publish_unknown_outcome_or_bad_output_fail_stops(
    tmp_path: Path, ffi_library: Path, fault: str
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("source", 32),))
    key = PrefixSemanticKey(b"h" * 32, b"i" * 32, 32)
    original_publish = manager.prefix_publish_batch
    committed: list[Any] = []

    def faulty_publish(_self: CtypesManager, items: Sequence[Any]) -> tuple[Any, ...]:
        outputs = tuple(original_publish(items))
        committed.extend(outputs)
        if fault == "raise-after-inner":
            raise OSError("simulated lost prefix publication return")
        return (replace(outputs[0], key=PrefixSemanticKey(b"z" * 32, b"i" * 32, 32)),)

    manager.prefix_publish_batch = MethodType(faulty_publish, manager)
    with pytest.raises(FailStopped, match="prefix publication"):
        runtime.prefix_publish_batch((("source", key),))
    assert len(committed) == 1
    assert runtime.failure_reason is not None

    manager.prefix_publish_batch = original_publish
    record = runtime._requests["source"]
    release = manager.release_batch((ReleaseBatchItem(record.lease, record.head),))
    assert release.retirements == ()
    manager.recycle_requests_batch((record.lease,))
    eviction = manager.prefix_evict_batch((committed[0].prefix,))
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(eviction.retirements)
    )
    manager.prefix_recycle_batch((committed[0].prefix,))
    runtime.close()


def test_publish_release_rejects_forged_prefix_resident_count_before_consumption(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    _step_batch(runtime, (("source", 32),))
    key = PrefixSemanticKey(b"m" * 32, b"o" * 32, 32)
    original = manager.prefix_publish_release_batch
    captured: list[Any] = []
    acknowledgements = manager.performance_counters[
        "acknowledge_reclamations_batch_calls"
    ]

    def corrupt(_self: CtypesManager, items: Sequence[Any]) -> Any:
        output = original(items)
        captured.append(output)
        first = output.outputs[0]
        publication = replace(
            first.publication,
            resident_count=first.publication.resident_count + 1,
        )
        return replace(output, outputs=(replace(first, publication=publication),))

    manager.prefix_publish_release_batch = MethodType(corrupt, manager)
    with pytest.raises(FailStopped, match="publish-release"):
        runtime.prefix_publish_release_batch((("source", key),))
    assert (
        manager.performance_counters["acknowledge_reclamations_batch_calls"]
        == acknowledgements
    )

    manager.prefix_publish_release_batch = original
    real = captured[0]
    record = runtime._requests["source"]
    manager.recycle_requests_batch((record.lease,))
    publication = real.outputs[0].publication
    eviction = manager.prefix_evict_batch((publication.prefix,))
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(eviction.retirements)
    )
    manager.prefix_recycle_batch((publication.prefix,))
    runtime.close()


def test_publish_release_bad_detach_does_not_partially_install_host_prefix(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    _step_batch(runtime, (("source", 32),))
    key = PrefixSemanticKey(b"p" * 32, b"q" * 32, 32)
    original = manager.prefix_publish_release_batch
    captured: list[Any] = []
    before_pages = dict(runtime._page_registry._pages)
    before_prefixes = dict(runtime._page_registry._prefixes)
    before_cursor = dict(runtime._requests["source"].cursor.pages)
    acknowledgements = manager.performance_counters[
        "acknowledge_reclamations_batch_calls"
    ]

    def corrupt(_self: CtypesManager, items: Sequence[Any]) -> Any:
        output = original(items)
        captured.append(output)
        first = output.outputs[0]
        return replace(
            output,
            outputs=(replace(first, release=replace(first.release, detached=())),),
        )

    manager.prefix_publish_release_batch = MethodType(corrupt, manager)
    with pytest.raises(FailStopped, match="publish-release"):
        runtime.prefix_publish_release_batch((("source", key),))
    assert runtime._page_registry._pages == before_pages
    assert runtime._page_registry._prefixes == before_prefixes
    assert runtime._requests["source"].cursor.pages == before_cursor
    assert (
        manager.performance_counters["acknowledge_reclamations_batch_calls"]
        == acknowledgements
    )

    manager.prefix_publish_release_batch = original
    real = captured[0]
    record = runtime._requests["source"]
    manager.recycle_requests_batch((record.lease,))
    publication = real.outputs[0].publication
    eviction = manager.prefix_evict_batch((publication.prefix,))
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(eviction.retirements)
    )
    manager.prefix_recycle_batch((publication.prefix,))
    runtime.close()


def test_native_prefix_recycle_raise_after_inner_permanently_fail_stops(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("source", 32),))
    key = PrefixSemanticKey(b"j" * 32, b"k" * 32, 32)
    published = runtime.prefix_publish_batch((("source", key),))[0]
    runtime.release_batch(("source",))
    runtime.bind_prefix_eviction_cleanup(OrderedMirror([]))
    runtime.prefix_evict_batch((published.prefix,))
    original_recycle = manager.prefix_recycle_batch

    def lost_recycle(_self: CtypesManager, prefixes: Sequence[PrefixLease]) -> None:
        original_recycle(prefixes)
        raise OSError("simulated lost prefix recycle return")

    manager.prefix_recycle_batch = MethodType(lost_recycle, manager)
    with pytest.raises(FailStopped, match="prefix recycling"):
        runtime.prefix_recycle_batch((published.prefix,))
    with pytest.raises(FailStopped):
        runtime.prefix_lookup_batch((key,))
    manager.prefix_recycle_batch = original_recycle
    runtime.close()


def test_success_then_invalid_completion_observation_fail_stops(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    original = manager.complete_batch

    def corrupt(_self: CtypesManager, receipt: Any, submissions: Any) -> CompletionBatch:
        output = original(receipt, submissions)
        item = replace(
            output.completions[0],
            published_boundary=output.completions[0].published_boundary + 1,
        )
        return CompletionBatch((item,), output.retirements)

    manager.complete_batch = MethodType(corrupt, manager)
    batch, _plans = runtime.prepare_batch((('request', 18),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    with pytest.raises(FailStopped, match="completion"):
        runtime.poll()
    assert runtime.failure_reason is not None
    manager.complete_batch = original
    # The real output committed before the corrupted Python observation.
    record = runtime._requests['request']
    record.cursor.snapshot = batch.records[0].prepared.target_snapshot
    record.cursor.view_version = batch.records[0].prepared.target_view_version
    record.cursor.boundary = batch.records[0].prepared.target_boundary
    record.pending = None
    _finish_failed_runtime(manager, runtime)


def test_retirement_without_a_matching_detach_fail_stops_before_ack(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("request", 18),))
    original_complete = manager.complete_batch
    original_ack = manager.acknowledge_reclamations_batch
    captured: list[CompletionBatch] = []
    ack_calls: list[int] = []

    def corrupt(
        _self: CtypesManager, receipt: Any, submissions: Any
    ) -> CompletionBatch:
        output = original_complete(receipt, submissions)
        captured.append(output)
        record = runtime.record_for("request")
        assert record.pending is not None
        projection = dict(record.cursor.pages)
        for shadow in record.pending.new_pages:
            projection.setdefault((shadow.class_id, shadow.logical_ordinal), shadow)
        return CompletionBatch(
            (replace(output.completions[0], resident_count=len(projection), detached=()),),
            output.retirements,
        )

    def traced_ack(_self: CtypesManager, receipts: Any) -> None:
        ack_calls.append(len(receipts))
        original_ack(receipts)

    manager.complete_batch = MethodType(corrupt, manager)
    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    batch, _plans = runtime.prepare_batch((("request", 48),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    with pytest.raises(FailStopped, match="completion"):
        runtime.poll()
    assert captured and captured[0].retirements
    assert ack_calls == []

    manager.complete_batch = original_complete
    manager.acknowledge_reclamations_batch = original_ack
    record = runtime._requests["request"]
    pending = record.pending
    assert pending is not None
    completion = captured[0].completions[0]
    projection = _completion_projection(runtime, record, pending, completion)
    page_refs = runtime._page_registry.plan(
        tuple(record.cursor.pages.values()), tuple(projection.values())
    )
    runtime._page_registry.commit(page_refs)
    record.cursor.pages = projection
    record.cursor.snapshot = completion.published_snapshot
    record.cursor.view_version = completion.published_view_version
    record.cursor.boundary = completion.published_boundary
    runtime._discard_candidates(pending)
    record.pending = None
    runtime._events.clear()
    original_ack(reclamation_receipts(captured[0].retirements))
    _finish_failed_runtime(manager, runtime)


def test_exact_completion_certificate_for_a_shared_page_has_zero_ack(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    _step_batch(runtime, (("source", 18),))
    runtime.request_acquire_batch(("target",))
    runtime.request_fork_batch((("source", "target"),))
    original_complete = manager.complete_batch
    original_ack = manager.acknowledge_reclamations_batch
    captured: list[CompletionBatch] = []
    ack_calls: list[int] = []

    def corrupt(
        _self: CtypesManager, receipt: Any, submissions: Any
    ) -> CompletionBatch:
        output = original_complete(receipt, submissions)
        captured.append(output)
        assert output.retirements == ()
        detached = output.completions[0].detached[0]
        certificate = ReclamationCertificate(
            reclamation=ReclamationLease(runtime.engine_epoch, 0, 1),
            page=detached.old,
            class_id=detached.class_id,
            backend_domain=detached.backend_domain,
            logical_ordinal=detached.logical_ordinal,
            backend_index=detached.old_backend_index,
            token_begin=detached.token_begin,
            token_end_exclusive=detached.token_end_exclusive,
            completion_domain=receipt.completion_domain,
            completion_value=receipt.completion_value,
        )
        return CompletionBatch(output.completions, (certificate,))

    def traced_ack(_self: CtypesManager, receipts: Sequence[Any]) -> None:
        ack_calls.append(len(receipts))
        original_ack(receipts)

    manager.complete_batch = MethodType(corrupt, manager)
    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    batch, _plans = runtime.prepare_batch((("source", 19),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    with pytest.raises(FailStopped, match="completion"):
        runtime.poll()
    assert captured and captured[0].retirements == ()
    assert ack_calls == []
    assert runtime.failure_reason is not None

    manager.complete_batch = original_complete
    manager.acknowledge_reclamations_batch = original_ack
    record = runtime._requests["source"]
    pending = record.pending
    assert pending is not None
    completion = captured[0].completions[0]
    projection = _completion_projection(runtime, record, pending, completion)
    page_refs = runtime._page_registry.plan(
        tuple(record.cursor.pages.values()),
        tuple(projection.values()),
        pending.new_pages,
    )
    runtime._page_registry.commit(page_refs)
    record.cursor.pages = projection
    record.cursor.snapshot = completion.published_snapshot
    record.cursor.view_version = completion.published_view_version
    record.cursor.boundary = completion.published_boundary
    runtime._discard_candidates(pending)
    record.pending = None
    runtime._events.clear()
    _finish_failed_runtime(manager, runtime)


@pytest.mark.parametrize(
    "fault", ["foreign-tail", "fresh-over-live", "live-write", "candidate-alias"]
)
def test_hostile_prepare_page_identity_fails_before_lowering(
    tmp_path: Path, ffi_library: Path, fault: str
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("a", 18), ("b", 18)))
    original_prepare = manager.prepare_batch
    original_quarantine = manager.quarantine_steps_batch
    abort_calls: list[int] = []

    def abort_for_cleanup(_self: CtypesManager, steps: Sequence[Any]) -> None:
        abort_calls.append(len(steps))
        manager.abort_steps_batch(tuple(SimpleNamespace(
            step=step, backend_unobserved=1, reserved=0
        ) for step in steps))

    def corrupt(_self: CtypesManager, items: Sequence[Any]) -> tuple[Any, ...]:
        outputs = list(original_prepare(items))
        prepared = outputs[0]
        actions = list(prepared.tail_actions)
        intents = list(prepared.write_intents)
        if fault == "foreign-tail":
            foreign = runtime.record_for("b").cursor.pages[(0, 1)].page
            actions[0] = replace(actions[0], source=foreign, destination=foreign)
        elif fault == "fresh-over-live":
            intent = intents[0]
            arena = runtime.arenas_by_class[0]
            destination = PageLease(
                runtime.engine_epoch,
                arena.pool_epoch,
                intent.page_generation,
                intent.page_id,
                arena.pool_id,
            )
            actions[0] = replace(
                actions[0],
                kind=TAIL_FRESH,
                valid_token_count=0,
                source=PageLease(0, 0, 0, 0, 0),
                destination=destination,
            )
        elif fault == "live-write":
            live = runtime.record_for("a").cursor.pages[(0, 1)].page
            intents[0] = replace(
                intents[0],
                page_id=live.page_id,
                page_generation=live.generation + 1,
            )
        outputs[0] = replace(
            prepared, tail_actions=tuple(actions), write_intents=tuple(intents)
        )
        if fault == "candidate-alias":
            second = outputs[1]
            second_intents = list(second.write_intents)
            second_intents[0] = replace(
                second_intents[0],
                page_id=intents[0].page_id,
                page_generation=intents[0].page_generation + 1,
            )
            outputs[1] = replace(second, write_intents=tuple(second_intents))
        return tuple(outputs)

    manager.prepare_batch = MethodType(corrupt, manager)
    manager.quarantine_steps_batch = MethodType(abort_for_cleanup, manager)
    target = 19 if fault == "foreign-tail" else 33
    values = (("a", target), ("b", target)) if fault == "candidate-alias" else (("a", target),)
    with pytest.raises(FailStopped, match="invalid prepare"):
        runtime.prepare_batch(values)
    assert abort_calls == [len(values)]
    assert not runtime._candidate_pages

    manager.prepare_batch = original_prepare
    manager.quarantine_steps_batch = original_quarantine
    _finish_failed_runtime(manager, runtime)


def test_lost_ack_return_permanently_fail_stops_before_future_allocation(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library)
    _step_batch(runtime, (('request', 18),))
    original_ack = manager.acknowledge_reclamations_batch

    def lost_return(_self: CtypesManager, receipts: Any) -> None:
        original_ack(receipts)
        raise OSError("simulated lost ACK return")

    manager.acknowledge_reclamations_batch = MethodType(lost_return, manager)
    batch, _plans = runtime.prepare_batch((('request', 48),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    with pytest.raises(FailStopped, match="completion"):
        runtime.poll()
    with pytest.raises(FailStopped):
        runtime.prepare_batch((('new', 1),))
    manager.acknowledge_reclamations_batch = original_ack
    runtime._requests['request'].pending = None
    _finish_failed_runtime(manager, runtime)


def test_lost_arena_census_return_fail_stops_runtime(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=False)
    original = manager.arena_stats

    def lost_return(_self: CtypesManager) -> Any:
        original()
        raise OSError("simulated lost arena census return")

    manager.arena_stats = MethodType(lost_return, manager)
    with pytest.raises(FailStopped, match="arena census"):
        runtime.arena_stats()
    assert runtime.failure_reason is not None
    manager.arena_stats = original
    runtime.close()


@pytest.mark.parametrize("class_id", [0, 1])
def test_release_rejects_a_short_detach_and_certificate_span(
    tmp_path: Path, ffi_library: Path, class_id: int
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=True)
    _step_batch(runtime, (("request", 18),))
    original_release = manager.release_batch
    original_ack = manager.acknowledge_reclamations_batch
    captured: list[Any] = []
    ack_calls: list[int] = []

    def corrupt(_self: CtypesManager, items: Any) -> Any:
        output = original_release(items)
        captured.append(output)
        release = output.releases[0]
        target = next(
            item
            for item in release.detached
            if item.class_id == class_id and item.logical_ordinal == 1
        )
        detached = tuple(
            replace(item, token_end_exclusive=17) if item is target else item
            for item in release.detached
        )
        retirements = tuple(
            replace(item, token_end_exclusive=17)
            if item.page == target.old
            else item
            for item in output.retirements
        )
        return replace(
            output,
            releases=(replace(release, detached=detached),),
            retirements=retirements,
        )

    def traced_ack(_self: CtypesManager, receipts: Any) -> None:
        ack_calls.append(len(receipts))
        original_ack(receipts)

    manager.release_batch = MethodType(corrupt, manager)
    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    with pytest.raises(FailStopped, match="request release"):
        runtime.release_batch(("request",))
    assert ack_calls == []

    manager.release_batch = original_release
    manager.acknowledge_reclamations_batch = original_ack
    original_ack(reclamation_receipts(captured[0].retirements))
    record = runtime._requests["request"]
    manager.recycle_requests_batch((record.lease,))
    manager.destroy()


class GlobalPrefixLut:
    """A fake global Full-to-SWA mirror keyed only by physical page identity."""

    def __init__(
        self,
        mapping: dict[PageLease, PageLease | None],
        log: list[str],
        *,
        late_mismatch: bool = False,
        fail_stage: str | None = None,
    ):
        self.mapping = mapping
        self.log = log
        self.late_mismatch = late_mismatch
        self.fail_stage = fail_stage
        self.commit_calls = 0
        self.sync_calls = 0
        self.finalize_calls = 0

    def preflight(
        self,
        items: Sequence[MirrorCleanupItem],
        retirements: Sequence[ReclamationCertificate],
    ) -> tuple[PageLease, ...]:
        self.log.append("preflight")
        if not items and self.late_mismatch:
            raise ManagerError("global LUT changed after prefix eviction planning")
        retired = {item.page for item in retirements}
        return tuple(
            source
            for source, target in self.mapping.items()
            if source in retired or target in retired
        )

    def commit(self, plan: tuple[PageLease, ...]) -> None:
        self.log.append("commit")
        if self.fail_stage == "commit":
            raise OSError("simulated global LUT commit failure")
        self.commit_calls += 1
        for source in plan:
            del self.mapping[source]

    def synchronize(self, _plan: tuple[PageLease, ...]) -> None:
        self.log.append("sync")
        if self.fail_stage == "sync":
            raise OSError("simulated global LUT synchronization failure")
        self.sync_calls += 1

    def finalize(self, _plan: tuple[PageLease, ...]) -> None:
        self.log.append("finalize")
        if self.fail_stage == "finalize":
            raise OSError("simulated global LUT finalization failure")
        self.finalize_calls += 1


def _global_lut(
    pages: Sequence[Any], *, hybrid: bool
) -> dict[PageLease, PageLease | None]:
    full = sorted(
        (item for item in pages if item.class_id == 0),
        key=lambda item: item.logical_ordinal,
    )
    if not hybrid:
        return {item.page: None for item in full}
    sliding = {
        item.logical_ordinal: item.page for item in pages if item.class_id == 1
    }
    return {item.page: sliding[item.logical_ordinal] for item in full}


@pytest.mark.parametrize("hybrid", [False, True])
def test_native_prefix_last_ref_clears_global_lut_before_one_ack(
    tmp_path: Path, ffi_library: Path, hybrid: bool
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=hybrid)
    _step_batch(runtime, (("source", 32),))
    record = runtime.record_for("source")
    key = PrefixSemanticKey(b"p" * 32, b"q" * 32, 32)
    published = runtime.prefix_publish_batch((("source", key),))[0]

    log: list[str] = []
    mapping = _global_lut(tuple(record.cursor.pages.values()), hybrid=hybrid)
    before_release = dict(mapping)
    coordinator = GlobalPrefixLut(mapping, log)
    runtime.bind_prefix_eviction_cleanup(coordinator)
    runtime.bind_reclamation_cleanup(
        "source", MirrorCleanupBinding(coordinator, "source")
    )
    original_ack = manager.acknowledge_reclamations_batch

    def traced_ack(_self: CtypesManager, receipts: Any) -> None:
        log.append("ack")
        original_ack(receipts)

    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    runtime.release_batch(("source",))
    assert mapping == before_release
    assert log == ["preflight", "commit", "sync", "finalize"]

    log.clear()
    syncs_before = coordinator.sync_calls
    output = runtime.prefix_evict_batch((published.prefix,))
    assert len(output.retirements) == published.resident_count
    assert mapping == {}
    assert coordinator.sync_calls == syncs_before + 1
    assert log == ["preflight", "commit", "sync", "finalize", "ack"]
    assert log.count("sync") == log.count("ack") == 1
    runtime.prefix_recycle_batch((published.prefix,))
    runtime.close()


@pytest.mark.parametrize("hybrid", [False, True])
def test_native_prefix_retirement_without_global_cleanup_has_zero_ack(
    tmp_path: Path, ffi_library: Path, hybrid: bool
) -> None:
    _config, manager, runtime = _runtime(tmp_path, ffi_library, hybrid=hybrid)
    _step_batch(runtime, (("source", 32),))
    key = PrefixSemanticKey(b"u" * 32, b"v" * 32, 32)
    published = runtime.prefix_publish_batch((("source", key),))[0]
    runtime.release_batch(("source",))

    captured: list[PrefixEvictionBatch] = []
    ack_calls: list[int] = []
    original_evict = manager.prefix_evict_batch
    original_ack = manager.acknowledge_reclamations_batch

    def capture_evict(
        _self: CtypesManager, prefixes: Sequence[PrefixLease]
    ) -> PrefixEvictionBatch:
        output = original_evict(prefixes)
        captured.append(output)
        return output

    def traced_ack(_self: CtypesManager, receipts: Sequence[Any]) -> None:
        ack_calls.append(len(receipts))
        original_ack(receipts)

    manager.prefix_evict_batch = MethodType(capture_evict, manager)
    manager.acknowledge_reclamations_batch = MethodType(traced_ack, manager)
    with pytest.raises(FailStopped, match="global cleanup authority"):
        runtime.prefix_evict_batch((published.prefix,))
    assert len(captured) == 1
    assert len(captured[0].retirements) == published.resident_count
    assert ack_calls == []

    manager.prefix_evict_batch = original_evict
    manager.acknowledge_reclamations_batch = original_ack
    original_ack(reclamation_receipts(captured[0].retirements))
    manager.prefix_recycle_batch((published.prefix,))
    runtime.close()


_UNUSED_MANAGER_METHODS = frozenset(
    {
        "abort_steps_batch",
        "complete_batch",
        "prefix_attach_batch",
        "prefix_lookup_batch",
        "prefix_publish_batch",
        "prefix_publish_release_batch",
        "prepare_batch",
        "quarantine_steps_batch",
        "quarantine_submissions_batch",
        "recycle_requests_batch",
        "release_batch",
        "request_acquire_batch",
        "request_fork_batch",
        "submit_batch",
    }
)


class FakePrefixManager:
    def __init__(self, *, hybrid: bool, forge_certificate: bool):
        class_count = 2 if hybrid else 1
        self.arenas = tuple(
            ArenaIdentity(
                engine_epoch=71,
                pool_epoch=81 + class_id,
                pool_id=class_id + 1,
                class_id=class_id,
                backend_domain=class_id + 1,
                page_count=4,
                page_tokens=16,
                backend_base_index=100 * (class_id + 1),
                first_page_id=10 * (class_id + 1),
            )
            for class_id in range(class_count)
        )
        self.arenas_by_class = {item.class_id: item for item in self.arenas}
        self.performance_counters: dict[str, int] = {}
        self.prefix = PrefixLease(71, 0, 1)
        self.pages = tuple(
            PageLease(
                71,
                item.pool_epoch,
                1,
                item.first_page_id,
                item.pool_id,
            )
            for item in self.arenas
        )
        certificates = tuple(
            ReclamationCertificate(
                reclamation=ReclamationLease(71, index, 1),
                page=page,
                class_id=arena.class_id,
                backend_domain=arena.backend_domain,
                logical_ordinal=0,
                backend_index=arena.backend_base_index,
                token_begin=0,
                token_end_exclusive=16,
                completion_domain=3,
                completion_value=9,
            )
            for index, (arena, page) in enumerate(
                zip(self.arenas, self.pages, strict=True)
            )
        )
        if forge_certificate:
            certificates = (
                replace(
                    certificates[0],
                    backend_index=certificates[0].backend_index + 1,
                ),
            ) + certificates[1:]
        key = PrefixSemanticKey(b"f" * 32, b"g" * 32, 16)
        self.output = PrefixEvictionBatch(
            (EvictedPrefix(self.prefix, key),), certificates
        )
        self.ack_calls = 0

    def __getattr__(self, name: str) -> Any:
        if name in _UNUSED_MANAGER_METHODS:
            return self._unused
        raise AttributeError(name)

    @staticmethod
    def _unused(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unused fake manager method")

    def arena_stats(self) -> tuple[ArenaStats, ...]:
        return tuple(
            ArenaStats(
                item.engine_epoch,
                item.pool_epoch,
                item.pool_id,
                item.page_count,
                item.class_id,
                item.backend_domain,
                item.first_page_id,
                item.page_count,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            for item in self.arenas
        )

    def stats(self) -> ManagerStats:
        page_count = sum(item.page_count for item in self.arenas)
        return ManagerStats(0, 0, 0, 0, 0, 0, page_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def prefix_evict_batch(
        self, prefixes: Sequence[PrefixLease]
    ) -> PrefixEvictionBatch:
        assert tuple(prefixes) == (self.prefix,)
        return self.output

    def acknowledge_reclamations_batch(self, receipts: Sequence[Any]) -> None:
        assert len(receipts) == len(self.output.retirements)
        self.ack_calls += 1

    def prefix_recycle_batch(self, prefixes: Sequence[PrefixLease]) -> None:
        assert tuple(prefixes) == (self.prefix,)

    def destroy(self) -> None:
        return None


def _fake_prefix_runtime(manager: FakePrefixManager) -> CanonicalRuntime:
    classes = tuple(
        SimpleNamespace(
            class_id=item.class_id,
            pool_id=item.pool_id,
            backend_domain=item.backend_domain,
            retention="full" if item.class_id == 0 else "sliding",
            period_blocks=None if item.class_id == 0 else 3,
        )
        for item in manager.arenas
    )
    runtime = CanonicalRuntime(
        SimpleNamespace(
            page_tokens=16,
            classes=classes,
            classes_by_id={item.class_id: item for item in classes},
        ),
        manager,
    )
    shadows = tuple(
        PageShadow(
            RequestLease(71, 99, 1),
            arena.class_id,
            0,
            page,
            arena.backend_base_index,
        )
        for arena, page in zip(manager.arenas, manager.pages, strict=True)
    )
    page_refs = runtime._page_registry.plan((), shadows)
    runtime._page_registry.commit(page_refs)
    runtime._page_registry.install_prefix(manager.prefix, shadows)
    return runtime


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("fault", ["forged-certificate", "late-mismatch"])
def test_fake_prefix_eviction_fault_has_zero_lut_mutation_and_zero_ack(
    hybrid: bool, fault: str
) -> None:
    manager = FakePrefixManager(
        hybrid=hybrid, forge_certificate=fault == "forged-certificate"
    )
    runtime = _fake_prefix_runtime(manager)
    mapping = {
        manager.pages[0]: manager.pages[1] if hybrid else None,
    }
    before = dict(mapping)
    coordinator = GlobalPrefixLut(
        mapping, [], late_mismatch=fault == "late-mismatch"
    )
    runtime.bind_prefix_eviction_cleanup(coordinator)

    with pytest.raises(FailStopped, match="prefix eviction"):
        runtime.prefix_evict_batch((manager.prefix,))
    assert mapping == before
    assert coordinator.commit_calls == 0
    assert coordinator.sync_calls == 0
    assert coordinator.finalize_calls == 0
    assert manager.ack_calls == 0
    assert runtime.failure_reason is not None


def test_fake_prefix_eviction_foreign_page_certificate_has_zero_ack() -> None:
    manager = FakePrefixManager(hybrid=False, forge_certificate=False)
    certificate = manager.output.retirements[0]
    arena = manager.arenas[0]
    foreign = replace(
        certificate.page,
        page_id=certificate.page.page_id + 1,
    )
    manager.output = replace(
        manager.output,
        retirements=(
            replace(
                certificate,
                page=foreign,
                backend_index=arena.backend_base_index + 1,
            ),
        ),
    )
    runtime = _fake_prefix_runtime(manager)
    mapping = {manager.pages[0]: None}
    coordinator = GlobalPrefixLut(mapping, [])
    runtime.bind_prefix_eviction_cleanup(coordinator)

    with pytest.raises(FailStopped, match="prefix eviction"):
        runtime.prefix_evict_batch((manager.prefix,))
    assert mapping == {manager.pages[0]: None}
    assert coordinator.commit_calls == 0
    assert coordinator.sync_calls == 0
    assert coordinator.finalize_calls == 0
    assert manager.ack_calls == 0
    assert runtime.failure_reason is not None
    runtime.close()


@pytest.mark.parametrize("fault", ["commit", "sync", "finalize", "ack"])
def test_fake_prefix_cleanup_or_ack_unknown_failure_is_fail_stopped(
    fault: str,
) -> None:
    manager = FakePrefixManager(hybrid=True, forge_certificate=False)
    runtime = _fake_prefix_runtime(manager)
    mapping = {manager.pages[0]: manager.pages[1]}
    coordinator = GlobalPrefixLut(
        mapping, [], fail_stage=fault if fault != "ack" else None
    )
    runtime.bind_prefix_eviction_cleanup(coordinator)
    if fault == "ack":
        original_ack = manager.acknowledge_reclamations_batch

        def lost_ack(receipts: Sequence[Any]) -> None:
            original_ack(receipts)
            raise OSError("simulated lost prefix ACK return")

        manager.acknowledge_reclamations_batch = lost_ack

    with pytest.raises(FailStopped, match="prefix eviction"):
        runtime.prefix_evict_batch((manager.prefix,))
    assert manager.ack_calls == (1 if fault == "ack" else 0)
    assert runtime.failure_reason is not None
