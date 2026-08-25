from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import pytest

from orbitkv_sglang.config import load_config
from orbitkv_sglang.ffi.manager import CtypesManager
from orbitkv_sglang.runtime import (
    CanonicalRuntime,
    ClassTokenDispositionUpdate,
    FailStopped,
    ManagerError,
    RelocationBatchItem,
    RelocationCopyBatch,
    RelocationCopyReceipt,
    RelocationCopyUnobserved,
    RelocationPolicy,
    RetryableConflict,
    PrefixSemanticKey,
    SnapshotLease,
    TokenDisposition,
    TokenDispositionKind,
)
from runtime_test_support import _policy_updates, _runtime, _step_batch, ffi_library

__all__ = ["ffi_library"]


def _relocation_batch_items(keys: Sequence[str]) -> tuple[RelocationBatchItem, ...]:
    return tuple(
        RelocationBatchItem(
            key,
            0,
            _policy_updates(),
            RelocationPolicy(3, 2, 250, True),
        )
        for key in keys
    )


def _copied_batch(
    prepared: Sequence[Any], completion_domain: int = 9
) -> RelocationCopyBatch:
    return RelocationCopyBatch(
        tuple(
            tuple(
                RelocationCopyReceipt(
                    item.relocation,
                    movement.token_id,
                    movement.source,
                    movement.destination,
                )
                for movement in item.moves
            )
            for item in prepared
        ),
        completion_domain,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_relocate_tokens_batch_b1_b4_is_one_ordered_native_transaction(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(batch_size))
    _step_batch(runtime, tuple((key, 48) for key in keys))
    before = dict(manager.performance_counters)
    callback_requests = []
    registry_commits = []
    head_replacements = []
    original_commit = runtime._page_registry.commit
    original_replace_heads = runtime._replace_heads

    def commit(plan: Any) -> None:
        registry_commits.append(plan)
        original_commit(plan)

    def replace_heads(
        _self: CanonicalRuntime,
        old: Sequence[SnapshotLease],
        new: Sequence[SnapshotLease],
    ) -> None:
        head_replacements.append((tuple(old), tuple(new)))
        original_replace_heads(old, new)

    def copy(prepared: tuple[Any, ...]) -> RelocationCopyBatch:
        callback_requests.append(tuple(item.request for item in prepared))
        return _copied_batch(prepared)

    monkeypatch.setattr(runtime._page_registry, "commit", commit)
    monkeypatch.setattr(
        runtime, "_replace_heads", MethodType(replace_heads, runtime)
    )
    publication = runtime.relocate_tokens_batch(
        _relocation_batch_items(keys), copy
    )

    assert tuple(item.key for item in publication.items) == keys
    assert callback_requests == [
        tuple(runtime.record_for(key).lease for key in keys)
    ]
    assert len(registry_commits) == len(head_replacements) == 1
    assert len(head_replacements[0][0]) == len(head_replacements[0][1]) == batch_size
    for name in (
        "mark_token_dispositions_batch_calls",
        "prepare_relocation_batch_calls",
        "submit_relocation_batch_calls",
        "complete_relocation_batch_calls",
    ):
        assert manager.performance_counters[name] - before[name] == 1
    for item in publication.items:
        assert len(item.retained_locations) == 24
        assert len(item.prepared.source_pages) == 3
        assert len(item.prepared.destination_pages) == 2
        assert len(item.prepared.moves) == 24
        assert item.prepared.projected_reclaimed_pages == 1
        assert len(item.retirements) == 3
    assert len(publication.retirements) == 3 * batch_size
    assert manager.stats().active_pages == 2 * batch_size
    assert manager.stats().pending_reclamations == 3 * batch_size

    runtime.acknowledge_relocation_batch(publication)
    assert (
        manager.performance_counters["acknowledge_reclamations_batch_calls"]
        - before["acknowledge_reclamations_batch_calls"]
        == 1
    )
    assert manager.stats().pending_reclamations == 0
    runtime.release_batch(keys)
    assert runtime.stats().free_pages == 64
    runtime.close()


def test_relocate_tokens_batch_aggregate_headroom_rejection_has_no_reservation(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(4))
    for boundary in (64, 128, 192, 208):
        _step_batch(runtime, tuple((key, boundary) for key in keys))
    updates = tuple(
        ClassTokenDispositionUpdate(
            0,
            token_id,
            TokenDisposition(TokenDispositionKind.POLICY_EVICTED, 7, 1, 99),
        )
        for token_id in range(208)
        if token_id % 16 >= 8
    )
    items = tuple(
        RelocationBatchItem(
            key, 0, updates, RelocationPolicy(13, 7, 500, True)
        )
        for key in keys
    )
    before = manager.stats()
    old_heads = tuple(runtime.record_for(key).head for key in keys)

    with pytest.raises(FailStopped, match="token relocation"):
        runtime.relocate_tokens_batch(
            items, lambda _prepared: pytest.fail("copy ran after headroom failure")
        )

    after = manager.stats()
    assert after.free_pages == before.free_pages
    assert after.reserved_pages == before.reserved_pages == 0
    assert after.active_pages == before.active_pages == 52
    assert after.quarantined_pages == before.quarantined_pages == 0
    assert tuple(record.head for record in runtime._requests.values()) == old_heads
    assert manager.performance_counters["prepare_relocation_batch_calls"] == 1
    assert manager.performance_counters["submit_relocation_batch_calls"] == 0
    runtime.close()


def test_relocate_tokens_batch_unobserved_callback_aborts_once_then_fail_stops(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(4))
    _step_batch(runtime, tuple((key, 48) for key in keys))

    def unobserved(_prepared: tuple[Any, ...]) -> RelocationCopyBatch:
        raise RelocationCopyUnobserved("copy was not enqueued")

    with pytest.raises(FailStopped, match="token relocation"):
        runtime.relocate_tokens_batch(_relocation_batch_items(keys), unobserved)

    stats = manager.stats()
    assert stats.active_pages == 12
    assert stats.reserved_pages == stats.quarantined_pages == 0
    assert manager.performance_counters["abort_relocations_batch_calls"] == 1
    assert manager.performance_counters["submit_relocation_batch_calls"] == 0
    assert manager.performance_counters["complete_relocation_batch_calls"] == 0
    runtime.close()


def test_relocate_tokens_batch_bad_later_receipt_quarantines_whole_batch(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(4))
    _step_batch(runtime, tuple((key, 48) for key in keys))

    def bad_later(prepared: tuple[Any, ...]) -> RelocationCopyBatch:
        copied = _copied_batch(prepared)
        groups = list(copied.receipts)
        last = list(groups[-1])
        last[-1] = replace(last[-1], copied=0)
        groups[-1] = tuple(last)
        return replace(copied, receipts=tuple(groups))

    with pytest.raises(FailStopped, match="token relocation"):
        runtime.relocate_tokens_batch(_relocation_batch_items(keys), bad_later)

    stats = manager.stats()
    assert stats.quarantined_pages == 8
    assert stats.reserved_pages == stats.writing_pages == 0
    assert manager.performance_counters["submit_relocation_batch_calls"] == 1
    assert manager.performance_counters["complete_relocation_batch_calls"] == 0
    assert manager.performance_counters["abort_relocations_batch_calls"] == 0
    assert manager.performance_counters["acknowledge_reclamations_batch_calls"] == 0
    runtime.close()


@pytest.mark.parametrize(
    "fault",
    [
        "source-backend",
        "source-offset",
        "destination-backend",
        "destination-offset",
        "destination-generation",
        "destination-slot",
        "missing-token",
        "wrong-token",
    ],
)
def test_malformed_prepared_move_fails_before_copy(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = ("first", "later")
    _step_batch(runtime, tuple((key, 48) for key in keys))
    original_prepare = manager.prepare_relocation_batch
    copy_calls = []

    def malformed(_self: CtypesManager, items: Sequence[Any]) -> Any:
        prepared = list(original_prepare(items))
        moves = list(prepared[-1].moves)
        index = 1 if fault == "destination-slot" else 0
        movement = moves[index]
        if fault == "source-backend":
            movement = replace(
                movement, source=replace(movement.source, backend_index=999)
            )
        elif fault == "source-offset":
            movement = replace(
                movement, source=replace(movement.source, offset=16)
            )
        elif fault == "destination-backend":
            movement = replace(
                movement, destination=replace(movement.destination, backend_index=999)
            )
        elif fault == "destination-offset":
            movement = replace(
                movement, destination=replace(movement.destination, offset=16)
            )
        elif fault == "destination-generation":
            movement = replace(
                movement,
                destination=replace(
                    movement.destination,
                    page=replace(movement.destination.page, generation=0),
                ),
            )
        elif fault == "destination-slot":
            movement = replace(movement, destination=moves[0].destination)
        elif fault == "missing-token":
            moves.pop()
        elif fault == "wrong-token":
            movement = replace(movement, token_id=47)
        if fault not in ("missing-token",):
            moves[index] = movement
        prepared[-1] = replace(prepared[-1], moves=tuple(moves))
        return tuple(prepared)

    def copied(_prepared: tuple[Any, ...]) -> RelocationCopyBatch:
        copy_calls.append(True)
        return _copied_batch(_prepared)

    monkeypatch.setattr(
        manager, "prepare_relocation_batch", MethodType(malformed, manager)
    )
    with pytest.raises(FailStopped, match="token relocation"):
        runtime.relocate_tokens_batch(_relocation_batch_items(keys), copied)

    assert copy_calls == []
    assert manager.performance_counters["submit_relocation_batch_calls"] == 0
    assert manager.performance_counters["complete_relocation_batch_calls"] == 0
    runtime.close()


@pytest.mark.parametrize("fault", ["token-id", "backend", "offset"])
def test_malformed_old_token_view_fails_before_mark_and_copy(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    _step_batch(runtime, (("request", 48),))
    original_views = manager.token_views_batch
    copy_calls = []

    def malformed(_self: CtypesManager, queries: Sequence[Any]) -> Any:
        views = list(original_views(queries))
        placements = list(views[0].placements)
        placement = placements[0]
        if fault == "token-id":
            placement = replace(placement, token_id=1)
        elif fault == "backend":
            placement = replace(
                placement, location=replace(placement.location, backend_index=999)
            )
        elif fault == "offset":
            placement = replace(
                placement, location=replace(placement.location, offset=16)
            )
        placements[0] = placement
        views[0] = replace(views[0], placements=tuple(placements))
        return tuple(views)

    monkeypatch.setattr(
        manager, "token_views_batch", MethodType(malformed, manager)
    )
    with pytest.raises(ManagerError, match="token view"):
        runtime.relocate_tokens_batch(
            _relocation_batch_items(("request",)),
            lambda prepared: copy_calls.append(prepared),
        )

    assert copy_calls == []
    assert manager.performance_counters["mark_token_dispositions_batch_calls"] == 0
    assert runtime.failure_reason is None
    monkeypatch.setattr(manager, "token_views_batch", original_views)
    runtime.release_batch(("request",))
    runtime.close()


def test_unacknowledged_relocation_blocks_append_release_and_close_until_ack(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    _step_batch(runtime, (("request", 48),))
    publication = runtime.relocate_tokens_batch(
        _relocation_batch_items(("request",)), _copied_batch
    )
    prepare_before = manager.performance_counters["prepare_batch_calls"]
    release_before = manager.performance_counters["release_batch_calls"]

    with pytest.raises(ManagerError, match="ready for append"):
        runtime.prepare_batch((("request", 49),))
    with pytest.raises(ManagerError, match="awaits relocation"):
        runtime.release_batch(("request",))
    with pytest.raises(ManagerError, match="awaits relocation"):
        runtime.relocate_tokens_batch(
            _relocation_batch_items(("request",)), _copied_batch
        )
    with pytest.raises(ManagerError, match="awaits relocation"):
        runtime.mark_token_dispositions("request", 0, _policy_updates())
    publish_release_before = manager.performance_counters[
        "prefix_publish_release_batch_calls"
    ]
    semantic = PrefixSemanticKey(b"r" * 32, b"a" * 32, 48)
    with pytest.raises(ManagerError, match="awaits relocation"):
        runtime.prefix_publish_release_batch((("request", semantic),))
    assert (
        manager.performance_counters["prefix_publish_release_batch_calls"]
        == publish_release_before
    )
    with pytest.raises(ManagerError, match="live requests"):
        runtime.close()
    assert manager.performance_counters["prepare_batch_calls"] == prepare_before
    assert manager.performance_counters["release_batch_calls"] == release_before

    runtime.acknowledge_relocation_batch(publication)
    assert runtime._pending_relocation_batches == {}
    assert runtime._pending_relocation_requests == {}
    _step_batch(runtime, (("request", 49),), domain=10)
    runtime.release_batch(("request",))
    runtime.close()


@pytest.mark.parametrize("fault", ["cardinality", "order"])
def test_relocate_tokens_batch_rejects_invalid_complete_output_before_host_commit(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(4))
    _step_batch(runtime, tuple((key, 48) for key in keys))
    old_heads = tuple(runtime.record_for(key).head for key in keys)
    original_complete = manager.complete_relocation_batch
    commits = []

    def invalid_complete(
        _self: CtypesManager, receipt: Any, relocations: Sequence[Any]
    ) -> Any:
        output = original_complete(receipt, relocations)
        publications = (
            output.publications[:-1]
            if fault == "cardinality"
            else tuple(reversed(output.publications))
        )
        return replace(output, publications=publications)

    def commit(_plan: Any) -> None:
        commits.append(True)

    monkeypatch.setattr(
        manager,
        "complete_relocation_batch",
        MethodType(invalid_complete, manager),
    )
    monkeypatch.setattr(runtime._page_registry, "commit", commit)

    with pytest.raises(FailStopped, match="publication"):
        runtime.relocate_tokens_batch(
            _relocation_batch_items(keys), _copied_batch
        )

    assert not commits
    assert tuple(runtime._requests[key].head for key in keys) == old_heads
    assert manager.stats().pending_reclamations == 12
    assert manager.performance_counters["acknowledge_reclamations_batch_calls"] == 0
    runtime.close()


def test_relocation_batch_lost_ack_return_fail_stops_after_one_native_ack(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(4))
    _step_batch(runtime, tuple((key, 48) for key in keys))
    publication = runtime.relocate_tokens_batch(
        _relocation_batch_items(keys), _copied_batch
    )
    original_ack = manager.acknowledge_reclamations_batch
    acknowledged = []

    def lost_return(_self: CtypesManager, receipts: Sequence[Any]) -> None:
        values = tuple(receipts)
        original_ack(values)
        acknowledged.append(values)
        raise OSError("simulated lost batch relocation ACK return")

    monkeypatch.setattr(
        manager,
        "acknowledge_reclamations_batch",
        MethodType(lost_return, manager),
    )
    with pytest.raises(FailStopped, match="relocation reclamation ACK"):
        runtime.acknowledge_relocation_batch(publication)

    assert len(acknowledged) == 1
    assert len(acknowledged[0]) == 12
    assert manager.stats().pending_reclamations == 0
    assert runtime._runtime_counters["fail_stop_count"] == 1
    with pytest.raises(FailStopped):
        runtime.acknowledge_relocation_batch(publication)
    runtime.close()


def test_relocation_batch_ack_rejects_tampering_repeat_and_partial_item(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=4
    )
    keys = tuple(f"request-{index}" for index in range(4))
    _step_batch(runtime, tuple((key, 48) for key in keys))
    publication = runtime.relocate_tokens_batch(
        _relocation_batch_items(keys), _copied_batch
    )
    ack_before = manager.performance_counters[
        "acknowledge_reclamations_batch_calls"
    ]

    with pytest.raises(ManagerError, match="singleton batch"):
        runtime.acknowledge_relocation(publication.items[0])
    with pytest.raises(ManagerError, match="pending batch"):
        runtime.acknowledge_relocation_batch(
            replace(publication, retirements=publication.retirements[:-1])
        )
    assert (
        manager.performance_counters["acknowledge_reclamations_batch_calls"]
        == ack_before
    )

    runtime.acknowledge_relocation_batch(publication)
    with pytest.raises(ManagerError, match="pending batch"):
        runtime.acknowledge_relocation_batch(publication)
    assert (
        manager.performance_counters["acknowledge_reclamations_batch_calls"]
        == ack_before + 1
    )
    runtime.release_batch(keys)
    runtime.close()


def test_mark_token_dispositions_rejects_missing_class_before_native_mark(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=True, window_tokens=128
    )
    _step_batch(runtime, (("request", 48),))
    updates = tuple(replace(item, class_id=1) for item in _policy_updates())
    before = manager.performance_counters["mark_token_dispositions_batch_calls"]

    with pytest.raises(ManagerError, match="requested class"):
        runtime.mark_token_dispositions("request", 0, updates)

    assert runtime.failure_reason is None
    assert manager.performance_counters["mark_token_dispositions_batch_calls"] == before
    runtime.release_batch(("request",))
    runtime.close()


def test_relocation_rejects_released_request_completion_fence_reuse(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False, requests=2
    )
    _step_batch(runtime, (("old", 16),), domain=9)
    old_value = runtime.record_for("old").completion_value
    assert old_value == 1
    runtime.release_batch(("old",))
    _step_batch(runtime, (("new", 48),), domain=10)
    submit_before = manager.performance_counters["submit_relocation_batch_calls"]

    with pytest.raises(ManagerError, match="completion value"):
        runtime.relocate_tokens(
            "new",
            0,
            _policy_updates(),
            RelocationPolicy(3, 2, 250, True),
            lambda _prepared: pytest.fail("copy ran with a stale fence"),
            9,
            old_value,
        )

    assert manager.performance_counters["mark_token_dispositions_batch_calls"] == 0
    assert manager.performance_counters["submit_relocation_batch_calls"] == submit_before
    assert runtime.failure_reason is None
    runtime.release_batch(("new",))
    runtime.close()


def test_token_reclamation_config_is_explicit_strict_and_full_only(
    tmp_path: Path, ffi_library: Path
) -> None:
    full_plan = tmp_path / "full-reclamation-plan.json"
    full_plan.write_text(
        json.dumps(
            {
                "page_tokens": 16,
                "classes": [
                    {
                        "name": "full",
                        "layers": [0],
                        "retention": "full",
                        "bytes_per_token_per_layer": 128,
                        "window_tokens": None,
                    }
                ],
            }
        )
    )
    base = {
        "ORBITKV_PLAN": str(full_plan),
        "ORBITKV_LIBRARY": str(ffi_library),
    }
    assert load_config(base).token_reclamation.mode == "off"
    profile = {
        "mode": "relocate",
        "trigger_tokens": 48,
        "retained_per_page": 8,
        "policy_id": 7,
        "policy_version": 1,
        "quality_contract": 99,
        "fragmentation_threshold_milli": 250,
        "maximum_source_pages": 3,
        "evacuation_headroom_pages": 2,
    }
    enabled = load_config(
        {**base, "ORBITKV_TOKEN_RECLAMATION": json.dumps(profile)}
    )
    assert enabled.token_reclamation.mode == "relocate"
    assert enabled.token_reclamation.retained_per_page == 8
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(
            {
                **base,
                "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                    {**profile, "silent_default": True}
                ),
            }
        )
    with pytest.raises(ValueError, match="below 16"):
        load_config(
            {
                **base,
                "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                    {**profile, "retained_per_page": 16}
                ),
            }
        )
    hybrid_plan = tmp_path / "hybrid-reclamation-plan.json"
    hybrid_plan.write_text(
        json.dumps(
            {
                "page_tokens": 16,
                "classes": [
                    {**json.loads(full_plan.read_text())["classes"][0], "layers": [0]},
                    {
                        "name": "swa",
                        "layers": [1],
                        "retention": "sliding",
                        "bytes_per_token_per_layer": 128,
                        "window_tokens": 128,
                    },
                ],
            }
        )
    )
    hybrid_base = {**base, "ORBITKV_PLAN": str(hybrid_plan)}
    assert load_config(
        {**hybrid_base, "ORBITKV_TOKEN_RECLAMATION": json.dumps(profile)}
    ).token_reclamation.mode == "relocate"
    with pytest.raises(ValueError, match="shared Full/SWA visibility"):
        load_config(
            {
                **hybrid_base,
                "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                    {
                        **profile,
                        "trigger_tokens": 128,
                        "maximum_source_pages": 8,
                        "evacuation_headroom_pages": 4,
                    }
                ),
            }
        )


def test_relocate_config_admission_matches_scheduled_page_geometry(
    tmp_path: Path, ffi_library: Path
) -> None:
    plan = tmp_path / "relocation-admission-plan.json"
    plan.write_text(
        json.dumps(
            {
                "page_tokens": 16,
                "classes": [
                    {
                        "name": "full",
                        "layers": [0],
                        "retention": "full",
                        "bytes_per_token_per_layer": 128,
                        "window_tokens": None,
                    }
                ],
            }
        )
    )
    base = {
        "ORBITKV_PLAN": str(plan),
        "ORBITKV_LIBRARY": str(ffi_library),
    }
    profile = {
        "mode": "relocate",
        "trigger_tokens": 48,
        "retained_per_page": 8,
        "policy_id": 7,
        "policy_version": 1,
        "quality_contract": 99,
        "fragmentation_threshold_milli": 500,
        "maximum_source_pages": 3,
        "evacuation_headroom_pages": 2,
    }

    exact = load_config(
        {**base, "ORBITKV_TOKEN_RECLAMATION": json.dumps(profile)}
    ).token_reclamation
    assert exact.maximum_source_pages == 3
    assert exact.evacuation_headroom_pages == 2
    assert exact.fragmentation_threshold_milli == 500

    invalid = (
        ({"maximum_source_pages": 2}, "3 source pages"),
        ({"evacuation_headroom_pages": 1}, "2 evacuation headroom pages"),
        ({"fragmentation_threshold_milli": 501}, "expected fragmentation 500"),
        (
            {
                "trigger_tokens": 16,
                "maximum_source_pages": 1,
                "evacuation_headroom_pages": 1,
            },
            "positive page gain",
        ),
    )
    for overrides, message in invalid:
        with pytest.raises(ValueError, match=message):
            load_config(
                {
                    **base,
                    "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                        {**profile, **overrides}
                    ),
                }
            )

    partial = {
        **profile,
        "trigger_tokens": 17,
        "maximum_source_pages": 2,
        "evacuation_headroom_pages": 1,
        "fragmentation_threshold_milli": 718,
    }
    assert load_config(
        {**base, "ORBITKV_TOKEN_RECLAMATION": json.dumps(partial)}
    ).token_reclamation.trigger_tokens == 17
    with pytest.raises(ValueError, match="expected fragmentation 718"):
        load_config(
            {
                **base,
                "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                    {**partial, "fragmentation_threshold_milli": 719}
                ),
            }
        )

    naive = load_config(
        {
            **base,
            "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                {
                    **profile,
                    "mode": "naive",
                    "maximum_source_pages": 1,
                    "evacuation_headroom_pages": 1,
                    "fragmentation_threshold_milli": 1000,
                }
            ),
        }
    ).token_reclamation
    assert naive.mode == "naive"
    assert naive.evacuation_headroom_pages == 1


@pytest.mark.parametrize("operation", ["mark", "relocate"])
@pytest.mark.parametrize("error_type", [RetryableConflict, ManagerError])
def test_token_disposition_known_precommit_rejection_does_not_fail_stop(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    error_type: type[ManagerError],
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    _step_batch(runtime, (("request", 48),))
    record = runtime.record_for("request")
    old_head = record.head

    def reject(_self: CtypesManager, _items: Sequence[Any]) -> Any:
        raise error_type("known native precommit rejection")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            manager, "mark_token_dispositions_batch", MethodType(reject, manager)
        )
        with pytest.raises(error_type, match="known native precommit rejection"):
            if operation == "mark":
                runtime.mark_token_dispositions("request", 0, _policy_updates())
            else:
                runtime.relocate_tokens(
                    "request",
                    0,
                    _policy_updates(),
                    RelocationPolicy(3, 2, 250, True),
                    lambda _prepared: pytest.fail(
                        "copy ran after a rejected disposition mark"
                    ),
                    9,
                    1,
                )

    assert runtime.failure_reason is None
    assert record.head == old_head
    assert runtime.token_view("request", 0).view_version == 1
    assert manager.performance_counters["prepare_relocation_batch_calls"] == 0
    runtime.release_batch(("request",))
    runtime.close()


@pytest.mark.parametrize(
    "fault",
    [
        "lost-mark-return",
        "publication-cardinality",
        "publication-identity",
        "lost-readback-return",
        "retained-location",
        "common-set",
        "head-mirror",
    ],
)
def test_mark_dispositions_postcommit_consumption_fault_fail_stops(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=True, window_tokens=128
    )
    _step_batch(runtime, (("request", 48),))
    record = runtime.record_for("request")
    old_head = record.head
    updates = tuple(
        ClassTokenDispositionUpdate(
            class_id,
            token_id,
            TokenDisposition(TokenDispositionKind.POLICY_EVICTED, 7, 1, 99),
        )
        for class_id in (0, 1)
        for token_id in range(48)
        if token_id % 16 >= 8
    )
    original_mark = manager.mark_token_dispositions_batch
    original_views = manager.token_views_batch
    original_replace_head = runtime._replace_head
    committed: list[Any] = []
    view_calls = 0

    def faulty_mark(_self: CtypesManager, items: Sequence[Any]) -> Any:
        outputs = tuple(original_mark(items))
        committed.extend(outputs)
        if fault == "lost-mark-return":
            raise OSError("simulated lost disposition mark return")
        if fault == "publication-cardinality":
            return outputs + outputs
        if fault == "publication-identity":
            return (replace(outputs[0], boundary=outputs[0].boundary + 1),)
        return outputs

    def faulty_views(_self: CtypesManager, queries: Sequence[Any]) -> Any:
        nonlocal view_calls
        outputs = list(original_views(queries))
        view_calls += 1
        if view_calls != 2:
            return tuple(outputs)
        if fault == "lost-readback-return":
            raise OSError("simulated lost disposition readback return")
        if fault == "retained-location":
            placements = list(outputs[0].placements)
            retained_index = next(
                index
                for index, placement in enumerate(placements)
                if placement.disposition.kind is TokenDispositionKind.RETAINED
            )
            placements[retained_index] = replace(
                placements[retained_index], location=None
            )
            outputs[0] = replace(outputs[0], placements=tuple(placements))
        if fault == "common-set":
            placements = list(outputs[1].placements)
            retained_index = next(
                index
                for index, placement in enumerate(placements)
                if placement.disposition.kind is TokenDispositionKind.RETAINED
            )
            placements[retained_index] = replace(
                placements[retained_index],
                disposition=TokenDisposition(
                    TokenDispositionKind.POLICY_EVICTED, 7, 1, 99
                ),
            )
            outputs[1] = replace(outputs[1], placements=tuple(placements))
        return tuple(outputs)

    def lost_head_mirror_return(
        _self: CanonicalRuntime, old: SnapshotLease, new: SnapshotLease
    ) -> None:
        original_replace_head(old, new)
        raise OSError("simulated lost head mirror return")

    monkeypatch.setattr(
        manager, "mark_token_dispositions_batch", MethodType(faulty_mark, manager)
    )
    monkeypatch.setattr(manager, "token_views_batch", MethodType(faulty_views, manager))
    if fault == "head-mirror":
        monkeypatch.setattr(
            runtime,
            "_replace_head",
            MethodType(lost_head_mirror_return, runtime),
        )

    with pytest.raises(FailStopped, match="token disposition"):
        runtime.mark_token_dispositions("request", 0, updates)
    assert len(committed) == 1
    assert committed[0].snapshot != old_head
    assert record.head == old_head
    assert runtime.failure_reason is not None
    assert runtime._runtime_counters["fail_stop_count"] == 1
    with pytest.raises(FailStopped):
        runtime.token_view("request", 0)
    runtime.close()


@pytest.mark.parametrize(
    "fault",
    [
        "lost-mark-return",
        "publication-cardinality",
        "publication-identity",
        "head-mirror",
        "prepare-rejection",
    ],
)
def test_relocate_tokens_fail_stops_from_successful_mark_return(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    _step_batch(runtime, (("request", 48),))
    record = runtime.record_for("request")
    old_head = record.head
    original_mark = manager.mark_token_dispositions_batch
    original_replace_heads = runtime._replace_heads
    committed: list[Any] = []
    prepare_calls: list[int] = []

    def faulty_mark(_self: CtypesManager, items: Sequence[Any]) -> Any:
        outputs = tuple(original_mark(items))
        committed.extend(outputs)
        if fault == "lost-mark-return":
            raise OSError("simulated lost relocation mark return")
        if fault == "publication-cardinality":
            return outputs + outputs
        if fault == "publication-identity":
            return (replace(outputs[0], boundary=outputs[0].boundary + 1),)
        return outputs

    def lost_head_mirror_return(
        _self: CanonicalRuntime, old: Sequence[SnapshotLease],
        new: Sequence[SnapshotLease],
    ) -> None:
        original_replace_heads(old, new)
        raise OSError("simulated lost relocation head mirror return")

    def reject_prepare(_self: CtypesManager, items: Sequence[Any]) -> Any:
        prepare_calls.append(len(items))
        raise ManagerError("simulated post-mark prepare rejection")

    monkeypatch.setattr(
        manager, "mark_token_dispositions_batch", MethodType(faulty_mark, manager)
    )
    if fault == "head-mirror":
        monkeypatch.setattr(
            runtime,
            "_replace_heads",
            MethodType(lost_head_mirror_return, runtime),
        )
    if fault == "prepare-rejection":
        monkeypatch.setattr(
            manager,
            "prepare_relocation_batch",
            MethodType(reject_prepare, manager),
        )

    def copied(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
        if fault != "head-mirror":
            pytest.fail("copy ran after injected failure")
        return tuple(
            RelocationCopyReceipt(
                prepared.relocation,
                movement.token_id,
                movement.source,
                movement.destination,
            )
            for movement in prepared.moves
        )

    with pytest.raises(FailStopped, match="token relocation"):
        runtime.relocate_tokens(
            "request",
            0,
            _policy_updates(),
            RelocationPolicy(3, 2, 250, True),
            copied,
            9,
            1,
        )
    assert len(committed) == 1
    assert committed[0].snapshot != old_head
    assert runtime.failure_reason is not None
    assert runtime._runtime_counters["fail_stop_count"] == 1
    assert prepare_calls == ([1] if fault == "prepare-rejection" else [])
    with pytest.raises(FailStopped):
        runtime.token_view("request", 0)
    runtime.close()


@pytest.mark.parametrize("mode", ["naive", "relocate"])
def test_runtime_same_victim_set_naive_and_relocation_preserve_absolute_boundary(
    tmp_path: Path, ffi_library: Path, mode: str
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    _step_batch(runtime, (("request", 48),))
    record = runtime.record_for("request")
    assert record.boundary == 48
    before = runtime.token_view("request", 0)
    assert len(before.placements) == 48
    updates = _policy_updates()

    if mode == "naive":
        output = runtime.mark_token_dispositions("request", 0, updates)
        assert len(output.retained_locations) == 24
        assert len(record.cursor.pages) == 3
        assert manager.stats().active_pages == 3
    else:
        def copied(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
            return tuple(
                RelocationCopyReceipt(
                    prepared.relocation,
                    movement.token_id,
                    movement.source,
                    movement.destination,
                )
                for movement in prepared.moves
            )

        output = runtime.relocate_tokens(
            "request",
            0,
            updates,
            RelocationPolicy(3, 2, 250, True),
            copied,
            9,
            1,
        )
        assert len(output.retained_locations) == 24
        assert len(output.prepared.source_pages) == 3
        assert len(output.prepared.destination_pages) == 2
        assert output.prepared.projected_reclaimed_pages == 1
        runtime.acknowledge_relocation(output)
        assert len(record.cursor.pages) == 2
        assert manager.stats().active_pages == 2

    assert record.boundary == 48
    assert runtime.active_kv_length("request", 0) == 24
    _step_batch(runtime, (("request", 49),), domain=10)
    assert record.boundary == 49
    assert runtime.active_kv_length("request", 0) == 25
    after = runtime.token_view("request", 0)
    assert len(after.placements) == 49
    assert after.placements[48].location is not None

    runtime.release_batch(("request",))
    stats = runtime.stats()
    assert stats.free_pages == 64
    assert stats.active_requests == stats.active_snapshots == 0
    assert stats.pending_reclamations == 0
    runtime.close()


def test_relocation_lost_ack_return_fail_stops_before_future_operation(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    _step_batch(runtime, (("request", 48),))

    def copied(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
        return tuple(
            RelocationCopyReceipt(
                prepared.relocation,
                movement.token_id,
                movement.source,
                movement.destination,
            )
            for movement in prepared.moves
        )

    publication = runtime.relocate_tokens(
        "request",
        0,
        _policy_updates(),
        RelocationPolicy(3, 2, 250, True),
        copied,
        9,
        1,
    )
    assert publication.retirements
    assert manager.stats().pending_reclamations == len(publication.retirements)
    original_ack = manager.acknowledge_reclamations_batch
    acknowledged: list[tuple[Any, ...]] = []

    def lost_return(_self: CtypesManager, receipts: Sequence[Any]) -> None:
        values = tuple(receipts)
        original_ack(values)
        acknowledged.append(values)
        raise OSError("simulated lost relocation ACK return")

    monkeypatch.setattr(
        manager,
        "acknowledge_reclamations_batch",
        MethodType(lost_return, manager),
    )
    with pytest.raises(FailStopped, match="relocation reclamation ACK"):
        runtime.acknowledge_relocation(publication)

    assert len(acknowledged) == 1
    assert len(acknowledged[0]) == len(publication.retirements)
    assert manager.stats().pending_reclamations == 0
    assert runtime.failure_reason is not None
    assert runtime._runtime_counters["fail_stop_count"] == 1
    with pytest.raises(FailStopped):
        runtime.request_acquire_batch(("new-request",))
    runtime.close()


def test_runtime_private_full_request_appends_and_relocates_twice(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )

    def copied(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
        return tuple(
            RelocationCopyReceipt(
                prepared.relocation,
                movement.token_id,
                movement.source,
                movement.destination,
            )
            for movement in prepared.moves
        )

    def dispose_packed_tail_halves() -> tuple[ClassTokenDispositionUpdate, ...]:
        view = runtime.token_view("request", 0)
        return tuple(
            ClassTokenDispositionUpdate(
                0,
                placement.token_id,
                TokenDisposition(TokenDispositionKind.POLICY_EVICTED, 7, 1, 99),
            )
            for placement in view.placements
            if placement.disposition.kind is TokenDispositionKind.RETAINED
            and placement.location is not None
            and placement.location.offset >= 8
        )

    _step_batch(runtime, (("request", 48),))
    first = runtime.relocate_tokens(
        "request",
        0,
        dispose_packed_tail_halves(),
        RelocationPolicy(3, 2, 250, True),
        copied,
        21,
        1,
    )
    assert first.key == "request"
    assert first.old_view.view_version == 1
    assert first.publication.view_version == 3
    assert len(first.prepared.source_pages) == 3
    assert len(first.prepared.destination_pages) == 2
    assert len(first.retained_locations) == 24
    assert first.class_retained_locations == ((0, first.retained_locations),)
    runtime.acknowledge_relocation(first)

    _step_batch(runtime, (("request", 65),), domain=22)
    record = runtime.record_for("request")
    assert record.boundary == 65
    assert record.cursor.layout_boundaries[0] == 41
    assert runtime.active_kv_length("request", 0) == 41

    second_updates = dispose_packed_tail_halves()
    assert len(second_updates) == 17
    second = runtime.relocate_tokens(
        "request",
        0,
        second_updates,
        RelocationPolicy(3, 2, 250, True),
        copied,
        23,
        2,
    )
    assert second.key == "request"
    assert second.old_view.view_version == 4
    assert second.publication.view_version == 6
    assert len(second.prepared.source_pages) == 3
    assert len(second.prepared.destination_pages) == 2
    assert len(second.retained_locations) == 24
    assert second.class_retained_locations == ((0, second.retained_locations),)
    assert tuple(
        (item.logical_ordinal, item.token_begin, item.token_end_exclusive)
        for item in second.retirements
    ) == ((0, 0, 16), (1, 16, 32), (2, 32, 41))
    runtime.acknowledge_relocation(second)

    assert record.boundary == 65
    assert record.cursor.layout_boundaries[0] == 24
    assert runtime.active_kv_length("request", 0) == 24
    assert len(record.cursor.pages) == 2
    assert manager.stats().active_pages == 2

    runtime.release_batch(("request",))
    stats = runtime.stats()
    assert stats.free_pages == 64
    assert stats.active_requests == stats.active_snapshots == 0
    assert stats.pending_reclamations == 0
    runtime.close()


def test_runtime_hybrid_full_relocation_preserves_swa_placements_and_appends(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=True, window_tokens=128
    )
    _step_batch(runtime, (("request", 48),))
    updates = tuple(
        ClassTokenDispositionUpdate(
            class_id,
            token_id,
            TokenDisposition(TokenDispositionKind.POLICY_EVICTED, 7, 1, 99),
        )
        for class_id in (0, 1)
        for token_id in range(48)
        if token_id % 16 >= 8
    )

    def copied(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
        return tuple(
            RelocationCopyReceipt(
                prepared.relocation,
                movement.token_id,
                movement.source,
                movement.destination,
            )
            for movement in prepared.moves
        )

    output = runtime.relocate_tokens(
        "request",
        0,
        updates,
        RelocationPolicy(3, 2, 250, True),
        copied,
        11,
        1,
    )
    class_locations = dict(output.class_retained_locations)
    assert len(class_locations[0]) == len(class_locations[1]) == 24
    assert class_locations[0] != class_locations[1]
    runtime.acknowledge_relocation(output)
    record = runtime.record_for("request")
    assert len([page for page in record.cursor.pages.values() if page.class_id == 0]) == 2
    assert len([page for page in record.cursor.pages.values() if page.class_id == 1]) == 3
    _step_batch(runtime, (("request", 49),), domain=12)
    assert runtime.active_kv_length("request", 0) == 25
    assert runtime.active_kv_length("request", 1) == 25
    views = (runtime.token_view("request", 0), runtime.token_view("request", 1))
    assert views[0].placements[48].location.offset == 8
    assert views[1].placements[48].location.offset == 0
    runtime.release_batch(("request",))
    assert runtime.stats().free_pages == 128
    runtime.close()
