from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.plugin.lowering as lowering  # noqa: E402
import orbitkv_sglang.plugin.state as state  # noqa: E402
from orbitkv_sglang.config import ClassConfig, ManagerPlanConfig  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    ClassLoweringSpec,
    CopyIntent,
    FailStopped,
    LoweringPlan,
    PageLease,
    RequestLease,
    SnapshotLease,
    TAIL_COPY_ON_WRITE,
    TailAction,
)


PAGE_TOKENS = 16


def _class(class_id: int, retention: str) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 1,
        name=retention,
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=32 if retention == "sliding" else None,
        period_blocks=3 if retention == "sliding" else None,
    )


class _MovePool:
    def __init__(self, name, events, *, fail=False):
        self.name = name
        self.events = events
        self.fail = fail

    def move_kv_cache(self, destinations, sources):
        self.events.append(
            (
                f"{self.name}_move",
                tuple(destinations.tolist()),
                tuple(sources.tolist()),
            )
        )
        if self.fail:
            raise RuntimeError(f"{self.name} launch failed")


class _Allocator:
    def __init__(self, events, *, fail_swa=False):
        self.full_to_swa_index_mapping = torch.zeros((2048,), dtype=torch.int64)
        self.kvcache = SimpleNamespace(
            full_kv_pool=_MovePool("full", events),
            swa_kv_pool=_MovePool("swa", events, fail=fail_swa),
        )
        self.events = events

    def get_kvcache(self):
        return self.kvcache

    def set_full_to_swa_mapping(self, full, sliding):
        self.events.append("new_token_lut")
        self.full_to_swa_index_mapping[full] = sliding


class _Runtime:
    def __init__(self, config, events):
        self.failure_reason = None
        self.events = events
        self.arenas_by_class = {
            item.class_id: ArenaIdentity(
                engine_epoch=1,
                pool_epoch=2 + item.class_id,
                pool_id=item.pool_id,
                class_id=item.class_id,
                backend_domain=item.backend_domain,
                page_count=64,
                page_tokens=PAGE_TOKENS,
                backend_base_index=0,
                first_page_id=1 + item.class_id * 64,
            )
            for item in config.classes
        }

    def mark_lowered(self, _batch):
        self.events.append("mark_lowered")

    def submit_batch(self, _batch):
        self.events.append("submit")
        return (object(),)

    def lowering_failed(self, _batch, error):
        self.events.append(("quarantine", str(error)))
        self.failure_reason = f"lowering: {error}"

    def candidate_mirror_failed(self, _batch, error):
        self.events.append(("mirror_failed", str(error)))
        self.failure_reason = f"mirror: {error}"

    def fail_stop(self, reason):
        self.failure_reason = reason


class _ReqPool:
    def __init__(self, events):
        self.req_to_token = torch.zeros((8, 128), dtype=torch.int32)
        self.events = events

    def write(self, indices, values):
        self.events.append("row_write")
        rows, columns = indices
        self.req_to_token[rows, columns] = values


def _install(*, fail_swa=False):
    events = []
    config = ManagerPlanConfig(
        plan_path=Path("plan.json"),
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:cow-test",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "full"), _class(1, "sliding")),
    )
    runtime = _Runtime(config, events)
    allocator = _Allocator(events, fail_swa=fail_swa)
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator
    return runtime, allocator, events


def _page(class_id: int, slot: int) -> PageLease:
    return PageLease(1, 2 + class_id, class_id + 1, slot, class_id + 1)


def _plan(index: int = 0, *, sliding_source_offset: int = 0) -> LoweringPlan:
    full_source_index = 2 + index * 4
    full_destination_index = full_source_index + 1
    swa_source_index = 34 + index * 4
    swa_destination_index = swa_source_index + 1
    specs = []
    for class_id, source_index, destination_index, source_offset in (
        (0, full_source_index, full_destination_index, 0),
        (1, swa_source_index, swa_destination_index, sliding_source_offset),
    ):
        source = _page(class_id, source_index + 1)
        destination = _page(class_id, destination_index + 1)
        action = TailAction(
            class_id,
            TAIL_COPY_ON_WRITE,
            8,
            0,
            source,
            destination,
        )
        intent = CopyIntent(
            class_id,
            class_id + 1,
            8,
            source_offset,
            0,
            source,
            destination,
            source_index,
            destination_index,
        )
        specs.append(
            ClassLoweringSpec(
                class_id,
                class_id + 1,
                (destination_index + 1) * PAGE_TOKENS + 7,
                (),
                action,
                (intent,),
            )
        )
    return LoweringPlan(
        RequestLease(1, index, 1),
        SnapshotLease(1, index, 1),
        SnapshotLease(1, 8 + index, 1),
        8,
        9,
        tuple(specs),
    )


def _batch(allocator, events, count=1):
    pool = _ReqPool(events)
    reqs = []
    plans = []
    for index in range(count):
        plan = _plan(index)
        full_spec = plan.by_class[0]
        swa_spec = plan.by_class[1]
        old_full = lowering._tail_locations(
            full_spec, source=True, device=torch.device("cpu")
        )
        old_swa = lowering._tail_locations(
            swa_spec, source=True, device=torch.device("cpu")
        )
        row = index + 1
        pool.req_to_token[row, :8] = old_full.to(torch.int32)
        allocator.full_to_swa_index_mapping[old_full] = old_swa
        reqs.append(
            SimpleNamespace(
                rid=f"cow-{index}",
                req_pool_idx=row,
                prefix_indices=torch.empty((0,), dtype=torch.int64),
                kv=SimpleNamespace(kv_allocated_len=8),
            )
        )
        plans.append(plan)
    return (
        SimpleNamespace(
            reqs=reqs,
            req_to_token_pool=pool,
            maybe_evict_swa=lambda: None,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            device=torch.device("cpu"),
        ),
        tuple(plans),
    )


def _patch_decode(monkeypatch, batch, plans, events):
    monkeypatch.setattr(lowering, "_validate_batch", lambda _batch: None)
    monkeypatch.setattr(
        lowering,
        "_preflight_decode_batch",
        lambda _batch: ([8] * len(plans), list(range(1, len(plans) + 1))),
    )
    monkeypatch.setattr(
        lowering, "_prepare_batch", lambda *_args: (object(), plans)
    )
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: {
            class_id: torch.tensor(
                [plan.by_class[class_id].last_location + 1 for plan in plans],
                dtype=torch.int64,
            )
            for class_id in (0, 1)
        },
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(
            current_stream=lambda _device: events.append("forward_stream")
        ),
    )


def test_hybrid_cow_moves_each_physical_subpool_before_submit_and_write(monkeypatch):
    _runtime, allocator, events = _install()
    batch, plans = _batch(allocator, events)
    _patch_decode(monkeypatch, batch, plans, events)

    lowering._alloc_for_decode(batch, 1)

    assert events[:5] == [
        "forward_stream",
        ("full_move", tuple(range(64, 72)), tuple(range(48, 56))),
        ("swa_move", tuple(range(576, 584)), tuple(range(560, 568))),
        "mark_lowered",
        "submit",
    ]
    assert events.index("submit") < events.index("row_write")
    assert events.index("submit") < events.index("new_token_lut")
    counters = state._activity_counters()
    assert counters["cow_copy_intents"] == 2
    assert counters["cow_move_calls"] == 2
    assert counters["cow_copied_tokens"] == 16


def test_second_class_copy_launch_failure_quarantines_without_submit_or_counter(
    monkeypatch,
):
    runtime, allocator, events = _install(fail_swa=True)
    batch, plans = _batch(allocator, events)
    before_row = batch.req_to_token_pool.req_to_token.clone()
    before_mapping = allocator.full_to_swa_index_mapping.clone()
    _patch_decode(monkeypatch, batch, plans, events)

    with pytest.raises(FailStopped, match="lowering"):
        lowering._alloc_for_decode(batch, 1)

    assert any(event[0] == "full_move" for event in events if isinstance(event, tuple))
    assert any(event[0] == "swa_move" for event in events if isinstance(event, tuple))
    assert any(event[0] == "quarantine" for event in events if isinstance(event, tuple))
    assert "submit" not in events
    assert "row_write" not in events
    assert torch.equal(batch.req_to_token_pool.req_to_token, before_row)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_mapping)
    assert runtime.failure_reason is not None
    assert state._activity_counters()["cow_copy_intents"] == 0


def test_joint_cow_offset_mismatch_is_rejected_before_any_move():
    _runtime, allocator, events = _install()
    batch, _plans = _batch(allocator, events)
    malformed = _plan(sliding_source_offset=1)

    with pytest.raises(RuntimeError, match="joint Full/SWA"):
        lowering._validate_joint_hybrid_tails((malformed,))

    assert events == []
    assert not torch.count_nonzero(batch.req_to_token_pool.req_to_token[:, 8:])


def test_b4_late_old_lut_fault_preflights_entire_batch_with_zero_mutation():
    _runtime, allocator, events = _install()
    batch, plans = _batch(allocator, events, count=4)
    last_old_full = lowering._tail_locations(
        plans[-1].by_class[0], source=True, device=batch.device
    )
    allocator.full_to_swa_index_mapping[last_old_full[-1]] += 1
    before_rows = batch.req_to_token_pool.req_to_token.clone()
    before_mapping = allocator.full_to_swa_index_mapping.clone()

    lowering._validate_joint_hybrid_tails(plans)
    with pytest.raises(RuntimeError, match="candidate mirror"):
        lowering._preflight_cow_mirrors(batch, plans, (False,) * 4)

    assert events == []
    assert torch.equal(batch.req_to_token_pool.req_to_token, before_rows)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_mapping)
    assert state._activity_counters()["cow_copy_intents"] == 0
