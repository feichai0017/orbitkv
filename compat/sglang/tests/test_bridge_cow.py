from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.lowering as lowering  # noqa: E402
import orbitkv_sglang.bridge.session_lowering as session_lowering  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402
from orbitkv_sglang.bridge.private_prefix import PrivatePrefixProvenance  # noqa: E402
from orbitkv_sglang.execution_plan import ClassSpec, StepPlan  # noqa: E402
from orbitkv_sglang.ffi.session_types import EngineRequestId  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    CopyIntent,
    FailStopped,
    PageLease,
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

    def submit(self, _evidence):
        self.events.append("submit")
        raise AssertionError("failed COW execution must not be submitted")

    def quarantine_prepared(self, batch_id):
        self.events.append(("quarantine", batch_id))
        self.failure_reason = "prepared execution was quarantined"
        raise FailStopped(self.failure_reason)

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


def _install(*, fail_swa=False, hybrid=True, pure_sliding=False):
    if hybrid and pure_sliding:
        raise ValueError("hybrid and pure_sliding are mutually exclusive")
    events = []
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:cow-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            (_class(0, "full"), _class(1, "sliding"))
            if hybrid
            else (
                (_class(0, "sliding"),)
                if pure_sliding
                else (_class(0, "full"),)
            )
        ),
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


def _plan(index: int = 0, *, sliding_source_offset: int = 0) -> StepPlan:
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
            ClassSpec(
                class_id,
                class_id + 1,
                (destination_index + 1) * PAGE_TOKENS + 7,
                (),
                action,
                (intent,),
                8,
                9,
            )
        )
    return StepPlan(
        EngineRequestId(index + 1),
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


def _compact_full_case(*, matches=True, private=True):
    _runtime, _allocator, events = _install(hybrid=False)
    source = _page(0, 3)
    destination = _page(0, 4)
    intent = CopyIntent(0, 1, 8, 0, 0, source, destination, 2, 3)
    plan = StepPlan(
        EngineRequestId(1),
        40,
        41,
        (
            ClassSpec(
                0, 1, 71, (),
                TailAction(0, TAIL_COPY_ON_WRITE, 8, 2, source, destination),
                (intent,), 40, 41,
            ),
        ),
    )
    retained = tuple(range(16, 28)) + ((48, 50, 52, 54) if matches else tuple(range(28, 32)))
    pool = _ReqPool(events)
    pool.req_to_token[1, : len(retained)] = torch.tensor(retained, dtype=torch.int32)
    req = SimpleNamespace(
        rid="compact-full", req_pool_idx=1,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        _orbitkv_active_kv_len=len(retained),
        _orbitkv_retained_locations=retained,
    )
    if private:
        key = ("str", req.rid)
        lease = plan.request_id
        req.prefix_indices = torch.tensor(retained, dtype=torch.int64)
        req.cache_protected_len = 0
        req._orbitkv_request_key = key
        req._orbitkv_request_lease = lease
        req._orbitkv_private_prefix = PrivatePrefixProvenance(
            req.prefix_indices, key, lease, 40
        )
    batch = SimpleNamespace(
        reqs=[req], req_to_token_pool=pool, device=torch.device("cpu")
    )
    return batch, req, plan, retained


def _patch_session_decode(monkeypatch, batch, plans, events):
    stream = object()
    device_module = SimpleNamespace(
        current_stream=lambda _device: (events.append("forward_stream"), stream)[1]
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    monkeypatch.setattr(session_lowering, "_validate_batch", lambda _batch: None)
    monkeypatch.setattr(session_lowering, "_validate_profile", lambda _batch: None)
    monkeypatch.setattr(
        session_lowering,
        "_preflight_decode_batch",
        lambda actual_batch: (
            (8,) * len(actual_batch.reqs),
            tuple(int(req.req_pool_idx) for req in actual_batch.reqs),
        ),
    )
    native_batch_id = object()
    prepared = SimpleNamespace(
        native_plan=SimpleNamespace(batch_id=native_batch_id),
        lowered=SimpleNamespace(steps=plans),
        bindings=(),
        cow_mirror_plan=object(),
    )
    monkeypatch.setattr(
        session_lowering, "_prepare_lowering", lambda *_args, **_kwargs: prepared
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
    monkeypatch.setattr(session_lowering, "_validate_locations", lambda *_args: None)
    return native_batch_id


def test_hybrid_cow_moves_each_physical_subpool_before_submit_and_write(monkeypatch):
    _runtime, allocator, events = _install()
    batch, plans = _batch(allocator, events)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(
            current_stream=lambda _device: events.append("forward_stream")
        ),
    )
    activity = lowering._execute_cow_copies(batch, plans)
    lowering._record_cow_activity(activity)

    assert events[:5] == [
        "forward_stream",
        ("full_move", tuple(range(64, 72)), tuple(range(48, 56))),
        ("swa_move", tuple(range(576, 584)), tuple(range(560, 568))),
    ]
    counters = state._activity_counters()
    assert counters["cow_copy_intents"] == 2
    assert counters["cow_move_calls"] == 2
    assert counters["cow_copied_tokens"] == 16


def test_pure_sliding_cow_uses_only_the_swa_leaf_pool(monkeypatch):
    _runtime, _allocator, events = _install(
        hybrid=False, pure_sliding=True
    )
    source = _page(0, 3)
    destination = _page(0, 4)
    intent = CopyIntent(0, 1, 8, 0, 0, source, destination, 2, 3)
    plan = StepPlan(
        EngineRequestId(1),
        8,
        9,
        (
            ClassSpec(
                0,
                1,
                71,
                (),
                TailAction(0, TAIL_COPY_ON_WRITE, 8, 0, source, destination),
                (intent,),
                8,
                9,
            ),
        ),
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(
            current_stream=lambda _device: events.append("forward_stream")
        ),
    )

    activity = lowering._execute_cow_copies(
        SimpleNamespace(device=torch.device("cpu")), (plan,)
    )

    assert activity == (1, 1, 8)
    assert events == [
        "forward_stream",
        ("swa_move", tuple(range(64, 72)), tuple(range(48, 56))),
    ]


def test_second_class_copy_launch_failure_quarantines_without_submit_or_counter(
    monkeypatch,
):
    runtime, allocator, events = _install(fail_swa=True)
    batch, plans = _batch(allocator, events)
    before_row = batch.req_to_token_pool.req_to_token.clone()
    before_mapping = allocator.full_to_swa_index_mapping.clone()
    native_batch_id = _patch_session_decode(monkeypatch, batch, plans, events)

    with pytest.raises(FailStopped, match="prepared execution") as stopped:
        lowering._alloc_for_decode(batch, 1)

    assert isinstance(stopped.value.__cause__, RuntimeError)
    assert "swa launch failed" in str(stopped.value.__cause__)
    assert any(event[0] == "full_move" for event in events if isinstance(event, tuple))
    assert any(event[0] == "swa_move" for event in events if isinstance(event, tuple))
    assert ("quarantine", native_batch_id) in events
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


def test_compact_hybrid_tail_uses_class_specific_geometry_and_lut_authority():
    _runtime, allocator, events = _install()
    full_locations = tuple(range(80, 104))
    swa_locations = tuple(range(208, 232))
    full_action = TailAction(0, 1, 8, 1, _page(0, 5), _page(0, 5))
    swa_action = TailAction(1, 3, 0, 3, PageLease(0, 0, 0, 0, 0), _page(1, 40))
    plan = StepPlan(
        EngineRequestId(1),
        48,
        49,
        (
            ClassSpec(0, 1, 88, (), full_action, (), 24, 25),
            ClassSpec(1, 2, -1, (40,), swa_action, (), 48, 49),
        ),
    )
    req = SimpleNamespace(
        rid="compact",
        req_pool_idx=1,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        _orbitkv_active_kv_len=24,
        _orbitkv_retained_locations=full_locations,
        _orbitkv_retained_swa_locations=swa_locations,
    )
    pool = _ReqPool(events)
    pool.req_to_token[1, :24] = torch.tensor(full_locations, dtype=torch.int32)
    allocator.full_to_swa_index_mapping[torch.tensor(full_locations)] = torch.tensor(
        swa_locations, dtype=torch.int64
    )
    batch = SimpleNamespace(
        reqs=[req], req_to_token_pool=pool, device=torch.device("cpu")
    )
    lowering._validate_joint_hybrid_tails((plan,))
    mirror = lowering._preflight_cow_mirrors(batch, (plan,), (False,))
    assert not mirror.assignments
    assert not mirror.mapping_assignments
    assert state._activity_counters()["cow_copy_intents"] == 0


def test_compact_full_cow_rewrites_overlapping_row_prefix_and_tuple():
    batch, req, plan, retained = _compact_full_case()
    row_before = batch.req_to_token_pool.req_to_token.clone()
    prefix_before = req.prefix_indices.clone()

    mirror = lowering._preflight_cow_mirrors(batch, (plan,), (False,))

    assert torch.equal(batch.req_to_token_pool.req_to_token, row_before)
    assert torch.equal(req.prefix_indices, prefix_before)
    assert req._orbitkv_retained_locations == retained
    lowering._commit_cow_mirrors(mirror)
    expected = retained[:12] + (64, 66, 68, 70)
    assert tuple(batch.req_to_token_pool.req_to_token[1, :16].tolist()) == expected
    assert tuple(req.prefix_indices.tolist()) == expected
    assert req._orbitkv_retained_locations == expected


def test_compact_full_cow_allows_policy_dead_source_with_zero_matches():
    batch, req, plan, retained = _compact_full_case(matches=False, private=False)
    before = batch.req_to_token_pool.req_to_token.clone()

    mirror = lowering._preflight_cow_mirrors(batch, (plan,), (False,))
    lowering._commit_cow_mirrors(mirror)

    assert not mirror.assignments
    assert torch.equal(batch.req_to_token_pool.req_to_token, before)
    assert req._orbitkv_retained_locations == retained


@pytest.mark.parametrize("fault", ("tuple", "row", "prefix"))
def test_compact_full_cow_corruption_fails_before_mutation(fault):
    batch, req, plan, retained = _compact_full_case()
    if fault == "tuple":
        req._orbitkv_retained_locations = retained[:-1] + (retained[-1] + 1,)
    elif fault == "row":
        batch.req_to_token_pool.req_to_token[1, 0] += 1
    else:
        req.prefix_indices[0] += 1
    row_before = batch.req_to_token_pool.req_to_token.clone()
    prefix_before = req.prefix_indices.clone()
    tuple_before = req._orbitkv_retained_locations

    with pytest.raises(RuntimeError):
        lowering._preflight_cow_mirrors(batch, (plan,), (False,))

    assert torch.equal(batch.req_to_token_pool.req_to_token, row_before)
    assert torch.equal(req.prefix_indices, prefix_before)
    assert req._orbitkv_retained_locations == tuple_before
