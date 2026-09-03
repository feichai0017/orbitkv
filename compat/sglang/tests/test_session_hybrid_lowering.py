from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

import test_session_lowering as base
import orbitkv_sglang.bridge.location_validation as location_validation
import orbitkv_sglang.bridge.lowering as lowering
import orbitkv_sglang.bridge.physical_lowering as physical_lowering
import orbitkv_sglang.bridge.state as state
from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.ffi.session_types import EngineStepPlan
from orbitkv_sglang.runtime import (
    TAIL_COPY_ON_WRITE,
    TAIL_IN_PLACE,
    TAIL_NONE,
    ArenaIdentity,
    ClassLowering,
    CopyIntent,
    FailStopped,
    PageLease,
    TailAction,
    WriteIntent,
)


def _class(class_id: int, retention: str) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 3,
        backend_domain=class_id + 1,
        name=retention,
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=32 if retention == "sliding" else None,
        period_blocks=3 if retention == "sliding" else None,
        storage="token_kv",
    )


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:session-hybrid-lowering-test",
        page_tokens=base.PAGE_TOKENS,
        classes=(_class(0, "full"), _class(1, "sliding")),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_sliding_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )


def _lease(arena: ArenaIdentity, page_id: int, generation: int) -> PageLease:
    return PageLease(
        arena.engine_epoch, arena.pool_epoch, generation, page_id, arena.pool_id
    )


class _MovePool:
    def __init__(self, name: str, events: list[Any], *, fail: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail = fail

    def move_kv_cache(self, destinations: Any, sources: Any) -> None:
        self.events.append(
            (self.name, tuple(destinations.tolist()), tuple(sources.tolist()))
        )
        if self.fail:
            raise RuntimeError(f"injected {self.name} failure")


class _Allocator:
    def __init__(
        self, events: list[Any], *, fail_swa: bool = False, fail_lut: bool = False
    ) -> None:
        self.events = events
        self.fail_lut = fail_lut
        self.full_to_swa_index_mapping = torch.zeros((256,), dtype=torch.int64)
        self.kvcache = SimpleNamespace(
            full_kv_pool=_MovePool("full-copy", events),
            swa_kv_pool=_MovePool("swa-copy", events, fail=fail_swa),
        )
        self._orbitkv_free_group_state = "idle"
        self.free_group = []

    def get_kvcache(self) -> Any:
        return self.kvcache

    def set_full_to_swa_mapping(self, full: Any, swa: Any) -> None:
        self.events.append("lut")
        if self.fail_lut:
            raise RuntimeError("injected LUT failure")
        self.full_to_swa_index_mapping[full] = swa


class _Runtime(base._Runtime):
    def __init__(self, events: list[Any], *, mismatched_tail: bool = False) -> None:
        super().__init__(events)
        self.mismatched_tail = mismatched_tail
        self.arenas = (
            ArenaIdentity(7, 11, 3, 0, 1, 32, base.PAGE_TOKENS, 0, 1),
            ArenaIdentity(7, 12, 4, 1, 2, 32, base.PAGE_TOKENS, 0, 33),
        )
        self.arenas_by_class = {item.class_id: item for item in self.arenas}

    def _step(self, view: Any, target: int) -> EngineStepPlan:
        previous = int(view.boundary)
        if previous == 0:
            actions = (
                TailAction(0, TAIL_NONE, 0, 0, base.ZERO_PAGE, base.ZERO_PAGE, 0),
                TailAction(1, TAIL_NONE, 0, 0, base.ZERO_PAGE, base.ZERO_PAGE, 0),
            )
            copies: tuple[CopyIntent, ...] = ()
            writes = (WriteIntent(1, 1, 0), WriteIntent(1, 35, 0))
            lowerings = (
                ClassLowering(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, target),
                ClassLowering(1, 0, 1, 1, 0, 0, 1, 1, 0, 0, target),
            )
        elif previous == 4:
            slot = int(view.request_id) - 1
            full_source_index = slot * 2
            full_destination_index = full_source_index + 1
            swa_source_index = full_source_index + 2
            swa_destination_index = swa_source_index + 1
            full_source = _lease(
                self.arenas[0], 1 + full_source_index, 1
            )
            full_destination = _lease(
                self.arenas[0], 1 + full_destination_index, 2
            )
            swa_source = _lease(self.arenas[1], 33 + swa_source_index, 1)
            swa_destination = _lease(
                self.arenas[1], 33 + swa_destination_index, 2
            )
            full_copy = CopyIntent(
                0, 1, 4, 0, 0, full_source, full_destination,
                full_source_index, full_destination_index, 0
            )
            if self.mismatched_tail:
                actions = (
                    TailAction(
                        0,
                        TAIL_COPY_ON_WRITE,
                        4,
                        0,
                        full_source,
                        full_destination,
                        0,
                    ),
                    TailAction(
                        1, TAIL_IN_PLACE, 4, 0, swa_source, swa_source, 0
                    ),
                )
                copies = (full_copy,)
                lowerings = (
                    ClassLowering(0, 0, 0, 1, 0, 1, 0, 0, 0, 4, target),
                    ClassLowering(1, 0, 1, 1, 1, 0, 0, 0, 0, 4, target),
                )
            else:
                swa_copy = CopyIntent(
                    1, 2, 4, 0, 0, swa_source, swa_destination,
                    swa_source_index, swa_destination_index, 0
                )
                actions = (
                    TailAction(
                        0,
                        TAIL_COPY_ON_WRITE,
                        4,
                        0,
                        full_source,
                        full_destination,
                        0,
                    ),
                    TailAction(
                        1,
                        TAIL_COPY_ON_WRITE,
                        4,
                        0,
                        swa_source,
                        swa_destination,
                        0,
                    ),
                )
                copies = (full_copy, swa_copy)
                lowerings = (
                    ClassLowering(0, 0, 0, 1, 0, 1, 0, 0, 0, 4, target),
                    ClassLowering(1, 0, 1, 1, 1, 1, 0, 0, 0, 4, target),
                )
            writes = ()
        else:  # pragma: no cover - fixture owns the supported geometries
            raise AssertionError(previous)
        return EngineStepPlan(
            view.request_id,
            view.view_version,
            view.view_version + 1,
            previous,
            target,
            lowerings,
            actions,
            copies,
            writes,
        )


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    previous: int,
    fail_row: bool = False,
    fail_swa: bool = False,
    fail_lut: bool = False,
    mismatched_tail: bool = False,
) -> tuple[_Runtime, Any, _Allocator, list[Any]]:
    import sglang.srt.mem_cache.allocation as allocation

    events: list[Any] = []
    runtime = _Runtime(events, mismatched_tail=mismatched_tail)
    allocator = _Allocator(events, fail_swa=fail_swa, fail_lut=fail_lut)
    state._install_test_state(
        config=_config(), limits=state.RuntimeLimits(8, 64, 64), runtime=runtime
    )
    state._ALLOCATOR = allocator
    monkeypatch.setattr(state, "_uses_runtime_session", lambda: True)
    batch = base._batch(events, allocator, runtime, previous=previous)
    batch.req_to_token_pool.fail_write = fail_row
    if previous:
        allocator.full_to_swa_index_mapping[16:20] = torch.arange(48, 52)

    def alloc_req_slots(_pool: Any, reqs: Any, _tree: Any) -> list[int]:
        for index, req in enumerate(reqs, start=1):
            if req.req_pool_idx is None:
                req.req_pool_idx = index
        return [int(req.req_pool_idx) for req in reqs]

    def write_cache_indices(
        out: Any,
        _rd: Any,
        rows: Any,
        _pd: Any,
        prefixes: Any,
        _td: Any,
        targets: Any,
        _ed: Any,
        _ec: Any,
        _pt: Any,
        pool: Any,
    ) -> None:
        events.append("mirror")
        offset = 0
        for row, begin, end in zip(rows, prefixes, targets, strict=True):
            count = int(end - begin)
            pool.req_to_token[int(row), int(begin):int(end)] = out[
                offset:offset + count
            ].to(torch.int32)
            offset += count

    monkeypatch.setattr(allocation, "alloc_req_slots", alloc_req_slots)
    monkeypatch.setattr(allocation, "write_cache_indices", write_cache_indices)
    device_module = base._DeviceModule()
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    batch._test_device_module = device_module
    return runtime, batch, allocator, events


def _expand_existing_decode_batch(
    batch: Any, allocator: _Allocator, count: int
) -> None:
    runtime = batch._test_runtime
    for row in range(2, count + 1):
        req = SimpleNamespace(
            rid=f"request-{row}",
            req_pool_idx=row,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            kv=SimpleNamespace(kv_allocated_len=4, swa_evicted_seqlen=0),
        )
        key = ("str", req.rid)
        request_id = runtime.seed(key, row, 4)
        req._orbitkv_request_key = key
        req._orbitkv_engine_request_id = request_id
        slot = int(request_id) - 1
        full_start = (slot * 2 + 1) * base.PAGE_TOKENS
        swa_start = (slot * 2 + 3) * base.PAGE_TOKENS
        full = torch.arange(full_start, full_start + 4, dtype=torch.int32)
        swa = torch.arange(swa_start, swa_start + 4, dtype=torch.int64)
        batch.req_to_token_pool.req_to_token[row, :4] = full
        allocator.full_to_swa_index_mapping[full.long()] = swa
        base._install_shared_prefix(
            req, 4, provisional=False, locations=full
        )
        batch.tree_cache._session_requests[key] = base.session_cache._RequestEntry(
            req, key, request_id, row
        )
        batch.reqs.append(req)
    rows = torch.arange(1, count + 1, dtype=torch.int64)
    batch.seq_lens_cpu = torch.full((count,), 4, dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.clone()
    batch.req_pool_indices_cpu = rows
    batch.req_pool_indices = rows.clone()


def _record_cow_commit(monkeypatch: pytest.MonkeyPatch, events: list[Any]) -> None:
    original = lowering._commit_cow_mirrors

    def commit(plan: Any, context: Any = None) -> None:
        events.append("cow-mirror")
        original(plan, context)

    monkeypatch.setattr(lowering, "_commit_cow_mirrors", commit)


def test_hybrid_extend_commits_full_row_then_lut_then_cow_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, allocator, events = _install(monkeypatch, previous=0)
    _record_cow_commit(monkeypatch, events)
    monkeypatch.setattr(
        lowering, "_lower_all_extend",
        lambda *_args: {0: torch.arange(16, 20), 1: torch.arange(48, 52)},
    )

    out, _rows_device, _rows_cpu = lowering._alloc_for_extend(batch)

    assert out.tolist() == [16, 17, 18, 19]
    assert events.index("submit") < events.index("mirror")
    assert (
        events.index("mirror")
        < events.index("lut")
        < events.index("cow-mirror")
    )
    assert batch.req_to_token_pool.req_to_token[1, :4].tolist() == out.int().tolist()
    assert allocator.full_to_swa_index_mapping[out].tolist() == [48, 49, 50, 51]
    assert runtime.failure_reason is None


@pytest.mark.parametrize("failure", ("lut", "cow-mirror"))
def test_hybrid_extend_mirror_failure_quarantines_exact_submitted_ticket(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    runtime, batch, _allocator, events = _install(
        monkeypatch, previous=0, fail_lut=failure == "lut"
    )
    if failure == "cow-mirror":

        def fail_commit(*_args: Any) -> None:
            raise RuntimeError("injected mirror failure")

        monkeypatch.setattr(lowering, "_commit_cow_mirrors", fail_commit)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: torch.arange(16, 20), 1: torch.arange(48, 52)},
    )

    with pytest.raises(FailStopped, match="submitted execution"):
        lowering._alloc_for_extend(batch)

    assert runtime.submitted and runtime.quarantined_submitted
    assert batch._orbitkv_session_ticket is runtime.quarantined_submitted[0]
    assert events.index("submit") < events.index("mirror") < events.index("lut")
    assert batch.req_to_token_pool.req_to_token[1, :4].tolist() == [
        16, 17, 18, 19
    ]
    assert batch.reqs[0].kv is None


def test_hybrid_decode_copies_both_pools_before_physical_mirror_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, allocator, events = _install(monkeypatch, previous=4)
    _record_cow_commit(monkeypatch, events)
    monkeypatch.setattr(
        lowering, "_lower_all_decode",
        lambda *_args: {0: torch.tensor([36]), 1: torch.tensor([68])},
    )

    lowering._alloc_for_decode(batch, 1)

    names = [item[0] if isinstance(item, tuple) else item for item in events]
    assert (
        names.index("full-copy")
        < names.index("swa-copy")
        < names.index("submit")
    )
    assert names.index("submit") < names.index("mirror") < names.index("lut")
    assert names.index("lut") < names.index("cow-mirror")
    assert batch.req_to_token_pool.req_to_token[1, :5].tolist() == [
        32, 33, 34, 35, 36
    ]
    assert allocator.full_to_swa_index_mapping[32:37].tolist() == [
        64, 65, 66, 67, 68
    ]
    receipts = runtime.submitted[0].steps[0].copy_receipts
    assert tuple(
        (
            item.class_id,
            item.backend_domain,
            item.token_count,
            item.source_token_offset,
            item.destination_token_offset,
            item.source_backend_index,
            item.destination_backend_index,
            item.ordered_before_writes,
        )
        for item in receipts
    ) == (
        (0, 1, 4, 0, 0, 0, 1, True),
        (1, 2, 4, 0, 0, 2, 3, True),
    )


def test_hybrid_joint_tail_mismatch_aborts_before_physical_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, allocator, events = _install(
        monkeypatch, previous=4, mismatched_tail=True
    )
    before_lut = allocator.full_to_swa_index_mapping.clone()
    monkeypatch.setattr(
        lowering, "_lower_all_decode",
        lambda *_args: pytest.fail("tensor lowering must not start"),
    )

    with pytest.raises(RuntimeError, match="joint Full/SWA"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.aborted and not runtime.submitted
    assert not runtime.quarantined_prepared and "full-copy" not in events
    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)


def test_hybrid_helpers_reject_noncanonical_class_order() -> None:
    config = _config()
    reversed_config = RuntimeConfig(
        library_path=config.library_path,
        plan_json=config.plan_json,
        plan_fingerprint=config.plan_fingerprint,
        page_tokens=config.page_tokens,
        classes=tuple(reversed(config.classes)),
        runtime_manifest_path=config.runtime_manifest_path,
        runtime_manifest_fingerprint=config.runtime_manifest_fingerprint,
        runtime_binding=config.runtime_binding,
        manager_plan_format=config.manager_plan_format,
    )
    shared = SimpleNamespace(_config=lambda: reversed_config)

    with pytest.raises(RuntimeError, match="exact ordered Full[+]SWA"):
        physical_lowering.validate_joint_hybrid_tails((), shared=shared)
    with pytest.raises(RuntimeError, match="exact ordered Full[+]SWA"):
        physical_lowering.write_hybrid_lut(
            {0: torch.tensor([16]), 1: torch.tensor([48])}, shared=shared
        )


def test_hybrid_swa_location_mismatch_quarantines_before_any_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, _allocator, events = _install(monkeypatch, previous=4)
    monkeypatch.setattr(location_validation, "VALIDATE_PHYSICAL_LOCATIONS", True)
    monkeypatch.setattr(
        lowering, "_lower_all_decode",
        lambda *_args: {0: torch.tensor([36]), 1: torch.tensor([69])},
    )

    with pytest.raises(FailStopped, match="prepared execution"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.quarantined_prepared and not runtime.submitted
    assert "full-copy" not in events and "swa-copy" not in events
    assert "mirror" not in events and "lut" not in events


def test_hybrid_b4_late_old_lut_fault_aborts_without_physical_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, allocator, events = _install(monkeypatch, previous=4)
    _expand_existing_decode_batch(batch, allocator, 4)
    last_slot = int(batch.reqs[-1]._orbitkv_engine_request_id) - 1
    last_full = (last_slot * 2 + 1) * base.PAGE_TOKENS
    allocator.full_to_swa_index_mapping[last_full + 3] += 1
    before_rows = batch.req_to_token_pool.req_to_token.clone()
    before_lut = allocator.full_to_swa_index_mapping.clone()
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: pytest.fail("tensor lowering must not start"),
    )

    with pytest.raises(RuntimeError, match="candidate mirror"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.aborted and not runtime.submitted
    assert not runtime.quarantined_prepared
    assert not any(
        item in ("full-copy", "swa-copy", "mirror", "lut")
        for item in (
            event[0] if isinstance(event, tuple) else event for event in events
        )
    )
    assert torch.equal(batch.req_to_token_pool.req_to_token, before_rows)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)


def test_hybrid_second_pool_copy_failure_never_submits_or_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, allocator, events = _install(
        monkeypatch, previous=4, fail_swa=True
    )
    before_row = batch.req_to_token_pool.req_to_token.clone()
    before_lut = allocator.full_to_swa_index_mapping.clone()
    monkeypatch.setattr(
        lowering, "_lower_all_decode",
        lambda *_args: {0: torch.tensor([36]), 1: torch.tensor([68])},
    )

    with pytest.raises(FailStopped, match="prepared execution"):
        lowering._alloc_for_decode(batch, 1)

    names = [item[0] if isinstance(item, tuple) else item for item in events]
    assert names.index("full-copy") < names.index("swa-copy")
    assert "submit" not in names and "mirror" not in names and "lut" not in names
    assert runtime.quarantined_prepared
    assert torch.equal(batch.req_to_token_pool.req_to_token, before_row)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)
    assert state._activity_counters()["cow_copy_intents"] == 0


@pytest.mark.parametrize("failure", ("row", "lut", "cow-mirror"))
def test_hybrid_post_submit_mirror_fault_quarantines_exact_ticket(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    runtime, batch, allocator, events = _install(
        monkeypatch,
        previous=4,
        fail_row=failure == "row",
        fail_lut=failure == "lut",
    )
    if failure == "cow-mirror":

        def fail_commit(*_args: Any) -> None:
            raise RuntimeError("injected mirror failure")

        monkeypatch.setattr(lowering, "_commit_cow_mirrors", fail_commit)
    monkeypatch.setattr(
        lowering, "_lower_all_decode",
        lambda *_args: {0: torch.tensor([36]), 1: torch.tensor([68])},
    )

    with pytest.raises(FailStopped, match="submitted execution"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.submitted and runtime.quarantined_submitted
    assert batch._orbitkv_session_ticket is runtime.quarantined_submitted[0]
    assert batch.reqs[0].kv.kv_allocated_len == 4
    if failure == "row":
        assert int(batch.req_to_token_pool.req_to_token[1, 4]) == 0
        assert "lut" not in events
        assert int(allocator.full_to_swa_index_mapping[36]) == 0
    else:
        assert int(batch.req_to_token_pool.req_to_token[1, 4]) == 36
    if failure == "cow-mirror":
        assert int(allocator.full_to_swa_index_mapping[36]) == 68
