from __future__ import annotations

import sys
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.plugin.mirror_cleanup as mirror_cleanup  # noqa: E402
import orbitkv_sglang.plugin.prefix_cache as prefix_cache  # noqa: E402
import orbitkv_sglang.plugin.state as state  # noqa: E402
from orbitkv_sglang.config import ClassConfig, ManagerPlanConfig  # noqa: E402
from orbitkv_sglang.config import load_config  # noqa: E402
from orbitkv_sglang.ffi import CtypesManagerFactory  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaRegistration,
    ArenaIdentity,
    DETACHED_CLEAR,
    DETACHED_COPY_ON_WRITE,
    DETACHED_REPLACE,
    DETACHED_RETENTION,
    DetachedBinding,
    MirrorCandidateTransition,
    MirrorCleanupBinding,
    MirrorCleanupItem,
    PageLease,
    ReclamationCertificate,
    ReclamationLease,
    CanonicalRuntime,
    ManagerCreateSettings,
)


PAGE_TOKENS = 16
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class _ReadyEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


@pytest.fixture(scope="session")
def native_ffi_library() -> Path:
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


def _install_hybrid():
    config = ManagerPlanConfig(
        plan_path=Path("plan.json"),
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:mirror-test",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "full"), _class(1, "sliding")),
    )
    arenas = {
        item.class_id: ArenaIdentity(
            engine_epoch=1,
            pool_epoch=2 + item.class_id,
            pool_id=item.pool_id,
            class_id=item.class_id,
            backend_domain=item.backend_domain,
            page_count=512,
            page_tokens=PAGE_TOKENS,
            backend_base_index=0,
            first_page_id=1 + item.class_id * 512,
        )
        for item in config.classes
    }
    runtime = SimpleNamespace(arenas_by_class=arenas)
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 512),
        runtime=runtime,
    )
    pool = SimpleNamespace(
        req_to_token=torch.zeros((8, 512), dtype=torch.int32),
        max_context_len=512,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace(
        full_to_swa_index_mapping=torch.zeros((8192,), dtype=torch.int64)
    )
    state._ALLOCATOR = allocator
    return pool, allocator


def _certificate(
    class_id: int,
    backend_index: int,
    *,
    begin: int = 0,
    end: int = PAGE_TOKENS,
) -> ReclamationCertificate:
    return ReclamationCertificate(
        reclamation=ReclamationLease(1, class_id * 512 + backend_index + 1, 1),
        page=PageLease(
            1,
            2 + class_id,
            class_id + 1,
            class_id * 512 + backend_index + 1,
            class_id + 1,
        ),
        class_id=class_id,
        backend_domain=class_id + 1,
        logical_ordinal=begin // PAGE_TOKENS,
        backend_index=backend_index,
        token_begin=begin,
        token_end_exclusive=end,
        completion_domain=1,
        completion_value=1,
    )


def _paired_retirements(count: int):
    full = tuple(_certificate(0, index + 1) for index in range(count))
    sliding = tuple(_certificate(1, index + 129) for index in range(count))
    return full, sliding


def _install_pairs(mapping, full, sliding) -> None:
    for full_certificate, sliding_certificate in zip(full, sliding, strict=True):
        full_start = (full_certificate.backend_index + 1) * PAGE_TOKENS
        sliding_start = (sliding_certificate.backend_index + 1) * PAGE_TOKENS
        mapping[full_start : full_start + PAGE_TOKENS] = torch.arange(
            sliding_start,
            sliding_start + PAGE_TOKENS,
            dtype=torch.int64,
        )


def test_cold_prefix_eviction_scans_global_aliases_once_per_collective(monkeypatch):
    pool, allocator = _install_hybrid()
    full, sliding = _paired_retirements(64)
    _install_pairs(allocator.full_to_swa_index_mapping, full, sliding)
    original_isin = torch.isin
    scans = []
    arange_sizes = []
    original_arange = torch.arange

    def tracked_isin(elements, test_elements, *args, **kwargs):
        scans.append((int(elements.numel()), int(test_elements.numel())))
        return original_isin(elements, test_elements, *args, **kwargs)

    def tracked_arange(*args, **kwargs):
        result = original_arange(*args, **kwargs)
        arange_sizes.append(int(result.numel()))
        return result

    monkeypatch.setattr(torch, "isin", tracked_isin)
    monkeypatch.setattr(torch, "arange", tracked_arange)
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    plan = coordinator.preflight((), full + sliding)

    assert scans == [(8192, 64 * PAGE_TOKENS)]
    assert 8192 not in arange_sizes
    assert state._activity_counters()["prefix_global_alias_scans"] == 1
    coordinator.commit(plan)
    assert not torch.count_nonzero(allocator.full_to_swa_index_mapping)


def test_online_swa_detach_uses_sparse_row_authority_without_global_scan(
    monkeypatch,
):
    pool, allocator = _install_hybrid()
    sliding = _certificate(1, 129)
    full_locations = torch.arange(32, 48, dtype=torch.int64)
    sliding_locations = torch.arange(2080, 2096, dtype=torch.int64)
    pool.req_to_token[1, :PAGE_TOKENS] = full_locations.to(torch.int32)
    allocator.full_to_swa_index_mapping[full_locations] = sliding_locations
    req = SimpleNamespace(
        req_pool_idx=1,
        prefix_indices=full_locations.clone(),
        kv=SimpleNamespace(kv_allocated_len=PAGE_TOKENS, swa_evicted_seqlen=0),
    )
    detached = DetachedBinding(
        old=sliding.page,
        replacement=PageLease(0, 0, 0, 0, 0),
        logical_ordinal=0,
        old_backend_index=sliding.backend_index,
        replacement_backend_index=0,
        token_begin=0,
        token_end_exclusive=PAGE_TOKENS,
        class_id=1,
        backend_domain=2,
        action=DETACHED_CLEAR,
        reason=DETACHED_RETENTION,
    )
    item = MirrorCleanupItem(
        context=mirror_cleanup._MirrorCleanupContext(req, 1),
        detached=(detached,),
        releasing=False,
        boundary=PAGE_TOKENS,
    )
    scans = []
    monkeypatch.setattr(torch, "isin", lambda *_args, **_kwargs: scans.append(True))
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    plan = coordinator.preflight((item,), (sliding,))
    coordinator.commit(plan)

    assert scans == []
    assert state._activity_counters()["prefix_global_alias_scans"] == 0
    assert not torch.count_nonzero(
        allocator.full_to_swa_index_mapping[full_locations]
    )
    assert torch.equal(pool.req_to_token[1, :PAGE_TOKENS], full_locations.int())


def test_cold_alias_fault_has_zero_mutation_and_no_success_counter():
    pool, allocator = _install_hybrid()
    full, sliding = _paired_retirements(4)
    _install_pairs(allocator.full_to_swa_index_mapping, full, sliding)
    external_full = 400 * PAGE_TOKENS
    allocator.full_to_swa_index_mapping[external_full] = (
        (sliding[0].backend_index + 1) * PAGE_TOKENS
    )
    before = allocator.full_to_swa_index_mapping.clone()
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        coordinator.preflight((), full + sliding)

    assert torch.equal(allocator.full_to_swa_index_mapping, before)
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 0
    assert counters["prefix_global_alias_scans"] == 0


def _page(class_id: int, backend_index: int) -> PageLease:
    return PageLease(
        1,
        2 + class_id,
        class_id + 1,
        class_id * 512 + backend_index + 1,
        class_id + 1,
    )


def _candidate(
    class_id: int,
    backend_index: int,
    ordinal: int,
    end: int,
    *,
    source_backend_index: int | None = None,
    copied_end: int = 0,
    retiring: bool = False,
) -> MirrorCandidateTransition:
    begin = ordinal * PAGE_TOKENS
    return MirrorCandidateTransition(
        destination=_page(class_id, backend_index),
        source=(
            PageLease(0, 0, 0, 0, 0)
            if source_backend_index is None
            else _page(class_id, source_backend_index)
        ),
        logical_ordinal=ordinal,
        destination_backend_index=backend_index,
        source_backend_index=source_backend_index or 0,
        token_begin=begin,
        token_end_exclusive=end,
        copied_token_begin=begin if source_backend_index is not None else 0,
        copied_token_end_exclusive=copied_end,
        class_id=class_id,
        backend_domain=class_id + 1,
        retiring=retiring,
    )


def _shared_tail_window_case():
    pool, allocator = _install_hybrid()
    row = 1
    full_old_zero = torch.arange(16, 32, dtype=torch.int64)
    full_old_tail = torch.arange(32, 34, dtype=torch.int64)
    full_dest_tail = torch.arange(48, 64, dtype=torch.int64)
    full_fresh = torch.arange(64, 80, dtype=torch.int64)
    full_last = torch.arange(80, 82, dtype=torch.int64)
    pool.req_to_token[row, 0:16] = full_old_zero.int()
    pool.req_to_token[row, 16:32] = full_dest_tail.int()
    pool.req_to_token[row, 32:48] = full_fresh.int()
    pool.req_to_token[row, 48:50] = full_last.int()
    allocator.full_to_swa_index_mapping[full_old_zero] = torch.arange(16, 32)
    allocator.full_to_swa_index_mapping[full_old_tail] = torch.arange(32, 34)
    allocator.full_to_swa_index_mapping[full_dest_tail] = torch.arange(48, 64)
    allocator.full_to_swa_index_mapping[full_fresh] = torch.arange(64, 80)
    allocator.full_to_swa_index_mapping[full_last] = torch.arange(80, 82)
    req = SimpleNamespace(
        req_pool_idx=row,
        prefix_indices=pool.req_to_token[row, :18].to(torch.int64).clone(),
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
    )
    detached = (
        DetachedBinding(
            old=_page(0, 1),
            replacement=_page(0, 2),
            logical_ordinal=1,
            old_backend_index=1,
            replacement_backend_index=2,
            token_begin=16,
            token_end_exclusive=18,
            class_id=0,
            backend_domain=1,
            action=DETACHED_REPLACE,
            reason=DETACHED_COPY_ON_WRITE,
        ),
        DetachedBinding(
            old=_page(1, 0),
            replacement=PageLease(0, 0, 0, 0, 0),
            logical_ordinal=0,
            old_backend_index=0,
            replacement_backend_index=0,
            token_begin=0,
            token_end_exclusive=16,
            class_id=1,
            backend_domain=2,
            action=DETACHED_CLEAR,
            reason=DETACHED_RETENTION,
        ),
        DetachedBinding(
            old=_page(1, 1),
            replacement=PageLease(0, 0, 0, 0, 0),
            logical_ordinal=1,
            old_backend_index=1,
            replacement_backend_index=0,
            token_begin=16,
            token_end_exclusive=18,
            class_id=1,
            backend_domain=2,
            action=DETACHED_CLEAR,
            reason=DETACHED_RETENTION,
        ),
    )
    candidates = (
        _candidate(0, 2, 1, 32, source_backend_index=1, copied_end=18),
        _candidate(
            1,
            2,
            1,
            32,
            source_backend_index=1,
            copied_end=18,
            retiring=True,
        ),
        _candidate(0, 3, 2, 48),
        _candidate(1, 3, 2, 48),
        _candidate(0, 4, 3, 50),
        _candidate(1, 4, 3, 50),
    )
    item = MirrorCleanupItem(
        context=mirror_cleanup._MirrorCleanupContext(req, row),
        detached=detached,
        releasing=False,
        boundary=50,
        candidates=candidates,
    )
    retirement = _certificate(1, 2, begin=16, end=32)
    return pool, allocator, req, item, retirement


def test_shared_cow_tail_leaving_swa_window_clears_only_transient_destination():
    pool, allocator, req, item, retirement = _shared_tail_window_case()
    before_row = pool.req_to_token.clone()
    before_prefix = req.prefix_indices.clone()
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    plan = coordinator.preflight((item,), (retirement,))
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(req.prefix_indices, before_prefix)
    mapping = allocator.full_to_swa_index_mapping
    assert torch.equal(mapping[32:34], torch.arange(32, 34))
    assert not torch.count_nonzero(mapping[48:64])
    assert torch.equal(mapping[64:80], torch.arange(64, 80))
    assert req.kv.swa_evicted_seqlen == 32


@pytest.mark.parametrize("fault", ("source", "destination", "span", "retiring"))
def test_candidate_authority_fault_is_zero_mutation_and_unacknowledged(fault):
    pool, allocator, req, item, retirement = _shared_tail_window_case()
    candidates = list(item.candidates)
    sliding = candidates[1]
    if fault == "source":
        sliding = replace(sliding, source=_page(1, 0))
    elif fault == "destination":
        sliding = replace(sliding, destination_backend_index=3)
    elif fault == "span":
        sliding = replace(sliding, token_end_exclusive=31)
    else:
        sliding = replace(sliding, retiring=False)
    candidates[1] = sliding
    corrupted = replace(item, candidates=tuple(candidates))
    before_row = pool.req_to_token.clone()
    before_prefix = req.prefix_indices.clone()
    before_mapping = allocator.full_to_swa_index_mapping.clone()
    acknowledgements = []
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    with pytest.raises(RuntimeError):
        plan = coordinator.preflight((corrupted,), (retirement,))
        coordinator.commit(plan)
        coordinator.synchronize(plan)
        coordinator.finalize(plan)
        acknowledgements.append(True)

    assert acknowledgements == []
    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(req.prefix_indices, before_prefix)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_mapping)
    assert req.kv.swa_evicted_seqlen == 0


def test_pure_swa_cow_source_leaving_window_clears_current_destination_row():
    config = ManagerPlanConfig(
        plan_path=Path("plan.json"),
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:pure-swa-candidate",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "sliding"),),
    )
    runtime = SimpleNamespace(
        arenas_by_class={
            0: ArenaIdentity(1, 2, 1, 0, 1, 64, 16, 0, 1)
        }
    )
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    pool = SimpleNamespace(
        req_to_token=torch.zeros((4, 128), dtype=torch.int32),
        max_context_len=128,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace()
    row = 1
    destination = torch.arange(48, 64, dtype=torch.int64)
    pool.req_to_token[row, 16:32] = destination.int()
    req = SimpleNamespace(
        req_pool_idx=row,
        prefix_indices=torch.cat(
            (torch.zeros(16, dtype=torch.int64), destination[:2])
        ),
        kv=SimpleNamespace(kv_allocated_len=32, swa_evicted_seqlen=0),
    )
    detached = DetachedBinding(
        old=_page(0, 1),
        replacement=PageLease(0, 0, 0, 0, 0),
        logical_ordinal=1,
        old_backend_index=1,
        replacement_backend_index=0,
        token_begin=16,
        token_end_exclusive=18,
        class_id=0,
        backend_domain=1,
        action=DETACHED_CLEAR,
        reason=DETACHED_RETENTION,
    )
    candidate = _candidate(
        0,
        2,
        1,
        32,
        source_backend_index=1,
        copied_end=18,
        retiring=True,
    )
    item = MirrorCleanupItem(
        mirror_cleanup._MirrorCleanupContext(req, row),
        (detached,),
        False,
        32,
        (candidate,),
    )
    retirement = _certificate(0, 2, begin=16, end=32)
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    plan = coordinator.preflight((item,), (retirement,))
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert not torch.count_nonzero(pool.req_to_token[row, 16:32])
    assert not torch.count_nonzero(req.prefix_indices)
    assert req.kv.swa_evicted_seqlen == 32


def _emulate_native_lowering(batch, plans, pool, allocator, row):
    vectors = {}
    previous = plans[0].previous_boundary
    target = plans[0].target_boundary
    for class_id, spec in plans[0].by_class.items():
        values = torch.zeros(target, dtype=torch.int64)
        if previous:
            full_existing = pool.req_to_token[row, :previous].to(torch.int64)
            values[:previous] = (
                full_existing
                if class_id == 0
                else allocator.full_to_swa_index_mapping[full_existing]
            )
        partial = previous % PAGE_TOKENS
        if partial:
            logical_begin = previous // PAGE_TOKENS * PAGE_TOKENS
            logical_end = min(logical_begin + PAGE_TOKENS, target)
            page_begin = spec.last_location - partial + 1
            values[logical_begin:logical_end] = torch.arange(
                page_begin,
                page_begin + logical_end - logical_begin,
                dtype=torch.int64,
            )
        first_new_ordinal = (previous + PAGE_TOKENS - 1) // PAGE_TOKENS
        for offset, page_id in enumerate(spec.exact_new_pages):
            logical_ordinal = first_new_ordinal + offset
            logical_begin = logical_ordinal * PAGE_TOKENS
            logical_end = min(logical_begin + PAGE_TOKENS, target)
            values[logical_begin:logical_end] = torch.arange(
                page_id * PAGE_TOKENS,
                page_id * PAGE_TOKENS + logical_end - logical_begin,
                dtype=torch.int64,
            )
        vectors[class_id] = values
    pool.req_to_token[row, :target] = vectors[0].int()
    allocator.full_to_swa_index_mapping[vectors[0]] = vectors[1]
    return batch


def test_native_shared_cow_tail_window_candidate_cleanup(
    tmp_path: Path, native_ffi_library: Path
):
    plan_path = tmp_path / "native-candidate-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "page_tokens": PAGE_TOKENS,
                "classes": [
                    {
                        "name": "full",
                        "layers": [0],
                        "retention": "full",
                        "bytes_per_token_per_layer": 128,
                        "window_tokens": None,
                    },
                    {
                        "name": "swa",
                        "layers": [1],
                        "retention": "sliding",
                        "bytes_per_token_per_layer": 128,
                        "window_tokens": 18,
                    },
                ],
            }
        )
    )
    config = load_config(
        {
            "ORBITKV_PLAN": str(plan_path),
            "ORBITKV_LIBRARY": str(native_ffi_library),
        }
    )
    registrations = tuple(
        ArenaRegistration(
            item.class_id, item.pool_id, item.backend_domain, 64, 0
        )
        for item in config.classes
    )
    manager = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(8, 4, 8, 128, 64),
        registrations,
    )
    runtime = CanonicalRuntime(config, manager)
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 128),
        runtime=runtime,
    )
    pool = SimpleNamespace(
        req_to_token=torch.zeros((4, 128), dtype=torch.int32),
        max_context_len=128,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace(
        full_to_swa_index_mapping=torch.zeros((2048,), dtype=torch.int64)
    )
    state._ALLOCATOR = allocator
    req = SimpleNamespace(
        req_pool_idx=1,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        kv=SimpleNamespace(kv_allocated_len=0, swa_evicted_seqlen=0),
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    runtime.bind_request_rows((("r", 1, True),))
    first, plans = runtime.prepare_batch((("r", 18),))
    runtime.bind_reclamation_cleanup(
        "r",
        MirrorCleanupBinding(
            coordinator, mirror_cleanup._MirrorCleanupContext(req, 1)
        ),
    )
    _emulate_native_lowering(first, plans, pool, allocator, 1)
    req.kv.kv_allocated_len = 18
    runtime.mark_lowered(first)
    runtime.submit_batch(first)
    runtime.mark_forward(first)
    runtime.register_event(first, _ReadyEvent(), 1)
    runtime.poll()
    assert runtime.record_for("r").boundary == 18

    runtime.request_acquire_batch(("t",))
    runtime.bind_request_rows((("t", 2, True),))
    runtime.request_fork_batch((("r", "t"),))
    pool.req_to_token[2, :18] = pool.req_to_token[1, :18]
    before_shared = allocator.full_to_swa_index_mapping[
        pool.req_to_token[2, 16:18].long()
    ].clone()
    second, plans = runtime.prepare_batch((("r", 50),))
    _emulate_native_lowering(second, plans, pool, allocator, 1)
    req.kv.kv_allocated_len = 50
    runtime.mark_lowered(second)
    runtime.submit_batch(second)
    runtime.mark_forward(second)
    runtime.register_event(second, _ReadyEvent(), 2)
    runtime.poll()

    assert runtime.record_for("r").boundary == 50
    assert req.kv.swa_evicted_seqlen == 32
    retired_full = pool.req_to_token[1, 16:32].long()
    assert not torch.count_nonzero(
        allocator.full_to_swa_index_mapping[retired_full]
    )
    shared_full = pool.req_to_token[2, 16:18].long()
    assert torch.equal(
        allocator.full_to_swa_index_mapping[shared_full], before_shared
    )
    assert manager.performance_counters["acknowledge_reclamations_batch_calls"] == 1
    runtime.fail_stop("native candidate cleanup test teardown")
    runtime.close()


def test_native_live_empty_tree_shutdown_destroys_handle_idempotently(
    tmp_path: Path, native_ffi_library: Path
):
    plan_path = tmp_path / "native-live-shutdown-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "page_tokens": PAGE_TOKENS,
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
    config = load_config(
        {
            "ORBITKV_PLAN": str(plan_path),
            "ORBITKV_LIBRARY": str(native_ffi_library),
        }
    )
    registration = ArenaRegistration(0, 1, 1, 8, 0)
    manager = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(2, 2, 8, 8, 32),
        (registration,),
    )
    runtime = CanonicalRuntime(config, manager)
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(2, 32, 64),
        runtime=runtime,
    )
    runtime.request_acquire_batch(("live",))
    cache = object.__new__(prefix_cache.OrbitKvPrefixCache)
    cache._released = False

    cache.release_host_resources()

    assert runtime.failure_reason == (
        "shutdown encountered live OrbitKV request ownership"
    )
    assert not manager._handle.value
    assert cache._released is True
    cache.release_host_resources()
    assert not manager._handle.value
