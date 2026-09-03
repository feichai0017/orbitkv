from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.mirror_cleanup as cleanup  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402
from orbitkv_sglang.ffi.session_types import EnginePrefixId  # noqa: E402
from orbitkv_sglang.bridge.private_prefix import (  # noqa: E402
    ENGINE_PREFIX_ID_MARKER,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    DETACHED_CLEAR,
    DETACHED_PREFIX_TRANSFER,
    DETACHED_REQUEST_RELEASE,
    DETACHED_RETENTION,
    DetachedBinding,
    MirrorCleanupItem,
    PageLease,
    PrefixSemanticKey,
    ReclamationCertificate,
)

PAGE_TOKENS = 16
ZERO_PAGE = PageLease(0, 0, 0, 0, 0)


@pytest.fixture(autouse=True)
def _reset_state():
    state._install_test_state()
    yield
    state._install_test_state()


def _class(class_id: int, retention: str) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 1,
        name=retention,
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=18 if retention == "sliding" else None,
        period_blocks=3 if retention == "sliding" else None,
    )


def _install():
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:session-hybrid-cleanup",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "full"), _class(1, "sliding")),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:session-hybrid-manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_sliding_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )
    arenas = {
        class_id: ArenaIdentity(
            7, 20 + class_id, class_id + 1, class_id, class_id + 1,
            128, PAGE_TOKENS, class_id * 256, class_id * 128 + 1,
        )
        for class_id in (0, 1)
    }
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 256),
        runtime=SimpleNamespace(arenas_by_class=arenas),
    )
    pool = SimpleNamespace(
        req_to_token=torch.zeros((8, 256), dtype=torch.int32),
        max_context_len=256,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace(
        full_to_swa_index_mapping=torch.zeros((4096,), dtype=torch.int64)
    )
    state._ALLOCATOR = allocator
    assert state._uses_runtime_session()
    return pool, allocator, arenas


def _page(class_id: int, backend_index: int, arena: ArenaIdentity) -> PageLease:
    return PageLease(
        arena.engine_epoch, arena.pool_epoch, 1,
        arena.first_page_id + backend_index - arena.backend_base_index,
        arena.pool_id,
    )


def _detach(
    class_id: int, ordinal: int, boundary: int, reason: int, arenas: dict[int, ArenaIdentity]
) -> DetachedBinding:
    backend_index = class_id * 256 + 20 + ordinal
    begin = ordinal * PAGE_TOKENS
    return DetachedBinding(
        _page(class_id, backend_index, arenas[class_id]), ZERO_PAGE, ordinal,
        backend_index, 0, begin, min(begin + PAGE_TOKENS, boundary), class_id,
        arenas[class_id].backend_domain, DETACHED_CLEAR, reason,
    )


def _certificate(detached: DetachedBinding) -> ReclamationCertificate:
    return ReclamationCertificate(
        detached.old,
        detached.class_id, detached.backend_domain, detached.logical_ordinal,
        detached.old_backend_index, detached.token_begin,
        detached.token_end_exclusive, 1, 1,
    )


def _case(reason: int, boundary: int = 35):
    pool, allocator, arenas = _install()
    full = tuple(
        _detach(0, ordinal, boundary, reason, arenas)
        for ordinal in range((boundary + PAGE_TOKENS - 1) // PAGE_TOKENS)
    )
    first_swa = max(0, boundary - 17) // PAGE_TOKENS
    swa = tuple(
        _detach(1, ordinal, boundary, reason, arenas)
        for ordinal in range(first_swa, (boundary + PAGE_TOKENS - 1) // PAGE_TOKENS)
    )
    for detached in full:
        begin, end = detached.token_begin, detached.token_end_exclusive
        start = (detached.old_backend_index + 1) * PAGE_TOKENS
        pool.req_to_token[1, begin:end] = torch.arange(
            start, start + end - begin, dtype=torch.int32
        )
    full_by_ordinal = {item.logical_ordinal: item for item in full}
    for detached in swa:
        owner = full_by_ordinal[detached.logical_ordinal]
        count = detached.token_end_exclusive - detached.token_begin
        full_start = (owner.old_backend_index + 1) * PAGE_TOKENS
        swa_start = (detached.old_backend_index - 256 + 1) * PAGE_TOKENS
        allocator.full_to_swa_index_mapping[full_start : full_start + count] = (
            torch.arange(swa_start, swa_start + count, dtype=torch.int64)
        )
    key = ("str", "session-hybrid")
    request_id = 91
    req = SimpleNamespace(
        req_pool_idx=1,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        cache_protected_len=0,
        _orbitkv_request_key=key,
        _orbitkv_engine_request_id=request_id,
        kv=SimpleNamespace(kv_allocated_len=boundary, swa_evicted_seqlen=0),
    )
    item = MirrorCleanupItem(
        cleanup._MirrorCleanupContext(req, 1, key, request_id),
        full + swa, True, boundary, (),
    )
    return pool, allocator, req, item, full, swa


def _install_shared_markers(req, pool, boundary: int) -> None:
    req.prefix_indices = pool.req_to_token[1, :boundary].to(torch.int64).clone()
    digest = b"h" * 32
    node = SimpleNamespace(boundary=boundary, digest=digest)
    req.cache_protected_len = boundary
    req.last_node = node
    req._orbitkv_prefix_node = node
    req._orbitkv_prefix_semantic = PrefixSemanticKey(b"n" * 32, digest, boundary)
    setattr(req, ENGINE_PREFIX_ID_MARKER, EnginePrefixId(7, 3))
    req._orbitkv_provisional_prefix_lock = False
    req._orbitkv_prefix_lock_held = True


@pytest.mark.parametrize("fault", ("missing_full", "extra_swa", "wrong_swa"))
def test_hybrid_release_rejects_hostile_class_coverage_without_mutation(fault):
    pool, allocator, _req, item, full, swa = _case(DETACHED_REQUEST_RELEASE)
    detached = list(item.detached)
    if fault == "missing_full":
        detached.remove(full[1])
    elif fault == "extra_swa":
        detached.append(replace(swa[-1], old=_page(1, 260, state._runtime().arenas_by_class[1]), old_backend_index=260))
    else:
        index = detached.index(swa[0])
        detached[index] = replace(swa[0], logical_ordinal=0, token_begin=0)
    before_row = pool.req_to_token.clone()
    before_lut = allocator.full_to_swa_index_mapping.clone()

    with pytest.raises(RuntimeError, match="coverage|transition"):
        cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (replace(item, detached=tuple(detached)),), ()
        )

    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)


def test_hybrid_release_rejects_shared_prefix_marker_drift():
    pool, allocator, req, item, _full, _swa = _case(
        DETACHED_PREFIX_TRANSFER, boundary=32
    )
    _install_shared_markers(req, pool, PAGE_TOKENS)
    req._orbitkv_prefix_lock_held = False
    before_row = pool.req_to_token.clone()
    before_lut = allocator.full_to_swa_index_mapping.clone()

    with pytest.raises(RuntimeError, match="shared Prefix metadata changed"):
        cleanup._MirrorCleanupCoordinator(pool, allocator).preflight((item,), ())
    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)


def test_hybrid_prefix_transfer_preserves_full_to_swa_lut():
    pool, allocator, req, item, _full, _swa = _case(
        DETACHED_PREFIX_TRANSFER, boundary=32
    )
    _install_shared_markers(req, pool, PAGE_TOKENS)
    before_lut = allocator.full_to_swa_index_mapping.clone()
    coordinator = cleanup._MirrorCleanupCoordinator(pool, allocator)

    with pytest.raises(RuntimeError, match="cannot carry retirement"):
        coordinator.preflight((item,), (_certificate(item.detached[0]),))
    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert torch.equal(allocator.full_to_swa_index_mapping, before_lut)
    assert not torch.count_nonzero(pool.req_to_token[1, : item.boundary])


@pytest.mark.parametrize("retire_class", (0, 1))
def test_hybrid_request_release_clears_only_retired_lut_entries(retire_class):
    pool, allocator, _req, item, full, swa = _case(DETACHED_REQUEST_RELEASE)
    shared_full = full[1]
    retired_full = full[2]
    before_shared = allocator.full_to_swa_index_mapping[
        (shared_full.old_backend_index + 1) * PAGE_TOKENS
        : (shared_full.old_backend_index + 1) * PAGE_TOKENS + PAGE_TOKENS
    ].clone()
    retiring = retired_full if retire_class == 0 else swa[-1]
    retirements = (_certificate(retiring),)
    coordinator = cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), retirements)
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    retired_start = (retired_full.old_backend_index + 1) * PAGE_TOKENS
    assert not torch.count_nonzero(
        allocator.full_to_swa_index_mapping[retired_start : retired_start + 3]
    )
    shared_start = (shared_full.old_backend_index + 1) * PAGE_TOKENS
    assert torch.equal(
        allocator.full_to_swa_index_mapping[shared_start : shared_start + PAGE_TOKENS],
        before_shared,
    )
    assert not torch.count_nonzero(pool.req_to_token[1, : item.boundary])


def test_swa_frontier_requires_sync_and_revalidates_old_value():
    pool, allocator, req, item, _full, swa = _case(DETACHED_REQUEST_RELEASE)
    retention = replace(swa[0], reason=DETACHED_RETENTION)
    online = replace(item, detached=(retention,), releasing=False)
    plan = cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
        (online,), (_certificate(retention),)
    )
    cleanup._MirrorCleanupCoordinator(pool, allocator).commit(plan)

    with pytest.raises(RuntimeError, match="before cleanup sync"):
        cleanup._MirrorCleanupCoordinator(pool, allocator).finalize(plan)
    assert req.kv.swa_evicted_seqlen == 0
    cleanup._MirrorCleanupCoordinator(pool, allocator).synchronize(plan)
    req.kv.swa_evicted_seqlen = 1
    with pytest.raises(RuntimeError, match="changed before finalization"):
        cleanup._MirrorCleanupCoordinator(pool, allocator).finalize(plan)
