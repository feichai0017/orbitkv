from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import orbitkv_sglang.bridge.mirror_cleanup as mirror_cleanup
import orbitkv_sglang.bridge.state as state
from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.runtime import (
    ArenaIdentity,
    DETACHED_CLEAR,
    DETACHED_RETENTION,
    DetachedBinding,
    MirrorCandidateTransition,
    MirrorCleanupItem,
    PageLease,
    ReclamationCertificate,
)


PAGE_TOKENS = 16
ZERO_PAGE = PageLease(0, 0, 0, 0, 0)


def _class() -> ClassConfig:
    return ClassConfig(
        class_id=0,
        pool_id=1,
        backend_domain=1,
        name="chunked",
        layers=(0,),
        retention="chunked",
        bytes_per_token_per_layer=128,
        window_tokens=None,
        period_blocks=None,
        chunk_tokens=32,
        blocks_per_epoch=2,
    )


def _page(backend_index: int) -> PageLease:
    return PageLease(1, 2, 1, backend_index + 1, 1)


def _certificate(
    backend_index: int, *, begin: int, end: int
) -> ReclamationCertificate:
    return ReclamationCertificate(
        page=_page(backend_index),
        class_id=0,
        backend_domain=1,
        logical_ordinal=begin // PAGE_TOKENS,
        backend_index=backend_index,
        token_begin=begin,
        token_end_exclusive=end,
        completion_domain=1,
        completion_value=1,
    )


def _candidate(backend_index: int, ordinal: int) -> MirrorCandidateTransition:
    begin = ordinal * PAGE_TOKENS
    return MirrorCandidateTransition(
        destination=_page(backend_index),
        source=ZERO_PAGE,
        logical_ordinal=ordinal,
        destination_backend_index=backend_index,
        source_backend_index=0,
        token_begin=begin,
        token_end_exclusive=begin + PAGE_TOKENS,
        copied_token_begin=0,
        copied_token_end_exclusive=0,
        class_id=0,
        backend_domain=1,
        retiring=True,
    )


def _install(monkeypatch: pytest.MonkeyPatch):
    class_config = _class()
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:session-chunked-cleanup-test",
        page_tokens=PAGE_TOKENS,
        classes=(class_config,),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:session-chunked-manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_chunked_token_kv"}
        ),
        manager_plan_format="retention_ir",
    )
    runtime = SimpleNamespace(
        arenas_by_class={
            0: ArenaIdentity(1, 2, 1, 0, 1, 64, PAGE_TOKENS, 0, 1)
        }
    )
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(4, 64, 128),
        runtime=runtime,
    )
    assert state._uses_runtime_session()
    pool = SimpleNamespace(
        req_to_token=torch.zeros((4, 128), dtype=torch.int32),
        max_context_len=128,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace()
    state._ALLOCATOR = allocator
    return pool, allocator


def _session_context(req: SimpleNamespace, row: int):
    key = ("str", req.rid)
    request_id = 100 + row
    req._orbitkv_request_key = key
    req._orbitkv_engine_request_id = request_id
    return mirror_cleanup._MirrorCleanupContext(req, row, key, request_id)


def _epoch_item(
    pool: SimpleNamespace,
    *,
    boundary: int = 32,
    row: int = 1,
    session: bool = True,
):
    epoch_begin = boundary - 32
    sources = tuple(
        _certificate(
            (row - 1) * 2 + ordinal % 2,
            begin=ordinal * PAGE_TOKENS,
            end=(ordinal + 1) * PAGE_TOKENS,
        )
        for ordinal in range(epoch_begin // PAGE_TOKENS, boundary // PAGE_TOKENS)
    )
    for source in sources:
        start = (source.backend_index + 1) * PAGE_TOKENS
        pool.req_to_token[
            row, source.token_begin : source.token_end_exclusive
        ] = torch.arange(start, start + PAGE_TOKENS, dtype=torch.int32)
    req = SimpleNamespace(
        rid=f"chunked-{boundary}-{row}",
        req_pool_idx=row,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        kv=SimpleNamespace(kv_allocated_len=boundary, swa_evicted_seqlen=0),
    )
    context = (
        _session_context(req, row)
        if session
        else mirror_cleanup._MirrorCleanupContext(req, row)
    )
    detached = tuple(
        DetachedBinding(
            old=source.page,
            replacement=ZERO_PAGE,
            logical_ordinal=source.logical_ordinal,
            old_backend_index=source.backend_index,
            replacement_backend_index=0,
            token_begin=source.token_begin,
            token_end_exclusive=source.token_end_exclusive,
            class_id=source.class_id,
            backend_domain=source.backend_domain,
            action=DETACHED_CLEAR,
            reason=DETACHED_RETENTION,
        )
        for source in sources
    )
    return req, MirrorCleanupItem(context, detached, False, boundary, ()), sources


def _candidate_item(pool: SimpleNamespace, *, row: int = 1):
    candidates = (
        _candidate(0, 0),
        _candidate(1, 1),
    )
    for candidate in candidates:
        start = (candidate.destination_backend_index + 1) * PAGE_TOKENS
        pool.req_to_token[
            row, candidate.token_begin : candidate.token_end_exclusive
        ] = torch.arange(start, start + PAGE_TOKENS, dtype=torch.int32)
    req = SimpleNamespace(
        rid="chunked-candidates",
        req_pool_idx=row,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        kv=SimpleNamespace(kv_allocated_len=32, swa_evicted_seqlen=0),
    )
    context = _session_context(req, row)
    certificates = tuple(
        _certificate(
            candidate.destination_backend_index,
            begin=candidate.token_begin,
            end=candidate.token_end_exclusive,
        )
        for candidate in candidates
    )
    return MirrorCleanupItem(context, (), False, 32, candidates), certificates


def test_session_chunked_detach_allows_certificate_subset_for_shared_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, allocator = _install(monkeypatch)
    _req, item, sources = _epoch_item(pool)
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), sources[:1])
    coordinator.commit(plan)

    assert not torch.count_nonzero(pool.req_to_token[1, :32])


def test_session_chunked_detach_without_certificate_keeps_fork_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, allocator = _install(monkeypatch)
    _first_req, first, _sources = _epoch_item(pool, row=1)
    _second_req, second, _second_sources = _epoch_item(pool, row=2)
    second = replace(second, detached=first.detached)
    pool.req_to_token[2, :32] = pool.req_to_token[1, :32]

    plan = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
        (first, second), ()
    )

    assert len(plan.zero_views) == 4


@pytest.mark.parametrize("fault", ("duplicate", "foreign"))
def test_session_chunked_certificate_must_be_unique_collective_source(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    pool, allocator = _install(monkeypatch)
    _req, item, sources = _epoch_item(pool)
    certificates = (sources[0], sources[0]) if fault == "duplicate" else (
        replace(sources[0], backend_domain=9),
    )

    with pytest.raises(RuntimeError, match="Chunked retirement coverage changed"):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (item,), certificates
        )


def test_session_chunked_retiring_candidates_are_certificate_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, allocator = _install(monkeypatch)
    item, certificates = _candidate_item(pool)
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), certificates)
    coordinator.commit(plan)

    assert not torch.count_nonzero(pool.req_to_token[1, :32])
    with pytest.raises(RuntimeError, match="Chunked retirement coverage changed"):
        coordinator.preflight(
            (item,), (replace(certificates[0], backend_domain=9), certificates[1])
        )


def test_session_chunked_rejects_mixed_context_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, allocator = _install(monkeypatch)
    _first_req, first, _first_sources = _epoch_item(pool, row=1, session=True)
    _second_req, second, _second_sources = _epoch_item(
        pool, row=2, session=False
    )

    with pytest.raises(RuntimeError, match="mixed context authorities"):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (first, second), ()
        )


def test_session_chunked_preserves_exact_epoch_span_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, allocator = _install(monkeypatch)
    _req, item, _sources = _epoch_item(pool)
    truncated = replace(item, detached=item.detached[:-1])

    with pytest.raises(RuntimeError, match="exact old epoch"):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (truncated,), ()
        )
