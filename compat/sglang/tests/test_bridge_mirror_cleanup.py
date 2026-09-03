from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.mirror_cleanup as mirror_cleanup  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
from orbitkv_sglang.bridge.private_prefix import (  # noqa: E402
    ENGINE_PREFIX_ID_MARKER,
    PrivatePrefixProvenance,
)
from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402
from orbitkv_sglang.ffi.session_types import EnginePrefixId  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    DETACHED_CLEAR,
    DETACHED_COPY_ON_WRITE,
    DETACHED_PREFIX_TRANSFER,
    DETACHED_REPLACE,
    DETACHED_REQUEST_RELEASE,
    DETACHED_RETENTION,
    DetachedBinding,
    MirrorCleanupItem,
    MirrorCandidateTransition,
    PageLease,
    PrefixSemanticKey,
    ReclamationCertificate,
)


PAGE_TOKENS = 16
_COMPONENT_CACHE_POLICIES: dict[int, str] = {}


def _install_component_cache_policy(
    config: RuntimeConfig, cache_policy: str
) -> None:
    if cache_policy not in ("request_private", "shared_prefix"):
        raise ValueError("invalid component cache policy")
    _COMPONENT_CACHE_POLICIES[id(config)] = cache_policy


@pytest.fixture(autouse=True)
def _component_cache_policy(monkeypatch: pytest.MonkeyPatch):
    production_policy = state._cache_sharing_policy
    _COMPONENT_CACHE_POLICIES.clear()

    def selected_policy(profile=None):
        if profile is not None:
            return production_policy(profile)
        cache_policy = _COMPONENT_CACHE_POLICIES.get(id(state._config()))
        if cache_policy is None:
            return production_policy()
        return production_policy(SimpleNamespace(cache_policy=cache_policy))

    monkeypatch.setattr(state, "_cache_sharing_policy", selected_policy)
    yield
    _COMPONENT_CACHE_POLICIES.clear()


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
        chunk_tokens=32 if retention == "chunked" else None,
        blocks_per_epoch=2 if retention == "chunked" else None,
    )


def _install_hybrid():
    config = RuntimeConfig(
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
    _install_component_cache_policy(config, "shared_prefix")
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


def _install_full():
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:mirror-full-test",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "full"),),
    )
    runtime = SimpleNamespace(
        arenas_by_class={
            0: ArenaIdentity(
                engine_epoch=1,
                pool_epoch=2,
                pool_id=1,
                class_id=0,
                backend_domain=1,
                page_count=512,
                page_tokens=PAGE_TOKENS,
                backend_base_index=0,
                first_page_id=1,
            )
        }
    )
    _install_component_cache_policy(config, "shared_prefix")
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
    allocator = SimpleNamespace()
    state._ALLOCATOR = allocator
    return pool, allocator


def _install_session_full():
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:mirror-session-full-test",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "full"),),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:mirror-session-manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )
    runtime = SimpleNamespace(
        arenas_by_class={
            0: ArenaIdentity(1, 2, 1, 0, 1, 512, PAGE_TOKENS, 0, 1)
        }
    )
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
    allocator = SimpleNamespace()
    state._ALLOCATOR = allocator
    return pool, allocator


def _session_full_empty_publication_case(*, boundary=PAGE_TOKENS):
    pool, allocator = _install_session_full()
    row = 1
    request_key = ("str", "session-full-publication")
    request_id = 41
    req = SimpleNamespace(
        rid=request_key[1],
        req_pool_idx=row,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        _orbitkv_request_key=request_key,
        _orbitkv_engine_request_id=request_id,
    )
    item = MirrorCleanupItem(
        mirror_cleanup._MirrorCleanupContext(
            req, row, request_key, request_id
        ),
        (),
        False,
        boundary,
        (),
    )
    return pool, allocator, item


def _install_chunked():
    config = RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:mirror-chunked-test",
        page_tokens=PAGE_TOKENS,
        classes=(_class(0, "chunked"),),
        manager_plan_format="retention_ir",
    )
    runtime = SimpleNamespace(
        arenas_by_class={
            0: ArenaIdentity(1, 2, 1, 0, 1, 64, PAGE_TOKENS, 0, 1)
        }
    )
    _install_component_cache_policy(config, "request_private")
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(4, 64, 128),
        runtime=runtime,
    )
    pool = SimpleNamespace(
        req_to_token=torch.zeros((4, 128), dtype=torch.int32),
        max_context_len=128,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace()
    state._ALLOCATOR = allocator
    return pool, allocator


def _chunked_item(pool, *, boundary, row=1):
    epoch_begin = boundary - 32
    certificates = tuple(
        _certificate(
            0,
            backend_index=(row - 1) * 2 + ordinal % 2,
            begin=ordinal * PAGE_TOKENS,
            end=(ordinal + 1) * PAGE_TOKENS,
        )
        for ordinal in range(epoch_begin // PAGE_TOKENS, boundary // PAGE_TOKENS)
    )
    for certificate in certificates:
        start = (certificate.backend_index + 1) * PAGE_TOKENS
        pool.req_to_token[
            row, certificate.token_begin : certificate.token_end_exclusive
        ] = torch.arange(start, start + PAGE_TOKENS, dtype=torch.int32)
    req = SimpleNamespace(
        rid=f"chunked-{boundary}",
        req_pool_idx=row,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        kv=SimpleNamespace(kv_allocated_len=boundary, swa_evicted_seqlen=0),
    )
    detached = tuple(
        DetachedBinding(
            old=certificate.page,
            replacement=PageLease(0, 0, 0, 0, 0),
            logical_ordinal=certificate.logical_ordinal,
            old_backend_index=certificate.backend_index,
            replacement_backend_index=0,
            token_begin=certificate.token_begin,
            token_end_exclusive=certificate.token_end_exclusive,
            class_id=certificate.class_id,
            backend_domain=certificate.backend_domain,
            action=DETACHED_CLEAR,
            reason=DETACHED_RETENTION,
        )
        for certificate in certificates
    )
    item = MirrorCleanupItem(
        mirror_cleanup._MirrorCleanupContext(req, row),
        detached,
        False,
        boundary,
        (),
    )
    return req, item, certificates


def test_chunked_epoch_end_clears_exact_absolute_row_without_swa_frontier(
    monkeypatch,
):
    pool, allocator = _install_chunked()
    req, item, certificates = _chunked_item(pool, boundary=32)
    observed = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: observed.append(pool.req_to_token[1, :32].clone()),
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), certificates)
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert len(observed) == 1
    assert not torch.count_nonzero(observed[0])
    assert not torch.count_nonzero(pool.req_to_token[1, :32])
    assert req.kv.swa_evicted_seqlen == 0


def test_chunked_epoch_end_geometry_fault_is_zero_mutation(monkeypatch):
    pool, allocator = _install_chunked()
    _req, item, certificates = _chunked_item(pool, boundary=32)
    pool.req_to_token[1, 17] += 1
    before = pool.req_to_token.clone()
    synchronized = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronized.append(True),
    )

    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (item,), certificates
        )

    assert torch.equal(pool.req_to_token, before)
    assert synchronized == []


def test_chunked_epoch_end_uses_each_requests_absolute_epoch():
    pool, allocator = _install_chunked()
    req_a, item_a, certificates_a = _chunked_item(pool, boundary=32, row=1)
    req_b, item_b, certificates_b = _chunked_item(pool, boundary=64, row=2)
    pool.req_to_token[2, :32] = 901
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(
        (item_a, item_b), certificates_a + certificates_b
    )
    coordinator.commit(plan)

    assert not torch.count_nonzero(pool.req_to_token[1, :32])
    assert torch.all(pool.req_to_token[2, :32] == 901)
    assert not torch.count_nonzero(pool.req_to_token[2, 32:64])
    assert req_a.kv.swa_evicted_seqlen == req_b.kv.swa_evicted_seqlen == 0


def test_chunked_new_epoch_candidate_validates_absolute_column_without_clear():
    pool, allocator = _install_chunked()
    row = 1
    destination = torch.arange(16, 17, dtype=torch.int64)
    pool.req_to_token[row, 32] = destination.int()[0]
    req = SimpleNamespace(
        rid="chunked-next-epoch",
        req_pool_idx=row,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        kv=SimpleNamespace(kv_allocated_len=33, swa_evicted_seqlen=0),
    )
    item = MirrorCleanupItem(
        mirror_cleanup._MirrorCleanupContext(req, row),
        (),
        False,
        33,
        (_candidate(0, 0, 2, 33),),
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)

    assert int(pool.req_to_token[row, 32]) == 16
    assert req.kv.swa_evicted_seqlen == 0


def _certificate(
    class_id: int,
    backend_index: int,
    *,
    begin: int = 0,
    end: int = PAGE_TOKENS,
) -> ReclamationCertificate:
    return ReclamationCertificate(
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


def test_compact_hybrid_release_validates_whole_row_and_lut_before_clear():
    pool, allocator = _install_hybrid()
    full_locations = tuple(range(80, 104))
    swa_locations = tuple(range(208, 232))
    row = 1
    pool.req_to_token[row, :24] = torch.tensor(full_locations, dtype=torch.int32)
    allocator.full_to_swa_index_mapping[torch.tensor(full_locations)] = torch.tensor(
        swa_locations, dtype=torch.int64
    )
    req = SimpleNamespace(
        rid="compact-hybrid",
        req_pool_idx=row,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        _orbitkv_retained_locations=full_locations,
        _orbitkv_retained_swa_locations=swa_locations,
    )
    detached = tuple(
        DetachedBinding(
            old=_page(class_id, ordinal),
            replacement=PageLease(0, 0, 0, 0, 0),
            logical_ordinal=ordinal,
            old_backend_index=ordinal,
            replacement_backend_index=0,
            token_begin=ordinal * 16,
            token_end_exclusive=(ordinal + 1) * 16,
            class_id=class_id,
            backend_domain=class_id + 1,
            action=DETACHED_CLEAR,
            reason=3,
        )
        for class_id in (0, 1)
        for ordinal in (0, 1)
    )
    item = MirrorCleanupItem(
        mirror_cleanup._MirrorCleanupContext(req, row),
        detached,
        True,
        32,
        (),
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)
    assert not torch.count_nonzero(pool.req_to_token[row, :24])
    assert not torch.count_nonzero(
        allocator.full_to_swa_index_mapping[torch.tensor(full_locations)]
    )

    pool.req_to_token[row, :24] = torch.tensor(full_locations, dtype=torch.int32)
    allocator.full_to_swa_index_mapping[torch.tensor(full_locations)] = torch.tensor(
        swa_locations, dtype=torch.int64
    )
    allocator.full_to_swa_index_mapping[full_locations[7]] += 1
    before_row = pool.req_to_token.clone()
    before_mapping = allocator.full_to_swa_index_mapping.clone()
    with pytest.raises(RuntimeError, match="SGLang mirror"):
        coordinator.preflight((item,), ())
    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(allocator.full_to_swa_index_mapping, before_mapping)


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
    sync_calls = []
    monkeypatch.setattr(torch, "isin", lambda *_args, **_kwargs: scans.append(True))
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    plan = coordinator.preflight((item,), (sliding,))
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert scans == []
    assert plan.requires_device_sync
    assert sync_calls == [True]
    assert state._activity_counters()["prefix_global_alias_scans"] == 0
    assert state._activity_counters()["mirror_syncs"] == 1
    assert not torch.count_nonzero(
        allocator.full_to_swa_index_mapping[full_locations]
    )
    assert torch.equal(pool.req_to_token[1, :PAGE_TOKENS], full_locations.int())
    assert req.kv.swa_evicted_seqlen == PAGE_TOKENS


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


def _full_fresh_candidate_case():
    pool, allocator = _install_full()
    row = 1
    locations = torch.arange(48, 64, dtype=torch.int64)
    pool.req_to_token[row, :PAGE_TOKENS] = locations.int()
    req = SimpleNamespace(
        rid="full-fresh-candidate",
        req_pool_idx=row,
        prefix_indices=locations.clone(),
    )
    item = MirrorCleanupItem(
        context=mirror_cleanup._MirrorCleanupContext(req, row),
        detached=(),
        releasing=False,
        boundary=PAGE_TOKENS,
        candidates=(_candidate(0, 2, 0, PAGE_TOKENS),),
    )
    return pool, allocator, req, item


def _full_b4_fresh_candidate_case():
    pool, allocator = _install_full()
    requests = []
    items = []
    for batch_index, row in enumerate(range(1, 5)):
        boundary = 25 + batch_index
        prefix_count = PAGE_TOKENS + 1 + batch_index
        candidates = []
        for ordinal, (begin, end) in enumerate(
            ((0, PAGE_TOKENS), (PAGE_TOKENS, boundary))
        ):
            backend_index = 8 + batch_index * 2 + ordinal
            start = (backend_index + 1) * PAGE_TOKENS
            pool.req_to_token[row, begin:end] = torch.arange(
                start, start + end - begin, dtype=torch.int32
            )
            candidates.append(_candidate(0, backend_index, ordinal, end))
        req = SimpleNamespace(
            req_pool_idx=row,
            prefix_indices=pool.req_to_token[row, :prefix_count]
            .to(torch.int64)
            .clone(),
        )
        requests.append(req)
        items.append(
            MirrorCleanupItem(
                context=mirror_cleanup._MirrorCleanupContext(req, row),
                detached=(),
                releasing=False,
                boundary=boundary,
                candidates=tuple(candidates),
            )
        )
    return pool, allocator, tuple(requests), tuple(items)


def _full_b4_release_case():
    pool, allocator = _install_full()
    requests = []
    items = []
    certificates = []
    for batch_index, row in enumerate(range(1, 5)):
        boundary = 25 + batch_index
        prefix_count = PAGE_TOKENS + 1 + batch_index
        detached = []
        for ordinal, (begin, end) in enumerate(
            ((0, PAGE_TOKENS), (PAGE_TOKENS, boundary))
        ):
            backend_index = 40 + batch_index * 2 + ordinal
            certificate = _certificate(0, backend_index, begin=begin, end=end)
            certificates.append(certificate)
            start = (backend_index + 1) * PAGE_TOKENS
            pool.req_to_token[row, begin:end] = torch.arange(
                start, start + end - begin, dtype=torch.int32
            )
            detached.append(
                DetachedBinding(
                    old=certificate.page,
                    replacement=PageLease(0, 0, 0, 0, 0),
                    logical_ordinal=ordinal,
                    old_backend_index=backend_index,
                    replacement_backend_index=0,
                    token_begin=begin,
                    token_end_exclusive=end,
                    class_id=0,
                    backend_domain=1,
                    action=DETACHED_CLEAR,
                    reason=DETACHED_REQUEST_RELEASE,
                )
            )
        # Release cleanup owns only the declared boundary.  This sentinel makes
        # an accidentally dense row clear externally visible.
        pool.req_to_token[row, boundary] = 900 + row
        req = SimpleNamespace(
            req_pool_idx=row,
            prefix_indices=pool.req_to_token[row, :prefix_count]
            .to(torch.int64)
            .clone(),
        )
        requests.append(req)
        items.append(
            MirrorCleanupItem(
                context=mirror_cleanup._MirrorCleanupContext(req, row),
                detached=tuple(detached),
                releasing=True,
                boundary=boundary,
                candidates=(),
            )
        )
    return (
        pool,
        allocator,
        tuple(requests),
        tuple(items),
        tuple(certificates),
    )


def _session_full_release_case(
    *, reason: int, boundary: int = PAGE_TOKENS * 2
):
    pool, allocator = _install_session_full()
    row = 1
    request_key = ("str", "session-full-release")
    request_id = 41
    detached = []
    for ordinal, begin in enumerate(range(0, boundary, PAGE_TOKENS)):
        end = min(begin + PAGE_TOKENS, boundary)
        backend_index = 40 + ordinal
        certificate = _certificate(0, backend_index, begin=begin, end=end)
        start = (backend_index + 1) * PAGE_TOKENS
        pool.req_to_token[row, begin:end] = torch.arange(
            start, start + end - begin, dtype=torch.int32
        )
        detached.append(
            DetachedBinding(
                old=certificate.page,
                replacement=PageLease(0, 0, 0, 0, 0),
                logical_ordinal=ordinal,
                old_backend_index=backend_index,
                replacement_backend_index=0,
                token_begin=begin,
                token_end_exclusive=end,
                class_id=0,
                backend_domain=1,
                action=DETACHED_CLEAR,
                reason=reason,
            )
        )
    pool.req_to_token[row, boundary] = 999
    prefix = pool.req_to_token[row, :boundary].to(torch.int64).clone()
    req = SimpleNamespace(
        rid=request_key[1],
        req_pool_idx=row,
        prefix_indices=prefix,
        cache_protected_len=0,
        _orbitkv_request_key=request_key,
        _orbitkv_engine_request_id=request_id,
    )
    item = MirrorCleanupItem(
        mirror_cleanup._MirrorCleanupContext(
            req, row, request_key, request_id
        ),
        tuple(detached),
        True,
        boundary,
        (),
    )
    return pool, allocator, req, item


def _install_shared_prefix_markers(req, boundary: int) -> None:
    req.prefix_indices = req.prefix_indices[:boundary].clone()
    digest = b"p" * 32
    prefix_id = EnginePrefixId(1, 9)
    node = SimpleNamespace(boundary=boundary, digest=digest)
    semantic = PrefixSemanticKey(b"mirror-test", digest, boundary)
    req.cache_protected_len = boundary
    req.last_node = node
    req._orbitkv_prefix_node = node
    req._orbitkv_prefix_semantic = semantic
    setattr(req, ENGINE_PREFIX_ID_MARKER, prefix_id)
    req._orbitkv_provisional_prefix_lock = False
    req._orbitkv_prefix_lock_held = True


def test_session_full_prefix_transfer_clears_exact_shared_mirrors_without_retirement(
    monkeypatch,
):
    pool, allocator, req, item = _session_full_release_case(
        reason=DETACHED_PREFIX_TRANSFER
    )
    _install_shared_prefix_markers(req, PAGE_TOKENS)
    original_prefix = req.prefix_indices
    synchronized = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronized.append(
            (
                pool.req_to_token[1, : item.boundary].clone(),
                original_prefix.clone(),
            )
        ),
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert len(synchronized) == 1
    assert not torch.count_nonzero(synchronized[0][0])
    assert not torch.count_nonzero(synchronized[0][1])
    assert not torch.count_nonzero(pool.req_to_token[1, : item.boundary])
    assert int(pool.req_to_token[1, item.boundary]) == 999
    assert req.prefix_indices is original_prefix
    assert not torch.count_nonzero(req.prefix_indices)
    assert getattr(req, "_orbitkv_prefix_node", None) is not None
    assert state._activity_counters()["mirror_validation_calls"] == 1


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("empty", "incomplete coverage"),
        ("missing", "incomplete coverage"),
        ("gap", "incomplete coverage"),
        ("duplicate", "duplicated"),
        ("backend", "page identity is invalid"),
        ("table", "disagrees with the SGLang mirror"),
        ("prefix", "disagrees with the SGLang mirror"),
        ("invalid", "invalid or mixed detach reasons"),
        ("mixed", "invalid or mixed detach reasons"),
    ),
)
def test_session_full_prefix_transfer_rejects_invalid_coverage_atomically(
    monkeypatch, fault, message
):
    pool, allocator, req, item = _session_full_release_case(
        reason=DETACHED_PREFIX_TRANSFER, boundary=PAGE_TOKENS * 3
    )
    _install_shared_prefix_markers(req, PAGE_TOKENS)
    detached = list(item.detached)
    if fault == "empty":
        detached.clear()
    elif fault == "missing":
        detached.pop()
    elif fault == "gap":
        detached.pop(1)
    elif fault == "duplicate":
        detached[1] = replace(
            detached[1],
            old=detached[0].old,
            old_backend_index=detached[0].old_backend_index,
        )
    elif fault == "backend":
        detached[1] = replace(
            detached[1], old_backend_index=detached[1].old_backend_index + 1
        )
    elif fault == "table":
        pool.req_to_token[1, PAGE_TOKENS + 1] += 1
    elif fault == "prefix":
        req.prefix_indices[1] += 1
    elif fault == "invalid":
        detached[0] = replace(detached[0], reason=DETACHED_RETENTION)
    else:
        detached[1] = replace(
            detached[1], reason=DETACHED_REQUEST_RELEASE
        )
    item = replace(item, detached=tuple(detached))
    before_row = pool.req_to_token.clone()
    before_prefix = req.prefix_indices.clone()
    synchronized = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronized.append(True),
    )

    with pytest.raises(RuntimeError, match=message):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (item,), ()
        )

    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(req.prefix_indices, before_prefix)
    assert synchronized == []


def test_session_full_prefix_transfer_rejects_retirement_certificate():
    pool, allocator, req, item = _session_full_release_case(
        reason=DETACHED_PREFIX_TRANSFER
    )
    _install_shared_prefix_markers(req, PAGE_TOKENS)
    detached = item.detached[0]
    certificate = _certificate(
        0,
        detached.old_backend_index,
        begin=detached.token_begin,
        end=detached.token_end_exclusive,
    )

    with pytest.raises(
        RuntimeError, match="cannot carry retirement certificates"
    ):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (item,), (certificate,)
        )


def test_session_full_prefix_transfer_rejects_changed_shared_marker_atomically(
    monkeypatch,
):
    pool, allocator, req, item = _session_full_release_case(
        reason=DETACHED_PREFIX_TRANSFER
    )
    _install_shared_prefix_markers(req, PAGE_TOKENS)
    req._orbitkv_prefix_lock_held = False
    before_row = pool.req_to_token.clone()
    before_prefix = req.prefix_indices.clone()
    synchronized = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronized.append(True),
    )

    with pytest.raises(RuntimeError, match="shared Prefix metadata changed"):
        mirror_cleanup._MirrorCleanupCoordinator(pool, allocator).preflight(
            (item,), ()
        )

    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(req.prefix_indices, before_prefix)
    assert synchronized == []


def test_session_full_ordinary_release_still_clears_shared_prefix():
    pool, allocator, req, item = _session_full_release_case(
        reason=DETACHED_REQUEST_RELEASE
    )
    _install_shared_prefix_markers(req, PAGE_TOKENS)
    original_prefix = req.prefix_indices
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert not torch.count_nonzero(pool.req_to_token[1, : item.boundary])
    assert req.prefix_indices is original_prefix
    assert not torch.count_nonzero(req.prefix_indices)
    assert getattr(req, "_orbitkv_prefix_node", None) is not None


def test_full_b4_fresh_candidates_preserve_rows_and_prefixes_without_sync(
    monkeypatch,
):
    pool, allocator, requests, items = _full_b4_fresh_candidate_case()
    before_table = pool.req_to_token.clone()
    before_prefixes = tuple(req.prefix_indices.clone() for req in requests)
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(items, ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert sync_calls == []
    assert torch.equal(pool.req_to_token, before_table)
    assert all(
        torch.equal(req.prefix_indices, before)
        for req, before in zip(requests, before_prefixes, strict=True)
    )
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 1
    assert counters["mirror_syncs"] == 0


def test_full_b4_request_release_clears_rows_and_prefixes_with_one_sync(
    monkeypatch,
):
    pool, allocator, requests, items, certificates = _full_b4_release_case()
    expected_table = pool.req_to_token.clone()
    for item in items:
        expected_table[item.context.request_row, : item.boundary] = 0
    sync_observations = []

    def observe_sync(_pool):
        sync_observations.append(
            (
                tuple(
                    int(
                        pool.req_to_token[
                            item.context.request_row, : item.boundary
                        ].count_nonzero()
                    )
                    for item in items
                ),
                tuple(
                    int(req.prefix_indices.count_nonzero()) for req in requests
                ),
            )
        )

    monkeypatch.setattr(mirror_cleanup, "_synchronize_mirror", observe_sync)
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(items, certificates)
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert sync_observations == [((0, 0, 0, 0), (0, 0, 0, 0))]
    assert torch.equal(pool.req_to_token, expected_table)
    assert all(not torch.count_nonzero(req.prefix_indices) for req in requests)
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 1
    assert counters["mirror_syncs"] == 1


def test_full_shared_release_allows_detach_without_last_reference_certificate(
    monkeypatch,
):
    pool, allocator, requests, items, _certificates = _full_b4_release_case()
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(items, ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert sync_calls == [True]
    assert all(
        not torch.count_nonzero(
            pool.req_to_token[item.context.request_row, : item.boundary]
        )
        for item in items
    )
    assert all(not torch.count_nonzero(req.prefix_indices) for req in requests)


def test_full_shared_page_release_accepts_one_last_reference_certificate(
    monkeypatch,
):
    pool, allocator = _install_full()
    certificate = _certificate(0, 40)
    start = (certificate.backend_index + 1) * PAGE_TOKENS
    locations = torch.arange(start, start + PAGE_TOKENS, dtype=torch.int64)
    requests = []
    items = []
    for row in (1, 2):
        pool.req_to_token[row, :PAGE_TOKENS] = locations.to(torch.int32)
        req = SimpleNamespace(
            req_pool_idx=row, prefix_indices=locations.clone()
        )
        requests.append(req)
        items.append(
            MirrorCleanupItem(
                context=mirror_cleanup._MirrorCleanupContext(req, row),
                detached=(
                    DetachedBinding(
                        old=certificate.page,
                        replacement=PageLease(0, 0, 0, 0, 0),
                        logical_ordinal=0,
                        old_backend_index=certificate.backend_index,
                        replacement_backend_index=0,
                        token_begin=0,
                        token_end_exclusive=PAGE_TOKENS,
                        class_id=0,
                        backend_domain=1,
                        action=DETACHED_CLEAR,
                        reason=DETACHED_REQUEST_RELEASE,
                    ),
                ),
                releasing=True,
                boundary=PAGE_TOKENS,
                candidates=(),
            )
        )
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(tuple(items), (certificate,))
    coordinator.commit(plan)
    coordinator.synchronize(plan)

    assert sync_calls == [True]
    assert not torch.count_nonzero(pool.req_to_token[1:3, :PAGE_TOKENS])
    assert all(not torch.count_nonzero(req.prefix_indices) for req in requests)


@pytest.mark.parametrize("corruption", ("table", "prefix"))
def test_full_b4_release_mirror_corruption_fails_atomically(
    monkeypatch, corruption
):
    pool, allocator, requests, items, certificates = _full_b4_release_case()
    if corruption == "table":
        pool.req_to_token[3, PAGE_TOKENS + 2] += 1
    else:
        requests[2].prefix_indices[PAGE_TOKENS + 1] += 1
    before_table = pool.req_to_token.clone()
    before_prefixes = tuple(req.prefix_indices.clone() for req in requests)
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        coordinator.preflight(items, certificates)

    assert sync_calls == []
    assert torch.equal(pool.req_to_token, before_table)
    assert all(
        torch.equal(req.prefix_indices, before)
        for req, before in zip(requests, before_prefixes, strict=True)
    )
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 0
    assert counters["mirror_syncs"] == 0


def test_full_b4_cow_falls_back_and_generic_cleanup_remains_valid(monkeypatch):
    pool, allocator = _install_full()
    requests = []
    items = []
    for batch_index, row in enumerate(range(1, 5)):
        source_backend_index = 80 + batch_index * 2
        destination_backend_index = source_backend_index + 1
        destination_start = (destination_backend_index + 1) * PAGE_TOKENS
        destination_locations = torch.arange(
            destination_start,
            destination_start + PAGE_TOKENS,
            dtype=torch.int64,
        )
        pool.req_to_token[row, :PAGE_TOKENS] = destination_locations.int()
        req = SimpleNamespace(
            req_pool_idx=row, prefix_indices=destination_locations.clone()
        )
        requests.append(req)
        items.append(
            MirrorCleanupItem(
                context=mirror_cleanup._MirrorCleanupContext(req, row),
                detached=(
                    DetachedBinding(
                        old=_page(0, source_backend_index),
                        replacement=_page(0, destination_backend_index),
                        logical_ordinal=0,
                        old_backend_index=source_backend_index,
                        replacement_backend_index=destination_backend_index,
                        token_begin=0,
                        token_end_exclusive=PAGE_TOKENS,
                        class_id=0,
                        backend_domain=1,
                        action=DETACHED_REPLACE,
                        reason=DETACHED_COPY_ON_WRITE,
                    ),
                ),
                releasing=False,
                boundary=PAGE_TOKENS,
                candidates=(
                    _candidate(
                        0,
                        destination_backend_index,
                        0,
                        PAGE_TOKENS,
                        source_backend_index=source_backend_index,
                        copied_end=PAGE_TOKENS,
                    ),
                ),
            )
        )
    requests = tuple(requests)
    items = tuple(items)
    before_table = pool.req_to_token.clone()
    before_prefixes = tuple(req.prefix_indices.clone() for req in requests)
    fast_path_results = []
    original = mirror_cleanup._MirrorCleanupCoordinator._preflight_full_batch

    def observe_fast_path(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        fast_path_results.append(result is not None)
        return result

    monkeypatch.setattr(
        mirror_cleanup._MirrorCleanupCoordinator,
        "_preflight_full_batch",
        observe_fast_path,
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(items, ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert fast_path_results == [False]
    assert torch.equal(pool.req_to_token, before_table)
    assert all(
        torch.equal(req.prefix_indices, before)
        for req, before in zip(requests, before_prefixes, strict=True)
    )
    assert state._activity_counters()["mirror_validation_calls"] == 1


def test_full_fresh_candidate_validates_without_device_sync(monkeypatch):
    pool, allocator, req, item = _full_fresh_candidate_case()
    before_row = pool.req_to_token.clone()
    before_prefix = req.prefix_indices.clone()
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert not plan.requires_device_sync
    assert sync_calls == []
    assert torch.equal(pool.req_to_token, before_row)
    assert torch.equal(req.prefix_indices, before_prefix)
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 1
    assert counters["mirror_syncs"] == 0


def test_session_full_empty_publication_runs_empty_four_phase_plan(
    monkeypatch,
):
    pool, allocator, item = _session_full_empty_publication_case()
    tensor_constructors = []
    sync_calls = []
    original_ones = torch.ones
    original_tensor = torch.tensor

    def track_ones(*args, **kwargs):
        tensor_constructors.append("ones")
        return original_ones(*args, **kwargs)

    def track_tensor(*args, **kwargs):
        tensor_constructors.append("tensor")
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "ones", track_ones)
    monkeypatch.setattr(torch, "tensor", track_tensor)
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert plan.zero_views == ()
    assert plan.mapping is None
    assert plan.mapping_indices == ()
    assert plan.frontier_updates == ()
    assert plan.indexed_zeroes == ()
    assert plan.private_prefixes == ()
    assert not plan.requires_device_sync
    assert plan.synchronized is True
    assert tensor_constructors == []
    assert sync_calls == []
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 1
    assert counters["mirror_syncs"] == 0


def test_full_fresh_candidate_fault_is_still_rejected_without_sync(monkeypatch):
    pool, allocator, _req, item = _full_fresh_candidate_case()
    pool.req_to_token[1, PAGE_TOKENS - 1] += 1
    corrupted_row = pool.req_to_token.clone()
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        coordinator.preflight((item,), ())

    assert sync_calls == []
    assert torch.equal(pool.req_to_token, corrupted_row)
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 0
    assert counters["mirror_syncs"] == 0


def test_frontier_only_cleanup_validates_and_finalizes_without_device_sync(
    monkeypatch,
):
    pool, allocator = _install_hybrid()
    full_locations = torch.arange(48, 64, dtype=torch.int64)
    swa_locations = torch.arange(16, 32, dtype=torch.int64)
    pool.req_to_token[1, :PAGE_TOKENS] = full_locations.int()
    allocator.full_to_swa_index_mapping[full_locations] = swa_locations
    req = SimpleNamespace(
        req_pool_idx=1,
        prefix_indices=full_locations.clone(),
        kv=SimpleNamespace(kv_allocated_len=PAGE_TOKENS, swa_evicted_seqlen=0),
    )
    sliding = _certificate(1, 0)
    item = MirrorCleanupItem(
        context=mirror_cleanup._MirrorCleanupContext(req, 1),
        detached=(
            DetachedBinding(
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
            ),
        ),
        releasing=False,
        boundary=PAGE_TOKENS,
    )
    sync_calls = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight((item,), ())
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert not plan.requires_device_sync
    assert sync_calls == []
    assert torch.equal(pool.req_to_token[1, :PAGE_TOKENS], full_locations.int())
    assert torch.equal(
        allocator.full_to_swa_index_mapping[full_locations], swa_locations
    )
    assert req.kv.swa_evicted_seqlen == PAGE_TOKENS
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 1
    assert counters["mirror_syncs"] == 0


def test_mixed_cleanup_batch_synchronizes_when_any_item_mutates(monkeypatch):
    pool, allocator, fresh_req, fresh_item = _full_fresh_candidate_case()
    release_locations = torch.arange(64, 80, dtype=torch.int64)
    pool.req_to_token[2, :PAGE_TOKENS] = release_locations.int()
    release_req = SimpleNamespace(
        req_pool_idx=2,
        prefix_indices=release_locations.clone(),
    )
    certificate = _certificate(0, 3)
    release_item = MirrorCleanupItem(
        context=mirror_cleanup._MirrorCleanupContext(release_req, 2),
        detached=(
            DetachedBinding(
                old=certificate.page,
                replacement=PageLease(0, 0, 0, 0, 0),
                logical_ordinal=0,
                old_backend_index=certificate.backend_index,
                replacement_backend_index=0,
                token_begin=0,
                token_end_exclusive=PAGE_TOKENS,
                class_id=0,
                backend_domain=1,
                action=DETACHED_CLEAR,
                reason=DETACHED_RETENTION,
            ),
        ),
        releasing=True,
        boundary=PAGE_TOKENS,
    )
    sync_observations = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: sync_observations.append(
            int(pool.req_to_token[2, :PAGE_TOKENS].count_nonzero())
        ),
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    plan = coordinator.preflight(
        (fresh_item, release_item),
        (certificate,),
    )
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)

    assert plan.requires_device_sync
    assert sync_observations == [0]
    assert torch.count_nonzero(pool.req_to_token[1, :PAGE_TOKENS])
    assert torch.equal(fresh_req.prefix_indices, torch.arange(48, 64))
    assert not torch.count_nonzero(release_req.prefix_indices)
    counters = state._activity_counters()
    assert counters["mirror_validation_calls"] == 1
    assert counters["mirror_syncs"] == 1


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
    config = RuntimeConfig(
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
    _install_component_cache_policy(config, "request_private")
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
        rid="pure-swa-cow",
        req_pool_idx=row,
        prefix_indices=torch.cat(
            (torch.zeros(16, dtype=torch.int64), destination[:2])
        ),
        kv=SimpleNamespace(kv_allocated_len=32, swa_evicted_seqlen=0),
    )
    req._orbitkv_request_key = ("str", req.rid)
    req._orbitkv_request_lease = SimpleNamespace(slot=1)
    req._orbitkv_private_prefix = PrivatePrefixProvenance(
        req.prefix_indices,
        req._orbitkv_request_key,
        req._orbitkv_request_lease,
        32,
    )
    req.cache_protected_len = 0
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
