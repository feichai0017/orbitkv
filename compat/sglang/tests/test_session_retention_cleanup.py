from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

import orbitkv_sglang.bridge.mirror_cleanup as mirror_cleanup
import orbitkv_sglang.bridge.state as state
from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.ffi.session_types import (
    EngineAppendIntent,
    EngineBindEvidence,
    EngineCompletionEvidence,
    EngineCopyEvidence,
    EnginePublicationEvidence,
    EngineRequestId,
    EngineStepExecutionEvidence,
    ExecutionEvidence,
    retirement_evidence,
)
from orbitkv_sglang.runtime import (
    ArenaIdentity,
    ArenaRegistration,
    CacheSharingPolicy,
    DETACHED_CLEAR,
    DETACHED_REQUEST_RELEASE,
    DETACHED_RETENTION,
    DetachedBinding,
    ManagerCreateSettings,
    MirrorCleanupItem,
    PageLease,
    ReclamationCertificate,
    SessionCreateSettings,
    TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
)
from ffi_test_support import ffi_library


__all__ = ["ffi_library"]

PAGE_TOKENS = 16


def _sliding_class() -> ClassConfig:
    return ClassConfig(
        class_id=0,
        pool_id=1,
        backend_domain=1,
        name="sliding",
        layers=(0,),
        retention="sliding",
        bytes_per_token_per_layer=128,
        window_tokens=18,
        period_blocks=3,
        storage="token_kv",
    )


def _session_config(library: Path) -> RuntimeConfig:
    class_config = _sliding_class()
    plan = {
        "page_tokens": PAGE_TOKENS,
        "classes": [
            {
                "name": class_config.name,
                "layers": list(class_config.layers),
                "retention": class_config.retention,
                "bytes_per_token_per_layer": (
                    class_config.bytes_per_token_per_layer
                ),
                "window_tokens": class_config.window_tokens,
            }
        ],
    }
    return RuntimeConfig(
        library_path=library,
        plan_json=json.dumps(plan, separators=(",", ":")).encode(),
        plan_fingerprint="sha256:session-retention-cleanup-test",
        page_tokens=PAGE_TOKENS,
        classes=(class_config,),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:retention-manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_sliding_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )


def _execution_evidence(plan: Any, arenas: tuple[Any, ...]) -> ExecutionEvidence:
    arenas_by_class = {arena.class_id: arena for arena in arenas}
    steps = []
    for step in plan.steps:
        binds = []
        for lowering in step.class_lowerings:
            arena = arenas_by_class[lowering.class_id]

            def backend_index(page_id: int) -> int:
                return arena.backend_base_index + page_id - arena.first_page_id

            tail_end = lowering.tail_offset + lowering.tail_count
            for action in step.tail_actions[lowering.tail_offset:tail_end]:
                if action.kind in (TAIL_COPY_ON_WRITE, TAIL_FRESH):
                    binds.append(
                        EngineBindEvidence(
                            action.destination,
                            arena.backend_domain,
                            True,
                            True,
                            backend_index(action.destination.page_id),
                        )
                    )
            write_end = lowering.write_offset + lowering.write_count
            for intent in step.write_intents[lowering.write_offset:write_end]:
                binds.append(
                    EngineBindEvidence(
                        PageLease(
                            arena.engine_epoch,
                            arena.pool_epoch,
                            intent.page_generation,
                            intent.page_id,
                            arena.pool_id,
                        ),
                        arena.backend_domain,
                        True,
                        True,
                        backend_index(intent.page_id),
                    )
                )
        copies = tuple(
            EngineCopyEvidence(
                intent.class_id,
                intent.backend_domain,
                intent.token_count,
                intent.source_token_offset,
                intent.destination_token_offset,
                True,
                True,
                True,
                intent.source,
                intent.destination,
                intent.source_backend_index,
                intent.destination_backend_index,
            )
            for intent in step.copy_intents
        )
        steps.append(
            EngineStepExecutionEvidence(step.request_id, tuple(binds), copies)
        )
    return ExecutionEvidence(plan.batch_id, tuple(steps))


def _cleanup_state(
    config: RuntimeConfig,
    arena: ArenaIdentity,
    request_id: EngineRequestId,
    *,
    boundary: int = 35,
) -> tuple[Any, Any, Any]:
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(1, 64, 64),
        runtime=SimpleNamespace(arenas_by_class={0: arena}),
    )
    pool = SimpleNamespace(
        req_to_token=torch.zeros((2, 64), dtype=torch.int32),
        max_context_len=64,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace()
    state._ALLOCATOR = allocator
    key = ("str", "retention-request")
    req = SimpleNamespace(
        rid=key[1],
        req_pool_idx=1,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        kv=SimpleNamespace(kv_allocated_len=boundary, swa_evicted_seqlen=0),
        _orbitkv_request_key=key,
        _orbitkv_engine_request_id=request_id,
    )
    context = mirror_cleanup._MirrorCleanupContext(req, 1, key, request_id)
    return pool, allocator, (req, context)


def test_session_pure_sliding_periodic_retirement_clears_without_lut(
    ffi_library: Path,
) -> None:
    config = _session_config(ffi_library)
    request_id = EngineRequestId(19)
    settings = SessionCreateSettings(
        ManagerCreateSettings(1, 2, 1, 3, 64),
        CacheSharingPolicy.REQUEST_PRIVATE,
    )
    registration = ArenaRegistration(0, 1, 1, 3, 0)

    with CtypesRuntimeSession.create(
        config, settings, (registration,)
    ) as session:
        session.acquire_requests((request_id,))
        first = session.prepare_append((EngineAppendIntent(request_id, 18),))
        session.submit_execution(_execution_evidence(first, session.arenas))
        first_publication = session.complete_execution(
            first.batch_id, EngineCompletionEvidence(7, 1, True)
        )
        session.confirm_publication(
            EnginePublicationEvidence(
                first_publication.publication_id,
                True,
                retirement_evidence(first_publication.retirements),
            )
        )

        second = session.prepare_append((EngineAppendIntent(request_id, 35),))
        session.submit_execution(_execution_evidence(second, session.arenas))
        publication = session.complete_execution(
            second.batch_id, EngineCompletionEvidence(7, 2, True)
        )

        assert len(publication.steps) == len(publication.retirements) == 1
        assert len(publication.steps[0].detached) == 1
        detached = publication.steps[0].detached[0]
        retirement = publication.retirements[0]
        assert detached.old == retirement.page
        assert detached.replacement == PageLease(0, 0, 0, 0, 0)
        assert detached.action == DETACHED_CLEAR
        assert detached.reason == DETACHED_RETENTION
        assert (
            detached.logical_ordinal,
            detached.old_backend_index,
            detached.token_begin,
            detached.token_end_exclusive,
        ) == (0, 0, 0, PAGE_TOKENS)
        assert (
            retirement.logical_ordinal,
            retirement.backend_index,
            retirement.token_begin,
            retirement.token_end_exclusive,
        ) == (0, 0, 0, PAGE_TOKENS)
        written_page_ids = {
            intent.page_id
            for step in second.steps
            for intent in step.write_intents
        }
        assert retirement.page.page_id not in written_page_ids

        pool, allocator, (req, context) = _cleanup_state(
            config, session.arenas[0], request_id
        )
        coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
        expected = coordinator._locations(
            detached.class_id,
            detached.old_backend_index,
            detached.token_begin,
            detached.token_end_exclusive,
        )
        pool.req_to_token[
            1, detached.token_begin : detached.token_end_exclusive
        ] = expected.to(torch.int32)
        item = MirrorCleanupItem(
            context, publication.steps[0].detached, False, 35, ()
        )
        assert item.candidates == ()

        cleanup = coordinator.preflight((item,), publication.retirements)
        coordinator.commit(cleanup)
        coordinator.synchronize(cleanup)
        coordinator.finalize(cleanup)
        assert not hasattr(allocator, "full_to_swa_index_mapping")
        assert not torch.count_nonzero(
            pool.req_to_token[
                1, detached.token_begin : detached.token_end_exclusive
            ]
        )
        assert req.kv.swa_evicted_seqlen == PAGE_TOKENS
        session.confirm_publication(
            EnginePublicationEvidence(
                publication.publication_id,
                True,
                retirement_evidence(publication.retirements),
            )
        )


def test_session_pure_sliding_release_requires_exact_live_window_coverage() -> None:
    config = _session_config(Path("liborbitkv_ffi.so"))
    arena = ArenaIdentity(1, 2, 1, 0, 1, 512, PAGE_TOKENS, 0, 1)
    request_id = EngineRequestId(19)
    pool, allocator, (req, context) = _cleanup_state(
        config, arena, request_id
    )
    detached = []
    retirements = []
    zero = PageLease(0, 0, 0, 0, 0)
    for ordinal, begin, end, backend_index in (
        (1, 16, 32, 0),
        (2, 32, 35, 1),
    ):
        page = PageLease(1, 2, 1, backend_index + 1, 1)
        start = (backend_index + 1) * PAGE_TOKENS + begin % PAGE_TOKENS
        pool.req_to_token[1, begin:end] = torch.arange(
            start, start + end - begin, dtype=torch.int32
        )
        detached.append(
            DetachedBinding(
                page,
                zero,
                ordinal,
                backend_index,
                0,
                begin,
                end,
                0,
                1,
                DETACHED_CLEAR,
                DETACHED_REQUEST_RELEASE,
            )
        )
        retirements.append(
            ReclamationCertificate(
                page,
                0,
                1,
                ordinal,
                backend_index,
                begin,
                end,
                7,
                1,
            )
        )
    item = MirrorCleanupItem(context, tuple(detached), True, 35, ())
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)

    cleanup = coordinator.preflight((item,), tuple(retirements))
    coordinator.commit(cleanup)
    coordinator.synchronize(cleanup)
    coordinator.finalize(cleanup)
    assert not torch.count_nonzero(pool.req_to_token[1, :35])

    truncated = replace(item, detached=item.detached[:-1])
    with pytest.raises(RuntimeError, match="Sliding release coverage changed"):
        coordinator.preflight((truncated,), tuple(retirements[:-1]))
