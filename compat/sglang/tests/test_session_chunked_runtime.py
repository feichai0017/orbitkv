from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.execution_plan import (
    BindingResult,
    StepExecutionResult,
    confirm_execution,
    expected_bindings,
    lower_batch_plan,
)
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.ffi.session_types import (
    EngineAppendIntent,
    EnginePublicationEvidence,
    EngineStepAbortEvidence,
)
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    ManagerCreateSettings,
    ManagerError,
    PrefixSemanticKey,
    SessionCreateSettings,
)
from orbitkv_sglang.runtime.snapshot_shadow import (
    CLASS_LOWERING_EPOCH_START,
    CLASS_LOWERING_RESETTABLE,
)
from orbitkv_sglang.session_runtime import SessionRuntime
from ffi_test_support import ffi_library


__all__ = ["ffi_library"]


PAGE_TOKENS = 16
CHUNK_TOKENS = 32
POOL_ID = 31
BACKEND_DOMAIN = 17
BACKEND_BASE_INDEX = 4000


def _config(library: Path) -> RuntimeConfig:
    retention = {
        "schema": "orbitkv.retention-ir.v1",
        "page_tokens": PAGE_TOKENS,
        "states": [
            {
                "name": "chunked",
                "layers": [0],
                "bytes_per_token_per_layer": 128,
                "may_read": {
                    "op": "equal",
                    "lhs": {
                        "op": "floor_div",
                        "value": {"op": "query_position"},
                        "divisor": CHUNK_TOKENS,
                    },
                    "rhs": {
                        "op": "floor_div",
                        "value": {"op": "key_position"},
                        "divisor": CHUNK_TOKENS,
                    },
                },
            }
        ],
    }
    return RuntimeConfig(
        library_path=library,
        plan_json=json.dumps(
            retention, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        plan_fingerprint="sha256:session-chunked-runtime-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=POOL_ID,
                backend_domain=BACKEND_DOMAIN,
                name="chunked",
                layers=(0,),
                retention="chunked",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
                chunk_tokens=CHUNK_TOKENS,
                blocks_per_epoch=CHUNK_TOKENS // PAGE_TOKENS,
            ),
        ),
        manager_plan_format="retention_ir",
    )


class _ReadyEvent:
    @staticmethod
    def query() -> bool:
        return True

    @staticmethod
    def synchronize() -> None:
        return None


def _execution_evidence(
    runtime: SessionRuntime, config: RuntimeConfig, plan: Any
) -> Any:
    views = {
        step.request_id: runtime.view_for(
            next(
                key
                for key in ("source", "competitor")
                if runtime.binding_for(key).request_id == step.request_id
            )
        )
        for step in plan.steps
    }
    lowered = lower_batch_plan(plan, config, runtime.arenas, views)
    bindings = expected_bindings(plan, lowered, runtime.arenas)
    results = tuple(
        StepExecutionResult(
            step.request_id,
            tuple(
                BindingResult(
                    item.page, item.backend_domain, item.backend_index, True, True
                )
                for item in selected
            ),
            tuple(
                intent
                for class_spec in lowered_step.class_specs
                for intent in class_spec.copy_intents
            ),
            True,
        )
        for step, lowered_step, selected in zip(
            plan.steps, lowered.steps, bindings, strict=True
        )
    )
    return confirm_execution(plan, lowered, runtime.arenas, results)


def _submit(
    runtime: SessionRuntime, config: RuntimeConfig, plan: Any, domain: int
) -> None:
    ticket = runtime.submit(_execution_evidence(runtime, config, plan))
    runtime.register_event(ticket, _ReadyEvent(), domain)


def test_exact_chunked_native_session_epoch_lifecycle(ffi_library: Path) -> None:
    config = _config(ffi_library)
    native = CtypesRuntimeSession.create(
        config,
        SessionCreateSettings(
            ManagerCreateSettings(2, 2, 2, 2, CHUNK_TOKENS),
            CacheSharingPolicy.REQUEST_PRIVATE,
        ),
        (ArenaRegistration(0, POOL_ID, BACKEND_DOMAIN, 2, BACKEND_BASE_INDEX),),
    )
    cleanup_calls: list[tuple[Any, Any]] = []
    lifecycle: list[str] = []
    runtime: SessionRuntime

    def cleanup(updates: tuple[Any, ...], retirements: tuple[Any, ...]) -> bool:
        if any(update.releasing for update in updates):
            lifecycle.append("release-cleanup")
            assert len(updates) == 2
            source = next(update for update in updates if update.key == "source")
            competitor = next(
                update for update in updates if update.key == "competitor"
            )
            assert source.boundary == CHUNK_TOKENS + 1
            assert len(source.detached) == len(retirements) == 1
            assert competitor.boundary == 0
            assert competitor.detached == ()
            return True
        assert len(updates) == 1
        update = updates[0]
        assert update.key == "source"
        if not update.detached and not retirements:
            return True
        cleanup_calls.append((updates, retirements))
        lifecycle.append("epoch-cleanup")
        assert update.boundary == CHUNK_TOKENS
        assert not update.releasing
        assert [(item.logical_ordinal, item.token_begin, item.token_end_exclusive)
                for item in update.detached] == [(0, 0, 16), (1, 16, 32)]
        assert [(item.logical_ordinal, item.token_begin, item.token_end_exclusive)
                for item in retirements] == [(0, 0, 16), (1, 16, 32)]

        competitor_id = runtime.binding_for("competitor").request_id
        with pytest.raises(ManagerError, match="capacity"):
            native.prepare_append((EngineAppendIntent(competitor_id, 1),))
        return True

    runtime = SessionRuntime(native, mirror_cleanup=cleanup)
    original_confirm_publication = native.confirm_publication

    def confirm_publication_with_bad_ack(evidence: EnginePublicationEvidence) -> None:
        if evidence.reclamation_receipts:
            bad = replace(
                evidence,
                reclamation_receipts=(
                    replace(evidence.reclamation_receipts[0], acknowledged=False),
                    *evidence.reclamation_receipts[1:],
                ),
            )
            with pytest.raises(ManagerError, match="evidence|receipt"):
                original_confirm_publication(bad)
            assert native.stats().retiring_pages == 2
            lifecycle.append("bad-ack-rejected")
        original_confirm_publication(evidence)
        if evidence.reclamation_receipts:
            lifecycle.append("ack")

    native.confirm_publication = confirm_publication_with_bad_ack
    drained = False
    try:
        runtime.acquire_unbound(("source", "competitor"))
        runtime.bind_request_rows((("source", 1), ("competitor", 2)))

        first = runtime.prepare((("source", CHUNK_TOKENS - 1),))
        assert first.steps[0].class_lowerings[0].flags == (
            CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START
        )
        old_pages = tuple(
            (intent.page_id, intent.page_generation)
            for intent in first.steps[0].write_intents
        )
        assert len(old_pages) == 2
        _submit(runtime, config, first, 7)
        assert runtime.poll()[0].retirements == ()
        assert cleanup_calls == []

        epoch_end = runtime.prepare((("source", CHUNK_TOKENS),))
        lowering = epoch_end.steps[0].class_lowerings[0]
        assert lowering.flags == CLASS_LOWERING_RESETTABLE
        assert (lowering.previous_layout_boundary, lowering.target_layout_boundary) == (31, 32)
        assert epoch_end.steps[0].write_intents == ()
        _submit(runtime, config, epoch_end, 7)
        publication = runtime.poll()[0]

        assert publication.steps[0].boundary == CHUNK_TOKENS
        assert publication.steps[0].resident_count == 0
        assert len(cleanup_calls) == 1
        assert lifecycle == ["epoch-cleanup", "bad-ack-rejected", "ack"]
        assert runtime.view_for("source").resident_count == 0

        competitor_id = runtime.binding_for("competitor").request_id
        reuse_probe = native.prepare_append((EngineAppendIntent(competitor_id, 1),))
        reused = reuse_probe.steps[0].write_intents[0]
        old_generations = dict(old_pages)
        assert reused.page_id in old_generations
        assert reused.page_generation == old_generations[reused.page_id] + 1
        native.abort_prepared(
            reuse_probe.batch_id,
            (EngineStepAbortEvidence(competitor_id, True),),
        )

        next_epoch = runtime.prepare((("source", CHUNK_TOKENS + 1),))
        next_lowering = next_epoch.steps[0].class_lowerings[0]
        assert next_lowering.flags == (
            CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START
        )
        assert (
            next_lowering.previous_layout_boundary,
            next_lowering.target_layout_boundary,
        ) == (CHUNK_TOKENS, CHUNK_TOKENS + 1)
        assert len(next_epoch.steps[0].write_intents) == 1
        next_page = next_epoch.steps[0].write_intents[0]
        assert next_page.page_id in old_generations
        assert next_page.page_id == reused.page_id
        assert next_page.page_generation == reused.page_generation + 1

        lowered = lower_batch_plan(
            next_epoch,
            config,
            runtime.arenas,
            {runtime.binding_for("source").request_id: runtime.view_for("source")},
        )
        assert lowered.steps[0].previous_boundary == CHUNK_TOKENS
        assert lowered.steps[0].target_boundary == CHUNK_TOKENS + 1
        assert lowered.steps[0].class_specs[0].previous_layout_boundary == CHUNK_TOKENS
        assert lowered.steps[0].class_specs[0].target_layout_boundary == CHUNK_TOKENS + 1
        assert lowered.steps[0].class_specs[0].exact_new_pages == (
            next_page.page_id,
        )
        absolute_column_locations = tuple(
            zip(
                range(
                    lowered.steps[0].previous_boundary,
                    lowered.steps[0].target_boundary,
                ),
                range(
                    next_page.page_id * PAGE_TOKENS,
                    next_page.page_id * PAGE_TOKENS + 1,
                ),
                strict=True,
            )
        )
        assert absolute_column_locations == (
            (CHUNK_TOKENS, next_page.page_id * PAGE_TOKENS),
        )

        _submit(runtime, config, next_epoch, 8)
        runtime.poll()

        key = PrefixSemanticKey(b"n" * 32, b"d" * 32, PAGE_TOKENS)
        with pytest.raises(
            ManagerError, match="request-private cache sharing policy"
        ):
            runtime.prefix_lookup((key,))
        with pytest.raises(
            ManagerError, match="request-private cache sharing policy"
        ):
            runtime.prefix_publish((("source", key),))

        release = runtime.prepare_release(("source", "competitor"))
        assert len(release.releases) == 2
        assert len(release.retirements) == 1
        runtime.confirm_release(release)
        stats = runtime.stats()
        assert stats.active_requests == 0
        assert stats.active_snapshots == 0
        assert stats.active_pages == 0
        assert stats.retiring_pages == 0
        assert stats.pending_reclamations == 0
        assert stats.free_pages == 2
        assert lifecycle[-1] == "release-cleanup"
        drained = True
    finally:
        if not drained and runtime.failure_reason is None:
            runtime.fail_stop("test cleanup after incomplete lifecycle")
        else:
            runtime.close()
