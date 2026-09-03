from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

from orbitkv_sglang.bridge import (
    prefix_runtime_helpers,
    session_cache,
    session_lowering,
    state,
)
from orbitkv_sglang.bridge.private_prefix import PrivatePrefixProvenance
from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.execution_plan import (
    BindingResult,
    StepExecutionResult,
    confirm_execution,
    expected_bindings,
    lower_batch_plan,
)
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.ffi.session_types import EnginePublicationEvidence
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    FailStopped,
    ManagerCreateSettings,
    SessionCreateSettings,
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
        ).encode(),
        plan_fingerprint="sha256:session-engine-integration",
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
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:session-engine-manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_chunked_token_kv"}
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


class _Cache(session_cache.SessionCacheMixin):
    def __init__(self, pool: Any, allocator: Any) -> None:
        self.req_to_token_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self._no_prefix = True


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def _bind_cache(
    runtime: SessionRuntime, config: RuntimeConfig
) -> tuple[_Cache, Any]:
    pool = SimpleNamespace(
        req_to_token=torch.zeros((2, 64), dtype=torch.int32),
        max_context_len=64,
        device=torch.device("cpu"),
    )
    allocator = SimpleNamespace()
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(1, CHUNK_TOKENS, 64),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator
    cache = _Cache(pool, allocator)
    session_cache.bind_session_cache(cache)
    return cache, pool


def _bind_request(runtime: SessionRuntime, cache: _Cache, key: Any) -> Any:
    req = SimpleNamespace(
        rid=key[1],
        req_pool_idx=None,
        kv=None,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        cache_protected_len=0,
    )
    assert prefix_runtime_helpers.ensure_session_request_identity(cache, req) == (
        key,
        runtime.binding_for(key).request_id,
    )
    req.req_pool_idx = 1
    runtime.bind_request_rows(((key, 1),))
    session_cache.promote_pending_requests(cache, (req,))
    req.kv = SimpleNamespace(kv_allocated_len=0, swa_evicted_seqlen=0)
    return req


def _evidence(
    runtime: SessionRuntime, config: RuntimeConfig, plan: Any, key: Any
) -> Any:
    request_id = runtime.binding_for(key).request_id
    lowered = lower_batch_plan(
        plan, config, runtime.arenas, {request_id: runtime.view_for(key)}
    )
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


def _write_locations(pool: Any, plan: Any) -> None:
    step = plan.steps[0]
    lowering = step.class_lowerings[0]
    begin = int(step.previous_boundary)
    target = int(step.target_boundary)
    locations = []
    if begin % PAGE_TOKENS:
        action = step.tail_actions[lowering.tail_offset]
        count = min(target - begin, PAGE_TOKENS - begin % PAGE_TOKENS)
        start = action.destination.page_id * PAGE_TOKENS + begin % PAGE_TOKENS
        locations.extend(range(start, start + count))
        begin += count
    for intent in step.write_intents[
        lowering.write_offset : lowering.write_offset + lowering.write_count
    ]:
        count = min(target - begin, PAGE_TOKENS)
        start = intent.page_id * PAGE_TOKENS
        locations.extend(range(start, start + count))
        begin += count
    pool.req_to_token[1, int(step.previous_boundary) : target] = torch.tensor(
        locations, dtype=torch.int32
    )


def _submit(
    runtime: SessionRuntime, config: RuntimeConfig, plan: Any, key: Any, domain: int
) -> None:
    ticket = runtime.submit(_evidence(runtime, config, plan, key))
    runtime.register_event(ticket, _ReadyEvent(), domain)


def test_epoch_cleanup_precedes_ack_and_private_continuation(
    ffi_library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ffi_library)
    native = CtypesRuntimeSession.create(
        config,
        SessionCreateSettings(
            ManagerCreateSettings(1, 2, 1, 2, CHUNK_TOKENS),
            CacheSharingPolicy.REQUEST_PRIVATE,
        ),
        (ArenaRegistration(0, POOL_ID, BACKEND_DOMAIN, 2, BACKEND_BASE_INDEX),),
    )
    runtime = SessionRuntime(native)
    cache, pool = _bind_cache(runtime, config)
    key = ("str", "request")
    req = _bind_request(runtime, cache, key)
    trace: list[str] = []
    coordinator = state._MIRROR_CLEANUP
    for phase in ("preflight", "commit", "synchronize", "finalize"):
        original = getattr(coordinator, phase)

        def traced(*args: Any, _phase: str = phase, _original: Any = original):
            trace.append(_phase)
            return _original(*args)

        monkeypatch.setattr(coordinator, phase, traced)
    original_ack = native.confirm_publication
    expect_epoch_cleanup = False

    def confirm_after_cleanup(evidence: EnginePublicationEvidence) -> None:
        if expect_epoch_cleanup:
            expected = ["preflight", "commit", "synchronize", "finalize"]
            assert trace[-4:] == expected
            assert not torch.count_nonzero(pool.req_to_token[1, :CHUNK_TOKENS])
            trace.append("ack")
        original_ack(evidence)

    native.confirm_publication = confirm_after_cleanup
    drained = False
    try:
        first = runtime.prepare(((key, CHUNK_TOKENS - 1),))
        _write_locations(pool, first)
        _submit(runtime, config, first, key, 7)
        runtime.poll()
        req.kv.kv_allocated_len = CHUNK_TOKENS - 1

        epoch_end = runtime.prepare(((key, CHUNK_TOKENS),))
        _write_locations(pool, epoch_end)
        req.kv.kv_allocated_len = CHUNK_TOKENS
        trace.clear()
        expect_epoch_cleanup = True
        _submit(runtime, config, epoch_end, key, 7)
        publication = runtime.poll()[0]

        assert publication.steps[0].resident_count == 0
        assert trace == [
            "preflight", "commit", "synchronize", "finalize", "ack"
        ]
        assert not torch.count_nonzero(pool.req_to_token[1, :CHUNK_TOKENS])

        session_cache.cache_unfinished_request(cache, req)
        assert type(req._orbitkv_private_prefix) is PrivatePrefixProvenance
        assert req.prefix_indices.shape == (CHUNK_TOKENS,)
        assert not torch.count_nonzero(req.prefix_indices)
        batch = SimpleNamespace(
            reqs=[req],
            tree_cache=cache,
            req_to_token_pool=pool,
            prefix_lens=[CHUNK_TOKENS],
            extend_lens=[1],
            extend_num_tokens=1,
            seq_lens_cpu=torch.tensor([CHUNK_TOKENS + 1]),
            seq_lens=torch.tensor([CHUNK_TOKENS + 1]),
            device=torch.device("cpu"),
        )
        prefixes, _extensions, targets = session_lowering._preflight_extend_batch(batch)
        pending, authoritative = session_lowering._preflight_pending_admission(
            batch, (key,), prefixes, (False,)
        )
        assert pending == (None,)
        assert authoritative == (False,)
        next_epoch = runtime.prepare(((key, targets[0]),))
        assert next_epoch.steps[0].previous_boundary == CHUNK_TOKENS
        assert next_epoch.steps[0].target_boundary == CHUNK_TOKENS + 1
        _write_locations(pool, next_epoch)
        expect_epoch_cleanup = False
        _submit(runtime, config, next_epoch, key, 8)
        next_publication = runtime.poll()[0]
        req.kv.kv_allocated_len = CHUNK_TOKENS + 1
        assert next_publication.steps[0].resident_count == 1
        assert int(pool.req_to_token[1, CHUNK_TOKENS]) > 0

        release = runtime.prepare_release((key,))
        runtime.confirm_release(release)
        assert not torch.count_nonzero(pool.req_to_token[1, : CHUNK_TOKENS + 1])
        assert not hasattr(req, "_orbitkv_private_prefix")
        drained = True
    finally:
        if not drained and runtime.failure_reason is None:
            runtime.fail_stop("test cleanup after incomplete bridge lifecycle")
        else:
            runtime.close()


@pytest.mark.parametrize(
    "failed_phase", ("preflight", "commit", "synchronize", "finalize")
)
def test_epoch_cleanup_phase_failure_stops_before_ack(
    ffi_library: Path, monkeypatch: pytest.MonkeyPatch, failed_phase: str
) -> None:
    config = _config(ffi_library)
    native = CtypesRuntimeSession.create(
        config,
        SessionCreateSettings(
            ManagerCreateSettings(1, 2, 1, 2, CHUNK_TOKENS),
            CacheSharingPolicy.REQUEST_PRIVATE,
        ),
        (ArenaRegistration(0, POOL_ID, BACKEND_DOMAIN, 2, BACKEND_BASE_INDEX),),
    )
    runtime = SessionRuntime(native)
    cache, pool = _bind_cache(runtime, config)
    key = ("str", f"failure-{failed_phase}")
    req = _bind_request(runtime, cache, key)

    first = runtime.prepare(((key, CHUNK_TOKENS - 1),))
    _write_locations(pool, first)
    _submit(runtime, config, first, key, 7)
    runtime.poll()
    req.kv.kv_allocated_len = CHUNK_TOKENS - 1

    coordinator = state._MIRROR_CLEANUP

    def fail_phase(*_args: Any) -> None:
        raise RuntimeError(f"{failed_phase} fault")

    monkeypatch.setattr(coordinator, failed_phase, fail_phase)
    acknowledgements: list[Any] = []
    original_ack = native.confirm_publication

    def record_ack(evidence: EnginePublicationEvidence) -> None:
        acknowledgements.append(evidence)
        original_ack(evidence)

    native.confirm_publication = record_ack
    epoch_end = runtime.prepare(((key, CHUNK_TOKENS),))
    _write_locations(pool, epoch_end)
    req.kv.kv_allocated_len = CHUNK_TOKENS
    _submit(runtime, config, epoch_end, key, 7)

    with pytest.raises(FailStopped, match=f"{failed_phase} fault"):
        runtime.poll()

    assert acknowledgements == []
    assert runtime.failure_reason is not None
