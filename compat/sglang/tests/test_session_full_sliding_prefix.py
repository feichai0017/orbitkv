from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402
from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineControlDisposition,
    EngineControlId,
    EngineControlOutcome,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixId,
    EngineRequestId,
    EngineRequestView,
)
from orbitkv_sglang.bridge import (  # noqa: E402
    prefix_runtime_helpers, private_prefix, session_cache, state,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity, PageLease, PrefixSemanticKey, SnapshotPage,
)
from orbitkv_sglang.session_runtime import (  # noqa: E402
    SessionMaterializationUpdate,
)


PAGE_TOKENS = 16
BOUNDARY = 32
WINDOW = 18
FRONTIER = BOUNDARY - (WINDOW - 1)


def _class(class_id: int, retention: str) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=3 + class_id,
        backend_domain=17 + class_id,
        name=retention,
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=WINDOW if retention == "sliding" else None,
        period_blocks=3 if retention == "sliding" else None,
        storage="token_kv",
    )


def _config(*retentions: str) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:full-sliding-prefix-test",
        page_tokens=PAGE_TOKENS,
        classes=tuple(_class(index, value) for index, value in enumerate(retentions)),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_sliding_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )


def _arena(class_config: ClassConfig) -> ArenaIdentity:
    return ArenaIdentity(
        7, 11 + class_config.class_id, class_config.pool_id,
        class_config.class_id, class_config.backend_domain, 32, PAGE_TOKENS,
        0, 1 + 32 * class_config.class_id,
    )


class _Allocator:
    def __init__(self) -> None:
        self.full_to_swa_index_mapping = torch.zeros((2048,), dtype=torch.int64)


def _install(*retentions: str) -> tuple[Any, Any, _Allocator]:
    config = _config(*retentions)
    arenas = tuple(_arena(item) for item in config.classes)
    runtime = SimpleNamespace(arenas_by_class={item.class_id: item for item in arenas})
    allocator = _Allocator()
    state._install_test_state(
        config=config, limits=state.RuntimeLimits(4, 64, 128), runtime=runtime
    )
    state._ALLOCATOR = allocator
    cache = SimpleNamespace(
        page_size=PAGE_TOKENS,
        device=torch.device("cpu"),
        _empty=torch.empty((0,), dtype=torch.int64),
    )
    return cache, runtime, allocator


def _page(
    runtime: Any, class_id: int, ordinal: int, *,
    visible_offset: int = 0, visible_count: int = PAGE_TOKENS,
) -> SnapshotPage:
    arena = runtime.arenas_by_class[class_id]
    slot = ordinal + 4 * class_id
    page_id = arena.first_page_id + slot
    return SnapshotPage(
        PageLease(arena.engine_epoch, arena.pool_epoch, 1, page_id, arena.pool_id),
        ordinal,
        ordinal if class_id == 0 else ordinal % 3,
        0 if class_id == 0 else ordinal // 3,
        arena.backend_base_index + slot,
        class_id,
        arena.backend_domain,
        PAGE_TOKENS,
        visible_offset,
        visible_count,
    )


def _full_sliding_pages(runtime: Any) -> tuple[SnapshotPage, ...]:
    return (
        _page(runtime, 0, 0),
        _page(runtime, 0, 1),
        _page(runtime, 1, 0, visible_offset=FRONTIER, visible_count=1),
        _page(runtime, 1, 1),
    )


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def test_real_helper_materializes_exact_full_sliding_tail_and_frontier() -> None:
    cache, runtime, allocator = _install("full", "sliding")
    pages = _full_sliding_pages(runtime)
    expected_full = torch.arange(16, 48, dtype=torch.int64)
    expected_swa = torch.arange(80, 112, dtype=torch.int64)
    allocator.full_to_swa_index_mapping[expected_full] = expected_swa

    result = prefix_runtime_helpers.validate_prefix_materialization(
        cache, pages, BOUNDARY, len(pages)
    )

    assert torch.equal(result.indices, expected_full)
    assert result.swa_evicted_seqlen == FRONTIER
    assert torch.equal(
        prefix_runtime_helpers.materialize_prefix_pages(
            cache, pages, BOUNDARY, len(pages)
        ),
        expected_full,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda pages: pages[:-1], "resident count"),
        (lambda pages: pages + (pages[-1],), "resident count"),
        (lambda pages: pages[:3] + (pages[2],), "page identity"),
        (lambda pages: pages[:2] + (replace(pages[2], class_id=0), pages[3]),
         "class arena"),
        (lambda pages: pages[:2] + (replace(pages[2], logical_ordinal=1), pages[3]),
         "retained tail"),
        (lambda pages: pages[:2] + (replace(pages[2], visible_token_offset=0), pages[3]),
         "retained tail"),
    ),
)
def test_real_helper_rejects_hostile_full_sliding_pages(
    mutation: Any, message: str
) -> None:
    cache, runtime, allocator = _install("full", "sliding")
    pages = _full_sliding_pages(runtime)
    allocator.full_to_swa_index_mapping[torch.arange(16, 48)] = torch.arange(80, 112)
    hostile = mutation(pages)

    with pytest.raises(RuntimeError, match=message):
        prefix_runtime_helpers.validate_prefix_materialization(
            cache, hostile, BOUNDARY, len(pages)
        )


def test_real_helper_rejects_every_disagreeing_swa_lut_location() -> None:
    cache, runtime, allocator = _install("full", "sliding")
    pages = _full_sliding_pages(runtime)
    full = torch.arange(16, 48)
    allocator.full_to_swa_index_mapping[full] = torch.arange(80, 112)
    allocator.full_to_swa_index_mapping[31] = 999

    with pytest.raises(RuntimeError, match="Full-to-SWA mirror"):
        prefix_runtime_helpers.validate_prefix_materialization(
            cache, pages, BOUNDARY, len(pages)
        )


def test_full_only_materialization_is_preserved() -> None:
    cache, runtime, _allocator = _install("full")
    pages = (_page(runtime, 0, 0), _page(runtime, 0, 1))

    result = prefix_runtime_helpers.validate_prefix_materialization(
        cache, pages, BOUNDARY, len(pages)
    )

    assert result.swa_evicted_seqlen == 0
    assert result.indices.tolist() == list(range(16, 48))


def test_confirmed_warm_attach_retains_frontier_for_kv_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    cache, runtime, allocator = _install("full", "sliding")
    pages = _full_sliding_pages(runtime)
    allocator.full_to_swa_index_mapping[torch.arange(16, 48)] = torch.arange(80, 112)
    materialized = prefix_runtime_helpers.validate_prefix_materialization(
        cache, pages, BOUNDARY, len(pages)
    )
    key = ("str", "warm")
    request_id = EngineRequestId(9)
    prefix_id = EnginePrefixId(7, 5)
    control_id = EngineControlId(7, 6)
    semantic = PrefixSemanticKey(b"n" * 32, b"d" * 32, BOUNDARY)
    node = SimpleNamespace(boundary=BOUNDARY, digest=semantic.digest,
                           resident_count=len(pages), lock_ref=2)
    req = SimpleNamespace(rid="warm", req_pool_idx=1, kv=None)
    table = torch.zeros((4, 64), dtype=torch.int32)
    runtime.binding_for = lambda _key: SimpleNamespace(
        request_id=request_id, request_row=1)
    confirmed = {"view": EngineRequestView(request_id, 1, 0, 0)}
    runtime.view_for = lambda _key: confirmed["view"]
    cache.req_to_token_pool = SimpleNamespace(req_to_token=table)
    cache._session_pending_requests = {
        key: session_cache._PendingRequestEntry(req, key, request_id)
    }
    cache._session_requests = {}
    cache._session_pending_shared_prefix = {}
    cache._session_active_shared_prefix = {}
    req._orbitkv_request_key = key
    req._orbitkv_engine_request_id = request_id
    private_prefix.install_shared_prefix_metadata(
        req, key=key, request_id=request_id, prefix_id=prefix_id, node=node,
        semantic=semantic, indices=materialized.indices, boundary=BOUNDARY,
        provisional=True,
    )
    plan = EngineMaterializationPlan(control_id, (EngineMaterializedRequest(
        request_id, 2, BOUNDARY, len(pages), pages),))
    entry = session_cache.register_pending_shared_prefix(
        cache, req=req, prefix_id=prefix_id, semantic=semantic, node=node,
        boundary=BOUNDARY, control_id=control_id, plan=plan,
        swa_evicted_seqlen=materialized.swa_evicted_seqlen,
    )
    cache._materialize_prefix_pages = lambda *_args: materialized.indices
    cache.dec_lock_ref = lambda current: setattr(
        current, "lock_ref", current.lock_ref - 1
    )

    def confirm_control(value: EngineControlId) -> EngineControlOutcome:
        assert value == control_id
        assert session_cache._materialization_callback(cache, (
            SessionMaterializationUpdate(
                key, request_id, 1, 2, BOUNDARY, len(pages), pages),
        )) is True
        confirmed["view"] = EngineRequestView(
            request_id, 2, BOUNDARY, len(pages)
        )
        return EngineControlOutcome(
            control_id, EngineControlDisposition.MATERIALIZED
        )

    runtime.confirm_control = confirm_control
    session_cache.confirm_pending_shared_prefix(cache, req)
    assert req.kv is None
    assert torch.equal(table[1, :BOUNDARY], materialized.indices.to(torch.int32))
    assert cache._session_active_shared_prefix[key] is entry
    assert cache._session_pending_shared_prefix == {}
    session_cache.promote_pending_requests(cache, (req,))

    assert type(req.kv) is ReqKvInfo
    assert req.kv.kv_allocated_len == BOUNDARY
    assert req.kv.swa_evicted_seqlen == FRONTIER
