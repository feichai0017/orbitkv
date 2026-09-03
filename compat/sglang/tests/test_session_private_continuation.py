from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

from orbitkv_sglang.bridge import session_cache, session_lowering, state
from orbitkv_sglang.bridge.private_prefix import PrivatePrefixProvenance
from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.ffi.session_types import EngineRequestId, EngineRequestView
from orbitkv_sglang.session_runtime import SessionRequestBinding


PAGE_TOKENS = 16


def _config(retention: str) -> RuntimeConfig:
    chunked = retention == "chunked"
    class_config = ClassConfig(
        class_id=0,
        pool_id=3,
        backend_domain=1,
        name=retention,
        layers=(0,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=None if chunked else 32,
        period_blocks=None if chunked else 3,
        storage="token_kv",
        chunk_tokens=32 if chunked else None,
        blocks_per_epoch=2 if chunked else None,
    )
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint=f"sha256:{retention}-private-continuation",
        page_tokens=PAGE_TOKENS,
        classes=(class_config,),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": f"whole_domain_{retention}_token_kv"}
        ),
        manager_plan_format="retention_ir" if chunked else "kv_plan",
    )


class _Runtime:
    def __init__(
        self,
        key: Any,
        request_id: EngineRequestId,
        row: int | None,
        boundary: int,
        resident_count: int | None = None,
    ):
        self.failure_reason = None
        self._binding = SessionRequestBinding(key, request_id, row)
        resident_count = (1 if boundary else 0) if resident_count is None else resident_count
        self._view = EngineRequestView(request_id, 2, boundary, resident_count)
        self.waited: list[tuple[Any, ...]] = []

    def binding_for(self, _key: Any) -> SessionRequestBinding:
        return self._binding

    def view_for(self, _key: Any) -> EngineRequestView:
        return self._view

    def wait_requests(self, keys: Any) -> None:
        self.waited.append(tuple(keys))


def _private_batch(
    retention: str, *, boundary: int, resident_count: int | None = None
) -> tuple[Any, Any, _Runtime]:
    key = ("str", f"{retention}-continuation")
    request_id = EngineRequestId(9)
    row_index = 1 if boundary else None
    runtime = _Runtime(key, request_id, row_index, boundary, resident_count)
    table = torch.zeros((4, 64), dtype=torch.int32)
    if boundary:
        table[1, :boundary] = torch.arange(16, 16 + boundary, dtype=torch.int32)
    req = SimpleNamespace(
        rid=key[1],
        req_pool_idx=row_index,
        kv=(
            None
            if boundary == 0
            else SimpleNamespace(kv_allocated_len=boundary, swa_evicted_seqlen=0)
        ),
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        cache_protected_len=0,
        _orbitkv_request_key=key,
        _orbitkv_engine_request_id=request_id,
    )
    cache = SimpleNamespace(
        _no_prefix=True,
        req_to_token_pool=SimpleNamespace(req_to_token=table, max_context_len=64),
        _session_requests=(
            {}
            if boundary == 0
            else {key: session_cache._RequestEntry(req, key, request_id, 1)}
        ),
        _session_pending_requests=(
            {key: session_cache._PendingRequestEntry(req, key, request_id)}
            if boundary == 0
            else {}
        ),
        _session_pending_shared_prefix={},
        _session_active_shared_prefix={},
    )
    target = boundary + 4
    batch = SimpleNamespace(
        reqs=[req],
        tree_cache=cache,
        req_to_token_pool=cache.req_to_token_pool,
        prefix_lens=[boundary],
        extend_lens=[4],
        extend_num_tokens=4,
        seq_lens_cpu=torch.tensor([target], dtype=torch.int64),
        seq_lens=torch.tensor([target], dtype=torch.int64),
        device=torch.device("cpu"),
    )
    state._install_test_state(
        config=_config(retention),
        limits=state.RuntimeLimits(4, 32, 64),
        runtime=runtime,
    )
    return batch, req, runtime


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


@pytest.mark.parametrize("retention", ("sliding", "chunked"))
def test_unfinished_private_request_reenters_extend_preflight(retention: str) -> None:
    batch, req, runtime = _private_batch(retention, boundary=16)
    key = req._orbitkv_request_key

    session_cache.cache_unfinished_request(batch.tree_cache, req)
    session_lowering._validate_profile(batch)
    prefixes, _extensions, _targets = session_lowering._preflight_extend_batch(batch)
    pending, authoritative = session_lowering._preflight_pending_admission(
        batch, (key,), prefixes, (False,)
    )

    assert pending == (None,)
    assert authoritative == (False,)
    assert runtime.waited == [(key,)]
    assert type(req._orbitkv_private_prefix) is PrivatePrefixProvenance
    assert req._orbitkv_private_prefix.tensor is req.prefix_indices
    assert torch.equal(
        req.prefix_indices,
        batch.req_to_token_pool.req_to_token[1, :16].to(torch.int64),
    )

    # The next extend performs the same admission preflight before row
    # allocation or native prepare.  The private snapshot stays non-shared.
    next_pending, next_authoritative = session_lowering._preflight_pending_admission(
        batch, (key,), prefixes, (False,)
    )
    assert next_pending == (None,)
    assert next_authoritative == (False,)


@pytest.mark.parametrize("retention", ("sliding", "chunked"))
def test_rowless_private_admission_requires_exact_empty_state(retention: str) -> None:
    batch, req, _runtime = _private_batch(retention, boundary=0)
    key = req._orbitkv_request_key

    pending, authoritative = session_lowering._preflight_pending_admission(
        batch, (key,), (0,), (True,)
    )

    assert pending == (batch.tree_cache._session_pending_requests[key],)
    assert authoritative == (False,)

    req._orbitkv_private_prefix = object()
    with pytest.raises(RuntimeError, match="exactly empty"):
        session_lowering._preflight_pending_admission(
            batch, (key,), (0,), (True,)
        )


def test_chunked_epoch_boundary_accepts_absolute_zero_private_prefix() -> None:
    batch, req, runtime = _private_batch(
        "chunked", boundary=32, resident_count=0
    )
    batch.req_to_token_pool.req_to_token[1, :32].zero_()
    key = req._orbitkv_request_key

    session_cache.cache_unfinished_request(batch.tree_cache, req)
    pending, authoritative = session_lowering._preflight_pending_admission(
        batch, (key,), (32,), (False,)
    )

    assert pending == (None,)
    assert authoritative == (False,)
    assert runtime.waited == [(key,)]
    assert req.prefix_indices.shape == (32,)
    assert not torch.count_nonzero(req.prefix_indices)
    assert req._orbitkv_private_prefix.boundary == 32


@pytest.mark.parametrize(
    ("retention", "boundary", "resident_count"),
    (("chunked", 32, 1), ("sliding", 32, 1)),
)
def test_private_continuation_rejects_wrong_resident_count(
    retention: str, boundary: int, resident_count: int
) -> None:
    batch, req, _runtime = _private_batch(
        retention, boundary=boundary, resident_count=resident_count
    )
    key = req._orbitkv_request_key

    with pytest.raises(RuntimeError, match="resident count differs"):
        session_cache.cache_unfinished_request(batch.tree_cache, req)
    with pytest.raises(RuntimeError, match="resident count differs"):
        session_lowering._preflight_pending_admission(
            batch, (key,), (boundary,), (False,)
        )
