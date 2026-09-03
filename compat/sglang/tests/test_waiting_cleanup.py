from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineControlId,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePendingAttachCancel,
    EnginePendingAttachCancelDisposition,
    EnginePendingAttachCancelOutcome,
    EnginePrefixId,
    EngineRequestId,
    EngineRequestView,
)
from orbitkv_sglang.bridge import session_cache, state, waiting_cleanup  # noqa: E402
from orbitkv_sglang.runtime import RetryableConflict  # noqa: E402
from orbitkv_sglang.session_runtime import SessionRequestBinding  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_bridge_state():
    state._install_test_state()
    yield
    state._install_test_state()


def _request(rid: str) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid, req_pool_idx=None, kv=None,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        cache_protected_len=0,
    )


def test_miss_only_retry_withholds_removal_then_completes(monkeypatch):
    req = _request("miss")
    key = ("str", "miss")
    pending = SimpleNamespace(request_id=EngineRequestId(3))
    cache = SimpleNamespace(
        _session_pending_requests={key: pending},
        _session_pending_shared_prefix={},
    )
    calls = []
    monkeypatch.setattr(state, "_uses_runtime_session", lambda: True)

    def cancel(current, request):
        calls.append((current, request))
        if len(calls) == 1:
            raise RetryableConflict("retry")

    monkeypatch.setattr(session_cache, "cancel_pending_request", cancel)

    assert waiting_cleanup.prepare_waiting_request_removal(req, cache) is False
    assert hasattr(req, "_orbitkv_pending_waiting_removal")
    assert waiting_cleanup.prepare_waiting_request_removal(req, cache) is True
    assert not hasattr(req, "_orbitkv_pending_waiting_removal")
    assert calls == [(cache, req), (cache, req)]


def test_unowned_waiting_request_is_a_true_noop(monkeypatch):
    req = _request("unowned")
    cache = SimpleNamespace(
        _session_pending_requests={}, _session_pending_shared_prefix={}
    )
    monkeypatch.setattr(state, "_uses_runtime_session", lambda: True)
    import orbitkv_sglang.bridge.lowering as lowering
    monkeypatch.setattr(
        lowering, "_release_kv_cache",
        lambda *_args, **_kwargs: pytest.fail("unowned request was released"),
    )

    assert waiting_cleanup.prepare_waiting_request_removal(req, cache) is True


def _committed_fixture(monkeypatch, *, finalize_retries: int = 0):
    req = _request("hit")
    key = ("str", "hit")
    request_id = EngineRequestId(41)
    control_id = EngineControlId(7, 9)
    prefix_id = EnginePrefixId(7, 11)
    node = SimpleNamespace(lock_ref=1)
    materialized = EngineMaterializedRequest(
        request_id, 2, 16, 1, (object(),)
    )
    pending = SimpleNamespace(req=req, key=key, request_id=request_id)
    shared = SimpleNamespace(
        req=req, key=key, request_id=request_id, control_id=control_id,
        prefix_id=prefix_id, node=node,
        plan=EngineMaterializationPlan(control_id, (materialized,)),
    )
    expected = EnginePendingAttachCancel(control_id, request_id, prefix_id, 2, 16, 1)
    events = []

    class Runtime:
        failure_reason = None
        retries = finalize_retries

        def binding_for(self, current):
            return SessionRequestBinding(current, request_id, None)

        def view_for(self, current):
            return EngineRequestView(request_id, 1, 0, 0)

        def cancel_pending_attach(self, actual, current):
            events.append(("cancel", actual, current))
            return EnginePendingAttachCancelOutcome(
                actual.control_id, actual.request_id, actual.prefix_id,
                actual.view_version, actual.boundary, actual.resident_count,
                EnginePendingAttachCancelDisposition.RECYCLE_PENDING,
            )

        def finalize_pending_attach_cancel(self, actual, current):
            events.append(("finalize", actual, current))
            if self.retries:
                self.retries -= 1
                raise RetryableConflict("retry finalize")
            return EnginePendingAttachCancelOutcome(
                actual.control_id, actual.request_id, actual.prefix_id,
                actual.view_version, actual.boundary, actual.resident_count,
                EnginePendingAttachCancelDisposition.FINALIZED,
            )

        def fail_stop(self, reason):
            self.failure_reason = reason

    runtime = Runtime()
    req._orbitkv_request_key = key
    req._orbitkv_engine_request_id = request_id
    req.prefix_indices = torch.arange(16, dtype=torch.int64)
    req._orbitkv_engine_prefix_id = prefix_id
    req._orbitkv_prefix_node = node
    req._orbitkv_prefix_semantic = object()
    req._orbitkv_provisional_prefix_lock = True
    req._orbitkv_prefix_lock_held = False
    req.last_node = req.last_host_node = req.best_match_node = node
    cache = SimpleNamespace(
        root_node=object(),
        _session_pending_requests={key: pending},
        _session_pending_shared_prefix={key: shared},
        _session_active_shared_prefix={},
    )
    cache._preflight_release_node = lambda request, provisional: node

    def commit(request, current_node, *, provisional):
        events.append(("drop-lock", current_node, provisional))
        current_node.lock_ref -= 1
        del request._orbitkv_provisional_prefix_lock

    cache._commit_release_node = commit
    monkeypatch.setattr(state, "_uses_runtime_session", lambda: True)
    monkeypatch.setattr(state, "_RUNTIME", runtime)
    monkeypatch.setattr(waiting_cleanup, "_runtime", lambda: runtime)
    monkeypatch.setattr(
        session_cache, "pending_shared_prefix", lambda current, request: shared
    )
    return req, cache, runtime, expected, node, events


def test_committed_attach_cancel_cleans_once_then_finalizes(monkeypatch):
    req, cache, runtime, expected, node, events = _committed_fixture(monkeypatch)

    assert waiting_cleanup.prepare_waiting_request_removal(req, cache) is True

    assert events == [
        ("cancel", expected, ("str", "hit")),
        ("drop-lock", node, True),
        ("finalize", expected, ("str", "hit")),
    ]
    assert node.lock_ref == 0
    assert cache._session_pending_requests == {}
    assert cache._session_pending_shared_prefix == {}
    assert req.prefix_indices.numel() == 0
    assert req.last_node is cache.root_node
    assert not hasattr(req, "_orbitkv_request_key")
    assert not hasattr(req, "_orbitkv_pending_waiting_removal")
    assert runtime.failure_reason is None


def test_finalize_retry_retains_work_and_does_not_repeat_local_cleanup(monkeypatch):
    req, cache, _runtime, expected, node, events = _committed_fixture(
        monkeypatch, finalize_retries=1
    )

    assert waiting_cleanup.prepare_waiting_request_removal(req, cache) is False
    assert node.lock_ref == 0
    assert cache._session_pending_requests == {}
    assert hasattr(req, "_orbitkv_pending_waiting_removal")

    assert waiting_cleanup.prepare_waiting_request_removal(req, cache) is True
    assert [event[0] for event in events] == [
        "cancel", "drop-lock", "finalize", "finalize"
    ]
    assert not hasattr(req, "_orbitkv_pending_waiting_removal")
    assert expected.request_id == EngineRequestId(41)


def test_deterministic_cancel_mismatch_fail_stops_instead_of_retrying(monkeypatch):
    req, cache, runtime, _expected, node, _events = _committed_fixture(monkeypatch)
    runtime.cancel_pending_attach = lambda *_args: (_ for _ in ()).throw(
        ValueError("identity mismatch")
    )

    with pytest.raises(Exception, match="waiting request removal"):
        waiting_cleanup.prepare_waiting_request_removal(req, cache)

    assert runtime.failure_reason is not None
    assert node.lock_ref == 1
