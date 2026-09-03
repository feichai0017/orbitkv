from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.lowering as lowering  # noqa: E402
import orbitkv_sglang.bridge.execution_context as execution_context  # noqa: E402
import orbitkv_sglang.bridge.location_validation as location_validation  # noqa: E402
import orbitkv_sglang.bridge.prefix_cache as prefix_cache  # noqa: E402
import orbitkv_sglang.bridge.session_cache as session_cache  # noqa: E402
import orbitkv_sglang.bridge.session_lowering as session_lowering  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
from orbitkv_sglang.config import ClassConfig, RuntimeConfig  # noqa: E402
from orbitkv_sglang.ffi.session_types import (  # noqa: E402
    EngineBatchId,
    EngineBatchPlan,
    EngineBatchTicket,
    EngineControlDisposition,
    EngineControlId,
    EngineControlOutcome,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixId,
    EngineReleaseId,
    EngineReleasePlan,
    EngineReleasedRequest,
    EngineRequestId,
    EngineRequestView,
    EngineStepPlan,
    ExecutionEvidence,
)
from orbitkv_sglang.runtime import (  # noqa: E402
    TAIL_COPY_ON_WRITE,
    TAIL_NONE,
    ArenaIdentity,
    ClassLowering,
    CopyIntent,
    FailStopped,
    ManagerError,
    PageLease,
    PrefixSemanticKey,
    SnapshotPage,
    TailAction,
    WriteIntent,
)
from orbitkv_sglang.session_runtime import SessionRequestBinding  # noqa: E402


PAGE_TOKENS = 16
ZERO_PAGE = PageLease(0, 0, 0, 0, 0)
FORWARD_CONTEXT_ATTR = "_orbitkv_forward_execution_context"


class _TestBaseException(BaseException):
    pass


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:session-lowering-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=3,
                backend_domain=1,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
                storage="token_kv",
            ),
        ),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )


def _pure_sliding_config() -> RuntimeConfig:
    full = _config().classes[0]
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:pure-sliding-session-lowering-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=full.pool_id,
                backend_domain=full.backend_domain,
                name="sliding",
                layers=full.layers,
                retention="sliding",
                bytes_per_token_per_layer=full.bytes_per_token_per_layer,
                window_tokens=32,
                period_blocks=3,
                storage="token_kv",
            ),
        ),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_sliding_token_kv"}
        ),
        manager_plan_format="kv_plan",
    )


def _latent_config() -> RuntimeConfig:
    full = _config().classes[0]
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:latent-session-lowering-test",
        page_tokens=PAGE_TOKENS,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=full.pool_id,
                backend_domain=full.backend_domain,
                name="latent",
                layers=full.layers,
                retention="full",
                bytes_per_token_per_layer=full.bytes_per_token_per_layer,
                window_tokens=None,
                period_blocks=None,
                storage="latent_kv",
                components=(("latent", 96), ("rope", 32)),
            ),
        ),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_latent_kv"}
        ),
        manager_plan_format="kv_plan",
    )


def _page(page_id: int, generation: int) -> PageLease:
    return PageLease(7, 11, generation, page_id, 3)


def _install_shared_prefix(
    req: Any,
    boundary: int,
    *,
    provisional: bool,
    locations: torch.Tensor | None = None,
) -> Any:
    digest = b"d" * 32
    node = SimpleNamespace(
        boundary=boundary,
        digest=digest,
        parent=object(),
        prefix=object(),
        resident_count=(boundary + PAGE_TOKENS - 1) // PAGE_TOKENS,
        evicted=False,
        lock_ref=2 if provisional else 1,
    )
    req.prefix_indices = (
        torch.arange(
            PAGE_TOKENS, PAGE_TOKENS + boundary, dtype=torch.int64
        )
        if locations is None
        else locations.to(dtype=torch.int64, copy=True)
    )
    req.last_node = node
    req._orbitkv_prefix_node = node
    req._orbitkv_prefix_semantic = PrefixSemanticKey(
        b"n" * 32, digest, boundary
    )
    req._orbitkv_provisional_prefix_lock = provisional
    req._orbitkv_prefix_lock_held = not provisional
    return node


class _MovePool:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    def move_kv_cache(self, destinations: Any, sources: Any) -> None:
        self.events.append(
            ("copy", tuple(destinations.tolist()), tuple(sources.tolist()))
        )


class _DeviceModule:
    def __init__(self) -> None:
        self.stream = object()
        self.calls: list[Any] = []

    def current_stream(self, device: Any) -> object:
        self.calls.append(device)
        return self.stream


class _Allocator:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.pool = _MovePool(events)
        self._orbitkv_free_group_state = "idle"
        self.free_group = []

    def get_kvcache(self) -> _MovePool:
        return self.pool


class _ReqToTokenPool:
    def __init__(self, events: list[Any], *, fail_write: bool = False) -> None:
        self.req_to_token = torch.zeros((8, 64), dtype=torch.int32)
        self.max_context_len = 64
        self.device = torch.device("cpu")
        self.events = events
        self.fail_write = fail_write
        self.freed: list[str] = []

    def write(self, indices: Any, values: Any) -> None:
        self.events.append("mirror")
        if self.fail_write:
            raise RuntimeError("injected mirror failure")
        self.req_to_token[indices] = values

    def free(self, req: Any) -> None:
        self.freed.append(req.rid)
        req.req_pool_idx = None


class _Runtime:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.failure_reason = None
        self.arenas = (
            ArenaIdentity(
                engine_epoch=7,
                pool_epoch=11,
                pool_id=3,
                class_id=0,
                backend_domain=1,
                page_count=32,
                page_tokens=PAGE_TOKENS,
                backend_base_index=0,
                first_page_id=1,
            ),
        )
        self.arenas_by_class = {0: self.arenas[0]}
        self.bindings: dict[Any, SessionRequestBinding] = {}
        self.views: dict[Any, EngineRequestView] = {}
        self.next_request_id = 1
        self.next_batch_id = 1
        self.submitted: list[ExecutionEvidence] = []
        self.aborted: list[Any] = []
        self.quarantined_prepared: list[Any] = []
        self.quarantined_submitted: list[Any] = []
        self.next_release_id = 1
        self.pending_controls: dict[EngineControlId, tuple[Any, EngineMaterializationPlan]] = {}

    def seed(
        self, key: Any, row: int | None, boundary: int
    ) -> EngineRequestId:
        request_id = EngineRequestId(self.next_request_id)
        self.next_request_id += 1
        self.bindings[key] = SessionRequestBinding(key, request_id, row)
        self.views[key] = EngineRequestView(
            request_id,
            1,
            boundary,
            (boundary + PAGE_TOKENS - 1) // PAGE_TOKENS,
        )
        return request_id

    def acquire(self, assignments: Any) -> tuple[EngineRequestView, ...]:
        values = tuple(assignments)
        self.events.append(("acquire", values))
        raise AssertionError("session lowering must not acquire after row allocation")

    def acquire_unbound(
        self, keys: Any
    ) -> tuple[EngineRequestView, ...]:
        values = tuple(keys)
        self.events.append(("acquire-unbound", values))
        if any(key in self.bindings for key in values):
            raise AssertionError("fake runtime acquired a request twice")
        result = []
        for key in values:
            self.seed(key, None, 0)
            result.append(self.views[key])
        return tuple(result)

    def bind_request_rows(self, assignments: Any) -> None:
        values = tuple(assignments)
        self.events.append(("bind-rows", values))
        if (
            not values
            or len({key for key, _row in values}) != len(values)
            or len({int(row) for _key, row in values}) != len(values)
            or any(
                self.bindings[key].request_row is not None
                for key, _row in values
            )
        ):
            raise RuntimeError("invalid fake request-row bind")
        replacements = tuple(
            SessionRequestBinding(
                key, self.bindings[key].request_id, int(row)
            )
            for key, row in values
        )
        for replacement in replacements:
            self.bindings[replacement.key] = replacement

    def binding_for(self, key: Any) -> SessionRequestBinding:
        return self.bindings[key]

    def view_for(self, key: Any) -> EngineRequestView:
        return self.views[key]

    def views_for(self, keys: Any) -> tuple[EngineRequestView, ...]:
        return tuple(self.views[key] for key in keys)

    def confirm_control(self, control_id: EngineControlId) -> EngineControlOutcome:
        key, plan = self.pending_controls.pop(control_id)
        item = plan.requests[0]
        binding = self.bindings[key]
        self.events.append(("confirm-control", control_id))
        callback = self.materialization
        if callback is None:
            raise RuntimeError("fake materialization callback is not bound")
        from orbitkv_sglang.session_runtime import SessionMaterializationUpdate

        assert callback((SessionMaterializationUpdate(
            key,
            item.request_id,
            int(binding.request_row),
            item.view_version,
            item.boundary,
            item.resident_count,
            item.pages,
        ),)) is True
        self.views[key] = EngineRequestView(
            item.request_id, item.view_version, item.boundary, item.resident_count
        )
        return EngineControlOutcome(
            control_id, EngineControlDisposition.MATERIALIZED
        )

    def quarantine_control(self, control_id: EngineControlId) -> None:
        self.events.append(("quarantine-control", control_id))
        self.pending_controls.pop(control_id, None)

    def prepare(self, appends: Any) -> EngineBatchPlan:
        values = tuple(appends)
        self.events.append(("prepare", values))
        if any(self.bindings[key].request_row is None for key, _target in values):
            raise AssertionError("fake runtime prepared an unbound request")
        batch_id = EngineBatchId(7, self.next_batch_id)
        self.next_batch_id += 1
        return EngineBatchPlan(
            batch_id,
            tuple(
                self._step(self.views[key], target) for key, target in values
            ),
        )

    def submit(self, evidence: ExecutionEvidence) -> EngineBatchTicket:
        self.events.append("submit")
        self.submitted.append(evidence)
        return EngineBatchTicket(
            evidence.batch_id, tuple(item.request_id for item in evidence.steps)
        )

    def abort_prepared(self, plan: EngineBatchPlan) -> None:
        self.events.append("abort")
        self.aborted.append(plan.batch_id)

    def quarantine_prepared(self, batch_id: Any) -> None:
        self.events.append("quarantine-prepared")
        self.quarantined_prepared.append(batch_id)
        self.failure_reason = "prepared execution was quarantined"
        raise FailStopped(self.failure_reason)

    def quarantine_submitted(self, ticket: Any) -> None:
        self.events.append("quarantine-submitted")
        self.quarantined_submitted.append(ticket)
        self.failure_reason = "submitted execution was quarantined"
        raise FailStopped(self.failure_reason)

    def poll(self) -> tuple[Any, ...]:
        self.events.append("poll")
        return ()

    @contextmanager
    def scheduler_turn(self):
        self.events.append("turn-enter")
        self.poll()
        try:
            yield
        finally:
            self.events.append("turn-exit")

    def wait_requests(self, keys: Any) -> None:
        self.events.append(("wait", tuple(keys)))

    def fail_stop(self, reason: str) -> None:
        self.failure_reason = reason

    def prepare_release(self, keys: Any) -> EngineReleasePlan:
        values = tuple(keys)
        self.events.append(("prepare-release", values))
        plan = EngineReleasePlan(
            EngineReleaseId(7, self.next_release_id),
            tuple(
                EngineReleasedRequest(
                    self.bindings[key].request_id, ()
                )
                for key in values
            ),
            (),
        )
        self.next_release_id += 1
        return plan

    def confirm_release(self, plan: EngineReleasePlan) -> None:
        self.events.append(("confirm-release", plan.release_id))
        released = {item.request_id for item in plan.releases}
        for key, binding in tuple(self.bindings.items()):
            if binding.request_id in released:
                del self.bindings[key]
                del self.views[key]

    @staticmethod
    def _step(view: EngineRequestView, target: int) -> EngineStepPlan:
        previous = view.boundary
        first_page = 1 + (int(view.request_id) - 1) * 2
        if previous == 0 or previous % PAGE_TOKENS == 0:
            action = TailAction(0, TAIL_NONE, 0, 0, ZERO_PAGE, ZERO_PAGE, 0)
            copies: tuple[CopyIntent, ...] = ()
            writes = (
                (
                    WriteIntent(
                        1, first_page + previous // PAGE_TOKENS, 0
                    ),
                )
                if (target + PAGE_TOKENS - 1) // PAGE_TOKENS
                > (previous + PAGE_TOKENS - 1) // PAGE_TOKENS
                else ()
            )
        else:
            source = _page(first_page, 1)
            destination = _page(first_page + 1, 2)
            action = TailAction(
                0, TAIL_COPY_ON_WRITE, previous, 0, source, destination, 0
            )
            copies = (
                CopyIntent(
                    0, 1, previous, 0, 0, source, destination, 0, 1, 0
                ),
            )
            writes = ()
        lowering_spec = ClassLowering(
            0,
            0,
            0,
            1,
            0,
            len(copies),
            0,
            len(writes),
            0,
            previous,
            target,
        )
        return EngineStepPlan(
            view.request_id,
            view.view_version,
            view.view_version + 1,
            previous,
            target,
            (lowering_spec,),
            (action,),
            copies,
            writes,
        )


def _batch(
    events: list[Any],
    allocator: _Allocator,
    runtime: _Runtime,
    *,
    previous: int,
    fail_write: bool = False,
) -> Any:
    pool = _ReqToTokenPool(events, fail_write=fail_write)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=None if previous == 0 else 1,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        cache_protected_len=0,
        kv=(
            None
            if previous == 0
            else SimpleNamespace(
                kv_allocated_len=previous, swa_evicted_seqlen=0
            )
        ),
    )
    key = ("str", req.rid)
    if previous:
        request_id = runtime.seed(key, 1, previous)
        req._orbitkv_request_key = key
        req._orbitkv_engine_request_id = request_id
        pool.req_to_token[1, :previous] = torch.arange(
            PAGE_TOKENS, PAGE_TOKENS + previous, dtype=torch.int32
        )
    tree_cache = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree_cache.token_to_kv_pool_allocator = allocator
    tree_cache.req_to_token_pool = pool
    tree_cache.page_size = PAGE_TOKENS
    tree_cache._no_prefix = False
    tree_cache.disable_finished_insert = True
    tree_cache.root_node = object()
    tree_cache._session_requests = {}
    tree_cache._session_pending_requests = {}
    tree_cache._session_pending_shared_prefix = {}
    tree_cache._session_active_shared_prefix = {}

    def preflight_release_node(candidate: Any, *, provisional: bool) -> Any:
        node = getattr(candidate, "_orbitkv_prefix_node", None)
        if node is None:
            if getattr(candidate, "last_node", None) is not tree_cache.root_node:
                raise RuntimeError("request prefix identity has no matching lock")
            return None
        marker = (
            "_orbitkv_provisional_prefix_lock"
            if provisional
            else "_orbitkv_prefix_lock_held"
        )
        if (
            getattr(candidate, "last_node", None) is not node
            or getattr(candidate, marker, None) is not True
            or int(node.lock_ref) <= 0
        ):
            raise RuntimeError("request prefix identity changed")
        return node

    def dec_lock_ref(node: Any) -> None:
        if int(node.lock_ref) <= 0:
            raise RuntimeError("prefix lock underflow")
        node.lock_ref -= 1

    def materialize_prefix_pages(
        pages: Any, boundary: int, resident_count: int
    ) -> torch.Tensor:
        assert len(tuple(pages)) == resident_count
        return torch.arange(
            PAGE_TOKENS, PAGE_TOKENS + boundary, dtype=torch.int64
        )

    def commit_release_node(
        candidate: Any, node: Any, *, provisional: bool
    ) -> None:
        assert preflight_release_node(candidate, provisional=provisional) is node
        if node is None:
            return
        dec_lock_ref(node)
        delattr(
            candidate,
            (
                "_orbitkv_provisional_prefix_lock"
                if provisional
                else "_orbitkv_prefix_lock_held"
            ),
        )

    tree_cache._preflight_release_node = preflight_release_node
    tree_cache._commit_release_node = commit_release_node
    tree_cache.dec_lock_ref = dec_lock_ref
    tree_cache._materialize_prefix_pages = materialize_prefix_pages
    runtime.materialization = tree_cache._session_materialization_callback

    def register_pending(candidate: Any) -> Any:
        candidate_key = ("str", candidate.rid)
        entry = session_cache._PendingRequestEntry(
            candidate,
            candidate_key,
            candidate._orbitkv_engine_request_id,
        )
        tree_cache._session_pending_requests[candidate_key] = entry
        return entry

    def promote_pending(candidate: Any) -> Any:
        candidate_key = ("str", candidate.rid)
        pending = tree_cache._session_pending_requests[candidate_key]
        binding = runtime.binding_for(candidate_key)
        assert pending.req is candidate
        assert pending.request_id == binding.request_id
        assert binding.request_row == candidate.req_pool_idx
        events.append(("promote", candidate_key, candidate.req_pool_idx))
        promoted = session_cache._RequestEntry(
            candidate, candidate_key, binding.request_id, candidate.req_pool_idx
        )
        tree_cache._session_requests[candidate_key] = promoted
        del tree_cache._session_pending_requests[candidate_key]
        return promoted

    def cancel_pending(candidate: Any) -> None:
        candidate_key = ("str", candidate.rid)
        pending = tree_cache._session_pending_requests[candidate_key]
        binding = runtime.binding_for(candidate_key)
        assert pending.req is candidate
        assert pending.request_id == binding.request_id
        assert binding.request_row is None
        plan = runtime.prepare_release((candidate_key,))
        runtime.confirm_release(plan)
        del tree_cache._session_pending_requests[candidate_key]
        delattr(candidate, "_orbitkv_request_key")
        delattr(candidate, "_orbitkv_engine_request_id")

    tree_cache._test_register_pending = register_pending
    def promote_pending_collective(candidates: Any) -> tuple[Any, ...]:
        return tuple(promote_pending(candidate) for candidate in candidates)

    tree_cache._promote_pending_session_request = promote_pending
    tree_cache._promote_pending_session_requests = promote_pending_collective
    tree_cache._cancel_pending_session_request = cancel_pending
    if previous:
        _install_shared_prefix(
            req,
            previous,
            provisional=False,
            locations=pool.req_to_token[1, :previous],
        )
        tree_cache._session_requests[key] = session_cache._RequestEntry(
            req, key, request_id, 1
        )
    else:
        view = runtime.acquire_unbound((key,))[0]
        req._orbitkv_request_key = key
        req._orbitkv_engine_request_id = view.request_id
        tree_cache._test_register_pending(req)
    target = previous + (1 if previous else 4)
    batch_boundary = previous if previous else target
    return SimpleNamespace(
        reqs=[req],
        req_to_token_pool=pool,
        tree_cache=tree_cache,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        enable_overlap=False,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        is_dllm=lambda: False,
        maybe_evict_swa=lambda: None,
        prefix_lens=[previous],
        extend_lens=[target - previous],
        extend_num_tokens=target - previous,
        seq_lens_cpu=torch.tensor([batch_boundary], dtype=torch.int64),
        seq_lens=torch.tensor([batch_boundary], dtype=torch.int64),
        req_pool_indices=torch.tensor([1], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
        device=torch.device("cpu"),
        _test_runtime=runtime,
    )


def _install(monkeypatch: pytest.MonkeyPatch, *, previous: int, fail_write=False):
    events: list[Any] = []
    config = _config()
    runtime = _Runtime(events)
    allocator = _Allocator(events)
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(8, 64, 64),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator
    batch = _batch(
        events, allocator, runtime, previous=previous, fail_write=fail_write
    )

    import sglang.srt.mem_cache.allocation as allocation

    def alloc_req_slots(_pool: Any, reqs: Any, _tree: Any) -> list[int]:
        used = {int(req.req_pool_idx) for req in reqs if req.req_pool_idx is not None}
        candidate = 1
        for req in reqs:
            if req.req_pool_idx is None:
                while candidate in used:
                    candidate += 1
                req.req_pool_idx = candidate
                used.add(candidate)
        return [int(req.req_pool_idx) for req in reqs]

    def write_cache_indices(
        out: Any,
        _rows_device: Any,
        rows_cpu: Any,
        _prefix_device: Any,
        prefix_cpu: Any,
        _targets_device: Any,
        targets_cpu: Any,
        _extend_device: Any,
        _extend_cpu: Any,
        _prefix_tensors: Any,
        pool: Any,
    ) -> None:
        events.append("mirror")
        offset = 0
        for row, begin, end in zip(
            rows_cpu, prefix_cpu, targets_cpu, strict=True
        ):
            count = int(end - begin)
            pool.req_to_token[int(row), int(begin) : int(end)] = out[
                offset : offset + count
            ].to(torch.int32)
            offset += count

    monkeypatch.setattr(allocation, "alloc_req_slots", alloc_req_slots)
    monkeypatch.setattr(allocation, "write_cache_indices", write_cache_indices)
    device_module = _DeviceModule()
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    batch._test_device_module = device_module
    return runtime, batch, events


def _add_new_request(batch: Any, rid: Any = "request-2") -> Any:
    existing = batch.reqs[0]
    existing.prefix_indices = batch.req_to_token_pool.req_to_token[
        int(existing.req_pool_idx), :4
    ].to(dtype=torch.int64, copy=True)
    req = SimpleNamespace(
        rid=rid,
        req_pool_idx=None,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        kv=None,
    )
    key = ("str", rid)
    view = batch._test_runtime.acquire_unbound((key,))[0]
    req._orbitkv_request_key = key
    req._orbitkv_engine_request_id = view.request_id
    batch.tree_cache._test_register_pending(req)
    batch.reqs.append(req)
    batch.prefix_lens = [4, 0]
    batch.extend_lens = [1, 4]
    batch.extend_num_tokens = 5
    batch.seq_lens_cpu = torch.tensor([5, 4], dtype=torch.int64)
    batch.seq_lens = torch.tensor([5, 4], dtype=torch.int64)
    batch.req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    batch.req_pool_indices_cpu = torch.tensor([1, 2], dtype=torch.int64)
    return req


def _make_pending_prefix_hit(batch: Any, boundary: int = 4) -> Any:
    req = batch.reqs[0]
    key = req._orbitkv_request_key
    request_id = req._orbitkv_engine_request_id
    node = _install_shared_prefix(req, boundary, provisional=True)
    prefix_id = EnginePrefixId(7, 100 + int(request_id))
    req._orbitkv_engine_prefix_id = prefix_id
    resident_count = node.resident_count
    pages = tuple(
        SnapshotPage(
            _page(1 + ordinal, 1),
            ordinal,
            0,
            0,
            1 + ordinal,
            0,
            1,
            PAGE_TOKENS,
            0,
            PAGE_TOKENS,
        )
        for ordinal in range(resident_count)
    )
    control_id = EngineControlId(7, 100 + int(request_id))
    plan = EngineMaterializationPlan(
        control_id,
        (
            EngineMaterializedRequest(
                request_id, 2, boundary, resident_count, pages
            ),
        ),
    )
    batch._test_runtime.pending_controls[control_id] = (key, plan)
    session_cache.register_pending_shared_prefix(
        batch.tree_cache,
        req=req,
        prefix_id=prefix_id,
        semantic=req._orbitkv_prefix_semantic,
        node=node,
        boundary=boundary,
        control_id=control_id,
        plan=plan,
    )
    target = boundary + 1
    batch.prefix_lens = [boundary]
    batch.extend_lens = [1]
    batch.extend_num_tokens = 1
    batch.seq_lens_cpu = torch.tensor([target], dtype=torch.int64)
    batch.seq_lens = torch.tensor([target], dtype=torch.int64)
    return req


def _make_pending_prefix_hit_for(
    batch: Any, req: Any, boundary: int
) -> Any:
    node = _install_shared_prefix(req, boundary, provisional=True)
    key = req._orbitkv_request_key
    request_id = req._orbitkv_engine_request_id
    prefix_id = EnginePrefixId(7, 200 + int(request_id))
    req._orbitkv_engine_prefix_id = prefix_id
    page = SnapshotPage(
        _page(1, 1), 0, 0, 0, 1, 0, 1, PAGE_TOKENS, 0, PAGE_TOKENS
    )
    control_id = EngineControlId(7, 200 + int(request_id))
    plan = EngineMaterializationPlan(
        control_id,
        (EngineMaterializedRequest(request_id, 2, boundary, 1, (page,)),),
    )
    batch._test_runtime.pending_controls[control_id] = (key, plan)
    session_cache.register_pending_shared_prefix(
        batch.tree_cache, req=req, prefix_id=prefix_id,
        semantic=req._orbitkv_prefix_semantic, node=node, boundary=boundary,
        control_id=control_id, plan=plan,
    )
    return node


def _set_pending_view_boundary(
    runtime: _Runtime, req: Any, boundary: int
) -> None:
    key = req._orbitkv_request_key
    runtime.views[key] = EngineRequestView(
        req._orbitkv_engine_request_id,
        1,
        boundary,
        (boundary + PAGE_TOKENS - 1) // PAGE_TOKENS,
    )


def _event_count(events: list[Any], name: str) -> int:
    return sum(
        isinstance(item, tuple) and item and item[0] == name
        for item in events
    )


def _forward_context(batch: Any) -> Any:
    return getattr(batch, FORWARD_CONTEXT_ATTR)


def _assert_context_cleared(batch: Any, context: Any | None = None) -> None:
    assert not hasattr(batch, FORWARD_CONTEXT_ATTR)
    if context is not None:
        assert context._batch is None


def test_first_append_uses_session_plan_and_commits_only_after_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)

    def lower(*_args: Any) -> dict[int, torch.Tensor]:
        events.append("tensor")
        return {0: torch.arange(16, 20, dtype=torch.int64)}

    monkeypatch.setattr(lowering, "_lower_all_extend", lower)

    out, _rows_device, _rows_cpu = lowering._alloc_for_extend(batch)

    req = batch.reqs[0]
    assert torch.equal(out, torch.arange(16, 20, dtype=torch.int64))
    assert events.index("tensor") < events.index("submit") < events.index("mirror")
    assert int(req._orbitkv_engine_request_id) == 1
    assert req._orbitkv_request_key == ("str", "request-1")
    assert not hasattr(req, "_orbitkv_request_lease")
    assert req.kv.kv_allocated_len == 4
    assert torch.equal(
        batch.req_to_token_pool.req_to_token[1, :4],
        torch.arange(16, 20, dtype=torch.int32),
    )
    assert isinstance(batch._orbitkv_session_ticket, EngineBatchTicket)
    assert not hasattr(batch, "_orbitkv_batch")
    assert runtime.submitted[0].steps[0].bind_receipts[0].page == _page(1, 1)
    key = ("str", req.rid)
    assert runtime.binding_for(key).request_row == 1
    assert batch.tree_cache._session_pending_requests == {}
    assert batch.tree_cache._session_requests[key].req is req
    assert _event_count(events, "acquire-unbound") == 1
    assert _event_count(events, "acquire") == 0


def test_existing_decode_executes_exact_cow_before_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)

    def lower(*_args: Any) -> dict[int, torch.Tensor]:
        events.append("tensor")
        return {0: torch.tensor([36], dtype=torch.int64)}

    monkeypatch.setattr(lowering, "_lower_all_decode", lower)

    out = lowering._alloc_for_decode(batch, 1)

    assert out.tolist() == [36]
    assert not any(isinstance(item, tuple) and item[0] == "acquire" for item in events)
    copy_index = next(
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "copy"
    )
    assert events.index("tensor") < copy_index < events.index("submit")
    assert events.index("submit") < events.index("mirror")
    assert batch.req_to_token_pool.req_to_token[1, :5].tolist() == [
        32, 33, 34, 35, 36
    ]
    evidence = runtime.submitted[0].steps[0]
    assert tuple(item.page for item in evidence.bind_receipts) == (_page(2, 2),)
    assert len(evidence.copy_receipts) == 1
    assert evidence.copy_receipts[0].ordered_before_writes is True
    assert batch.reqs[0].kv.kv_allocated_len == 5
    assert _event_count(events, "acquire-unbound") == 0
    assert _event_count(events, "acquire") == 0


def test_shared_prefix_authority_is_forwarded_to_cow_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    req = _make_pending_prefix_hit(batch)
    captured: list[tuple[bool, ...]] = []
    original = lowering._preflight_cow_mirrors

    def preflight(
        actual_batch: Any, plans: Any, flags: Any, context: Any
    ) -> Any:
        values = tuple(flags)
        captured.append(values)
        return original(actual_batch, plans, values, context)

    monkeypatch.setattr(lowering, "_preflight_cow_mirrors", preflight)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: torch.tensor([36], dtype=torch.int64)},
    )

    out, _rows_device, _rows_cpu = lowering._alloc_for_extend(batch)

    key = ("str", req.rid)
    assert out.tolist() == [36]
    assert captured == [(True,)]
    assert runtime.binding_for(key).request_row == 1
    assert batch.tree_cache._session_pending_requests == {}
    assert batch.tree_cache._session_requests[key].req is req
    assert _event_count(events, "acquire-unbound") == 1
    assert _event_count(events, "acquire") == 0


def test_fresh_pending_request_has_no_shared_prefix_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    captured: list[tuple[bool, ...]] = []
    original = lowering._preflight_cow_mirrors

    def preflight(
        actual_batch: Any, plans: Any, flags: Any, context: Any
    ) -> Any:
        values = tuple(flags)
        captured.append(values)
        return original(actual_batch, plans, values, context)

    monkeypatch.setattr(lowering, "_preflight_cow_mirrors", preflight)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: torch.arange(16, 20, dtype=torch.int64)},
    )

    lowering._alloc_for_extend(batch)

    assert captured == [(False,)]
    assert _event_count(events, "acquire-unbound") == 1
    assert _event_count(events, "acquire") == 0
    assert runtime.failure_reason is None


def test_pure_sliding_session_lowering_never_writes_full_to_swa_lut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    state._CONFIG = _pure_sliding_config()
    batch.tree_cache._no_prefix = True
    entered: list[Any] = []
    real_session_lowering = session_lowering.alloc_for_extend

    def enter_session_lowering(actual_batch: Any) -> tuple[Any, Any, Any]:
        entered.append(actual_batch)
        return real_session_lowering(actual_batch)

    monkeypatch.setattr(
        session_lowering, "alloc_for_extend", enter_session_lowering
    )
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: torch.arange(16, 20, dtype=torch.int64)},
    )
    monkeypatch.setattr(
        lowering,
        "_write_hybrid_lut",
        lambda *_args: pytest.fail("pure Sliding requested a Full-to-SWA LUT"),
    )
    monkeypatch.setattr(
        lowering,
        "_validate_joint_hybrid_tails",
        lambda *_args: pytest.fail("pure Sliding used Hybrid tail validation"),
    )

    out, _rows_device, _rows_cpu = lowering._alloc_for_extend(batch)

    assert state._uses_runtime_session()
    assert entered == [batch]
    assert out.tolist() == [16, 17, 18, 19]
    assert events.index("submit") < events.index("mirror")
    assert runtime.failure_reason is None


def test_full_latent_profile_is_request_private_and_accepted_for_lowering() -> None:
    config = _latent_config()
    state._install_test_state(config=config)
    batch = SimpleNamespace(tree_cache=SimpleNamespace(_no_prefix=True))

    assert tuple(
        (item.class_id, item.retention, item.storage) for item in config.classes
    ) == ((0, "full", "latent_kv"),)
    assert state._uses_runtime_session()
    assert state._requires_disabled_radix_cache()

    session_lowering._validate_profile(batch)


def _malform_shared_prefix(req: Any, case: str) -> None:
    node = req._orbitkv_prefix_node
    semantic = req._orbitkv_prefix_semantic
    if case == "request-key":
        req._orbitkv_request_key = ("str", "foreign")
    elif case == "request-id":
        req._orbitkv_engine_request_id = EngineRequestId(0)
    elif case == "legacy-lease":
        req._orbitkv_request_lease = object()
    elif case == "prefix-length":
        req.prefix_indices = req.prefix_indices[:-1]
    elif case == "node":
        req._orbitkv_prefix_node = None
    elif case == "last-node":
        req.last_node = SimpleNamespace()
    elif case == "semantic":
        req._orbitkv_prefix_semantic = None
    elif case == "semantic-boundary":
        req._orbitkv_prefix_semantic = PrefixSemanticKey(
            semantic.namespace, semantic.digest, semantic.boundary + 1
        )
    elif case == "semantic-digest":
        req._orbitkv_prefix_semantic = PrefixSemanticKey(
            semantic.namespace, b"x" * 32, semantic.boundary
        )
    elif case == "node-boundary":
        node.boundary += 1
    elif case == "provisional-lock":
        req._orbitkv_provisional_prefix_lock = False
    elif case == "held-lock":
        req._orbitkv_prefix_lock_held = True
    else:  # pragma: no cover - test table owns the values
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "request-key",
        "request-id",
        "legacy-lease",
        "prefix-length",
        "node",
        "last-node",
        "semantic",
        "semantic-boundary",
        "semantic-digest",
        "node-boundary",
        "provisional-lock",
        "held-lock",
    ),
)
def test_malformed_shared_prefix_authority_rejected_before_native_prepare(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    req = _make_pending_prefix_hit(batch)
    _malform_shared_prefix(req, case)
    if case == "prefix-length":
        monkeypatch.setattr(
            session_lowering,
            "_preflight_extend_batch",
            lambda _batch: ((4,), (1,), (5,)),
        )
    acquire_count = _event_count(events, "acquire-unbound")
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: pytest.fail("tensor lowering must not start"),
    )

    expected = (
        "shared Prefix pending registry"
        if case == "request-id"
        else "shared Prefix authority"
    )
    with pytest.raises((RuntimeError, FailStopped), match=expected):
        lowering._alloc_for_extend(batch)

    assert _event_count(events, "prepare") == 0
    assert _event_count(events, "acquire-unbound") == acquire_count == 1
    assert _event_count(events, "acquire") == 0
    assert runtime.submitted == []
    assert runtime.quarantined_prepared == []


def _malform_zero_prefix(req: Any, case: str) -> None:
    if case == "prefix":
        req.prefix_indices = torch.tensor([16], dtype=torch.int64)
        return
    node = SimpleNamespace(
        boundary=0,
        digest=b"d" * 32,
        parent=object(),
        prefix=object(),
        evicted=False,
        lock_ref=1,
    )
    if case == "node":
        req._orbitkv_prefix_node = node
    elif case == "semantic":
        req._orbitkv_prefix_semantic = PrefixSemanticKey(
            b"n" * 32, node.digest, 0
        )
    elif case == "last-node":
        req.last_node = node
    elif case == "provisional-lock":
        req._orbitkv_provisional_prefix_lock = True
    elif case == "false-provisional-lock":
        req._orbitkv_provisional_prefix_lock = False
    elif case == "held-lock":
        req._orbitkv_prefix_lock_held = True
    elif case == "false-held-lock":
        req._orbitkv_prefix_lock_held = False
    else:  # pragma: no cover - test table owns the values
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "prefix",
        "node",
        "semantic",
        "last-node",
        "provisional-lock",
        "false-provisional-lock",
        "held-lock",
        "false-held-lock",
    ),
)
def test_zero_prefix_rejects_partial_shared_authority_before_prepare(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    _malform_zero_prefix(batch.reqs[0], case)
    if case == "prefix":
        monkeypatch.setattr(
            session_lowering,
            "_preflight_extend_batch",
            lambda _batch: ((0,), (4,), (4,)),
        )
    acquire_count = _event_count(events, "acquire-unbound")

    with pytest.raises(RuntimeError, match="shared Prefix authority"):
        lowering._alloc_for_extend(batch)

    assert _event_count(events, "prepare") == 0
    assert _event_count(events, "acquire-unbound") == acquire_count == 1
    assert _event_count(events, "acquire") == 0
    assert runtime.submitted == []


def test_mixed_batch_binds_only_pending_row_and_promotes_before_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    existing = batch.reqs[0]
    pending = _add_new_request(batch)
    captured: list[tuple[bool, ...]] = []
    original = lowering._preflight_cow_mirrors

    def preflight(
        actual_batch: Any, plans: Any, flags: Any, context: Any
    ) -> Any:
        values = tuple(flags)
        captured.append(values)
        return original(actual_batch, plans, values, context)

    monkeypatch.setattr(lowering, "_preflight_cow_mirrors", preflight)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {
            0: torch.tensor([36, 48, 49, 50, 51], dtype=torch.int64)
        },
    )

    out, _rows_device, _rows_cpu = lowering._alloc_for_extend(batch)

    existing_key = ("str", existing.rid)
    pending_key = ("str", pending.rid)
    bind = ("bind-rows", ((pending_key, 2),))
    assert out.tolist() == [36, 48, 49, 50, 51]
    assert captured == [(True, False)]
    assert bind in events
    assert events.index(bind) < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "promote"
    ) < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "prepare"
    )
    assert runtime.binding_for(existing_key).request_row == 1
    assert runtime.binding_for(pending_key).request_row == 2
    assert batch.tree_cache._session_pending_requests == {}
    assert set(batch.tree_cache._session_requests) == {existing_key, pending_key}
    assert _event_count(events, "acquire-unbound") == 1
    assert _event_count(events, "acquire") == 0


def test_mixed_pending_hit_and_miss_bind_confirm_activate_then_promote_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    hit = batch.reqs[0]
    miss = SimpleNamespace(
        rid="request-miss", req_pool_idx=None,
        prefix_indices=torch.empty((0,), dtype=torch.int64), kv=None,
    )
    miss_key = ("str", miss.rid)
    miss_view = runtime.acquire_unbound((miss_key,))[0]
    miss._orbitkv_request_key = miss_key
    miss._orbitkv_engine_request_id = miss_view.request_id
    batch.tree_cache._test_register_pending(miss)
    batch.reqs.append(miss)
    node = _make_pending_prefix_hit_for(batch, hit, PAGE_TOKENS)
    batch.reqs = [hit, miss]
    batch.prefix_lens = [PAGE_TOKENS, 0]
    batch.extend_lens = [1, 4]
    batch.extend_num_tokens = 5
    batch.seq_lens_cpu = torch.tensor([PAGE_TOKENS + 1, 4], dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.clone()
    batch.req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    observed: list[Any] = []
    original_confirm = runtime.confirm_control

    def confirm(control_id: EngineControlId) -> EngineControlOutcome:
        hit_key = hit._orbitkv_request_key
        observed.append((
            "before-confirm",
            runtime.view_for(hit_key).boundary,
            bool(hit._orbitkv_provisional_prefix_lock),
            hit_key in batch.tree_cache._session_pending_shared_prefix,
            set(batch.tree_cache._session_pending_requests),
        ))
        return original_confirm(control_id)

    monkeypatch.setattr(runtime, "confirm_control", confirm)
    monkeypatch.setattr(
        lowering, "_lower_all_extend",
        lambda *_args: {0: torch.tensor([32, 48, 49, 50, 51])},
    )

    lowering._alloc_for_extend(batch)

    hit_key = hit._orbitkv_request_key
    miss_key = miss._orbitkv_request_key
    bind = ("bind-rows", ((hit_key, 1), (miss_key, 2)))
    confirm_event = next(
        item for item in events
        if isinstance(item, tuple) and item[0] == "confirm-control"
    )
    promote_events = [
        item for item in events
        if isinstance(item, tuple) and item[0] == "promote"
    ]
    prepare_index = next(
        index for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "prepare"
    )
    assert observed == [(
        "before-confirm", 0, True, True, {hit_key, miss_key}
    )]
    assert events.index(bind) < events.index(confirm_event)
    assert events.index(confirm_event) < events.index(promote_events[0])
    assert events.index(promote_events[-1]) < prepare_index
    assert promote_events == [
        ("promote", hit_key, 1), ("promote", miss_key, 2)
    ]
    assert batch.tree_cache._session_pending_requests == {}
    assert set(batch.tree_cache._session_requests) == {hit_key, miss_key}
    assert batch.tree_cache._session_pending_shared_prefix == {}
    assert batch.tree_cache._session_active_shared_prefix[hit_key].req is hit
    assert hit._orbitkv_provisional_prefix_lock is False
    assert hit._orbitkv_prefix_lock_held is True
    assert node.lock_ref == 1


def test_bind_base_exception_fail_stops_without_duplicate_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    req = batch.reqs[0]
    key = req._orbitkv_request_key
    acquire_count = _event_count(events, "acquire-unbound")
    attempts = 0

    def bind(_assignments: Any) -> None:
        nonlocal attempts
        attempts += 1
        events.append(("bind-rows-failed", key))
        raise _TestBaseException("injected bind interruption")

    monkeypatch.setattr(runtime, "bind_request_rows", bind)

    with pytest.raises(FailStopped, match="binding became uncertain") as stopped:
        lowering._alloc_for_extend(batch)

    assert isinstance(stopped.value.__cause__, _TestBaseException)
    assert attempts == 1
    assert runtime.failure_reason is not None
    assert runtime.binding_for(key).request_row is None
    assert req.req_pool_idx == 1
    assert batch.req_to_token_pool.freed == []
    assert batch.tree_cache._session_pending_requests == {}
    assert batch.tree_cache._session_requests == {}
    assert _event_count(events, "prepare") == 0
    assert _event_count(events, "acquire-unbound") == acquire_count == 1
    assert _event_count(events, "acquire") == 0


def test_bind_manager_error_cancels_pending_identity_and_frees_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    req = batch.reqs[0]
    acquire_count = _event_count(events, "acquire-unbound")
    attempts = 0

    def bind(_assignments: Any) -> None:
        nonlocal attempts
        attempts += 1
        raise ManagerError("injected safe bind rejection")

    monkeypatch.setattr(runtime, "bind_request_rows", bind)

    with pytest.raises(ManagerError, match="safe bind rejection"):
        lowering._alloc_for_extend(batch)

    assert attempts == 1
    assert runtime.failure_reason is None
    assert runtime.bindings == {}
    assert runtime.views == {}
    assert req.req_pool_idx is None
    assert batch.req_to_token_pool.freed == [req.rid]
    assert batch.tree_cache._session_pending_requests == {}
    assert batch.tree_cache._session_requests == {}
    assert _event_count(events, "prepare") == 0
    assert _event_count(events, "acquire-unbound") == acquire_count == 1
    assert _event_count(events, "acquire") == 0


def test_promotion_base_exception_fail_stops_without_duplicate_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    req = batch.reqs[0]
    key = req._orbitkv_request_key
    acquire_count = _event_count(events, "acquire-unbound")
    attempts = 0

    def promote(_reqs: Any) -> None:
        nonlocal attempts
        attempts += 1
        events.append(("promote-failed", key))
        raise _TestBaseException("injected promotion interruption")

    monkeypatch.setattr(
        batch.tree_cache, "_promote_pending_session_requests", promote
    )

    with pytest.raises(FailStopped, match="attach handoff") as stopped:
        lowering._alloc_for_extend(batch)

    assert isinstance(stopped.value.__cause__, _TestBaseException)
    assert attempts == 1
    assert runtime.failure_reason is not None
    assert runtime.binding_for(key).request_row == 1
    assert req.req_pool_idx == 1
    assert batch.req_to_token_pool.freed == []
    assert batch.tree_cache._session_pending_requests[key].req is req
    assert batch.tree_cache._session_requests == {}
    assert _event_count(events, "bind-rows") == 1
    assert _event_count(events, "prepare") == 0
    assert _event_count(events, "acquire-unbound") == acquire_count == 1
    assert _event_count(events, "acquire") == 0


@pytest.mark.parametrize(
    ("previous", "helper_name", "expected"),
    (
        (0, "_lower_all_extend", torch.arange(16, 20, dtype=torch.int64)),
        (4, "_lower_all_decode", torch.tensor([36], dtype=torch.int64)),
    ),
)
def test_forward_context_is_captured_before_first_device_work_and_threaded_exactly(
    monkeypatch: pytest.MonkeyPatch,
    previous: int,
    helper_name: str,
    expected: torch.Tensor,
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=previous)
    device_module = batch._test_device_module
    captured: list[Any] = []
    issued: list[EngineBatchTicket] = []
    observed_unsubmitted_device_work = False
    original_to = torch.Tensor.to

    def observe_to(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        nonlocal observed_unsubmitted_device_work
        if args and args[0] == batch.device:
            context = _forward_context(batch)
            assert context.device_module is device_module
            assert context.device == batch.device
            assert context.stream is device_module.stream
            if not issued:
                assert context.ticket is None
                assert not hasattr(batch, "_orbitkv_session_ticket")
                observed_unsubmitted_device_work = True
            captured.append(context)
        return original_to(tensor, *args, **kwargs)

    def lower(*args: Any) -> dict[int, torch.Tensor]:
        context = args[-1]
        assert context is _forward_context(batch)
        assert context is captured[0]
        return {0: expected}

    original_cow = lowering._execute_cow_copies

    def execute_cow(
        actual_batch: Any, plans: Any, context: Any
    ) -> tuple[int, int, int]:
        assert actual_batch is batch
        assert context is captured[0]
        return original_cow(actual_batch, plans, context)

    original_submit = runtime.submit

    def submit(evidence: ExecutionEvidence) -> EngineBatchTicket:
        context = _forward_context(batch)
        assert context.ticket is None
        assert not hasattr(batch, "_orbitkv_session_ticket")
        ticket = original_submit(evidence)
        issued.append(ticket)
        return ticket

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    monkeypatch.setattr(lowering, helper_name, lower)
    monkeypatch.setattr(lowering, "_execute_cow_copies", execute_cow)
    monkeypatch.setattr(runtime, "submit", submit)

    result = (
        lowering._alloc_for_extend(batch)[0]
        if previous == 0
        else lowering._alloc_for_decode(batch, 1)
    )

    assert torch.equal(result, expected)
    assert captured
    assert observed_unsubmitted_device_work
    context = _forward_context(batch)
    ticket = batch._orbitkv_session_ticket
    assert context is captured[0]
    assert len(issued) == 1
    assert context.ticket is ticket is issued[0]
    assert ticket.batch_id == runtime.submitted[0].batch_id


@pytest.mark.parametrize(("previous", "phase"), ((4, "prepared"), (0, "submitted")))
def test_stream_drift_quarantines_the_exact_phase_and_clears_context(
    monkeypatch: pytest.MonkeyPatch, previous: int, phase: str
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=previous)
    device_module = batch._test_device_module
    captured: list[Any] = []

    if phase == "prepared":
        def lower(*_args: Any) -> dict[int, torch.Tensor]:
            captured.append(_forward_context(batch))
            device_module.stream = object()
            return {0: torch.tensor([36], dtype=torch.int64)}

        monkeypatch.setattr(lowering, "_lower_all_decode", lower)
    else:
        original_submit = runtime.submit

        def submit(evidence: ExecutionEvidence) -> EngineBatchTicket:
            captured.append(_forward_context(batch))
            ticket = original_submit(evidence)
            device_module.stream = object()
            return ticket

        monkeypatch.setattr(runtime, "submit", submit)
        monkeypatch.setattr(
            lowering,
            "_lower_all_extend",
            lambda *_args: {0: torch.arange(16, 20, dtype=torch.int64)},
        )

    with pytest.raises(FailStopped, match=f"{phase} execution"):
        (
            lowering._alloc_for_decode(batch, 1)
            if phase == "prepared"
            else lowering._alloc_for_extend(batch)
        )

    assert len(captured) == 1
    _assert_context_cleared(batch, captured[0])
    if phase == "prepared":
        assert runtime.quarantined_prepared
        assert not runtime.submitted
        assert not runtime.quarantined_submitted
        assert not hasattr(batch, "_orbitkv_session_ticket")
    else:
        assert not runtime.quarantined_prepared
        assert runtime.submitted
        assert runtime.quarantined_submitted
        assert runtime.quarantined_submitted[0].batch_id == runtime.submitted[0].batch_id
        assert batch._orbitkv_session_ticket is runtime.quarantined_submitted[0]


@pytest.mark.parametrize("phase", ("prepared", "submitted"))
def test_base_exception_is_contained_at_prepared_and_submitted_phases(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    runtime, batch, _events = _install(
        monkeypatch, previous=4, fail_write=phase == "submitted"
    )
    captured: list[Any] = []

    if phase == "prepared":
        def lower(*_args: Any) -> dict[int, torch.Tensor]:
            captured.append(_forward_context(batch))
            raise _TestBaseException("prepared base failure")

        monkeypatch.setattr(lowering, "_lower_all_decode", lower)
    else:
        monkeypatch.setattr(
            lowering,
            "_lower_all_decode",
            lambda *_args: {0: torch.tensor([36], dtype=torch.int64)},
        )
        original_write = batch.req_to_token_pool.write

        def write(*args: Any, **kwargs: Any) -> None:
            captured.append(_forward_context(batch))
            try:
                original_write(*args, **kwargs)
            except RuntimeError:
                raise _TestBaseException("submitted base failure")

        monkeypatch.setattr(batch.req_to_token_pool, "write", write)

    with pytest.raises(FailStopped, match=f"{phase} execution") as stopped:
        lowering._alloc_for_decode(batch, 1)

    assert isinstance(stopped.value.__cause__, _TestBaseException)
    assert len(captured) == 1
    _assert_context_cleared(batch, captured[0])
    if phase == "prepared":
        assert runtime.quarantined_prepared
        assert not runtime.submitted
        assert not runtime.quarantined_submitted
        assert not hasattr(batch, "_orbitkv_session_ticket")
    else:
        assert not runtime.quarantined_prepared
        assert runtime.submitted
        assert runtime.quarantined_submitted
        assert batch._orbitkv_session_ticket is runtime.quarantined_submitted[0]


@pytest.mark.parametrize("existing_kind", ("stale", "duplicate"))
def test_stale_or_duplicate_context_is_rejected_before_device_work(
    monkeypatch: pytest.MonkeyPatch, existing_kind: str
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=4)
    device_module = batch._test_device_module
    existing = None
    if existing_kind == "duplicate":
        existing = execution_context.ForwardExecutionContext(
            device_module, batch.device, device_module.stream
        )
        existing._batch = batch
    setattr(batch, FORWARD_CONTEXT_ATTR, existing)

    with pytest.raises(RuntimeError, match="already has a forward execution context"):
        lowering._alloc_for_decode(batch, 1)

    assert getattr(batch, FORWARD_CONTEXT_ATTR) is existing
    assert device_module.calls == []
    assert not runtime.submitted
    assert not runtime.quarantined_prepared
    assert not runtime.quarantined_submitted
    assert not hasattr(batch, "_orbitkv_session_ticket")


def test_unsubmitted_context_cleanup_failure_fail_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=0)
    captured: list[Any] = []

    def fail_clear(actual_batch: Any, context: Any) -> None:
        assert actual_batch is batch
        assert context is _forward_context(batch)
        captured.append(context)
        raise _TestBaseException("cleanup base failure")

    monkeypatch.setattr(execution_context, "clear_forward_context", fail_clear)
    monkeypatch.setattr(session_lowering, "clear_forward_context", fail_clear)
    monkeypatch.setattr(
        session_lowering,
        "_request_row_tensors",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("row tensor failed")),
    )

    with pytest.raises(FailStopped, match="context cleanup") as stopped:
        lowering._alloc_for_extend(batch)

    assert isinstance(stopped.value.__cause__, _TestBaseException)
    assert len(captured) == 1
    assert runtime.failure_reason is not None
    assert "cleanup base failure" in runtime.failure_reason
    assert _forward_context(batch) is captured[0]
    assert not runtime.submitted
    assert not runtime.quarantined_prepared
    assert not runtime.quarantined_submitted
    assert not hasattr(batch, "_orbitkv_session_ticket")


def test_unobserved_cow_preflight_failure_aborts_without_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    batch.req_to_token_pool.req_to_token[1, 0] = 99
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: pytest.fail("tensor lowering must not start"),
    )

    with pytest.raises(RuntimeError, match="candidate mirror"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.aborted
    assert not runtime.quarantined_prepared
    assert not runtime.quarantined_submitted
    assert "submit" not in events


def test_tensor_failure_quarantines_prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("tensor failed")),
    )

    with pytest.raises(FailStopped, match="prepared execution"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.quarantined_prepared
    assert not runtime.aborted
    assert not runtime.quarantined_submitted
    assert "submit" not in events


@pytest.mark.parametrize(
    ("locations", "message"),
    (
        ({}, "every compiled KV class"),
        ({0: torch.tensor([[36]], dtype=torch.int64)}, "non-vector"),
        ({0: torch.tensor([36, 37], dtype=torch.int64)}, "cardinality"),
        ({0: torch.tensor([36.0], dtype=torch.float32)}, "non-integer"),
        ({0: torch.empty((1,), dtype=torch.int64, device="meta")}, "foreign device"),
    ),
)
def test_location_metadata_mismatch_quarantines_prepared_before_submit(
    monkeypatch: pytest.MonkeyPatch,
    locations: Any,
    message: str,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    monkeypatch.setattr(lowering, "_lower_all_decode", lambda *_args: locations)

    with pytest.raises(FailStopped, match="prepared execution") as stopped:
        lowering._alloc_for_decode(batch, 1)

    assert message in str(stopped.value.__cause__)
    assert runtime.quarantined_prepared
    assert not runtime.submitted
    assert "copy" not in events
    assert "submit" not in events


def test_diagnostic_location_validation_accepts_exact_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=4)
    monkeypatch.setattr(location_validation, "VALIDATE_PHYSICAL_LOCATIONS", True)
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: {0: torch.tensor([36], dtype=torch.int64)},
    )

    lowering._alloc_for_decode(batch, 1)

    assert len(runtime.submitted) == 1
    assert not runtime.quarantined_prepared


def test_tensor_location_mismatch_quarantines_prepared_before_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    monkeypatch.setattr(location_validation, "VALIDATE_PHYSICAL_LOCATIONS", True)
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: {0: torch.tensor([99], dtype=torch.int64)},
    )

    with pytest.raises(FailStopped, match="prepared execution"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.quarantined_prepared
    assert not runtime.submitted
    assert "copy" not in events
    assert batch.req_to_token_pool.req_to_token[1, :4].tolist() == [
        16, 17, 18, 19
    ]


def test_extend_tensor_location_mismatch_quarantines_before_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    monkeypatch.setattr(location_validation, "VALIDATE_PHYSICAL_LOCATIONS", True)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: torch.tensor([17, 18, 19, 20], dtype=torch.int64)},
    )

    with pytest.raises(FailStopped, match="prepared execution"):
        lowering._alloc_for_extend(batch)

    assert runtime.quarantined_prepared
    assert not runtime.submitted
    assert "copy" not in events
    assert "mirror" not in events
    assert batch.reqs[0].kv is None


@pytest.mark.parametrize("previous", (0, 4))
def test_session_rejects_legacy_request_lease(
    monkeypatch: pytest.MonkeyPatch, previous: int
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=previous)
    batch.reqs[0]._orbitkv_request_lease = object()

    with pytest.raises(
        RuntimeError, match="legacy request (?:lease|identity)"
    ):
        (
            lowering._alloc_for_extend(batch)
            if previous == 0
            else lowering._alloc_for_decode(batch, 1)
        )

    assert not runtime.submitted
    assert not runtime.quarantined_prepared
    assert "submit" not in events
    if previous == 0:
        assert batch.req_to_token_pool.freed == []
        assert runtime.binding_for(("str", "request-1")).request_row is None


def test_post_submit_mirror_failure_quarantines_issued_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(
        monkeypatch, previous=4, fail_write=True
    )
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: {0: torch.tensor([36], dtype=torch.int64)},
    )

    with pytest.raises(FailStopped, match="submitted execution"):
        lowering._alloc_for_decode(batch, 1)

    assert runtime.submitted
    assert not runtime.quarantined_prepared
    assert runtime.quarantined_submitted
    assert runtime.quarantined_submitted[0].batch_id == runtime.submitted[0].batch_id
    assert batch._orbitkv_session_ticket is runtime.quarantined_submitted[0]
    _assert_context_cleared(batch)
    assert batch.reqs[0].kv.kv_allocated_len == 4


def test_session_swa_hook_only_polls_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)

    lowering._manager_maybe_evict_swa(batch)

    assert events[-2:] == [
        "poll",
        ("wait", (("str", "request-1"),)),
    ]


def test_request_row_capacity_failure_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, _events = _install(monkeypatch, previous=0)

    import sglang.srt.mem_cache.allocation as allocation

    monkeypatch.setattr(
        allocation,
        "alloc_req_slots",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("out of rows")),
    )

    with pytest.raises(RuntimeError, match="out of rows"):
        lowering._alloc_for_extend(batch)

    assert runtime.failure_reason is None
    assert runtime.bindings == {}
    assert batch.tree_cache._session_requests == {}
    assert batch.reqs[0].req_pool_idx is None


def test_invalid_request_key_fail_stops_and_preserves_pending_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=0)
    batch.reqs[0].rid = True

    with pytest.raises(
        FailStopped, match="pending session admission identity"
    ) as stopped:
        lowering._alloc_for_extend(batch)

    assert isinstance(stopped.value.__cause__, RuntimeError)
    assert "rid must not be boolean" in str(stopped.value.__cause__)
    assert runtime.failure_reason is not None
    key = ("str", "request-1")
    assert runtime.binding_for(key).request_row is None
    assert batch.tree_cache._session_requests == {}
    assert batch.tree_cache._session_pending_requests == {}
    assert batch.req_to_token_pool.freed == []
    assert batch.reqs[0].req_pool_idx is None
    assert "submit" not in events


@pytest.mark.parametrize("failure", ("row-tensor", "request-key"))
def test_mixed_batch_pre_acquire_failure_only_reclaims_new_row(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    existing = batch.reqs[0]
    new = _add_new_request(batch, True if failure == "request-key" else "request-2")

    import sglang.srt.mem_cache.allocation as allocation

    monkeypatch.setattr(
        allocation,
        "alloc_req_slots",
        lambda _pool, _reqs, _tree: (setattr(new, "req_pool_idx", 2), [1, 2])[1],
    )
    if failure == "row-tensor":
        monkeypatch.setattr(
            session_lowering,
            "_request_row_tensors",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("row tensor failed")),
        )

    if failure == "row-tensor":
        with pytest.raises(RuntimeError, match="row tensor failed"):
            lowering._alloc_for_extend(batch)
    else:
        with pytest.raises(
            FailStopped, match="pending session admission identity"
        ) as stopped:
            lowering._alloc_for_extend(batch)
        assert isinstance(stopped.value.__cause__, RuntimeError)
        assert "rid must not be boolean" in str(stopped.value.__cause__)

    key = ("str", existing.rid)
    assert runtime.binding_for(key).request_row == 1
    assert existing.req_pool_idx == 1
    assert new.req_pool_idx is None
    new_key = ("str", "request-2")
    if failure == "row-tensor":
        assert batch.tree_cache._session_requests[key].req is existing
        assert new_key not in runtime.bindings
        assert new_key not in batch.tree_cache._session_pending_requests
        assert batch.req_to_token_pool.freed == [new.rid]
    else:
        assert batch.tree_cache._session_requests == {}
        original_key = ("str", True)
        assert runtime.binding_for(original_key).request_row is None
        assert batch.tree_cache._session_pending_requests == {}
        assert batch.req_to_token_pool.freed == []
    assert not any(isinstance(item, tuple) and item[0] == "acquire" for item in events)
    if failure == "row-tensor":
        assert runtime.failure_reason is None
    else:
        assert runtime.failure_reason is not None

def test_existing_request_requires_exact_session_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, batch, events = _install(monkeypatch, previous=4)
    batch.tree_cache._session_requests.clear()

    with pytest.raises(RuntimeError, match="not registered"):
        lowering._alloc_for_decode(batch, 1)

    assert not runtime.submitted
    assert not runtime.quarantined_prepared
    assert "submit" not in events
