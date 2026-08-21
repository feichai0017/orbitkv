from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Sequence

from ..runtime import (
    TAIL_COPY_ON_WRITE,
    TAIL_NONE,
    BatchRecord,
    FailStopped,
    LoweringPlan,
    MirrorCleanupBinding,
    sglang_page_id,
)
from . import state as _state
from .mirror_cleanup import _mirror_cleanup_coordinator, _MirrorCleanupContext
from .state import _config, _request_key, _runtime
from .validation import (
    _integer_vector,
    _preflight_decode_batch,
    _preflight_extend_batch,
    _validate_batch,
)


@dataclass(frozen=True, slots=True)
class _ReleaseCandidate:
    req: Any
    tree_cache: Any
    req_to_token_pool: Any
    key: tuple[str, str | bytes | int]
    lease: Any
    row: int
    prefix_indices: Any
    empty_prefix_indices: Any
    prefix_node: Any | None
    is_insert: bool


def _wait_previous_steps(batch: Any) -> None:
    runtime = _runtime()
    runtime.poll()
    keys = []
    for req in batch.reqs:
        if hasattr(req, "_orbitkv_request_lease"):
            key = _request_key(req)
            if getattr(req, "_orbitkv_request_key", None) != key:
                raise RuntimeError("SGLang request rid changed after KV acquisition")
            keys.append(key)
    if keys:
        runtime.wait_batch(tuple(keys))


def _worst_case_prepare_pages(
    previous_boundaries: Sequence[int], targets: Sequence[int]
) -> int:
    page_size = _config().page_tokens
    total = 0
    for previous, target in zip(previous_boundaries, targets, strict=True):
        old_pages = (int(previous) + page_size - 1) // page_size
        new_pages = (int(target) + page_size - 1) // page_size
        # A non-aligned live tail may be shared by a persistent snapshot.  The
        # pre-prepare capacity gate reserves for COW even when it proves private.
        cow_page = int(int(previous) > 0 and int(previous) % page_size != 0)
        total += new_pages - old_pages + cow_page
    return total


def _ensure_prepare_capacity(
    batch: Any,
    previous_boundaries: Sequence[int],
    targets: Sequence[int],
) -> None:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    runtime = _runtime()
    runtime.poll()
    _manager_stats, arena_values = runtime.census()
    required_pages = _worst_case_prepare_pages(previous_boundaries, targets)
    if required_pages <= 0:
        return
    stats = {item.class_id: item for item in arena_values}
    config = _config()
    if set(stats) != set(config.classes_by_id):
        runtime.fail_stop("manager arena census changed before prepare")
        raise FailStopped(runtime.failure_reason or "invalid arena census")
    full = config.full_class
    sliding = config.sliding_class
    full_deficit = (
        0
        if full is None
        else max(0, required_pages - stats[full.class_id].free_pages)
    )
    swa_deficit = (
        0
        if sliding is None
        else max(0, required_pages - stats[sliding.class_id].free_pages)
    )
    if full_deficit == 0 and swa_deficit == 0:
        return
    page_size = config.page_tokens
    result = batch.tree_cache.evict(
        EvictParams(
            num_tokens=full_deficit * page_size,
            swa_num_tokens=swa_deficit * page_size,
        )
    )
    if (
        int(result.num_tokens_evicted) < full_deficit * page_size
        or int(result.swa_num_tokens_evicted) < swa_deficit * page_size
    ):
        raise RuntimeError("OrbitKV has insufficient evictable prefix capacity")
    _manager_stats, arena_values = runtime.census()
    after = {item.class_id: item for item in arena_values}
    if any(
        item.class_id not in after
        or after[item.class_id].free_pages < required_pages
        for item in config.classes
    ):
        runtime.fail_stop("prefix eviction did not publish reusable arena capacity")
        raise FailStopped(runtime.failure_reason or "prefix capacity did not recycle")


def _prepare_batch(
    batch: Any, previous_boundaries: Sequence[int], targets: Sequence[int]
) -> tuple[BatchRecord, tuple[LoweringPlan, ...]]:
    if len(batch.reqs) != len(previous_boundaries) or len(batch.reqs) != len(targets):
        raise RuntimeError("OrbitKV batch boundary cardinality changed")
    runtime = _runtime()
    keys: list[Any] = []
    acquired: list[bool] = []
    batch_record: BatchRecord | None = None
    try:
        for req, previous, target in zip(
            batch.reqs, previous_boundaries, targets, strict=True
        ):
            key = _request_key(req)
            was_acquired = hasattr(req, "_orbitkv_request_lease")
            if runtime.has_request(key) != was_acquired:
                raise RuntimeError("duplicate live request rid or stale OrbitKV lease")
            previous_key = getattr(req, "_orbitkv_request_key", key)
            if previous_key != key:
                raise RuntimeError("SGLang request rid changed after KV acquisition")
            if was_acquired:
                manager_record = runtime.record_for(key)
                if req._orbitkv_request_lease != manager_record.lease:
                    raise RuntimeError("SGLang request carries a foreign OrbitKV lease")
                if int(previous) != manager_record.boundary:
                    raise RuntimeError(
                        "SGLang prefix boundary differs from the manager-published root"
                    )
            elif int(previous) != 0:
                raise RuntimeError("new SGLang request has a nonzero KV boundary")
            keys.append(key)
            acquired.append(not was_acquired)

        batch_record, plans = runtime.prepare_batch(
            tuple(zip(keys, (int(value) for value in targets), strict=True))
        )
        for req, key, pending, _is_new in zip(
            batch.reqs, keys, batch_record.records, acquired, strict=True
        ):
            manager_record = runtime.record_for(key)
            if pending.key != key or pending.prepared.request != manager_record.lease:
                raise RuntimeError(
                    "manager batch order differs from SGLang request order"
                )
            req._orbitkv_request_key = key
            req._orbitkv_request_lease = manager_record.lease
            if manager_record.reclamation_cleanup is None:
                # Prefix-attached requests are acquired before their SGLang row;
                # bind the sole mirror authority as soon as that row exists.
                runtime.bind_reclamation_cleanup(
                    key,
                    MirrorCleanupBinding(
                        coordinator=_mirror_cleanup_coordinator(
                            batch.req_to_token_pool, _state._ALLOCATOR
                        ),
                        context=_MirrorCleanupContext(
                            req=req, request_row=int(req.req_pool_idx)
                        ),
                    ),
                )
    except Exception:
        if runtime.failure_reason is None:
            if batch_record is not None:
                runtime.abort_unobserved(batch_record)
            new_requests = tuple(
                req for req, is_new in zip(batch.reqs, acquired, strict=False) if is_new
            )
            if new_requests:
                runtime.release_batch(tuple(_request_key(req) for req in new_requests))
                for req in new_requests:
                    if hasattr(req, "_orbitkv_request_lease"):
                        delattr(req, "_orbitkv_request_lease")
                    if hasattr(req, "_orbitkv_request_key"):
                        delattr(req, "_orbitkv_request_key")
        raise
    assert batch_record is not None
    return batch_record, plans


def _free_new_req_rows(batch: Any, new_req_slots: Sequence[bool]) -> None:
    """Return only SGLang's non-authoritative request-table rows."""

    values = tuple(
        (req, _request_key(req), int(req.req_pool_idx))
        for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
        if is_new and req.req_pool_idx is not None
    )
    if values:
        try:
            for req, _key, _row in values:
                batch.req_to_token_pool.free(req)
            _runtime().rollback_request_rows(
                tuple((key, row) for _req, key, row in values)
            )
        except Exception as error:
            _runtime().fail_stop(f"ReqToToken row rollback became uncertain: {error}")
            raise FailStopped(
                _runtime().failure_reason or "request-row rollback failed"
            ) from error


def _rollback_admission_locks(batch: Any) -> None:
    """Undo SGLang's run lock while retaining the waiting-prefix lock."""

    entries = tuple(
        (req, getattr(req, "_orbitkv_prefix_node", None))
        for req in batch.reqs
        if bool(getattr(req, "_orbitkv_provisional_prefix_lock", False))
    )
    if any(node is None or int(getattr(node, "lock_ref", 0)) < 2 for _req, node in entries):
        raise RuntimeError("SGLang admission lock cannot be rolled back")
    for _req, node in entries:
        batch.tree_cache.dec_lock_ref(node)


def _preflight_prefix_locks(batch: Any) -> None:
    entries = tuple(
        getattr(req, "_orbitkv_prefix_node", None)
        for req in batch.reqs
        if bool(getattr(req, "_orbitkv_provisional_prefix_lock", False))
    )
    if any(node is None or int(getattr(node, "lock_ref", 0)) < 2 for node in entries):
        raise RuntimeError("SGLang admitted a prefix without its official lock")


def _promote_prefix_locks(batch: Any) -> None:
    """Commit preflighted run locks and drop provisional waiting locks."""

    _preflight_prefix_locks(batch)
    try:
        for req in batch.reqs:
            if not bool(getattr(req, "_orbitkv_provisional_prefix_lock", False)):
                continue
            node = req._orbitkv_prefix_node
            batch.tree_cache.dec_lock_ref(node)
            delattr(req, "_orbitkv_provisional_prefix_lock")
            req._orbitkv_prefix_lock_held = True
    except Exception as error:
        _runtime().fail_stop(f"prefix lock promotion became uncertain: {error}")
        raise FailStopped(
            _runtime().failure_reason or "prefix lock promotion failed"
        ) from error


def _lower_extend_class(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[LoweringPlan],
    class_id: int,
) -> Any:
    import torch
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.srt.utils import next_power_of_2

    class_plans = [plan.by_class[class_id] for plan in plans]
    bs = len(plans)
    prefix_lens = prefix_lens_cpu.to(batch.device, non_blocking=True)
    targets = targets_cpu.to(batch.device, non_blocking=True)
    last_loc = torch.tensor(
        [class_plan.last_location for class_plan in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    page_ids = [
        page for class_plan in class_plans for page in class_plan.exact_new_pages
    ]
    exact_pages = torch.tensor(page_ids, dtype=torch.int64, device=batch.device)
    out_cache_loc = torch.empty(
        (int(extend_num_tokens),), dtype=torch.int64, device=batch.device
    )
    alloc_extend_kernel[(bs,)](
        prefix_lens,
        targets,
        last_loc,
        exact_pages,
        out_cache_loc,
        next_power_of_2(bs),
        _config().page_tokens,
    )
    return out_cache_loc


def _lower_decode_class(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[LoweringPlan],
    class_id: int,
) -> Any:
    import torch
    from sglang.kernels.ops.memory.allocator import alloc_decode_kernel
    from sglang.srt.utils import next_power_of_2

    class_plans = [plan.by_class[class_id] for plan in plans]
    bs = len(plans)
    targets = targets_cpu.to(batch.device, non_blocking=True)
    last_loc = torch.tensor(
        [class_plan.last_location for class_plan in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    page_ids = [
        page for class_plan in class_plans for page in class_plan.exact_new_pages
    ]
    exact_pages = torch.tensor(page_ids, dtype=torch.int64, device=batch.device)
    out_cache_loc = torch.empty((bs,), dtype=torch.int64, device=batch.device)
    alloc_decode_kernel[(bs,)](
        targets,
        last_loc,
        exact_pages,
        out_cache_loc,
        next_power_of_2(bs),
        _config().page_tokens,
    )
    return out_cache_loc


def _lower_all_extend(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[LoweringPlan],
) -> dict[int, Any]:
    return {
        item.class_id: _lower_extend_class(
            batch,
            prefix_lens_cpu,
            targets_cpu,
            extend_num_tokens,
            plans,
            item.class_id,
        )
        for item in _config().classes
    }


def _lower_all_decode(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[LoweringPlan],
) -> dict[int, Any]:
    return {
        item.class_id: _lower_decode_class(batch, targets_cpu, plans, item.class_id)
        for item in _config().classes
    }


def _primary_locations(locations: dict[int, Any]) -> Any:
    class_config = _config().full_class or _config().sliding_class
    if class_config is None or set(locations) != {
        item.class_id for item in _config().classes
    }:
        raise RuntimeError("lowering did not return every compiled KV class")
    return locations[class_config.class_id]


def _write_hybrid_lut(locations: dict[int, Any]) -> None:
    import torch

    full = _config().full_class
    sliding = _config().sliding_class
    if full is None or sliding is None:
        return
    full_locations = locations[full.class_id]
    swa_locations = locations[sliding.class_id]
    if int(full_locations.numel()) != int(swa_locations.numel()):
        raise RuntimeError("Full and SWA lowering cardinalities differ")
    _state._ALLOCATOR.set_full_to_swa_mapping(
        full_locations.to(dtype=torch.int64),
        swa_locations.to(dtype=torch.int64),
    )


def _class_kv_pool(class_id: int) -> Any:
    kvcache = _state._ALLOCATOR.get_kvcache()
    class_config = _config().classes_by_id[class_id]
    if _config().full_class is not None and _config().sliding_class is not None:
        name = "full_kv_pool" if class_config.retention == "full" else "swa_kv_pool"
        pool = getattr(kvcache, name, None)
    else:
        pool = kvcache
    if pool is None or not callable(getattr(pool, "move_kv_cache", None)):
        raise RuntimeError("SGLang KV pool cannot execute an exact COW copy")
    return pool


def _validate_joint_hybrid_tails(plans: Sequence[LoweringPlan]) -> None:
    full = _config().full_class
    sliding = _config().sliding_class
    if full is None or sliding is None:
        return
    for plan in plans:
        full_spec = plan.by_class[full.class_id]
        swa_spec = plan.by_class[sliding.class_id]
        full_action = full_spec.tail_action
        swa_action = swa_spec.tail_action
        full_cow = full_action.kind == TAIL_COPY_ON_WRITE
        swa_cow = swa_action.kind == TAIL_COPY_ON_WRITE
        full_intent = (
            full_spec.copy_intents[0]
            if full_cow and len(full_spec.copy_intents) == 1
            else None
        )
        swa_intent = (
            swa_spec.copy_intents[0]
            if swa_cow and len(swa_spec.copy_intents) == 1
            else None
        )
        if (
            full_action.kind != swa_action.kind
            or full_action.valid_token_count != swa_action.valid_token_count
            or full_action.logical_ordinal != swa_action.logical_ordinal
            or full_cow != swa_cow
            or full_cow
            and (
                full_intent is None
                or swa_intent is None
                or full_intent.token_count != full_action.valid_token_count
                or swa_intent.token_count != swa_action.valid_token_count
                or full_intent.source_token_offset
                != swa_intent.source_token_offset
                or full_intent.destination_token_offset
                != swa_intent.destination_token_offset
            )
        ):
            raise RuntimeError("Hybrid tail transition is not a joint Full/SWA action")


def _intent_locations(intent: Any, *, destination: bool, device: Any) -> Any:
    import torch

    arena = _runtime().arenas_by_class[intent.class_id]
    backend_index = (
        intent.destination_backend_index
        if destination
        else intent.source_backend_index
    )
    offset = (
        intent.destination_token_offset
        if destination
        else intent.source_token_offset
    )
    start = (
        sglang_page_id(backend_index, arena.backend_base_index)
        * _config().page_tokens
        + offset
    )
    return torch.arange(
        start,
        start + intent.token_count,
        dtype=torch.int64,
        device=device,
    )


def _execute_cow_copies(
    batch: Any, plans: Sequence[LoweringPlan]
) -> tuple[int, int, int]:
    """Enqueue every exact manager copy on the current forward stream."""

    import torch

    device_module = torch.get_device_module(batch.device)
    device_module.current_stream(batch.device)
    intent_count = move_calls = copied_tokens = 0
    for class_config in _config().classes:
        sources = []
        destinations = []
        for plan in plans:
            for intent in plan.by_class[class_config.class_id].copy_intents:
                sources.append(
                    _intent_locations(
                        intent, destination=False, device=batch.device
                    )
                )
                destinations.append(
                    _intent_locations(
                        intent, destination=True, device=batch.device
                    )
                )
        if sources:
            _class_kv_pool(class_config.class_id).move_kv_cache(
                torch.cat(destinations), torch.cat(sources)
            )
            class_intents = sum(
                len(plan.by_class[class_config.class_id].copy_intents)
                for plan in plans
            )
            intent_count += class_intents
            move_calls += 1
            copied_tokens += sum(
                intent.token_count
                for plan in plans
                for intent in plan.by_class[class_config.class_id].copy_intents
            )
    return intent_count, move_calls, copied_tokens


def _record_cow_activity(activity: tuple[int, int, int]) -> None:
    intent_count, move_calls, copied_tokens = activity
    if intent_count:
        _state._counter_add("cow_copy_intents", intent_count)
        _state._counter_add("cow_move_calls", move_calls)
        _state._counter_add("cow_copied_tokens", copied_tokens)


def _tail_locations(spec: Any, *, source: bool, device: Any) -> Any | None:
    import torch

    count = int(spec.tail_action.valid_token_count)
    if spec.tail_action.kind == TAIL_NONE or count == 0:
        return None
    if source and spec.copy_intents:
        return _intent_locations(
            spec.copy_intents[0], destination=False, device=device
        )
    end = int(spec.last_location) + 1
    return torch.arange(
        end - count, end, dtype=torch.int64, device=device
    )


@dataclass(frozen=True, slots=True)
class _CowMirrorPlan:
    assignments: tuple[tuple[Any, Any], ...]
    mapping: Any | None
    mapping_assignments: tuple[tuple[Any, Any], ...]


def _preflight_cow_mirrors(
    batch: Any,
    plans: Sequence[LoweringPlan],
    use_prefix_mirror: Sequence[bool],
) -> _CowMirrorPlan:
    """Validate the entire B4 source mirror before copy or manager submit."""

    import torch

    config = _config()
    primary = config.full_class or config.sliding_class
    if primary is None:
        raise RuntimeError("OrbitKV plan has no primary KV class")
    comparisons: list[tuple[Any, Any]] = []
    assignments: list[tuple[Any, Any]] = []
    mapping_assignments: list[tuple[Any, Any]] = []
    mapping = (
        _state._ALLOCATOR.full_to_swa_index_mapping
        if config.full_class is not None and config.sliding_class is not None
        else None
    )
    for req, plan, prefix_authoritative in zip(
        batch.reqs, plans, use_prefix_mirror, strict=True
    ):
        primary_spec = plan.by_class[primary.class_id]
        count = int(primary_spec.tail_action.valid_token_count)
        if count == 0:
            continue
        logical_begin = primary_spec.tail_action.logical_ordinal * config.page_tokens
        logical_end = logical_begin + count
        old_primary = _tail_locations(
            primary_spec, source=True, device=batch.device
        )
        new_primary = _tail_locations(
            primary_spec, source=False, device=batch.device
        )
        assert old_primary is not None and new_primary is not None
        row_view = batch.req_to_token_pool.req_to_token[
            int(req.req_pool_idx), logical_begin:logical_end
        ]
        prefix = getattr(req, "prefix_indices", None)
        if prefix_authoritative:
            if prefix is None or int(prefix.numel()) < logical_end:
                raise RuntimeError("new request lost its attached prefix mirror")
            comparisons.append((prefix[logical_begin:logical_end], old_primary))
        else:
            comparisons.append((row_view.to(dtype=torch.int64), old_primary))
        assignments.append((row_view, new_primary.to(dtype=row_view.dtype)))
        prefix_end = min(logical_end, int(prefix.numel()) if prefix is not None else 0)
        if prefix_end > logical_begin:
            prefix_view = prefix[logical_begin:prefix_end]
            prefix_count = prefix_end - logical_begin
            comparisons.append((prefix_view, old_primary[:prefix_count]))
            assignments.append((prefix_view, new_primary[:prefix_count]))

        if mapping is not None:
            sliding = config.sliding_class
            assert sliding is not None
            sliding_spec = plan.by_class[sliding.class_id]
            if int(sliding_spec.tail_action.valid_token_count) != count:
                raise RuntimeError("Full and SWA tail visibility differs")
            old_sliding = _tail_locations(
                sliding_spec, source=True, device=batch.device
            )
            new_sliding = _tail_locations(
                sliding_spec, source=False, device=batch.device
            )
            assert old_sliding is not None and new_sliding is not None
            comparisons.append((mapping[old_primary], old_sliding))
            mapping_assignments.append((new_primary, new_sliding))

    if comparisons:
        actual = torch.cat(tuple(item[0].to(dtype=torch.int64) for item in comparisons))
        expected = torch.cat(tuple(item[1].to(dtype=torch.int64) for item in comparisons))
        if not torch.equal(actual, expected):
            raise RuntimeError("COW intent disagrees with the SGLang candidate mirror")
    return _CowMirrorPlan(tuple(assignments), mapping, tuple(mapping_assignments))


def _commit_cow_mirrors(plan: _CowMirrorPlan) -> None:
    for target, replacement in plan.assignments:
        target.copy_(replacement)
    if plan.mapping is not None:
        for full_locations, swa_locations in plan.mapping_assignments:
            plan.mapping[full_locations] = swa_locations


def _submit_batch(batch_record: BatchRecord) -> tuple[Any, ...]:
    return _runtime().submit_batch(batch_record)


def _alloc_for_extend(batch: Any) -> tuple[Any, Any, Any]:
    import sglang.srt.mem_cache.allocation as allocation
    import torch
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    _validate_batch(batch)
    prefix_values, extend_values, target_values = _preflight_extend_batch(batch)
    batch.maybe_evict_swa()
    _ensure_prepare_capacity(batch, prefix_values, target_values)
    prefix_tensors = [req.prefix_indices for req in batch.reqs]
    prefix_lens_cpu = torch.tensor(prefix_values, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(extend_values, dtype=torch.int64)
    targets_cpu = torch.tensor(target_values, dtype=torch.int64)
    targets_device = targets_cpu.to(batch.device, non_blocking=True)
    batch.seq_lens = targets_device

    new_req_slots = [req.req_pool_idx is None for req in batch.reqs]
    try:
        req_pool_indices = allocation.alloc_req_slots(
            batch.req_to_token_pool, batch.reqs, batch.tree_cache
        )
        req_pool_values = _integer_vector(
            "allocated req_pool_indices", req_pool_indices, len(batch.reqs)
        )
        row_capacity = int(batch.req_to_token_pool.req_to_token.shape[0])
        if len(set(req_pool_values)) != len(req_pool_values):
            raise RuntimeError("SGLang allocated duplicate request-pool rows")
        if any(value <= 0 or value >= row_capacity for value in req_pool_values):
            raise RuntimeError("SGLang allocated an out-of-range request-pool row")
        if any(
            req.req_pool_idx is None or int(req.req_pool_idx) != value
            for req, value in zip(batch.reqs, req_pool_values, strict=True)
        ):
            raise RuntimeError("SGLang allocated request-pool identity is inconsistent")
    except Exception as error:
        _runtime().fail_stop(f"SGLang request-row allocation became uncertain: {error}")
        raise FailStopped(
            _runtime().failure_reason or "request-row allocation became uncertain"
        ) from error
    _runtime().bind_request_rows(
        tuple(
            (_request_key(req), row, is_new)
            for req, row, is_new in zip(
                batch.reqs, req_pool_values, new_req_slots, strict=True
            )
        )
    )
    batch_record: BatchRecord | None = None
    try:
        _preflight_prefix_locks(batch)
        req_pool_indices_cpu = torch.tensor(req_pool_values, dtype=torch.int64)
        req_pool_indices_device = req_pool_indices_cpu.to(
            batch.device, non_blocking=True
        )
        batch_record, plans = _prepare_batch(
            batch,
            prefix_values,
            target_values,
        )
        _promote_prefix_locks(batch)
    except Exception:
        if _runtime().failure_reason is None:
            try:
                _rollback_admission_locks(batch)
                _free_new_req_rows(batch, new_req_slots)
            except Exception as rollback_error:
                _runtime().fail_stop(
                    f"prefix admission rollback became uncertain: {rollback_error}"
                )
                raise FailStopped(
                    _runtime().failure_reason or "prefix admission rollback failed"
                ) from rollback_error
        raise

    try:
        locations = _lower_all_extend(
            batch,
            prefix_lens_cpu,
            targets_cpu,
            int(batch.extend_num_tokens),
            plans,
        )
        out_cache_loc = _primary_locations(locations)
        assert batch_record is not None
        _validate_joint_hybrid_tails(plans)
        cow_mirror_plan = _preflight_cow_mirrors(
            batch, plans, new_req_slots
        )
        cow_activity = _execute_cow_copies(batch, plans)
        _runtime().mark_lowered(batch_record)
    except Exception as error:
        if batch_record is not None:
            _runtime().lowering_failed(batch_record, error)
        raise FailStopped(
            _runtime().failure_reason or "extend lowering failed"
        ) from error

    _submit_batch(batch_record)
    _record_cow_activity(cow_activity)

    try:
        allocation.write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_cpu.to(batch.device, non_blocking=True),
            prefix_lens_cpu,
            targets_device,
            targets_cpu,
            extend_lens_cpu.to(batch.device, non_blocking=True),
            extend_lens_cpu,
            prefix_tensors,
            batch.req_to_token_pool,
        )
        _write_hybrid_lut(locations)
        _commit_cow_mirrors(cow_mirror_plan)
        for req, target in zip(batch.reqs, targets_cpu.tolist(), strict=True):
            if req.kv is None:
                req.kv = ReqKvInfo(kv_allocated_len=int(target), swa_evicted_seqlen=0)
            else:
                req.kv.kv_allocated_len = int(target)
        batch._orbitkv_batch = batch_record
    except Exception as error:
        _runtime().candidate_mirror_failed(batch_record, error)
        raise FailStopped(
            _runtime().failure_reason or "candidate mirror failed"
        ) from error
    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _alloc_for_decode(batch: Any, token_per_req: int) -> Any:
    _validate_batch(batch)
    if int(token_per_req) != 1:
        raise RuntimeError("OrbitKV supports one decode token per request")
    previous, req_pool_values = _preflight_decode_batch(batch)
    batch.maybe_evict_swa()
    targets = [value + 1 for value in previous]

    import torch

    targets_cpu = torch.tensor(targets, dtype=torch.int64)
    previous_device = torch.tensor(previous, dtype=torch.int64).to(
        batch.device, non_blocking=True
    )
    req_pool_indices_device = torch.tensor(req_pool_values, dtype=torch.int64).to(
        batch.device, non_blocking=True
    )
    batch.seq_lens = previous_device
    batch.req_pool_indices = req_pool_indices_device
    batch_record, plans = _prepare_batch(batch, previous, targets)
    try:
        locations = _lower_all_decode(batch, targets_cpu, plans)
        out_cache_loc = _primary_locations(locations)
        _validate_joint_hybrid_tails(plans)
        cow_mirror_plan = _preflight_cow_mirrors(
            batch, plans, (False,) * len(plans)
        )
        cow_activity = _execute_cow_copies(batch, plans)
        _runtime().mark_lowered(batch_record)
    except Exception as error:
        _runtime().lowering_failed(batch_record, error)
        raise FailStopped(
            _runtime().failure_reason or "decode lowering failed"
        ) from error

    _submit_batch(batch_record)
    _record_cow_activity(cow_activity)

    try:
        if batch.model_config.is_encoder_decoder:
            raise RuntimeError("OrbitKV does not support encoder-decoder models")
        batch.req_to_token_pool.write(
            (batch.req_pool_indices, previous_device),
            out_cache_loc.to(torch.int32),
        )
        _write_hybrid_lut(locations)
        _commit_cow_mirrors(cow_mirror_plan)
        for req in batch.reqs:
            req.kv.kv_allocated_len += 1
        batch._orbitkv_batch = batch_record
    except Exception as error:
        _runtime().candidate_mirror_failed(batch_record, error)
        raise FailStopped(
            _runtime().failure_reason or "decode mirror failed"
        ) from error
    return out_cache_loc


def _manager_maybe_evict_swa(batch: Any) -> None:
    _validate_batch(batch)
    _wait_previous_steps(batch)


def _completion_domain(scheduler: Any) -> int:
    device = getattr(scheduler, "device", None)
    if isinstance(device, str):
        index = None
    else:
        index = getattr(device, "index", None)
    if index is None and isinstance(device, str) and ":" in device:
        suffix = device.rsplit(":", 1)[-1]
        index = int(suffix) if suffix.isdigit() else 0
    return int(index or 0) + 1


def _get_next_batch_to_run(
    original_fn: Callable[..., Any], scheduler: Any, *args: Any, **kwargs: Any
) -> Any:
    try:
        return original_fn(scheduler, *args, **kwargs)
    except Exception as error:
        _runtime().pre_forward_failed(error)


def _run_batch(
    original_fn: Callable[..., Any],
    scheduler: Any,
    batch: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    runtime = _runtime()
    batch_record = getattr(batch, "_orbitkv_batch", None)
    try:
        _validate_batch(batch)
        runtime.poll()
        if batch.reqs and not isinstance(batch_record, BatchRecord):
            raise RuntimeError("OrbitKV forward has no submitted manager step")
        expected_keys = tuple(_request_key(req) for req in batch.reqs)
        if (
            not isinstance(batch_record, BatchRecord)
            or batch_record.keys != expected_keys
        ):
            raise RuntimeError(
                "OrbitKV forward records do not match batch request order"
            )
        runtime.mark_forward(batch_record)
    except Exception as error:
        if isinstance(batch_record, BatchRecord):
            runtime.forward_failed(batch_record, error)
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            runtime.failure_reason or "pre-forward manager state became uncertain"
        ) from error
    try:
        result = original_fn(scheduler, batch, *args, **kwargs)
    except Exception as error:
        runtime.forward_failed(batch_record, error)
        raise FailStopped(runtime.failure_reason or "forward failed") from error
    try:
        launch_stream = scheduler.device_module.current_stream(scheduler.device)
        event = scheduler.device_module.Event()
        event.record(stream=launch_stream)
        runtime.register_event(batch_record, event, _completion_domain(scheduler))
        batch._orbitkv_batch = None
    except Exception as error:
        runtime.event_registration_failed(batch_record, error)
        raise FailStopped(
            runtime.failure_reason or "event registration failed"
        ) from error
    return result


def _release_candidate(
    req: Any, tree_cache: Any, *, is_insert: bool
) -> _ReleaseCandidate | None:
    runtime = _runtime()
    key = _request_key(req)
    if req.req_pool_idx is None:
        carries_lease = hasattr(req, "_orbitkv_request_lease") or hasattr(
            req, "_orbitkv_request_key"
        )
        if req.kv is None and not carries_lease and not runtime.has_request(key):
            return None
        runtime.fail_stop(
            "SGLang dropped a ReqToToken row while OrbitKV request state remained"
        )
        raise FailStopped(runtime.failure_reason or "release identity was lost")
    from .prefix_cache import OrbitKvPrefixCache

    if type(tree_cache) is not OrbitKvPrefixCache:
        raise RuntimeError("OrbitKV release requires its radix-cache backend")
    if bool(tree_cache.supports_swa()) != (_config().sliding_class is not None):
        raise RuntimeError("SGLang release cache type differs from the plan")
    if tree_cache.token_to_kv_pool_allocator is not _state._ALLOCATOR:
        raise RuntimeError("SGLang release references a foreign KV allocator")
    if not hasattr(req, "_orbitkv_request_lease"):
        raise RuntimeError("SGLang tried to release KV without an OrbitKV lease")
    if getattr(req, "_orbitkv_request_key", None) != key:
        raise RuntimeError("SGLang request rid changed before release")

    raw_row = req.req_pool_idx
    if isinstance(raw_row, bool) or not isinstance(raw_row, Integral):
        raise RuntimeError("SGLang release row is not an integer")
    row = int(raw_row)
    if row <= 0:
        raise RuntimeError("SGLang release named the dummy ReqToToken row")
    lease = req._orbitkv_request_lease
    if runtime.record_for(key).lease != lease:
        raise RuntimeError("SGLang release lease differs from manager authority")
    req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
    if req_to_token_pool is None or not callable(
        getattr(req_to_token_pool, "free", None)
    ):
        raise RuntimeError("SGLang release lost its ReqToToken pool")
    prefix_indices = getattr(req, "prefix_indices", None)
    try:
        empty_prefix_indices = prefix_indices[:0]
    except Exception as error:
        raise RuntimeError("SGLang release prefix is not sliceable") from error
    prefix_node = tree_cache._preflight_release_node(req, provisional=False)
    return _ReleaseCandidate(
        req=req,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        key=key,
        lease=lease,
        row=row,
        prefix_indices=prefix_indices,
        empty_prefix_indices=empty_prefix_indices,
        prefix_node=prefix_node,
        is_insert=bool(is_insert),
    )


def _release_group_identity_changed(
    candidate: _ReleaseCandidate, runtime: Any
) -> bool:
    req = candidate.req
    return bool(
        req.req_pool_idx != candidate.row
        or getattr(req, "_orbitkv_request_key", None) != candidate.key
        or getattr(req, "_orbitkv_request_lease", None) != candidate.lease
        or getattr(req, "prefix_indices", None) is not candidate.prefix_indices
        or runtime.record_for(candidate.key).lease != candidate.lease
        or candidate.tree_cache.token_to_kv_pool_allocator is not _state._ALLOCATOR
        or candidate.tree_cache._preflight_release_node(
            req, provisional=False
        )
        is not candidate.prefix_node
    )


def _flush_release_group(candidates: Sequence[_ReleaseCandidate]) -> None:
    values = tuple(candidates)
    if not values:
        raise RuntimeError("OrbitKV release group must be nonempty")
    runtime = _runtime()
    keys = {candidate.key for candidate in values}
    rows = {candidate.row for candidate in values}
    requests = {id(candidate.req) for candidate in values}
    pool = values[0].req_to_token_pool
    if (
        len(keys) != len(values)
        or len(rows) != len(values)
        or len(requests) != len(values)
        or any(candidate.req_to_token_pool is not pool for candidate in values)
    ):
        runtime.fail_stop("SGLang release group contains aliased identities")
        raise FailStopped(runtime.failure_reason or "aliased release group")
    for candidate in values:
        if _release_group_identity_changed(candidate, runtime):
            runtime.fail_stop("SGLang release group identity changed before flush")
            raise FailStopped(runtime.failure_reason or "release group changed")

    try:
        runtime.wait_batch(tuple(candidate.key for candidate in values))
        for candidate in values:
            effective_boundary = candidate.req.effective_kv_committed_len()
            if (
                isinstance(effective_boundary, bool)
                or not isinstance(effective_boundary, Integral)
                or int(effective_boundary)
                != runtime.record_for(candidate.key).boundary
            ):
                raise RuntimeError(
                    "finished request boundary differs from manager authority"
                )
        publications = []
        for candidate in values:
            publication = candidate.tree_cache.publication_for_release(
                candidate.req, is_insert=candidate.is_insert
            )
            publications.append((candidate, publication))
        publishable = (
            all(publication is not None for _candidate, publication in publications)
            and len(
                {
                    publication.semantic
                    for _candidate, publication in publications
                    if publication is not None
                }
            )
            == len(values)
        )
        # All user/model callbacks have now run.  Revalidate the entire group
        # once more immediately before the single native commit so a hostile
        # callback cannot turn a postcommit lock release into the first point
        # where identity corruption is observed.
        if any(
            _release_group_identity_changed(candidate, runtime)
            for candidate in values
        ):
            raise RuntimeError("release group changed during precommit callbacks")
        runtime.bind_request_rows(
            tuple((candidate.key, candidate.row, False) for candidate in values)
        )
        if publishable:
            output = runtime.prefix_publish_release_batch(
                tuple(
                    (candidate.key, publication.semantic)
                    for candidate, publication in publications
                    if publication is not None
                )
            )
            if len(output.outputs) != len(publications):
                raise RuntimeError("publish-release cardinality changed")
            for (candidate, pending), completed in zip(
                publications, output.outputs, strict=True
            ):
                assert pending is not None
                candidate.tree_cache.accept_release_publication(
                    completed.publication, pending.tokens
                )
        else:
            # ABI6 has no heterogeneous publish-or-release transaction.  Keep
            # one official free_group atomic: duplicates or opt-outs sacrifice
            # this insertion instead of splitting the group across commits.
            runtime.release_batch(tuple(candidate.key for candidate in values))
        for candidate in values:
            candidate.tree_cache._commit_release_node(
                candidate.req, candidate.prefix_node, provisional=False
            )
            candidate.req_to_token_pool.free(candidate.req)
        runtime.unbind_request_rows(
            tuple((candidate.key, candidate.row) for candidate in values)
        )
        for candidate in values:
            req = candidate.req
            req.kv = None
            req.prefix_indices = candidate.empty_prefix_indices
            for name in (
                "_orbitkv_request_lease",
                "_orbitkv_request_key",
                "_orbitkv_prefix_node",
                "_orbitkv_prefix_semantic",
                "_orbitkv_provisional_prefix_lock",
                "_orbitkv_prefix_lock_held",
            ):
                if hasattr(req, name):
                    delattr(req, name)
    except Exception as error:
        runtime.fail_stop(f"ReqToToken release group became uncertain: {error}")
        raise FailStopped(runtime.failure_reason or "release group failed") from error


def _release_kv_cache(req: Any, tree_cache: Any, is_insert: bool = True) -> None:
    runtime = _runtime()
    key = _request_key(req)
    if req.req_pool_idx is None and runtime.has_request(key):
        if req.kv is not None:
            runtime.fail_stop("request lost its row but retained KV metadata")
            raise FailStopped(runtime.failure_reason or "release identity was lost")
        prefix_node = tree_cache._preflight_release_node(req, provisional=True)
        # Commit native release first.  Retryable pre-commit failures must
        # leave the local node protected and the request identity unchanged.
        runtime.release_batch((key,))
        try:
            tree_cache._commit_release_node(
                req, prefix_node, provisional=True
            )
        except Exception as error:
            runtime.fail_stop(
                f"released waiting-prefix lock became uncertain: {error}"
            )
            raise FailStopped(
                runtime.failure_reason or "waiting-prefix release failed"
            ) from error
        for name in (
            "_orbitkv_request_lease",
            "_orbitkv_request_key",
            "_orbitkv_prefix_node",
            "_orbitkv_prefix_semantic",
            "_orbitkv_provisional_prefix_lock",
            "_orbitkv_prefix_lock_held",
        ):
            if hasattr(req, name):
                delattr(req, name)
        return
    candidate = _release_candidate(req, tree_cache, is_insert=is_insert)
    if candidate is None:
        return
    allocator = _state._ALLOCATOR
    if getattr(allocator, "_orbitkv_free_group_state", "idle") == "collecting":
        if any(
            existing.key == candidate.key
            or existing.row == candidate.row
            or existing.req is candidate.req
            for existing in allocator.free_group
        ):
            runtime = _runtime()
            runtime.fail_stop("SGLang duplicated a request inside a release group")
            raise FailStopped(runtime.failure_reason or "duplicate release")
        allocator.free_group.append(candidate)
        return
    if getattr(allocator, "_orbitkv_free_group_state", "idle") != "idle":
        runtime = _runtime()
        runtime.fail_stop("SGLang released KV while a group was flushing")
        raise FailStopped(runtime.failure_reason or "release raced group flush")
    _flush_release_group((candidate,))
