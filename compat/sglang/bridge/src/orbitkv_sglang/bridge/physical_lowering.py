"""Tensor lowering and exact physical COW execution for SGLang batches.

The public bridge hooks remain in :mod:`lowering`.  A caller-supplied module
namespace keeps those compatibility hooks (and their test overrides) as the
single dispatch surface without introducing a reverse import cycle.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..execution_plan import StepPlan
from ..runtime import TAIL_COPY_ON_WRITE


def _sglang_page_id(backend_index: int, backend_base_index: int) -> int:
    page_id = backend_index - backend_base_index + 1
    if page_id <= 0:
        raise RuntimeError(
            "native backend index cannot lower into the SGLang arena"
        )
    return page_id


def lower_extend_class(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[StepPlan],
    class_id: int,
    execution_context: Any | None,
    *,
    shared: Any,
) -> Any:
    import torch
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.srt.utils import next_power_of_2

    class_plans = [plan.by_class[class_id] for plan in plans]
    bs = len(plans)
    if int(prefix_lens_cpu.numel()) != len(class_plans) or int(
        targets_cpu.numel()
    ) != len(class_plans):
        raise RuntimeError("absolute and physical lowering cardinalities differ")
    physical_prefixes = torch.tensor(
        [int(item.previous_layout_boundary) for item in class_plans],
        dtype=torch.int64,
    )
    physical_targets = torch.tensor(
        [int(item.target_layout_boundary) for item in class_plans],
        dtype=torch.int64,
    )
    if execution_context is not None:
        execution_context.validate_current()
    prefix_lens = physical_prefixes.to(batch.device, non_blocking=True)
    if execution_context is not None:
        execution_context.validate_current()
    targets = physical_targets.to(batch.device, non_blocking=True)
    if execution_context is not None:
        execution_context.validate_current()
    last_loc = torch.tensor(
        [class_plan.last_location for class_plan in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    page_ids = [
        page for class_plan in class_plans for page in class_plan.exact_new_pages
    ]
    if execution_context is not None:
        execution_context.validate_current()
    exact_pages = torch.tensor(page_ids, dtype=torch.int64, device=batch.device)
    if execution_context is not None:
        execution_context.validate_current()
    out_cache_loc = torch.empty(
        (int(extend_num_tokens),), dtype=torch.int64, device=batch.device
    )
    if execution_context is not None:
        execution_context.validate_current()
    alloc_extend_kernel[(bs,)](
        prefix_lens,
        targets,
        last_loc,
        exact_pages,
        out_cache_loc,
        next_power_of_2(bs),
        shared._config().page_tokens,
    )
    return out_cache_loc


def lower_decode_class(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[StepPlan],
    class_id: int,
    execution_context: Any | None,
    *,
    shared: Any,
) -> Any:
    import torch
    from sglang.kernels.ops.memory.allocator import alloc_decode_kernel
    from sglang.srt.utils import next_power_of_2

    class_plans = [plan.by_class[class_id] for plan in plans]
    bs = len(plans)
    if int(targets_cpu.numel()) != len(class_plans):
        raise RuntimeError("absolute and physical decode cardinalities differ")
    if execution_context is not None:
        execution_context.validate_current()
    targets = torch.tensor(
        [int(item.target_layout_boundary) for item in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    if execution_context is not None:
        execution_context.validate_current()
    last_loc = torch.tensor(
        [class_plan.last_location for class_plan in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    page_ids = [
        page for class_plan in class_plans for page in class_plan.exact_new_pages
    ]
    if execution_context is not None:
        execution_context.validate_current()
    exact_pages = torch.tensor(page_ids, dtype=torch.int64, device=batch.device)
    if execution_context is not None:
        execution_context.validate_current()
    out_cache_loc = torch.empty((bs,), dtype=torch.int64, device=batch.device)
    if execution_context is not None:
        execution_context.validate_current()
    alloc_decode_kernel[(bs,)](
        targets,
        last_loc,
        exact_pages,
        out_cache_loc,
        next_power_of_2(bs),
        shared._config().page_tokens,
    )
    return out_cache_loc


def lower_all_extend(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[StepPlan],
    execution_context: Any | None,
    *,
    shared: Any,
) -> dict[int, Any]:
    return {
        item.class_id: shared._lower_extend_class(
            batch,
            prefix_lens_cpu,
            targets_cpu,
            extend_num_tokens,
            plans,
            item.class_id,
            execution_context,
        )
        for item in shared._config().classes
    }


def lower_all_decode(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[StepPlan],
    execution_context: Any | None,
    *,
    shared: Any,
) -> dict[int, Any]:
    return {
        item.class_id: shared._lower_decode_class(
            batch, targets_cpu, plans, item.class_id, execution_context
        )
        for item in shared._config().classes
    }


def primary_locations(locations: dict[int, Any], *, shared: Any) -> Any:
    class_config = shared._config().primary_class
    if class_config is None or set(locations) != {
        item.class_id for item in shared._config().classes
    }:
        raise RuntimeError("lowering did not return every compiled KV class")
    return locations[class_config.class_id]


def write_hybrid_lut(locations: dict[int, Any], *, shared: Any) -> None:
    import torch

    config = shared._config()
    full = config.full_class
    sliding = config.sliding_class
    if full is None or sliding is None:
        return
    if (
        tuple((item.class_id, item.retention) for item in config.classes)
        != ((0, "full"), (1, "sliding"))
        or any(
            getattr(item, "storage", "token_kv") != "token_kv"
            for item in config.classes
        )
        or set(locations) != {full.class_id, sliding.class_id}
    ):
        raise RuntimeError(
            "hybrid LUT publication requires exact ordered Full+SWA locations"
        )
    full_locations = locations[full.class_id]
    swa_locations = locations[sliding.class_id]
    if (
        getattr(full_locations, "ndim", None) != 1
        or getattr(swa_locations, "ndim", None) != 1
        or int(full_locations.numel()) != int(swa_locations.numel())
    ):
        raise RuntimeError("Full and SWA lowering cardinalities differ")
    shared._state._ALLOCATOR.set_full_to_swa_mapping(
        full_locations.to(dtype=torch.int64),
        swa_locations.to(dtype=torch.int64),
    )


def class_kv_pool(class_id: int, *, shared: Any) -> Any:
    kvcache = shared._state._ALLOCATOR.get_kvcache()
    config = shared._config()
    class_config = config.classes_by_id[class_id]
    if config.full_class is not None and config.sliding_class is not None:
        name = "full_kv_pool" if class_config.retention == "full" else "swa_kv_pool"
        pool = getattr(kvcache, name, None)
    elif config.full_class is None and config.sliding_class is not None:
        pool = getattr(kvcache, "swa_kv_pool", None)
    else:
        pool = kvcache
    if pool is None or not callable(getattr(pool, "move_kv_cache", None)):
        raise RuntimeError("SGLang KV pool cannot execute an exact COW copy")
    return pool


def validate_joint_hybrid_tails(
    plans: Sequence[StepPlan], *, shared: Any
) -> None:
    config = shared._config()
    full = config.full_class
    sliding = config.sliding_class
    if full is None or sliding is None:
        return
    if (
        tuple((item.class_id, item.retention) for item in config.classes)
        != ((0, "full"), (1, "sliding"))
        or any(
            getattr(item, "storage", "token_kv") != "token_kv"
            for item in config.classes
        )
    ):
        raise RuntimeError(
            "hybrid lowering requires exact ordered Full+SWA token_kv classes"
        )
    for plan in plans:
        if set(plan.by_class) != {full.class_id, sliding.class_id}:
            raise RuntimeError(
                "hybrid lowering requires exact Full and SWA class plans"
            )
        full_spec = plan.by_class[full.class_id]
        swa_spec = plan.by_class[sliding.class_id]
        compact = (
            full_spec.previous_layout_boundary is not None
            and full_spec.previous_layout_boundary != plan.previous_boundary
        )
        if compact:
            if (
                full_spec.previous_layout_boundary is None
                or full_spec.target_layout_boundary is None
                or swa_spec.previous_layout_boundary != plan.previous_boundary
                or swa_spec.target_layout_boundary != plan.target_boundary
                or full_spec.target_layout_boundary
                - full_spec.previous_layout_boundary
                != plan.target_boundary - plan.previous_boundary
                or full_spec.copy_intents
                or swa_spec.copy_intents
            ):
                raise RuntimeError(
                    "compact Hybrid transition changed class-specific append geometry"
                )
            continue
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


def intent_locations(
    intent: Any, *, destination: bool, device: Any, shared: Any
) -> Any:
    import torch

    arena = shared._runtime().arenas_by_class[intent.class_id]
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
        _sglang_page_id(backend_index, arena.backend_base_index)
        * shared._config().page_tokens
        + offset
    )
    return torch.arange(
        start,
        start + intent.token_count,
        dtype=torch.int64,
        device=device,
    )


def execute_cow_copies(
    batch: Any,
    plans: Sequence[StepPlan],
    execution_context: Any | None,
    *,
    shared: Any,
) -> tuple[int, int, int]:
    """Enqueue every exact manager copy on the current forward stream."""

    import torch

    device_module = torch.get_device_module(batch.device)
    if execution_context is None:
        # Preserve the legacy path exactly: establish its current device stream
        # without imposing the runtime-session execution identity contract.
        device_module.current_stream(batch.device)
    else:
        if (
            execution_context.device_module is not device_module
            or execution_context.device != batch.device
        ):
            raise RuntimeError(
                "COW execution context does not match the batch device"
            )
        execution_context.validate_current(
            device_module=device_module, device=batch.device
        )
    intent_count = move_calls = copied_tokens = 0
    for class_config in shared._config().classes:
        sources = []
        destinations = []
        for plan in plans:
            for intent in plan.by_class[class_config.class_id].copy_intents:
                if execution_context is not None:
                    execution_context.validate_current(
                        device_module=device_module, device=batch.device
                    )
                sources.append(
                    shared._intent_locations(
                        intent, destination=False, device=batch.device
                    )
                )
                if execution_context is not None:
                    execution_context.validate_current(
                        device_module=device_module, device=batch.device
                    )
                destinations.append(
                    shared._intent_locations(
                        intent, destination=True, device=batch.device
                    )
                )
        if sources:
            if execution_context is None:
                shared._class_kv_pool(class_config.class_id).move_kv_cache(
                    torch.cat(destinations), torch.cat(sources)
                )
            else:
                execution_context.validate_current(
                    device_module=device_module, device=batch.device
                )
                destination_locations = torch.cat(destinations)
                execution_context.validate_current(
                    device_module=device_module, device=batch.device
                )
                source_locations = torch.cat(sources)
                execution_context.validate_current(
                    device_module=device_module, device=batch.device
                )
                shared._class_kv_pool(class_config.class_id).move_kv_cache(
                    destination_locations, source_locations
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


def record_cow_activity(
    activity: tuple[int, int, int], *, shared: Any
) -> None:
    intent_count, move_calls, copied_tokens = activity
    if intent_count:
        shared._state._counter_add("cow_copy_intents", intent_count)
        shared._state._counter_add("cow_move_calls", move_calls)
        shared._state._counter_add("cow_copied_tokens", copied_tokens)
