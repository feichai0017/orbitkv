"""SGLang hooks for the native runtime-session product path.

The hook names remain stable because the source overlay imports this module.
All lifecycle authority lives in ``session_lowering``; only physical tensor
lowering and COW mirror helpers are shared here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..execution_plan import StepPlan
from ..runtime import TAIL_NONE
from . import physical_lowering as _physical_lowering
from . import session_lowering, state as _state
from .cow_mirror import preflight_compact_cow
from .state import _config

# ``physical_lowering`` receives this module as its shared dispatch namespace.
# Keep the runtime accessor explicit even though static call sites live there.
_runtime = _state._runtime


def _lower_extend_class(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[StepPlan],
    class_id: int,
    execution_context: Any | None = None,
) -> Any:
    return _physical_lowering.lower_extend_class(
        batch,
        prefix_lens_cpu,
        targets_cpu,
        extend_num_tokens,
        plans,
        class_id,
        execution_context,
        shared=sys.modules[__name__],
    )


def _lower_decode_class(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[StepPlan],
    class_id: int,
    execution_context: Any | None = None,
) -> Any:
    return _physical_lowering.lower_decode_class(
        batch,
        targets_cpu,
        plans,
        class_id,
        execution_context,
        shared=sys.modules[__name__],
    )


def _lower_all_extend(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[StepPlan],
    execution_context: Any | None = None,
) -> dict[int, Any]:
    return _physical_lowering.lower_all_extend(
        batch,
        prefix_lens_cpu,
        targets_cpu,
        extend_num_tokens,
        plans,
        execution_context,
        shared=sys.modules[__name__],
    )


def _lower_all_decode(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[StepPlan],
    execution_context: Any | None = None,
) -> dict[int, Any]:
    return _physical_lowering.lower_all_decode(
        batch,
        targets_cpu,
        plans,
        execution_context,
        shared=sys.modules[__name__],
    )


def _primary_locations(locations: dict[int, Any]) -> Any:
    return _physical_lowering.primary_locations(
        locations, shared=sys.modules[__name__]
    )


def _write_hybrid_lut(locations: dict[int, Any]) -> None:
    _physical_lowering.write_hybrid_lut(
        locations, shared=sys.modules[__name__]
    )


def _class_kv_pool(class_id: int) -> Any:
    return _physical_lowering.class_kv_pool(
        class_id, shared=sys.modules[__name__]
    )


def _validate_joint_hybrid_tails(plans: Sequence[StepPlan]) -> None:
    _physical_lowering.validate_joint_hybrid_tails(
        plans, shared=sys.modules[__name__]
    )


def _intent_locations(intent: Any, *, destination: bool, device: Any) -> Any:
    return _physical_lowering.intent_locations(
        intent,
        destination=destination,
        device=device,
        shared=sys.modules[__name__],
    )


def _execute_cow_copies(
    batch: Any,
    plans: Sequence[StepPlan],
    execution_context: Any | None = None,
) -> tuple[int, int, int]:
    return _physical_lowering.execute_cow_copies(
        batch, plans, execution_context, shared=sys.modules[__name__]
    )


def _record_cow_activity(activity: tuple[int, int, int]) -> None:
    _physical_lowering.record_cow_activity(
        activity, shared=sys.modules[__name__]
    )


def _tail_locations(
    spec: Any, *, source: bool, device: Any
) -> Any | None:
    import torch

    count = int(spec.tail_action.valid_token_count)
    if spec.tail_action.kind == TAIL_NONE or count == 0:
        return None
    if source and spec.copy_intents:
        return _intent_locations(
            spec.copy_intents[0], destination=False, device=device
        )
    end = int(spec.last_location) + 1
    return torch.arange(end - count, end, dtype=torch.int64, device=device)


@dataclass(frozen=True, slots=True)
class _CowMirrorPlan:
    assignments: tuple[tuple[Any, Any], ...]
    retained_assignments: tuple[
        tuple[Any, tuple[int, ...], tuple[int, ...]], ...
    ]
    mapping: Any | None
    mapping_assignments: tuple[tuple[Any, Any], ...]


def _preflight_cow_mirrors(
    batch: Any,
    plans: Sequence[StepPlan],
    use_prefix_mirror: Sequence[bool],
    execution_context: Any | None = None,
) -> _CowMirrorPlan:
    """Validate every source mirror before native submission."""

    import torch

    config = _config()
    primary = config.primary_class
    if primary is None:
        raise RuntimeError("OrbitKV plan has no primary KV class")
    comparisons: list[tuple[Any, Any]] = []
    assignments: list[tuple[Any, Any]] = []
    retained_assignments: list[
        tuple[Any, tuple[int, ...], tuple[int, ...]]
    ] = []
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
        compact_request = hasattr(req, "_orbitkv_active_kv_len")
        if compact_request:
            row = batch.req_to_token_pool.req_to_token[int(req.req_pool_idx)]
            substitutions_values = []
            for intent in primary_spec.copy_intents:
                if execution_context is not None:
                    execution_context.validate_current()
                old_locations = _intent_locations(
                    intent, destination=False, device=batch.device
                )
                if execution_context is not None:
                    execution_context.validate_current()
                new_locations = _intent_locations(
                    intent, destination=True, device=batch.device
                )
                substitutions_values.extend(
                    (int(old), int(new))
                    for old, new in zip(
                        old_locations, new_locations, strict=True
                    )
                )
            substitutions = tuple(substitutions_values)
            if execution_context is not None:
                execution_context.validate_current()
            compact = preflight_compact_cow(
                req,
                row,
                int(plan.previous_boundary),
                substitutions,
                prefix_authoritative=prefix_authoritative,
            )
            if execution_context is not None:
                execution_context.validate_current()
            assignments.extend(compact.assignments)
            if compact.replacement != compact.previous:
                retained_assignments.append(
                    (req, compact.previous, compact.replacement)
                )

        if (
            config.full_class is not None
            and config.sliding_class is not None
            and compact_request
        ):
            retained_full = getattr(req, "_orbitkv_retained_locations", None)
            retained_swa = getattr(
                req, "_orbitkv_retained_swa_locations", None
            )
            packed = (
                primary_spec.previous_layout_boundary is not None
                and primary_spec.previous_layout_boundary
                != plan.previous_boundary
            )
            if (
                not isinstance(retained_full, tuple)
                or not isinstance(retained_swa, tuple)
                or len(retained_full) != len(retained_swa)
                or packed
                and primary_spec.previous_layout_boundary != len(retained_full)
                or any(spec.copy_intents for spec in plan.class_specs)
            ):
                raise RuntimeError(
                    "compact Hybrid request lost its class-specific mirror authority"
                )
            row = batch.req_to_token_pool.req_to_token[
                int(req.req_pool_idx), : len(retained_full)
            ]
            if execution_context is not None:
                execution_context.validate_current()
            row = row.to(torch.int64)
            if execution_context is not None:
                execution_context.validate_current()
            expected_full = torch.tensor(
                retained_full, dtype=torch.int64, device=batch.device
            )
            if execution_context is not None:
                execution_context.validate_current()
            expected_swa = torch.tensor(
                retained_swa, dtype=torch.int64, device=batch.device
            )
            comparisons.append((row, expected_full))
            if execution_context is not None:
                execution_context.validate_current()
            mapped_full = mapping[expected_full]
            if execution_context is not None:
                execution_context.validate_current()
            comparisons.append((mapped_full.to(torch.int64), expected_swa))
            continue
        if compact_request:
            continue
        count = int(primary_spec.tail_action.valid_token_count)
        if count == 0:
            continue
        logical_begin = (
            primary_spec.tail_action.logical_ordinal * config.page_tokens
        )
        logical_end = logical_begin + count
        if execution_context is not None:
            execution_context.validate_current()
        old_primary = _tail_locations(
            primary_spec, source=True, device=batch.device
        )
        if execution_context is not None:
            execution_context.validate_current()
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
                raise RuntimeError(
                    "new request lost its attached prefix mirror"
                )
            comparisons.append(
                (prefix[logical_begin:logical_end], old_primary)
            )
        else:
            if execution_context is not None:
                execution_context.validate_current()
            comparisons.append((row_view.to(dtype=torch.int64), old_primary))
        if execution_context is not None:
            execution_context.validate_current()
        assignments.append((row_view, new_primary.to(dtype=row_view.dtype)))
        prefix_end = min(
            logical_end, int(prefix.numel()) if prefix is not None else 0
        )
        if prefix_end > logical_begin:
            prefix_view = prefix[logical_begin:prefix_end]
            prefix_count = prefix_end - logical_begin
            comparisons.append((prefix_view, old_primary[:prefix_count]))
            if execution_context is not None:
                execution_context.validate_current()
            assignments.append(
                (prefix_view, new_primary[:prefix_count])
            )

        if mapping is not None:
            sliding = config.sliding_class
            assert sliding is not None
            sliding_spec = plan.by_class[sliding.class_id]
            if int(sliding_spec.tail_action.valid_token_count) != count:
                raise RuntimeError("Full and SWA tail visibility differs")
            if execution_context is not None:
                execution_context.validate_current()
            old_sliding = _tail_locations(
                sliding_spec, source=True, device=batch.device
            )
            if execution_context is not None:
                execution_context.validate_current()
            new_sliding = _tail_locations(
                sliding_spec, source=False, device=batch.device
            )
            assert old_sliding is not None and new_sliding is not None
            if execution_context is not None:
                execution_context.validate_current()
            comparisons.append((mapping[old_primary], old_sliding))
            mapping_assignments.append((new_primary, new_sliding))

    if comparisons:
        actual_values = []
        expected_values = []
        for actual_value, expected_value in comparisons:
            if execution_context is not None:
                execution_context.validate_current()
            actual_values.append(actual_value.to(dtype=torch.int64))
            if execution_context is not None:
                execution_context.validate_current()
            expected_values.append(expected_value.to(dtype=torch.int64))
        if execution_context is not None:
            execution_context.validate_current()
        actual = torch.cat(tuple(actual_values))
        if execution_context is not None:
            execution_context.validate_current()
        expected = torch.cat(tuple(expected_values))
        if execution_context is not None:
            execution_context.validate_current()
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "COW intent disagrees with the SGLang candidate mirror"
            )
    return _CowMirrorPlan(
        tuple(assignments),
        tuple(retained_assignments),
        mapping,
        tuple(mapping_assignments),
    )


def _commit_cow_mirrors(
    plan: _CowMirrorPlan, execution_context: Any | None = None
) -> None:
    if any(
        getattr(req, "_orbitkv_retained_locations", None) is not previous
        for req, previous, _replacement in plan.retained_assignments
    ):
        raise RuntimeError(
            "compact retained locations changed before COW mirror commit"
        )
    for target, replacement in plan.assignments:
        if execution_context is not None:
            execution_context.validate_current()
        target.copy_(replacement)
    if plan.mapping is not None:
        for full_locations, swa_locations in plan.mapping_assignments:
            if execution_context is not None:
                execution_context.validate_current()
            plan.mapping[full_locations] = swa_locations
    for req, _previous, replacement in plan.retained_assignments:
        req._orbitkv_retained_locations = replacement


def _alloc_for_extend(batch: Any) -> tuple[Any, Any, Any]:
    return session_lowering.alloc_for_extend(batch)


def _alloc_for_decode(batch: Any, token_per_req: int) -> Any:
    return session_lowering.alloc_for_decode(batch, token_per_req)


def _manager_maybe_evict_swa(batch: Any) -> None:
    session_lowering.manager_maybe_evict_swa(batch)


def _get_next_batch_to_run(
    original_fn: Callable[..., Any],
    scheduler: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return session_lowering.get_next_batch_to_run(
        original_fn, scheduler, *args, **kwargs
    )


def _flush_release_group(candidates: Sequence[Any]) -> None:
    session_lowering.flush_release_group(candidates)


def _release_kv_cache(
    req: Any, tree_cache: Any, is_insert: bool = True
) -> None:
    session_lowering.release_kv_cache(
        req, tree_cache, is_insert=bool(is_insert)
    )


__all__ = [
    "_alloc_for_decode",
    "_alloc_for_extend",
    "_flush_release_group",
    "_get_next_batch_to_run",
    "_manager_maybe_evict_swa",
    "_release_kv_cache",
]
