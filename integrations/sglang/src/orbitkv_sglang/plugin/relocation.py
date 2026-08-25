from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

from ..runtime import (
    ClassTokenDispositionUpdate,
    FailStopped,
    RelocationBatchItem,
    RelocationCopyBatch,
    RelocationCopyReceipt,
    RelocationCopyUnobserved,
    RelocationPolicy,
    TokenDisposition,
    TokenDispositionKind,
)
from . import state as _state
from .private_prefix import (
    PrivatePrefixProvenance,
    replace_private_prefix,
    validate_private_prefix,
)
from .state import _config, _request_key, _runtime
from .validation import _validate_mla_pool_geometry


_NEXT_RECLAMATION_BOUNDARY = "_orbitkv_token_reclamation_next_boundary"


@dataclass(frozen=True, slots=True)
class _CompactMirrorPlan:
    req: Any
    row: Any
    boundary: int
    retained_count: int
    replacement: Any
    full_locations: tuple[int, ...]
    mapping: Any | None
    old_full_indices: Any | None
    new_full_indices: Any | None
    swa_replacement: Any | None
    swa_locations: tuple[int, ...] | None
    private_prefix: PrivatePrefixProvenance | None


def _retained_placements(view: Any) -> tuple[Any, ...]:
    result = []
    for placement in view.placements:
        try:
            kind = TokenDispositionKind(placement.disposition.kind)
        except (TypeError, ValueError) as error:
            raise RuntimeError("token view has an invalid disposition") from error
        if kind is not TokenDispositionKind.RETAINED:
            continue
        if placement.location is None:
            raise RuntimeError("retained token lacks a physical location")
        result.append(placement)
    return tuple(result)


def _victim_updates(full_view: Any) -> tuple[ClassTokenDispositionUpdate, ...]:
    config = _config()
    policy = config.token_reclamation
    full = config.full_class
    if full is None or full_view.class_id != full.class_id:
        raise RuntimeError("token-reclamation victims require the Full token view")
    victims = tuple(
        placement.token_id
        for active_ordinal, placement in enumerate(_retained_placements(full_view))
        if active_ordinal % config.page_tokens >= policy.retained_per_page
    )
    return tuple(
        ClassTokenDispositionUpdate(
            class_id=class_config.class_id,
            token_id=token_id,
            disposition=TokenDisposition(
                TokenDispositionKind.POLICY_EVICTED,
                policy.policy_id,
                policy.policy_version,
                policy.quality_contract,
            ),
        )
        for class_config in config.classes
        for token_id in victims
    )


def _view_locations(view: Any) -> tuple[int, ...]:
    arena = _runtime().arenas_by_class[view.class_id]
    result = []
    for placement in _retained_placements(view):
        location = placement.location
        assert location is not None
        if not 0 <= location.offset < _config().page_tokens:
            raise RuntimeError("retained token location offset is out of range")
        page = location.backend_index - arena.backend_base_index + 1
        result.append(page * _config().page_tokens + location.offset)
    return tuple(result)


def _publication_locations(name: str, values: Any) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise RuntimeError(
            f"token reclamation {name} locations are not a sequence"
        ) from error
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RuntimeError(
                f"token reclamation {name} locations are not integers"
            )
        value = int(value)
        if value < 0:
            raise RuntimeError(
                f"token reclamation {name} locations must be nonnegative"
            )
        result.append(value)
    return tuple(result)


def _preflight_request(
    req: Any, row: Any, boundary: int, active: int, old_views: Any
) -> PrivatePrefixProvenance | None:
    import torch

    private_prefix = validate_private_prefix(
        req,
        row,
        getattr(req, "_orbitkv_request_key", None),
        getattr(req, "_orbitkv_request_lease", None),
        boundary,
    )
    if private_prefix is not None:
        key = _request_key(req)
        record = _runtime().record_for(key)
        if (
            private_prefix.request_key != key
            or private_prefix.request_lease != getattr(record, "lease", None)
        ):
            raise RuntimeError("request-private prefix provenance changed")
    if int(row.numel()) < boundary:
        raise RuntimeError("ReqToToken row is shorter than the absolute boundary")
    views = tuple(old_views)
    classes = tuple(_config().classes)
    if tuple(view.class_id for view in views) != tuple(
        item.class_id for item in classes
    ):
        raise RuntimeError("pre-reclamation token views are not class ordered")
    if any(
        view.page_tokens != _config().page_tokens
        or len(view.placements) != boundary
        or tuple(placement.token_id for placement in view.placements)
        != tuple(range(boundary))
        for view in views
    ):
        raise RuntimeError("pre-reclamation token view shape changed")
    retained_ids = tuple(
        tuple(placement.token_id for placement in _retained_placements(view))
        for view in views
    )
    if not retained_ids or any(ids != retained_ids[0] for ids in retained_ids[1:]):
        raise RuntimeError("attention classes do not share one retained token set")
    locations = {view.class_id: _view_locations(view) for view in views}
    full = _config().full_class
    assert full is not None
    expected = locations[full.class_id]
    if len(expected) != active:
        raise RuntimeError("Full token view differs from the active KV length")
    compacted = hasattr(req, "_orbitkv_active_kv_len")
    mirrored_active = getattr(req, "_orbitkv_active_kv_len", active)
    if (
        isinstance(mirrored_active, bool)
        or not isinstance(mirrored_active, Integral)
        or int(mirrored_active) != active
    ):
        raise RuntimeError("active KV length mirror differs from manager authority")
    mirrored_locations = getattr(req, "_orbitkv_retained_locations", None)
    if compacted and not isinstance(mirrored_locations, tuple):
        raise RuntimeError(
            "compact request lost its retained Full locations mirror"
        )
    if mirrored_locations is None:
        mirrored_locations = expected
    try:
        mirrored_locations = tuple(mirrored_locations)
    except TypeError as error:
        raise RuntimeError(
            "retained Full locations mirror is not a sequence"
        ) from error
    if mirrored_locations != expected:
        raise RuntimeError("retained Full locations differ from the authority view")
    expected_tensor = torch.tensor(expected, dtype=torch.int64, device=row.device)
    if not torch.equal(row[:active].to(torch.int64), expected_tensor):
        raise RuntimeError("victim set disagrees with the ReqToToken authority mirror")
    if int(torch.count_nonzero(row[active:boundary]).item()) != 0:
        raise RuntimeError("compact ReqToToken tail is not clear")
    sliding = _config().sliding_class
    if sliding is not None:
        mapping = _state._ALLOCATOR.full_to_swa_index_mapping
        full_tensor = expected_tensor
        swa_tensor = torch.tensor(
            locations[sliding.class_id], dtype=torch.int64, device=row.device
        )
        if not torch.equal(mapping[full_tensor].to(torch.int64), swa_tensor):
            raise RuntimeError("victim set disagrees with the Full-to-SWA LUT")
        mirrored_swa = getattr(req, "_orbitkv_retained_swa_locations", None)
        if compacted and not isinstance(mirrored_swa, tuple):
            raise RuntimeError(
                "compact Full+SWA request lost its retained SWA locations mirror"
            )
        if mirrored_swa is not None:
            try:
                mirrored_swa = tuple(mirrored_swa)
            except TypeError as error:
                raise RuntimeError(
                    "retained SWA locations mirror is not a sequence"
                ) from error
            if mirrored_swa != locations[sliding.class_id]:
                raise RuntimeError(
                    "retained SWA locations differ from the authority view"
                )
    return private_prefix


def _plan_compact_row(
    req: Any,
    row: Any,
    publication: Any,
    old_full: tuple[int, ...],
    boundary: int,
    private_prefix: PrivatePrefixProvenance | None = None,
) -> _CompactMirrorPlan:
    import torch

    raw_locations = tuple(publication.class_retained_locations)
    class_ids = tuple(item.class_id for item in _config().classes)
    if (
        len(raw_locations) != len(class_ids)
        or tuple(class_id for class_id, _locations in raw_locations) != class_ids
    ):
        raise RuntimeError(
            "token reclamation publication changed the configured class order"
        )
    class_locations = {
        class_id: _publication_locations(
            _config().classes_by_id[class_id].name
            if hasattr(_config(), "classes_by_id")
            else f"class {class_id}",
            locations,
        )
        for class_id, locations in raw_locations
    }
    full = _config().full_class
    assert full is not None
    locations = class_locations[full.class_id]
    count = len(locations)
    if not 0 < count < len(old_full) <= boundary:
        raise RuntimeError("token reclamation did not produce a strict retained subset")
    if not isinstance(row, torch.Tensor) or row.ndim != 1:
        raise RuntimeError("token reclamation ReqToToken row is not a vector")
    try:
        row_limit = torch.iinfo(row.dtype).max
    except TypeError as error:
        raise RuntimeError(
            "token reclamation ReqToToken row is not integer typed"
        ) from error
    if locations and max(locations) > row_limit:
        raise RuntimeError(
            "token reclamation Full locations exceed the ReqToToken dtype"
        )
    replacement = torch.tensor(locations, dtype=row.dtype, device=row.device)
    sliding = _config().sliding_class
    mapping = None
    full_locations = None
    swa_locations = None
    old_full_locations = None
    if sliding is not None:
        mapping = _state._ALLOCATOR.full_to_swa_index_mapping
        if (
            not isinstance(mapping, torch.Tensor)
            or mapping.ndim != 1
            or mapping.device != row.device
        ):
            raise RuntimeError(
                "token reclamation Full-to-SWA LUT geometry changed"
            )
        try:
            mapping_limit = torch.iinfo(mapping.dtype).max
        except TypeError as error:
            raise RuntimeError(
                "token reclamation Full-to-SWA LUT is not integer typed"
            ) from error
        old_full = _publication_locations("old Full", old_full)
        if any(
            location >= int(mapping.numel())
            for location in (*old_full, *locations)
        ):
            raise RuntimeError(
                "token reclamation Full location is outside the Full-to-SWA LUT"
            )
        sliding_locations = class_locations[sliding.class_id]
        if sliding_locations and max(sliding_locations) > mapping_limit:
            raise RuntimeError(
                "token reclamation SWA locations exceed the Full-to-SWA LUT dtype"
            )
        full_locations = torch.tensor(
            locations, dtype=torch.int64, device=mapping.device
        )
        swa_locations = torch.tensor(
            sliding_locations, dtype=mapping.dtype, device=mapping.device
        )
        if int(full_locations.numel()) != int(swa_locations.numel()):
            raise RuntimeError("Full and SWA retained cardinalities differ")
        old_full_locations = torch.tensor(
            old_full, dtype=torch.int64, device=mapping.device
        )
    if any(len(class_locations[class_id]) != count for class_id in class_ids):
        raise RuntimeError(
            "token reclamation class retained cardinalities differ"
        )

    return _CompactMirrorPlan(
        req=req,
        row=row,
        boundary=boundary,
        retained_count=count,
        replacement=replacement,
        full_locations=locations,
        mapping=mapping,
        old_full_indices=old_full_locations,
        new_full_indices=full_locations,
        swa_replacement=swa_locations,
        swa_locations=(
            tuple(class_locations[sliding.class_id])
            if sliding is not None
            else None
        ),
        private_prefix=private_prefix,
    )


def _commit_compact_row(plan: _CompactMirrorPlan) -> None:
    plan.row[: plan.retained_count].copy_(plan.replacement)
    plan.row[plan.retained_count : plan.boundary].zero_()
    plan.req._orbitkv_active_kv_len = plan.retained_count
    plan.req._orbitkv_retained_locations = plan.full_locations
    if plan.private_prefix is not None:
        replace_private_prefix(
            plan.req,
            plan.private_prefix,
            plan.replacement,
            boundary=plan.boundary,
        )
    if plan.mapping is not None:
        assert (
            plan.old_full_indices is not None
            and plan.new_full_indices is not None
            and plan.swa_replacement is not None
            and plan.swa_locations is not None
        )
        plan.mapping[plan.old_full_indices] = 0
        plan.mapping[plan.new_full_indices] = plan.swa_replacement
        plan.req._orbitkv_retained_swa_locations = plan.swa_locations


def _publish_compact_row(
    req: Any,
    row: Any,
    publication: Any,
    old_full: tuple[int, ...],
    boundary: int,
    private_prefix: PrivatePrefixProvenance | None = None,
) -> int:
    plan = _plan_compact_row(
        req, row, publication, old_full, boundary, private_prefix
    )
    _commit_compact_row(plan)
    return plan.retained_count


def _copy_callback(batch: Any) -> Callable[[tuple[Any, ...]], RelocationCopyBatch]:
    def execute(prepared: tuple[Any, ...]) -> RelocationCopyBatch:
        import torch

        enqueued = False
        try:
            prepared = tuple(prepared)
            if not prepared:
                raise RuntimeError(
                    "relocation copy callback received an empty batch"
                )
            sources = []
            destinations = []
            receipt_groups = []
            for item in prepared:
                arena = _runtime().arenas_by_class[item.class_id]
                receipts = []
                for movement in item.moves:
                    source_page = (
                        movement.source.backend_index
                        - arena.backend_base_index
                        + 1
                    )
                    destination_page = (
                        movement.destination.backend_index
                        - arena.backend_base_index
                        + 1
                    )
                    sources.append(
                        source_page * _config().page_tokens
                        + movement.source.offset
                    )
                    destinations.append(
                        destination_page * _config().page_tokens
                        + movement.destination.offset
                    )
                    receipts.append(
                        RelocationCopyReceipt(
                            item.relocation,
                            movement.token_id,
                            movement.source,
                            movement.destination,
                        )
                    )
                receipt_groups.append(tuple(receipts))
            if not sources:
                raise RuntimeError(
                    "relocation copy callback received no token moves"
                )
            device = batch.device
            device_module = torch.get_device_module(device)
            source = torch.tensor(sources, dtype=torch.int64, device=device)
            destination = torch.tensor(
                destinations, dtype=torch.int64, device=device
            )
            kvcache = _state._ALLOCATOR.get_kvcache()
            pool = (
                kvcache.full_kv_pool
                if _config().sliding_class is not None
                else kvcache
            )
            move_kv_cache = getattr(pool, "move_kv_cache", None)
            if not callable(move_kv_cache):
                raise RuntimeError(
                    "relocation KV pool does not expose move_kv_cache"
                )
            for class_id in {item.class_id for item in prepared}:
                class_config = _config().classes_by_id[class_id]
                if class_config.storage == "latent_kv":
                    _validate_mla_pool_geometry(pool, class_config)
            producer_stream = device_module.current_stream(device)
            producer_event = device_module.Event()
            stream = device_module.Stream(device=device)
            completion_event = device_module.Event()
            # Relocation runs on a private stream, while the KV rows being moved
            # were produced on SGLang's current forward stream.  Record the
            # dependency after materializing the index tensors and make the copy
            # stream wait before it may read source KV.
            producer_event.record(stream=producer_stream)
            stream.wait_event(producer_event)
            with device_module.stream(stream):
                # Once this call begins, an exception cannot prove that the backend
                # observed no copy.  Only failures above this point are abortable.
                enqueued = True
                move_kv_cache(destination, source)
                completion_event.record(stream=stream)
            completion_event.synchronize()
        except Exception as error:
            if not enqueued:
                raise RelocationCopyUnobserved(
                    f"relocation CUDA copy was not enqueued: {error}"
                ) from error
            raise RuntimeError(f"relocation CUDA copy/event failed: {error}") from error
        _state._counter_add("relocation_copy_events")
        _state._counter_add("relocation_copy_tokens", len(sources))
        return RelocationCopyBatch(
            tuple(receipt_groups),
            int(source.device.index or 0) + 65_537,
        )

    return execute


def _maybe_reclaim_decode_batch(batch: Any) -> None:
    policy = _config().token_reclamation
    if policy.mode == "off" or not batch.reqs:
        return
    runtime = _runtime()
    full = _config().full_class
    if full is None:
        raise RuntimeError("token reclamation requires a Full class")
    candidates = []
    candidate_state = []
    for req in batch.reqs:
        key = _request_key(req)
        if not runtime.has_request(key):
            continue
        record = runtime.record_for(key)
        boundary = record.boundary
        if (
            isinstance(boundary, bool)
            or not isinstance(boundary, Integral)
            or int(boundary) <= 0
        ):
            raise RuntimeError("manager request boundary is invalid")
        boundary = int(boundary)
        allocated = getattr(getattr(req, "kv", None), "kv_allocated_len", None)
        if (
            isinstance(allocated, bool)
            or not isinstance(allocated, Integral)
            or int(allocated) != boundary
        ):
            raise RuntimeError(
                "published boundary differs from the SGLang KV boundary"
            )
        active = runtime.active_kv_length(key, full.class_id)
        if isinstance(active, bool) or not isinstance(active, Integral) or active <= 0:
            raise RuntimeError("active Full KV length is invalid")
        compacted = hasattr(req, "_orbitkv_active_kv_len")
        if compacted and not hasattr(req, _NEXT_RECLAMATION_BOUNDARY):
            raise RuntimeError("compacted request lost its reclamation schedule")
        next_boundary = (
            getattr(req, _NEXT_RECLAMATION_BOUNDARY)
            if compacted
            else policy.trigger_tokens
        )
        if (
            isinstance(next_boundary, bool)
            or not isinstance(next_boundary, Integral)
            or next_boundary <= 0
        ):
            raise RuntimeError("next token-reclamation boundary is invalid")
        next_boundary = int(next_boundary)
        if compacted:
            mirrored_active = getattr(req, "_orbitkv_active_kv_len")
            if (
                isinstance(mirrored_active, bool)
                or not isinstance(mirrored_active, Integral)
                or int(mirrored_active) != active
            ):
                raise RuntimeError(
                    "active KV length mirror differs from manager authority"
                )
            expected_next = boundary + policy.trigger_tokens - active
            if next_boundary != expected_next:
                raise RuntimeError("next token-reclamation boundary changed")
        if boundary < next_boundary:
            continue
        if boundary > next_boundary:
            raise RuntimeError("request passed the next reclamation boundary")
        if active != policy.trigger_tokens:
            raise RuntimeError(
                "scheduled reclamation boundary has the wrong active KV length"
            )
        candidates.append(req)
        candidate_state.append((key, boundary, int(active)))
    if not candidates:
        return
    table = batch.req_to_token_pool.req_to_token
    preflighted = []
    row_indices = set()
    for req, (key, boundary, active) in zip(
        candidates, candidate_state, strict=True
    ):
        raw_row_index = req.req_pool_idx
        if (
            isinstance(raw_row_index, bool)
            or not isinstance(raw_row_index, Integral)
        ):
            raise RuntimeError("token-reclamation ReqToToken row is not an integer")
        row_index = int(raw_row_index)
        if row_index <= 0 or row_index >= int(table.shape[0]):
            raise RuntimeError("token-reclamation ReqToToken row is out of range")
        if row_index in row_indices:
            raise RuntimeError(
                "token-reclamation batch aliases a ReqToToken row"
            )
        row_indices.add(row_index)
        row = table[row_index]
        old_views = tuple(
            runtime.token_view(key, item.class_id) for item in _config().classes
        )
        private_prefix = _preflight_request(
            req, row, boundary, active, old_views
        )
        old_full_view = next(
            view for view in old_views if view.class_id == full.class_id
        )
        updates = _victim_updates(old_full_view)
        if not updates:
            raise RuntimeError("configured victim policy produced no updates")
        old_full_locations = _view_locations(old_full_view)
        preflighted.append(
            (
                req,
                key,
                boundary,
                row,
                updates,
                old_full_locations,
                private_prefix,
            )
        )

    # Freeze every candidate's views, mirrors, prefix state, and victim set before
    # the first native mutation.  The naive mode still has only scalar runtime
    # disposition publication, so its boundary is preflight atomicity rather than
    # multi-request commit atomicity.
    if policy.mode == "naive":
        for (
            req,
            key,
            boundary,
            row,
            updates,
            old_full_locations,
            private_prefix,
        ) in preflighted:
            publication = runtime.mark_token_dispositions(key, full.class_id, updates)
            try:
                retained = _publish_compact_row(
                    req,
                    row,
                    publication,
                    old_full_locations,
                    boundary,
                    private_prefix,
                )
                setattr(
                    req,
                    _NEXT_RECLAMATION_BOUNDARY,
                    boundary + policy.trigger_tokens - retained,
                )
                req.skip_radix_cache_insert = True
                import torch

                torch.get_device_module(batch.device).current_stream(
                    batch.device
                ).synchronize()
            except Exception as error:
                runtime.fail_stop(
                    f"naive disposition mirror publication became uncertain: {error}"
                )
                raise FailStopped(
                    runtime.failure_reason or "naive mirror publication failed"
                ) from error
    elif policy.mode == "relocate":
        relocation_policy = RelocationPolicy(
            policy.maximum_source_pages,
            policy.evacuation_headroom_pages,
            policy.fragmentation_threshold_milli,
            True,
        )
        publication_batch = runtime.relocate_tokens_batch(
            tuple(
                RelocationBatchItem(key, full.class_id, updates, relocation_policy)
                for (
                    _req, key, _boundary, _row, updates, _old_full, _private
                ) in preflighted
            ),
            _copy_callback(batch),
        )
        try:
            publications = tuple(publication_batch.items)
            if len(publications) != len(preflighted) or any(
                publication.key != key
                for publication, (
                    _req, key, _boundary, _row, _updates, _old_full, _private
                ) in zip(publications, preflighted, strict=True)
            ):
                raise RuntimeError(
                    "relocation batch publication order changed"
                )
            mirror_plans = tuple(
                _plan_compact_row(
                    req, row, publication, old_full, boundary, private_prefix
                )
                for publication, (
                    req,
                    _key,
                    boundary,
                    row,
                    _updates,
                    old_full,
                    private_prefix,
                ) in zip(publications, preflighted, strict=True)
            )
            for mirror_plan in mirror_plans:
                _commit_compact_row(mirror_plan)
                setattr(
                    mirror_plan.req,
                    _NEXT_RECLAMATION_BOUNDARY,
                    mirror_plan.boundary
                    + policy.trigger_tokens
                    - mirror_plan.retained_count,
                )
                mirror_plan.req.skip_radix_cache_insert = True

            import torch

            torch.get_device_module(batch.device).current_stream(
                batch.device
            ).synchronize()
            runtime.acknowledge_relocation_batch(publication_batch)
        except Exception as error:
            runtime.fail_stop(
                f"relocation batch mirror publication became uncertain: {error}"
            )
            raise FailStopped(
                runtime.failure_reason or "relocation batch mirror publication failed"
            ) from error
        _state._counter_add("relocation_batches")
        _state._counter_add(
            "relocation_moves",
            sum(len(publication.prepared.moves) for publication in publications),
        )
        _state._counter_add(
            "relocation_reclaimed_pages",
            sum(
                publication.prepared.projected_reclaimed_pages
                for publication in publications
            ),
        )
    else:
        raise RuntimeError("unknown token-reclamation mode")
    _state._counter_add("token_disposition_batches")
    _state._counter_add(
        "token_policy_evictions",
        sum(
            len(updates)
            for _req, _key, _boundary, _row, updates, _old, _private in preflighted
        ),
    )


def _active_forward_lengths(
    result: Any, _cls: Any, batch: Any, _model_runner: Any, **_kwargs: Any
) -> Any:
    import torch

    absolute = [int(value) for value in batch.seq_lens_cpu.tolist()]
    active = [
        int(getattr(req, "_orbitkv_active_kv_len", value))
        for req, value in zip(batch.reqs, absolute, strict=True)
    ]
    if active == absolute:
        return result
    if len(active) != int(result.batch_size):
        raise RuntimeError("active KV length cardinality changed")
    result.absolute_seq_lens = result.seq_lens
    result.absolute_seq_lens_cpu = result.seq_lens_cpu
    result.absolute_seq_lens_sum = result.seq_lens_sum
    result.seq_lens = torch.tensor(
        active,
        dtype=result.absolute_seq_lens.dtype,
        device=result.absolute_seq_lens.device,
    )
    result.seq_lens_cpu = torch.tensor(active, dtype=torch.int64)
    result.seq_lens_sum = sum(active)
    return result


__all__ = ["_active_forward_lengths", "_maybe_reclaim_decode_batch"]
