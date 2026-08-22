from __future__ import annotations

from typing import Any, Callable

from ..runtime import (
    ClassTokenDispositionUpdate,
    FailStopped,
    RelocationCopyReceipt,
    RelocationPolicy,
    TokenDisposition,
    TokenDispositionKind,
)
from . import state as _state
from .state import _config, _request_key, _runtime


def _victim_updates(boundary: int) -> tuple[ClassTokenDispositionUpdate, ...]:
    policy = _config().token_reclamation
    if boundary != policy.trigger_tokens:
        return ()
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
        for class_config in _config().classes
        for token_id in range(boundary)
        if token_id % _config().page_tokens >= policy.retained_per_page
    )


def _view_locations(view: Any) -> tuple[int, ...]:
    arena = _runtime().arenas_by_class[view.class_id]
    result = []
    for placement in view.placements:
        if placement.location is None:
            raise RuntimeError("pre-reclamation token lacks a physical location")
        location = placement.location
        page = location.backend_index - arena.backend_base_index + 1
        result.append(page * _config().page_tokens + location.offset)
    return tuple(result)


def _preflight_request(req: Any, row: Any, boundary: int, old_views: Any) -> None:
    import torch

    prefix = getattr(req, "prefix_indices", None)
    if prefix is not None and int(prefix.numel()) != 0:
        raise RuntimeError("first token-reclamation profile forbids Prefix mirrors")
    if int(row.numel()) < boundary:
        raise RuntimeError("ReqToToken row is shorter than the absolute boundary")
    views = tuple(old_views)
    if tuple(view.class_id for view in views) != tuple(
        item.class_id for item in _config().classes
    ):
        raise RuntimeError("pre-reclamation token views are not class ordered")
    locations = {view.class_id: _view_locations(view) for view in views}
    full = _config().full_class
    assert full is not None
    expected = locations[full.class_id]
    expected_tensor = torch.tensor(expected, dtype=torch.int64, device=row.device)
    if not torch.equal(row[:boundary].to(torch.int64), expected_tensor):
        raise RuntimeError("victim set disagrees with the ReqToToken authority mirror")
    sliding = _config().sliding_class
    if sliding is not None:
        mapping = _state._ALLOCATOR.full_to_swa_index_mapping
        full_tensor = expected_tensor
        swa_tensor = torch.tensor(
            locations[sliding.class_id], dtype=torch.int64, device=row.device
        )
        if not torch.equal(mapping[full_tensor].to(torch.int64), swa_tensor):
            raise RuntimeError("victim set disagrees with the Full-to-SWA LUT")


def _publish_compact_row(
    req: Any, row: Any, publication: Any, old_full: tuple[int, ...], boundary: int
) -> None:
    import torch

    class_locations = dict(publication.class_retained_locations)
    full = _config().full_class
    assert full is not None
    locations = class_locations[full.class_id]
    count = len(locations)
    if not 0 < count < boundary:
        raise RuntimeError("token reclamation did not produce a strict retained subset")
    replacement = torch.tensor(locations, dtype=row.dtype, device=row.device)
    row[:count].copy_(replacement)
    row[count:boundary].zero_()
    req._orbitkv_active_kv_len = count
    req._orbitkv_retained_locations = tuple(locations)
    sliding = _config().sliding_class
    if sliding is not None:
        mapping = _state._ALLOCATOR.full_to_swa_index_mapping
        full_locations = torch.tensor(
            locations, dtype=torch.int64, device=row.device
        )
        swa_locations = torch.tensor(
            class_locations[sliding.class_id], dtype=torch.int64, device=row.device
        )
        if int(full_locations.numel()) != int(swa_locations.numel()):
            raise RuntimeError("Full and SWA retained cardinalities differ")
        old_full_locations = torch.tensor(
            old_full, dtype=torch.int64, device=row.device
        )
        mapping[old_full_locations] = 0
        mapping[full_locations] = swa_locations
        req._orbitkv_retained_swa_locations = tuple(class_locations[sliding.class_id])


def _copy_callback(batch: Any) -> Callable[[Any], tuple[RelocationCopyReceipt, ...]]:
    def execute(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
        import torch

        arena = _runtime().arenas_by_class[prepared.class_id]
        sources = []
        destinations = []
        for movement in prepared.moves:
            source_page = movement.source.backend_index - arena.backend_base_index + 1
            destination_page = (
                movement.destination.backend_index - arena.backend_base_index + 1
            )
            sources.append(source_page * _config().page_tokens + movement.source.offset)
            destinations.append(
                destination_page * _config().page_tokens + movement.destination.offset
            )
        device = batch.device
        device_module = torch.get_device_module(device)
        source = torch.tensor(sources, dtype=torch.int64, device=device)
        destination = torch.tensor(destinations, dtype=torch.int64, device=device)
        stream = device_module.Stream(device=device)
        event = device_module.Event()
        try:
            with device_module.stream(stream):
                kvcache = _state._ALLOCATOR.get_kvcache()
                pool = (
                    kvcache.full_kv_pool
                    if _config().sliding_class is not None
                    else kvcache
                )
                pool.move_kv_cache(destination, source)
                event.record(stream=stream)
            event.synchronize()
        except Exception as error:
            raise RuntimeError(f"relocation CUDA copy/event failed: {error}") from error
        _state._counter_add("relocation_copy_events")
        _state._counter_add("relocation_copy_tokens", len(sources))
        return tuple(
            RelocationCopyReceipt(
                prepared.relocation,
                movement.token_id,
                movement.source,
                movement.destination,
            )
            for movement in prepared.moves
        )

    return execute


def _maybe_reclaim_decode_batch(batch: Any) -> None:
    policy = _config().token_reclamation
    if policy.mode == "off" or not batch.reqs:
        return
    runtime = _runtime()
    candidates = tuple(
        req
        for req in batch.reqs
        if runtime.has_request(_request_key(req))
        and getattr(getattr(req, "kv", None), "kv_allocated_len", None)
        == policy.trigger_tokens
        and not bool(getattr(req, "_orbitkv_token_reclamation_done", False))
    )
    if not candidates:
        return
    if any(
        runtime.record_for(_request_key(req)).boundary != policy.trigger_tokens
        for req in candidates
    ):
        raise RuntimeError("published boundary differs from reclamation trigger")
    table = batch.req_to_token_pool.req_to_token
    for req in candidates:
        key = _request_key(req)
        record = runtime.record_for(key)
        row_index = int(req.req_pool_idx)
        row = table[row_index]
        updates = _victim_updates(record.boundary)
        if not updates:
            raise RuntimeError("configured victim policy produced no updates")
        old_views = tuple(
            runtime.token_view(key, item.class_id) for item in _config().classes
        )
        _preflight_request(req, row, record.boundary, old_views)
        old_full_locations = _view_locations(old_views[0])
        if policy.mode == "naive":
            publication = runtime.mark_token_dispositions(key, 0, updates)
            try:
                _publish_compact_row(
                    req, row, publication, old_full_locations, record.boundary
                )
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
            publication = runtime.relocate_tokens(
                key,
                0,
                updates,
                RelocationPolicy(
                    policy.maximum_source_pages,
                    policy.evacuation_headroom_pages,
                    policy.fragmentation_threshold_milli,
                    True,
                ),
                _copy_callback(batch),
                int(getattr(batch.device, "index", 0) or 0) + 65_537,
                record.boundary,
            )
            try:
                _publish_compact_row(
                    req, row, publication, old_full_locations, record.boundary
                )
                import torch

                torch.get_device_module(batch.device).current_stream(
                    batch.device
                ).synchronize()
                runtime.acknowledge_relocation(publication)
            except Exception as error:
                runtime.fail_stop(f"relocation mirror publication became uncertain: {error}")
                raise FailStopped(
                    runtime.failure_reason or "relocation mirror publication failed"
                ) from error
            _state._counter_add("relocation_batches")
            _state._counter_add("relocation_moves", len(publication.prepared.moves))
            _state._counter_add(
                "relocation_reclaimed_pages",
                publication.prepared.projected_reclaimed_pages,
            )
        else:
            raise RuntimeError("unknown token-reclamation mode")
        req._orbitkv_token_reclamation_done = True
        req.skip_radix_cache_insert = True
        _state._counter_add("token_disposition_batches")
        _state._counter_add("token_policy_evictions", len(updates))


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
        active, dtype=result.absolute_seq_lens.dtype, device=result.absolute_seq_lens.device
    )
    result.seq_lens_cpu = torch.tensor(active, dtype=torch.int64)
    result.seq_lens_sum = sum(active)
    return result


__all__ = ["_active_forward_lengths", "_maybe_reclaim_decode_batch"]
