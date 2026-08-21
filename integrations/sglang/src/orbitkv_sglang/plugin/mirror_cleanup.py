from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

from ..runtime import (
    DETACHED_CLEAR,
    DETACHED_COPY_ON_WRITE,
    DETACHED_REPLACE,
    DETACHED_RETENTION,
    DetachedBinding,
    MirrorCleanupItem,
    ReclamationCertificate,
    sglang_page_id,
)
from . import state as _state
from .state import _config, _runtime


def _sorted_membership(needles: Any, sorted_haystack: Any) -> Any:
    import torch

    if int(sorted_haystack.numel()) == 0:
        return torch.zeros_like(needles, dtype=torch.bool)
    contiguous_needles = needles.contiguous()
    positions = torch.searchsorted(sorted_haystack, contiguous_needles)
    valid = positions < int(sorted_haystack.numel())
    safe = positions.clamp(max=int(sorted_haystack.numel()) - 1)
    return valid & (sorted_haystack[safe] == contiguous_needles)


def _synchronize_mirror(req_to_token_pool: Any) -> None:
    device = getattr(req_to_token_pool, "device", None)
    if device is not None and str(device).startswith("cuda"):
        import torch

        torch.get_device_module(device).current_stream(device).synchronize()


@dataclass(frozen=True, slots=True)
class _MirrorCleanupContext:
    req: Any
    request_row: int


@dataclass(frozen=True, slots=True)
class _MirrorCleanupPlan:
    zero_views: tuple[Any, ...]
    mapping: Any | None
    mapping_indices: tuple[Any, ...]
    frontier_updates: tuple[tuple[Any, int], ...]


class _MirrorCleanupCoordinator:
    """One all-batch validation/mutation/sync boundary for SGLang mirrors."""

    def __init__(self, req_to_token_pool: Any, allocator: Any):
        self.req_to_token_pool = req_to_token_pool
        self.allocator = allocator

    def matches(self, req_to_token_pool: Any, allocator: Any) -> bool:
        return (
            self.req_to_token_pool is req_to_token_pool and self.allocator is allocator
        )

    def preflight(
        self,
        items: Sequence[MirrorCleanupItem],
        retirements: Sequence[ReclamationCertificate],
    ) -> _MirrorCleanupPlan:
        import torch

        values = tuple(items)
        certificates = tuple(retirements)
        if not values and not certificates:
            raise RuntimeError("mirror cleanup transaction must be nonempty")
        pool = self.req_to_token_pool
        table = getattr(pool, "req_to_token", None)
        if (
            type(table) is not torch.Tensor
            or table.ndim != 2
            or table.dtype is not torch.int32
        ):
            raise RuntimeError("SGLang ReqToToken mirror is not two-dimensional")
        try:
            pool_device = torch.device(pool.device)
        except Exception as error:
            raise RuntimeError("SGLang ReqToToken device is invalid") from error
        maximum = int(pool.max_context_len)
        if (
            maximum <= 0
            or maximum > int(table.shape[1])
            or table.device.type != pool_device.type
            or (
                pool_device.index is not None
                and table.device.index != pool_device.index
            )
        ):
            raise RuntimeError("SGLang ReqToToken capacity changed")

        config = _config()
        classes = config.classes_by_id
        full = config.full_class
        sliding = config.sliding_class
        primary = full or sliding
        hybrid = full is not None and sliding is not None
        mapping = self.allocator.full_to_swa_index_mapping if hybrid else None
        if hybrid and (
            type(mapping) is not torch.Tensor
            or mapping.ndim != 1
            or mapping.dtype is not torch.int64
            or mapping.device != table.device
            or int(mapping.numel()) <= 1
        ):
            raise RuntimeError("SGLang Full-to-SWA mapping changed")

        checks: list[Any] = []
        zero_views: list[Any] = []
        mapping_indices: list[Any] = []
        frontier_updates: list[tuple[Any, int]] = []
        rows: set[int] = set()
        retired_keys = {
            (
                item.page,
                item.logical_ordinal,
                item.token_begin,
                item.token_end_exclusive,
            )
            for item in certificates
        }
        covered_swa_keys = set()
        cold_alias_scan = False

        for item in values:
            context = item.context
            if not isinstance(context, _MirrorCleanupContext):
                raise RuntimeError("mirror cleanup lost its request context")
            req = context.req
            raw_row = getattr(req, "req_pool_idx", None)
            if isinstance(raw_row, bool) or not isinstance(raw_row, Integral):
                raise RuntimeError("ReqToToken cleanup row is not an integer")
            row = int(raw_row)
            if (
                row != context.request_row
                or not 0 < row < int(table.shape[0])
                or row in rows
            ):
                raise RuntimeError("ReqToToken cleanup names a dummy or aliased row")
            rows.add(row)
            if (
                isinstance(item.boundary, bool)
                or not isinstance(item.boundary, Integral)
                or not 0 <= int(item.boundary) <= maximum
            ):
                raise RuntimeError("manager cleanup boundary exceeds ReqToToken")
            boundary = int(item.boundary)
            mirror = table[row]
            prefix = getattr(req, "prefix_indices", None)
            if prefix is not None and (
                type(prefix) is not torch.Tensor
                or prefix.ndim != 1
                or prefix.dtype is not torch.int64
                or prefix.device != table.device
            ):
                raise RuntimeError("SGLang prefix mirror is not a device int64 vector")
            prefix_count = int(prefix.numel()) if prefix is not None else 0
            if prefix_count > boundary:
                raise RuntimeError("SGLang prefix mirror exceeds its KV boundary")
            retention_frontier = 0
            candidates = tuple(item.candidates)
            candidate_by_class_ordinal: dict[tuple[int, int], Any] = {}
            for candidate in candidates:
                self._validate_candidate(candidate, boundary, classes)
                key = (candidate.class_id, candidate.logical_ordinal)
                if key in candidate_by_class_ordinal:
                    raise RuntimeError("candidate mirror transition is duplicated")
                candidate_by_class_ordinal[key] = candidate

            cow_swa_sources: dict[
                tuple[Any, int, int, int], tuple[Any, Any]
            ] = {}
            retiring_primary_sources: set[tuple[Any, int, int, int]] = set()
            if hybrid and candidates:
                ordinals = {candidate.logical_ordinal for candidate in candidates}
                if any(
                    (full.class_id, ordinal) not in candidate_by_class_ordinal
                    or (sliding.class_id, ordinal)
                    not in candidate_by_class_ordinal
                    for ordinal in ordinals
                ):
                    raise RuntimeError("Hybrid candidate transition is not joint")
                for ordinal in ordinals:
                    full_candidate = candidate_by_class_ordinal[
                        (full.class_id, ordinal)
                    ]
                    swa_candidate = candidate_by_class_ordinal[
                        (sliding.class_id, ordinal)
                    ]
                    if (
                        full_candidate.token_begin != swa_candidate.token_begin
                        or full_candidate.token_end_exclusive
                        != swa_candidate.token_end_exclusive
                        or full_candidate.copied_token_begin
                        != swa_candidate.copied_token_begin
                        or full_candidate.copied_token_end_exclusive
                        != swa_candidate.copied_token_end_exclusive
                        or self._is_zero_page(full_candidate.source)
                        != self._is_zero_page(swa_candidate.source)
                        or full_candidate.retiring
                    ):
                        raise RuntimeError(
                            "Hybrid candidate transition changed joint ownership"
                        )
                    if self._is_zero_page(full_candidate.source):
                        continue
                    copied_begin = full_candidate.copied_token_begin
                    copied_end = full_candidate.copied_token_end_exclusive
                    full_detached = tuple(
                        detached
                        for detached in item.detached
                        if detached.class_id == full.class_id
                        and detached.action == DETACHED_REPLACE
                        and detached.reason == DETACHED_COPY_ON_WRITE
                        and detached.old == full_candidate.source
                        and detached.replacement == full_candidate.destination
                        and detached.old_backend_index
                        == full_candidate.source_backend_index
                        and detached.replacement_backend_index
                        == full_candidate.destination_backend_index
                        and detached.logical_ordinal == ordinal
                        and detached.token_begin == copied_begin
                        and detached.token_end_exclusive == copied_end
                    )
                    swa_detached = tuple(
                        detached
                        for detached in item.detached
                        if detached.class_id == sliding.class_id
                        and detached.old == swa_candidate.source
                        and detached.old_backend_index
                        == swa_candidate.source_backend_index
                        and detached.logical_ordinal == ordinal
                        and detached.token_begin == copied_begin
                        and detached.token_end_exclusive == copied_end
                    )
                    if len(full_detached) != 1 or len(swa_detached) != 1:
                        raise RuntimeError("COW candidate lost its detached source pair")
                    swa_transition = swa_detached[0]
                    valid_swa_transition = (
                        swa_candidate.retiring
                        and swa_transition.action == DETACHED_CLEAR
                        and swa_transition.reason == DETACHED_RETENTION
                    ) or (
                        not swa_candidate.retiring
                        and swa_transition.action == DETACHED_REPLACE
                        and swa_transition.reason == DETACHED_COPY_ON_WRITE
                        and swa_transition.replacement == swa_candidate.destination
                        and swa_transition.replacement_backend_index
                        == swa_candidate.destination_backend_index
                    )
                    if not valid_swa_transition:
                        raise RuntimeError("SWA COW detach differs from its candidate")
                    full_source = self._locations(
                        full.class_id,
                        full_candidate.source_backend_index,
                        copied_begin,
                        copied_end,
                    )
                    swa_source = self._locations(
                        sliding.class_id,
                        swa_candidate.source_backend_index,
                        copied_begin,
                        copied_end,
                    )
                    checks.append(
                        torch.all(mapping[full_source].to(torch.int64) == swa_source)
                    )
                    cow_swa_sources[
                        (
                            swa_candidate.source,
                            ordinal,
                            copied_begin,
                            copied_end,
                        )
                    ] = (full_source, swa_source)
            elif candidates:
                for candidate in candidates:
                    if self._is_zero_page(candidate.source):
                        continue
                    source_key = (
                        candidate.source,
                        candidate.logical_ordinal,
                        candidate.copied_token_begin,
                        candidate.copied_token_end_exclusive,
                    )
                    matches = tuple(
                        detached
                        for detached in item.detached
                        if detached.class_id == candidate.class_id
                        and detached.old == candidate.source
                        and detached.old_backend_index
                        == candidate.source_backend_index
                        and detached.logical_ordinal == candidate.logical_ordinal
                        and detached.token_begin == candidate.copied_token_begin
                        and detached.token_end_exclusive
                        == candidate.copied_token_end_exclusive
                    )
                    if len(matches) != 1:
                        raise RuntimeError("COW candidate lost its detached source")
                    transition = matches[0]
                    valid = (
                        candidate.retiring
                        and transition.action == DETACHED_CLEAR
                        and transition.reason == DETACHED_RETENTION
                    ) or (
                        not candidate.retiring
                        and transition.action == DETACHED_REPLACE
                        and transition.reason == DETACHED_COPY_ON_WRITE
                        and transition.replacement == candidate.destination
                        and transition.replacement_backend_index
                        == candidate.destination_backend_index
                    )
                    if not valid:
                        raise RuntimeError("COW candidate detach changed identity")
                    if candidate.retiring:
                        retiring_primary_sources.add(source_key)

            for detached in item.detached:
                self._validate_detached(detached, boundary, classes)
                begin = detached.token_begin
                end = detached.token_end_exclusive
                count = end - begin
                detached_key = (
                    detached.old,
                    detached.logical_ordinal,
                    begin,
                    end,
                )
                old_locations = self._locations(
                    detached.class_id,
                    detached.old_backend_index,
                    begin,
                    end,
                )
                replacement_locations = (
                    self._locations(
                        detached.class_id,
                        detached.replacement_backend_index,
                        begin,
                        end,
                    )
                    if detached.action == DETACHED_REPLACE
                    else None
                )

                if primary is not None and detached.class_id == primary.class_id:
                    if detached_key not in retiring_primary_sources:
                        target = mirror[begin:end].to(dtype=torch.int64)
                        expected = (
                            old_locations
                            if detached.action == DETACHED_CLEAR
                            else replacement_locations
                        )
                        assert expected is not None and int(expected.numel()) == count
                        checks.append(torch.all(target == expected))
                        if detached.action == DETACHED_CLEAR:
                            zero_views.append(mirror[begin:end])
                        prefix_begin = min(begin, prefix_count)
                        prefix_end = min(end, prefix_count)
                        if prefix_begin < prefix_end:
                            prefix_view = prefix[prefix_begin:prefix_end]
                            offset = prefix_begin - begin
                            prefix_expected = expected[
                                offset : offset + (prefix_end - prefix_begin)
                            ]
                            checks.append(
                                torch.all(
                                    prefix_view.to(dtype=torch.int64)
                                    == prefix_expected
                                )
                            )
                            if detached.action == DETACHED_CLEAR:
                                zero_views.append(prefix_view)
                elif hybrid and detached.class_id == sliding.class_id:
                    cow_source = cow_swa_sources.get(detached_key)
                    if cow_source is not None:
                        safe_locations, expected = cow_source
                    else:
                        full_locations = mirror[begin:end].to(dtype=torch.int64)
                        valid = (full_locations > 0) & (
                            full_locations < int(mapping.numel())
                        )
                        checks.append(torch.all(valid))
                        safe_locations = full_locations.clamp(
                            min=0, max=int(mapping.numel()) - 1
                        )
                        expected = (
                            old_locations
                            if detached.action == DETACHED_CLEAR
                            else replacement_locations
                        )
                        assert expected is not None
                        checks.append(
                            torch.all(
                                mapping[safe_locations].to(dtype=torch.int64)
                                == expected
                            )
                        )
                    if (
                        detached.action == DETACHED_CLEAR
                        and detached_key in retired_keys
                    ):
                        mapping_indices.append(safe_locations)
                        covered_swa_keys.add(detached_key)
                if (
                    sliding is not None
                    and detached.class_id == sliding.class_id
                    and detached.action == DETACHED_CLEAR
                    and detached.reason == DETACHED_RETENTION
                ):
                    retention_frontier = max(retention_frontier, end)

            if candidates:
                if hybrid:
                    for ordinal in {
                        candidate.logical_ordinal for candidate in candidates
                    }:
                        full_candidate = candidate_by_class_ordinal[
                            (full.class_id, ordinal)
                        ]
                        swa_candidate = candidate_by_class_ordinal[
                            (sliding.class_id, ordinal)
                        ]
                        begin = full_candidate.token_begin
                        end = full_candidate.token_end_exclusive
                        full_destination = self._locations(
                            full.class_id,
                            full_candidate.destination_backend_index,
                            begin,
                            end,
                        )
                        swa_destination = self._locations(
                            sliding.class_id,
                            swa_candidate.destination_backend_index,
                            begin,
                            end,
                        )
                        row_view = mirror[begin:end].to(dtype=torch.int64)
                        checks.append(torch.all(row_view == full_destination))
                        prefix_begin = min(begin, prefix_count)
                        prefix_end = min(end, prefix_count)
                        if prefix_begin < prefix_end:
                            offset = prefix_begin - begin
                            checks.append(
                                torch.all(
                                    prefix[prefix_begin:prefix_end].to(torch.int64)
                                    == full_destination[
                                        offset : offset + prefix_end - prefix_begin
                                    ]
                                )
                            )
                        checks.append(
                            torch.all(
                                mapping[full_destination].to(torch.int64)
                                == swa_destination
                            )
                        )
                        full_key = (
                            full_candidate.destination,
                            ordinal,
                            begin,
                            end,
                        )
                        swa_key = (
                            swa_candidate.destination,
                            ordinal,
                            begin,
                            end,
                        )
                        if full_key in retired_keys:
                            raise RuntimeError("Full candidate retired before publication")
                        if swa_candidate.retiring != (swa_key in retired_keys):
                            raise RuntimeError(
                                "SWA candidate retirement authority changed"
                            )
                        if swa_candidate.retiring:
                            mapping_indices.append(full_destination)
                            covered_swa_keys.add(swa_key)
                            retention_frontier = max(retention_frontier, end)
                else:
                    for candidate in candidates:
                        begin = candidate.token_begin
                        end = candidate.token_end_exclusive
                        destination = self._locations(
                            candidate.class_id,
                            candidate.destination_backend_index,
                            begin,
                            end,
                        )
                        row_view = mirror[begin:end]
                        checks.append(
                            torch.all(row_view.to(torch.int64) == destination)
                        )
                        candidate_key = (
                            candidate.destination,
                            candidate.logical_ordinal,
                            begin,
                            end,
                        )
                        if candidate.retiring != (candidate_key in retired_keys):
                            raise RuntimeError(
                                "candidate retirement authority changed"
                            )
                        if candidate.retiring:
                            if full is not None:
                                raise RuntimeError(
                                    "Full candidate retired before publication"
                                )
                            zero_views.append(row_view)
                            prefix_begin = min(begin, prefix_count)
                            prefix_end = min(end, prefix_count)
                            if prefix_begin < prefix_end:
                                zero_views.append(prefix[prefix_begin:prefix_end])
                            retention_frontier = max(retention_frontier, end)

            if retention_frontier:
                kv = getattr(req, "kv", None)
                if kv is None:
                    raise RuntimeError("SWA retention cleanup lost request KV metadata")
                allocated = getattr(kv, "kv_allocated_len", None)
                current = getattr(kv, "swa_evicted_seqlen", None)
                if (
                    isinstance(allocated, bool)
                    or not isinstance(allocated, Integral)
                    or isinstance(current, bool)
                    or not isinstance(current, Integral)
                    or int(allocated) != boundary
                    or not 0 <= int(current) <= retention_frontier <= boundary
                ):
                    raise RuntimeError("SWA retention frontier is invalid")
                frontier_updates.append((kv, max(int(current), retention_frontier)))

        if hybrid:
            full_groups: dict[tuple[int, int, int], list[Any]] = {}
            swa_groups: dict[tuple[int, int, int], list[Any]] = {}
            for certificate in certificates:
                span = (
                    certificate.logical_ordinal,
                    certificate.token_begin,
                    certificate.token_end_exclusive,
                )
                if certificate.class_id == full.class_id:
                    full_groups.setdefault(span, []).append(certificate)
                elif certificate.class_id == sliding.class_id:
                    swa_groups.setdefault(span, []).append(certificate)
                else:
                    raise RuntimeError("reclamation names an unknown KV class")

            all_full_locations: list[Any] = []
            all_swa_locations: list[Any] = []
            for span in full_groups.keys() | swa_groups.keys():
                _ordinal, begin, end = span
                count = end - begin
                full_certificates = full_groups.get(span, ())
                swa_certificates = swa_groups.get(span, ())
                full_locations = tuple(
                    self._locations(
                        full.class_id,
                        certificate.backend_index,
                        begin,
                        end,
                    )
                    for certificate in full_certificates
                )
                swa_locations = tuple(
                    self._locations(
                        sliding.class_id,
                        certificate.backend_index,
                        begin,
                        end,
                    )
                    for certificate in swa_certificates
                )
                all_full_locations.extend(full_locations)
                all_swa_locations.extend(swa_locations)
                mapping_indices.extend(full_locations)

                swa_first = torch.tensor(
                    [
                        self._location_start(
                            sliding.class_id,
                            certificate.backend_index,
                            begin,
                        )
                        for certificate in swa_certificates
                    ],
                    dtype=torch.int64,
                    device=mapping.device,
                )
                if int(swa_first.numel()) > 1:
                    ordered_swa = torch.sort(swa_first).values
                    checks.append(torch.all(ordered_swa[1:] != ordered_swa[:-1]))
                else:
                    ordered_swa = swa_first

                if full_locations:
                    mapped = torch.stack(
                        tuple(mapping[locations] for locations in full_locations)
                    ).to(dtype=torch.int64)
                    first = mapped[:, 0]
                    zero = torch.all(mapped == 0, dim=1)
                    offsets = torch.arange(
                        count, dtype=torch.int64, device=mapping.device
                    )
                    contiguous = torch.all(
                        mapped == first[:, None] + offsets[None, :], dim=1
                    )
                    known_swa = _sorted_membership(first, ordered_swa)
                    correct_offset = (
                        first.remainder(config.page_tokens)
                        == begin % config.page_tokens
                    )
                    checks.append(
                        torch.all(zero | (contiguous & known_swa & correct_offset))
                    )
                    if int(first.numel()) > 1:
                        ordered_full = torch.sort(first).values
                        checks.append(
                            torch.all(
                                (ordered_full[:-1] == 0)
                                | (ordered_full[1:] != ordered_full[:-1])
                            )
                        )
                    else:
                        ordered_full = first
                else:
                    first = torch.empty(
                        (0,), dtype=torch.int64, device=mapping.device
                    )
                    ordered_full = first

                if swa_certificates:
                    mapped_coverage = _sorted_membership(
                        swa_first, ordered_full
                    )
                    detached_coverage = torch.tensor(
                        [
                            (
                                certificate.page,
                                certificate.logical_ordinal,
                                certificate.token_begin,
                                certificate.token_end_exclusive,
                            )
                            in covered_swa_keys
                            for certificate in swa_certificates
                        ],
                        dtype=torch.bool,
                        device=mapping.device,
                    )
                    checks.append(torch.all(mapped_coverage | detached_coverage))

            if not values and all_swa_locations:
                # Cold prefix eviction has no request row from which to recover
                # reverse ownership.  One aggregate full-table scan proves that
                # no non-retiring Full location still aliases a retiring SWA page.
                retiring_swa = torch.cat(tuple(all_swa_locations))
                actual_aliases = torch.isin(mapping, retiring_swa)
                expected_aliases = torch.zeros_like(actual_aliases)
                if all_full_locations:
                    retiring_full = torch.cat(tuple(all_full_locations))
                    expected_aliases[retiring_full] = mapping[retiring_full] != 0
                checks.append(torch.all(actual_aliases == expected_aliases))
                cold_alias_scan = True

        verified = (
            torch.stack(tuple(check.reshape(()) for check in checks)).all()
            if checks
            else torch.ones((), dtype=torch.bool, device=table.device)
        )
        if not torch.equal(
            verified,
            torch.ones((), dtype=torch.bool, device=table.device),
        ):
            raise RuntimeError("DetachedBinding disagrees with the SGLang mirror")
        _state._counter_add("mirror_validation_calls")
        if cold_alias_scan:
            _state._counter_add("prefix_global_alias_scans")
        return _MirrorCleanupPlan(
            tuple(zero_views),
            mapping,
            tuple(mapping_indices),
            tuple(frontier_updates),
        )

    def commit(self, plan: _MirrorCleanupPlan) -> None:
        for target in plan.zero_views:
            target.zero_()
        if plan.mapping is not None:
            for indices in plan.mapping_indices:
                plan.mapping[indices] = 0

    def synchronize(self, _plan: _MirrorCleanupPlan) -> None:
        _synchronize_mirror(self.req_to_token_pool)
        _state._counter_add("mirror_syncs")

    @staticmethod
    def finalize(plan: _MirrorCleanupPlan) -> None:
        for state, frontier in plan.frontier_updates:
            state.swa_evicted_seqlen = frontier

    @staticmethod
    def _validate_detached(
        detached: DetachedBinding, boundary: int, classes: dict[int, Any]
    ) -> None:
        if detached.class_id not in classes:
            raise RuntimeError("DetachedBinding names an unknown KV class")
        arena = _runtime().arenas_by_class[detached.class_id]
        if (
            detached.backend_domain != arena.backend_domain
            or detached.reserved != 0
            or detached.logical_ordinal < 0
            or detached.token_begin
            != detached.logical_ordinal * _config().page_tokens
            or not detached.token_begin < detached.token_end_exclusive
            or detached.token_end_exclusive
            > min(boundary, (detached.logical_ordinal + 1) * _config().page_tokens)
            or detached.action not in (DETACHED_CLEAR, DETACHED_REPLACE)
        ):
            raise RuntimeError("DetachedBinding is not a valid mirror transition")

    def _validate_candidate(
        self, candidate: Any, boundary: int, classes: dict[int, Any]
    ) -> None:
        if candidate.class_id not in classes:
            raise RuntimeError("candidate transition names an unknown KV class")
        arena = _runtime().arenas_by_class[candidate.class_id]
        begin = candidate.token_begin
        end = candidate.token_end_exclusive
        copied_begin = candidate.copied_token_begin
        copied_end = candidate.copied_token_end_exclusive
        if (
            candidate.backend_domain != arena.backend_domain
            or candidate.reserved != 0
            or type(candidate.retiring) is not bool
            or candidate.logical_ordinal < 0
            or begin != candidate.logical_ordinal * _config().page_tokens
            or not begin < end <= min(
                boundary, begin + _config().page_tokens
            )
            or self._is_zero_page(candidate.destination)
            or candidate.destination == candidate.source
            or not self._page_matches_backend(
                candidate.destination,
                candidate.destination_backend_index,
                arena,
            )
        ):
            raise RuntimeError("candidate mirror transition is invalid")
        if self._is_zero_page(candidate.source):
            if (
                candidate.source_backend_index != 0
                or copied_begin != 0
                or copied_end != 0
            ):
                raise RuntimeError("fresh candidate carries COW source authority")
        elif (
            not self._page_matches_backend(
                candidate.source, candidate.source_backend_index, arena
            )
            or not begin <= copied_begin < copied_end <= end
        ):
            raise RuntimeError("COW candidate source authority is invalid")

    @staticmethod
    def _page_matches_backend(page: Any, backend_index: int, arena: Any) -> bool:
        return (
            page.engine_epoch == arena.engine_epoch
            and page.pool_epoch == arena.pool_epoch
            and page.pool_id == arena.pool_id
            and page.generation > 0
            and backend_index
            == arena.backend_base_index + page.page_id - arena.first_page_id
            and arena.first_page_id
            <= page.page_id
            < arena.first_page_id + arena.page_count
        )

    @staticmethod
    def _is_zero_page(page: Any) -> bool:
        return (
            page.engine_epoch == 0
            and page.pool_epoch == 0
            and page.pool_id == 0
            and page.page_id == 0
            and page.generation == 0
        )

    def _locations(
        self, class_id: int, backend_index: int, begin: int, end: int
    ) -> Any:
        import torch

        start = self._location_start(class_id, backend_index, begin)
        return torch.arange(
            start,
            start + (end - begin),
            dtype=torch.int64,
            device=self.req_to_token_pool.device,
        )

    @staticmethod
    def _location_start(class_id: int, backend_index: int, begin: int) -> int:
        arena = _runtime().arenas_by_class[class_id]
        page = sglang_page_id(backend_index, arena.backend_base_index)
        return page * _config().page_tokens + begin % _config().page_tokens


def _mirror_cleanup_coordinator(
    req_to_token_pool: Any, allocator: Any
) -> _MirrorCleanupCoordinator:
    if _state._MIRROR_CLEANUP is None:
        _state._MIRROR_CLEANUP = _MirrorCleanupCoordinator(req_to_token_pool, allocator)
    elif not _state._MIRROR_CLEANUP.matches(req_to_token_pool, allocator):
        raise RuntimeError("SGLang reclamation mirror authority changed")
    return _state._MIRROR_CLEANUP
