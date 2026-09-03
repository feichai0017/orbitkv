from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

from ..runtime import (
    DETACHED_CLEAR,
    DETACHED_COPY_ON_WRITE,
    DETACHED_PREFIX_TRANSFER,
    DETACHED_REPLACE,
    DETACHED_REQUEST_RELEASE,
    DETACHED_RETENTION,
    DetachedBinding,
    MirrorCleanupItem,
    ReclamationCertificate,
    sglang_page_id,
)
from . import state as _state
from .mirror_cleanup_support import (
    MirrorCleanupContext as _MirrorCleanupContext,
    MirrorCleanupPlan as _MirrorCleanupPlan,
    is_zero_page as _is_zero_page,
    page_matches_backend as _page_matches_backend,
    preflight_empty_publication as _preflight_empty_publication,
    retirement_identities as _retirement_identities,
    sorted_membership as _sorted_membership,
    synchronize_mirror as _synchronize_mirror,
    validate_session_context_count as _validate_session_context_count,
    validate_session_hybrid_release as _validate_session_hybrid_release,
    validate_session_retirement_sources as _validate_session_retirement_sources,
)
from .private_prefix import (
    ENGINE_PREFIX_ID_MARKER,
    ENGINE_REQUEST_ID_MARKER,
    PRIVATE_PREFIX_MARKER,
    SHARED_PREFIX_MARKERS,
    PrivatePrefixProvenance,
    clear_private_prefix,
    request_identity,
    validate_private_prefix,
    validate_shared_prefix_metadata,
)
from .state import _config, _runtime


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
        chunked = getattr(config, "chunked_class", None)
        primary = config.primary_class
        hybrid = full is not None and sliding is not None
        session_hybrid_profile = hybrid and _state._uses_runtime_session()
        session_sliding_profile = (
            full is None
            and sliding is not None
            and len(classes) == 1
            and _state._uses_runtime_session()
        )
        session_chunked_profile = (
            chunked is not None
            and len(classes) == 1
            and _state._uses_runtime_session()
        )
        mapping = self.allocator.full_to_swa_index_mapping if hybrid else None
        if hybrid and (
            type(mapping) is not torch.Tensor
            or mapping.ndim != 1
            or mapping.dtype is not torch.int64
            or mapping.device != table.device
            or int(mapping.numel()) <= 1
        ):
            raise RuntimeError("SGLang Full-to-SWA mapping changed")
        fast_plan = self._preflight_full_batch(
            values, certificates, table, maximum, full, sliding
        )
        if fast_plan is not None:
            _state._counter_add("mirror_validation_calls")
            return fast_plan

        checks: list[Any] = []
        zero_views: list[Any] = []
        mapping_indices: list[Any] = []
        frontier_updates: list[tuple[Any, int, int]] = []
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
        compact_hybrid_releases = 0
        private_prefixes: list[tuple[Any, PrivatePrefixProvenance]] = []
        session_release_reasons: set[int] = set()
        session_release_detached: set[tuple[Any, ...]] = set()
        session_retirement_sources: set[tuple[Any, ...]] = set()
        session_hybrid_items = 0
        session_sliding_items = 0
        session_chunked_items = 0

        for item in values:
            req, row = self._validate_context(item.context, table, rows)
            if (
                isinstance(item.boundary, bool)
                or not isinstance(item.boundary, Integral)
                or not 0 <= int(item.boundary) <= maximum
            ):
                raise RuntimeError("manager cleanup boundary exceeds ReqToToken")
            boundary = int(item.boundary)
            mirror = table[row]
            has_session_context = (
                item.context.request_key is not None
                or item.context.request_id is not None
                or hasattr(req, "_orbitkv_request_key")
                or hasattr(req, ENGINE_REQUEST_ID_MARKER)
            )
            session_hybrid = session_hybrid_profile and has_session_context
            session_hybrid_items += int(session_hybrid)
            session_sliding = session_sliding_profile and has_session_context
            session_sliding_items += int(session_sliding)
            session_chunked = session_chunked_profile and has_session_context
            session_chunked_items += int(session_chunked)
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
            private_prefix = None
            if session_hybrid:
                if (
                    item.context.request_key is None
                    or item.context.request_id is None
                ):
                    raise RuntimeError(
                        "runtime-session Hybrid cleanup lost request identity"
                    )
                private_prefix = self._validate_session_full_prefix(
                    req, mirror, item.context, boundary
                )
                if item.releasing:
                    reason, identities = _validate_session_hybrid_release(
                        item, boundary, full, sliding, classes=classes,
                        arenas=_runtime().arenas_by_class,
                        page_tokens=int(config.page_tokens),
                        validate_detached=self._validate_detached,
                        page_matches_backend=_page_matches_backend,
                        is_zero_page=_is_zero_page,
                    )
                    if reason is not None:
                        session_release_reasons.add(reason)
                    session_release_detached.update(identities)
                    if private_prefix is not None:
                        private_prefixes.append((req, private_prefix))
            elif _state._requires_disabled_radix_cache():
                private_prefix = validate_private_prefix(
                    req,
                    mirror,
                    getattr(req, "_orbitkv_request_key", None),
                    request_identity(req),
                    boundary,
                )
                if item.releasing and private_prefix is not None:
                    private_prefixes.append((req, private_prefix))
                    zero_views.append(prefix)
                if (session_sliding or session_chunked) and (
                    item.context.request_key is None
                    or item.context.request_id is None
                ):
                    raise RuntimeError(
                        "runtime-session local cleanup lost request identity"
                    )
            retention_frontier = 0
            chunked_epoch_end = (
                chunked is not None
                and not item.releasing
                and boundary > 0
                and boundary % int(chunked.chunk_tokens) == 0
            )
            compact_locations = getattr(req, "_orbitkv_retained_locations", None)
            compact_swa_locations = getattr(
                req, "_orbitkv_retained_swa_locations", None
            )
            compact_release = compact_locations is not None and item.releasing
            if compact_release:
                if (
                    (prefix_count != 0 and private_prefix is None)
                    or not isinstance(compact_locations, tuple)
                    or not compact_locations
                    or len(compact_locations) > boundary
                    or any(
                        isinstance(location, bool)
                        or not isinstance(location, Integral)
                        or int(location) <= 0
                        for location in compact_locations
                    )
                ):
                    raise RuntimeError("compact release metadata is invalid")
                if private_prefix is not None and prefix_count > len(compact_locations):
                    raise RuntimeError("compact private-prefix length is invalid")
                compact_view = mirror[: len(compact_locations)]
                expected_compact = torch.tensor(
                    compact_locations, dtype=torch.int64, device=table.device
                )
                checks.append(
                    torch.all(compact_view.to(torch.int64) == expected_compact)
                )
                zero_views.append(compact_view)
                if hybrid:
                    if (
                        not isinstance(compact_swa_locations, tuple)
                        or len(compact_swa_locations) != len(compact_locations)
                    ):
                        raise RuntimeError("compact SWA release metadata is invalid")
                    full_tensor = expected_compact
                    swa_tensor = torch.tensor(
                        compact_swa_locations,
                        dtype=torch.int64,
                        device=table.device,
                    )
                    checks.append(
                        torch.all(mapping[full_tensor].to(torch.int64) == swa_tensor)
                    )
                    mapping_indices.append(full_tensor)
                    compact_hybrid_releases += 1
            candidates = tuple(item.candidates)
            candidate_by_class_ordinal: dict[tuple[int, int], Any] = {}
            for candidate in candidates:
                self._validate_candidate(candidate, boundary, classes)
                key = (candidate.class_id, candidate.logical_ordinal)
                if key in candidate_by_class_ordinal:
                    raise RuntimeError("candidate mirror transition is duplicated")
                candidate_by_class_ordinal[key] = candidate
                if (
                    session_hybrid or session_sliding or session_chunked
                ) and candidate.retiring:
                    session_retirement_sources.add((
                        candidate.destination, candidate.class_id,
                        candidate.backend_domain, candidate.logical_ordinal,
                        candidate.destination_backend_index,
                        candidate.token_begin, candidate.token_end_exclusive,
                    ))

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
                        or _is_zero_page(full_candidate.source)
                        != _is_zero_page(swa_candidate.source)
                        or full_candidate.retiring
                    ):
                        raise RuntimeError(
                            "Hybrid candidate transition changed joint ownership"
                        )
                    if _is_zero_page(full_candidate.source):
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
                    if _is_zero_page(candidate.source):
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
                if session_hybrid or session_sliding or session_chunked:
                    session_retirement_sources.add((
                        detached.old, detached.class_id, detached.backend_domain,
                        detached.logical_ordinal, detached.old_backend_index,
                        begin, end,
                    ))
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
                    if compact_release:
                        continue
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
                    if chunked is not None and not item.releasing:
                        if (
                            detached.class_id != chunked.class_id
                            or detached.action != DETACHED_CLEAR
                            or detached.reason != DETACHED_RETENTION
                            or not chunked_epoch_end
                            or not (
                                boundary - int(chunked.chunk_tokens)
                                <= begin
                                < end
                                <= boundary
                            )
                        ):
                            raise RuntimeError(
                                "chunked EpochEnd detach authority changed"
                            )
                elif hybrid and detached.class_id == sliding.class_id:
                    if compact_release:
                        continue
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

            if session_sliding and item.releasing:
                expected_spans = {
                    (ordinal, ordinal * int(config.page_tokens), min(
                        (ordinal + 1) * int(config.page_tokens), boundary
                    ))
                    for ordinal in range(
                        max(0, boundary - (int(sliding.window_tokens) - 1))
                        // int(config.page_tokens),
                        (boundary + int(config.page_tokens) - 1)
                        // int(config.page_tokens),
                    )
                }
                actual_spans = {
                    (entry.logical_ordinal, entry.token_begin, entry.token_end_exclusive)
                    for entry in item.detached
                    if entry.class_id == sliding.class_id
                    and entry.action == DETACHED_CLEAR
                    and entry.reason == DETACHED_REQUEST_RELEASE
                    and _is_zero_page(entry.replacement)
                    and entry.replacement_backend_index == 0
                }
                if len(item.detached) != len(expected_spans) or actual_spans != expected_spans:
                    raise RuntimeError(
                        "runtime-session Sliding release coverage changed"
                    )

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
                            if chunked is not None and not chunked_epoch_end:
                                raise RuntimeError(
                                    "chunked candidate retired outside EpochEnd"
                                )
                            zero_views.append(row_view)
                            prefix_begin = min(begin, prefix_count)
                            prefix_end = min(end, prefix_count)
                            if prefix_begin < prefix_end:
                                zero_views.append(prefix[prefix_begin:prefix_end])
                            if sliding is not None:
                                retention_frontier = max(retention_frontier, end)

            if chunked_epoch_end:
                epoch_begin = boundary - int(chunked.chunk_tokens)
                if any(not entry.retiring for entry in candidates):
                    raise RuntimeError(
                        "chunked EpochEnd retained a transient candidate"
                    )
                detached_spans = {
                    (entry.token_begin, entry.token_end_exclusive)
                    for entry in item.detached
                    if entry.class_id == chunked.class_id
                    and entry.action == DETACHED_CLEAR
                    and entry.reason == DETACHED_RETENTION
                    and (
                        entry.old,
                        entry.logical_ordinal,
                        entry.token_begin,
                        entry.token_end_exclusive,
                    )
                    not in retiring_primary_sources
                }
                candidate_spans = {
                    (entry.token_begin, entry.token_end_exclusive)
                    for entry in candidates
                    if entry.class_id == chunked.class_id and entry.retiring
                }
                cursor = epoch_begin
                for begin, end in sorted(detached_spans | candidate_spans):
                    if begin != cursor:
                        raise RuntimeError(
                            "chunked EpochEnd does not cover the exact old epoch"
                        )
                    cursor = end
                if cursor != boundary:
                    raise RuntimeError(
                        "chunked EpochEnd does not cover the exact old epoch"
                    )
            elif chunked is not None and any(
                entry.reason == DETACHED_RETENTION
                for entry in item.detached
            ):
                raise RuntimeError(
                    "chunked retention detach occurred outside EpochEnd"
                )

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
                frontier_updates.append(
                    (kv, int(current), max(int(current), retention_frontier))
                )

        session_retirement_identities = _retirement_identities(certificates)
        _validate_session_context_count(
            "Hybrid", session_hybrid_items, len(values)
        )
        _validate_session_context_count(
            "Sliding", session_sliding_items, len(values)
        )
        _validate_session_context_count(
            "Chunked", session_chunked_items, len(values)
        )
        if session_sliding_items:
            _validate_session_retirement_sources(
                "Sliding", session_retirement_identities,
                session_retirement_sources,
            )
        if session_hybrid_items:
            _validate_session_retirement_sources(
                "Hybrid", session_retirement_identities,
                session_retirement_sources,
            )
        if session_chunked_profile and (session_chunked_items or certificates):
            _validate_session_retirement_sources(
                "Chunked", session_retirement_identities,
                session_retirement_sources,
            )
        if session_hybrid_items and any(item.releasing for item in values):
            if not all(item.releasing for item in values) or len(session_release_reasons) > 1:
                raise RuntimeError(
                    "runtime-session Hybrid cleanup mixed release states"
                )
            if session_release_reasons == {DETACHED_PREFIX_TRANSFER}:
                if certificates:
                    raise RuntimeError(
                        "prefix-transfer release cannot carry retirement certificates"
                    )
            elif (
                not set(session_retirement_identities).issubset(
                    session_release_detached
                )
            ):
                raise RuntimeError(
                    "runtime-session Hybrid release retirement coverage changed"
                )

        if hybrid and compact_hybrid_releases not in (0, len(values)):
            raise RuntimeError(
                "compact and dense Hybrid releases require separate cleanup batches"
            )
        if hybrid and compact_hybrid_releases == 0:
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
                        torch.all(
                            zero
                            | (
                                contiguous
                                & correct_offset
                                & (True if session_hybrid_items else known_swa)
                            )
                        )
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
            private_prefixes=tuple(private_prefixes),
        )

    def commit(self, plan: _MirrorCleanupPlan) -> None:
        for target in plan.zero_views:
            target.zero_()
        for target, indices in plan.indexed_zeroes:
            target[indices] = 0
        if plan.mapping is not None:
            for indices in plan.mapping_indices:
                plan.mapping[indices] = 0

    def synchronize(self, plan: _MirrorCleanupPlan) -> None:
        if plan.requires_device_sync:
            _synchronize_mirror(self.req_to_token_pool)
            _state._counter_add("mirror_syncs")
        plan.synchronized = True

    @staticmethod
    def finalize(plan: _MirrorCleanupPlan) -> None:
        if plan.frontier_updates and not plan.synchronized:
            raise RuntimeError(
                "SWA retention frontier cannot advance before cleanup sync"
            )
        for state, expected, frontier in plan.frontier_updates:
            current = getattr(state, "swa_evicted_seqlen", None)
            if (
                isinstance(current, bool)
                or not isinstance(current, Integral)
                or int(current) != expected
            ):
                raise RuntimeError(
                    "SWA retention frontier changed before finalization"
                )
            state.swa_evicted_seqlen = frontier
        for req, marker in plan.private_prefixes:
            clear_private_prefix(req, marker)

    def _preflight_full_batch(
        self,
        values: tuple[MirrorCleanupItem, ...],
        certificates: tuple[ReclamationCertificate, ...],
        table: Any,
        maximum: int,
        full: Any | None,
        sliding: Any | None,
    ) -> _MirrorCleanupPlan | None:
        """Aggregate the strict Full-only fresh/release mirror profile."""

        import torch

        session_full = _state._uses_runtime_session()
        if (
            full is None
            or sliding is not None
            or _state._requires_disabled_radix_cache() and not session_full
            or len(_config().classes_by_id) != 1
            or full.storage != "token_kv"
            or not values
        ):
            return None
        empty_plan = (
            _preflight_empty_publication(
                values, certificates, table, maximum, self._validate_context
            )
            if session_full
            else None
        )
        if empty_plan is not None:
            return empty_plan
        if not table.is_contiguous():
            return None
        if session_full:
            if all(
                item.releasing and not item.candidates for item in values
            ):
                fresh = False
                releasing = True
            elif all(
                not item.releasing and not item.candidates for item in values
            ):
                return None
            else:
                raise RuntimeError(
                    "request-private Full cleanup mixed release states"
                )
        else:
            fresh = all(
                not item.releasing
                and not item.detached
                and bool(item.candidates)
                for item in values
            )
            releasing = all(
                item.releasing
                and not item.candidates
                and bool(item.detached)
                for item in values
            )
        if not (fresh or releasing) or fresh and certificates:
            return None
        if fresh and any(
            candidate.class_id != full.class_id
            or not _is_zero_page(candidate.source)
            or candidate.retiring
            or getattr(item.context.req, "_orbitkv_retained_locations", None)
            is not None
            for item in values
            for candidate in item.candidates
        ):
            return None
        release_reasons = (
            {
                detached.reason
                for item in values
                for detached in item.detached
            }
            if releasing
            else set()
        )
        if session_full and release_reasons and release_reasons not in (
            {DETACHED_REQUEST_RELEASE},
            {DETACHED_PREFIX_TRANSFER},
        ):
            raise RuntimeError(
                "runtime-session Full release has invalid or mixed detach reasons"
            )
        valid_release_reasons = (
            (DETACHED_REQUEST_RELEASE, DETACHED_PREFIX_TRANSFER)
            if session_full
            else (DETACHED_REQUEST_RELEASE,)
        )
        invalid_release = releasing and (
            bool(release_reasons.difference(valid_release_reasons))
            or any(
            detached.class_id != full.class_id
            or detached.action != DETACHED_CLEAR
            or not _is_zero_page(detached.replacement)
            or detached.replacement_backend_index != 0
            or getattr(item.context.req, "_orbitkv_retained_locations", None)
            is not None
            for item in values
            for detached in item.detached
            )
        )
        if invalid_release and session_full:
            raise RuntimeError(
                "request-private Full release transition is invalid"
            )
        if invalid_release:
            return None

        spans: list[tuple[int, int, int, int]] = []
        prefix_actual: list[Any] = []
        prefix_spans: list[tuple[int, int]] = []
        prefix_zeroes: list[Any] = []
        private_prefixes: list[
            tuple[Any, PrivatePrefixProvenance]
        ] = []
        rows: set[int] = set()
        width = int(table.shape[1])
        for item in values:
            req, row = self._validate_context(item.context, table, rows)
            if (
                isinstance(item.boundary, bool)
                or not isinstance(item.boundary, Integral)
                or not 0 <= int(item.boundary) <= maximum
            ):
                raise RuntimeError("manager cleanup boundary exceeds ReqToToken")
            boundary = int(item.boundary)
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
            private_prefix = None
            if session_full and releasing:
                private_prefix = self._validate_session_full_prefix(
                    req,
                    table[row],
                    item.context,
                    boundary,
                )

            transitions = item.candidates if fresh else item.detached
            ordinals: set[int] = set()
            pages: set[Any] = set()
            covered: list[tuple[int, int]] = []
            for transition in transitions:
                if fresh:
                    self._validate_candidate(
                        transition, boundary, {full.class_id: full}
                    )
                    backend_index = transition.destination_backend_index
                else:
                    self._validate_detached(transition, boundary, {full.class_id: full})
                    backend_index = transition.old_backend_index
                    if session_full and not _page_matches_backend(
                        transition.old,
                        backend_index,
                        _runtime().arenas_by_class[full.class_id],
                    ):
                        raise RuntimeError(
                            "request-private Full release page identity is invalid"
                        )
                if (
                    transition.logical_ordinal in ordinals
                    or not fresh
                    and transition.old in pages
                ):
                    raise RuntimeError("Full mirror transition is duplicated")
                ordinals.add(transition.logical_ordinal)
                if not fresh:
                    pages.add(transition.old)
                begin = transition.token_begin
                end = transition.token_end_exclusive
                covered.append((begin, end))
                start = self._location_start(full.class_id, backend_index, begin)
                spans.append((row, begin, end, start))
                prefix_end = min(end, prefix_count)
                if begin < prefix_end:
                    assert prefix is not None
                    prefix_actual.append(prefix[begin:prefix_end])
                    prefix_spans.append((start, prefix_end - begin))
            if releasing:
                cursor = 0
                for begin, end in sorted(covered):
                    if begin != cursor:
                        if session_full:
                            raise RuntimeError(
                                "request-private Full release has incomplete coverage"
                            )
                        return None
                    cursor = end
                if cursor != boundary:
                    if session_full:
                        raise RuntimeError(
                            "request-private Full release has incomplete coverage"
                        )
                    return None
                if prefix_count:
                    assert prefix is not None
                    prefix_zeroes.append(prefix[:prefix_count])
                if private_prefix is not None:
                    private_prefixes.append((req, private_prefix))

        prefix_transfer = (
            session_full
            and releasing
            and release_reasons == {DETACHED_PREFIX_TRANSFER}
        )
        if prefix_transfer:
            if certificates:
                raise RuntimeError(
                    "prefix-transfer release cannot carry retirement certificates"
                )
        if session_full and releasing:
            detached_keys = tuple(
                (
                    transition.old,
                    transition.class_id,
                    transition.backend_domain,
                    transition.logical_ordinal,
                    transition.old_backend_index,
                    transition.token_begin,
                    transition.token_end_exclusive,
                )
                for item in values
                for transition in item.detached
            )
            retirement_keys = tuple(
                (
                    item.page,
                    item.class_id,
                    item.backend_domain,
                    item.logical_ordinal,
                    item.backend_index,
                    item.token_begin,
                    item.token_end_exclusive,
                )
                for item in certificates
            )
            if (
                len(set(retirement_keys)) != len(retirement_keys)
                or not set(retirement_keys).issubset(detached_keys)
            ):
                raise RuntimeError(
                    "request-private Full release retirement coverage changed"
                )
        if not spans:
            if session_full:
                return _MirrorCleanupPlan.empty()
            return None
        total = sum(end - begin for _row, begin, end, _start in spans)
        indices_cpu = torch.empty(total, dtype=torch.int64)
        expected_cpu = torch.empty(total, dtype=torch.int64)
        offsets = torch.arange(_config().page_tokens, dtype=torch.int64)
        cursor = 0
        for row, begin, end, start in spans:
            count = end - begin
            indices_cpu[cursor : cursor + count] = (
                row * width + begin + offsets[:count]
            )
            expected_cpu[cursor : cursor + count] = start + offsets[:count]
            cursor += count
        indices = indices_cpu.to(device=table.device)
        expected = expected_cpu.to(device=table.device)
        actual_parts = [table.reshape(-1)[indices].to(torch.int64)]
        expected_parts = [expected]
        if prefix_actual:
            actual_parts.append(torch.cat(prefix_actual).to(torch.int64))
            prefix_total = sum(count for _start, count in prefix_spans)
            prefix_expected_cpu = torch.empty(prefix_total, dtype=torch.int64)
            cursor = 0
            for start, count in prefix_spans:
                prefix_expected_cpu[cursor : cursor + count] = (
                    start + offsets[:count]
                )
                cursor += count
            expected_parts.append(
                prefix_expected_cpu.to(device=table.device)
            )
        actual = actual_parts[0] if len(actual_parts) == 1 else torch.cat(actual_parts)
        expected = (
            expected_parts[0]
            if len(expected_parts) == 1
            else torch.cat(expected_parts)
        )
        if not torch.equal(actual, expected):
            raise RuntimeError("DetachedBinding disagrees with the SGLang mirror")
        return _MirrorCleanupPlan(
            tuple(prefix_zeroes),
            None,
            (),
            (),
            ((table.reshape(-1), indices),) if releasing else (),
            tuple(private_prefixes),
        )

    @staticmethod
    def _validate_session_full_prefix(
        req: Any, row: Any, context: _MirrorCleanupContext, boundary: int
    ) -> PrivatePrefixProvenance | None:
        """Validate either private or radix-owned session prefix metadata.

        Shared-prefix registry membership is owned by ``session_cache``.  This
        layer deliberately validates only the request-local identity and mirror
        markers, then proves the tensor contents against the detached pages in
        the aggregate Full cleanup below.
        """

        shared = any(hasattr(req, name) for name in SHARED_PREFIX_MARKERS)
        if not shared:
            return validate_private_prefix(
                req, row, context.request_key, context.request_id, boundary
            )

        prefix = getattr(req, "prefix_indices", None)
        if prefix is None:
            raise RuntimeError("shared Prefix mirror metadata changed")
        prefix_count = int(prefix.numel())
        node = getattr(req, "_orbitkv_prefix_node", None)
        semantic = getattr(req, "_orbitkv_prefix_semantic", None)
        prefix_id = getattr(req, ENGINE_PREFIX_ID_MARKER, None)
        if (
            any(not hasattr(req, name) for name in SHARED_PREFIX_MARKERS)
            or getattr(req, PRIVATE_PREFIX_MARKER, None) is not None
            or context.request_key is None
            or context.request_id is None
            or getattr(req, "_orbitkv_request_key", None)
            != context.request_key
            or request_identity(req) != context.request_id
            or prefix_count <= 0
            or prefix_count > boundary
            or type(getattr(req, "cache_protected_len", None)) is not int
            or req.cache_protected_len != prefix_count
            or node is None
            or semantic is None
            or prefix_id is None
        ):
            raise RuntimeError("shared Prefix mirror metadata changed")
        validate_shared_prefix_metadata(
            req,
            key=context.request_key,
            request_id=context.request_id,
            prefix_id=prefix_id,
            boundary=prefix_count,
            node=node,
            semantic=semantic,
            provisional=False,
        )
        return None

    @staticmethod
    def _validate_context(
        context: Any, table: Any, rows: set[int]
    ) -> tuple[Any, int]:
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
            raise RuntimeError(
                "ReqToToken cleanup names a dummy or aliased row"
            )
        session_context = (
            context.request_key is not None or context.request_id is not None
        )
        if session_context and (
            context.request_key is None
            or context.request_id is None
            or getattr(req, "_orbitkv_request_key", None)
            != context.request_key
            or getattr(req, ENGINE_REQUEST_ID_MARKER, None)
            != context.request_id
            or request_identity(req) != context.request_id
        ):
            raise RuntimeError(
                "session mirror cleanup request identity changed"
            )
        rows.add(row)
        return req, row

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
            or _is_zero_page(candidate.destination)
            or candidate.destination == candidate.source
            or not _page_matches_backend(
                candidate.destination,
                candidate.destination_backend_index,
                arena,
            )
        ):
            raise RuntimeError("candidate mirror transition is invalid")
        if _is_zero_page(candidate.source):
            if (
                candidate.source_backend_index != 0
                or copied_begin != 0
                or copied_end != 0
            ):
                raise RuntimeError("fresh candidate carries COW source authority")
        elif (
            not _page_matches_backend(
                candidate.source, candidate.source_backend_index, arena
            )
            or not begin <= copied_begin < copied_end <= end
        ):
            raise RuntimeError("COW candidate source authority is invalid")

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
