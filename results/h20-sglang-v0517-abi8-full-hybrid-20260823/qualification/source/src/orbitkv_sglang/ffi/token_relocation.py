from __future__ import annotations

import ctypes
from typing import Any, Sequence

from orbitkv_sglang.runtime import (
    BatchCompletionReceipt,
    CompletedRelocationBatch,
    ManagerError,
    PageLease,
    PrepareRelocationItem,
    PreparedRelocation,
    ReclamationCertificate,
    ReclamationLease,
    RelocationCopyReceipt,
    RelocationLease,
    RelocationUnobservedReceipt,
    RequestLease,
    RequestView,
    SnapshotLease,
    SubmittedRelocation,
    TokenDisposition,
    TokenDispositionBatchItem,
    TokenDispositionKind,
    TokenLocation,
    TokenMove,
    TokenPlacement,
    TokenView,
    TokenViewQuery,
)

from . import layouts as L
from .library import STATUS_BUFFER_TOO_SMALL
from .workspace import array


def _uint(name: str, value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ManagerError(f"{name} is outside uint{bits}_t")
    return value


def _lease_c(layout: Any, value: Any) -> Any:
    return layout(
        _uint("lease engine epoch", value.engine_epoch, 64),
        _uint("lease slot", value.slot, 32),
        _uint("lease generation", value.generation, 32),
    )


def _request(value: Any) -> RequestLease:
    return RequestLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _snapshot(value: Any) -> SnapshotLease:
    return SnapshotLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _relocation(value: Any) -> RelocationLease:
    return RelocationLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _reclamation(value: Any) -> ReclamationLease:
    return ReclamationLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _page_c(value: PageLease) -> L.PageLeaseLayout:
    return L.PageLeaseLayout(
        _uint("page engine epoch", value.engine_epoch, 64),
        _uint("page pool epoch", value.pool_epoch, 64),
        _uint("page generation", value.generation, 64),
        _uint("page id", value.page_id, 32),
        _uint("page pool id", value.pool_id, 32),
    )


def _page(value: Any) -> PageLease:
    return PageLease(
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.page_id),
        int(value.pool_id),
    )


def _location_c(value: TokenLocation) -> L.TokenLocationLayout:
    return L.TokenLocationLayout(
        _page_c(value.page),
        _uint("token backend index", value.backend_index, 64),
        _uint("token offset", value.offset, 32),
        0,
    )


def _location(value: Any) -> TokenLocation:
    if int(value.reserved) != 0:
        raise ManagerError("token location reserved field is nonzero")
    return TokenLocation(_page(value.page), int(value.backend_index), int(value.offset))


def _disposition_c(value: TokenDisposition, *, allow_retained: bool) -> L.TokenDispositionLayout:
    try:
        kind = TokenDispositionKind(value.kind)
    except (TypeError, ValueError) as error:
        raise ManagerError("token disposition kind is invalid") from error
    if not allow_retained and kind is TokenDispositionKind.RETAINED:
        raise ManagerError("disposition updates cannot mark a token retained")
    if kind is TokenDispositionKind.RETAINED:
        if value.policy_or_proof_id or value.version or value.quality_contract:
            raise ManagerError("retained token disposition carries policy evidence")
    elif value.policy_or_proof_id == 0:
        raise ManagerError("non-retained token disposition lacks policy evidence")
    elif kind is TokenDispositionKind.SEMANTICALLY_DEAD and value.quality_contract != 0:
        raise ManagerError("semantic-death proof carries an approximate quality contract")
    elif kind is TokenDispositionKind.POLICY_EVICTED and value.quality_contract == 0:
        raise ManagerError("policy eviction lacks a quality contract")
    return L.TokenDispositionLayout(
        _uint("policy or proof id", value.policy_or_proof_id, 64),
        _uint("policy version", value.version, 64),
        _uint("quality contract", value.quality_contract, 64),
        int(kind),
        0,
        0,
    )


def _disposition(value: Any) -> TokenDisposition:
    if int(value.reserved16) or int(value.reserved32):
        raise ManagerError("token disposition reserved field is nonzero")
    try:
        kind = TokenDispositionKind(int(value.kind))
    except ValueError as error:
        raise ManagerError("token disposition kind is invalid") from error
    result = TokenDisposition(
        kind,
        int(value.policy_or_proof_id),
        int(value.version),
        int(value.quality_contract),
    )
    _disposition_c(result, allow_retained=True)
    return result


def _request_view(value: Any) -> RequestView:
    if int(value.reserved) != 0:
        raise ManagerError("request view reserved field is nonzero")
    return RequestView(
        _request(value.request),
        _snapshot(value.snapshot),
        int(value.view_version),
        int(value.boundary),
        int(value.resident_count),
    )


def _certificate(value: Any) -> ReclamationCertificate:
    if int(value.reserved32) != 0:
        raise ManagerError("reclamation certificate reserved field is nonzero")
    return ReclamationCertificate(
        _reclamation(value.reclamation),
        _page(value.page),
        int(value.class_id),
        int(value.backend_domain),
        int(value.logical_ordinal),
        int(value.backend_index),
        int(value.token_begin),
        int(value.token_end_exclusive),
        int(value.completion_domain),
        int(value.completion_value),
    )


def _copy_receipt_c(value: RelocationCopyReceipt) -> L.RelocationCopyReceiptLayout:
    return L.RelocationCopyReceiptLayout(
        _lease_c(L.RelocationLeaseLayout, value.relocation),
        _uint("relocation token id", value.token_id, 64),
        _location_c(value.source),
        _location_c(value.destination),
        _uint("relocation observed", value.observed, 8),
        _uint("relocation copied", value.copied, 8),
        _uint("relocation receipt reserved16", value.reserved16, 16),
        _uint("relocation receipt reserved32", value.reserved32, 32),
    )


class TokenRelocationMixin:
    """ABI8 token-view and relocation calls shared by the ctypes manager."""

    def token_views_batch(
        self, queries: Sequence[TokenViewQuery]
    ) -> tuple[TokenView, ...]:
        with self._lock:
            values = tuple(queries)
            count = self._count(values, "token view", self._operation_capacity)
            raw = (L.TokenViewQueryLayout * count)(
                *(
                    L.TokenViewQueryLayout(
                        _lease_c(L.RequestLeaseLayout, item.request),
                        _lease_c(L.SnapshotLeaseLayout, item.expected_snapshot),
                        _uint("token view class", item.class_id, 16),
                        0,
                        0,
                    )
                    for item in values
                )
            )
            expected_placements = sum(
                _uint("token view expected boundary", item.expected_boundary, 64)
                for item in values
            )
            if expected_placements >= 1 << 32:
                raise ManagerError("token view placement count exceeds uint32_t")
            required = [ctypes.c_uint32(), ctypes.c_uint32()]
            status = self._call(
                "token views preflight",
                self._library.orbitkv_manager_token_views_batch,
                self._require_handle(),
                raw,
                count,
                None,
                0,
                ctypes.byref(required[0]),
                None,
                0,
                ctypes.byref(required[1]),
                allow_short=True,
                counter="token_views_batch_calls",
            )
            if (
                status != STATUS_BUFFER_TOO_SMALL
                or int(required[0].value) != count
                or int(required[1].value) != expected_placements
            ):
                raise self._poison("token view preflight changed the logical history bound")
            views = array(L.TokenViewLayout, count)
            placements = array(L.TokenPlacementLayout, expected_placements)
            self._counters["cold_workspace_allocations"] += 1
            out = [ctypes.c_uint32(), ctypes.c_uint32()]
            self._call(
                "token views batch",
                self._library.orbitkv_manager_token_views_batch,
                self._require_handle(),
                raw,
                count,
                views,
                count,
                ctypes.byref(out[0]),
                placements,
                expected_placements,
                ctypes.byref(out[1]),
            )
            if tuple(int(item.value) for item in out) != (count, expected_placements):
                raise self._poison("token view second pass changed capacities")
            cursor = 0
            result = []
            try:
                for index, query in enumerate(values):
                    view = views[index]
                    if int(view.reserved16) or int(view.reserved32):
                        raise ManagerError("token view reserved field is nonzero")
                    begin = int(view.placement_offset)
                    end = begin + int(view.placement_count)
                    if (
                        begin != cursor
                        or end > expected_placements
                        or int(view.placement_count) != query.expected_boundary
                        or int(view.class_id) != query.class_id
                        or int(view.page_tokens) != self._page_tokens
                    ):
                        raise ManagerError("token view span or identity is invalid")
                    converted = []
                    for token_id in range(begin, end):
                        item = placements[token_id]
                        if int(item.reserved) or int(item.location_present) not in (0, 1):
                            raise ManagerError("token placement reserved/presence field is invalid")
                        logical_id = token_id - begin
                        if int(item.token_id) != logical_id:
                            raise ManagerError("token placement ids are not canonical")
                        location = _location(item.location) if item.location_present else None
                        converted.append(
                            TokenPlacement(
                                logical_id,
                                _disposition(item.disposition),
                                location,
                            )
                        )
                    result.append(
                        TokenView(
                            int(view.class_id),
                            int(view.view_version),
                            int(view.page_tokens),
                            tuple(converted),
                        )
                    )
                    cursor = end
                if cursor != expected_placements:
                    raise ManagerError("token view spans do not cover the flat buffer")
                return tuple(result)
            except Exception as error:
                raise self._poison(f"token view output is invalid: {error}") from error

    def mark_token_dispositions_batch(
        self, items: Sequence[TokenDispositionBatchItem]
    ) -> tuple[RequestView, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "token disposition", self._operation_capacity)
            updates = tuple(update for item in values for update in item.updates)
            if not updates or len(updates) >= 1 << 32:
                raise ManagerError("token disposition update cardinality is invalid")
            raw_items = array(L.TokenDispositionBatchItemLayout, count)
            raw_updates = array(L.ClassTokenDispositionUpdateLayout, len(updates))
            offset = 0
            for index, item in enumerate(values):
                if not item.updates:
                    raise ManagerError("token disposition item must not be empty")
                raw_items[index] = L.TokenDispositionBatchItemLayout(
                    _lease_c(L.RequestLeaseLayout, item.request),
                    _lease_c(L.SnapshotLeaseLayout, item.expected_snapshot),
                    offset,
                    len(item.updates),
                )
                for update in item.updates:
                    raw_updates[offset] = L.ClassTokenDispositionUpdateLayout(
                        _uint("disposition token id", update.token_id, 64),
                        _disposition_c(update.disposition, allow_retained=False),
                        _uint("disposition class", update.class_id, 16),
                        0,
                        0,
                    )
                    offset += 1
            output = array(L.RequestViewLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "mark token dispositions batch",
                self._library.orbitkv_manager_mark_token_dispositions_batch,
                self._require_handle(),
                raw_items,
                count,
                raw_updates,
                len(updates),
                output,
                count,
                ctypes.byref(out),
                counter="mark_token_dispositions_batch_calls",
            )
            if int(out.value) != count:
                raise self._poison("token disposition output cardinality changed")
            try:
                return tuple(_request_view(output[index]) for index in range(count))
            except Exception as error:
                raise self._poison(f"token disposition output is invalid: {error}") from error

    def prepare_relocation_batch(
        self, items: Sequence[PrepareRelocationItem]
    ) -> tuple[PreparedRelocation, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "prepare relocation", self._operation_capacity)
            raw = (L.PrepareRelocationItemLayout * count)(
                *(
                    L.PrepareRelocationItemLayout(
                        _lease_c(L.RequestLeaseLayout, item.request),
                        _lease_c(L.SnapshotLeaseLayout, item.expected_snapshot),
                        L.RelocationPolicyLayout(
                            _uint("maximum source pages", item.policy.maximum_source_pages, 32),
                            _uint("evacuation headroom pages", item.policy.evacuation_headroom_pages, 32),
                            _uint("fragmentation threshold", item.policy.fragmentation_threshold_milli, 16),
                            int(item.policy.full_evacuation),
                            0,
                            0,
                        ),
                        _uint("relocation class", item.class_id, 16),
                        0,
                        0,
                    )
                    for item in values
                )
            )
            if any(
                not item.policy.full_evacuation
                or item.policy.maximum_source_pages > self._physical_pages
                or item.policy.evacuation_headroom_pages > self._physical_pages
                for item in values
            ):
                raise ManagerError("ABI8 first relocation transaction requires bounded full evacuation")
            move_capacity = self._physical_pages * self._page_tokens
            prepared = array(L.PreparedRelocationLayout, count)
            sources = array(L.PageLeaseLayout, self._physical_pages)
            destinations = array(L.PageLeaseLayout, self._physical_pages)
            moves = array(L.TokenMoveLayout, move_capacity)
            self._counters["cold_workspace_allocations"] += 1
            out = [ctypes.c_uint32() for _ in range(4)]
            self._call(
                "prepare relocation batch",
                self._library.orbitkv_manager_prepare_relocation_batch,
                self._require_handle(),
                raw,
                count,
                prepared,
                count,
                ctypes.byref(out[0]),
                sources,
                self._physical_pages,
                ctypes.byref(out[1]),
                destinations,
                self._physical_pages,
                ctypes.byref(out[2]),
                moves,
                move_capacity,
                ctypes.byref(out[3]),
                counter="prepare_relocation_batch_calls",
            )
            actual = tuple(int(item.value) for item in out)
            if actual[0] != count or actual[1] > self._physical_pages or actual[2] > self._physical_pages or actual[3] > move_capacity:
                raise self._poison("prepare relocation exceeded its bounded outputs")
            source_cursor = destination_cursor = move_cursor = 0
            result = []
            try:
                for index in range(count):
                    item = prepared[index]
                    if int(item.reserved32) != 0:
                        raise ManagerError("prepared relocation reserved field is nonzero")
                    source_end = self._span(int(item.source_offset), int(item.source_count), source_cursor, actual[1], "relocation source")
                    destination_end = self._span(int(item.destination_offset), int(item.destination_count), destination_cursor, actual[2], "relocation destination")
                    move_end = self._span(int(item.move_offset), int(item.move_count), move_cursor, actual[3], "relocation move")
                    result.append(
                        PreparedRelocation(
                            _relocation(item.relocation),
                            _request(item.request),
                            _snapshot(item.base_snapshot),
                            _snapshot(item.target_snapshot),
                            int(item.base_view_version),
                            int(item.target_view_version),
                            tuple(_page(sources[p]) for p in range(source_cursor, source_end)),
                            tuple(_page(destinations[p]) for p in range(destination_cursor, destination_end)),
                            tuple(
                                TokenMove(
                                    int(moves[p].token_id),
                                    _location(moves[p].source),
                                    _location(moves[p].destination),
                                )
                                for p in range(move_cursor, move_end)
                            ),
                            int(item.projected_reclaimed_pages),
                            int(item.fragmentation_milli),
                            int(item.class_id),
                        )
                    )
                    source_cursor = source_end
                    destination_cursor = destination_end
                    move_cursor = move_end
                if (source_cursor, destination_cursor, move_cursor) != actual[1:]:
                    raise ManagerError("prepared relocation spans do not cover flat buffers")
                return tuple(result)
            except Exception as error:
                raise self._poison(f"prepared relocation output is invalid: {error}") from error

    def submit_relocation_batch(
        self,
        items: Sequence[tuple[RelocationLease, Sequence[RelocationCopyReceipt]]],
    ) -> tuple[SubmittedRelocation, ...]:
        with self._lock:
            values = tuple((lease, tuple(receipts)) for lease, receipts in items)
            count = self._count(values, "submit relocation", self._operation_capacity)
            receipts = tuple(receipt for _lease, group in values for receipt in group)
            if len(receipts) > self._physical_pages * self._page_tokens:
                raise ManagerError("relocation receipt cardinality exceeds its physical bound")
            leases = (L.RelocationLeaseLayout * count)(
                *(_lease_c(L.RelocationLeaseLayout, lease) for lease, _receipts in values)
            )
            raw_receipts = (L.RelocationCopyReceiptLayout * len(receipts))(
                *(_copy_receipt_c(receipt) for receipt in receipts)
            )
            output = array(L.SubmittedRelocationLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "submit relocation batch",
                self._library.orbitkv_manager_submit_relocation_batch,
                self._require_handle(),
                leases,
                count,
                raw_receipts,
                len(receipts),
                output,
                count,
                ctypes.byref(out),
                counter="submit_relocation_batch_calls",
            )
            if int(out.value) != count:
                raise self._poison("submitted relocation cardinality changed")
            try:
                return tuple(
                    SubmittedRelocation(
                        _relocation(output[index].relocation),
                        _request(output[index].request),
                        _snapshot(output[index].target_snapshot),
                    )
                    for index in range(count)
                )
            except Exception as error:
                raise self._poison(f"submitted relocation output is invalid: {error}") from error

    def complete_relocation_batch(
        self,
        receipt: BatchCompletionReceipt,
        relocations: Sequence[RelocationLease],
    ) -> CompletedRelocationBatch:
        with self._lock:
            values = tuple(relocations)
            count = self._count(values, "complete relocation", self._operation_capacity)
            completion = L.CompletionReceiptLayout(
                _uint("completion engine epoch", receipt.engine_epoch, 64),
                _uint("completion domain", receipt.completion_domain, 64),
                _uint("completion value", receipt.completion_value, 64),
                _uint("completion confirmed", receipt.confirmed, 32),
                _uint("completion reserved", receipt.reserved, 32),
            )
            leases = (L.RelocationLeaseLayout * count)(
                *(_lease_c(L.RelocationLeaseLayout, value) for value in values)
            )
            publications = array(L.RequestViewLayout, count)
            retirements = array(L.ReclamationCertificateLayout, self._physical_pages)
            out = [ctypes.c_uint32(), ctypes.c_uint32()]
            self._call(
                "complete relocation batch",
                self._library.orbitkv_manager_complete_relocation_batch,
                self._require_handle(),
                ctypes.byref(completion),
                leases,
                count,
                publications,
                count,
                ctypes.byref(out[0]),
                retirements,
                self._physical_pages,
                ctypes.byref(out[1]),
                counter="complete_relocation_batch_calls",
            )
            if int(out[0].value) != count or int(out[1].value) > self._physical_pages:
                raise self._poison("complete relocation exceeded its bounded outputs")
            try:
                return CompletedRelocationBatch(
                    tuple(_request_view(publications[index]) for index in range(count)),
                    tuple(_certificate(retirements[index]) for index in range(int(out[1].value))),
                )
            except Exception as error:
                raise self._poison(f"complete relocation output is invalid: {error}") from error

    def abort_relocations_batch(
        self, receipts: Sequence[RelocationUnobservedReceipt]
    ) -> None:
        with self._lock:
            values = tuple(receipts)
            count = self._count(values, "abort relocation", self._operation_capacity)
            raw = (L.RelocationUnobservedReceiptLayout * count)(
                *(
                    L.RelocationUnobservedReceiptLayout(
                        _lease_c(L.RelocationLeaseLayout, item.relocation),
                        _uint("backend unobserved", item.backend_unobserved, 32),
                        _uint("relocation abort reserved", item.reserved, 32),
                    )
                    for item in values
                )
            )
            self._call(
                "abort relocations batch",
                self._library.orbitkv_manager_abort_relocations_batch,
                self._require_handle(),
                raw,
                count,
                counter="abort_relocations_batch_calls",
            )


__all__ = ["TokenRelocationMixin"]
