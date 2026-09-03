from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Sequence

from orbitkv_sglang.runtime import ManagerError
from orbitkv_sglang.runtime.token_state import (
    RelocationPolicy,
    TokenDisposition,
    TokenDispositionKind,
    TokenLocation,
    TokenMove,
    TokenPlacement,
)

from . import layouts as L
from .codec import page as _page
from .codec import page_to_c as _page_to_c
from .codec import uint as _uint
from .library import STATUS_BUFFER_TOO_SMALL
from .session_support import (
    bounded_output_counts,
    canonical_span_end,
    decode_operation_id,
    require_batch_count,
)
from .session_types import (
    EngineCompletionEvidence,
    EnginePrepareRelocationItem,
    EnginePreparedRelocation,
    EngineRelocationAbortEvidence,
    EngineRelocationCopyEvidence,
    EngineRelocationExecutionEvidence,
    EngineRelocationId,
    EngineRelocationPlan,
    EngineRelocationPublication,
    EngineRelocationPublicationEvidence,
    EngineRelocationRequestEvidence,
    EngineRelocationRequestPublication,
    EngineRelocationTicket,
    EngineRequestId,
    EngineRequestView,
    EngineRetirement,
    EngineRetirementEvidence,
    EngineTokenDispositionBatchItem,
    EngineTokenDispositionUpdate,
    EngineTokenView,
    EngineTokenViewQuery,
    _retirement,
    _retirement_evidence_c,
    _session_id_c,
    _wire_bool,
)
from .workspace import array, checked_product


def _location_c(value: TokenLocation) -> L.TokenLocationLayout:
    if type(value) is not TokenLocation:
        raise ManagerError("token location must be TokenLocation")
    return L.TokenLocationLayout(
        _page_to_c(value.page),
        _uint("token backend index", value.backend_index, 64),
        _uint("token offset", value.offset, 32),
        0,
    )


def _location(value: Any) -> TokenLocation:
    if int(value.reserved) != 0:
        raise ManagerError("token location reserved field is nonzero")
    return TokenLocation(
        _page(value.page), int(value.backend_index), int(value.offset)
    )


def _disposition_c(
    value: TokenDisposition, *, allow_retained: bool
) -> L.TokenDispositionLayout:
    if type(value) is not TokenDisposition:
        raise ManagerError("token disposition must be TokenDisposition")
    try:
        kind = TokenDispositionKind(value.kind)
    except (TypeError, ValueError) as error:
        raise ManagerError("token disposition kind is invalid") from error
    policy_or_proof_id = _uint(
        "policy or proof id", value.policy_or_proof_id, 64
    )
    version = _uint("policy version", value.version, 64)
    quality_contract = _uint(
        "quality contract", value.quality_contract, 64
    )
    if not allow_retained and kind is TokenDispositionKind.RETAINED:
        raise ManagerError("disposition updates cannot mark a token retained")
    if kind is TokenDispositionKind.RETAINED:
        if policy_or_proof_id or version or quality_contract:
            raise ManagerError(
                "retained token disposition carries policy evidence"
            )
    elif policy_or_proof_id == 0:
        raise ManagerError(
            "non-retained token disposition lacks policy evidence"
        )
    elif (
        kind is TokenDispositionKind.SEMANTICALLY_DEAD
        and quality_contract != 0
    ):
        raise ManagerError(
            "semantic-death proof carries an approximate quality contract"
        )
    elif (
        kind is TokenDispositionKind.POLICY_EVICTED
        and quality_contract == 0
    ):
        raise ManagerError("policy eviction lacks a quality contract")
    return L.TokenDispositionLayout(
        policy_or_proof_id, version, quality_contract, int(kind), 0, 0
    )


def _disposition(value: Any) -> TokenDisposition:
    if int(value.reserved16) != 0 or int(value.reserved32) != 0:
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


def _relocation_policy_c(value: RelocationPolicy) -> L.RelocationPolicyLayout:
    if type(value) is not RelocationPolicy:
        raise ManagerError("relocation policy must be RelocationPolicy")
    if value.full_evacuation is not True:
        raise ManagerError("session relocation requires full evacuation")
    return L.RelocationPolicyLayout(
        _uint("maximum source pages", value.maximum_source_pages, 32),
        _uint(
            "evacuation headroom pages",
            value.evacuation_headroom_pages,
            32,
        ),
        _uint(
            "fragmentation threshold",
            value.fragmentation_threshold_milli,
            16,
        ),
        _wire_bool("relocation full evacuation", value.full_evacuation, 8),
        0,
        0,
    )


def _move(value: Any) -> TokenMove:
    return TokenMove(
        int(value.token_id),
        _location(value.source),
        _location(value.destination),
    )


@dataclass(frozen=True, slots=True)
class _PendingRelocation:
    prepared: EnginePreparedRelocation
    phase: str
    publication: EngineRelocationPublication | None = None


class SessionRelocationMixin:
    """Capability-safe token and relocation calls for RuntimeSession."""

    _library: Any
    _lock: Any
    _handle: ctypes.c_void_p
    _operation_capacity: int
    _physical_pages: int
    _page_tokens: int
    _session_epoch: int
    _request_views: Any
    _retirement_evidence: Any
    _pending_relocations: dict[EngineRelocationId, _PendingRelocation]
    _relocation_plans: Any
    _relocation_source_pages: Any
    _relocation_destination_pages: Any
    _relocation_moves: Any
    _relocation_publications: Any
    _token_view_queries: Any
    _token_disposition_items: Any
    _token_disposition_updates: Any
    _relocation_prepare_items: Any
    _relocation_request_evidence: Any
    _relocation_copy_evidence: Any
    _relocation_abort_evidence: Any

    def _initialize_relocation_state(self) -> None:
        move_capacity = self._relocation_move_capacity
        batch_capacity = self._operation_capacity
        self._token_view_queries = array(
            L.SessionTokenViewQueryLayout, self._request_capacity
        )
        self._token_disposition_items = array(
            L.SessionTokenDispositionBatchItemLayout, batch_capacity
        )
        self._token_disposition_updates = array(
            L.ClassTokenDispositionUpdateLayout, move_capacity
        )
        self._relocation_prepare_items = array(
            L.SessionPrepareRelocationItemLayout, batch_capacity
        )
        self._relocation_plans = array(
            L.SessionRelocationPlanLayout, batch_capacity
        )
        self._relocation_source_pages = array(
            L.PageLeaseLayout, self._physical_pages
        )
        self._relocation_destination_pages = array(
            L.PageLeaseLayout, self._physical_pages
        )
        self._relocation_moves = array(L.TokenMoveLayout, move_capacity)
        self._relocation_request_evidence = array(
            L.SessionRelocationRequestEvidenceLayout, batch_capacity
        )
        self._relocation_copy_evidence = array(
            L.SessionRelocationCopyEvidenceLayout, move_capacity
        )
        self._relocation_abort_evidence = array(
            L.SessionRelocationAbortEvidenceLayout, batch_capacity
        )
        self._relocation_publications = array(
            L.SessionRelocationRequestPublicationLayout, batch_capacity
        )
        self._pending_relocations = {}

    def _require_handle(
        self, *, allow_poisoned: bool = False
    ) -> ctypes.c_void_p: ...

    def _call(
        self,
        operation: str,
        function: Any,
        *args: Any,
        require_handle: bool = True,
        allow_poisoned: bool = False,
        allow_short: bool = False,
    ) -> int: ...

    def _poison(self, reason: str) -> BaseException: ...

    def _validate_retirement_evidence(
        self,
        retirements: tuple[EngineRetirement, ...],
        evidence: tuple[EngineRetirementEvidence, ...],
        operation: str,
    ) -> None: ...

    @property
    def _relocation_move_capacity(self) -> int:
        return checked_product(
            "session relocation move bound",
            self._physical_pages,
            self._page_tokens,
        )

    @staticmethod
    def _relocation_id_c(
        value: EngineRelocationId,
    ) -> L.SessionRelocationIdLayout:
        return _session_id_c(
            value,
            EngineRelocationId,
            L.SessionRelocationIdLayout,
            "relocation id",
        )

    def token_views_batch(
        self, queries: Sequence[EngineTokenViewQuery]
    ) -> tuple[EngineTokenView, ...]:
        with self._lock:
            self._require_handle()
            values = tuple(queries)
            count = require_batch_count(
                values, "session token view", self._request_capacity
            )
            if any(type(item) is not EngineTokenViewQuery for item in values):
                raise ManagerError(
                    "session token view items must be EngineTokenViewQuery values"
                )
            identities = tuple(
                (
                    _uint("engine request id", item.request_id, 64),
                    _uint("token view class", item.class_id, 16),
                    _uint(
                        "token view expected boundary",
                        item.expected_boundary,
                        64,
                    ),
                )
                for item in values
            )
            if len({item[:2] for item in identities}) != count:
                raise ManagerError(
                    "session token view contains duplicate request/class queries"
                )
            expected_placements = sum(item[2] for item in identities)
            if expected_placements >= 1 << 32:
                raise ManagerError(
                    "session token view placement count exceeds uint32_t"
                )
            for index, (request_id, class_id, expected_boundary) in enumerate(
                identities
            ):
                self._token_view_queries[index] = (
                    L.SessionTokenViewQueryLayout(
                        request_id, expected_boundary, class_id, 0, 0
                    )
                )
            required = [ctypes.c_uint32(), ctypes.c_uint32()]
            status = self._call(
                "runtime session token views preflight",
                self._library.function("orbitkv_session_token_views_batch"),
                self._require_handle(),
                self._token_view_queries,
                count,
                None,
                0,
                ctypes.byref(required[0]),
                None,
                0,
                ctypes.byref(required[1]),
                allow_short=True,
            )
            view_count = int(required[0].value)
            placement_count = int(required[1].value)
            if (
                status != STATUS_BUFFER_TOO_SMALL
                or view_count != count
                or placement_count != expected_placements
            ):
                raise self._poison(
                    "runtime session token view preflight returned invalid bounds"
                )
            try:
                views = array(L.SessionTokenViewLayout, view_count)
                placements = array(L.TokenPlacementLayout, placement_count)
            except BaseException as error:
                raise self._poison(
                    "runtime session token view workspace allocation failed: "
                    f"{error}"
                ) from error
            actual = [ctypes.c_uint32(), ctypes.c_uint32()]
            self._call(
                "runtime session token views batch",
                self._library.function("orbitkv_session_token_views_batch"),
                self._require_handle(),
                self._token_view_queries,
                count,
                views,
                view_count,
                ctypes.byref(actual[0]),
                placements,
                placement_count,
                ctypes.byref(actual[1]),
            )
            if tuple(int(item.value) for item in actual) != (
                view_count,
                placement_count,
            ):
                raise self._poison(
                    "runtime session token view second pass changed capacities"
                )
            cursor = 0
            result = []
            try:
                for index, (identity, expected) in enumerate(
                    zip(identities, values, strict=True)
                ):
                    request_id, class_id, expected_boundary = identity
                    view = views[index]
                    if int(view.reserved16) or int(view.reserved32):
                        raise ManagerError(
                            "session token view reserved field is nonzero"
                        )
                    end = canonical_span_end(
                        int(view.placement_offset),
                        int(view.placement_count),
                        cursor,
                        placement_count,
                        "session token view placement",
                    )
                    if (
                        int(view.request_id) != request_id
                        or int(view.class_id) != class_id
                        or int(view.page_tokens) != self._page_tokens
                        or int(view.placement_count) != expected_boundary
                    ):
                        raise ManagerError(
                            "session token view identity or page size changed"
                        )
                    converted = []
                    for token_id, position in enumerate(range(cursor, end)):
                        item = placements[position]
                        present = int(item.location_present)
                        if int(item.reserved) != 0 or present not in (0, 1):
                            raise ManagerError(
                                "session token placement reserved/presence field is invalid"
                            )
                        if int(item.token_id) != token_id:
                            raise ManagerError(
                                "session token placement ids are not canonical"
                            )
                        converted.append(
                            TokenPlacement(
                                token_id,
                                _disposition(item.disposition),
                                _location(item.location) if present else None,
                            )
                        )
                    result.append(
                        EngineTokenView(
                            expected.request_id,
                            expected.class_id,
                            int(view.view_version),
                            int(view.page_tokens),
                            tuple(converted),
                        )
                    )
                    cursor = end
                if cursor != placement_count:
                    raise ManagerError(
                        "session token view spans do not cover the flat buffer"
                    )
                return tuple(result)
            except Exception as error:
                raise self._poison(
                    f"runtime session token view output is invalid: {error}"
                ) from error

    def mark_token_dispositions_batch(
        self, items: Sequence[EngineTokenDispositionBatchItem]
    ) -> tuple[EngineRequestView, ...]:
        with self._lock:
            self._require_handle()
            values = tuple(items)
            count = require_batch_count(
                values, "session token disposition", self._operation_capacity
            )
            if any(
                type(item) is not EngineTokenDispositionBatchItem
                for item in values
            ):
                raise ManagerError(
                    "session token disposition items must be "
                    "EngineTokenDispositionBatchItem values"
                )
            request_ids = tuple(
                _uint("engine request id", item.request_id, 64)
                for item in values
            )
            if len(set(request_ids)) != count:
                raise ManagerError(
                    "session token disposition contains duplicate request ids"
                )
            update_count = sum(len(item.updates) for item in values)
            if not update_count or update_count > self._relocation_move_capacity:
                raise ManagerError(
                    "session token disposition update cardinality exceeds its "
                    "configured bound"
                )
            offset = 0
            for index, (request_id, item) in enumerate(
                zip(request_ids, values, strict=True)
            ):
                updates = tuple(item.updates)
                if not updates:
                    raise ManagerError(
                        "session token disposition item must not be empty"
                    )
                self._token_disposition_items[index] = (
                    L.SessionTokenDispositionBatchItemLayout(
                    request_id, offset, len(updates)
                    )
                )
                for update in updates:
                    if type(update) is not EngineTokenDispositionUpdate:
                        raise ManagerError(
                            "session token disposition update must be "
                            "EngineTokenDispositionUpdate"
                        )
                    self._token_disposition_updates[offset] = (
                        L.ClassTokenDispositionUpdateLayout(
                        _uint(
                            "disposition token id", update.token_id, 64
                        ),
                        _disposition_c(
                            update.disposition, allow_retained=False
                        ),
                        _uint("disposition class", update.class_id, 16),
                        0,
                        0,
                        )
                    )
                    offset += 1
            out = ctypes.c_uint32()
            self._call(
                "runtime session mark token dispositions batch",
                self._library.function(
                    "orbitkv_session_mark_token_dispositions_batch"
                ),
                self._require_handle(),
                self._token_disposition_items,
                count,
                self._token_disposition_updates,
                update_count,
                self._request_views,
                self._operation_capacity,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison(
                    "runtime session token disposition cardinality changed"
                )
            try:
                result = []
                for request_id, raw in zip(
                    request_ids, self._request_views[:count], strict=True
                ):
                    if int(raw.reserved) != 0:
                        raise ManagerError(
                            "session request view reserved field is nonzero"
                        )
                    if int(raw.request_id) != request_id:
                        raise ManagerError(
                            "session token disposition request ordering changed"
                        )
                    result.append(
                        EngineRequestView(
                            EngineRequestId(request_id),
                            int(raw.view_version),
                            int(raw.boundary),
                            int(raw.resident_count),
                        )
                    )
                return tuple(result)
            except Exception as error:
                raise self._poison(
                    "runtime session token disposition output is invalid: "
                    f"{error}"
                ) from error

    def prepare_relocation_batch(
        self, items: Sequence[EnginePrepareRelocationItem]
    ) -> EnginePreparedRelocation:
        with self._lock:
            self._require_handle()
            values = tuple(items)
            count = require_batch_count(
                values, "session relocation prepare", self._operation_capacity
            )
            if any(type(item) is not EnginePrepareRelocationItem for item in values):
                raise ManagerError(
                    "session relocation prepare items must be "
                    "EnginePrepareRelocationItem values"
                )
            request_ids = tuple(
                _uint("engine request id", item.request_id, 64)
                for item in values
            )
            if len(set(request_ids)) != count:
                raise ManagerError(
                    "session relocation prepare contains duplicate request ids"
                )
            for index, (request_id, item) in enumerate(
                zip(request_ids, values, strict=True)
            ):
                if (
                    item.policy.maximum_source_pages <= 0
                    or item.policy.evacuation_headroom_pages <= 0
                    or item.policy.maximum_source_pages > self._physical_pages
                    or item.policy.evacuation_headroom_pages
                    > self._physical_pages
                    or not 0
                    <= item.policy.fragmentation_threshold_milli
                    <= 1000
                ):
                    raise ManagerError(
                        "session relocation policy exceeds configured bounds"
                    )
                self._relocation_prepare_items[index] = (
                    L.SessionPrepareRelocationItemLayout(
                        request_id,
                        _relocation_policy_c(item.policy),
                        _uint("relocation class", item.class_id, 16),
                        0,
                        0,
                    )
                )
            relocation_id = L.SessionRelocationIdLayout()
            counts = [ctypes.c_uint32() for _ in range(4)]
            self._call(
                "runtime session prepare relocation batch",
                self._library.function(
                    "orbitkv_session_prepare_relocation_batch"
                ),
                self._require_handle(),
                self._relocation_prepare_items,
                count,
                ctypes.byref(relocation_id),
                self._relocation_plans,
                self._operation_capacity,
                ctypes.byref(counts[0]),
                self._relocation_source_pages,
                self._physical_pages,
                ctypes.byref(counts[1]),
                self._relocation_destination_pages,
                self._physical_pages,
                ctypes.byref(counts[2]),
                self._relocation_moves,
                self._relocation_move_capacity,
                ctypes.byref(counts[3]),
            )
            try:
                totals = bounded_output_counts(
                    counts,
                    (
                        self._operation_capacity,
                        self._physical_pages,
                        self._physical_pages,
                        self._relocation_move_capacity,
                    ),
                    "session relocation prepare",
                )
                if totals[0] != count:
                    raise ManagerError(
                        "session relocation prepare cardinality changed"
                    )
                operation_id = decode_operation_id(
                    relocation_id,
                    EngineRelocationId,
                    L.SessionRelocationIdLayout,
                    "relocation id",
                )
                if operation_id.session_epoch != self._session_epoch:
                    raise ManagerError(
                        "session relocation prepare returned a foreign id"
                    )
                if operation_id in self._pending_relocations:
                    raise ManagerError(
                        "session relocation prepare returned a duplicate id"
                    )
                cursors = [0, 0, 0]
                plans = []
                for index, expected in enumerate(values):
                    item = self._relocation_plans[index]
                    if int(item.reserved32) != 0:
                        raise ManagerError(
                            "session relocation plan reserved field is nonzero"
                        )
                    ends = [
                        canonical_span_end(
                            int(item.source_offset),
                            int(item.source_count),
                            cursors[0],
                            totals[1],
                            "session relocation source",
                        ),
                        canonical_span_end(
                            int(item.destination_offset),
                            int(item.destination_count),
                            cursors[1],
                            totals[2],
                            "session relocation destination",
                        ),
                        canonical_span_end(
                            int(item.move_offset),
                            int(item.move_count),
                            cursors[2],
                            totals[3],
                            "session relocation move",
                        ),
                    ]
                    if (
                        int(item.request_id) != expected.request_id
                        or int(item.class_id) != expected.class_id
                        or int(item.target_view_version)
                        != int(item.base_view_version) + 1
                        or int(item.projected_reclaimed_pages) == 0
                        or int(item.fragmentation_milli) > 1000
                    ):
                        raise ManagerError(
                            "session relocation plan identity or metadata is invalid"
                        )
                    plans.append(
                        EngineRelocationPlan(
                            expected.request_id,
                            expected.class_id,
                            int(item.base_view_version),
                            int(item.target_view_version),
                            int(item.fragmentation_milli),
                            tuple(
                                _page(self._relocation_source_pages[position])
                                for position in range(cursors[0], ends[0])
                            ),
                            tuple(
                                _page(
                                    self._relocation_destination_pages[position]
                                )
                                for position in range(cursors[1], ends[1])
                            ),
                            tuple(
                                _move(self._relocation_moves[position])
                                for position in range(cursors[2], ends[2])
                            ),
                            int(item.projected_reclaimed_pages),
                        )
                    )
                    cursors = ends
                if tuple(cursors) != totals[1:]:
                    raise ManagerError(
                        "session relocation spans do not cover flat buffers"
                    )
                prepared = EnginePreparedRelocation(
                    operation_id, tuple(plans)
                )
                self._pending_relocations[operation_id] = _PendingRelocation(
                    prepared, "prepared"
                )
                return prepared
            except Exception as error:
                raise self._poison(
                    f"runtime session relocation prepare output is invalid: {error}"
                ) from error

    def abort_prepared_relocation(
        self,
        relocation_id: EngineRelocationId,
        evidence: Sequence[EngineRelocationAbortEvidence],
    ) -> None:
        with self._lock:
            self._require_handle()
            pending = self._pending_relocations.get(relocation_id)
            if pending is None or pending.phase != "prepared":
                raise ManagerError(
                    "session relocation is not locally prepared"
                )
            values = tuple(evidence)
            if len(values) != len(pending.prepared.plans):
                raise ManagerError(
                    "relocation abort evidence cardinality differs from plan"
                )
            if any(type(item) is not EngineRelocationAbortEvidence for item in values):
                raise ManagerError(
                    "relocation abort evidence must be "
                    "EngineRelocationAbortEvidence"
                )
            for index, item in enumerate(values):
                self._relocation_abort_evidence[index] = (
                    L.SessionRelocationAbortEvidenceLayout(
                        _uint("engine request id", item.request_id, 64),
                        _wire_bool(
                            "relocation backend unobserved",
                            item.backend_unobserved,
                            32,
                        ),
                        0,
                    )
                )
            if any(
                item.request_id != plan.request_id
                for item, plan in zip(
                    values, pending.prepared.plans, strict=True
                )
            ):
                raise ManagerError(
                    "relocation abort evidence ordering differs from plan"
                )
            self._call(
                "runtime session abort prepared relocation",
                self._library.function(
                    "orbitkv_session_abort_prepared_relocation"
                ),
                self._require_handle(),
                self._relocation_id_c(relocation_id),
                self._relocation_abort_evidence,
                len(values),
            )
            self._pending_relocations.pop(relocation_id)

    def quarantine_relocation(
        self, relocation_id: EngineRelocationId
    ) -> None:
        with self._lock:
            self._require_handle()
            if relocation_id not in self._pending_relocations:
                raise ManagerError(
                    "session relocation is not locally pending"
                )
            try:
                self._call(
                    "runtime session quarantine relocation",
                    self._library.function(
                        "orbitkv_session_quarantine_relocation"
                    ),
                    self._require_handle(),
                    self._relocation_id_c(relocation_id),
                )
                raise self._poison(
                    "runtime session relocation quarantine returned success"
                )
            finally:
                if self._poisoned is not None:
                    self._pending_relocations.pop(relocation_id, None)

    def submit_relocation(
        self, evidence: EngineRelocationExecutionEvidence
    ) -> EngineRelocationTicket:
        with self._lock:
            self._require_handle()
            if type(evidence) is not EngineRelocationExecutionEvidence:
                raise ManagerError(
                    "relocation evidence must be "
                    "EngineRelocationExecutionEvidence"
                )
            pending = self._pending_relocations.get(evidence.relocation_id)
            if pending is None or pending.phase != "prepared":
                raise ManagerError(
                    "session relocation is not locally prepared"
                )
            requests = tuple(evidence.requests)
            if len(requests) != len(pending.prepared.plans):
                raise ManagerError(
                    "relocation request evidence cardinality differs from plan"
                )
            if any(
                type(item) is not EngineRelocationRequestEvidence
                for item in requests
            ):
                raise ManagerError(
                    "relocation request evidence must be "
                    "EngineRelocationRequestEvidence"
                )
            copies = tuple(copy for item in requests for copy in item.copies)
            if len(copies) > self._relocation_move_capacity:
                raise ManagerError(
                    "relocation copy evidence exceeds its configured bound"
                )
            offset = 0
            for index, (request, plan) in enumerate(
                zip(requests, pending.prepared.plans, strict=True)
            ):
                if (
                    request.request_id != plan.request_id
                    or len(request.copies) != len(plan.moves)
                ):
                    raise ManagerError(
                        "relocation request evidence differs from plan"
                    )
                self._relocation_request_evidence[index] = (
                    L.SessionRelocationRequestEvidenceLayout(
                    _uint("engine request id", request.request_id, 64),
                    offset,
                    len(request.copies),
                    )
                )
                for copy, move in zip(
                    request.copies, plan.moves, strict=True
                ):
                    if type(copy) is not EngineRelocationCopyEvidence:
                        raise ManagerError(
                            "relocation copy evidence must be "
                            "EngineRelocationCopyEvidence"
                        )
                    if (
                        copy.token_id != move.token_id
                        or copy.source != move.source
                        or copy.destination != move.destination
                    ):
                        raise ManagerError(
                            "relocation copy evidence differs from plan"
                        )
                    self._relocation_copy_evidence[offset] = (
                        L.SessionRelocationCopyEvidenceLayout(
                        _uint("relocation token id", copy.token_id, 64),
                        _location_c(copy.source),
                        _location_c(copy.destination),
                        _wire_bool("relocation observed", copy.observed, 8),
                        _wire_bool("relocation copied", copy.copied, 8),
                        0,
                        0,
                        )
                    )
                    offset += 1
            try:
                self._call(
                    "runtime session submit relocation",
                    self._library.function(
                        "orbitkv_session_submit_relocation"
                    ),
                    self._require_handle(),
                    self._relocation_id_c(evidence.relocation_id),
                    self._relocation_request_evidence,
                    len(requests),
                    self._relocation_copy_evidence,
                    len(copies),
                )
            finally:
                if self._poisoned is not None:
                    self._pending_relocations.pop(
                        evidence.relocation_id, None
                    )
            self._pending_relocations[evidence.relocation_id] = (
                _PendingRelocation(pending.prepared, "submitted")
            )
            return EngineRelocationTicket(evidence.relocation_id)

    def complete_relocation(
        self,
        relocation_id: EngineRelocationId,
        evidence: EngineCompletionEvidence,
    ) -> EngineRelocationPublication:
        with self._lock:
            self._require_handle()
            pending = self._pending_relocations.get(relocation_id)
            if pending is None or pending.phase != "submitted":
                raise ManagerError(
                    "session relocation is not locally submitted"
                )
            if type(evidence) is not EngineCompletionEvidence:
                raise ManagerError(
                    "completion evidence must be EngineCompletionEvidence"
                )
            raw_completion = L.SessionCompletionEvidenceLayout(
                _uint("completion domain", evidence.completion_domain, 64),
                _uint("completion value", evidence.completion_value, 64),
                _wire_bool("completion confirmed", evidence.confirmed, 32),
                0,
            )
            counts = [ctypes.c_uint32(), ctypes.c_uint32()]
            self._call(
                "runtime session complete relocation",
                self._library.function("orbitkv_session_complete_relocation"),
                self._require_handle(),
                self._relocation_id_c(relocation_id),
                raw_completion,
                self._relocation_publications,
                self._operation_capacity,
                ctypes.byref(counts[0]),
                self._retirements,
                self._physical_pages,
                ctypes.byref(counts[1]),
            )
            try:
                totals = bounded_output_counts(
                    counts,
                    (self._operation_capacity, self._physical_pages),
                    "session relocation completion",
                )
                if totals[0] != len(pending.prepared.plans):
                    raise ManagerError(
                        "session relocation publication cardinality changed"
                    )
                publications = []
                for plan, raw in zip(
                    pending.prepared.plans,
                    self._relocation_publications[: totals[0]],
                    strict=True,
                ):
                    if (
                        int(raw.reserved) != 0
                        or int(raw.request_id) != plan.request_id
                        or int(raw.view_version) != plan.target_view_version
                    ):
                        raise ManagerError(
                            "session relocation publication ordering is invalid"
                        )
                    publications.append(
                        EngineRelocationRequestPublication(
                            plan.request_id,
                            int(raw.view_version),
                            int(raw.boundary),
                            int(raw.resident_count),
                        )
                    )
                publication = EngineRelocationPublication(
                    relocation_id,
                    tuple(publications),
                    tuple(
                        _retirement(self._retirements[index])
                        for index in range(totals[1])
                    ),
                )
                self._pending_relocations[relocation_id] = _PendingRelocation(
                    pending.prepared, "publication_pending", publication
                )
                return publication
            except Exception as error:
                raise self._poison(
                    "runtime session relocation completion output is invalid: "
                    f"{error}"
                ) from error

    def confirm_relocation_publication(
        self, evidence: EngineRelocationPublicationEvidence
    ) -> None:
        with self._lock:
            self._require_handle()
            if type(evidence) is not EngineRelocationPublicationEvidence:
                raise ManagerError(
                    "relocation publication evidence must be "
                    "EngineRelocationPublicationEvidence"
                )
            pending = self._pending_relocations.get(evidence.relocation_id)
            if (
                pending is None
                or pending.phase != "publication_pending"
                or pending.publication is None
            ):
                raise ManagerError(
                    "session relocation publication is not locally pending"
                )
            receipts = tuple(evidence.reclamation_receipts)
            retirements = pending.publication.retirements
            if len(receipts) != len(retirements):
                raise ManagerError(
                    "relocation retirement evidence cardinality differs from plan"
                )
            self._validate_retirement_evidence(
                retirements, receipts, "relocation"
            )
            for index, item in enumerate(receipts):
                self._retirement_evidence[index] = (
                    _retirement_evidence_c(item)
                )
            raw = L.SessionRelocationPublicationEvidenceLayout(
                self._relocation_id_c(evidence.relocation_id),
                _wire_bool(
                    "relocation mirror cleanup",
                    evidence.mirror_cleanup_confirmed,
                    32,
                ),
                0,
            )
            self._call(
                "runtime session confirm relocation publication",
                self._library.function(
                    "orbitkv_session_confirm_relocation_publication"
                ),
                self._require_handle(),
                raw,
                self._retirement_evidence,
                len(receipts),
            )
            self._pending_relocations.pop(evidence.relocation_id)


__all__ = ["SessionRelocationMixin"]
