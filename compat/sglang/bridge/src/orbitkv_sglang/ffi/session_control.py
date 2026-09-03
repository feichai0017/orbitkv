from __future__ import annotations

import ctypes
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

from orbitkv_sglang.runtime import (
    CacheSharingPolicy,
    ManagerError,
    PrefixSemanticKey,
)

from . import layouts as L
from .codec import key as _key
from .codec import key_to_c as _key_to_c
from .codec import snapshot_page as _snapshot_page
from .codec import uint as _uint
from .session_support import canonical_span_end, decode_operation_id, require_batch_count
from .session_types import (
    EngineControlDisposition,
    EngineControlEvidence,
    EngineControlId,
    EngineControlKind,
    EngineControlOutcome,
    EngineControlPlanInfo,
    EngineMaterializedRequest,
    EngineMaterializationPlan,
    EnginePendingAttachCancel,
    EnginePendingAttachCancelDisposition,
    EnginePendingAttachCancelOutcome,
    EnginePrefixAttachItem,
    EnginePrefixEvictionPlan,
    EnginePrefixId,
    EnginePrefixLookup,
    EnginePrefixPublishItem,
    EnginePublishedPrefix,
    EngineRequestForkItem,
    EngineRequestId,
    EngineRetirement,
    EngineRetirementEvidence,
    _retirement,
    _retirement_evidence_c,
    _session_id_c,
    _wire_bool,
)
from .workspace import array


@dataclass(frozen=True, slots=True)
class _PendingControl:
    plan_info: EngineControlPlanInfo | None = None
    plan: EngineMaterializationPlan | EnginePrefixEvictionPlan | None = None
    retirements: tuple[EngineRetirement, ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingAttachCancelReplay:
    outcome: EnginePendingAttachCancelOutcome


class SessionControlMixin:
    _library: Any
    _lock: Any
    _handle: ctypes.c_void_p
    _request_capacity: int
    _operation_capacity: int
    _configured_operation_capacity: int
    _cache_sharing_policy: CacheSharingPolicy
    _prefix_capacity: int
    _physical_pages: int
    _session_epoch: int
    _retirement_evidence: Any
    _pending_controls: dict[EngineControlId, _PendingControl]
    _pending_attach_cancels: dict[EngineControlId, _PendingAttachCancelReplay]
    _finalized_attach_cancels: OrderedDict[
        EngineControlId, EnginePendingAttachCancelOutcome
    ]
    _finalized_attach_cancel_limit: int

    @property
    def control_batch_capacity(self) -> int: ...

    @property
    def prefix_eviction_batch_capacity(self) -> int: ...

    def _require_handle(self, *, allow_poisoned: bool = False) -> ctypes.c_void_p: ...
    def _call(
        self,
        operation: str,
        function: Any,
        *args: Any,
        require_handle: bool = True,
        allow_poisoned: bool = False,
    ) -> int: ...
    def _poison(self, reason: str) -> BaseException: ...
    def _validate_retirement_evidence(
        self,
        retirements: tuple[EngineRetirement, ...],
        evidence: tuple[EngineRetirementEvidence, ...],
        operation: str,
    ) -> None: ...

    def _ensure_control_state(self) -> None:
        if not hasattr(self, "_pending_controls"):
            self._pending_controls = {}
        if not hasattr(self, "_pending_attach_cancels"):
            self._pending_attach_cancels = {}
        if not hasattr(self, "_finalized_attach_cancels"):
            self._finalized_attach_cancels = OrderedDict()
        if not hasattr(self, "_finalized_attach_cancel_limit"):
            self._finalized_attach_cancel_limit = max(
                1, int(getattr(self, "_operation_capacity", 1))
            )

    def _require_shared_cache_policy(self, operation: str) -> None:
        if self._cache_sharing_policy is not CacheSharingPolicy.SHARED_PREFIX:
            raise ManagerError(
                f"{operation} is unavailable under request-private cache "
                "sharing policy"
            )

    @staticmethod
    def _prefix_id_c(value: EnginePrefixId) -> L.SessionPrefixIdLayout:
        return _session_id_c(
            value, EnginePrefixId, L.SessionPrefixIdLayout, "prefix id"
        )

    @staticmethod
    def _control_id_c(value: EngineControlId) -> L.SessionControlIdLayout:
        return _session_id_c(
            value, EngineControlId, L.SessionControlIdLayout, "control id"
        )

    def _pending_attach_cancel_c(
        self, value: EnginePendingAttachCancel
    ) -> L.SessionPendingAttachCancelLayout:
        if type(value) is not EnginePendingAttachCancel:
            raise ManagerError(
                "pending attach cancel must be EnginePendingAttachCancel"
            )
        return L.SessionPendingAttachCancelLayout(
            self._control_id_c(value.control_id),
            _uint("pending attach request id", value.request_id, 64),
            self._prefix_id_c(value.prefix_id),
            _uint("pending attach view version", value.view_version, 64),
            _uint("pending attach boundary", value.boundary, 64),
            _uint("pending attach resident count", value.resident_count, 32),
        )

    def _decode_pending_attach_cancel_outcome(
        self,
        value: Any,
        expected: EnginePendingAttachCancel,
    ) -> EnginePendingAttachCancelOutcome:
        control_id = decode_operation_id(
            value.control_id,
            EngineControlId,
            L.SessionControlIdLayout,
            "pending attach cancel control id",
        )
        if control_id != expected.control_id:
            raise ManagerError(
                "pending attach cancel outcome control id differs from request"
            )
        if control_id.session_epoch != self._session_epoch:
            raise ManagerError(
                "pending attach cancel outcome returned a foreign control id"
            )
        prefix_id = decode_operation_id(
            value.prefix_id,
            EnginePrefixId,
            L.SessionPrefixIdLayout,
            "pending attach cancel prefix id",
        )
        if prefix_id != expected.prefix_id:
            raise ManagerError(
                "pending attach cancel outcome prefix id differs from request"
            )
        if prefix_id.session_epoch != self._session_epoch:
            raise ManagerError(
                "pending attach cancel outcome returned a foreign prefix id"
            )
        outcome = EnginePendingAttachCancelOutcome(
            control_id,
            EngineRequestId(int(value.request_id)),
            prefix_id,
            int(value.view_version),
            int(value.boundary),
            int(value.resident_count),
            EnginePendingAttachCancelDisposition(int(value.disposition)),
        )
        if (
            outcome.control_id != expected.control_id
            or outcome.request_id != expected.request_id
            or outcome.prefix_id != expected.prefix_id
            or outcome.view_version != expected.view_version
            or outcome.boundary != expected.boundary
            or outcome.resident_count != expected.resident_count
        ):
            raise ManagerError(
                "pending attach cancel outcome identity differs from request"
            )
        return outcome

    @staticmethod
    def _matches_pending_attach_expected(
        outcome: EnginePendingAttachCancelOutcome,
        expected: EnginePendingAttachCancel,
    ) -> bool:
        return (
            outcome.control_id == expected.control_id
            and outcome.request_id == expected.request_id
            and outcome.prefix_id == expected.prefix_id
            and outcome.view_version == expected.view_version
            and outcome.boundary == expected.boundary
            and outcome.resident_count == expected.resident_count
        )

    def _remember_finalized_attach_cancel(
        self, outcome: EnginePendingAttachCancelOutcome
    ) -> None:
        self._finalized_attach_cancels.pop(outcome.control_id, None)
        self._finalized_attach_cancels[outcome.control_id] = outcome
        while (
            len(self._finalized_attach_cancels)
            > self._finalized_attach_cancel_limit
        ):
            self._finalized_attach_cancels.popitem(last=False)

    def _decode_prefix_lookup(self, value: Any) -> EnginePrefixLookup:
        present = int(value.candidate_present)
        if present not in (0, 1):
            raise ManagerError("session prefix lookup candidate presence is invalid")
        if int(value.reserved0) != 0 or int(value.reserved1) != 0:
            raise ManagerError("session prefix lookup reserved field is nonzero")
        candidate = None
        if present == 0:
            if (
                int(value.candidate.session_epoch) != 0
                or int(value.candidate.sequence) != 0
                or int(value.resident_count) != 0
            ):
                raise ManagerError(
                    "session prefix lookup miss must return zero candidate and residents"
                )
        else:
            candidate = decode_operation_id(
                value.candidate,
                EnginePrefixId,
                L.SessionPrefixIdLayout,
                "prefix lookup candidate",
            )
            if candidate.session_epoch != self._session_epoch:
                raise ManagerError("session prefix lookup returned a foreign prefix id")
        return EnginePrefixLookup(_key(value.key), candidate, int(value.resident_count))

    @staticmethod
    def _decode_plan_info(value: Any) -> EngineControlPlanInfo:
        if int(value.reserved) != 0:
            raise ManagerError("session control plan info reserved field is nonzero")
        return EngineControlPlanInfo(
            decode_operation_id(
                value.id, EngineControlId, L.SessionControlIdLayout, "control id"
            ),
            EngineControlKind(int(value.kind)),
            int(value.request_count),
            int(value.page_count),
            int(value.prefix_count),
            int(value.retirement_count),
        )

    @staticmethod
    def _validate_plan_shape(plan_info: EngineControlPlanInfo) -> None:
        if plan_info.kind is EngineControlKind.MATERIALIZATION:
            if (
                plan_info.request_count <= 0
                or plan_info.page_count < 0
                or plan_info.prefix_count != 0
                or plan_info.retirement_count != 0
            ):
                raise ManagerError("materialization control plan counts are invalid")
        elif plan_info.kind is EngineControlKind.PREFIX_EVICTION:
            if plan_info.request_count != 0 or plan_info.page_count != 0:
                raise ManagerError(
                    "eviction control plan cannot contain materialized requests"
                )
            if plan_info.prefix_count <= 0:
                raise ManagerError("eviction control plan must contain prefix ids")

    def prefix_lookup_batch(
        self, keys: Sequence[PrefixSemanticKey]
    ) -> tuple[EnginePrefixLookup, ...]:
        with self._lock:
            self._require_shared_cache_policy("session prefix lookup")
            self._require_handle()
            values = tuple(keys)
            count = require_batch_count(
                values, "session prefix lookup", self._prefix_capacity
            )
            raw = (L.PrefixKeyLayout * count)(*(_key_to_c(value) for value in values))
            output = array(L.SessionPrefixLookupLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "runtime session prefix lookup batch",
                self._library.function("orbitkv_session_prefix_lookup_batch"),
                self._require_handle(),
                raw,
                count,
                output,
                count,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison("runtime session prefix lookup cardinality changed")
            try:
                result = tuple(
                    self._decode_prefix_lookup(output[index]) for index in range(count)
                )
                if any(
                    item.key != expected
                    for item, expected in zip(result, values, strict=True)
                ):
                    raise ManagerError("session prefix lookup key ordering changed")
                seen: set[EnginePrefixId] = set()
                for item in result:
                    if item.candidate is None:
                        continue
                    if item.candidate in seen:
                        raise ManagerError(
                            "session prefix lookup returned duplicate prefix ids"
                        )
                    seen.add(item.candidate)
                return result
            except Exception as error:
                raise self._poison(
                    f"runtime session prefix lookup output is invalid: {error}"
                ) from error

    def prefix_publish_batch(
        self, items: Sequence[EnginePrefixPublishItem]
    ) -> tuple[EnginePublishedPrefix, ...]:
        with self._lock:
            self._require_shared_cache_policy("session prefix publish")
            self._require_handle()
            values = tuple(items)
            count = require_batch_count(
                values, "session prefix publish", self._prefix_capacity
            )
            if any(type(item) is not EnginePrefixPublishItem for item in values):
                raise ManagerError(
                    "session prefix publish items must be EnginePrefixPublishItem values"
                )
            raw = (L.SessionPrefixPublishItemLayout * count)(
                *(
                    L.SessionPrefixPublishItemLayout(
                        _uint("engine request id", item.request_id, 64),
                        _key_to_c(item.key),
                    )
                    for item in values
                )
            )
            output = array(L.SessionPublishedPrefixLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "runtime session prefix publish batch",
                self._library.function("orbitkv_session_prefix_publish_batch"),
                self._require_handle(),
                raw,
                count,
                output,
                count,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison("runtime session prefix publish cardinality changed")
            try:
                result: list[EnginePublishedPrefix] = []
                seen: set[EnginePrefixId] = set()
                for index in range(count):
                    item = output[index]
                    if int(item.reserved) != 0:
                        raise ManagerError(
                            "session published prefix reserved field is nonzero"
                        )
                    prefix_id = decode_operation_id(
                        item.prefix_id,
                        EnginePrefixId,
                        L.SessionPrefixIdLayout,
                        "published prefix id",
                    )
                    if prefix_id.session_epoch != self._session_epoch:
                        raise ManagerError(
                            "session prefix publish returned a foreign prefix id"
                        )
                    if prefix_id in seen:
                        raise ManagerError(
                            "session prefix publish returned duplicate prefix ids"
                        )
                    seen.add(prefix_id)
                    result.append(
                        EnginePublishedPrefix(prefix_id, _key(item.key), int(item.resident_count))
                    )
                if any(
                    item.key != expected.key
                    for item, expected in zip(result, values, strict=True)
                ):
                    raise ManagerError(
                        "session prefix publish output key ordering changed"
                    )
                return tuple(result)
            except Exception as error:
                raise self._poison(
                    f"runtime session prefix publish output is invalid: {error}"
                ) from error

    def _prepare_control(
        self,
        operation: str,
        symbol: str,
        raw_items: Any,
        count: int,
    ) -> EngineControlId:
        control_id = L.SessionControlIdLayout()
        self._call(
            operation,
            self._library.function(symbol),
            self._require_handle(),
            raw_items,
            count,
            ctypes.byref(control_id),
        )
        result = decode_operation_id(
            control_id, EngineControlId, L.SessionControlIdLayout, "control id"
        )
        if result.session_epoch != self._session_epoch:
            raise self._poison(f"{operation} returned a foreign control id")
        if result in self._pending_controls:
            raise self._poison(f"{operation} returned a duplicate control id")
        self._pending_controls[result] = _PendingControl()
        return result

    def prepare_prefix_attach(
        self, items: Sequence[EnginePrefixAttachItem]
    ) -> EngineControlId:
        with self._lock:
            self._require_shared_cache_policy("session prefix attach")
            self._require_handle()
            self._ensure_control_state()
            values = tuple(items)
            count = require_batch_count(
                values,
                "session prepare prefix attach",
                min(self._prefix_capacity, self._operation_capacity),
            )
            if any(type(item) is not EnginePrefixAttachItem for item in values):
                raise ManagerError(
                    "session prefix attach items must be EnginePrefixAttachItem values"
                )
            raw = (L.SessionPrefixAttachItemLayout * count)(
                *(
                    L.SessionPrefixAttachItemLayout(
                        _uint("target request id", item.target_request_id, 64),
                        self._prefix_id_c(item.prefix_id),
                        _key_to_c(item.key),
                        _uint("resident count", item.resident_count, 32),
                        0,
                    )
                    for item in values
                )
            )
            return self._prepare_control(
                "runtime session prepare prefix attach",
                "orbitkv_session_prepare_prefix_attach",
                raw,
                count,
            )

    def prepare_request_fork(
        self, items: Sequence[EngineRequestForkItem]
    ) -> EngineControlId:
        with self._lock:
            self._require_shared_cache_policy("session request fork")
            self._require_handle()
            self._ensure_control_state()
            values = tuple(items)
            count = require_batch_count(
                values,
                "session prepare request fork",
                self._operation_capacity,
            )
            if any(type(item) is not EngineRequestForkItem for item in values):
                raise ManagerError(
                    "session request fork items must be EngineRequestForkItem values"
                )
            raw = (L.SessionRequestForkItemLayout * count)(
                *(
                    L.SessionRequestForkItemLayout(
                        _uint("source request id", item.source_request_id, 64),
                        _uint("target request id", item.target_request_id, 64),
                    )
                    for item in values
                )
            )
            return self._prepare_control(
                "runtime session prepare request fork",
                "orbitkv_session_prepare_request_fork",
                raw,
                count,
            )

    def prepare_prefix_evict(
        self, prefix_ids: Sequence[EnginePrefixId]
    ) -> EngineControlId:
        with self._lock:
            self._require_shared_cache_policy("session prefix eviction")
            self._require_handle()
            self._ensure_control_state()
            values = tuple(prefix_ids)
            count = require_batch_count(
                values,
                "session prepare prefix evict",
                self.prefix_eviction_batch_capacity,
            )
            if any(type(item) is not EnginePrefixId for item in values):
                raise ManagerError(
                    "session prefix evict ids must be EnginePrefixId values"
                )
            raw = (L.SessionPrefixIdLayout * count)(
                *(self._prefix_id_c(item) for item in values)
            )
            return self._prepare_control(
                "runtime session prepare prefix evict",
                "orbitkv_session_prepare_prefix_evict",
                raw,
                count,
            )

    def abort_control(self, control_id: EngineControlId) -> None:
        with self._lock:
            self._require_shared_cache_policy("session control abort")
            self._require_handle()
            self._ensure_control_state()
            pending = self._pending_controls.get(control_id)
            if pending is None or pending.plan_info is not None:
                raise ManagerError("session control is not locally prepared")
            self._call(
                "runtime session abort control",
                self._library.function("orbitkv_session_abort_control"),
                self._require_handle(),
                self._control_id_c(control_id),
            )
            self._pending_controls.pop(control_id, None)

    def commit_control(self, control_id: EngineControlId) -> EngineControlPlanInfo:
        with self._lock:
            self._require_shared_cache_policy("session control commit")
            self._require_handle()
            self._ensure_control_state()
            pending = self._pending_controls.get(control_id)
            if pending is None:
                raise ManagerError("session control is not locally pending")
            if pending.plan_info is not None:
                return pending.plan_info
            raw_plan = L.SessionControlPlanInfoLayout()
            self._call(
                "runtime session commit control",
                self._library.function("orbitkv_session_commit_control"),
                self._require_handle(),
                self._control_id_c(control_id),
                ctypes.byref(raw_plan),
            )
            try:
                plan_info = self._decode_plan_info(raw_plan)
                if plan_info.control_id != control_id:
                    raise ManagerError(
                        "session control plan info returned the wrong control id"
                    )
                self._validate_plan_shape(plan_info)
                self._pending_controls[control_id] = _PendingControl(
                    plan_info, None, ()
                )
                return plan_info
            except Exception as error:
                raise self._poison(
                    f"runtime session control plan info is invalid: {error}"
                ) from error

    def read_control_plan(
        self, control_id: EngineControlId
    ) -> EngineMaterializationPlan | EnginePrefixEvictionPlan:
        with self._lock:
            self._require_shared_cache_policy("session control read")
            self._require_handle()
            self._ensure_control_state()
            pending = self._pending_controls.get(control_id)
            if pending is None or pending.plan_info is None:
                raise ManagerError("session control is not locally committed")
            if pending.plan is not None:
                return pending.plan
            plan_info = pending.plan_info
            exact_counts = (
                plan_info.request_count,
                plan_info.page_count,
                plan_info.prefix_count,
                plan_info.retirement_count,
            )
            requests = array(L.SessionMaterializedRequestLayout, exact_counts[0])
            pages = array(L.SnapshotPageLayout, exact_counts[1])
            prefixes = array(L.SessionPrefixIdLayout, exact_counts[2])
            retirements = array(L.SessionRetirementLayout, exact_counts[3])
            out = [ctypes.c_uint32() for _ in range(4)]
            self._call(
                "runtime session read control plan",
                self._library.function("orbitkv_session_read_control_plan"),
                self._require_handle(),
                self._control_id_c(control_id),
                requests,
                exact_counts[0],
                ctypes.byref(out[0]),
                pages,
                exact_counts[1],
                ctypes.byref(out[1]),
                prefixes,
                exact_counts[2],
                ctypes.byref(out[2]),
                retirements,
                exact_counts[3],
                ctypes.byref(out[3]),
            )
            actual_counts = tuple(int(item.value) for item in out)
            if actual_counts != exact_counts:
                raise self._poison(
                    "runtime session control read changed committed exact counts"
                )
            try:
                page_cursor = 0
                requests_out: list[EngineMaterializedRequest] = []
                for index in range(actual_counts[0]):
                    item = requests[index]
                    if int(item.reserved) != 0:
                        raise ManagerError(
                            "session materialized request reserved field is nonzero"
                        )
                    if int(item.resident_count) != int(item.page_count):
                        raise ManagerError(
                            "session materialized request resident_count must equal page_count"
                        )
                    page_end = canonical_span_end(
                        int(item.page_offset),
                        int(item.page_count),
                        page_cursor,
                        actual_counts[1],
                        "session control pages",
                    )
                    requests_out.append(
                        EngineMaterializedRequest(
                            EngineRequestId(int(item.request_id)),
                            int(item.view_version),
                            int(item.boundary),
                            int(item.resident_count),
                            tuple(
                                _snapshot_page(pages[position])
                                for position in range(page_cursor, page_end)
                            ),
                        )
                    )
                    page_cursor = page_end
                if page_cursor != actual_counts[1]:
                    raise ManagerError(
                        "session control page spans do not cover output"
                    )
                prefix_ids = tuple(
                    decode_operation_id(
                        prefixes[index],
                        EnginePrefixId,
                        L.SessionPrefixIdLayout,
                        "session control prefix id",
                    )
                    for index in range(actual_counts[2])
                )
                if any(item.session_epoch != self._session_epoch for item in prefix_ids):
                    raise ManagerError("session control returned a foreign prefix id")
                if len(set(prefix_ids)) != len(prefix_ids):
                    raise ManagerError("session control returned duplicate prefix ids")
                plan_retirements = tuple(
                    _retirement(retirements[index])
                    for index in range(actual_counts[3])
                )
                if plan_info.kind is EngineControlKind.MATERIALIZATION:
                    if actual_counts[2] != 0 or actual_counts[3] != 0:
                        raise ManagerError(
                            "materialization control plan returned inactive outputs"
                        )
                    plan = EngineMaterializationPlan(
                        control_id,
                        tuple(requests_out),
                    )
                else:
                    if actual_counts[0] != 0 or actual_counts[1] != 0:
                        raise ManagerError(
                            "eviction control plan returned inactive outputs"
                        )
                    plan = EnginePrefixEvictionPlan(
                        control_id,
                        prefix_ids,
                        plan_retirements,
                    )
                self._pending_controls[control_id] = _PendingControl(
                    plan_info, plan, plan_retirements
                )
                return plan
            except Exception as error:
                raise self._poison(
                    f"runtime session control plan output is invalid: {error}"
                ) from error

    def cancel_pending_attach(
        self, expected: EnginePendingAttachCancel
    ) -> EnginePendingAttachCancelOutcome:
        with self._lock:
            self._require_shared_cache_policy(
                "session pending attach cancel"
            )
            self._require_handle()
            self._ensure_control_state()
            replay = self._pending_attach_cancels.get(expected.control_id)
            if replay is not None:
                if not self._matches_pending_attach_expected(
                    replay.outcome, expected
                ):
                    raise ManagerError(
                        "pending attach cancel expectation differs from replay"
                    )
                return replay.outcome
            finalized = self._finalized_attach_cancels.get(expected.control_id)
            if finalized is not None:
                if not self._matches_pending_attach_expected(
                    finalized, expected
                ):
                    raise ManagerError(
                        "pending attach cancel expectation differs from replay"
                    )
                return finalized
            raw_expected = self._pending_attach_cancel_c(expected)
            raw_outcome = L.SessionPendingAttachCancelOutcomeLayout()
            self._call(
                "runtime session cancel pending attach",
                self._library.function("orbitkv_session_cancel_pending_attach"),
                self._require_handle(),
                raw_expected,
                ctypes.byref(raw_outcome),
            )
            try:
                outcome = self._decode_pending_attach_cancel_outcome(
                    raw_outcome, expected
                )
                if (
                    outcome.disposition
                    not in (
                        EnginePendingAttachCancelDisposition.RECYCLE_PENDING,
                        EnginePendingAttachCancelDisposition.FINALIZED,
                    )
                ):
                    raise ManagerError(
                        "pending attach cancel outcome disposition is invalid"
                    )
            except Exception as error:
                raise self._poison(
                    f"runtime session pending attach cancel outcome is invalid: {error}"
                ) from error
            if (
                outcome.disposition
                is EnginePendingAttachCancelDisposition.RECYCLE_PENDING
            ):
                self._pending_attach_cancels[expected.control_id] = (
                    _PendingAttachCancelReplay(outcome)
                )
            else:
                self._pending_attach_cancels.pop(expected.control_id, None)
                self._remember_finalized_attach_cancel(outcome)
            self._pending_controls.pop(expected.control_id, None)
            return outcome

    def finalize_pending_attach_cancel(
        self, expected: EnginePendingAttachCancel
    ) -> EnginePendingAttachCancelOutcome:
        with self._lock:
            self._require_shared_cache_policy(
                "session pending attach cancel finalize"
            )
            self._require_handle()
            self._ensure_control_state()
            finalized = self._finalized_attach_cancels.get(expected.control_id)
            if finalized is not None:
                if not self._matches_pending_attach_expected(
                    finalized, expected
                ):
                    raise ManagerError(
                        "pending attach finalize expectation differs from replay"
                    )
                return finalized
            replay = self._pending_attach_cancels.get(expected.control_id)
            if replay is not None and not self._matches_pending_attach_expected(
                replay.outcome, expected
            ):
                raise ManagerError(
                    "pending attach finalize expectation differs from replay"
                )
            raw_expected = self._pending_attach_cancel_c(expected)
            raw_outcome = L.SessionPendingAttachCancelOutcomeLayout()
            self._call(
                "runtime session finalize pending attach cancel",
                self._library.function(
                    "orbitkv_session_finalize_pending_attach_cancel"
                ),
                self._require_handle(),
                raw_expected,
                ctypes.byref(raw_outcome),
            )
            try:
                outcome = self._decode_pending_attach_cancel_outcome(
                    raw_outcome, expected
                )
                if (
                    outcome.disposition
                    is not EnginePendingAttachCancelDisposition.FINALIZED
                ):
                    raise ManagerError(
                        "pending attach finalize outcome must be finalized"
                    )
            except Exception as error:
                raise self._poison(
                    "runtime session pending attach finalize outcome is invalid: "
                    f"{error}"
                ) from error
            self._pending_attach_cancels.pop(expected.control_id, None)
            self._remember_finalized_attach_cancel(outcome)
            return outcome

    def confirm_control(
        self, evidence: EngineControlEvidence
    ) -> EngineControlOutcome:
        with self._lock:
            self._require_shared_cache_policy("session control confirmation")
            self._require_handle()
            self._ensure_control_state()
            if type(evidence) is not EngineControlEvidence:
                raise ManagerError("control evidence must be EngineControlEvidence")
            pending = self._pending_controls.get(evidence.control_id)
            if pending is None or pending.plan_info is None or pending.plan is None:
                raise ManagerError("session control plan is not locally read")
            receipts = tuple(evidence.reclamation_receipts)
            expected_retirements = pending.retirements
            if len(receipts) != len(expected_retirements):
                raise ManagerError(
                    "control retirement evidence cardinality differs from plan"
                )
            self._validate_retirement_evidence(
                expected_retirements,
                receipts,
                "control",
            )
            for index, item in enumerate(receipts):
                self._retirement_evidence[index] = _retirement_evidence_c(item)
            raw = L.SessionControlEvidenceLayout(
                self._control_id_c(evidence.control_id),
                _wire_bool(
                    "control mirror cleanup",
                    evidence.mirror_cleanup_confirmed,
                    32,
                ),
                0,
            )
            raw_outcome = L.SessionControlOutcomeLayout()
            self._call(
                "runtime session confirm control",
                self._library.function("orbitkv_session_confirm_control"),
                self._require_handle(),
                raw,
                self._retirement_evidence,
                len(receipts),
                ctypes.byref(raw_outcome),
            )
            try:
                if int(raw_outcome.reserved) != 0:
                    raise ManagerError(
                        "session control outcome reserved field is nonzero"
                    )
                outcome_id = decode_operation_id(
                    raw_outcome.id,
                    EngineControlId,
                    L.SessionControlIdLayout,
                    "control outcome id",
                )
                if outcome_id != evidence.control_id:
                    raise ManagerError(
                        "session control outcome id differs from confirmed control"
                    )
                outcome = EngineControlOutcome(
                    outcome_id,
                    EngineControlDisposition(int(raw_outcome.disposition)),
                )
                if (
                    pending.plan_info.kind is EngineControlKind.MATERIALIZATION
                    and outcome.disposition
                    is not EngineControlDisposition.MATERIALIZED
                ) or (
                    pending.plan_info.kind is EngineControlKind.PREFIX_EVICTION
                    and outcome.disposition
                    is not EngineControlDisposition.EVICTED
                ):
                    raise ManagerError(
                        "session control outcome disposition does not match plan kind"
                    )
            except Exception as error:
                raise self._poison(
                    f"runtime session control outcome is invalid: {error}"
                ) from error
            self._pending_controls.pop(evidence.control_id, None)
            return outcome

    def quarantine_control(self, control_id: EngineControlId) -> None:
        with self._lock:
            self._require_shared_cache_policy("session control quarantine")
            self._require_handle()
            self._ensure_control_state()
            pending = self._pending_controls.get(control_id)
            if pending is None or pending.plan_info is None:
                raise ManagerError("session control is not locally committed")
            self._call(
                "runtime session quarantine control",
                self._library.function("orbitkv_session_quarantine_control"),
                self._require_handle(),
                self._control_id_c(control_id),
            )
            self._pending_controls.pop(control_id, None)
