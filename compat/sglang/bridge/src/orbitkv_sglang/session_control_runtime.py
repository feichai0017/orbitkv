from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence

from .ffi.session_types import (
    EngineControlDisposition,
    EngineControlEvidence,
    EngineControlId,
    EngineControlKind,
    EngineControlOutcome,
    EngineControlPlanInfo,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
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
    EngineRequestView,
    EngineRetirement,
    retirement_evidence,
)
from .runtime import FailStopped, ManagerError, PrefixSemanticKey, SnapshotPage


@dataclass(frozen=True, slots=True)
class SessionMaterializationUpdate:
    """One native materialization translated to its engine mirror identity."""

    key: Hashable
    request_id: EngineRequestId
    request_row: int
    view_version: int
    boundary: int
    resident_count: int
    pages: tuple[SnapshotPage, ...]


MaterializationCallback = Callable[
    [tuple[SessionMaterializationUpdate, ...]], bool
]


def unbound_materialization(
    _updates: tuple[SessionMaterializationUpdate, ...],
) -> bool:
    """Fail-closed bootstrap marker accepted only for two-phase wiring."""

    raise FailStopped("session materialization authority is not bound")


@dataclass(slots=True)
class _ControlGroup:
    control_id: EngineControlId
    request_keys: tuple[Hashable, ...]
    target_keys: tuple[Hashable, ...]
    prefix_ids: tuple[EnginePrefixId, ...]
    plan_info: EngineControlPlanInfo | None = None
    plan: EngineMaterializationPlan | EnginePrefixEvictionPlan | None = None
    mirror_done: bool = False


class SessionControlRuntimeMixin:
    _materialization: MaterializationCallback | None
    _bindings: dict[Hashable, Any]
    _views: dict[Hashable, EngineRequestView]
    _releases: dict[Any, Any]
    _request_occupancy: dict[Hashable, EngineControlId]
    _prefix_occupancy: dict[EnginePrefixId, EngineControlId]
    _quarantined_requests: set[Hashable]
    _controls: dict[EngineControlId, _ControlGroup]
    _pending_attach_cancels: dict[EngineControlId, tuple[Hashable, Any, EnginePendingAttachCancel]]
    _prefixes: dict[PrefixSemanticKey, EnginePublishedPrefix]
    _lock: Any
    _session: Any

    def _healthy(self) -> None: ...
    def _binding(self, key: Hashable) -> Any: ...
    def _fail_stop(self, reason: str, error: BaseException | None = None) -> Any: ...
    def _run_cleanup(
        self,
        updates: tuple[Any, ...],
        retirements: Sequence[EngineRetirement],
        operation: str,
    ) -> None: ...
    def _require_shared_cache_policy(self, operation: str) -> None: ...

    @property
    def materialization_bound(self) -> bool:
        return self._materialization is not None

    def bind_materialization(
        self, materialization: MaterializationCallback
    ) -> None:
        """Install the sole materialization authority before first use."""

        with self._lock:
            self._healthy()
            if not callable(materialization):
                raise TypeError("materialization must be callable")
            if self._materialization is not None:
                raise ManagerError("materialization authority is already bound")
            self._materialization = materialization

    def prefix_lookup(
        self, keys: Sequence[PrefixSemanticKey]
    ) -> tuple[EnginePrefixLookup, ...]:
        with self._lock:
            self._require_shared_cache_policy("session prefix lookup")
            self._healthy()
            values = tuple(keys)
            if not values:
                raise ManagerError("session prefix lookup batch must be nonempty")
            try:
                lookups = tuple(self._session.prefix_lookup_batch(values))
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native prefix lookup became uncertain", error)
            if len(lookups) != len(values):
                self._fail_stop("native prefix lookup changed cardinality")
            seen: set[EnginePrefixId] = set()
            for expected, item in zip(values, lookups, strict=True):
                if type(item) is not EnginePrefixLookup or item.key != expected:
                    self._fail_stop(
                        "native prefix lookup changed identity or ordering"
                    )
                if item.candidate is not None:
                    if item.candidate in seen:
                        self._fail_stop(
                            "native prefix lookup duplicated a prefix identity"
                        )
                    seen.add(item.candidate)
            return lookups

    def prefix_publish(
        self, items: Sequence[tuple[Hashable, PrefixSemanticKey]]
    ) -> tuple[EnginePublishedPrefix, ...]:
        with self._lock:
            self._require_shared_cache_policy("session prefix publish")
            self._healthy()
            values = tuple(items)
            if not values:
                raise ManagerError("session prefix publish batch must be nonempty")
            raw: list[EnginePrefixPublishItem] = []
            keys: list[Hashable] = []
            semantics: list[PrefixSemanticKey] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ManagerError(
                        "session prefix publish items must be (key, semantic) pairs"
                    )
                key, semantic = item
                binding = self._binding(key)
                self._require_control_free(key)
                raw.append(EnginePrefixPublishItem(binding.request_id, semantic))
                keys.append(key)
                semantics.append(semantic)
            if len(set(keys)) != len(keys):
                raise ManagerError(
                    "session prefix publish contains duplicate request keys"
                )
            if len(set(semantics)) != len(semantics):
                raise ManagerError(
                    "session prefix publish contains duplicate semantic keys"
                )
            try:
                outputs = tuple(self._session.prefix_publish_batch(tuple(raw)))
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native prefix publish became uncertain", error)
            if len(outputs) != len(values):
                self._fail_stop("native prefix publish changed cardinality")
            next_prefixes = dict(self._prefixes)
            for semantic, output in zip(semantics, outputs, strict=True):
                if type(output) is not EnginePublishedPrefix or output.key != semantic:
                    self._fail_stop(
                        "native prefix publish changed identity or ordering"
                    )
                if output.prefix_id in self._prefix_occupancy:
                    self._fail_stop("native prefix publish returned an occupied prefix")
                next_prefixes[semantic] = output
            self._prefixes = next_prefixes
            return outputs

    def prepare_prefix_attach(
        self, items: Sequence[tuple[Hashable, EnginePrefixLookup]]
    ) -> EngineControlId:
        with self._lock:
            self._require_shared_cache_policy("session prefix attach")
            self._healthy()
            values = tuple(items)
            if not values:
                raise ManagerError("session prefix attach batch must be nonempty")
            raw: list[EnginePrefixAttachItem] = []
            request_keys: list[Hashable] = []
            target_keys: list[Hashable] = []
            prefix_ids: list[EnginePrefixId] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ManagerError(
                        "session prefix attach items must be (key, lookup) pairs"
                    )
                key, lookup = item
                if type(lookup) is not EnginePrefixLookup or lookup.candidate is None:
                    raise ManagerError("session prefix attach requires a lookup hit")
                binding = self._binding(key)
                self._require_control_free(key)
                request_keys.append(key)
                target_keys.append(key)
                prefix_ids.append(lookup.candidate)
                raw.append(
                    EnginePrefixAttachItem(
                        binding.request_id,
                        lookup.candidate,
                        lookup.key,
                        lookup.resident_count,
                    )
                )
            if len(set(target_keys)) != len(target_keys):
                raise ManagerError(
                    "session prefix attach contains duplicate request keys"
                )
            return self._prepare_control_group(
                self._session.prepare_prefix_attach,
                tuple(raw),
                tuple(request_keys),
                tuple(target_keys),
                tuple(prefix_ids),
                "native prefix attach preparation became uncertain",
            )

    def prepare_request_fork(
        self, items: Sequence[tuple[Hashable, Hashable]]
    ) -> EngineControlId:
        with self._lock:
            self._require_shared_cache_policy("session request fork")
            self._healthy()
            values = tuple(items)
            if not values:
                raise ManagerError("session request fork batch must be nonempty")
            raw: list[EngineRequestForkItem] = []
            request_keys: list[Hashable] = []
            target_keys: list[Hashable] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ManagerError(
                        "session request fork items must be (source_key, target_key) pairs"
                    )
                source_key, target_key = item
                source = self._binding(source_key)
                target = self._binding(target_key)
                self._require_control_free(source_key)
                self._require_control_free(target_key)
                request_keys.extend((source_key, target_key))
                target_keys.append(target_key)
                raw.append(
                    EngineRequestForkItem(source.request_id, target.request_id)
                )
            if len(set(target_keys)) != len(target_keys):
                raise ManagerError("session request fork duplicates a target request")
            return self._prepare_control_group(
                self._session.prepare_request_fork,
                tuple(raw),
                tuple(request_keys),
                tuple(target_keys),
                (),
                "native request fork preparation became uncertain",
            )

    def prepare_prefix_evict(
        self, prefix_ids: Sequence[EnginePrefixId]
    ) -> EngineControlId:
        with self._lock:
            self._require_shared_cache_policy("session prefix eviction")
            self._healthy()
            values = tuple(prefix_ids)
            if not values:
                raise ManagerError("session prefix eviction batch must be nonempty")
            if len(set(values)) != len(values):
                raise ManagerError(
                    "session prefix eviction contains duplicate prefix ids"
                )
            for prefix_id in values:
                owner = self._prefix_owner(prefix_id)
                if owner is not None:
                    raise ManagerError("prefix belongs to a pending control group")
            return self._prepare_control_group(
                self._session.prepare_prefix_evict,
                values,
                (),
                (),
                values,
                "native prefix eviction preparation became uncertain",
            )

    def commit_control(
        self, control_id: EngineControlId
    ) -> EngineControlPlanInfo:
        with self._lock:
            self._require_shared_cache_policy("session control commit")
            self._healthy()
            group = self._control_group(control_id)
            if group.plan_info is not None:
                return group.plan_info
            try:
                plan_info = self._session.commit_control(control_id)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native control commit became uncertain", error)
            if type(plan_info) is not EngineControlPlanInfo:
                self._fail_stop(
                    "native control commit returned a malformed plan info"
                )
            if plan_info.control_id != control_id:
                self._fail_stop("native control commit returned the wrong control id")
            group.plan_info = plan_info
            return plan_info

    def read_control(
        self, control_id: EngineControlId
    ) -> EngineMaterializationPlan | EnginePrefixEvictionPlan:
        with self._lock:
            self._require_shared_cache_policy("session control read")
            self._healthy()
            group = self._control_group(control_id)
            if group.plan_info is None:
                raise ManagerError("session control is not locally committed")
            if group.plan is not None:
                return group.plan
            try:
                plan = self._session.read_control_plan(control_id)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native control read became uncertain", error)
            kind = group.plan_info.kind
            if kind is EngineControlKind.MATERIALIZATION:
                if (
                    type(plan) is not EngineMaterializationPlan
                    or plan.control_id != control_id
                ):
                    self._fail_stop("native materialization read changed identity")
                if len(plan.requests) != group.plan_info.request_count:
                    self._fail_stop(
                        "native materialization read changed committed request count"
                    )
                group.plan = plan
                return plan
            if kind is not EngineControlKind.PREFIX_EVICTION:
                self._fail_stop("native control read returned an unsupported plan kind")
            if (
                type(plan) is not EnginePrefixEvictionPlan
                or plan.control_id != control_id
            ):
                self._fail_stop("native eviction read changed identity")
            if group.plan_info.prefix_count != len(plan.prefix_ids):
                self._fail_stop(
                    "native eviction read changed committed prefix count"
                )
            if tuple(plan.prefix_ids) != group.prefix_ids:
                self._fail_stop("native eviction read changed prefix identity")
            group.plan = plan
            return plan

    def abort_control(self, control_id: EngineControlId) -> None:
        with self._lock:
            self._require_shared_cache_policy("session control abort")
            self._healthy()
            group = self._control_group(control_id)
            if group.plan_info is not None:
                raise ManagerError("session control is not locally prepared")
            try:
                self._session.abort_control(control_id)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native control abort became uncertain", error)
            self._drop_control_group(control_id)

    def quarantine_control(self, control_id: EngineControlId) -> None:
        with self._lock:
            self._require_shared_cache_policy("session control quarantine")
            self._healthy()
            group = self._control_group(control_id)
            if group.plan_info is None:
                raise ManagerError("session control is not locally committed")
            try:
                self._session.quarantine_control(control_id)
            except BaseException as error:
                self._fail_stop("control quarantine became uncertain", error)
            self._mark_group_quarantined(group)
            self._drop_control_group(control_id)

    def cancel_pending_attach(
        self, expected: EnginePendingAttachCancel, key: Hashable
    ) -> EnginePendingAttachCancelOutcome:
        """Cancel one exact committed rowless Prefix attach."""

        with self._lock:
            self._require_shared_cache_policy(
                "session pending attach cancel"
            )
            self._healthy()
            if type(expected) is not EnginePendingAttachCancel:
                raise ManagerError(
                    "pending attach cancel requires exact typed identity"
                )
            retained = self._pending_attach_cancels.get(expected.control_id)
            if retained is not None:
                retained_key, binding, retained_expected = retained
                if retained_key != key or retained_expected != expected:
                    raise ManagerError(
                        "pending attach cancel differs from retained authority"
                    )
                outcome = self._session.cancel_pending_attach(expected)
                self._validate_pending_attach_cancel_outcome(
                    expected, outcome, allow_finalized=True
                )
                return outcome
            group = self._controls.get(expected.control_id)
            binding = self._binding(key)
            plan = None if group is None else group.plan
            if (
                group is None
                or group.control_id != expected.control_id
                or group.request_keys != (key,)
                or group.target_keys != (key,)
                or group.prefix_ids != (expected.prefix_id,)
                or group.plan_info is None
                or group.plan_info.kind is not EngineControlKind.MATERIALIZATION
                or type(plan) is not EngineMaterializationPlan
                or len(plan.requests) != 1
                or plan.requests[0].request_id != expected.request_id
                or int(plan.requests[0].view_version) != int(expected.view_version)
                or int(plan.requests[0].boundary) != int(expected.boundary)
                or int(plan.requests[0].resident_count)
                != int(expected.resident_count)
                or binding.request_id != expected.request_id
                or binding.request_row is not None
            ):
                raise ManagerError(
                    "pending attach cancel differs from local control authority"
                )
            try:
                outcome = self._session.cancel_pending_attach(expected)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop(
                    "native pending attach cancel became uncertain", error
                )
            self._validate_pending_attach_cancel_outcome(
                expected, outcome, allow_finalized=True
            )
            self._pending_attach_cancels[expected.control_id] = (
                key,
                binding,
                expected,
            )
            self._drop_control_group(expected.control_id)
            return outcome

    def finalize_pending_attach_cancel(
        self, expected: EnginePendingAttachCancel, key: Hashable
    ) -> EnginePendingAttachCancelOutcome:
        """Recycle an exact canceled attach target, with replay-safe output."""

        with self._lock:
            self._require_shared_cache_policy(
                "session pending attach cancel finalize"
            )
            self._healthy()
            retained = self._pending_attach_cancels.get(expected.control_id)
            if (
                retained is None
                or retained[0] != key
                or retained[2] != expected
            ):
                raise ManagerError(
                    "pending attach finalize has no retained local authority"
                )
            try:
                outcome = self._session.finalize_pending_attach_cancel(expected)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop(
                    "native pending attach cancel finalize became uncertain",
                    error,
                )
            self._validate_pending_attach_cancel_outcome(
                expected, outcome, allow_finalized=False
            )
            binding = retained[1]
            if (
                self._bindings.get(key) is not binding
                or self._views.get(key) is None
                or self._request_keys.get(binding.request_id) != key
            ):
                self._fail_stop(
                    "pending attach finalize lost retained request identity"
                )
            self._bindings.pop(key)
            self._views.pop(key)
            self._request_keys.pop(binding.request_id)
            self._pending_attach_cancels.pop(expected.control_id)
            return outcome

    def _validate_pending_attach_cancel_outcome(
        self,
        expected: EnginePendingAttachCancel,
        outcome: Any,
        *,
        allow_finalized: bool,
    ) -> None:
        valid_dispositions = (
            (
                EnginePendingAttachCancelDisposition.RECYCLE_PENDING,
                EnginePendingAttachCancelDisposition.FINALIZED,
            )
            if allow_finalized
            else (EnginePendingAttachCancelDisposition.FINALIZED,)
        )
        if (
            type(outcome) is not EnginePendingAttachCancelOutcome
            or outcome.identity != expected
            or outcome.disposition not in valid_dispositions
        ):
            self._fail_stop(
                "native pending attach cancel returned a hostile outcome"
            )

    def confirm_control(
        self,
        control_id: EngineControlId,
    ) -> EngineControlOutcome:
        with self._lock:
            self._require_shared_cache_policy("session control confirmation")
            self._healthy()
            group = self._control_group(control_id)
            if group.plan_info is None or group.plan is None:
                raise ManagerError("session control plan is not locally read")
            kind = group.plan_info.kind
            if kind is EngineControlKind.MATERIALIZATION:
                assert isinstance(group.plan, EngineMaterializationPlan)
                try:
                    updates = self._materialization_updates_for_plan(group, group.plan)
                except ManagerError as error:
                    self._best_effort_quarantine_control(control_id)
                    self._fail_stop(
                        "materialization mirror preflight became uncertain", error
                    )
                if not group.mirror_done:
                    self._run_materialization(group, updates)
                    group.mirror_done = True
                try:
                    outcome = self._session.confirm_control(
                        EngineControlEvidence(control_id, True, ())
                    )
                except ManagerError:
                    raise
                except BaseException as error:
                    self._fail_stop(
                        "native materialization confirmation became uncertain",
                        error,
                    )
                if (
                    type(outcome) is not EngineControlOutcome
                    or outcome.control_id != control_id
                ):
                    self._fail_stop(
                        "native materialization confirmation returned a malformed outcome"
                    )
                if outcome.disposition is not EngineControlDisposition.MATERIALIZED:
                    self._best_effort_quarantine_control(control_id)
                    self._fail_stop(
                        "native materialization confirmation returned a hostile outcome"
                    )
                self._install_materialized_views(updates)
            else:
                assert isinstance(group.plan, EnginePrefixEvictionPlan)
                self._run_cleanup((), group.plan.retirements, "control eviction")
                try:
                    outcome = self._session.confirm_control(
                        EngineControlEvidence(
                            control_id,
                            True,
                            retirement_evidence(group.plan.retirements),
                        )
                    )
                except ManagerError:
                    raise
                except BaseException as error:
                    self._fail_stop(
                        "native eviction confirmation became uncertain", error
                    )
                if (
                    type(outcome) is not EngineControlOutcome
                    or outcome.control_id != control_id
                ):
                    self._fail_stop(
                        "native eviction confirmation returned a malformed outcome"
                    )
                if outcome.disposition is not EngineControlDisposition.EVICTED:
                    self._best_effort_quarantine_control(control_id)
                    self._fail_stop(
                        "native eviction confirmation returned a hostile outcome"
                    )
                next_prefixes = dict(self._prefixes)
                for semantic, publication in tuple(next_prefixes.items()):
                    if publication.prefix_id in group.prefix_ids:
                        del next_prefixes[semantic]
                self._prefixes = next_prefixes
            self._drop_control_group(control_id)
            return outcome

    def _prepare_control_group(
        self,
        native_prepare: Callable[[Any], EngineControlId],
        native_items: Any,
        request_keys: tuple[Hashable, ...],
        target_keys: tuple[Hashable, ...],
        prefix_ids: tuple[EnginePrefixId, ...],
        uncertainty_reason: str,
    ) -> EngineControlId:
        if len(set(request_keys)) != len(request_keys):
            raise ManagerError("session control participants must be unique")
        if len(set(prefix_ids)) != len(prefix_ids):
            raise ManagerError("session control prefixes must be unique")
        try:
            control_id = native_prepare(native_items)
        except ManagerError:
            raise
        except BaseException as error:
            self._fail_stop(uncertainty_reason, error)
        if type(control_id) is not EngineControlId:
            self._fail_stop("native control preparation returned a malformed id")
        if control_id in self._controls:
            self._fail_stop("native control preparation returned a duplicate id")
        group = _ControlGroup(
            control_id=control_id,
            request_keys=request_keys,
            target_keys=target_keys,
            prefix_ids=prefix_ids,
        )
        self._controls[control_id] = group
        for key in request_keys:
            self._request_occupancy[key] = control_id
        for prefix_id in prefix_ids:
            self._prefix_occupancy[prefix_id] = control_id
        return control_id

    def _control_group(self, control_id: EngineControlId) -> _ControlGroup:
        try:
            return self._controls[control_id]
        except KeyError as error:
            raise ManagerError("unknown session control id") from error

    def _drop_control_group(self, control_id: EngineControlId) -> None:
        group = self._controls.pop(control_id, None)
        if group is None:
            return
        for key in group.request_keys:
            if self._request_occupancy.get(key) == control_id:
                del self._request_occupancy[key]
        for prefix_id in group.prefix_ids:
            if self._prefix_occupancy.get(prefix_id) == control_id:
                del self._prefix_occupancy[prefix_id]

    def _mark_group_quarantined(self, group: _ControlGroup) -> None:
        for key in group.target_keys:
            self._quarantined_requests.add(key)

    def _best_effort_quarantine_control(self, control_id: EngineControlId) -> None:
        try:
            self._session.quarantine_control(control_id)
        except BaseException:
            return
        group = self._controls.get(control_id)
        if group is not None:
            self._mark_group_quarantined(group)
            self._drop_control_group(control_id)

    def _require_control_free(self, key: Hashable) -> None:
        if key in self._quarantined_requests:
            raise ManagerError("request is quarantined")
        if key in self._request_occupancy:
            raise ManagerError("request participates in a pending control group")
        if any(key in group.keys for group in self._releases.values()):
            raise ManagerError("session request belongs to a pending release group")

    def _require_row_binding_allowed(self, key: Hashable) -> None:
        """Allow only a free request or one exact materialization target."""

        if key in self._quarantined_requests:
            raise ManagerError("request is quarantined")
        if any(key in group.keys for group in self._releases.values()):
            raise ManagerError("session request belongs to a pending release group")

        owner = self._request_occupancy.get(key)
        participants = tuple(
            (control_id, group)
            for control_id, group in self._controls.items()
            if key in group.request_keys or key in group.target_keys
        )
        control_participants = tuple(
            (participant, control_id)
            for control_id, group in self._controls.items()
            for participant in group.request_keys
        )
        if (
            any(
                group.control_id != control_id
                or len(set(group.request_keys)) != len(group.request_keys)
                or len(set(group.target_keys)) != len(group.target_keys)
                or any(
                    target not in group.request_keys
                    for target in group.target_keys
                )
                for control_id, group in self._controls.items()
            )
            or len({participant for participant, _ in control_participants})
            != len(control_participants)
        ):
            raise ManagerError(
                "session control occupancy is inconsistent for row binding"
            )
        expected_occupancy = {
            participant: control_id for participant, control_id in control_participants
        }
        if self._request_occupancy != expected_occupancy:
            raise ManagerError(
                "session control occupancy is inconsistent for row binding"
            )
        if owner is None:
            return
        if len(participants) != 1:
            raise ManagerError(
                "request participates in an unsafe pending control group"
            )

        control_id, group = participants[0]
        plan_info = group.plan_info
        plan = group.plan
        if (
            control_id != owner
            or group.control_id != owner
            or self._controls.get(owner) is not group
            or group.request_keys.count(key) != 1
            or group.target_keys.count(key) != 1
            or type(plan_info) is not EngineControlPlanInfo
            or plan_info.control_id != owner
            or plan_info.kind is not EngineControlKind.MATERIALIZATION
            or type(plan) is not EngineMaterializationPlan
            or plan.control_id != owner
            or plan_info.request_count != len(group.target_keys)
            or len(plan.requests) != len(group.target_keys)
            or any(
                type(item) is not EngineMaterializedRequest
                or item.request_id != self._binding(target).request_id
                for target, item in zip(
                    group.target_keys, plan.requests, strict=True
                )
            )
        ):
            raise ManagerError(
                "request participates in an unsafe pending control group"
            )

    def _prefix_owner(self, prefix_id: EnginePrefixId) -> EngineControlId | None:
        return self._prefix_occupancy.get(prefix_id)

    def _materialization_updates_for_plan(
        self,
        group: _ControlGroup,
        plan: EngineMaterializationPlan,
    ) -> tuple[SessionMaterializationUpdate, ...]:
        updates: list[SessionMaterializationUpdate] = []
        seen_rows: set[int] = set()
        if len(plan.requests) != len(group.target_keys):
            raise ManagerError("native materialization read changed target cardinality")
        for key, item in zip(group.target_keys, plan.requests, strict=True):
            if type(item) is not EngineMaterializedRequest:
                raise ManagerError(
                    "native materialization read returned an invalid request"
                )
            binding = self._binding(key)
            if item.request_id != binding.request_id:
                raise ManagerError(
                    "native materialization read changed request identity"
                )
            if binding.request_row is None:
                raise ManagerError(
                    "materialization mirror requires a bound target ReqToToken row"
                )
            if binding.request_row in seen_rows:
                raise ManagerError("native materialization duplicated a target row")
            seen_rows.add(binding.request_row)
            updates.append(
                SessionMaterializationUpdate(
                    key,
                    binding.request_id,
                    binding.request_row,
                    int(item.view_version),
                    int(item.boundary),
                    int(item.resident_count),
                    tuple(item.pages),
                )
            )
        return tuple(updates)

    def _run_materialization(
        self,
        group: _ControlGroup,
        updates: tuple[SessionMaterializationUpdate, ...],
    ) -> None:
        if self._materialization is None:
            self._best_effort_quarantine_control(group.control_id)
            self._fail_stop("materialization has no bound authority")
        try:
            confirmed = self._materialization(updates)
        except ManagerError:
            raise
        except BaseException as error:
            self._best_effort_quarantine_control(group.control_id)
            self._fail_stop("materialization mirror became uncertain", error)
        if confirmed is not True:
            self._best_effort_quarantine_control(group.control_id)
            self._fail_stop("materialization mirror was not explicitly confirmed")

    def _install_materialized_views(
        self, updates: Sequence[SessionMaterializationUpdate]
    ) -> None:
        next_views = dict(self._views)
        for item in updates:
            next_views[item.key] = EngineRequestView(
                item.request_id,
                item.view_version,
                item.boundary,
                item.resident_count,
            )
        self._views = next_views


__all__ = [
    "MaterializationCallback",
    "SessionControlRuntimeMixin",
    "SessionMaterializationUpdate",
    "unbound_materialization",
]
