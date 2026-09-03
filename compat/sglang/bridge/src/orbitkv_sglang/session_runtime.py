from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock, get_ident
from typing import Any, Callable, Hashable, Iterator, NoReturn, Sequence

from .ffi.session import CtypesRuntimeSession
from .ffi.session_types import (
    EngineAppendIntent,
    EngineBatchId,
    EngineBatchPlan,
    EngineBatchPublication,
    EngineBatchTicket,
    EngineCompletionEvidence,
    EngineControlId, EngineControlOutcome,
    EnginePrefixId, EnginePrefixEvictionPlan,
    EnginePublishedPrefix,
    EnginePublicationEvidence,
    EngineReleaseDisposition,
    EngineReleaseEvidence,
    EngineReleaseId,
    EngineReleaseOutcome,
    EngineReleasePlan,
    EngineRequestId,
    EngineRequestView,
    EngineRetirement,
    EngineStepAbortEvidence,
    ExecutionEvidence,
    retirement_evidence,
)
from .runtime import (
    CacheSharingPolicy,
    DetachedBinding,
    FailStopped,
    ManagerError,
    PrefixSemanticKey,
)
from .session_activity import SessionActivityTracker, SessionSwaActivity
from .session_control_runtime import (
    MaterializationCallback,
    SessionControlRuntimeMixin,
    SessionMaterializationUpdate,
    _ControlGroup,
    unbound_materialization,
)
from .session_capacity_runtime import CapacityInvalidatingSession
from .session_release_runtime import (
    SessionReleaseRuntimeMixin,
    _ReleaseGroup,
)


_UINT64_LIMIT = 1 << 64


@dataclass(frozen=True, slots=True)
class SessionRequestBinding:
    """Stable engine identity and ReqToToken row for one live request."""

    key: Hashable
    request_id: EngineRequestId
    request_row: int | None


@dataclass(frozen=True, slots=True)
class SessionMirrorUpdate:
    """One native publication translated to its engine mirror identity."""

    key: Hashable
    request_id: EngineRequestId
    request_row: int
    boundary: int
    detached: tuple[DetachedBinding, ...]
    releasing: bool


MirrorCleanupCallback = Callable[
    [tuple[SessionMirrorUpdate, ...], tuple[EngineRetirement, ...]], bool
]
MirrorContextFactory = Callable[[SessionMirrorUpdate], Any]


def unbound_mirror_cleanup(
    _updates: tuple[SessionMirrorUpdate, ...],
    _retirements: tuple[EngineRetirement, ...],
) -> bool:
    """Fail-closed bootstrap marker accepted only for two-phase wiring."""

    raise FailStopped("session mirror cleanup authority is not bound")


def collective_mirror_cleanup(
    coordinator: Any, context_for: MirrorContextFactory
) -> MirrorCleanupCallback:
    """Adapt the existing collective mirror protocol to session updates.

    ``context_for`` supplies the engine-specific request context for an update.
    Session retirements intentionally carry no reclamation lease, but the
    mirror coordinator consumes only their physical identity and token span;
    exact ACK authority remains private to the native session.
    """

    if not callable(context_for):
        raise TypeError("mirror cleanup context factory must be callable")
    for name in ("preflight", "commit", "synchronize", "finalize"):
        if not callable(getattr(coordinator, name, None)):
            raise TypeError(
                "mirror cleanup coordinator lacks the collective protocol"
            )

    def cleanup(
        updates: tuple[SessionMirrorUpdate, ...],
        retirements: tuple[EngineRetirement, ...],
    ) -> bool:
        from .runtime import MirrorCleanupItem

        items = tuple(
            MirrorCleanupItem(
                context_for(update),
                update.detached,
                update.releasing,
                update.boundary,
                (),
            )
            for update in updates
            if update.detached or update.releasing
        )
        if not items and not retirements:
            return True
        plan = coordinator.preflight(items, retirements)
        coordinator.commit(plan)
        coordinator.synchronize(plan)
        coordinator.finalize(plan)
        return True

    return cleanup


@dataclass(slots=True)
class _EventGroup:
    ticket: EngineBatchTicket
    event: Any
    completion_domain: int


class ReleaseRetryPending(ManagerError):
    """The exact release plan must be retained and retried."""

    def __init__(self, plan: EngineReleasePlan, message: str) -> None:
        self.plan = plan
        super().__init__(message)

    @property
    def release_id(self) -> EngineReleaseId:
        return self.plan.release_id


class ReleaseRecyclePending(ReleaseRetryPending):
    """The release ACK committed, but native request recycling is pending."""

    def __init__(self, plan: EngineReleasePlan) -> None:
        super().__init__(
            plan,
            "release ACK was accepted but request recycling remains pending: "
            f"{plan.release_id}"
        )


class ReleaseAckRetryPending(ReleaseRetryPending):
    """Mirror cleanup completed, but the release ACK did not commit."""

    def __init__(self, plan: EngineReleasePlan) -> None:
        super().__init__(
            plan,
            "release cleanup completed but the native ACK must be retried: "
            f"{plan.release_id}",
        )


class SessionRuntime(SessionReleaseRuntimeMixin, SessionControlRuntimeMixin):
    """Thin engine coordinator around the native runtime session.

    The native session remains the only lifecycle and page-ownership
    authority.  This layer owns only identities that exist in the embedding
    engine: stable request keys, ReqToToken rows, submitted batch tickets,
    completion events, and completion high-water marks.

    ``mirror_cleanup`` is one collective, synchronous transaction.  Before it
    returns ``True`` it must validate and apply every supplied mirror update,
    synchronize any device work, and finish its host-side bookkeeping.  A
    false/non-boolean result or an exception is fail-stop because native
    publication or release has already begun.
    """

    def __init__(
        self,
        session: CtypesRuntimeSession,
        *,
        mirror_cleanup: MirrorCleanupCallback | None = unbound_mirror_cleanup,
        materialization: MaterializationCallback | None = unbound_materialization,
    ) -> None:
        if mirror_cleanup is not None and not callable(mirror_cleanup):
            raise TypeError("mirror_cleanup must be callable")
        if materialization is not None and not callable(materialization):
            raise TypeError("materialization must be callable")
        try:
            cache_sharing_policy = session.cache_sharing_policy
        except AttributeError as error:
            raise ManagerError(
                "runtime session does not expose its cache sharing policy"
            ) from error
        if type(cache_sharing_policy) is not CacheSharingPolicy:
            raise ManagerError(
                "runtime session cache sharing policy must be CacheSharingPolicy"
            )
        self._session = session
        self._mirror_cleanup = (
            None
            if mirror_cleanup is unbound_mirror_cleanup
            else mirror_cleanup
        )
        self._materialization = (
            None
            if materialization is unbound_materialization
            else materialization
        )
        self._arenas = tuple(session.arenas)
        try:
            prefix_capacity = session.prefix_capacity
            control_batch_capacity = session.control_batch_capacity
            prefix_eviction_batch_capacity = (
                session.prefix_eviction_batch_capacity
            )
        except AttributeError as error:
            raise ManagerError(
                "runtime session does not expose its control batch capacity"
            ) from error
        if type(prefix_capacity) is not int or prefix_capacity <= 0:
            raise ManagerError(
                "runtime session Prefix capacity must be a positive integer"
            )
        if (
            type(control_batch_capacity) is not int
            or control_batch_capacity <= 0
        ):
            raise ManagerError(
                "runtime session control batch capacity must be a positive integer"
            )
        if (
            type(prefix_eviction_batch_capacity) is not int
            or prefix_eviction_batch_capacity <= 0
            or prefix_eviction_batch_capacity > prefix_capacity
            or prefix_eviction_batch_capacity > control_batch_capacity
        ):
            raise ManagerError(
                "runtime session Prefix eviction batch capacity is invalid"
            )
        self._prefix_capacity = prefix_capacity
        self._cache_sharing_policy = cache_sharing_policy
        self._control_batch_capacity = control_batch_capacity
        self._prefix_eviction_batch_capacity = (
            prefix_eviction_batch_capacity
        )
        if not self._arenas:
            raise ManagerError("runtime session has no arenas")
        epochs = {item.engine_epoch for item in self._arenas}
        page_tokens = {item.page_tokens for item in self._arenas}
        class_ids = {item.class_id for item in self._arenas}
        if (
            len(epochs) != 1
            or len(page_tokens) != 1
            or len(class_ids) != len(self._arenas)
        ):
            raise ManagerError("runtime session arena identities are inconsistent")
        self._engine_epoch = epochs.pop()
        self._page_tokens = page_tokens.pop()
        self._arenas_by_class = {item.class_id: item for item in self._arenas}
        self._bindings: dict[Hashable, SessionRequestBinding] = {}
        self._views: dict[Hashable, EngineRequestView] = {}
        self._request_keys: dict[EngineRequestId, Hashable] = {}
        self._row_owners: dict[int, Hashable] = {}
        self._tickets: dict[EngineBatchId, EngineBatchTicket] = {}
        self._events: list[_EventGroup] = []
        self._releases: dict[EngineReleaseId, _ReleaseGroup] = {}
        self._prefixes: dict[PrefixSemanticKey, EnginePublishedPrefix] = {}
        self._request_occupancy: dict[Hashable, EngineControlId] = {}
        self._prefix_occupancy: dict[EnginePrefixId, EngineControlId] = {}
        self._quarantined_requests: set[Hashable] = set()
        self._controls: dict[EngineControlId, _ControlGroup] = {}
        self._pending_attach_cancels = {}
        self._next_request_id = 1
        self._next_completion_value = 1
        self._completion_high_water: dict[int, int] = {}
        self._counters = {
            "forward_events": 0,
            "completion_values": 0,
            "event_queries": 0,
            "event_waits": 0,
            "fail_stop_count": 0,
        }
        self._activity = SessionActivityTracker(self._arenas, self._page_tokens)
        self._failure: str | None = None
        self._closed = False
        self._lock = RLock()
        self._capacity_revision = 0
        self._arena_stats_cache: tuple[int, tuple[Any, ...]] | None = None
        self._scheduler_turn_owner: int | None = None
        self._scheduler_turn_polled = False
        self._session = CapacityInvalidatingSession(
            session, self._invalidate_capacity_cache
        )

    @property
    def failure_reason(self) -> str | None:
        return self._failure

    @property
    def arenas(self) -> tuple[Any, ...]:
        return self._arenas

    @property
    def arenas_by_class(self) -> dict[int, Any]:
        return dict(self._arenas_by_class)

    @property
    def engine_epoch(self) -> int:
        return self._engine_epoch

    @property
    def page_tokens(self) -> int:
        return self._page_tokens

    @property
    def cache_sharing_policy(self) -> CacheSharingPolicy:
        return self._cache_sharing_policy

    def _require_shared_cache_policy(self, operation: str) -> None:
        if self._cache_sharing_policy is not CacheSharingPolicy.SHARED_PREFIX:
            raise ManagerError(
                f"{operation} is unavailable under request-private cache "
                "sharing policy"
            )

    @property
    def mirror_cleanup_bound(self) -> bool:
        return self._mirror_cleanup is not None

    def performance_counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def swa_activity(self, classes: Sequence[Any]) -> SessionSwaActivity:
        """Return exact cumulative Sliding activity from native events.

        Retirement certificates are counted from successful append
        publications, reclaimed pages only after their exact publication ACK,
        wraps from confirmed request-boundary transitions, and reuse from a
        later native reservation of an ACKed physical page at a newer
        generation.  The first generation observed for a page is a baseline,
        never a reuse event.
        """

        with self._lock:
            self._healthy()
            return self._activity.activity(classes)

    def completion_evidence(self, *, external: bool = False) -> dict[str, object]:
        del external
        with self._lock:
            return {
                "event_backend": "cuda_event_current_forward_stream",
                "pending_events": len(self._events),
                "completion_high_water": [
                    {"domain": domain, "value": value}
                    for domain, value in sorted(
                        self._completion_high_water.items()
                    )
                ],
            }

    def binding_for(self, key: Hashable) -> SessionRequestBinding:
        with self._lock:
            self._healthy()
            return self._binding(key)

    def view_for(self, key: Hashable) -> EngineRequestView:
        """Return the last native view whose publication was confirmed."""

        with self._lock:
            self._healthy()
            self._binding(key)
            try:
                return self._views[key]
            except KeyError:
                self._fail_stop(
                    "session request view mirror changed unexpectedly"
                )

    def views_for(
        self, keys: Sequence[Hashable]
    ) -> tuple[EngineRequestView, ...]:
        """Return confirmed views in the exact requested key order."""

        with self._lock:
            self._healthy()
            bindings = self._bindings_for_keys(keys, "session view")
            return tuple(self.view_for(item.key) for item in bindings)

    def bind_mirror_cleanup(self, cleanup: MirrorCleanupCallback) -> None:
        """Install the sole mirror cleanup authority before first use."""

        with self._lock:
            self._healthy()
            if not callable(cleanup):
                raise TypeError("cleanup must be callable")
            if self._mirror_cleanup is not None:
                raise ManagerError("mirror cleanup authority is already bound")
            self._mirror_cleanup = cleanup

    def stats(self) -> Any:
        with self._lock:
            self._healthy()
            try:
                return self._session.stats()
            except BaseException as error:
                self._fail_stop("native session stats became uncertain", error)

    def arena_stats(self) -> tuple[Any, ...]:
        with self._lock:
            self._healthy()
            in_turn = self._scheduler_turn_owner == get_ident()
            cached = self._arena_stats_cache
            if (
                in_turn
                and cached is not None
                and cached[0] == self._capacity_revision
            ):
                return cached[1]
            try:
                values = tuple(self._session.arena_stats())
            except BaseException as error:
                self._fail_stop("native arena stats became uncertain", error)
            if in_turn:
                self._arena_stats_cache = (self._capacity_revision, values)
            return values

    def capacity_arena_stats(self) -> tuple[Any, ...]:
        """Poll as needed and return the authoritative capacity census."""

        with self._lock:
            self._healthy()
            if self._scheduler_turn_owner != get_ident():
                self.poll()
            return self.arena_stats()

    @contextmanager
    def scheduler_turn(self) -> Iterator[None]:
        """Poll once, then keep capacity reads coherent for one scheduler turn.

        The cache is visible only to the scheduler thread that owns this
        explicit scope.  A native mutation advances the capacity revision, so
        the next availability read in the same turn obtains a fresh native
        arena census without querying asynchronous events a second time.
        """

        owner = get_ident()
        with self._lock:
            self._healthy()
            if self._scheduler_turn_owner is not None:
                raise ManagerError("session scheduler turn is already active")
            self._scheduler_turn_owner = owner
            self._scheduler_turn_polled = False
            self._arena_stats_cache = None
        try:
            self.poll()
            yield
        finally:
            with self._lock:
                if self._scheduler_turn_owner == owner:
                    self._scheduler_turn_owner = None
                    self._scheduler_turn_polled = False
                    self._arena_stats_cache = None

    @property
    def prefix_capacity(self) -> int:
        """Return the configured native Prefix identity capacity."""

        return self._prefix_capacity

    @property
    def control_batch_capacity(self) -> int:
        """Return the configured native control-item batch capacity."""

        return self._control_batch_capacity

    @property
    def prefix_eviction_batch_capacity(self) -> int:
        """Return the exact native prefix-eviction batch capacity."""

        return self._prefix_eviction_batch_capacity

    def census(self) -> tuple[Any, tuple[Any, ...]]:
        with self._lock:
            self._healthy()
            return self.stats(), self.arena_stats()

    def fail_stop(self, reason: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = str(reason)
                self._counters["fail_stop_count"] += 1
            if not self._closed:
                self._closed = True
                try:
                    self._session.close()
                except BaseException:
                    pass
                finally:
                    self._clear_local_state()

    def acquire_unbound(
        self, keys: Sequence[Hashable]
    ) -> tuple[EngineRequestView, ...]:
        """Acquire native request identities without claiming engine rows."""

        with self._lock:
            self._healthy()
            values = tuple(keys)
            if not values:
                raise ManagerError("session acquire batch must be nonempty")
            for key in values:
                self._require_hashable(key)
            if len(set(values)) != len(values):
                raise ManagerError("session acquire contains duplicate request keys")
            if any(key in self._bindings for key in values):
                raise ManagerError("session request key is already acquired")
            request_ids = self._allocate_request_ids(len(values))
            try:
                acquired_views = self._session.acquire_requests(request_ids)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop(
                    "native request acquisition became uncertain", error
                )
            try:
                views = tuple(acquired_views)
                if any(type(view) is not EngineRequestView for view in views):
                    raise RuntimeError(
                        "native request acquisition returned an invalid view"
                    )
                if (
                    len(views) != len(values)
                    or tuple(item.request_id for item in views) != request_ids
                    or any(
                        int(view.boundary) != 0
                        or int(view.resident_count) != 0
                        for view in views
                    )
                ):
                    raise RuntimeError(
                        "native request acquisition changed initial state or ordering"
                    )
                bindings = tuple(
                    SessionRequestBinding(key, request_id, None)
                    for key, request_id in zip(
                        values, request_ids, strict=True
                    )
                )
                installed_bindings = dict(self._bindings)
                installed_views = dict(self._views)
                installed_request_keys = dict(self._request_keys)
                for binding, view in zip(bindings, views, strict=True):
                    if (
                        binding.key in installed_bindings
                        or binding.request_id in installed_request_keys
                    ):
                        raise RuntimeError(
                            "native request acquisition duplicated local identity"
                        )
                    installed_bindings[binding.key] = binding
                    installed_views[binding.key] = view
                    installed_request_keys[binding.request_id] = binding.key
            except BaseException as error:
                self._fail_stop(
                    "native request acquisition could not be installed locally",
                    error,
                )
            try:
                self._bindings = installed_bindings
                self._views = installed_views
                self._request_keys = installed_request_keys
            except BaseException as error:
                self._fail_stop(
                    "native request acquisition could not be installed locally",
                    error,
                )
            return views

    def bind_request_rows(
        self, assignments: Sequence[tuple[Hashable, int]]
    ) -> None:
        """Collectively bind positive, unique rows exactly once."""

        with self._lock:
            self._healthy()
            if self._mirror_cleanup is None:
                raise ManagerError(
                    "session mirror cleanup authority must be bound before row binding"
                )
            values = tuple(assignments)
            if not values:
                raise ManagerError("request-row bind batch must be nonempty")
            keys: list[Hashable] = []
            rows: list[int] = []
            bindings: list[SessionRequestBinding] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ManagerError(
                        "request-row bind items must be (key, request_row) pairs"
                    )
                key, row = item
                self._require_hashable(key)
                if isinstance(row, bool) or not isinstance(row, int) or row <= 0:
                    raise ManagerError(
                        "ReqToToken request row must be a positive integer"
                    )
                binding = self._binding(key)
                if binding.request_row is not None:
                    raise ManagerError("session request row is already bound")
                self._require_row_binding_allowed(key)
                keys.append(key)
                rows.append(row)
                bindings.append(binding)
            if len(set(keys)) != len(keys):
                raise ManagerError("request-row bind contains duplicate request keys")
            if len(set(rows)) != len(rows):
                raise ManagerError("request-row bind aliases a ReqToToken row")
            if any(row in self._row_owners for row in rows):
                raise ManagerError("ReqToToken row is already owned")
            if any(
                key in group.keys
                for group in self._releases.values()
                for key in keys
            ):
                raise ManagerError(
                    "session request belongs to a pending release group"
                )
            try:
                replacements = tuple(
                    SessionRequestBinding(binding.key, binding.request_id, row)
                    for binding, row in zip(bindings, rows, strict=True)
                )
                installed_bindings = dict(self._bindings)
                installed_row_owners = dict(self._row_owners)
                for binding in replacements:
                    installed_bindings[binding.key] = binding
                    assert binding.request_row is not None
                    installed_row_owners[binding.request_row] = binding.key
            except BaseException as error:
                self._fail_stop(
                    "request-row binding local commit became uncertain", error
                )
            try:
                self._bindings = installed_bindings
                self._row_owners = installed_row_owners
            except BaseException as error:
                self._fail_stop(
                    "request-row binding local commit became uncertain", error
                )

    def prepare(
        self, appends: Sequence[tuple[Hashable, int]]
    ) -> EngineBatchPlan:
        """Prepare an ordered append batch without duplicating its phase."""

        with self._lock:
            self._healthy()
            if self._mirror_cleanup is None:
                raise ManagerError(
                    "session mirror cleanup authority must be bound before prepare"
                )
            values = tuple(appends)
            if not values:
                raise ManagerError("session append batch must be nonempty")
            keys: list[Hashable] = []
            intents: list[EngineAppendIntent] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ManagerError(
                        "session append items must be (key, target_boundary) pairs"
                    )
                key, boundary = item
                binding = self._binding(key)
                if binding.request_row is None:
                    raise ManagerError(
                        "session append requires a bound ReqToToken row"
                    )
                self._require_control_free(key)
                if (
                    isinstance(boundary, bool)
                    or not isinstance(boundary, int)
                    or not 0 <= boundary < _UINT64_LIMIT
                ):
                    raise ManagerError(
                        "append target boundary must fit in uint64_t"
                    )
                keys.append(key)
                intents.append(
                    EngineAppendIntent(binding.request_id, boundary)
                )
            if len(set(keys)) != len(keys):
                raise ManagerError("session append contains duplicate request keys")
            try:
                plan = self._session.prepare_append(tuple(intents))
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native append preparation became uncertain", error)
            if (
                not isinstance(plan, EngineBatchPlan)
                or tuple(step.request_id for step in plan.steps)
                != tuple(intent.request_id for intent in intents)
            ):
                self._fail_stop(
                    "native append preparation changed identity or ordering"
                )
            try:
                self._activity.observe_plan(plan)
            except BaseException as error:
                self._fail_stop(
                    "native append preparation returned invalid page observations",
                    error,
                )
            return plan

    def submit(self, evidence: ExecutionEvidence) -> EngineBatchTicket:
        """Submit exact engine evidence and retain only its batch ticket."""

        with self._lock:
            self._healthy()
            try:
                ticket = self._session.submit_execution(evidence)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native execution submission became uncertain", error)
            if not isinstance(ticket, EngineBatchTicket) or not ticket.requests:
                self._fail_stop("native execution returned an invalid batch ticket")
            if ticket.batch_id in self._tickets:
                self._fail_stop("native execution returned a duplicate batch ticket")
            if len(set(ticket.requests)) != len(ticket.requests):
                self._fail_stop("native execution ticket aliases a request")
            live_ticket_requests = {
                request_id
                for value in self._tickets.values()
                for request_id in value.requests
            }
            if any(
                request_id not in self._request_keys
                or request_id in live_ticket_requests
                for request_id in ticket.requests
            ):
                self._fail_stop(
                    "native execution ticket names an unknown or busy request"
                )
            if any(
                self._binding_for_request_id(request_id).request_row is None
                for request_id in ticket.requests
            ):
                self._fail_stop(
                    "native execution ticket names an unbound request"
                )
            self._tickets[ticket.batch_id] = ticket
            return ticket

    def abort_prepared(self, plan: EngineBatchPlan) -> None:
        """Abort a plan only when the backend provably observed nothing."""

        with self._lock:
            self._healthy()
            if not isinstance(plan, EngineBatchPlan) or not plan.steps:
                raise ManagerError("prepared abort requires a nonempty batch plan")
            evidence = tuple(
                EngineStepAbortEvidence(item.request_id, True)
                for item in plan.steps
            )
            try:
                self._session.abort_prepared(plan.batch_id, evidence)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native prepared abort became uncertain", error)

    def quarantine_prepared(self, batch_id: EngineBatchId) -> None:
        """Quarantine a prepared batch after ambiguous backend observation."""

        with self._lock:
            self._healthy()
            try:
                self._session.quarantine_prepared(batch_id)
            except BaseException as error:
                self._fail_stop("prepared quarantine became uncertain", error)
            self._fail_stop("prepared execution was quarantined")

    def quarantine_submitted(self, ticket: EngineBatchTicket) -> None:
        """Quarantine submitted work and make the coordinator unusable."""

        with self._lock:
            self._healthy()
            if not isinstance(ticket, EngineBatchTicket):
                raise ManagerError("submitted quarantine requires a batch ticket")
            if self._tickets.get(ticket.batch_id) is not ticket:
                raise ManagerError(
                    "submitted quarantine requires the issued batch ticket"
                )
            try:
                self._session.quarantine_submitted(ticket.batch_id)
            except BaseException as error:
                self._fail_stop("submitted quarantine became uncertain", error)
            self._fail_stop("submitted execution was quarantined")

    def register_event(
        self,
        ticket: EngineBatchTicket,
        event: Any,
        completion_domain: int,
    ) -> None:
        """Associate a submitted ticket with its post-forward CUDA event."""

        with self._lock:
            self._healthy()
            if not isinstance(ticket, EngineBatchTicket):
                raise ManagerError("event registration requires a batch ticket")
            stored = self._tickets.get(ticket.batch_id)
            if stored is not ticket:
                raise ManagerError(
                    "event registration requires the issued batch ticket"
                )
            if any(group.ticket.batch_id == ticket.batch_id for group in self._events):
                raise ManagerError("batch ticket already has a completion event")
            if (
                isinstance(completion_domain, bool)
                or not isinstance(completion_domain, int)
                or not 0 < completion_domain < _UINT64_LIMIT
            ):
                raise ManagerError("completion domain must be a positive uint64_t")
            if not callable(getattr(event, "query", None)) or not callable(
                getattr(event, "synchronize", None)
            ):
                self._fail_stop(
                    "submitted execution has no usable completion event"
                )
            self._events.append(_EventGroup(ticket, event, completion_domain))
            self._counters["forward_events"] += 1

    def poll(self) -> tuple[EngineBatchPublication, ...]:
        """Query pending events and fully confirm every ready publication."""

        with self._lock:
            self._healthy()
            in_turn = self._scheduler_turn_owner == get_ident()
            if in_turn and self._scheduler_turn_polled:
                return ()
            if in_turn:
                # Mark the attempt before callbacks can re-enter availability.
                self._scheduler_turn_polled = True
            ready: list[_EventGroup] = []
            for group in tuple(self._events):
                try:
                    self._counters["event_queries"] += 1
                    complete = bool(group.event.query())
                except BaseException as error:
                    self._fail_stop(
                        "completion event query became uncertain", error
                    )
                if complete:
                    ready.append(group)
            return tuple(self._complete_group(group) for group in ready)

    def wait_requests(self, keys: Sequence[Hashable]) -> None:
        """Synchronize and publish every event batch touching ``keys``."""

        with self._lock:
            self._healthy()
            bindings = self._bindings_for_keys(keys, "session wait")
            request_ids = {item.request_id for item in bindings}
            groups = [
                group
                for group in self._events
                if request_ids.intersection(group.ticket.requests)
            ]
            registered = {
                request_id
                for group in groups
                for request_id in group.ticket.requests
            }
            if any(
                request_id in request_ids and request_id not in registered
                for ticket in self._tickets.values()
                for request_id in ticket.requests
            ):
                raise ManagerError(
                    "submitted request has no registered completion event"
                )
            for group in groups:
                try:
                    self._counters["event_waits"] += 1
                    group.event.synchronize()
                except BaseException as error:
                    self._fail_stop(
                        "completion event synchronization became uncertain",
                        error,
                    )
                self._complete_group(group)

    def prepare_release(
        self, keys: Sequence[Hashable]
    ) -> EngineReleasePlan:
        """Wait for execution, then begin one native release transaction."""

        with self._lock:
            self._healthy()
            bindings = self._bindings_for_keys(keys, "session release")
            ordered_keys = tuple(item.key for item in bindings)
            existing = tuple(
                group
                for group in self._releases.values()
                if group.keys == ordered_keys
            )
            if len(existing) > 1:
                self._fail_stop(
                    "multiple pending releases alias the same request group"
                )
            if existing:
                return existing[0].plan
            if any(
                key in group.keys
                for group in self._releases.values()
                for key in ordered_keys
            ):
                raise ManagerError(
                    "session request belongs to another pending release group"
                )
            for key in ordered_keys:
                self._require_control_free(key)
            self.wait_requests(ordered_keys)
            request_ids = tuple(
                self._binding(key).request_id for key in ordered_keys
            )
            try:
                plan = self._session.prepare_release(request_ids)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop("native request release became uncertain", error)
            if (
                type(plan) is not EngineReleasePlan
                or type(plan.release_id) is not EngineReleaseId
                or plan.release_id.session_epoch != self._engine_epoch
                or type(plan.release_id.sequence) is not int
                or plan.release_id.sequence <= 0
                or type(plan.releases) is not tuple
                or type(plan.retirements) is not tuple
                or tuple(item.request_id for item in plan.releases) != request_ids
            ):
                self._fail_stop(
                    "native request release changed identity or ordering"
                )
            if plan.release_id in self._releases:
                self._fail_stop("native request release returned a duplicate id")
            self._releases[plan.release_id] = _ReleaseGroup(
                plan, ordered_keys
            )
            return plan

    def pending_release(
        self, keys: Sequence[Hashable]
    ) -> EngineReleasePlan | None:
        """Return an atomic publish-release or cleanup-complete retry."""

        with self._lock:
            self._healthy()
            values = tuple(keys)
            if not values:
                raise ManagerError("pending release batch must be nonempty")
            for key in values:
                self._require_hashable(key)
            if len(set(values)) != len(values):
                raise ManagerError(
                    "pending release contains duplicate request keys"
                )
            matches = tuple(
                group
                for group in self._releases.values()
                if group.keys == values
                and (
                    group.publish_release is not None
                    or group.cleanup_confirmed
                )
            )
            if len(matches) > 1:
                self._fail_stop(
                    "multiple pending releases alias one request group"
                )
            if not matches:
                self._bindings_for_keys(values, "pending release")
            return matches[0].plan if matches else None

    def confirm_release(self, plan: EngineReleasePlan) -> None:
        """Clean released mirrors, confirm ACKs, then drop engine identities."""

        with self._lock:
            self._healthy()
            if type(plan) is not EngineReleasePlan:
                raise ManagerError("release confirmation requires a release plan")
            group = self._releases.get(plan.release_id)
            if group is None or group.plan is not plan:
                raise ManagerError(
                    "release confirmation requires the issued release plan"
                )
            if not group.cleanup_confirmed:
                updates = self._release_updates(group)
                self._run_cleanup(updates, plan.retirements, "release")
                group.cleanup_confirmed = True
            try:
                outcome = self._confirm_release_native(
                    plan, acknowledged_retry=group.ack_committed
                )
            except ManagerError as error:
                if group.ack_committed:
                    raise ReleaseRecyclePending(plan) from error
                raise ReleaseAckRetryPending(plan) from error
            if (
                not group.ack_committed
                and outcome.disposition
                is EngineReleaseDisposition.RECYCLE_PENDING
            ):
                # Only a typed pending result proves that ACK committed. Keep
                # cleanup completion separate so a pre-ACK ManagerError can
                # retry the full evidence without replaying mirror cleanup.
                group.ack_committed = True
                self._record_release_ack_activity(plan)
                try:
                    outcome = self._confirm_release_native(
                        plan, acknowledged_retry=True
                    )
                except ManagerError as error:
                    raise ReleaseRecyclePending(plan) from error
            if outcome.disposition is EngineReleaseDisposition.RECYCLE_PENDING:
                raise ReleaseRecyclePending(plan)
            if not group.ack_committed:
                self._record_release_ack_activity(plan)
            try:
                bindings = tuple(self._binding(key) for key in group.keys)
                retained_bindings = dict(self._bindings)
                retained_views = dict(self._views)
                retained_request_keys = dict(self._request_keys)
                retained_row_owners = dict(self._row_owners)
                retained_releases = dict(self._releases)
                for binding in bindings:
                    del retained_bindings[binding.key]
                    del retained_views[binding.key]
                    del retained_request_keys[binding.request_id]
                    if binding.request_row is not None:
                        del retained_row_owners[binding.request_row]
                    self._activity.forget_requests((binding.request_id,))
                del retained_releases[plan.release_id]
                self._bindings = retained_bindings
                self._views = retained_views
                self._request_keys = retained_request_keys
                self._row_owners = retained_row_owners
                self._releases = retained_releases
            except BaseException as error:
                self._fail_stop(
                    "confirmed request release local cleanup became uncertain",
                    error,
                )

    def confirm_control(
        self, control_id: EngineControlId
    ) -> EngineControlOutcome:
        """Confirm native control and retain ACKed pages for reuse proof."""

        with self._lock:
            self._healthy()
            group = self._controls.get(control_id)
            plan = None if group is None else group.plan
            outcome = super().confirm_control(control_id)
            if type(plan) is EnginePrefixEvictionPlan:
                try:
                    self._activity.acknowledge_reuse_candidates(
                        plan.retirements
                    )
                except BaseException as error:
                    self._fail_stop(
                        "native eviction ACK activity became inconsistent",
                        error,
                    )
            return outcome

    def _record_release_ack_activity(self, plan: EngineReleasePlan) -> None:
        try:
            self._activity.acknowledge_reuse_candidates(plan.retirements)
        except BaseException as error:
            self._fail_stop("release ACK activity became inconsistent", error)

    def _confirm_release_native(
        self, plan: EngineReleasePlan, *, acknowledged_retry: bool
    ) -> EngineReleaseOutcome:
        evidence = EngineReleaseEvidence(
            plan.release_id,
            not acknowledged_retry,
            ()
            if acknowledged_retry
            else retirement_evidence(plan.retirements),
            acknowledged_retry=acknowledged_retry,
        )
        try:
            outcome = self._session.confirm_release(evidence)
        except ManagerError:
            raise
        except BaseException as error:
            self._fail_stop(
                "native release confirmation became uncertain", error
            )
        if (
            not isinstance(outcome, EngineReleaseOutcome)
            or outcome.release_id != plan.release_id
            or type(outcome.disposition) is not EngineReleaseDisposition
        ):
            self._fail_stop(
                "native release confirmation returned a malformed outcome"
            )
        return outcome

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._failure is None and (
                self._bindings
                or self._tickets
                or self._events
                or self._releases
                or self._controls
            ):
                raise ManagerError(
                    "cannot close a session runtime with live requests"
                )
            self._closed = True
            try:
                self._session.close()
            except BaseException as error:
                if self._failure is None:
                    self._failure = (
                        "native session close outcome is unknown: "
                        f"{type(error).__name__}: {error}"
                    )
                self._clear_local_state()
                raise FailStopped(self._failure) from error
            self._clear_local_state()

    def __enter__(self) -> SessionRuntime:
        with self._lock:
            self._healthy()
            return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _complete_group(
        self, group: _EventGroup
    ) -> EngineBatchPublication:
        if group not in self._events:
            self._fail_stop("completion event disappeared before publication")
        ticket = self._tickets.get(group.ticket.batch_id)
        if ticket is not group.ticket:
            self._fail_stop("completion event lost its issued batch ticket")
        completion_value = self._allocate_completion_value(
            group.completion_domain
        )
        try:
            publication = self._session.complete_execution(
                group.ticket.batch_id,
                EngineCompletionEvidence(
                    group.completion_domain, completion_value, True
                ),
            )
        except BaseException as error:
            self._fail_stop(
                "native execution completion became uncertain", error
            )
        if (
            not isinstance(publication, EngineBatchPublication)
            or publication.batch_id != group.ticket.batch_id
            or tuple(item.request_id for item in publication.steps)
            != group.ticket.requests
        ):
            self._fail_stop(
                "native execution publication changed identity or ordering"
            )
        updates = tuple(
            self._publication_update(item) for item in publication.steps
        )
        transitions = tuple(
            (
                item.request_id,
                int(
                    self._views[
                        self._binding_for_request_id(item.request_id).key
                    ].boundary
                ),
                int(item.boundary),
            )
            for item in publication.steps
        )
        try:
            self._activity.observe_publication(publication, transitions)
        except BaseException as error:
            self._fail_stop(
                "native publication activity became inconsistent", error
            )
        self._run_cleanup(updates, publication.retirements, "publication")
        try:
            self._session.confirm_publication(
                EnginePublicationEvidence(
                    publication.publication_id,
                    True,
                    retirement_evidence(publication.retirements),
                )
            )
        except BaseException as error:
            self._fail_stop(
                "native publication confirmation became uncertain", error
            )
        try:
            self._activity.acknowledge_publication(publication.retirements)
        except BaseException as error:
            self._fail_stop(
                "native publication ACK activity became inconsistent", error
            )
        for item in publication.steps:
            binding = self._binding_for_request_id(item.request_id)
            self._views[binding.key] = EngineRequestView(
                item.request_id,
                int(item.view_version),
                int(item.boundary),
                int(item.resident_count),
            )
        self._completion_high_water[group.completion_domain] = (
            completion_value
        )
        self._counters["completion_values"] += 1
        del self._tickets[group.ticket.batch_id]
        self._events.remove(group)
        return publication

    def _publication_update(self, item: Any) -> SessionMirrorUpdate:
        binding = self._binding_for_request_id(item.request_id)
        if binding.request_row is None:
            self._fail_stop(
                "native publication named an unbound session request"
            )
        return SessionMirrorUpdate(
            binding.key,
            binding.request_id,
            binding.request_row,
            int(item.boundary),
            tuple(item.detached),
            False,
        )

    def _release_updates(
        self, group: _ReleaseGroup
    ) -> tuple[SessionMirrorUpdate, ...]:
        updates = []
        for key, released in zip(
            group.keys, group.plan.releases, strict=True
        ):
            binding = self._binding(key)
            view = self.view_for(key)
            if released.request_id != binding.request_id:
                self._fail_stop(
                    "pending release changed request identity before cleanup"
                )
            if binding.request_row is None:
                if (
                    int(view.boundary) != 0
                    or int(view.resident_count) != 0
                    or released.detached
                ):
                    self._fail_stop(
                        "unbound session release returned physical state"
                    )
                continue
            updates.append(
                SessionMirrorUpdate(
                    key,
                    binding.request_id,
                    binding.request_row,
                    int(view.boundary),
                    tuple(released.detached),
                    True,
                )
            )
        if not updates and group.plan.retirements:
            self._fail_stop(
                "unbound session release returned physical retirements"
            )
        return tuple(updates)

    def _run_cleanup(
        self,
        updates: tuple[SessionMirrorUpdate, ...],
        retirements: Sequence[EngineRetirement],
        operation: str,
    ) -> None:
        values = tuple(retirements)
        if not updates and not values:
            return
        if self._mirror_cleanup is None:
            self._fail_stop(
                f"{operation} has no bound mirror cleanup authority"
            )
        try:
            confirmed = self._mirror_cleanup(updates, values)
        except BaseException as error:
            self._fail_stop(
                f"{operation} mirror cleanup became uncertain", error
            )
        if confirmed is not True:
            self._fail_stop(
                f"{operation} mirror cleanup was not explicitly confirmed"
            )

    def _allocate_request_ids(
        self, count: int
    ) -> tuple[EngineRequestId, ...]:
        end = self._next_request_id + count
        if end > _UINT64_LIMIT:
            raise ManagerError("engine request identity space is exhausted")
        result = tuple(
            EngineRequestId(value)
            for value in range(self._next_request_id, end)
        )
        # IDs are burned before crossing the native boundary.  A failed call
        # can therefore never make a possibly observed identity reusable.
        self._next_request_id = end
        return result

    def _allocate_completion_value(self, completion_domain: int) -> int:
        value = max(
            self._next_completion_value,
            self._completion_high_water.get(completion_domain, 0) + 1,
        )
        if value >= _UINT64_LIMIT - 1:
            raise ManagerError("completion identity space is exhausted")
        # Reserve before calling native code so an uncertain call cannot cause
        # a completion point to be issued twice.
        self._next_completion_value = value + 1
        return value

    def _bindings_for_keys(
        self, keys: Sequence[Hashable], operation: str
    ) -> tuple[SessionRequestBinding, ...]:
        values = tuple(keys)
        if not values:
            raise ManagerError(f"{operation} batch must be nonempty")
        for key in values:
            self._require_hashable(key)
        if len(set(values)) != len(values):
            raise ManagerError(f"{operation} contains duplicate request keys")
        return tuple(self._binding(key) for key in values)

    def _binding(self, key: Hashable) -> SessionRequestBinding:
        self._require_hashable(key)
        try:
            binding = self._bindings[key]
        except KeyError as error:
            raise ManagerError("unknown session request key") from error
        if (
            self._request_keys.get(binding.request_id) != key
            or binding.request_row is None
            and key in self._row_owners.values()
            or binding.request_row is not None
            and (
                isinstance(binding.request_row, bool)
                or not isinstance(binding.request_row, int)
                or binding.request_row <= 0
                or self._row_owners.get(binding.request_row) != key
            )
        ):
            self._fail_stop("session request identity mirror changed unexpectedly")
        return binding

    def _binding_for_request_id(
        self, request_id: EngineRequestId
    ) -> SessionRequestBinding:
        try:
            key = self._request_keys[request_id]
        except KeyError:
            self._fail_stop("native operation named an unknown engine request id")
        return self._binding(key)

    @staticmethod
    def _require_hashable(key: Hashable) -> None:
        try:
            hash(key)
        except BaseException as error:
            raise ManagerError("session request key must be hashable") from error

    def _healthy(self) -> None:
        if self._failure is not None:
            raise FailStopped(
                "OrbitKV session runtime is fail-stopped: " + self._failure
            )
        if self._closed:
            raise ManagerError("OrbitKV session runtime is closed")

    def _fail_stop(
        self, reason: str, error: BaseException | None = None
    ) -> NoReturn:
        if self._failure is None:
            detail = (
                ""
                if error is None
                else f": {type(error).__name__}: {error}"
            )
            self._failure = reason + detail
            self._counters["fail_stop_count"] += 1
        self._closed = True
        try:
            self._session.close()
        except BaseException:
            pass
        finally:
            self._clear_local_state()
        failure = FailStopped(
            "OrbitKV session runtime is fail-stopped: " + self._failure
        )
        if error is None:
            raise failure
        raise failure from error

    def _clear_local_state(self) -> None:
        self._arena_stats_cache = None
        self._bindings.clear()
        self._views.clear()
        self._request_keys.clear()
        self._row_owners.clear()
        self._tickets.clear()
        self._events.clear()
        self._releases.clear()
        self._prefixes.clear()
        self._request_occupancy.clear()
        self._prefix_occupancy.clear()
        self._controls.clear()
        self._pending_attach_cancels.clear()
        self._activity.clear()

    def _invalidate_capacity_cache(self) -> None:
        """Advance the revision before any potentially mutating native call."""

        with self._lock:
            self._capacity_revision += 1
            self._arena_stats_cache = None


__all__ = [
    "MaterializationCallback",
    "MirrorCleanupCallback",
    "MirrorContextFactory",
    "ReleaseAckRetryPending",
    "ReleaseRecyclePending",
    "ReleaseRetryPending",
    "SessionMaterializationUpdate",
    "SessionMirrorUpdate",
    "SessionRequestBinding",
    "SessionRuntime",
    "SessionSwaActivity",
    "collective_mirror_cleanup",
    "unbound_materialization",
    "unbound_mirror_cleanup",
]
