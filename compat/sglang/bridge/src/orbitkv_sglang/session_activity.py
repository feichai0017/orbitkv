from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .ffi.session_types import (
    EngineBatchPlan,
    EngineBatchPublication,
    EngineRequestId,
    EngineRetirement,
)
from .runtime import (
    ArenaIdentity,
    ManagerError,
    PageLease,
    TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
)


_PageIdentity = tuple[int, int, int, int, int]
_ClassSignature = tuple[tuple[int, str, int | None], ...]


@dataclass(frozen=True, slots=True)
class SessionSwaActivity:
    """Cumulative native-session activity for configured Sliding classes."""

    applicable: bool
    retirement_certificates: int
    pages_reclaimed: int
    wrap_events: int
    page_reuse_events: int


class SessionActivityTracker:
    """Observe native lifecycle facts without becoming lifecycle authority.

    Class metadata intentionally arrives through :meth:`activity`; the native
    session wrapper does not retain a second copy of ``RuntimeConfig``. Events
    observed before the first report stay partitioned by class and are folded
    once the canonical class metadata is supplied.
    """

    def __init__(
        self, arenas: Sequence[ArenaIdentity], page_tokens: int
    ) -> None:
        self._arenas_by_class = {item.class_id: item for item in arenas}
        self._page_tokens = page_tokens
        self._retirement_certificates: dict[int, int] = {}
        self._pages_reclaimed: dict[int, int] = {}
        self._wrap_events: dict[int, int] = {}
        self._page_reuse_events: dict[int, int] = {}
        self._page_generations: dict[_PageIdentity, int] = {}
        self._acknowledged_generations: dict[_PageIdentity, int] = {}
        self._class_signature: _ClassSignature | None = None
        self._period_blocks: dict[int, int] | None = None
        self._pending_boundary_transitions: list[
            tuple[EngineRequestId, int, int]
        ] = []
        self._request_cycles: dict[EngineRequestId, dict[int, int]] = {}
        self._forgotten_requests: set[EngineRequestId] = set()

    def activity(self, classes: Sequence[Any]) -> SessionSwaActivity:
        periods = self._configure_classes(classes)
        sliding_ids = frozenset(periods)
        return SessionSwaActivity(
            bool(sliding_ids),
            self._sum(self._retirement_certificates, sliding_ids),
            self._sum(self._pages_reclaimed, sliding_ids),
            self._sum(self._wrap_events, sliding_ids),
            self._sum(self._page_reuse_events, sliding_ids),
        )

    def observe_plan(self, plan: EngineBatchPlan) -> None:
        """Record generations of physical pages reserved by native prepare."""

        observed: set[_PageIdentity] = set()
        reservations: list[tuple[int, _PageIdentity, int]] = []
        for step in plan.steps:
            for lowering in step.class_lowerings:
                class_id = int(lowering.class_id)
                try:
                    arena = self._arenas_by_class[class_id]
                except KeyError as error:
                    raise RuntimeError(
                        "prepared page observation names an unknown class"
                    ) from error
                tail_begin = int(lowering.tail_offset)
                tail_end = tail_begin + int(lowering.tail_count)
                write_begin = int(lowering.write_offset)
                write_end = write_begin + int(lowering.write_count)
                if (
                    tail_begin < 0
                    or tail_end > len(step.tail_actions)
                    or write_begin < 0
                    or write_end > len(step.write_intents)
                ):
                    raise RuntimeError(
                        "prepared page observation span is out of bounds"
                    )
                pages = [
                    action.destination
                    for action in step.tail_actions[tail_begin:tail_end]
                    if int(action.kind) in (TAIL_COPY_ON_WRITE, TAIL_FRESH)
                ]
                pages.extend(
                    PageLease(
                        arena.engine_epoch,
                        arena.pool_epoch,
                        int(intent.page_generation),
                        int(intent.page_id),
                        arena.pool_id,
                    )
                    for intent in step.write_intents[write_begin:write_end]
                )
                for page in pages:
                    identity = self._page_identity(class_id, page)
                    if identity in observed:
                        raise RuntimeError(
                            "prepared page observation duplicated a reservation"
                        )
                    observed.add(identity)
                    generation = int(page.generation)
                    previous = self._page_generations.get(identity)
                    if previous is not None and generation <= previous:
                        raise RuntimeError(
                            "prepared page generation did not advance"
                        )
                    reservations.append((class_id, identity, generation))

        for class_id, identity, generation in reservations:
            previous = self._page_generations.get(identity)
            acknowledged = self._acknowledged_generations.get(identity)
            if (
                previous is not None
                and acknowledged == previous
                and generation > previous
            ):
                self._increment(self._page_reuse_events, class_id)
                del self._acknowledged_generations[identity]
            # The first observed generation establishes a baseline only.
            self._page_generations[identity] = generation

    def observe_publication(
        self,
        publication: EngineBatchPublication,
        transitions: Sequence[tuple[EngineRequestId, int, int]],
    ) -> None:
        """Record issued certificates and completed boundary transitions."""

        values = tuple(transitions)
        if len(values) != len(publication.steps):
            raise RuntimeError(
                "SWA publication transition cardinality changed"
            )
        for retirement in publication.retirements:
            self._page_identity(int(retirement.class_id), retirement.page)
        if tuple(item[0] for item in values) != tuple(
            item.request_id for item in publication.steps
        ):
            raise RuntimeError(
                "SWA publication transition identity changed"
            )
        for _request_id, previous, target in values:
            if target < previous:
                raise RuntimeError(
                    "native publication moved a request boundary backwards"
                )

        projected_certificates = dict(self._retirement_certificates)
        for retirement in publication.retirements:
            self._increment(
                projected_certificates, int(retirement.class_id)
            )
        periods = self._period_blocks
        if periods is None:
            projected_pending = [*self._pending_boundary_transitions, *values]
            projected_wraps = self._wrap_events
            projected_cycles = self._request_cycles
        else:
            projected_pending = self._pending_boundary_transitions
            projected_wraps = dict(self._wrap_events)
            projected_cycles = {
                request_id: dict(cycles)
                for request_id, cycles in self._request_cycles.items()
            }
            for request_id, previous, target in values:
                self._record_wraps_into(
                    request_id, previous, target, periods,
                    projected_wraps, projected_cycles,
                )
        self._retirement_certificates = projected_certificates
        self._pending_boundary_transitions = projected_pending
        self._wrap_events = projected_wraps
        self._request_cycles = projected_cycles

    def acknowledge_publication(
        self, retirements: Sequence[EngineRetirement]
    ) -> None:
        """Record pages whose exact publication ACK native accepted."""

        self._record_acknowledged_pages(retirements, count_reclaimed=True)

    def acknowledge_reuse_candidates(
        self, retirements: Sequence[EngineRetirement]
    ) -> None:
        """Remember non-publication ACKs solely as future reuse evidence."""

        self._record_acknowledged_pages(retirements, count_reclaimed=False)

    def _record_acknowledged_pages(
        self,
        retirements: Sequence[EngineRetirement],
        *,
        count_reclaimed: bool,
    ) -> None:

        values = tuple(retirements)
        observations = []
        seen: set[_PageIdentity] = set()
        for retirement in values:
            class_id = int(retirement.class_id)
            identity = self._page_identity(class_id, retirement.page)
            generation = int(retirement.page.generation)
            observed = self._page_generations.get(identity)
            if observed is not None and generation < observed:
                raise RuntimeError(
                    "publication ACK named an older observed page generation"
                )
            if identity in seen:
                raise RuntimeError("page ACK observation duplicated a page")
            seen.add(identity)
            observations.append((class_id, identity, generation))

        for class_id, identity, generation in observations:
            if count_reclaimed:
                self._increment(self._pages_reclaimed, class_id)
            self._page_generations.setdefault(identity, generation)
            self._acknowledged_generations[identity] = generation

    def forget_requests(self, request_ids: Sequence[EngineRequestId]) -> None:
        for request_id in request_ids:
            self._request_cycles.pop(request_id, None)
            if self._period_blocks is None:
                self._forgotten_requests.add(request_id)

    def clear(self) -> None:
        self._pending_boundary_transitions.clear()
        self._request_cycles.clear()
        self._forgotten_requests.clear()
        self._page_generations.clear()
        self._acknowledged_generations.clear()

    def _configure_classes(self, classes: Sequence[Any]) -> dict[int, int]:
        values = tuple(classes)
        arena_ids = tuple(self._arenas_by_class)
        if len(values) != len(arena_ids):
            raise ManagerError(
                "SWA activity classes differ from native session arenas"
            )
        signature: list[tuple[int, str, int | None]] = []
        periods: dict[int, int] = {}
        for item, arena_id in zip(values, arena_ids, strict=True):
            class_id = getattr(item, "class_id", None)
            retention = getattr(item, "retention", None)
            raw_period = getattr(item, "period_blocks", None)
            if (
                isinstance(class_id, bool)
                or not isinstance(class_id, int)
                or class_id != arena_id
                or retention not in ("full", "sliding", "chunked")
            ):
                raise ManagerError(
                    "SWA activity classes differ from native session arenas"
                )
            period = None
            if retention == "sliding":
                if (
                    isinstance(raw_period, bool)
                    or not isinstance(raw_period, int)
                    or raw_period <= 0
                ):
                    raise ManagerError(
                        "SWA activity requires a positive Sliding period"
                    )
                period = raw_period
                periods[class_id] = period
            signature.append((class_id, retention, period))

        class_signature = tuple(signature)
        if self._class_signature is not None:
            if class_signature != self._class_signature:
                raise ManagerError(
                    "SWA activity class configuration changed during the session"
                )
            assert self._period_blocks is not None
            return self._period_blocks
        projected_wraps = dict(self._wrap_events)
        projected_cycles = {
            request_id: dict(cycles)
            for request_id, cycles in self._request_cycles.items()
        }
        for request_id, previous, target in self._pending_boundary_transitions:
            self._record_wraps_into(
                request_id, previous, target, periods,
                projected_wraps, projected_cycles,
            )
        self._class_signature = class_signature
        self._period_blocks = periods
        self._wrap_events = projected_wraps
        self._request_cycles = projected_cycles
        self._pending_boundary_transitions.clear()
        for request_id in self._forgotten_requests:
            self._request_cycles.pop(request_id, None)
        self._forgotten_requests.clear()
        return periods

    def _record_wraps_into(
        self,
        request_id: EngineRequestId,
        previous_boundary: int,
        target_boundary: int,
        periods: dict[int, int],
        wrap_events: dict[int, int],
        request_cycles: dict[EngineRequestId, dict[int, int]],
    ) -> None:
        cycles = request_cycles.setdefault(request_id, {})
        for class_id, period_blocks in periods.items():
            period_tokens = self._page_tokens * period_blocks
            previous_cycle = (
                0
                if previous_boundary == 0
                else (previous_boundary - 1) // period_tokens
            )
            target_cycle = (
                0 if target_boundary == 0 else (target_boundary - 1) // period_tokens
            )
            recorded = cycles.get(class_id)
            if recorded is not None and recorded != previous_cycle:
                raise RuntimeError(
                    "SWA request cycle differs from its confirmed boundary"
                )
            if target_cycle < previous_cycle:
                raise RuntimeError("SWA temporal cycle moved backwards")
            wrap_events[class_id] = (
                wrap_events.get(class_id, 0)
                + target_cycle
                - previous_cycle
            )
            cycles[class_id] = target_cycle

    def _page_identity(
        self, class_id: int, page: PageLease
    ) -> _PageIdentity:
        try:
            arena = self._arenas_by_class[class_id]
        except KeyError as error:
            raise RuntimeError(
                "page observation names an unknown native arena"
            ) from error
        if (
            page.engine_epoch != arena.engine_epoch
            or page.pool_epoch != arena.pool_epoch
            or page.pool_id != arena.pool_id
            or not (
                arena.first_page_id
                <= page.page_id
                < arena.first_page_id + arena.page_count
            )
        ):
            raise RuntimeError(
                "page observation differs from its native arena"
            )
        return (
            class_id,
            int(page.engine_epoch),
            int(page.pool_epoch),
            int(page.pool_id),
            int(page.page_id),
        )

    @staticmethod
    def _increment(counters: dict[int, int], class_id: int) -> None:
        counters[class_id] = counters.get(class_id, 0) + 1

    @staticmethod
    def _sum(counters: dict[int, int], class_ids: frozenset[int]) -> int:
        return sum(counters.get(class_id, 0) for class_id in class_ids)


__all__ = ["SessionActivityTracker", "SessionSwaActivity"]
