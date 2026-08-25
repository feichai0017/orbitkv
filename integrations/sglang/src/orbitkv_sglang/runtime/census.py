from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Hashable, Sequence

from .identity import ArenaStats, FailStopped, ManagerError, ManagerStats, PageLease
from .pressure import (
    ArenaPressureSample,
    disabled_pressure_report,
    semantic_live_tokens_for_request,
)


@dataclass(frozen=True, slots=True)
class SwaActivity:
    retirement_certificates: int
    pages_reclaimed: int
    wrap_events: int


class CensusRuntimeMixin:
    def _healthy(self) -> None:
        if self._failure is not None:
            raise FailStopped(
                "OrbitKV manager is fail-stopped: " + self._failure
            )

    @property
    def failure_reason(self) -> str | None:
        return self._failure

    def fail_stop(self, reason: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = str(reason)
                self._runtime_counters["fail_stop_count"] += 1

    def record_for(self, key: Hashable) -> Any:
        with self._lock:
            self._healthy()
            try:
                record = self._requests[key]
            except KeyError as error:
                raise ManagerError("request is not acquired") from error
            self._require_indexed_record(record)
            return record

    def has_request(self, key: Hashable) -> bool:
        with self._lock:
            self._healthy()
            return key in self._requests

    def close(self) -> None:
        with self._lock:
            if self._failure is None and (
                self._events
                or self._requests
                or self._candidate_pages
                or self._page_registry
                or self._request_rows
                or self._row_owners
                or self._identity_indexes_live()
                or getattr(self, "_pending_relocation_batches", {})
            ):
                raise ManagerError("cannot destroy a manager with live requests")
            self.manager.destroy()

    @staticmethod
    def _zero_page() -> PageLease:
        return PageLease(0, 0, 0, 0, 0)

    def performance_counters(self) -> dict[str, int]:
        with self._lock:
            counters = dict(getattr(self.manager, "performance_counters", {}))
            counters.update(self._runtime_counters)
            return counters

    def swa_activity(self) -> SwaActivity:
        with self._lock:
            self._healthy()
            return SwaActivity(
                self._swa_retirement_certificates,
                self._swa_pages_reclaimed,
                self._swa_wrap_events,
            )

    @property
    def pressure_telemetry_enabled(self) -> bool:
        return getattr(self, "_pressure", None) is not None

    def pressure_report(self) -> dict[str, object]:
        """Return the event-driven pressure report without taking a sample."""

        with self._lock:
            collector = getattr(self, "_pressure", None)
            return (
                disabled_pressure_report()
                if collector is None
                else collector.report()
            )

    def _pressure_require_request_private(self, operation: str) -> None:
        if getattr(self, "_pressure", None) is not None:
            raise ManagerError(
                "request-private pressure telemetry does not support "
                f"{operation}; shared Prefix/fork RA requires deduplicated "
                "semantic ownership"
            )

    def _pressure_state(
        self,
    ) -> tuple[
        int,
        tuple[ArenaPressureSample, ...],
        dict[int, int],
        dict[int, int],
    ]:
        stats, arena_stats = self.census()
        if stats.active_requests != len(self._requests):
            raise ManagerError(
                "native and request-journal active counts disagree"
            )
        if getattr(self._page_registry, "_prefixes", None):
            raise ManagerError(
                "request-private pressure telemetry observed a shared Prefix"
            )
        if stats.active_prefixes or stats.total_prefix_page_refs or any(
            item.prefix_page_refs for item in arena_stats
        ):
            raise ManagerError(
                "request-private pressure telemetry observed native Prefix refs"
            )

        reachable: dict[
            int, dict[tuple[int, int], tuple[PageLease, object]]
        ] = {
            item.class_id: {} for item in self.arenas
        }
        semantic = {item.class_id: 0 for item in self.arenas}
        for record in self._requests.values():
            for page in record.cursor.pages.values():
                pages = reachable.get(page.class_id)
                if pages is None:
                    raise ManagerError(
                        "request journal references an unknown class"
                    )
                physical = (page.page.pool_id, page.page.page_id)
                previous = pages.get(physical)
                if previous is not None and (
                    previous[0] != page.page or previous[1] != record.lease
                ):
                    raise ManagerError(
                        "request-private pressure telemetry observed a shared or "
                        "conflicting physical page"
                    )
                pages[physical] = (page.page, record.lease)
            for class_config in self.config.classes:
                active = record.cursor.active_kv_lengths.get(
                    class_config.class_id
                )
                semantic[class_config.class_id] += (
                    semantic_live_tokens_for_request(
                        boundary=record.boundary,
                        retention=class_config.retention,
                        window_tokens=class_config.window_tokens,
                        active_kv_length=active,
                    )
                )

        samples = tuple(
            ArenaPressureSample(
                class_id=item.class_id,
                page_count=item.page_count,
                free_pages=item.free_pages,
                reserved_pages=item.reserved_pages,
                writing_pages=item.writing_pages,
                active_pages=item.active_pages,
                retiring_pages=item.retiring_pages,
                quarantined_pages=item.quarantined_pages,
                exhausted_pages=item.exhausted_pages,
            )
            for item in arena_stats
        )
        return (
            stats.active_requests,
            samples,
            {class_id: len(pages) for class_id, pages in reachable.items()},
            semantic,
        )

    def _pressure_checkpoint(self, event: str) -> None:
        """Sample a committed transition only when explicitly enabled."""

        if getattr(self, "_pressure", None) is None:
            # Keep the normal runtime path free of native stats calls.
            return
        with self._lock:
            try:
                active_requests, samples, reachable, semantic = (
                    self._pressure_state()
                )
                self._pressure.sample(
                    event,
                    active_requests=active_requests,
                    arenas=samples,
                    request_reachable_unique_pages=reachable,
                    semantic_live_tokens=semantic,
                )
            except FailStopped:
                raise
            except Exception as error:
                self.fail_stop(
                    f"pressure telemetry became uncertain at {event}: {error}"
                )
                raise FailStopped(
                    self._failure or "pressure telemetry failed"
                ) from error

    def pressure_checkpoint(self, event: str) -> None:
        """Record one caller-defined, successfully committed lifecycle event."""

        self._pressure_checkpoint(event)

    def stats(self) -> ManagerStats:
        return self.census()[0]

    def census(self) -> tuple[ManagerStats, tuple[ArenaStats, ...]]:
        """Return one aggregate/per-arena census from one validated sample."""

        with self._lock:
            self._healthy()
            try:
                stats = self.manager.stats()
                counts = tuple(
                    getattr(stats, item.name) for item in fields(ManagerStats)
                )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in counts
                ):
                    raise ManagerError("manager stats contain an invalid counter")
                arena_stats = self._arena_stats_unlocked()
                self._validate_aggregate_stats(stats, arena_stats)
                return stats, arena_stats
            except Exception as error:
                self.fail_stop(f"manager census became uncertain: {error}")
                raise FailStopped(self._failure or "manager census failed") from error

    def arena_stats(self) -> tuple[ArenaStats, ...]:
        with self._lock:
            self._healthy()
            try:
                return self._arena_stats_unlocked()
            except Exception as error:
                self.fail_stop(f"manager arena census became uncertain: {error}")
                raise FailStopped(self._failure or "manager arena census failed") from error

    def _arena_stats_unlocked(self) -> tuple[ArenaStats, ...]:
        arena_stats = tuple(self.manager.arena_stats())
        if len(arena_stats) != len(self.arenas):
            raise ManagerError("manager returned the wrong arena-stats count")
        phase_names = (
            "free_pages",
            "reserved_pages",
            "writing_pages",
            "active_pages",
            "retiring_pages",
            "quarantined_pages",
            "exhausted_pages",
        )
        counter_names = phase_names + (
            "request_page_refs",
            "prefix_page_refs",
            "reader_pins",
        )
        for identity, item in zip(self.arenas, arena_stats, strict=True):
            if any(
                getattr(item, name) != getattr(identity, name)
                for name in (
                    "engine_epoch",
                    "pool_epoch",
                    "pool_id",
                    "page_count",
                    "class_id",
                    "backend_domain",
                    "first_page_id",
                )
            ):
                raise ManagerError("arena stats changed arena identity")
            counters = tuple(getattr(item, name) for name in counter_names)
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counters
            ):
                raise ManagerError("arena stats contain an invalid counter")
            if sum(getattr(item, name) for name in phase_names) != identity.page_count:
                raise ManagerError("arena page census does not match its identity")
        return arena_stats

    @staticmethod
    def _validate_aggregate_stats(
        stats: ManagerStats, arena_stats: Sequence[ArenaStats]
    ) -> None:
        phase_names = (
            "free_pages",
            "reserved_pages",
            "writing_pages",
            "active_pages",
            "retiring_pages",
            "quarantined_pages",
            "exhausted_pages",
        )
        totals = {name: 0 for name in phase_names}
        for item in arena_stats:
            for name in phase_names:
                totals[name] += getattr(item, name)
        if any(getattr(stats, name) != totals[name] for name in phase_names):
            raise ManagerError("aggregate stats disagree with per-arena census")
        if (
            stats.total_request_page_refs
            != sum(item.request_page_refs for item in arena_stats)
            or stats.total_prefix_page_refs
            != sum(item.prefix_page_refs for item in arena_stats)
            or stats.total_reader_pins != sum(item.reader_pins for item in arena_stats)
        ):
            raise ManagerError("aggregate reference census disagrees with arenas")


__all__ = ["CensusRuntimeMixin", "SwaActivity"]
