from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

from .identity import ArenaStats, FailStopped, ManagerError, ManagerStats


@dataclass(frozen=True, slots=True)
class SwaActivity:
    retirement_certificates: int
    pages_reclaimed: int
    wrap_events: int


class CensusRuntimeMixin:
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
