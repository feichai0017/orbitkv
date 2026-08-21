from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .identity import ManagerError, PageLease, PrefixLease
from .snapshot_shadow import PageShadow


PhysicalPageKey = tuple[int, int]
ShadowIdentity = tuple[int, int, PageLease, int]


def shadow_identity(value: PageShadow) -> ShadowIdentity:
    return value.class_id, value.logical_ordinal, value.page, value.backend_index


def canonical_shadows(values: Sequence[PageShadow]) -> tuple[PageShadow, ...]:
    result = tuple(sorted(values, key=lambda item: (item.class_id, item.logical_ordinal)))
    if len({(item.class_id, item.logical_ordinal) for item in result}) != len(result):
        raise ManagerError("page shadows duplicate a logical identity")
    return result


def shadow_identities(values: Sequence[PageShadow]) -> tuple[ShadowIdentity, ...]:
    return tuple(map(shadow_identity, canonical_shadows(values)))


@dataclass(frozen=True, slots=True)
class PageRefPlan:
    old: Counter[PageLease]
    new: Counter[PageLease]
    # Manager-reserved destinations that either become `new` references or
    # retire before ever entering a published snapshot.
    transient: Counter[PageLease]


class PhysicalPageRegistry:
    """Incremental host witness for request and Prefix page references."""

    def __init__(self) -> None:
        self._pages: dict[PhysicalPageKey, tuple[PageLease, int]] = {}
        self._prefixes: dict[PrefixLease, tuple[PageShadow, ...]] = {}

    @staticmethod
    def physical_key(value: PageShadow | PageLease) -> PhysicalPageKey:
        page = value.page if isinstance(value, PageShadow) else value
        return page.pool_id, page.page_id

    def contains_physical(self, value: PageShadow | PageLease) -> bool:
        return self.physical_key(value) in self._pages

    def plan(
        self,
        old_values: Sequence[PageShadow | PageLease],
        new_values: Sequence[PageShadow | PageLease],
        transient_values: Sequence[PageShadow | PageLease] = (),
    ) -> PageRefPlan:
        old = Counter(
            value.page if isinstance(value, PageShadow) else value
            for value in old_values
        )
        new = Counter(
            value.page if isinstance(value, PageShadow) else value
            for value in new_values
        )
        transient = Counter(
            value.page if isinstance(value, PageShadow) else value
            for value in transient_values
        )
        for lease, count in old.items():
            current = self._pages.get(self.physical_key(lease))
            if current is None or current[0] != lease or current[1] < count:
                raise ManagerError("published page reference journal changed identity")
        additions: dict[PhysicalPageKey, PageLease] = {}
        for lease in new:
            physical = self.physical_key(lease)
            prior = additions.setdefault(physical, lease)
            current = self._pages.get(physical)
            if prior != lease or (current is not None and current[0] != lease):
                raise ManagerError("published views alias one physical page generation")
        transient_physical: dict[PhysicalPageKey, PageLease] = {}
        for lease, count in transient.items():
            physical = self.physical_key(lease)
            prior = transient_physical.setdefault(physical, lease)
            current = self._pages.get(physical)
            published = additions.get(physical)
            if (
                count != 1
                or prior != lease
                or current is not None
                or published is not None and published != lease
                or new[lease] not in (0, 1)
            ):
                raise ManagerError("transient candidate aliases a published page")
        return PageRefPlan(old, new, transient)

    def commit(self, plan: PageRefPlan) -> None:
        for lease, count in plan.old.items():
            physical = self.physical_key(lease)
            current_lease, current_count = self._pages[physical]
            assert current_lease == lease and current_count >= count
            remaining = current_count - count
            if remaining:
                self._pages[physical] = (lease, remaining)
            else:
                del self._pages[physical]
        for lease, count in plan.new.items():
            physical = self.physical_key(lease)
            current = self._pages.get(physical)
            assert current is None or current[0] == lease
            self._pages[physical] = (
                lease,
                count + (0 if current is None else current[1]),
            )

    def expected_retirements(self, plan: PageRefPlan) -> frozenset[PageLease]:
        expected: set[PageLease] = set()
        for lease in plan.old.keys() | plan.new.keys() | plan.transient.keys():
            current = self._pages.get(self.physical_key(lease))
            current_count = 0 if current is None else current[1]
            after = current_count - plan.old[lease] + plan.new[lease]
            if after < 0:
                raise ManagerError("page reference plan underflowed")
            loses_last_published_ref = plan.old[lease] > plan.new[lease]
            retires_unpublished_candidate = (
                plan.transient[lease] > plan.new[lease]
            )
            if (loses_last_published_ref or retires_unpublished_candidate) and after == 0:
                expected.add(lease)
        return frozenset(expected)

    def has_prefix(self, prefix: PrefixLease) -> bool:
        return prefix in self._prefixes

    def prefix_pages(self, prefix: PrefixLease) -> tuple[PageShadow, ...] | None:
        return self._prefixes.get(prefix)

    def install_prefix(
        self, prefix: PrefixLease, pages: Sequence[PageShadow]
    ) -> None:
        if prefix in self._prefixes:
            raise ManagerError("prefix identity is already live")
        self._prefixes[prefix] = canonical_shadows(pages)

    def remove_prefix(self, prefix: PrefixLease) -> None:
        if self._prefixes.pop(prefix, None) is None:
            raise ManagerError("prefix identity is not live")

    def __bool__(self) -> bool:
        return bool(self._pages or self._prefixes)


__all__ = [
    "PageRefPlan",
    "PhysicalPageRegistry",
    "canonical_shadows",
    "shadow_identities",
]
