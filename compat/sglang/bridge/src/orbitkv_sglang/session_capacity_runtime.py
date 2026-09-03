from __future__ import annotations

from typing import Any, Callable


_CAPACITY_READS = frozenset(("arena_identities", "arena_stats", "stats"))


class CapacityInvalidatingSession:
    """Invalidate cached capacity before crossing a native mutation boundary.

    The wrapper deliberately treats every native call other than the three
    census/identity reads as a possible capacity mutation.  This is broader
    than the current native implementation requires, but guarantees that a
    newly added release, control, or Prefix operation cannot accidentally
    leave scheduler admission using a stale, over-generous arena census.
    """

    __slots__ = ("_invalidate", "_session")

    def __init__(self, session: Any, invalidate: Callable[[], None]) -> None:
        self._session = session
        self._invalidate = invalidate

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._session, name)
        if name in _CAPACITY_READS or not callable(value):
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            self._invalidate()
            return value(*args, **kwargs)

        return call


__all__ = ["CapacityInvalidatingSession"]
