from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .completion import StepRecord
from .identity import RequestLease, SnapshotLease
from .reclamation import MirrorCleanupBinding, MirrorCleanupProtocol
from .snapshot_shadow import RequestCursor


@dataclass(slots=True)
class RequestRecord:
    cursor: RequestCursor
    pending: StepRecord | None = None
    completion_domain: int = 0
    completion_value: int = 0
    reclamation_cleanup: MirrorCleanupBinding | None = None
    swa_temporal_cycles: dict[int, int] = field(default_factory=dict)

    @property
    def lease(self) -> RequestLease:
        return self.cursor.lease

    @property
    def head(self) -> SnapshotLease:
        return self.cursor.snapshot

    @property
    def boundary(self) -> int:
        return self.cursor.boundary


@dataclass(frozen=True, slots=True)
class MirrorTransaction:
    entries: tuple[tuple[MirrorCleanupProtocol, Any], ...]


__all__ = ["MirrorTransaction", "RequestRecord"]
