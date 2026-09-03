from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from .identity import PageLease


@dataclass(frozen=True, slots=True)
class DetachedBinding:
    old: PageLease
    replacement: PageLease
    logical_ordinal: int
    old_backend_index: int
    replacement_backend_index: int
    token_begin: int
    token_end_exclusive: int
    class_id: int
    backend_domain: int
    action: int
    reason: int
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class ReclamationCertificate:
    page: PageLease
    class_id: int
    backend_domain: int
    logical_ordinal: int
    backend_index: int
    token_begin: int
    token_end_exclusive: int
    completion_domain: int
    completion_value: int


@dataclass(frozen=True, slots=True)
class MirrorCandidateTransition:
    destination: PageLease
    source: PageLease
    logical_ordinal: int
    destination_backend_index: int
    source_backend_index: int
    token_begin: int
    token_end_exclusive: int
    copied_token_begin: int
    copied_token_end_exclusive: int
    class_id: int
    backend_domain: int
    retiring: bool
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class MirrorCleanupItem:
    context: Any
    detached: tuple[DetachedBinding, ...]
    releasing: bool
    boundary: int
    candidates: tuple[MirrorCandidateTransition, ...] = ()


@runtime_checkable
class MirrorCleanupProtocol(Protocol):
    def preflight(
        self,
        items: Sequence[MirrorCleanupItem],
        retirements: Sequence[ReclamationCertificate],
    ) -> Any: ...

    def commit(self, plan: Any) -> None: ...

    def synchronize(self, plan: Any) -> None: ...

    def finalize(self, plan: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class MirrorCleanupBinding:
    coordinator: MirrorCleanupProtocol
    context: Any


__all__ = [
    "DetachedBinding",
    "MirrorCandidateTransition",
    "MirrorCleanupBinding",
    "MirrorCleanupItem",
    "MirrorCleanupProtocol",
    "ReclamationCertificate",
]
