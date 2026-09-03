from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .identity import PageLease


class TokenDispositionKind(IntEnum):
    RETAINED = 0
    SEMANTICALLY_DEAD = 1
    POLICY_EVICTED = 2


@dataclass(frozen=True, slots=True)
class TokenDisposition:
    kind: TokenDispositionKind
    policy_or_proof_id: int = 0
    version: int = 0
    quality_contract: int = 0


@dataclass(frozen=True, slots=True)
class TokenLocation:
    page: PageLease
    backend_index: int
    offset: int


@dataclass(frozen=True, slots=True)
class TokenPlacement:
    token_id: int
    disposition: TokenDisposition
    location: TokenLocation | None


@dataclass(frozen=True, slots=True)
class RelocationPolicy:
    maximum_source_pages: int = 32
    evacuation_headroom_pages: int = 8
    fragmentation_threshold_milli: int = 250
    full_evacuation: bool = True


@dataclass(frozen=True, slots=True)
class TokenMove:
    token_id: int
    source: TokenLocation
    destination: TokenLocation


__all__ = [
    "RelocationPolicy",
    "TokenDisposition",
    "TokenDispositionKind",
    "TokenLocation",
    "TokenMove",
    "TokenPlacement",
]
