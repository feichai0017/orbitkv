from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


TAIL_NONE = 0
TAIL_IN_PLACE = 1
TAIL_COPY_ON_WRITE = 2
TAIL_FRESH = 3
CLASS_LOWERING_PACKED = 1

DETACHED_CLEAR = 1
DETACHED_REPLACE = 2
DETACHED_RETENTION = 1
DETACHED_COPY_ON_WRITE = 2
DETACHED_REQUEST_RELEASE = 3
DETACHED_PREFIX_TRANSFER = 4


class CacheSharingPolicy(IntEnum):
    """Cross-request cache visibility selected when a session is created."""

    REQUEST_PRIVATE = 1
    SHARED_PREFIX = 2


class ManagerError(RuntimeError):
    """The native runtime rejected an operation or returned invalid data."""


class RetryableConflict(ManagerError):
    """A proven precommit conflict left the whole manager batch unchanged."""


class FailStopped(RuntimeError):
    """The adapter quarantined uncertain state and cannot safely continue."""


@dataclass(frozen=True, slots=True)
class ArenaIdentity:
    engine_epoch: int
    pool_epoch: int
    pool_id: int
    class_id: int
    backend_domain: int
    page_count: int
    page_tokens: int
    backend_base_index: int
    first_page_id: int


@dataclass(frozen=True, slots=True)
class ArenaRegistration:
    class_id: int
    pool_id: int
    backend_domain: int
    page_count: int
    backend_base_index: int = 0


@dataclass(frozen=True, slots=True)
class ArenaStats:
    engine_epoch: int
    pool_epoch: int
    pool_id: int
    page_count: int
    class_id: int
    backend_domain: int
    first_page_id: int
    free_pages: int
    reserved_pages: int
    writing_pages: int
    active_pages: int
    retiring_pages: int
    quarantined_pages: int
    exhausted_pages: int
    request_page_refs: int
    prefix_page_refs: int
    reader_pins: int


@dataclass(frozen=True, slots=True)
class ManagerCreateSettings:
    maximum_requests: int
    maximum_operations: int
    maximum_prefixes: int
    maximum_reclamations: int
    maximum_step_tokens: int


@dataclass(frozen=True, slots=True)
class SessionCreateSettings:
    manager: ManagerCreateSettings
    cache_sharing_policy: CacheSharingPolicy


@dataclass(frozen=True, slots=True)
class PageLease:
    engine_epoch: int
    pool_epoch: int
    generation: int
    page_id: int
    pool_id: int


@dataclass(frozen=True, slots=True)
class PrefixSemanticKey:
    namespace: bytes
    digest: bytes
    boundary: int


@dataclass(frozen=True, slots=True)
class ManagerStats:
    active_requests: int
    active_snapshots: int
    active_prefixes: int
    evicted_prefixes: int
    prepared_steps: int
    submitted_steps: int
    free_pages: int
    reserved_pages: int
    writing_pages: int
    active_pages: int
    retiring_pages: int
    quarantined_pages: int
    exhausted_pages: int
    pending_reclamations: int
    total_request_page_refs: int
    total_prefix_page_refs: int
    total_reader_pins: int


__all__ = [
    "ArenaIdentity",
    "ArenaRegistration",
    "ArenaStats",
    "CacheSharingPolicy",
    "FailStopped",
    "ManagerCreateSettings",
    "ManagerError",
    "ManagerStats",
    "PageLease",
    "PrefixSemanticKey",
    "RetryableConflict",
    "SessionCreateSettings",
]
