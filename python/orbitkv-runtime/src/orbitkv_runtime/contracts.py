"""Typed, engine-neutral contracts for OrbitKV tensor-arena adapters.

This module intentionally depends only on the Python standard library.  It is
the boundary between manager-issued identities and an engine-owned data plane;
it is not an allocator or an ownership manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Protocol, Sequence, runtime_checkable


BytesLike = bytes | bytearray | memoryview


def _uint(name: str, value: int, bits: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    if value >= 1 << bits:
        raise ValueError(f"{name} exceeds uint{bits}")


def _u16(name: str, value: int, *, positive: bool = False) -> None:
    _uint(name, value, 16, positive=positive)


def _u32(name: str, value: int, *, positive: bool = False) -> None:
    _uint(name, value, 32, positive=positive)


def _u64(name: str, value: int, *, positive: bool = False) -> None:
    _uint(name, value, 64, positive=positive)


@dataclass(frozen=True, slots=True)
class PageLease:
    """Generation-bearing page authority issued by the OrbitKV manager."""

    engine_epoch: int
    pool_epoch: int
    generation: int
    page_id: int
    pool_id: int

    def __post_init__(self) -> None:
        for name in ("engine_epoch", "pool_epoch", "generation"):
            _u64(name, getattr(self, name), positive=True)
        for name in ("page_id", "pool_id"):
            _u32(name, getattr(self, name), positive=True)


@dataclass(frozen=True, slots=True)
class RequestLease:
    engine_epoch: int
    slot: int
    generation: int

    def __post_init__(self) -> None:
        _u64("engine_epoch", self.engine_epoch, positive=True)
        _u32("slot", self.slot)
        _u32("generation", self.generation, positive=True)


@dataclass(frozen=True, slots=True)
class StepLease:
    """Generation-bearing append transaction identity."""

    engine_epoch: int
    slot: int
    generation: int

    def __post_init__(self) -> None:
        _u64("engine_epoch", self.engine_epoch, positive=True)
        _u32("slot", self.slot)
        _u32("generation", self.generation, positive=True)


@dataclass(frozen=True, slots=True)
class RelocationLease:
    """Generation-bearing token-relocation transaction identity."""

    engine_epoch: int
    slot: int
    generation: int

    def __post_init__(self) -> None:
        _u64("engine_epoch", self.engine_epoch, positive=True)
        _u32("slot", self.slot)
        _u32("generation", self.generation, positive=True)


class DataPlaneOperation(str, Enum):
    APPEND = "append"
    RELOCATE = "relocate"


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Exact request and manager transaction that authorized an effect."""

    request: RequestLease
    transaction: StepLease | RelocationLease
    operation: DataPlaneOperation

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestLease):
            raise TypeError("request must be a RequestLease")
        if not isinstance(self.operation, DataPlaneOperation):
            raise TypeError("operation must be a DataPlaneOperation")
        expected_type = (
            StepLease
            if self.operation is DataPlaneOperation.APPEND
            else RelocationLease
        )
        if not isinstance(self.transaction, expected_type):
            raise TypeError(
                f"{self.operation.value} requires {expected_type.__name__}"
            )
        if self.request.engine_epoch != self.transaction.engine_epoch:
            raise ValueError("request and transaction engine epochs differ")


@dataclass(frozen=True, slots=True)
class ReclamationLease:
    engine_epoch: int
    slot: int
    generation: int

    def __post_init__(self) -> None:
        _u64("engine_epoch", self.engine_epoch, positive=True)
        _u32("slot", self.slot)
        _u32("generation", self.generation, positive=True)


@dataclass(frozen=True, slots=True)
class ArenaRegistration:
    """Geometry and identity of one externally allocated tensor arena.

    ``backend_index`` values occupy ``[backend_base_index,
    backend_base_index + page_count)``.  Page ids occupy an equally sized
    range beginning at ``first_page_id``.  No engine-specific dummy-page
    offset is implied.
    """

    engine_epoch: int
    pool_epoch: int
    pool_id: int
    class_id: int
    backend_domain: int
    page_count: int
    page_tokens: int
    token_bytes: int
    backend_base_index: int = 0
    first_page_id: int = 1

    def __post_init__(self) -> None:
        for name in (
            "engine_epoch",
            "pool_epoch",
            "token_bytes",
        ):
            _u64(name, getattr(self, name), positive=True)
        for name in ("pool_id", "page_count", "page_tokens", "first_page_id"):
            _u32(name, getattr(self, name), positive=True)
        _u16("class_id", self.class_id)
        _u16("backend_domain", self.backend_domain)
        _u64("backend_base_index", self.backend_base_index)
        if self.first_page_id + self.page_count > 1 << 32:
            raise ValueError("arena page-id range exceeds uint32")
        if self.backend_base_index + self.page_count > 1 << 64:
            raise ValueError("arena backend-index range exceeds uint64")


@dataclass(frozen=True, slots=True)
class BackendPageAddress:
    page: PageLease
    class_id: int
    backend_domain: int
    backend_index: int

    def __post_init__(self) -> None:
        _u16("class_id", self.class_id)
        _u16("backend_domain", self.backend_domain)
        _u64("backend_index", self.backend_index)


@dataclass(frozen=True, slots=True)
class BackendTokenAddress:
    """Exact manager lease plus its engine-facing page/token coordinate."""

    page: PageLease
    class_id: int
    backend_domain: int
    backend_index: int
    token_offset: int

    def __post_init__(self) -> None:
        _u16("class_id", self.class_id)
        _u16("backend_domain", self.backend_domain)
        _u64("backend_index", self.backend_index)
        _u32("token_offset", self.token_offset)

    @property
    def page_address(self) -> BackendPageAddress:
        return BackendPageAddress(
            self.page, self.class_id, self.backend_domain, self.backend_index
        )


@dataclass(frozen=True, slots=True)
class ResolvedPageAddress:
    address: BackendPageAddress
    arena: object
    arena_page_index: int
    byte_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.address, BackendPageAddress):
            raise TypeError("address must be a BackendPageAddress")
        _u32("arena_page_index", self.arena_page_index)
        _u64("byte_length", self.byte_length, positive=True)


@dataclass(frozen=True, slots=True)
class ResolvedTokenAddress:
    address: BackendTokenAddress
    arena: object
    arena_page_index: int
    token_index: int
    byte_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.address, BackendTokenAddress):
            raise TypeError("address must be a BackendTokenAddress")
        _u32("arena_page_index", self.arena_page_index)
        _u64("token_index", self.token_index)
        _u64("byte_length", self.byte_length, positive=True)


@dataclass(frozen=True, slots=True)
class TokenCopy:
    context: OperationContext
    token_id: int
    source: BackendTokenAddress
    destination: BackendTokenAddress

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise TypeError("context must be an OperationContext")
        if not isinstance(self.source, BackendTokenAddress) or not isinstance(
            self.destination, BackendTokenAddress
        ):
            raise TypeError("copy addresses must be BackendTokenAddress values")
        _u64("token_id", self.token_id)


@dataclass(frozen=True, slots=True)
class TokenWrite:
    context: OperationContext
    token_id: int
    destination: BackendTokenAddress
    payload: BytesLike

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise TypeError("context must be an OperationContext")
        if not isinstance(self.destination, BackendTokenAddress):
            raise TypeError("destination must be a BackendTokenAddress")
        _u64("token_id", self.token_id)
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")


@dataclass(frozen=True, slots=True)
class TokenWriteReceipt:
    context: OperationContext
    token_id: int
    destination: BackendTokenAddress
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise TypeError("context must be an OperationContext")
        if not isinstance(self.destination, BackendTokenAddress):
            raise TypeError("destination must be a BackendTokenAddress")
        _u64("token_id", self.token_id)
        _u64("byte_count", self.byte_count, positive=True)


@dataclass(frozen=True, slots=True)
class ExternalTokenWrite:
    """One exact destination an engine kernel is authorized to write."""

    context: OperationContext
    token_id: int
    destination: BackendTokenAddress
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise TypeError("context must be an OperationContext")
        if self.context.operation is not DataPlaneOperation.APPEND:
            raise ValueError("external writes require an append context")
        if not isinstance(self.destination, BackendTokenAddress):
            raise TypeError("destination must be a BackendTokenAddress")
        _u64("token_id", self.token_id)
        _u64("byte_count", self.byte_count, positive=True)


@dataclass(frozen=True, slots=True)
class ExternalAppendTicket:
    """Adapter-issued authorization for engine-owned append mutation.

    The adapter must additionally reject tickets it did not issue.  Carrying
    its identity, ticket id, original writes, and exact resolved destinations
    gives implementations all information needed to perform that check.
    """

    adapter_id: str
    ticket_id: int
    writes: tuple[ExternalTokenWrite, ...]
    resolved_destinations: tuple[ResolvedTokenAddress, ...]
    completion_domain: int = 1
    launch_context: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str):
            raise TypeError("adapter_id must be a string")
        if not self.adapter_id:
            raise ValueError("adapter_id must not be empty")
        _u64("ticket_id", self.ticket_id, positive=True)
        _u64("completion_domain", self.completion_domain, positive=True)
        if not isinstance(self.writes, tuple) or any(
            not isinstance(item, ExternalTokenWrite) for item in self.writes
        ):
            raise TypeError("writes must be ExternalTokenWrite values")
        if not isinstance(self.resolved_destinations, tuple) or any(
            not isinstance(item, ResolvedTokenAddress)
            for item in self.resolved_destinations
        ):
            raise TypeError(
                "resolved_destinations must be ResolvedTokenAddress values"
            )
        if not self.writes:
            raise ValueError("writes must not be empty")
        if len(self.writes) != len(self.resolved_destinations):
            raise ValueError(
                "writes and resolved_destinations must have matching lengths"
            )
        for write, resolved in zip(self.writes, self.resolved_destinations):
            if write.context.operation is not DataPlaneOperation.APPEND:
                raise ValueError("external writes require append contexts")
            if write.destination != resolved.address:
                raise ValueError("resolved destination does not match write")
            if write.byte_count != resolved.byte_length:
                raise ValueError("resolved byte length does not match write")


@dataclass(frozen=True, slots=True)
class CompletionFence:
    """Adapter-issued point on one totally ordered completion domain."""

    adapter_id: str
    engine_epoch: int
    completion_domain: int
    completion_value: int
    fence_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            raise ValueError("adapter_id must not be empty")
        for name in (
            "engine_epoch",
            "completion_domain",
            "completion_value",
            "fence_id",
        ):
            _u64(name, getattr(self, name), positive=True)


@dataclass(frozen=True, slots=True)
class CompletionEvidence:
    """Proof that a fence was confirmed by the adapter that issued it."""

    fence: CompletionFence
    covered_pages: tuple[BackendPageAddress, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fence, CompletionFence):
            raise TypeError("fence must be a CompletionFence")
        if not isinstance(self.covered_pages, tuple) or any(
            not isinstance(item, BackendPageAddress) for item in self.covered_pages
        ):
            raise TypeError("covered_pages must be BackendPageAddress values")


@dataclass(frozen=True, slots=True)
class DataPlaneEvidence:
    operation_id: int
    operation: DataPlaneOperation
    copies: tuple[TokenCopy, ...]
    writes: tuple[TokenWriteReceipt, ...]
    completion: CompletionFence

    def __post_init__(self) -> None:
        _u64("operation_id", self.operation_id, positive=True)
        if not isinstance(self.operation, DataPlaneOperation):
            raise TypeError("operation must be a DataPlaneOperation")
        if not isinstance(self.copies, tuple) or any(
            not isinstance(item, TokenCopy) for item in self.copies
        ):
            raise TypeError("copies must be TokenCopy values")
        if not isinstance(self.writes, tuple) or any(
            not isinstance(item, TokenWriteReceipt) for item in self.writes
        ):
            raise TypeError("writes must be TokenWriteReceipt values")
        if not isinstance(self.completion, CompletionFence):
            raise TypeError("completion must be a CompletionFence")


@dataclass(frozen=True, slots=True)
class PageMirrorKey:
    request_key: Hashable
    class_id: int
    logical_ordinal: int

    def __post_init__(self) -> None:
        hash(self.request_key)
        _u16("class_id", self.class_id)
        _u64("logical_ordinal", self.logical_ordinal)


@dataclass(frozen=True, slots=True)
class TokenMirrorKey:
    request_key: Hashable
    class_id: int
    token_id: int

    def __post_init__(self) -> None:
        hash(self.request_key)
        _u16("class_id", self.class_id)
        _u64("token_id", self.token_id)


@dataclass(frozen=True, slots=True)
class PageMirrorUpdate:
    key: PageMirrorKey
    expected: BackendPageAddress | None
    replacement: BackendPageAddress | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, PageMirrorKey):
            raise TypeError("key must be a PageMirrorKey")
        if self.expected is not None and not isinstance(
            self.expected, BackendPageAddress
        ):
            raise TypeError("expected must be a BackendPageAddress or None")
        if self.replacement is not None and not isinstance(
            self.replacement, BackendPageAddress
        ):
            raise TypeError("replacement must be a BackendPageAddress or None")


@dataclass(frozen=True, slots=True)
class TokenMirrorUpdate:
    key: TokenMirrorKey
    expected: BackendTokenAddress | None
    replacement: BackendTokenAddress | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, TokenMirrorKey):
            raise TypeError("key must be a TokenMirrorKey")
        if self.expected is not None and not isinstance(
            self.expected, BackendTokenAddress
        ):
            raise TypeError("expected must be a BackendTokenAddress or None")
        if self.replacement is not None and not isinstance(
            self.replacement, BackendTokenAddress
        ):
            raise TypeError("replacement must be a BackendTokenAddress or None")


@dataclass(frozen=True, slots=True)
class MirrorEvidence:
    operation_id: int
    page_updates: tuple[PageMirrorUpdate, ...]
    token_updates: tuple[TokenMirrorUpdate, ...]
    ordered_after: CompletionEvidence
    completion: CompletionFence

    def __post_init__(self) -> None:
        _u64("operation_id", self.operation_id, positive=True)
        if not isinstance(self.page_updates, tuple) or any(
            not isinstance(item, PageMirrorUpdate) for item in self.page_updates
        ):
            raise TypeError("page_updates must be PageMirrorUpdate values")
        if not isinstance(self.token_updates, tuple) or any(
            not isinstance(item, TokenMirrorUpdate) for item in self.token_updates
        ):
            raise TypeError("token_updates must be TokenMirrorUpdate values")
        if not isinstance(self.ordered_after, CompletionEvidence):
            raise TypeError("ordered_after must be CompletionEvidence")
        if not isinstance(self.completion, CompletionFence):
            raise TypeError("completion must be a CompletionFence")


@dataclass(frozen=True, slots=True)
class ReclamationCertificate:
    """Exact manager certificate; the adapter may verify but not mint it."""

    reclamation: ReclamationLease
    page: PageLease
    class_id: int
    backend_domain: int
    logical_ordinal: int
    backend_index: int
    token_begin: int
    token_end_exclusive: int
    completion_domain: int
    completion_value: int

    def __post_init__(self) -> None:
        if not isinstance(self.reclamation, ReclamationLease):
            raise TypeError("reclamation must be a ReclamationLease")
        if not isinstance(self.page, PageLease):
            raise TypeError("page must be a PageLease")
        if self.reclamation.engine_epoch != self.page.engine_epoch:
            raise ValueError("reclamation and page engine epochs differ")
        _u16("class_id", self.class_id)
        _u16("backend_domain", self.backend_domain)
        for name in (
            "logical_ordinal",
            "backend_index",
            "token_begin",
            "token_end_exclusive",
            "completion_domain",
            "completion_value",
        ):
            _u64(
                name,
                getattr(self, name),
                positive=name in (
                    "token_end_exclusive",
                    "completion_domain",
                    "completion_value",
                ),
            )
        if self.token_end_exclusive <= self.token_begin:
            raise ValueError("retirement token span must not be empty")

    @property
    def page_address(self) -> BackendPageAddress:
        return BackendPageAddress(
            self.page, self.class_id, self.backend_domain, self.backend_index
        )


@dataclass(frozen=True, slots=True)
class ReuseEvidence:
    adapter_id: str
    evidence_id: int
    certificates: tuple[ReclamationCertificate, ...]
    last_use: tuple[CompletionEvidence, ...]
    mirror_cleanup: CompletionEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            raise ValueError("adapter_id must not be empty")
        _u64("evidence_id", self.evidence_id, positive=True)
        if not self.certificates:
            raise ValueError("certificates must not be empty")
        if not self.last_use:
            raise ValueError("last_use must not be empty")
        if not isinstance(self.certificates, tuple) or any(
            not isinstance(item, ReclamationCertificate)
            for item in self.certificates
        ):
            raise TypeError("certificates must be ReclamationCertificate values")
        if not isinstance(self.last_use, tuple) or any(
            not isinstance(item, CompletionEvidence) for item in self.last_use
        ):
            raise TypeError("last_use must be CompletionEvidence values")
        if not isinstance(self.mirror_cleanup, CompletionEvidence):
            raise TypeError("mirror_cleanup must be CompletionEvidence")


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    exact_append: bool = True
    exact_cow: bool = True
    exact_token_relocation: bool = True
    page_mirrors: bool = True
    token_mirrors: bool = True
    completion_fences: bool = True
    ack_gated_reuse: bool = True
    cpu_arenas: bool = True
    cuda_arenas: bool = False
    external_kernel_writes: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@runtime_checkable
class KvDataPlaneAdapter(Protocol):
    """Batch-first effect boundary implemented by an engine integration."""

    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def resolve_pages(
        self, addresses: Sequence[BackendPageAddress]
    ) -> tuple[ResolvedPageAddress, ...]: ...

    def resolve_tokens(
        self, addresses: Sequence[BackendTokenAddress]
    ) -> tuple[ResolvedTokenAddress, ...]: ...

    def append(
        self,
        writes: Sequence[TokenWrite],
        *,
        cow_copies: Sequence[TokenCopy] = (),
        completion_domain: int = 1,
    ) -> DataPlaneEvidence: ...

    def relocate(
        self,
        moves: Sequence[TokenCopy],
        *,
        completion_domain: int = 1,
    ) -> DataPlaneEvidence: ...

    def query_completion(
        self, fence: CompletionFence
    ) -> CompletionEvidence | None: ...

    def wait_completion(self, fence: CompletionFence) -> CompletionEvidence: ...

    def record_completion(
        self,
        pages: Sequence[BackendPageAddress],
        *,
        completion_domain: int = 1,
    ) -> CompletionFence: ...

    def update_mirrors(
        self,
        page_updates: Sequence[PageMirrorUpdate],
        token_updates: Sequence[TokenMirrorUpdate],
        *,
        after: CompletionEvidence,
        completion_domain: int = 1,
    ) -> MirrorEvidence: ...

    def prepare_reuse(
        self,
        certificates: Sequence[ReclamationCertificate],
        *,
        last_use: Sequence[CompletionEvidence],
        mirror_cleanup: CompletionEvidence,
    ) -> ReuseEvidence: ...

    def note_reuse_acknowledged(self, evidence: ReuseEvidence) -> None: ...

    def poison(self, reason: str) -> None: ...


@runtime_checkable
class ExternalWriteCompletionAdapter(Protocol):
    """Optional extension for writes performed by an engine kernel."""

    def prepare_external_append(
        self,
        writes: Sequence[ExternalTokenWrite],
        *,
        completion_domain: int = 1,
    ) -> ExternalAppendTicket: ...

    def record_external_data_ready(
        self,
        ticket: ExternalAppendTicket,
    ) -> DataPlaneEvidence: ...

    def record_last_use(
        self,
        pages: Sequence[BackendPageAddress],
        *,
        completion_domain: int = 1,
    ) -> CompletionFence: ...


__all__ = [
    "AdapterCapabilities",
    "ArenaRegistration",
    "BackendPageAddress",
    "BackendTokenAddress",
    "BytesLike",
    "CompletionEvidence",
    "CompletionFence",
    "DataPlaneOperation",
    "DataPlaneEvidence",
    "ExternalAppendTicket",
    "ExternalTokenWrite",
    "ExternalWriteCompletionAdapter",
    "KvDataPlaneAdapter",
    "MirrorEvidence",
    "PageLease",
    "PageMirrorKey",
    "PageMirrorUpdate",
    "OperationContext",
    "ReclamationCertificate",
    "ReclamationLease",
    "RelocationLease",
    "RequestLease",
    "ResolvedPageAddress",
    "ResolvedTokenAddress",
    "ReuseEvidence",
    "StepLease",
    "TokenCopy",
    "TokenMirrorKey",
    "TokenMirrorUpdate",
    "TokenWrite",
    "TokenWriteReceipt",
]
