from __future__ import annotations

import ctypes
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Sequence

from .runtime import (
    ArenaIdentity,
    ArenaRegistration,
    ArenaStats,
    BackendBindReceipt,
    BackendUnobservedReceipt,
    CompletionReceipt,
    DeviceKvView,
    DeviceViewEntry,
    DeviceViewHeader,
    ManagerCreateSettings,
    ManagerError,
    ManagerFactoryProtocol,
    ManagerProtocol,
    ManagerStats,
    PageLease,
    PreparedStep,
    ReclamationCertificate,
    ReclamationLease,
    ReclamationReceipt,
    ReleaseCompletion,
    RequestLease,
    StepCompletion,
    StepLease,
    SubmissionLease,
    SubmittedStep,
)


ABI_VERSION = 4
STATUS_OK = 0
STATUS_BUFFER_TOO_SMALL = 1
ERROR_BUFFER_BYTES = 2048


class CanonicalAbiUnavailable(RuntimeError):
    """The configured library is not the one canonical OrbitKV ABI."""


class _RequestLease(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("slot", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
    ]


class _StepLease(ctypes.Structure):
    _fields_ = _RequestLease._fields_


class _SubmissionLease(ctypes.Structure):
    _fields_ = _RequestLease._fields_


class _ReclamationLease(ctypes.Structure):
    _fields_ = _RequestLease._fields_


class _PageLease(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("pool_epoch", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("page_id", ctypes.c_uint32),
        ("pool_id", ctypes.c_uint32),
    ]


class _DeviceViewHeader(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("engine_epoch", ctypes.c_uint64),
        ("request_slot", ctypes.c_uint32),
        ("request_generation", ctypes.c_uint32),
        ("view_version", ctypes.c_uint64),
        ("base_frontier", ctypes.c_uint64),
        ("target_frontier", ctypes.c_uint64),
        ("page_tokens", ctypes.c_uint32),
        ("entry_count", ctypes.c_uint32),
    ]


class _DeviceViewEntry(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_uint16),
        ("backend_domain", ctypes.c_uint16),
        ("access_flags", ctypes.c_uint32),
        ("logical_ordinal", ctypes.c_uint64),
        ("token_begin", ctypes.c_uint64),
        ("valid_token_count", ctypes.c_uint32),
        ("visible_token_offset", ctypes.c_uint32),
        ("visible_token_count", ctypes.c_uint32),
        ("pool_id", ctypes.c_uint32),
        ("temporal_cell_index", ctypes.c_uint64),
        ("temporal_cycle", ctypes.c_uint64),
        ("pool_epoch", ctypes.c_uint64),
        ("page_generation", ctypes.c_uint64),
        ("backend_index", ctypes.c_uint64),
        ("page_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _BackendArenaRegistration(ctypes.Structure):
    _fields_ = [
        ("pool_id", ctypes.c_uint32),
        ("class_id", ctypes.c_uint16),
        ("backend_domain", ctypes.c_uint16),
        ("page_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("backend_base_index", ctypes.c_uint64),
    ]


class _ManagerConfig(ctypes.Structure):
    _fields_ = [
        ("maximum_requests", ctypes.c_uint32),
        ("maximum_operations", ctypes.c_uint32),
        ("maximum_reclamations", ctypes.c_uint32),
        ("maximum_step_tokens", ctypes.c_uint32),
    ]


class _ArenaIdentity(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("pool_epoch", ctypes.c_uint64),
        ("backend_base_index", ctypes.c_uint64),
        ("pool_id", ctypes.c_uint32),
        ("page_count", ctypes.c_uint32),
        ("page_tokens", ctypes.c_uint32),
        ("class_id", ctypes.c_uint16),
        ("backend_domain", ctypes.c_uint16),
        ("first_page_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _ArenaStats(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("pool_epoch", ctypes.c_uint64),
        ("pool_id", ctypes.c_uint32),
        ("page_count", ctypes.c_uint32),
        ("class_id", ctypes.c_uint16),
        ("backend_domain", ctypes.c_uint16),
        ("first_page_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("reserved_padding", ctypes.c_uint32),
        ("free_pages", ctypes.c_uint64),
        ("reserved_pages", ctypes.c_uint64),
        ("writing_pages", ctypes.c_uint64),
        ("active_pages", ctypes.c_uint64),
        ("retiring_pages", ctypes.c_uint64),
        ("quarantined_pages", ctypes.c_uint64),
        ("exhausted_pages", ctypes.c_uint64),
    ]


class _PreparedStep(ctypes.Structure):
    _fields_ = [
        ("step", _StepLease),
        ("request", _RequestLease),
        ("base_view_version", ctypes.c_uint64),
        ("target_view_version", ctypes.c_uint64),
        ("previous_boundary", ctypes.c_uint64),
        ("target_boundary", ctypes.c_uint64),
        ("view", _DeviceViewHeader),
    ]


class _BackendBindReceipt(ctypes.Structure):
    _fields_ = [
        ("step", _StepLease),
        ("page", _PageLease),
        ("backend_domain", ctypes.c_uint16),
        ("mapped", ctypes.c_uint8),
        ("writable", ctypes.c_uint8),
        ("reserved", ctypes.c_uint32),
        ("backend_index", ctypes.c_uint64),
    ]


class _SubmittedStep(ctypes.Structure):
    _fields_ = [
        ("submission", _SubmissionLease),
        ("request", _RequestLease),
        ("view", _DeviceViewHeader),
    ]


class _CompletionReceipt(ctypes.Structure):
    _fields_ = [
        ("submission", _SubmissionLease),
        ("completion_domain", ctypes.c_uint64),
        ("completion_value", ctypes.c_uint64),
        ("confirmed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _BackendUnobservedReceipt(ctypes.Structure):
    _fields_ = [
        ("step", _StepLease),
        ("backend_unobserved", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _ReclamationCertificate(ctypes.Structure):
    _fields_ = [
        ("reclamation", _ReclamationLease),
        ("request", _RequestLease),
        ("page", _PageLease),
        ("class_id", ctypes.c_uint16),
        ("backend_domain", ctypes.c_uint16),
        ("logical_ordinal", ctypes.c_uint64),
        ("backend_index", ctypes.c_uint64),
        ("token_begin", ctypes.c_uint64),
        ("token_end_exclusive", ctypes.c_uint64),
        ("completion_domain", ctypes.c_uint64),
        ("completion_value", ctypes.c_uint64),
    ]


class _ReclamationReceipt(ctypes.Structure):
    _fields_ = [
        ("reclamation", _ReclamationLease),
        ("page", _PageLease),
        ("backend_domain", ctypes.c_uint16),
        ("acknowledged", ctypes.c_uint8),
        ("reserved8", ctypes.c_uint8),
        ("reserved32", ctypes.c_uint32),
        ("backend_index", ctypes.c_uint64),
    ]


class _StepCompletion(ctypes.Structure):
    _fields_ = [
        ("submission", _SubmissionLease),
        ("request", _RequestLease),
        ("published_view", _DeviceViewHeader),
    ]


class _ReleaseCompletion(ctypes.Structure):
    _fields_ = [("request", _RequestLease)]


class _ManagerStats(ctypes.Structure):
    _fields_ = [
        ("active_requests", ctypes.c_uint64),
        ("prepared_steps", ctypes.c_uint64),
        ("submitted_steps", ctypes.c_uint64),
        ("free_pages", ctypes.c_uint64),
        ("reserved_pages", ctypes.c_uint64),
        ("writing_pages", ctypes.c_uint64),
        ("active_pages", ctypes.c_uint64),
        ("retiring_pages", ctypes.c_uint64),
        ("quarantined_pages", ctypes.c_uint64),
        ("exhausted_pages", ctypes.c_uint64),
        ("pending_reclamations", ctypes.c_uint64),
    ]


_EXPECTED_SIZES = {
    _RequestLease: 16,
    _StepLease: 16,
    _SubmissionLease: 16,
    _ReclamationLease: 16,
    _PageLease: 32,
    _DeviceViewHeader: 56,
    _DeviceViewEntry: 88,
    _BackendArenaRegistration: 24,
    _ManagerConfig: 16,
    _ArenaIdentity: 48,
    _ArenaStats: 96,
    _PreparedStep: 120,
    _BackendBindReceipt: 64,
    _SubmittedStep: 88,
    _CompletionReceipt: 40,
    _BackendUnobservedReceipt: 24,
    _ReclamationCertificate: 120,
    _ReclamationReceipt: 64,
    _StepCompletion: 88,
    _ReleaseCompletion: 16,
    _ManagerStats: 88,
}
for _structure, _expected_size in _EXPECTED_SIZES.items():
    if ctypes.sizeof(_structure) != _expected_size:
        raise CanonicalAbiUnavailable(
            f"ctypes layout mismatch for {_structure.__name__}: "
            f"{ctypes.sizeof(_structure)} != {_expected_size}"
        )


def _bounded(name: str, value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManagerError(f"{name} must be an integer")
    if value < 0 or value >= 1 << bits:
        raise ManagerError(f"{name} does not fit uint{bits}")
    return value


def _u8(name: str, value: int) -> int:
    return _bounded(name, value, 8)


def _u16(name: str, value: int) -> int:
    return _bounded(name, value, 16)


def _u32(name: str, value: int) -> int:
    return _bounded(name, value, 32)


def _u64(name: str, value: int) -> int:
    return _bounded(name, value, 64)


def _zero(name: str, value: int) -> int:
    if value != 0:
        raise ManagerError(f"{name} must be zero")
    return 0


def _request_to_c(value: RequestLease) -> _RequestLease:
    return _RequestLease(
        _u64("request.engine_epoch", value.engine_epoch),
        _u32("request.slot", value.slot),
        _u32("request.generation", value.generation),
    )


def _request_from_c(value: _RequestLease) -> RequestLease:
    return RequestLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _step_to_c(value: StepLease) -> _StepLease:
    return _StepLease(
        _u64("step.engine_epoch", value.engine_epoch),
        _u32("step.slot", value.slot),
        _u32("step.generation", value.generation),
    )


def _step_from_c(value: _StepLease) -> StepLease:
    return StepLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _submission_to_c(value: SubmissionLease) -> _SubmissionLease:
    return _SubmissionLease(
        _u64("submission.engine_epoch", value.engine_epoch),
        _u32("submission.slot", value.slot),
        _u32("submission.generation", value.generation),
    )


def _submission_from_c(value: _SubmissionLease) -> SubmissionLease:
    return SubmissionLease(
        int(value.engine_epoch), int(value.slot), int(value.generation)
    )


def _reclamation_to_c(value: ReclamationLease) -> _ReclamationLease:
    return _ReclamationLease(
        _u64("reclamation.engine_epoch", value.engine_epoch),
        _u32("reclamation.slot", value.slot),
        _u32("reclamation.generation", value.generation),
    )


def _reclamation_from_c(value: _ReclamationLease) -> ReclamationLease:
    return ReclamationLease(
        int(value.engine_epoch), int(value.slot), int(value.generation)
    )


def _page_to_c(value: PageLease) -> _PageLease:
    return _PageLease(
        _u64("page.engine_epoch", value.engine_epoch),
        _u64("page.pool_epoch", value.pool_epoch),
        _u64("page.generation", value.generation),
        _u32("page.page_id", value.page_id),
        _u32("page.pool_id", value.pool_id),
    )


def _page_from_c(value: _PageLease) -> PageLease:
    return PageLease(
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.page_id),
        int(value.pool_id),
    )


def _header_from_c(value: _DeviceViewHeader) -> DeviceViewHeader:
    return DeviceViewHeader(
        abi_version=int(value.abi_version),
        flags=int(value.flags),
        engine_epoch=int(value.engine_epoch),
        request_slot=int(value.request_slot),
        request_generation=int(value.request_generation),
        view_version=int(value.view_version),
        base_frontier=int(value.base_frontier),
        target_frontier=int(value.target_frontier),
        page_tokens=int(value.page_tokens),
        entry_count=int(value.entry_count),
    )


def _entry_from_c(value: _DeviceViewEntry) -> DeviceViewEntry:
    _zero("device view entry reserved", int(value.reserved))
    return DeviceViewEntry(
        class_id=int(value.class_id),
        backend_domain=int(value.backend_domain),
        access_flags=int(value.access_flags),
        logical_ordinal=int(value.logical_ordinal),
        token_begin=int(value.token_begin),
        valid_token_count=int(value.valid_token_count),
        visible_token_offset=int(value.visible_token_offset),
        visible_token_count=int(value.visible_token_count),
        pool_id=int(value.pool_id),
        temporal_cell_index=int(value.temporal_cell_index),
        temporal_cycle=int(value.temporal_cycle),
        pool_epoch=int(value.pool_epoch),
        page_generation=int(value.page_generation),
        backend_index=int(value.backend_index),
        page_id=int(value.page_id),
        reserved=0,
    )


def _certificate_from_c(value: _ReclamationCertificate) -> ReclamationCertificate:
    return ReclamationCertificate(
        reclamation=_reclamation_from_c(value.reclamation),
        request=_request_from_c(value.request),
        page=_page_from_c(value.page),
        class_id=int(value.class_id),
        backend_domain=int(value.backend_domain),
        logical_ordinal=int(value.logical_ordinal),
        backend_index=int(value.backend_index),
        token_begin=int(value.token_begin),
        token_end_exclusive=int(value.token_end_exclusive),
        completion_domain=int(value.completion_domain),
        completion_value=int(value.completion_value),
    )


def _arena_identity_from_c(value: _ArenaIdentity) -> ArenaIdentity:
    _zero("arena identity reserved", int(value.reserved))
    return ArenaIdentity(
        engine_epoch=int(value.engine_epoch),
        pool_epoch=int(value.pool_epoch),
        pool_id=int(value.pool_id),
        class_id=int(value.class_id),
        backend_domain=int(value.backend_domain),
        page_count=int(value.page_count),
        page_tokens=int(value.page_tokens),
        backend_base_index=int(value.backend_base_index),
        first_page_id=int(value.first_page_id),
    )


def _arena_stats_from_c(value: _ArenaStats) -> ArenaStats:
    _zero("arena stats reserved", int(value.reserved))
    _zero("arena stats reserved padding", int(value.reserved_padding))
    return ArenaStats(
        engine_epoch=int(value.engine_epoch),
        pool_epoch=int(value.pool_epoch),
        pool_id=int(value.pool_id),
        page_count=int(value.page_count),
        class_id=int(value.class_id),
        backend_domain=int(value.backend_domain),
        first_page_id=int(value.first_page_id),
        free_pages=int(value.free_pages),
        reserved_pages=int(value.reserved_pages),
        writing_pages=int(value.writing_pages),
        active_pages=int(value.active_pages),
        retiring_pages=int(value.retiring_pages),
        quarantined_pages=int(value.quarantined_pages),
        exhausted_pages=int(value.exhausted_pages),
    )


def _registration_to_c(value: ArenaRegistration) -> _BackendArenaRegistration:
    page_count = _u32("arena.page_count", value.page_count)
    if page_count == 0:
        raise ManagerError("arena.page_count must be positive")
    return _BackendArenaRegistration(
        _u32("arena.pool_id", value.pool_id),
        _u16("arena.class_id", value.class_id),
        _u16("arena.backend_domain", value.backend_domain),
        page_count,
        0,
        _u64("arena.backend_base_index", value.backend_base_index),
    )


def _bind_to_c(value: BackendBindReceipt) -> _BackendBindReceipt:
    _zero("binding receipt reserved", value.reserved)
    if value.mapped != 1 or value.writable != 1:
        raise ManagerError("binding receipt must prove mapped and writable")
    return _BackendBindReceipt(
        _step_to_c(value.step),
        _page_to_c(value.page),
        _u16("binding backend_domain", value.backend_domain),
        _u8("binding mapped", value.mapped),
        _u8("binding writable", value.writable),
        0,
        _u64("binding backend_index", value.backend_index),
    )


def _reclamation_receipt_to_c(value: ReclamationReceipt) -> _ReclamationReceipt:
    _zero("reclamation receipt reserved8", value.reserved8)
    _zero("reclamation receipt reserved32", value.reserved32)
    if value.acknowledged != 1:
        raise ManagerError("reclamation receipt must be acknowledged")
    return _ReclamationReceipt(
        _reclamation_to_c(value.reclamation),
        _page_to_c(value.page),
        _u16("reclamation backend_domain", value.backend_domain),
        _u8("reclamation acknowledged", value.acknowledged),
        0,
        0,
        _u64("reclamation backend_index", value.backend_index),
    )


def _configure_library(library: Any) -> None:
    handle = ctypes.c_void_p
    error = ctypes.POINTER(ctypes.c_char)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    signatures: dict[str, tuple[list[Any], Any]] = {
        "orbitkv_manager_create": (
            [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.POINTER(_ManagerConfig),
                ctypes.POINTER(_BackendArenaRegistration),
                ctypes.c_uint32,
                ctypes.POINTER(handle),
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_arena_identities": (
            [
                handle,
                ctypes.POINTER(_ArenaIdentity),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_arena_stats": (
            [
                handle,
                ctypes.POINTER(_ArenaStats),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_request_acquire": (
            [handle, ctypes.POINTER(_RequestLease), error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_current_view": (
            [handle, _RequestLease, ctypes.POINTER(_DeviceViewHeader), ctypes.POINTER(_DeviceViewEntry), ctypes.c_uint32, u32p, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_prepare_step": (
            [handle, _RequestLease, ctypes.c_uint64, ctypes.POINTER(_PreparedStep), ctypes.POINTER(_DeviceViewEntry), ctypes.c_uint32, u32p, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_submit_step": (
            [handle, _StepLease, ctypes.POINTER(_BackendBindReceipt), ctypes.c_uint32, ctypes.POINTER(_SubmittedStep), ctypes.POINTER(_DeviceViewEntry), ctypes.c_uint32, u32p, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_complete_step": (
            [handle, _CompletionReceipt, ctypes.POINTER(_StepCompletion), ctypes.POINTER(_DeviceViewEntry), ctypes.c_uint32, u32p, ctypes.POINTER(_ReclamationCertificate), ctypes.c_uint32, u32p, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_abort_step": (
            [handle, _BackendUnobservedReceipt, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_quarantine_step": (
            [handle, _StepLease, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_quarantine_submission": (
            [handle, _SubmissionLease, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_release_request": (
            [handle, _RequestLease, ctypes.POINTER(_ReleaseCompletion), ctypes.POINTER(_ReclamationCertificate), ctypes.c_uint32, u32p, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_acknowledge_reclamations": (
            [handle, ctypes.POINTER(_ReclamationReceipt), ctypes.c_uint32, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_recycle_request": (
            [handle, _RequestLease, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_stats": (
            [handle, ctypes.POINTER(_ManagerStats), error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_destroy": (
            [handle, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
    }
    try:
        library.orbitkv_abi_version.argtypes = []
        library.orbitkv_abi_version.restype = ctypes.c_uint32
        version = int(library.orbitkv_abi_version())
        if version != ABI_VERSION:
            raise CanonicalAbiUnavailable(
                f"OrbitKV ABI version {version} is not required version {ABI_VERSION}"
            )
        for name, (argtypes, restype) in signatures.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype
    except AttributeError as error_value:
        raise CanonicalAbiUnavailable(
            f"canonical OrbitKV C ABI symbol is missing: {error_value}"
        ) from error_value


class _CtypesManager(ManagerProtocol):
    def __init__(
        self,
        library: Any,
        handle: ctypes.c_void_p,
        registrations: Sequence[ArenaRegistration],
        page_tokens: int,
    ):
        self._library = library
        self._handle = handle
        self._arena_count = len(registrations)
        self._capacity = sum(item.page_count for item in registrations)
        self._lock = RLock()
        if self._arena_count <= 0 or self._capacity <= 0:
            raise ManagerError("ABI4 manager requires at least one nonempty arena")
        identity_values = (_ArenaIdentity * self._arena_count)()
        identity_count = ctypes.c_uint32()
        self._checked_call(
            "arena identities",
            self._library.orbitkv_manager_arena_identities,
            self._handle,
            identity_values,
            self._arena_count,
            ctypes.byref(identity_count),
        )
        if int(identity_count.value) != self._arena_count:
            raise ManagerError("manager returned the wrong arena-identity count")
        self._arenas = tuple(
            _arena_identity_from_c(identity_values[index])
            for index in range(self._arena_count)
        )
        class_ids = tuple(item.class_id for item in self._arenas)
        if class_ids != tuple(range(self._arena_count)):
            raise ManagerError("manager arena identities are not class-id ordered")
        if len({item.pool_id for item in self._arenas}) != self._arena_count:
            raise ManagerError("manager returned duplicate arena pool identities")
        if len({item.engine_epoch for item in self._arenas}) != 1:
            raise ManagerError("manager arenas disagree on engine epoch")
        if self._arenas[0].engine_epoch <= 0 or any(
            item.pool_epoch <= 0 for item in self._arenas
        ):
            raise ManagerError("manager returned a zero arena epoch")
        page_ranges: list[tuple[int, int]] = []
        backend_ranges: dict[int, list[tuple[int, int]]] = {}
        for identity, registration in zip(
            self._arenas, registrations, strict=True
        ):
            if (
                identity.class_id != registration.class_id
                or identity.pool_id != registration.pool_id
                or identity.backend_domain != registration.backend_domain
                or identity.page_count != registration.page_count
                or identity.backend_base_index != registration.backend_base_index
                or identity.page_tokens != page_tokens
            ):
                raise ManagerError(
                    "manager returned a different registered arena identity"
                )
            if identity.first_page_id <= 0:
                raise ManagerError("manager returned a zero arena page id")
            page_end = identity.first_page_id + identity.page_count
            if page_end > 1 << 32 or any(
                identity.first_page_id < other_end
                and other_start < page_end
                for other_start, other_end in page_ranges
            ):
                raise ManagerError("manager arena page-id ranges overlap or overflow")
            page_ranges.append((identity.first_page_id, page_end))

            backend_end = identity.backend_base_index + identity.page_count
            ranges = backend_ranges.setdefault(identity.backend_domain, [])
            if backend_end > 1 << 64 or any(
                identity.backend_base_index < other_end
                and other_start < backend_end
                for other_start, other_end in ranges
            ):
                raise ManagerError(
                    "manager arena backend ranges overlap or overflow within a domain"
                )
            ranges.append((identity.backend_base_index, backend_end))
        self._arenas_by_class = {item.class_id: item for item in self._arenas}

    @property
    def arenas(self) -> tuple[ArenaIdentity, ...]:
        return self._arenas

    @property
    def arenas_by_class(self) -> dict[int, ArenaIdentity]:
        return dict(self._arenas_by_class)

    def arena_stats(self) -> tuple[ArenaStats, ...]:
        with self._lock:
            values = (_ArenaStats * self._arena_count)()
            count = ctypes.c_uint32()
            self._checked_call(
                "arena stats",
                self._library.orbitkv_manager_arena_stats,
                self._require_open(),
                values,
                self._arena_count,
                ctypes.byref(count),
            )
            if int(count.value) != self._arena_count:
                raise ManagerError("manager returned the wrong arena-stats count")
            result = tuple(
                _arena_stats_from_c(values[index])
                for index in range(self._arena_count)
            )
            if tuple(item.class_id for item in result) != tuple(
                range(self._arena_count)
            ):
                raise ManagerError("manager arena stats are not class-id ordered")
            for stats, identity in zip(result, self._arenas, strict=True):
                if (
                    stats.engine_epoch != identity.engine_epoch
                    or stats.pool_epoch != identity.pool_epoch
                    or stats.pool_id != identity.pool_id
                    or stats.page_count != identity.page_count
                    or stats.class_id != identity.class_id
                    or stats.backend_domain != identity.backend_domain
                    or stats.first_page_id != identity.first_page_id
                ):
                    raise ManagerError("manager arena stats changed arena identity")
                page_census = sum(
                    (
                        stats.free_pages,
                        stats.reserved_pages,
                        stats.writing_pages,
                        stats.active_pages,
                        stats.retiring_pages,
                        stats.quarantined_pages,
                        stats.exhausted_pages,
                    )
                )
                if page_census != identity.page_count:
                    raise ManagerError("manager arena stats have an invalid page census")
            return result

    def _require_open(self) -> ctypes.c_void_p:
        if not self._handle or not self._handle.value:
            raise ManagerError("OrbitKV manager handle is closed")
        return self._handle

    def _checked_call(self, operation: str, function: Callable[..., Any], *args: Any) -> None:
        error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        status = int(function(*args, error, len(error)))
        if status == STATUS_OK:
            if error.value:
                raise ManagerError(f"{operation} returned success with an error payload")
            return
        message = error.value.decode("utf-8", errors="replace") or "no error detail"
        if status == STATUS_BUFFER_TOO_SMALL:
            raise ManagerError(
                f"{operation} rejected the registered full-capacity output buffer"
            )
        raise ManagerError(f"{operation} failed with status {status}: {message}")

    def _entry_buffer(self) -> Any:
        return (_DeviceViewEntry * self._capacity)()

    def _certificate_buffer(self) -> Any:
        return (_ReclamationCertificate * self._capacity)()

    def _view(self, header: _DeviceViewHeader, entries: Any, count: int) -> DeviceKvView:
        if count < 0 or count > self._capacity or int(header.entry_count) != count:
            raise ManagerError("C ABI returned an inconsistent device-view count")
        return DeviceKvView(
            _header_from_c(header),
            tuple(_entry_from_c(entries[index]) for index in range(count)),
        )

    def request_acquire(self) -> RequestLease:
        with self._lock:
            request = _RequestLease()
            self._checked_call(
                "request acquire",
                self._library.orbitkv_manager_request_acquire,
                self._require_open(),
                ctypes.byref(request),
            )
            return _request_from_c(request)

    def current_view(self, request: RequestLease) -> DeviceKvView:
        with self._lock:
            header = _DeviceViewHeader()
            entries = self._entry_buffer()
            count = ctypes.c_uint32()
            self._checked_call(
                "current view",
                self._library.orbitkv_manager_current_view,
                self._require_open(),
                _request_to_c(request),
                ctypes.byref(header),
                entries,
                self._capacity,
                ctypes.byref(count),
            )
            return self._view(header, entries, int(count.value))

    def prepare_step(self, request: RequestLease, target_boundary: int) -> PreparedStep:
        with self._lock:
            prepared = _PreparedStep()
            entries = self._entry_buffer()
            count = ctypes.c_uint32()
            self._checked_call(
                "prepare step",
                self._library.orbitkv_manager_prepare_step,
                self._require_open(),
                _request_to_c(request),
                _u64("target boundary", target_boundary),
                ctypes.byref(prepared),
                entries,
                self._capacity,
                ctypes.byref(count),
            )
            view = self._view(prepared.view, entries, int(count.value))
            return PreparedStep(
                step=_step_from_c(prepared.step),
                request=_request_from_c(prepared.request),
                base_view_version=int(prepared.base_view_version),
                target_view_version=int(prepared.target_view_version),
                previous_boundary=int(prepared.previous_boundary),
                target_boundary=int(prepared.target_boundary),
                view=view,
            )

    def submit_step(
        self, step: StepLease, receipts: Sequence[BackendBindReceipt]
    ) -> SubmittedStep:
        with self._lock:
            if len(receipts) > self._capacity:
                raise ManagerError("binding receipt count exceeds the arena")
            values = tuple(_bind_to_c(receipt) for receipt in receipts)
            receipt_array = (_BackendBindReceipt * len(values))(*values) if values else None
            submitted = _SubmittedStep()
            entries = self._entry_buffer()
            count = ctypes.c_uint32()
            self._checked_call(
                "submit step",
                self._library.orbitkv_manager_submit_step,
                self._require_open(),
                _step_to_c(step),
                receipt_array,
                len(values),
                ctypes.byref(submitted),
                entries,
                self._capacity,
                ctypes.byref(count),
            )
            return SubmittedStep(
                submission=_submission_from_c(submitted.submission),
                request=_request_from_c(submitted.request),
                view=self._view(submitted.view, entries, int(count.value)),
            )

    def complete_step(self, receipt: CompletionReceipt) -> StepCompletion:
        with self._lock:
            _zero("completion receipt reserved", receipt.reserved)
            if receipt.confirmed != 1:
                raise ManagerError("completion receipt must be confirmed")
            value = _CompletionReceipt(
                _submission_to_c(receipt.submission),
                _u64("completion domain", receipt.completion_domain),
                _u64("completion value", receipt.completion_value),
                _u32("completion confirmed", receipt.confirmed),
                0,
            )
            completion = _StepCompletion()
            entries = self._entry_buffer()
            entry_count = ctypes.c_uint32()
            certificates = self._certificate_buffer()
            certificate_count = ctypes.c_uint32()
            self._checked_call(
                "complete step",
                self._library.orbitkv_manager_complete_step,
                self._require_open(),
                value,
                ctypes.byref(completion),
                entries,
                self._capacity,
                ctypes.byref(entry_count),
                certificates,
                self._capacity,
                ctypes.byref(certificate_count),
            )
            retire_count = int(certificate_count.value)
            if retire_count > self._capacity:
                raise ManagerError("C ABI returned too many reclamation certificates")
            return StepCompletion(
                submission=_submission_from_c(completion.submission),
                request=_request_from_c(completion.request),
                published_view=self._view(
                    completion.published_view, entries, int(entry_count.value)
                ),
                retirements=tuple(
                    _certificate_from_c(certificates[index])
                    for index in range(retire_count)
                ),
            )

    def abort_step(self, receipt: BackendUnobservedReceipt) -> None:
        with self._lock:
            _zero("abort receipt reserved", receipt.reserved)
            if receipt.backend_unobserved != 1:
                raise ManagerError("abort receipt must prove backend unobserved")
            value = _BackendUnobservedReceipt(
                _step_to_c(receipt.step),
                _u32("backend_unobserved", receipt.backend_unobserved),
                0,
            )
            self._checked_call(
                "abort step",
                self._library.orbitkv_manager_abort_step,
                self._require_open(),
                value,
            )

    def quarantine_step(self, step: StepLease) -> None:
        with self._lock:
            self._checked_call(
                "quarantine step",
                self._library.orbitkv_manager_quarantine_step,
                self._require_open(),
                _step_to_c(step),
            )

    def quarantine_submission(self, submission: SubmissionLease) -> None:
        with self._lock:
            self._checked_call(
                "quarantine submission",
                self._library.orbitkv_manager_quarantine_submission,
                self._require_open(),
                _submission_to_c(submission),
            )

    def release_request(self, request: RequestLease) -> ReleaseCompletion:
        with self._lock:
            release = _ReleaseCompletion()
            certificates = self._certificate_buffer()
            count = ctypes.c_uint32()
            self._checked_call(
                "release request",
                self._library.orbitkv_manager_release_request,
                self._require_open(),
                _request_to_c(request),
                ctypes.byref(release),
                certificates,
                self._capacity,
                ctypes.byref(count),
            )
            certificate_count = int(count.value)
            if certificate_count > self._capacity:
                raise ManagerError("C ABI returned too many release certificates")
            return ReleaseCompletion(
                request=_request_from_c(release.request),
                retirements=tuple(
                    _certificate_from_c(certificates[index])
                    for index in range(certificate_count)
                ),
            )

    def commit_reclamations(
        self, receipts: Sequence[ReclamationReceipt]
    ) -> None:
        with self._lock:
            if len(receipts) > self._capacity:
                raise ManagerError("reclamation receipt count exceeds the arena")
            values = tuple(_reclamation_receipt_to_c(receipt) for receipt in receipts)
            receipt_array = (
                (_ReclamationReceipt * len(values))(*values) if values else None
            )
            self._checked_call(
                "acknowledge reclamations",
                self._library.orbitkv_manager_acknowledge_reclamations,
                self._require_open(),
                receipt_array,
                len(values),
            )

    def recycle_request(self, request: RequestLease) -> None:
        with self._lock:
            self._checked_call(
                "recycle request",
                self._library.orbitkv_manager_recycle_request,
                self._require_open(),
                _request_to_c(request),
            )

    def stats(self) -> ManagerStats:
        with self._lock:
            value = _ManagerStats()
            self._checked_call(
                "manager stats",
                self._library.orbitkv_manager_stats,
                self._require_open(),
                ctypes.byref(value),
            )
            return ManagerStats(*(int(getattr(value, name)) for name, _ in value._fields_))

    def destroy(self) -> None:
        with self._lock:
            if not self._handle or not self._handle.value:
                return
            self._checked_call(
                "destroy manager",
                self._library.orbitkv_manager_destroy,
                self._handle,
            )
            self._handle = ctypes.c_void_p()


class CtypesManagerFactory(ManagerFactoryProtocol):
    def create(
        self,
        config: Any,
        settings: ManagerCreateSettings,
        arenas: Sequence[ArenaRegistration],
    ) -> ManagerProtocol:
        library_path = Path(config.library_path)
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise CanonicalAbiUnavailable(
                f"cannot load canonical OrbitKV C ABI {library_path}: {error}"
            ) from error
        _configure_library(library)

        plan = bytes(config.plan_json)
        if not plan:
            raise ManagerError("canonical KvPlanInput JSON is empty")
        plan_buffer = (ctypes.c_uint8 * len(plan)).from_buffer_copy(plan)
        registrations = tuple(arenas)
        if not registrations or len(registrations) != len(config.classes):
            raise ManagerError("one ABI4 arena is required for every plan class")
        if tuple(item.class_id for item in registrations) != tuple(
            range(len(registrations))
        ):
            raise ManagerError("ABI4 arena registrations must be class-id ordered")
        for registration, class_config in zip(
            registrations, config.classes, strict=True
        ):
            if (
                registration.class_id != class_config.class_id
                or registration.pool_id != class_config.pool_id
                or registration.backend_domain != class_config.backend_domain
                or registration.backend_base_index != 0
            ):
                raise ManagerError("ABI4 arena registration differs from the plan")
        total_page_count = sum(item.page_count for item in registrations)
        if total_page_count <= 0 or settings.maximum_reclamations < total_page_count:
            raise ManagerError(
                "maximum_reclamations must cover every registered arena page"
            )
        manager_config = _ManagerConfig(
            _u32("maximum_requests", settings.maximum_requests),
            _u32("maximum_operations", settings.maximum_operations),
            _u32("maximum_reclamations", settings.maximum_reclamations),
            _u32("maximum_step_tokens", settings.maximum_step_tokens),
        )
        backend_values = tuple(_registration_to_c(item) for item in registrations)
        backend_array = (_BackendArenaRegistration * len(backend_values))(
            *backend_values
        )
        handle = ctypes.c_void_p()
        error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        status = int(
            library.orbitkv_manager_create(
                plan_buffer,
                len(plan),
                ctypes.byref(manager_config),
                backend_array,
                len(backend_values),
                ctypes.byref(handle),
                error_buffer,
                len(error_buffer),
            )
        )
        if status != STATUS_OK or not handle.value:
            message = (
                error_buffer.value.decode("utf-8", errors="replace")
                or "manager creation returned a null handle"
            )
            raise ManagerError(f"manager create failed with status {status}: {message}")

        def destroy_after_failed_construction(primary: BaseException) -> None:
            cleanup_error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
            cleanup_status = int(
                library.orbitkv_manager_destroy(
                    handle, cleanup_error, len(cleanup_error)
                )
            )
            if cleanup_status != STATUS_OK:
                detail = cleanup_error.value.decode("utf-8", errors="replace")
                raise ManagerError(
                    f"{primary}; additionally manager cleanup failed with "
                    f"status {cleanup_status}: {detail or 'no error detail'}"
                ) from primary

        if error_buffer.value:
            failure = ManagerError(
                "manager create returned success with an error payload"
            )
            destroy_after_failed_construction(failure)
            raise failure
        try:
            return _CtypesManager(
                library,
                handle,
                registrations,
                int(config.page_tokens),
            )
        except Exception as error:
            destroy_after_failed_construction(error)
            raise


__all__ = [
    "ABI_VERSION",
    "CanonicalAbiUnavailable",
    "CtypesManagerFactory",
]
