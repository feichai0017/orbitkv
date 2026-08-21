from __future__ import annotations

import ctypes
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Sequence

from .runtime import (
    ArenaIdentity,
    ArenaRegistration,
    ArenaStats,
    BatchCompletionReceipt,
    BackendBindReceipt,
    BackendUnobservedReceipt,
    ClassLowering,
    ManagerCreateSettings,
    ManagerError,
    ManagerFactoryProtocol,
    ManagerProtocol,
    ManagerStats,
    PageLease,
    PrepareBatchItem,
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
    WriteIntent,
)


ABI_VERSION = 5
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


class _PrepareBatchItem(ctypes.Structure):
    _fields_ = [
        ("request", _RequestLease),
        ("target_boundary", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64),
    ]


class _PreparedBatchItem(ctypes.Structure):
    _fields_ = [
        ("step", _StepLease),
        ("request", _RequestLease),
        ("base_view_version", ctypes.c_uint64),
        ("target_view_version", ctypes.c_uint64),
        ("previous_boundary", ctypes.c_uint64),
        ("target_boundary", ctypes.c_uint64),
        ("class_offset", ctypes.c_uint32),
        ("class_count", ctypes.c_uint32),
        ("write_offset", ctypes.c_uint32),
        ("write_count", ctypes.c_uint32),
    ]


class _ClassLowering(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("write_offset", ctypes.c_uint32),
        ("write_count", ctypes.c_uint32),
        ("previous_tail_page_id", ctypes.c_uint32),
        ("previous_tail_generation", ctypes.c_uint64),
    ]


class _WriteIntent(ctypes.Structure):
    _fields_ = [
        ("page_generation", ctypes.c_uint64),
        ("page_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
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


class _SubmitBatchItem(ctypes.Structure):
    _fields_ = [
        ("step", _StepLease),
        ("receipt_offset", ctypes.c_uint32),
        ("receipt_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


class _SubmittedBatchItem(ctypes.Structure):
    _fields_ = [
        ("submission", _SubmissionLease),
        ("request", _RequestLease),
    ]


class _BatchCompletionReceipt(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("completion_domain", ctypes.c_uint64),
        ("completion_value", ctypes.c_uint64),
        ("confirmed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _CompleteBatchItem(ctypes.Structure):
    _fields_ = [("submission", _SubmissionLease)]


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
        ("reserved32", ctypes.c_uint32),
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


class _CompletedBatchItem(ctypes.Structure):
    _fields_ = [
        ("submission", _SubmissionLease),
        ("request", _RequestLease),
        ("published_view_version", ctypes.c_uint64),
        ("published_boundary", ctypes.c_uint64),
        ("resident_count", ctypes.c_uint32),
        ("retirement_offset", ctypes.c_uint32),
        ("retirement_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _ReleaseBatchItem(ctypes.Structure):
    _fields_ = [("request", _RequestLease), ("reserved", ctypes.c_uint64)]


class _ReleasedBatchItem(ctypes.Structure):
    _fields_ = [
        ("request", _RequestLease),
        ("retirement_offset", ctypes.c_uint32),
        ("retirement_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


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
    _BackendArenaRegistration: 24,
    _ManagerConfig: 16,
    _ArenaIdentity: 48,
    _ArenaStats: 96,
    _PrepareBatchItem: 32,
    _PreparedBatchItem: 80,
    _ClassLowering: 24,
    _WriteIntent: 16,
    _BackendBindReceipt: 64,
    _SubmitBatchItem: 32,
    _SubmittedBatchItem: 32,
    _BatchCompletionReceipt: 32,
    _CompleteBatchItem: 16,
    _BackendUnobservedReceipt: 24,
    _ReclamationCertificate: 120,
    _ReclamationReceipt: 64,
    _CompletedBatchItem: 64,
    _ReleaseBatchItem: 24,
    _ReleasedBatchItem: 32,
    _ManagerStats: 88,
}
for _structure, _expected_size in _EXPECTED_SIZES.items():
    if ctypes.sizeof(_structure) != _expected_size:
        raise CanonicalAbiUnavailable(
            f"ctypes layout mismatch for {_structure.__name__}: "
            f"{ctypes.sizeof(_structure)} != {_expected_size}"
        )

_EXPECTED_OFFSETS = {
    (_PrepareBatchItem, "request"): 0,
    (_PrepareBatchItem, "target_boundary"): 16,
    (_PrepareBatchItem, "reserved"): 24,
    (_PreparedBatchItem, "class_offset"): 64,
    (_PreparedBatchItem, "class_count"): 68,
    (_PreparedBatchItem, "write_offset"): 72,
    (_PreparedBatchItem, "write_count"): 76,
    (_ClassLowering, "previous_tail_generation"): 16,
    (_WriteIntent, "reserved"): 12,
    (_SubmitBatchItem, "receipt_offset"): 16,
    (_SubmitBatchItem, "receipt_count"): 20,
    (_SubmitBatchItem, "reserved"): 24,
    (_BatchCompletionReceipt, "confirmed"): 24,
    (_ReclamationCertificate, "reserved32"): 68,
    (_ReclamationCertificate, "logical_ordinal"): 72,
    (_CompletedBatchItem, "published_view_version"): 32,
    (_CompletedBatchItem, "published_boundary"): 40,
    (_CompletedBatchItem, "resident_count"): 48,
    (_CompletedBatchItem, "retirement_offset"): 52,
    (_CompletedBatchItem, "retirement_count"): 56,
    (_CompletedBatchItem, "reserved"): 60,
    (_ReleasedBatchItem, "retirement_offset"): 16,
    (_ReleasedBatchItem, "retirement_count"): 20,
    (_ReleasedBatchItem, "reserved"): 24,
}
for (_structure, _field), _expected_offset in _EXPECTED_OFFSETS.items():
    _actual_offset = int(getattr(_structure, _field).offset)
    if _actual_offset != _expected_offset:
        raise CanonicalAbiUnavailable(
            f"ctypes offset mismatch for {_structure.__name__}.{_field}: "
            f"{_actual_offset} != {_expected_offset}"
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


def _certificate_from_c(value: _ReclamationCertificate) -> ReclamationCertificate:
    _zero("reclamation certificate reserved32", int(value.reserved32))
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
    # Do not interpret any semantic bind field here. Once submit_batch has
    # resolved the ordered step set, the core must see a malformed receipt so
    # it can quarantine the entire candidate batch atomically.
    return _BackendBindReceipt(
        _step_to_c(value.step),
        _page_to_c(value.page),
        _u16("binding backend_domain", value.backend_domain),
        _u8("binding mapped", value.mapped),
        _u8("binding writable", value.writable),
        _u32("binding reserved", value.reserved),
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
        "orbitkv_manager_request_acquire_batch": (
            [
                handle,
                ctypes.c_uint32,
                ctypes.POINTER(_RequestLease),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_prepare_batch": (
            [
                handle,
                ctypes.POINTER(_PrepareBatchItem),
                ctypes.c_uint32,
                ctypes.POINTER(_PreparedBatchItem),
                ctypes.c_uint32,
                u32p,
                ctypes.POINTER(_ClassLowering),
                ctypes.c_uint32,
                u32p,
                ctypes.POINTER(_WriteIntent),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_submit_batch": (
            [
                handle,
                ctypes.POINTER(_SubmitBatchItem),
                ctypes.c_uint32,
                ctypes.POINTER(_BackendBindReceipt),
                ctypes.c_uint32,
                ctypes.POINTER(_SubmittedBatchItem),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_complete_batch": (
            [
                handle,
                _BatchCompletionReceipt,
                ctypes.POINTER(_CompleteBatchItem),
                ctypes.c_uint32,
                ctypes.POINTER(_CompletedBatchItem),
                ctypes.c_uint32,
                u32p,
                ctypes.POINTER(_ReclamationCertificate),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_abort_steps": (
            [
                handle,
                ctypes.POINTER(_BackendUnobservedReceipt),
                ctypes.c_uint32,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_quarantine_steps": (
            [handle, ctypes.POINTER(_StepLease), ctypes.c_uint32, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_quarantine_submissions": (
            [
                handle,
                ctypes.POINTER(_SubmissionLease),
                ctypes.c_uint32,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_release_batch": (
            [
                handle,
                ctypes.POINTER(_ReleaseBatchItem),
                ctypes.c_uint32,
                ctypes.POINTER(_ReleasedBatchItem),
                ctypes.c_uint32,
                u32p,
                ctypes.POINTER(_ReclamationCertificate),
                ctypes.c_uint32,
                u32p,
                error,
                ctypes.c_size_t,
            ],
            ctypes.c_int32,
        ),
        "orbitkv_manager_acknowledge_reclamations": (
            [handle, ctypes.POINTER(_ReclamationReceipt), ctypes.c_uint32, error, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "orbitkv_manager_recycle_requests": (
            [handle, ctypes.POINTER(_RequestLease), ctypes.c_uint32, error, ctypes.c_size_t],
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
        settings: ManagerCreateSettings,
        sliding_class_count: int,
    ):
        self._library = library
        self._handle = handle
        self._arena_count = len(registrations)
        self._capacity = sum(item.page_count for item in registrations)
        self._request_capacity = int(settings.maximum_requests)
        self._operation_capacity = min(
            self._request_capacity, int(settings.maximum_operations)
        )
        maximum_step_tokens = int(settings.maximum_step_tokens)
        self._lock = RLock()
        if (
            self._arena_count <= 0
            or self._capacity <= 0
            or self._request_capacity <= 0
            or self._operation_capacity <= 0
            or page_tokens <= 0
            or maximum_step_tokens <= 0
            or not 0 <= sliding_class_count <= self._arena_count
        ):
            raise ManagerError("ABI5 manager requires nonempty item and page arenas")

        pages_per_step_class = (maximum_step_tokens + page_tokens - 1) // page_tokens
        maximum_writes_per_item = self._arena_count * pages_per_step_class
        maximum_retirements_per_item = sliding_class_count * (
            pages_per_step_class + 1
        )
        self._class_capacity = self._operation_capacity * self._arena_count
        self._write_capacity = min(
            self._capacity,
            self._operation_capacity * maximum_writes_per_item,
        )
        self._completion_retirement_capacity = min(
            self._capacity,
            self._operation_capacity * maximum_retirements_per_item,
        )
        if any(
            value >= 1 << 32
            for value in (
                self._class_capacity,
                self._write_capacity,
                self._completion_retirement_capacity,
            )
        ):
            raise ManagerError("ABI5 compact workspace bound exceeds uint32")

        # Fixed ABI5 workspaces. No capacity-sized ctypes array or error buffer
        # is allocated on a lifecycle hot call.
        self._request_workspace = (_RequestLease * self._request_capacity)()
        self._prepare_input_workspace = (
            _PrepareBatchItem * self._operation_capacity
        )()
        self._prepared_workspace = (_PreparedBatchItem * self._operation_capacity)()
        self._submit_input_workspace = (_SubmitBatchItem * self._operation_capacity)()
        self._submitted_workspace = (_SubmittedBatchItem * self._operation_capacity)()
        self._complete_input_workspace = (
            _CompleteBatchItem * self._operation_capacity
        )()
        self._completed_workspace = (_CompletedBatchItem * self._operation_capacity)()
        self._release_input_workspace = (_ReleaseBatchItem * self._request_capacity)()
        self._released_workspace = (_ReleasedBatchItem * self._request_capacity)()
        self._abort_workspace = (
            _BackendUnobservedReceipt * self._operation_capacity
        )()
        self._step_workspace = (_StepLease * self._operation_capacity)()
        self._submission_workspace = (
            _SubmissionLease * self._operation_capacity
        )()
        self._class_workspace = (_ClassLowering * self._class_capacity)()
        self._write_workspace = (
            _WriteIntent * max(1, self._write_capacity)
        )()
        self._bind_workspace = (
            _BackendBindReceipt * max(1, self._write_capacity)
        )()
        self._completion_certificate_workspace = (
            _ReclamationCertificate
            * max(1, self._completion_retirement_capacity)
        )()
        self._release_certificate_workspace = (
            _ReclamationCertificate * self._capacity
        )()
        self._reclamation_workspace = (_ReclamationReceipt * self._capacity)()
        self._error_workspace = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        self._counters: dict[str, int] = {
            "request_acquire_batch_calls": 0,
            "prepare_batch_calls": 0,
            "submit_batch_calls": 0,
            "complete_batch_calls": 0,
            "release_batch_calls": 0,
            "acknowledge_reclamations_calls": 0,
            "recycle_requests_calls": 0,
            "abort_steps_calls": 0,
            "quarantine_steps_calls": 0,
            "quarantine_submissions_calls": 0,
            "acquired_items": 0,
            "prepared_items": 0,
            "submitted_items": 0,
            "completed_items": 0,
            "released_items": 0,
            "hot_workspace_allocations": 0,
            "capacity_memset_bytes": 0,
            "root_entries_crossed": 0,
            "materialized_page_objects": 0,
            "buffer_too_small_failures": 0,
        }
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

    @property
    def performance_counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

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

    def _checked_call(
        self,
        operation: str,
        function: Callable[..., Any],
        *args: Any,
        counter: str | None = None,
    ) -> None:
        if counter is not None:
            self._counters[counter] += 1
        error = self._error_workspace
        ctypes.memset(error, 0, len(error))
        status = int(function(*args, error, len(error)))
        if status == STATUS_OK:
            if error.value:
                raise ManagerError(f"{operation} returned success with an error payload")
            return
        message = error.value.decode("utf-8", errors="replace") or "no error detail"
        if status == STATUS_BUFFER_TOO_SMALL:
            self._counters["buffer_too_small_failures"] += 1
            raise ManagerError(
                f"{operation} rejected the registered full-capacity output buffer"
            )
        raise ManagerError(f"{operation} failed with status {status}: {message}")

    def _batch_count(
        self, values: Sequence[Any], label: str, capacity: int
    ) -> int:
        count = len(values)
        if count <= 0:
            raise ManagerError(f"{label} must be nonempty")
        if count > capacity:
            raise ManagerError(f"{label} exceeds its configured arena capacity")
        return count

    @staticmethod
    def _span(
        *, offset: int, count: int, cursor: int, total: int, label: str
    ) -> int:
        if offset != cursor or count < 0:
            raise ManagerError(f"C ABI returned a noncanonical {label} span")
        end = offset + count
        if end < offset or end > total:
            raise ManagerError(f"C ABI returned an out-of-range {label} span")
        return end

    def request_acquire_batch(self, request_count: int) -> tuple[RequestLease, ...]:
        with self._lock:
            count = _u32("request count", request_count)
            if count == 0 or count > self._request_capacity:
                raise ManagerError("request batch cardinality exceeds maximum_requests")
            out_count = ctypes.c_uint32()
            self._checked_call(
                "request acquire batch",
                self._library.orbitkv_manager_request_acquire_batch,
                self._require_open(),
                count,
                self._request_workspace,
                self._request_capacity,
                ctypes.byref(out_count),
                counter="request_acquire_batch_calls",
            )
            if int(out_count.value) != count:
                raise ManagerError("C ABI returned the wrong acquired-request count")
            result = tuple(
                _request_from_c(self._request_workspace[index])
                for index in range(count)
            )
            self._counters["acquired_items"] += count
            return result

    def prepare_batch(
        self, items: Sequence[PrepareBatchItem]
    ) -> tuple[PreparedStep, ...]:
        with self._lock:
            values = tuple(items)
            count = self._batch_count(
                values, "prepare batch", self._operation_capacity
            )
            for index, item in enumerate(values):
                self._prepare_input_workspace[index] = _PrepareBatchItem(
                    _request_to_c(item.request),
                    _u64("prepare target boundary", item.target_boundary),
                    0,
                )
            out_count = ctypes.c_uint32()
            out_class_count = ctypes.c_uint32()
            out_write_count = ctypes.c_uint32()
            self._checked_call(
                "prepare batch",
                self._library.orbitkv_manager_prepare_batch,
                self._require_open(),
                self._prepare_input_workspace,
                count,
                self._prepared_workspace,
                self._operation_capacity,
                ctypes.byref(out_count),
                self._class_workspace,
                self._class_capacity,
                ctypes.byref(out_class_count),
                self._write_workspace,
                self._write_capacity,
                ctypes.byref(out_write_count),
                counter="prepare_batch_calls",
            )
            class_total = int(out_class_count.value)
            write_total = int(out_write_count.value)
            if (
                int(out_count.value) != count
                or class_total != count * self._arena_count
                or class_total > self._class_capacity
                or write_total > self._write_capacity
            ):
                raise ManagerError("C ABI returned invalid prepare cardinality")
            class_cursor = 0
            write_cursor = 0
            result: list[PreparedStep] = []
            for index in range(count):
                item = self._prepared_workspace[index]
                item_class_offset = int(item.class_offset)
                item_class_count = int(item.class_count)
                item_write_offset = int(item.write_offset)
                item_write_count = int(item.write_count)
                class_end = self._span(
                    offset=item_class_offset,
                    count=item_class_count,
                    cursor=class_cursor,
                    total=class_total,
                    label="class lowering",
                )
                write_end = self._span(
                    offset=item_write_offset,
                    count=item_write_count,
                    cursor=write_cursor,
                    total=write_total,
                    label="write intent",
                )
                if item_class_count != self._arena_count:
                    raise ManagerError("C ABI returned the wrong per-item class count")
                local_write_cursor = item_write_offset
                class_values: list[ClassLowering] = []
                for class_index, position in enumerate(
                    range(item_class_offset, class_end)
                ):
                    lowering = self._class_workspace[position]
                    if int(lowering.class_id) != class_index:
                        raise ManagerError("C ABI class lowerings are not canonical")
                    flags = int(lowering.flags)
                    if flags & ~1:
                        raise ManagerError("C ABI returned unknown class-lowering flags")
                    if not flags and (
                        int(lowering.previous_tail_page_id)
                        or int(lowering.previous_tail_generation)
                    ):
                        raise ManagerError("C ABI returned a tail without its flag")
                    next_write = self._span(
                        offset=int(lowering.write_offset),
                        count=int(lowering.write_count),
                        cursor=local_write_cursor,
                        total=write_end,
                        label="class write",
                    )
                    class_values.append(
                        ClassLowering(
                            class_id=int(lowering.class_id),
                            flags=flags,
                            write_offset=int(lowering.write_offset)
                            - item_write_offset,
                            write_count=int(lowering.write_count),
                            previous_tail_page_id=int(
                                lowering.previous_tail_page_id
                            ),
                            previous_tail_generation=int(
                                lowering.previous_tail_generation
                            ),
                        )
                    )
                    local_write_cursor = next_write
                if local_write_cursor != write_end:
                    raise ManagerError("C ABI class spans do not cover item writes")
                write_values = []
                for position in range(item_write_offset, write_end):
                    intent = self._write_workspace[position]
                    _zero("write intent reserved", int(intent.reserved))
                    write_values.append(
                        WriteIntent(
                            page_generation=int(intent.page_generation),
                            page_id=int(intent.page_id),
                        )
                    )
                result.append(
                    PreparedStep(
                        step=_step_from_c(item.step),
                        request=_request_from_c(item.request),
                        base_view_version=int(item.base_view_version),
                        target_view_version=int(item.target_view_version),
                        previous_boundary=int(item.previous_boundary),
                        target_boundary=int(item.target_boundary),
                        class_lowerings=tuple(class_values),
                        write_intents=tuple(write_values),
                    )
                )
                class_cursor = class_end
                write_cursor = write_end
            if class_cursor != class_total or write_cursor != write_total:
                raise ManagerError("C ABI prepare spans do not cover flat outputs")
            self._counters["prepared_items"] += count
            return tuple(result)

    def submit_batch(
        self,
        items: Sequence[tuple[StepLease, Sequence[BackendBindReceipt]]],
    ) -> tuple[SubmittedStep, ...]:
        with self._lock:
            values = tuple((step, tuple(receipts)) for step, receipts in items)
            count = self._batch_count(
                values, "submit batch", self._operation_capacity
            )
            receipt_count = sum(len(receipts) for _, receipts in values)
            if receipt_count > self._write_capacity:
                raise ManagerError("binding receipt batch exceeds compact write bound")
            receipt_cursor = 0
            for index, (step, receipts) in enumerate(values):
                self._submit_input_workspace[index] = _SubmitBatchItem(
                    _step_to_c(step), receipt_cursor, len(receipts), 0
                )
                for receipt in receipts:
                    self._bind_workspace[receipt_cursor] = _bind_to_c(receipt)
                    receipt_cursor += 1
            out_count = ctypes.c_uint32()
            self._checked_call(
                "submit batch",
                self._library.orbitkv_manager_submit_batch,
                self._require_open(),
                self._submit_input_workspace,
                count,
                self._bind_workspace,
                receipt_count,
                self._submitted_workspace,
                self._operation_capacity,
                ctypes.byref(out_count),
                counter="submit_batch_calls",
            )
            if int(out_count.value) != count:
                raise ManagerError("C ABI returned invalid submit cardinality")
            result: list[SubmittedStep] = []
            for index in range(count):
                item = self._submitted_workspace[index]
                result.append(
                    SubmittedStep(
                        submission=_submission_from_c(item.submission),
                        request=_request_from_c(item.request),
                    )
                )
            self._counters["submitted_items"] += count
            return tuple(result)

    def complete_batch(
        self,
        receipt: BatchCompletionReceipt,
        submissions: Sequence[SubmissionLease],
    ) -> tuple[StepCompletion, ...]:
        with self._lock:
            values = tuple(submissions)
            count = self._batch_count(
                values, "completion batch", self._operation_capacity
            )
            for index, submission in enumerate(values):
                self._complete_input_workspace[index] = _CompleteBatchItem(
                    _submission_to_c(submission)
                )
            value = _BatchCompletionReceipt(
                _u64("completion engine epoch", receipt.engine_epoch),
                _u64("completion domain", receipt.completion_domain),
                _u64("completion value", receipt.completion_value),
                _u32("completion confirmed", receipt.confirmed),
                _u32("completion reserved", receipt.reserved),
            )
            out_count = ctypes.c_uint32()
            out_retirement_count = ctypes.c_uint32()
            self._checked_call(
                "complete batch",
                self._library.orbitkv_manager_complete_batch,
                self._require_open(),
                value,
                self._complete_input_workspace,
                count,
                self._completed_workspace,
                self._operation_capacity,
                ctypes.byref(out_count),
                self._completion_certificate_workspace,
                self._completion_retirement_capacity,
                ctypes.byref(out_retirement_count),
                counter="complete_batch_calls",
            )
            retirement_total = int(out_retirement_count.value)
            if (
                int(out_count.value) != count
                or retirement_total > self._completion_retirement_capacity
            ):
                raise ManagerError("C ABI returned invalid completion cardinality")
            retirement_cursor = 0
            result: list[StepCompletion] = []
            for index in range(count):
                item = self._completed_workspace[index]
                _zero("completed batch item reserved", int(item.reserved))
                retirement_count = int(item.retirement_count)
                retirement_cursor = self._span(
                    offset=int(item.retirement_offset),
                    count=retirement_count,
                    cursor=retirement_cursor,
                    total=retirement_total,
                    label="completion retirement",
                )
                result.append(
                    StepCompletion(
                        submission=_submission_from_c(item.submission),
                        request=_request_from_c(item.request),
                        published_view_version=int(item.published_view_version),
                        published_boundary=int(item.published_boundary),
                        resident_count=int(item.resident_count),
                        retirements=tuple(
                            _certificate_from_c(
                                self._completion_certificate_workspace[position]
                            )
                            for position in range(
                                int(item.retirement_offset), retirement_cursor
                            )
                        ),
                    )
                )
            if retirement_cursor != retirement_total:
                raise ManagerError("C ABI completion spans do not cover flat output")
            self._counters["completed_items"] += count
            return tuple(result)

    def abort_steps(self, receipts: Sequence[BackendUnobservedReceipt]) -> None:
        with self._lock:
            values = tuple(receipts)
            count = self._batch_count(
                values, "abort batch", self._operation_capacity
            )
            for index, receipt in enumerate(values):
                self._abort_workspace[index] = _BackendUnobservedReceipt(
                    _step_to_c(receipt.step),
                    _u32("backend_unobserved", receipt.backend_unobserved),
                    _u32("abort reserved", receipt.reserved),
                )
            self._checked_call(
                "abort steps",
                self._library.orbitkv_manager_abort_steps,
                self._require_open(),
                self._abort_workspace,
                count,
                counter="abort_steps_calls",
            )

    def quarantine_steps(self, steps: Sequence[StepLease]) -> None:
        with self._lock:
            values = tuple(steps)
            count = self._batch_count(
                values, "step quarantine batch", self._operation_capacity
            )
            for index, step in enumerate(values):
                self._step_workspace[index] = _step_to_c(step)
            self._checked_call(
                "quarantine steps",
                self._library.orbitkv_manager_quarantine_steps,
                self._require_open(),
                self._step_workspace,
                count,
                counter="quarantine_steps_calls",
            )

    def quarantine_submissions(
        self, submissions: Sequence[SubmissionLease]
    ) -> None:
        with self._lock:
            values = tuple(submissions)
            count = self._batch_count(
                values, "submission quarantine batch", self._operation_capacity
            )
            for index, submission in enumerate(values):
                self._submission_workspace[index] = _submission_to_c(submission)
            self._checked_call(
                "quarantine submissions",
                self._library.orbitkv_manager_quarantine_submissions,
                self._require_open(),
                self._submission_workspace,
                count,
                counter="quarantine_submissions_calls",
            )

    def release_batch(
        self, requests: Sequence[RequestLease]
    ) -> tuple[ReleaseCompletion, ...]:
        with self._lock:
            values = tuple(requests)
            count = self._batch_count(
                values, "release batch", self._request_capacity
            )
            for index, request in enumerate(values):
                self._release_input_workspace[index] = _ReleaseBatchItem(
                    _request_to_c(request), 0
                )
            out_count = ctypes.c_uint32()
            out_retirement_count = ctypes.c_uint32()
            self._checked_call(
                "release batch",
                self._library.orbitkv_manager_release_batch,
                self._require_open(),
                self._release_input_workspace,
                count,
                self._released_workspace,
                self._request_capacity,
                ctypes.byref(out_count),
                self._release_certificate_workspace,
                self._capacity,
                ctypes.byref(out_retirement_count),
                counter="release_batch_calls",
            )
            retirement_total = int(out_retirement_count.value)
            if int(out_count.value) != count or retirement_total > self._capacity:
                raise ManagerError("C ABI returned invalid release cardinality")
            cursor = 0
            result: list[ReleaseCompletion] = []
            for index in range(count):
                item = self._released_workspace[index]
                _zero("released batch item reserved", int(item.reserved))
                retirement_count = int(item.retirement_count)
                cursor = self._span(
                    offset=int(item.retirement_offset),
                    count=retirement_count,
                    cursor=cursor,
                    total=retirement_total,
                    label="release retirement",
                )
                result.append(
                    ReleaseCompletion(
                        request=_request_from_c(item.request),
                        retirements=tuple(
                            _certificate_from_c(
                                self._release_certificate_workspace[position]
                            )
                            for position in range(int(item.retirement_offset), cursor)
                        ),
                    )
                )
            if cursor != retirement_total:
                raise ManagerError("C ABI release spans do not cover the flat output")
            self._counters["released_items"] += count
            return tuple(result)

    def acknowledge_reclamations(
        self, receipts: Sequence[ReclamationReceipt]
    ) -> None:
        with self._lock:
            values = tuple(receipts)
            if not values or len(values) > self._capacity:
                raise ManagerError("reclamation receipt batch cardinality is invalid")
            for index, receipt in enumerate(values):
                self._reclamation_workspace[index] = _reclamation_receipt_to_c(receipt)
            self._checked_call(
                "acknowledge reclamations",
                self._library.orbitkv_manager_acknowledge_reclamations,
                self._require_open(),
                self._reclamation_workspace,
                len(values),
                counter="acknowledge_reclamations_calls",
            )

    def recycle_requests(self, requests: Sequence[RequestLease]) -> None:
        with self._lock:
            values = tuple(requests)
            count = self._batch_count(
                values, "request recycle batch", self._request_capacity
            )
            for index, request in enumerate(values):
                self._request_workspace[index] = _request_to_c(request)
            self._checked_call(
                "recycle requests",
                self._library.orbitkv_manager_recycle_requests,
                self._require_open(),
                self._request_workspace,
                count,
                counter="recycle_requests_calls",
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
            raise ManagerError("one ABI5 arena is required for every plan class")
        if tuple(item.class_id for item in registrations) != tuple(
            range(len(registrations))
        ):
            raise ManagerError("ABI5 arena registrations must be class-id ordered")
        for registration, class_config in zip(
            registrations, config.classes, strict=True
        ):
            if (
                registration.class_id != class_config.class_id
                or registration.pool_id != class_config.pool_id
                or registration.backend_domain != class_config.backend_domain
                or registration.backend_base_index != 0
            ):
                raise ManagerError("ABI5 arena registration differs from the plan")
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
                settings,
                sum(
                    item.retention == "sliding"
                    for item in config.classes
                ),
            )
        except Exception as error:
            destroy_after_failed_construction(error)
            raise


__all__ = [
    "ABI_VERSION",
    "CanonicalAbiUnavailable",
    "CtypesManagerFactory",
]
