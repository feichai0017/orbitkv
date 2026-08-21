from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from . import layouts as L


ABI_VERSION = 6
STATUS_OK = 0
STATUS_BUFFER_TOO_SMALL = 1
STATUS_RETRYABLE_CONFLICT = 2
STATUS_INVALID_ARGUMENT = -1
STATUS_MANAGER_ERROR = -2
STATUS_PANIC = -3
STATUS_FAIL_STOPPED = -4

ERROR_BUFFER_BYTES = 4096


class CanonicalAbiUnavailable(RuntimeError):
    """The configured library does not expose the frozen ABI6 surface."""


HANDLE = ctypes.c_void_p
PCHAR = ctypes.POINTER(ctypes.c_char)
PU32 = ctypes.POINTER(ctypes.c_uint32)


def _ptr(layout: Any) -> Any:
    return ctypes.POINTER(layout)


FUNCTION_SPECS: dict[str, tuple[Any, ...]] = {
    "orbitkv_manager_create": (
        _ptr(ctypes.c_uint8), ctypes.c_size_t,
        _ptr(L.ManagerConfigLayout), _ptr(L.BackendArenaRegistrationLayout),
        ctypes.c_uint32, _ptr(HANDLE), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_arena_identities": (
        HANDLE, _ptr(L.ArenaIdentityLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_arena_stats": (
        HANDLE, _ptr(L.ArenaStatsLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_request_acquire_batch": (
        HANDLE, ctypes.c_uint32, _ptr(L.RequestViewLayout), ctypes.c_uint32,
        PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_request_fork_batch": (
        HANDLE, _ptr(L.RequestForkItemLayout), ctypes.c_uint32,
        _ptr(L.ForkedItemLayout), ctypes.c_uint32, PU32,
        _ptr(L.SnapshotPageLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prepare_batch": (
        HANDLE, _ptr(L.PrepareItemLayout), ctypes.c_uint32,
        _ptr(L.PreparedItemLayout), ctypes.c_uint32, PU32,
        _ptr(L.ClassLoweringLayout), ctypes.c_uint32, PU32,
        _ptr(L.TailActionLayout), ctypes.c_uint32, PU32,
        _ptr(L.CopyIntentLayout), ctypes.c_uint32, PU32,
        _ptr(L.WriteIntentLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_submit_batch": (
        HANDLE, _ptr(L.SubmitItemLayout), ctypes.c_uint32,
        _ptr(L.BindReceiptLayout), ctypes.c_uint32,
        _ptr(L.CopyReceiptLayout), ctypes.c_uint32,
        _ptr(L.SubmittedItemLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_complete_batch": (
        HANDLE, L.CompletionReceiptLayout, _ptr(L.CompleteItemLayout), ctypes.c_uint32,
        _ptr(L.CompletedItemLayout), ctypes.c_uint32, PU32,
        _ptr(L.DetachedBindingLayout), ctypes.c_uint32, PU32,
        _ptr(L.ReclamationCertificateLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_abort_steps_batch": (
        HANDLE, _ptr(L.UnobservedReceiptLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_quarantine_steps_batch": (
        HANDLE, _ptr(L.StepLeaseLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_quarantine_submissions_batch": (
        HANDLE, _ptr(L.SubmissionLeaseLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_release_batch": (
        HANDLE, _ptr(L.ReleaseItemLayout), ctypes.c_uint32,
        _ptr(L.ReleasedItemLayout), ctypes.c_uint32, PU32,
        _ptr(L.DetachedBindingLayout), ctypes.c_uint32, PU32,
        _ptr(L.ReclamationCertificateLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_acknowledge_reclamations_batch": (
        HANDLE, _ptr(L.ReclamationReceiptLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_recycle_requests_batch": (
        HANDLE, _ptr(L.RequestLeaseLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prefix_lookup_batch": (
        HANDLE, _ptr(L.PrefixKeyLayout), ctypes.c_uint32,
        _ptr(L.PrefixLookupHintLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prefix_attach_batch": (
        HANDLE, _ptr(L.PrefixAttachItemLayout), ctypes.c_uint32,
        _ptr(L.AttachedPrefixLayout), ctypes.c_uint32, PU32,
        _ptr(L.SnapshotPageLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prefix_publish_batch": (
        HANDLE, _ptr(L.PrefixPublishItemLayout), ctypes.c_uint32,
        _ptr(L.PublishedPrefixLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prefix_publish_release_batch": (
        HANDLE, _ptr(L.PrefixPublishItemLayout), ctypes.c_uint32,
        _ptr(L.PrefixPublishReleaseLayout), ctypes.c_uint32, PU32,
        _ptr(L.DetachedBindingLayout), ctypes.c_uint32, PU32,
        _ptr(L.ReclamationCertificateLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prefix_evict_batch": (
        HANDLE, _ptr(L.PrefixLeaseLayout), ctypes.c_uint32,
        _ptr(L.EvictedPrefixLayout), ctypes.c_uint32, PU32,
        _ptr(L.ReclamationCertificateLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_prefix_recycle_batch": (
        HANDLE, _ptr(L.PrefixLeaseLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_stats": (
        HANDLE, _ptr(L.ManagerStatsLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_manager_destroy": (HANDLE, PCHAR, ctypes.c_size_t),
}


EXACT_SYMBOL_ALLOWLIST = frozenset({"orbitkv_abi_version", *FUNCTION_SPECS})


class LoadedLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        try:
            library = ctypes.CDLL(str(self.path))
        except OSError as error:
            raise CanonicalAbiUnavailable(
                f"cannot load OrbitKV ABI6 library {self.path}: {error}"
            ) from error
        try:
            abi = library.orbitkv_abi_version
            abi.argtypes = []
            abi.restype = ctypes.c_uint32
            actual = int(abi())
            if actual != ABI_VERSION:
                raise CanonicalAbiUnavailable(
                    f"OrbitKV ABI mismatch: library={actual}, adapter={ABI_VERSION}"
                )
            for name, argtypes in FUNCTION_SPECS.items():
                function = getattr(library, name)
                function.argtypes = list(argtypes)
                function.restype = ctypes.c_int32
        except AttributeError as error:
            raise CanonicalAbiUnavailable(
                f"OrbitKV library is missing an ABI6 symbol: {error}"
            ) from error
        self.cdll = library

    def function(self, name: str) -> Any:
        if name not in EXACT_SYMBOL_ALLOWLIST:
            raise KeyError(f"symbol is outside the frozen ABI6 allowlist: {name}")
        return getattr(self.cdll, name)


__all__ = [
    "ABI_VERSION",
    "CanonicalAbiUnavailable",
    "ERROR_BUFFER_BYTES",
    "EXACT_SYMBOL_ALLOWLIST",
    "FUNCTION_SPECS",
    "LoadedLibrary",
    "STATUS_BUFFER_TOO_SMALL",
    "STATUS_FAIL_STOPPED",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_MANAGER_ERROR",
    "STATUS_OK",
    "STATUS_PANIC",
    "STATUS_RETRYABLE_CONFLICT",
]
