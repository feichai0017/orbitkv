from __future__ import annotations

import ctypes
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from orbitkv_sglang.runtime import (
    FailStopped,
    ManagerError,
    RetryableConflict,
    StateCompletionReceipt,
    StateCopyIntent,
    StateCopyReceipt,
    StatePoolConfig,
    StatePoolIdentity,
    StatePoolStats,
    StatePublication,
    StateRetirementCertificate,
    StateRetirementLease,
    StateSlotLease,
    StateTransitionLease,
)

from . import layouts as L
from .codec import uint as _uint
from .library import (
    ERROR_BUFFER_BYTES,
    STATUS_BUFFER_TOO_SMALL,
    STATUS_FAIL_STOPPED,
    STATUS_INVALID_ARGUMENT,
    STATUS_MANAGER_ERROR,
    STATUS_OK,
    STATUS_PANIC,
    STATUS_RETRYABLE_CONFLICT,
    LoadedLibrary,
)
from .workspace import array


def _slot_c(value: StateSlotLease | None) -> L.StateSlotLeaseLayout:
    if value is None:
        return L.StateSlotLeaseLayout()
    return L.StateSlotLeaseLayout(
        _uint("state slot engine epoch", value.engine_epoch, 64),
        _uint("state slot pool epoch", value.pool_epoch, 64),
        _uint("state slot generation", value.generation, 64),
        _uint("state slot id", value.slot_id, 32),
        _uint("state slot pool id", value.pool_id, 32),
    )


def _slot(value: Any) -> StateSlotLease:
    return StateSlotLease(
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.slot_id),
        int(value.pool_id),
    )


def _transition_c(value: StateTransitionLease) -> L.StateTransitionLeaseLayout:
    return L.StateTransitionLeaseLayout(
        _uint("state transition engine epoch", value.engine_epoch, 64),
        _uint("state transition slot", value.slot, 32),
        _uint("state transition generation", value.generation, 32),
    )


def _transition(value: Any) -> StateTransitionLease:
    return StateTransitionLease(
        int(value.engine_epoch), int(value.slot), int(value.generation)
    )


def _retirement(value: Any) -> StateRetirementLease:
    return StateRetirementLease(
        int(value.engine_epoch), int(value.slot), int(value.generation)
    )


def _certificate_c(
    value: StateRetirementCertificate,
) -> L.StateRetirementCertificateLayout:
    return L.StateRetirementCertificateLayout(
        L.StateRetirementLeaseLayout(
            _uint("state retirement engine epoch", value.retirement.engine_epoch, 64),
            _uint("state retirement slot", value.retirement.slot, 32),
            _uint("state retirement generation", value.retirement.generation, 32),
        ),
        _slot_c(value.slot),
        _uint("state retirement byte count", value.byte_count, 64),
        _uint("state retirement completion domain", value.completion_domain, 64),
        _uint("state retirement completion value", value.completion_value, 64),
    )


def _completion_c(value: StateCompletionReceipt) -> L.StateCompletionReceiptLayout:
    return L.StateCompletionReceiptLayout(
        _uint("state completion engine epoch", value.engine_epoch, 64),
        _uint("state completion domain", value.completion_domain, 64),
        _uint("state completion value", value.completion_value, 64),
        _uint("state completion confirmed", value.confirmed, 1),
        0,
    )


def _certificate(value: Any) -> StateRetirementCertificate:
    return StateRetirementCertificate(
        _retirement(value.retirement),
        _slot(value.slot),
        int(value.byte_count),
        int(value.completion_domain),
        int(value.completion_value),
    )


def _discard_created_handle(loaded: LoadedLibrary, handle: ctypes.c_void_p) -> None:
    if not handle.value:
        return
    error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
    try:
        loaded.function("orbitkv_state_pool_destroy")(handle, error, len(error))
    except BaseException:
        pass
    finally:
        handle.value = None


class CtypesStatePool:
    def __init__(self, library_path: Path, config: StatePoolConfig):
        self._library = LoadedLibrary(library_path)
        self._error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        self._lock = RLock()
        self._handle = ctypes.c_void_p()
        self._poisoned: str | None = None
        self._slot_count = _uint("state slot count", config.slot_count, 32)
        raw = L.StatePoolConfigLayout(
            _uint("state engine epoch", config.engine_epoch, 64),
            _uint("state pool epoch", config.pool_epoch, 64),
            _uint("state byte count", config.byte_count, 64),
            _uint("state pool id", config.pool_id, 32),
            self._slot_count,
        )
        try:
            self._call(
                "create state pool",
                self._library.function("orbitkv_state_pool_create"),
                ctypes.byref(raw),
                ctypes.byref(self._handle),
                require_handle=False,
            )
            if not self._handle.value:
                raise ManagerError("state pool create returned a null handle")
            self.identity = self._load_identity()
            expected_identity = StatePoolIdentity(
                config.engine_epoch,
                config.pool_epoch,
                config.byte_count,
                config.pool_id,
                config.slot_count,
            )
            if self.identity != expected_identity:
                raise self._poison("state pool identity differs from create config")
        except BaseException:
            _discard_created_handle(self._library, self._handle)
            raise

    def _require_handle(self, *, allow_poisoned: bool = False) -> ctypes.c_void_p:
        if not self._handle.value:
            raise ManagerError("OrbitKV state pool handle is closed")
        if self._poisoned is not None and not allow_poisoned:
            raise FailStopped("OrbitKV state pool is poisoned: " + self._poisoned)
        return self._handle

    def _poison(self, reason: str) -> FailStopped:
        if self._poisoned is None:
            self._poisoned = reason
        return FailStopped("OrbitKV state pool is poisoned: " + self._poisoned)

    def _call(
        self,
        operation: str,
        function: Any,
        *args: Any,
        require_handle: bool = True,
        allow_poisoned: bool = False,
    ) -> int:
        if require_handle:
            self._require_handle(allow_poisoned=allow_poisoned)
        ctypes.memset(self._error, 0, len(self._error))
        try:
            status = int(function(*args, self._error, len(self._error)))
        except BaseException as error:
            raise self._poison(f"{operation} outcome is unknown: {error}") from error
        message = self._error.value.decode("utf-8", errors="replace")
        if status == STATUS_OK:
            if message:
                raise self._poison(f"{operation} succeeded with an error payload")
            return status
        if status == STATUS_BUFFER_TOO_SMALL:
            raise self._poison(f"{operation} unexpectedly returned a short buffer")
        if status == STATUS_RETRYABLE_CONFLICT:
            raise RetryableConflict(f"{operation}: {message}")
        if status in (STATUS_INVALID_ARGUMENT, STATUS_MANAGER_ERROR):
            raise ManagerError(f"{operation} failed with status {status}: {message}")
        if status in (STATUS_PANIC, STATUS_FAIL_STOPPED):
            raise self._poison(f"{operation} failed with status {status}: {message}")
        raise self._poison(f"{operation} returned unknown status {status}: {message}")

    def _count(self, values: Sequence[Any], label: str) -> int:
        count = len(values)
        if not 0 < count <= self._slot_count:
            raise ManagerError(f"{label} exceeds state pool capacity")
        return count

    def _load_identity(self) -> StatePoolIdentity:
        raw = L.StatePoolIdentityLayout()
        self._call(
            "state pool identity",
            self._library.function("orbitkv_state_pool_identity"),
            self._require_handle(),
            ctypes.byref(raw),
        )
        return StatePoolIdentity(
            int(raw.engine_epoch),
            int(raw.pool_epoch),
            int(raw.byte_count),
            int(raw.pool_id),
            int(raw.slot_count),
        )

    def stats(self) -> StatePoolStats:
        with self._lock:
            raw = L.StatePoolStatsLayout()
            self._call(
                "state pool stats",
                self._library.function("orbitkv_state_pool_stats"),
                self._require_handle(allow_poisoned=True),
                ctypes.byref(raw),
                allow_poisoned=True,
            )
            identity = StatePoolIdentity(
                int(raw.identity.engine_epoch),
                int(raw.identity.pool_epoch),
                int(raw.identity.byte_count),
                int(raw.identity.pool_id),
                int(raw.identity.slot_count),
            )
            if identity != self.identity:
                raise self._poison("state pool stats identity drifted")
            return StatePoolStats(
                identity,
                int(raw.free_slots),
                int(raw.reserved_slots),
                int(raw.relocating_slots),
                int(raw.live_slots),
                int(raw.retiring_slots),
                int(raw.quarantined_slots),
                int(raw.active_owners),
                int(raw.pending_transitions),
                int(raw.pending_retirements),
            )

    def prepare_batch(
        self, items: Sequence[tuple[int, StateSlotLease | None]]
    ) -> tuple[StateCopyIntent, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "state prepare")
            raw = (L.StatePrepareItemLayout * count)(
                *(
                    L.StatePrepareItemLayout(
                        _uint("state owner id", owner_id, 64),
                        _slot_c(expected),
                        int(expected is not None),
                        0,
                    )
                    for owner_id, expected in values
                )
            )
            output = array(L.StateCopyIntentLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "prepare state batch",
                self._library.function("orbitkv_state_pool_prepare_batch"),
                self._require_handle(),
                raw,
                count,
                output,
                count,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison("state prepare cardinality changed")
            if any(
                int(item.source_present) not in (0, 1) or int(item.reserved) != 0
                for item in output
            ):
                raise self._poison("state prepare output flags are invalid")
            return tuple(
                StateCopyIntent(
                    _transition(item.transition),
                    int(item.owner_id),
                    _slot(item.source) if int(item.source_present) else None,
                    _slot(item.destination),
                    int(item.byte_count),
                )
                for item in output
            )

    def submit_batch(self, receipts: Sequence[StateCopyReceipt]) -> None:
        with self._lock:
            values = tuple(receipts)
            count = self._count(values, "state submit")
            raw = (L.StateCopyReceiptLayout * count)(
                *(
                    L.StateCopyReceiptLayout(
                        _transition_c(item.transition),
                        _slot_c(item.source),
                        _slot_c(item.destination),
                        _uint("state byte count", item.byte_count, 64),
                        int(item.source is not None),
                        _uint("state observed", item.observed, 8),
                        _uint("state written", item.written, 8),
                        0,
                        0,
                    )
                    for item in values
                )
            )
            self._call(
                "submit state batch",
                self._library.function("orbitkv_state_pool_submit_batch"),
                self._require_handle(),
                raw,
                count,
            )

    def complete_batch(
        self,
        receipt: StateCompletionReceipt,
        transitions: Sequence[StateTransitionLease],
    ) -> tuple[StatePublication, ...]:
        with self._lock:
            values = tuple(transitions)
            count = self._count(values, "state complete")
            completion = _completion_c(receipt)
            raw = (L.StateTransitionLeaseLayout * count)(
                *(_transition_c(item) for item in values)
            )
            output = array(L.StatePublicationLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "complete state batch",
                self._library.function("orbitkv_state_pool_complete_batch"),
                self._require_handle(),
                ctypes.byref(completion),
                raw,
                count,
                output,
                count,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison("state publication cardinality changed")
            if any(
                int(item.retirement_present) not in (0, 1)
                or int(item.reserved) != 0
                for item in output
            ):
                raise self._poison("state publication output flags are invalid")
            return tuple(
                StatePublication(
                    int(item.owner_id),
                    _slot(item.slot),
                    _certificate(item.retirement)
                    if int(item.retirement_present)
                    else None,
                )
                for item in output
            )

    def abort_batch(self, transitions: Sequence[StateTransitionLease]) -> None:
        with self._lock:
            values = tuple(transitions)
            count = self._count(values, "state abort")
            raw = (L.StateAbortItemLayout * count)(
                *(L.StateAbortItemLayout(_transition_c(item), 1, 0) for item in values)
            )
            self._call(
                "abort state batch",
                self._library.function("orbitkv_state_pool_abort_batch"),
                self._require_handle(),
                raw,
                count,
            )

    def retire_owners_batch(
        self,
        receipt: StateCompletionReceipt,
        items: Sequence[tuple[int, StateSlotLease]],
    ) -> tuple[StateRetirementCertificate, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "state retire")
            completion = _completion_c(receipt)
            raw = (L.StateRetireOwnerItemLayout * count)(
                *(
                    L.StateRetireOwnerItemLayout(
                        _uint("state owner id", owner_id, 64), _slot_c(slot)
                    )
                    for owner_id, slot in values
                )
            )
            output = array(L.StateRetirementCertificateLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "retire state owners",
                self._library.function("orbitkv_state_pool_retire_owners_batch"),
                self._require_handle(),
                ctypes.byref(completion),
                raw,
                count,
                output,
                count,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison("state retirement cardinality changed")
            return tuple(_certificate(item) for item in output)

    def acknowledge_batch(
        self, certificates: Sequence[StateRetirementCertificate]
    ) -> None:
        with self._lock:
            values = tuple(certificates)
            count = self._count(values, "state acknowledge")
            raw = (L.StateRetirementCertificateLayout * count)(
                *(_certificate_c(item) for item in values)
            )
            self._call(
                "acknowledge state batch",
                self._library.function("orbitkv_state_pool_acknowledge_batch"),
                self._require_handle(),
                raw,
                count,
            )

    def current_batch(
        self, owner_ids: Sequence[int]
    ) -> tuple[tuple[int, StateSlotLease | None], ...]:
        with self._lock:
            values = tuple(owner_ids)
            count = self._count(values, "state current")
            raw = (ctypes.c_uint64 * count)(
                *(_uint("state owner id", item, 64) for item in values)
            )
            output = array(L.StateCurrentLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "state current batch",
                self._library.function("orbitkv_state_pool_current_batch"),
                self._require_handle(),
                raw,
                count,
                output,
                count,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison("state current cardinality changed")
            if any(
                int(item.owner_id) != owner_id
                or int(item.present) not in (0, 1)
                or int(item.reserved) != 0
                for owner_id, item in zip(values, output)
            ):
                raise self._poison("state current output is invalid")
            return tuple(
                (int(item.owner_id), _slot(item.slot) if int(item.present) else None)
                for item in output
            )

    def close(self) -> None:
        with self._lock:
            if not self._handle.value:
                return
            handle = self._handle
            try:
                self._call(
                    "destroy state pool",
                    self._library.function("orbitkv_state_pool_destroy"),
                    handle,
                    require_handle=False,
                )
            finally:
                self._handle = ctypes.c_void_p()


__all__ = ["CtypesStatePool"]
