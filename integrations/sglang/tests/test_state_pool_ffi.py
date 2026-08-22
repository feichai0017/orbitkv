from __future__ import annotations

from pathlib import Path
import ctypes

import pytest

from orbitkv_sglang.ffi import CtypesStatePool
from orbitkv_sglang.runtime import (
    FailStopped,
    ManagerError,
    StateCompletionReceipt,
    StateCopyReceipt,
    StatePoolConfig,
)
from test_multi_arena_ffi import ffi_library


def _pool(library: Path) -> CtypesStatePool:
    return CtypesStatePool(library, StatePoolConfig(11, 12, 4096, 7, 4))


def _receipt(intent) -> StateCopyReceipt:
    return StateCopyReceipt(
        intent.transition, intent.source, intent.destination, intent.byte_count
    )


def test_native_state_pool_replace_retire_ack_and_generation_reuse(ffi_library):
    pool = _pool(ffi_library)
    assert pool.identity.slot_count == 4
    first = pool.prepare_batch(((101, None),))[0]
    assert first.destination.slot_id == 0
    pool.submit_batch((_receipt(first),))
    publication = pool.complete_batch(
        StateCompletionReceipt(11, 3, 1), (first.transition,)
    )[0]
    assert publication.retirement is None
    assert pool.current_batch((101,)) == ((101, publication.slot),)

    replacement = pool.prepare_batch(((101, publication.slot),))[0]
    pool.submit_batch((_receipt(replacement),))
    second = pool.complete_batch(
        StateCompletionReceipt(11, 3, 2), (replacement.transition,)
    )[0]
    assert second.retirement is not None
    pool.acknowledge_batch((second.retirement,))
    retirement = pool.retire_owners_batch(
        StateCompletionReceipt(11, 3, 3), ((101, second.slot),)
    )[0]
    pool.acknowledge_batch((retirement,))
    stats = pool.stats()
    assert stats.free_slots == stats.identity.slot_count == 4
    assert stats.active_owners == stats.pending_retirements == 0

    reused = pool.prepare_batch(((202, None),))[0]
    assert reused.destination.generation > publication.slot.generation
    pool.abort_batch((reused.transition,))
    pool.close()


def test_native_state_pool_short_output_is_zero_mutation(ffi_library):
    pool = _pool(ffi_library)
    before = pool.stats()
    loaded = pool._library
    raw = loaded.function("orbitkv_state_pool_prepare_batch")
    from orbitkv_sglang.ffi import layouts as L
    import ctypes

    item = L.StatePrepareItemLayout(101, L.StateSlotLeaseLayout(), 0, 0)
    count = ctypes.c_uint32()
    error = ctypes.create_string_buffer(256)
    status = raw(pool._handle, ctypes.byref(item), 1, None, 0, ctypes.byref(count), error, 256)
    assert status == 1
    assert count.value == 1
    assert pool.stats() == before
    pool.close()


def test_native_state_receipt_fault_quarantines_and_poison_client(ffi_library):
    pool = _pool(ffi_library)
    prepared = pool.prepare_batch(((101, None), (202, None)))
    receipts = [_receipt(item) for item in prepared]
    receipts[1] = StateCopyReceipt(
        receipts[1].transition,
        receipts[1].source,
        receipts[1].destination,
        receipts[1].byte_count + 1,
    )
    with pytest.raises(FailStopped, match="status -4"):
        pool.submit_batch(receipts)
    stats = pool.stats()
    assert stats.quarantined_slots == 2
    with pytest.raises(FailStopped, match="poisoned"):
        pool.current_batch((101,))
    pool.close()


def test_native_state_abort_unknown_quarantines_batch(ffi_library):
    pool = _pool(ffi_library)
    prepared = pool.prepare_batch(((101, None), (202, None)))
    loaded = pool._library
    raw = loaded.function("orbitkv_state_pool_abort_batch")
    from orbitkv_sglang.ffi import layouts as L
    import ctypes

    items = (L.StateAbortItemLayout * 2)(
        L.StateAbortItemLayout(
            L.StateTransitionLeaseLayout(
                prepared[0].transition.engine_epoch,
                prepared[0].transition.slot,
                prepared[0].transition.generation,
            ),
            1,
            0,
        ),
        L.StateAbortItemLayout(
            L.StateTransitionLeaseLayout(
                prepared[1].transition.engine_epoch,
                prepared[1].transition.slot,
                prepared[1].transition.generation,
            ),
            0,
            0,
        ),
    )
    error = ctypes.create_string_buffer(256)
    assert raw(pool._handle, items, 2, error, 256) == -4
    stats = pool.stats()
    assert stats.quarantined_slots == 2
    assert stats.free_slots == 2
    assert stats.reserved_slots == stats.pending_transitions == 0
    with pytest.raises(FailStopped, match="status -4"):
        pool.prepare_batch(((101, None),))
    pool.close()


def test_state_python_client_rejects_integer_truncation(ffi_library):
    pool = _pool(ffi_library)
    initial = pool.prepare_batch(((101, None),))[0]
    pool.submit_batch((_receipt(initial),))
    with pytest.raises(ManagerError, match="outside uint64_t"):
        pool.complete_batch(
            StateCompletionReceipt(11, 1 << 64, 1), (initial.transition,)
        )
    publication = pool.complete_batch(
        StateCompletionReceipt(11, 3, 1), (initial.transition,)
    )[0]
    retirement = pool.retire_owners_batch(
        StateCompletionReceipt(11, 3, 2), ((101, publication.slot),)
    )[0]
    pool.acknowledge_batch((retirement,))
    pool.close()


@pytest.mark.parametrize(("status", "message"), ((0, b"bad success"), (-4, b"")))
def test_unusable_state_create_consumes_returned_handle(
    tmp_path, monkeypatch, status, message
):
    library_path = tmp_path / "hostile-state.so"
    library_path.write_bytes(b"")
    destroyed = []

    class HostileLibrary:
        @staticmethod
        def function(name):
            if name == "orbitkv_state_pool_create":
                def create(_config, out_handle, error, _error_len):
                    ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_void_p))[0] = (
                        ctypes.c_void_p(0x1234)
                    )
                    error.value = message
                    return status

                return create
            if name == "orbitkv_state_pool_destroy":
                def destroy(handle, _error, _error_len):
                    destroyed.append(int(handle.value or 0))
                    return 0

                return destroy
            raise AssertionError(name)

    monkeypatch.setattr(
        "orbitkv_sglang.ffi.state_pool.LoadedLibrary",
        lambda _path: HostileLibrary(),
    )
    with pytest.raises(FailStopped):
        _pool(library_path)
    assert destroyed == [0x1234]


__all__ = ["ffi_library"]
