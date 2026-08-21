from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]

from orbitkv_sglang.config import load_config
from orbitkv_sglang.ffi import (
    ABI_VERSION,
    STATUS_BUFFER_TOO_SMALL,
    STATUS_FAIL_STOPPED,
    CtypesManagerFactory,
)
from orbitkv_sglang.ffi.layouts import (
    FROZEN_LAYOUTS,
    RequestLeaseLayout,
    SnapshotLeaseLayout,
    ReleaseItemLayout,
    assert_frozen_layouts,
)
from orbitkv_sglang.ffi.library import ERROR_BUFFER_BYTES, EXACT_SYMBOL_ALLOWLIST
from orbitkv_sglang.ffi.manager import CtypesManager
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    BatchCompletionReceipt,
    FailStopped,
    ManagerError,
    ManagerCreateSettings,
    PrefixAttachItem,
    PrefixLease,
    PrefixLookupHint,
    PrefixPublishItem,
    PrefixSemanticKey,
    PrepareBatchItem,
    ReleaseBatchItem,
    RequestCursor,
    RequestForkItem,
    RequestView,
    RetryableConflict,
    bind_receipts,
    copy_receipts,
    reclamation_receipts,
)
from orbitkv_sglang.runtime.snapshot_shadow import (
    _decode_prepared,
    page_shadow_from_snapshot,
)


@pytest.fixture(scope="session")
def ffi_library() -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(REPOSITORY_ROOT / "crates/orbitkv-ffi/Cargo.toml"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    value = REPOSITORY_ROOT / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"
    assert value.is_file()
    return value


def _config(tmp_path: Path, library: Path, *, hybrid: bool = True) -> Any:
    classes = [
        {
            "name": "full",
            "layers": [0],
            "retention": "full",
            "bytes_per_token_per_layer": 128,
            "window_tokens": None,
        }
    ]
    if hybrid:
        classes.append(
            {
                "name": "swa",
                "layers": [1],
                "retention": "sliding",
                "bytes_per_token_per_layer": 128,
                "window_tokens": 18,
            }
        )
    plan = tmp_path / ("hybrid-plan.json" if hybrid else "full-plan.json")
    plan.write_text(json.dumps({"page_tokens": 16, "classes": classes}))
    return load_config(
        {"ORBITKV_PLAN": str(plan), "ORBITKV_LIBRARY": str(library)}
    )


def _manager(
    tmp_path: Path,
    library: Path,
    *,
    hybrid: bool = True,
    maximum_requests: int = 16,
) -> tuple[Any, CtypesManager]:
    config = _config(tmp_path, library, hybrid=hybrid)
    arenas = tuple(
        ArenaRegistration(
            item.class_id,
            item.pool_id,
            item.backend_domain,
            64,
            0,
        )
        for item in config.classes
    )
    value = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(
            maximum_requests=maximum_requests,
            maximum_operations=4,
            maximum_prefixes=maximum_requests,
            maximum_reclamations=64 * len(arenas),
            maximum_step_tokens=64,
        ),
        arenas,
    )
    assert isinstance(value, CtypesManager)
    return config, value


@pytest.mark.parametrize(
    ("status", "message"),
    ((0, b"post-commit create error"), (STATUS_FAIL_STOPPED, b"")),
)
def test_unusable_create_result_consumes_any_returned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    message: bytes,
) -> None:
    library_path = tmp_path / "hostile-create.so"
    library_path.write_bytes(b"")
    config = _config(tmp_path, library_path, hybrid=False)
    destroyed: list[int] = []

    class HostileCdll:
        @staticmethod
        def orbitkv_manager_create(*args: Any) -> int:
            out_handle = args[5]
            error = args[6]
            ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
                0x1234
            )
            error.value = message
            return status

        @staticmethod
        def orbitkv_manager_destroy(handle: ctypes.c_void_p, *_args: Any) -> int:
            destroyed.append(int(handle.value or 0))
            return 0

    class HostileLibrary:
        cdll = HostileCdll()

    monkeypatch.setattr(
        "orbitkv_sglang.ffi.manager.LoadedLibrary", lambda _path: HostileLibrary()
    )
    arena = ArenaRegistration(
        config.classes[0].class_id,
        config.classes[0].pool_id,
        config.classes[0].backend_domain,
        8,
        0,
    )
    with pytest.raises(FailStopped, match="create outcome is unusable"):
        CtypesManagerFactory().create(
            config,
            ManagerCreateSettings(2, 1, 2, 8, 16),
            (arena,),
        )
    assert destroyed == [0x1234]


@pytest.mark.parametrize(
    "operation", ("fork", "attach", "release", "publish-release", "evict")
)
def test_cold_preflight_rejects_unbounded_native_counts_before_allocation(
    tmp_path: Path,
    ffi_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    _config_value, manager = _manager(tmp_path, ffi_library, hybrid=False)
    source, target = manager.request_acquire_batch(2)
    key = PrefixSemanticKey(b"n" * 32, b"d" * 32, 16)
    prefix = PrefixLease(source.request.engine_epoch, 0, 1)
    pointer_indexes = {
        "fork": (5, 8),
        "attach": (5, 8),
        "release": (5, 8, 11),
        "publish-release": (5, 8, 11),
        "evict": (5, 8),
    }[operation]
    symbol = {
        "fork": "orbitkv_manager_request_fork_batch",
        "attach": "orbitkv_manager_prefix_attach_batch",
        "release": "orbitkv_manager_release_batch",
        "publish-release": "orbitkv_manager_prefix_publish_release_batch",
        "evict": "orbitkv_manager_prefix_evict_batch",
    }[operation]
    calls: list[int] = []

    def hostile_preflight(*args: Any) -> int:
        calls.append(1)
        for position, pointer_index in enumerate(pointer_indexes):
            value = 1 if position == 0 else (1 << 32) - 1
            ctypes.cast(
                args[pointer_index], ctypes.POINTER(ctypes.c_uint32)
            )[0] = value
        return STATUS_BUFFER_TOO_SMALL

    allocations: list[str] = []

    def unexpected_allocation(*_args: Any, **_kwargs: Any) -> Any:
        allocations.append("attempted")
        raise AssertionError("cold allocation happened before count validation")

    setattr(manager._library, symbol, hostile_preflight)
    monkeypatch.setattr(
        "orbitkv_sglang.ffi.manager.cold_materialization", unexpected_allocation
    )
    monkeypatch.setattr(
        "orbitkv_sglang.ffi.manager.cold_reclamation", unexpected_allocation
    )
    monkeypatch.setattr("orbitkv_sglang.ffi.manager.array", unexpected_allocation)

    with pytest.raises(FailStopped, match="cold output bound"):
        if operation == "fork":
            manager.request_fork_batch(
                (
                    RequestForkItem(
                        source.request,
                        source.snapshot,
                        target.request,
                        target.snapshot,
                    ),
                )
            )
        elif operation == "attach":
            manager.prefix_attach_batch(
                (
                    PrefixAttachItem(
                        target.request,
                        target.snapshot,
                        PrefixLookupHint(key, prefix, 0),
                    ),
                )
            )
        elif operation == "release":
            manager.release_batch((ReleaseBatchItem(source.request, source.snapshot),))
        elif operation == "publish-release":
            manager.prefix_publish_release_batch(
                (PrefixPublishItem(source.request, source.snapshot, key),)
            )
        else:
            manager.prefix_evict_batch((prefix,))

    assert calls == [1]
    assert allocations == []
    manager.destroy()


def _commit(
    manager: CtypesManager,
    config: Any,
    view: RequestView,
    target: int,
    *,
    completion_value: int,
    cursor: RequestCursor | None = None,
) -> tuple[RequestView, Any, Any]:
    prepared = manager.prepare_batch(
        (PrepareBatchItem(view.request, view.snapshot, target),)
    )[0]
    request_cursor = RequestCursor.from_view(view) if cursor is None else cursor
    _plan, pages = _decode_prepared(
        request_cursor, prepared, manager.arenas_by_class, config
    )
    submitted = manager.submit_batch(
        (
            (
                prepared.step,
                bind_receipts(prepared, pages, manager.arenas_by_class),
                copy_receipts(prepared),
            ),
        )
    )[0]
    completion = manager.complete_batch(
        BatchCompletionReceipt(
            view.request.engine_epoch, 1, completion_value
        ),
        (submitted.submission,),
    ).completions[0]
    return (
        RequestView(
            completion.request,
            completion.published_snapshot,
            completion.published_view_version,
            completion.published_boundary,
            completion.resident_count,
        ),
        prepared,
        completion,
    )


def _release_all(manager: CtypesManager, views: tuple[RequestView, ...]) -> None:
    output = manager.release_batch(
        tuple(ReleaseBatchItem(item.request, item.snapshot) for item in views)
    )
    if output.retirements:
        manager.acknowledge_reclamations_batch(
            reclamation_receipts(output.retirements)
        )
    manager.recycle_requests_batch(tuple(item.request for item in views))


def test_frozen_layouts_and_exact_symbol_allowlist(ffi_library: Path) -> None:
    assert ABI_VERSION == 6
    assert len(FROZEN_LAYOUTS) == 43
    assert_frozen_layouts()
    output = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(ffi_library)], text=True
    )
    exported = {
        line.split()[-1]
        for line in output.splitlines()
        if line.split() and line.split()[-1].startswith("orbitkv_")
    }
    assert exported == EXACT_SYMBOL_ALLOWLIST


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("hybrid", [False, True])
def test_real_b2_b4_lifecycle_has_exact_zero_final_census(
    tmp_path: Path, ffi_library: Path, batch_size: int, hybrid: bool
) -> None:
    config, manager = _manager(tmp_path, ffi_library, hybrid=hybrid)
    acquired = manager.request_acquire_batch(batch_size)
    current = tuple(
        _commit(
            manager,
            config,
            view,
            18,
            completion_value=index + 1,
        )[0]
        for index, view in enumerate(acquired)
    )
    _release_all(manager, current)
    stats = manager.stats()
    assert stats.free_pages == 64 * len(config.classes)
    assert stats.active_requests == stats.active_snapshots == 0
    assert stats.pending_reclamations == 0
    assert stats.total_request_page_refs == stats.total_prefix_page_refs == 0
    assert stats.total_reader_pins == 0
    manager.destroy()


def test_release_short_buffer_is_zero_mutation_and_reports_exact_census(
    tmp_path: Path, ffi_library: Path
) -> None:
    config, manager = _manager(tmp_path, ffi_library)
    view, _prepared, _completion = _commit(
        manager,
        config,
        manager.request_acquire_batch(1)[0],
        18,
        completion_value=1,
    )
    before = manager.stats()
    raw = ReleaseItemLayout(
        RequestLeaseLayout(
            view.request.engine_epoch, view.request.slot, view.request.generation
        ),
        SnapshotLeaseLayout(
            view.snapshot.engine_epoch, view.snapshot.slot, view.snapshot.generation
        ),
    )
    counts = [ctypes.c_uint32() for _ in range(3)]
    error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
    status = int(
        manager._library.orbitkv_manager_release_batch(
            manager._handle,
            ctypes.byref(raw),
            1,
            None,
            0,
            ctypes.byref(counts[0]),
            None,
            0,
            ctypes.byref(counts[1]),
            None,
            0,
            ctypes.byref(counts[2]),
            error,
            len(error),
        )
    )
    assert status == STATUS_BUFFER_TOO_SMALL
    assert not error.value
    assert tuple(item.value for item in counts) == (1, 4, 4)
    assert manager.stats() == before
    _release_all(manager, (view,))
    manager.destroy()


def test_retryable_duplicate_prefix_is_typed_and_zero_mutation(
    tmp_path: Path, ffi_library: Path
) -> None:
    config, manager = _manager(tmp_path, ffi_library)
    view, _prepared, _completion = _commit(
        manager,
        config,
        manager.request_acquire_batch(1)[0],
        32,
        completion_value=1,
    )
    key = PrefixSemanticKey(b"n" * 32, b"d" * 32, 32)
    published = manager.prefix_publish_batch(
        (PrefixPublishItem(view.request, view.snapshot, key),)
    )[0]
    before_stats = manager.stats()
    before_arenas = manager.arena_stats()
    with pytest.raises(RetryableConflict):
        manager.prefix_publish_batch(
            (PrefixPublishItem(view.request, view.snapshot, key),)
        )
    assert manager.stats() == before_stats
    assert manager.arena_stats() == before_arenas
    _release_all(manager, (view,))
    evicted = manager.prefix_evict_batch((published.prefix,))
    if evicted.retirements:
        manager.acknowledge_reclamations_batch(
            reclamation_receipts(evicted.retirements)
        )
    manager.prefix_recycle_batch((published.prefix,))
    manager.destroy()


def test_b4_fork_joint_cow_and_prefix_reference_lifecycle(
    tmp_path: Path, ffi_library: Path
) -> None:
    config, manager = _manager(tmp_path, ffi_library)
    acquired = manager.request_acquire_batch(5)
    source, _prepared, _completion = _commit(
        manager, config, acquired[0], 18, completion_value=1
    )
    forked = manager.request_fork_batch(
        tuple(
            RequestForkItem(
                source.request,
                source.snapshot,
                target.request,
                target.snapshot,
            )
            for target in acquired[1:]
        )
    )
    assert len(forked) == 4
    assert all(len(item.target.pages) == 4 for item in forked)
    source_release = manager.release_batch(
        (ReleaseBatchItem(source.request, source.snapshot),)
    )
    assert not source_release.retirements
    manager.recycle_requests_batch((source.request,))

    cursor = RequestCursor.from_view(forked[0].target.view)
    for page in forked[0].target.pages:
        shadow = page_shadow_from_snapshot(
            cursor.lease, page, manager.arenas_by_class[page.class_id]
        )
        cursor.pages[(shadow.class_id, shadow.logical_ordinal)] = shadow
    extended, prepared, completion = _commit(
        manager,
        config,
        forked[0].target.view,
        19,
        completion_value=2,
        cursor=cursor,
    )
    assert [item.kind for item in prepared.tail_actions] == [2, 2]
    assert len(prepared.copy_intents) == 2
    assert [(item.action, item.reason) for item in completion.detached] == [
        (2, 2),
        (2, 2),
    ]

    prefix_source, _prepared, _completion = _commit(
        manager,
        config,
        manager.request_acquire_batch(1)[0],
        32,
        completion_value=3,
    )
    key = PrefixSemanticKey(b"p" * 32, b"k" * 32, 32)
    publication = manager.prefix_publish_batch(
        (PrefixPublishItem(prefix_source.request, prefix_source.snapshot, key),)
    )[0]
    hint = manager.prefix_lookup_batch((key,))[0]
    empty = manager.request_acquire_batch(1)[0]
    attached = manager.prefix_attach_batch(
        (PrefixAttachItem(empty.request, empty.snapshot, hint),)
    )[0]
    assert attached.target.view.boundary == 32
    assert len(attached.target.pages) == publication.resident_count

    remaining = (extended,) + tuple(
        item.target.view for item in forked[1:]
    ) + (prefix_source, attached.target.view)
    _release_all(manager, remaining)
    evicted = manager.prefix_evict_batch((publication.prefix,))
    assert len(evicted.retirements) == publication.resident_count
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(evicted.retirements)
    )
    manager.prefix_recycle_batch((publication.prefix,))
    assert manager.stats().free_pages == 128
    manager.destroy()


def test_unknown_python_call_outcome_permanently_poisons_handle(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager = _manager(tmp_path, ffi_library, hybrid=False)

    def lost_return(*_args: Any) -> int:
        raise OSError("simulated lost FFI return")

    with pytest.raises(FailStopped, match="outcome is unknown"):
        manager._call("fault injection", lost_return, manager._handle)
    with pytest.raises(FailStopped, match="poisoned"):
        manager.request_acquire_batch(1)
    with pytest.raises(FailStopped, match="poisoned"):
        manager.arena_stats()
    # Aggregate stats and a destroy attempt are the only allowed follow-ups.
    assert manager.stats().free_pages == 64
    manager.destroy()


def test_lost_destroy_return_invalidates_the_consumed_native_pointer(
    tmp_path: Path, ffi_library: Path
) -> None:
    _config_value, manager = _manager(tmp_path, ffi_library, hybrid=False)
    original = manager._library.orbitkv_manager_destroy

    def lost_destroy(*args: Any) -> int:
        status = int(original(*args))
        assert status == 0
        raise OSError("simulated lost destroy return")

    manager._library.orbitkv_manager_destroy = lost_destroy
    with pytest.raises(FailStopped, match="outcome is unknown"):
        manager.destroy()
    assert not manager._handle or not manager._handle.value
    with pytest.raises(ManagerError, match="handle is closed"):
        manager.stats()
    with pytest.raises(ManagerError, match="handle is closed"):
        manager.arena_stats()
    manager.destroy()


@pytest.mark.parametrize("status", [STATUS_FAIL_STOPPED, -999])
def test_fail_stopped_or_unknown_native_status_permanently_poisons_handle(
    tmp_path: Path, ffi_library: Path, status: int
) -> None:
    _config_value, manager = _manager(tmp_path, ffi_library, hybrid=False)

    def hostile_status(*_args: Any) -> int:
        return status

    with pytest.raises(FailStopped, match="poisoned"):
        manager._call("fault injection", hostile_status, manager._handle)
    with pytest.raises(FailStopped, match="poisoned"):
        manager.request_acquire_batch(1)
    with pytest.raises(FailStopped, match="poisoned"):
        manager.arena_stats()
    assert manager.stats().free_pages == 64
    manager.destroy()
