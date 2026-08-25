from __future__ import annotations

import ctypes
import json
import os
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
    ClassTokenDispositionUpdate,
    FailStopped,
    ManagerError,
    ManagerCreateSettings,
    PrefixAttachItem,
    PrefixLease,
    PrefixLookupHint,
    PrefixPublishItem,
    PrefixSemanticKey,
    PrepareBatchItem,
    PrepareRelocationItem,
    RelocationPolicy,
    ReleaseBatchItem,
    RequestCursor,
    RequestForkItem,
    RequestView,
    RetryableConflict,
    bind_receipts,
    CLASS_LOWERING_PACKED,
    copy_receipts,
    DETACHED_COPY_ON_WRITE,
    DETACHED_REPLACE,
    reclamation_receipts,
    relocation_copy_receipts,
    TAIL_COPY_ON_WRITE,
    TAIL_IN_PLACE,
    TokenDisposition,
    TokenDispositionBatchItem,
    TokenDispositionKind,
    TokenViewQuery,
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
    target = os.environ.get("CARGO_TARGET_DIR")
    target_dir = (
        Path(target).resolve()
        if target is not None
        else REPOSITORY_ROOT / "crates/orbitkv-ffi/target"
    )
    value = target_dir / "release/liborbitkv_ffi.so"
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
    assert ABI_VERSION == 8
    assert len(FROZEN_LAYOUTS) == 73
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


def test_abi8_full_evacuation_relocates_live_tokens_and_reclaims_sources(
    tmp_path: Path, ffi_library: Path
) -> None:
    config, manager = _manager(tmp_path, ffi_library, hybrid=False)
    current, _prepared, _completion = _commit(
        manager,
        config,
        manager.request_acquire_batch(1)[0],
        48,
        completion_value=1,
    )
    before = manager.token_views_batch(
        (TokenViewQuery(current.request, current.snapshot, 0, current.boundary),)
    )[0]
    assert len(before.placements) == 48
    assert all(
        item.disposition.kind is TokenDispositionKind.RETAINED
        and item.location is not None
        for item in before.placements
    )

    victims = tuple(range(8, 16)) + tuple(range(24, 32)) + tuple(range(40, 48))
    current = manager.mark_token_dispositions_batch(
        (
            TokenDispositionBatchItem(
                current.request,
                current.snapshot,
                tuple(
                    ClassTokenDispositionUpdate(
                        0,
                        token_id,
                        TokenDisposition(
                            TokenDispositionKind.POLICY_EVICTED,
                            policy_or_proof_id=7,
                            version=1,
                            quality_contract=99,
                        ),
                    )
                    for token_id in victims
                ),
            ),
        )
    )[0]
    naive = manager.token_views_batch(
        (TokenViewQuery(current.request, current.snapshot, 0, current.boundary),)
    )[0]
    assert sum(
        item.disposition.kind is TokenDispositionKind.POLICY_EVICTED
        for item in naive.placements
    ) == 24
    assert all(naive.placements[token_id].location is not None for token_id in victims)

    prepared = manager.prepare_relocation_batch(
        (
            PrepareRelocationItem(
                current.request,
                current.snapshot,
                0,
                RelocationPolicy(
                    maximum_source_pages=3,
                    evacuation_headroom_pages=2,
                    fragmentation_threshold_milli=250,
                    full_evacuation=True,
                ),
            ),
        )
    )[0]
    assert len(prepared.source_pages) == 3
    assert len(prepared.destination_pages) == 2
    assert len(prepared.moves) == 24
    assert prepared.projected_reclaimed_pages == 1
    submitted = manager.submit_relocation_batch(
        ((prepared.relocation, relocation_copy_receipts(prepared)),)
    )[0]
    completed = manager.complete_relocation_batch(
        BatchCompletionReceipt(current.request.engine_epoch, 2, 2),
        (submitted.relocation,),
    )
    assert len(completed.publications) == 1
    assert len(completed.retirements) == 3
    current = completed.publications[0]
    packed = manager.token_views_batch(
        (TokenViewQuery(current.request, current.snapshot, 0, current.boundary),)
    )[0]
    assert [
        item.token_id for item in packed.placements if item.location is not None
    ] == [token_id for token_id in range(48) if token_id not in victims]
    assert all(packed.placements[token_id].location is None for token_id in victims)

    manager.acknowledge_reclamations_batch(
        reclamation_receipts(completed.retirements)
    )
    prepared_append = manager.prepare_batch(
        (PrepareBatchItem(current.request, current.snapshot, 49),)
    )[0]
    assert prepared_append.class_lowerings[0].flags == CLASS_LOWERING_PACKED
    assert prepared_append.tail_actions[0].logical_ordinal == 1
    assert prepared_append.tail_actions[0].valid_token_count == 8
    assert not prepared_append.write_intents
    submitted_append = manager.submit_batch(
        (
            (
                prepared_append.step,
                bind_receipts(prepared_append, (), manager.arenas_by_class),
                copy_receipts(prepared_append),
            ),
        )
    )[0]
    append_completion = manager.complete_batch(
        BatchCompletionReceipt(current.request.engine_epoch, 3, 3),
        (submitted_append.submission,),
    ).completions[0]
    current = RequestView(
        append_completion.request,
        append_completion.published_snapshot,
        append_completion.published_view_version,
        append_completion.published_boundary,
        append_completion.resident_count,
    )
    appended = manager.token_views_batch(
        (TokenViewQuery(current.request, current.snapshot, 0, current.boundary),)
    )[0]
    assert appended.placements[48].location is not None
    assert appended.placements[48].location.offset == 8

    released = manager.release_batch(
        (ReleaseBatchItem(current.request, current.snapshot),)
    )
    assert tuple(
        (item.token_begin, item.token_end_exclusive)
        for item in released.retirements
    ) == ((0, 16), (16, 25))
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(released.retirements)
    )
    manager.recycle_requests_batch((current.request,))
    stats = manager.stats()
    assert stats.free_pages == 64
    assert stats.active_requests == stats.active_snapshots == 0
    assert stats.pending_reclamations == 0
    assert stats.total_request_page_refs == stats.total_prefix_page_refs == 0
    assert stats.total_reader_pins == 0
    manager.destroy()


def test_abi8_packed_partial_b4_fork_cow_append_preserves_token_locations(
    tmp_path: Path, ffi_library: Path
) -> None:
    config, manager = _manager(
        tmp_path, ffi_library, hybrid=False, maximum_requests=5
    )
    acquired = manager.request_acquire_batch(5)
    source, _prepared, _completion = _commit(
        manager, config, acquired[0], 48, completion_value=1
    )
    victims = tuple(range(8, 16)) + tuple(range(24, 32)) + tuple(range(40, 48))
    source = manager.mark_token_dispositions_batch(
        (
            TokenDispositionBatchItem(
                source.request,
                source.snapshot,
                tuple(
                    ClassTokenDispositionUpdate(
                        0,
                        token_id,
                        TokenDisposition(
                            TokenDispositionKind.POLICY_EVICTED,
                            policy_or_proof_id=17,
                            version=1,
                            quality_contract=99,
                        ),
                    )
                    for token_id in victims
                ),
            ),
        )
    )[0]
    prepared = manager.prepare_relocation_batch(
        (
            PrepareRelocationItem(
                source.request,
                source.snapshot,
                0,
                RelocationPolicy(
                    maximum_source_pages=3,
                    evacuation_headroom_pages=2,
                    fragmentation_threshold_milli=250,
                    full_evacuation=True,
                ),
            ),
        )
    )[0]
    submitted = manager.submit_relocation_batch(
        ((prepared.relocation, relocation_copy_receipts(prepared)),)
    )[0]
    relocated = manager.complete_relocation_batch(
        BatchCompletionReceipt(source.request.engine_epoch, 2, 2),
        (submitted.relocation,),
    )
    source = relocated.publications[0]
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(relocated.retirements)
    )
    source = manager.mark_token_dispositions_batch(
        (
            TokenDispositionBatchItem(
                source.request,
                source.snapshot,
                (
                    ClassTokenDispositionUpdate(
                        0,
                        35,
                        TokenDisposition(
                            TokenDispositionKind.POLICY_EVICTED,
                            policy_or_proof_id=91,
                            version=2,
                            quality_contract=101,
                        ),
                    ),
                ),
            ),
        )
    )[0]
    source_tokens = manager.token_views_batch(
        (TokenViewQuery(source.request, source.snapshot, 0, source.boundary),)
    )[0]
    assert source_tokens.placements[35].location is not None
    assert (
        source_tokens.placements[35].disposition.kind
        is TokenDispositionKind.POLICY_EVICTED
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
    assert all(item.source == source.request for item in forked)
    assert all(item.target.view.boundary == 48 for item in forked)
    assert all(item.target.view.resident_count == 2 for item in forked)
    assert all(
        tuple(
            (page.logical_ordinal, page.valid_token_count, page.visible_token_count)
            for page in item.target.pages
        )
        == ((0, 16, 16), (1, 8, 8))
        for item in forked
    )
    target_tokens = manager.token_views_batch(
        tuple(
            TokenViewQuery(
                item.target.view.request,
                item.target.view.snapshot,
                0,
                item.target.view.boundary,
            )
            for item in forked
        )
    )
    assert all(view.placements == source_tokens.placements for view in target_tokens)
    assert all(
        view.view_version == item.target.view.view_version
        for view, item in zip(target_tokens, forked, strict=True)
    )

    old_tail = forked[0].target.pages[1]
    cursors = []
    for item in forked:
        cursor = RequestCursor.from_view(item.target.view)
        cursor.layout_boundaries[0] = 24
        for page in item.target.pages:
            shadow = page_shadow_from_snapshot(
                cursor.lease, page, manager.arenas_by_class[page.class_id]
            )
            cursor.pages[(shadow.class_id, shadow.logical_ordinal)] = shadow
        cursors.append(cursor)

    prepared_cow = manager.prepare_batch(
        tuple(
            PrepareBatchItem(
                item.target.view.request, item.target.view.snapshot, 49
            )
            for item in forked
        )
    )
    decoded_cow = tuple(
        _decode_prepared(cursor, prepared, manager.arenas_by_class, config)[1]
        for cursor, prepared in zip(cursors, prepared_cow, strict=True)
    )
    assert all(
        prepared.class_lowerings[0].flags == CLASS_LOWERING_PACKED
        and len(prepared.tail_actions) == 1
        and prepared.tail_actions[0].kind == TAIL_COPY_ON_WRITE
        and prepared.tail_actions[0].valid_token_count == 8
        and prepared.tail_actions[0].source == old_tail.page
        and len(prepared.copy_intents) == 1
        and prepared.copy_intents[0].token_count == 8
        and prepared.copy_intents[0].source_token_offset == 0
        and prepared.copy_intents[0].destination_token_offset == 0
        and prepared.copy_intents[0].source == old_tail.page
        and prepared.copy_intents[0].source_backend_index
        == old_tail.backend_index
        and not prepared.write_intents
        for prepared in prepared_cow
    )
    assert len(
        {prepared.copy_intents[0].destination for prepared in prepared_cow}
    ) == len(forked)

    submitted_cow = manager.submit_batch(
        tuple(
            (
                prepared.step,
                bind_receipts(prepared, pages, manager.arenas_by_class),
                copy_receipts(prepared),
            )
            for prepared, pages in zip(prepared_cow, decoded_cow, strict=True)
        )
    )
    completed_cow = manager.complete_batch(
        BatchCompletionReceipt(source.request.engine_epoch, 3, 3),
        tuple(item.submission for item in submitted_cow),
    )
    assert not completed_cow.retirements
    assert all(
        len(completion.detached) == 1
        and completion.detached[0].old == old_tail.page
        and completion.detached[0].old_backend_index == old_tail.backend_index
        and completion.detached[0].replacement
        == prepared.copy_intents[0].destination
        and completion.detached[0].replacement_backend_index
        == prepared.copy_intents[0].destination_backend_index
        and completion.detached[0].logical_ordinal == 1
        and completion.detached[0].token_begin == 16
        and completion.detached[0].token_end_exclusive == 24
        and completion.detached[0].action == DETACHED_REPLACE
        and completion.detached[0].reason == DETACHED_COPY_ON_WRITE
        for prepared, completion in zip(
            prepared_cow, completed_cow.completions, strict=True
        )
    )

    cow_views = tuple(
        RequestView(
            completion.request,
            completion.published_snapshot,
            completion.published_view_version,
            completion.published_boundary,
            completion.resident_count,
        )
        for completion in completed_cow.completions
    )
    cow_tokens = manager.token_views_batch(
        tuple(
            TokenViewQuery(view.request, view.snapshot, 0, view.boundary)
            for view in cow_views
        )
    )
    for prepared, view in zip(prepared_cow, cow_tokens, strict=True):
        destination = prepared.copy_intents[0].destination
        destination_backend = prepared.copy_intents[0].destination_backend_index
        for before, after in zip(
            source_tokens.placements, view.placements[:48], strict=True
        ):
            assert after.token_id == before.token_id
            assert after.disposition == before.disposition
            if (
                before.location is not None
                and before.location.page == old_tail.page
                and before.location.backend_index == old_tail.backend_index
            ):
                assert after.location is not None
                assert after.location.page == destination
                assert after.location.backend_index == destination_backend
                assert after.location.offset == before.location.offset
            else:
                assert after.location == before.location
        appended = view.placements[48]
        assert appended.token_id == 48
        assert appended.disposition.kind is TokenDispositionKind.RETAINED
        assert appended.location is not None
        assert appended.location.page == destination
        assert appended.location.backend_index == destination_backend
        assert appended.location.offset == 8

    source_after_cow = manager.token_views_batch(
        (TokenViewQuery(source.request, source.snapshot, 0, source.boundary),)
    )[0]
    assert source_after_cow.placements == source_tokens.placements

    second_cursor = RequestCursor.from_view(cow_views[0])
    second_cursor.layout_boundaries[0] = 25
    for key, shadow in cursors[0].pages.items():
        second_cursor.pages[key] = shadow
    cow_tail = decoded_cow[0][0]
    second_cursor.pages[(cow_tail.class_id, cow_tail.logical_ordinal)] = cow_tail
    extended, second_prepared, second_completion = _commit(
        manager,
        config,
        cow_views[0],
        50,
        completion_value=4,
        cursor=second_cursor,
    )
    assert second_prepared.class_lowerings[0].flags == CLASS_LOWERING_PACKED
    assert second_prepared.tail_actions[0].kind == TAIL_IN_PLACE
    assert second_prepared.tail_actions[0].valid_token_count == 9
    assert second_prepared.tail_actions[0].source == cow_tail.page
    assert second_prepared.tail_actions[0].destination == cow_tail.page
    assert not second_prepared.copy_intents
    assert not second_prepared.write_intents
    assert not second_completion.detached
    extended_tokens = manager.token_views_batch(
        (TokenViewQuery(extended.request, extended.snapshot, 0, extended.boundary),)
    )[0]
    assert extended_tokens.placements[:49] == cow_tokens[0].placements
    assert extended_tokens.placements[49].token_id == 49
    assert (
        extended_tokens.placements[49].disposition.kind
        is TokenDispositionKind.RETAINED
    )
    assert extended_tokens.placements[49].location is not None
    assert extended_tokens.placements[49].location.page == cow_tail.page
    assert extended_tokens.placements[49].location.backend_index == (
        prepared_cow[0].copy_intents[0].destination_backend_index
    )
    assert extended_tokens.placements[49].location.offset == 9

    source_release = manager.release_batch(
        (ReleaseBatchItem(source.request, source.snapshot),)
    )
    assert tuple(
        (
            item.page,
            item.backend_index,
            item.logical_ordinal,
            item.token_begin,
            item.token_end_exclusive,
            item.completion_domain,
            item.completion_value,
        )
        for item in source_release.retirements
    ) == ((old_tail.page, old_tail.backend_index, 1, 16, 24, 3, 3),)
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(source_release.retirements)
    )
    manager.recycle_requests_batch((source.request,))

    sibling_views = cow_views[1:]
    sibling_release = manager.release_batch(
        tuple(ReleaseBatchItem(view.request, view.snapshot) for view in sibling_views)
    )
    assert len(sibling_release.retirements) == 3
    assert all(
        item.logical_ordinal == 1
        and item.token_begin == 16
        and item.token_end_exclusive == 25
        and item.completion_domain == 3
        and item.completion_value == 3
        for item in sibling_release.retirements
    )
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(sibling_release.retirements)
    )
    manager.recycle_requests_batch(tuple(view.request for view in sibling_views))

    final_release = manager.release_batch(
        (ReleaseBatchItem(extended.request, extended.snapshot),)
    )
    assert tuple(
        (item.logical_ordinal, item.token_begin, item.token_end_exclusive)
        for item in final_release.retirements
    ) == ((0, 0, 16), (1, 16, 26))
    assert len({item.reclamation for item in final_release.retirements}) == 2
    manager.acknowledge_reclamations_batch(
        reclamation_receipts(final_release.retirements)
    )
    manager.recycle_requests_batch((extended.request,))
    stats = manager.stats()
    assert stats.free_pages == 64
    assert stats.active_requests == stats.active_snapshots == 0
    assert stats.active_prefixes == stats.evicted_prefixes == 0
    assert stats.prepared_steps == stats.submitted_steps == 0
    assert stats.reserved_pages == stats.writing_pages == stats.active_pages == 0
    assert stats.retiring_pages == stats.quarantined_pages == 0
    assert stats.exhausted_pages == 0
    assert stats.pending_reclamations == 0
    assert stats.total_request_page_refs == stats.total_prefix_page_refs == 0
    assert stats.total_reader_pins == 0
    manager.destroy()


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
