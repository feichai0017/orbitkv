from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

import orbitkv_sglang.ffi as public_ffi
from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.ffi import layouts as L
from orbitkv_sglang.ffi import session as session_ffi
from orbitkv_sglang.ffi import session_types as session_dtos
from orbitkv_sglang.ffi.library import (
    EXACT_SYMBOL_ALLOWLIST, FUNCTION_SPECS, STATUS_BUFFER_TOO_SMALL,
    STATUS_FAIL_STOPPED, STATUS_INVALID_ARGUMENT, STATUS_MANAGER_ERROR,
    STATUS_OK, STATUS_RETRYABLE_CONFLICT, WIRE_VERSION,
)
from orbitkv_sglang.ffi.session import CtypesRuntimeSession
from orbitkv_sglang.ffi.session_types import (
    EngineAppendIntent, EngineBatchId, EngineBindEvidence,
    EngineCompletionEvidence, EngineControlId, EngineCopyEvidence,
    EnginePrefixId, EnginePublicationEvidence, EnginePublicationId,
    EngineReleaseDisposition,
    EngineReleaseEvidence, EngineReleaseId, EngineReleaseOutcome,
    EngineRequestId, EngineRetirementEvidence,
    EngineStepExecutionEvidence, ExecutionEvidence, retirement_evidence,
)
from orbitkv_sglang.runtime import (
    ArenaRegistration, CacheSharingPolicy, FailStopped, ManagerCreateSettings, ManagerError,
    PageLease, RetryableConflict, SessionCreateSettings, TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
)
from ffi_test_support import ffi_library


__all__ = ["ffi_library"]


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def ffi_test_support_library(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    target_dir = tmp_path_factory.mktemp("orbitkv-ffi-test-support-target")
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--features",
            "test-support",
            "--manifest-path",
            str(REPOSITORY_ROOT / "core/ffi/Cargo.toml"),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    library = target_dir / "release/liborbitkv_ffi.so"
    assert library.is_file()
    return library


SESSION_LAYOUTS = {
    L.SessionCreateConfigLayout: (40, ("manager", "cache_sharing_policy", "reserved")),
    L.SessionBatchIdLayout: (16, ("session_epoch", "sequence")),
    L.SessionPublicationIdLayout: (16, ("session_epoch", "sequence")),
    L.SessionReleaseIdLayout: (16, ("session_epoch", "sequence")),
    L.SessionRequestViewLayout: (
        32,
        ("request_id", "view_version", "boundary", "resident_count", "reserved"),
    ),
    L.SessionAppendIntentLayout: (16, ("request_id", "target_boundary")),
    L.SessionPreparedStepLayout: (
        72,
        (
            "request_id",
            "base_view_version",
            "target_view_version",
            "previous_boundary",
            "target_boundary",
            "class_offset",
            "class_count",
            "tail_offset",
            "tail_count",
            "copy_offset",
            "copy_count",
            "write_offset",
            "write_count",
        ),
    ),
    L.SessionStepExecutionEvidenceLayout: (
        32,
        (
            "request_id",
            "bind_offset",
            "bind_count",
            "copy_offset",
            "copy_count",
            "reserved",
        ),
    ),
    L.SessionBindEvidenceLayout: (
        48,
        (
            "page",
            "backend_domain",
            "mapped",
            "writable",
            "reserved",
            "backend_index",
        ),
    ),
    L.SessionCopyEvidenceLayout: (
        104,
        (
            "class_id",
            "backend_domain",
            "token_count",
            "source_token_offset",
            "destination_token_offset",
            "observed",
            "copied",
            "ordered_before_writes",
            "reserved8",
            "reserved32",
            "source",
            "destination",
            "source_backend_index",
            "destination_backend_index",
        ),
    ),
    L.SessionStepAbortEvidenceLayout: (
        16,
        ("request_id", "backend_unobserved", "reserved"),
    ),
    L.SessionCompletionEvidenceLayout: (
        24,
        ("completion_domain", "completion_value", "confirmed", "reserved"),
    ),
    L.SessionStepPublicationLayout: (
        40,
        (
            "request_id",
            "view_version",
            "boundary",
            "resident_count",
            "detached_offset",
            "detached_count",
            "reserved",
        ),
    ),
    L.SessionRetirementLayout: (
        88,
        (
            "page",
            "class_id",
            "backend_domain",
            "reserved32",
            "logical_ordinal",
            "backend_index",
            "token_begin",
            "token_end_exclusive",
            "completion_domain",
            "completion_value",
        ),
    ),
    L.SessionRetirementEvidenceLayout: (
        48,
        (
            "page",
            "backend_domain",
            "acknowledged",
            "reserved8",
            "reserved32",
            "backend_index",
        ),
    ),
    L.SessionPublicationEvidenceLayout: (
        24,
        ("publication_id", "mirror_cleanup_confirmed", "reserved"),
    ),
    L.SessionReleasedRequestLayout: (
        16,
        ("request_id", "detached_offset", "detached_count"),
    ),
    L.SessionReleaseEvidenceLayout: (
        24,
        ("release_id", "mirror_cleanup_confirmed", "reserved"),
    ),
    L.SessionReleaseOutcomeLayout: (
        24,
        ("release_id", "disposition", "reserved"),
    ),
    L.SessionPrefixIdLayout: (16, ("session_epoch", "sequence")),
    L.SessionControlIdLayout: (16, ("session_epoch", "sequence")),
    L.SessionPrefixLookupLayout: (
        104,
        ("key", "candidate", "resident_count", "candidate_present", "reserved0", "reserved1"),
    ),
    L.SessionPrefixPublishItemLayout: (80, ("request_id", "key")),
    L.SessionPublishedPrefixLayout: (
        96,
        ("prefix_id", "key", "resident_count", "reserved"),
    ),
    L.SessionPublishedPrefixReleaseLayout: (
        112,
        ("request_id", "prefix_id", "key", "resident_count", "detached_offset", "detached_count", "reserved"),
    ),
    L.SessionPrefixAttachItemLayout: (
        104,
        ("target_request_id", "prefix_id", "key", "resident_count", "reserved"),
    ),
    L.SessionRequestForkItemLayout: (
        16,
        ("source_request_id", "target_request_id"),
    ),
    L.SessionControlPlanInfoLayout: (
        40,
        ("id", "kind", "reserved", "request_count", "page_count", "prefix_count", "retirement_count"),
    ),
    L.SessionMaterializedRequestLayout: (
        40,
        ("request_id", "view_version", "boundary", "resident_count", "page_offset", "page_count", "reserved"),
    ),
    L.SessionPendingAttachCancelLayout: (
        64,
        ("control_id", "request_id", "prefix_id", "view_version", "boundary", "resident_count"),
    ),
    L.SessionPendingAttachCancelOutcomeLayout: (
        64,
        ("control_id", "request_id", "prefix_id", "view_version", "boundary", "resident_count", "disposition"),
    ),
    L.SessionControlEvidenceLayout: (
        24,
        ("id", "mirror_cleanup_confirmed", "reserved"),
    ),
    L.SessionControlOutcomeLayout: (
        24,
        ("id", "disposition", "reserved"),
    ),
}

SESSION_SYMBOL_ARITIES = {
    "orbitkv_session_create": 8,
    "orbitkv_session_arena_identities": 6,
    "orbitkv_session_arena_stats": 6,
    "orbitkv_session_stats": 4,
    "orbitkv_session_acquire_requests": 8,
    "orbitkv_session_prepare_append": 21,
    "orbitkv_session_submit_execution": 10,
    "orbitkv_session_abort_prepared": 6,
    "orbitkv_session_quarantine_prepared": 4,
    "orbitkv_session_quarantine_submitted": 4,
    "orbitkv_session_complete_execution": 15,
    "orbitkv_session_confirm_publication": 6,
    "orbitkv_session_prepare_release": 15,
    "orbitkv_session_confirm_release": 7,
    "orbitkv_session_prefix_lookup_batch": 8,
    "orbitkv_session_prefix_publish_batch": 8,
    "orbitkv_session_prefix_publish_release_batch": 12,
    "orbitkv_session_prepare_prefix_attach": 6,
    "orbitkv_session_prepare_request_fork": 6,
    "orbitkv_session_prepare_prefix_evict": 6,
    "orbitkv_session_abort_control": 4,
    "orbitkv_session_commit_control": 5,
    "orbitkv_session_read_control_plan": 16,
    "orbitkv_session_cancel_pending_attach": 5,
    "orbitkv_session_finalize_pending_attach_cancel": 5,
    "orbitkv_session_confirm_control": 7,
    "orbitkv_session_quarantine_control": 4,
    "orbitkv_session_destroy": 3,
}

PUBLIC_DTO_NAMES = (
    "EngineBatchId",
    "EnginePublicationId",
    "EngineReleaseId",
    "EnginePrefixId",
    "EngineControlId",
    "EngineRequestView",
    "EngineAppendIntent",
    "EngineStepPlan",
    "EngineBatchPlan",
    "EngineBindEvidence",
    "EngineCopyEvidence",
    "EngineStepExecutionEvidence",
    "ExecutionEvidence",
    "EngineStepAbortEvidence",
    "EngineBatchTicket",
    "EngineCompletionEvidence",
    "EngineRetirement",
    "EngineRetirementEvidence",
    "EngineStepPublication",
    "EngineBatchPublication",
    "EnginePublicationEvidence",
    "EngineReleasedRequest",
    "EngineReleasePlan",
    "EngineReleaseEvidence",
    "EngineReleaseOutcome",
    "EnginePendingAttachCancel",
    "EnginePendingAttachCancelOutcome",
    "EnginePrefixLookup",
    "EnginePrefixPublishItem",
    "EnginePrefixPublishReleasePlan",
    "EnginePublishedPrefixRelease",
    "EnginePublishedPrefix",
    "EnginePrefixAttachItem",
    "EngineRequestForkItem",
    "EngineControlPlanInfo",
    "EngineMaterializationPlan",
    "EnginePrefixEvictionPlan",
    "EngineControlEvidence",
    "EngineControlOutcome",
)

INTERNAL_LEASE_NAMES = (
    "RequestLease",
    "SnapshotLease",
    "StepLease",
    "SubmissionLease",
    "ReclamationLease",
    "RelocationLease",
    "PrefixLease",
)


def _config(path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=path,
        plan_json=b'{"page_tokens":16,"classes":[]}',
        plan_fingerprint="sha256:session-fake",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=11,
                backend_domain=3,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
            ClassConfig(
                class_id=1,
                pool_id=22,
                backend_domain=4,
                name="swa",
                layers=(1,),
                retention="sliding",
                bytes_per_token_per_layer=128,
                window_tokens=18,
                period_blocks=2,
            ),
        ),
    )


SETTINGS = SessionCreateSettings(ManagerCreateSettings(4, 2, 0, 6, 32), CacheSharingPolicy.SHARED_PREFIX)
CONTROL_SETTINGS = SessionCreateSettings(ManagerCreateSettings(4, 2, 4, 6, 32), CacheSharingPolicy.SHARED_PREFIX)
REGISTRATIONS = (
    ArenaRegistration(0, 11, 3, 3, 100),
    ArenaRegistration(1, 22, 4, 3, 200),
)


def _layout(layout: type[ctypes.Structure], **values: Any) -> Any:
    result = layout()
    for name, value in values.items():
        setattr(result, name, value)
    return result


def _store(pointer: Any, layout: type[Any], value: Any) -> None:
    ctypes.cast(pointer, ctypes.POINTER(layout))[0] = value


def _count(pointer: Any, value: int) -> None:
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32))[0] = value


def _raw_page(page_id: int, pool_id: int, generation: int | None = None) -> Any:
    return L.PageLeaseLayout(
        77,
        900 + pool_id,
        page_id + 100 if generation is None else generation,
        page_id,
        pool_id,
    )


def _page_values(value: Any) -> tuple[int, int, int, int, int]:
    return (
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.page_id),
        int(value.pool_id),
    )


def _raw_detached(ordinal: int, *, reserved: int = 0) -> Any:
    pool_id = 11 if ordinal % 2 == 0 else 22
    return _layout(
        L.DetachedBindingLayout,
        old=_raw_page(10 + ordinal, pool_id),
        replacement=_raw_page(20 + ordinal, pool_id),
        logical_ordinal=ordinal,
        old_backend_index=1000 + ordinal,
        replacement_backend_index=2000 + ordinal,
        token_begin=ordinal * 16,
        token_end_exclusive=(ordinal + 1) * 16,
        class_id=ordinal % 2,
        backend_domain=3 + (ordinal % 2),
        action=2,
        reason=2,
        reserved=reserved,
    )


def _raw_retirement(ordinal: int, *, reserved: int = 0) -> Any:
    return _layout(
        L.SessionRetirementLayout,
        page=_raw_page(30 + ordinal, 11 if ordinal == 0 else 22),
        class_id=ordinal % 2,
        backend_domain=3 + (ordinal % 2),
        reserved32=reserved,
        logical_ordinal=ordinal,
        backend_index=3000 + ordinal,
        token_begin=ordinal * 16,
        token_end_exclusive=(ordinal + 1) * 16,
        completion_domain=9,
        completion_value=90 + ordinal,
    )


def _raw_key(token: int) -> Any:
    namespace = bytes([token]) * 32
    digest = bytes([token + 1]) * 32
    return _layout(
        L.PrefixKeyLayout,
        namespace_bytes=(ctypes.c_uint8 * 32)(*namespace),
        digest=(ctypes.c_uint8 * 32)(*digest),
        boundary=16 * token,
    )


class FakeSessionLibrary:
    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.invocations: list[str] = []
        self.destroyed: list[int] = []
        self.destroy_behavior: str | None = None
        self.created: dict[str, Any] | None = None
        self.submitted: dict[str, Any] | None = None
        self.completed: dict[str, Any] | None = None
        self.confirmed_publication: dict[str, Any] | None = None
        self.confirmed_release: dict[str, Any] | None = None
        self.confirmed_releases: list[dict[str, Any]] = []
        self.committed_controls: list[tuple[int, int]] = []
        self.confirmed_control: dict[str, Any] | None = None
        self.quarantined_controls: list[tuple[int, int]] = []

    def function(self, name: str) -> Any:
        try:
            return getattr(self, name)
        except AttributeError as error:
            raise AssertionError(f"unexpected symbol lookup: {name}") from error

    def orbitkv_session_create(
        self,
        plan: Any,
        plan_len: int,
        config: Any,
        registrations: Any,
        registration_count: int,
        out_handle: Any,
        error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("create")
        raw_config = ctypes.cast(config, ctypes.POINTER(L.SessionCreateConfigLayout))[0]
        raw_manager = raw_config.manager
        self.created = {
            "plan": bytes(plan[:plan_len]),
            "manager": tuple(int(getattr(raw_manager, name)) for name, _ctype in raw_manager._fields_),
            "cache_sharing_policy": int(raw_config.cache_sharing_policy),
            "reserved": int(raw_config.reserved),
            "registrations": tuple(
                tuple(
                    int(getattr(registrations[index], name))
                    for name, _ctype in L.BackendArenaRegistrationLayout._fields_
                )
                for index in range(registration_count)
            ),
        }
        null_statuses = {
            "create_buffer_null": STATUS_BUFFER_TOO_SMALL,
            "create_invalid_null": STATUS_INVALID_ARGUMENT,
            "create_manager_null": STATUS_MANAGER_ERROR,
            "create_retryable_null": STATUS_RETRYABLE_CONFLICT,
        }
        if self.fault in null_statuses:
            error.value = b"ordinary create rejection"
            return null_statuses[self.fault]
        _store(out_handle, ctypes.c_void_p, ctypes.c_void_p(0x5150))
        if self.fault == "create_raise":
            raise RuntimeError("lost create return")
        if self.fault == "create_message":
            error.value = b"error payload after create"
        if self.fault == "create_invalid_handle":
            error.value = b"invalid config"
            return STATUS_INVALID_ARGUMENT
        if self.fault == "create_retryable_handle":
            error.value = b"try again"
            return STATUS_RETRYABLE_CONFLICT
        if self.fault == "create_buffer_handle":
            error.value = b"workspace rejected"
            return STATUS_BUFFER_TOO_SMALL
        if self.fault == "create_manager_handle":
            error.value = b"manager rejected"
            return STATUS_MANAGER_ERROR
        return STATUS_FAIL_STOPPED if self.fault == "create_status" else STATUS_OK

    def orbitkv_session_arena_identities(
        self,
        _handle: Any,
        output: Any,
        capacity: int,
        out_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("arena_identities")
        assert capacity >= 2
        for index, registration in enumerate(REGISTRATIONS):
            output[index] = _layout(
                L.ArenaIdentityLayout,
                engine_epoch=77,
                pool_epoch=900 + registration.pool_id,
                backend_base_index=registration.backend_base_index,
                pool_id=registration.pool_id,
                page_count=registration.page_count,
                page_tokens=16,
                class_id=registration.class_id,
                backend_domain=registration.backend_domain,
                first_page_id=10 + index * 10,
                reserved=1 if self.fault == "identity_reserved" else 0,
            )
        _count(out_count, 2)
        return STATUS_OK

    def orbitkv_session_arena_stats(
        self,
        _handle: Any,
        output: Any,
        capacity: int,
        out_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("arena_stats")
        assert capacity >= 2
        for index, registration in enumerate(REGISTRATIONS):
            output[index] = _layout(
                L.ArenaStatsLayout,
                engine_epoch=77,
                pool_epoch=900 + registration.pool_id,
                class_id=registration.class_id,
                backend_domain=registration.backend_domain,
                pool_id=registration.pool_id,
                page_count=registration.page_count,
                first_page_id=10 + index * 10,
                reserved=0,
                reserved_padding=0,
                free_pages=2,
                reserved_pages=0,
                writing_pages=0,
                active_pages=1,
                retiring_pages=0,
                quarantined_pages=0,
                exhausted_pages=0,
                request_page_refs=1,
                prefix_page_refs=0,
                reader_pins=0,
            )
        _count(out_count, 2)
        return STATUS_OK

    def orbitkv_session_stats(
        self, _handle: Any, output: Any, _error: Any, _error_len: int
    ) -> int:
        self.invocations.append("stats")
        raw = L.ManagerStatsLayout()
        for index, (name, _ctype) in enumerate(raw._fields_, start=1):
            setattr(raw, name, index)
        _store(output, L.ManagerStatsLayout, raw)
        return STATUS_OK

    def orbitkv_session_acquire_requests(
        self,
        _handle: Any,
        request_ids: Any,
        request_count: int,
        output: Any,
        capacity: int,
        out_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("acquire")
        assert capacity >= request_count
        values = [int(request_ids[index]) for index in range(request_count)]
        returned = list(reversed(values)) if self.fault == "acquire_order" else values
        for index, request_id in enumerate(returned):
            output[index] = L.SessionRequestViewLayout(
                request_id,
                1 + index * 3,
                index * 16,
                index,
                1 if self.fault == "acquire_reserved" and index == 0 else 0,
            )
        _count(out_count, request_count)
        return STATUS_OK

    def orbitkv_session_prepare_append(
        self,
        _handle: Any,
        intents: Any,
        intent_count: int,
        batch_id: Any,
        steps: Any,
        step_capacity: int,
        step_count: Any,
        classes: Any,
        class_capacity: int,
        class_count: Any,
        tails: Any,
        tail_capacity: int,
        tail_count: Any,
        copies: Any,
        copy_capacity: int,
        copy_count: Any,
        writes: Any,
        write_capacity: int,
        write_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prepare")
        assert intent_count == 2
        assert [
            (int(intents[index].request_id), int(intents[index].target_boundary))
            for index in range(2)
        ] == [(101, 17), (202, 32)]
        assert step_capacity >= 2 and class_capacity >= 4
        assert tail_capacity >= 2 and copy_capacity >= 2 and write_capacity >= 3
        _store(batch_id, L.SessionBatchIdLayout, L.SessionBatchIdLayout(77, 1))
        steps[0] = L.SessionPreparedStepLayout(
            101, 1, 2, 0, 17, 0, 2, 0, 1, 0, 1, 0, 2
        )
        steps[1] = L.SessionPreparedStepLayout(
            202, 4, 5, 16, 32, 2, 2, 1, 1, 1, 1, 2, 1
        )
        if self.fault == "prepare_order":
            steps[0].request_id = 202
        if self.fault == "prepare_span":
            steps[1].class_offset = 1
        class_values = (
            (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 17),
            (1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 17),
            (0, 1, 1, 0, 1, 0, 2, 0, 0, 16, 32),
            (1, 1, 1, 1, 1, 1, 2, 1, 0, 16, 32),
        )
        for index, value in enumerate(class_values):
            classes[index] = L.ClassLoweringLayout(*value)
        tails[0] = L.TailActionLayout(
            0, 2, 1, 0, _raw_page(1, 11), _raw_page(2, 11), 0
        )
        tails[1] = L.TailActionLayout(
            1, 3, 16, 1, _raw_page(3, 22), _raw_page(4, 22), 0
        )
        if self.fault == "prepare_reserved":
            tails[0].reserved = 9
        copies[0] = L.CopyIntentLayout(
            0,
            3,
            8,
            1,
            2,
            0,
            _raw_page(5, 11),
            _raw_page(6, 11),
            1005,
            1006,
        )
        copies[1] = L.CopyIntentLayout(
            1,
            4,
            16,
            0,
            0,
            0,
            _raw_page(7, 22),
            _raw_page(8, 22),
            2007,
            2008,
        )
        writes[0] = L.WriteIntentLayout(102, 2, 0)
        writes[1] = L.WriteIntentLayout(106, 6, 0)
        writes[2] = L.WriteIntentLayout(108, 8, 0)
        _count(step_count, step_capacity + 1 if self.fault == "prepare_count" else 2)
        _count(class_count, 4)
        _count(tail_count, 2)
        _count(copy_count, 2)
        _count(write_count, 3)
        return STATUS_OK

    def orbitkv_session_submit_execution(
        self,
        _handle: Any,
        batch_id: Any,
        steps: Any,
        step_count: int,
        binds: Any,
        bind_count: int,
        copies: Any,
        copy_count: int,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("submit")
        self.submitted = {
            "batch_id": (int(batch_id.session_epoch), int(batch_id.sequence)),
            "steps": tuple(
                tuple(
                    int(getattr(steps[index], name))
                    for name, _ctype in steps[index]._fields_
                )
                for index in range(step_count)
            ),
            "binds": tuple(
                (
                    _page_values(binds[index].page),
                    int(binds[index].backend_domain),
                    int(binds[index].mapped),
                    int(binds[index].writable),
                    int(binds[index].reserved),
                    int(binds[index].backend_index),
                )
                for index in range(bind_count)
            ),
            "copies": tuple(
                (
                    int(copies[index].class_id),
                    int(copies[index].backend_domain),
                    int(copies[index].token_count),
                    int(copies[index].source_token_offset),
                    int(copies[index].destination_token_offset),
                    int(copies[index].observed),
                    int(copies[index].copied),
                    int(copies[index].ordered_before_writes),
                    int(copies[index].reserved8),
                    int(copies[index].reserved32),
                    _page_values(copies[index].source),
                    _page_values(copies[index].destination),
                    int(copies[index].source_backend_index),
                    int(copies[index].destination_backend_index),
                )
                for index in range(copy_count)
            ),
        }
        return STATUS_OK

    def orbitkv_session_complete_execution(
        self,
        _handle: Any,
        batch_id: Any,
        evidence: Any,
        publication_id: Any,
        publications: Any,
        publication_capacity: int,
        publication_count: Any,
        detached: Any,
        detached_capacity: int,
        detached_count: Any,
        retirements: Any,
        retirement_capacity: int,
        retirement_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("complete")
        assert publication_capacity >= 2 and detached_capacity >= 2
        assert retirement_capacity >= 2
        self.completed = {
            "batch_id": (int(batch_id.session_epoch), int(batch_id.sequence)),
            "evidence": tuple(
                int(getattr(evidence, name)) for name, _ctype in evidence._fields_
            ),
        }
        _store(
            publication_id,
            L.SessionPublicationIdLayout,
            L.SessionPublicationIdLayout(77, 2),
        )
        publications[0] = L.SessionStepPublicationLayout(101, 2, 17, 2, 0, 1, 0)
        publications[1] = L.SessionStepPublicationLayout(202, 5, 32, 3, 1, 1, 0)
        if self.fault == "completion_order":
            publications[0].request_id = 202
        if self.fault == "completion_reserved":
            publications[0].reserved = 1
        detached[0] = _raw_detached(0)
        detached[1] = _raw_detached(1)
        retirements[0] = _raw_retirement(0)
        retirements[1] = _raw_retirement(1)
        _count(publication_count, 2)
        _count(detached_count, 2)
        _count(retirement_count, 2)
        return STATUS_OK

    def orbitkv_session_confirm_publication(
        self,
        _handle: Any,
        evidence: Any,
        receipts: Any,
        receipt_count: int,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("confirm_publication")
        self.confirmed_publication = {
            "evidence": (
                int(evidence.publication_id.session_epoch),
                int(evidence.publication_id.sequence),
                int(evidence.mirror_cleanup_confirmed),
                int(evidence.reserved),
            ),
            "receipts": tuple(
                (
                    _page_values(receipts[index].page),
                    int(receipts[index].backend_domain),
                    int(receipts[index].acknowledged),
                    int(receipts[index].reserved8),
                    int(receipts[index].reserved32),
                    int(receipts[index].backend_index),
                )
                for index in range(receipt_count)
            ),
        }
        return STATUS_OK

    def orbitkv_session_prepare_release(
        self,
        _handle: Any,
        request_ids: Any,
        request_count: int,
        release_id: Any,
        releases: Any,
        release_capacity: int,
        release_count: Any,
        detached: Any,
        detached_capacity: int,
        detached_count: Any,
        retirements: Any,
        retirement_capacity: int,
        retirement_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("release")
        assert [int(request_ids[index]) for index in range(request_count)] == [101, 202]
        assert release_capacity >= 2 and detached_capacity >= 2
        assert retirement_capacity >= 1
        _store(release_id, L.SessionReleaseIdLayout, L.SessionReleaseIdLayout(77, 3))
        releases[0] = L.SessionReleasedRequestLayout(101, 0, 1)
        releases[1] = L.SessionReleasedRequestLayout(202, 1, 1)
        if self.fault == "release_span":
            releases[1].detached_offset = 0
        detached[0] = _raw_detached(
            2, reserved=1 if self.fault == "release_reserved" else 0
        )
        detached[1] = _raw_detached(3)
        retirements[0] = _raw_retirement(0)
        _count(release_count, 2)
        _count(detached_count, 2)
        _count(retirement_count, 1)
        return STATUS_OK

    def orbitkv_session_confirm_release(
        self,
        _handle: Any,
        evidence: Any,
        receipts: Any,
        receipt_count: int,
        out_outcome: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("confirm_release")
        confirmation = {
            "evidence": (
                int(evidence.release_id.session_epoch),
                int(evidence.release_id.sequence),
                int(evidence.mirror_cleanup_confirmed),
                int(evidence.reserved),
            ),
            "receipts": tuple(
                (
                    _page_values(receipts[index].page),
                    int(receipts[index].backend_domain),
                    int(receipts[index].acknowledged),
                    int(receipts[index].reserved8),
                    int(receipts[index].reserved32),
                    int(receipts[index].backend_index),
                )
                for index in range(receipt_count)
            ),
        }
        self.confirmed_release = confirmation
        self.confirmed_releases.append(confirmation)
        release_id = L.SessionReleaseIdLayout(
            int(evidence.release_id.session_epoch),
            int(evidence.release_id.sequence),
        )
        disposition = EngineReleaseDisposition.COMPLETED
        if (
            self.fault == "release_recycle_pending"
            and len(self.confirmed_releases) == 1
        ):
            disposition = EngineReleaseDisposition.RECYCLE_PENDING
        if self.fault == "release_outcome_id":
            release_id.sequence += 1
        raw_outcome = L.SessionReleaseOutcomeLayout(
            release_id,
            99
            if self.fault == "release_outcome_disposition"
            else int(disposition),
            1 if self.fault == "release_outcome_reserved" else 0,
        )
        _store(out_outcome, L.SessionReleaseOutcomeLayout, raw_outcome)
        if self.fault == "release_outcome_raise":
            raise RuntimeError("lost release confirmation return")
        return STATUS_OK

    def orbitkv_session_prefix_lookup_batch(
        self,
        _handle: Any,
        keys: Any,
        key_count: int,
        output: Any,
        capacity: int,
        out_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prefix_lookup")
        assert capacity >= key_count
        for index in range(key_count):
            key = keys[index]
            if index == 0:
                output[index] = L.SessionPrefixLookupLayout(
                    key,
                    L.SessionPrefixIdLayout(77, 11),
                    2,
                    1,
                    0,
                    0,
                )
            else:
                output[index] = L.SessionPrefixLookupLayout(
                    key,
                    L.SessionPrefixIdLayout(0, 0),
                    0 if self.fault != "lookup_miss_resident" else 1,
                    0,
                    0,
                    0,
                )
        _count(out_count, key_count)
        return STATUS_OK

    def orbitkv_session_prefix_publish_batch(
        self,
        _handle: Any,
        items: Any,
        item_count: int,
        output: Any,
        capacity: int,
        out_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prefix_publish")
        assert capacity >= item_count
        for index in range(item_count):
            output[index] = L.SessionPublishedPrefixLayout(
                L.SessionPrefixIdLayout(77, 20 + index),
                items[index].key,
                1 + index,
                0,
            )
        _count(out_count, item_count)
        return STATUS_OK

    def orbitkv_session_prefix_publish_release_batch(
        self,
        _handle: Any,
        items: Any,
        item_count: int,
        release_id: Any,
        output: Any,
        capacity: int,
        out_count: Any,
        detached: Any,
        detached_capacity: int,
        detached_count: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prefix_publish_release")
        assert capacity >= item_count and detached_capacity >= item_count
        _store(release_id, L.SessionReleaseIdLayout, L.SessionReleaseIdLayout(77, 4))
        for index in range(item_count):
            output[index] = L.SessionPublishedPrefixReleaseLayout(
                int(items[index].request_id),
                L.SessionPrefixIdLayout(77, 30 + index),
                items[index].key,
                1 + index,
                index,
                1,
                0,
            )
            detached[index] = _raw_detached(10 + index)
        _count(out_count, item_count)
        _count(detached_count, item_count)
        return STATUS_OK

    def orbitkv_session_prepare_prefix_attach(
        self,
        _handle: Any,
        _items: Any,
        _item_count: int,
        out_control: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prepare_prefix_attach")
        _store(out_control, L.SessionControlIdLayout, L.SessionControlIdLayout(77, 41))
        return STATUS_OK

    def orbitkv_session_prepare_request_fork(
        self,
        _handle: Any,
        _items: Any,
        _item_count: int,
        out_control: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prepare_request_fork")
        _store(out_control, L.SessionControlIdLayout, L.SessionControlIdLayout(77, 42))
        return STATUS_OK

    def orbitkv_session_prepare_prefix_evict(
        self,
        _handle: Any,
        _prefixes: Any,
        _prefix_count: int,
        out_control: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("prepare_prefix_evict")
        _store(out_control, L.SessionControlIdLayout, L.SessionControlIdLayout(77, 43))
        return STATUS_OK

    def orbitkv_session_abort_control(
        self,
        _handle: Any,
        control_id: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("abort_control")
        self.quarantined_controls.append(
            (int(control_id.session_epoch), int(control_id.sequence))
        )
        return STATUS_OK

    def orbitkv_session_commit_control(
        self,
        _handle: Any,
        control_id: Any,
        out_plan: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("commit_control")
        control = (int(control_id.session_epoch), int(control_id.sequence))
        self.committed_controls.append(control)
        if control[1] == 43:
            plan = L.SessionControlPlanInfoLayout(
                L.SessionControlIdLayout(*control), 2, 0, 0, 0, 1, 1
            )
        else:
            plan = L.SessionControlPlanInfoLayout(
                L.SessionControlIdLayout(*control), 1, 0, 1, 2, 0, 0
            )
        _store(out_plan, L.SessionControlPlanInfoLayout, plan)
        return STATUS_OK

    def orbitkv_session_read_control_plan(
        self,
        _handle: Any,
        control_id: Any,
        requests: Any,
        request_capacity: int,
        out_requests: Any,
        pages: Any,
        page_capacity: int,
        out_pages: Any,
        prefixes: Any,
        prefix_capacity: int,
        out_prefixes: Any,
        retirements: Any,
        retirement_capacity: int,
        out_retirements: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("read_control_plan")
        sequence = int(control_id.sequence)
        if sequence == 43:
            assert prefix_capacity >= 1 and retirement_capacity >= 1
            prefixes[0] = L.SessionPrefixIdLayout(77, 11)
            retirements[0] = _raw_retirement(0)
            _count(out_requests, 0)
            _count(out_pages, 0)
            _count(out_prefixes, 1)
            _count(out_retirements, 1)
        else:
            assert request_capacity >= 1 and page_capacity >= 2
            requests[0] = L.SessionMaterializedRequestLayout(202, 9, 32, 2, 0, 2, 0)
            pages[0] = _layout(
                L.SnapshotPageLayout,
                page=_raw_page(1, 11),
                logical_ordinal=0,
                temporal_cell_index=0,
                temporal_cycle=0,
                backend_index=1001,
                class_id=0,
                backend_domain=3,
                valid_token_count=16,
                visible_token_offset=0,
                visible_token_count=16,
                reserved=0,
            )
            pages[1] = _layout(
                L.SnapshotPageLayout,
                page=_raw_page(2, 11),
                logical_ordinal=1,
                temporal_cell_index=0,
                temporal_cycle=0,
                backend_index=1002,
                class_id=0,
                backend_domain=3,
                valid_token_count=16,
                visible_token_offset=0,
                visible_token_count=16,
                reserved=0,
            )
            _count(out_requests, 1)
            _count(out_pages, 2)
            _count(out_prefixes, 0)
            _count(out_retirements, 0)
        return STATUS_OK

    def orbitkv_session_confirm_control(
        self,
        _handle: Any,
        evidence: Any,
        receipts: Any,
        receipt_count: int,
        out_outcome: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("confirm_control")
        self.confirmed_control = {
            "evidence": (
                int(evidence.id.session_epoch),
                int(evidence.id.sequence),
                int(evidence.mirror_cleanup_confirmed),
                int(evidence.reserved),
            ),
            "receipts": tuple(
                (
                    _page_values(receipts[index].page),
                    int(receipts[index].backend_domain),
                    int(receipts[index].acknowledged),
                    int(receipts[index].reserved8),
                    int(receipts[index].reserved32),
                    int(receipts[index].backend_index),
                )
                for index in range(receipt_count)
            ),
        }
        disposition = 2 if int(evidence.id.sequence) == 43 else 1
        raw = L.SessionControlOutcomeLayout(
            L.SessionControlIdLayout(
                int(evidence.id.session_epoch), int(evidence.id.sequence)
            ),
            disposition,
            0,
        )
        _store(out_outcome, L.SessionControlOutcomeLayout, raw)
        return STATUS_OK

    def orbitkv_session_quarantine_control(
        self,
        _handle: Any,
        control_id: Any,
        _error: Any,
        _error_len: int,
    ) -> int:
        self.invocations.append("quarantine_control")
        self.quarantined_controls.append(
            (int(control_id.session_epoch), int(control_id.sequence))
        )
        return STATUS_OK

    def orbitkv_session_destroy(
        self, handle: Any, _error: Any, _error_len: int
    ) -> int:
        self.invocations.append("destroy")
        self.destroyed.append(int(handle.value or 0))
        if self.destroy_behavior == "raise":
            raise RuntimeError("lost destroy return")
        if self.destroy_behavior == "status":
            return STATUS_FAIL_STOPPED
        return STATUS_OK


def _open_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeSessionLibrary
) -> CtypesRuntimeSession:
    return _open_session_with_settings(
        tmp_path, monkeypatch, fake, SETTINGS
    )


def _open_session_with_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeSessionLibrary,
    settings: SessionCreateSettings,
) -> CtypesRuntimeSession:
    monkeypatch.setattr(session_ffi, "LoadedLibrary", lambda _path: fake)
    return CtypesRuntimeSession.create(
        _config(tmp_path / "fake.so"), settings, REGISTRATIONS
    )


def _dto_page(page_id: int, pool_id: int) -> PageLease:
    return PageLease(77, 900 + pool_id, page_id + 100, page_id, pool_id)


def _execution_evidence(plan: Any) -> ExecutionEvidence:
    copies = []
    for step in plan.steps:
        item = step.copy_intents[0]
        copies.append(
            EngineCopyEvidence(
                item.class_id,
                item.backend_domain,
                item.token_count,
                item.source_token_offset,
                item.destination_token_offset,
                True,
                True,
                True,
                item.source,
                item.destination,
                item.source_backend_index,
                item.destination_backend_index,
            )
        )
    return ExecutionEvidence(
        plan.batch_id,
        (
            EngineStepExecutionEvidence(
                EngineRequestId(101),
                (
                    EngineBindEvidence(_dto_page(40, 11), 3, True, True, 1040),
                    EngineBindEvidence(_dto_page(41, 11), 3, True, False, 1041),
                ),
                (copies[0],),
            ),
            EngineStepExecutionEvidence(
                EngineRequestId(202),
                (EngineBindEvidence(_dto_page(42, 22), 4, True, True, 2042),),
                (copies[1],),
            ),
        ),
    )


def _native_config(library: Path) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=library,
        plan_json=(
            b'{"page_tokens":16,"classes":[{"name":"full","layers":[0],'
            b'"retention":"full","bytes_per_token_per_layer":128,'
            b'"window_tokens":null}]}'
        ),
        plan_fingerprint="sha256:session-native-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=21,
                backend_domain=10,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
        ),
    )


def _native_execution_evidence(plan: Any, arenas: tuple[Any, ...]) -> ExecutionEvidence:
    arenas_by_class = {arena.class_id: arena for arena in arenas}
    steps = []
    for step in plan.steps:
        binds = []
        for lowering in step.class_lowerings:
            arena = arenas_by_class[lowering.class_id]

            def backend_index(page_id: int) -> int:
                return arena.backend_base_index + page_id - arena.first_page_id

            tail_end = lowering.tail_offset + lowering.tail_count
            for action in step.tail_actions[lowering.tail_offset:tail_end]:
                if action.kind in (TAIL_COPY_ON_WRITE, TAIL_FRESH):
                    binds.append(
                        EngineBindEvidence(
                            action.destination,
                            arena.backend_domain,
                            True,
                            True,
                            backend_index(action.destination.page_id),
                        )
                    )
            write_end = lowering.write_offset + lowering.write_count
            for intent in step.write_intents[lowering.write_offset:write_end]:
                binds.append(
                    EngineBindEvidence(
                        PageLease(
                            arena.engine_epoch,
                            arena.pool_epoch,
                            intent.page_generation,
                            intent.page_id,
                            arena.pool_id,
                        ),
                        arena.backend_domain,
                        True,
                        True,
                        backend_index(intent.page_id),
                    )
                )
        copies = tuple(
            EngineCopyEvidence(
                intent.class_id,
                intent.backend_domain,
                intent.token_count,
                intent.source_token_offset,
                intent.destination_token_offset,
                True,
                True,
                True,
                intent.source,
                intent.destination,
                intent.source_backend_index,
                intent.destination_backend_index,
            )
            for intent in step.copy_intents
        )
        steps.append(
            EngineStepExecutionEvidence(step.request_id, tuple(binds), copies)
        )
    return ExecutionEvidence(plan.batch_id, tuple(steps))


def _assert_no_internal_leases(value: Any) -> None:
    if type(value).__name__ in INTERNAL_LEASE_NAMES:
        pytest.fail(f"public session value leaked {type(value).__name__}")
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_no_internal_leases(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_internal_leases(item)


def test_session_wire_layouts_and_symbols_are_frozen() -> None:
    L.assert_frozen_layouts()
    for layout, (size, names) in SESSION_LAYOUTS.items():
        assert L.FROZEN_LAYOUTS[layout] == (size, 8)
        assert ctypes.sizeof(layout) == size
        assert ctypes.alignment(layout) == 8
        assert tuple(name for name, _ctype in layout._fields_) == names

    assert WIRE_VERSION == 14
    assert set(SESSION_SYMBOL_ARITIES) <= EXACT_SYMBOL_ALLOWLIST
    assert {
        name for name in FUNCTION_SPECS if name.startswith("orbitkv_session_")
    } >= set(SESSION_SYMBOL_ARITIES)
    for name, arity in SESSION_SYMBOL_ARITIES.items():
        assert len(FUNCTION_SPECS[name]) == arity


def test_public_session_dtos_are_frozen_and_do_not_expose_manager_leases() -> None:
    for name in PUBLIC_DTO_NAMES:
        dto = getattr(session_dtos, name)
        assert getattr(public_ffi, name) is dto
        assert dto.__dataclass_params__.frozen
        assert "__slots__" in dto.__dict__
        annotations = " ".join(str(field.type) for field in fields(dto))
        assert not any(forbidden in annotations for forbidden in INTERNAL_LEASE_NAMES)

    identifiers = (
        EngineBatchId(77, 1),
        EnginePublicationId(77, 1),
        EngineReleaseId(77, 1),
        EnginePrefixId(77, 1),
        EngineControlId(77, 1),
    )
    assert len(set(identifiers)) == 5
    assert not hasattr(public_ffi, "Engine" + "OperationId")
    for operation_id in identifiers:
        with pytest.raises(FrozenInstanceError):
            operation_id.sequence = 2  # type: ignore[misc]


def test_cross_kind_id_collisions_are_rejected_before_native_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)
    session.acquire_requests((101, 202))
    plan = session.prepare_append((EngineAppendIntent(EngineRequestId(101), 17), EngineAppendIntent(EngineRequestId(202), 32)))
    execution = _execution_evidence(plan)

    for wrong_id in (
        EnginePublicationId(plan.batch_id.session_epoch, plan.batch_id.sequence),
        EngineReleaseId(plan.batch_id.session_epoch, plan.batch_id.sequence),
    ):
        before = tuple(fake.invocations)
        with pytest.raises(ManagerError, match="must be an EngineBatchId"):
            session.submit_execution(
                ExecutionEvidence(wrong_id, execution.steps)  # type: ignore[arg-type]
            )
        assert tuple(fake.invocations) == before

    session.submit_execution(execution)
    publication = session.complete_execution(
        plan.batch_id, EngineCompletionEvidence(9, 99, True)
    )
    publication_receipts = retirement_evidence(publication.retirements)
    for wrong_id in (
        EngineBatchId(
            publication.publication_id.session_epoch,
            publication.publication_id.sequence,
        ),
        EngineReleaseId(
            publication.publication_id.session_epoch,
            publication.publication_id.sequence,
        ),
    ):
        before = tuple(fake.invocations)
        with pytest.raises(ManagerError, match="must be an EnginePublicationId"):
            session.confirm_publication(
                EnginePublicationEvidence(  # type: ignore[arg-type]
                    wrong_id, True, publication_receipts
                )
            )
        assert tuple(fake.invocations) == before

    session.confirm_publication(
        EnginePublicationEvidence(
            publication.publication_id, True, publication_receipts
        )
    )
    release = session.prepare_release((101, 202))
    release_receipts = retirement_evidence(release.retirements)
    for wrong_id in (
        EngineBatchId(release.release_id.session_epoch, release.release_id.sequence),
        EnginePublicationId(
            release.release_id.session_epoch, release.release_id.sequence
        ),
    ):
        before = tuple(fake.invocations)
        with pytest.raises(ManagerError, match="must be an EngineReleaseId"):
            session.confirm_release(
                EngineReleaseEvidence(  # type: ignore[arg-type]
                    wrong_id, True, release_receipts
                )
            )
        assert tuple(fake.invocations) == before

    session.confirm_release(
        EngineReleaseEvidence(release.release_id, True, release_receipts)
    )
    session.close()


def test_fake_session_happy_lifecycle_preserves_order_spans_and_reserved_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)

    assert fake.created == {
        "plan": b'{"page_tokens":16,"classes":[]}',
        "manager": (4, 2, 0, 6, 32, 1, 0),
        "cache_sharing_policy": 2,
        "reserved": 0,
        "registrations": ((11, 0, 3, 3, 0, 100), (22, 1, 4, 3, 0, 200)),
    }
    assert session.cache_sharing_policy is CacheSharingPolicy.SHARED_PREFIX
    assert tuple(item.class_id for item in session.arenas) == (0, 1)
    assert session.arena_identities() == session.arenas
    assert tuple(item.free_pages for item in session.arena_stats()) == (2, 2)
    stats = session.stats()
    assert tuple(getattr(stats, field.name) for field in fields(stats)) == tuple(
        range(1, 18)
    )

    views = session.acquire_requests((101, 202))
    assert [
        (int(item.request_id), item.view_version, item.boundary, item.resident_count)
        for item in views
    ] == [(101, 1, 0, 0), (202, 4, 16, 1)]
    plan = session.prepare_append((EngineAppendIntent(EngineRequestId(101), 17), EngineAppendIntent(EngineRequestId(202), 32)))
    assert plan.batch_id == EngineBatchId(77, 1)
    assert [int(item.request_id) for item in plan.steps] == [101, 202]
    assert [
        (item.previous_boundary, item.target_boundary) for item in plan.steps
    ] == [(0, 17), (16, 32)]
    assert [
        [
            (
                item.tail_offset,
                item.tail_count,
                item.copy_offset,
                item.copy_count,
                item.write_offset,
                item.write_count,
            )
            for item in step.class_lowerings
        ]
        for step in plan.steps
    ] == [
        [(0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 1)],
        [(0, 0, 0, 0, 0, 0), (0, 1, 0, 1, 0, 1)],
    ]
    assert [[item.page_id for item in step.write_intents] for step in plan.steps] == [
        [2, 6],
        [8],
    ]

    evidence = _execution_evidence(plan)
    ticket = session.submit_execution(evidence)
    assert ticket.batch_id == plan.batch_id
    assert tuple(map(int, ticket.requests)) == (101, 202)
    assert fake.submitted is not None
    assert fake.submitted["batch_id"] == (77, 1)
    assert fake.submitted["steps"] == (
        (101, 0, 2, 0, 1, 0),
        (202, 2, 1, 1, 1, 0),
    )
    assert [item[4] for item in fake.submitted["binds"]] == [0, 0, 0]
    assert [(item[8], item[9]) for item in fake.submitted["copies"]] == [
        (0, 0),
        (0, 0),
    ]

    publication = session.complete_execution(
        plan.batch_id, EngineCompletionEvidence(9, 99, True)
    )
    assert fake.completed == {
        "batch_id": (77, 1),
        "evidence": (9, 99, 1, 0),
    }
    assert publication.publication_id == EnginePublicationId(77, 2)
    assert publication.batch_id == plan.batch_id
    assert [int(item.request_id) for item in publication.steps] == [101, 202]
    assert [len(item.detached) for item in publication.steps] == [1, 1]
    assert [item.logical_ordinal for item in publication.retirements] == [0, 1]
    publication_receipts = retirement_evidence(publication.retirements)
    session.confirm_publication(
        EnginePublicationEvidence(publication.publication_id, True, publication_receipts)
    )
    assert fake.confirmed_publication is not None
    assert fake.confirmed_publication["evidence"] == (77, 2, 1, 0)
    assert [
        (item[2], item[3], item[4])
        for item in fake.confirmed_publication["receipts"]
    ] == [(1, 0, 0), (1, 0, 0)]

    release = session.prepare_release((101, 202))
    assert release.release_id == EngineReleaseId(77, 3)
    assert [int(item.request_id) for item in release.releases] == [101, 202]
    assert [len(item.detached) for item in release.releases] == [1, 1]
    assert len(release.retirements) == 1
    release_receipts = retirement_evidence(release.retirements)
    release_outcome = session.confirm_release(
        EngineReleaseEvidence(release.release_id, True, release_receipts)
    )
    assert release_outcome == EngineReleaseOutcome(
        release.release_id, EngineReleaseDisposition.COMPLETED
    )
    assert fake.confirmed_release is not None
    assert fake.confirmed_release["evidence"] == (77, 3, 1, 0)
    assert [
        (item[2], item[3], item[4]) for item in fake.confirmed_release["receipts"]
    ] == [(1, 0, 0)]

    _assert_no_internal_leases((views, plan, evidence, ticket, publication, release))
    assert not session._prepared
    assert not session._submitted
    assert not session._pending_publications
    assert not session._pending_releases
    session.close()
    session.close()
    assert fake.destroyed == [0x5150]


def test_native_session_full_lifecycle(ffi_library: Path) -> None:
    config = _native_config(ffi_library)
    settings = SessionCreateSettings(ManagerCreateSettings(1, 1, 1, 1, 16), CacheSharingPolicy.SHARED_PREFIX)
    registrations = (ArenaRegistration(0, 21, 10, 1, 1000),)

    with CtypesRuntimeSession.create(config, settings, registrations) as session:
        request_id = EngineRequestId(101)
        views = session.acquire_requests((request_id,))
        assert len(views) == 1
        assert views[0].request_id == request_id
        assert views[0].boundary == 0
        assert views[0].resident_count == 0

        plan = session.prepare_append((EngineAppendIntent(request_id, 16),))
        assert len(plan.steps) == 1
        assert plan.steps[0].request_id == request_id
        assert plan.steps[0].previous_boundary == 0
        assert plan.steps[0].target_boundary == 16
        execution = _native_execution_evidence(plan, session.arenas)
        assert sum(len(step.bind_receipts) for step in execution.steps) == 1
        assert sum(len(step.copy_receipts) for step in execution.steps) == 0

        ticket = session.submit_execution(execution)
        assert ticket.batch_id == plan.batch_id
        publication = session.complete_execution(
            plan.batch_id, EngineCompletionEvidence(7, 1, True)
        )
        assert publication.batch_id == plan.batch_id
        assert len(publication.steps) == 1
        assert publication.steps[0].request_id == request_id
        assert publication.steps[0].boundary == 16
        assert publication.steps[0].resident_count == 1
        assert publication.retirements == ()
        session.confirm_publication(
            EnginePublicationEvidence(
                publication.publication_id,
                True,
                retirement_evidence(publication.retirements),
            )
        )

        release = session.prepare_release((request_id,))
        assert len(release.releases) == 1
        assert release.releases[0].request_id == request_id
        assert len(release.releases[0].detached) == 1
        assert len(release.retirements) == 1
        release_outcome = session.confirm_release(
            EngineReleaseEvidence(
                release.release_id,
                True,
                retirement_evidence(release.retirements),
            )
        )
        assert release_outcome == EngineReleaseOutcome(
            release.release_id, EngineReleaseDisposition.COMPLETED
        )
        assert session.stats().active_requests == 0


def test_native_release_recycle_pending_then_id_only_retry_completes(
    ffi_test_support_library: Path,
) -> None:
    config = _native_config(ffi_test_support_library)
    settings = SessionCreateSettings(ManagerCreateSettings(1, 1, 1, 1, 16), CacheSharingPolicy.SHARED_PREFIX)
    registrations = (ArenaRegistration(0, 21, 10, 1, 1000),)
    symbol = "orbitkv_session_test_inject_release_recycle_once"
    assert symbol not in EXACT_SYMBOL_ALLOWLIST
    test_support = ctypes.CDLL(str(ffi_test_support_library))
    inject_recycle_once = getattr(test_support, symbol)
    inject_recycle_once.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    inject_recycle_once.restype = ctypes.c_int32

    with CtypesRuntimeSession.create(config, settings, registrations) as session:
        request_id = EngineRequestId(101)
        session.acquire_requests((request_id,))
        plan = session.prepare_append((EngineAppendIntent(request_id, 16),))
        session.submit_execution(_native_execution_evidence(plan, session.arenas))
        publication = session.complete_execution(
            plan.batch_id, EngineCompletionEvidence(7, 1, True)
        )
        session.confirm_publication(
            EnginePublicationEvidence(
                publication.publication_id,
                True,
                retirement_evidence(publication.retirements),
            )
        )
        release = session.prepare_release((request_id,))

        error = ctypes.create_string_buffer(4096)
        status = int(
            inject_recycle_once(session._handle, error, ctypes.sizeof(error))
        )
        assert status == STATUS_OK, error.value.decode("utf-8", errors="replace")

        first = session.confirm_release(
            EngineReleaseEvidence(
                release.release_id,
                True,
                retirement_evidence(release.retirements),
            )
        )
        assert first == EngineReleaseOutcome(
            release.release_id, EngineReleaseDisposition.RECYCLE_PENDING
        )
        assert session.stats().active_requests == 1

        second = session.confirm_release(
            EngineReleaseEvidence(
                release.release_id, False, (), acknowledged_retry=True
            )
        )
        assert second == EngineReleaseOutcome(
            release.release_id, EngineReleaseDisposition.COMPLETED
        )
        assert session.stats().active_requests == 0


@pytest.mark.parametrize(
    "fault,stage",
    (
        ("acquire_order", "acquire"),
        ("acquire_reserved", "acquire"),
        ("prepare_count", "prepare"),
        ("prepare_order", "prepare"),
        ("prepare_span", "prepare"),
        ("prepare_reserved", "prepare"),
        ("completion_order", "complete"),
        ("completion_reserved", "complete"),
        ("release_span", "release"),
        ("release_reserved", "release"),
    ),
)
def test_malformed_native_output_poisoning_is_sticky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    stage: str,
) -> None:
    fake = FakeSessionLibrary(fault)
    session = _open_session(tmp_path, monkeypatch, fake)

    with pytest.raises(FailStopped, match="poisoned"):
        if stage == "acquire":
            session.acquire_requests((101, 202))
        elif stage == "prepare":
            session.prepare_append(
                (
                    EngineAppendIntent(EngineRequestId(101), 17),
                    EngineAppendIntent(EngineRequestId(202), 32),
                )
            )
        elif stage == "complete":
            plan = session.prepare_append(
                (
                    EngineAppendIntent(EngineRequestId(101), 17),
                    EngineAppendIntent(EngineRequestId(202), 32),
                )
            )
            session.submit_execution(_execution_evidence(plan))
            session.complete_execution(
                plan.batch_id, EngineCompletionEvidence(9, 99, True)
            )
        else:
            session.prepare_release((101, 202))

    if stage == "prepare":
        assert not session._prepared
    elif stage == "complete":
        assert len(session._submitted) == 1
        assert not session._pending_publications
    elif stage == "release":
        assert not session._pending_releases

    before = tuple(fake.invocations)
    with pytest.raises(FailStopped, match="poisoned"):
        session.acquire_requests((303,))
    assert tuple(fake.invocations) == before
    session.close()
    assert fake.destroyed == [0x5150]


@pytest.mark.parametrize(
    "fault",
    ("create_message", "create_status", "create_raise", "identity_reserved", "create_buffer_handle", "create_invalid_handle", "create_manager_handle", "create_retryable_handle"),
)
def test_unusable_create_always_consumes_a_returned_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    fake = FakeSessionLibrary(fault)
    monkeypatch.setattr(session_ffi, "LoadedLibrary", lambda _path: fake)

    with pytest.raises(FailStopped):
        CtypesRuntimeSession.create(_config(tmp_path / "hostile.so"), SETTINGS, REGISTRATIONS)

    assert fake.destroyed == [0x5150]
    assert fake.invocations.count("destroy") == 1


@pytest.mark.parametrize(
    "fault,error_type",
    (("create_buffer_null", ManagerError), ("create_invalid_null", ManagerError), ("create_manager_null", ManagerError), ("create_retryable_null", RetryableConflict)),
)
def test_known_create_failure_with_message_and_null_handle_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    error_type: type[Exception],
) -> None:
    fake = FakeSessionLibrary(fault)
    monkeypatch.setattr(session_ffi, "LoadedLibrary", lambda _path: fake)

    with pytest.raises(error_type, match="ordinary create rejection"):
        CtypesRuntimeSession.create(_config(tmp_path / "rejected.so"), SETTINGS, REGISTRATIONS)

    assert fake.destroyed == []
    assert "destroy" not in fake.invocations


def test_confirm_release_rejects_missing_or_mismatched_receipts_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)
    release = session.prepare_release((101, 202))
    exact = retirement_evidence(release.retirements)
    assert len(exact) == 1

    before = tuple(fake.invocations)
    with pytest.raises(ManagerError, match="cardinality differs from plan"):
        session.confirm_release(EngineReleaseEvidence(release.release_id, True, ()))
    assert tuple(fake.invocations) == before
    assert release.release_id in session._pending_releases

    retirement = exact[0]
    mismatched = EngineRetirementEvidence(
        retirement.page,
        retirement.backend_domain,
        retirement.acknowledged,
        retirement.backend_index + 1,
    )
    with pytest.raises(ManagerError, match="differs from plan"):
        session.confirm_release(
            EngineReleaseEvidence(release.release_id, True, (mismatched,))
        )
    assert tuple(fake.invocations) == before
    assert release.release_id in session._pending_releases

    session.confirm_release(EngineReleaseEvidence(release.release_id, True, exact))
    assert fake.invocations[-1] == "confirm_release"
    session.close()

def test_recycle_pending_outcome_automatically_registers_acknowledged_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSessionLibrary("release_recycle_pending")
    session = _open_session(tmp_path, monkeypatch, fake)
    release = session.prepare_release((101, 202))
    outcome = session.confirm_release(
        EngineReleaseEvidence(release.release_id, True, retirement_evidence(release.retirements))
    )
    assert outcome == EngineReleaseOutcome(
        release.release_id, EngineReleaseDisposition.RECYCLE_PENDING
    )
    assert release.release_id in session._pending_releases

    retry_outcome = session.confirm_release(
        EngineReleaseEvidence(release.release_id, False, (), acknowledged_retry=True)
    )
    assert retry_outcome == EngineReleaseOutcome(
        release.release_id, EngineReleaseDisposition.COMPLETED
    )
    assert release.release_id not in session._pending_releases

    assert fake.confirmed_releases == [
        {
            "evidence": (77, 3, 1, 0),
            "receipts": (
                ((_page_values(_raw_page(30, 11))), 3, 1, 0, 0, 3000),
            ),
        },
        {"evidence": (77, 3, 0, 0), "receipts": ()},
    ]
    session.close()


@pytest.mark.parametrize(
    "fault,match",
    (("release_outcome_id", "release outcome is invalid"), ("release_outcome_disposition", "release outcome is invalid"), ("release_outcome_reserved", "release outcome is invalid"), ("release_outcome_raise", "outcome is unknown")),
)
def test_malformed_or_lost_release_outcome_fail_stops_stickily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    match: str,
) -> None:
    fake = FakeSessionLibrary(fault)
    session = _open_session(tmp_path, monkeypatch, fake)
    release = session.prepare_release((101, 202))

    with pytest.raises(FailStopped, match=match):
        session.confirm_release(
            EngineReleaseEvidence(release.release_id, True, retirement_evidence(release.retirements))
        )

    assert release.release_id in session._pending_releases
    before = tuple(fake.invocations)
    with pytest.raises(FailStopped, match="poisoned"):
        session.acquire_requests((303,))
    assert tuple(fake.invocations) == before
    session.close()


@pytest.mark.parametrize("destroy_behavior", ("status", "raise"))
def test_close_is_one_shot_even_when_destroy_outcome_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destroy_behavior: str,
) -> None:
    fake = FakeSessionLibrary()
    session = _open_session(tmp_path, monkeypatch, fake)
    plan = session.prepare_append((EngineAppendIntent(EngineRequestId(101), 17), EngineAppendIntent(EngineRequestId(202), 32)))
    assert plan.batch_id in session._prepared
    fake.destroy_behavior = destroy_behavior

    with pytest.raises(FailStopped):
        session.close()

    assert not session._handle.value
    assert not session._prepared
    session.close()
    assert fake.destroyed == [0x5150]
