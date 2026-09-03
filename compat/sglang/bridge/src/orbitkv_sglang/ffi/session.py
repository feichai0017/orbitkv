from __future__ import annotations

import ctypes
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Sequence

from orbitkv_sglang.config import MANAGER_PLAN_JSON_MAX_BYTES, RuntimeConfig
from orbitkv_sglang.runtime import (
    ArenaIdentity, ArenaRegistration, ArenaStats, CacheSharingPolicy, FailStopped,
    ManagerCreateSettings, ManagerError, ManagerStats, RetryableConflict,
    SessionCreateSettings,
)

from . import layouts as L
from .codec import key as _key
from .codec import key_to_c as _key_to_c
from .codec import page_to_c as _page_to_c
from .codec import uint as _uint
from .library import (
    ERROR_BUFFER_BYTES, LoadedLibrary, MANAGER_PLAN_FORMAT_KV_PLAN,
    MANAGER_PLAN_FORMAT_RETENTION_IR, STATUS_BUFFER_TOO_SMALL,
    STATUS_FAIL_STOPPED, STATUS_INVALID_ARGUMENT, STATUS_MANAGER_ERROR,
    STATUS_OK, STATUS_PANIC, STATUS_RETRYABLE_CONFLICT,
)
from .session_support import (
    bounded_output_counts, canonical_span_end, decode_operation_id,
    decode_prepared_steps, require_batch_count,
)
from .session_create import require_session_create_settings, session_create_config
from .session_types import (
    EngineAppendIntent, EngineBatchId, EngineBatchPlan, EngineBatchPublication,
    EngineBatchTicket, EngineBindEvidence, EngineCompletionEvidence,
    EngineCopyEvidence, EnginePrefixId, EnginePrefixPublishItem,
    EnginePrefixPublishReleasePlan, EnginePublicationEvidence,
    EnginePublicationId, EnginePublishedPrefixRelease,
    EngineReleaseDisposition, EngineReleaseEvidence, EngineReleaseId,
    EngineReleasedRequest, EngineReleaseOutcome, EngineReleasePlan,
    EngineRequestId, EngineRequestView, EngineRetirement,
    EngineRetirementEvidence, EngineStepAbortEvidence,
    EngineStepExecutionEvidence, EngineStepPlan as EngineStepPlan,
    EngineStepPublication, ExecutionEvidence, _detached, _operation_id_c,
    _retirement, _retirement_evidence_c, _wire_bool,
)
from .session_control import SessionControlMixin
from .session_relocation import SessionRelocationMixin
from .workspace import HotBounds, array, checked_product


def _discard_created_handle(loaded: LoadedLibrary, handle: ctypes.c_void_p) -> None:
    if not handle.value:
        return
    error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
    try:
        loaded.function("orbitkv_session_destroy")(handle, error, len(error))
    except BaseException:
        pass
    finally:
        handle.value = None


class CtypesRuntimeSession(SessionRelocationMixin, SessionControlMixin):
    """Locked, fail-stop ctypes client for the RuntimeSession wire."""

    @classmethod
    def create(
        cls,
        config: RuntimeConfig,
        settings: SessionCreateSettings,
        backend_registrations: Sequence[ArenaRegistration],
    ) -> CtypesRuntimeSession:
        manager_settings, cache_sharing_policy = (
            require_session_create_settings(settings)
        )
        plan = bytes(config.plan_json)
        if not plan:
            raise ManagerError("runtime session plan JSON is empty")
        if len(plan) > MANAGER_PLAN_JSON_MAX_BYTES:
            raise ManagerError(
                "runtime session plan JSON exceeds the "
                f"{MANAGER_PLAN_JSON_MAX_BYTES}-byte (1 MiB) native limit"
            )
        registrations = tuple(backend_registrations)
        if not registrations or len(registrations) != len(config.classes):
            raise ManagerError("one arena is required for every plan class")
        if tuple(item.class_id for item in registrations) != tuple(
            range(len(registrations))
        ):
            raise ManagerError("arena registrations must be class-id ordered")
        total_pages = sum(item.page_count for item in registrations)
        for registration, class_config in zip(
            registrations, config.classes, strict=True
        ):
            if (
                registration.class_id != class_config.class_id
                or registration.pool_id != class_config.pool_id
                or registration.backend_domain != class_config.backend_domain
                or registration.page_count <= 0
                or registration.backend_base_index < 0
            ):
                raise ManagerError("arena registration differs from the plan")
        raw_plan_format = getattr(config, "manager_plan_format", "kv_plan")
        if raw_plan_format == "kv_plan":
            plan_format = MANAGER_PLAN_FORMAT_KV_PLAN
        elif raw_plan_format == "retention_ir":
            plan_format = MANAGER_PLAN_FORMAT_RETENTION_IR
        elif type(raw_plan_format) is int and raw_plan_format in (
            MANAGER_PLAN_FORMAT_KV_PLAN,
            MANAGER_PLAN_FORMAT_RETENTION_IR,
        ):
            plan_format = raw_plan_format
        else:
            raise ManagerError(
                "manager plan format must be kv_plan/1 or retention_ir/2"
            )
        total_pages = _uint("total physical pages", total_pages, 32)
        if not total_pages:
            raise ManagerError("runtime session requires physical pages")
        raw_config = session_create_config(
            manager_settings, cache_sharing_policy, plan_format, total_pages
        )
        raw_registrations = (
            L.BackendArenaRegistrationLayout * len(registrations)
        )(
            *(
                L.BackendArenaRegistrationLayout(
                    _uint("pool id", item.pool_id, 32),
                    _uint("class id", item.class_id, 16),
                    _uint("backend domain", item.backend_domain, 16),
                    _uint("page count", item.page_count, 32),
                    0,
                    _uint(
                        "backend base index", item.backend_base_index, 64
                    ),
                )
                for item in registrations
            )
        )
        loaded = LoadedLibrary(Path(config.library_path))
        plan_buffer = (ctypes.c_uint8 * len(plan)).from_buffer_copy(plan)
        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        try:
            status = int(
                loaded.function("orbitkv_session_create")(
                    plan_buffer,
                    len(plan),
                    ctypes.byref(raw_config),
                    raw_registrations,
                    len(registrations),
                    ctypes.byref(handle),
                    error,
                    len(error),
                )
            )
        except BaseException as failure:
            _discard_created_handle(loaded, handle)
            raise FailStopped(
                f"runtime session create outcome is unknown: {failure}"
            ) from failure
        message = error.value.decode("utf-8", errors="replace")
        if status != STATUS_OK or not handle.value or message:
            returned_handle = bool(handle.value)
            _discard_created_handle(loaded, handle)
            if status == STATUS_RETRYABLE_CONFLICT and not returned_handle:
                raise RetryableConflict(
                    "runtime session create: "
                    f"{message or 'retryable conflict'}"
                )
            if status in (
                STATUS_BUFFER_TOO_SMALL,
                STATUS_INVALID_ARGUMENT,
                STATUS_MANAGER_ERROR,
            ) and not returned_handle:
                raise ManagerError(
                    f"runtime session create failed with status {status}: "
                    f"{message or 'no detail'}"
                )
            raise FailStopped(
                f"runtime session create outcome is unusable (status {status}): "
                f"{message or 'no detail'}"
            )
        try:
            return cls(
                loaded,
                handle,
                config,
                manager_settings,
                cache_sharing_policy,
                registrations,
                total_pages,
            )
        except BaseException:
            _discard_created_handle(loaded, handle)
            raise

    def __init__(
        self,
        loaded: LoadedLibrary,
        handle: ctypes.c_void_p,
        config: RuntimeConfig,
        settings: ManagerCreateSettings,
        cache_sharing_policy: CacheSharingPolicy,
        registrations: tuple[ArenaRegistration, ...],
        physical_pages: int,
    ) -> None:
        self._library = loaded
        self._handle = handle
        self._error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        self._lock = RLock()
        self._poisoned: str | None = None
        self._destroy_attempted = False
        self._registrations = registrations
        self._arena_count = len(registrations)
        self._physical_pages = physical_pages
        self._cache_sharing_policy = cache_sharing_policy
        self._page_tokens = _uint("page tokens", config.page_tokens, 32)
        if not self._page_tokens:
            raise ManagerError("page tokens must be positive")
        self._request_capacity = int(settings.maximum_requests)
        self._configured_operation_capacity = int(settings.maximum_operations)
        self._operation_capacity = min(
            self._request_capacity, self._configured_operation_capacity
        )
        self._prefix_capacity = int(settings.maximum_prefixes)
        self._hot_bounds = HotBounds.compile(
            maximum_batch=self._operation_capacity,
            class_count=self._arena_count,
            maximum_step_tokens=int(settings.maximum_step_tokens),
            page_tokens=self._page_tokens,
            physical_pages=physical_pages,
        )
        b = self._hot_bounds
        release_detached = checked_product(
            "session release detached bound",
            self._request_capacity,
            self._physical_pages,
        )
        self._arena_identities = array(L.ArenaIdentityLayout, self._arena_count)
        self._arena_stats = array(L.ArenaStatsLayout, self._arena_count)
        self._request_ids = array(ctypes.c_uint64, self._request_capacity)
        self._request_views = array(
            L.SessionRequestViewLayout, self._request_capacity
        )
        self._initialize_relocation_state()
        self._append_intents = array(
            L.SessionAppendIntentLayout, self._operation_capacity
        )
        self._prepared_steps = array(
            L.SessionPreparedStepLayout, self._operation_capacity
        )
        self._class_lowerings = array(
            L.ClassLoweringLayout, b.class_outputs
        )
        self._tail_actions = array(L.TailActionLayout, b.class_outputs)
        self._copy_intents = array(L.CopyIntentLayout, b.copy_outputs)
        self._write_intents = array(L.WriteIntentLayout, b.write_outputs)
        self._step_evidence = array(
            L.SessionStepExecutionEvidenceLayout, self._operation_capacity
        )
        self._bind_evidence = array(
            L.SessionBindEvidenceLayout, b.bind_outputs
        )
        self._copy_evidence = array(
            L.SessionCopyEvidenceLayout, b.copy_outputs
        )
        self._abort_evidence = array(
            L.SessionStepAbortEvidenceLayout, self._operation_capacity
        )
        self._publications = array(
            L.SessionStepPublicationLayout, self._operation_capacity
        )
        completion_detached = b.completion_detached
        self._detached = array(
            L.DetachedBindingLayout,
            max(completion_detached, release_detached),
        )
        self._retirements = array(
            L.SessionRetirementLayout, self._physical_pages
        )
        self._retirement_evidence = array(
            L.SessionRetirementEvidenceLayout, self._physical_pages
        )
        self._released = array(
            L.SessionReleasedRequestLayout, self._request_capacity
        )
        self._published_releases = array(
            L.SessionPublishedPrefixReleaseLayout, self._request_capacity
        )
        self._prepared: dict[EngineBatchId, EngineBatchPlan] = {}
        self._submitted: dict[EngineBatchId, EngineBatchTicket] = {}
        self._pending_publications: dict[
            EnginePublicationId, EngineBatchPublication
        ] = {}
        self._pending_releases: dict[EngineReleaseId, EngineReleasePlan] = {}
        self._pending_controls = {}
        self._pending_attach_cancels = {}
        self._finalized_attach_cancels = OrderedDict()
        self._finalized_attach_cancel_limit = self._operation_capacity
        # These ids are capabilities minted only from a typed native
        # RECYCLE_PENDING outcome. They authorize an ID-only retry without
        # duplicating the native release phase or reclamation authority.
        self._acknowledged_release_retries: set[EngineReleaseId] = set()
        self._arenas = self._read_arena_identities()

    @property
    def arenas(self) -> tuple[ArenaIdentity, ...]:
        return self._arenas

    @property
    def prefix_capacity(self) -> int:
        return self._prefix_capacity

    @property
    def cache_sharing_policy(self) -> CacheSharingPolicy:
        return self._cache_sharing_policy

    @property
    def control_batch_capacity(self) -> int:
        return self._configured_operation_capacity

    @property
    def prefix_eviction_batch_capacity(self) -> int:
        return min(self._prefix_capacity, self._configured_operation_capacity)

    def _require_handle(
        self, *, allow_poisoned: bool = False
    ) -> ctypes.c_void_p:
        if not self._handle.value:
            raise ManagerError("OrbitKV runtime session handle is closed")
        if self._poisoned is not None and not allow_poisoned:
            raise FailStopped(
                "OrbitKV runtime session is poisoned: " + self._poisoned
            )
        return self._handle

    def _poison(self, reason: str) -> FailStopped:
        if self._poisoned is None:
            self._poisoned = reason
        return FailStopped(
            "OrbitKV runtime session is poisoned: " + self._poisoned
        )

    def _call(
        self,
        operation: str,
        function: Callable[..., Any],
        *args: Any,
        require_handle: bool = True,
        allow_poisoned: bool = False,
        allow_short: bool = False,
    ) -> int:
        if require_handle:
            self._require_handle(allow_poisoned=allow_poisoned)
        ctypes.memset(self._error, 0, len(self._error))
        try:
            status = int(function(*args, self._error, len(self._error)))
        except BaseException as error:
            raise self._poison(
                f"{operation} outcome is unknown: {error}"
            ) from error
        message = self._error.value.decode("utf-8", errors="replace")
        if status == STATUS_OK:
            if message:
                raise self._poison(
                    f"{operation} succeeded with an error payload"
                )
            return status
        if status == STATUS_BUFFER_TOO_SMALL:
            if allow_short:
                if message:
                    raise self._poison(
                        f"{operation} preflight returned an error payload"
                    )
                return status
            raise self._poison(
                f"{operation} exceeded its fixed configured workspace"
            )
        if status == STATUS_RETRYABLE_CONFLICT:
            raise RetryableConflict(
                f"{operation}: {message or 'retryable conflict'}"
            )
        if status in (STATUS_INVALID_ARGUMENT, STATUS_MANAGER_ERROR):
            raise ManagerError(
                f"{operation} failed with status {status}: "
                f"{message or 'no detail'}"
            )
        if status in (STATUS_PANIC, STATUS_FAIL_STOPPED):
            raise self._poison(
                f"{operation} failed with status {status}: "
                f"{message or 'no detail'}"
            )
        raise self._poison(
            f"{operation} returned unknown status {status}: {message}"
        )

    def _read_arena_identities(self) -> tuple[ArenaIdentity, ...]:
        count = ctypes.c_uint32()
        self._call(
            "runtime session arena identities",
            self._library.function("orbitkv_session_arena_identities"),
            self._require_handle(allow_poisoned=True),
            self._arena_identities,
            self._arena_count,
            ctypes.byref(count),
            allow_poisoned=True,
        )
        if int(count.value) != self._arena_count:
            raise self._poison(
                "runtime session arena identity cardinality changed"
            )
        try:
            result = []
            for raw, registration in zip(
                self._arena_identities, self._registrations, strict=True
            ):
                if int(raw.reserved) != 0:
                    raise ManagerError(
                        "runtime session arena identity reserved field is nonzero"
                    )
                item = ArenaIdentity(
                    int(raw.engine_epoch),
                    int(raw.pool_epoch),
                    int(raw.pool_id),
                    int(raw.class_id),
                    int(raw.backend_domain),
                    int(raw.page_count),
                    int(raw.page_tokens),
                    int(raw.backend_base_index),
                    int(raw.first_page_id),
                )
                if (
                    item.class_id != registration.class_id
                    or item.pool_id != registration.pool_id
                    or item.backend_domain != registration.backend_domain
                    or item.page_count != registration.page_count
                    or item.backend_base_index
                    != registration.backend_base_index
                    or item.page_tokens != self._page_tokens
                ):
                    raise ManagerError(
                        "runtime session arena identity differs from registration"
                    )
                result.append(item)
            if len({item.engine_epoch for item in result}) != 1:
                raise ManagerError(
                    "runtime session arena identities have different epochs"
                )
            self._session_epoch = result[0].engine_epoch
            return tuple(result)
        except Exception as error:
            raise self._poison(
                f"runtime session arena identities are invalid: {error}"
            ) from error

    def arena_identities(self) -> tuple[ArenaIdentity, ...]:
        with self._lock:
            current = self._read_arena_identities()
            if current != self._arenas:
                raise self._poison(
                    "runtime session arena identities drifted after create"
                )
            return current

    def arena_stats(self) -> tuple[ArenaStats, ...]:
        with self._lock:
            count = ctypes.c_uint32()
            self._call(
                "runtime session arena stats",
                self._library.function("orbitkv_session_arena_stats"),
                self._require_handle(allow_poisoned=True),
                self._arena_stats,
                self._arena_count,
                ctypes.byref(count),
                allow_poisoned=True,
            )
            if int(count.value) != self._arena_count:
                raise self._poison(
                    "runtime session arena stats cardinality changed"
                )
            try:
                result = []
                for raw, identity in zip(
                    self._arena_stats, self._arenas, strict=True
                ):
                    if int(raw.reserved) or int(raw.reserved_padding):
                        raise ManagerError(
                            "runtime session arena stats reserved field is nonzero"
                        )
                    item = ArenaStats(
                        int(raw.engine_epoch),
                        int(raw.pool_epoch),
                        int(raw.pool_id),
                        int(raw.page_count),
                        int(raw.class_id),
                        int(raw.backend_domain),
                        int(raw.first_page_id),
                        int(raw.free_pages),
                        int(raw.reserved_pages),
                        int(raw.writing_pages),
                        int(raw.active_pages),
                        int(raw.retiring_pages),
                        int(raw.quarantined_pages),
                        int(raw.exhausted_pages),
                        int(raw.request_page_refs),
                        int(raw.prefix_page_refs),
                        int(raw.reader_pins),
                    )
                    if (
                        item.engine_epoch != identity.engine_epoch
                        or item.pool_epoch != identity.pool_epoch
                        or item.pool_id != identity.pool_id
                        or item.page_count != identity.page_count
                        or item.class_id != identity.class_id
                        or item.backend_domain != identity.backend_domain
                        or item.first_page_id != identity.first_page_id
                    ):
                        raise ManagerError(
                            "runtime session arena stats identity drifted"
                        )
                    result.append(item)
                return tuple(result)
            except Exception as error:
                raise self._poison(
                    f"runtime session arena stats are invalid: {error}"
                ) from error

    def stats(self) -> ManagerStats:
        with self._lock:
            raw = L.ManagerStatsLayout()
            self._call(
                "runtime session stats",
                self._library.function("orbitkv_session_stats"),
                self._require_handle(allow_poisoned=True),
                ctypes.byref(raw),
                allow_poisoned=True,
            )
            return ManagerStats(
                *(int(getattr(raw, name)) for name, _ctype in raw._fields_)
            )

    def acquire_requests(
        self, request_ids: Sequence[int]
    ) -> tuple[EngineRequestView, ...]:
        with self._lock:
            self._require_handle()
            values = tuple(request_ids)
            count = require_batch_count(
                values, "session acquire", self._request_capacity
            )
            encoded = tuple(
                _uint("engine request id", value, 64) for value in values
            )
            if len(set(encoded)) != count:
                raise ManagerError("session acquire contains duplicate request ids")
            for index, value in enumerate(encoded):
                self._request_ids[index] = value
            out = ctypes.c_uint32()
            self._call(
                "runtime session acquire requests",
                self._library.function("orbitkv_session_acquire_requests"),
                self._require_handle(),
                self._request_ids,
                count,
                self._request_views,
                self._request_capacity,
                ctypes.byref(out),
            )
            if int(out.value) != count:
                raise self._poison(
                    "runtime session acquire cardinality changed"
                )
            try:
                result = []
                for expected, raw in zip(
                    encoded, self._request_views[:count], strict=True
                ):
                    if (
                        int(raw.request_id) != expected
                        or int(raw.reserved) != 0
                    ):
                        raise ManagerError(
                            "runtime session acquire request ordering is invalid"
                        )
                    result.append(
                        EngineRequestView(
                            EngineRequestId(int(raw.request_id)),
                            int(raw.view_version),
                            int(raw.boundary),
                            int(raw.resident_count),
                        )
                    )
                return tuple(result)
            except Exception as error:
                raise self._poison(
                    f"runtime session acquire output is invalid: {error}"
                ) from error

    def prepare_append(
        self, intents: Sequence[EngineAppendIntent]
    ) -> EngineBatchPlan:
        with self._lock:
            self._require_handle()
            values = tuple(intents)
            count = require_batch_count(
                values, "session append", self._operation_capacity
            )
            if any(not isinstance(item, EngineAppendIntent) for item in values):
                raise ManagerError(
                    "session append items must be EngineAppendIntent values"
                )
            request_ids = tuple(item.request_id for item in values)
            if len(set(request_ids)) != count:
                raise ManagerError("session append contains duplicate request ids")
            for index, item in enumerate(values):
                self._append_intents[index] = L.SessionAppendIntentLayout(
                    _uint("engine request id", item.request_id, 64),
                    _uint("append target boundary", item.target_boundary, 64),
                )
            batch_id = L.SessionBatchIdLayout()
            counts = [ctypes.c_uint32() for _ in range(5)]
            b = self._hot_bounds
            self._call(
                "runtime session prepare append",
                self._library.function("orbitkv_session_prepare_append"),
                self._require_handle(),
                self._append_intents,
                count,
                ctypes.byref(batch_id),
                self._prepared_steps,
                self._operation_capacity,
                ctypes.byref(counts[0]),
                self._class_lowerings,
                b.class_outputs,
                ctypes.byref(counts[1]),
                self._tail_actions,
                b.class_outputs,
                ctypes.byref(counts[2]),
                self._copy_intents,
                b.copy_outputs,
                ctypes.byref(counts[3]),
                self._write_intents,
                b.write_outputs,
                ctypes.byref(counts[4]),
            )
            try:
                totals = bounded_output_counts(
                    counts,
                    (
                        self._operation_capacity,
                        b.class_outputs,
                        b.class_outputs,
                        b.copy_outputs,
                        b.write_outputs,
                    ),
                    "session prepare",
                )
                if totals[0] != count:
                    raise ManagerError(
                        "session prepare output cardinality changed"
                    )
                operation_id = decode_operation_id(
                    batch_id, EngineBatchId, L.SessionBatchIdLayout, "batch id"
                )
                if operation_id.session_epoch != self._session_epoch:
                    raise ManagerError(
                        "session prepare returned a foreign batch id"
                    )
                steps: tuple[EngineStepPlan, ...] = decode_prepared_steps(
                    request_ids,
                    count,
                    self._prepared_steps,
                    self._class_lowerings,
                    self._tail_actions,
                    self._copy_intents,
                    self._write_intents,
                    *totals[1:],
                )
                plan = EngineBatchPlan(operation_id, steps)
                if (
                    operation_id in self._prepared
                    or operation_id in self._submitted
                ):
                    raise ManagerError("session returned a duplicate batch id")
                self._prepared[operation_id] = plan
                return plan
            except Exception as error:
                raise self._poison(
                    f"runtime session prepare output is invalid: {error}"
                ) from error

    def submit_execution(
        self, evidence: ExecutionEvidence
    ) -> EngineBatchTicket:
        with self._lock:
            self._require_handle()
            if not isinstance(evidence, ExecutionEvidence):
                raise ManagerError(
                    "session execution evidence has the wrong type"
                )
            batch_id = _operation_id_c(
                evidence.batch_id,
                EngineBatchId,
                L.SessionBatchIdLayout,
                "batch id",
            )
            plan = self._prepared.get(evidence.batch_id)
            if plan is None:
                raise ManagerError("session batch is not locally prepared")
            steps = tuple(evidence.steps)
            count = require_batch_count(
                steps, "session execution evidence", self._operation_capacity
            )
            if any(
                not isinstance(item, EngineStepExecutionEvidence)
                for item in steps
            ):
                raise ManagerError(
                    "session steps must be EngineStepExecutionEvidence values"
                )
            expected = tuple(item.request_id for item in plan.steps)
            actual = tuple(item.request_id for item in steps)
            if actual != expected:
                raise ManagerError(
                    "session execution evidence request ordering differs from plan"
                )
            bind_total = sum(len(item.bind_receipts) for item in steps)
            copy_total = sum(len(item.copy_receipts) for item in steps)
            if (
                bind_total > self._hot_bounds.bind_outputs
                or copy_total > self._hot_bounds.copy_outputs
            ):
                raise ManagerError(
                    "session execution evidence exceeds fixed workspace bounds"
                )
            bind_cursor = copy_cursor = 0
            for index, item in enumerate(steps):
                binds = tuple(item.bind_receipts)
                copies = tuple(item.copy_receipts)
                self._step_evidence[index] = (
                    L.SessionStepExecutionEvidenceLayout(
                        _uint("engine request id", item.request_id, 64),
                        bind_cursor,
                        len(binds),
                        copy_cursor,
                        len(copies),
                        0,
                    )
                )
                for value in binds:
                    if not isinstance(value, EngineBindEvidence):
                        raise ManagerError(
                            "bind receipts must be EngineBindEvidence values"
                        )
                    self._bind_evidence[bind_cursor] = (
                        L.SessionBindEvidenceLayout(
                            _page_to_c(value.page),
                            _uint(
                                "bind backend domain", value.backend_domain, 16
                            ),
                            _wire_bool("bind mapped", value.mapped, 8),
                            _wire_bool("bind writable", value.writable, 8),
                            0,
                            _uint(
                                "bind backend index", value.backend_index, 64
                            ),
                        )
                    )
                    bind_cursor += 1
                for value in copies:
                    if not isinstance(value, EngineCopyEvidence):
                        raise ManagerError(
                            "copy receipts must be EngineCopyEvidence values"
                        )
                    self._copy_evidence[copy_cursor] = (
                        L.SessionCopyEvidenceLayout(
                            _uint("copy class", value.class_id, 16),
                            _uint(
                                "copy backend domain",
                                value.backend_domain,
                                16,
                            ),
                            _uint("copy token count", value.token_count, 32),
                            _uint(
                                "copy source offset",
                                value.source_token_offset,
                                32,
                            ),
                            _uint(
                                "copy destination offset",
                                value.destination_token_offset,
                                32,
                            ),
                            _wire_bool("copy observed", value.observed, 8),
                            _wire_bool("copy completed", value.copied, 8),
                            _wire_bool(
                                "copy ordering",
                                value.ordered_before_writes,
                                8,
                            ),
                            0,
                            0,
                            _page_to_c(value.source),
                            _page_to_c(value.destination),
                            _uint(
                                "copy source backend",
                                value.source_backend_index,
                                64,
                            ),
                            _uint(
                                "copy destination backend",
                                value.destination_backend_index,
                                64,
                            ),
                        )
                    )
                    copy_cursor += 1
            self._call(
                "runtime session submit execution",
                self._library.function("orbitkv_session_submit_execution"),
                self._require_handle(),
                batch_id,
                self._step_evidence,
                count,
                self._bind_evidence,
                bind_total,
                self._copy_evidence,
                copy_total,
            )
            ticket = EngineBatchTicket(evidence.batch_id, expected)
            self._prepared.pop(evidence.batch_id)
            self._submitted[evidence.batch_id] = ticket
            return ticket

    def abort_prepared(
        self,
        batch_id: EngineBatchId,
        evidence: Sequence[EngineStepAbortEvidence],
    ) -> None:
        with self._lock:
            self._require_handle()
            raw_batch_id = _operation_id_c(
                batch_id, EngineBatchId, L.SessionBatchIdLayout, "batch id"
            )
            plan = self._prepared.get(batch_id)
            if plan is None:
                raise ManagerError("session batch is not locally prepared")
            values = tuple(evidence)
            count = require_batch_count(
                values, "session abort evidence", self._operation_capacity
            )
            if any(
                not isinstance(item, EngineStepAbortEvidence) for item in values
            ):
                raise ManagerError(
                    "abort evidence must be EngineStepAbortEvidence values"
                )
            expected = tuple(item.request_id for item in plan.steps)
            if tuple(item.request_id for item in values) != expected:
                raise ManagerError(
                    "session abort evidence request ordering differs from plan"
                )
            for index, item in enumerate(values):
                self._abort_evidence[index] = L.SessionStepAbortEvidenceLayout(
                    _uint("engine request id", item.request_id, 64),
                    _wire_bool(
                        "abort backend unobserved",
                        item.backend_unobserved,
                        32,
                    ),
                    0,
                )
            self._call(
                "runtime session abort prepared execution",
                self._library.function("orbitkv_session_abort_prepared"),
                self._require_handle(),
                raw_batch_id,
                self._abort_evidence,
                count,
            )
            self._prepared.pop(batch_id)

    def quarantine_prepared(self, batch_id: EngineBatchId) -> None:
        with self._lock:
            self._require_handle()
            raw_batch_id = _operation_id_c(
                batch_id, EngineBatchId, L.SessionBatchIdLayout, "batch id"
            )
            if batch_id not in self._prepared:
                raise ManagerError("session batch is not locally prepared")
            try:
                self._call(
                    "runtime session quarantine prepared execution",
                    self._library.function("orbitkv_session_quarantine_prepared"),
                    self._require_handle(),
                    raw_batch_id,
                )
                raise self._poison(
                    "runtime session prepared quarantine returned success"
                )
            finally:
                if self._poisoned is not None:
                    self._prepared.pop(batch_id, None)

    def quarantine_submitted(self, batch_id: EngineBatchId) -> None:
        with self._lock:
            self._require_handle()
            raw_batch_id = _operation_id_c(
                batch_id, EngineBatchId, L.SessionBatchIdLayout, "batch id"
            )
            if batch_id not in self._submitted:
                raise ManagerError("session batch is not locally submitted")
            try:
                self._call(
                    "runtime session quarantine submitted execution",
                    self._library.function(
                        "orbitkv_session_quarantine_submitted"
                    ),
                    self._require_handle(),
                    raw_batch_id,
                )
                raise self._poison(
                    "runtime session submitted quarantine returned success"
                )
            finally:
                if self._poisoned is not None:
                    self._submitted.pop(batch_id, None)

    def complete_execution(
        self,
        batch_id: EngineBatchId,
        evidence: EngineCompletionEvidence,
    ) -> EngineBatchPublication:
        with self._lock:
            self._require_handle()
            raw_batch_id = _operation_id_c(
                batch_id, EngineBatchId, L.SessionBatchIdLayout, "batch id"
            )
            ticket = self._submitted.get(batch_id)
            if ticket is None:
                raise ManagerError("session batch is not locally submitted")
            if not isinstance(evidence, EngineCompletionEvidence):
                raise ManagerError(
                    "completion evidence must be EngineCompletionEvidence"
                )
            raw_evidence = L.SessionCompletionEvidenceLayout(
                _uint(
                    "completion domain", evidence.completion_domain, 64
                ),
                _uint("completion value", evidence.completion_value, 64),
                _wire_bool("completion confirmed", evidence.confirmed, 32),
                0,
            )
            publication_id = L.SessionPublicationIdLayout()
            counts = [ctypes.c_uint32() for _ in range(3)]
            detached_capacity = len(self._detached)
            self._call(
                "runtime session complete execution",
                self._library.function("orbitkv_session_complete_execution"),
                self._require_handle(),
                raw_batch_id,
                raw_evidence,
                ctypes.byref(publication_id),
                self._publications,
                self._operation_capacity,
                ctypes.byref(counts[0]),
                self._detached,
                detached_capacity,
                ctypes.byref(counts[1]),
                self._retirements,
                self._physical_pages,
                ctypes.byref(counts[2]),
            )
            try:
                totals = bounded_output_counts(
                    counts,
                    (
                        self._operation_capacity,
                        detached_capacity,
                        self._physical_pages,
                    ),
                    "session completion",
                )
                if totals[0] != len(ticket.requests):
                    raise ManagerError(
                        "session completion cardinality changed"
                    )
                operation_id = decode_operation_id(
                    publication_id,
                    EnginePublicationId,
                    L.SessionPublicationIdLayout,
                    "publication id",
                )
                if operation_id.session_epoch != batch_id.session_epoch:
                    raise ManagerError(
                        "session completion returned a foreign publication id"
                    )
                if operation_id.session_epoch != self._session_epoch:
                    raise ManagerError(
                        "session completion returned the wrong session epoch"
                    )
                cursor = 0
                steps = []
                for expected, raw in zip(
                    ticket.requests,
                    self._publications[: totals[0]],
                    strict=True,
                ):
                    if (
                        int(raw.request_id) != expected
                        or int(raw.reserved) != 0
                    ):
                        raise ManagerError(
                            "session completion request ordering is invalid"
                        )
                    end = canonical_span_end(
                        int(raw.detached_offset),
                        int(raw.detached_count),
                        cursor,
                        totals[1],
                        "session completion detached",
                    )
                    steps.append(
                        EngineStepPublication(
                            EngineRequestId(int(raw.request_id)),
                            int(raw.view_version),
                            int(raw.boundary),
                            int(raw.resident_count),
                            tuple(
                                _detached(self._detached[position])
                                for position in range(cursor, end)
                            ),
                        )
                    )
                    cursor = end
                if cursor != totals[1]:
                    raise ManagerError(
                        "session completion detached spans do not cover output"
                    )
                publication = EngineBatchPublication(
                    operation_id,
                    batch_id,
                    tuple(steps),
                    tuple(
                        _retirement(self._retirements[position])
                        for position in range(totals[2])
                    ),
                )
                if operation_id in self._pending_publications:
                    raise ManagerError("session returned a duplicate publication id")
                self._submitted.pop(batch_id)
                self._pending_publications[operation_id] = publication
                return publication
            except Exception as error:
                raise self._poison(
                    f"runtime session completion output is invalid: {error}"
                ) from error

    def confirm_publication(
        self, evidence: EnginePublicationEvidence
    ) -> None:
        with self._lock:
            self._require_handle()
            if not isinstance(evidence, EnginePublicationEvidence):
                raise ManagerError(
                    "publication evidence must be EnginePublicationEvidence"
                )
            raw_publication_id = _operation_id_c(
                evidence.publication_id,
                EnginePublicationId,
                L.SessionPublicationIdLayout,
                "publication id",
            )
            publication = self._pending_publications.get(
                evidence.publication_id
            )
            if publication is None:
                raise ManagerError("session publication is not locally pending")
            receipts = tuple(evidence.reclamation_receipts)
            if len(receipts) != len(publication.retirements):
                raise ManagerError(
                    "publication retirement evidence cardinality differs from plan"
                )
            self._validate_retirement_evidence(
                publication.retirements, receipts, "publication"
            )
            self._encode_retirement_evidence(receipts)
            raw = L.SessionPublicationEvidenceLayout(
                raw_publication_id,
                _wire_bool(
                    "publication mirror cleanup",
                    evidence.mirror_cleanup_confirmed,
                    32,
                ),
                0,
            )
            self._call(
                "runtime session confirm publication",
                self._library.function("orbitkv_session_confirm_publication"),
                self._require_handle(),
                raw,
                self._retirement_evidence,
                len(receipts),
            )
            self._pending_publications.pop(evidence.publication_id)

    def prepare_release(
        self, request_ids: Sequence[int]
    ) -> EngineReleasePlan:
        with self._lock:
            self._require_handle()
            values = tuple(request_ids)
            count = require_batch_count(
                values, "session release", self._request_capacity
            )
            encoded = tuple(
                _uint("engine request id", value, 64) for value in values
            )
            if len(set(encoded)) != count:
                raise ManagerError("session release contains duplicate request ids")
            for index, value in enumerate(encoded):
                self._request_ids[index] = value
            release_id = L.SessionReleaseIdLayout()
            counts = [ctypes.c_uint32() for _ in range(3)]
            detached_capacity = len(self._detached)
            self._call(
                "runtime session prepare release",
                self._library.function("orbitkv_session_prepare_release"),
                self._require_handle(),
                self._request_ids,
                count,
                ctypes.byref(release_id),
                self._released,
                self._request_capacity,
                ctypes.byref(counts[0]),
                self._detached,
                detached_capacity,
                ctypes.byref(counts[1]),
                self._retirements,
                self._physical_pages,
                ctypes.byref(counts[2]),
            )
            try:
                totals = bounded_output_counts(
                    counts,
                    (
                        self._request_capacity,
                        detached_capacity,
                        self._physical_pages,
                    ),
                    "session release",
                )
                if totals[0] != count:
                    raise ManagerError("session release cardinality changed")
                operation_id = decode_operation_id(
                    release_id,
                    EngineReleaseId,
                    L.SessionReleaseIdLayout,
                    "release id",
                )
                if operation_id.session_epoch != self._session_epoch:
                    raise ManagerError(
                        "session release returned a foreign release id"
                    )
                cursor = 0
                releases = []
                for expected, raw in zip(
                    encoded, self._released[: totals[0]], strict=True
                ):
                    if int(raw.request_id) != expected:
                        raise ManagerError(
                            "session release request ordering is invalid"
                        )
                    end = canonical_span_end(
                        int(raw.detached_offset),
                        int(raw.detached_count),
                        cursor,
                        totals[1],
                        "session release detached",
                    )
                    releases.append(
                        EngineReleasedRequest(
                            EngineRequestId(int(raw.request_id)),
                            tuple(
                                _detached(self._detached[position])
                                for position in range(cursor, end)
                            ),
                        )
                    )
                    cursor = end
                if cursor != totals[1]:
                    raise ManagerError(
                        "session release detached spans do not cover output"
                    )
                plan = EngineReleasePlan(
                    operation_id,
                    tuple(releases),
                    tuple(
                        _retirement(self._retirements[position])
                        for position in range(totals[2])
                    ),
                )
                if operation_id in self._pending_releases:
                    raise ManagerError("session returned a duplicate release id")
                self._pending_releases[operation_id] = plan
                return plan
            except Exception as error:
                raise self._poison(
                    f"runtime session release output is invalid: {error}"
                ) from error

    def prefix_publish_release_batch(
        self, items: Sequence[EnginePrefixPublishItem]
    ) -> EnginePrefixPublishReleasePlan:
        with self._lock:
            self._require_shared_cache_policy(
                "session prefix publish-release"
            )
            self._require_handle()
            values = tuple(items)
            count = require_batch_count(
                values,
                "session prefix publish-release",
                min(self._prefix_capacity, self._request_capacity),
            )
            if any(type(item) is not EnginePrefixPublishItem for item in values):
                raise ManagerError("session prefix publish-release items must be EnginePrefixPublishItem values")
            if len({item.request_id for item in values}) != count:
                raise ManagerError("session prefix publish-release contains duplicate request ids")
            raw = (L.SessionPrefixPublishItemLayout * count)(*(
                L.SessionPrefixPublishItemLayout(
                    _uint("engine request id", item.request_id, 64),
                    _key_to_c(item.key),
                )
                for item in values
            ))
            release_id = L.SessionReleaseIdLayout()
            counts = [ctypes.c_uint32() for _ in range(3)]
            detached_capacity = len(self._detached)
            self._call(
                "runtime session prefix publish-release batch",
                self._library.function("orbitkv_session_prefix_publish_release_batch"),
                self._require_handle(),
                raw,
                count,
                ctypes.byref(release_id),
                self._published_releases,
                self._request_capacity,
                ctypes.byref(counts[0]),
                self._detached,
                detached_capacity,
                ctypes.byref(counts[1]),
            )
            try:
                totals = bounded_output_counts(
                    counts[:2], (self._request_capacity, detached_capacity),
                    "session prefix publish-release",
                )
                if totals[0] != count:
                    raise ManagerError("session prefix publish-release cardinality changed")
                operation_id = decode_operation_id(
                    release_id,
                    EngineReleaseId,
                    L.SessionReleaseIdLayout,
                    "release id",
                )
                if operation_id.session_epoch != self._session_epoch:
                    raise ManagerError("session prefix publish-release returned a foreign release id")
                cursor = 0
                outputs = []
                seen_prefix_ids: set[EnginePrefixId] = set()
                seen_release_requests: set[EngineRequestId] = set()
                for expected, raw_item in zip(
                    values, self._published_releases[: totals[0]], strict=True
                ):
                    if int(raw_item.reserved) != 0:
                        raise ManagerError("session prefix publish-release reserved field is nonzero")
                    prefix_id = decode_operation_id(
                        raw_item.prefix_id,
                        EnginePrefixId,
                        L.SessionPrefixIdLayout,
                        "published prefix id",
                    )
                    if prefix_id.session_epoch != self._session_epoch:
                        raise ManagerError("session prefix publish-release returned a foreign prefix id")
                    if prefix_id in seen_prefix_ids:
                        raise ManagerError("session prefix publish-release returned duplicate prefix ids")
                    seen_prefix_ids.add(prefix_id)
                    request_id = EngineRequestId(int(raw_item.request_id))
                    if request_id != expected.request_id:
                        raise ManagerError("session prefix publish-release request ordering is invalid")
                    if request_id in seen_release_requests:
                        raise ManagerError("session prefix publish-release returned duplicate request ids")
                    seen_release_requests.add(request_id)
                    key = _key(raw_item.key)
                    if key != expected.key:
                        raise ManagerError("session prefix publish-release key ordering changed")
                    end = canonical_span_end(
                        int(raw_item.detached_offset),
                        int(raw_item.detached_count),
                        cursor,
                        totals[1],
                        "session prefix publish-release detached",
                    )
                    outputs.append(
                        EnginePublishedPrefixRelease(
                            prefix_id,
                            key,
                            int(raw_item.resident_count),
                            EngineReleasedRequest(
                                request_id,
                                tuple(
                                    _detached(self._detached[position])
                                    for position in range(cursor, end)
                                ),
                            ),
                        )
                    )
                    cursor = end
                if cursor != totals[1]:
                    raise ManagerError("session prefix publish-release detached spans do not cover output")
                plan = EnginePrefixPublishReleasePlan(operation_id, tuple(outputs))
                if operation_id in self._pending_releases:
                    raise ManagerError("session returned a duplicate release id")
                self._pending_releases[operation_id] = EngineReleasePlan(operation_id, tuple(item.release for item in outputs), ())
                return plan
            except Exception as error:
                raise self._poison(
                    "runtime session prefix publish-release output is invalid: "
                    f"{error}"
                ) from error

    def confirm_release(
        self, evidence: EngineReleaseEvidence
    ) -> EngineReleaseOutcome:
        with self._lock:
            self._require_handle()
            if not isinstance(evidence, EngineReleaseEvidence):
                raise ManagerError(
                    "release evidence must be EngineReleaseEvidence"
                )
            raw_release_id = _operation_id_c(
                evidence.release_id,
                EngineReleaseId,
                L.SessionReleaseIdLayout,
                "release id",
            )
            if evidence.release_id not in self._pending_releases:
                raise ManagerError("session release is not locally pending")
            receipts = tuple(evidence.reclamation_receipts)
            retirements = self._pending_releases[evidence.release_id].retirements
            if evidence.acknowledged_retry:
                if evidence.release_id not in self._acknowledged_release_retries:
                    raise ManagerError(
                        "release is not eligible for an acknowledged retry"
                    )
                if receipts or evidence.mirror_cleanup_confirmed:
                    raise ManagerError(
                        "acknowledged release retry requires empty evidence"
                    )
            else:
                if evidence.release_id in self._acknowledged_release_retries:
                    raise ManagerError(
                        "release requires an acknowledged ID-only retry"
                    )
                if len(receipts) != len(retirements):
                    raise ManagerError(
                        "release retirement evidence cardinality differs from plan"
                    )
                self._validate_retirement_evidence(
                    retirements, receipts, "release"
                )
            self._encode_retirement_evidence(receipts)
            raw = L.SessionReleaseEvidenceLayout(
                raw_release_id,
                _wire_bool(
                    "release mirror cleanup",
                    evidence.mirror_cleanup_confirmed,
                    32,
                ),
                0,
            )
            raw_outcome = L.SessionReleaseOutcomeLayout()
            self._call(
                "runtime session confirm release",
                self._library.function("orbitkv_session_confirm_release"),
                self._require_handle(),
                raw,
                self._retirement_evidence,
                len(receipts),
                ctypes.byref(raw_outcome),
            )
            try:
                if int(raw_outcome.reserved) != 0:
                    raise ManagerError(
                        "release outcome reserved field is nonzero"
                    )
                release_id = decode_operation_id(
                    raw_outcome.release_id,
                    EngineReleaseId,
                    L.SessionReleaseIdLayout,
                    "release outcome id",
                )
                if release_id != evidence.release_id:
                    raise ManagerError(
                        "release outcome id differs from the confirmed release"
                    )
                disposition = EngineReleaseDisposition(
                    int(raw_outcome.disposition)
                )
                outcome = EngineReleaseOutcome(release_id, disposition)
            except Exception as error:
                raise self._poison(
                    f"runtime session release outcome is invalid: {error}"
                ) from error
            if disposition is EngineReleaseDisposition.RECYCLE_PENDING:
                self._acknowledged_release_retries.add(evidence.release_id)
            else:
                self._pending_releases.pop(evidence.release_id)
                self._acknowledged_release_retries.discard(evidence.release_id)
            return outcome

    def _encode_retirement_evidence(
        self, values: tuple[EngineRetirementEvidence, ...]
    ) -> None:
        if len(values) > self._physical_pages:
            raise ManagerError(
                "session retirement evidence exceeds physical page capacity"
            )
        for index, item in enumerate(values):
            self._retirement_evidence[index] = _retirement_evidence_c(item)

    @staticmethod
    def _validate_retirement_evidence(
        retirements: tuple[EngineRetirement, ...],
        evidence: tuple[EngineRetirementEvidence, ...],
        operation: str,
    ) -> None:
        if any(
            item.page != retirement.page
            or item.backend_domain != retirement.backend_domain
            or item.backend_index != retirement.backend_index
            or item.acknowledged is not True
            for retirement, item in zip(retirements, evidence, strict=True)
        ):
            raise ManagerError(
                f"{operation} retirement evidence differs from plan"
            )

    def close(self) -> None:
        with self._lock:
            if not self._handle.value or self._destroy_attempted:
                return
            handle = self._handle
            self._destroy_attempted = True
            try:
                self._call(
                    "destroy runtime session",
                    self._library.function("orbitkv_session_destroy"),
                    handle,
                    require_handle=False,
                    allow_poisoned=True,
                )
            finally:
                self._handle = ctypes.c_void_p()
                self._prepared.clear()
                self._submitted.clear()
                self._pending_publications.clear()
                self._pending_releases.clear()
                self._pending_controls.clear()
                self._pending_relocations.clear()
                self._pending_attach_cancels.clear()
                self._finalized_attach_cancels.clear()
                self._acknowledged_release_retries.clear()

    def __enter__(self) -> CtypesRuntimeSession:
        self._require_handle()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
