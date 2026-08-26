"""Fail-closed completion tracking for SGLang-owned KV writes.

The canonical ABI8 runtime remains the allocator and reclamation authority.
This module only authorizes writes into already allocated structured arenas and
turns SGLang stream ordering into engine-neutral completion fences.  In
particular, data readiness and final use are deliberately separate events.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from orbitkv_runtime import (
    ArenaRegistration,
    BackendPageAddress,
    BackendTokenAddress,
    CompletionFence,
    DataPlaneEvidence,
    DataPlaneOperation,
    ExternalAppendTicket,
    ExternalTokenWrite,
    OperationContext,
    RequestLease,
    ResolvedTokenAddress,
    StepLease,
    TokenWriteReceipt,
)

from .external_validation import (
    ExternalAppendError,
    ExternalAppendPoisonedError,
    canonical_integer as _integer,
    copy_page as _copy_page,
    copy_registration as _copy_registration,
    copy_write as _copy_write,
    device_key as _device_key,
    same_stream as _same_stream,
)


_U64_LIMIT = 1 << 64
_MISSING = object()


@dataclass(frozen=True, slots=True)
class _ArenaState:
    arena: object
    registration: ArenaRegistration
    storage_token_bias: int
    fingerprint: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _Timeline:
    completion_domain: int
    completion_value: int
    stream: object | None


@dataclass(slots=True)
class _FenceState:
    issued: CompletionFence
    canonical: CompletionFence
    event: object | None
    purpose: str
    covered_pages: tuple[BackendPageAddress, ...]


@dataclass(slots=True)
class _PendingAppend:
    ticket: ExternalAppendTicket
    ticket_id: int
    writes: tuple[ExternalTokenWrite, ...]
    resolved: tuple[ResolvedTokenAddress, ...]
    transactions: tuple[StepLease, ...]
    covered_pages: tuple[BackendPageAddress, ...]
    timeline: _Timeline
    launch_locations: tuple[tuple[int, tuple[int, ...]], ...] | None = None


class SglangExternalWriteAdapter:
    """Two-phase external-write coordinator for structured SGLang arenas.

    The adapter never allocates, retires, reuses, or mutates an arena.  It
    verifies manager-issued addresses, reserves their authority until the
    engine records data readiness, and owns only the CUDA event timeline.
    """

    def __init__(
        self,
        arenas: Sequence[object] | Mapping[int, object],
        *,
        device_module: object | None = None,
        device: object | None = None,
        adapter_id: str | None = None,
    ) -> None:
        if adapter_id is not None and (
            not isinstance(adapter_id, str) or not adapter_id
        ):
            raise ExternalAppendError("adapter_id must be a nonempty string")
        values = self._arena_values(arenas)
        if not values:
            raise ExternalAppendError("at least one structured arena is required")

        states: list[_ArenaState] = []
        observed_devices: list[tuple[str, int | None]] = []
        storage_spans: list[tuple[int, int]] = []
        for arena in values:
            registration = _copy_registration(
                getattr(arena, "registration", None)
            )
            bias = _integer(
                "structured arena storage_token_bias",
                getattr(arena, "storage_token_bias", None),
            )
            if bias != registration.page_tokens:
                raise ExternalAppendError(
                    "structured arena must reserve exactly one dummy page"
                )
            fingerprint, component_devices, component_spans = self._arena_fingerprint(
                arena, registration, bias
            )
            observed_devices.extend(component_devices)
            storage_spans.extend(component_spans)
            states.append(
                _ArenaState(arena, registration, bias, fingerprint)
            )

        pool_ids = tuple(item.registration.pool_id for item in states)
        class_ids = tuple(item.registration.class_id for item in states)
        identities = tuple(
            (item.registration.backend_domain, item.registration.class_id)
            for item in states
        )
        epochs = {item.registration.engine_epoch for item in states}
        if len(set(pool_ids)) != len(pool_ids):
            raise ExternalAppendError("structured arenas have duplicate pool ids")
        if len(set(class_ids)) != len(class_ids):
            raise ExternalAppendError("structured arenas have duplicate class ids")
        if len(set(identities)) != len(identities):
            raise ExternalAppendError(
                "structured arenas have duplicate class/domain identities"
            )
        if len(epochs) != 1:
            raise ExternalAppendError(
                "structured arenas must belong to one engine epoch"
            )
        if not observed_devices:
            raise ExternalAppendError(
                "structured arena components do not expose a device"
            )
        observed_device = observed_devices[0]
        if observed_device[0] == "cuda" and observed_device[1] is None:
            raise ExternalAppendError(
                "structured arena CUDA device must have a concrete index"
            )
        if any(observed_device != item for item in observed_devices[1:]):
            raise ExternalAppendError(
                "structured arena components span multiple devices"
            )
        configured_device = (
            observed_device if device is None else _device_key(device)
        )
        if configured_device[0] == "cuda" and configured_device[1] is None:
            raise ExternalAppendError(
                "configured CUDA device must have a concrete index"
            )
        if configured_device != observed_device:
            raise ExternalAppendError(
                "configured device differs from structured arena storage"
            )
        ordered_spans = sorted(storage_spans)
        for index, (begin, _end) in enumerate(ordered_spans):
            if index and begin < ordered_spans[index - 1][1]:
                raise ExternalAppendError(
                    "structured arena component storage aliases"
                )

        if device is None:
            first_component = tuple(getattr(values[0], "components"))[0]
            device = getattr(first_component.tensor, "device")
        if configured_device[0] == "cuda":
            if device_module is None:
                try:
                    import torch
                except ImportError as error:
                    raise ExternalAppendError(
                        "CUDA structured arenas require a device module"
                    ) from error
                get_module = getattr(torch, "get_device_module", None)
                device_module = (
                    get_module(device) if callable(get_module) else torch.cuda
                )
            elif callable(getattr(device_module, "get_device_module", None)):
                device_module = device_module.get_device_module(device)
            if not callable(getattr(device_module, "current_stream", None)):
                raise ExternalAppendError(
                    "CUDA device module lacks current_stream"
                )
            if not callable(getattr(device_module, "Event", None)):
                raise ExternalAppendError("CUDA device module lacks Event")

        label = adapter_id if adapter_id is not None else "sglang-external"
        self._adapter_id = f"{label}:{uuid4().hex}"
        self._engine_epoch = epochs.pop()
        self._device = device
        self._device_key = observed_device
        self._device_module = device_module
        self._is_cuda = observed_device[0] == "cuda"
        self._completion_domain = (observed_device[1] or 0) + 1
        self._arenas = {item.registration.pool_id: item for item in states}
        self._arenas_by_class = {
            item.registration.class_id: item for item in states
        }
        self._class_ids = class_ids
        self._lock = RLock()
        self._poison_reason: str | None = None
        self._next_ticket_id = 1
        self._next_fence_id = 1
        self._next_operation_id = 1
        self._pending: dict[int, _PendingAppend] = {}
        self._reserved_transactions: dict[StepLease, int] = {}
        self._reserved_pages: dict[BackendPageAddress, int] = {}
        self._reserved_domains: dict[int, int] = {}
        self._observed_transactions: dict[tuple[int, int], int] = {}
        self._generations: dict[tuple[int, int], int] = {}
        self._domain_values: dict[int, int] = {}
        self._domain_last_fence: dict[int, int] = {}
        self._latest_page_fence: dict[BackendPageAddress, int] = {}
        self._fences: dict[int, _FenceState] = {}
        self._event_identities: set[int] = set()

    @staticmethod
    def _arena_values(
        arenas: Sequence[object] | Mapping[int, object],
    ) -> tuple[object, ...]:
        try:
            if isinstance(arenas, Mapping):
                items = tuple(arenas.items())
                values = tuple(item[1] for item in items)
                for key, arena in items:
                    registration = getattr(arena, "registration", None)
                    if key != getattr(registration, "class_id", _MISSING):
                        raise ExternalAppendError(
                            "structured arena mapping key differs from class id"
                        )
            else:
                values = tuple(arenas)
            return tuple(
                sorted(
                    values,
                    key=lambda arena: _integer(
                        "structured arena class id",
                        getattr(
                            getattr(arena, "registration", None),
                            "class_id",
                            None,
                        ),
                    ),
                )
            )
        except ExternalAppendError:
            raise
        except Exception as error:
            raise ExternalAppendError("structured arenas are unreadable") from error

    @staticmethod
    def _arena_fingerprint(
        arena: object, registration: ArenaRegistration, bias: int
    ) -> tuple[
        tuple[object, ...],
        tuple[tuple[str, int | None], ...],
        tuple[tuple[int, int], ...],
    ]:
        try:
            components = tuple(arena.components)
        except Exception as error:
            raise ExternalAppendError(
                "structured arena components are unreadable"
            ) from error
        if not components:
            raise ExternalAppendError(
                "structured arena components must not be empty"
            )
        offset = 0
        component_values: list[tuple[object, ...]] = []
        devices: list[tuple[str, int | None]] = []
        spans: list[tuple[int, int]] = []
        tensors: set[int] = set()
        for component in components:
            try:
                byte_offset = _integer(
                    "component token_byte_offset",
                    component.token_byte_offset,
                )
                byte_count = _integer(
                    "component token_byte_count",
                    component.token_byte_count,
                    positive=True,
                )
                if byte_offset != offset:
                    raise ExternalAppendError(
                        "structured component byte offsets are not contiguous"
                    )
                tensor = component.tensor
                tensor_id = id(tensor)
                if tensor_id in tensors:
                    raise ExternalAppendError(
                        "structured arena aliases a component tensor"
                    )
                tensors.add(tensor_id)
                device = _device_key(getattr(tensor, "device", None))
                if device[0] == "cuda" and device[1] is None:
                    raise ExternalAppendError(
                        "structured component CUDA device lacks an index"
                    )
                shape = tuple(
                    _integer("component tensor shape", item, positive=True)
                    for item in getattr(tensor, "shape", ())
                )
                if not shape:
                    raise ExternalAppendError(
                        "structured component tensor shape is missing"
                    )
                dtype = getattr(tensor, "dtype", _MISSING)
                if dtype is _MISSING:
                    raise ExternalAppendError(
                        "structured component tensor dtype is missing"
                    )
                contiguous = getattr(tensor, "is_contiguous", None)
                if not callable(contiguous) or contiguous() is not True:
                    raise ExternalAppendError(
                        "structured component tensor is not contiguous"
                    )
                if getattr(tensor, "requires_grad", False) is not False:
                    raise ExternalAppendError(
                        "structured component tensor requires gradients"
                    )
                element_size = getattr(tensor, "element_size", None)
                numel = getattr(tensor, "numel", None)
                data_ptr = getattr(tensor, "data_ptr", None)
                if not all(
                    callable(item) for item in (element_size, numel, data_ptr)
                ):
                    raise ExternalAppendError(
                        "structured component tensor storage metadata is missing"
                    )
                item_bytes = _integer(
                    "component tensor element size",
                    element_size(),
                    positive=True,
                )
                element_count = _integer(
                    "component tensor element count", numel(), positive=True
                )
                expected_count = 1
                for extent in shape:
                    expected_count *= extent
                if element_count != expected_count:
                    raise ExternalAppendError(
                        "structured component tensor element count changed"
                    )
                nbytes = _integer(
                    "component tensor byte span",
                    getattr(tensor, "nbytes", element_count * item_bytes),
                    positive=True,
                )
                if nbytes != element_count * item_bytes:
                    raise ExternalAppendError(
                        "structured component tensor byte span changed"
                    )
                pointer = _integer(
                    "component tensor pointer", data_ptr(), positive=True
                )
                storage_offset_method = getattr(tensor, "storage_offset", None)
                storage_offset = (
                    _integer(
                        "component tensor storage offset",
                        storage_offset_method(),
                    )
                    if callable(storage_offset_method)
                    else 0
                )
                stride_method = getattr(tensor, "stride", None)
                stride = (
                    tuple(
                        _integer("component tensor stride", item)
                        for item in stride_method()
                    )
                    if callable(stride_method)
                    else ()
                )
                if stride and len(stride) != len(shape):
                    raise ExternalAppendError(
                        "structured component tensor stride rank changed"
                    )
                devices.append(device)
                spans.append((pointer, pointer + nbytes))
                component_values.append(
                    (
                        _integer(
                            "component global_layer_id",
                            component.global_layer_id,
                        ),
                        _integer(
                            "component local_layer_id",
                            component.local_layer_id,
                        ),
                        str(component.name),
                        tensor_id,
                        byte_offset,
                        byte_count,
                        device,
                        shape,
                        type(dtype),
                        repr(dtype),
                        item_bytes,
                        element_count,
                        nbytes,
                        pointer,
                        storage_offset,
                        stride,
                    )
                )
                offset += byte_count
            except AttributeError as error:
                raise ExternalAppendError(
                    "structured arena component is invalid"
                ) from error
        if offset != registration.token_bytes:
            raise ExternalAppendError(
                "structured component bytes differ from aggregate token width"
            )
        registration_values = tuple(
            getattr(registration, name)
            for name in (
                "engine_epoch",
                "pool_epoch",
                "pool_id",
                "class_id",
                "backend_domain",
                "page_count",
                "page_tokens",
                "token_bytes",
                "backend_base_index",
                "first_page_id",
            )
        )
        return (
            (registration_values, bias, tuple(component_values)),
            tuple(devices),
            tuple(spans),
        )

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def engine_epoch(self) -> int:
        return self._engine_epoch

    @property
    def completion_domain(self) -> int:
        return self._completion_domain

    @property
    def poison_reason(self) -> str | None:
        with self._lock:
            return self._poison_reason

    def poison(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("poison reason must be a nonempty string")
        with self._lock:
            if self._poison_reason is None:
                self._poison_reason = reason

    def close(self) -> None:
        """Require quiescence before dropping adapter-owned evidence."""

        with self._lock:
            if self._pending or self._fences:
                self.poison("shutdown encountered outstanding external evidence")
                raise ExternalAppendPoisonedError(
                    "SGLang external-write adapter has outstanding evidence"
                )

    def _poisoned(
        self, reason: str, error: BaseException | None = None
    ) -> ExternalAppendPoisonedError:
        self.poison(reason)
        failure = ExternalAppendPoisonedError(
            f"SGLang external-write adapter is poisoned: {self._poison_reason}"
        )
        if error is not None:
            failure.__cause__ = error
        return failure

    def _ensure_healthy(self) -> None:
        if self._poison_reason is not None:
            raise ExternalAppendPoisonedError(
                "SGLang external-write adapter is poisoned: "
                + self._poison_reason
            )

    def _assert_arena_stable(self, state: _ArenaState) -> None:
        registration = _copy_registration(
            getattr(state.arena, "registration", None)
        )
        bias = _integer(
            "structured arena storage_token_bias",
            getattr(state.arena, "storage_token_bias", None),
        )
        fingerprint, _devices, _spans = self._arena_fingerprint(
            state.arena, registration, bias
        )
        if (
            registration != state.registration
            or bias != state.storage_token_bias
            or fingerprint != state.fingerprint
        ):
            raise ExternalAppendError(
                "structured arena changed after adapter construction"
            )

    def _validate_page(
        self, address: BackendPageAddress
    ) -> tuple[_ArenaState, int]:
        if not isinstance(address, BackendPageAddress):
            raise ExternalAppendError(
                "page address must be a BackendPageAddress"
            )
        try:
            state = self._arenas[address.page.pool_id]
        except (AttributeError, KeyError) as error:
            raise ExternalAppendError(
                "page names an unregistered structured arena"
            ) from error
        self._assert_arena_stable(state)
        registration = state.registration
        page_index = address.page.page_id - registration.first_page_id
        backend_page_index = (
            address.backend_index - registration.backend_base_index
        )
        if (
            address.page.engine_epoch != self._engine_epoch
            or address.page.engine_epoch != registration.engine_epoch
            or address.page.pool_epoch != registration.pool_epoch
            or address.page.pool_id != registration.pool_id
            or address.class_id != registration.class_id
            or address.backend_domain != registration.backend_domain
            or not 0 <= page_index < registration.page_count
            or page_index != backend_page_index
        ):
            raise ExternalAppendError(
                "page address does not match its structured arena identity"
            )
        return state, page_index

    def _resolve_token(
        self, address: BackendTokenAddress
    ) -> ResolvedTokenAddress:
        if not isinstance(address, BackendTokenAddress):
            raise ExternalAppendError(
                "destination must be a BackendTokenAddress"
            )
        state, page_index = self._validate_page(address.page_address)
        if address.token_offset >= state.registration.page_tokens:
            raise ExternalAppendError("token offset is outside its page")
        return ResolvedTokenAddress(
            address=address,
            arena=state.arena,
            arena_page_index=page_index,
            token_index=(
                page_index * state.registration.page_tokens
                + address.token_offset
            ),
            byte_length=state.registration.token_bytes,
        )

    @staticmethod
    def _physical_page_key(address: BackendPageAddress) -> tuple[int, int]:
        return address.page.pool_id, address.page.page_id

    def _plan_generations(
        self, pages: Sequence[BackendPageAddress]
    ) -> dict[tuple[int, int], int]:
        generations = dict(self._generations)
        batch: dict[tuple[int, int], int] = {}
        reserved_physical = {
            self._physical_page_key(item) for item in self._reserved_pages
        }
        for page in pages:
            self._validate_page(page)
            key = self._physical_page_key(page)
            generation = page.page.generation
            if key in reserved_physical:
                raise ExternalAppendError(
                    "page generation is reserved by a pending external append"
                )
            if key in batch and batch[key] != generation:
                raise ExternalAppendError(
                    "one operation names multiple generations of a physical page"
                )
            batch[key] = generation
            previous = generations.get(key)
            if previous is not None and generation < previous:
                raise ExternalAppendError(
                    "page address carries a stale generation"
                )
            if previous is None or generation > previous:
                generations[key] = generation
        return generations

    @staticmethod
    def _validate_context(context: OperationContext, engine_epoch: int) -> None:
        if (
            not isinstance(context, OperationContext)
            or context.operation is not DataPlaneOperation.APPEND
            or not isinstance(context.request, RequestLease)
            or not isinstance(context.transaction, StepLease)
            or context.request.engine_epoch != engine_epoch
            or context.transaction.engine_epoch != engine_epoch
        ):
            raise ExternalAppendError(
                "external write context does not match the adapter epoch"
            )

    @staticmethod
    def _reject_duplicates(name: str, values: Sequence[object]) -> None:
        if len(set(values)) != len(values):
            raise ExternalAppendError(f"duplicate {name}")

    def _current_stream(self) -> object | None:
        if not self._is_cuda:
            return None
        assert self._device_module is not None
        return self._device_module.current_stream(self._device)

    def _validate_domain(self, completion_domain: object) -> int:
        return _integer(
            "completion domain", completion_domain, positive=True
        )

    def _dependencies(
        self, completion_domain: int, pages: Sequence[BackendPageAddress]
    ) -> tuple[object, ...]:
        fence_ids: list[int] = []
        predecessor = self._domain_last_fence.get(completion_domain)
        if predecessor is not None:
            fence_ids.append(predecessor)
        fence_ids.extend(
            self._latest_page_fence[page]
            for page in pages
            if page in self._latest_page_fence
        )
        events: list[object] = []
        seen: set[int] = set()
        for fence_id in fence_ids:
            state = self._fences.get(fence_id)
            if state is None:
                raise ExternalAppendError(
                    "completion dependency is unknown"
                )
            if state.event is not None and id(state.event) not in seen:
                seen.add(id(state.event))
                events.append(state.event)
        return tuple(events)

    def _prepare_timeline(
        self,
        completion_domain: object,
        pages: Sequence[BackendPageAddress],
    ) -> _Timeline:
        domain = self._validate_domain(completion_domain)
        if domain in self._reserved_domains:
            raise ExternalAppendError(
                "completion domain is reserved by a pending external append"
            )
        value = self._domain_values.get(domain, 0) + 1
        if value >= _U64_LIMIT:
            raise ExternalAppendError("completion sequence is exhausted")
        if self._next_fence_id >= _U64_LIMIT:
            raise ExternalAppendError("completion fence sequence is exhausted")
        stream = self._current_stream()
        if self._is_cuda:
            if stream is None or not callable(getattr(stream, "wait_event", None)):
                raise ExternalAppendError(
                    "current CUDA stream cannot wait for dependencies"
                )
            for event in self._dependencies(domain, pages):
                stream.wait_event(event)
        return _Timeline(domain, value, stream)

    def _new_event(self, stream: object | None) -> object | None:
        if not self._is_cuda:
            return None
        assert self._device_module is not None and stream is not None
        event = self._device_module.Event()
        if event is None or id(event) in self._event_identities:
            raise ExternalAppendError(
                "CUDA event factory did not return a distinct event"
            )
        record = getattr(event, "record", None)
        if not callable(record):
            raise ExternalAppendError("CUDA event cannot be recorded")
        record(stream)
        return event

    def _record_fence(
        self,
        timeline: _Timeline,
        *,
        purpose: str,
        pages: Sequence[BackendPageAddress],
    ) -> CompletionFence:
        canonical_pages = tuple(pages)
        self._reject_duplicates("completion page generation", canonical_pages)
        fence_id = self._next_fence_id
        canonical = CompletionFence(
            self._adapter_id,
            self._engine_epoch,
            timeline.completion_domain,
            timeline.completion_value,
            fence_id,
        )
        issued = CompletionFence(
            canonical.adapter_id,
            canonical.engine_epoch,
            canonical.completion_domain,
            canonical.completion_value,
            canonical.fence_id,
        )
        event = self._new_event(timeline.stream)
        state = _FenceState(issued, canonical, event, purpose, canonical_pages)
        self._fences[fence_id] = state
        self._next_fence_id += 1
        self._domain_values[timeline.completion_domain] = timeline.completion_value
        self._domain_last_fence[timeline.completion_domain] = fence_id
        if event is not None:
            self._event_identities.add(id(event))
        for page in canonical_pages:
            self._latest_page_fence[page] = fence_id
        return issued

    def prepare_external_append(
        self,
        writes: Sequence[ExternalTokenWrite],
        *,
        completion_domain: int = 1,
    ) -> ExternalAppendTicket:
        """Preflight and reserve a complete SGLang append before mutation."""

        with self._lock:
            self._ensure_healthy()
            try:
                supplied = tuple(writes)
            except Exception as error:
                raise ExternalAppendError(
                    "external append batch is unreadable"
                ) from error
            if not supplied:
                raise ExternalAppendError(
                    "external append batch must not be empty"
                )
            private_writes = tuple(_copy_write(item) for item in supplied)

            transaction_owners: dict[StepLease, RequestLease] = {}
            transactions: list[StepLease] = []
            for write in private_writes:
                context = write.context
                self._validate_context(context, self._engine_epoch)
                previous = transaction_owners.setdefault(
                    context.transaction, context.request
                )
                if previous != context.request:
                    raise ExternalAppendError(
                        "one transaction cannot authorize multiple requests"
                    )
                if context.transaction not in transactions:
                    transactions.append(context.transaction)
                transaction_key = (
                    context.transaction.slot, context.transaction.engine_epoch
                )
                if context.transaction.generation <= self._observed_transactions.get(
                    transaction_key, 0
                ):
                    raise ExternalAppendError(
                        "manager transaction was already observed by the adapter"
                    )
                if context.transaction in self._reserved_transactions:
                    raise ExternalAppendError(
                        "manager transaction is reserved by an external append"
                    )

            pages = tuple(
                write.destination.page_address for write in private_writes
            )
            generations = self._plan_generations(pages)
            resolved = tuple(
                self._resolve_token(write.destination)
                for write in private_writes
            )
            for write, destination in zip(
                private_writes, resolved, strict=True
            ):
                if write.byte_count != destination.byte_length:
                    raise ExternalAppendError(
                        "external write byte count differs from aggregate arena width"
                    )
            self._reject_duplicates(
                "external write destination",
                tuple(write.destination for write in private_writes),
            )
            self._reject_duplicates(
                "logical transaction/class/token write",
                tuple(
                    (write.context, write.destination.class_id, write.token_id)
                    for write in private_writes
                ),
            )
            self._validate_class_coverage(private_writes)

            if self._next_ticket_id >= _U64_LIMIT:
                raise ExternalAppendError("external ticket sequence is exhausted")
            covered_pages = tuple(dict.fromkeys(pages))
            timeline = self._prepare_timeline(
                completion_domain, covered_pages
            )
            ticket_id = self._next_ticket_id
            public_writes = tuple(_copy_write(item) for item in private_writes)
            public_resolved = tuple(
                ResolvedTokenAddress(
                    public_write.destination,
                    private_destination.arena,
                    private_destination.arena_page_index,
                    private_destination.token_index,
                    private_destination.byte_length,
                )
                for public_write, private_destination in zip(
                    public_writes, resolved, strict=True
                )
            )
            ticket = ExternalAppendTicket(
                self._adapter_id,
                ticket_id,
                public_writes,
                public_resolved,
                timeline.completion_domain,
                timeline.stream,
            )
            self._pending[ticket_id] = _PendingAppend(
                ticket,
                ticket_id,
                private_writes,
                resolved,
                tuple(transactions),
                covered_pages,
                timeline,
            )
            self._next_ticket_id += 1
            self._generations = generations
            for transaction in transactions:
                self._reserved_transactions[transaction] = ticket_id
            for page in covered_pages:
                self._reserved_pages[page] = ticket_id
                self._latest_page_fence.pop(page, None)
            self._reserved_domains[timeline.completion_domain] = ticket_id
            return ticket

    def _validate_class_coverage(
        self, writes: Sequence[ExternalTokenWrite]
    ) -> None:
        seen: set[tuple[OperationContext, int]] = set()
        current_key: tuple[OperationContext, int] | None = None
        current_classes: list[int] = []
        for write in writes:
            key = write.context, write.token_id
            if current_key is None:
                current_key = key
            if key != current_key:
                if tuple(current_classes) != self._class_ids:
                    raise ExternalAppendError(
                        "external append does not cover every structured class in order"
                    )
                seen.add(current_key)
                if key in seen:
                    raise ExternalAppendError(
                        "external append logical token groups are not canonical"
                    )
                current_key = key
                current_classes = []
            current_classes.append(write.destination.class_id)
        if tuple(current_classes) != self._class_ids:
            raise ExternalAppendError(
                "external append does not cover every structured class in order"
            )

    def _find_pending(self, ticket: object) -> _PendingAppend:
        if not isinstance(ticket, ExternalAppendTicket):
            raise ExternalAppendError(
                "external append ticket has an invalid type"
            )
        for pending in self._pending.values():
            if pending.ticket is ticket:
                return pending
        try:
            adapter_id = ticket.adapter_id
            ticket_id = ticket.ticket_id
        except Exception as error:
            raise ExternalAppendError(
                "external append ticket identity is unreadable"
            ) from error
        if adapter_id != self._adapter_id:
            raise ExternalAppendError(
                "external append ticket belongs to another adapter"
            )
        if isinstance(ticket_id, bool) or not isinstance(ticket_id, int):
            raise ExternalAppendError(
                "external append ticket id is invalid"
            )
        if ticket_id in self._pending:
            raise ExternalAppendError(
                "external append ticket is forged or reconstructed"
            )
        raise ExternalAppendError(
            "external append ticket is unknown or already consumed"
        )

    def _validate_issued_ticket(self, pending: _PendingAppend) -> None:
        ticket = pending.ticket
        if (
            ticket.adapter_id != self._adapter_id
            or ticket.ticket_id != pending.ticket_id
            or ticket.completion_domain != pending.timeline.completion_domain
            or ticket.launch_context is not pending.timeline.stream
            or not isinstance(ticket.writes, tuple)
            or not isinstance(ticket.resolved_destinations, tuple)
        ):
            raise ValueError("external append ticket identity changed after issue")
        writes = tuple(_copy_write(item) for item in ticket.writes)
        if writes != pending.writes or len(ticket.resolved_destinations) != len(
            pending.resolved
        ):
            raise ValueError("external append ticket writes changed after issue")
        for actual, expected in zip(
            ticket.resolved_destinations, pending.resolved, strict=True
        ):
            if (
                not isinstance(actual, ResolvedTokenAddress)
                or actual.address != expected.address
                or actual.arena is not expected.arena
                or actual.arena_page_index != expected.arena_page_index
                or actual.token_index != expected.token_index
                or actual.byte_length != expected.byte_length
            ):
                raise ValueError(
                    "external append resolved destination changed after issue"
                )

    def _validate_reservations(self, pending: _PendingAppend) -> None:
        for transaction in pending.transactions:
            if self._reserved_transactions.get(transaction) != pending.ticket_id:
                raise RuntimeError(
                    "external append transaction reservation was lost"
                )
        for page in pending.covered_pages:
            if self._reserved_pages.get(page) != pending.ticket_id:
                raise RuntimeError("external append page reservation was lost")
        if (
            self._reserved_domains.get(pending.timeline.completion_domain)
            != pending.ticket_id
        ):
            raise RuntimeError(
                "external append completion-domain reservation was lost"
            )

    def _checked_pending(self, ticket: object) -> _PendingAppend:
        pending = self._find_pending(ticket)
        try:
            self._validate_issued_ticket(pending)
            self._validate_reservations(pending)
            for destination in pending.resolved:
                state = self._arenas[destination.address.page.pool_id]
                self._assert_arena_stable(state)
        except Exception as error:
            raise self._poisoned(
                "issued external append ticket failed integrity validation",
                error,
            )
        return pending

    @staticmethod
    def _location_vector(name: str, value: object) -> tuple[int, ...]:
        try:
            ndim = getattr(value, "ndim", None)
            if ndim is not None and _integer(f"{name} rank", ndim) != 1:
                raise ExternalAppendError(f"{name} must be one-dimensional")
            current = (
                value.detach()
                if callable(getattr(value, "detach", None))
                else value
            )
            current = (
                current.cpu()
                if callable(getattr(current, "cpu", None))
                else current
            )
            raw = (
                current.tolist()
                if callable(getattr(current, "tolist", None))
                else list(current)
            )
        except ExternalAppendError:
            raise
        except Exception as error:
            raise ExternalAppendError(f"{name} is unreadable") from error
        if not isinstance(raw, (tuple, list)):
            raise ExternalAppendError(f"{name} is not a vector")
        return tuple(_integer(name, item) for item in raw)

    def _canonical_locations(
        self, value: object, *, label: str, require_mapping: bool
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        if isinstance(value, Mapping):
            if set(value) != set(self._class_ids):
                raise ExternalAppendError(
                    f"{label} classes differ from structured arenas"
                )
            return tuple(
                (
                    class_id,
                    self._location_vector(
                        f"{label} class {class_id}", value[class_id]
                    ),
                )
                for class_id in self._class_ids
            )
        if require_mapping:
            raise ExternalAppendError(f"{label} must be a mapping")
        try:
            items = tuple(value)  # type: ignore[arg-type]
        except Exception as error:
            raise ExternalAppendError(f"{label} is unreadable") from error
        if len(items) != len(self._class_ids):
            raise ExternalAppendError(
                f"{label} classes differ from structured arenas"
            )
        result = []
        for expected_class, item in zip(self._class_ids, items, strict=True):
            try:
                class_id, locations = item
            except Exception as error:
                raise ExternalAppendError(f"{label} entry is invalid") from error
            if class_id != expected_class:
                raise ExternalAppendError(
                    f"{label} is not in structured arena order"
                )
            result.append(
                (
                    class_id,
                    self._location_vector(
                        f"{label} class {class_id}", locations
                    ),
                )
            )
        return tuple(result)

    def _ticket_locations(
        self, pending: _PendingAppend
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        locations = {class_id: [] for class_id in self._class_ids}
        for destination in pending.resolved:
            class_id = destination.address.class_id
            state = self._arenas_by_class[class_id]
            locations[class_id].append(
                destination.token_index + state.storage_token_bias
            )
        return tuple(
            (class_id, tuple(locations[class_id]))
            for class_id in self._class_ids
        )

    def validate_launch(
        self,
        ticket: ExternalAppendTicket,
        expected_locations_by_class: (
            Mapping[int, object] | Sequence[tuple[int, object]]
        ),
        live_locations_by_class: Mapping[int, object],
        *,
        current_stream: object | None,
        completion_domain: int | None = None,
    ) -> None:
        """Bind the issued ticket to the exact live SGLang launch inputs."""

        with self._lock:
            self._ensure_healthy()
            pending = self._checked_pending(ticket)
            try:
                if completion_domain is not None and self._validate_domain(
                    completion_domain
                ) != pending.timeline.completion_domain:
                    raise ExternalAppendError(
                        "launch completion domain differs from its ticket"
                    )
                if not _same_stream(current_stream, pending.timeline.stream):
                    raise ExternalAppendError(
                        "launch stream differs from the prepared stream"
                    )
                expected = self._canonical_locations(
                    expected_locations_by_class,
                    label="expected launch locations",
                    require_mapping=False,
                )
                live = self._canonical_locations(
                    live_locations_by_class,
                    label="live launch locations",
                    require_mapping=True,
                )
                if expected != self._ticket_locations(pending):
                    raise ExternalAppendError(
                        "expected launch locations differ from ticket destinations"
                    )
                if live != expected:
                    raise ExternalAppendError(
                        "live SGLang locations changed after ticket preparation"
                    )
                if (
                    pending.launch_locations is not None
                    and pending.launch_locations != expected
                ):
                    raise ExternalAppendError(
                        "external append launch was already validated differently"
                    )
                pending.launch_locations = expected
            except ExternalAppendPoisonedError:
                raise
            except Exception as error:
                raise self._poisoned(
                    "external append launch validation failed after ticket issue",
                    error,
                )

    def record_external_data_ready(
        self, ticket: ExternalAppendTicket
    ) -> DataPlaneEvidence:
        """Record data readiness on the stream captured at preparation."""

        with self._lock:
            self._ensure_healthy()
            pending = self._checked_pending(ticket)
            try:
                if pending.launch_locations is None:
                    raise RuntimeError(
                        "external append launch was not validated"
                    )
                if self._next_operation_id >= _U64_LIMIT:
                    raise RuntimeError("external operation sequence is exhausted")
                receipts = tuple(
                    TokenWriteReceipt(
                        write.context,
                        write.token_id,
                        write.destination,
                        write.byte_count,
                    )
                    for write in pending.writes
                )
                fence = self._record_fence(
                    pending.timeline,
                    purpose="data_ready",
                    pages=pending.covered_pages,
                )
                evidence = DataPlaneEvidence(
                    self._next_operation_id,
                    DataPlaneOperation.APPEND,
                    (),
                    receipts,
                    fence,
                )
            except Exception as error:
                raise self._poisoned(
                    "external append completion became uncertain", error
                )

            self._next_operation_id += 1
            del self._pending[pending.ticket_id]
            for transaction in pending.transactions:
                if self._reserved_transactions.pop(transaction) != pending.ticket_id:
                    raise AssertionError(
                        "validated transaction reservation changed during commit"
                    )
                transaction_key = (transaction.slot, transaction.engine_epoch)
                self._observed_transactions[transaction_key] = max(
                    self._observed_transactions.get(transaction_key, 0),
                    transaction.generation,
                )
            for page in pending.covered_pages:
                if self._reserved_pages.pop(page) != pending.ticket_id:
                    raise AssertionError(
                        "validated page reservation changed during commit"
                    )
            if (
                self._reserved_domains.pop(pending.timeline.completion_domain)
                != pending.ticket_id
            ):
                raise AssertionError(
                    "validated domain reservation changed during commit"
                )
            return evidence

    def record_last_use(
        self,
        pages: Sequence[BackendPageAddress],
        *,
        completion_domain: int = 1,
    ) -> CompletionFence:
        """Record final consumers after waiting on each page's latest event."""

        with self._lock:
            self._ensure_healthy()
            try:
                supplied = tuple(pages)
            except Exception as error:
                raise ExternalAppendError("last-use pages are unreadable") from error
            if not supplied:
                raise ExternalAppendError(
                    "last-use fence must cover at least one exact page generation"
                )
            canonical = tuple(_copy_page(item) for item in supplied)
            self._reject_duplicates("last-use page generation", canonical)
            generations = self._plan_generations(canonical)
            domain = self._validate_domain(completion_domain)
            if domain in self._reserved_domains:
                raise ExternalAppendError(
                    "completion domain is reserved by a pending external append"
                )
            try:
                timeline = self._prepare_timeline(domain, canonical)
                fence = self._record_fence(
                    timeline, purpose="last_use", pages=canonical
                )
            except Exception as error:
                raise self._poisoned(
                    "last-use event recording became uncertain", error
                )
            self._generations = generations
            return fence

    def _find_fence(self, fence: object) -> _FenceState:
        if not isinstance(fence, CompletionFence):
            raise ExternalAppendError("completion fence has an invalid type")
        for state in self._fences.values():
            if state.issued is fence:
                try:
                    if fence != state.canonical:
                        raise ValueError(
                            "completion fence changed after issue"
                        )
                except Exception as error:
                    raise self._poisoned(
                        "issued completion fence failed integrity validation",
                        error,
                    )
                return state
        try:
            adapter_id = fence.adapter_id
            fence_id = fence.fence_id
        except Exception as error:
            raise ExternalAppendError("completion fence is unreadable") from error
        if adapter_id != self._adapter_id:
            raise ExternalAppendError(
                "completion fence belongs to another adapter"
            )
        if fence_id in self._fences:
            raise ExternalAppendError(
                "completion fence is forged or reconstructed"
            )
        raise ExternalAppendError("completion fence is unknown")

    def event_for(self, fence: CompletionFence) -> object | None:
        """Return the exact raw event associated with an issued fence."""

        with self._lock:
            self._ensure_healthy()
            return self._find_fence(fence).event

    def retire_completion(self, fence: CompletionFence) -> None:
        """Release completed evidence after every consumer accepted it."""

        with self._lock:
            self._ensure_healthy()
            state = self._find_fence(fence)
            if state.purpose != "last_use":
                raise ExternalAppendError(
                    "only last-use completion may be retired"
                )
            if state.event is not None:
                query = getattr(state.event, "query", None)
                if not callable(query):
                    raise self._poisoned(
                        "last-use event cannot confirm retirement"
                    )
                try:
                    if not bool(query()):
                        raise ExternalAppendError(
                            "last-use completion is not ready for retirement"
                        )
                except ExternalAppendError:
                    raise
                except Exception as error:
                    raise self._poisoned(
                        "last-use event query became uncertain", error
                    )
            fence_ids = {state.issued.fence_id}
            for page in state.covered_pages:
                if self._latest_page_fence.get(page) == state.issued.fence_id:
                    del self._latest_page_fence[page]
            for candidate in tuple(self._fences.values()):
                if (
                    candidate.purpose == "data_ready"
                    and candidate.covered_pages
                    and set(candidate.covered_pages).issubset(state.covered_pages)
                    and candidate.issued.completion_domain
                    == state.issued.completion_domain
                    and candidate.issued.completion_value
                    < state.issued.completion_value
                ):
                    fence_ids.add(candidate.issued.fence_id)
            for fence_id in fence_ids:
                retired = self._fences.pop(fence_id)
                if retired.event is not None:
                    self._event_identities.discard(id(retired.event))
            domain = state.issued.completion_domain
            if self._domain_last_fence.get(domain) in fence_ids:
                replacement = max(
                    (
                        candidate.issued.fence_id
                        for candidate in self._fences.values()
                        if candidate.issued.completion_domain == domain
                    ),
                    default=None,
                )
                if replacement is None:
                    self._domain_last_fence.pop(domain, None)
                else:
                    self._domain_last_fence[domain] = replacement


__all__ = [
    "ExternalAppendError",
    "ExternalAppendPoisonedError",
    "SglangExternalWriteAdapter",
]
