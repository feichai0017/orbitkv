from __future__ import annotations

from dataclasses import replace

import pytest

from orbitkv_runtime import (
    ArenaRegistration,
    BackendTokenAddress,
    DataPlaneOperation,
    ExternalAppendTicket,
    ExternalTokenWrite,
    ExternalWriteCompletionAdapter,
    OperationContext,
    PageLease,
    RequestLease,
    StepLease,
)
from orbitkv_sglang.plugin.external_append import (
    ExternalAppendError,
    ExternalAppendPoisonedError,
    SglangExternalWriteAdapter,
)
from orbitkv_sglang.plugin.structured_arena import (
    SglangArenaComponent,
    SglangStructuredArena,
)


ENGINE_EPOCH = 7
POOL_EPOCH = 11
PAGE_TOKENS = 4


class _Tensor:
    _next_pointer = 4096

    def __init__(self, device: str, *, pointer: int | None = None) -> None:
        self.device = device
        self.shape = (36, 1, 1)
        self.dtype = "fake-bf16"
        self.requires_grad = False
        self.nbytes = 72
        self._pointer = (
            type(self)._next_pointer if pointer is None else pointer
        )
        type(self)._next_pointer += self.nbytes + 4096

    def is_contiguous(self) -> bool:
        return True

    def element_size(self) -> int:
        return 2

    def numel(self) -> int:
        return 36

    def data_ptr(self) -> int:
        return self._pointer

    def storage_offset(self) -> int:
        return 0

    def stride(self) -> tuple[int, int, int]:
        return (1, 1, 1)


class _Stream:
    def __init__(self, stream_id: int, device: str = "cuda:0") -> None:
        self.stream_id = stream_id
        self.device = device
        self.actions: list[tuple[str, object]] = []

    def wait_event(self, event: object) -> None:
        self.actions.append(("wait", event))


class _Event:
    def __init__(self) -> None:
        self.recorded_on: object | None = None

    def record(self, stream: _Stream) -> None:
        self.recorded_on = stream
        stream.actions.append(("record", self))

    def query(self) -> bool:
        return True


class _DeviceModule:
    def __init__(self) -> None:
        self.stream = _Stream(1)
        self.events: list[_Event] = []

    def current_stream(self, _device: object) -> _Stream:
        return self.stream

    def Event(self) -> _Event:
        event = _Event()
        self.events.append(event)
        return event


class _FailingEvent(_Event):
    def record(self, stream: _Stream) -> None:
        raise RuntimeError("injected record failure")


class _FailingDeviceModule(_DeviceModule):
    def Event(self) -> _Event:
        event = _FailingEvent()
        self.events.append(event)
        return event


def _arena(
    *,
    class_id: int = 0,
    pool_id: int = 5,
    backend_domain: int = 13,
    backend_base_index: int = 20,
    first_page_id: int = 101,
    widths: tuple[int, int] = (4, 8),
    device: str = "cpu",
) -> SglangStructuredArena:
    key = _Tensor(device)
    value = _Tensor(device)
    return SglangStructuredArena(
        registration=ArenaRegistration(
            ENGINE_EPOCH,
            POOL_EPOCH,
            pool_id,
            class_id,
            backend_domain,
            8,
            PAGE_TOKENS,
            sum(widths),
            backend_base_index,
            first_page_id,
        ),
        retention="full" if class_id == 0 else "sliding",
        storage="token_kv",
        storage_token_bias=PAGE_TOKENS,
        components=(
            SglangArenaComponent(0, 0, "key", key, 0, widths[0]),
            SglangArenaComponent(0, 0, "value", value, widths[0], widths[1]),
        ),
    )


def _context(transaction: int = 1, request: int = 1) -> OperationContext:
    return OperationContext(
        RequestLease(ENGINE_EPOCH, request, 1),
        StepLease(ENGINE_EPOCH, transaction, 1),
        DataPlaneOperation.APPEND,
    )


def _address(
    arena: SglangStructuredArena,
    *,
    page_index: int = 0,
    token_offset: int = 0,
    generation: int = 1,
) -> BackendTokenAddress:
    registration = arena.registration
    return BackendTokenAddress(
        PageLease(
            ENGINE_EPOCH,
            POOL_EPOCH,
            generation,
            registration.first_page_id + page_index,
            registration.pool_id,
        ),
        registration.class_id,
        registration.backend_domain,
        registration.backend_base_index + page_index,
        token_offset,
    )


def _write(
    arena: SglangStructuredArena,
    *,
    context: OperationContext | None = None,
    token_id: int = 0,
    page_index: int = 0,
    token_offset: int = 0,
    byte_count: int | None = None,
) -> ExternalTokenWrite:
    return ExternalTokenWrite(
        context if context is not None else _context(),
        token_id,
        _address(
            arena, page_index=page_index, token_offset=token_offset
        ),
        arena.token_bytes if byte_count is None else byte_count,
    )


def _locations(ticket: ExternalAppendTicket) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for destination in ticket.resolved_destinations:
        result.setdefault(destination.address.class_id, []).append(
            destination.token_index + destination.arena.storage_token_bias
        )
    return result


def _validate(
    adapter: SglangExternalWriteAdapter,
    ticket: ExternalAppendTicket,
    *,
    stream: object | None = None,
    completion_domain: int | None = None,
) -> dict[int, list[int]]:
    live = _locations(ticket)
    expected = tuple(
        (class_id, tuple(values)) for class_id, values in live.items()
    )
    adapter.validate_launch(
        ticket,
        expected,
        live,
        current_stream=stream,
        completion_domain=completion_domain,
    )
    return live


def test_cpu_external_append_resolves_normalized_structured_destination() -> None:
    arena = _arena()
    adapter = SglangExternalWriteAdapter((arena,), adapter_id="test")
    write = _write(arena, token_id=17, page_index=2, token_offset=3)

    ticket = adapter.prepare_external_append((write,), completion_domain=37)

    assert isinstance(adapter, ExternalWriteCompletionAdapter)
    assert ticket.adapter_id.startswith("test:")
    assert ticket.launch_context is None
    assert ticket.resolved_destinations == (
        ticket.resolved_destinations[0],
    )
    resolved = ticket.resolved_destinations[0]
    assert resolved.address == write.destination
    assert resolved.arena is arena
    assert resolved.arena_page_index == 2
    assert resolved.token_index == 2 * PAGE_TOKENS + 3
    assert resolved.byte_length == sum((4, 8))

    live = _validate(adapter, ticket, completion_domain=37)
    assert live == {0: [resolved.token_index + PAGE_TOKENS]}
    data = adapter.record_external_data_ready(ticket)
    last_use = adapter.record_last_use(
        (write.destination.page_address,), completion_domain=37
    )

    assert data.operation is DataPlaneOperation.APPEND
    assert data.copies == ()
    assert data.writes[0].destination == write.destination
    assert data.writes[0].byte_count == arena.token_bytes
    assert data.completion.completion_value == 1
    assert last_use.completion_value == 2
    assert last_use is not data.completion
    assert adapter.event_for(data.completion) is None
    assert adapter.event_for(last_use) is None


@pytest.mark.parametrize(
    ("first_device", "second_device"),
    (("cpu", "cuda:0"), ("cuda:0", "cuda:1")),
)
def test_external_adapter_rejects_component_device_mismatch(
    first_device: str, second_device: str
) -> None:
    arena = _arena(device=first_device)
    arena.components[1].tensor.device = second_device

    with pytest.raises(ExternalAppendError, match="span multiple devices"):
        SglangExternalWriteAdapter((arena,), device_module=_DeviceModule())


def test_external_adapter_rejects_configured_device_mismatch() -> None:
    arena = _arena()

    with pytest.raises(ExternalAppendError, match="differs from structured arena"):
        SglangExternalWriteAdapter((arena,), device="cuda:0")


def test_external_adapter_rejects_ambiguous_cuda_device() -> None:
    arena = _arena(device="cuda")

    with pytest.raises(ExternalAppendError, match="lacks an index"):
        SglangExternalWriteAdapter((arena,), device_module=_DeviceModule())


def test_multi_class_ticket_requires_canonical_full_swa_coverage() -> None:
    full = _arena()
    sliding = _arena(
        class_id=1,
        pool_id=6,
        backend_domain=14,
        backend_base_index=40,
        first_page_id=201,
        widths=(6, 10),
    )
    adapter = SglangExternalWriteAdapter((full, sliding))
    context = _context(2)
    writes = (
        _write(full, context=context, token_id=20, token_offset=1),
        _write(sliding, context=context, token_id=20, token_offset=2),
        _write(full, context=context, token_id=21, token_offset=2),
        _write(sliding, context=context, token_id=21, token_offset=3),
    )

    ticket = adapter.prepare_external_append(writes)
    assert [item.arena for item in ticket.resolved_destinations] == [
        full,
        sliding,
        full,
        sliding,
    ]
    assert [item.byte_length for item in ticket.resolved_destinations] == [
        full.token_bytes,
        sliding.token_bytes,
        full.token_bytes,
        sliding.token_bytes,
    ]
    _validate(adapter, ticket)
    adapter.record_external_data_ready(ticket)

    another = SglangExternalWriteAdapter((full, sliding))
    with pytest.raises(ExternalAppendError, match="every structured class"):
        another.prepare_external_append((writes[0],))
    with pytest.raises(ExternalAppendError, match="every structured class"):
        another.prepare_external_append((writes[1], writes[0]))


def test_mapping_constructor_canonicalizes_class_order() -> None:
    full = _arena()
    sliding = _arena(
        class_id=1,
        pool_id=6,
        backend_domain=14,
        backend_base_index=40,
        first_page_id=201,
    )
    adapter = SglangExternalWriteAdapter({1: sliding, 0: full})
    context = _context(15)

    ticket = adapter.prepare_external_append(
        (
            _write(full, context=context),
            _write(sliding, context=context),
        )
    )

    assert tuple(
        item.address.class_id for item in ticket.resolved_destinations
    ) == (0, 1)


def test_tensor_storage_drift_after_ticket_issue_poisons() -> None:
    arena = _arena()
    adapter = SglangExternalWriteAdapter((arena,))
    ticket = adapter.prepare_external_append(
        (_write(arena, context=_context(16)),)
    )
    arena.components[0].tensor._pointer += 4096

    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        adapter.validate_launch(
            ticket,
            ((0, (PAGE_TOKENS,)),),
            {0: [PAGE_TOKENS]},
            current_stream=None,
        )


def test_prepare_preflight_is_atomic_and_reserves_exact_authority() -> None:
    arena = _arena()
    adapter = SglangExternalWriteAdapter((arena,))
    context = _context(3)

    with pytest.raises(ExternalAppendError, match="aggregate arena width"):
        adapter.prepare_external_append(
            (_write(arena, context=context, byte_count=arena.token_bytes - 1),)
        )
    ticket = adapter.prepare_external_append(
        (_write(arena, context=context),), completion_domain=9
    )

    with pytest.raises(ExternalAppendError, match="transaction is reserved"):
        adapter.prepare_external_append(
            (_write(arena, context=context, page_index=1),),
            completion_domain=10,
        )
    with pytest.raises(ExternalAppendError, match="page generation is reserved"):
        adapter.prepare_external_append(
            (_write(arena, context=_context(4), token_offset=1),),
            completion_domain=10,
        )
    with pytest.raises(ExternalAppendError, match="completion domain is reserved"):
        adapter.prepare_external_append(
            (_write(arena, context=_context(5), page_index=1),),
            completion_domain=9,
        )

    _validate(adapter, ticket, completion_domain=9)
    adapter.record_external_data_ready(ticket)
    with pytest.raises(ExternalAppendError, match="already observed"):
        adapter.prepare_external_append(
            (_write(arena, context=context, page_index=1),)
        )


def test_prepare_rejects_bad_address_duplicates_and_stale_generation() -> None:
    arena = _arena()
    context = _context(6)

    adapter = SglangExternalWriteAdapter((arena,))
    bad = replace(
        _address(arena),
        backend_index=arena.registration.backend_base_index + 1,
    )
    with pytest.raises(ExternalAppendError, match="arena identity"):
        adapter.prepare_external_append(
            (ExternalTokenWrite(context, 0, bad, arena.token_bytes),)
        )

    duplicate = _write(arena, context=context)
    with pytest.raises(ExternalAppendError, match="destination"):
        adapter.prepare_external_append((duplicate, duplicate))

    first = adapter.prepare_external_append(
        (_write(arena, context=_context(7), page_index=1),)
    )
    _validate(adapter, first)
    adapter.record_external_data_ready(first)
    newer = _write(
        arena, context=_context(8), page_index=1, token_offset=1
    )
    newer = replace(
        newer,
        destination=replace(
            newer.destination,
            page=replace(newer.destination.page, generation=2),
        ),
    )
    second = adapter.prepare_external_append((newer,))
    _validate(adapter, second)
    adapter.record_external_data_ready(second)
    with pytest.raises(ExternalAppendError, match="stale generation"):
        adapter.record_last_use(
            (_address(arena, page_index=1).page_address,)
        )


def test_cuda_prepare_waits_and_ready_uses_saved_stream() -> None:
    arena = _arena(device="cuda:0")
    device_module = _DeviceModule()
    adapter = SglangExternalWriteAdapter(
        (arena,),
        device_module=device_module,
        device="cuda:0",
    )
    page = _address(arena).page_address
    predecessor = adapter.record_last_use((page,), completion_domain=19)
    predecessor_event = adapter.event_for(predecessor)
    first_stream = device_module.stream
    first_stream.actions.clear()

    ticket = adapter.prepare_external_append(
        (_write(arena, context=_context(9)),), completion_domain=19
    )

    assert ticket.launch_context is first_stream
    assert adapter.completion_domain == 1
    assert first_stream.actions == [("wait", predecessor_event)]
    _validate(adapter, ticket, stream=first_stream, completion_domain=19)
    first_stream.actions.append(("kernel", ticket.ticket_id))
    second_stream = _Stream(2)
    device_module.stream = second_stream

    data = adapter.record_external_data_ready(ticket)
    data_event = adapter.event_for(data.completion)
    assert data_event is device_module.events[-1]
    assert data_event is not predecessor_event
    assert first_stream.actions[-2:] == [
        ("kernel", ticket.ticket_id),
        ("record", data_event),
    ]
    assert second_stream.actions == []

    last_use = adapter.record_last_use((page,), completion_domain=23)
    last_use_event = adapter.event_for(last_use)
    assert last_use_event is not data_event
    assert second_stream.actions == [
        ("wait", data_event),
        ("record", last_use_event),
    ]


@pytest.mark.parametrize(
    "failure",
    (
        "locations",
        "stream",
        "domain",
    ),
)
def test_launch_mismatch_after_ticket_issue_poisons(failure: str) -> None:
    arena = _arena(device="cuda:0")
    device_module = _DeviceModule()
    adapter = SglangExternalWriteAdapter(
        (arena,), device_module=device_module, device="cuda:0"
    )
    ticket = adapter.prepare_external_append(
        (_write(arena, context=_context(10)),), completion_domain=29
    )
    live = _locations(ticket)
    expected = tuple((key, tuple(value)) for key, value in live.items())
    stream: object = ticket.launch_context
    domain = 29
    if failure == "locations":
        live[0][0] += 1
    elif failure == "stream":
        stream = _Stream(99)
    else:
        domain += 1

    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        adapter.validate_launch(
            ticket,
            expected,
            live,
            current_stream=stream,
            completion_domain=domain,
        )
    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        adapter.prepare_external_append(
            (_write(arena, context=_context(11), page_index=1),)
        )


def test_forgery_replay_and_issued_ticket_tamper_fail_closed() -> None:
    arena = _arena()
    adapter = SglangExternalWriteAdapter((arena,))
    ticket = adapter.prepare_external_append(
        (_write(arena, context=_context(12)),)
    )

    with pytest.raises(ExternalAppendError, match="forged|reconstructed"):
        adapter.record_external_data_ready(replace(ticket))
    foreign = replace(ticket, adapter_id="foreign")
    with pytest.raises(ExternalAppendError, match="another adapter"):
        adapter.record_external_data_ready(foreign)

    _validate(adapter, ticket)
    adapter.record_external_data_ready(ticket)
    with pytest.raises(ExternalAppendError, match="already consumed"):
        adapter.record_external_data_ready(ticket)

    poisoned = SglangExternalWriteAdapter((arena,))
    issued = poisoned.prepare_external_append(
        (_write(arena, context=_context(13)),)
    )
    object.__setattr__(issued.resolved_destinations[0], "token_index", 999)
    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        poisoned.validate_launch(
            issued,
            ((0, (PAGE_TOKENS,)),),
            {0: [PAGE_TOKENS]},
            current_stream=None,
        )
    object.__setattr__(issued.resolved_destinations[0], "token_index", 0)
    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        poisoned.record_external_data_ready(issued)


def test_cuda_event_record_failure_after_launch_poisons_without_retry() -> None:
    arena = _arena(device="cuda:0")
    device_module = _FailingDeviceModule()
    adapter = SglangExternalWriteAdapter(
        (arena,), device_module=device_module, device="cuda:0"
    )
    ticket = adapter.prepare_external_append(
        (_write(arena, context=_context(14)),)
    )
    _validate(adapter, ticket, stream=ticket.launch_context)

    with pytest.raises(ExternalAppendPoisonedError, match="uncertain"):
        adapter.record_external_data_ready(ticket)
    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        adapter.record_external_data_ready(ticket)


def test_event_for_rejects_reconstructed_and_tampered_fences() -> None:
    arena = _arena()
    adapter = SglangExternalWriteAdapter((arena,))
    fence = adapter.record_last_use((_address(arena).page_address,))

    with pytest.raises(ExternalAppendError, match="forged|reconstructed"):
        adapter.event_for(replace(fence))
    object.__setattr__(fence, "completion_value", 99)
    with pytest.raises(ExternalAppendPoisonedError, match="poisoned"):
        adapter.event_for(fence)


def test_many_completed_steps_keep_evidence_and_transaction_state_bounded() -> None:
    arena = _arena(device="cuda:0")
    device_module = _DeviceModule()
    adapter = SglangExternalWriteAdapter(
        (arena,), device_module=device_module, device="cuda:0"
    )
    page = _address(arena).page_address

    for generation in range(1, 65):
        context = OperationContext(
            RequestLease(ENGINE_EPOCH, 1, 1),
            StepLease(ENGINE_EPOCH, 3, generation),
            DataPlaneOperation.APPEND,
        )
        ticket = adapter.prepare_external_append(
            (_write(arena, context=context, token_id=generation),),
            completion_domain=1,
        )
        _validate(adapter, ticket, stream=ticket.launch_context)
        adapter.record_external_data_ready(ticket)
        last_use = adapter.record_last_use((page,), completion_domain=1)
        adapter.retire_completion(last_use)

        assert adapter._pending == {}
        assert adapter._fences == {}
        assert adapter._event_identities == set()
        assert len(adapter._observed_transactions) == 1

    adapter.close()
