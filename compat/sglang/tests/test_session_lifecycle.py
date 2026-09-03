from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from orbitkv_sglang.ffi.session_types import (
    EngineBatchId,
    EngineBatchTicket,
)
from orbitkv_sglang.bridge import execution_context, session_lifecycle, state
from orbitkv_sglang.bridge.execution_context import ForwardExecutionContext
from orbitkv_sglang.runtime import FailStopped


class _Fatal(BaseException):
    pass


class _Stream:
    def __init__(self, handle: int, device: str) -> None:
        self.cuda_stream = handle
        self.device = device


class _Event:
    def __init__(
        self, trace: list[Any], *, record_error: BaseException | None = None
    ) -> None:
        self._trace = trace
        self._record_error = record_error

    def record(self, *, stream: Any) -> None:
        self._trace.append(("record", stream))
        if self._record_error is not None:
            raise self._record_error

    def query(self) -> bool:
        return False

    def synchronize(self) -> None:
        raise AssertionError("forward registration must not synchronize the event")


class _DeviceModule:
    def __init__(
        self,
        trace: list[Any],
        *,
        current_error: BaseException | None = None,
        event_error: BaseException | None = None,
        record_error: BaseException | None = None,
    ) -> None:
        self._trace = trace
        self.current_error = current_error
        self._event_error = event_error
        self._record_error = record_error
        self.stream = object()

    def current_stream(self, device: Any) -> object:
        self._trace.append(("current_stream", device))
        if self.current_error is not None:
            raise self.current_error
        return self.stream

    def Event(self) -> _Event:
        self._trace.append("event")
        if self._event_error is not None:
            raise self._event_error
        return _Event(self._trace, record_error=self._record_error)


class _Runtime:
    def __init__(
        self,
        trace: list[Any],
        *,
        register_error: BaseException | None = None,
        register_fail_stopped: bool = False,
    ) -> None:
        self.trace = trace
        self.failure_reason: str | None = None
        self.register_error = register_error
        self.register_fail_stopped = register_fail_stopped

    def poll(self) -> tuple[()]:
        self.trace.append("poll")
        return ()

    def register_event(
        self, ticket: EngineBatchTicket, event: _Event, domain: int
    ) -> None:
        self.trace.append(("register_event", ticket, event, domain))
        if self.register_error is not None:
            if self.register_fail_stopped:
                self.failure_reason = "native event registration failed"
            raise self.register_error

    def quarantine_submitted(self, ticket: EngineBatchTicket) -> None:
        self.trace.append(("quarantine_submitted", ticket))
        self.failure_reason = "submitted execution was quarantined"
        raise FailStopped(self.failure_reason)

    def fail_stop(self, reason: str) -> None:
        self.trace.append(("fail_stop", reason))
        self.failure_reason = reason


def _ticket() -> EngineBatchTicket:
    return EngineBatchTicket(EngineBatchId(7, 3), (11,))


def _batch(ticket: Any = None, *, include_ticket: bool = True) -> SimpleNamespace:
    attributes: dict[str, Any] = {
        "reqs": [SimpleNamespace(rid="request")]
    }
    if include_ticket:
        attributes["_orbitkv_session_ticket"] = ticket
    return SimpleNamespace(**attributes)


class _StickyBatch(SimpleNamespace):
    def __init__(self, sticky_attribute: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sticky_attribute = sticky_attribute

    def __delattr__(self, name: str) -> None:
        if name == self._sticky_attribute:
            return
        super().__delattr__(name)


def _attach_context(
    batch: Any,
    device_module: _DeviceModule,
    ticket: EngineBatchTicket | None,
    *,
    device: Any = "cuda:0",
) -> ForwardExecutionContext:
    context = ForwardExecutionContext(
        device_module=device_module,
        device=device,
        stream=device_module.stream,
    )
    context._batch = batch
    if ticket is not None:
        context.bind_ticket(ticket)
    batch._orbitkv_forward_execution_context = context
    return context


def _contextual_batch(
    ticket: EngineBatchTicket,
    device_module: _DeviceModule,
    *,
    device: Any = "cuda:0",
) -> tuple[SimpleNamespace, ForwardExecutionContext]:
    batch = _batch(ticket)
    context = _attach_context(batch, device_module, ticket, device=device)
    return batch, context


def _install(
    monkeypatch: pytest.MonkeyPatch, runtime: _Runtime, trace: list[Any]
) -> None:
    state._install_test_state(runtime=runtime)
    monkeypatch.setattr(state, "_uses_runtime_session", lambda: True)
    monkeypatch.setattr(
        "orbitkv_sglang.bridge.validation._validate_batch",
        lambda _batch: trace.append("validate"),
    )


@pytest.fixture(autouse=True)
def _reset_state():
    state._install_test_state()
    yield
    state._install_test_state()


def test_session_forward_registers_recorded_event_without_completing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch, context = _contextual_batch(
        ticket, device_module, device="cuda:3"
    )
    scheduler = SimpleNamespace(
        device="cuda:3",
        ps=SimpleNamespace(gpu_id=8),
        device_module=device_module,
    )

    def forward(received_scheduler: Any, received_batch: Any) -> str:
        assert received_scheduler is scheduler
        assert received_batch is batch
        assert batch._orbitkv_session_ticket is ticket
        trace.append("forward")
        return "result"

    assert (
        session_lifecycle.run_scheduled_batch(forward, scheduler, batch)
        == "result"
    )

    assert trace[:3] == ["validate", "poll", ("current_stream", "cuda:3")]
    assert trace[3:7] == [
        "forward",
        ("current_stream", "cuda:3"),
        "event",
        ("record", context.stream),
    ]
    registration = trace[7]
    assert registration[0] == "register_event"
    assert registration[1] is ticket
    assert registration[3] == 4
    assert not hasattr(batch, "_orbitkv_session_ticket")
    assert not hasattr(batch, "_orbitkv_forward_execution_context")
    assert context._batch is None
    assert not any(item == "complete" for item in trace)


def test_session_event_records_on_captured_stream_for_equivalent_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    captured = _Stream(17, "cuda:2")
    equivalent = _Stream(17, "cuda:2")
    device_module.stream = equivalent
    batch = _batch(ticket)
    context = _attach_context(
        batch, device_module, ticket, device="cuda:2"
    )
    context.stream = captured
    scheduler = SimpleNamespace(device="cuda:2", device_module=device_module)

    assert session_lifecycle.run_scheduled_batch(
        lambda *_args: "result", scheduler, batch
    ) == "result"

    assert captured is not equivalent
    assert ("record", captured) in trace
    assert ("record", equivalent) not in trace


def test_session_validation_failure_quarantines_submitted_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    monkeypatch.setattr(
        "orbitkv_sglang.bridge.validation._validate_batch",
        lambda _batch: (_ for _ in ()).throw(RuntimeError("invalid batch")),
    )
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch, _context = _contextual_batch(ticket, device_module)
    scheduler = SimpleNamespace(
        device="cuda:0", device_module=device_module
    )

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: pytest.fail("invalid batch must not reach forward"),
            scheduler,
            batch,
        )

    assert trace == [("quarantine_submitted", ticket)]
    assert batch._orbitkv_session_ticket is ticket


@pytest.mark.parametrize(
    "context_state",
    (
        "missing",
        "invalid",
        "foreign",
        "unbound",
        "swapped_ticket",
        "missing_raw_ticket",
    ),
)
def test_session_requires_valid_context_and_exact_ticket_identity(
    monkeypatch: pytest.MonkeyPatch, context_state: str
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    raw_ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch = _batch(
        raw_ticket, include_ticket=context_state != "missing_raw_ticket"
    )
    expected_quarantine = raw_ticket
    if context_state == "invalid":
        batch._orbitkv_forward_execution_context = object()
    elif context_state == "foreign":
        foreign_ticket = EngineBatchTicket(EngineBatchId(7, 4), (12,))
        foreign_batch = _batch(foreign_ticket)
        foreign_context = _attach_context(
            foreign_batch, device_module, foreign_ticket
        )
        batch._orbitkv_forward_execution_context = foreign_context
    elif context_state == "unbound":
        _attach_context(batch, device_module, None)
    elif context_state == "swapped_ticket":
        context_ticket = _ticket()
        assert context_ticket == raw_ticket and context_ticket is not raw_ticket
        _attach_context(batch, device_module, context_ticket)
        expected_quarantine = context_ticket
    elif context_state == "missing_raw_ticket":
        _attach_context(batch, device_module, raw_ticket)

    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)
    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: pytest.fail("invalid identity must not run forward"),
            scheduler,
            batch,
        )

    assert len(trace) == 1
    assert trace[0][0] == "quarantine_submitted"
    assert trace[0][1] is expected_quarantine


@pytest.mark.parametrize(
    ("include_ticket", "raw_ticket"),
    ((False, None), (True, object())),
)
def test_session_missing_or_invalid_ticket_without_canonical_context_ticket(
    monkeypatch: pytest.MonkeyPatch, include_ticket: bool, raw_ticket: Any
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    device_module = _DeviceModule(trace)
    batch = _batch(raw_ticket, include_ticket=include_ticket)
    _attach_context(batch, device_module, None)
    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)

    with pytest.raises(
        FailStopped, match="forward execution context has no batch ticket"
    ):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: pytest.fail("missing ticket must not run forward"),
            scheduler,
            batch,
        )

    assert trace and trace[0][0] == "fail_stop"
    assert not any(
        isinstance(item, tuple) and item[0] == "quarantine_submitted"
        for item in trace
    )


def test_session_model_failure_quarantines_submitted_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch, _context = _contextual_batch(ticket, device_module)
    scheduler = SimpleNamespace(
        device="cuda:0", device_module=device_module
    )

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("kernel failed")),
            scheduler,
            batch,
        )

    assert trace == [
        "validate",
        "poll",
        ("current_stream", "cuda:0"),
        ("quarantine_submitted", ticket),
    ]
    assert batch._orbitkv_session_ticket is ticket


def test_session_pre_forward_stream_drift_quarantines_before_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch, _context = _contextual_batch(ticket, device_module)
    device_module.stream = object()
    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: pytest.fail("stream drift must not run forward"),
            scheduler,
            batch,
        )

    assert trace == [
        "validate",
        "poll",
        ("current_stream", "cuda:0"),
        ("quarantine_submitted", ticket),
    ]


def test_session_post_forward_stream_drift_quarantines_before_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch, _context = _contextual_batch(ticket, device_module)
    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)

    def forward(*_args: Any) -> str:
        trace.append("forward")
        device_module.stream = object()
        return "result"

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(forward, scheduler, batch)

    assert trace == [
        "validate",
        "poll",
        ("current_stream", "cuda:0"),
        "forward",
        ("current_stream", "cuda:0"),
        ("quarantine_submitted", ticket),
    ]
    assert "event" not in trace


@pytest.mark.parametrize(
    "failure_stage",
    ("event", "record", "register"),
)
def test_session_completion_uncertainty_quarantines_submitted_ticket(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    trace: list[Any] = []
    error = RuntimeError(f"{failure_stage} failed")
    runtime = _Runtime(
        trace, register_error=error if failure_stage == "register" else None
    )
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(
        trace,
        event_error=error if failure_stage == "event" else None,
        record_error=error if failure_stage == "record" else None,
    )
    batch, _context = _contextual_batch(ticket, device_module)
    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: trace.append("forward") or "result",
            scheduler,
            batch,
        )

    assert trace[-1] == ("quarantine_submitted", ticket)
    assert batch._orbitkv_session_ticket is ticket


@pytest.mark.parametrize(
    "sticky_attribute",
    (
        "_orbitkv_session_ticket",
        "_orbitkv_forward_execution_context",
    ),
)
def test_session_cleanup_failure_quarantines_registered_batch(
    monkeypatch: pytest.MonkeyPatch, sticky_attribute: str
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch = _StickyBatch(
        sticky_attribute,
        reqs=[SimpleNamespace(rid="request")],
        _orbitkv_session_ticket=ticket,
    )
    context = _attach_context(batch, device_module, ticket)
    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: "result", scheduler, batch
        )

    registration_index = next(
        index
        for index, item in enumerate(trace)
        if isinstance(item, tuple) and item[0] == "register_event"
    )
    quarantine_index = trace.index(("quarantine_submitted", ticket))
    assert registration_index < quarantine_index
    assert getattr(batch, sticky_attribute) is (
        ticket
        if sticky_attribute == "_orbitkv_session_ticket"
        else context
    )


@pytest.mark.parametrize(
    "failure_stage",
    ("pre", "model", "post", "event", "cleanup"),
)
def test_session_contains_base_exception_from_every_forward_phase(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    trace: list[Any] = []
    fatal = _Fatal(f"{failure_stage} fatal")
    runtime = _Runtime(trace)
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(
        trace, event_error=fatal if failure_stage == "event" else None
    )
    batch, _context = _contextual_batch(ticket, device_module)
    scheduler = SimpleNamespace(device="cuda:0", device_module=device_module)
    if failure_stage == "pre":
        monkeypatch.setattr(
            "orbitkv_sglang.bridge.validation._validate_batch",
            lambda _batch: (_ for _ in ()).throw(fatal),
        )
    if failure_stage == "cleanup":
        monkeypatch.setattr(
            execution_context,
            "clear_forward_context",
            lambda _batch, _context: (_ for _ in ()).throw(fatal),
        )

    def forward(*_args: Any) -> str:
        if failure_stage == "model":
            raise fatal
        if failure_stage == "post":
            device_module.current_error = fatal
        return "result"

    with pytest.raises(FailStopped, match="submitted execution was quarantined"):
        session_lifecycle.run_scheduled_batch(forward, scheduler, batch)

    assert trace[-1] == ("quarantine_submitted", ticket)


def test_session_register_fail_stop_is_not_overwritten_by_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    runtime = _Runtime(
        trace,
        register_error=FailStopped("native event registration failed"),
        register_fail_stopped=True,
    )
    _install(monkeypatch, runtime, trace)
    ticket = _ticket()
    device_module = _DeviceModule(trace)
    batch, _context = _contextual_batch(ticket, device_module)
    scheduler = SimpleNamespace(
        device="cuda:0", device_module=device_module
    )

    with pytest.raises(FailStopped, match="native event registration failed"):
        session_lifecycle.run_scheduled_batch(
            lambda *_args: "result", scheduler, batch
        )

    assert not any(
        isinstance(item, tuple) and item[0] == "quarantine_submitted"
        for item in trace
    )
    assert runtime.failure_reason == "native event registration failed"
    assert batch._orbitkv_session_ticket is ticket
