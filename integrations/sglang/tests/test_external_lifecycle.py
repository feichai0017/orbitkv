from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbitkv_runtime import CompletionFence
from orbitkv_sglang.config import ClassConfig, ManagerPlanConfig
from orbitkv_sglang.plugin import external_lifecycle, state
from orbitkv_sglang.runtime import BatchRecord


def _class(class_id: int, retention: str) -> ClassConfig:
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 11,
        name=retention,
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=32,
        window_tokens=None if retention == "full" else 32,
        period_blocks=None if retention == "full" else 3,
        components=(("key", 16), ("value", 16)),
    )


def _config(*retentions: str) -> ManagerPlanConfig:
    return ManagerPlanConfig(
        plan_path=SimpleNamespace(),
        library_path=SimpleNamespace(),
        plan_json=b"{}",
        plan_fingerprint="sha256:test",
        page_tokens=16,
        classes=tuple(
            _class(index, retention)
            for index, retention in enumerate(retentions)
        ),
    )


class _Runtime:
    engine_epoch = 7

    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.failure_reason = None

    def poll(self) -> None:
        self.events.append("poll")

    def mark_forward(self, batch: BatchRecord) -> None:
        self.events.append(("mark_forward", batch))

    def register_external_event(
        self, batch: BatchRecord, fence: CompletionFence, event: object,
        *, adapter
    ) -> None:
        assert adapter is state._DATA_PLANE
        self.events.append(("register_external", batch, fence, event))

    def register_event(
        self, batch: BatchRecord, event: object, domain: int
    ) -> None:
        self.events.append(("register_legacy", batch, event, domain))

    def forward_failed(self, _batch: BatchRecord, error: BaseException) -> None:
        self.failure_reason = f"forward: {error}"
        self.events.append(("forward_failed", str(error)))

    def event_registration_failed(
        self, _batch: BatchRecord, error: BaseException
    ) -> None:
        self.failure_reason = f"event: {error}"
        self.events.append(("event_failed", str(error)))


class _DataPlane:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.adapter_id = "adapter"
        self.poison_reason = None
        self.ticket = object()
        self.data_fence = CompletionFence("adapter", 7, 4, 1, 1)
        self.last_use = CompletionFence("adapter", 7, 4, 2, 2)
        self.event = object()
        self.completion_domain = 4

    def prepare_external_append(self, writes, *, completion_domain=1):
        self.events.append(("prepare_external", tuple(writes), completion_domain))
        return self.ticket

    def validate_launch(
        self,
        ticket,
        expected,
        live,
        *,
        current_stream,
        completion_domain=None,
    ) -> None:
        self.events.append(
            (
                "validate_launch",
                ticket,
                expected,
                live,
                current_stream,
                completion_domain,
            )
        )

    def record_external_data_ready(self, ticket):
        self.events.append(("data_ready", ticket))
        return SimpleNamespace(completion=self.data_fence)

    def record_last_use(self, pages, *, completion_domain=1):
        self.events.append(("last_use", tuple(pages), completion_domain))
        return self.last_use

    def event_for(self, fence):
        self.events.append(("event_for", fence))
        return self.event

    def poison(self, reason: str) -> None:
        self.poison_reason = self.poison_reason or reason
        self.events.append(("poison", reason))


class _FixedState:
    def __init__(self, records, events):
        self.records = tuple(records)
        self.events = events

    def records_for_schedule_batch(self, _batch):
        return self.records

    def register_event(
        self,
        keys,
        records,
        event,
        domain,
        _device_module,
        _device,
        completion_value=None,
    ):
        self.events.append(
            ("fixed_register", tuple(keys), tuple(records), event, domain, completion_value)
        )

    def register_external_event(
        self, keys, records, event, fence, _device_module, _device,
        *, adapter
    ):
        assert adapter is state._DATA_PLANE
        self.events.append(
            (
                "fixed_register",
                tuple(keys),
                tuple(records),
                event,
                fence.completion_domain,
                fence.completion_value,
            )
        )

    def pre_forward_failed(self, error):
        self.events.append(("fixed_pre_failed", str(error)))

    def forward_failed(self, records, error):
        self.events.append(("fixed_forward_failed", tuple(records), str(error)))

    def event_registration_failed(self, records, error):
        self.events.append(("fixed_event_failed", tuple(records), str(error)))


class _DeviceModule:
    def __init__(self, stream: object) -> None:
        self.stream = stream

    def current_stream(self, _device):
        return self.stream


class _LocationMap(dict):
    def __getitem__(self, key):
        if isinstance(key, list):
            return [dict.__getitem__(self, item) for item in key]
        return dict.__getitem__(self, key)


def _batch(config: ManagerPlanConfig, record: object) -> SimpleNamespace:
    req = SimpleNamespace(rid="request")
    batch_record = BatchRecord((("str", "request"),), (record,))
    return SimpleNamespace(
        reqs=[req],
        _orbitkv_batch=batch_record,
        _orbitkv_external_ticket=None,
        _orbitkv_external_manifest=None,
        out_cache_loc=[20, 21],
        device="cuda:3",
        tree_cache=SimpleNamespace(),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        is_dllm=lambda: False,
        enable_overlap=False,
        _config=config,
    )


def _install(
    monkeypatch, config: ManagerPlanConfig, runtime: _Runtime, data_plane: _DataPlane
) -> None:
    state._install_test_state(config=config, runtime=runtime)
    state._DATA_PLANE = data_plane
    state._STRUCTURED_ARENAS = (object(),) * len(config.classes)
    monkeypatch.setattr(
        "orbitkv_sglang.plugin.validation._validate_batch", lambda _batch: None
    )


def test_structured_data_plane_is_explicit_opt_in(monkeypatch) -> None:
    state._install_test_state(config=_config("full"), runtime=_Runtime([]))
    monkeypatch.delenv("ORBITKV_STRUCTURED_DATA_PLANE", raising=False)
    assert not state._uses_structured_data_plane()
    monkeypatch.setenv("ORBITKV_STRUCTURED_DATA_PLANE", "1")
    assert state._uses_structured_data_plane()
    monkeypatch.setenv("ORBITKV_STRUCTURED_DATA_PLANE", "invalid")
    with pytest.raises(RuntimeError, match="0/1"):
        state._uses_structured_data_plane()
    pure_sliding = _config("sliding")
    state._install_test_state(config=pure_sliding, runtime=_Runtime([]))
    monkeypatch.setenv("ORBITKV_STRUCTURED_DATA_PLANE", "1")
    assert state._uses_structured_data_plane()


def test_completion_domain_tracks_nonzero_cuda_device() -> None:
    assert external_lifecycle.completion_domain_for_device("cuda:3") == 4
    assert (
        external_lifecycle.completion_domain_for_device(
            SimpleNamespace(index=7)
        )
        == 8
    )
    assert external_lifecycle.completion_domain_for_device("cuda", 5) == 6


def test_run_batch_uses_live_primary_and_full_to_swa_locations(monkeypatch) -> None:
    events: list[object] = []
    config = _config("full", "sliding")
    runtime = _Runtime(events)
    data_plane = _DataPlane(events)
    _install(monkeypatch, config, runtime, data_plane)
    state._ALLOCATOR = SimpleNamespace(
        full_to_swa_index_mapping=_LocationMap({20: 40, 21: 41})
    )
    record = SimpleNamespace(key=("str", "request"))
    batch = _batch(config, record)
    manifest = SimpleNamespace(
        expected_locations_by_class=((0, (20, 21)), (1, (40, 41))),
        last_use_pages=(object(),),
    )
    batch._orbitkv_external_ticket = data_plane.ticket
    batch._orbitkv_external_manifest = manifest
    stream = object()
    scheduler = SimpleNamespace(
        device="cuda:3", device_module=_DeviceModule(stream)
    )

    result = external_lifecycle.run_scheduled_batch(
        lambda _scheduler, _batch: events.append("forward") or "ok",
        scheduler,
        batch,
    )

    assert result == "ok"
    launch = next(item for item in events if isinstance(item, tuple) and item[0] == "validate_launch")
    assert launch[3] == {0: [20, 21], 1: [40, 41]}
    assert launch[4:] == (stream, 4)
    assert events.index("forward") < next(
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "data_ready"
    )
    registered = next(
        item for item in events if isinstance(item, tuple) and item[0] == "register_external"
    )
    assert registered[2] is data_plane.last_use
    assert registered[3] is data_plane.event
    assert batch._orbitkv_batch is None
    assert not hasattr(batch, "_orbitkv_external_ticket")


def test_live_lut_drift_fails_before_model_forward_and_poisons(monkeypatch) -> None:
    events: list[object] = []
    config = _config("full", "sliding")
    runtime = _Runtime(events)
    data_plane = _DataPlane(events)
    _install(monkeypatch, config, runtime, data_plane)
    state._ALLOCATOR = SimpleNamespace(
        full_to_swa_index_mapping=_LocationMap({20: 99, 21: 98})
    )
    batch = _batch(config, SimpleNamespace(key=("str", "request")))
    batch._orbitkv_external_ticket = data_plane.ticket
    batch._orbitkv_external_manifest = SimpleNamespace(
        expected_locations_by_class=((0, (20, 21)), (1, (40, 41))),
        last_use_pages=(object(),),
    )
    scheduler = SimpleNamespace(
        device="cuda:3", device_module=_DeviceModule(object())
    )

    def reject(_ticket, _expected, live, **_kwargs):
        if live[1] != [40, 41]:
            raise RuntimeError("live LUT drift")

    data_plane.validate_launch = reject
    with pytest.raises(Exception, match="live LUT drift|pre-forward"):
        external_lifecycle.run_scheduled_batch(
            lambda *_args: pytest.fail("model forward must not run"),
            scheduler,
            batch,
        )
    assert data_plane.poison_reason is not None
    assert any(
        isinstance(item, tuple) and item[0] == "forward_failed"
        for item in events
    )


def test_post_ticket_model_failure_poison_and_quarantines(monkeypatch) -> None:
    events: list[object] = []
    config = _config("full")
    runtime = _Runtime(events)
    data_plane = _DataPlane(events)
    _install(monkeypatch, config, runtime, data_plane)
    state._ALLOCATOR = SimpleNamespace()
    batch = _batch(config, SimpleNamespace(key=("str", "request")))
    batch._orbitkv_external_ticket = data_plane.ticket
    batch._orbitkv_external_manifest = SimpleNamespace(
        expected_locations_by_class=((0, (20, 21)),),
        last_use_pages=(object(),),
    )
    scheduler = SimpleNamespace(
        device="cuda:3", device_module=_DeviceModule(object())
    )

    with pytest.raises(Exception, match="kernel failure|forward failed"):
        external_lifecycle.run_scheduled_batch(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("kernel failure")),
            scheduler,
            batch,
        )
    assert "kernel failure" in data_plane.poison_reason
    assert runtime.failure_reason == "forward: kernel failure"


def test_token_and_fixed_state_share_exact_event_domain_and_value(monkeypatch) -> None:
    events: list[object] = []
    config = _config("full")
    runtime = _Runtime(events)
    data_plane = _DataPlane(events)
    _install(monkeypatch, config, runtime, data_plane)
    state._ALLOCATOR = SimpleNamespace()
    fixed_record = object()
    state._FIXED_STATE = _FixedState((fixed_record,), events)
    batch = _batch(config, SimpleNamespace(key=("str", "request")))
    batch._orbitkv_state_records = (fixed_record,)
    batch._orbitkv_external_ticket = data_plane.ticket
    batch._orbitkv_external_manifest = SimpleNamespace(
        expected_locations_by_class=((0, (20, 21)),),
        last_use_pages=(object(),),
    )
    scheduler = SimpleNamespace(
        device="cuda:3", device_module=_DeviceModule(object())
    )

    external_lifecycle.run_scheduled_batch(
        lambda *_args: "ok", scheduler, batch
    )

    token = next(
        item for item in events if isinstance(item, tuple) and item[0] == "register_external"
    )
    fixed = next(
        item for item in events if isinstance(item, tuple) and item[0] == "fixed_register"
    )
    assert token[2].completion_domain == fixed[4] == 4
    assert token[2].completion_value == fixed[5] == 2
    assert token[3] is fixed[3] is data_plane.event
