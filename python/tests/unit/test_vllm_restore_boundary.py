"""vLLM page initialization and recurrent migration bracket Restore submission."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from orbitkv.vllm.runtime import install_restore_boundary


@pytest.fixture
def runtime(monkeypatch):
    events = []

    class GPUModelRunner:
        def update_requests(self, output):
            events.append("zero-and-copy-on-write")
            return output

    class OrbitKVConnector:
        pass

    class MultiConnector:
        def __init__(self, connectors):
            self._connectors = connectors

    class MultiKVConnectorMetadata(SimpleNamespace):
        pass

    multi_module = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector")
    multi_module.MultiConnector = MultiConnector
    multi_module.MultiKVConnectorMetadata = MultiKVConnectorMetadata
    monkeypatch.setitem(sys.modules, multi_module.__name__, multi_module)
    runner_module = ModuleType("vllm.v1.worker.gpu.model_runner")
    runner_module.GPUModelRunner = GPUModelRunner
    connector_module = ModuleType("orbitkv.vllm.connector")
    connector_module.OrbitKVConnector = OrbitKVConnector
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    monkeypatch.setitem(sys.modules, connector_module.__name__, connector_module)
    install_restore_boundary()
    installed = GPUModelRunner.update_requests
    install_restore_boundary()
    assert GPUModelRunner.update_requests is installed
    return GPUModelRunner, OrbitKVConnector, events


def test_restore_precedes_recurrent_migration_after_page_initialization(runtime):
    runner_class, connector_class, events = runtime
    connector = connector_class()
    metadata = SimpleNamespace(preempted_req_ids={"evicted"})

    def drain_saves(request_ids):
        assert request_ids == {"evicted"}
        events.append("save-drained")

    def restore(value):
        assert value is metadata
        events.append("restore-recorded-and-recurrent-waits-enqueued")

    connector._worker = SimpleNamespace(handle_preemptions=drain_saves, start_load_kv=restore)
    runner = runner_class()
    runner.kv_connector = SimpleNamespace(kv_connector=connector, _disabled=False)
    output = SimpleNamespace(
        kv_connector_metadata=metadata,
        has_sync_kv_loads=True,
        total_num_scheduled_tokens=1,
    )
    assert runner.update_requests(output) is output
    events.append("migrate-recurrent-state")
    assert events == [
        "save-drained",
        "zero-and-copy-on-write",
        "restore-recorded-and-recurrent-waits-enqueued",
        "migrate-recurrent-state",
    ]


@pytest.mark.parametrize("disabled,ours", [(True, True), (False, False)])
def test_other_connectors_and_profiling_keep_the_engine_boundary(runtime, disabled, ours):
    runner_class, connector_class, events = runtime
    runner = runner_class()
    runner.kv_connector = SimpleNamespace(
        kv_connector=connector_class() if ours else object(), _disabled=disabled
    )
    output = object()
    assert runner.update_requests(output) is output
    assert events == ["zero-and-copy-on-write"]


@pytest.mark.parametrize("sync,tokens", [(False, 1), (True, 0)])
def test_no_forward_or_no_restore_keeps_native_connector_lifecycle(runtime, sync, tokens):
    runner_class, connector_class, events = runtime
    connector = connector_class()
    connector._worker = SimpleNamespace(handle_preemptions=lambda _: events.append("save-drained"))
    runner = runner_class()
    runner.kv_connector = SimpleNamespace(kv_connector=connector, _disabled=False)
    output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(preempted_req_ids=set()),
        has_sync_kv_loads=sync,
        total_num_scheduled_tokens=tokens,
    )
    runner.update_requests(output)
    assert events == ["save-drained", "zero-and-copy-on-write"]


def test_composed_cache_uses_its_own_metadata_before_recurrent_migration(runtime):
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
        MultiKVConnectorMetadata,
    )

    runner_class, connector_class, events = runtime
    connector = connector_class()
    metadata = SimpleNamespace(preempted_req_ids={"cache-owner"})
    connector._worker = SimpleNamespace(
        handle_preemptions=lambda ids: events.append(("save-drained", ids)),
        start_load_kv=lambda value: events.append(("restore", value)),
    )
    runner = runner_class()
    runner.kv_connector = SimpleNamespace(
        kv_connector=MultiConnector([object(), MultiConnector([connector])]),
        _disabled=False,
    )
    output = SimpleNamespace(
        kv_connector_metadata=MultiKVConnectorMetadata(
            metadata=[object(), MultiKVConnectorMetadata(metadata=[metadata])]
        ),
        has_sync_kv_loads=True,
        total_num_scheduled_tokens=1,
    )
    runner.update_requests(output)
    assert events == [
        ("save-drained", {"cache-owner"}),
        "zero-and-copy-on-write",
        ("restore", metadata),
    ]
    output.kv_connector_metadata.metadata.clear()
    events.clear()
    with pytest.raises(ValueError, match="zip"):
        runner.update_requests(output)
    assert not events
