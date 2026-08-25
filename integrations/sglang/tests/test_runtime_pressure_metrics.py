from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orbitkv_sglang.runtime import (
    ArenaPressureSample,
    ManagerError,
    PressureClassDescriptor,
    PressureTelemetry,
    pressure_enabled_from_environment,
    retention_amplification_milli,
    semantic_live_tokens_for_request,
)
from orbitkv_sglang.runtime.census import CensusRuntimeMixin
from runtime_test_support import ReadyEvent, _runtime, ffi_library


__all__ = ["ffi_library"]


def _arena(
    class_id: int,
    *,
    free: int,
    reserved: int = 0,
    writing: int = 0,
    active: int = 0,
    retiring: int = 0,
    quarantined: int = 0,
    exhausted: int = 0,
) -> ArenaPressureSample:
    return ArenaPressureSample(
        class_id,
        free + reserved + writing + active + retiring + quarantined + exhausted,
        free,
        reserved,
        writing,
        active,
        retiring,
        quarantined,
        exhausted,
    )


def test_pressure_is_default_off_and_environment_is_strict() -> None:
    assert not pressure_enabled_from_environment({})
    assert pressure_enabled_from_environment({"ORBITKV_PRESSURE_TELEMETRY": "yes"})
    assert not pressure_enabled_from_environment(
        {"ORBITKV_PRESSURE_TELEMETRY": "OFF"}
    )
    with pytest.raises(ValueError, match="ORBITKV_PRESSURE_TELEMETRY"):
        pressure_enabled_from_environment({"ORBITKV_PRESSURE_TELEMETRY": "maybe"})


def test_pressure_phases_separate_consumed_resident_and_reachable() -> None:
    sample = _arena(0, free=2, reserved=1, writing=2, active=3, retiring=1, exhausted=1)
    assert sample.page_count == 10
    assert sample.consumed_pages == 8
    assert sample.resident_data_pages == 6

    collector = PressureTelemetry((PressureClassDescriptor(0, "full", 16, 10, 8),))
    collector.sample(
        "step_prepared",
        active_requests=2,
        arenas=(sample,),
        request_reachable_unique_pages={0: 3},
        semantic_live_tokens={0: 40},
    )
    report = collector.report()
    current = report["classes"][0]
    assert current["consumed_pages"] == 8
    assert current["resident_data_pages"] == 6
    assert current["request_reachable_unique_pages"] == 3
    assert current["resident_data_bytes"] == 6 * 16 * 8
    assert current["semantic_live_bytes"] == 40 * 8
    assert current["retention_amplification_milli"] == 2400
    assert current["min_free_bytes"] == 2 * 16 * 8
    assert current["high_water_consumed_bytes"] == 8 * 16 * 8
    assert current["high_water_resident_data_bytes"] == 6 * 16 * 8
    assert collector.report()["scope"]["fixed_state_bytes"] == "excluded"


def test_high_water_and_minimum_free_accumulate_per_class_and_global() -> None:
    collector = PressureTelemetry(
        (
            PressureClassDescriptor(0, "full", 16, 8, 4),
            PressureClassDescriptor(1, "swa", 16, 6, 8),
        )
    )
    collector.sample(
        "runtime_initialized",
        active_requests=0,
        arenas=(_arena(0, free=8), _arena(1, free=6)),
        request_reachable_unique_pages={0: 0, 1: 0},
        semantic_live_tokens={0: 0, 1: 0},
    )
    collector.sample(
        "step_completed",
        active_requests=3,
        arenas=(
            _arena(0, free=3, active=5),
            _arena(1, free=2, active=3, retiring=1),
        ),
        request_reachable_unique_pages={0: 5, 1: 3},
        semantic_live_tokens={0: 65, 1: 35},
    )
    collector.sample(
        "request_released",
        active_requests=1,
        arenas=(
            _arena(0, free=6, active=2),
            _arena(1, free=4, active=1, quarantined=1),
        ),
        request_reachable_unique_pages={0: 2, 1: 1},
        semantic_live_tokens={0: 17, 1: 9},
    )

    report = collector.report()
    assert report["sample_count"] == 3
    assert report["max_active_requests"] == 3
    assert report["event_counts"] == {
        "request_released": 1,
        "runtime_initialized": 1,
        "step_completed": 1,
    }
    assert report["global"]["min_free_pages"] == 5
    assert report["global"]["high_water_consumed_pages"] == 9
    assert report["global"]["high_water_resident_data_pages"] == 9
    assert report["global"]["high_water_request_reachable_unique_pages"] == 8
    assert report["classes"][1]["min_free_pages"] == 2


def test_semantic_live_and_ra_are_request_private_and_zero_is_undefined() -> None:
    assert semantic_live_tokens_for_request(
        boundary=48, retention="full", window_tokens=None
    ) == 48
    assert semantic_live_tokens_for_request(
        boundary=48, retention="sliding", window_tokens=18
    ) == 17
    assert semantic_live_tokens_for_request(
        boundary=48,
        retention="full",
        window_tokens=None,
        active_kv_length=24,
    ) == 24
    assert retention_amplification_milli(1536, 1024) == 1500
    assert retention_amplification_milli(0, 0) is None
    with pytest.raises(ValueError, match="exceeds"):
        semantic_live_tokens_for_request(
            boundary=8,
            retention="full",
            window_tokens=None,
            active_kv_length=9,
        )


def test_request_private_fail_closed_rejects_prefix_and_fork_only_when_enabled() -> None:
    disabled = SimpleNamespace(_pressure=None)
    CensusRuntimeMixin._pressure_require_request_private(disabled, "request fork")

    enabled = SimpleNamespace(_pressure=object())
    with pytest.raises(ManagerError, match="does not support request fork"):
        CensusRuntimeMixin._pressure_require_request_private(
            enabled, "request fork"
        )
    with pytest.raises(ManagerError, match="does not support Prefix attach"):
        CensusRuntimeMixin._pressure_require_request_private(
            enabled, "Prefix attach"
        )


def test_opt_in_runtime_rejects_shared_prefix_and_fork_before_native_calls(
    tmp_path, ffi_library, monkeypatch
) -> None:
    monkeypatch.setenv("ORBITKV_PRESSURE_TELEMETRY", "true")
    _config, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    manager.prefix_lookup_batch = Mock(
        side_effect=AssertionError("native Prefix lookup was reached")
    )
    manager.request_fork_batch = Mock(
        side_effect=AssertionError("native fork was reached")
    )
    with pytest.raises(ManagerError, match="does not support Prefix lookup"):
        runtime.prefix_lookup_batch(())
    with pytest.raises(ManagerError, match="does not support request fork"):
        runtime.request_fork_batch(())
    manager.prefix_lookup_batch.assert_not_called()
    manager.request_fork_batch.assert_not_called()
    runtime.close()


def test_disabled_checkpoint_does_not_call_census() -> None:
    class Runtime(CensusRuntimeMixin):
        _pressure = None

        def census(self):
            raise AssertionError("disabled telemetry called the native census")

    Runtime().pressure_checkpoint("event")


def test_default_off_runtime_does_not_add_census_calls(
    tmp_path, ffi_library, monkeypatch
) -> None:
    monkeypatch.delenv("ORBITKV_PRESSURE_TELEMETRY", raising=False)
    _config, manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    assert not runtime.pressure_telemetry_enabled
    original_stats = manager.stats
    original_arena_stats = manager.arena_stats
    manager.stats = Mock(side_effect=original_stats)
    manager.arena_stats = Mock(side_effect=original_arena_stats)

    batch, _plans = runtime.prepare_batch((("request", 18),))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    runtime.poll()
    runtime.release_batch(("request",))

    manager.stats.assert_not_called()
    manager.arena_stats.assert_not_called()
    runtime.close()


def test_collector_rejects_reachable_pages_outside_protocol_residency() -> None:
    collector = PressureTelemetry((PressureClassDescriptor(0, "full", 16, 4, 8),))
    with pytest.raises(ValueError, match="request-reachable"):
        collector.sample(
            "invalid",
            active_requests=2,
            arenas=(_arena(0, free=2, active=2),),
            request_reachable_unique_pages={0: 3},
            semantic_live_tokens={0: 17},
        )


def test_opt_in_real_runtime_samples_successful_lifecycle_events(
    tmp_path, ffi_library, monkeypatch
) -> None:
    monkeypatch.setenv("ORBITKV_PRESSURE_TELEMETRY", "1")
    _config, _manager, runtime = _runtime(
        tmp_path, ffi_library, hybrid=False
    )
    assert runtime.pressure_telemetry_enabled

    batch, _plans = runtime.prepare_batch((("r0", 18), ("r1", 33)))
    runtime.mark_lowered(batch)
    runtime.submit_batch(batch)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), 1)
    runtime.poll()
    report = runtime.pressure_report()

    assert report["max_active_requests"] == 2
    assert report["event_counts"]["request_acquired"] == 1
    assert report["event_counts"]["step_prepared"] == 1
    assert report["event_counts"]["step_submitted"] == 1
    assert report["event_counts"]["step_completed"] == 1
    assert report["global"]["request_reachable_unique_pages"] == 5
    assert report["global"]["semantic_live_tokens"] == 51

    runtime.release_batch(("r0", "r1"))
    final = runtime.pressure_report()
    assert final["active_requests"] == 0
    assert final["event_counts"]["request_released"] == 1
    assert final["global"]["free_pages"] == runtime.page_count
    runtime.close()
