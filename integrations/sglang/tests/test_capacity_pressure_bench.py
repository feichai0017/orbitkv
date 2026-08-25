from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))

import bench_capacity_pressure as bench  # noqa: E402


def _pressure(
    *,
    enabled: bool = True,
    sample_count: int = 9,
    max_active: int = 2,
    active: int = 0,
) -> dict[str, object]:
    values = {
        "capacity_pages": 64,
        "free_pages": 64,
        "consumed_pages": 0,
        "resident_data_pages": 0,
        "request_reachable_unique_pages": 0,
        "capacity_bytes": 65536,
        "free_bytes": 65536,
        "consumed_bytes": 0,
        "resident_data_bytes": 0,
        "request_reachable_unique_bytes": 0,
        "semantic_live_tokens": 0,
        "semantic_live_bytes": 0,
        "retention_amplification_milli": None,
        "min_free_pages": 56,
        "min_free_bytes": 57344,
        "high_water_consumed_pages": 8,
        "high_water_consumed_bytes": 8192,
        "high_water_resident_data_pages": 6,
        "high_water_resident_data_bytes": 6144,
        "high_water_request_reachable_unique_pages": 6,
        "high_water_request_reachable_unique_bytes": 6144,
        "high_water_semantic_live_bytes": 4096,
        "high_water_retention_amplification_milli": 1500,
        "sample_count": sample_count,
    }
    return {
        "schema": bench.PRESSURE_SCHEMA,
        "enabled": enabled,
        "mode": "event_driven_high_water",
        "scope": dict(bench._PRESSURE_SCOPE),
        "sample_count": sample_count,
        "last_event": "request_released",
        "event_counts": {"request_released": sample_count},
        "active_requests": active,
        "max_active_requests": max_active,
        "global": dict(values),
        "classes": [{"class_id": 0, "name": "full", **values}],
    }


def _manager_state(
    *,
    pressure: dict[str, object] | None = None,
    counters: dict[str, int] | None = None,
    fixed_state_bytes: int = 0,
) -> dict[str, object]:
    all_counters = {
        name: 0
        for name in (
            *bench._PREFIX_AND_FORK_COUNTERS,
            *bench._FAILURE_COUNTERS,
            *bench._FIXED_STATE_COUNTERS,
        )
    }
    all_counters.update(counters or {})
    stats = {
        name: 0
        for name in (
            "active_requests",
            "active_snapshots",
            "active_prefixes",
            "evicted_prefixes",
            "prepared_steps",
            "submitted_steps",
            "reserved_pages",
            "writing_pages",
            "active_pages",
            "retiring_pages",
            "quarantined_pages",
            "exhausted_pages",
            "pending_reclamations",
            "total_request_page_refs",
            "total_prefix_page_refs",
            "total_reader_pins",
        )
    }
    stats["free_pages"] = 64
    manager = {
        "abi_version": 8,
        "fixed_state_byte_count": fixed_state_bytes,
        "fixed_state_descriptors": (
            [] if fixed_state_bytes == 0 else [{"name": "gdn"}]
        ),
        "batch_counters": all_counters,
        "manager_stats": stats,
        "arena_stats": [
            {
                "class_id": 0,
                "page_count": 64,
                "free_pages": 64,
                "prefix_page_refs": 0,
            }
        ],
        "pressure": _pressure() if pressure is None else pressure,
    }
    return {"orbitkv_manager": manager}


def _output(rid: str, ids: list[int] | None = None) -> dict[str, object]:
    return {
        "output_ids": [11, 12] if ids is None else ids,
        "meta_info": {"id": rid, "cached_tokens": 0},
    }


def test_arrival_schedule_supports_burst_seeded_poisson_and_trace() -> None:
    assert bench.arrival_schedule(3, kind="burst") == (0.0, 0.0, 0.0)
    assert bench.arrival_schedule(
        3, kind="poisson", request_rate=math.inf
    ) == (0.0, 0.0, 0.0)
    first = bench.arrival_schedule(4, kind="poisson", request_rate=2.0, seed=7)
    second = bench.arrival_schedule(4, kind="poisson", request_rate=2.0, seed=7)
    assert first == second
    assert first[0] == 0.0
    assert all(right >= left for left, right in zip(first, first[1:]))
    assert bench.arrival_schedule(
        3, kind="trace", trace_offsets_seconds=(0, 1, 1)
    ) == (0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"request_count": 0, "kind": "burst"},
        {"request_count": 2, "kind": "burst", "request_rate": 1},
        {"request_count": 2, "kind": "poisson", "request_rate": 0},
        {
            "request_count": 2,
            "kind": "trace",
            "trace_offsets_seconds": (1, 2),
        },
        {
            "request_count": 2,
            "kind": "trace",
            "trace_offsets_seconds": (0, -1),
        },
    ),
)
def test_arrival_schedule_fails_closed_on_invalid_inputs(kwargs) -> None:
    with pytest.raises(ValueError):
        bench.arrival_schedule(**kwargs)


def test_sweep_line_uses_half_open_actual_request_intervals() -> None:
    result = bench.sweep_line_concurrency(((0, 2), (1, 3), (2, 4), (4, 4)))
    assert result.interval_semantics == "[submitted_seconds, completed_seconds)"
    assert result.interval_count == 4
    assert result.peak_concurrency == 2
    assert result.area_request_seconds == 6.0
    assert result.busy_seconds == 4.0
    assert result.observation_seconds == 4.0
    assert result.time_weighted_mean_concurrency == 1.5


def test_sweep_line_touching_intervals_do_not_overlap() -> None:
    result = bench.sweep_line_concurrency(((3, 4), (0, 1), (1, 3)))
    assert result.peak_concurrency == 1
    assert result.area_request_seconds == 4.0
    assert result.time_weighted_mean_concurrency == 1.0
    assert bench.sweep_line_concurrency(((1, 1),)).peak_concurrency == 0
    with pytest.raises(ValueError, match="completed before"):
        bench.sweep_line_concurrency(((2, 1),))


def test_host_record_is_explicitly_nonexecuted_and_not_performance_go() -> None:
    traces = (
        {
            "request_index": 0,
            "rid": "r0",
            "submitted_seconds": 0.0,
            "completed_seconds": 2.0,
        },
        {
            "request_index": 1,
            "rid": "r1",
            "submitted_seconds": 1.0,
            "completed_seconds": 3.0,
        },
    )
    record = bench.capacity_pressure_record(
        arrival={"kind": "trace", "scheduled_offsets_seconds": [0, 1]},
        request_traces=traces,
        pressure=None,
        runner_capabilities={"async_generate": False},
        scope="host_schedule_only",
    )
    assert record["schema"] == bench.RECORD_SCHEMA
    assert record["executed"] is False
    assert record["diagnostic_only"] is True
    assert record["performance_go"] is False
    assert record["request_traces"] == list(traces)
    assert record["concurrency"]["peak_concurrency"] == 2
    assert record["gpu_allocator_peak"]["status"] == "unavailable"
    assert any(
        "not SGLang or GPU evidence" in item for item in record["claim_boundary"]
    )


def test_engine_record_requires_outputs_and_summarizes_pressure() -> None:
    trace = {
        "request_index": 0,
        "rid": "r0",
        "returned_rid": "r0",
        "cached_tokens": 0,
        "output_ids": [1, 2],
        "submitted_seconds": 0.0,
        "completed_seconds": 2.0,
    }
    with pytest.raises(ValueError, match="must include pressure"):
        bench.capacity_pressure_record(
            arrival={}, request_traces=(trace,), pressure=None,
            runner_capabilities={"async_generate": True},
            scope="single_device_eager",
        )
    record = bench.capacity_pressure_record(
        arrival={}, request_traces=(trace,), pressure=_pressure(max_active=1),
        runner_capabilities={"async_generate": True},
        scope="single_device_eager",
    )
    assert record["pressure_summary"] == {
        "max_active": 1,
        "peak_consumed_bytes": 8192,
        "peak_resident_bytes": 6144,
        "peak_request_reachable_unique_bytes": 6144,
        "peak_semantic_live_bytes": 4096,
        "request_private_retention_amplification_milli": 1500,
        "request_private_retention_amplification_ratio": 1.5,
        "retention_amplification_scope": "request_private_kv_only",
        "sample_count": 9,
    }
    bad_trace = dict(trace, cached_tokens=16)
    with pytest.raises(ValueError, match="Prefix reuse"):
        bench.capacity_pressure_record(
            arrival={}, request_traces=(bad_trace,), pressure=_pressure(max_active=1),
            runner_capabilities={"async_generate": True},
            scope="single_device_eager",
        )


def test_pressure_summary_rejects_schema_drift_and_bad_aggregates() -> None:
    malformed = _pressure(max_active=1)
    malformed["mode"] = "periodic"
    with pytest.raises(ValueError, match="event-driven"):
        bench._pressure_summary(
            malformed, request_count=1, require_activity=True
        )

    malformed = _pressure(max_active=1)
    malformed["classes"][0].pop("high_water_consumed_bytes")
    with pytest.raises(ValueError, match="noncanonical field set"):
        bench._pressure_summary(
            malformed, request_count=1, require_activity=True
        )

    malformed = _pressure(max_active=1)
    malformed["classes"][0]["consumed_bytes"] = 1024
    with pytest.raises(ValueError, match="page/byte geometry"):
        bench._pressure_summary(
            malformed, request_count=1, require_activity=True
        )

    malformed = _pressure(max_active=1)
    malformed["global"]["retention_amplification_milli"] = 1
    with pytest.raises(ValueError, match="current retention amplification"):
        bench._pressure_summary(
            malformed, request_count=1, require_activity=True
        )
def test_async_runner_uses_staggered_deadlines_and_preserves_output_order() -> None:
    calls: list[tuple[str, float]] = []
    now = [10.0]

    async def fake_sleep(delay: float) -> None:
        now[0] += delay
        await asyncio.sleep(0)

    class Engine:
        async def async_generate(self, **kwargs):
            calls.append((kwargs["rid"], now[0]))
            await asyncio.sleep(0)
            return _output(kwargs["rid"], [len(calls)])

    traces, outputs = asyncio.run(
        bench.run_async_arrivals(
            Engine(),
            (
                {"input_ids": [1], "rid": "first"},
                {"input_ids": [2], "rid": "second"},
            ),
            (0.0, 0.25),
            expected_output_tokens=1,
            clock=lambda: now[0],
            sleep=fake_sleep,
        )
    )
    assert tuple(item["request_index"] for item in traces) == (0, 1)
    assert tuple(item["returned_rid"] for item in traces) == ("first", "second")
    assert [item["output_ids"] for item in traces] == [[1], [2]]
    assert calls == [("first", 10.0), ("second", 10.25)]
    assert traces[1]["submitted_seconds"] == pytest.approx(0.25)
    assert tuple(item["meta_info"]["id"] for item in outputs) == (
        "first", "second"
    )


def test_async_runner_requires_capability_and_rejects_foreign_outputs() -> None:
    with pytest.raises(RuntimeError, match="lacks async_generate"):
        asyncio.run(
            bench.run_async_arrivals(object(), ({"input_ids": [1]},), (0.0,))
        )

    class SyncEngine:
        def async_generate(self, **kwargs):
            return _output(kwargs["rid"])

    with pytest.raises(RuntimeError, match="did not return an awaitable"):
        asyncio.run(
            bench.run_async_arrivals(
                SyncEngine(), ({"input_ids": [1], "rid": "r0"},), (0.0,)
            )
        )

    class ForeignEngine:
        async def async_generate(self, **kwargs):
            return _output("foreign")

    with pytest.raises(RuntimeError, match="foreign request id"):
        asyncio.run(
            bench.run_async_arrivals(
                ForeignEngine(), ({"input_ids": [1], "rid": "r0"},), (0.0,)
            )
        )


def test_async_runner_cancels_and_reaps_siblings_on_failure() -> None:
    cancelled = asyncio.Event()

    class Engine:
        async def async_generate(self, **kwargs):
            if kwargs["rid"] == "bad":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            await bench.run_async_arrivals(
                Engine(),
                (
                    {"input_ids": [1], "rid": "slow"},
                    {"input_ids": [2], "rid": "bad"},
                ),
                (0.0, 0.0),
            )
        assert cancelled.is_set()

    asyncio.run(exercise())


def test_sglang_runner_reads_async_worker_state_and_reports_outputs() -> None:
    class Engine:
        async def async_generate(self, **kwargs):
            return _output(kwargs["rid"], [7, 8])

    async def readback() -> dict[str, object]:
        await asyncio.sleep(0)
        return _manager_state(pressure=_pressure(max_active=1))

    record, outputs = asyncio.run(
        bench.run_sglang_capacity_pressure(
            Engine(),
            ({"input_ids": [1], "rid": "r0"},),
            (0.0,),
            arrival={"kind": "burst"},
            internal_state=readback,
            expected_output_tokens=2,
        )
    )
    assert record["scope"] == "single_device_eager"
    assert record["pressure_summary"]["max_active"] == 1
    assert record["runner_capabilities"]["pressure_internal_state"]
    assert record["performance_go"] is False
    assert outputs == (_output("r0", [7, 8]),)


@pytest.mark.parametrize(
    ("state", "message"),
    (
        ({}, "lacks orbitkv_manager"),
        (_manager_state(pressure=_pressure(enabled=False)), "not enabled"),
        (
            _manager_state(counters={"request_fork_batch_calls": 1}),
            "request_fork_batch_calls=1",
        ),
        (_manager_state(fixed_state_bytes=4096), "does not support fixed-state"),
    ),
)
def test_sglang_runner_rejects_missing_or_unsupported_readback(
    state, message
) -> None:
    class Engine:
        async def async_generate(self, **kwargs):
            return _output(kwargs["rid"])

    with pytest.raises((RuntimeError, ValueError), match=message):
        asyncio.run(
            bench.run_sglang_capacity_pressure(
                Engine(),
                ({"input_ids": [1], "rid": "r0"},),
                (0.0,),
                arrival={"kind": "burst"},
                internal_state=lambda: state,
                expected_output_tokens=2,
            )
        )


def _arguments(tmp_path: Path, **overrides) -> argparse.Namespace:
    sglang = tmp_path / "sglang"
    model = tmp_path / "model"
    (sglang / "python/sglang").mkdir(parents=True)
    model.mkdir()
    (sglang / "python/sglang/__init__.py").write_text("", encoding="utf-8")
    (model / "config.json").write_text("{}", encoding="utf-8")
    plan = tmp_path / "plan.json"
    library = tmp_path / "liborbitkv_ffi.so"
    plan.write_text("{}", encoding="utf-8")
    library.write_bytes(b"ffi")
    values = {
        "sglang_root": str(sglang),
        "model": str(model),
        "plan": str(plan),
        "library": str(library),
        "requests": 8,
        "max_running_requests": 4,
        "prompt_tokens": 128,
        "decode_tokens": 32,
        "chunked_prefill_size": 128,
        "context_length": 256,
        "max_total_tokens": 1024,
        "reclamation_mode": "relocate",
        "reclamation_trigger_tokens": 128,
        "retained_per_page": 8,
        "fragmentation_threshold_milli": 500,
        "maximum_source_pages": 8,
        "evacuation_headroom_pages": 4,
        "arrival": "poisson",
        "request_rate": 4.0,
        "seed": 7,
        "attention_backend": "fa3",
        "fp8_gemm_backend": None,
        "mem_fraction_static": None,
        "output": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_arguments_accept_small_smoke_and_long_context_profiles(tmp_path) -> None:
    smoke = _arguments(tmp_path)
    paths = bench.validate_arguments(smoke)
    assert paths["state_plan"] is None
    assert paths["model"].name == "model"

    long_context = argparse.Namespace(
        **{
            **vars(smoke),
            "requests": 64,
            "max_running_requests": 16,
            "prompt_tokens": 32768,
            "reclamation_trigger_tokens": 32768,
            "decode_tokens": 1024,
            "chunked_prefill_size": 8192,
            "context_length": 65536,
            "max_total_tokens": 16 * (32768 + 1024),
            "maximum_source_pages": 2048,
            "evacuation_headroom_pages": 1024,
            "request_rate": 1.5,
        }
    )
    bench.validate_arguments(long_context)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"requests": 0}, "--requests must be a positive integer"),
        (
            {"max_running_requests": 9},
            "max-running-requests cannot exceed",
        ),
        ({"context_length": 160}, "leave one context slot"),
        ({"reclamation_trigger_tokens": 48}, "must equal --prompt-tokens"),
        ({"chunked_prefill_size": 127}, "must be page aligned"),
        ({"max_total_tokens": 1000}, "must be page aligned"),
        ({"max_total_tokens": 496}, "cannot hold every configured request prefill"),
        ({"retained_per_page": 16}, "below page size"),
        ({"maximum_source_pages": 7}, "below the reclamation trigger"),
        ({"evacuation_headroom_pages": 1}, "below the retained footprint"),
        ({"fragmentation_threshold_milli": 501}, "exceeds expected"),
        (
            {"reclamation_mode": "naive", "attention_backend": "fa3"},
            "token-indexed FlashInfer",
        ),
        ({"arrival": "poisson", "request_rate": None}, "requires a positive"),
        ({"arrival": "burst", "request_rate": 1.0}, "must be omitted"),
        ({"mem_fraction_static": 0.0}, "must be in"),
    ),
)
def test_argument_validation_fails_closed(tmp_path, overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        bench.validate_arguments(_arguments(tmp_path, **overrides))


def test_fixed_state_plan_is_rejected_before_engine_construction() -> None:
    class Config:
        fixed_states = (object(),)
        fixed_state_byte_count = 4096

    with pytest.raises(RuntimeError, match="rejects fixed-state RA"):
        bench._reject_fixed_state(Config(), {"fixed_states": [{}]})


def test_reclamation_activity_is_mode_specific_and_nonempty() -> None:
    names = (
        "token_disposition_batches",
        "token_policy_evictions",
        "relocation_batches",
        "relocation_moves",
        "relocation_reclaimed_pages",
        "relocation_copy_events",
        "relocation_copy_tokens",
        "prepare_relocation_batch_calls",
        "submit_relocation_batch_calls",
        "complete_relocation_batch_calls",
        "abort_relocations_batch_calls",
    )
    empty = {name: 0 for name in names}
    with pytest.raises(RuntimeError, match="no token disposition"):
        bench._validate_reclamation_activity(empty, "relocate")

    naive = dict(empty, token_disposition_batches=2, token_policy_evictions=32)
    assert bench._validate_reclamation_activity(naive, "naive")[
        "token_policy_evictions"
    ] == 32
    with pytest.raises(RuntimeError, match="reported relocation"):
        bench._validate_reclamation_activity(
            dict(naive, relocation_batches=1), "naive"
        )

    relocate = dict(
        naive,
        token_disposition_batches=2,
        relocation_batches=2,
        relocation_moves=32,
        relocation_reclaimed_pages=2,
        relocation_copy_events=2,
        relocation_copy_tokens=32,
        prepare_relocation_batch_calls=2,
        submit_relocation_batch_calls=2,
        complete_relocation_batch_calls=2,
    )
    assert bench._validate_reclamation_activity(relocate, "relocate")[
        "relocation_reclaimed_pages"
    ] == 2
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        bench._validate_reclamation_activity(
            dict(relocate, relocation_copy_events=1), "relocate"
        )


def test_run_uses_engine_loop_async_generate_and_two_worker_readbacks(
    tmp_path, monkeypatch
) -> None:
    args = _arguments(
        tmp_path,
        requests=2,
        max_running_requests=2,
        prompt_tokens=48,
        reclamation_trigger_tokens=48,
        decode_tokens=2,
        context_length=64,
        chunked_prefill_size=48,
        arrival="burst",
        request_rate=None,
    )
    paths = bench.validate_arguments(args)
    calls: list[str] = []
    engine_kwargs: dict[str, object] = {}

    class ClassConfig:
        name = "full"

    config = SimpleNamespace(
        fixed_states=(),
        fixed_state_byte_count=0,
        classes=(ClassConfig(),),
        token_reclamation=SimpleNamespace(**bench._reclamation_policy(args)),
    )
    contract = {
        "architecture": "Qwen2ForCausalLM",
        "attention_backend": "fa3",
        "backend_profile": {"attention_backend": "fa3"},
        "workload_profile": "prefix_reuse",
        "state_ownership": "token_prefix_shareable",
        "fixed_states": [],
        "max_position_embeddings": 128,
        "vocab_size": 256,
        "prompt_token_upper_bound": 256,
        "control_token_ids": {},
    }
    final_pressure = _pressure(sample_count=9, max_active=2)
    manager_load = {"pressure": _pressure(sample_count=1, max_active=0)}
    manager_final = {
        "pressure": final_pressure,
        "batch_counters": {
            "token_disposition_batches": 1,
            "token_policy_evictions": 16,
            "relocation_batches": 1,
            "relocation_moves": 16,
            "relocation_reclaimed_pages": 1,
            "relocation_copy_events": 1,
            "relocation_copy_tokens": 16,
            "prepare_relocation_batch_calls": 1,
            "submit_relocation_batch_calls": 1,
            "complete_relocation_batch_calls": 1,
            "abort_relocations_batch_calls": 0,
        },
    }

    class FakeEngine:
        def __init__(self, **kwargs):
            engine_kwargs.update(kwargs)
            self.loop = asyncio.new_event_loop()
            calls.append("engine_init")

        def __enter__(self):
            calls.append("engine_enter")
            return self

        def __exit__(self, *_):
            calls.append("engine_exit")
            self.loop.close()

        async def async_generate(self, **kwargs):
            calls.append(f"generate:{kwargs['rid']}")
            await asyncio.sleep(0)
            return _output(kwargs["rid"], [7, 8])

        def get_server_info(self):
            calls.append("readback")
            return {"stage": calls.count("readback")}

    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=FakeEngine, __version__="0.5.17"),
    )
    monkeypatch.setenv("ORBITKV_PRESSURE_TELEMETRY", "previous")
    monkeypatch.setenv("ORBITKV_TOKEN_RECLAMATION", "previous")
    monkeypatch.setattr(
        bench.common, "configure_environment", lambda *_: {"base": "frozen"}
    )
    monkeypatch.setattr(bench.common, "verify_sglang_source", lambda *_: {})
    monkeypatch.setattr(bench.common, "verify_manager_entrypoint", lambda: {})
    monkeypatch.setattr(bench.common, "verify_pinned_module_constants", lambda: {})
    monkeypatch.setattr(bench.common, "sha256_file", lambda _: "sha256")
    monkeypatch.setattr(bench.common, "_adapter_identity", lambda: {})
    monkeypatch.setattr(bench.common, "artifact_identity", lambda _: {})
    monkeypatch.setattr(bench.common, "manager_plan_identity", lambda _: {})
    monkeypatch.setattr(
        bench.common, "checkpoint_contract", lambda *_: (contract, {"id": "model"})
    )
    monkeypatch.setattr(
        bench.common,
        "fresh_input_ids",
        lambda **kwargs: [
            [3] * kwargs["prompt_tokens"] for _ in range(kwargs["requests"])
        ],
    )

    def engine_arguments(*_, diagnostic_attention_backend=None):
        assert diagnostic_attention_backend == "fa3"
        assert os.environ["ORBITKV_PRESSURE_TELEMETRY"] == "1"
        assert "\"mode\":\"relocate\"" in os.environ[
            "ORBITKV_TOKEN_RECLAMATION"
        ]
        return {"fake_engine_arg": True}

    monkeypatch.setattr(bench.common, "engine_arguments", engine_arguments)
    monkeypatch.setattr(
        bench.common,
        "verify_runtime_contract",
        lambda *_args, **_kwargs: {"full_tokens": 1024},
    )
    monkeypatch.setattr(
        bench.common,
        "gpu_snapshot",
        lambda label: {"label": label, "gpus": [{"name": "fake H20"}]},
    )

    def pressure_state(info, **kwargs):
        stage = kwargs["stage"]
        calls.append(f"validate:{stage}")
        if info["stage"] == 1:
            return manager_load, {"sample_count": 1, "max_active": 0}
        return manager_final, {"sample_count": 9, "max_active": 2}

    monkeypatch.setattr(bench, "validate_private_pressure_state", pressure_state)
    import orbitkv_sglang.config as config_module

    monkeypatch.setattr(config_module, "load_config", lambda: config)

    record = bench.run(args, paths)

    assert engine_kwargs == {"fake_engine_arg": True}
    assert calls == [
        "engine_init",
        "engine_enter",
        "readback",
        "validate:after_load",
        "generate:orbitkv-pressure-7-0",
        "generate:orbitkv-pressure-7-1",
        "readback",
        "validate:after_workload",
        "engine_exit",
    ]
    assert record["executed"] is True
    assert record["performance_go"] is False
    assert record["request_output_ids"] == [[7, 8], [7, 8]]
    assert record["workload"]["observed_reclamation_activity"][
        "relocation_batches"
    ] == 1
    assert record["runtime_identity"]["execution_profile"] == (
        "sglang_offline_engine_eager_tp1"
    )
    assert [item["label"] for item in record["gpu_snapshots"]] == [
        "before_engine",
        "after_load",
        "after_workload",
        "after_shutdown",
    ]
