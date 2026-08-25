from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))

import bench_capacity_pressure as bench  # noqa: E402


def test_arrival_schedule_supports_burst_seeded_poisson_and_trace() -> None:
    assert bench.arrival_schedule(3, kind="burst") == (0.0, 0.0, 0.0)
    assert bench.arrival_schedule(3, kind="poisson", request_rate=math.inf) == (
        0.0,
        0.0,
        0.0,
    )
    first = bench.arrival_schedule(4, kind="poisson", request_rate=2.0, seed=7)
    second = bench.arrival_schedule(4, kind="poisson", request_rate=2.0, seed=7)
    assert first == second
    assert first[0] == 0.0
    assert all(right >= left for left, right in zip(first, first[1:]))
    assert bench.arrival_schedule(3, kind="trace", trace_offsets_seconds=(0, 1, 1)) == (
        0.0,
        1.0,
        1.0,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"request_count": 0, "kind": "burst"},
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


def test_record_schema_preserves_raw_intervals_and_claim_boundary() -> None:
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
    assert record["request_traces"] == list(traces)
    assert record["concurrency"]["peak_concurrency"] == 2
    assert any("not SGLang or GPU evidence" in item for item in record["claim_boundary"])
    with pytest.raises(ValueError, match="must include pressure"):
        bench.capacity_pressure_record(
            arrival={},
            request_traces=traces,
            pressure=None,
            runner_capabilities={"async_generate": True},
            scope="single_device_eager",
        )
    with pytest.raises(ValueError, match="enabled sampled"):
        bench.capacity_pressure_record(
            arrival={},
            request_traces=traces,
            pressure={
                "schema": "orbitkv.runtime-pressure.v1",
                "enabled": False,
                "sample_count": 0,
            },
            runner_capabilities={"async_generate": True},
            scope="single_device_eager",
        )


def test_async_runner_requires_capability_and_preserves_order() -> None:
    with pytest.raises(RuntimeError, match="lacks async_generate"):
        asyncio.run(
            bench.run_async_arrivals(
                object(), ({"input_ids": [1]},), (0.0,)
            )
        )

    class Engine:
        async def async_generate(self, **kwargs):
            await asyncio.sleep(0)
            return {"rid": kwargs["rid"]}

    traces, outputs = asyncio.run(
        bench.run_async_arrivals(
            Engine(),
            (
                {"input_ids": [1], "rid": "first"},
                {"input_ids": [2], "rid": "second"},
            ),
            (0.0, 0.0),
        )
    )
    assert tuple(item["request_index"] for item in traces) == (0, 1)
    assert tuple(item["rid"] for item in outputs) == ("first", "second")
    assert all(item["completed_seconds"] >= item["submitted_seconds"] for item in traces)


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


def test_sglang_runner_is_pressure_readback_capability_gated() -> None:
    class Engine:
        async def async_generate(self, **kwargs):
            return {"rid": kwargs["rid"]}

    request = ({"input_ids": [1], "rid": "r0"},)
    with pytest.raises(RuntimeError, match="internal-state readback"):
        asyncio.run(
            bench.run_sglang_capacity_pressure(
                Engine(), request, (0.0,), arrival={"kind": "burst"},
                internal_state=None,
            )
        )
    with pytest.raises(ValueError, match="enabled sampled"):
        asyncio.run(
            bench.run_sglang_capacity_pressure(
                Engine(), request, (0.0,), arrival={"kind": "burst"},
                internal_state=lambda: {
                    "orbitkv_manager": {
                        "pressure": {
                            "schema": "orbitkv.runtime-pressure.v1",
                            "enabled": False,
                            "sample_count": 0,
                        }
                    }
                },
            )
        )

    record, outputs = asyncio.run(
        bench.run_sglang_capacity_pressure(
            Engine(), request, (0.0,), arrival={"kind": "burst"},
            internal_state=lambda: {
                "orbitkv_manager": {
                    "pressure": {
                        "schema": "orbitkv.runtime-pressure.v1",
                        "enabled": True,
                        "sample_count": 1,
                    }
                }
            },
        )
    )
    assert record["scope"] == "single_device_eager"
    assert record["runner_capabilities"]["pressure_internal_state"]
    assert outputs == ({"rid": "r0"},)
