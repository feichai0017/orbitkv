#!/usr/bin/env python3
"""Capacity-pressure workload helpers and capability-gated async runner.

The schedule and concurrency functions are intentionally host-only.  The
SGLang engine is supplied by a caller and is accepted only when it exposes the
pinned asynchronous request API; this module never pretends a synchronous or
GPU end-to-end run happened.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


RECORD_SCHEMA = "orbitkv.sglang-v0517-capacity-pressure.v1"
INTERVAL_SEMANTICS = "[submitted_seconds, completed_seconds)"


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def arrival_schedule(
    request_count: int,
    *,
    kind: str,
    request_rate: float | None = None,
    seed: int = 0,
    trace_offsets_seconds: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Create reproducible absolute arrival offsets without sleeping."""

    count = _positive_int("request count", request_count)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("arrival seed must be an integer")
    if kind == "burst":
        if trace_offsets_seconds is not None:
            raise ValueError("burst arrival does not accept trace offsets")
        if request_rate is not None and request_rate != math.inf:
            raise ValueError("burst request rate must be omitted or infinity")
        return (0.0,) * count
    if kind == "poisson":
        if trace_offsets_seconds is not None:
            raise ValueError("Poisson arrival does not accept trace offsets")
        if request_rate == math.inf:
            return (0.0,) * count
        if request_rate is None:
            raise ValueError("Poisson arrival requires request_rate")
        if isinstance(request_rate, bool):
            raise ValueError("Poisson request_rate must be positive or infinity")
        rate = float(request_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Poisson request_rate must be positive or infinity")
        generator = random.Random(seed)
        offsets = [0.0]
        for _ in range(1, count):
            offsets.append(offsets[-1] + generator.expovariate(rate))
        return tuple(offsets)
    if kind == "trace":
        if request_rate is not None:
            raise ValueError("trace arrival does not accept request_rate")
        if trace_offsets_seconds is None or len(trace_offsets_seconds) != count:
            raise ValueError("trace arrival cardinality differs from request count")
        offsets = tuple(
            _finite_nonnegative("trace arrival offset", value)
            for value in trace_offsets_seconds
        )
        if offsets[0] != 0.0:
            raise ValueError("trace arrival must start at zero")
        if any(right < left for left, right in zip(offsets, offsets[1:])):
            raise ValueError("trace arrival offsets must be monotonic")
        return offsets
    raise ValueError("arrival kind must be burst, poisson, or trace")


@dataclass(frozen=True, slots=True)
class ConcurrencySummary:
    interval_semantics: str
    interval_count: int
    peak_concurrency: int
    time_weighted_mean_concurrency: float
    busy_seconds: float
    observation_seconds: float
    area_request_seconds: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "interval_semantics": self.interval_semantics,
            "interval_count": self.interval_count,
            "peak_concurrency": self.peak_concurrency,
            "time_weighted_mean_concurrency": (
                self.time_weighted_mean_concurrency
            ),
            "busy_seconds": self.busy_seconds,
            "observation_seconds": self.observation_seconds,
            "area_request_seconds": self.area_request_seconds,
        }


def sweep_line_concurrency(
    intervals: Sequence[tuple[float, float]],
) -> ConcurrencySummary:
    """Aggregate real request intervals using half-open semantics."""

    values: list[tuple[float, float]] = []
    events: dict[float, int] = {}
    for submitted, completed in intervals:
        begin = _finite_nonnegative("submitted time", submitted)
        end = _finite_nonnegative("completed time", completed)
        if end < begin:
            raise ValueError("request completed before it was submitted")
        values.append((begin, end))
        if end == begin:
            continue
        events[begin] = events.get(begin, 0) + 1
        events[end] = events.get(end, 0) - 1

    if not events:
        return ConcurrencySummary(
            INTERVAL_SEMANTICS, len(values), 0, 0.0, 0.0, 0.0, 0.0
        )

    active = peak = 0
    area = busy = 0.0
    ordered = sorted(events.items())
    previous = ordered[0][0]
    for timestamp, delta in ordered:
        duration = timestamp - previous
        area += active * duration
        if active:
            busy += duration
        active += delta
        if active < 0:
            raise ValueError("concurrency event stream underflowed")
        peak = max(peak, active)
        previous = timestamp
    if active != 0:
        raise ValueError("concurrency event stream did not drain")
    observation = ordered[-1][0] - ordered[0][0]
    return ConcurrencySummary(
        INTERVAL_SEMANTICS,
        len(values),
        peak,
        0.0 if observation == 0 else area / observation,
        busy,
        observation,
        area,
    )


def capacity_pressure_record(
    *,
    arrival: Mapping[str, object],
    request_traces: Sequence[Mapping[str, object]],
    pressure: Mapping[str, object] | None,
    runner_capabilities: Mapping[str, bool],
    scope: str,
) -> dict[str, object]:
    """Build one strict JSON-safe benchmark record from observed intervals."""

    traces = [dict(item) for item in request_traces]
    intervals: list[tuple[float, float]] = []
    for index, trace in enumerate(traces):
        if trace.get("request_index") != index:
            raise ValueError("request traces are not in request-index order")
        submitted = trace.get("submitted_seconds")
        completed = trace.get("completed_seconds")
        if not isinstance(submitted, (int, float)) or isinstance(submitted, bool):
            raise ValueError("request trace lacks submitted_seconds")
        if not isinstance(completed, (int, float)) or isinstance(completed, bool):
            raise ValueError("request trace lacks completed_seconds")
        intervals.append((float(submitted), float(completed)))
    if scope not in ("host_schedule_only", "single_device_eager"):
        raise ValueError("unknown capacity-pressure benchmark scope")
    if scope == "single_device_eager" and pressure is None:
        raise ValueError("an engine run must include pressure telemetry")
    if scope == "single_device_eager" and (
        pressure.get("schema") != "orbitkv.runtime-pressure.v1"
        or pressure.get("enabled") is not True
        or not isinstance(pressure.get("sample_count"), int)
        or pressure.get("sample_count") <= 0
    ):
        raise ValueError("an engine run requires enabled sampled pressure telemetry")
    return {
        "schema": RECORD_SCHEMA,
        "scope": scope,
        "executed": scope == "single_device_eager",
        "claim_boundary": [
            "request-private state only",
            "shared Prefix and request fork RA are unsupported",
            "fixed-state bytes are excluded from KV pressure and RA",
            "host_schedule_only records are not SGLang or GPU evidence",
            "single-run telemetry does not establish a memory-saving claim",
        ],
        "runner_capabilities": dict(runner_capabilities),
        "arrival": dict(arrival),
        "request_traces": traces,
        "concurrency": sweep_line_concurrency(intervals).as_dict(),
        "pressure": None if pressure is None else dict(pressure),
    }


async def run_async_arrivals(
    engine: Any,
    requests: Sequence[Mapping[str, object]],
    offsets_seconds: Sequence[float],
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[tuple[dict[str, object], ...], tuple[Any, ...]]:
    """Drive absolute-deadline arrivals through ``Engine.async_generate``.

    The function is capability-gated and has no synchronous fallback.  It can
    be host-tested with a fake engine; calling it with a real engine is the
    explicit device/E2E boundary.
    """

    generate = getattr(engine, "async_generate", None)
    if not callable(generate):
        raise RuntimeError("SGLang Engine lacks async_generate capability")
    values = tuple(dict(item) for item in requests)
    offsets = tuple(
        _finite_nonnegative("arrival offset", value)
        for value in offsets_seconds
    )
    if len(values) != len(offsets) or not values:
        raise ValueError("request and arrival cardinalities must match and be nonempty")
    if offsets[0] != 0.0 or any(
        right < left for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("arrival offsets must start at zero and be monotonic")

    start = clock()

    async def one(index: int, request: dict[str, object]) -> tuple[dict[str, object], Any]:
        target = start + offsets[index]
        delay = target - clock()
        if delay > 0:
            await sleep(delay)
        submitted = clock() - start
        rid = request.get("rid", f"pressure-{index}")
        kwargs = dict(request)
        kwargs["rid"] = rid
        result = generate(**kwargs)
        if not inspect.isawaitable(result):
            raise RuntimeError("SGLang async_generate did not return an awaitable")
        output = await result
        completed = clock() - start
        return (
            {
                "request_index": index,
                "rid": rid,
                "scheduled_arrival_seconds": offsets[index],
                "submitted_seconds": submitted,
                "completed_seconds": completed,
                "success": True,
            },
            output,
        )

    tasks = [
        asyncio.create_task(one(index, request))
        for index, request in enumerate(values)
    ]
    try:
        pairs = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return (
        tuple(pair[0] for pair in pairs),
        tuple(pair[1] for pair in pairs),
    )


async def run_sglang_capacity_pressure(
    engine: Any,
    requests: Sequence[Mapping[str, object]],
    offsets_seconds: Sequence[float],
    *,
    arrival: Mapping[str, object],
    internal_state: Callable[[], Mapping[str, object]] | None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[dict[str, object], tuple[Any, ...]]:
    """Run the gated async seam and bind it to sampled pressure evidence.

    ``internal_state`` must read the worker's ``orbitkv_manager`` namespace
    after all requests complete.  A missing capability or disabled/empty
    pressure report is rejected; there is no synchronous fallback.
    """

    if not callable(internal_state):
        raise RuntimeError(
            "capacity-pressure E2E requires an internal-state readback capability"
        )
    traces, outputs = await run_async_arrivals(
        engine, requests, offsets_seconds, clock=clock, sleep=sleep
    )
    state = internal_state()
    if not isinstance(state, Mapping):
        raise RuntimeError("internal-state readback is not a mapping")
    manager = state.get("orbitkv_manager", state)
    if not isinstance(manager, Mapping):
        raise RuntimeError("internal-state readback lacks orbitkv_manager")
    pressure = manager.get("pressure")
    if not isinstance(pressure, Mapping):
        raise RuntimeError("internal-state readback lacks pressure telemetry")
    record = capacity_pressure_record(
        arrival=arrival,
        request_traces=traces,
        pressure=pressure,
        runner_capabilities={
            "async_generate": True,
            "scheduled_arrivals": True,
            "interval_concurrency": True,
            "pressure_internal_state": True,
        },
        scope="single_device_eager",
    )
    return record, outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a host-only capacity-pressure workload schedule"
    )
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument(
        "--arrival", choices=("burst", "poisson"), required=True
    )
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    offsets = arrival_schedule(
        args.requests,
        kind=args.arrival,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    record = {
        "schema": RECORD_SCHEMA,
        "scope": "host_schedule_only",
        "executed": False,
        "claim_boundary": (
            "schedule scaffold only; no SGLang engine or GPU was run"
        ),
        "arrival": {
            "kind": args.arrival,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "scheduled_offsets_seconds": offsets,
        },
        "runner_capabilities": {
            "async_generate_required": True,
            "synchronous_fallback": False,
        },
    }
    encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
