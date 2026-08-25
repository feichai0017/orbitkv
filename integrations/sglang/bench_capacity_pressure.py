#!/usr/bin/env python3
"""Run one request-private SGLang capacity-pressure diagnostic.

The device path is deliberately fail closed.  It requires the pinned SGLang
checkout, the canonical OrbitKV manager, ``Engine.async_generate``, and ABI8
worker pressure readback.  Prefix sharing, request forks, and fixed-state
retention amplification are outside this schema.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import bench_canonical_manager as common


RECORD_SCHEMA = "orbitkv.sglang-v0517-capacity-pressure.v2"
PRESSURE_SCHEMA = "orbitkv.runtime-pressure.v1"
INTERVAL_SEMANTICS = "[submitted_seconds, completed_seconds)"
_PREFIX_AND_FORK_COUNTERS = (
    "request_fork_batch_calls",
    "prefix_lookup_batch_calls",
    "prefix_attach_batch_calls",
    "prefix_publish_batch_calls",
    "prefix_publish_release_batch_calls",
    "prefix_evict_batch_calls",
    "prefix_recycle_batch_calls",
)
_FAILURE_COUNTERS = (
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "abort_steps_batch_calls",
    "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls",
    "retryable_conflicts",
    "fail_stops",
    "fail_stop_count",
    "quarantine_count",
)
_FIXED_STATE_COUNTERS = (
    "fixed_state_prepares",
    "fixed_state_clears",
    "fixed_state_copies",
    "fixed_state_events",
    "fixed_state_retirements",
    "fixed_state_acks",
)
_PRESSURE_SCOPE = {
    "state_ownership": "request_private",
    "shared_prefix_deduplication": "unsupported_fail_closed",
    "request_fork": "unsupported_fail_closed",
    "consumed_pages": "capacity_pages - free_pages",
    "resident_data_pages": (
        "writing_pages + active_pages + retiring_pages + quarantined_pages"
    ),
    "retention_amplification": (
        "resident_data_bytes / request_private_semantic_live_bytes"
    ),
    "retiring_and_quarantined_are_safety_cost": True,
    "fixed_state_bytes": "excluded",
}
_PRESSURE_CURRENT_FIELDS = (
    "capacity_pages",
    "free_pages",
    "consumed_pages",
    "resident_data_pages",
    "request_reachable_unique_pages",
    "capacity_bytes",
    "free_bytes",
    "consumed_bytes",
    "resident_data_bytes",
    "request_reachable_unique_bytes",
    "semantic_live_tokens",
    "semantic_live_bytes",
)
_PRESSURE_WATER_FIELDS = (
    "min_free_pages",
    "min_free_bytes",
    "high_water_consumed_pages",
    "high_water_consumed_bytes",
    "high_water_resident_data_pages",
    "high_water_resident_data_bytes",
    "high_water_request_reachable_unique_pages",
    "high_water_request_reachable_unique_bytes",
    "high_water_semantic_live_bytes",
)
_POLICY_ID = 260825001


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _nonnegative_record(
    value: Mapping[str, object], names: Sequence[str], label: str
) -> dict[str, int]:
    return {
        name: _nonnegative_int(f"{label}.{name}", value.get(name))
        for name in names
    }


def _expected_amplification(resident: int, semantic: int) -> int | None:
    return None if semantic == 0 else resident * 1000 // semantic


def _validate_pressure_values(
    value: Mapping[str, object],
    *,
    label: str,
    sample_count: int,
    page_geometry: bool,
) -> dict[str, int]:
    expected_keys = {
        *_PRESSURE_CURRENT_FIELDS,
        *_PRESSURE_WATER_FIELDS,
        "retention_amplification_milli",
        "high_water_retention_amplification_milli",
        "sample_count",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{label} has a noncanonical field set")
    current = _nonnegative_record(value, _PRESSURE_CURRENT_FIELDS, label)
    water = _nonnegative_record(value, _PRESSURE_WATER_FIELDS, label)
    if value.get("sample_count") != sample_count:
        raise ValueError(f"{label} sample_count differs from pressure telemetry")
    if current["capacity_pages"] <= 0 or current["capacity_bytes"] <= 0:
        raise ValueError(f"{label} capacity must be positive")
    if page_geometry:
        if current["capacity_bytes"] % current["capacity_pages"]:
            raise ValueError(f"{label} byte geometry is not page integral")
        page_bytes = current["capacity_bytes"] // current["capacity_pages"]
        for page_name, byte_name in (
            ("free_pages", "free_bytes"),
            ("consumed_pages", "consumed_bytes"),
            ("resident_data_pages", "resident_data_bytes"),
            (
                "request_reachable_unique_pages",
                "request_reachable_unique_bytes",
            ),
            ("min_free_pages", "min_free_bytes"),
            ("high_water_consumed_pages", "high_water_consumed_bytes"),
            (
                "high_water_resident_data_pages",
                "high_water_resident_data_bytes",
            ),
            (
                "high_water_request_reachable_unique_pages",
                "high_water_request_reachable_unique_bytes",
            ),
        ):
            source = current if page_name in current else water
            target = current if byte_name in current else water
            if source[page_name] * page_bytes != target[byte_name]:
                raise ValueError(
                    f"{label} page/byte geometry differs at {page_name}"
                )
    if (
        current["free_pages"] + current["consumed_pages"]
        != current["capacity_pages"]
        or current["resident_data_pages"] > current["consumed_pages"]
        or current["request_reachable_unique_pages"]
        > current["resident_data_pages"]
        or current["resident_data_bytes"] > current["consumed_bytes"]
        or current["semantic_live_bytes"]
        > current["request_reachable_unique_bytes"]
        or current["request_reachable_unique_bytes"]
        > current["resident_data_bytes"]
    ):
        raise ValueError(f"{label} current pressure ordering is invalid")
    if (
        water["min_free_pages"] > current["free_pages"]
        or water["high_water_consumed_pages"] < current["consumed_pages"]
        or water["high_water_consumed_pages"] > current["capacity_pages"]
        or water["high_water_resident_data_pages"]
        < current["resident_data_pages"]
        or water["high_water_resident_data_pages"]
        > water["high_water_consumed_pages"]
        or water["high_water_request_reachable_unique_pages"]
        < current["request_reachable_unique_pages"]
        or water["high_water_request_reachable_unique_pages"]
        > water["high_water_resident_data_pages"]
        or water["high_water_resident_data_bytes"]
        > water["high_water_consumed_bytes"]
        or water["high_water_request_reachable_unique_bytes"]
        > water["high_water_resident_data_bytes"]
        or water["high_water_semantic_live_bytes"]
        < current["semantic_live_bytes"]
        or water["high_water_semantic_live_bytes"]
        > water["high_water_request_reachable_unique_bytes"]
    ):
        raise ValueError(f"{label} pressure high-water ordering is invalid")
    if (
        water["min_free_pages"] + water["high_water_consumed_pages"]
        != current["capacity_pages"]
        or water["min_free_bytes"] + water["high_water_consumed_bytes"]
        != current["capacity_bytes"]
    ):
        raise ValueError(f"{label} free/consumed high waters are inconsistent")
    current_ra = value.get("retention_amplification_milli")
    expected_ra = _expected_amplification(
        current["resident_data_bytes"], current["semantic_live_bytes"]
    )
    if current_ra != expected_ra:
        raise ValueError(f"{label} current retention amplification is invalid")
    high_ra = value.get("high_water_retention_amplification_milli")
    if high_ra is not None:
        high_ra = _nonnegative_int(
            f"{label}.high_water_retention_amplification_milli", high_ra
        )
        if high_ra < 1000 or (current_ra is not None and high_ra < current_ra):
            raise ValueError(f"{label} high-water retention amplification is invalid")
    return {**current, **water}


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
        if request_rate is None or isinstance(request_rate, bool):
            raise ValueError("Poisson arrival requires a positive request_rate")
        rate = float(request_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Poisson request_rate must be positive and finite")
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


def _pressure_summary(
    pressure: Mapping[str, object],
    *,
    request_count: int,
    require_activity: bool,
    expected_classes: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate the request-private pressure schema and extract high waters."""

    if pressure.get("schema") != PRESSURE_SCHEMA:
        raise ValueError("pressure telemetry has an unsupported schema")
    if pressure.get("enabled") is not True:
        raise ValueError("pressure telemetry is not enabled")
    if pressure.get("mode") != "event_driven_high_water":
        raise ValueError("pressure telemetry mode is not event-driven high-water")
    if set(pressure) != {
        "schema",
        "enabled",
        "mode",
        "scope",
        "sample_count",
        "last_event",
        "event_counts",
        "active_requests",
        "max_active_requests",
        "global",
        "classes",
    }:
        raise ValueError("pressure telemetry has a noncanonical field set")
    sample_count = _nonnegative_int(
        "pressure sample_count", pressure.get("sample_count")
    )
    if sample_count == 0:
        raise ValueError("pressure telemetry has no samples")
    event_counts = pressure.get("event_counts")
    if (
        not isinstance(event_counts, Mapping)
        or not event_counts
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for name, count in event_counts.items()
        )
        or sum(event_counts.values()) != sample_count
        or pressure.get("last_event") not in event_counts
    ):
        raise ValueError("pressure event census is invalid")
    scope = pressure.get("scope")
    if not isinstance(scope, Mapping) or dict(scope) != _PRESSURE_SCOPE:
        raise ValueError(
            "pressure telemetry is not request-private Prefix/fork-fail-closed KV"
        )
    active = _nonnegative_int(
        "pressure active_requests", pressure.get("active_requests")
    )
    maximum = _nonnegative_int(
        "pressure max_active_requests", pressure.get("max_active_requests")
    )
    if active != 0:
        raise ValueError("pressure telemetry was read before requests drained")
    if maximum > request_count:
        raise ValueError("pressure max_active_requests exceeds submitted requests")
    global_values = pressure.get("global")
    if not isinstance(global_values, Mapping):
        raise ValueError("pressure telemetry lacks a global high-water record")
    high = _validate_pressure_values(
        global_values,
        label="pressure.global",
        sample_count=sample_count,
        page_geometry=False,
    )
    amplification = global_values.get(
        "high_water_retention_amplification_milli"
    )
    if amplification is not None:
        amplification = _nonnegative_int(
            "pressure high-water retention amplification", amplification
        )
        if amplification < 1000:
            raise ValueError("pressure retention amplification is below one")
    if require_activity and (
        maximum == 0
        or high["high_water_consumed_bytes"] == 0
        or high["high_water_resident_data_bytes"] == 0
        or high["high_water_semantic_live_bytes"] == 0
        or amplification is None
    ):
        raise ValueError("pressure telemetry did not observe request activity")
    classes = pressure.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("pressure telemetry lacks per-class high waters")
    expected = tuple(expected_classes or ())
    if expected and len(classes) != len(expected):
        raise ValueError("pressure class cardinality differs from manager arenas")
    class_ids: set[int] = set()
    class_values: list[dict[str, int]] = []
    for index, raw_class in enumerate(classes):
        if not isinstance(raw_class, Mapping):
            raise ValueError(f"pressure class {index} is not a mapping")
        expected_keys = {"class_id", "name"} | {
            *_PRESSURE_CURRENT_FIELDS,
            *_PRESSURE_WATER_FIELDS,
            "retention_amplification_milli",
            "high_water_retention_amplification_milli",
            "sample_count",
        }
        if set(raw_class) != expected_keys:
            raise ValueError(f"pressure class {index} has a noncanonical field set")
        class_id = _nonnegative_int(
            f"pressure class {index}.class_id", raw_class.get("class_id")
        )
        name = raw_class.get("name")
        if not isinstance(name, str) or not name or class_id in class_ids:
            raise ValueError("pressure class identities are invalid or duplicated")
        class_ids.add(class_id)
        if expected:
            arena = expected[index]
            if (
                class_id != arena.get("class_id")
                or raw_class.get("capacity_pages") != arena.get("page_count")
                or (
                    arena.get("name") is not None
                    and name != arena.get("name")
                )
            ):
                raise ValueError(
                    "pressure class identity/capacity differs from manager arena"
                )
        values = _validate_pressure_values(
            {
                name: value
                for name, value in raw_class.items()
                if name not in ("class_id", "name")
            },
            label=f"pressure.classes[{index}]",
            sample_count=sample_count,
            page_geometry=True,
        )
        class_values.append(values)
    if any(
        sum(item[name] for item in class_values) != high[name]
        for name in _PRESSURE_CURRENT_FIELDS
    ):
        raise ValueError("pressure global current values differ from class totals")
    for name in (
        "high_water_consumed_pages",
        "high_water_consumed_bytes",
        "high_water_resident_data_pages",
        "high_water_resident_data_bytes",
        "high_water_request_reachable_unique_pages",
        "high_water_request_reachable_unique_bytes",
        "high_water_semantic_live_bytes",
    ):
        class_highs = [item[name] for item in class_values]
        if not max(class_highs) <= high[name] <= sum(class_highs):
            raise ValueError(
                f"pressure global {name} is outside per-class high-water bounds"
            )
    for name in ("min_free_pages", "min_free_bytes"):
        if high[name] < sum(item[name] for item in class_values):
            raise ValueError(
                f"pressure global {name} is below per-class minimum bounds"
            )
    return {
        "max_active": maximum,
        "peak_consumed_bytes": high["high_water_consumed_bytes"],
        "peak_resident_bytes": high["high_water_resident_data_bytes"],
        "peak_request_reachable_unique_bytes": high[
            "high_water_request_reachable_unique_bytes"
        ],
        "peak_semantic_live_bytes": high["high_water_semantic_live_bytes"],
        "request_private_retention_amplification_milli": amplification,
        "request_private_retention_amplification_ratio": (
            None if amplification is None else amplification / 1000.0
        ),
        "retention_amplification_scope": "request_private_kv_only",
        "sample_count": sample_count,
    }


def _manager_namespace(state: Mapping[str, object]) -> Mapping[str, object]:
    if "internal_states" in state:
        worker_state = common._state(dict(state))
        manager = worker_state.get("orbitkv_manager")
    elif "orbitkv_manager" in state:
        manager = state.get("orbitkv_manager")
    elif "pressure" in state:
        manager = state
    else:
        manager = None
    if not isinstance(manager, Mapping):
        raise RuntimeError("internal-state readback lacks orbitkv_manager")
    return manager


def validate_private_pressure_state(
    state: Mapping[str, object],
    *,
    stage: str,
    request_count: int,
    require_activity: bool,
    manager_config: object | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate ABI, ownership, drain, and pressure evidence together."""

    manager = _manager_namespace(state)
    if manager.get("abi_version") != 8:
        raise RuntimeError(f"ABI8 manager readback is missing at {stage}")
    if (
        manager.get("fixed_state_byte_count") != 0
        or manager.get("fixed_state_descriptors") != []
        or "fixed_state" in manager
    ):
        raise RuntimeError(
            "capacity-pressure RA does not support fixed-state bytes"
        )
    if manager_config is not None:
        common.verify_worker_plan_readback(manager, manager_config, stage)
    expected_top_level = {
        "abi_version",
        "plan_fingerprint",
        "state_plan_fingerprint",
        "fixed_state_byte_count",
        "fixed_state_descriptors",
        "tree_cache_type",
        "identities",
        "arena_stats",
        "manager_stats",
        "swa_activity",
        "batch_counters",
        "pressure",
    }
    if manager_config is not None and set(manager) != expected_top_level:
        raise RuntimeError(f"manager top-level schema is invalid at {stage}")

    counters = manager.get("batch_counters")
    if not isinstance(counters, Mapping):
        raise RuntimeError(f"manager batch counters are missing at {stage}")
    for name in (
        *_PREFIX_AND_FORK_COUNTERS,
        *_FAILURE_COUNTERS,
        *_FIXED_STATE_COUNTERS,
    ):
        value = counters.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"manager counter {name} is missing at {stage}")
        if value != 0:
            raise RuntimeError(
                f"capacity-pressure request-private gate rejected {name}={value}"
            )

    stats = manager.get("manager_stats")
    if not isinstance(stats, Mapping):
        raise RuntimeError(f"manager census is missing at {stage}")
    drained_fields = (
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
    dirty: dict[str, object] = {}
    for name in drained_fields:
        value = stats.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"manager census field {name} is missing at {stage}")
        if value != 0:
            dirty[name] = value
    if dirty:
        raise RuntimeError(f"manager did not drain at {stage}: {dirty}")

    arenas = manager.get("arena_stats")
    if not isinstance(arenas, list) or not arenas:
        raise RuntimeError(f"manager arena census is missing at {stage}")
    for index, arena in enumerate(arenas):
        if not isinstance(arena, Mapping):
            raise RuntimeError(f"manager arena {index} is malformed at {stage}")
        page_count = arena.get("page_count")
        free_pages = arena.get("free_pages")
        prefix_refs = arena.get("prefix_page_refs")
        if (
            isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count <= 0
            or free_pages != page_count
            or prefix_refs != 0
        ):
            raise RuntimeError(
                f"manager arena {index} is not request-private and drained at {stage}"
            )
    if sum(
        arena["free_pages"]
        for arena in arenas
        if isinstance(arena, Mapping)
    ) != stats.get("free_pages"):
        raise RuntimeError(f"manager aggregate free pages differ at {stage}")

    pressure = manager.get("pressure")
    if not isinstance(pressure, Mapping):
        raise RuntimeError("internal-state readback lacks pressure telemetry")
    expected_pressure_classes: list[Mapping[str, object]] = list(arenas)
    if manager_config is not None:
        config_classes = tuple(getattr(manager_config, "classes", ()))
        if len(config_classes) != len(expected_pressure_classes):
            raise RuntimeError(
                f"manager plan/arena cardinality differs at {stage}"
            )
        expected_pressure_classes = [
            {**dict(arena), "name": config_class.name}
            for arena, config_class in zip(
                expected_pressure_classes, config_classes, strict=True
            )
        ]
    summary = _pressure_summary(
        pressure,
        request_count=request_count,
        require_activity=require_activity,
        expected_classes=expected_pressure_classes,
    )
    return dict(manager), summary


def _gpu_allocator_peak_unavailable() -> dict[str, object]:
    return {
        "status": "unavailable",
        "bytes": None,
        "source": "worker_internal_state",
        "reason": (
            "the pinned worker hook does not expose a reset-scoped allocator "
            "peak; process-external current-memory samples are not substituted"
        ),
    }


def capacity_pressure_record(
    *,
    arrival: Mapping[str, object],
    request_traces: Sequence[Mapping[str, object]],
    pressure: Mapping[str, object] | None,
    runner_capabilities: Mapping[str, bool],
    scope: str,
) -> dict[str, object]:
    """Build one strict JSON-safe diagnostic from observed intervals."""

    if scope not in ("host_schedule_only", "single_device_eager"):
        raise ValueError("unknown capacity-pressure benchmark scope")
    if scope == "single_device_eager" and pressure is None:
        raise ValueError("an engine run must include pressure telemetry")
    traces = [dict(item) for item in request_traces]
    intervals: list[tuple[float, float]] = []
    rids: set[str] = set()
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
        rid = trace.get("rid", trace.get("submitted_rid"))
        if not isinstance(rid, str) or not rid or rid in rids:
            raise ValueError("request traces require unique request ids")
        rids.add(rid)
        if scope == "single_device_eager":
            output_ids = trace.get("output_ids")
            if (
                not isinstance(output_ids, list)
                or not all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in output_ids
                )
            ):
                raise ValueError("engine request trace lacks output_ids")
            if trace.get("returned_rid") != rid:
                raise ValueError("engine request trace contains a foreign request id")
            if trace.get("cached_tokens") != 0:
                raise ValueError("engine request trace contains Prefix reuse")
    pressure_summary = None
    if scope == "single_device_eager":
        assert pressure is not None
        pressure_summary = _pressure_summary(
            pressure, request_count=len(traces), require_activity=True
        )
    if any(not isinstance(value, bool) for value in runner_capabilities.values()):
        raise ValueError("runner capabilities must be boolean")
    return {
        "schema": RECORD_SCHEMA,
        "scope": scope,
        "executed": scope == "single_device_eager",
        "diagnostic_only": True,
        "qualification_scope": "diagnostic_only",
        "performance_go": False,
        "performance_go_reason": (
            "one diagnostic run without independent repeated matched controls"
        ),
        "claim_boundary": [
            "request-private KV state only",
            "shared Prefix and request fork retention amplification are rejected",
            "fixed-state retention amplification is rejected",
            "host_schedule_only records are not SGLang or GPU evidence",
            "single-run telemetry does not establish a memory-saving claim",
        ],
        "runner_capabilities": dict(runner_capabilities),
        "arrival": dict(arrival),
        "request_traces": traces,
        "concurrency": sweep_line_concurrency(intervals).as_dict(),
        "pressure": None if pressure is None else dict(pressure),
        "pressure_summary": pressure_summary,
        "gpu_allocator_peak": _gpu_allocator_peak_unavailable(),
    }


def _request_values(
    requests: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    values = tuple(dict(item) for item in requests)
    if not values:
        raise ValueError("requests must be nonempty")
    seen: set[str] = set()
    for index, request in enumerate(values):
        rid = request.setdefault("rid", f"pressure-{index}")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("requests require unique nonempty rid values")
        seen.add(rid)
        input_ids = request.get("input_ids")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in input_ids
            )
        ):
            raise ValueError("each pressure request requires integer input_ids")
        if request.get("stream", False) is not False:
            raise ValueError("capacity-pressure requests must be non-streaming")
        if (
            request.get("session_id") is not None
            or request.get("session_params") is not None
        ):
            raise ValueError("capacity-pressure requests forbid session/fork state")
    return values


def _trace_output(
    output: object, *, rid: str, expected_output_tokens: int | None
) -> tuple[dict[str, object], Mapping[str, object]]:
    if not isinstance(output, Mapping):
        raise RuntimeError("SGLang async_generate returned a non-mapping output")
    output_ids = output.get("output_ids")
    if (
        not isinstance(output_ids, list)
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in output_ids
        )
    ):
        raise RuntimeError("SGLang async_generate output_ids are missing or invalid")
    if expected_output_tokens is not None and len(output_ids) != expected_output_tokens:
        raise RuntimeError("SGLang async_generate returned the wrong token count")
    meta = output.get("meta_info")
    if not isinstance(meta, Mapping) or meta.get("id") != rid:
        raise RuntimeError("SGLang async_generate returned a foreign request id")
    cached_tokens = meta.get("cached_tokens")
    if cached_tokens != 0:
        raise RuntimeError("SGLang async_generate unexpectedly reused Prefix KV")
    return (
        {
            "returned_rid": rid,
            "cached_tokens": 0,
            "output_ids": list(output_ids),
            "output_ids_sha256": _canonical_digest(output_ids),
        },
        output,
    )


async def run_async_arrivals(
    engine: Any,
    requests: Sequence[Mapping[str, object]],
    offsets_seconds: Sequence[float],
    *,
    expected_output_tokens: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[tuple[dict[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Drive absolute-deadline arrivals through ``Engine.async_generate``."""

    generate = getattr(engine, "async_generate", None)
    if not callable(generate):
        raise RuntimeError("SGLang Engine lacks async_generate capability")
    values = _request_values(requests)
    offsets = tuple(
        _finite_nonnegative("arrival offset", value)
        for value in offsets_seconds
    )
    if len(values) != len(offsets):
        raise ValueError("request and arrival cardinalities must match")
    if offsets[0] != 0.0 or any(
        right < left for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("arrival offsets must start at zero and be monotonic")
    if expected_output_tokens is not None:
        _positive_int("expected output tokens", expected_output_tokens)

    start = clock()

    async def one(
        index: int, request: dict[str, object]
    ) -> tuple[dict[str, object], Mapping[str, object]]:
        target = start + offsets[index]
        delay = target - clock()
        if delay > 0:
            await sleep(delay)
        submitted = clock() - start
        rid = request["rid"]
        assert isinstance(rid, str)
        result = generate(**request)
        if not inspect.isawaitable(result):
            raise RuntimeError("SGLang async_generate did not return an awaitable")
        output = await result
        completed = clock() - start
        output_trace, normalized = _trace_output(
            output, rid=rid, expected_output_tokens=expected_output_tokens
        )
        trace: dict[str, object] = {
            "request_index": index,
            "rid": rid,
            "submitted_rid": rid,
            "scheduled_arrival_seconds": offsets[index],
            "submitted_seconds": submitted,
            "completed_seconds": completed,
            "latency_seconds": completed - submitted,
            "arrival_lateness_seconds": max(0.0, submitted - offsets[index]),
            "timestamps_seconds": {
                "scheduled_arrival": offsets[index],
                "submitted": submitted,
                "completed": completed,
            },
            "submitted_input_ids_sha256": _canonical_digest(
                request["input_ids"]
            ),
            "success": True,
            **output_trace,
        }
        return trace, normalized

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
    internal_state: Callable[
        [], Mapping[str, object] | Awaitable[Mapping[str, object]]
    ] | None,
    expected_output_tokens: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    """Host-testable async seam with strict post-workload readback."""

    if not callable(internal_state):
        raise RuntimeError(
            "capacity-pressure E2E requires an internal-state readback capability"
        )
    traces, outputs = await run_async_arrivals(
        engine,
        requests,
        offsets_seconds,
        expected_output_tokens=expected_output_tokens,
        clock=clock,
        sleep=sleep,
    )
    state_result = internal_state()
    state = await state_result if inspect.isawaitable(state_result) else state_result
    if not isinstance(state, Mapping):
        raise RuntimeError("internal-state readback is not a mapping")
    manager, _ = validate_private_pressure_state(
        state,
        stage="after_workload",
        request_count=len(requests),
        require_activity=True,
    )
    pressure = manager["pressure"]
    assert isinstance(pressure, Mapping)
    record = capacity_pressure_record(
        arrival=arrival,
        request_traces=traces,
        pressure=pressure,
        runner_capabilities={
            "async_generate": True,
            "scheduled_arrivals": True,
            "interval_concurrency": True,
            "pressure_internal_state": True,
            "request_private_ra": True,
            "worker_gpu_allocator_peak": False,
        },
        scope="single_device_eager",
    )
    return record, outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one capability-gated request-private SGLang async pressure "
            "diagnostic. Dimensions are explicit so the same runner supports "
            "small H20 smoke and long-context workloads."
        )
    )
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument(
        "--reclamation-mode", choices=("naive", "relocate"), required=True
    )
    parser.add_argument("--reclamation-trigger-tokens", type=int, required=True)
    parser.add_argument("--retained-per-page", type=int, required=True)
    parser.add_argument(
        "--fragmentation-threshold-milli", type=int, default=500
    )
    parser.add_argument("--maximum-source-pages", type=int, required=True)
    parser.add_argument("--evacuation-headroom-pages", type=int, required=True)
    parser.add_argument(
        "--arrival", choices=("burst", "poisson"), default="poisson"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="requests/second; required for Poisson arrivals",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--attention-backend",
        choices=tuple(
            dict.fromkeys(common.ATTENTION_BACKENDS_BY_ARCHITECTURE.values())
        ),
        required=True,
    )
    parser.add_argument(
        "--fp8-gemm-backend", choices=common.FP8_GEMM_BACKENDS, default=None
    )
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def validate_arguments(args: argparse.Namespace) -> dict[str, Path | None]:
    for name in (
        "requests",
        "max_running_requests",
        "prompt_tokens",
        "decode_tokens",
        "chunked_prefill_size",
        "context_length",
        "max_total_tokens",
        "reclamation_trigger_tokens",
        "retained_per_page",
        "maximum_source_pages",
        "evacuation_headroom_pages",
    ):
        _positive_int(f"--{name.replace('_', '-')}", getattr(args, name))
    if args.reclamation_mode not in ("naive", "relocate"):
        raise ValueError("--reclamation-mode must be naive or relocate")
    if args.max_running_requests > args.requests:
        raise ValueError("--max-running-requests cannot exceed --requests")
    if args.prompt_tokens + args.decode_tokens >= args.context_length:
        raise ValueError(
            "prompt plus decode tokens must leave one context slot unused"
        )
    if args.reclamation_trigger_tokens != args.prompt_tokens:
        raise ValueError(
            "--reclamation-trigger-tokens must equal --prompt-tokens so the "
            "first decode step reaches the boundary exactly"
        )
    if args.chunked_prefill_size % common.PAGE_TOKENS:
        raise ValueError("--chunked-prefill-size must be page aligned")
    if args.max_total_tokens % common.PAGE_TOKENS:
        raise ValueError("--max-total-tokens must be page aligned")
    required_prefill_tokens = args.max_running_requests * args.prompt_tokens
    if required_prefill_tokens > args.max_total_tokens:
        raise ValueError(
            "--max-total-tokens cannot hold every configured request prefill"
        )
    if args.chunked_prefill_size > args.max_total_tokens:
        raise ValueError("--chunked-prefill-size exceeds --max-total-tokens")
    if args.retained_per_page >= common.PAGE_TOKENS:
        raise ValueError("--retained-per-page must be below page size 16")
    if (
        isinstance(args.fragmentation_threshold_milli, bool)
        or not isinstance(args.fragmentation_threshold_milli, int)
        or not 0 <= args.fragmentation_threshold_milli <= 1000
    ):
        raise ValueError("--fragmentation-threshold-milli must be in [0, 1000]")
    source_pages = (
        args.reclamation_trigger_tokens + common.PAGE_TOKENS - 1
    ) // common.PAGE_TOKENS
    retained_tokens = (
        args.reclamation_trigger_tokens // common.PAGE_TOKENS
        * args.retained_per_page
        + min(
            args.reclamation_trigger_tokens % common.PAGE_TOKENS,
            args.retained_per_page,
        )
    )
    retained_pages = (retained_tokens + common.PAGE_TOKENS - 1) // common.PAGE_TOKENS
    slots = source_pages * common.PAGE_TOKENS
    expected_fragmentation = (slots - retained_tokens) * 1000 // slots
    if args.reclamation_mode == "relocate":
        if source_pages > args.maximum_source_pages:
            raise ValueError(
                "--maximum-source-pages is below the reclamation trigger "
                "footprint"
            )
        if retained_pages > args.evacuation_headroom_pages:
            raise ValueError(
                "--evacuation-headroom-pages is below the retained footprint"
            )
        if source_pages <= retained_pages:
            raise ValueError("relocate policy must project a positive page gain")
        if args.fragmentation_threshold_milli > expected_fragmentation:
            raise ValueError(
                "--fragmentation-threshold-milli exceeds expected "
                f"fragmentation {expected_fragmentation}"
            )
    if (
        args.reclamation_mode == "naive"
        and args.attention_backend != "flashinfer"
    ):
        raise ValueError(
            "naive reclamation requires the token-indexed FlashInfer backend"
        )
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.mem_fraction_static is not None and not (
        0.0 < args.mem_fraction_static <= 1.0
    ):
        raise ValueError("--mem-fraction-static must be in (0, 1]")
    if args.attention_backend not in set(
        common.ATTENTION_BACKENDS_BY_ARCHITECTURE.values()
    ):
        raise ValueError("--attention-backend is unsupported")
    fp8_backend = getattr(args, "fp8_gemm_backend", None)
    if fp8_backend is not None and fp8_backend not in common.FP8_GEMM_BACKENDS:
        raise ValueError("--fp8-gemm-backend is unsupported")
    arrival_schedule(
        args.requests,
        kind=args.arrival,
        request_rate=args.request_rate,
        seed=args.seed,
    )

    sglang_root = common._directory(args.sglang_root, "--sglang-root")
    if not (sglang_root / "python/sglang/__init__.py").is_file():
        raise ValueError("--sglang-root is not an SGLang source checkout")
    model = common._directory(args.model, "--model")
    common._regular_file(str(model / "config.json"), "checkpoint config")
    return {
        "sglang_root": sglang_root,
        "model": model,
        "plan": common._regular_file(args.plan, "--plan"),
        "state_plan": None,
        "library": common._regular_file(args.library, "--library"),
    }


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mode="manager",
        attention_backend=args.attention_backend,
        fp8_gemm_backend=getattr(args, "fp8_gemm_backend", None),
        context_length=args.context_length,
        chunked_prefill_size=args.chunked_prefill_size,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        mem_fraction_static=args.mem_fraction_static,
        seed=args.seed,
    )


def _execution_contract(
    contract: Mapping[str, object], attention_backend: str
) -> dict[str, object]:
    values = dict(contract)
    backend_profile = dict(values.get("backend_profile", {}))
    backend_profile["attention_backend"] = attention_backend
    values["attention_backend"] = attention_backend
    values["backend_profile"] = backend_profile
    values["workload_profile"] = "fresh_prompt"
    values["state_ownership"] = "request_private"
    values["qualification_scope"] = "diagnostic_only"
    return values


def _reclamation_policy(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.reclamation_mode,
        "trigger_tokens": args.reclamation_trigger_tokens,
        "retained_per_page": args.retained_per_page,
        "policy_id": _POLICY_ID,
        "policy_version": 1,
        "quality_contract": 1,
        "fragmentation_threshold_milli": (
            args.fragmentation_threshold_milli
        ),
        "maximum_source_pages": args.maximum_source_pages,
        "evacuation_headroom_pages": args.evacuation_headroom_pages,
    }


def _verify_reclamation_config(
    manager_config: object, policy: Mapping[str, object]
) -> None:
    configured = getattr(manager_config, "token_reclamation", None)
    actual = {
        name: getattr(configured, name, None)
        for name in policy
    }
    if actual != dict(policy) or actual.get("mode") == "off":
        raise RuntimeError(
            "loaded manager token-reclamation policy differs from the "
            "request-private pressure workload"
        )


def _validate_reclamation_activity(
    counters: Mapping[str, object], mode: str
) -> dict[str, int]:
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
    values = _nonnegative_record(counters, names, "manager.batch_counters")
    if values["token_disposition_batches"] <= 0:
        raise RuntimeError("pressure workload observed no token disposition batch")
    if values["token_policy_evictions"] <= 0:
        raise RuntimeError("pressure workload observed no token policy evictions")
    if values["abort_relocations_batch_calls"] != 0:
        raise RuntimeError("pressure workload observed aborted relocation work")
    relocation = (
        "relocation_batches",
        "relocation_moves",
        "relocation_reclaimed_pages",
        "relocation_copy_events",
        "relocation_copy_tokens",
        "prepare_relocation_batch_calls",
        "submit_relocation_batch_calls",
        "complete_relocation_batch_calls",
    )
    if mode == "naive":
        if any(values[name] for name in relocation):
            raise RuntimeError("naive pressure workload reported relocation activity")
    elif mode == "relocate":
        if any(values[name] <= 0 for name in relocation):
            raise RuntimeError("relocate pressure workload did not execute relocation")
        batches = values["relocation_batches"]
        if (
            values["token_disposition_batches"] != batches
            or values["relocation_copy_events"] != batches
            or values["prepare_relocation_batch_calls"] != batches
            or values["submit_relocation_batch_calls"] != batches
            or values["complete_relocation_batch_calls"] != batches
            or values["relocation_moves"]
            != values["relocation_copy_tokens"]
        ):
            raise RuntimeError(
                "relocate pressure workload counters are internally inconsistent"
            )
    else:
        raise RuntimeError("unknown pressure reclamation mode")
    return values


def _reject_fixed_state(config: object, contract: Mapping[str, object]) -> None:
    fixed = tuple(getattr(config, "fixed_states", ()))
    fixed_bytes = getattr(config, "fixed_state_byte_count", None)
    contract_fixed = contract.get("fixed_states")
    if fixed or fixed_bytes != 0 or contract_fixed not in ([], ()):
        raise RuntimeError(
            "capacity-pressure rejects fixed-state RA; use an attention-only plan"
        )


def _engine_loop(engine: object) -> asyncio.AbstractEventLoop:
    loop = getattr(engine, "loop", None)
    if not isinstance(loop, asyncio.AbstractEventLoop):
        raise RuntimeError("SGLang Engine lacks its pinned asynchronous event loop")
    if loop.is_closed() or loop.is_running():
        raise RuntimeError("SGLang Engine event loop is unavailable for the workload")
    return loop


def _manager_without_pressure(manager: Mapping[str, object]) -> dict[str, object]:
    return {name: value for name, value in manager.items() if name != "pressure"}


def run(
    args: argparse.Namespace, paths: dict[str, Path | None]
) -> dict[str, object]:
    run_started = time.perf_counter()
    base = _base_args(args)
    environment = common.configure_environment(base, paths)
    os.environ["ORBITKV_PRESSURE_TELEMETRY"] = "1"
    environment["ORBITKV_PRESSURE_TELEMETRY"] = "1"
    policy = _reclamation_policy(args)
    encoded_policy = json.dumps(
        policy, sort_keys=True, separators=(",", ":")
    )
    os.environ["ORBITKV_TOKEN_RECLAMATION"] = encoded_policy
    environment["ORBITKV_TOKEN_RECLAMATION"] = encoded_policy

    source = common.verify_sglang_source(paths["sglang_root"], "manager")
    source["plugin_selection"] = common.verify_manager_entrypoint()
    source["pinned_contract"] = common.verify_pinned_module_constants()
    source["harness_sha256"] = common.sha256_file(Path(__file__).resolve())
    source["adapter"] = common._adapter_identity()
    source["library"] = common.artifact_identity(paths["library"])

    from orbitkv_sglang.config import load_config

    manager_config = load_config()
    _verify_reclamation_config(manager_config, policy)
    source["plan"] = common.manager_plan_identity(manager_config)
    raw_contract, checkpoint = common.checkpoint_contract(
        paths["model"], manager_config
    )
    _reject_fixed_state(manager_config, raw_contract)
    contract = _execution_contract(raw_contract, args.attention_backend)
    if args.context_length > contract["max_position_embeddings"]:
        raise RuntimeError("--context-length exceeds the checkpoint limit")

    prompts = common.fresh_input_ids(
        requests=args.requests,
        prompt_tokens=args.prompt_tokens,
        vocab_size=contract["vocab_size"],
        seed=args.seed,
        iteration=0,
        forbidden_token_ids=tuple(contract["control_token_ids"].values()),
        token_upper_bound=contract["prompt_token_upper_bound"],
    )
    sampling = {
        "temperature": 0,
        "max_new_tokens": args.decode_tokens,
        "min_new_tokens": args.decode_tokens,
        "ignore_eos": True,
        "sampling_seed": args.seed,
    }
    requests = tuple(
        {
            "input_ids": list(prompt),
            "rid": f"orbitkv-pressure-{args.seed}-{index}",
            "sampling_params": dict(sampling),
        }
        for index, prompt in enumerate(prompts)
    )
    offsets = arrival_schedule(
        args.requests,
        kind=args.arrival,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    arrival = {
        "kind": args.arrival,
        "request_rate_per_second": (
            None
            if args.request_rate is None or args.request_rate == math.inf
            else args.request_rate
        ),
        "seed": args.seed,
        "scheduled_offsets_seconds": list(offsets),
        "clock": "time.perf_counter",
        "deadline_semantics": "absolute_from_workload_start",
    }
    engine_args = common.engine_arguments(
        base,
        paths["model"],
        contract,
        diagnostic_attention_backend=args.attention_backend,
    )

    before_engine = common.gpu_snapshot("before_engine")
    import sglang as sgl

    started_at = datetime.now(timezone.utc).isoformat()
    load_started = time.perf_counter()
    with sgl.Engine(**engine_args) as engine:
        load_seconds = time.perf_counter() - load_started
        after_load = common.gpu_snapshot("after_load")
        if not callable(getattr(engine, "async_generate", None)):
            raise RuntimeError("SGLang Engine lacks async_generate capability")
        if not callable(getattr(engine, "get_server_info", None)):
            raise RuntimeError("SGLang Engine lacks internal-state readback")
        loop = _engine_loop(engine)

        info_load = engine.get_server_info()
        if not isinstance(info_load, dict):
            raise RuntimeError("SGLang get_server_info returned a non-mapping")
        capacities = common.verify_runtime_contract(
            base,
            info_load,
            contract,
            diagnostic_attention_backend=args.attention_backend,
        )
        manager_load, load_pressure = validate_private_pressure_state(
            info_load,
            stage="after_load",
            request_count=args.requests,
            require_activity=False,
            manager_config=manager_config,
        )

        traces, outputs = loop.run_until_complete(
            run_async_arrivals(
                engine,
                requests,
                offsets,
                expected_output_tokens=args.decode_tokens,
            )
        )
        info_final = engine.get_server_info()
        if not isinstance(info_final, dict):
            raise RuntimeError("SGLang get_server_info returned a non-mapping")
        common.verify_runtime_contract(
            base,
            info_final,
            contract,
            diagnostic_attention_backend=args.attention_backend,
        )
        manager_final, pressure_summary = validate_private_pressure_state(
            info_final,
            stage="after_workload",
            request_count=args.requests,
            require_activity=True,
            manager_config=manager_config,
        )
        if pressure_summary["max_active"] > args.max_running_requests:
            raise RuntimeError(
                "worker pressure exceeded configured max-running-requests"
            )
        if pressure_summary["sample_count"] <= load_pressure["sample_count"]:
            raise RuntimeError("worker pressure sample count did not advance")
        reclamation_activity = _validate_reclamation_activity(
            manager_final["batch_counters"], args.reclamation_mode
        )
        after_workload = common.gpu_snapshot("after_workload")
    after_shutdown = common.gpu_snapshot("after_shutdown")

    pressure = manager_final["pressure"]
    assert isinstance(pressure, Mapping)
    record = capacity_pressure_record(
        arrival=arrival,
        request_traces=traces,
        pressure=pressure,
        runner_capabilities={
            "pinned_sglang_source": True,
            "canonical_manager_entrypoint": True,
            "async_generate": True,
            "scheduled_arrivals": True,
            "interval_concurrency": True,
            "abi8_pressure_internal_state": True,
            "request_private_ra": True,
            "prefix_ra": False,
            "request_fork_ra": False,
            "fixed_state_ra": False,
            "worker_gpu_allocator_peak": False,
        },
        scope="single_device_eager",
    )
    record.update(
        {
            "started_at_utc": started_at,
            "command": [
                sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]
            ],
            "environment": environment,
            "source_identity": source,
            "runtime_identity": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "sglang_version": sgl.__version__,
                "execution_profile": "sglang_offline_engine_eager_tp1",
            },
            "checkpoint": checkpoint,
            "checkpoint_contract": contract,
            "engine_args": engine_args,
            "capacity": capacities,
            "sampling_params": sampling,
            "workload": {
                "requests": args.requests,
                "max_running_requests": args.max_running_requests,
                "prompt_tokens": args.prompt_tokens,
                "decode_tokens": args.decode_tokens,
                "context_length": args.context_length,
                "max_total_tokens": args.max_total_tokens,
                "chunked_prefill_size": args.chunked_prefill_size,
                "seed": args.seed,
                "state_ownership": "request_private",
                "token_reclamation": policy,
                "observed_reclamation_activity": reclamation_activity,
                "input_token_digests_sha256": [
                    _canonical_digest(prompt) for prompt in prompts
                ],
            },
            "load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - run_started,
            "gpu_snapshots": [
                before_engine,
                after_load,
                after_workload,
                after_shutdown,
            ],
            "gpu_snapshot_scope": (
                "device-wide point observations for provenance only; not "
                "allocator peaks"
            ),
            "request_output_ids": [
                list(output["output_ids"]) for output in outputs
            ],
            "output_token_digest_sha256": _canonical_digest(
                [output["output_ids"] for output in outputs]
            ),
            "manager": {
                "after_load": _manager_without_pressure(manager_load),
                "after_workload": _manager_without_pressure(manager_final),
            },
        }
    )
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = validate_arguments(args)
        result = run(args, paths)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
