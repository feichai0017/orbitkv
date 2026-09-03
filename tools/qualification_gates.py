"""Independent gate derivation for current qualification evidence."""

from __future__ import annotations

import math
import random
import re
import statistics
from typing import Any, Mapping, Sequence


_CAPACITY_ALLOW = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bkv cache pool is full\b",
        r"\bprefill out of memory\b",
        r"\bdecode out of memory\b",
        r"\bout of memory even after retracting all other requests\b",
        r"\btry to allocate [0-9]+ tokens\b",
    )
)
_CAPACITY_BLOCK = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"cuda out of memory",
        r"cublas.*alloc",
        r"hip out of memory",
        r"allocator.*device memory",
    )
)


def gate(qualified: bool, *reasons: str) -> dict[str, Any]:
    """Return one total gate result; every false result explains itself."""

    if qualified:
        return {"qualified": True, "reasons": []}
    filtered = [reason for reason in reasons if isinstance(reason, str) and reason]
    return {
        "qualified": False,
        "reasons": filtered or ["qualification evidence is incomplete"],
    }


def classify_capacity_message(message: Any) -> str:
    """Classify an untrusted raw message as allow, block, or unknown."""

    if not isinstance(message, str) or not message.strip():
        return "unknown"
    if any(pattern.search(message) for pattern in _CAPACITY_BLOCK):
        return "block"
    if any(pattern.search(message) for pattern in _CAPACITY_ALLOW):
        return "allow"
    return "unknown"


def is_capacity_exhaustion(failure: Any) -> bool:
    """Classify raw text independently of its untrusted type label."""

    return (
        isinstance(failure, dict)
        and set(failure) == {"type", "message"}
        and classify_capacity_message(failure.get("message")) == "allow"
    )


def _record_outputs(record: Mapping[str, Any]) -> list[list[Mapping[str, Any]]]:
    return [row["requests"] for row in record["outputs"]["iterations"]]


def _stable_outputs(record: Mapping[str, Any]) -> bool:
    rows = _record_outputs(record)
    if not rows:
        return False
    canonical = [
        (request["input_sha256"], request["output_ids"])
        for request in rows[0]
    ]
    return all(
        [(request["input_sha256"], request["output_ids"]) for request in row]
        == canonical
        for row in rows[1:]
    )


def _pair_outputs_equal(
    stock: Mapping[str, Any], manager: Mapping[str, Any]
) -> bool:
    stock_rows = _record_outputs(stock)
    manager_rows = _record_outputs(manager)
    if len(stock_rows) != len(manager_rows):
        return False
    for stock_row, manager_row in zip(stock_rows, manager_rows, strict=True):
        if len(stock_row) != len(manager_row):
            return False
        for stock_request, manager_request in zip(
            stock_row, manager_row, strict=True
        ):
            if (
                stock_request["input_sha256"]
                != manager_request["input_sha256"]
                or stock_request["output_ids"]
                != manager_request["output_ids"]
            ):
                return False
    return True


def _manager_stream_gate(manager: Mapping[str, Any]) -> dict[str, Any]:
    after_load, after_workload, final = manager["manager"]["snapshots"]
    initial = after_load["batch_counters"]
    active = after_workload["batch_counters"]
    delta = {name: active[name] - initial[name] for name in active}
    workload = manager["workload"]
    workload_batches = workload["iterations"] * workload["decode_tokens"]
    reasons: list[str] = []
    failures = (
        "retryable_conflicts",
        "fail_stops",
        "fail_stop_count",
        "quarantine_count",
        "quarantine_steps_batch_calls",
        "quarantine_submissions_batch_calls",
    )
    if any(delta[name] != 0 for name in failures):
        reasons.append("workload failure or quarantine counter delta is nonzero")
    lifecycle = (
        "prepare_batch_calls",
        "submit_batch_calls",
        "complete_batch_calls",
        "forward_events",
        "completion_values",
    )
    if any(delta[name] != workload_batches for name in lifecycle):
        reasons.append("after-load lifecycle deltas do not match the workload")
    if delta["event_queries"] + delta["event_waits"] < delta["forward_events"]:
        reasons.append("workload completion events were not all observed")
    crossings = (
        workload["chunk_geometry"]["epoch_end_crossings_per_request"]
        * workload["requests"]
        * workload["iterations"]
    )
    if delta["acknowledge_reclamations_batch_calls"] < crossings:
        reasons.append("workload retirement acknowledgements are incomplete")
    # Aggregate snapshots cannot establish reuse in every iteration.  Do not
    # infer it from total logical pages across independent fresh requests.
    reasons.append("no per-iteration physical page reuse trace was recorded")
    pressure = final["pressure"]
    if pressure["enabled"] is not True:
        reasons.append("event-driven pressure evidence is disabled")
    else:
        for event in ("step_completed", "request_released"):
            if pressure["event_counts"].get(event, 0) <= 0:
                reasons.append(f"pressure event {event} was not observed")
    drain_fields = (
        "active_requests", "active_snapshots", "active_prefixes",
        "prepared_steps", "submitted_steps", "reserved_pages",
        "writing_pages", "active_pages", "retiring_pages",
        "quarantined_pages", "pending_reclamations",
        "total_request_page_refs", "total_prefix_page_refs",
        "total_reader_pins",
    )
    final_stats = final["manager_stats"]
    final_arena = final["arena_stats"][0]
    if any(final_stats[name] != 0 for name in drain_fields) or (
        final_arena["free_pages"] != final_arena["page_count"]
    ):
        reasons.append("manager state did not fully drain")
    return gate(not reasons, *reasons)


_SWA_COUNTERS = (
    "swa_retirement_certificates",
    "swa_pages_reclaimed",
    "swa_wrap_events",
    "swa_page_reuse_events",
)


def _frontier(snapshot: Mapping[str, Any]) -> dict[int, int]:
    return {
        point["domain"]: point["value"]
        for point in snapshot["completion_evidence"]["completion_high_water"]
    }


def _native_manager_stream_gate(manager: Mapping[str, Any]) -> dict[str, Any]:
    after_load, after_workload, final = manager["manager"]["snapshots"]
    initial = after_load["batch_counters"]
    active = after_workload["batch_counters"]
    delta = {name: active[name] - initial[name] for name in active}
    reasons: list[str] = []
    if delta["fail_stop_count"] != 0:
        reasons.append("workload fail-stop counter delta is nonzero")
    if (
        delta["forward_events"] <= 0
        or delta["completion_values"] <= 0
        or delta["forward_events"] != delta["completion_values"]
    ):
        reasons.append("runtime-session forward/completion lifecycle did not advance")
    if delta["event_queries"] + delta["event_waits"] <= 0:
        reasons.append("runtime-session completion events were not observed")

    before_frontier = _frontier(after_load)
    workload_frontier = _frontier(after_workload)
    if not any(
        value > before_frontier.get(domain, 0)
        for domain, value in workload_frontier.items()
    ):
        reasons.append("completion frontier did not advance during the workload")

    before_swa = after_load["swa_activity"]
    workload_swa = after_workload["swa_activity"]
    for snapshot in (after_load, after_workload, final):
        activity = snapshot["swa_activity"]
        if (
            activity["status"] != "exposed"
            or activity["applicable"] is not True
            or activity["source"] != "native_runtime_session"
            or activity["derived"] is not False
        ):
            reasons.append("SWA activity is not direct native-session evidence")
            break
    stalled = [
        name
        for name in _SWA_COUNTERS
        if workload_swa[name] - before_swa[name] <= 0
    ]
    if stalled:
        reasons.append(
            "Sliding workload did not advance all direct SWA counters: "
            + ", ".join(stalled)
        )

    drain_fields = (
        "active_requests", "active_snapshots", "active_prefixes",
        "prepared_steps", "submitted_steps", "reserved_pages",
        "writing_pages", "active_pages", "retiring_pages",
        "quarantined_pages", "pending_reclamations",
        "total_request_page_refs", "total_prefix_page_refs",
        "total_reader_pins",
    )
    if any(final["manager_stats"][name] != 0 for name in drain_fields) or any(
        arena["free_pages"] != arena["page_count"]
        for arena in final["arena_stats"]
    ):
        reasons.append("native-session manager state did not fully drain")
    return gate(not reasons, *reasons)


def pair_gates(
    stock: Mapping[str, Any], manager: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Independently derive all four pair-level gates from raw records."""

    case = stock["workload"]["case"]
    stock_failed = stock["server_capacity"]["status"] == "failed"
    manager_failed = manager["server_capacity"]["status"] == "failed"
    if case == "roomy":
        native = (
            manager.get("schema")
            == "orbitkv.sglang-native-session-single-run.v1"
        )
        correct = (
            not stock_failed
            and not manager_failed
            and (native or _stable_outputs(stock))
            and (native or _stable_outputs(manager))
            and _pair_outputs_equal(stock, manager)
        )
        correctness = gate(
            correct,
            "roomy records must succeed, repeat stably, and have exact equal outputs",
        )
        stream = (
            (
                _native_manager_stream_gate(manager)
                if manager.get("schema")
                == "orbitkv.sglang-native-session-single-run.v1"
                else _manager_stream_gate(manager)
            )
            if not manager_failed
            else gate(False, "manager roomy execution failed")
        )
        capacity = gate(False, "roomy records alone do not qualify capacity")
    else:
        correctness = gate(
            False,
            "exact-floor capacity records do not replace roomy correctness evidence",
        )
        stream = gate(
            False,
            "exact-floor capacity records do not replace roomy stream evidence",
        )
        raw_exhaustion = stock_failed and is_capacity_exhaustion(
            stock["server_capacity"].get("failure")
        )
        native = (
            manager.get("schema")
            == "orbitkv.sglang-native-session-single-run.v1"
        )
        native_capacity = True
        if native and not manager_failed:
            manager_capacity = manager["server_capacity"]
            stock_capacity = stock["server_capacity"]
            floor = manager_capacity.get("floor")
            classes = manager_capacity.get("class_capacities")
            native_capacity = (
                isinstance(floor, dict)
                and isinstance(classes, dict)
                and classes
                == {
                    "full_tokens": floor.get("expected_full_tokens"),
                    "sliding_tokens": floor.get("expected_sliding_tokens"),
                }
                and stock_capacity.get("class_capacities") == classes
                and floor.get("configured_max_total_tokens")
                == manager_capacity.get("requested_tokens")
            )
        capacity = gate(
            not manager_failed and raw_exhaustion and native_capacity,
            (
                "capacity requires manager success, exact native class floors, "
                "and an allowed raw stock failure message"
                if native
                else "capacity requires manager success and an allowed raw stock failure message"
            ),
        )
    return {
        "correctness_qualified": correctness,
        "stream_event_qualified": stream,
        "capacity_qualified": capacity,
        "throughput_go": gate(
            False, "throughput requires a multi-epoch summary"
        ),
    }


def paired_regressions(
    stock: Mapping[str, Any], manager: Mapping[str, Any]
) -> list[float]:
    stock_values = stock["timings"]["iteration_seconds"]
    manager_values = manager["timings"]["iteration_seconds"]
    if len(stock_values) != len(manager_values):
        raise RuntimeError("roomy pair timing cardinalities differ")
    values = [
        1.0 - float(stock_value) / float(manager_value)
        for stock_value, manager_value in zip(
            stock_values, manager_values, strict=True
        )
    ]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("paired regression is non-finite")
    return values


def bootstrap_upper(
    values: Sequence[float], *, confidence: float, resamples: int, seed: int
) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("cannot bootstrap invalid paired samples")
    rng = random.Random(seed)
    count = len(values)
    medians = sorted(
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    index = min(len(medians) - 1, math.ceil(confidence * len(medians)) - 1)
    return float(medians[index])


def summary_gates(
    pairs: Sequence[dict[str, Any]],
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    thresholds: Mapping[str, int | float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Independently derive all summary statistics and gates."""

    roomy = [
        (pair, raw)
        for pair, raw in zip(pairs, records, strict=True)
        if pair["case"] == "roomy"
    ]
    floor = [pair for pair in pairs if pair["case"] == "exact-floor"]
    correctness_ok = bool(roomy) and all(
        pair["gates"]["correctness_qualified"]["qualified"]
        for pair, _ in roomy
    )
    stream_ok = bool(roomy) and all(
        pair["gates"]["stream_event_qualified"]["qualified"]
        for pair, _ in roomy
    )
    capacity_ok = bool(floor) and all(
        pair["gates"]["capacity_qualified"]["qualified"] for pair in floor
    )
    regressions: list[float] = []
    samples_ok = True
    for _, (stock, manager) in roomy:
        values = paired_regressions(stock, manager)
        samples_ok &= len(values) >= thresholds["minimum_samples_per_epoch"]
        regressions.extend(values)
    enough_epochs = len(roomy) >= thresholds["minimum_roomy_epochs"]
    ready = bool(regressions) and enough_epochs and samples_ok
    median = statistics.median(regressions) if regressions else None
    upper = (
        bootstrap_upper(
            regressions,
            confidence=float(thresholds["confidence_level"]),
            resamples=int(thresholds["bootstrap_resamples"]),
            seed=int(thresholds["bootstrap_seed"]),
        )
        if ready
        else None
    )
    throughput_ok = bool(
        ready
        and correctness_ok
        and median is not None
        and upper is not None
        and median <= thresholds["maximum_median_regression_fraction"]
        and upper <= thresholds["maximum_paired_upper_regression_fraction"]
    )
    throughput_reasons: list[str] = []
    if not correctness_ok:
        throughput_reasons.append("roomy correctness evidence is incomplete")
    if not regressions:
        throughput_reasons.append("no paired roomy timing samples were recorded")
    if not enough_epochs:
        throughput_reasons.append("too few independent roomy epochs")
    if not samples_ok:
        throughput_reasons.append("a roomy epoch has too few paired samples")
    if median is not None and median > thresholds["maximum_median_regression_fraction"]:
        throughput_reasons.append("paired median regression exceeds threshold")
    if upper is not None and upper > thresholds["maximum_paired_upper_regression_fraction"]:
        throughput_reasons.append("paired bootstrap upper regression exceeds threshold")
    statistics_value = {
        "roomy_epoch_count": len(roomy),
        "paired_sample_count": len(regressions),
        "paired_median_regression_fraction": median,
        "paired_bootstrap_upper_regression_fraction": upper,
    }
    return statistics_value, {
        "correctness_qualified": gate(
            correctness_ok, "not every roomy pair has exact stable outputs"
        ),
        "stream_event_qualified": gate(
            stream_ok, "not every roomy manager record proves event-safe reuse"
        ),
        "capacity_qualified": gate(
            capacity_ok, "no exact-floor manager-success/stock-exhaustion pair"
        ),
        "throughput_go": gate(throughput_ok, *throughput_reasons),
    }


def all_gates_qualified(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(gates) and all(result.get("qualified") is True for result in gates.values())


__all__ = [
    "all_gates_qualified",
    "bootstrap_upper",
    "classify_capacity_message",
    "gate",
    "is_capacity_exhaustion",
    "pair_gates",
    "paired_regressions",
    "summary_gates",
]
