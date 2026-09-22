"""Extract measured-request transfer observations without mixing host clocks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import percentile


def collect(directory: Path, samples: list[dict]) -> dict:
    request_ids = {sample["request_id"] for sample in samples if sample.get("request_id")}
    observed = []
    by_request = defaultdict(list)
    for filename in ("manager.log", "engine.log"):
        for line in (directory / filename).read_text().splitlines():
            _, marker, payload = line.partition("cache_timeline ")
            if not marker:
                continue
            event, _ = json.JSONDecoder().raw_decode(payload)
            event["source"] = filename
            observed.append(event)

    def match_request(rid):
        # vLLM appends a completion index and an engine-unique suffix.
        while rid not in request_ids and "-" in rid:
            rid = rid.rsplit("-", 1)[0]
        return rid if rid in request_ids else None

    restores = defaultdict(set)
    for event in observed:
        if event["stage"] == "restore_link" and (rid := match_request(event["request_id"])):
            restores[event["restore_key"]].add(rid)
    events = []
    for event in observed:
        rid = match_request(event.get("request_id", ""))
        owners = {rid} if rid else restores.get(event.get("restore_key"), set())
        if not owners:
            continue
        events.append(event)
        for owner in owners:
            by_request[owner].append(event)

    if request_ids and not by_request:
        raise ValueError("No transfer timeline records matched the measured request IDs")

    intervals = defaultdict(list)
    for group in by_request.values():
        for start, end, label in (
            ("queued", "first_use", "queued_to_first_use_ms"),
            ("restore_submit", "gpu_ready", "restore_ms"),
        ):
            starts = {}
            for event in group:
                if "monotonic_ns" not in event:
                    continue
                pid, now = event["pid"], event["monotonic_ns"]
                if event["stage"] == start:
                    starts[pid] = now
                elif event["stage"] == end and pid in starts:
                    intervals[label].append((now - starts.pop(pid)) / 1e6)
    for event in events:
        if event["stage"] == "host_ready":
            label = (
                "prepared_read_ms"
                if event.get("prepare")
                else "warmup_prepare_ms"
                if event["warmup"]
                else "demand_prepare_ms"
            )
            intervals[label].append(event["elapsed_us"] / 1000)
        elif event["stage"] in (
            "discovery_ready",
            "restore_complete",
            "restore_notification",
            "restore_delivered",
        ):
            labels = {
                "discovery_ready": "candidate_discovery_ms",
                "restore_complete": "manager_restore_ms",
                "restore_notification": "completion_signal_ms",
                "restore_delivered": "completion_delivery_ms",
            }
            intervals[labels[event["stage"]]].append(event["elapsed_us"] / 1000)

    with (directory / "timeline.jsonl").open("w") as output:
        for event in events:
            output.write(json.dumps(event) + "\n")
    summary = {
        "measured_requests": len(samples),
        "traced_requests": len(by_request),
        "stage_counts": dict(Counter(event["stage"] for event in events)),
        "intervals": {
            label: {
                "count": len(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
            for label, values in intervals.items()
        },
        "notes": "Durations use one process's monotonic clock or Manager-local elapsed_us. Manager restore includes submission/worker queue and synchronized copy; completion signal/delivery start at the worker's terminal timestamp. Manager batches are counted once even when several requests share them. Engine callback observations are not GPU kernel timestamps. Dense lookup can fuse discovery with preparation; missing stages are not zero latency.",
    }
    (directory / "timeline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
