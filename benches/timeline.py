"""Extract measured-request transfer observations without mixing host clocks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import percentile


def collect(directory: Path, samples: list[dict]) -> dict:
    request_ids = {sample["request_id"] for sample in samples if sample.get("request_id")}
    events = []
    by_request = defaultdict(list)
    for filename in ("manager.log", "engine.log"):
        for line in (directory / filename).read_text().splitlines():
            _, marker, payload = line.partition("cache_timeline ")
            if not marker:
                continue
            event, _ = json.JSONDecoder().raw_decode(payload)
            rid = event["request_id"]
            # vLLM appends a completion index and an engine-unique suffix.
            matched = rid
            while matched not in request_ids and "-" in matched:
                matched = matched.rsplit("-", 1)[0]
            if matched not in request_ids:
                continue
            event["source"] = filename
            events.append(event)
            by_request[matched].append(event)

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
            label = "warmup_prepare_ms" if event["warmup"] else "demand_prepare_ms"
            intervals[label].append(event["elapsed_us"] / 1000)

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
            }
            for label, values in intervals.items()
        },
        "notes": "Durations use one process's monotonic clock or Manager-local elapsed_us. Engine callback observations are not GPU kernel timestamps. Missing stages are not zero latency.",
    }
    (directory / "timeline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
