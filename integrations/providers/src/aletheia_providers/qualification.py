"""Portable Aletheia evidence helpers for qualification runners."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no samples")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_samples(samples_micros: Iterable[float], tokens_per_sample: int) -> dict[str, Any]:
    samples = [float(value) for value in samples_micros]
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    if tokens_per_sample <= 0:
        raise ValueError("tokens_per_sample must be positive")
    mean = sum(samples) / len(samples)
    return {
        "samples_micros": samples,
        "samples": len(samples),
        "p50_latency_micros": math.ceil(percentile(samples, 0.50)),
        "p99_latency_micros": math.ceil(percentile(samples, 0.99)),
        "goodput_tokens_per_second": tokens_per_sample * 1_000_000.0 / mean,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"{path}:{line_number}: event must be an object")
        events.append(event)
    if not events:
        raise ValueError(f"{path}: trace has no events")
    return events


def trace_covers(events: list[dict[str, Any]], point: dict[str, Any]) -> bool:
    for event in events:
        workload = event.get("workload")
        if not isinstance(workload, dict):
            continue
        rows = workload.get("rows_per_sequence") or {}
        context = workload.get("context_tokens") or {}
        row_min, row_max = rows.get("min"), rows.get("max")
        context_min, context_max = context.get("min"), context.get("max")
        if (
            event.get("outcome") == "ok"
            and workload.get("phase") == point["phase"]
            and workload.get("batch") == point["batch"]
            and isinstance(row_min, int)
            and isinstance(row_max, int)
            and row_min == point["rows_per_sequence"] == row_max
            and isinstance(context_min, int)
            and isinstance(context_max, int)
            and context_min == point["context_tokens"] == context_max
        ):
            return True
    return False
