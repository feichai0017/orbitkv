#!/usr/bin/env python3
"""Summarize matched JSON Lines samples without discarding raw evidence."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main(paths: list[str]) -> None:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict] = {}
    for path_string in paths:
        path = Path(path_string)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["provider"], record["case"])
            samples[key].extend(record["samples_us"])
            metadata[key] = record

    cases = sorted({case for _, case in samples})
    summaries = []
    for case in cases:
        providers = {}
        for provider in ("loom-infer", "flashinfer"):
            values = samples.get((provider, case))
            if values:
                providers[provider] = {
                    "samples": len(values),
                    "median_us": statistics.median(values),
                    "p10_us": percentile(values, 0.10),
                    "p90_us": percentile(values, 0.90),
                    "min_us": min(values),
                    "max_us": max(values),
                    "provider_version": metadata[(provider, case)][
                        "provider_version"
                    ],
                    "provider_commit": metadata[(provider, case)][
                        "provider_commit"
                    ],
                }
        summary = {"case": case, "providers": providers}
        if "loom-infer" in providers and "flashinfer" in providers:
            summary["flashinfer_over_loom_median"] = (
                providers["flashinfer"]["median_us"]
                / providers["loom-infer"]["median_us"]
            )
        summaries.append(summary)
    print(json.dumps({"schema_version": 1, "cases": summaries}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize.py RESULTS.jsonl [RESULTS.jsonl ...]")
    main(sys.argv[1:])
