"""Rebuild comparable per-run CSV/JSON summaries without starting a GPU runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .metrics import cache_source, summarize, workload_phases


def collect_run(directory: Path) -> dict:
    if (directory / "failure.json").exists():
        raise ValueError(f"Failed run has no complete latency result: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    args = manifest["arguments"]
    timeline = directory / "timeline-summary.json"
    timeline_summary = json.loads(timeline.read_text()) if timeline.exists() else None
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    if args.get("workload") in ("concurrent", "sustained"):
        from . import concurrent, sustained

        workload = sustained if args["workload"] == "sustained" else concurrent
        evidence = "windows.jsonl" if args["workload"] == "sustained" else "batches.jsonl"
        batches = [json.loads(line) for line in (directory / evidence).read_text().splitlines()]
        workload.validate(args, samples, batches)
        return {
            "directory": str(directory.resolve()),
            "manifest": manifest,
            "timeline": timeline_summary,
            "storage": json.loads((directory / "storage.json").read_text())
            if args.get("ssd_gib", 0)
            else None,
            "summary": workload.summarize(samples, batches),
        }
    expected = {
        (length, repeat, phase)
        for length in args["lengths"]
        for repeat in range(args["repeats"])
        for phase in workload_phases(bool(args.get("ssd_gib", 0)))
    }
    actual = {(s["length"], s["repeat"], s["phase"]) for s in samples}
    if actual != expected or len(samples) != len(expected):
        raise ValueError(f"Incomplete or duplicated workload: {directory}")
    for sample in samples:
        for metric in ("ttft_ms", "e2e_ms"):
            if not math.isfinite(sample[metric]) or sample[metric] < 0:
                raise ValueError(f"Invalid {metric} in {directory}")
        sample["cache_source"] = cache_source(args["engine"], sample)
    return {
        "directory": str(directory.resolve()),
        "manifest": manifest,
        "timeline": timeline_summary,
        "storage": json.loads((directory / "storage.json").read_text())
        if args.get("ssd_gib", 0)
        else None,
        "summary": summarize(samples, args["lengths"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Empty report directory")
    args = parser.parse_args()
    runs = [collect_run(directory) for directory in args.runs]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in runs:
        config = run["manifest"]["arguments"]
        for summary in run["summary"]:
            rows.append(
                {
                    "run": run["directory"],
                    "engine": config["engine"],
                    "backend": config["backend"],
                    "gpu_tokens": config["gpu_tokens"],
                    "host_gib": config["host_gib"],
                    "ssd_gib": config.get("ssd_gib", 0),
                    "query_budget_gib": config.get("query_budget_gib"),
                    "prefill_tokens": config.get("prefill_tokens", 8192),
                    "cache_protected_percent": config.get("cache_protected_percent", 0),
                    "ssd_write_policy": config.get("ssd_write_policy", "all"),
                    "queue_warmup": config.get("queue_warmup"),
                    "prepare_requests": config.get("prepare_requests"),
                    "read_batch_mib": config.get("read_batch_mib"),
                    "read_timeout_ms": config.get("read_timeout_ms"),
                    "read_max_batches": config.get("read_max_batches"),
                    **summary,
                    "cache_sources": json.dumps(summary["cache_sources"], sort_keys=True),
                }
            )
    (args.output / "summary.json").write_text(json.dumps(runs, indent=2, allow_nan=False) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
