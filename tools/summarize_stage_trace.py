#!/usr/bin/env python3
"""Summarize synchronous compiler/runtime spans without double-counting children.

These are CPU-side wall intervals, not GPU timings or CPU utilization. Different
threads can overlap; their summed durations are not process elapsed time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def covered(intervals: list[tuple[int, int]]) -> int:
    total = 0
    right = 0
    for start, end in sorted(intervals):
        if end > right:
            total += end - max(start, right)
            right = end
    return total


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (not rows or rows[0].get("event") != "trace_started" or rows[0].get("schema") != 1
            or rows[-1].get("event") != "trace_completed" or rows[-1].get("open_spans") != 0):
        raise ValueError("expected a complete schema-1 stage trace")
    spans = {}
    for row in rows[1:-1]:
        if row.get("event") != "stage" or row.get("panicking") is not False:
            raise ValueError("unexpected or panicking stage record")
        for field in ("id", "start_ns", "wall_duration_ns"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"invalid stage {field}")
        if row["id"] in spans:
            raise ValueError("duplicate stage id")
        if not isinstance(row.get("fields"), dict) or not isinstance(row.get("thread"), str):
            raise ValueError("missing stage metadata")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError("missing stage name")
        parent = row.get("parent")
        if parent is not None and (type(parent) is not int or parent < 0):
            raise ValueError("invalid parent id")
        spans[row["id"]] = row
    if not spans:
        raise ValueError("stage trace contains no spans")
    children = defaultdict(list)
    for row in spans.values():
        parent = row["parent"]
        if parent is not None:
            if parent not in spans:
                raise ValueError("missing parent stage")
            ancestor = spans[parent]
            if (row["start_ns"] < ancestor["start_ns"]
                    or row["start_ns"] + row["wall_duration_ns"] > ancestor["start_ns"] + ancestor["wall_duration_ns"]):
                raise ValueError("child interval escapes parent")
            children[parent].append(row)
        seen = {row["id"]}
        while parent is not None:
            if parent in seen:
                raise ValueError("cyclic stage parents")
            seen.add(parent)
            parent = spans[parent]["parent"] if parent in spans else None
    totals = defaultdict(lambda: {"count": 0, "inclusive_wall_ns": 0, "self_wall_ns": 0, "maximum_wall_ns": 0})
    executions = []
    candidates = []
    for row in spans.values():
        duration = row["wall_duration_ns"]
        child_time = covered([(child["start_ns"], child["start_ns"] + child["wall_duration_ns"])
                              for child in children[row["id"]] if child["thread"] == row["thread"]])
        group = totals[row["name"]]
        group["count"] += 1
        group["inclusive_wall_ns"] += duration
        group["self_wall_ns"] += duration - child_time
        group["maximum_wall_ns"] = max(group["maximum_wall_ns"], duration)
        compact = {key: row[key] for key in ("id", "parent", "start_ns", "wall_duration_ns", "fields")}
        if row["name"] == "cuda.execute" and not row["fields"].get("profiling", True):
            executions.append(compact)
        if "program" in row["fields"]:
            candidates.append({"stage": row["name"], **compact})
    return {
        "schema": "luminal.stage-summary.v1", "status": "passed", "span_count": len(spans),
        "clock": "CPU-side monotonic wall time; no added device synchronization",
        "self_time_rule": "subtract union of direct child intervals on the same thread; other threads may overlap",
        "stages": [{"name": name, **data} for name, data in sorted(totals.items(), key=lambda pair: -pair[1]["self_wall_ns"])],
        "roots": [row for row in spans.values() if row["parent"] is None],
        "non_profiled_executions": sorted(executions, key=lambda row: row["start_ns"]),
        "candidate_program_spans": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.trace)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
