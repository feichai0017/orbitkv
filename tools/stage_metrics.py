"""Interpret reported compiler and device measurements, separately from CPU spans."""

from __future__ import annotations

from collections import defaultdict
import math


def integer(fields: dict, name: str) -> int:
    value = fields.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid measurement {name}")
    return value


def text(fields: dict, name: str, *, allow_empty: bool = False) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"invalid measurement {name}")
    return value


def duration(fields: dict, name: str) -> float:
    value = fields.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid measurement {name}")
    return value


def ancestor(row: dict, spans: dict, name: str) -> dict | None:
    parent = row["parent"]
    while parent is not None:
        stage = spans[parent]
        if stage["name"] == name:
            return stage
        parent = stage["parent"]
    return None


def aggregate(rows: list[dict], identity: tuple[str, ...], counters: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in identity)
        group = groups.setdefault(key, {**dict(zip(identity, key)), "records": 0,
                                        **dict.fromkeys(counters, 0)})
        group["records"] += 1
        for counter in counters:
            group[counter] += row[counter]
    return sorted(groups.values(), key=lambda row: (-row[counters[0]], *(row[field] for field in identity)))


def egglog_runs(spans: dict, metrics: list[dict]) -> list[dict]:
    rules = defaultdict(list)
    rulesets = defaultdict(list)
    seen = set()
    for metric in metrics:
        kind = metric["name"]
        if kind not in ("luminal.egglog.rule", "luminal.egglog.ruleset"):
            continue
        parent = spans.get(metric["parent"])
        if parent is None or parent["name"] != "luminal.egglog.schedule":
            raise ValueError("egglog measurement has no schedule parent")
        fields = metric["fields"]
        identity = "rule" if kind.endswith(".rule") else "ruleset"
        name = text(fields, identity)
        key = (metric["parent"], kind, name)
        if key in seen:
            raise ValueError("duplicate egglog measurement within a schedule")
        seen.add(key)
        row = {identity: name, "search_apply_ns": integer(fields, "search_apply_ns")}
        if identity == "rule":
            row["matches"] = integer(fields, "matches")
            rules[parent["id"]].append(row)
        else:
            row.update(merge_ns=integer(fields, "merge_ns"), rebuild_ns=integer(fields, "rebuild_ns"))
            rulesets[parent["id"]].append(row)

    phases = defaultdict(list)
    for stage in sorted(spans.values(), key=lambda row: row["start_ns"]):
        if stage["name"] != "luminal.egglog.schedule":
            continue
        run = ancestor(stage, spans, "luminal.egglog.run")
        if run is None:
            raise ValueError("egglog schedule has no run parent")
        fields = stage["fields"]
        elapsed = integer(fields, "run_wall_ns")
        if elapsed > stage["wall_duration_ns"]:
            raise ValueError("reported schedule time exceeds its enclosing span")
        before, after = integer(fields, "tuples_before"), integer(fields, "tuples_after")
        phases[run["id"]].append({
            "stage_id": stage["id"], "phase": text(fields, "phase"),
            "schedule": text(fields, "schedule"), "run_wall_ns": elapsed,
            "observed_wall_ns": stage["wall_duration_ns"],
            "iterations": integer(fields, "iterations"),
            "tuples_before": before, "tuples_after": after, "tuple_delta": after - before,
            "rules": sorted(rules[stage["id"]], key=lambda row: (-row["search_apply_ns"], row["rule"])),
            "rulesets": sorted(rulesets[stage["id"]], key=lambda row: (-row["search_apply_ns"], row["ruleset"])),
        })
    output = []
    for stage in sorted(spans.values(), key=lambda row: row["start_ns"]):
        if stage["name"] != "luminal.egglog.run":
            continue
        bucket = ancestor(stage, spans, "luminal.egglog.bucket")
        run_phases = phases[stage["id"]]
        if not run_phases or len(run_phases) != integer(stage["fields"], "schedule_count"):
            raise ValueError("egglog schedule measurements are incomplete")
        output.append({
            "stage_id": stage["id"], "bucket": bucket["fields"] if bucket else None,
            "wall_duration_ns": stage["wall_duration_ns"], "fields": stage["fields"],
            "phases": run_phases,
            "rules": aggregate([rule for phase in run_phases for rule in phase["rules"]],
                               ("rule",), ("search_apply_ns", "matches")),
            "rulesets": aggregate([row for phase in run_phases for row in phase["rulesets"]],
                                  ("ruleset",), ("search_apply_ns", "merge_ns", "rebuild_ns")),
        })
    return output


def cuda_graph_profiles(spans: dict, metrics: list[dict]) -> list[dict]:
    steps = defaultdict(list)
    dimensions = defaultdict(dict)
    for metric in metrics:
        kind = metric["name"]
        if kind not in ("cuda.graph.step", "cuda.graph.dimension"):
            continue
        parent = spans.get(metric["parent"])
        if parent is None or parent["name"] != "cuda.graph.profile":
            raise ValueError("CUDA measurement has no graph profile parent")
        fields = metric["fields"]
        if kind == "cuda.graph.dimension":
            name = text(fields, "dimension")
            if name in dimensions[parent["id"]]:
                raise ValueError("duplicate graph dimension")
            dimensions[parent["id"]][name] = integer(fields, "value")
        else:
            steps[parent["id"]].append({
                "index": integer(fields, "index"), "operator": text(fields, "operator"),
                "implementation": text(fields, "implementation", allow_empty=True),
                "duration_ms": duration(fields, "duration_ms"),
            })
    output = []
    for stage in sorted(spans.values(), key=lambda row: row["start_ns"]):
        if stage["name"] != "cuda.graph.profile":
            continue
        fields = stage["fields"]
        if fields.get("status") != "measured" or type(fields.get("detailed")) is not bool:
            raise ValueError("CUDA graph measurement is incomplete")
        count = integer(fields, "step_count")
        graph_steps = sorted(steps[stage["id"]], key=lambda row: row["index"])
        if [row["index"] for row in graph_steps] != list(range(count)):
            raise ValueError("missing or duplicate CUDA graph step")
        total = duration(fields, "total_device_ms")
        if not math.isclose(total, math.fsum(row["duration_ms"] for row in graph_steps),
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("CUDA graph total disagrees with its step measurements")
        execution = ancestor(stage, spans, "cuda.execute")
        if execution is None:
            raise ValueError("CUDA graph profile has no execution parent")
        workload = ancestor(stage, spans, "orbitkv.qualification.step")
        output.append({
            "stage_id": stage["id"], "execution_stage_id": execution["id"],
            "execution": execution["fields"], "program": text(execution["fields"], "program"),
            "graph_node": integer(fields, "graph_node"),
            "dimensions": dimensions[stage["id"]], "detailed": fields["detailed"],
            "workload": workload["fields"] if workload else None,
            "total_device_ms": total, "steps": graph_steps,
            "operations": aggregate(graph_steps, ("operator", "implementation"), ("duration_ms",)),
        })
    return output
