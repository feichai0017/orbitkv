import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_stage_trace.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("summarize_stage_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def span(identity, parent, start, duration, *, thread="main", name="work", fields=None):
    return {"event": "stage", "id": identity, "parent": parent, "start_ns": start,
            "wall_duration_ns": duration, "thread": thread, "name": name,
            "fields": fields or {}, "panicking": False}


def metric(parent, at, name, **fields):
    return {"event": "metric", "parent": parent, "at_ns": at, "name": name,
            "thread": "main", "panicking": False, "fields": fields}


class StageSummaryTest(unittest.TestCase):
    def summarize(self, stages, complete=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [{"event": "trace_started", "schema": 2}, *stages,
                    {"event": "trace_completed" if complete else "trace_incomplete", "open_spans": 0}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return MODULE.summarize(path)

    def test_child_union_and_thread_overlap_do_not_double_subtract(self):
        summary = self.summarize([
            span(0, None, 0, 100, name="parent"),
            span(1, 0, 10, 50), span(2, 0, 40, 40),
            span(3, 0, 20, 70, thread="worker"),
        ])
        parent = next(row for row in summary["stages"] if row["name"] == "parent")
        self.assertEqual(parent["self_wall_ns"], 30)
        self.assertEqual(parent["inclusive_wall_ns"], 100)
        self.assertEqual(summary["span_count"], 4)

    def test_rejects_incomplete_empty_or_invalid_parent_graphs(self):
        invalid = [[], [span(0, 9, 0, 10)], [span(0, 0, 0, 10)],
                   [span(0, None, 0, 10), span(0, None, 0, 10)],
                   [span(0, None, 0, 10), span(1, 0, 9, 10)],
                   [{**span(0, None, 0, 10), "panicking": True}]]
        for stages in invalid:
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                self.summarize(stages)
        with self.assertRaises(ValueError):
            self.summarize([span(0, None, 0, 10)], complete=False)

    def test_preserves_program_identity_and_separates_candidate_execution(self):
        summary = self.summarize([
            span(0, None, 0, 100, name="candidate", fields={"program": "program-a", "bucket": 0}),
            span(1, 0, 10, 20, name="cuda.execute", fields={"profiling": True}),
            span(2, None, 120, 30, name="cuda.execute", fields={"profiling": False, "bucket": 1}),
        ])
        self.assertEqual(summary["program_spans"][0]["fields"]["program"], "program-a")
        self.assertEqual([row["id"] for row in summary["non_profiled_executions"]], [2])

    def test_reported_measurements_do_not_change_cpu_self_time(self):
        result = self.summarize([
            span(0, None, 0, 100),
            metric(0, 50, "independent.device.counter", duration_ns=9000),
        ])
        self.assertEqual(result["span_count"], 1)
        self.assertEqual(result["stages"][0]["self_wall_ns"], 100)
        self.assertEqual(result["metric_counts"], {"independent.device.counter": 1})

    def test_rule_aggregation_retains_full_names_and_zero_match_cost(self):
        name = "long-untruncated-rule-identity-" * 8
        phase = {"phase": "main-1", "schedule": "(run rules)", "run_wall_ns": 80,
                 "iterations": 1, "tuples_before": 10, "tuples_after": 12}
        rows = [
            span(0, None, 0, 500, name="orbitkv.compiler.egglog.bucket", fields={"bucket": 3}),
            span(1, 0, 0, 500, name="orbitkv.compiler.egglog.run", fields={"schedule_count": 2}),
            span(2, 1, 10, 100, name="orbitkv.compiler.egglog.schedule", fields=phase),
            metric(2, 95, "orbitkv.compiler.egglog.rule", rule=name, search_apply_ns=60, matches=2),
            span(3, 1, 120, 100, name="orbitkv.compiler.egglog.schedule", fields={**phase, "phase": "main-2"}),
            metric(3, 205, "orbitkv.compiler.egglog.rule", rule=name, search_apply_ns=70, matches=0),
            metric(3, 206, "orbitkv.compiler.egglog.ruleset", ruleset="rules", search_apply_ns=75, merge_ns=2, rebuild_ns=3),
        ]
        result = self.summarize(rows)
        run = result["egglog_runs"][0]
        self.assertEqual(run["bucket"], {"bucket": 3})
        self.assertEqual(run["rules"], [{"rule": name, "records": 2, "search_apply_ns": 130, "matches": 2}])
        self.assertEqual(run["phases"][1]["rules"][0]["matches"], 0)
        self.assertEqual(run["rulesets"][0]["rebuild_ns"], 3)
        with self.assertRaisesRegex(ValueError, "duplicate egglog"):
            self.summarize([*rows, rows[-1]])
        with self.assertRaisesRegex(ValueError, "schedule measurements are incomplete"):
            self.summarize(rows[:2])

    def graph_measurements(self):
        return [
            span(0, None, 0, 100, name="orbitkv.qualification.step", fields={"phase": "decode", "batch_size": 8}),
            span(1, 0, 5, 90, name="cuda.execute", fields={"program": "selected-program", "bucket": 2, "profiling": False}),
            span(2, 1, 10, 50, name="cuda.graph.profile", fields={"graph_node": 7, "step_count": 3,
                 "total_device_ms": 3.5, "status": "measured", "detailed": True}),
            metric(2, 20, "cuda.graph.dimension", dimension="b", value=8),
            metric(2, 21, "cuda.graph.step", index=0, operator="CustomLinear", implementation="shape A", duration_ms=1.25),
            metric(2, 22, "cuda.graph.step", index=1, operator="CustomLinear", implementation="shape B", duration_ms=2.0),
            metric(2, 23, "cuda.graph.step", index=2, operator="CustomLinear", implementation="shape A", duration_ms=0.25),
        ]

    def test_gpu_profiles_preserve_program_workload_order_and_implementation(self):
        profile = self.summarize(self.graph_measurements())["cuda_graph_profiles"][0]
        self.assertEqual(profile["program"], "selected-program")
        self.assertEqual(profile["workload"], {"phase": "decode", "batch_size": 8})
        self.assertEqual(profile["dimensions"], {"b": 8})
        self.assertEqual([row["index"] for row in profile["steps"]], [0, 1, 2])
        self.assertEqual([(row["implementation"], row["records"], row["duration_ms"]) for row in profile["operations"]],
                         [("shape B", 1, 2.0), ("shape A", 2, 1.5)])

    def test_invalid_or_incomplete_device_measurements_cannot_qualify(self):
        for issue in ("missing_step", "duplicate_step", "negative_time", "nonfinite_time", "wrong_total",
                      "missing_program", "missing_execution", "duplicate_dimension", "unavailable"):
            rows = self.graph_measurements()
            if issue == "missing_step": rows.pop()
            elif issue == "duplicate_step": rows.append(rows[-1])
            elif issue == "negative_time": rows[-1]["fields"]["duration_ms"] = -1
            elif issue == "nonfinite_time": rows[-1]["fields"]["duration_ms"] = float("nan")
            elif issue == "wrong_total": rows[2]["fields"]["total_device_ms"] = 9
            elif issue == "missing_program": del rows[1]["fields"]["program"]
            elif issue == "missing_execution": rows[1]["name"] = "unrelated"
            elif issue == "duplicate_dimension": rows.append(rows[3])
            elif issue == "unavailable": rows[2]["fields"]["status"] = "unavailable"
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                self.summarize(rows)

    def test_rejects_orphaned_escaped_or_mistyped_measurements(self):
        invalid = [metric(99, 50, "counter"), metric(0, 101, "counter"),
                   {**metric(0, 50, "counter"), "at_ns": True},
                   metric(0, 50, "orbitkv.compiler.egglog.rule", rule="rule", search_apply_ns=1, matches=1)]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.summarize([span(0, None, 0, 100), row])
