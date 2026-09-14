import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_stage_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_stage_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def span(identity, parent, start, duration, *, thread="main", name="work", fields=None):
    return {"event": "stage", "id": identity, "parent": parent, "start_ns": start,
            "wall_duration_ns": duration, "thread": thread, "name": name,
            "fields": fields or {}, "panicking": False}


class StageSummaryTest(unittest.TestCase):
    def summarize(self, stages, complete=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [{"event": "trace_started", "schema": 1}, *stages,
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
        self.assertEqual(summary["candidate_program_spans"][0]["fields"]["program"], "program-a")
        self.assertEqual([row["id"] for row in summary["non_profiled_executions"]], [2])
