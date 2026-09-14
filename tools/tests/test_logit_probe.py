import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("logit_probe", Path(__file__).parents[1] / "logit_probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class LogitProbeTests(unittest.TestCase):
    def test_comparison_reports_actual_winner_margin_and_error(self):
        result = PROBE.compare_rows([1.0, 1.125, -5.0], [1.25, 1.125, -4.5], 2)
        self.assertFalse(result["argmax_equal"])
        self.assertEqual(result["reference_top2_margin"], 0.125)
        self.assertEqual(result["reference_preference_for_its_winner"], 0.125)
        self.assertEqual(result["candidate_preference_for_its_winner"], 0.125)
        self.assertEqual(result["maximum_absolute_error"], 0.5)
        self.assertEqual(result["worst_error_token_id"], 2)
        self.assertAlmostEqual(result["root_mean_square_error"], math.sqrt(0.3125 / 3))

    def test_ties_use_lowest_index_but_record_zero_margin(self):
        result = PROBE.compare_rows([2.0, 2.0], [2.0, 2.0], 2)
        self.assertEqual(result["reference_top"][0]["token_id"], 0)
        self.assertEqual(result["reference_top2_margin"], 0.0)
        self.assertEqual(result["candidate_maximum_tie_count"], 2)

    def test_nonfinite_or_mismatched_rows_are_rejected(self):
        for left, right in [([0, float("nan")], [0, 1]), ([0, 1], [0]), ([0, 1], [float("inf"), 0])]:
            with self.assertRaises(ValueError):
                PROBE.compare_rows(left, right, 2)

    def test_trace_comparison_rejects_different_teacher_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            left, right = Path(root) / "left", Path(root) / "right"
            for directory in (left, right):
                directory.mkdir()
                (directory / "row.f32").write_bytes(struct.pack("<ff", 1.0, 2.0))
                trace = {"schema": 1, "teacher_forced": True, "vocabulary_size": 2,
                         "cases": [{"id": "case", "prompt_token_ids": [0], "drain_passed": True,
                                    "steps": [{"step": 0, "input_token_ids": [0], "selected_token_id": 1, "logits_file": "row.f32"}]}]}
                (directory / "trace.json").write_text(json.dumps(trace))
            result = PROBE.compare_traces(left, right, 2)
            self.assertIsNone(result["cases"][0]["first_argmax_difference"])
            # A device reduction can choose a different tied maximum even when
            # canonical argmax computed from the exported logits is identical.
            (right / "row.f32").write_bytes(struct.pack("<ff", 2.0, 2.0))
            trace["cases"][0]["steps"][0]["selected_token_id"] = 0
            (right / "trace.json").write_text(json.dumps(trace))
            result = PROBE.compare_traces(left, right, 2)
            self.assertEqual(result["cases"][0]["first_selected_token_difference"], 0)
            self.assertTrue(result["cases"][0]["steps"][0]["selected_token_is_logit_maximum"])
            trace["cases"][0]["steps"][0]["input_token_ids"] = [1]
            (right / "trace.json").write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "teacher-forced input"):
                PROBE.compare_traces(left, right, 2)

    def test_manifest_rejects_duplicate_cases_or_empty_sequences(self):
        case = {"id": "same", "prompt_token_ids": [1], "continuation_token_ids": [0]}
        for cases in ([case, case], [{**case, "continuation_token_ids": []}], [{**case, "prompt_token_ids": [True]}]):
            with self.assertRaises(ValueError):
                PROBE.validate_manifest({"schema": 1, "cases": cases})


if __name__ == "__main__":
    unittest.main()
