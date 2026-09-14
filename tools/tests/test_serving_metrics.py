import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving_metrics import ProcessMemorySampler, engine_report, validate_drain
from run_model_serving import summarize
from run_matched_serving import METRICS


class ServingMetricsTest(unittest.TestCase):
    def report(self):
        return {
            "stats": {"admitted_requests": 4, "completed_requests": 4,
                      "active_requests": 0, "queued_requests": 0, "cancelled_requests": 0,
                      "manager": {"free_pages": 32, "live_pages": 0, "reader_pins": 0}},
            "fixed_states": [[0, {"identity": {"slot_count": 4}, "free_slots": 4,
                                   "owned_slots": 0, "pending_transitions": 0}]],
        }

    def test_drain_rejects_live_owners_pending_work_and_cancellation(self):
        report = self.report()
        validate_drain(report, 4)
        for path in [("stats", "active_requests"), ("stats", "cancelled_requests"),
                     ("stats", "manager", "live_pages"), ("stats", "manager", "reader_pins")]:
            changed = copy.deepcopy(report)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = 1
            with self.assertRaises(ValueError):
                validate_drain(changed, 4)
        report["fixed_states"][0][1]["pending_transitions"] = 1
        with self.assertRaises(ValueError):
            validate_drain(report, 4)

    def test_report_marker_must_be_unique_and_structured(self):
        self.assertEqual(engine_report('other\nREPORT {"value":1}\n', "REPORT"), {"value": 1})
        for text in ("", "REPORT []", "REPORT {}\nREPORT {}"):
            with self.assertRaises(ValueError):
                engine_report(text, "REPORT")

    def test_missing_gpu_memory_is_not_reported_as_zero(self):
        sampler = ProcessMemorySampler(1, 1)
        sampler.samples = [{"host_rss_bytes": 100}, {"host_rss_bytes": 120}]
        self.assertIsNone(sampler.report()["sampled_max_gpu_bytes"])
        self.assertEqual(sampler.report()["sampled_max_host_rss_bytes"], 120)

    def test_device_memory_does_not_impersonate_process_memory(self):
        sampler = ProcessMemorySampler(1, 1, device_index=0)
        sampler.samples = [{"device_gpu_bytes": 1000}, {"device_gpu_bytes": 1500}]
        report = sampler.report()
        self.assertIsNone(report["sampled_max_gpu_bytes"])
        self.assertEqual(report["sampled_max_device_gpu_bytes"], 1500)
        self.assertEqual(report["device_index"], 0)

    def test_summary_keeps_runs_and_output_repeatability_separate(self):
        runs = [{"metrics": {name: value for name in METRICS},
                 "generated_texts_sha256": digest, "input_lengths": [31],
                 "memory": {"sampled_max_gpu_bytes": None, "sampled_max_host_rss_bytes": 100}}
                for value, digest in [(1, "a"), (3, "b"), (8, "a")]]
        summary = summarize(runs)
        self.assertEqual(summary["metrics"]["p95_tpot_ms"], 3)
        self.assertFalse(summary["output_repeatable"])
        self.assertIsNone(summary["sampled_max_gpu_bytes"])


if __name__ == "__main__":
    unittest.main()
