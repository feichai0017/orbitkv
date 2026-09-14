import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_serving_comparison import balanced_schedule, comparison_summary, load_servers, load_traces
from run_matched_serving import Workload


class ServingComparisonTests(unittest.TestCase):
    def test_trace_rejects_changed_lengths_missing_hashes_and_request_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.jsonl"
            row = {"timestamp": 0, "input_length": 2, "output_length": 3, "hash_ids": [17, 29]}
            path.write_text(json.dumps(row) + "\n")
            workloads = {"p": Workload(2, 3, 1, 1)}
            self.assertEqual(load_traces(Path(directory), workloads), {"p": path})
            for key, value in (("input_length", 1), ("output_length", 4), ("timestamp", 1),
                               ("hash_ids", [17]), ("hash_ids", [17, True])):
                path.write_text(json.dumps({**row, key: value}) + "\n")
                with self.assertRaises(ValueError):
                    load_traces(Path(directory), workloads)
            path.write_text(2 * (json.dumps(row) + "\n"))
            with self.assertRaises(ValueError):
                load_traces(Path(directory), workloads)

    def test_each_engine_occupies_each_position(self):
        names = ["orbitkv", "vllm", "sglang"]
        schedule = balanced_schedule(names, 6)
        for position in range(len(names)):
            self.assertEqual(sorted(row[position] for row in schedule), sorted(names * 2))
        for epochs in (0, -3, 2, 4):
            with self.assertRaises(ValueError):
                balanced_schedule(names, epochs)

    def test_server_config_preserves_literal_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servers.json"
            entries = [{"name": name, "command": ["engine", "--model", "path with $data and `text`"],
                        "base_url": "http://localhost:8000/"} for name in ("a", "b")]
            path.write_text(json.dumps({"servers": entries}))
            servers = load_servers(path, dry_run=True)
            self.assertEqual(servers[0][0].command[-1], "path with $data and `text`")
            self.assertEqual(servers[0][0].base_url, "http://localhost:8000")
            for key, value in (("name", "../a"), ("name", "b"), ("command", "engine"),
                               ("lifecycle", "unknown")):
                bad = copy.deepcopy(entries)
                bad[0][key] = value
                path.write_text(json.dumps({"servers": bad}))
                with self.assertRaises(ValueError):
                    load_servers(path, dry_run=True)

    @staticmethod
    def observations():
        return [{"profile": "p", "server": name, "epoch": epoch,
                 "input_lengths": [4, 3], "generated_texts_sha256": name,
                 "metrics": {metric: float(epoch) for metric in (
                     "request_throughput", "output_throughput", "median_ttft_ms", "p95_ttft_ms",
                     "p99_ttft_ms", "median_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
                     "median_itl_ms", "p95_itl_ms", "p99_itl_ms", "median_e2el_ms",
                     "p95_e2el_ms", "p99_e2el_ms")},
                 "memory": {"sampled_max_gpu_bytes": None, "sampled_max_host_rss_bytes": 10,
                            "sampled_max_device_gpu_bytes": 100}}
                for name in ("a", "b") for epoch in (1, 2)]

    def test_summary_keeps_memory_scope_ranges_and_output_disagreement(self):
        rows = comparison_summary(self.observations(), ["a", "b"], ["p"], 2)
        for row in rows:
            self.assertTrue(row["output_repeatable"])
            self.assertFalse(row["outputs_match_across_engines"])
            self.assertIsNone(row["sampled_max_gpu_bytes"])
            self.assertEqual(row["sampled_max_device_gpu_bytes"], 100)
            self.assertEqual(row["metrics"]["output_throughput"], 1.5)
            self.assertEqual(row["metric_ranges"]["output_throughput"], {"minimum": 1.0, "maximum": 2.0})

    def test_summary_rejects_incomplete_epochs_and_changed_request_order(self):
        runs = self.observations()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            comparison_summary(runs[:-1], ["a", "b"], ["p"], 2)
        runs[-1]["epoch"] = 1
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            comparison_summary(runs, ["a", "b"], ["p"], 2)
        runs = self.observations()
        runs[-1]["input_lengths"].reverse()
        with self.assertRaisesRegex(RuntimeError, "input lengths"):
            comparison_summary(runs, ["a", "b"], ["p"], 2)


if __name__ == "__main__":
    unittest.main()
