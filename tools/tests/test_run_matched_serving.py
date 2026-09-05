import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_matched_serving.py"
SPEC = importlib.util.spec_from_file_location("run_matched_serving", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def result(metrics=None, generated=None):
    payload = {
        "completed": 4,
        "failed": 0,
        "request_throughput": 2.0,
        "output_throughput": 20.0,
        "median_ttft_ms": 10.0,
        "p95_ttft_ms": 12.0,
        "p99_ttft_ms": 14.0,
        "median_tpot_ms": 5.0,
        "p95_tpot_ms": 6.0,
        "p99_tpot_ms": 7.0,
        "median_itl_ms": 4.0,
        "p95_itl_ms": 5.0,
        "p99_itl_ms": 6.0,
        "median_e2el_ms": 100.0,
        "p95_e2el_ms": 120.0,
        "p99_e2el_ms": 140.0,
    }
    if metrics:
        payload.update(metrics)
    if generated is not None:
        payload["generated_texts"] = generated
    return payload


class MatchedServingTest(unittest.TestCase):
    def test_command_parsing_accepts_relative_paths_during_dry_run(self):
        command = MODULE.parse_command(
            "target/release/orbitkv-server --port 8000",
            "candidate",
            require_executable=False,
        )
        self.assertEqual(command[0], "target/release/orbitkv-server")
        self.assertEqual(command[-1], "8000")

    def test_bench_command_fixes_random_lengths_and_tail_metrics(self):
        args = type(
            "Args",
            (),
            {
                "client_style": "rust",
                "backend": "openai",
                "endpoint": "/v1/completions",
                "model": "model",
                "tokenizer": "tokenizer",
                "request_rate": "inf",
                "seed": 7,
                "bench_arg": [],
            },
        )()
        command = MODULE.bench_command(
            ("vllm-bench",),
            MODULE.ServerSpec(
                "candidate", ("candidate",), "http://127.0.0.1:8000"
            ),
            args,
            MODULE.Workload(128, 32, 8, 2),
            Path("/tmp/results"),
            "benchmark.json",
        )
        self.assertEqual(command[0], "vllm-bench")
        self.assertEqual(command[command.index("--random-range-ratio") + 1], "0.0")
        self.assertEqual(command[command.index("--metric-percentiles") + 1], "95,99")

    def test_load_workload_rejects_invalid_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "bad": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "requests": 1,
                                "max_concurrency": 2,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                MODULE.load_workload(path, "bad")

    def test_result_gate_requires_all_requests_and_metrics(self):
        metrics, digest = MODULE.benchmark_result(
            result(generated=["a", "b", "c", "d"]), 4
        )
        self.assertEqual(metrics["median_tpot_ms"], 5.0)
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(RuntimeError, "request gate failed"):
            MODULE.benchmark_result(result({"failed": 1}), 4)
        with self.assertRaisesRegex(ValueError, "median_itl_ms"):
            MODULE.benchmark_result(result({"median_itl_ms": None}), 4)

    def test_paired_summary_preserves_metric_direction(self):
        candidate_metrics, digest = MODULE.benchmark_result(
            result(
                {
                    "request_throughput": 3.0,
                    "output_throughput": 30.0,
                    "median_ttft_ms": 8.0,
                    "p95_ttft_ms": 9.6,
                    "p99_ttft_ms": 11.2,
                    "median_tpot_ms": 4.0,
                    "p95_tpot_ms": 4.8,
                    "p99_tpot_ms": 5.6,
                    "median_itl_ms": 3.0,
                    "p95_itl_ms": 4.0,
                    "p99_itl_ms": 5.0,
                    "median_e2el_ms": 80.0,
                    "p95_e2el_ms": 96.0,
                    "p99_e2el_ms": 112.0,
                },
                ["same"] * 4,
            ),
            4,
        )
        baseline_metrics, baseline_digest = MODULE.benchmark_result(
            result(generated=["same"] * 4), 4
        )
        runs = []
        for epoch in (1, 2):
            runs.extend(
                [
                    {
                        "epoch": epoch,
                        "server": "candidate",
                        "metrics": candidate_metrics,
                        "generated_texts_sha256": digest,
                    },
                    {
                        "epoch": epoch,
                        "server": "baseline",
                        "metrics": baseline_metrics,
                        "generated_texts_sha256": baseline_digest,
                    },
                ]
            )
        summary = MODULE.paired_summary(runs, 2)
        ratios = summary["median_candidate_over_baseline"]
        self.assertEqual(ratios["request_throughput"], 1.5)
        self.assertEqual(ratios["median_ttft_ms"], 0.8)
        self.assertEqual(
            summary["ratio_interpretation"]["latency_metrics"],
            "less_than_one_favors_candidate",
        )
        self.assertTrue(summary["output_equivalence_evaluated"])
        self.assertTrue(summary["output_equivalence_passed"])
        self.assertFalse(summary["performance_qualified"])


if __name__ == "__main__":
    unittest.main()
