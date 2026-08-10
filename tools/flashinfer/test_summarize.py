from __future__ import annotations

import copy
import unittest

from summarize import summarize_records


def record(provider: str, run_label: str) -> dict:
    return {
        "schema_version": 1,
        "provider": provider,
        "provider_version": "1.0.0",
        "provider_commit": "a" * 40,
        "run_label": run_label,
        "measurement": "eager_stream_batch_cuda_event",
        "operator": "paged_prefill",
        "case": "fixed_case",
        "dtype": "bf16",
        "layout": "NHD_D128_page16",
        "shape": {"batch_size": 1, "head_dim": 128},
        "fixture_id": "fixture-v1",
        "fixture_digests": {"query": "1234"},
        "execution": {"algorithm": "direct", "correctness": {"max_abs": 0.0}},
        "warmup_launches": 10,
        "launches_per_sample": 100,
        "samples_us": [2.0, 4.0],
    }


class SummarizeTests(unittest.TestCase):
    def test_keeps_provider_and_run_label_groups_separate(self) -> None:
        records = [
            record("loom-infer", "loom_first"),
            record("loom-infer", "loom_second"),
            record("flashinfer", "flashinfer_second"),
        ]

        result = summarize_records(records)

        self.assertEqual(result["schema_version"], 2)
        runs = result["cases"][0]["runs"]
        self.assertEqual(len(runs), 3)
        self.assertEqual(
            {(run["provider"], run["run_label"]) for run in runs},
            {
                ("loom-infer", "loom_first"),
                ("loom-infer", "loom_second"),
                ("flashinfer", "flashinfer_second"),
            },
        )

    def test_merges_only_identical_run_keys(self) -> None:
        first = record("loom-infer", "same_run")
        second = copy.deepcopy(first)
        second["samples_us"] = [6.0]

        result = summarize_records([first, second])

        run = result["cases"][0]["runs"][0]
        self.assertEqual(run["samples"], 3)
        self.assertEqual(run["median_us"], 4.0)
        self.assertEqual(run["execution"]["correctness"], {"max_abs": 0.0})

    def test_rejects_correctness_drift_within_one_run(self) -> None:
        first = record("loom-infer", "same_run")
        second = copy.deepcopy(first)
        second["execution"]["correctness"] = {"max_abs": 1.0}

        with self.assertRaisesRegex(ValueError, "inconsistent correctness"):
            summarize_records([first, second])

    def test_keeps_provider_algorithms_separate(self) -> None:
        direct = record("loom-infer", "same_run")
        tiled = copy.deepcopy(direct)
        tiled["execution"]["algorithm"] = "tiled"

        result = summarize_records([direct, tiled])

        runs = result["cases"][0]["runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual(
            {run["execution"]["algorithm"] for run in runs},
            {"direct", "tiled"},
        )

    def test_rejects_eager_and_graph_metadata_for_same_case(self) -> None:
        eager = record("loom-infer", "eager")
        graph = record("flashinfer", "graph")
        graph["measurement"] = "fixed_address_cuda_graph_single_replay_event"
        graph["launches_per_sample"] = 1

        with self.assertRaisesRegex(
            ValueError, "incompatible metadata.*measurement.*launches_per_sample"
        ):
            summarize_records([eager, graph])

    def test_rejects_shape_or_fixture_drift(self) -> None:
        first = record("loom-infer", "first")
        second = record("flashinfer", "second")
        second["shape"] = {"batch_size": 2, "head_dim": 128}
        second["fixture_digests"] = {"query": "5678"}

        with self.assertRaisesRegex(
            ValueError, "incompatible metadata.*shape.*fixture_digests"
        ):
            summarize_records([first, second])

    def test_rejects_invalid_samples(self) -> None:
        invalid = record("loom-infer", "bad")
        invalid["samples_us"] = [0.0]

        with self.assertRaisesRegex(ValueError, "positive finite"):
            summarize_records([invalid])


if __name__ == "__main__":
    unittest.main()
