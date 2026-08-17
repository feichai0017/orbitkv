from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
IDENTITY = ROOT / "integrations/sglang/checkpoint_identity.py"
ABLATION = ROOT / "integrations/sglang/bench_real_model_ablation.py"
UNIFORM_SWA = ROOT / "integrations/sglang/bench_uniform_swa_ab.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load OrbitKV benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RealModelBenchTests(unittest.TestCase):
    def test_checkpoint_identity_requires_every_indexed_shard(self):
        bench = load_module("orbitkv_checkpoint_identity", IDENTITY)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model-00001.safetensors").write_bytes(b"headerpayload")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 14},
                        "weight_map": {
                            "layer.0": "model-00001.safetensors",
                            "layer.1": "model-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )

            incomplete = bench.checkpoint_identity(root, "auto")
            self.assertFalse(incomplete["indexed_weights_complete"])
            self.assertEqual(
                incomplete["missing_indexed_weights"],
                ["model-00002.safetensors"],
            )

            (root / "model-00002.safetensors").write_bytes(b"more")
            complete = bench.checkpoint_identity(root, "auto")
            self.assertTrue(complete["indexed_weights_complete"])
            self.assertEqual(complete["observed_indexed_weight_bytes"], 17)
            self.assertEqual(complete["indexed_weight_container_overhead_bytes"], 3)

    def test_four_way_ablation_separates_native_and_orbitkv_modes(self):
        ablation = load_module("orbitkv_real_ablation", ABLATION)
        self.assertEqual(ablation.bench_mode("stock128"), ("stock", 128))
        self.assertEqual(ablation.bench_mode("stock32"), ("native_policy", 32))
        self.assertEqual(ablation.bench_mode("policy32"), ("policy", 32))
        self.assertEqual(ablation.bench_mode("owner32"), ("owner", 32))
        self.assertEqual(
            set(ablation.EXECUTION_ORDERS[0]),
            {"stock128", "stock32", "policy32", "owner32"},
        )

    def test_uniform_swa_runner_uses_mode_specific_pool_sizes(self):
        runner = load_module("orbitkv_uniform_swa_ab", UNIFORM_SWA)
        args = type(
            "Args",
            (),
            {
                "python": "python",
                "orbitkv_bin": "orbitkv",
                "model": "/tmp/model",
                "requests": 4,
                "prompt_tokens": 12000,
                "decode_tokens": 32,
                "context_length": 16384,
                "page_size": 16,
                "chunked_prefill_tokens": 2048,
                "eviction_interval": 128,
                "execute_pool_tokens": 19077,
                "reference_pool_tokens": 50000,
                "attention_backend": "flashinfer",
                "sglang_revision": "revision",
                "timeout": 30,
            },
        )()
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            mode = command[command.index("--mode") + 1]
            capacity = 19077 if mode == "state_plan" else 50000
            record = {
                "checkpoint": {"identity": "same"},
                "output_digest": "digest",
                "completion_tokens": 128,
                "num_retractions": [0, 0, 0, 0],
                "iteration_seconds": [8.0],
                "server_memory": {"token_capacity": capacity},
            }
            return type(
                "Completed",
                (),
                {"stdout": json.dumps(record) + "\n", "stderr": ""},
            )()

        with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
            execute = runner.run_once(
                args, "state_plan", Path("/tmp/state-plan.json")
            )
            reference = runner.run_once(
                args, "kernel_reference", Path("/tmp/state-plan.json")
            )

        self.assertEqual(execute["server_memory"]["token_capacity"], 19077)
        self.assertEqual(reference["server_memory"]["token_capacity"], 50000)
        self.assertIn("19077", calls[0])
        self.assertIn("50000", calls[1])


if __name__ == "__main__":
    unittest.main()
