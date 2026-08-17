from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IDENTITY = ROOT / "integrations/sglang/checkpoint_identity.py"
ABLATION = ROOT / "integrations/sglang/bench_real_model_ablation.py"


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


if __name__ == "__main__":
    unittest.main()
