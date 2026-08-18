from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from orbitkv_sglang.capsule_wire import decode_cpu_tensors, encode_cpu_tensors
else:
    decode_cpu_tensors = None
    encode_cpu_tensors = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CapsuleWireTests(unittest.TestCase):
    def test_nested_hybrid_cpu_copy_round_trips(self):
        value = {
            "full": [
                [
                    [
                        torch.arange(12, dtype=torch.float16).reshape(3, 2, 2),
                        torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2),
                    ]
                ]
            ],
            "swa": [[[
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
            ]]],
            "swa_mask": torch.tensor([True, False, True]),
        }
        payload = encode_cpu_tensors(value)
        restored = decode_cpu_tensors(payload)
        self.assertEqual(set(restored), set(value))
        self.assertTrue(torch.equal(restored["full"][0][0][0], value["full"][0][0][0]))
        self.assertTrue(torch.equal(restored["full"][0][0][1], value["full"][0][0][1]))
        self.assertTrue(torch.equal(restored["swa"][0][0][0], value["swa"][0][0][0]))
        self.assertTrue(torch.equal(restored["swa_mask"], value["swa_mask"]))

    def test_rejects_gpu_or_unsupported_values(self):
        with self.assertRaises(TypeError):
            encode_cpu_tensors({"bad": 1})

    def test_rejects_trailing_bytes(self):
        payload = encode_cpu_tensors(torch.tensor([1, 2], dtype=torch.int64))
        with self.assertRaises(ValueError):
            decode_cpu_tensors(payload + b"x")


if __name__ == "__main__":
    unittest.main()
