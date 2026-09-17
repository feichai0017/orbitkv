"""Guard the numeric execution contracts and semantic names of the Qwen examples."""
import hashlib
import json
import pathlib
import unittest

from kern_manifest import resolve_constants
from qwen_constants import name_constants

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Canonical JSON hashes of the checked-in examples (last: unified attention
# reads the kv state, `inout` -> `in`, 2026-09-16). Changes to kernel wiring,
# capacities, float literals or even artifact pins require review.
BASELINES = {
    "qwen3-4b-dspark.json": "444001bf6fa6991128128a878c0cc139b8ecf71800c6ccfea0d344c1ad169133",
    "qwen3-4b-silu-mined.json": "e14198af226849f46826bc624c4a6e8a208a4e8d7b160ffc611b817d2915d7b8",
    "qwen3-4b.json": "792666b476938520a21598bec6bdf5461cb62dd8fdf52cd5998b33a7388c2e69",
    "qwen3.8-27b-dflash2.json": "134839dd1b25fc8fb5a5cb97391a02576d2624f2f0ccc5431af6e10c9dd679dd",
    "qwen3.8-27b.json": "a9cc996a7548dabc756b30f9ef848459c2a5c2043075b79e11091054d1a30443"
}


class QwenConstantsTests(unittest.TestCase):
    def test_execution_contracts_and_reproducible_generation(self):
        for name, digest in BASELINES.items():
            with self.subTest(name=name):
                named = json.loads((ROOT / "examples" / name).read_text())
                literal = resolve_constants(named)
                canonical = json.dumps(literal, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
                self.assertEqual(name_constants(literal), named)
                self.assertEqual(name_constants(named), named)

    def test_equal_numbers_keep_distinct_meanings(self):
        qwen3 = json.loads((ROOT / "examples/qwen3-4b.json").read_text())
        self.assertEqual(qwen3["buffers"]["block_table"]["shape"][-1], "MAX_BLOCKS_PER_SEQ")
        self.assertEqual(qwen3["buffers"]["x"]["shape"][-1], "HIDDEN_SIZE")
        draft = json.loads((ROOT / "examples/qwen3.8-27b-dflash2.json").read_text())
        for name, expected in [("q_n", "Q_DIM"), ("gdn_v", "GDN_V_DIM"), ("d_qkv", "DRAFT_QKV_DIM")]:
            self.assertEqual(draft["buffers"][name]["shape"][-1], expected)
        self.assertEqual(draft["buffers"]["hidden_r"]["shape"], ["MAX_VERIFY_TOKENS", "SELECTOR_RANK"])
        self.assertEqual(draft["buffers"]["model.layers.3.self_attn.q_norm.weight_p1"]["shape"], ["HEAD_DIM"])

    def test_reference_names_are_not_substituted(self):
        m = json.loads((ROOT / "examples/minimal.json").read_text())
        m["constants"] = {"x": 64, "scale": 3}
        m["buffers"]["w"]["shape"] = ["x"]
        resolved = resolve_constants(m)
        self.assertEqual(resolved["buffers"]["w"]["shape"], [64])
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][0], {"buf": "x"})
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][-1], {"var": "tokens"})
