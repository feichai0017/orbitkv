import json
import os
import sys
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))


class HookType(Enum):
    AROUND = "around"


class HookRegistry:
    registered = []

    @classmethod
    def register(cls, target, hook, hook_type):
        cls.registered.append((target, hook, hook_type))


def install_fake_sglang():
    hook_registry = types.ModuleType("sglang.srt.plugins.hook_registry")
    hook_registry.HookRegistry = HookRegistry
    hook_registry.HookType = HookType
    plugins = types.ModuleType("sglang.srt.plugins")
    plugins.hook_registry = hook_registry
    srt = types.ModuleType("sglang.srt")
    srt.plugins = plugins
    sglang = types.ModuleType("sglang")
    sglang.srt = srt
    sys.modules.update(
        {
            "sglang": sglang,
            "sglang.srt": srt,
            "sglang.srt.plugins": plugins,
            "sglang.srt.plugins.hook_registry": hook_registry,
        }
    )


install_fake_sglang()
from aletheia_sglang import plugin  # noqa: E402


class Request:
    pass


class Batch:
    reqs = [Request(), Request()]
    forward_mode = "decode"
    seq_lens_cpu = [17, 33]
    extend_lens = None


class Worker:
    pass


class PluginTests(unittest.TestCase):
    def setUp(self):
        HookRegistry.registered.clear()

    def test_registers_real_tp_worker_hook(self):
        plugin.register()
        target, hook, hook_type = HookRegistry.registered[0]
        self.assertEqual(
            target,
            "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
        )
        self.assertIs(hook, plugin._around_forward_batch_generation)
        self.assertIs(hook_type, HookType.AROUND)

    def test_trace_wraps_real_call_shape_without_changing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory, "trace.jsonl")
            with patch.dict(os.environ, {"ALETHEIA_TRACE": str(trace)}):
                result = plugin._around_forward_batch_generation(
                    lambda worker, batch, forward_batch: "result",
                    Worker(),
                    Batch(),
                    None,
                )
            self.assertEqual(result, "result")
            event = json.loads(trace.read_text())
            self.assertEqual(event["batch_size"], 2)
            self.assertEqual(event["phase"], "decode")
            self.assertEqual(event["outcome"], "ok")
            self.assertIn("host_elapsed_micros", event)
            self.assertEqual(event["workload"]["rows_per_sequence"], {"min": 1, "max": 1})
            self.assertEqual(event["workload"]["context_tokens"], {"min": 16, "max": 32})

    def test_prefill_shape_uses_cpu_extend_lengths(self):
        batch = Batch()
        batch.forward_mode = "extend"
        batch.seq_lens_cpu = [10, 20]
        batch.extend_lens = [4, 8]
        facts = plugin._batch_facts(batch, None)
        self.assertEqual(facts["workload"]["phase"], "prefill")
        self.assertEqual(facts["workload"]["rows_per_sequence"], {"min": 4, "max": 8})
        self.assertEqual(facts["workload"]["context_tokens"], {"min": 6, "max": 12})

    @patch("aletheia_sglang.plugin.Path.open", side_effect=OSError("disk full"))
    def test_trace_failure_does_not_fail_inference(self, _open):
        with patch.dict(os.environ, {"ALETHEIA_TRACE": "trace.jsonl"}):
            with self.assertLogs("aletheia_sglang.plugin", level="ERROR"):
                result = plugin._around_forward_batch_generation(
                    lambda worker, batch, forward_batch: "ok", Worker(), Batch(), None
                )
        self.assertEqual(result, "ok")

    def test_disabled_trace_has_no_side_effect(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                plugin._around_forward_batch_generation(
                    lambda worker, batch, forward_batch: 7, Worker(), Batch(), None
                ),
                7,
            )


if __name__ == "__main__":
    unittest.main()
