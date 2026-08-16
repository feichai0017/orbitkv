from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


class FakeTensor:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size


class FakeSWATokenToKVPoolAllocator:
    page_size = 16
    size_full = 4096
    size_swa = 2048

    def __init__(self):
        self.full_available = self.size_full
        self.swa_available = self.size_swa

    def full_available_size(self) -> int:
        return self.full_available

    def swa_available_size(self) -> int:
        return self.swa_available

    def available_size(self) -> int:
        return min(self.full_available, self.swa_available)

    def alloc(self, size: int) -> FakeTensor:
        self.full_available -= size
        self.swa_available -= size
        return FakeTensor(size)

    def alloc_extend(self, *args, **kwargs) -> FakeTensor:
        self.full_available -= 32
        self.swa_available -= 32
        return FakeTensor(32)

    def alloc_decode(self, *args, **kwargs) -> FakeTensor:
        self.full_available -= 16
        self.swa_available -= 16
        return FakeTensor(16)

    def free(self, value: FakeTensor) -> None:
        self.full_available += value.numel()
        self.swa_available += value.numel()

    def free_swa(self, value: FakeTensor) -> None:
        self.swa_available += value.numel()


def _module(name: str, *, package: bool = True) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    if "." in name:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], attribute, module)
    return module


def _load_hook_registry(sglang_root: Path):
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.plugins",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.mem_cache.allocator.swa",
    ):
        _module(name)

    allocator_module = sys.modules["sglang.srt.mem_cache.allocator.swa"]
    allocator_module.SWATokenToKVPoolAllocator = FakeSWATokenToKVPoolAllocator

    path = sglang_root / "python/sglang/srt/plugins/hook_registry.py"
    spec = importlib.util.spec_from_file_location(
        "sglang.srt.plugins.hook_registry", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SGLang hook registry from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules["sglang.srt.plugins"].hook_registry = module
    spec.loader.exec_module(module)
    return module


class ShadowPluginTests(unittest.TestCase):
    def test_loads_validated_policy_from_rust(self):
        from orbitkv_sglang import plugin

        orbitkv_bin = os.environ.get("ORBITKV_BIN")
        orbitkv_plan = os.environ.get("ORBITKV_PLAN")
        if not orbitkv_bin or not orbitkv_plan:
            self.skipTest("ORBITKV_BIN and ORBITKV_PLAN are required")

        old_values = {
            name: os.environ.get(name)
            for name in (
                "ORBITKV_BIN",
                "ORBITKV_SGLANG_POLICY",
                "ORBITKV_SGLANG_EVICTION_INTERVAL",
            )
        }
        try:
            os.environ["ORBITKV_BIN"] = orbitkv_bin
            os.environ["ORBITKV_SGLANG_POLICY"] = orbitkv_plan
            os.environ["ORBITKV_SGLANG_EVICTION_INTERVAL"] = "32"
            policy = plugin._load_policy()
        finally:
            for name, value in old_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.assertEqual(policy["schema"], "orbitkv.sglang-policy.v1")
        self.assertEqual(policy["swa_eviction_interval_tokens"], 32)
        self.assertEqual(policy["max_persistent_swa_token_slots_per_request"], 1040)

    def test_real_sglang_hook_registry_preserves_allocator_results(self):
        root = Path(os.environ["ORBITKV_SGLANG_ROOT"])
        registry_module = _load_hook_registry(root)

        from orbitkv_sglang import plugin

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            os.environ["ORBITKV_TRACE_PATH"] = str(trace_path)
            plugin.register()
            registry_module.HookRegistry.apply_hooks()

            allocator = FakeSWATokenToKVPoolAllocator()
            allocated = allocator.alloc(64)
            extended = allocator.alloc_extend()
            decoded = allocator.alloc_decode()
            allocator.free(decoded)
            allocator.free_swa(FakeTensor(16))
            plugin._stop_writer()

            self.assertEqual(allocated.numel(), 64)
            self.assertEqual(extended.numel(), 32)
            self.assertEqual(decoded.numel(), 16)

            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "plugin_loaded",
                    "alloc",
                    "alloc_extend",
                    "alloc_decode",
                    "free",
                    "free_swa",
                ],
            )
            self.assertEqual(events[1]["output_tokens"], 64)
            self.assertEqual(events[1]["swa_available_before"], 2048)
            self.assertEqual(events[1]["swa_available_after"], 1984)

            orbitkv_bin = os.environ.get("ORBITKV_BIN")
            orbitkv_plan = os.environ.get("ORBITKV_PLAN")
            if orbitkv_bin and orbitkv_plan:
                completed = subprocess.run(
                    [
                        orbitkv_bin,
                        "analyze-sglang",
                        orbitkv_plan,
                        str(trace_path),
                        "--max-active-requests",
                        "1",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                summary = json.loads(completed.stdout)
                self.assertEqual(summary["events"], 6)
                self.assertEqual(summary["peak_swa_used_tokens"], 112)
                self.assertEqual(summary["peak_full_used_tokens"], 112)
                self.assertEqual(summary["minimum_expected_swa_slots"], 1040)


if __name__ == "__main__":
    unittest.main()
