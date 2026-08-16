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


class FakeOwner:
    def __init__(self):
        self.commands = []

    def command(self, command):
        self.commands.append(command)
        if command["op"] == "plan_reclamation":
            return {
                "status": "reclamation",
                "certificate": {
                    "schema": "orbitkv.sglang-retirement-certificate.v1",
                    "plan_fingerprint": "fnv1a64:test",
                    "certificate_id": 7,
                    "request_id": command["request_id"],
                    "class_name": "swa",
                    "page_tokens": 16,
                    "token_start": command["observed_evicted_seqlen"],
                    "token_end_exclusive": 32,
                    "semantic_proof": {
                        "kind": "sliding_window",
                        "semantic_frontier": command["semantic_frontier"],
                        "window_tokens": 32,
                        "maximum_reclaimable_end": 32,
                    },
                    "execution_proof": {
                        "kind": "non_overlap_scheduler_barrier",
                        "execution_epoch": command["execution_epoch"],
                    },
                },
            }
        if command["op"] == "commit_reclamations":
            return {
                "status": "committed",
                "certificate_ids": command["certificate_ids"],
            }
        if command["op"] == "release_request":
            return {"status": "released", "request_id": command["request_id"]}
        raise AssertionError(command)


class FakeReqToToken:
    def __getitem__(self, key):
        _, token_range = key
        return FakeTensor(token_range.stop - token_range.start)


class FakeTreeCache:
    page_size = 16

    @staticmethod
    def is_chunk_cache() -> bool:
        return True


class FakeSpecAlgorithm:
    @staticmethod
    def is_none() -> bool:
        return True


class FakeOwningAllocator:
    def __init__(self, events, fail_free=False):
        self.events = events
        self.fail_free = fail_free

    def free_swa(self, value):
        self.events.append(("physical_free", value.numel()))
        if self.fail_free:
            raise RuntimeError("injected physical free failure")


class FakeOwningReq:
    rid = "r0"
    req_pool_idx = 0
    decode_batch_idx = 3
    extend_batch_idx = 0

    def __init__(self):
        self.kv = types.SimpleNamespace(swa_evicted_seqlen=0)


class FakeOwningBatch:
    enable_overlap = False
    tree_cache = FakeTreeCache()
    spec_algorithm = FakeSpecAlgorithm()
    req_to_token_pool = types.SimpleNamespace(req_to_token=FakeReqToToken())

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator


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

    def test_owner_commits_only_after_physical_free_group(self):
        from orbitkv_sglang import plugin

        events = []
        owner = FakeOwner()
        batch = FakeOwningBatch(FakeOwningAllocator(events))
        req = FakeOwningReq()
        old_owner = plugin._OWNER
        old_policy = plugin._POLICY
        old_trace = os.environ.get("ORBITKV_TRACE_ALLOCATIONS")
        try:
            plugin._OWNER = owner
            plugin._POLICY = {
                "plan_fingerprint": "fnv1a64:test",
                "page_tokens": 16,
            }
            os.environ["ORBITKV_TRACE_ALLOCATIONS"] = "0"

            def original(current_batch):
                plugin._own_swa_reclamation(
                    lambda *_args, **_kwargs: None,
                    current_batch,
                    req,
                    64,
                )
                events.append(("free_group_end", None))

            plugin._commit_swa_reclamations(original, batch)
        finally:
            plugin._OWNER = old_owner
            plugin._POLICY = old_policy
            if old_trace is None:
                os.environ.pop("ORBITKV_TRACE_ALLOCATIONS", None)
            else:
                os.environ["ORBITKV_TRACE_ALLOCATIONS"] = old_trace

        self.assertEqual(
            [command["op"] for command in owner.commands],
            ["plan_reclamation", "commit_reclamations"],
        )
        self.assertEqual(events, [("physical_free", 32), ("free_group_end", None)])
        self.assertEqual(req.kv.swa_evicted_seqlen, 32)

    def test_owner_does_not_commit_failed_physical_free(self):
        from orbitkv_sglang import plugin

        owner = FakeOwner()
        batch = FakeOwningBatch(FakeOwningAllocator([], fail_free=True))
        req = FakeOwningReq()
        old_owner = plugin._OWNER
        old_policy = plugin._POLICY
        old_trace = os.environ.get("ORBITKV_TRACE_ALLOCATIONS")
        try:
            plugin._OWNER = owner
            plugin._POLICY = {
                "plan_fingerprint": "fnv1a64:test",
                "page_tokens": 16,
            }
            os.environ["ORBITKV_TRACE_ALLOCATIONS"] = "0"

            def original(current_batch):
                plugin._own_swa_reclamation(
                    lambda *_args, **_kwargs: None,
                    current_batch,
                    req,
                    64,
                )

            with self.assertRaisesRegex(
                RuntimeError, "injected physical free failure"
            ):
                plugin._commit_swa_reclamations(original, batch)
        finally:
            plugin._OWNER = old_owner
            plugin._POLICY = old_policy
            if old_trace is None:
                os.environ.pop("ORBITKV_TRACE_ALLOCATIONS", None)
            else:
                os.environ["ORBITKV_TRACE_ALLOCATIONS"] = old_trace

        self.assertEqual(
            [command["op"] for command in owner.commands],
            ["plan_reclamation"],
        )
        self.assertEqual(req.kv.swa_evicted_seqlen, 0)


if __name__ == "__main__":
    unittest.main()
