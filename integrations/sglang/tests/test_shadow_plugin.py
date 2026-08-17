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


class FakeBaseSWAKVPool:
    pass


class FakeGeneralSwaAllocator:
    def __init__(
        self,
        size,
        size_swa,
        page_size,
        dtype,
        device,
        kvcache,
        need_sort,
    ):
        self._size_full = size
        self._size_swa = size_swa
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

    @property
    def size_full(self):
        return self._size_full

    @property
    def size_swa(self):
        return self._size_swa


class FakePureSwaAllocator(FakeGeneralSwaAllocator):
    pass


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
                    "plan_fingerprint": f"sha256:{'00' * 32}",
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


class PureSWATokenToKVPoolAllocator:
    page_size = 1
    size_swa = 20_000


class WrongUniformAllocator:
    page_size = 1


class FakeOwningAllocator:
    def __init__(self, events, fail_free=False):
        self.events = events
        self.fail_free = fail_free
        self.size_full = 4096
        self.size_swa = 2048

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
        "sglang.srt.mem_cache.base_swa_memory_pool",
    ):
        _module(name)

    allocator_module = sys.modules["sglang.srt.mem_cache.allocator.swa"]
    allocator_module.SWATokenToKVPoolAllocator = FakeSWATokenToKVPoolAllocator
    allocator_module.PureSWATokenToKVPoolAllocator = FakePureSwaAllocator
    sys.modules[
        "sglang.srt.mem_cache.base_swa_memory_pool"
    ].BaseSWAKVPool = FakeBaseSWAKVPool

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
    def test_paged_periodic_allocator_separates_logical_and_physical_slots(self):
        root = Path(os.environ["ORBITKV_SGLANG_ROOT"])
        _load_hook_registry(root)

        from orbitkv_sglang import plugin

        old_contract = plugin._UNIFORM_SWA_CONTRACT
        try:
            plugin._UNIFORM_SWA_CONTRACT = {
                "physical_backend": "paged_periodic",
                "page_tokens": 16,
                "logical_index_tokens": 32_768,
                "minimum_pool_tokens": 19_152,
            }
            allocator_type = plugin._build_paged_periodic_allocator()
            allocator = allocator_type(
                19_152,
                16,
                "bf16",
                "cpu",
                FakeBaseSWAKVPool(),
                False,
            )
            self.assertEqual(allocator.size_full, 32_768)
            self.assertEqual(allocator.size_swa, 19_152)
            self.assertEqual(allocator.page_size, 16)
            with self.assertRaisesRegex(RuntimeError, "below the compiled minimum"):
                allocator_type(
                    19_136,
                    16,
                    "bf16",
                    "cpu",
                    FakeBaseSWAKVPool(),
                    False,
                )
        finally:
            plugin._UNIFORM_SWA_CONTRACT = old_contract

    def test_uniform_swa_state_plan_binds_model_kernel_and_allocator(self):
        from orbitkv_sglang import plugin

        orbitkv_bin = os.environ.get("ORBITKV_BIN")
        if not orbitkv_bin:
            self.skipTest("ORBITKV_BIN is required")
        root = Path(__file__).resolve().parents[3]
        config = root / "fixtures/mistral-uniform-swa-tiny/config.json"
        artifact = json.loads(
            subprocess.check_output(
                [
                    orbitkv_bin,
                    "compile-hf-state-plan",
                    str(config),
                    "--page-tokens",
                    "1",
                    "--kv-dtype-bytes",
                    "2",
                    "--boundary",
                    "8192",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-tokens",
                    "2048",
                    "--eviction-interval",
                    "128",
                    "--decode-headroom-tokens",
                    "32",
                ],
                text=True,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            model_config_path = model / "config.json"
            model_config_path.write_bytes(config.read_bytes())
            artifact_path = Path(directory) / "state-plan.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            old_path = os.environ.get("ORBITKV_SGLANG_STATE_PLAN")
            old_contract = plugin._UNIFORM_SWA_CONTRACT
            old_mode = plugin._STATE_PLAN_MODE
            try:
                os.environ["ORBITKV_SGLANG_STATE_PLAN"] = str(artifact_path)
                plugin._load_state_plan()
                tampered = json.loads(json.dumps(artifact))
                tampered["sglang_lowering"]["contract"][
                    "maximum_running_requests"
                ] += 1
                artifact_path.write_text(
                    json.dumps(tampered), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    plugin._load_state_plan()
                artifact_path.write_text(
                    json.dumps(artifact), encoding="utf-8"
                )
                plugin._load_state_plan()
                model_config = types.SimpleNamespace(
                    model_path=str(model),
                    hf_config=types.SimpleNamespace(
                        architectures=["MistralForCausalLM"]
                    ),
                    hf_text_config=types.SimpleNamespace(num_hidden_layers=4),
                    sliding_window_size=4096,
                    is_hybrid_swa=False,
                )
                plugin._activate_uniform_swa_model_config(
                    lambda *_args, **_kwargs: None,
                    model_config,
                )
                self.assertTrue(model_config.is_hybrid_swa)
                self.assertFalse(model_config.is_deepseek_v4_arch)
                self.assertEqual(model_config.sliding_window_size, 4095)
                self.assertEqual(model_config.swa_attention_layer_ids, [0, 1, 2, 3])
                self.assertEqual(model_config.full_attention_layer_ids, [])

                attention = types.SimpleNamespace(
                    total_num_kv_heads=8,
                    head_dim=128,
                    attn=types.SimpleNamespace(sliding_window_size=-1),
                )
                plugin._activate_uniform_swa_kernel(
                    lambda *_args, **_kwargs: None,
                    attention,
                    types.SimpleNamespace(
                        architectures=["MistralForCausalLM"]
                    ),
                )
                self.assertEqual(attention.attn.sliding_window_size, 4095)

                server_args = types.SimpleNamespace(
                    disable_radix_cache=True,
                    disable_overlap_schedule=True,
                    disable_cuda_graph=True,
                    disaggregation_mode="null",
                    max_running_requests=1,
                    chunked_prefill_size=2048,
                )
                configurator = types.SimpleNamespace(
                    server_args=server_args,
                    spec_algorithm=FakeSpecAlgorithm(),
                    page_size=1,
                )
                result = types.SimpleNamespace(
                    token_to_kv_pool_allocator=PureSWATokenToKVPoolAllocator()
                )
                returned = plugin._validate_uniform_swa_runtime(
                    lambda *_args, **_kwargs: result,
                    configurator,
                )
                self.assertIs(returned, result)
                too_many_requests = types.SimpleNamespace(
                    server_args=types.SimpleNamespace(
                        disable_radix_cache=True,
                        disable_overlap_schedule=True,
                        disable_cuda_graph=True,
                        disaggregation_mode="null",
                        max_running_requests=5,
                        chunked_prefill_size=2048,
                    ),
                    spec_algorithm=FakeSpecAlgorithm(),
                    page_size=1,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "maximum_running_requests"
                ):
                    plugin._validate_uniform_swa_runtime(
                        lambda *_args, **_kwargs: result,
                        too_many_requests,
                    )
                with self.assertRaisesRegex(
                    RuntimeError, "compiled minimum"
                ):
                    below_minimum_allocator = PureSWATokenToKVPoolAllocator()
                    below_minimum_allocator.size_swa = 19_076
                    plugin._validate_uniform_swa_runtime(
                        lambda *_args, **_kwargs: types.SimpleNamespace(
                            token_to_kv_pool_allocator=below_minimum_allocator
                        ),
                        configurator,
                    )

                worker = types.SimpleNamespace(
                    model_config=types.SimpleNamespace(context_len=16384)
                )
                worker_info = plugin._expand_uniform_swa_worker_info(
                    lambda *_args, **_kwargs: (
                        8192,
                        16384,
                        1,
                        1,
                        8191,
                        8186,
                        0,
                        "cuda",
                        None,
                        1,
                        16384,
                        8192,
                    ),
                    worker,
                )
                self.assertEqual(worker_info[4], 16383)
                self.assertEqual(worker_info[5], 16378)

                request = types.SimpleNamespace(
                    origin_input_ids=list(range(12000)),
                    sampling_params=types.SimpleNamespace(
                        max_new_tokens=8,
                        min_new_tokens=4,
                    ),
                )
                scheduler = types.SimpleNamespace(
                    max_new_tokens_limit=None,
                    max_req_len=16383,
                )
                plugin._init_uniform_swa_max_new_tokens(
                    lambda _scheduler, req: setattr(
                        req.sampling_params, "max_new_tokens", 0
                    ),
                    scheduler,
                    request,
                )
                self.assertEqual(request.sampling_params.max_new_tokens, 8)
                self.assertEqual(request.sampling_params.min_new_tokens, 4)

                admission_request = types.SimpleNamespace(
                    full_untruncated_fill_ids=list(range(12000)),
                    prefix_indices=[],
                    sampling_params=types.SimpleNamespace(max_new_tokens=8),
                )
                class FakePureSwaAdder:
                    is_all_swa = True
                    max_running_requests = 1
                    page_size = 1

                    def __init__(self):
                        self.rem_total_token_offset = 0

                    @property
                    def rem_total_tokens(self):
                        return 8192 - self.rem_total_token_offset

                adder = FakePureSwaAdder()

                def admit(current_adder, *_args, **_kwargs):
                    self.assertGreater(current_adder.rem_total_tokens, 12009)
                    current_adder.rem_total_token_offset += 2048
                    return "admitted"

                self.assertEqual(
                    plugin._admit_uniform_swa_request(
                        admit,
                        adder,
                        admission_request,
                        False,
                        None,
                    ),
                    "admitted",
                )
                self.assertEqual(adder.rem_total_token_offset, 2048)
                with self.assertRaisesRegex(
                    RuntimeError, "did not produce its compiled allocator"
                ):
                    plugin._validate_uniform_swa_runtime(
                        lambda *_args, **_kwargs: types.SimpleNamespace(
                            token_to_kv_pool_allocator=WrongUniformAllocator()
                        ),
                        configurator,
                    )

                model_config_path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "config hash"):
                    plugin._activate_uniform_swa_model_config(
                        lambda *_args, **_kwargs: None,
                        model_config,
                    )

                model_config_path.write_bytes(config.read_bytes())
                plugin._STATE_PLAN_MODE = "kernel_reference"
                reference_model_config = types.SimpleNamespace(
                    model_path=str(model),
                    hf_config=types.SimpleNamespace(
                        architectures=["MistralForCausalLM"]
                    ),
                    hf_text_config=types.SimpleNamespace(num_hidden_layers=4),
                    sliding_window_size=4096,
                    is_hybrid_swa=False,
                )
                plugin._activate_uniform_swa_model_config(
                    lambda *_args, **_kwargs: None,
                    reference_model_config,
                )
                self.assertFalse(reference_model_config.is_hybrid_swa)
                self.assertEqual(reference_model_config.sliding_window_size, 4095)
                self.assertEqual(
                    plugin._resolve_uniform_swa_window(
                        lambda *_args, **_kwargs: None,
                        object(),
                        reference_model_config,
                    ),
                    4095,
                )
                reference_result = types.SimpleNamespace(
                    token_to_kv_pool_allocator=WrongUniformAllocator()
                )
                self.assertIs(
                    plugin._validate_uniform_swa_runtime(
                        lambda *_args, **_kwargs: reference_result,
                        configurator,
                    ),
                    reference_result,
                )
            finally:
                plugin._STATE_PLAN_MODE = old_mode
                plugin._UNIFORM_SWA_CONTRACT = old_contract
                if old_path is None:
                    os.environ.pop("ORBITKV_SGLANG_STATE_PLAN", None)
                else:
                    os.environ["ORBITKV_SGLANG_STATE_PLAN"] = old_path

    def test_loads_and_enforces_compiled_physical_plan(self):
        from orbitkv_sglang import plugin

        orbitkv_bin = os.environ.get("ORBITKV_BIN")
        if not orbitkv_bin:
            self.skipTest("ORBITKV_BIN is required")
        config = Path(__file__).resolve().parents[3] / (
            "fixtures/gpt-oss-hybrid-tiny/config.json"
        )
        command = [
            orbitkv_bin,
            "compile-hf-physical-plan",
            str(config),
            "--page-tokens",
            "16",
            "--kv-dtype-bytes",
            "2",
            "--available-kv-bytes",
            "120881152",
            "--max-running-requests",
            "8",
            "--attention-dp-size",
            "1",
            "--chunked-prefill-tokens",
            "2048",
            "--workload-requests",
            "8",
            "--prompt-tokens",
            "6000",
            "--decode-tokens",
            "32",
            "--candidate-intervals",
            "16,32,64,128",
            "--max-reclamation-calls",
            "4",
            "--min-admitted-requests",
            "8",
            "--objective",
            "capacity",
        ]
        artifact = json.loads(subprocess.check_output(command, text=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical-plan.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            old_path = os.environ.get("ORBITKV_SGLANG_PHYSICAL_PLAN")
            old_interval = os.environ.get("ORBITKV_SGLANG_EVICTION_INTERVAL")
            old_physical = plugin._PHYSICAL_PLAN
            old_policy = plugin._POLICY
            try:
                os.environ["ORBITKV_SGLANG_PHYSICAL_PLAN"] = str(path)
                os.environ.pop("ORBITKV_SGLANG_EVICTION_INTERVAL", None)
                policy = plugin._load_policy()
                plugin._POLICY = policy
                self.assertEqual(policy["swa_eviction_interval_tokens"], 32)
                selected_cost = artifact["physical_plan"]["selected"]["cost"]
                allocator = FakeOwningAllocator([])
                allocator.size_full = selected_cost["full_token_capacity"]
                allocator.size_swa = selected_cost["physical_swa_token_slots"]
                batch = FakeOwningBatch(allocator)
                plugin._validate_physical_contract(batch)

                allocator.size_full += 16
                with self.assertRaisesRegex(
                    RuntimeError, "Full capacity does not match"
                ):
                    plugin._validate_physical_contract(batch)
            finally:
                plugin._PHYSICAL_PLAN = old_physical
                plugin._POLICY = old_policy
                if old_path is None:
                    os.environ.pop("ORBITKV_SGLANG_PHYSICAL_PLAN", None)
                else:
                    os.environ["ORBITKV_SGLANG_PHYSICAL_PLAN"] = old_path
                if old_interval is None:
                    os.environ.pop("ORBITKV_SGLANG_EVICTION_INTERVAL", None)
                else:
                    os.environ["ORBITKV_SGLANG_EVICTION_INTERVAL"] = old_interval

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
                "plan_fingerprint": f"sha256:{'00' * 32}",
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
                "plan_fingerprint": f"sha256:{'00' * 32}",
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

    def test_ffi_and_sidecar_owner_transports_are_equivalent(self):
        from orbitkv_sglang import plugin

        orbitkv_bin = os.environ.get("ORBITKV_BIN")
        orbitkv_plan = os.environ.get("ORBITKV_PLAN")
        owner_library = os.environ.get("ORBITKV_OWNER_LIB")
        if not orbitkv_bin or not orbitkv_plan or not owner_library:
            self.skipTest(
                "ORBITKV_BIN, ORBITKV_PLAN, and ORBITKV_OWNER_LIB are required"
            )

        old_bin = os.environ.get("ORBITKV_BIN")
        old_policy = os.environ.get("ORBITKV_SGLANG_POLICY")
        try:
            os.environ["ORBITKV_BIN"] = orbitkv_bin
            os.environ["ORBITKV_SGLANG_POLICY"] = orbitkv_plan
            policy = plugin._load_policy()
        finally:
            if old_bin is None:
                os.environ.pop("ORBITKV_BIN", None)
            else:
                os.environ["ORBITKV_BIN"] = old_bin
            if old_policy is None:
                os.environ.pop("ORBITKV_SGLANG_POLICY", None)
            else:
                os.environ["ORBITKV_SGLANG_POLICY"] = old_policy

        ffi = plugin.FfiOwnerClient(owner_library, orbitkv_plan, policy)
        sidecar = plugin.SidecarOwnerClient(orbitkv_bin, orbitkv_plan)
        try:
            plan_command = {
                "op": "plan_reclamation",
                "request_id": "r0",
                "observed_evicted_seqlen": 0,
                "semantic_frontier": 2048,
                "execution_epoch": 7,
                "cache_kind": "chunk",
            }
            ffi_plan = ffi.command(plan_command)
            sidecar_plan = sidecar.command(plan_command)
            self.assertEqual(ffi_plan, sidecar_plan)
            certificate_id = ffi_plan["certificate"]["certificate_id"]

            commit_command = {
                "op": "commit_reclamations",
                "certificate_ids": [certificate_id],
            }
            self.assertEqual(
                ffi.command(commit_command),
                sidecar.command(commit_command),
            )
            self.assertEqual(
                ffi.command({"op": "stats"}),
                sidecar.command({"op": "stats"}),
            )
            release_command = {"op": "release_request", "request_id": "r0"}
            self.assertEqual(
                ffi.command(release_command),
                sidecar.command(release_command),
            )
        finally:
            ffi.close()
            sidecar.close()


if __name__ == "__main__":
    unittest.main()
