from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


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


class FakeCapsules:
    def __init__(self, events):
        self.events = events
        self.commands = []

    def command(self, command):
        self.commands.append(command)
        if command["op"] == "restore":
            return {"status": "miss"}
        self.events.append(("capsule_publish", None))
        return {
            "status": "published",
            "prefix_token_count": len(command["token_ids"]),
            "payload_bytes": Path(command["payload_path"]).stat().st_size,
            "created": True,
        }


class FakeHydrationCapsules:
    def __init__(
        self,
        payload_path: Path,
        prefix_tokens: int,
        live_tokens: int | None = None,
    ):
        payload = payload_path.read_bytes()
        digest = list(hashlib.sha256(payload).digest())
        live_tokens = prefix_tokens if live_tokens is None else live_tokens
        self.commands = []
        self.response = {
            "status": "restored",
            "manifest": {
                "schema": "orbitkv.continuation-capsule.v1",
                "prefix_token_count": prefix_tokens,
                "live_token_count": live_tokens,
                "payload_bytes": len(payload),
                "payload_digest": digest,
                "components": [
                    {
                        "state_class": "sglang-kv",
                        "offset_bytes": 0,
                        "length_bytes": len(payload),
                        "checksum": digest,
                    }
                ],
            },
            "payload_path": str(payload_path),
        }

    def command(self, command):
        self.commands.append(command)
        return self.response


class FakeExistingCapsules:
    def __init__(self, prefix_tokens):
        self.prefix_tokens = prefix_tokens
        self.commands = []

    def command(self, command):
        self.commands.append(command)
        return {
            "status": "restored",
            "manifest": {
                "schema": "orbitkv.continuation-capsule.v1",
                "prefix_token_count": self.prefix_tokens,
            },
            "payload_path": "/not-read-for-publication-reuse",
        }


class FakeHydrationAllocator:
    page_size = 16
    device = "cpu"

    def __init__(self, fail_load=False, hybrid=True):
        self.fail_load = fail_load
        self.loaded = []
        self.freed = []
        self.tail_lengths = []
        self._kvcache = types.SimpleNamespace(
            full_kv_pool=object() if hybrid else None
        )

    def alloc_extend(
        self,
        _prefix_lens,
        _prefix_lens_cpu,
        _seq_lens,
        _seq_lens_cpu,
        _last_loc,
        extend_num_tokens,
    ):
        return torch.arange(1, extend_num_tokens + 1, dtype=torch.int64)

    def alloc_extend_swa_tail(
        self,
        _prefix_lens,
        _prefix_lens_cpu,
        _seq_lens,
        _seq_lens_cpu,
        _last_loc,
        extend_num_tokens,
        swa_tail_len,
    ):
        self.tail_lengths.append(swa_tail_len)
        return torch.arange(1, extend_num_tokens + 1, dtype=torch.int64)

    def load_cpu_copy(self, value, indices, mamba_indices=None):
        if self.fail_load:
            raise RuntimeError("injected capsule load failure")
        self.loaded.append((value, indices.clone(), mamba_indices))

    def free(self, indices):
        self.freed.append(indices.clone())


class FakeHydrationReq:
    rid = "hydrate-r0"
    req_pool_idx = None
    kv = None
    mamba_pool_idx = None
    is_retracted = False
    cache_protected_len = 0

    def __init__(self, input_tokens=65):
        self.full_untruncated_fill_ids = list(range(input_tokens))
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)

    @staticmethod
    def _compute_max_prefix_len(input_len):
        return max(input_len - 1, 0)


class FakeReqToToken:
    def __getitem__(self, key):
        _, token_range = key
        start = 0 if token_range.start is None else token_range.start
        return FakeTensor(token_range.stop - start)


class FakeTreeCache:
    page_size = 16

    @staticmethod
    def is_chunk_cache() -> bool:
        return True


class FakeSpecAlgorithm:
    @staticmethod
    def is_none() -> bool:
        return True


def fake_cuda_graph_config(decode="disabled", prefill="disabled", batch_sizes=None):
    return types.SimpleNamespace(
        decode=types.SimpleNamespace(
            backend=decode,
            bs=[] if batch_sizes is None else batch_sizes,
        ),
        prefill=types.SimpleNamespace(backend=prefill),
    )


class PureSWATokenToKVPoolAllocator:
    page_size = 1
    size_swa = 20_000


class WrongUniformAllocator:
    page_size = 1


class OrbitKvPagedPeriodicAllocator:
    def __init__(self, page_size, size_swa, size_full):
        self.page_size = page_size
        self.size_swa = size_swa
        self.size_full = size_full


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
        self.kv_committed_len = 4
        self.origin_input_ids = [1, 2, 3, 4]
        self.output_ids = []
        self.mamba_pool_idx = None

    def effective_kv_committed_len(self):
        return self.kv_committed_len

    def finished(self):
        return True


class FakeOwningBatch:
    enable_overlap = False
    tree_cache = FakeTreeCache()
    spec_algorithm = FakeSpecAlgorithm()
    req_to_token_pool = types.SimpleNamespace(req_to_token=FakeReqToToken())

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator


class FakeCapsuleAllocator(FakeOwningAllocator):
    def __init__(self, events):
        super().__init__(events)
        self._kvcache = types.SimpleNamespace(full_kv_pool=object())

    def get_cpu_copy(self, indices, mamba_indices=None):
        self.events.append(("cpu_copy", indices.numel()))
        return {"fake": True}


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
    def test_decode_graph_contract_validates_runtime_and_records_replay(self):
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
                    "16",
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
                    "--cuda-graph-mode",
                    "decode",
                ],
                text=True,
            )
        )
        contract = artifact["sglang_lowering"]["contract"]
        old_contract = plugin._UNIFORM_SWA_CONTRACT
        old_mode = plugin._STATE_PLAN_MODE
        try:
            plugin._UNIFORM_SWA_CONTRACT = contract
            plugin._STATE_PLAN_MODE = "execute"
            server_args = types.SimpleNamespace(
                disable_radix_cache=True,
                disable_overlap_schedule=True,
                disaggregation_mode="null",
                max_running_requests=4,
                chunked_prefill_size=2048,
                cuda_graph_config=fake_cuda_graph_config(
                    decode="full",
                    prefill="disabled",
                    batch_sizes=[1, 2, 3, 4],
                ),
            )
            configurator = types.SimpleNamespace(
                server_args=server_args,
                spec_algorithm=FakeSpecAlgorithm(),
                page_size=16,
            )
            allocator = OrbitKvPagedPeriodicAllocator(
                page_size=16,
                size_swa=contract["minimum_pool_tokens"],
                size_full=contract["logical_index_tokens"],
            )
            result = types.SimpleNamespace(token_to_kv_pool_allocator=allocator)
            self.assertIs(
                plugin._validate_uniform_swa_runtime(
                    lambda *_args, **_kwargs: result,
                    configurator,
                ),
                result,
            )

            events = []
            original_emit = plugin._emit
            plugin._emit = events.append
            try:
                replay_result = plugin._record_decode_graph_replay(
                    lambda *_args, **_kwargs: "replayed",
                    object(),
                    types.SimpleNamespace(
                        batch_size=4,
                        forward_mode="DECODE",
                    ),
                )
            finally:
                plugin._emit = original_emit
            self.assertEqual(replay_result, "replayed")
            self.assertEqual(events[0]["event"], "decode_graph_replay")
            self.assertEqual(events[0]["batch_size"], 4)
        finally:
            plugin._UNIFORM_SWA_CONTRACT = old_contract
            plugin._STATE_PLAN_MODE = old_mode

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
                    "--cuda-graph-mode",
                    "disabled",
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
                    cuda_graph_config=fake_cuda_graph_config(),
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
                        cuda_graph_config=fake_cuda_graph_config(),
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

    def test_capsule_exports_before_physical_release(self):
        from orbitkv_sglang import plugin

        events = []
        owner = FakeOwner()
        capsules = FakeCapsules(events)
        allocator = FakeCapsuleAllocator(events)
        tree_cache = types.SimpleNamespace(
            is_chunk_cache=lambda: True,
            req_to_token_pool=types.SimpleNamespace(req_to_token=FakeReqToToken()),
            token_to_kv_pool_allocator=allocator,
        )
        req = FakeOwningReq()
        old_capsules = plugin._CAPSULES
        old_owner = plugin._OWNER
        old_encoder = plugin._encode_capsule_payload
        environment = {
            "ORBITKV_CAPSULE_STORE": tempfile.mkdtemp(),
            "ORBITKV_CAPSULE_CHUNK_TOKENS": "4",
            "ORBITKV_TRACE_ALLOCATIONS": "0",
            "ORBITKV_CAPSULE_IDENTITY": json.dumps(
                {
                    "namespace": "dGVuYW50",
                    "model_fingerprint": f"sha256:{'01' * 32}",
                    "tokenizer_fingerprint": f"sha256:{'02' * 32}",
                    "adapter_fingerprint": f"sha256:{'03' * 32}",
                    "state_plan_fingerprint": f"sha256:{'04' * 32}",
                }
            ),
        }
        old_environment = {key: os.environ.get(key) for key in environment}
        try:
            plugin._CAPSULES = capsules
            plugin._OWNER = owner
            plugin._encode_capsule_payload = lambda value: b"wire"
            os.environ.update(environment)

            def original(released_req, released_cache):
                self.assertIs(released_req, req)
                self.assertIs(released_cache, tree_cache)
                events.append(("physical_release", None))
                return "released"

            result = plugin._release_owned_request(
                original,
                req,
                tree_cache,
            )
        finally:
            plugin._CAPSULES = old_capsules
            plugin._OWNER = old_owner
            plugin._encode_capsule_payload = old_encoder
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result, "released")
        self.assertEqual(
            events,
            [
                ("cpu_copy", 4),
                ("capsule_publish", None),
                ("physical_release", None),
            ],
        )
        self.assertEqual(
            [command["op"] for command in owner.commands],
            ["release_request"],
        )
        self.assertEqual(
            [command["op"] for command in capsules.commands],
            ["restore", "publish"],
        )
        command = capsules.commands[1]
        self.assertEqual(command["token_ids"], [1, 2, 3, 4])
        self.assertEqual(command["components"][0]["length_bytes"], 4)

    def test_capsule_skips_non_insert_release(self):
        from orbitkv_sglang import plugin

        req = FakeOwningReq()
        capsules = FakeCapsules([])
        owner = FakeOwner()
        old_capsules = plugin._CAPSULES
        old_owner = plugin._OWNER
        try:
            plugin._CAPSULES = capsules
            plugin._OWNER = owner
            result = plugin._release_owned_request(
                lambda *_args, **_kwargs: "released",
                req,
                types.SimpleNamespace(),
                is_insert=False,
            )
        finally:
            plugin._CAPSULES = old_capsules
            plugin._OWNER = old_owner
        self.assertEqual(result, "released")
        self.assertEqual(capsules.commands, [])
        self.assertEqual(
            [command["op"] for command in owner.commands],
            ["release_request"],
        )

    def test_capsule_does_not_republish_unchanged_hydrated_boundary(self):
        from orbitkv_sglang import plugin

        req = FakeOwningReq()
        req._orbitkv_capsule_prefix_tokens = 4
        capsules = FakeCapsules([])
        owner = FakeOwner()
        old_capsules = plugin._CAPSULES
        old_owner = plugin._OWNER
        environment = {
            "ORBITKV_CAPSULE_STORE": tempfile.mkdtemp(),
            "ORBITKV_CAPSULE_CHUNK_TOKENS": "4",
        }
        old_environment = {key: os.environ.get(key) for key in environment}
        try:
            plugin._CAPSULES = capsules
            plugin._OWNER = owner
            os.environ.update(environment)
            result = plugin._release_owned_request(
                lambda *_args, **_kwargs: "released",
                req,
                types.SimpleNamespace(is_chunk_cache=lambda: True),
            )
        finally:
            plugin._CAPSULES = old_capsules
            plugin._OWNER = old_owner
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result, "released")
        self.assertEqual(capsules.commands, [])
        self.assertEqual(
            [command["op"] for command in owner.commands],
            ["release_request"],
        )

    def test_capsule_reuses_existing_published_boundary(self):
        from orbitkv_sglang import plugin

        req = FakeOwningReq()
        capsules = FakeExistingCapsules(4)
        owner = FakeOwner()
        old_capsules = plugin._CAPSULES
        old_owner = plugin._OWNER
        environment = {
            "ORBITKV_CAPSULE_STORE": tempfile.mkdtemp(),
            "ORBITKV_CAPSULE_CHUNK_TOKENS": "4",
            "ORBITKV_TRACE_ALLOCATIONS": "0",
            "ORBITKV_CAPSULE_IDENTITY": json.dumps(
                {
                    "namespace": "dGVuYW50",
                    "model_fingerprint": f"sha256:{'01' * 32}",
                    "tokenizer_fingerprint": f"sha256:{'02' * 32}",
                    "adapter_fingerprint": f"sha256:{'03' * 32}",
                    "state_plan_fingerprint": f"sha256:{'04' * 32}",
                }
            ),
        }
        old_environment = {key: os.environ.get(key) for key in environment}
        try:
            plugin._CAPSULES = capsules
            plugin._OWNER = owner
            os.environ.update(environment)
            result = plugin._release_owned_request(
                lambda *_args, **_kwargs: "released",
                req,
                types.SimpleNamespace(is_chunk_cache=lambda: True),
            )
        finally:
            plugin._CAPSULES = old_capsules
            plugin._OWNER = old_owner
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result, "released")
        self.assertEqual(
            [command["op"] for command in capsules.commands],
            ["restore"],
        )
        self.assertEqual(
            [command["op"] for command in owner.commands],
            ["release_request"],
        )

    def test_pure_swa_capsule_exports_only_live_tail(self):
        from orbitkv_sglang import plugin

        events = []
        owner = FakeOwner()
        capsules = FakeCapsules(events)
        allocator = FakeCapsuleAllocator(events)
        tree_cache = types.SimpleNamespace(
            is_chunk_cache=lambda: True,
            req_to_token_pool=types.SimpleNamespace(req_to_token=FakeReqToToken()),
            token_to_kv_pool_allocator=allocator,
        )
        req = FakeOwningReq()
        req.kv_committed_len = 64
        req.origin_input_ids = list(range(64))
        old_capsules = plugin._CAPSULES
        old_owner = plugin._OWNER
        old_policy = plugin._POLICY
        old_encoder = plugin._encode_capsule_payload
        environment = {
            "ORBITKV_CAPSULE_STORE": tempfile.mkdtemp(),
            "ORBITKV_CAPSULE_CHUNK_TOKENS": "16",
            "ORBITKV_TRACE_ALLOCATIONS": "0",
            "ORBITKV_CAPSULE_IDENTITY": json.dumps(
                {
                    "namespace": "dGVuYW50",
                    "model_fingerprint": f"sha256:{'01' * 32}",
                    "tokenizer_fingerprint": f"sha256:{'02' * 32}",
                    "adapter_fingerprint": f"sha256:{'03' * 32}",
                    "state_plan_fingerprint": f"sha256:{'04' * 32}",
                }
            ),
        }
        old_environment = {key: os.environ.get(key) for key in environment}
        try:
            plugin._CAPSULES = capsules
            plugin._OWNER = owner
            plugin._POLICY = {
                "page_tokens": 16,
                "unbounded_classes": [],
                "bounded_classes": [{"name": "swa", "window_tokens": 32}],
            }
            plugin._encode_capsule_payload = lambda value: b"x" * value["tokens"]
            os.environ.update(environment)
            allocator.get_cpu_copy = lambda indices, mamba_indices=None: {
                "tokens": indices.numel()
            }
            result = plugin._release_owned_request(
                lambda *_args, **_kwargs: "released",
                req,
                tree_cache,
            )
        finally:
            plugin._CAPSULES = old_capsules
            plugin._OWNER = old_owner
            plugin._POLICY = old_policy
            plugin._encode_capsule_payload = old_encoder
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result, "released")
        publish = capsules.commands[1]
        self.assertEqual(publish["token_ids"], list(range(64)))
        self.assertEqual(publish["live_token_count"], 32)
        self.assertEqual(publish["components"][0]["length_bytes"], 32)

    def test_capsule_hydration_commits_only_after_admission(self):
        from orbitkv_sglang import plugin
        from orbitkv_sglang.capsule_wire import encode_cpu_tensors

        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capsule.payload"
            payload_path.write_bytes(
                encode_cpu_tensors({"full": [[torch.ones((64, 1))]], "swa": None})
            )
            capsules = FakeHydrationCapsules(payload_path, 64)
            allocator = FakeHydrationAllocator()
            tree_cache = types.SimpleNamespace(
                is_chunk_cache=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
            adder = types.SimpleNamespace(tree_cache=tree_cache, can_run_list=[])
            req = FakeHydrationReq()
            old_capsules = plugin._CAPSULES
            old_policy = plugin._POLICY
            environment = {
                "ORBITKV_CAPSULE_STORE": directory,
                "ORBITKV_CAPSULE_CHUNK_TOKENS": "16",
                "ORBITKV_TRACE_ALLOCATIONS": "0",
                "ORBITKV_CAPSULE_IDENTITY": json.dumps(
                    {
                        "namespace": "dGVuYW50",
                        "model_fingerprint": f"sha256:{'01' * 32}",
                        "tokenizer_fingerprint": f"sha256:{'02' * 32}",
                        "adapter_fingerprint": f"sha256:{'03' * 32}",
                        "state_plan_fingerprint": f"sha256:{'04' * 32}",
                    }
                ),
            }
            old_environment = {key: os.environ.get(key) for key in environment}
            try:
                plugin._CAPSULES = capsules
                plugin._POLICY = {
                    "page_tokens": 16,
                    "unbounded_classes": ["full"],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}]
                }
                os.environ.update(environment)

                def original(current_adder, current_req):
                    self.assertEqual(len(current_req.prefix_indices), 64)
                    current_adder.can_run_list.append(current_req)
                    return "continue"

                result = plugin._hydrate_capsule_for_admission(
                    original, adder, req
                )
            finally:
                plugin._CAPSULES = old_capsules
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(result, "continue")
        self.assertEqual(allocator.tail_lengths, [32])
        self.assertEqual(len(allocator.loaded), 1)
        self.assertEqual(allocator.freed, [])
        self.assertEqual(req._orbitkv_capsule_prefix_tokens, 64)
        self.assertEqual(capsules.commands[0]["token_ids"], list(range(64)))

    def test_capsule_hydration_rolls_back_rejected_admission(self):
        from orbitkv_sglang import plugin
        from orbitkv_sglang.capsule_wire import encode_cpu_tensors

        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capsule.payload"
            payload_path.write_bytes(encode_cpu_tensors([torch.ones((64, 1))]))
            capsules = FakeHydrationCapsules(payload_path, 64)
            allocator = FakeHydrationAllocator()
            tree_cache = types.SimpleNamespace(
                is_chunk_cache=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
            adder = types.SimpleNamespace(tree_cache=tree_cache, can_run_list=[])
            req = FakeHydrationReq()
            old_capsules = plugin._CAPSULES
            old_policy = plugin._POLICY
            environment = {
                "ORBITKV_CAPSULE_STORE": directory,
                "ORBITKV_CAPSULE_CHUNK_TOKENS": "16",
                "ORBITKV_TRACE_ALLOCATIONS": "0",
                "ORBITKV_CAPSULE_IDENTITY": json.dumps(
                    {
                        "namespace": "dGVuYW50",
                        "model_fingerprint": f"sha256:{'01' * 32}",
                        "tokenizer_fingerprint": f"sha256:{'02' * 32}",
                        "adapter_fingerprint": f"sha256:{'03' * 32}",
                        "state_plan_fingerprint": f"sha256:{'04' * 32}",
                    }
                ),
            }
            old_environment = {key: os.environ.get(key) for key in environment}
            try:
                plugin._CAPSULES = capsules
                plugin._POLICY = {
                    "page_tokens": 16,
                    "unbounded_classes": ["full"],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}]
                }
                os.environ.update(environment)
                result = plugin._hydrate_capsule_for_admission(
                    lambda *_args, **_kwargs: "no_token",
                    adder,
                    req,
                )
            finally:
                plugin._CAPSULES = old_capsules
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(result, "no_token")
        self.assertEqual(len(allocator.loaded), 1)
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(len(req.prefix_indices), 0)

    def test_pure_swa_capsule_hydrates_only_live_tail(self):
        from orbitkv_sglang import plugin
        from orbitkv_sglang.capsule_wire import encode_cpu_tensors

        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capsule.payload"
            payload_path.write_bytes(encode_cpu_tensors([torch.ones((32, 1))]))
            capsules = FakeHydrationCapsules(
                payload_path,
                prefix_tokens=64,
                live_tokens=32,
            )
            allocator = FakeHydrationAllocator(hybrid=False)
            tree_cache = types.SimpleNamespace(
                is_chunk_cache=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
            adder = types.SimpleNamespace(tree_cache=tree_cache, can_run_list=[])
            req = FakeHydrationReq()
            old_capsules = plugin._CAPSULES
            old_policy = plugin._POLICY
            environment = {
                "ORBITKV_CAPSULE_STORE": directory,
                "ORBITKV_CAPSULE_CHUNK_TOKENS": "16",
                "ORBITKV_TRACE_ALLOCATIONS": "0",
                "ORBITKV_CAPSULE_IDENTITY": json.dumps(
                    {
                        "namespace": "dGVuYW50",
                        "model_fingerprint": f"sha256:{'01' * 32}",
                        "tokenizer_fingerprint": f"sha256:{'02' * 32}",
                        "adapter_fingerprint": f"sha256:{'03' * 32}",
                        "state_plan_fingerprint": f"sha256:{'04' * 32}",
                    }
                ),
            }
            old_environment = {key: os.environ.get(key) for key in environment}
            try:
                plugin._CAPSULES = capsules
                plugin._POLICY = {
                    "page_tokens": 16,
                    "unbounded_classes": [],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}],
                }
                os.environ.update(environment)

                def original(current_adder, current_req):
                    self.assertEqual(len(current_req.prefix_indices), 64)
                    self.assertTrue(
                        torch.equal(
                            current_req.prefix_indices[:32],
                            torch.zeros((32,), dtype=torch.int64),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            current_req.prefix_indices[32:],
                            torch.arange(1, 33, dtype=torch.int64),
                        )
                    )
                    current_adder.can_run_list.append(current_req)
                    return "continue"

                result = plugin._hydrate_capsule_for_admission(
                    original, adder, req
                )
            finally:
                plugin._CAPSULES = old_capsules
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(result, "continue")
        self.assertEqual(req.cache_protected_len, 32)
        self.assertEqual(req._orbitkv_capsule_prefix_tokens, 64)
        self.assertEqual(req._orbitkv_capsule_live_tokens, 32)
        self.assertEqual(len(allocator.loaded[0][1]), 32)
        self.assertEqual(allocator.freed, [])

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
