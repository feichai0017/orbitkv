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
from array import array
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


class FakeTensor:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError(key)
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("FakeTensor only supports contiguous slices")
        return FakeTensor(max(0, stop - start))


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
        self.next_submission_id = 1
        self.domain_sequences = {}

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
        if command["op"] == "register_execution":
            domain = command["completion_domain"]
            sequence = self.domain_sequences.get(domain, 0) + 1
            self.domain_sequences[domain] = sequence
            submission_id = self.next_submission_id
            self.next_submission_id += 1
            return {
                "status": "execution_registered",
                "ticket": {
                    "submission_id": submission_id,
                    "completion_domain": domain,
                    "domain_sequence": sequence,
                    "request_ids": sorted(command["request_ids"]),
                },
            }
        if command["op"] == "complete_executions":
            return {
                "status": "executions_completed",
                "completion_domain": command["completion_domain"],
                "submission_ids": command["submission_ids"],
            }
        raise AssertionError(command)


class FakeBindings:
    def __init__(self, *, hybrid=True, window_tokens=32, fail_commit=False):
        self.hybrid = hybrid
        self.window_tokens = window_tokens
        self.fail_commit = fail_commit
        self.commands = []
        self.next_binding_id = 1

    def command(self, command):
        self.commands.append(command)
        operation = command["op"]
        if operation == "prepare_binding":
            prefix_tokens = int(command["prefix_tokens"])
            start = max(0, prefix_tokens - self.window_tokens)
            components = []
            if self.hybrid:
                components.append(
                    {
                        "state_class": "full",
                        "token_start": 0,
                        "token_end_exclusive": prefix_tokens,
                        "physical_tokens": prefix_tokens,
                    }
                )
            components.append(
                {
                    "state_class": "swa",
                    "token_start": start,
                    "token_end_exclusive": prefix_tokens,
                    "physical_tokens": prefix_tokens - start,
                }
            )
            binding_id = self.next_binding_id
            self.next_binding_id += 1
            return {
                "status": "binding_prepared",
                "intent": {
                    "schema": "orbitkv.state-binding-intent.v1",
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "binding_id": binding_id,
                    "request_id": command["request_id"],
                    "components": components,
                },
            }
        if operation == "commit_binding":
            if self.fail_commit:
                raise RuntimeError("injected binding commit failure")
            return {
                "status": "binding_committed",
                "binding_id": command["receipt"]["binding_id"],
            }
        if operation == "commit_binding_and_acquire_prefix":
            if self.fail_commit:
                raise RuntimeError("injected Prefix binding commit failure")
            return {
                "status": "prefix_binding_committed",
                "binding_id": command["receipt"]["binding_id"],
                "lease_id": 1,
                "object_id": command["receipt"].get(
                    "object_id",
                    self.commands[-2]["prefix"]["object_id"],
                ),
                "state_classes": sorted(command["state_classes"]),
            }
        if operation == "release_prefix":
            return {
                "status": "prefix_released",
                "lease_id": command["lease_id"],
            }
        if operation == "release_prefix_component":
            return {
                "status": "prefix_component_released",
                "lease_id": command["lease_id"],
                "state_class": command["state_class"],
            }
        if operation == "attach_prefix_component":
            return {
                "status": "prefix_component_attached",
                "lease_id": command["lease_id"],
                "state_class": command["state_class"],
            }
        if operation == "tombstone_prefix_component":
            return {
                "status": "prefix_component_tombstoned",
                "object_id": command["object_id"],
                "state_class": command["state_class"],
            }
        if operation == "recover_prefix_component":
            return {
                "status": "prefix_component_recovered",
                "object_id": command["object_id"],
                "state_class": command["state_class"],
            }
        if operation == "abort_binding":
            return {
                "status": "binding_aborted",
                "binding_id": command["binding_id"],
            }
        if operation == "register_prefix":
            return {
                "status": "prefix_registered",
                "object_id": command["prefix"]["object_id"],
            }
        raise AssertionError(command)


class FakeDenseRuntime:
    def __init__(self):
        self.commands = []
        self.next_submission_id = 1

    def command(self, command):
        self.commands.append(command)
        operation = command["op"]
        if operation == "submit_view":
            submission_id = self.next_submission_id
            self.next_submission_id += 1
            return {
                "status": "view_submitted",
                "view": {
                    "request": command["request"],
                    "submission_id": submission_id,
                    "blocks": [],
                },
            }
        if operation == "complete_submission":
            return {
                "status": "submission_completed",
                "submission_id": command["submission_id"],
                "certificates": [],
            }
        if operation in (
            "advance_semantic_frontier",
            "advance_resident_frontier",
        ):
            return {
                "status": "semantic_frontier_advanced",
                "request": command["request"],
                "boundary": command["boundary"],
                "certificates": [],
            }
        if operation == "commit_binding":
            return {
                "status": "binding_committed",
                "binding_id": command["receipt"]["binding_id"],
                "blocks": [],
            }
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


def _prefix_snapshot(manifest):
    capsule_id = manifest.setdefault(
        "capsule_id", list(hashlib.sha256(b"capsule-id").digest())
    )
    object_id = list(hashlib.sha256(b"prefix-object-id").digest())
    return {
        "schema": "orbitkv.prefix-object-snapshot.v1",
        "object_id": object_id,
        "prefix_token_count": manifest["prefix_token_count"],
        "availability": "restorable",
        "components": [
            {
                "spec": {
                    "state_class": component["state_class"],
                    "token_range": {
                        "start": component["token_start"],
                        "end_exclusive": component["token_end_exclusive"],
                    },
                },
                "device": {"state": "absent"},
                "device_completeness": "missing",
                "persistent": {
                    "capsule_id": capsule_id,
                    "payload_digest": manifest["payload_digest"],
                    "component_checksum": component["checksum"],
                    "offset_bytes": component["offset_bytes"],
                    "length_bytes": component["length_bytes"],
                },
                "persistent_completeness": "complete",
                "lease_count": 0,
            }
            for component in manifest["components"]
        ],
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
                        "state_class": "swa",
                        "offset_bytes": 0,
                        "length_bytes": len(payload),
                        "checksum": digest,
                        "token_start": prefix_tokens - live_tokens,
                        "token_end_exclusive": prefix_tokens,
                    }
                ],
            },
            "payload_path": str(payload_path),
        }
        self.response["prefix"] = _prefix_snapshot(self.response["manifest"])

    def command(self, command):
        self.commands.append(command)
        return self.response


class FakeHybridHydrationCapsules:
    def __init__(
        self,
        payload_path: Path,
        full_length: int,
        prefix_tokens: int,
        live_start: int,
    ):
        payload = payload_path.read_bytes()
        full_payload = payload[:full_length]
        swa_payload = payload[full_length:]
        self.commands = []
        self.response = {
            "status": "restored",
            "manifest": {
                "schema": "orbitkv.continuation-capsule.v1",
                "prefix_token_count": prefix_tokens,
                "live_token_count": prefix_tokens,
                "payload_bytes": len(payload),
                "payload_digest": list(hashlib.sha256(payload).digest()),
                "components": [
                    {
                        "state_class": "full",
                        "offset_bytes": 0,
                        "length_bytes": len(full_payload),
                        "checksum": list(hashlib.sha256(full_payload).digest()),
                        "token_start": 0,
                        "token_end_exclusive": prefix_tokens,
                    },
                    {
                        "state_class": "swa",
                        "offset_bytes": len(full_payload),
                        "length_bytes": len(swa_payload),
                        "checksum": list(hashlib.sha256(swa_payload).digest()),
                        "token_start": live_start,
                        "token_end_exclusive": prefix_tokens,
                    },
                ],
            },
            "payload_path": str(payload_path),
        }
        self.response["prefix"] = _prefix_snapshot(self.response["manifest"])

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


class FakeTorchReqToToken:
    def __getitem__(self, key):
        _, token_range = key
        start = 0 if token_range.start is None else token_range.start
        return torch.arange(start, token_range.stop, dtype=torch.int64)


class FakeTreeCache:
    page_size = 16

    @staticmethod
    def is_chunk_cache() -> bool:
        return True


class FakeRadixTreeCache:
    page_size = 16

    @staticmethod
    def is_chunk_cache() -> bool:
        return False

    @staticmethod
    def supports_swa() -> bool:
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


class FakeExecutionEvent:
    def __init__(self):
        self.ready = False
        self.recorded_stream = None
        self.synchronized = False

    def record(self, stream=None):
        self.recorded_stream = stream

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronized = True
        self.ready = True


class FakeExecutionDeviceModule:
    def __init__(self):
        self.events = []

    def Event(self):
        event = FakeExecutionEvent()
        self.events.append(event)
        return event


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


class FakeHybridPool:
    def __init__(self, rows):
        self.rows = rows
        self.copies = []

    def get_cpu_copy(self, indices):
        self.copies.append(indices.clone())
        return [[indices.to(torch.float32).reshape(-1, 1)]]


class FakeHybridCapsuleAllocator(FakeOwningAllocator):
    def __init__(self, events, prefix_tokens, live_start):
        super().__init__(events)
        self.full_pool = FakeHybridPool(prefix_tokens)
        self.swa_pool = FakeHybridPool(prefix_tokens - live_start)
        mapping = torch.zeros((prefix_tokens + 1,), dtype=torch.int64)
        mapping[live_start:prefix_tokens] = torch.arange(
            1, prefix_tokens - live_start + 1, dtype=torch.int64
        )
        self._kvcache = types.SimpleNamespace(
            full_kv_pool=self.full_pool,
            swa_kv_pool=self.swa_pool,
            full_to_swa_index_mapping=mapping,
        )


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


def _load_swa_radix_types(sglang_root: Path):
    required_modules = (
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.mem_cache.allocator.swa",
        "sglang.srt.mem_cache.base_prefix_cache",
        "sglang.srt.mem_cache.cache_init_params",
        "sglang.srt.mem_cache.events",
        "sglang.srt.mem_cache.utils",
        "sglang.srt.environ",
    )
    for name in required_modules:
        if name not in sys.modules:
            _module(name)

    allocator_module = sys.modules["sglang.srt.mem_cache.allocator.swa"]
    allocator_module.SWATokenToKVPoolAllocator = FakeSWATokenToKVPoolAllocator
    base_module = sys.modules["sglang.srt.mem_cache.base_prefix_cache"]
    base_module.BasePrefixCache = object
    for name in (
        "DecLockRefParams",
        "DecLockRefResult",
        "EvictParams",
        "EvictResult",
        "IncLockRefResult",
        "InsertParams",
        "InsertResult",
        "MatchPrefixParams",
        "MatchResult",
    ):
        setattr(base_module, name, type(name, (), {}))
    sys.modules["sglang.srt.mem_cache.cache_init_params"].CacheInitParams = type(
        "CacheInitParams", (), {}
    )
    sys.modules["sglang.srt.mem_cache.events"].KVCacheEventMixin = type(
        "KVCacheEventMixin", (), {}
    )
    utilities = sys.modules["sglang.srt.mem_cache.utils"]
    utilities.get_eviction_strategy = lambda _policy: None
    utilities.get_hash_str = lambda *_args, **_kwargs: ""
    utilities.split_node_hash_value = lambda *_args, **_kwargs: (None, None)
    sys.modules["sglang.srt.environ"].envs = types.SimpleNamespace()

    radix_path = sglang_root / "python/sglang/srt/mem_cache/radix_cache.py"
    radix_spec = importlib.util.spec_from_file_location(
        "sglang.srt.mem_cache.radix_cache", radix_path
    )
    if radix_spec is None or radix_spec.loader is None:
        raise RuntimeError(f"cannot load SGLang RadixKey from {radix_path}")
    radix_module = importlib.util.module_from_spec(radix_spec)
    sys.modules[radix_spec.name] = radix_module
    radix_spec.loader.exec_module(radix_module)

    swa_path = sglang_root / "python/sglang/srt/mem_cache/swa_radix_cache.py"
    swa_spec = importlib.util.spec_from_file_location(
        "sglang.srt.mem_cache.swa_radix_cache", swa_path
    )
    if swa_spec is None or swa_spec.loader is None:
        raise RuntimeError(f"cannot load SGLang SWARadix TreeNode from {swa_path}")
    swa_module = importlib.util.module_from_spec(swa_spec)
    sys.modules[swa_spec.name] = swa_module
    swa_spec.loader.exec_module(swa_module)
    return radix_module.RadixKey, swa_module.TreeNode


class ShadowPluginTests(unittest.TestCase):
    def test_pinned_unified_radix_component_contract_is_present(self):
        root = Path(os.environ["ORBITKV_SGLANG_ROOT"])
        cache = (
            root
            / "python/sglang/srt/mem_cache/unified_radix_cache.py"
        ).read_text(encoding="utf-8")
        core = (
            root
            / "python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py"
        ).read_text(encoding="utf-8")
        component = (
            root
            / "python/sglang/srt/mem_cache/unified_cache/components/tree_component.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class UnifiedRadixCache", cache)
        self.assertIn("def insert(", cache)
        self.assertIn("def evict(", cache)
        self.assertIn("class UnifiedTreeNode", core)
        self.assertIn("self.component_data", core)
        self.assertIn("class ComponentData", component)
        self.assertIn("value: Optional[torch.Tensor] = None", component)

    def test_pinned_overlap_forward_stream_contract_is_present(self):
        root = Path(os.environ["ORBITKV_SGLANG_ROOT"])
        scheduler = (
            root / "python/sglang/srt/managers/scheduler.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def run_batch(", scheduler)
        self.assertIn("with self.forward_stream_ctx:", scheduler)
        self.assertIn(
            "self.model_worker.forward_batch_generation(",
            scheduler,
        )
        self.assertIn("def event_loop_overlap(", scheduler)
        self.assertIn("self.result_queue.append((batch.copy(), batch_result))", scheduler)

    def test_runtime_state_plan_is_the_single_plugin_contract(self):
        from orbitkv_sglang import plugin

        orbitkv_bin = os.environ.get("ORBITKV_BIN")
        if not orbitkv_bin:
            self.skipTest("ORBITKV_BIN is required")
        root = Path(__file__).resolve().parents[3]
        plan = root / "examples/gpt_oss_hybrid_tiny.json"
        artifact = json.loads(
            subprocess.check_output(
                [
                    orbitkv_bin,
                    "compile-runtime-state-plan",
                    str(plan),
                    "--eviction-interval",
                    "32",
                    "--execution-mode",
                    "owner",
                    "--owner-transport",
                    "sidecar",
                    "--capsule-enabled",
                    "true",
                    "--capsule-chunk-tokens",
                    "128",
                    "--capsule-max-payload-bytes",
                    "1073741824",
                    "--prefix-mode",
                    "capsule_backed_swa_radix",
                ],
                text=True,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime-state-plan.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            old_runtime = os.environ.get("ORBITKV_RUNTIME_STATE_PLAN")
            old_policy = os.environ.get("ORBITKV_SGLANG_POLICY")
            old_bin = os.environ.get("ORBITKV_BIN")
            old_loaded = plugin._RUNTIME_STATE_PLAN
            try:
                os.environ["ORBITKV_BIN"] = orbitkv_bin
                os.environ["ORBITKV_RUNTIME_STATE_PLAN"] = str(path)
                os.environ.pop("ORBITKV_SGLANG_POLICY", None)
                loaded = plugin._load_runtime_state_plan()
                plugin._RUNTIME_STATE_PLAN = loaded
                self.assertEqual(
                    loaded["artifact_fingerprint"],
                    artifact["artifact_fingerprint"],
                )
                self.assertTrue(plugin._owner_enabled())
                self.assertEqual(plugin._owner_transport(), "sidecar")
                self.assertTrue(plugin._capsules_enabled())
                self.assertTrue(plugin._prefix_radix_enabled())
                self.assertEqual(plugin._capsule_chunk_tokens(), 128)
                self.assertEqual(plugin._capsule_payload_limit(), 1073741824)

                os.environ["ORBITKV_SGLANG_POLICY"] = str(plan)
                with self.assertRaisesRegex(RuntimeError, "conflicts with legacy"):
                    plugin._load_runtime_state_plan()
            finally:
                plugin._RUNTIME_STATE_PLAN = old_loaded
                for name, value in (
                    ("ORBITKV_RUNTIME_STATE_PLAN", old_runtime),
                    ("ORBITKV_SGLANG_POLICY", old_policy),
                    ("ORBITKV_BIN", old_bin),
                ):
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

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

    def test_cuda_event_frontier_registers_and_polls_real_submission(self):
        from orbitkv_sglang import plugin

        owner = FakeOwner()
        device_module = FakeExecutionDeviceModule()
        req = FakeOwningReq()
        batch = types.SimpleNamespace(reqs=[req])
        scheduler = types.SimpleNamespace(
            device=types.SimpleNamespace(index=0),
            device_module=device_module,
            forward_stream=object(),
        )
        old_owner = plugin._OWNER
        old_runtime = plugin._RUNTIME_STATE_PLAN
        old_trace = os.environ.get("ORBITKV_TRACE_ALLOCATIONS")
        with plugin._EXECUTION_EVENTS_LOCK:
            old_events = dict(plugin._EXECUTION_EVENTS)
            plugin._EXECUTION_EVENTS.clear()
        try:
            plugin._OWNER = owner
            plugin._RUNTIME_STATE_PLAN = {
                "execution": {
                    "mode": "owner",
                    "owner_transport": "sidecar",
                    "frontier": "cuda_event",
                },
                "capsule": {"enabled": False},
            }
            os.environ["ORBITKV_TRACE_ALLOCATIONS"] = "0"
            result = plugin._track_forward_execution(
                lambda *_args, **_kwargs: "forward-result",
                scheduler,
                batch,
            )
            self.assertEqual(result, "forward-result")
            self.assertEqual(len(device_module.events), 1)
            self.assertIs(
                device_module.events[0].recorded_stream,
                scheduler.forward_stream,
            )
            plugin._poll_execution_events()
            self.assertEqual(
                [command["op"] for command in owner.commands],
                ["register_execution"],
            )

            device_module.events[0].ready = True
            plugin._poll_execution_events()
            self.assertEqual(
                [command["op"] for command in owner.commands],
                ["register_execution", "complete_executions"],
            )
            with plugin._EXECUTION_EVENTS_LOCK:
                self.assertEqual(plugin._EXECUTION_EVENTS, {})
        finally:
            plugin._OWNER = old_owner
            plugin._RUNTIME_STATE_PLAN = old_runtime
            with plugin._EXECUTION_EVENTS_LOCK:
                plugin._EXECUTION_EVENTS.clear()
                plugin._EXECUTION_EVENTS.update(old_events)
            if old_trace is None:
                os.environ.pop("ORBITKV_TRACE_ALLOCATIONS", None)
            else:
                os.environ["ORBITKV_TRACE_ALLOCATIONS"] = old_trace

    def test_dense_capsule_receipt_binds_full_and_swa_backend_pages(self):
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin

        indices = torch.arange(16, 80, dtype=torch.int64)
        mapping = torch.zeros((96,), dtype=torch.int64)
        mapping[indices] = torch.arange(48, 112, dtype=torch.int64)
        allocator = types.SimpleNamespace(
            _kvcache=types.SimpleNamespace(
                full_to_swa_index_mapping=mapping
            )
        )
        transaction = {
            "request": {"slot": 0, "generation": 1},
            "index_origin_tokens": 0,
            "intent": {
                "binding_id": 7,
                "pending_blocks": [
                    {
                        "logical": {
                            "request": {"slot": 0, "generation": 1},
                            "class_id": 0,
                            "ordinal": 0,
                        },
                        "physical": {
                            "class_id": 0,
                            "slot": 0,
                            "generation": 1,
                        },
                    },
                    {
                        "logical": {
                            "request": {"slot": 0, "generation": 1},
                            "class_id": 1,
                            "ordinal": 0,
                        },
                        "physical": {
                            "class_id": 1,
                            "slot": 0,
                            "generation": 1,
                        },
                    },
                ],
            },
        }
        old_runtime = plugin._RUNTIME_STATE_PLAN
        old_policy = plugin._POLICY
        try:
            plugin._RUNTIME_STATE_PLAN = {
                "dense_runtime": {
                    "artifact_fingerprint": f"sha256:{'11' * 32}",
                    "page_tokens": 16,
                    "classes": [
                        {"class_id": 0, "name": "full"},
                        {"class_id": 1, "name": "swa"},
                    ],
                }
            }
            plugin._POLICY = {
                "unbounded_classes": ["full"],
                "bounded_classes": [{"name": "swa"}],
            }
            receipt = plugin._dense_binding_receipt(
                transaction,
                allocator,
                indices,
            )
        finally:
            plugin._RUNTIME_STATE_PLAN = old_runtime
            plugin._POLICY = old_policy

        self.assertEqual(receipt["blocks"][0]["backend"], {"domain": 0, "index": 1})
        self.assertEqual(receipt["blocks"][1]["backend"], {"domain": 1, "index": 3})

    def test_cuda_event_completion_advances_dense_runtime(self):
        from orbitkv_sglang import plugin

        owner = FakeOwner()
        dense = FakeDenseRuntime()
        device_module = FakeExecutionDeviceModule()
        req = FakeOwningReq()
        req._orbitkv_dense_request = {"slot": 0, "generation": 1}
        req._orbitkv_dense_boundary = 4
        req.kv.kv_allocated_len = 4
        batch = types.SimpleNamespace(reqs=[req])
        scheduler = types.SimpleNamespace(
            device=types.SimpleNamespace(index=0),
            device_module=device_module,
            forward_stream=object(),
        )
        old_owner = plugin._OWNER
        old_dense = plugin._DENSE_RUNTIME
        old_runtime = plugin._RUNTIME_STATE_PLAN
        old_trace = os.environ.get("ORBITKV_TRACE_ALLOCATIONS")
        with plugin._EXECUTION_EVENTS_LOCK:
            old_events = dict(plugin._EXECUTION_EVENTS)
            plugin._EXECUTION_EVENTS.clear()
        try:
            plugin._OWNER = owner
            plugin._DENSE_RUNTIME = dense
            plugin._RUNTIME_STATE_PLAN = {
                "execution": {
                    "mode": "owner",
                    "owner_transport": "sidecar",
                    "frontier": "cuda_event",
                },
                "capsule": {"enabled": True},
                "dense_runtime": {
                    "page_tokens": 16,
                    "classes": [{"class_id": 0, "name": "full"}],
                },
            }
            os.environ["ORBITKV_TRACE_ALLOCATIONS"] = "0"
            result = plugin._track_forward_execution(
                lambda *_args, **_kwargs: "forward-result",
                scheduler,
                batch,
            )
            self.assertEqual(result, "forward-result")
            device_module.events[0].ready = True
            plugin._poll_execution_events()
        finally:
            plugin._OWNER = old_owner
            plugin._DENSE_RUNTIME = old_dense
            plugin._RUNTIME_STATE_PLAN = old_runtime
            with plugin._EXECUTION_EVENTS_LOCK:
                plugin._EXECUTION_EVENTS.clear()
                plugin._EXECUTION_EVENTS.update(old_events)
            if old_trace is None:
                os.environ.pop("ORBITKV_TRACE_ALLOCATIONS", None)
            else:
                os.environ["ORBITKV_TRACE_ALLOCATIONS"] = old_trace

        self.assertEqual(
            [command["op"] for command in dense.commands],
            [
                "submit_view",
                "complete_submission",
                "advance_resident_frontier",
            ],
        )

    def test_dense_capsule_rejects_more_than_one_continuation_token(self):
        from orbitkv_sglang import plugin

        req = FakeOwningReq()
        req._orbitkv_dense_request = {"slot": 0, "generation": 1}
        req._orbitkv_dense_hydration_boundary = 4096
        req.kv.kv_allocated_len = 4097
        batch = types.SimpleNamespace(
            enable_overlap=False,
            reqs=[req],
        )
        old_runtime = plugin._RUNTIME_STATE_PLAN
        try:
            plugin._RUNTIME_STATE_PLAN = {
                "execution": {"frontier": "cuda_event"},
                "capsule": {"enabled": True},
                "dense_runtime": {"page_tokens": 16},
            }
            with self.assertRaisesRegex(RuntimeError, "one continuation token"):
                plugin._stage_dense_bindings_after_prepare(
                    lambda *_args, **_kwargs: None,
                    batch,
                )
        finally:
            plugin._RUNTIME_STATE_PLAN = old_runtime

    def test_request_release_waits_only_for_its_cuda_event(self):
        from orbitkv_sglang import plugin

        owner = FakeOwner()
        event_r0 = FakeExecutionEvent()
        event_r1 = FakeExecutionEvent()
        old_owner = plugin._OWNER
        old_runtime = plugin._RUNTIME_STATE_PLAN
        with plugin._EXECUTION_EVENTS_LOCK:
            old_events = dict(plugin._EXECUTION_EVENTS)
            plugin._EXECUTION_EVENTS.clear()
            plugin._EXECUTION_EVENTS.update(
                {
                    1: {
                        "event": event_r0,
                        "completion_domain": "cuda:0:forward",
                        "submission_id": 1,
                        "domain_sequence": 1,
                        "request_ids": ["r0"],
                    },
                    2: {
                        "event": event_r1,
                        "completion_domain": "cuda:0:forward",
                        "submission_id": 2,
                        "domain_sequence": 2,
                        "request_ids": ["r1"],
                    },
                }
            )
        try:
            plugin._OWNER = owner
            plugin._RUNTIME_STATE_PLAN = {
                "execution": {
                    "mode": "owner",
                    "owner_transport": "sidecar",
                    "frontier": "cuda_event",
                },
                "capsule": {"enabled": False},
            }
            plugin._wait_execution_for_request("r0")
            self.assertTrue(event_r0.synchronized)
            self.assertFalse(event_r1.synchronized)
            with plugin._EXECUTION_EVENTS_LOCK:
                self.assertNotIn(1, plugin._EXECUTION_EVENTS)
                self.assertIn(2, plugin._EXECUTION_EVENTS)
        finally:
            plugin._OWNER = old_owner
            plugin._RUNTIME_STATE_PLAN = old_runtime
            with plugin._EXECUTION_EVENTS_LOCK:
                plugin._EXECUTION_EVENTS.clear()
                plugin._EXECUTION_EVENTS.update(old_events)

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
        old_policy = plugin._POLICY
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
            plugin._POLICY = {
                "page_tokens": 16,
                "unbounded_classes": [],
                "bounded_classes": [{"name": "swa", "window_tokens": 32}],
            }
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
            plugin._POLICY = old_policy
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
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin

        events = []
        owner = FakeOwner()
        capsules = FakeCapsules(events)
        allocator = FakeCapsuleAllocator(events)
        tree_cache = types.SimpleNamespace(
            is_chunk_cache=lambda: True,
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=FakeTorchReqToToken()
            ),
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

    def test_hybrid_capsule_exports_full_history_and_swa_tail_components(self):
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin

        events = []
        owner = FakeOwner()
        capsules = FakeCapsules(events)
        allocator = FakeHybridCapsuleAllocator(events, prefix_tokens=64, live_start=32)
        tree_cache = types.SimpleNamespace(
            is_chunk_cache=lambda: True,
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=FakeTorchReqToToken()
            ),
            token_to_kv_pool_allocator=allocator,
        )
        req = FakeOwningReq()
        req.kv_committed_len = 64
        req.origin_input_ids = list(range(64))
        old_capsules = plugin._CAPSULES
        old_owner = plugin._OWNER
        old_policy = plugin._POLICY
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
                "unbounded_classes": ["full"],
                "bounded_classes": [{"name": "swa", "window_tokens": 32}],
            }
            os.environ.update(environment)
            result = plugin._release_owned_request(
                lambda *_args, **_kwargs: "released",
                req,
                tree_cache,
            )
        finally:
            plugin._CAPSULES = old_capsules
            plugin._OWNER = old_owner
            plugin._POLICY = old_policy
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result, "released")
        publish = capsules.commands[1]
        self.assertEqual(publish["live_token_count"], 64)
        self.assertEqual(
            publish["components"],
            [
                {
                    "state_class": "full",
                    "length_bytes": publish["components"][0]["length_bytes"],
                    "token_start": 0,
                    "token_end_exclusive": 64,
                },
                {
                    "state_class": "swa",
                    "length_bytes": publish["components"][1]["length_bytes"],
                    "token_start": 32,
                    "token_end_exclusive": 64,
                },
            ],
        )
        self.assertEqual(allocator.full_pool.copies[0].tolist(), list(range(64)))
        self.assertEqual(allocator.swa_pool.copies[0].tolist(), list(range(1, 33)))

    def test_capsule_hydration_commits_only_after_admission(self):
        if torch is None:
            self.skipTest("torch is unavailable")
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
            old_bindings = plugin._BINDINGS
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
                bindings = FakeBindings(hybrid=False)
                plugin._BINDINGS = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "page_tokens": 16,
                    "unbounded_classes": [],
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
                plugin._BINDINGS = old_bindings
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(result, "continue")
        self.assertEqual(allocator.tail_lengths, [])
        self.assertEqual(len(allocator.loaded), 1)
        self.assertEqual(allocator.freed, [])
        self.assertEqual(req._orbitkv_capsule_prefix_tokens, 64)
        self.assertEqual(capsules.commands[0]["token_ids"], list(range(64)))
        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["prepare_binding", "commit_binding"],
        )

    def test_capsule_hydration_rolls_back_rejected_admission(self):
        if torch is None:
            self.skipTest("torch is unavailable")
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
            old_bindings = plugin._BINDINGS
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
                bindings = FakeBindings(hybrid=False)
                plugin._BINDINGS = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "page_tokens": 16,
                    "unbounded_classes": [],
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
                plugin._BINDINGS = old_bindings
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

    def test_radix_capsule_binding_atomically_acquires_prefix_lease(self):
        if torch is None:
            self.skipTest("torch is unavailable")
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
                is_chunk_cache=lambda: False,
                supports_swa=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
            adder = types.SimpleNamespace(tree_cache=tree_cache, can_run_list=[])
            req = FakeHydrationReq()
            old_capsules = plugin._CAPSULES
            old_bindings = plugin._BINDINGS
            old_owner = plugin._OWNER
            old_policy = plugin._POLICY
            old_runtime_plan = plugin._RUNTIME_STATE_PLAN
            with plugin._PREFIX_OBJECTS_LOCK:
                old_prefix_objects = dict(plugin._PREFIX_OBJECTS)
                plugin._PREFIX_OBJECTS.clear()
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
                bindings = FakeBindings(hybrid=False)
                plugin._BINDINGS = bindings
                plugin._OWNER = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "page_tokens": 16,
                    "unbounded_classes": [],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}],
                }
                plugin._RUNTIME_STATE_PLAN = {
                    "artifact_fingerprint": f"sha256:{'04' * 32}",
                    "capsule": {
                        "enabled": True,
                        "chunk_tokens": 16,
                        "maximum_payload_bytes": 1 << 30,
                    },
                    "execution": {"owner_transport": "sidecar"},
                    "prefix": {"mode": "capsule_backed_swa_radix"},
                }
                os.environ.update(environment)

                def original(current_adder, current_req):
                    current_adder.can_run_list.append(current_req)
                    return "continue"

                result = plugin._hydrate_capsule_for_admission(original, adder, req)
            finally:
                plugin._CAPSULES = old_capsules
                plugin._BINDINGS = old_bindings
                plugin._OWNER = old_owner
                plugin._POLICY = old_policy
                plugin._RUNTIME_STATE_PLAN = old_runtime_plan
                with plugin._PREFIX_OBJECTS_LOCK:
                    plugin._PREFIX_OBJECTS.clear()
                    plugin._PREFIX_OBJECTS.update(old_prefix_objects)
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(result, "continue")
        self.assertEqual(req._orbitkv_prefix_lease_id, 1)
        self.assertEqual(req._orbitkv_prefix_state_classes, ["swa"])
        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["prepare_binding", "commit_binding_and_acquire_prefix"],
        )
        self.assertEqual(
            bindings.commands[0]["prefix"],
            capsules.response["prefix"],
        )

    def test_real_swa_radix_nodes_drive_component_tombstone_and_recovery(self):
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin

        sglang_root = Path(os.environ["ORBITKV_SGLANG_ROOT"])
        RadixKey, TreeNode = _load_swa_radix_types(sglang_root)
        root = TreeNode()
        root.key = RadixKey(token_ids=array("q"))
        child = TreeNode()
        child.parent = root
        child.key = RadixKey(token_ids=array("q", range(64)))
        child.value = torch.arange(64, dtype=torch.int64)
        child.swa_tombstone = True
        root.children[child.key.child_key(16)] = child
        tree_cache = types.SimpleNamespace(root_node=root, page_size=16)
        object_id = list(hashlib.sha256(b"prefix-object").digest())
        entry = {
            "object_id": object_id,
            "prefix_token_count": 64,
            "token_ids": tuple(range(64)),
            "token_digest": plugin._prefix_token_digest(tuple(range(64))),
            "extra_key": None,
            "cache_salt": None,
            "tree_materialized": True,
            "components": {
                "full": {
                    "token_start": 0,
                    "token_end_exclusive": 64,
                    "device_resident": True,
                },
                "swa": {
                    "token_start": 32,
                    "token_end_exclusive": 64,
                    "device_resident": True,
                },
            },
        }
        key = plugin._prefix_registry_key(tuple(range(64)), None, None)
        bindings = FakeBindings()
        old_bindings = plugin._BINDINGS
        old_policy = plugin._POLICY
        with plugin._PREFIX_OBJECTS_LOCK:
            old_prefix_objects = dict(plugin._PREFIX_OBJECTS)
            plugin._PREFIX_OBJECTS.clear()
            plugin._PREFIX_OBJECTS[key] = entry
        try:
            plugin._BINDINGS = bindings
            plugin._POLICY = {
                "page_tokens": 16,
                "unbounded_classes": ["full"],
                "bounded_classes": [{"name": "swa", "window_tokens": 32}],
            }
            plugin._sync_radix_prefix_components(tree_cache)
            with plugin._PREFIX_OBJECTS_LOCK:
                self.assertTrue(
                    plugin._PREFIX_OBJECTS[key]["components"]["full"][
                        "device_resident"
                    ]
                )
                self.assertFalse(
                    plugin._PREFIX_OBJECTS[key]["components"]["swa"][
                        "device_resident"
                    ]
                )

            child.swa_tombstone = False
            plugin._sync_radix_prefix_components(tree_cache)
            with plugin._PREFIX_OBJECTS_LOCK:
                self.assertTrue(
                    plugin._PREFIX_OBJECTS[key]["components"]["swa"][
                        "device_resident"
                    ]
                )
        finally:
            plugin._BINDINGS = old_bindings
            plugin._POLICY = old_policy
            with plugin._PREFIX_OBJECTS_LOCK:
                plugin._PREFIX_OBJECTS.clear()
                plugin._PREFIX_OBJECTS.update(old_prefix_objects)

        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["tombstone_prefix_component", "recover_prefix_component"],
        )

    def test_capsule_load_failure_frees_physical_state_and_aborts_binding(self):
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin
        from orbitkv_sglang.capsule_wire import encode_cpu_tensors

        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capsule.payload"
            payload_path.write_bytes(
                encode_cpu_tensors({"full": [[torch.ones((64, 1))]], "swa": None})
            )
            capsules = FakeHydrationCapsules(payload_path, 64)
            allocator = FakeHydrationAllocator(fail_load=True)
            tree_cache = types.SimpleNamespace(
                is_chunk_cache=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
            req = FakeHydrationReq()
            old_capsules = plugin._CAPSULES
            old_bindings = plugin._BINDINGS
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
                bindings = FakeBindings(hybrid=False)
                plugin._CAPSULES = capsules
                plugin._BINDINGS = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "page_tokens": 16,
                    "unbounded_classes": [],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}],
                }
                os.environ.update(environment)
                with self.assertRaisesRegex(RuntimeError, "injected capsule load"):
                    plugin._try_hydrate_capsule(req, tree_cache)
            finally:
                plugin._CAPSULES = old_capsules
                plugin._BINDINGS = old_bindings
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["prepare_binding", "abort_binding"],
        )

    def test_binding_commit_failure_removes_admission_and_rolls_back(self):
        if torch is None:
            self.skipTest("torch is unavailable")
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
            old_bindings = plugin._BINDINGS
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
                bindings = FakeBindings(hybrid=False, fail_commit=True)
                plugin._CAPSULES = capsules
                plugin._BINDINGS = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "page_tokens": 16,
                    "unbounded_classes": [],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}],
                }
                os.environ.update(environment)

                def original(current_adder, current_req):
                    current_adder.can_run_list.append(current_req)
                    return "continue"

                with self.assertRaisesRegex(RuntimeError, "binding commit failure"):
                    plugin._hydrate_capsule_for_admission(original, adder, req)
            finally:
                plugin._CAPSULES = old_capsules
                plugin._BINDINGS = old_bindings
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["prepare_binding", "commit_binding", "abort_binding"],
        )

    def test_pure_swa_capsule_hydrates_only_live_tail(self):
        if torch is None:
            self.skipTest("torch is unavailable")
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
            old_bindings = plugin._BINDINGS
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
                bindings = FakeBindings(hybrid=False)
                plugin._BINDINGS = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
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
                plugin._BINDINGS = old_bindings
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
        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["prepare_binding", "commit_binding"],
        )

    def test_hybrid_capsule_restores_full_history_and_swa_tail(self):
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin
        from orbitkv_sglang.capsule_wire import encode_cpu_tensors

        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capsule.payload"
            full_payload = encode_cpu_tensors([[torch.ones((64, 1))]])
            swa_mask = torch.zeros((64,), dtype=torch.bool)
            swa_mask[32:] = True
            swa_payload = encode_cpu_tensors(
                {
                    "swa": [[torch.ones((32, 1))]],
                    "swa_mask": swa_mask,
                }
            )
            payload_path.write_bytes(full_payload + swa_payload)
            capsules = FakeHybridHydrationCapsules(
                payload_path,
                full_length=len(full_payload),
                prefix_tokens=64,
                live_start=32,
            )
            allocator = FakeHydrationAllocator()
            tree_cache = types.SimpleNamespace(
                is_chunk_cache=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
            adder = types.SimpleNamespace(tree_cache=tree_cache, can_run_list=[])
            req = FakeHydrationReq()
            old_capsules = plugin._CAPSULES
            old_bindings = plugin._BINDINGS
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
                bindings = FakeBindings()
                plugin._BINDINGS = bindings
                plugin._POLICY = {
                    "plan_fingerprint": f"sha256:{'00' * 32}",
                    "page_tokens": 16,
                    "unbounded_classes": ["full"],
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}],
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
                plugin._BINDINGS = old_bindings
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(result, "continue")
        self.assertEqual(allocator.tail_lengths, [32])
        loaded, indices, _ = allocator.loaded[0]
        self.assertEqual(len(indices), 64)
        self.assertEqual(len(loaded["full"][0][0]), 64)
        self.assertEqual(len(loaded["swa"][0][0]), 32)
        self.assertFalse(bool(loaded["swa_mask"][:32].any().item()))
        self.assertTrue(bool(loaded["swa_mask"][32:].all().item()))
        self.assertEqual(req.cache_protected_len, 0)
        self.assertEqual(
            [command["op"] for command in bindings.commands],
            ["prepare_binding", "commit_binding"],
        )

    def test_hybrid_capsule_rejects_wrong_swa_component_range(self):
        if torch is None:
            self.skipTest("torch is unavailable")
        from orbitkv_sglang import plugin
        from orbitkv_sglang.capsule_wire import encode_cpu_tensors

        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capsule.payload"
            full_payload = encode_cpu_tensors([[torch.ones((64, 1))]])
            swa_mask = torch.zeros((64,), dtype=torch.bool)
            swa_mask[32:] = True
            swa_payload = encode_cpu_tensors(
                {
                    "swa": [[torch.ones((32, 1))]],
                    "swa_mask": swa_mask,
                }
            )
            payload_path.write_bytes(full_payload + swa_payload)
            capsules = FakeHybridHydrationCapsules(
                payload_path,
                full_length=len(full_payload),
                prefix_tokens=64,
                live_start=32,
            )
            capsules.response["manifest"]["components"][1]["token_start"] = 16
            req = FakeHydrationReq()
            allocator = FakeHydrationAllocator()
            tree_cache = types.SimpleNamespace(
                is_chunk_cache=lambda: True,
                token_to_kv_pool_allocator=allocator,
            )
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
                    "bounded_classes": [{"name": "swa", "window_tokens": 32}],
                }
                os.environ.update(environment)
                with self.assertRaisesRegex(
                    RuntimeError, "Prefix component proof is invalid"
                ):
                    plugin._try_hydrate_capsule(req, tree_cache)
            finally:
                plugin._CAPSULES = old_capsules
                plugin._POLICY = old_policy
                for key, value in old_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(allocator.loaded, [])

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
