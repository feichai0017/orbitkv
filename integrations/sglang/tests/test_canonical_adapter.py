from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.plugin.facade as facade  # noqa: E402
import orbitkv_sglang.plugin.lowering as lowering  # noqa: E402
import orbitkv_sglang.plugin.mirror_cleanup as mirror_cleanup  # noqa: E402
import orbitkv_sglang.plugin.prefix_cache as prefix_cache  # noqa: E402
import orbitkv_sglang.plugin.state as state  # noqa: E402
import orbitkv_sglang.plugin.validation as validation  # noqa: E402
from orbitkv_sglang.config import ClassConfig, ManagerPlanConfig  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaIdentity,
    ArenaStats,
    BatchRecord,
    ClassLoweringSpec,
    DETACHED_CLEAR,
    DETACHED_RETENTION,
    DetachedBinding,
    FailStopped,
    LoweringPlan,
    ManagerStats,
    MirrorCleanupItem,
    PageLease,
    ReclamationCertificate,
    ReclamationLease,
    RequestLease,
    SnapshotLease,
    TAIL_NONE,
    TailAction,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo  # noqa: E402


def _class(class_id: int, retention: str) -> ClassConfig:
    window = 32 if retention == "sliding" else None
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 1,
        name="swa" if retention == "sliding" else "full",
        layers=(class_id,),
        retention=retention,
        bytes_per_token_per_layer=128,
        window_tokens=window,
        period_blocks=3 if window is not None else None,
    )


def _config(retentions: tuple[str, ...]) -> ManagerPlanConfig:
    return ManagerPlanConfig(
        plan_path=Path("plan.json"),
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:test",
        page_tokens=16,
        classes=tuple(_class(index, value) for index, value in enumerate(retentions)),
    )


class FakeManager:
    def __init__(self, config, registrations):
        first_ids = (7, 101)
        self.arenas = tuple(
            ArenaIdentity(
                engine_epoch=11,
                pool_epoch=20 + item.class_id,
                pool_id=item.pool_id,
                class_id=item.class_id,
                backend_domain=item.backend_domain,
                page_count=item.page_count,
                page_tokens=config.page_tokens,
                backend_base_index=item.backend_base_index,
                first_page_id=first_ids[item.class_id],
            )
            for item in registrations
        )
        self.arenas_by_class = {item.class_id: item for item in self.arenas}
        self.free_pages = {item.class_id: item.page_count for item in self.arenas}

    def arena_stats(self):
        return tuple(
            ArenaStats(
                engine_epoch=item.engine_epoch,
                pool_epoch=item.pool_epoch,
                pool_id=item.pool_id,
                page_count=item.page_count,
                class_id=item.class_id,
                backend_domain=item.backend_domain,
                first_page_id=item.first_page_id,
                free_pages=self.free_pages[item.class_id],
                reserved_pages=0,
                writing_pages=0,
                active_pages=item.page_count - self.free_pages[item.class_id],
                retiring_pages=0,
                quarantined_pages=0,
                exhausted_pages=0,
                request_page_refs=0,
                prefix_page_refs=0,
                reader_pins=0,
            )
            for item in self.arenas
        )

    def stats(self):
        arena_stats = self.arena_stats()
        return ManagerStats(
            active_requests=0,
            active_snapshots=0,
            active_prefixes=0,
            evicted_prefixes=0,
            prepared_steps=0,
            submitted_steps=0,
            free_pages=sum(item.free_pages for item in arena_stats),
            reserved_pages=0,
            writing_pages=0,
            active_pages=sum(item.active_pages for item in arena_stats),
            retiring_pages=0,
            quarantined_pages=0,
            exhausted_pages=0,
            pending_reclamations=0,
            total_request_page_refs=0,
            total_prefix_page_refs=0,
            total_reader_pins=0,
        )

    @property
    def performance_counters(self):
        return {}

    def request_acquire_batch(self, _count):
        raise AssertionError("not used")

    def request_fork_batch(self, _items):
        raise AssertionError("not used")

    def prepare_batch(self, _items):
        raise AssertionError("not used")

    def submit_batch(self, _items):
        raise AssertionError("not used")

    def complete_batch(self, _receipt, _submissions):
        raise AssertionError("not used")

    def abort_steps_batch(self, _receipts):
        raise AssertionError("not used")

    def quarantine_steps_batch(self, _steps):
        raise AssertionError("not used")

    def quarantine_submissions_batch(self, _submissions):
        raise AssertionError("not used")

    def release_batch(self, _requests):
        raise AssertionError("not used")

    def acknowledge_reclamations_batch(self, _receipts):
        raise AssertionError("not used")

    def recycle_requests_batch(self, _requests):
        raise AssertionError("not used")

    def prefix_lookup_batch(self, _keys):
        raise AssertionError("not used")

    def prefix_attach_batch(self, _items):
        raise AssertionError("not used")

    def prefix_publish_batch(self, _items):
        raise AssertionError("not used")

    def prefix_publish_release_batch(self, _items):
        raise AssertionError("not used")

    def prefix_evict_batch(self, _prefixes):
        raise AssertionError("not used")

    def prefix_recycle_batch(self, _prefixes):
        raise AssertionError("not used")

    def destroy(self):
        return None


class FakeFactory:
    def __init__(self):
        self.calls = []
        self.manager = None

    def create(self, config, settings, arenas):
        registrations = tuple(arenas)
        self.calls.append((config, settings, registrations))
        self.manager = FakeManager(config, registrations)
        return self.manager


class FakeKvPool:
    def __init__(self):
        self.mapping = None

    def register_mapping(self, mapping):
        self.mapping = mapping


def _install(config, factory):
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(
            maximum_running_requests=4,
            chunked_prefill_tokens=64,
            maximum_context_tokens=256,
        ),
        runtime=None,
        factory=factory,
    )


def _configurator(*, hybrid: bool):
    return SimpleNamespace(
        page_size=16,
        device="cuda:0",
        kv_cache_dtype=torch.bfloat16,
        is_hybrid_swa=hybrid,
        is_draft_worker=False,
    )


@pytest.fixture
def cpu_mapping_tensors(monkeypatch):
    facade._facade_types()
    original_zeros = torch.zeros
    original_arange = torch.arange
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: original_zeros(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *args, **kwargs: original_arange(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )


def test_full_builder_installs_paged_facade_and_one_arena():
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    config = _config(("full",))
    factory = FakeFactory()
    _install(config, factory)
    allocator = facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=False),
        sizes=SimpleNamespace(max_total_num_tokens=128),
        token_to_kv_pool=FakeKvPool(),
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )
    assert isinstance(allocator, PagedTokenToKVPoolAllocator)
    assert allocator.size == 128
    assert tuple(item.page_count for item in factory.calls[0][2]) == (8,)
    assert allocator.available_size() == 128
    factory.manager.free_pages[0] = 2
    assert allocator.available_size() == 32
    with pytest.raises(RuntimeError, match="native KV authority"):
        allocator.alloc(16)
    with pytest.raises(RuntimeError, match="native KV authority"):
        allocator.free(object())


def test_hybrid_builder_installs_swa_facade_and_independent_admission(
    cpu_mapping_tensors,
):
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    config = _config(("full", "sliding"))
    factory = FakeFactory()
    pool = FakeKvPool()
    _install(config, factory)
    allocator = facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=True),
        sizes=SimpleNamespace(
            max_total_num_tokens=64,
            full_max_total_num_tokens=128,
            swa_max_total_num_tokens=320,
        ),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )
    assert isinstance(allocator, SWATokenToKVPoolAllocator)
    assert tuple(item.page_count for item in factory.calls[0][2]) == (8, 20)
    assert pool.mapping is allocator.full_to_swa_index_mapping
    assert int(pool.mapping[-1]) == -1
    factory.manager.free_pages.update({0: 3, 1: 1})
    assert allocator.full_available_size() == 48
    assert allocator.swa_available_size() == 16
    assert allocator.available_size() == 16
    assert allocator.new_pages_available(3, 1)
    assert not allocator.new_pages_available(4, 1)
    with pytest.raises(RuntimeError, match="native KV authority"):
        allocator.free_swa(object())


def test_pure_swa_builder_uses_identity_lut_and_one_arena(cpu_mapping_tensors):
    from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator

    config = _config(("sliding",))
    factory = FakeFactory()
    pool = FakeKvPool()
    _install(config, factory)
    allocator = facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=True),
        sizes=SimpleNamespace(
            max_total_num_tokens=64,
            full_max_total_num_tokens=0,
            swa_max_total_num_tokens=320,
        ),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )
    assert isinstance(allocator, PureSWATokenToKVPoolAllocator)
    assert tuple(item.page_count for item in factory.calls[0][2]) == (20,)
    assert torch.equal(pool.mapping[:320], torch.arange(320))
    assert int(pool.mapping[-1]) == -1
    assert allocator.swa_available_size() == 320


def test_hybrid_builder_accepts_exact_b1_swa_floor_and_rejects_below_it(
    cpu_mapping_tensors,
):
    base = _config(("full", "sliding"))
    config = replace(
        base,
        classes=(
            base.classes[0],
            replace(base.classes[1], window_tokens=128, period_blocks=19),
        ),
    )
    limits = state.RuntimeLimits(
        maximum_running_requests=1,
        chunked_prefill_tokens=256,
        maximum_context_tokens=1024,
    )
    factory = FakeFactory()
    state._install_test_state(config=config, limits=limits, factory=factory)
    facade._build_token_to_kv_pool_allocator(
        _configurator(hybrid=True),
        sizes=SimpleNamespace(
            max_total_num_tokens=512,
            full_max_total_num_tokens=512,
            swa_max_total_num_tokens=560,
        ),
        token_to_kv_pool=FakeKvPool(),
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )
    assert tuple(item.page_count for item in factory.calls[0][2]) == (32, 35)

    rejected_factory = FakeFactory()
    state._install_test_state(
        config=config, limits=limits, factory=rejected_factory
    )
    with pytest.raises(RuntimeError, match=r"capacity=544 minimum=560"):
        facade._build_token_to_kv_pool_allocator(
            _configurator(hybrid=True),
            sizes=SimpleNamespace(
                max_total_num_tokens=512,
                full_max_total_num_tokens=512,
                swa_max_total_num_tokens=544,
            ),
            token_to_kv_pool=FakeKvPool(),
            is_dsv4_model=False,
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=None,
        )
    assert rejected_factory.calls == []


@pytest.mark.parametrize(
    "rid, expected",
    (("request-1", ("str", "request-1")), (b"request-1", ("bytes", b"request-1")), (7, ("int", 7))),
)
def test_request_key_uses_stable_typed_rid(rid, expected):
    assert state._request_key(SimpleNamespace(rid=rid)) == expected


@pytest.mark.parametrize("rid", (None, "", b"", -1, True, object()))
def test_request_key_rejects_unstable_or_empty_identity(rid):
    with pytest.raises(RuntimeError, match="rid"):
        state._request_key(SimpleNamespace(rid=rid))


class TransactionRuntime:
    def __init__(self, config):
        self.config = config
        self.records = {}
        self.callbacks = {}
        self.calls = []
        self.failure_reason = None
        self.request_rows = {}
        self.row_owners = {}
        self.arenas_by_class = {
            item.class_id: ArenaIdentity(
                engine_epoch=1,
                pool_epoch=2 + item.class_id,
                pool_id=item.pool_id,
                class_id=item.class_id,
                backend_domain=item.backend_domain,
                page_count=8,
                page_tokens=16,
                backend_base_index=0,
                first_page_id=1 + item.class_id * 16,
            )
            for item in config.classes
        }
        self.manager = self

    def poll(self):
        return None

    def wait_batch(self, _keys):
        return None

    def stats(self):
        arena_stats = self.arena_stats()
        return ManagerStats(
            active_requests=len(self.records),
            active_snapshots=len(self.records),
            active_prefixes=0,
            evicted_prefixes=0,
            prepared_steps=0,
            submitted_steps=0,
            free_pages=sum(item.free_pages for item in arena_stats),
            reserved_pages=0,
            writing_pages=0,
            active_pages=0,
            retiring_pages=0,
            quarantined_pages=0,
            exhausted_pages=0,
            pending_reclamations=0,
            total_request_page_refs=0,
            total_prefix_page_refs=0,
            total_reader_pins=0,
        )

    def arena_stats(self):
        return tuple(
            ArenaStats(
                engine_epoch=arena.engine_epoch,
                pool_epoch=arena.pool_epoch,
                pool_id=arena.pool_id,
                page_count=arena.page_count,
                class_id=arena.class_id,
                backend_domain=arena.backend_domain,
                first_page_id=arena.first_page_id,
                free_pages=arena.page_count,
                reserved_pages=0,
                writing_pages=0,
                active_pages=0,
                retiring_pages=0,
                quarantined_pages=0,
                exhausted_pages=0,
                request_page_refs=0,
                prefix_page_refs=0,
                reader_pins=0,
            )
            for arena in self.arenas_by_class.values()
        )

    def census(self):
        return self.stats(), self.arena_stats()

    def has_request(self, key):
        return key in self.records

    def bind_request_rows(self, assignments):
        for key, row, is_new in assignments:
            if is_new:
                if (
                    key in self.request_rows
                    or row in self.row_owners
                ):
                    self.fail_stop("ReqToToken row ownership became uncertain")
                    raise FailStopped(self.failure_reason)
            elif (
                key not in self.records
                or self.request_rows.get(key) != row
                or self.row_owners.get(row) != key
            ):
                self.fail_stop("ReqToToken row ownership became uncertain")
                raise FailStopped(self.failure_reason)
        for key, row, is_new in assignments:
            if is_new:
                self.request_rows[key] = row
                self.row_owners[row] = key

    def rollback_request_rows(self, assignments):
        for key, row in assignments:
            if self.request_rows.get(key) != row or self.row_owners.get(row) != key:
                raise RuntimeError("row rollback identity changed")
        for key, row in assignments:
            del self.request_rows[key]
            del self.row_owners[row]

    def unbind_request_rows(self, assignments):
        values = tuple(assignments)
        if any(
            key in self.records
            or self.request_rows.get(key) != row
            or self.row_owners.get(row) != key
            for key, row in values
        ):
            self.fail_stop("ReqToToken row release changed ownership identity")
            raise FailStopped(self.failure_reason)
        for key, row in values:
            del self.request_rows[key]
            del self.row_owners[row]

    def record_for(self, key):
        return self.records[key]

    def bind_reclamation_cleanup(self, key, callback):
        self.callbacks[key] = callback
        self.records[key].reclamation_cleanup = callback

    def prepare_batch(self, items):
        keys = tuple(key for key, _ in items)
        new_keys = tuple(key for key in keys if key not in self.records)
        if new_keys:
            self.calls.append(("acquire_batch", len(new_keys)))
        for key in new_keys:
            self.records[key] = SimpleNamespace(
                lease=RequestLease(1, len(self.records), 1),
                boundary=0,
                reclamation_cleanup=None,
            )
        pending = tuple(
            SimpleNamespace(
                key=key,
                prepared=SimpleNamespace(request=self.records[key].lease),
            )
            for key in keys
        )
        lowering = tuple(
            LoweringPlan(
                self.records[key].lease,
                SnapshotLease(1, len(self.records), 1),
                SnapshotLease(1, len(self.records) + 1, 1),
                self.records[key].boundary,
                target,
                tuple(
                    ClassLoweringSpec(
                        item.class_id,
                        item.pool_id,
                        -1,
                        (1,),
                        TailAction(
                            item.class_id,
                            TAIL_NONE,
                            0,
                            0,
                            PageLease(0, 0, 0, 0, 0),
                            PageLease(0, 0, 0, 0, 0),
                        ),
                        (),
                    )
                    for item in self.config.classes
                ),
            )
            for key, target in items
        )
        self.calls.append(("prepare_batch", len(keys)))
        return BatchRecord(keys, pending), lowering

    def mark_lowered(self, batch):
        self.calls.append(("lowered_batch", len(batch.records)))

    def submit_batch(self, batch):
        self.calls.append(("submit_batch", len(batch.records)))
        return tuple(object() for _ in batch.records)

    def abort_unobserved(self, batch):
        self.calls.append(("abort_batch", len(batch.records)))

    def release_batch(self, keys):
        values = tuple(keys)
        self.calls.append(("release_batch", values))
        for key in values:
            self.records.pop(key, None)

    def lowering_failed(self, _batch, error):
        self.failure_reason = f"lowering: {error}"

    def candidate_mirror_failed(self, batch, error):
        self.calls.append(("mirror_failed", len(batch.records)))
        self.failure_reason = f"mirror: {error}"

    def fail_stop(self, reason):
        self.calls.append(("fail_stop", reason))
        self.failure_reason = reason


class FakeHybridAllocator:
    def __init__(self, mapping_size=512):
        self.full_to_swa_index_mapping = torch.zeros(
            mapping_size, dtype=torch.int64
        )

    def set_full_to_swa_mapping(self, full, swa):
        self.full_to_swa_index_mapping[full] = swa


class FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.zeros((8, 128), dtype=torch.int32)
        self.max_context_len = 128
        self.device = torch.device("cpu")
        self.freed = []

    def write(self, indices, values):
        rows, columns = indices
        self.req_to_token[rows, columns] = values

    def free(self, req):
        self.freed.append(req.rid)
        req.req_pool_idx = None


def _batch(config, allocator, *, decode=False):
    req_pool = FakeReqToTokenPool()
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1 if decode else None,
        prefix_indices=(
            torch.arange(16, 32, dtype=torch.int64)
            if decode
            else torch.empty((0,), dtype=torch.int64)
        ),
        kv=(
            ReqKvInfo(kv_allocated_len=16, swa_evicted_seqlen=0)
            if decode
            else None
        ),
    )
    tree_cache = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree_cache.token_to_kv_pool_allocator = allocator
    tree_cache.req_to_token_pool = req_pool
    tree_cache.page_size = 16
    tree_cache.disable_finished_insert = True
    req.effective_kv_committed_len = lambda: (
        0 if req.kv is None else int(req.kv.kv_allocated_len)
    )
    return SimpleNamespace(
        reqs=[req],
        req_to_token_pool=req_pool,
        tree_cache=tree_cache,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        enable_overlap=False,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        is_dllm=lambda: False,
        maybe_evict_swa=lambda: None,
        prefix_lens=[0],
        extend_lens=[16],
        extend_num_tokens=16,
        seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
        seq_lens=torch.tensor([16], dtype=torch.int64),
        req_pool_indices=torch.tensor([1], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
        device=torch.device("cpu"),
    )


def _install_transaction(config, runtime, allocator):
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(4, 64, 128),
        runtime=runtime,
    )
    state._ALLOCATOR = allocator


def _patch_extend_allocation(monkeypatch, *, fail_write=False):
    import sglang.srt.mem_cache.allocation as allocation

    def alloc_req_slots(pool, reqs, _tree):
        for index, req in enumerate(reqs):
            req.req_pool_idx = index + 1
        return list(range(1, len(reqs) + 1))

    def write_cache_indices(
        out,
        _rows_device,
        rows_cpu,
        _prefix_device,
        prefix_cpu,
        _seq_device,
        seq_cpu,
        _extend_device,
        _extend_cpu,
        _prefix_tensors,
        pool,
    ):
        if fail_write:
            raise RuntimeError("injected mirror failure")
        offset = 0
        for row, begin, end in zip(rows_cpu, prefix_cpu, seq_cpu, strict=True):
            length = int(end - begin)
            pool.req_to_token[int(row), int(begin) : int(end)] = out[
                offset : offset + length
            ].to(torch.int32)
            offset += length

    monkeypatch.setattr(allocation, "alloc_req_slots", alloc_req_slots)
    monkeypatch.setattr(allocation, "write_cache_indices", write_cache_indices)


def test_qwen_full_extend_writes_full_req_to_token(monkeypatch):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator)
    batch.seq_lens = torch.tensor([99], dtype=torch.int64)
    full_locations = torch.arange(16, 32, dtype=torch.int64)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: full_locations},
    )
    _patch_extend_allocation(monkeypatch)
    out, _, _ = lowering._alloc_for_extend(batch)
    assert torch.equal(out, full_locations)
    assert torch.equal(
        batch.req_to_token_pool.req_to_token[1, :16],
        full_locations.to(torch.int32),
    )
    assert torch.equal(batch.seq_lens, torch.tensor([16], dtype=torch.int64))
    assert [item[0] for item in runtime.calls] == [
        "acquire_batch",
        "prepare_batch",
        "lowered_batch",
        "submit_batch",
    ]


def test_gpt_oss_hybrid_decode_writes_full_row_and_swa_lut(monkeypatch):
    config = _config(("full", "sliding"))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator, decode=True)
    key = state._request_key(batch.reqs[0])
    lease = RequestLease(1, 0, 1)
    runtime.records[key] = SimpleNamespace(
        lease=lease, boundary=16, reclamation_cleanup=None
    )
    runtime.request_rows[key] = 1
    runtime.row_owners[1] = key
    batch.reqs[0]._orbitkv_request_key = key
    batch.reqs[0]._orbitkv_request_lease = lease
    batch.seq_lens = torch.tensor([99], dtype=torch.int64)
    batch.req_pool_indices = torch.tensor([3], dtype=torch.int64)
    full_locations = torch.tensor([32], dtype=torch.int64)
    swa_locations = torch.tensor([48], dtype=torch.int64)
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: {0: full_locations, 1: swa_locations},
    )
    out = lowering._alloc_for_decode(batch, 1)
    assert torch.equal(out, full_locations)
    assert int(batch.req_to_token_pool.req_to_token[1, 16]) == 32
    assert torch.equal(batch.seq_lens, torch.tensor([16], dtype=torch.int64))
    assert torch.equal(batch.req_pool_indices, torch.tensor([1], dtype=torch.int64))
    assert int(allocator.full_to_swa_index_mapping[32]) == 48
    assert batch.reqs[0].kv.kv_allocated_len == 17


def test_extend_rejects_a_row_owned_by_another_live_request_before_prepare(
    monkeypatch,
):
    import sglang.srt.mem_cache.allocation as allocation

    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    owner_key = ("str", "request-a")
    runtime.records[owner_key] = SimpleNamespace(
        lease=RequestLease(1, 0, 1), boundary=16, reclamation_cleanup=None
    )
    runtime.request_rows[owner_key] = 1
    runtime.row_owners[1] = owner_key
    batch = _batch(config, allocator)
    batch.reqs[0].rid = "request-b"
    lowering_calls = []

    def alias_live_row(_pool, reqs, _tree):
        reqs[0].req_pool_idx = 1
        return [1]

    monkeypatch.setattr(allocation, "alloc_req_slots", alias_live_row)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: lowering_calls.append(True),
    )
    with pytest.raises(FailStopped, match="row ownership"):
        lowering._alloc_for_extend(batch)
    assert list(runtime.records) == [owner_key]
    assert not any(
        item[0] in ("acquire_batch", "prepare_batch", "submit_batch")
        for item in runtime.calls
    )
    assert lowering_calls == []
    assert batch.req_to_token_pool.freed == []


def test_decode_rejects_live_request_row_migration_before_prepare(monkeypatch):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator, decode=True)
    key = state._request_key(batch.reqs[0])
    lease = RequestLease(1, 0, 1)
    runtime.records[key] = SimpleNamespace(
        lease=lease, boundary=16, reclamation_cleanup=None
    )
    runtime.request_rows[key] = 1
    runtime.row_owners[1] = key
    batch.reqs[0]._orbitkv_request_key = key
    batch.reqs[0]._orbitkv_request_lease = lease
    batch.reqs[0].req_pool_idx = 2
    batch.req_pool_indices_cpu = torch.tensor([2], dtype=torch.int64)
    batch.req_pool_indices = torch.tensor([2], dtype=torch.int64)
    lowering_calls = []
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: lowering_calls.append(True),
    )
    with pytest.raises(FailStopped, match="row ownership"):
        lowering._alloc_for_decode(batch, 1)
    assert not any(
        item[0] in ("prepare_batch", "submit_batch")
        for item in runtime.calls
    )
    assert lowering_calls == []


def test_post_submit_mirror_fault_fail_stops_transaction(monkeypatch):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator)
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: {0: torch.arange(16, 32, dtype=torch.int64)},
    )
    _patch_extend_allocation(monkeypatch, fail_write=True)
    with pytest.raises(FailStopped, match="mirror"):
        lowering._alloc_for_extend(batch)
    assert any(item[0] == "submit_batch" for item in runtime.calls)
    assert runtime.calls[-1] == ("mirror_failed", 1)


def test_extend_preflight_rejects_misaligned_vectors_without_side_effects(
    monkeypatch,
):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator)
    batch.extend_lens = [15]
    lowering_calls = []
    monkeypatch.setattr(
        lowering,
        "_lower_all_extend",
        lambda *_args: lowering_calls.append(True),
    )
    with pytest.raises(RuntimeError, match="extend boundaries"):
        lowering._alloc_for_extend(batch)
    assert runtime.calls == []
    assert lowering_calls == []
    assert batch.reqs[0].req_pool_idx is None


def test_extend_device_vector_preflight_never_reads_back_to_cpu():
    class CpuForbiddenTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        def cpu(self, *args, **kwargs):
            raise AssertionError("device metadata validation must not read back")

        def tolist(self):
            raise AssertionError("device metadata validation must not read values")

    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator)
    batch.seq_lens = CpuForbiddenTensor(
        torch.tensor([16], dtype=torch.int64)
    )
    assert validation._preflight_extend_batch(batch) == ((0,), (16,), (16,))


@pytest.mark.parametrize("allocated", ([0, 0], [99]))
def test_untrusted_request_row_allocation_fail_stops_without_free(
    monkeypatch, allocated
):
    import sglang.srt.mem_cache.allocation as allocation

    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator)

    def corrupt_rows(_pool, reqs, _tree):
        reqs[0].req_pool_idx = allocated[0]
        return allocated

    monkeypatch.setattr(allocation, "alloc_req_slots", corrupt_rows)
    with pytest.raises(FailStopped, match="request-row allocation"):
        lowering._alloc_for_extend(batch)
    assert batch.req_to_token_pool.freed == []
    assert runtime.failure_reason is not None


def test_decode_preflight_rejects_device_cardinality_without_side_effects(
    monkeypatch,
):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator, decode=True)
    batch.seq_lens = torch.tensor([[16]], dtype=torch.int64)
    lowering_calls = []
    monkeypatch.setattr(
        lowering,
        "_lower_all_decode",
        lambda *_args: lowering_calls.append(True),
    )
    with pytest.raises(RuntimeError, match="seq_lens cardinality"):
        lowering._alloc_for_decode(batch, 1)
    assert runtime.calls == []
    assert lowering_calls == []


def test_duplicate_live_rid_fails_before_prepare():
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    key = ("str", "request-1")
    runtime.records[key] = SimpleNamespace(lease=RequestLease(1, 0, 1), boundary=0)
    batch = _batch(config, allocator)
    with pytest.raises(RuntimeError, match="duplicate live request rid"):
        lowering._prepare_batch(batch, [0], [16])
    assert not any(item[0] == "prepare_batch" for item in runtime.calls)


def _certificate(class_id, *, backend_index, begin, end):
    return ReclamationCertificate(
        reclamation=ReclamationLease(1, backend_index + 1, 1),
        page=PageLease(
            1,
            2 + class_id,
            class_id + 1,
            1 + class_id * 16 + backend_index,
            1,
        ),
        class_id=class_id,
        backend_domain=class_id + 1,
        logical_ordinal=begin // 16,
        backend_index=backend_index,
        token_begin=begin,
        token_end_exclusive=end,
        completion_domain=1,
        completion_value=1,
    )


def _detached(certificate, *, reason=DETACHED_RETENTION):
    return DetachedBinding(
        old=certificate.page,
        replacement=PageLease(0, 0, 0, 0, 0),
        logical_ordinal=certificate.logical_ordinal,
        old_backend_index=certificate.backend_index,
        replacement_backend_index=0,
        token_begin=certificate.token_begin,
        token_end_exclusive=certificate.token_end_exclusive,
        class_id=certificate.class_id,
        backend_domain=certificate.backend_domain,
        action=DETACHED_CLEAR,
        reason=reason,
    )


def _run_mirror_cleanup(req, pool, allocator, certificates, releasing):
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    context = mirror_cleanup._MirrorCleanupContext(req, int(req.req_pool_idx))
    boundary = state._runtime().record_for(state._request_key(req)).boundary
    plan = coordinator.preflight(
        (
            MirrorCleanupItem(
                context=context,
                detached=tuple(_detached(item) for item in certificates),
                releasing=releasing,
                boundary=boundary,
            ),
        ),
        tuple(certificates),
    )
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)


def _b4_hybrid_cleanup_case():
    config = _config(("full", "sliding"))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    pool = FakeReqToTokenPool()
    pool.req_to_token = torch.zeros((5, 128), dtype=torch.int32)
    requests = []
    items = []
    retirements = []
    full_spans = []
    for index in range(4):
        row = index + 1
        full_begin = 64 + index * 64
        full_locations = torch.arange(
            full_begin, full_begin + 34, dtype=torch.int32
        )
        pool.req_to_token[row, :34] = full_locations
        swa_begin = (index + 1) * 16
        allocator.full_to_swa_index_mapping[
            full_begin : full_begin + 16
        ] = torch.arange(swa_begin, swa_begin + 16, dtype=torch.int64)
        req = SimpleNamespace(
            rid=f"request-{index}",
            req_pool_idx=row,
            prefix_indices=torch.arange(34, dtype=torch.int64) + full_begin,
            kv=ReqKvInfo(kv_allocated_len=34, swa_evicted_seqlen=0),
        )
        runtime.records[state._request_key(req)] = SimpleNamespace(
            lease=RequestLease(1, index, 1), boundary=34, reclamation_cleanup=None
        )
        requests.append(req)
        full_spans.append((full_begin, full_begin + 16))
        certificate = _certificate(1, backend_index=index, begin=0, end=16)
        retirements.append(certificate)
        items.append(
            MirrorCleanupItem(
                context=mirror_cleanup._MirrorCleanupContext(req, row),
                detached=(_detached(certificate),),
                releasing=False,
                boundary=34,
            )
        )
    return (
        pool,
        allocator,
        tuple(requests),
        tuple(items),
        tuple(retirements),
        tuple(full_spans),
    )


def test_b4_hybrid_collective_cleanup_uses_one_compare_and_one_sync(monkeypatch):
    pool, allocator, requests, items, retirements, full_spans = (
        _b4_hybrid_cleanup_case()
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    original_equal = torch.equal
    equal_calls = []
    sync_calls = []

    def tracked_equal(actual, expected):
        equal_calls.append((int(actual.numel()), int(expected.numel())))
        return original_equal(actual, expected)

    monkeypatch.setattr(torch, "equal", tracked_equal)
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    plan = coordinator.preflight(items, retirements)
    coordinator.commit(plan)
    coordinator.synchronize(plan)
    coordinator.finalize(plan)
    monkeypatch.setattr(torch, "equal", original_equal)

    assert equal_calls == [(1, 1)]
    assert sync_calls == [True]
    for req, (begin, end) in zip(requests, full_spans, strict=True):
        assert original_equal(
            allocator.full_to_swa_index_mapping[begin:end],
            torch.zeros(16, dtype=torch.int64),
        )
        assert req.kv.swa_evicted_seqlen == 16


def test_b4_hybrid_late_fault_is_zero_mutation_and_unacknowledged(monkeypatch):
    pool, allocator, requests, items, retirements, full_spans = (
        _b4_hybrid_cleanup_case()
    )
    fourth_begin, _ = full_spans[-1]
    allocator.full_to_swa_index_mapping[fourth_begin + 7] += 1
    original_rows = pool.req_to_token.clone()
    original_mapping = allocator.full_to_swa_index_mapping.clone()
    original_prefixes = tuple(req.prefix_indices.clone() for req in requests)
    original_equal = torch.equal
    equal_calls = []
    sync_calls = []
    acknowledgements = []

    def tracked_equal(actual, expected):
        equal_calls.append((int(actual.numel()), int(expected.numel())))
        return original_equal(actual, expected)

    monkeypatch.setattr(torch, "equal", tracked_equal)
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        plan = coordinator.preflight(items, retirements)
        coordinator.commit(plan)
        coordinator.synchronize(plan)
        coordinator.finalize(plan)
        acknowledgements.append(True)
    monkeypatch.setattr(torch, "equal", original_equal)

    assert equal_calls == [(1, 1)]
    assert sync_calls == []
    assert acknowledgements == []
    assert original_equal(pool.req_to_token, original_rows)
    assert original_equal(allocator.full_to_swa_index_mapping, original_mapping)
    assert all(
        original_equal(req.prefix_indices, prefix)
        for req, prefix in zip(requests, original_prefixes, strict=True)
    )
    assert all(req.kv.swa_evicted_seqlen == 0 for req in requests)


def _b4_pure_swa_cleanup_case():
    config = _config(("sliding",))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    pool = FakeReqToTokenPool()
    pool.req_to_token = torch.zeros((5, 128), dtype=torch.int32)
    requests = []
    items = []
    retirements = []
    for index in range(4):
        row = index + 1
        locations = torch.arange(
            (index + 1) * 16, (index + 2) * 16, dtype=torch.int64
        )
        pool.req_to_token[row, :16] = locations.to(torch.int32)
        req = SimpleNamespace(
            rid=f"pure-swa-{index}",
            req_pool_idx=row,
            prefix_indices=locations.clone(),
            kv=ReqKvInfo(kv_allocated_len=16, swa_evicted_seqlen=0),
        )
        runtime.records[state._request_key(req)] = SimpleNamespace(
            lease=RequestLease(1, index, 1), boundary=16, reclamation_cleanup=None
        )
        requests.append(req)
        certificate = _certificate(0, backend_index=index, begin=0, end=16)
        retirements.append(certificate)
        items.append(
            MirrorCleanupItem(
                context=mirror_cleanup._MirrorCleanupContext(req, row),
                detached=(_detached(certificate),),
                releasing=False,
                boundary=16,
            )
        )
    return pool, allocator, tuple(requests), tuple(items), tuple(retirements)


def test_b4_pure_swa_late_prefix_fault_is_zero_mutation(monkeypatch):
    pool, allocator, requests, items, retirements = _b4_pure_swa_cleanup_case()
    requests[-1].prefix_indices = requests[-1].prefix_indices.reshape(2, 8)
    original_rows = pool.req_to_token.clone()
    original_mapping = allocator.full_to_swa_index_mapping.clone()
    original_prefixes = tuple(req.prefix_indices.clone() for req in requests)
    original_frontiers = tuple(req.kv.swa_evicted_seqlen for req in requests)
    sync_calls = []
    acknowledgements = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )

    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    with pytest.raises(RuntimeError, match="prefix mirror"):
        plan = coordinator.preflight(items, retirements)
        coordinator.commit(plan)
        coordinator.synchronize(plan)
        coordinator.finalize(plan)
        acknowledgements.append(True)

    assert sync_calls == []
    assert acknowledgements == []
    assert torch.equal(pool.req_to_token, original_rows)
    assert torch.equal(allocator.full_to_swa_index_mapping, original_mapping)
    assert all(
        torch.equal(req.prefix_indices, original)
        for req, original in zip(requests, original_prefixes, strict=True)
    )
    assert tuple(req.kv.swa_evicted_seqlen for req in requests) == original_frontiers


def test_b4_pure_swa_late_frontier_fault_is_zero_mutation(monkeypatch):
    pool, allocator, requests, items, retirements = _b4_pure_swa_cleanup_case()
    requests[-1].kv.swa_evicted_seqlen = True
    original_rows = pool.req_to_token.clone()
    original_mapping = allocator.full_to_swa_index_mapping.clone()
    original_prefixes = tuple(req.prefix_indices.clone() for req in requests)
    original_frontiers = tuple(req.kv.swa_evicted_seqlen for req in requests)
    sync_calls = []
    acknowledgements = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )

    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    with pytest.raises(RuntimeError, match="retention frontier"):
        plan = coordinator.preflight(items, retirements)
        coordinator.commit(plan)
        coordinator.synchronize(plan)
        coordinator.finalize(plan)
        acknowledgements.append(True)

    assert sync_calls == []
    assert acknowledgements == []
    assert torch.equal(pool.req_to_token, original_rows)
    assert torch.equal(allocator.full_to_swa_index_mapping, original_mapping)
    assert all(
        torch.equal(req.prefix_indices, original)
        for req, original in zip(requests, original_prefixes, strict=True)
    )
    assert tuple(req.kv.swa_evicted_seqlen for req in requests) == original_frontiers


def test_cleanup_certificate_cannot_cross_target_boundary():
    pool, allocator, _requests, items, retirements = _b4_pure_swa_cleanup_case()
    original_rows = pool.req_to_token.clone()
    invalid_items = (*items[:-1], replace(items[-1], boundary=15))

    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    with pytest.raises(RuntimeError, match="exceeds its KV boundary"):
        coordinator.preflight(invalid_items, retirements)

    assert torch.equal(pool.req_to_token, original_rows)


def test_release_prefix_fault_is_zero_mutation_and_unacknowledged(monkeypatch):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    pool = FakeReqToTokenPool()
    locations = torch.arange(16, 32, dtype=torch.int64)
    pool.req_to_token[1, :16] = locations.to(torch.int32)
    req = SimpleNamespace(
        rid="release-prefix-fault",
        req_pool_idx=1,
        prefix_indices=locations.clone(),
        kv=ReqKvInfo(kv_allocated_len=16, swa_evicted_seqlen=0),
    )
    req.prefix_indices[-1] += 1
    original_row = pool.req_to_token.clone()
    original_prefix = req.prefix_indices.clone()
    sync_calls = []
    acknowledgements = []
    monkeypatch.setattr(
        mirror_cleanup, "_synchronize_mirror", lambda _pool: sync_calls.append(True)
    )
    certificate = _certificate(0, backend_index=0, begin=0, end=16)
    item = MirrorCleanupItem(
        context=mirror_cleanup._MirrorCleanupContext(req, 1),
        detached=(_detached(certificate),),
        releasing=True,
        boundary=16,
    )

    coordinator = mirror_cleanup._MirrorCleanupCoordinator(pool, allocator)
    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        plan = coordinator.preflight((item,), (certificate,))
        coordinator.commit(plan)
        coordinator.synchronize(plan)
        coordinator.finalize(plan)
        acknowledgements.append(True)

    assert sync_calls == []
    assert acknowledgements == []
    assert torch.equal(pool.req_to_token, original_row)
    assert torch.equal(req.prefix_indices, original_prefix)


def test_hybrid_swa_completion_clears_only_lut_then_advances_frontier(
    monkeypatch,
):
    config = _config(("full", "sliding"))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1,
        prefix_indices=torch.arange(48, 64, dtype=torch.int64),
        kv=ReqKvInfo(kv_allocated_len=16, swa_evicted_seqlen=0),
    )
    key = state._request_key(req)
    runtime.records[key] = SimpleNamespace(lease=RequestLease(1, 0, 1), boundary=16)
    pool = FakeReqToTokenPool()
    full_locations = torch.arange(48, 64, dtype=torch.int32)
    pool.req_to_token[1, :16] = full_locations
    allocator.full_to_swa_index_mapping[48:64] = torch.arange(16, 32)
    observations = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: observations.append(
            (
                int(allocator.full_to_swa_index_mapping[48:64].sum()),
                req.kv.swa_evicted_seqlen,
            )
        ),
    )
    _run_mirror_cleanup(
        req,
        pool,
        allocator,
        (_certificate(1, backend_index=0, begin=0, end=16),),
        False,
    )
    assert torch.equal(pool.req_to_token[1, :16], full_locations)
    assert torch.equal(req.prefix_indices, torch.arange(48, 64))
    assert observations == [(0, 0)]
    assert req.kv.swa_evicted_seqlen == 16


def test_hybrid_swa_batches_multiple_certificates_before_one_cleanup_sync(
    monkeypatch,
):
    config = _config(("full", "sliding"))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1,
        prefix_indices=torch.arange(80, 128, dtype=torch.int64),
        kv=ReqKvInfo(kv_allocated_len=48, swa_evicted_seqlen=0),
    )
    runtime.records[state._request_key(req)] = SimpleNamespace(
        lease=RequestLease(1, 0, 1), boundary=48, reclamation_cleanup=None
    )
    pool = FakeReqToTokenPool()
    full_locations = torch.arange(80, 128, dtype=torch.int32)
    pool.req_to_token[1, :48] = full_locations
    allocator.full_to_swa_index_mapping[80:128] = torch.arange(16, 64)
    certificates = tuple(
        _certificate(
            1,
            backend_index=ordinal,
            begin=ordinal * 16,
            end=(ordinal + 1) * 16,
        )
        for ordinal in range(3)
    )

    original_equal = torch.equal
    equal_calls = []

    def tracked_equal(actual, expected):
        equal_calls.append((int(actual.numel()), int(expected.numel())))
        return original_equal(actual, expected)

    monkeypatch.setattr(torch, "equal", tracked_equal)
    synchronization_observations = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronization_observations.append(
            (
                int(allocator.full_to_swa_index_mapping[80:128].sum()),
                req.kv.swa_evicted_seqlen,
            )
        ),
    )
    _run_mirror_cleanup(
        req,
        pool,
        allocator,
        certificates,
        False,
    )
    monkeypatch.setattr(torch, "equal", original_equal)

    assert equal_calls == [(1, 1)]
    assert synchronization_observations == [(0, 0)]
    assert torch.equal(pool.req_to_token[1, :48], full_locations)
    assert torch.equal(req.prefix_indices, torch.arange(80, 128))
    assert req.kv.swa_evicted_seqlen == 48


def test_hybrid_swa_forged_middle_certificate_cannot_partially_clear(
    monkeypatch,
):
    config = _config(("full", "sliding"))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1,
        prefix_indices=torch.arange(80, 128, dtype=torch.int64),
        kv=ReqKvInfo(kv_allocated_len=48, swa_evicted_seqlen=0),
    )
    runtime.records[state._request_key(req)] = SimpleNamespace(
        lease=RequestLease(1, 0, 1), boundary=48, reclamation_cleanup=None
    )
    pool = FakeReqToTokenPool()
    pool.req_to_token[1, :48] = torch.arange(80, 128, dtype=torch.int32)
    allocator.full_to_swa_index_mapping[80:128] = torch.arange(16, 64)
    certificates = (
        _certificate(1, backend_index=0, begin=0, end=16),
        _certificate(1, backend_index=7, begin=16, end=32),
        _certificate(1, backend_index=2, begin=32, end=48),
    )
    original_mirror = pool.req_to_token[1].clone()
    original_mapping = allocator.full_to_swa_index_mapping.clone()
    original_prefix = req.prefix_indices.clone()

    original_equal = torch.equal
    equal_calls = []

    def tracked_equal(actual, expected):
        equal_calls.append((int(actual.numel()), int(expected.numel())))
        return original_equal(actual, expected)

    monkeypatch.setattr(torch, "equal", tracked_equal)
    synchronization_calls = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronization_calls.append(True),
    )
    with pytest.raises(RuntimeError, match="disagrees with the SGLang mirror"):
        _run_mirror_cleanup(
            req,
            pool,
            allocator,
            certificates,
            False,
        )
    monkeypatch.setattr(torch, "equal", original_equal)

    assert equal_calls == [(1, 1)]
    assert synchronization_calls == []
    assert torch.equal(pool.req_to_token[1], original_mirror)
    assert torch.equal(allocator.full_to_swa_index_mapping, original_mapping)
    assert torch.equal(req.prefix_indices, original_prefix)
    assert req.kv.swa_evicted_seqlen == 0


def test_pure_swa_completion_clears_req_row_and_prefix_before_ack(monkeypatch):
    config = _config(("sliding",))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1,
        prefix_indices=torch.arange(16, 32, dtype=torch.int64),
        kv=ReqKvInfo(kv_allocated_len=16, swa_evicted_seqlen=0),
    )
    runtime.records[state._request_key(req)] = SimpleNamespace(
        lease=RequestLease(1, 0, 1), boundary=16, reclamation_cleanup=None
    )
    pool = FakeReqToTokenPool()
    pool.req_to_token[1, :16] = torch.arange(16, 32, dtype=torch.int32)
    observed = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: observed.append(
            (
                int(pool.req_to_token[1, :16].sum()),
                int(req.prefix_indices.sum()),
            )
        ),
    )
    _run_mirror_cleanup(
        req,
        pool,
        allocator,
        (_certificate(0, backend_index=0, begin=0, end=16),),
        False,
    )
    assert observed == [(0, 0)]
    assert req.kv.swa_evicted_seqlen == 16


def test_pure_swa_batches_multiple_certificates_before_one_cleanup_sync(
    monkeypatch,
):
    config = _config(("sliding",))
    runtime = TransactionRuntime(config)
    allocator = FakeHybridAllocator()
    _install_transaction(config, runtime, allocator)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1,
        prefix_indices=torch.arange(16, 64, dtype=torch.int64),
        kv=ReqKvInfo(kv_allocated_len=48, swa_evicted_seqlen=0),
    )
    runtime.records[state._request_key(req)] = SimpleNamespace(
        lease=RequestLease(1, 0, 1), boundary=48, reclamation_cleanup=None
    )
    pool = FakeReqToTokenPool()
    pool.req_to_token[1, :48] = torch.arange(16, 64, dtype=torch.int32)
    certificates = tuple(
        _certificate(
            0,
            backend_index=ordinal,
            begin=ordinal * 16,
            end=(ordinal + 1) * 16,
        )
        for ordinal in range(3)
    )

    original_equal = torch.equal
    equal_calls = []

    def tracked_equal(actual, expected):
        equal_calls.append((int(actual.numel()), int(expected.numel())))
        return original_equal(actual, expected)

    monkeypatch.setattr(torch, "equal", tracked_equal)
    synchronization_observations = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: synchronization_observations.append(
            (
                int(pool.req_to_token[1, :48].sum()),
                int(req.prefix_indices.sum()),
                req.kv.swa_evicted_seqlen,
            )
        ),
    )
    _run_mirror_cleanup(
        req,
        pool,
        allocator,
        certificates,
        False,
    )
    monkeypatch.setattr(torch, "equal", original_equal)

    assert equal_calls == [(1, 1)]
    assert synchronization_observations == [(0, 0, 0)]
    assert req.kv.swa_evicted_seqlen == 48


def test_full_release_validates_then_clears_row_and_prefix_before_ack(monkeypatch):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    req = SimpleNamespace(
        rid="request-1",
        req_pool_idx=1,
        prefix_indices=torch.arange(16, 32, dtype=torch.int64),
        kv=ReqKvInfo(kv_allocated_len=16, swa_evicted_seqlen=0),
    )
    runtime.records[state._request_key(req)] = SimpleNamespace(
        lease=RequestLease(1, 0, 1), boundary=16, reclamation_cleanup=None
    )
    pool = FakeReqToTokenPool()
    pool.req_to_token[1, :16] = torch.arange(16, 32, dtype=torch.int32)
    observed = []
    monkeypatch.setattr(
        mirror_cleanup,
        "_synchronize_mirror",
        lambda _pool: observed.append(
            (
                int(pool.req_to_token[1, :16].sum()),
                int(req.prefix_indices.sum()),
            )
        ),
    )
    _run_mirror_cleanup(
        req,
        pool,
        allocator,
        (_certificate(0, backend_index=0, begin=0, end=16),),
        True,
    )
    assert observed == [(0, 0)]


def test_release_empty_request_is_idempotent():
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = object()
    _install_transaction(config, runtime, allocator)
    batch = _batch(config, allocator)
    lowering._release_kv_cache(batch.reqs[0], batch.tree_cache)
    assert runtime.calls == []


def test_sglang_nested_free_group_fail_stops_without_changing_collection_state():
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = facade._NativeAuthorityForbidden()
    _install_transaction(config, runtime, allocator)
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []

    allocator.free_group_begin()
    with pytest.raises(FailStopped, match="nested an OrbitKV release group"):
        allocator.free_group_begin()

    assert runtime.failure_reason == "SGLang nested an OrbitKV release group"
    assert runtime.calls == [
        ("fail_stop", "SGLang nested an OrbitKV release group")
    ]
    assert allocator._orbitkv_free_group_state == "collecting"
    assert not allocator.is_not_in_free_group
    assert allocator.free_group == []


def test_sglang_inactive_free_group_end_fail_stops_without_entering_flush():
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = facade._NativeAuthorityForbidden()
    _install_transaction(config, runtime, allocator)
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []

    with pytest.raises(FailStopped, match="inactive OrbitKV release group"):
        allocator.free_group_end()

    assert runtime.failure_reason == "SGLang ended an inactive OrbitKV release group"
    assert runtime.calls == [
        ("fail_stop", "SGLang ended an inactive OrbitKV release group")
    ]
    assert allocator._orbitkv_free_group_state == "idle"
    assert allocator.is_not_in_free_group
    assert allocator.free_group == []


def test_sglang_free_group_flush_fault_is_terminal_and_stays_flushing(
    monkeypatch,
):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = facade._NativeAuthorityForbidden()
    _install_transaction(config, runtime, allocator)
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    batch = _batch(config, allocator, decode=True)
    req = batch.reqs[0]
    key = state._request_key(req)
    lease = RequestLease(1, 0, 1)
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease
    runtime.records[key] = SimpleNamespace(
        lease=lease, boundary=16, reclamation_cleanup=None
    )
    runtime.request_rows[key] = 1
    runtime.row_owners[1] = key

    allocator.free_group_begin()
    lowering._release_kv_cache(req, batch.tree_cache)

    def fail_flush(candidates):
        assert len(candidates) == 1
        raise RuntimeError("injected grouped flush fault")

    monkeypatch.setattr(lowering, "_flush_release_group", fail_flush)
    with pytest.raises(FailStopped, match="release-group flush failed"):
        allocator.free_group_end()

    assert runtime.failure_reason == (
        "OrbitKV release-group flush failed: injected grouped flush fault"
    )
    assert runtime.calls == [
        (
            "fail_stop",
            "OrbitKV release-group flush failed: injected grouped flush fault",
        )
    ]
    assert allocator._orbitkv_free_group_state == "flushing"
    assert allocator.is_not_in_free_group
    assert allocator.free_group == []
    assert runtime.records == {
        key: SimpleNamespace(lease=lease, boundary=16, reclamation_cleanup=None)
    }
    assert runtime.request_rows == {key: 1}
    assert runtime.row_owners == {1: key}
    assert batch.req_to_token_pool.freed == []
    assert req.req_pool_idx == 1


def test_sglang_b1_release_outside_group_runs_each_release_stage_once(
    monkeypatch,
):
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = facade._NativeAuthorityForbidden()
    _install_transaction(config, runtime, allocator)
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    batch = _batch(config, allocator, decode=True)
    req = batch.reqs[0]
    key = state._request_key(req)
    lease = RequestLease(1, 0, 1)
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease
    runtime.records[key] = SimpleNamespace(
        lease=lease, boundary=16, reclamation_cleanup=None
    )
    runtime.request_rows[key] = 1
    runtime.row_owners[1] = key
    events = []
    original_release_batch = runtime.release_batch
    original_free = batch.req_to_token_pool.free
    original_unbind = runtime.unbind_request_rows

    def release_batch(keys):
        values = tuple(keys)
        events.append(("release_batch", values))
        original_release_batch(values)

    def free(released_req):
        events.append(("free", released_req.rid))
        original_free(released_req)

    def unbind(assignments):
        values = tuple(assignments)
        events.append(("unbind", values))
        original_unbind(values)

    monkeypatch.setattr(runtime, "release_batch", release_batch)
    monkeypatch.setattr(batch.req_to_token_pool, "free", free)
    monkeypatch.setattr(runtime, "unbind_request_rows", unbind)

    lowering._release_kv_cache(req, batch.tree_cache)

    assert events == [
        ("release_batch", (key,)),
        ("free", "request-1"),
        ("unbind", ((key, 1),)),
    ]
    assert runtime.calls == [("release_batch", (key,))]
    assert runtime.records == {}
    assert runtime.request_rows == {}
    assert runtime.row_owners == {}
    assert batch.req_to_token_pool.freed == ["request-1"]
    assert req.req_pool_idx is None
    assert req.kv is None
    assert not hasattr(req, "_orbitkv_request_key")
    assert not hasattr(req, "_orbitkv_request_lease")


def test_sglang_free_group_flushes_one_b4_release_transaction():
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = facade._NativeAuthorityForbidden()
    _install_transaction(config, runtime, allocator)
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    pool = FakeReqToTokenPool()
    tree_cache = object.__new__(prefix_cache.OrbitKvPrefixCache)
    tree_cache.token_to_kv_pool_allocator = allocator
    tree_cache.req_to_token_pool = pool
    tree_cache.disable_finished_insert = True
    requests = []
    keys = []
    for index in range(4):
        req = SimpleNamespace(
            rid=f"release-{index}",
            req_pool_idx=index + 1,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            kv=SimpleNamespace(),
        )
        key = state._request_key(req)
        lease = RequestLease(1, index, 1)
        req._orbitkv_request_key = key
        req._orbitkv_request_lease = lease
        req.effective_kv_committed_len = lambda: 0
        runtime.records[key] = SimpleNamespace(
            lease=lease, boundary=0, reclamation_cleanup=None
        )
        runtime.request_rows[key] = index + 1
        runtime.row_owners[index + 1] = key
        requests.append(req)
        keys.append(key)

    allocator.free_group_begin()
    for req in requests:
        lowering._release_kv_cache(req, tree_cache)
    assert runtime.calls == []
    assert pool.freed == []
    assert all(req.req_pool_idx is not None for req in requests)

    allocator.free_group_end()
    assert runtime.calls == [("release_batch", tuple(keys))]
    assert pool.freed == [f"release-{index}" for index in range(4)]
    assert allocator._orbitkv_free_group_state == "idle"
    assert allocator.is_not_in_free_group
    assert allocator.free_group == []
    assert runtime.records == {}
    assert runtime.request_rows == {}
    assert runtime.row_owners == {}
    assert all(req.req_pool_idx is None and req.kv is None for req in requests)
    assert all(req.prefix_indices.numel() == 0 for req in requests)

    reused_req = SimpleNamespace(rid=7001)
    reused_key = state._request_key(reused_req)
    reused_lease = RequestLease(1, 0, 2)
    assert reused_key == ("int", 7001)
    runtime.bind_request_rows(((reused_key, 1, True),))
    runtime.records[reused_key] = SimpleNamespace(
        lease=reused_lease,
        boundary=0,
    )
    assert all(key not in runtime.records for key in keys)
    assert all(key not in runtime.request_rows for key in keys)
    assert runtime.request_rows == {reused_key: 1}
    assert runtime.row_owners == {1: reused_key}
    assert runtime.records[reused_key].lease == RequestLease(1, 0, 2)


def test_sglang_free_group_rejects_duplicate_before_any_release_mutation():
    config = _config(("full",))
    runtime = TransactionRuntime(config)
    allocator = facade._NativeAuthorityForbidden()
    _install_transaction(config, runtime, allocator)
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    batch = _batch(config, allocator, decode=True)
    req = batch.reqs[0]
    key = state._request_key(req)
    lease = RequestLease(1, 0, 1)
    req._orbitkv_request_key = key
    req._orbitkv_request_lease = lease
    runtime.records[key] = SimpleNamespace(
        lease=lease, boundary=16, reclamation_cleanup=None
    )
    runtime.request_rows[key] = 1
    runtime.row_owners[1] = key

    allocator.free_group_begin()
    lowering._release_kv_cache(req, batch.tree_cache)
    with pytest.raises(FailStopped, match="duplicated a request"):
        lowering._release_kv_cache(req, batch.tree_cache)
    assert runtime.has_request(key)
    assert req.req_pool_idx == 1
    assert batch.req_to_token_pool.freed == []


def test_qwen_and_gpt_oss_geometry_gates_allow_native_attention_partition():
    for retentions, architecture, layers in (
        (("full",), "Qwen2ForCausalLM", ((0,), ())),
        (("full", "sliding"), "GptOssForCausalLM", ((0,), (1,))),
    ):
        config = _config(retentions)
        state._install_test_state(config=config)
        model = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            hf_text_config=SimpleNamespace(
                num_hidden_layers=len(retentions),
                num_key_value_heads=2,
            ),
            head_dim=16,
            v_head_dim=16,
            swa_head_dim=16,
            swa_v_head_dim=16,
            is_hybrid_swa=len(retentions) == 2,
            full_attention_layer_ids=list(layers[0]),
            swa_attention_layer_ids=list(layers[1]),
            sliding_window_size=32,
            disable_hybrid_swa_memory=False,
            is_deepseek_v4_arch=False,
            is_hybrid_swa_compress=False,
            attention_chunk_size=None,
            has_attention_sinks=architecture == "GptOssForCausalLM",
        )
        validation._validate_checkpoint_geometry(
            SimpleNamespace(model_config=model, kv_cache_dtype=torch.bfloat16)
        )


@pytest.mark.parametrize(
    "architecture, expected, rejected",
    (
        ("Qwen2ForCausalLM", ("flashinfer", "flashinfer"), ("fa3", "fa3")),
        ("GptOssForCausalLM", ("fa3", "fa3"), ("flashinfer", "flashinfer")),
    ),
)
def test_attention_backend_contract_is_exact_per_architecture(
    architecture, expected, rejected
):
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture])
    )
    configurator = SimpleNamespace(
        model_config=model,
        server_args=SimpleNamespace(get_attention_backends=lambda: expected),
    )
    assert validation._validate_attention_backend_contract(configurator) == architecture

    configurator.server_args.get_attention_backends = lambda: rejected
    with pytest.raises(RuntimeError, match=rf"{architecture} requires"):
        validation._validate_attention_backend_contract(configurator)


@pytest.mark.parametrize("backend", ("flashinfer", "fa3"))
def test_pinned_server_args_reports_explicit_uniform_backend_pair(backend):
    from sglang.srt.server_args import ServerArgs

    server = SimpleNamespace(
        attention_backend=backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )
    assert ServerArgs.get_attention_backends(server) == (backend, backend)
