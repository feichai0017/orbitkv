from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import MambaPool

from orbitkv_sglang.ffi import CtypesStatePool
from orbitkv_sglang.config import (
    ClassConfig,
    FixedStateConfig,
    RuntimeConfig,
    load_config,
)
import orbitkv_sglang.bridge.state as bridge_state
from orbitkv_sglang.bridge.fixed_state import (
    FixedStateCoordinator,
    _execute_fixed_state_deferred,
    _fixed_state_alloc,
    _fixed_state_free,
    _fixed_state_pool_clear,
)
from orbitkv_sglang.bridge.prefix_cache import OrbitKvPrefixCache
from orbitkv_sglang.bridge.validation import (
    _validate_checkpoint_geometry,
    _validate_fixed_state_options,
    _validate_fixed_state_pool,
    _validate_gdn_fixed_state_backend_contract,
    _validate_radix_cache_contract,
)
from orbitkv_sglang.runtime import (
    CacheSharingPolicy,
    FailStopped,
    ManagerError,
    StatePoolConfig,
)
from manifest_test_support import runtime_manifest_environment
from ffi_test_support import ffi_library


class _MambaPool:
    def __init__(self, size: int):
        self.size = size
        self.mamba_cache = SimpleNamespace(
            conv=[torch.zeros((1, size + 1, 2, 3), dtype=torch.bfloat16)],
            temporal=torch.zeros((1, size + 1, 2, 2, 2), dtype=torch.float32),
        )
        self.enable_linear_replayssm = False
        self.enable_linear_replayssm_spec = False
        self.calls = []

    def clear_slots(self, indices):
        self.calls.append(("clear", tuple(int(value) for value in indices)))
        for tensor in (*self.mamba_cache.conv, self.mamba_cache.temporal):
            tensor[:, indices] = 0

    def copy_from(self, sources, destinations):
        self.calls.append(
            (
                "copy",
                tuple(int(value) for value in sources),
                tuple(int(value) for value in destinations),
            )
        )
        for tensor in (*self.mamba_cache.conv, self.mamba_cache.temporal):
            tensor[:, destinations] = tensor[:, sources]


class _NativeAllocator:
    def __init__(self, size: int):
        self.size = size
        self.device = "cpu"


class _ReqPool:
    def __init__(self, size: int):
        self.device = "cpu"
        self.mamba_pool = _MambaPool(size)
        self.mamba_allocator = _NativeAllocator(size)
        self.mamba_ckpt_pool = None
        self.enable_mamba_extra_buffer = False
        self.enable_mamba_extra_buffer_lazy = False
        self.req_to_token = torch.zeros((size + 1, 16), dtype=torch.int32)
        self.free_slots = list(range(1, size + 1))
        self.req_generation = torch.zeros(size + 1, dtype=torch.int64)
        self.req_index_to_mamba_index_mapping = torch.zeros(
            size + 1, dtype=torch.int32
        )


class _Event:
    def __init__(self, ready=True, *, fail_wait=False):
        self.ready = ready
        self.fail_wait = fail_wait
        self.recorded = []
        self.waits = 0

    def record(self, stream=None):
        self.recorded.append(stream)

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        if self.fail_wait:
            raise RuntimeError("event not ready")
        self.ready = True


class _DeviceModule:
    def __init__(self):
        self.events = []

    def current_stream(self, _device):
        return "forward-stream"

    def Event(self):
        event = _Event()
        self.events.append(event)
        return event


class _FaultEvent(_Event):
    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage

    def record(self, stream=None):
        if self.stage == "record":
            raise RuntimeError("injected event record failure")
        super().record(stream=stream)

    def synchronize(self):
        if self.stage == "synchronize":
            raise RuntimeError("injected event synchronize failure")
        super().synchronize()


class _FaultDeviceModule(_DeviceModule):
    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage

    def Event(self):
        if self.stage == "construct":
            raise RuntimeError("injected event construction failure")
        event = _FaultEvent(self.stage)
        self.events.append(event)
        return event


class _Mode:
    def __init__(self, *, extend=True, target_verify=False, draft_extend_v2=False):
        self._extend = extend
        self._target_verify = target_verify
        self._draft_extend_v2 = draft_extend_v2

    def is_extend(self):
        return self._extend

    def is_target_verify(self):
        return self._target_verify

    def is_draft_extend_v2(self):
        return self._draft_extend_v2


def _coordinator(library: Path, size: int = 4):
    req_pool = _ReqPool(size)
    native = CtypesStatePool(library, StatePoolConfig(11, 13, 56, 9, size))
    coordinator = FixedStateCoordinator(
        native, req_pool, device_module=_DeviceModule()
    )
    coordinator.install_allocator_facade()
    return coordinator, req_pool


def _request(rid: str, slot: int, generation: int = 1):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=slot,
        mamba_pool_idx=None,
        mamba_needs_clear=False,
        mamba_cow_src_index=None,
        _orbitkv_engine_request_id=slot,
    )


def _forward(records):
    keys = tuple(record.key for record in records)
    rows = tuple(int(record.request.req_pool_idx) for record in records)
    return SimpleNamespace(
        _orbitkv_state_records=tuple(records),
        _orbitkv_state_forward_keys=keys,
        _orbitkv_state_forward_rows=rows,
        rids=[record.request.rid for record in records],
        forward_mode=_Mode(),
        mamba_clear_indices=torch.tensor(
            [record.intent.destination.slot_id + 1 for record in records],
            dtype=torch.int64,
        ),
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
    )


def _publish_initial(coordinator, req_pool, req):
    records = coordinator.prepare_for_allocated_rows((req,), (req.req_pool_idx,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    )
    coordinator.register_event(
        tuple(record.key for record in records),
        records,
        _Event(),
        1,
        coordinator._device_module,
        "cpu",
    )
    coordinator.poll()
    return records


def test_decode_ignores_stale_deferred_indices_without_owner_records(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = SimpleNamespace(
        _orbitkv_state_records=(),
        forward_mode=_Mode(extend=False),
        mamba_clear_indices=torch.tensor([1], dtype=torch.int64),
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
    )
    calls = []

    result = _execute_fixed_state_deferred(
        lambda _runner, value: calls.append(value) or "decode-noop",
        runner,
        forward,
    )

    assert result == "decode-noop"
    assert calls == [forward]
    assert coordinator.failed is None
    assert coordinator.stats().free_slots == 2
    coordinator.close()


def test_target_extend_rejects_unowned_deferred_indices(ffi_library):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = SimpleNamespace(
        _orbitkv_state_records=(),
        forward_mode=_Mode(),
        mamba_clear_indices=torch.tensor([1], dtype=torch.int64),
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
    )

    with pytest.raises(ManagerError, match="unowned deferred Mamba operation"):
        coordinator.preflight_deferred(runner, forward)

    assert coordinator.failed is None
    coordinator.close()


def test_owned_transition_rejects_decode_forward_mode(ffi_library):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    req = _request("a", 1)
    records = coordinator.prepare_for_allocated_rows((req,), (1,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    forward.forward_mode = _Mode(extend=False)

    with pytest.raises(ManagerError, match="unsupported forward mode"):
        coordinator.preflight_deferred(runner, forward)

    coordinator.abort_batch(records)
    coordinator.close()


@pytest.mark.parametrize(
    ("tampered_section", "message"),
    (
        (
            "source",
            "attention_state_plan differs from its attention-state source",
        ),
        ("derived", "bytes_per_token_per_layer differs from the manager input"),
    ),
)
def test_runtime_manifest_rejects_resealed_source_or_derived_tampering(
    tmp_path, ffi_library, tampered_section, message
):
    attention_state = {
        "page_tokens": 16,
        "states": [
            {
                "name": "full_mha",
                "layers": [0],
                "storage": {
                    "kind": "token_kv",
                    "key_bytes_per_token_per_layer": 64,
                    "value_bytes_per_token_per_layer": 64,
                    "retention": "full",
                    "window_tokens": None,
                },
            },
            {
                "name": "mamba_state",
                "layers": [1],
                "storage": {
                    "kind": "recurrent",
                    "family": "mamba",
                    "state_bytes_per_layer": 96,
                    "checkpoint_slots_per_request": 2,
                },
            },
        ],
    }
    environment = runtime_manifest_environment(
        tmp_path,
        ffi_library,
        attention_state=attention_state,
        stem=f"fixed-state-{tampered_section}",
    )
    config = load_config(environment)
    assert config.num_hidden_layers == 2
    assert config.fixed_state_byte_count == 96
    assert config.fixed_states[0].kind == "mamba"
    manifest_path = Path(environment["ORBITKV_RUNTIME_MANIFEST"])
    assert config.runtime_manifest_path == manifest_path.resolve()
    assert config.runtime_manifest_fingerprint.startswith("sha256:")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tampered_section == "source":
        manifest["source"]["input"]["states"][0]["storage"][
            "value_bytes_per_token_per_layer"
        ] = 32
    else:
        manifest["token_manager_plan"]["layout"]["classes"][0][
            "bytes_per_token_per_layer"
        ] = 96
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["fingerprint"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(environment)


def test_real_sglang_mamba_pool_methods_copy_and_clear_cpu_tensors():
    pool = object.__new__(MambaPool)
    pool.debug_memory_pool = False
    pool.replayssm_write_pos = None
    pool.replayssm_cache_base = None
    pool.replayssm_is_flush = None
    pool.mamba_cache = SimpleNamespace(
        conv=[torch.zeros((2, 3, 4, 2), dtype=torch.bfloat16)],
        temporal=torch.zeros((2, 3, 2, 2, 2), dtype=torch.float32),
    )
    pool.mamba_cache.conv[0][:, 1].fill_(3)
    pool.mamba_cache.temporal[:, 1].fill_(5)
    source = torch.tensor([1], dtype=torch.int64)
    destination = torch.tensor([2], dtype=torch.int64)

    MambaPool.copy_from(pool, source, destination)
    assert torch.equal(
        pool.mamba_cache.conv[0][:, 2], pool.mamba_cache.conv[0][:, 1]
    )
    assert torch.equal(
        pool.mamba_cache.temporal[:, 2], pool.mamba_cache.temporal[:, 1]
    )
    MambaPool.clear_slots(pool, destination)
    assert not torch.count_nonzero(pool.mamba_cache.conv[0][:, 2])
    assert not torch.count_nonzero(pool.mamba_cache.temporal[:, 2])
    assert torch.count_nonzero(pool.mamba_cache.conv[0][:, 1])
    assert torch.count_nonzero(pool.mamba_cache.temporal[:, 1])
    assert not torch.count_nonzero(pool.mamba_cache.conv[0][:, 0])


def test_fixed_state_initial_forward_event_release_and_generation_reuse(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    records = coordinator.prepare_for_allocated_rows((req,), (1,))
    assert int(req.mamba_pool_idx) == records[0].intent.destination.slot_id + 1 == 1
    assert int(req_pool.req_index_to_mamba_index_mapping[1]) == 1

    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    assert _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    ) is None
    assert req_pool.mamba_pool.calls == [("clear", (1,))]
    event = _Event()
    coordinator.register_event(
        (("str", "a"),), records, event, 1, coordinator._device_module, "cpu"
    )
    coordinator.poll()
    first = req._orbitkv_state_lease
    assert coordinator.stats().live_slots == 1

    req_pool.mamba_pool.mamba_cache.conv[0][:, 1].fill_(7)
    coordinator.retire_batch(((("str", "a"), req),))
    assert req.mamba_pool_idx is None
    assert int(req_pool.req_index_to_mamba_index_mapping[1]) == 0
    assert coordinator.stats().free_slots == 2

    reused = _request("b", 1, generation=2)
    next_record = coordinator.prepare_for_allocated_rows((reused,), (1,))[0]
    assert next_record.intent.destination.slot_id == first.slot_id
    assert next_record.intent.destination.generation > first.generation
    coordinator.abort_batch((next_record,))
    coordinator.close()


def test_same_owner_replacement_uses_real_mamba_copy(ffi_library, monkeypatch):
    coordinator, req_pool = _coordinator(ffi_library, 3)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    initial = coordinator.prepare_for_allocated_rows((req,), (1,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    first_forward = _forward(initial)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        first_forward,
    )
    coordinator.register_event(
        (("str", "a"),),
        initial,
        _Event(),
        1,
        coordinator._device_module,
        "cpu",
    )
    coordinator.poll()
    old_slot = int(req.mamba_pool_idx)
    req_pool.mamba_pool.mamba_cache.conv[0][:, old_slot].fill_(3)
    req_pool.mamba_pool.mamba_cache.temporal[:, old_slot].fill_(5)

    replacement = coordinator.prepare_replacement_batch((req,), (1,))
    replacement_forward = SimpleNamespace(
        _orbitkv_state_records=replacement,
        _orbitkv_state_forward_keys=(("str", "a"),),
        _orbitkv_state_forward_rows=(1,),
        rids=[req.rid],
        forward_mode=_Mode(),
        mamba_clear_indices=None,
        mamba_cow_src_indices=torch.tensor([old_slot], dtype=torch.int64),
        mamba_cow_dst_indices=torch.tensor(
            [int(req.mamba_pool_idx)], dtype=torch.int64
        ),
    )
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.copy_from(
                value.mamba_cow_src_indices, value.mamba_cow_dst_indices
            ),
            setattr(value, "mamba_cow_src_indices", None),
            setattr(value, "mamba_cow_dst_indices", None),
        )[0],
        runner,
        replacement_forward,
    )
    new_slot = int(req.mamba_pool_idx)
    assert new_slot != old_slot
    assert torch.equal(
        req_pool.mamba_pool.mamba_cache.conv[0][:, new_slot],
        req_pool.mamba_pool.mamba_cache.conv[0][:, old_slot],
    )
    coordinator.register_event(
        (("str", "a"),),
        replacement,
        _Event(),
        1,
        coordinator._device_module,
        "cpu",
    )
    coordinator.poll()
    assert coordinator.stats().live_slots == 1
    coordinator.retire_batch(((("str", "a"), req),))
    coordinator.close()


def test_mixed_continuing_and_new_prepare_keeps_ordered_new_subset_and_mirrors(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 4)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    continuing = _request("continuing", 1)
    _publish_initial(coordinator, req_pool, continuing)
    continuing_lease = continuing._orbitkv_state_lease
    continuing_slot = int(continuing.mamba_pool_idx)

    new_b = _request("new-b", 2, generation=2)
    new_c = _request("new-c", 3, generation=3)
    records = coordinator.prepare_for_allocated_rows(
        (continuing, new_b, new_c), (1, 2, 3)
    )

    assert tuple(record.key for record in records) == (
        ("str", "new-b"),
        ("str", "new-c"),
    )
    assert tuple(record.request for record in records) == (new_b, new_c)
    assert continuing._orbitkv_state_lease == continuing_lease
    assert int(continuing.mamba_pool_idx) == continuing_slot
    assert int(req_pool.req_index_to_mamba_index_mapping[1]) == continuing_slot
    for row, request, record in zip((2, 3), (new_b, new_c), records, strict=True):
        destination = record.intent.destination.slot_id + 1
        assert int(request.mamba_pool_idx) == destination
        assert int(req_pool.req_index_to_mamba_index_mapping[row]) == destination

    forward = _forward(records)
    forward.rids = [continuing.rid, new_b.rid, new_c.rid]
    forward._orbitkv_state_forward_keys = all_keys = (
        ("str", "continuing"),
        ("str", "new-b"),
        ("str", "new-c"),
    )
    forward._orbitkv_state_forward_rows = (1, 2, 3)
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    )
    assert req_pool.mamba_pool.calls[-1] == (
        "clear",
        tuple(record.intent.destination.slot_id + 1 for record in records),
    )
    coordinator.register_event(
        all_keys, records, _Event(), 1, coordinator._device_module, "cpu"
    )
    coordinator.poll()
    coordinator.retire_batch(
        tuple(
            (key, request)
            for key, request in zip(
                all_keys, (continuing, new_b, new_c), strict=True
            )
        )
    )
    coordinator.close()


@pytest.mark.parametrize("drift", ("row", "destination"))
def test_mixed_continuing_and_new_prepare_rejects_row_or_destination_drift(
    ffi_library, monkeypatch, drift
):
    coordinator, req_pool = _coordinator(ffi_library, 3)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    continuing = _request("continuing", 1)
    _publish_initial(coordinator, req_pool, continuing)
    new = _request("new", 2, generation=2)
    if drift == "row":
        req_pool.req_index_to_mamba_index_mapping[1] = 0
    else:
        continuing.mamba_pool_idx = torch.tensor(3, dtype=torch.int64)

    with pytest.raises(ManagerError, match="live request identity changed"):
        coordinator.prepare_for_allocated_rows((continuing, new), (1, 2))

    assert new.mamba_pool_idx is None
    assert int(req_pool.req_index_to_mamba_index_mapping[2]) == 0
    continuing.mamba_pool_idx = torch.tensor(
        continuing._orbitkv_state_lease.slot_id + 1, dtype=torch.int64
    )
    req_pool.req_index_to_mamba_index_mapping[1] = continuing.mamba_pool_idx
    coordinator.retire_batch(((("str", "continuing"), continuing),))
    coordinator.close()


def test_initial_prepare_rejects_stale_mamba_mirror(ffi_library):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    req = _request("stale", 1)
    req.mamba_pool_idx = torch.tensor(1, dtype=torch.int64)
    req_pool.req_index_to_mamba_index_mapping[1] = 1

    with pytest.raises(ManagerError, match="lost its published slot"):
        coordinator.prepare_for_allocated_rows((req,), (1,))

    req.mamba_pool_idx = None
    req_pool.req_index_to_mamba_index_mapping[1] = 0
    coordinator.close()


def test_same_owner_replacement_abort_restores_published_source(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 3)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    _publish_initial(coordinator, req_pool, req)
    key = ("str", "a")
    source_lease = req._orbitkv_state_lease
    source_slot = int(req.mamba_pool_idx)
    source_mapping = int(req_pool.req_index_to_mamba_index_mapping[1])
    source_owner = req._orbitkv_state_owner_id

    replacement = coordinator.prepare_replacement_batch((req,), (1,))
    assert int(req.mamba_pool_idx) != source_slot
    coordinator.abort_batch(replacement)

    assert req._orbitkv_state_lease == source_lease
    assert int(req.mamba_pool_idx) == source_slot
    assert int(req_pool.req_index_to_mamba_index_mapping[1]) == source_mapping
    assert req._orbitkv_state_owner_id == source_owner
    assert req._orbitkv_state_key == key
    assert coordinator._owners[key] == source_owner
    assert coordinator._owner_keys[source_owner] == key
    assert coordinator._state_pool.current_batch((source_owner,)) == (
        (source_owner, source_lease),
    )
    assert coordinator.stats().live_slots == 1
    assert coordinator.stats().reserved_slots == 0

    coordinator.retire_batch(((key, req),))
    coordinator.close()


def test_fixed_state_event_defers_publication_and_failure_never_reuses(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    records = coordinator.prepare_for_allocated_rows((req,), (1,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    )
    event = _Event(ready=False, fail_wait=True)
    coordinator.register_event(
        (("str", "a"),), records, event, 1, coordinator._device_module, "cpu"
    )
    coordinator.poll()
    assert not hasattr(req, "_orbitkv_state_lease")
    with pytest.raises(FailStopped, match="event wait failed"):
        coordinator.wait_keys((("str", "a"),))
    assert coordinator.stats().relocating_slots == 1
    coordinator.close()


def test_decode_forward_event_advances_release_frontier(ffi_library, monkeypatch):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    records = coordinator.prepare_for_allocated_rows((req,), (1,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    )
    first = _Event()
    coordinator.register_event(
        (("str", "a"),), records, first, 1, coordinator._device_module, "cpu"
    )
    coordinator.poll()

    decode = _Event(ready=False, fail_wait=True)
    coordinator.register_event(
        (("str", "a"),), (), decode, 1, coordinator._device_module, "cpu"
    )
    with pytest.raises(FailStopped, match="event wait failed"):
        coordinator.retire_batch(((("str", "a"), req),))
    assert coordinator.stats().live_slots == 1
    coordinator.close()


def test_ready_decode_event_is_required_before_release(ffi_library, monkeypatch):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    records = coordinator.prepare_for_allocated_rows((req,), (1,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    )
    coordinator.register_event(
        (("str", "a"),),
        records,
        _Event(),
        1,
        coordinator._device_module,
        "cpu",
    )
    coordinator.poll()
    decode = _Event(ready=False)
    coordinator.register_event(
        (("str", "a"),),
        (),
        decode,
        1,
        coordinator._device_module,
        "cpu",
    )
    coordinator.retire_batch(((("str", "a"), req),))
    assert decode.waits == 1
    assert coordinator.stats().free_slots == 2
    coordinator.close()


def test_failed_retirement_clear_never_acknowledges_or_reuses(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    records = coordinator.prepare_for_allocated_rows((req,), (1,))
    runner = SimpleNamespace(req_to_token_pool=req_pool, is_draft_worker=False)
    forward = _forward(records)
    _execute_fixed_state_deferred(
        lambda _runner, value: (
            req_pool.mamba_pool.clear_slots(value.mamba_clear_indices),
            setattr(value, "mamba_clear_indices", None),
        )[0],
        runner,
        forward,
    )
    coordinator.register_event(
        (("str", "a"),),
        records,
        _Event(),
        1,
        coordinator._device_module,
        "cpu",
    )
    coordinator.poll()
    original_clear = req_pool.mamba_pool.clear_slots

    def fail_clear(_indices):
        raise RuntimeError("injected clear failure")

    req_pool.mamba_pool.clear_slots = fail_clear
    with pytest.raises(FailStopped, match="retirement clear/ACK failed"):
        coordinator.retire_batch(((("str", "a"), req),))
    stats = coordinator.stats()
    assert stats.retiring_slots == stats.pending_retirements == 1
    assert stats.free_slots == 1
    req_pool.mamba_pool.clear_slots = original_clear
    coordinator.close()


@pytest.mark.parametrize("stage", ("construct", "record", "synchronize", "ack"))
def test_retirement_event_or_ack_failure_never_reuses_slot(
    ffi_library, monkeypatch, stage
):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    _publish_initial(coordinator, req_pool, req)
    retired_slot = req._orbitkv_state_lease.slot_id
    coordinator._device_module = _FaultDeviceModule(stage)
    coordinator._current_stream = lambda: "retirement-stream"
    if stage == "ack":
        monkeypatch.setattr(
            coordinator._state_pool,
            "acknowledge_batch",
            lambda _certificates: (_ for _ in ()).throw(
                RuntimeError("injected acknowledge failure")
            ),
        )

    with pytest.raises(FailStopped, match="retirement clear/ACK failed"):
        coordinator.retire_batch(((("str", "a"), req),))

    stats = coordinator.stats()
    assert stats.retiring_slots == stats.pending_retirements == 1
    assert stats.free_slots == 1
    other = _request("b", 2, generation=2)
    intent = coordinator._state_pool.prepare_batch(
        ((coordinator._request_owner(other), None),)
    )[0]
    assert intent.destination.slot_id != retired_slot
    coordinator._state_pool.abort_batch((intent.transition,))
    coordinator.close()


def test_fixed_state_rejects_native_allocator_and_dummy_slot(ffi_library):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    assert coordinator.available_size() == 2
    with pytest.raises(RuntimeError, match="bypassed"):
        req_pool.mamba_allocator.alloc(1)
    with pytest.raises(RuntimeError, match="bypassed"):
        req_pool.mamba_allocator.free(torch.tensor([1]))
    assert int(req_pool.req_index_to_mamba_index_mapping[0]) == 0
    coordinator.close()


def test_fixed_state_free_hook_fails_closed_without_native_release(
    ffi_library, monkeypatch
):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    native_calls = []

    with pytest.raises(RuntimeError, match="bypassed OrbitKV retirement/ACK"):
        _fixed_state_free(
            lambda *_args: native_calls.append("native"), req_pool, req
        )

    assert native_calls == []
    assert req.req_pool_idx == 1
    assert req.mamba_pool_idx is None
    coordinator.close()


def test_fixed_state_alloc_hook_uses_only_request_rows(ffi_library, monkeypatch):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    req.req_pool_idx = None
    req.inflight_middle_chunks = 0
    req.kv_committed_len = 0
    rows = _fixed_state_alloc(lambda *_args: pytest.fail("native alloc called"), req_pool, (req,))
    assert rows == [1]
    assert req.req_pool_idx == 1
    assert req.mamba_pool_idx is None
    req_pool.req_index_to_mamba_index_mapping[1] = 0
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    ReqToTokenPool.free(req_pool, req)
    coordinator.close()


def test_fixed_state_profile_never_advertises_mamba_prefix_sharing():
    cache = object.__new__(OrbitKvPrefixCache)
    assert cache.supports_mamba() is False
    assert cache.mamba_evictable_size() == cache.mamba_protected_size() == 0


@pytest.mark.parametrize(
    "field",
    (
        "extra",
        "lazy",
        "replayssm",
        "replayssm_spec",
        "int8",
    ),
)
def test_restricted_fixed_state_options_fail_closed(field, monkeypatch):
    config = SimpleNamespace(fixed_states=(object(),))
    monkeypatch.setattr(bridge_state, "_CONFIG", config)
    values = {
        name: False
        for name in ("extra", "lazy", "replayssm", "replayssm_spec", "int8")
    }
    values[field] = True
    server = SimpleNamespace(
        enable_mamba_extra_buffer=lambda: values["extra"],
        enable_mamba_extra_buffer_lazy=lambda: values["lazy"],
        enable_linear_replayssm=values["replayssm"],
        enable_linear_replayssm_spec=values["replayssm_spec"],
        enable_int8_mamba_checkpoint=values["int8"],
    )
    configurator = SimpleNamespace(
        server_args=server,
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(is_kda=False)
        ),
    )
    with pytest.raises(RuntimeError, match="restricted fixed-state profile rejects"):
        _validate_fixed_state_options(configurator)


@pytest.mark.parametrize(
    ("kind", "admitted_policy", "disabled"),
    (
        (None, CacheSharingPolicy.SHARED_PREFIX, False),
        ("mamba", CacheSharingPolicy.REQUEST_PRIVATE, True),
        ("gdn", CacheSharingPolicy.REQUEST_PRIVATE, True),
    ),
)
def test_radix_cache_contract_keeps_orbitkv_backend(
    monkeypatch, kind, admitted_policy, disabled
):
    monkeypatch.setattr(
        bridge_state,
        "_CONFIG",
        SimpleNamespace(
            fixed_states=(SimpleNamespace(kind=kind),) if kind else ()
        ),
    )
    monkeypatch.setattr(
        bridge_state, "_cache_sharing_policy", lambda: admitted_policy
    )
    configurator = SimpleNamespace(
        server_args=SimpleNamespace(
            disable_radix_cache=disabled, radix_cache_backend="orbitkv"
        )
    )
    _validate_radix_cache_contract(configurator)

    configurator.server_args.radix_cache_backend = None
    with pytest.raises(RuntimeError, match="radix-cache-backend"):
        _validate_radix_cache_contract(configurator)


@pytest.mark.parametrize(
    ("value", "message"),
    (("true", "does not support"), ("invalid", "valid boolean")),
)
def test_radix_cache_contract_rejects_force_miss(monkeypatch, value, message):
    monkeypatch.setenv("SGLANG_RADIX_FORCE_MISS", value)
    monkeypatch.setattr(bridge_state, "_CONFIG", SimpleNamespace(fixed_states=()))
    configurator = SimpleNamespace(
        server_args=SimpleNamespace(
            disable_radix_cache=False, radix_cache_backend="orbitkv"
        )
    )

    with pytest.raises(RuntimeError, match=message):
        _validate_radix_cache_contract(configurator)


@pytest.mark.parametrize(
    ("kind", "admitted_policy", "disabled"),
    (
        (None, CacheSharingPolicy.SHARED_PREFIX, True),
        ("mamba", CacheSharingPolicy.REQUEST_PRIVATE, False),
        ("gdn", CacheSharingPolicy.REQUEST_PRIVATE, False),
    ),
)
def test_radix_cache_contract_rejects_wrong_disable_mode(
    monkeypatch, kind, admitted_policy, disabled
):
    monkeypatch.setattr(
        bridge_state,
        "_CONFIG",
        SimpleNamespace(
            fixed_states=(SimpleNamespace(kind=kind),) if kind else ()
        ),
    )
    monkeypatch.setattr(
        bridge_state, "_cache_sharing_policy", lambda: admitted_policy
    )
    configurator = SimpleNamespace(
        server_args=SimpleNamespace(
            disable_radix_cache=disabled, radix_cache_backend="orbitkv"
        )
    )
    with pytest.raises(RuntimeError, match="disable-radix-cache"):
        _validate_radix_cache_contract(configurator)


def _mixed_fixed_state_config(
    *,
    recurrent_kind="gdn",
    recurrent_layers=(1,),
    convolution_layers=(1,),
    recurrent_bytes=8,
    convolution_bytes=6,
    recurrent_slots=2,
    convolution_slots=2,
    kernel_width=4,
):
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:fixed-state-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                name="full_mha",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
        ),
        fixed_states=(
            FixedStateConfig(
                "recurrent",
                recurrent_kind,
                recurrent_layers,
                recurrent_bytes,
                recurrent_slots,
            ),
            FixedStateConfig(
                "conv",
                "convolution",
                convolution_layers,
                convolution_bytes,
                convolution_slots,
                kernel_width,
            ),
        ),
    )


def _fixed_state_params(
    *,
    layers=(1,),
    conv_shape=(1, 3),
    temporal_shape=(1, 1, 2),
    conv_kernel=4,
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,
    is_kda=False,
):
    shape = SimpleNamespace(
        conv=[conv_shape], temporal=temporal_shape, conv_kernel=conv_kernel
    )
    dtype = SimpleNamespace(conv=conv_dtype, temporal=temporal_dtype)
    conv_bytes = sum(
        torch.empty((), dtype=conv_dtype).element_size()
        * torch.Size(value).numel()
        for value in shape.conv
    )
    recurrent_bytes = (
        torch.empty((), dtype=temporal_dtype).element_size()
        * torch.Size(temporal_shape).numel()
    )
    return SimpleNamespace(
        layers=list(layers),
        shape=shape,
        dtype=dtype,
        is_kda=is_kda,
        mamba_cache_per_req=(conv_bytes + recurrent_bytes) * len(layers),
    )


def _checkpoint_configurator(*, architecture, params, hybrid_gdn):
    mambaish = SimpleNamespace(
        full_attention_layer_ids=[0],
        mamba2_cache_params=params,
        linear_conv_kernel_dim=params.shape.conv_kernel,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            hf_text_config=SimpleNamespace(
                num_hidden_layers=2, num_key_value_heads=2
            ),
            head_dim=16,
            v_head_dim=16,
            swa_head_dim=16,
            swa_v_head_dim=16,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[],
            sliding_window_size=32,
            disable_hybrid_swa_memory=False,
            is_deepseek_v4_arch=False,
            is_hybrid_swa_compress=False,
            attention_chunk_size=None,
        ),
        kv_cache_dtype=torch.bfloat16,
        use_mla_backend=False,
        mambaish_config=mambaish,
        hybrid_gdn_config=mambaish if hybrid_gdn else None,
    )


def _gdn_production_backend_configurator(**overrides):
    values = {
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "linear_attn_prefill_backend": "triton",
        "mamba_ssm_dtype": "float32",
        "mamba_radix_cache_strategy": "no_buffer",
        "disable_radix_cache": True,
    }
    values.update(overrides)
    attention_backends = values.pop("attention_backends", ("fa3", "fa3"))
    params = values.pop("params", _fixed_state_params())
    configurator = _checkpoint_configurator(
        architecture="RenamedGdnForConditionalGeneration",
        params=params,
        hybrid_gdn=True,
    )
    configurator.server_args = SimpleNamespace(
        get_attention_backends=lambda: attention_backends, **values
    )
    return configurator


def _real_shape_hybrid_pool(size=2):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_ckpt_pool = None
    pool.enable_mamba_extra_buffer = False
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_pool = object.__new__(MambaPool)
    pool.mamba_pool.size = size
    pool.mamba_pool.enable_linear_replayssm = False
    pool.mamba_pool.enable_linear_replayssm_spec = False
    pool.mamba_pool.mamba_cache = SimpleNamespace(
        conv=[torch.zeros((1, size + 1, 1, 3), dtype=torch.bfloat16)],
        temporal=torch.zeros((1, size + 1, 1, 1, 2), dtype=torch.float32),
    )
    return pool


def test_fixed_state_pool_accepts_exact_component_tensors():
    _validate_fixed_state_pool(
        _real_shape_hybrid_pool(),
        _mixed_fixed_state_config(),
        _fixed_state_params(),
    )


def test_fixed_state_pool_keeps_legacy_mamba_aggregate_geometry():
    config = _mixed_fixed_state_config(recurrent_kind="mamba")
    recurrent = FixedStateConfig("recurrent", "mamba", (1,), 14, 2)
    config = RuntimeConfig(
        library_path=config.library_path,
        plan_json=config.plan_json,
        plan_fingerprint=config.plan_fingerprint,
        page_tokens=config.page_tokens,
        classes=config.classes,
        fixed_states=(recurrent,),
    )
    _validate_fixed_state_pool(
        _real_shape_hybrid_pool(), config, _fixed_state_params()
    )


def test_fixed_state_pool_rejects_mamba_component_split_drift_at_same_total():
    config = _mixed_fixed_state_config(
        recurrent_kind="mamba", recurrent_bytes=6, convolution_bytes=8
    )

    with pytest.raises(RuntimeError, match="component geometry differs"):
        _validate_fixed_state_pool(
            _real_shape_hybrid_pool(), config, _fixed_state_params()
        )


@pytest.mark.parametrize("component", ("recurrent", "convolution"))
def test_fixed_state_pool_rejects_component_byte_drift(component):
    config = _mixed_fixed_state_config()
    recurrent, convolution = config.fixed_states
    if component == "recurrent":
        recurrent = FixedStateConfig(
            recurrent.name, recurrent.kind, recurrent.layers, 7, 2
        )
    else:
        convolution = FixedStateConfig(
            convolution.name,
            convolution.kind,
            convolution.layers,
            5,
            2,
            convolution.kernel_width,
        )
    config = RuntimeConfig(
        library_path=config.library_path,
        plan_json=config.plan_json,
        plan_fingerprint=config.plan_fingerprint,
        page_tokens=config.page_tokens,
        classes=config.classes,
        fixed_states=(recurrent, convolution),
    )

    with pytest.raises(RuntimeError, match="component geometry differs"):
        _validate_fixed_state_pool(
            _real_shape_hybrid_pool(), config, _fixed_state_params()
        )


def test_checkpoint_geometry_keeps_legacy_mamba_aggregate_profile(monkeypatch):
    config = _mixed_fixed_state_config(recurrent_kind="mamba")
    config = RuntimeConfig(
        library_path=config.library_path,
        plan_json=config.plan_json,
        plan_fingerprint=config.plan_fingerprint,
        page_tokens=config.page_tokens,
        classes=config.classes,
        fixed_states=(FixedStateConfig("recurrent", "mamba", (1,), 14, 2),),
    )
    monkeypatch.setattr(bridge_state, "_CONFIG", config)
    configurator = _checkpoint_configurator(
        architecture="LegacyMambaForCausalLM",
        params=_fixed_state_params(),
        hybrid_gdn=False,
    )
    _validate_checkpoint_geometry(configurator)


def test_checkpoint_geometry_accepts_renamed_exact_gdn_profile(monkeypatch):
    monkeypatch.setattr(bridge_state, "_CONFIG", _mixed_fixed_state_config())
    configurator = _checkpoint_configurator(
        architecture="RenamedGdnForConditionalGeneration",
        params=_fixed_state_params(),
        hybrid_gdn=True,
    )
    _validate_checkpoint_geometry(configurator)


def test_gdn_production_backend_profile_accepts_renamed_architecture(monkeypatch):
    monkeypatch.setattr(bridge_state, "_CONFIG", _mixed_fixed_state_config())
    configurator = _gdn_production_backend_configurator()

    _validate_gdn_fixed_state_backend_contract(configurator)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("attention_backends", ("flashinfer", "flashinfer"), "Full attention FA3"),
        ("linear_attn_backend", None, "linear_attn_backend=triton"),
        ("linear_attn_decode_backend", None, "linear_attn_decode_backend=triton"),
        ("linear_attn_prefill_backend", None, "linear_attn_prefill_backend=triton"),
        ("mamba_ssm_dtype", None, "mamba_ssm_dtype=float32"),
        ("mamba_radix_cache_strategy", "auto", "mamba_radix_cache_strategy=no_buffer"),
    ),
)
def test_gdn_production_backend_profile_rejects_drift(
    monkeypatch, field, value, message
):
    monkeypatch.setattr(bridge_state, "_CONFIG", _mixed_fixed_state_config())
    configurator = _gdn_production_backend_configurator(**{field: value})

    with pytest.raises(RuntimeError, match=message):
        _validate_gdn_fixed_state_backend_contract(configurator)


def test_gdn_production_backend_profile_rejects_actual_temporal_dtype(
    monkeypatch,
):
    monkeypatch.setattr(bridge_state, "_CONFIG", _mixed_fixed_state_config())
    configurator = _gdn_production_backend_configurator(
        params=_fixed_state_params(temporal_dtype=torch.bfloat16)
    )

    with pytest.raises(
        RuntimeError, match="mamba2_cache_params.dtype.temporal=float32"
    ):
        _validate_gdn_fixed_state_backend_contract(configurator)


@pytest.mark.parametrize(
    "drift",
    (
        "missing_gdn",
        "foreign_gdn",
        "kda",
        "layers",
        "slots",
        "kernel",
        "component_bytes",
        "conv_dtype",
        "temporal_dtype",
    ),
)
def test_checkpoint_geometry_rejects_inexact_gdn_profile(monkeypatch, drift):
    config_kwargs = {}
    params_kwargs = {}
    architecture = "RenamedGdnForConditionalGeneration"
    if drift == "kda":
        params_kwargs["is_kda"] = True
    elif drift == "layers":
        config_kwargs["convolution_layers"] = (0,)
    elif drift == "slots":
        config_kwargs["convolution_slots"] = 3
    elif drift == "kernel":
        config_kwargs["kernel_width"] = 5
    elif drift == "component_bytes":
        config_kwargs.update(recurrent_bytes=7, convolution_bytes=7)
    elif drift == "conv_dtype":
        params_kwargs["conv_dtype"] = torch.float32
        config_kwargs.update(recurrent_bytes=8, convolution_bytes=12)
    elif drift == "temporal_dtype":
        params_kwargs["temporal_dtype"] = torch.bfloat16
        config_kwargs.update(recurrent_bytes=4, convolution_bytes=6)
    config = _mixed_fixed_state_config(**config_kwargs)
    params = _fixed_state_params(**params_kwargs)
    configurator = _checkpoint_configurator(
        architecture=architecture, params=params, hybrid_gdn=drift != "missing_gdn"
    )
    if drift == "foreign_gdn":
        configurator.hybrid_gdn_config = SimpleNamespace(linear_conv_kernel_dim=4)
    monkeypatch.setattr(bridge_state, "_CONFIG", config)

    with pytest.raises(RuntimeError):
        _validate_checkpoint_geometry(configurator)


@pytest.mark.parametrize(
    "drift",
    (
        "conv_count",
        "conv_shape",
        "temporal_shape",
        "conv_dtype",
        "temporal_dtype",
        "layer_axis",
        "slot_axis",
    ),
)
def test_fixed_state_pool_rejects_tensor_geometry_drift(drift):
    pool = _real_shape_hybrid_pool()
    if drift == "conv_count":
        pool.mamba_pool.mamba_cache.conv.append(
            torch.zeros((1, 3, 3, 1), dtype=torch.bfloat16)
        )
    elif drift == "conv_shape":
        pool.mamba_pool.mamba_cache.conv[0] = torch.zeros(
            (1, 3, 3, 1), dtype=torch.bfloat16
        )
    elif drift == "temporal_shape":
        pool.mamba_pool.mamba_cache.temporal = torch.zeros(
            (1, 3, 1, 2), dtype=torch.float32
        )
    elif drift == "conv_dtype":
        pool.mamba_pool.mamba_cache.conv[0] = torch.zeros(
            (1, 3, 1, 3), dtype=torch.float32
        )
    elif drift == "temporal_dtype":
        pool.mamba_pool.mamba_cache.temporal = torch.zeros(
            (1, 3, 1, 1, 2), dtype=torch.bfloat16
        )
    elif drift == "layer_axis":
        pool.mamba_pool.mamba_cache.temporal = torch.zeros(
            (2, 3, 1, 1, 2), dtype=torch.float32
        )
    else:
        pool.mamba_pool.mamba_cache.conv[0] = torch.zeros(
            (1, 4, 1, 3), dtype=torch.bfloat16
        )
    with pytest.raises(RuntimeError, match="tensor geometry changed"):
        _validate_fixed_state_pool(
            pool, _mixed_fixed_state_config(), _fixed_state_params()
        )


def test_fixed_state_pool_clear_requires_quiescence(ffi_library, monkeypatch):
    coordinator, req_pool = _coordinator(ffi_library, 2)
    monkeypatch.setattr(bridge_state, "_FIXED_STATE", coordinator)
    req = _request("a", 1)
    coordinator.prepare_for_allocated_rows((req,), (1,))
    with pytest.raises(FailStopped, match="non-quiescent"):
        _fixed_state_pool_clear(lambda _pool: pytest.fail("native clear called"), req_pool)
    assert coordinator.stats().reserved_slots == 1
    coordinator.close()


__all__ = ["ffi_library"]
