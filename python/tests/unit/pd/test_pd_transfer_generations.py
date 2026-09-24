"""Queued P/D writes must retain the authorization that produced their ranges."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .pd_connector_test_utils import (
    RELEASE_CONSUMER_ABORT,
    RELEASE_PRODUCER_PREEMPTED,
    BlockRegionSlice,
    FakeMooncakeTransferEngine,
    FakeTensor,
    LayerBlockSlices,
    PdConnectorMetadata,
    PdHandshake,
    PdPrefillWorkerConnector,
    PushReqMeta,
    RealMooncakePort,
    hnd_remote_layer,
    prefill_worker_mod,
)


@pytest.fixture
def push_state():
    engine = FakeMooncakeTransferEngine()
    transfer = RealMooncakePort(engine)
    layer = hnd_remote_layer(block_ids=(0, 1))
    transfer.register_local_layers((layer,))
    handshake = PdHandshake(
        request_id="remote-old",
        engine_id="decode",
        transfer_endpoint="old-peer:1",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(layer,),
    )
    blocks = [
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=0, src_offset_bytes=0, bytes=0x400),
                BlockRegionSlice(block_id=0, src_offset_bytes=0x7000, bytes=0x400),
            )
        )
    ]
    return engine, transfer, handshake, blocks


def test_queued_push_cannot_borrow_reopened_request_authorization(push_state):
    engine, transfer, handshake, blocks = push_state
    generation = transfer.open_request("req", handshake)
    event_entered = threading.Event()
    ready = threading.Event()

    class Event:
        def synchronize(self):
            event_entered.set()
            assert ready.wait(timeout=3)

    sender = prefill_worker_mod._AsyncLayerPushSender(max_workers=1)
    try:
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                transfer=transfer,
                req_id="req",
                layer_idx=0,
                block_slices=blocks,
                request_generation=generation,
                event=Event(),
            )
        )
        assert event_entered.wait(timeout=3)
        transfer.close_request("req")
        replacement = replace(handshake, request_id="remote-new", transfer_endpoint="new-peer:1")
        new_generation = transfer.open_request("req", replacement)
        assert new_generation != generation
        ready.set()
        with pytest.raises(RuntimeError, match="stale Mooncake push generation"):
            sender.wait_req("req")
        assert engine.writes == []
        assert transfer.write_stats("req")["bytes"] == 0
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                transfer=transfer,
                req_id="req",
                layer_idx=0,
                block_slices=blocks,
                request_generation=new_generation,
            )
        )
        sender.wait_req("req")
        assert [endpoint for endpoint, _, _ in engine.writes] == ["new-peer:1"]
        assert transfer.write_stats("req")["bytes"] == 0x800
    finally:
        ready.set()
        sender.close()


def test_reopen_during_descriptor_construction_rejects_before_transport(push_state, monkeypatch):
    import orbitkv.vllm.pd.mooncake as mooncake

    engine, transfer, handshake, blocks = push_state
    generation = transfer.open_request("req", handshake)
    original = mooncake._layer_blocks_to_native

    def replace_during_construction(blocks):
        transfer.close_request("req")
        transfer.open_request("req", replace(handshake, transfer_endpoint="new-peer:1"))
        return original(blocks)

    monkeypatch.setattr(mooncake, "_layer_blocks_to_native", replace_during_construction)
    with pytest.raises(RuntimeError, match="stale Mooncake push generation"):
        transfer.push_layer("req", 0, blocks, request_generation=generation)
    assert engine.writes == []
    assert transfer.write_stats("req")["bytes"] == 0


def test_admitted_write_completion_cannot_update_replacement_statistics(push_state, monkeypatch):
    engine, transfer, handshake, blocks = push_state
    generation = transfer.open_request("req", handshake)
    submitted = threading.Event()
    completed = threading.Event()
    drained = threading.Event()
    errors = []
    original = engine.write

    def blocked_write(*args, **kwargs):
        submitted.set()
        assert completed.wait(timeout=3)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "write", blocked_write)
    sender = prefill_worker_mod._AsyncLayerPushSender(max_workers=1)

    def wait_for_completion():
        try:
            sender.wait_req("req")
        except BaseException as error:
            errors.append(error)
        finally:
            drained.set()

    waiter = threading.Thread(target=wait_for_completion)
    try:
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                transfer=transfer,
                req_id="req",
                layer_idx=0,
                block_slices=blocks,
                request_generation=generation,
            )
        )
        assert submitted.wait(timeout=3)
        waiter.start()
        # Replacement must not retarget work already admitted with old pointers.
        transfer.close_request("req")
        transfer.open_request("req", replace(handshake, transfer_endpoint="new-peer:1"))
        assert not drained.wait(timeout=0.05)
        completed.set()
        assert drained.wait(timeout=3)
        assert errors == []
        assert [endpoint for endpoint, _, _ in engine.writes] == ["old-peer:1"]
        assert transfer.write_stats("req")["bytes"] == 0
    finally:
        completed.set()
        if waiter.ident is not None:
            waiter.join(timeout=3)
        sender.close()


@pytest.mark.parametrize("changed", ["authorization", "target", "empty_targets"])
def test_active_chunks_cannot_replace_push_authorization(push_state, monkeypatch, changed):
    _, transfer, handshake, _ = push_state
    handshake = replace(handshake, layers=(hnd_remote_layer(block_ids=(0, 1), block_len=4096),))
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        transfer=transfer,
    )
    worker.register_kv_caches(
        {"layer.0": FakeTensor(shape=(2, 8, 16, 4, 32), stride=(16384, 2048, 32, 512, 1))}
    )
    first = PushReqMeta(
        local_block_ids=([1],), target_request_id="remote-old", handshakes=(handshake,)
    )
    try:
        worker.start_load_kv(PdConnectorMetadata(reqs_to_push={"req": first}), None)
        original = worker._prefill._push_authorizations["req"]
        replacement = replace(first, local_block_ids=([2],))
        if changed == "authorization":
            replacement = replace(
                replacement, handshakes=(replace(handshake, transfer_endpoint="new-peer:1"),)
            )
        elif changed == "target":
            monkeypatch.setattr(worker._prefill, "_physical_req_ids", lambda *args: ("req#new",))
        else:
            plan = worker._prefill._push_plans["req"]
            monkeypatch.setattr(
                worker._prefill, "_build_push_layout_plan", lambda *args: replace(plan, targets=())
            )
        with pytest.raises(RuntimeError, match="changed before release"):
            worker.start_load_kv(PdConnectorMetadata(reqs_to_push={"req": replacement}), None)
        assert worker._prefill._push_authorizations["req"] == original
        assert worker._prefill._logical_to_physical["req"] == ("req",)
        assert worker._prefill.push_reqs["req"] == first
        assert "req" not in worker._prefill._completed_pushes
    finally:
        worker.shutdown()


@pytest.mark.parametrize("reason", [RELEASE_CONSUMER_ABORT, RELEASE_PRODUCER_PREEMPTED])
def test_sender_error_cannot_retire_authorization_before_admitted_write_drains(
    push_state, monkeypatch, reason
):
    engine, transfer, handshake, _ = push_state
    handshake = replace(handshake, layers=(hnd_remote_layer(block_ids=(0, 1), block_len=4096),))
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        transfer=transfer,
    )
    tensor = FakeTensor(shape=(2, 8, 16, 4, 32), stride=(16384, 2048, 32, 512, 1))
    worker.register_kv_caches({"layer.0": tensor})
    submitted, completed, draining, retired = (threading.Event() for _ in range(4))
    errors = []
    sender = worker._push_sender
    original_write, original_wait = engine.write, sender.wait_req

    def blocked_write(*args, **kwargs):
        submitted.set()
        assert completed.wait(timeout=3)
        return original_write(*args, **kwargs)

    def observe_drain(req_id):
        draining.set()
        original_wait(req_id)

    def release():
        try:
            worker.start_load_kv(
                PdConnectorMetadata(reqs_to_release={"req"}, release_reasons={"req": reason}),
                None,
            )
        except BaseException as error:
            errors.append(error)
        finally:
            retired.set()

    monkeypatch.setattr(engine, "write", blocked_write)
    monkeypatch.setattr(sender, "wait_req", observe_drain)
    releaser = threading.Thread(target=release)
    try:
        worker.start_load_kv(
            PdConnectorMetadata(
                reqs_to_push={
                    "req": PushReqMeta(
                        local_block_ids=([1],),
                        target_request_id="remote-old",
                        handshakes=(handshake,),
                    )
                }
            ),
            None,
        )
        prepared = worker._prefill._push_layer_plans["req"][0].target_pushes[0]
        worker.save_kv_layer("layer.0", tensor, SimpleNamespace())
        assert submitted.wait(timeout=3)
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                transfer=transfer,
                req_id="req",
                layer_idx=0,
                block_slices=prepared.block_slices,
                request_generation=prepared.request_generation - 1,
            )
        )
        with sender._condition:
            assert sender._condition.wait_for(lambda: sender._error is not None, timeout=3)
        releaser.start()
        assert draining.wait(timeout=3)
        assert not retired.wait(timeout=0.05)
        assert transfer._request_generations["req"] == prepared.request_generation
        assert "req" in worker._prefill._push_authorizations
        assert engine.notifications == []
        completed.set()
        assert retired.wait(timeout=3)
        assert errors == []
        assert "req" not in transfer.peer_handshakes
        assert "req" not in worker._prefill._push_authorizations
        assert sender.is_idle()
        assert [endpoint for endpoint, _, _ in engine.writes] == ["old-peer:1"]
    finally:
        completed.set()
        if releaser.ident is not None:
            releaser.join(timeout=3)
        worker.shutdown()
