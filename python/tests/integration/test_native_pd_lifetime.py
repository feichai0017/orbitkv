"""Real GPU payloads and native P/D owners under partial-write/shutdown faults.

These small native-owner gates complement model serving tests. They use real
TENT and ZMQ TCP; the model/layout fixture is one two-page dense allocation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def gpu_pages(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU and the built TENT artifact")
    from orbitkv.vllm.transport import TentTransferEngine

    monkeypatch.setenv("MC_FORCE_TCP", "1")
    engines = [
        TentTransferEngine(hostname="127.0.0.1", protocol="tcp", device_name="") for _ in range(2)
    ]
    source = torch.arange(8192, device="cuda", dtype=torch.int32).remainder(251).to(torch.uint8)
    destination = torch.zeros_like(source)
    torch.cuda.synchronize()
    for engine, tensor in zip(engines, (source, destination), strict=True):
        engine.batch_register_memory([tensor.data_ptr()], [tensor.numel()])
    try:
        yield torch, source, destination, engines
    finally:
        for engine, tensor in zip(engines, (source, destination), strict=True):
            engine._engine.unregister_memory([tensor.data_ptr()])


class _WriteGate:
    def __init__(self, engine, fail):
        self.engine = engine
        self.fail = fail
        self.started = threading.Event()
        self.resume = threading.Event()

    def batch_transfer_sync_write(self, endpoint, sources, destinations, lengths):
        assert len(lengths) == 1 and lengths[0] == 8192
        assert self.engine.batch_transfer_sync_write(endpoint, sources, destinations, [4096]) == 0
        self.started.set()
        assert self.resume.wait(30), "test did not release the physical-write gate"
        if self.fail:
            return -1
        return self.engine.batch_transfer_sync_write(
            endpoint, [sources[0] + 4096], [destinations[0] + 4096], [4096]
        )


def _native_vllm_worker(tensor, engine, producer):
    import msgspec
    import zmq.asyncio
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnectorWorker,
        MooncakeKVConnectorStats,
        MooncakeXferMetadata,
        MooncakeXferResponse,
    )

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._closing = worker._shutdown_complete = False
    worker._receive_tasks = set()
    worker._pending_receives = {}
    worker._sender_listener_task = None
    worker.is_kv_producer, worker.is_kv_consumer = producer, not producer
    worker.hostname, worker.rpc_port = "127.0.0.1", 0
    worker.transfer_engine_name = "orbitkv.vllm.transport.TentTransferEngine"
    worker.transfer_endpoint = engine.endpoint
    worker.engine = engine
    worker.device_id = 0
    worker.tp_rank = worker.pp_rank = 0
    worker.tp_size = worker.pp_size = 1
    worker.use_mla = worker._is_hma_required = False
    worker._physical_blocks_per_logical_kv_block = 1
    worker.opaque_packed_storages = set()
    worker._prepared_transfer_regions = {}
    worker.kv_caches_base_addr = [tensor.data_ptr()]
    worker.block_len_per_layer = worker.kv_block_len_per_layer = [4096]
    worker.registered_layer_names = ["layer"]
    worker.registered_layer_indices = worker.registered_group_indices = [0]
    worker.region_shared_groups = [()]
    worker.region_row_offsets = [0]
    worker.kv_cache_config = SimpleNamespace(
        has_mamba_layers=False,
        transfer_group_index_by_layer={"layer": 0},
        transfer_groups=[SimpleNamespace(kv_cache_spec=object())],
    )
    worker.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda size: [0],
        local_replicates_kv_cache=False,
        total_num_kv_heads=1,
    )
    worker._tp_size = {"producer": 1}
    worker._remote_agents = {}
    worker._pending_bootstrap_queries = {}
    worker.reqs_need_send = {}
    worker.finished_sending_reqs = set()
    worker.finished_recving_reqs = set()
    worker._invalid_block_ids = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker.xfer_stats = MooncakeKVConnectorStats()
    worker.async_zmq_ctx = zmq.asyncio.Context()
    worker._encoder = msgspec.msgpack.Encoder()
    worker._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    worker._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    if producer:
        worker.sender_loop, worker._sender_listener_t = loop, thread
        worker.num_sender_tasks = 2
        worker.sender_worker_queue = asyncio.Queue()
        worker._sender_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            initializer=worker._bind_sender_thread_device,
        )

        async def register():
            pass

        worker.register_worker_with_bootstrap = register
        ready = threading.Event()
        asyncio.run_coroutine_threadsafe(worker._mooncake_sender_listener(ready), loop)
        assert ready.wait(10)
    else:
        worker.receiver_loop, worker._mooncake_receiver_t = loop, thread
    return worker


@pytest.mark.parametrize("partial_failure", [False, True], ids=["delayed-ack", "partial-write"])
def test_vllm_native_pages_wait_for_write_and_shutdown_drain(gpu_pages, tmp_path, partial_failure):
    pytest.importorskip("vllm")
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnectorMetadata,
    )

    torch, source, destination, engines = gpu_pages
    sender = _native_vllm_worker(source, engines[0], True)
    receiver = _native_vllm_worker(destination, engines[1], False)
    gate = _WriteGate(engines[0], partial_failure)
    sender.engine = gate
    receiver._remote_agents = {"producer": {0: {0: f"tcp://127.0.0.1:{sender.side_channel_port}"}}}
    params = {
        "transfer_id": "attempt-1",
        "remote_engine_id": "producer",
        "remote_bootstrap_addr": "unused",
    }
    initial = MooncakeConnectorMetadata()
    initial.reqs_to_send["p"] = ("attempt-1", [])
    ready = MooncakeConnectorMetadata()
    ready.reqs_to_send["p"] = ("attempt-1", [[0, 1]])
    incoming = MooncakeConnectorMetadata()
    incoming.add_new_req("d", [[0, 1]], params)
    shutdown_done = threading.Event()
    shutdown_errors = []
    shutdown_thread = None
    try:
        asyncio.run_coroutine_threadsafe(
            sender.record_send_reqs(initial), sender.sender_loop
        ).result(5)
        asyncio.run_coroutine_threadsafe(sender.record_send_reqs(ready), sender.sender_loop).result(
            5
        )
        receiver.start_load_kv(incoming)
        assert gate.started.wait(10)
        torch.cuda.synchronize()
        assert torch.equal(source[:4096], destination[:4096])
        assert torch.count_nonzero(destination[4096:]).item() == 0

        # A cancellation/timeout cannot finish the receive or invalidate pages yet.
        async def cancel_receive():
            receiver._handle_failed_recv(
                incoming.reqs_to_recv["producer"], {"d"}, "test cancellation"
            )

        asyncio.run_coroutine_threadsafe(cancel_receive(), receiver.receiver_loop).result(5)
        assert not receiver.get_transfer_results().finished_recving
        assert not receiver.get_block_ids_with_load_errors()
        sender.reqs_need_send["attempt-1"].expire_time = float("-inf")
        assert not sender.get_transfer_results().finished_sending

        def shutdown():
            try:
                sender.shutdown()
            except BaseException as error:
                shutdown_errors.append(error)
            finally:
                shutdown_done.set()

        shutdown_thread = threading.Thread(target=shutdown, daemon=True)
        shutdown_thread.start()
        assert not shutdown_done.wait(0.1), (
            "shutdown returned while a WRITE could still target the pages"
        )
        gate.resume.set()
        assert shutdown_done.wait(10)
        assert not shutdown_errors
        deadline = time.monotonic() + 10
        while True:
            result = receiver.get_transfer_results()
            if result.finished_recving:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result.finished_recving == {"d"}
        assert receiver.get_block_ids_with_load_errors() == {0, 1}
        assert not receiver.get_transfer_results().finished_recving
        torch.cuda.synchronize()
        if not partial_failure:
            assert torch.equal(source, destination)
        else:
            assert torch.count_nonzero(destination[4096:]).item() == 0
        receiver.shutdown()
        destination.fill_(173)
        torch.cuda.synchronize()
        assert torch.all(destination == 173).item()
        (tmp_path / "native-lifetime.json").write_text(
            json.dumps(
                {
                    "partial_failure": partial_failure,
                    "first_write_bytes": 4096,
                    "held_blocks": [0, 1],
                    "shutdown_waited": True,
                    "terminal_once": True,
                    "reuse_value": 173,
                },
                indent=2,
            )
        )
    finally:
        gate.resume.set()
        if shutdown_thread is not None:
            shutdown_thread.join(10)
        sender.shutdown()
        receiver.shutdown()
