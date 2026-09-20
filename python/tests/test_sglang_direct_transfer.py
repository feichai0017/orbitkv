"""Prove the SGLang-style GPU buffer layout survives a real D2H/H2D round trip."""

from __future__ import annotations

import hashlib
import queue
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("failure", ["submit", "poll", "timeout"])
def test_linker_failure_never_acknowledges_gpu_destinations(failure):
    pytest.importorskip("sglang")
    from orbitkv.sglang.linker import OrbitKVLinker, _LayerDoneCounter, _Load

    linker = object.__new__(OrbitKVLinker)
    linker.instance_id = "failed-transfer"
    linker.device_id = 0
    linker._layer_names = ["kv:0"]
    linker._load_error = None
    linker._load_queue = queue.Queue()
    linker._completed_loads = queue.Queue()
    linker.layer_done_counter = _LayerDoneCounter(1)
    linker.client = MagicMock()
    if failure == "submit":
        linker.client.start_restore.side_effect = ConnectionError("lost acknowledgement")
    elif failure == "poll":
        linker.client.wait_restore.side_effect = ConnectionError("lost completion")
    else:
        linker.client.wait_restore.side_effect = TimeoutError("restore deadline")

    index = linker.layer_done_counter.update_producer()
    linker._load_queue.put(
        (
            index,
            [_Load("first", b"attempted", (1,)), _Load("second", b"unsubmitted", (2,))],
            SimpleNamespace(synchronize=lambda: None),
        )
    )
    linker._load_queue.put(None)
    thread = threading.Thread(target=linker._load_worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert linker._completed_loads.empty()
    linker.client.release.assert_called_once_with(b"unsubmitted")
    for observe in (linker.num_completed_loads, linker.pop_completed_load):
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            observe()
    linker.layer_done_counter.set_consumer(index)
    with pytest.raises((ConnectionError, TimeoutError)):
        linker.layer_done_counter.wait_until(0)


@pytest.mark.parametrize(
    ("layer_count", "page_count", "page_first"),
    [(2, 2, False), (36, 64, False), (36, 64, True)],
)
def test_direct_page_transfer_overwrites_poisoned_gpu_slots(
    channel_server, layer_count, page_count, page_first
):
    torch = pytest.importorskip("torch")
    from orbitkv import QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
    from orbitkv.client.manager import CacheManagerClient

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    page_size = 64
    num_blocks = page_count + 4
    shape = (page_size * num_blocks, 2, 128)
    tensors = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for _ in range(layer_count)]
    expected = []
    for layer, tensor in enumerate(tensors):
        values = torch.arange(page_size * page_count * 2 * 128, device="cuda").reshape(
            page_size * page_count, 2, 128
        )
        tensor[page_size : page_size * (page_count + 1)] = (values + layer * 13).to(torch.bfloat16)
        expected.append(tensor[page_size : page_size * (page_count + 1)].clone())
    torch.cuda.synchronize()

    names = [f"kv:{layer}" for layer in range(layer_count)]
    hashes = [hashlib.sha256(f"page-{i}".encode()).digest() for i in range(page_count)]
    instance = f"sglang-layout-{uuid.uuid4().hex}"
    namespace = f"sglang-layout:{instance}"
    client = CacheManagerClient(channel_server.bootstrap_socket)
    try:
        client.start_session_watcher(instance, namespace, 1, 1)
        wrappers = [serialize_gpu_buffer(tensor) for tensor in tensors]
        block_bytes = page_size * tensors[0].stride(0) * tensors[0].element_size()
        ok, message = client.register_context_batch(
            instance,
            namespace,
            0,
            0,
            1,
            1,
            resolve_device_id(),
            names,
            wrappers,
            [num_blocks] * layer_count,
            [block_bytes] * layer_count,
            [0] * layer_count,
            [1] * layer_count,
            "direct",
            page_first,
        )
        assert ok, message

        success, message = client.save(
            instance,
            0,
            0,
            resolve_device_id(),
            [(name, list(range(1, page_count + 1)), hashes) for name in names],
        )
        assert success, message
        for tensor in tensors:
            tensor.zero_()
        torch.cuda.synchronize()

        lookup = client.query_prefetch(instance, hashes, "sglang-poison-test")
        assert isinstance(lookup, QueryReady)
        assert lookup.num_hit_blocks == page_count
        restore = client.start_restore(
            instance,
            0,
            resolve_device_id(),
            [names],
            [(lookup.lease, [list(range(3, page_count + 3))])],
        )
        status = client.wait_restore(restore, timeout=30)
        assert status.success, status.message
        torch.cuda.synchronize()
        for tensor, original in zip(tensors, expected, strict=True):
            assert torch.equal(tensor[page_size * 3 : page_size * (page_count + 3)], original)
    finally:
        client.unregister_context(instance)
        client.close()
