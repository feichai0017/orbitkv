"""Prove the SGLang-style GPU buffer layout survives a real D2H/H2D round trip."""

from __future__ import annotations

import hashlib
import time
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_direct_page_transfer_overwrites_poisoned_gpu_slots(channel_server):
    torch = pytest.importorskip("torch")
    from orbitkv import QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
    from orbitkv.client.manager import CacheManagerClient

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    page_size = 64
    shape = (page_size * 8, 2, 128)
    tensors = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    expected = []
    for layer, tensor in enumerate(tensors):
        values = torch.arange(page_size * 2 * 2 * 128, device="cuda").reshape(page_size * 2, 2, 128)
        tensor[page_size : page_size * 3] = (values + layer * 13).to(torch.bfloat16)
        expected.append(tensor[page_size : page_size * 3].clone())
    torch.cuda.synchronize()

    names = ["kv:0", "kv:1"]
    hashes = [hashlib.sha256(f"page-{i}".encode()).digest() for i in range(2)]
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
            [8, 8],
            [block_bytes, block_bytes],
            [0, 0],
            [1, 1],
            "direct",
            False,
        )
        assert ok, message

        success, message = client.save(
            instance, 0, 0, resolve_device_id(), [(name, [1, 2], hashes) for name in names]
        )
        assert success, message
        for tensor in tensors:
            tensor.zero_()
        torch.cuda.synchronize()

        lookup = client.query_prefetch(instance, hashes, "sglang-poison-test")
        assert isinstance(lookup, QueryReady)
        assert lookup.num_hit_blocks == 2
        restore = client.start_restore(
            instance, 0, resolve_device_id(), [names], [(lookup.lease, [[3, 4]])]
        )
        deadline = time.monotonic() + 30
        while True:
            status = client.poll_restore(restore)
            if status.done:
                assert status.success, status.message
                break
            assert time.monotonic() < deadline, "GPU restore timed out"
            time.sleep(0.01)
        torch.cuda.synchronize()
        for tensor, original in zip(tensors, expected, strict=True):
            assert torch.equal(tensor[page_size * 3 : page_size * 5], original)
    finally:
        client.unregister_context(instance)
        client.close()
