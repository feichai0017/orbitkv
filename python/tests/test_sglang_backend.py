"""Real Cache Manager round trip through SGLang's dynamic HiCache contract."""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_host_page_protocol_round_trip(local_control_server):
    from orbitkv import LocalQueryClient

    client = LocalQueryClient(local_control_server.local_bootstrap_socket)
    namespace = f"sglang-test:{uuid.uuid4().hex}"
    key = b"page-key"
    assert not client.has_host_page(namespace, key)
    assert client.get_host_page(namespace, key) is None
    data = b"\x00\x01\xff" * 701
    client.put_host_page(namespace, key, data)
    assert client.has_host_page(namespace, key)
    assert client.get_host_page(namespace, key) == data
    assert client.get_host_page(namespace + ":other", key) is None
    client.close()


def test_sglang_hicache_kv_and_trailing_pool_round_trip(local_control_server):
    torch = pytest.importorskip("torch")
    storage_module = pytest.importorskip("orbitkv.sglang.storage")
    hicache = pytest.importorskip("sglang.srt.mem_cache.hicache_storage")

    class HostPool:
        def __init__(self, page_size: int, dtype: torch.dtype):
            self.page_size = page_size
            self.dtype = dtype
            self.layout = "page_first"
            self.buffer = torch.zeros(16 * page_size, dtype=dtype)

        def get_dummy_flat_data_page(self):
            return torch.zeros(self.page_size, dtype=self.dtype)

        def get_data_page(self, index: int, flat: bool = True):
            return self.buffer[index : index + self.page_size]

        def set_from_flat_data_page(self, index: int, data_page: torch.Tensor):
            self.buffer[index : index + self.page_size] = data_page

    def make_config(model_name: str):
        return hicache.HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name=model_name,
            extra_config={
                "endpoint": f"unix://{local_control_server.local_bootstrap_socket}",
                "allocator": "shm",
                "namespace": "integration-test",
            },
        )

    kv_pool = HostPool(4, torch.bfloat16)
    mamba_pool = HostPool(1, torch.float32)
    backend = storage_module.OrbitKVHiCacheStorage(make_config("model-a"))
    backend.register_mem_pool_host(kv_pool)
    backend.register_mem_host_pool_v2(mamba_pool, hicache.PoolName.MAMBA)
    keys = ["a", "b", "c"]
    kv_indices = torch.arange(12)
    kv_pool.buffer[:12] = torch.arange(12, dtype=torch.float32).to(torch.bfloat16)
    mamba_pool.buffer[2] = 42

    assert backend.batch_set_v1(keys, kv_indices) == [True] * 3
    assert backend.batch_set_v2(
        [
            hicache.PoolTransfer(
                name=hicache.PoolName.MAMBA, keys=["c"], host_indices=torch.tensor([2])
            )
        ]
    ) == {"mamba": [True]}
    result = backend.batch_exists_v2(
        keys,
        [
            hicache.PoolTransfer(
                name=hicache.PoolName.MAMBA,
                keys=["c"],
                hit_policy=hicache.PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )
    assert result.kv_hit_pages == 3
    assert result.restorable_prefix_pages == [3]
    missing_tail = backend.batch_exists_v2(
        keys,
        [
            hicache.PoolTransfer(
                name=hicache.PoolName.MAMBA,
                keys=["a", "b", "c"],
                hit_policy=hicache.PoolHitPolicy.ALL_PAGES,
            )
        ],
    )
    assert missing_tail.kv_hit_pages == 0
    assert missing_tail.restorable_prefix_pages == []
    assert missing_tail.extra_pool_hit_pages == {"kv": 3, "mamba": 0}

    kv_pool.buffer.zero_()
    mamba_pool.buffer.zero_()
    assert backend.batch_get_v1(keys, kv_indices) == [True] * 3
    assert backend.batch_get_v2(
        [
            hicache.PoolTransfer(
                name=hicache.PoolName.MAMBA, keys=["c"], host_indices=torch.tensor([2])
            )
        ]
    ) == {"mamba": [True]}
    assert torch.equal(
        kv_pool.buffer[:12], torch.arange(12, dtype=torch.float32).to(torch.bfloat16)
    )
    assert mamba_pool.buffer[2] == 42

    other_model = storage_module.OrbitKVHiCacheStorage(make_config("model-b"))
    other_model.register_mem_pool_host(HostPool(4, torch.bfloat16))
    assert other_model.batch_exists(keys) == 0
    other_model.close()
    backend.close()
