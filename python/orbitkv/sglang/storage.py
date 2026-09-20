"""SGLang 0.5.20 HiCache storage backed by the node-local Cache Manager."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

from orbitkv.client.data_plane import LocalDataClient
from orbitkv.sglang.config import OrbitKVSGLangConfig
from orbitkv.sglang.pools import component_for_pool

logger = logging.getLogger(__name__)


def _pool_name(pool: object) -> str:
    return str(getattr(pool, "value", pool))


class OrbitKVHiCacheStorage(HiCacheStorage):
    """Persistent L3 pages for HiCache, with model- and layout-scoped identities.

    SGLang still owns its GPU and L2 host pools. OrbitKV copies a completed L2
    page into its bounded, tiered store before SGLang may reuse that slot.
    """

    def __init__(self, storage_config: HiCacheStorageConfig, _factory_kwargs: dict | None = None):
        self.config = storage_config
        self.options = OrbitKVSGLangConfig.from_extra_config(storage_config.extra_config)
        self._client = LocalDataClient(self.options.bootstrap_socket)
        self._client.health()
        self.registered_pools: dict[str, Any] = {}
        self._namespaces: dict[str, str] = {}
        self._page_bytes: dict[str, int] = {}

    def close(self) -> None:
        self._client.close()

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        super().register_mem_pool_host(mem_pool_host)
        self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name: object) -> None:
        name = _pool_name(host_pool_name)
        self.registered_pools[name] = host_pool
        template = host_pool.get_dummy_flat_data_page()
        if template.device.type != "cpu":
            raise ValueError(f"SGLang host pool {name} must be on CPU")
        page_bytes = template.numel() * template.element_size()
        if page_bytes <= 0:
            raise ValueError(f"SGLang host pool {name} has an empty page")
        self._page_bytes[name] = page_bytes
        identity = {
            "adapter": "sglang-hicache-v2",
            "model": self.options.namespace or self.config.model_name,
            "model_name": self.config.model_name,
            "tp": [self.config.tp_rank, self.config.tp_size],
            "pp": [self.config.pp_rank, self.config.pp_size],
            "cp": [self.config.attn_cp_rank, self.config.attn_cp_size],
            "mla": self.config.is_mla_model,
            "tp_lcm_size": self.config.tp_lcm_size,
            "split_heads": self.config.should_split_heads,
            "pool": name,
            "component": component_for_pool(name),
            "layout": getattr(host_pool, "layout", None),
            "page_size": host_pool.page_size,
            "page_bytes": page_bytes,
            "dtype": str(template.dtype),
        }
        if not identity["model"]:
            raise ValueError("SGLang OrbitKV requires a model name or an explicit namespace")
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._namespaces[name] = f"sglang:{digest}"

    @staticmethod
    def _key(key: str) -> bytes:
        return hashlib.sha256(key.encode()).digest()

    def _put(self, pool: str, key: str, data: bytes) -> bool:
        if len(data) != self._page_bytes[pool]:
            logger.error("HiCache page size mismatch for %s: got %s", pool, len(data))
            return False
        try:
            self._client.put_host_page(self._namespaces[pool], self._key(key), data)
        except Exception:
            logger.exception("HiCache page write failed for pool %s", pool)
            return False
        return True

    def _get(self, pool: str, key: str) -> bytes | None:
        try:
            data = self._client.get_host_page(self._namespaces[pool], self._key(key))
        except Exception:
            logger.exception("HiCache page read failed for pool %s", pool)
            return None
        if data is not None and len(data) != self._page_bytes[pool]:
            logger.error("HiCache cached page has incompatible size for pool %s", pool)
            return None
        return data

    def _exists(self, pool: str, key: str) -> bool:
        if pool not in self._namespaces:
            return False
        try:
            return self._client.has_host_page(self._namespaces[pool], self._key(key))
        except Exception:
            logger.exception("HiCache page lookup failed for pool %s", pool)
            return False

    def exists(self, key: str) -> bool:
        return self._exists(PoolName.KV.value, key)

    def batch_exists(self, keys: list[str], extra_info: Any = None) -> int:
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def batch_exists_v2(
        self,
        keys: list[str],
        pool_transfers: list[PoolTransfer] | None = None,
        extra_info: Any = None,
    ) -> PoolTransferResult:
        kv_hit = self.batch_exists(keys)
        restorable = set(range(1, kv_hit + 1))
        pool_hits = {PoolName.KV.value: kv_hit} if kv_hit else {}
        for transfer in pool_transfers or []:
            if not restorable:
                break
            pool = _pool_name(transfer.name)
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                # Only the first missing page matters for an all-pages pool.
                first_missing = next(
                    (i for i, key in enumerate(keys[:kv_hit]) if not self._exists(pool, key)),
                    kv_hit,
                )
                pool_hits[pool] = first_missing
                restorable.intersection_update(range(1, first_missing + 1))
            else:
                # A trailing window may make 3 restorable while 2 is not.
                # Query each component once; do not turn a long prompt into
                # quadratic round trips to the Cache Manager.
                missing_prefix = [0]
                for key in keys[:kv_hit]:
                    missing_prefix.append(missing_prefix[-1] + (not self._exists(pool, key)))
                tail = max(1, len(transfer.keys or []))
                pool_restorable = {
                    length
                    for length in range(1, kv_hit + 1)
                    if missing_prefix[length] == missing_prefix[max(0, length - tail)]
                }
                pool_hits[pool] = max(pool_restorable, default=0)
                restorable.intersection_update(pool_restorable)
        restorable_pages = sorted(restorable)
        # The controller uses kv_hit_pages to decide how many pages to load.
        # A KV-only hit cannot be returned when a required auxiliary pool is
        # absent, even if the primary KV pages are present.
        max_hit = max(restorable_pages, default=0)
        return PoolTransferResult(max_hit, pool_hits, restorable_prefix_pages=restorable_pages)

    def _read_into_pool(self, pool: str, key: str, index: int) -> bool:
        data = self._get(pool, key)
        if data is None:
            return False
        host_pool = self.registered_pools[pool]
        tensor = torch.frombuffer(bytearray(data), dtype=host_pool.dtype)
        try:
            host_pool.set_from_flat_data_page(index, tensor)
        except Exception:
            logger.exception("HiCache host page restore failed for pool %s", pool)
            return False
        return True

    def _write_from_pool(self, pool: str, key: str, index: int) -> bool:
        host_pool = self.registered_pools[pool]
        try:
            tensor = host_pool.get_data_page(index, flat=True)
            if tensor.device.type != "cpu":
                raise ValueError("HiCache host page must be on CPU")
            data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        except Exception:
            logger.exception("HiCache host page snapshot failed for pool %s", pool)
            return False
        return self._put(pool, key, data)

    def _batch_io_v2(self, transfers: list[PoolTransfer], *, write: bool) -> dict[str, list[bool]]:
        results = {}
        for transfer in transfers:
            pool = _pool_name(transfer.name)
            keys = transfer.keys or []
            host_pool = self.registered_pools.get(pool)
            indices = transfer.host_indices
            if (
                host_pool is None
                or indices is None
                or indices.numel() != len(keys) * host_pool.page_size
            ):
                results[pool] = [False] * len(keys)
                continue
            operation = self._write_from_pool if write else self._read_into_pool
            results[pool] = [
                operation(pool, key, int(indices[i * host_pool.page_size].item()))
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self, transfers: list[PoolTransfer], extra_info: Any = None
    ) -> dict[str, list[bool]]:
        return self._batch_io_v2(transfers, write=False)

    def batch_set_v2(
        self, transfers: list[PoolTransfer], extra_info: Any = None
    ) -> dict[str, list[bool]]:
        return self._batch_io_v2(transfers, write=True)

    def batch_get_v1(
        self, keys: list[str], host_indices: torch.Tensor, extra_info: Any = None
    ) -> list[bool]:
        return self.batch_get_v2(
            [PoolTransfer(name=PoolName.KV, keys=keys, host_indices=host_indices)]
        )[PoolName.KV.value]

    def batch_set_v1(
        self, keys: list[str], host_indices: torch.Tensor, extra_info: Any = None
    ) -> list[bool]:
        return self.batch_set_v2(
            [PoolTransfer(name=PoolName.KV, keys=keys, host_indices=host_indices)]
        )[PoolName.KV.value]

    def get(
        self, key: str, target_location: Any = None, target_sizes: Any = None
    ) -> torch.Tensor | None:
        data = self._get(PoolName.KV.value, key)
        if data is None:
            return None
        if target_location is not None:
            if target_location.numel() * target_location.element_size() != len(data):
                return None
            target_location.copy_(
                torch.frombuffer(bytearray(data), dtype=target_location.dtype).reshape(
                    target_location.shape
                )
            )
            return target_location
        return torch.frombuffer(
            bytearray(data), dtype=self.registered_pools[PoolName.KV.value].dtype
        )

    def batch_get(
        self, keys: list[str], target_locations: Any = None, target_sizes: Any = None
    ) -> list[torch.Tensor | None]:
        if target_locations is None:
            return [self.get(key) for key in keys]
        return [self.get(key, target) for key, target in zip(keys, target_locations, strict=True)]

    def set(
        self, key: str, value: Any = None, target_location: Any = None, target_sizes: Any = None
    ) -> bool:
        tensor = value if value is not None else target_location
        if tensor is None:
            return False
        try:
            data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        except Exception:
            logger.exception("HiCache page snapshot failed")
            return False
        return self._put(PoolName.KV.value, key, data)

    def batch_set(
        self,
        keys: list[str],
        values: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        pages = values if values is not None else target_locations
        if pages is None or len(pages) != len(keys):
            return False
        return all(self.set(key, page) for key, page in zip(keys, pages, strict=True))
