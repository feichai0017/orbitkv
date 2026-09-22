"""Bridge SGLang tree checkpoints and OrbitKV's compiled recovery contract."""

from __future__ import annotations

import torch
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinkerWrapper


class RecurrentComponent(MambaComponent):
    """Persist sealed tree checkpoints, then copy restored state into request slots."""

    def build_external_linker_transfer(self, phase, node, keys):
        if phase == LinkerTransferPhase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            value = node.component_data[self.component_type].value
            if value is None:
                return None
            return PoolTransfer(
                name=PoolName.MAMBA,
                keys=[node.hash_value[-1]],
                device_indices=value.to(torch.int64),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        if not keys:
            return None
        transfer = PoolTransfer(
            name=PoolName.MAMBA, keys=list(keys), hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        if phase == LinkerTransferPhase.LOAD:
            allocator = self.cache.req_to_token_pool.mamba_allocator
            slots = allocator.alloc(1)
            if slots is None:
                self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
                slots = allocator.alloc(1)
            if slots is None:
                return None
            transfer.keys = [keys[-1]]
            transfer.device_indices = slots.to(torch.int64)
        return transfer

    def update_external_linker_load(
        self,
        phase,
        req,
        full_transfer,
        transfer,
        prefix_len,
        *,
        insert_result=None,
        canonical_full=None,
    ):
        pool = self.cache.req_to_token_pool
        if phase == ExternalLinkerLoadPhase.ABORT:
            pool.mamba_allocator.free(transfer.device_indices)
            return None
        if phase == ExternalLinkerLoadPhase.PREPARE:
            if not req.kv.holds_mamba:
                slots = pool.mamba_allocator.alloc(1)
                if slots is None:
                    self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
                    slots = pool.mamba_allocator.alloc(1)
                if slots is None:
                    raise RuntimeError("No request slot for restored recurrent state")
                req.kv.mamba_pool_idx = slots[0]
            return transfer
        if phase != ExternalLinkerLoadPhase.COMMIT or insert_result is None:
            raise ValueError("Invalid recurrent checkpoint handoff")
        source = self.tree_core.get_component_device_value(
            insert_result.last_device_node, self.component_type
        )
        if source is None:
            raise RuntimeError("Restored tree has no recurrent checkpoint slot")
        req.kv.mamba_cow_src_index = source
        req.kv.mamba_needs_clear = False
        # SGLang freed the redundant destination after insert. Its canonical
        # checkpoint is already present and remains protected by the tree lock.
        if insert_result.mamba_exist:
            return None
        transfer.device_indices = source.to(torch.int64)
        return transfer


class RecoveryLinkerWrapper(UnifiedCacheLinkerWrapper):
    """Carry absolute query boundaries and support checkpoint-sized state slots."""

    def __init__(self, cache, cache_linker):
        supported = {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}
        if not set(cache.tree_components) <= supported:
            raise ValueError("OrbitKV has no recovery contract for these tree components")
        if is_swa_req_ring(cache.token_to_kv_pool_allocator):
            raise ValueError("SWA request rings require a separate recovery contract")
        if ComponentType.MAMBA in cache.components:
            component = cache.components[ComponentType.MAMBA]
            if (
                not isinstance(component, RecurrentComponent)
                or component.int8_ckpt_pool is not None
            ):
                raise ValueError("Unsupported recurrent checkpoint representation")
        self.cache = cache
        self.cache_linker = cache_linker
        self._skip_swa = False
        self._components = cache._components_tuple
        self.hit_markers = {}
        self.pending_loads = {}
        self.pending_offloads = []
        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    def match(self, key, req, result):
        self.cache_linker._origins[req.rid] = int(result.device_indices.numel())
        try:
            return super().match(key, req, result)
        finally:
            self.cache_linker._origins.pop(req.rid, None)

    def load_back(self, req):
        hit = self.hit_markers.get(req.rid)
        if hit is not None:
            self.cache_linker._load_boundaries[req.rid] = (
                hit.device_hit_len + len(hit.tail_hashes) * self.cache.page_size,
                {},
            )
        try:
            return super().load_back(req)
        finally:
            # Allocation may fail, or another request may have populated every
            # destination already. Neither path consumes the lookup in load().
            self.cache_linker._load_boundaries.pop(req.rid, None)
            self.cache_linker.cancel_query(req.rid)

    def release_request(self, rid):
        self.hit_markers.pop(rid, None)
        linker = self.cache_linker
        if rid in linker._queued_loads:
            # load_back already published these slots into the tree. Finish
            # their DMA before cancellation can make them reusable by a match.
            index = linker.start_layer_wise_loading()
            linker._load_queue.join()
            linker._check_load_failure()
            self.drain_loads(self.num_completed_loads())
            linker.layer_done_counter._futures.pop(index, None)
            linker.layer_done_counter.request_ids.pop(index, None)
        linker.cancel_queued_load(rid)

    def _update_load(
        self,
        phase,
        req,
        component_transfers,
        prefix_len,
        *,
        insert_result=None,
        canonical_full=None,
    ):
        requested = {transfer.name: set(transfer.keys or ()) for _, transfer in component_transfers}
        ordinary = [(c, t) for c, t in component_transfers if t.name != PoolName.MAMBA]
        result = super()._update_load(
            phase,
            req,
            ordinary,
            prefix_len,
            insert_result=insert_result,
            canonical_full=canonical_full,
        )
        for component, transfer in component_transfers:
            if transfer.name != PoolName.MAMBA:
                continue
            updated = component.update_external_linker_load(
                phase,
                req,
                component_transfers[0][1],
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=canonical_full,
            )
            if updated is not None:
                result.append(updated)
        if phase == ExternalLinkerLoadPhase.COMMIT:
            copied = {transfer.name: set(transfer.keys or ()) for transfer in result}
            self.cache_linker._load_boundaries[req.rid] = (
                prefix_len,
                {name: keys - copied.get(name, set()) for name, keys in requested.items()},
            )
        return result
