"""Nonblocking request admission through SGLang's general plugin hooks."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from orbitkv.logging_utils import get_connector_logger, trace_transfer


def enqueue_request(original: Callable, scheduler: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
    result = original(scheduler, req, *args, **kwargs)
    # This hook only prepares requests accepted by the ordinary serving queue.
    if not scheduler.waiting_queue or scheduler.waiting_queue[-1] is not req:
        return result
    from .linker import OrbitKVLinker

    wrapper = getattr(scheduler.tree_cache, "linker", None)
    linker = getattr(wrapper, "cache_linker", None)
    if not isinstance(linker, OrbitKVLinker) or req.positional_embed_overrides is not None:
        return result
    trace_transfer("queued", req.rid, engine="sglang")
    if os.environ.get("ORBITKV_QUEUE_WARMUP") != "1":
        return result
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    tokens = req.origin_input_ids + req.output_ids
    key = RadixKey(
        token_ids=tokens,
        extra_key=req.extra_key,
        cache_salt=req.cache_salt,
        limit=req._compute_max_prefix_len(len(tokens)),
    )
    # req=None keeps this HBM-only: no external lookup, allocation or load marker.
    resident = scheduler.tree_cache.match_prefix(MatchPrefixParams(key=key, cow_mamba=False))
    keys = wrapper._tail_hashes(key, resident, int(resident.device_indices.numel()))
    try:
        linker.client.warm_prefix(linker.instance_id, linker._hashes(keys), req.rid)
    except (RuntimeError, OSError):
        get_connector_logger().warning(
            "Queued warmup failed for request %s", req.rid, exc_info=True
        )
    return result


def abort_request(original: Callable, scheduler: Any, req: Any) -> Any:
    from .linker import OrbitKVLinker

    wrapper = getattr(scheduler.tree_cache, "linker", None)
    linker = getattr(wrapper, "cache_linker", None)
    if isinstance(linker, OrbitKVLinker):
        linker.cancel_query(req.rid)
    return original(scheduler, req)


def admit_request(original: Callable, adder: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
    from .linker import OrbitKVLinker

    wrapper = getattr(adder.tree_cache, "linker", None)
    linker = getattr(wrapper, "cache_linker", None)
    if isinstance(linker, OrbitKVLinker):
        import torch
        from sglang.srt.managers.schedule_policy import AddReqResult

        match_limit = req._compute_max_prefix_len(len(req.full_untruncated_fill_ids))
        restorable_tokens = match_limit // linker.page_size * linker.page_size
        if len(req.prefix_indices) >= restorable_tokens:
            linker.cancel_query(req.rid)
        # Every attention rank takes the same admission decision even when its
        # SSD read finishes in a different scheduler iteration.
        state = torch.tensor([linker.query_state(req.rid)], dtype=torch.int)
        adder.tree_cache._all_reduce_attn_groups(state, torch.distributed.ReduceOp.MAX)
        if state.item() == 1:
            return AddReqResult.CONTINUE
        if state.item() == 2:
            linker.expire_query(req.rid)
    return original(adder, req, *args, **kwargs)
