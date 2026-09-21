"""Nonblocking request admission through SGLang's general plugin hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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
