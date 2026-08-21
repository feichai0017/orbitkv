from __future__ import annotations

import hashlib
from typing import Any


def validate_prefix_cache(cache: Any) -> None:
    """Recompute the complete non-owning radix and census invariants."""

    root = cache.root_node
    if (
        root.boundary != 0
        or root.edge
        or root.digest
        or root.prefix is not None
        or root.resident_count != 0
        or root.swa_ref_count != 0
        or root.parent is not None
        or root.evicted
        or root.lock_ref != 1
    ):
        raise RuntimeError("OrbitKV Prefix root invariant failed")

    full_total = full_evictable = full_protected = 0
    swa_total = swa_evictable = swa_protected = 0
    seen_nodes: set[int] = set()
    seen_prefixes: set[Any] = set()
    published = []
    stack = [(root, hashlib.sha256())]
    while stack:
        parent, parent_hash = stack.pop()
        for child_key, node in parent.children.items():
            identity = id(node)
            if identity in seen_nodes:
                raise RuntimeError("OrbitKV Prefix graph is not a tree")
            seen_nodes.add(identity)
            if (
                node.parent is not parent
                or child_key != node.digest
                or len(node.edge) != cache.page_size
                or node.boundary != parent.boundary + cache.page_size
                or cache._nodes.get((node.boundary, node.digest)) is not node
                or node.evicted
                or node.lock_ref < 0
                or node.swa_ref_count < 0
                or node.resident_count < 0
            ):
                raise RuntimeError("OrbitKV Prefix topology invariant failed")
            node_hash = parent_hash.copy()
            for token in node.edge:
                node_hash.update(token.to_bytes(8, "little", signed=False))
            if node.digest != node_hash.digest():
                raise RuntimeError("OrbitKV Prefix digest invariant failed")
            if node.prefix is None:
                if node.resident_count != 0:
                    raise RuntimeError("structural Prefix node retained residency")
            else:
                if node.prefix in seen_prefixes:
                    raise RuntimeError("OrbitKV Prefix lease is duplicated")
                seen_prefixes.add(node.prefix)
                full_pages = node.boundary // cache.page_size
                swa_pages = node.resident_count - full_pages
                if (
                    node.resident_count <= 0
                    or swa_pages < 0
                    or swa_pages > full_pages
                    or (cache.sliding_window_size is None and swa_pages != 0)
                ):
                    raise RuntimeError("OrbitKV Prefix residency invariant failed")
                published.append(node)
            full_total += cache.page_size
            if node.lock_ref == 0:
                full_evictable += cache.page_size
            else:
                full_protected += cache.page_size
            stack.append((node, node_hash))

    if len(seen_nodes) != len(cache._nodes):
        raise RuntimeError("OrbitKV Prefix index contains unreachable nodes")

    expected_swa_refs = {node: 0 for node in cache._nodes.values()}
    for node in published:
        current = node
        full_pages = node.boundary // cache.page_size
        for _ in range(node.resident_count - full_pages):
            expected_swa_refs[current] += 1
            if current.parent is None:
                raise RuntimeError("OrbitKV Prefix SWA path is too short")
            current = current.parent
    for node, expected_refs in expected_swa_refs.items():
        if node.swa_ref_count != expected_refs:
            raise RuntimeError("OrbitKV Prefix SWA reference invariant failed")
        if expected_refs:
            swa_total += cache.page_size
            if node.lock_ref == 0:
                swa_evictable += cache.page_size
            else:
                swa_protected += cache.page_size

    actual = (
        cache._full_total_tokens,
        cache._full_evictable_tokens,
        cache._full_protected_tokens,
        cache._swa_total_tokens,
        cache._swa_evictable_tokens,
        cache._swa_protected_tokens,
    )
    expected = (
        full_total,
        full_evictable,
        full_protected,
        swa_total,
        swa_evictable,
        swa_protected,
    )
    if actual != expected:
        raise RuntimeError("OrbitKV Prefix incremental census invariant failed")


__all__ = ["validate_prefix_cache"]
