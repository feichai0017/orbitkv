from __future__ import annotations

from typing import Any

from orbitkv_sglang.runtime import (
    ManagerError,
    PageLease,
    PrefixSemanticKey,
    SnapshotPage,
)

from . import layouts as L


def uint(name: str, value: int, bits: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << bits
    ):
        raise ManagerError(f"{name} is outside uint{bits}_t")
    return value


def page_to_c(value: PageLease) -> L.PageLeaseLayout:
    return L.PageLeaseLayout(
        uint("page engine epoch", value.engine_epoch, 64),
        uint("page pool epoch", value.pool_epoch, 64),
        uint("page generation", value.generation, 64),
        uint("page id", value.page_id, 32),
        uint("page pool id", value.pool_id, 32),
    )


def page(value: Any) -> PageLease:
    return PageLease(
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.page_id),
        int(value.pool_id),
    )


def key_to_c(value: PrefixSemanticKey) -> L.PrefixKeyLayout:
    if not isinstance(value.namespace, bytes) or len(value.namespace) != 32:
        raise ManagerError("prefix namespace must contain exactly 32 bytes")
    if not isinstance(value.digest, bytes) or len(value.digest) != 32:
        raise ManagerError("prefix digest must contain exactly 32 bytes")
    result = L.PrefixKeyLayout()
    result.namespace_bytes[:] = value.namespace
    result.digest[:] = value.digest
    result.boundary = uint("prefix boundary", value.boundary, 64)
    return result


def key(value: L.PrefixKeyLayout) -> PrefixSemanticKey:
    return PrefixSemanticKey(
        bytes(value.namespace_bytes), bytes(value.digest), int(value.boundary)
    )


def snapshot_page(value: L.SnapshotPageLayout) -> SnapshotPage:
    if int(value.reserved) != 0:
        raise ManagerError("snapshot page reserved field is nonzero")
    return SnapshotPage(
        page(value.page),
        int(value.logical_ordinal),
        int(value.temporal_cell_index),
        int(value.temporal_cycle),
        int(value.backend_index),
        int(value.class_id),
        int(value.backend_domain),
        int(value.valid_token_count),
        int(value.visible_token_offset),
        int(value.visible_token_count),
    )


__all__ = ["key", "key_to_c", "page", "page_to_c", "snapshot_page", "uint"]
