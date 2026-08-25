from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True, slots=True, eq=False)
class PrivatePrefixProvenance:
    """Proof that ``prefix_indices`` is a request-private row snapshot."""

    tensor: Any
    request_key: tuple[str, str | bytes | int]
    request_lease: Any
    boundary: int


PRIVATE_PREFIX_MARKER = "_orbitkv_private_prefix"
SHARED_PREFIX_MARKERS = (
    "_orbitkv_prefix_node",
    "_orbitkv_prefix_semantic",
    "_orbitkv_provisional_prefix_lock",
    "_orbitkv_prefix_lock_held",
)


def validate_private_prefix(
    req: Any,
    row: Any,
    key: tuple[str, str | bytes | int],
    lease: Any,
    boundary: int,
) -> PrivatePrefixProvenance | None:
    """Return exact request-private provenance or reject ambiguous Prefix state."""

    import torch

    if any(hasattr(req, name) for name in SHARED_PREFIX_MARKERS):
        raise RuntimeError("current token-reclamation profile forbids Prefix mirrors")
    prefix = getattr(req, "prefix_indices", None)
    if (
        type(prefix) is not torch.Tensor
        or prefix.ndim != 1
        or prefix.dtype is not torch.int64
        or prefix.device != row.device
    ):
        raise RuntimeError("request-private prefix mirror is not a device int64 vector")
    marker = getattr(req, PRIVATE_PREFIX_MARKER, None)
    if int(prefix.numel()) == 0:
        if marker is not None:
            raise RuntimeError("empty request retained private-prefix provenance")
        return None
    if type(marker) is not PrivatePrefixProvenance:
        raise RuntimeError("current token-reclamation profile forbids Prefix mirrors")
    if (
        marker.tensor is not prefix
        or marker.request_key != key
        or marker.request_lease != lease
        or getattr(req, "_orbitkv_request_key", None) != key
        or getattr(req, "_orbitkv_request_lease", None) != lease
        or isinstance(marker.boundary, bool)
        or not isinstance(marker.boundary, Integral)
        or not 0 < marker.boundary <= boundary
        or int(prefix.numel()) > marker.boundary
        or type(getattr(req, "cache_protected_len", None)) is not int
        or req.cache_protected_len != 0
    ):
        raise RuntimeError("request-private prefix provenance changed")
    if (
        type(row) is not torch.Tensor
        or row.ndim != 1
        or int(row.numel()) < int(prefix.numel())
        or row.untyped_storage().data_ptr() == prefix.untyped_storage().data_ptr()
        or not torch.equal(prefix, row[: int(prefix.numel())].to(dtype=torch.int64))
    ):
        raise RuntimeError("request-private prefix differs from its ReqToToken row")
    return marker


def replace_private_prefix(
    req: Any,
    marker: PrivatePrefixProvenance,
    replacement: Any,
    *,
    boundary: int,
) -> None:
    import torch

    if (
        getattr(req, PRIVATE_PREFIX_MARKER, None) is not marker
        or getattr(req, "prefix_indices", None) is not marker.tensor
        or any(hasattr(req, name) for name in SHARED_PREFIX_MARKERS)
    ):
        raise RuntimeError("request-private prefix identity changed before publication")
    private = replacement.to(dtype=torch.int64, copy=True)
    req.prefix_indices = private
    req.cache_protected_len = 0
    setattr(
        req,
        PRIVATE_PREFIX_MARKER,
        PrivatePrefixProvenance(
            private, marker.request_key, marker.request_lease, boundary
        ),
    )


def clear_private_prefix(req: Any, marker: PrivatePrefixProvenance) -> None:
    if (
        getattr(req, PRIVATE_PREFIX_MARKER, None) is not marker
        or getattr(req, "prefix_indices", None) is not marker.tensor
    ):
        raise RuntimeError("request-private prefix identity changed before cleanup")
    req.prefix_indices = marker.tensor[:0]
    req.cache_protected_len = 0
    delattr(req, PRIVATE_PREFIX_MARKER)
