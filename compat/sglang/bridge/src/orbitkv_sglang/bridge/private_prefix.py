from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from ..ffi.session_types import EnginePrefixId


@dataclass(frozen=True, slots=True, eq=False)
class PrivatePrefixProvenance:
    """Proof that ``prefix_indices`` is a request-private row snapshot."""

    tensor: Any
    request_key: tuple[str, str | bytes | int]
    request_lease: Any
    boundary: int


PRIVATE_PREFIX_MARKER = "_orbitkv_private_prefix"
ENGINE_REQUEST_ID_MARKER = "_orbitkv_engine_request_id"
ENGINE_PREFIX_ID_MARKER = "_orbitkv_engine_prefix_id"
SHARED_PREFIX_MARKERS = (
    "_orbitkv_prefix_node",
    "_orbitkv_prefix_semantic",
    ENGINE_PREFIX_ID_MARKER,
    "_orbitkv_provisional_prefix_lock",
    "_orbitkv_prefix_lock_held",
)
_MISSING = object()


def request_identity(req: Any) -> Any:
    """Return the one request identity carried by the engine object."""

    engine_request_id = getattr(req, ENGINE_REQUEST_ID_MARKER, _MISSING)
    request_lease = getattr(req, "_orbitkv_request_lease", _MISSING)
    if engine_request_id is not _MISSING:
        if request_lease is not _MISSING:
            raise RuntimeError(
                "request carries both session and legacy OrbitKV identities"
            )
        return engine_request_id
    if request_lease is not _MISSING:
        return request_lease
    return None


def validate_private_prefix(
    req: Any,
    row: Any,
    key: tuple[str, str | bytes | int],
    identity: Any,
    boundary: int,
) -> PrivatePrefixProvenance | None:
    """Return exact request-private provenance or reject ambiguous Prefix state."""

    import torch

    if any(hasattr(req, name) for name in SHARED_PREFIX_MARKERS):
        raise RuntimeError("current request-private profile forbids Prefix mirrors")
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
        raise RuntimeError("current request-private profile forbids Prefix mirrors")
    if (
        marker.tensor is not prefix
        or marker.request_key != key
        or identity is None
        or marker.request_lease != identity
        or getattr(req, "_orbitkv_request_key", None) != key
        or request_identity(req) != identity
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


def update_private_prefix(
    req: Any,
    row: Any,
    key: tuple[str, str | bytes | int],
    identity: Any,
    boundary: int,
) -> PrivatePrefixProvenance | None:
    """Validate the old snapshot and copy the confirmed row prefix."""

    import torch

    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, Integral)
        or int(boundary) < 0
        or type(row) is not torch.Tensor
        or row.ndim != 1
        or int(boundary) > int(row.numel())
    ):
        raise RuntimeError(
            "request-private prefix boundary exceeds its ReqToToken row"
        )
    boundary = int(boundary)
    validate_private_prefix(req, row, key, identity, boundary)
    private = row[:boundary].to(dtype=torch.int64, copy=True)
    req.prefix_indices = private
    req.cache_protected_len = 0
    if boundary == 0:
        return None
    marker = PrivatePrefixProvenance(private, key, identity, boundary)
    setattr(req, PRIVATE_PREFIX_MARKER, marker)
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


def clear_request_identity(req: Any) -> None:
    """Remove all request identity and prefix metadata after release."""

    for name in (
        "_orbitkv_request_key",
        "_orbitkv_request_lease",
        ENGINE_REQUEST_ID_MARKER,
        ENGINE_PREFIX_ID_MARKER,
        *SHARED_PREFIX_MARKERS,
        PRIVATE_PREFIX_MARKER,
    ):
        if hasattr(req, name):
            delattr(req, name)


def validate_shared_prefix_metadata(
    req: Any,
    *,
    key: tuple[str, str | bytes | int],
    request_id: Any,
    prefix_id: EnginePrefixId,
    boundary: int,
    node: Any,
    semantic: Any,
    provisional: bool,
) -> None:
    """Validate one exact shared-prefix marker set stored on the request."""

    if (
        getattr(req, "_orbitkv_request_key", None) != key
        or request_identity(req) != request_id
        or getattr(req, ENGINE_PREFIX_ID_MARKER, None) != prefix_id
        or getattr(req, "_orbitkv_prefix_node", None) is not node
        or getattr(req, "_orbitkv_prefix_semantic", None) != semantic
        or getattr(req, "last_node", None) is not node
        or type(getattr(req, "_orbitkv_provisional_prefix_lock", None)) is not bool
        or type(getattr(req, "_orbitkv_prefix_lock_held", None)) is not bool
        or getattr(req, "_orbitkv_provisional_prefix_lock") != provisional
        or getattr(req, "_orbitkv_prefix_lock_held") == provisional
        or getattr(node, "boundary", None) != boundary
        or getattr(semantic, "boundary", None) != boundary
        or getattr(semantic, "digest", None) != getattr(node, "digest", None)
    ):
        raise RuntimeError("shared Prefix metadata changed")


def install_shared_prefix_metadata(
    req: Any,
    *,
    key: tuple[str, str | bytes | int],
    request_id: Any,
    prefix_id: EnginePrefixId,
    node: Any,
    semantic: Any,
    indices: Any,
    boundary: int,
    provisional: bool,
) -> None:
    """Install one exact shared-prefix mirror and ownership marker set."""

    import torch

    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, Integral)
        or int(boundary) <= 0
        or type(prefix_id) is not EnginePrefixId
        or type(indices) is not torch.Tensor
        or indices.ndim != 1
        or indices.dtype is not torch.int64
        or int(indices.numel()) != int(boundary)
    ):
        raise RuntimeError("shared Prefix installation received invalid metadata")
    req.prefix_indices = indices
    req.cache_protected_len = int(boundary)
    req.last_node = node
    req.last_host_node = node
    req.best_match_node = node
    req._orbitkv_request_key = key
    req._orbitkv_engine_request_id = request_id
    req._orbitkv_engine_prefix_id = prefix_id
    req._orbitkv_prefix_node = node
    req._orbitkv_prefix_semantic = semantic
    req._orbitkv_provisional_prefix_lock = bool(provisional)
    req._orbitkv_prefix_lock_held = not bool(provisional)
    validate_shared_prefix_metadata(
        req,
        key=key,
        request_id=request_id,
        prefix_id=prefix_id,
        boundary=int(boundary),
        node=node,
        semantic=semantic,
        provisional=bool(provisional),
    )
