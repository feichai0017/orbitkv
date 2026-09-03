from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

from .private_prefix import validate_private_prefix


@dataclass(frozen=True, slots=True)
class CompactCowMirror:
    assignments: tuple[tuple[Any, Any], ...]
    previous: tuple[int, ...]
    replacement: tuple[int, ...]


def preflight_compact_cow(
    req: Any,
    row: Any,
    absolute_boundary: int,
    substitutions: Sequence[tuple[int, int]],
    *,
    prefix_authoritative: bool,
) -> CompactCowMirror:
    """Validate and stage physical COW substitutions for one packed row."""

    import torch

    active = getattr(req, "_orbitkv_active_kv_len")
    retained = getattr(req, "_orbitkv_retained_locations", None)
    if (
        prefix_authoritative
        or isinstance(active, bool)
        or not isinstance(active, Integral)
        or not 0 <= int(active) <= absolute_boundary
        or not isinstance(retained, tuple)
        or len(retained) != int(active)
        or int(row.numel()) < absolute_boundary
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in retained
        )
        or len(set(retained)) != len(retained)
    ):
        raise RuntimeError("compact request lost its packed Full mirror authority")
    active = int(active)
    expected = torch.tensor(retained, dtype=torch.int64, device=row.device)
    if not torch.equal(row[:active].to(torch.int64), expected):
        raise RuntimeError("compact retained locations disagree with ReqToToken")
    if int(torch.count_nonzero(row[active:absolute_boundary]).item()):
        raise RuntimeError("compact ReqToToken tail is not clear")

    private = validate_private_prefix(
        req,
        row,
        getattr(req, "_orbitkv_request_key", None),
        getattr(req, "_orbitkv_request_lease", None),
        absolute_boundary,
    )
    prefix = getattr(req, "prefix_indices")
    prefix_count = int(prefix.numel())
    if prefix_count > active:
        raise RuntimeError("compact private prefix exceeds the active KV mirror")

    pairs = tuple((int(old), int(new)) for old, new in substitutions)
    mapping = dict(pairs)
    destinations = {new for _old, new in pairs}
    if (
        len(mapping) != len(pairs)
        or len(destinations) != len(pairs)
        or mapping.keys() & destinations
    ):
        raise RuntimeError("compact COW mapping is duplicate or ambiguous")
    replacement = tuple(mapping.get(int(value), int(value)) for value in retained)
    if len(set(replacement)) != len(replacement):
        raise RuntimeError("compact COW mapping aliases retained locations")

    assignments: list[tuple[Any, Any]] = []
    if replacement != retained:
        values = torch.tensor(replacement, dtype=row.dtype, device=row.device)
        assignments.append((row[:active], values))
        if private is not None:
            assignments.append((prefix, values[:prefix_count].to(prefix.dtype)))
    return CompactCowMirror(tuple(assignments), retained, replacement)
