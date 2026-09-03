from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from ..runtime import (
    DETACHED_CLEAR,
    DETACHED_PREFIX_TRANSFER,
    DETACHED_REQUEST_RELEASE,
)


def sorted_membership(needles: Any, sorted_haystack: Any) -> Any:
    import torch

    if int(sorted_haystack.numel()) == 0:
        return torch.zeros_like(needles, dtype=torch.bool)
    needles = needles.contiguous()
    positions = torch.searchsorted(sorted_haystack, needles)
    valid = positions < int(sorted_haystack.numel())
    safe = positions.clamp(max=int(sorted_haystack.numel()) - 1)
    return valid & (sorted_haystack[safe] == needles)


def synchronize_mirror(req_to_token_pool: Any) -> None:
    device = getattr(req_to_token_pool, "device", None)
    if device is not None and str(device).startswith("cuda"):
        import torch

        torch.get_device_module(device).current_stream(device).synchronize()


def page_matches_backend(page: Any, backend_index: int, arena: Any) -> bool:
    """Return whether one leased page names the exact physical backend slot."""

    return (
        page.engine_epoch == arena.engine_epoch
        and page.pool_epoch == arena.pool_epoch
        and page.pool_id == arena.pool_id
        and page.generation > 0
        and backend_index
        == arena.backend_base_index + page.page_id - arena.first_page_id
        and arena.first_page_id
        <= page.page_id
        < arena.first_page_id + arena.page_count
    )


def is_zero_page(page: Any) -> bool:
    """Return whether a page is the canonical absence sentinel."""

    return (
        page.engine_epoch == 0
        and page.pool_epoch == 0
        and page.pool_id == 0
        and page.page_id == 0
        and page.generation == 0
    )


def retirement_identities(certificates: tuple[Any, ...]) -> tuple[tuple[Any, ...], ...]:
    """Project certificates onto the physical sources a cleanup can prove."""

    return tuple(
        (
            item.page,
            item.class_id,
            item.backend_domain,
            item.logical_ordinal,
            item.backend_index,
            item.token_begin,
            item.token_end_exclusive,
        )
        for item in certificates
    )


def validate_session_context_count(profile: str, count: int, total: int) -> None:
    """Reject a collective that mixes session and non-session authorities."""

    if count not in (0, total):
        raise RuntimeError(
            f"runtime-session {profile} cleanup mixed context authorities"
        )


def validate_session_retirement_sources(
    profile: str,
    identities: tuple[tuple[Any, ...], ...],
    sources: set[tuple[Any, ...]],
) -> None:
    """Require unique certificates sourced by this exact cleanup collective."""

    if len(set(identities)) != len(identities) or not set(identities).issubset(
        sources
    ):
        raise RuntimeError(
            f"runtime-session {profile} retirement coverage changed"
        )


def preflight_empty_publication(
    items: tuple[Any, ...],
    retirements: tuple[Any, ...],
    table: Any,
    maximum: int,
    validate_context: Any,
) -> MirrorCleanupPlan | None:
    """Validate a publication with no mirror effect without device work."""

    if not items or retirements or not table.is_contiguous() or any(
        item.releasing or item.detached or item.candidates for item in items
    ):
        return None

    import torch

    rows: set[int] = set()
    for item in items:
        req, _row = validate_context(item.context, table, rows)
        if (
            isinstance(item.boundary, bool)
            or not isinstance(item.boundary, Integral)
            or not 0 <= int(item.boundary) <= maximum
        ):
            raise RuntimeError("manager cleanup boundary exceeds ReqToToken")
        boundary = int(item.boundary)
        prefix = getattr(req, "prefix_indices", None)
        if prefix is not None and (
            type(prefix) is not torch.Tensor
            or prefix.ndim != 1
            or prefix.dtype is not torch.int64
            or prefix.device != table.device
        ):
            raise RuntimeError(
                "SGLang prefix mirror is not a device int64 vector"
            )
        if prefix is not None and int(prefix.numel()) > boundary:
            raise RuntimeError("SGLang prefix mirror exceeds its KV boundary")
    return MirrorCleanupPlan.empty()


@dataclass(frozen=True, slots=True)
class MirrorCleanupContext:
    req: Any
    request_row: int
    request_key: Any | None = None
    request_id: Any | None = None


@dataclass(slots=True)
class MirrorCleanupPlan:
    zero_views: tuple[Any, ...]
    mapping: Any | None
    mapping_indices: tuple[Any, ...]
    frontier_updates: tuple[tuple[Any, int, int], ...]
    indexed_zeroes: tuple[tuple[Any, Any], ...] = ()
    private_prefixes: tuple[tuple[Any, Any], ...] = ()
    synchronized: bool = field(default=False, init=False)

    @classmethod
    def empty(cls) -> MirrorCleanupPlan:
        """Return a plan whose four phases have no mirror side effects."""

        return cls((), None, (), ())

    @property
    def requires_device_sync(self) -> bool:
        return any(
            int(target.numel()) > 0
            for targets in (self.zero_views, self.mapping_indices)
            for target in targets
        ) or any(
            int(indices.numel()) > 0 for _target, indices in self.indexed_zeroes
        )


def validate_session_hybrid_release(
    item: Any,
    boundary: int,
    full: Any,
    sliding: Any,
    *,
    classes: dict[int, Any],
    arenas: dict[int, Any],
    page_tokens: int,
    validate_detached: Any,
    page_matches_backend: Any,
    is_zero_page: Any,
) -> tuple[int | None, tuple[tuple[Any, ...], ...]]:
    """Prove the exact dense Full/SWA release image from topology."""

    context = item.context
    if (
        context.request_key is None
        or context.request_id is None
        or item.candidates
        or getattr(context.req, "_orbitkv_retained_locations", None) is not None
        or getattr(context.req, "_orbitkv_retained_swa_locations", None) is not None
    ):
        raise RuntimeError("runtime-session Hybrid release context is invalid")
    detached = tuple(item.detached)
    if boundary == 0:
        if detached:
            raise RuntimeError(
                "runtime-session Hybrid release class coverage changed"
            )
        return None, ()
    reasons = {entry.reason for entry in detached}
    if reasons not in ({DETACHED_REQUEST_RELEASE}, {DETACHED_PREFIX_TRANSFER}):
        raise RuntimeError(
            "runtime-session Hybrid release has invalid or mixed detach reasons"
        )
    end_ordinal = (boundary + page_tokens - 1) // page_tokens
    window = getattr(sliding, "window_tokens", None)
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise RuntimeError("runtime-session Hybrid SWA geometry is invalid")
    first_swa = max(0, boundary - (int(window) - 1)) // page_tokens
    expected = {
        full.class_id: {
            (ordinal, ordinal * page_tokens, min((ordinal + 1) * page_tokens, boundary))
            for ordinal in range(end_ordinal)
        },
        sliding.class_id: {
            (ordinal, ordinal * page_tokens, min((ordinal + 1) * page_tokens, boundary))
            for ordinal in range(first_swa, end_ordinal)
        },
    }
    actual: dict[int, list[tuple[int, int, int]]] = {class_id: [] for class_id in expected}
    identities = []
    pages = set()
    for entry in detached:
        validate_detached(entry, boundary, classes)
        arena = arenas.get(entry.class_id)
        if (
            entry.class_id not in actual
            or entry.action != DETACHED_CLEAR
            or not is_zero_page(entry.replacement)
            or entry.replacement_backend_index != 0
            or arena is None
            or not page_matches_backend(entry.old, entry.old_backend_index, arena)
            or entry.old in pages
        ):
            raise RuntimeError(
                "runtime-session Hybrid release transition is invalid"
            )
        pages.add(entry.old)
        actual[entry.class_id].append(
            (entry.logical_ordinal, entry.token_begin, entry.token_end_exclusive)
        )
        identities.append((
            entry.old, entry.class_id, entry.backend_domain, entry.logical_ordinal,
            entry.old_backend_index, entry.token_begin, entry.token_end_exclusive,
        ))
    for class_id, expected_spans in expected.items():
        actual_spans = actual[class_id]
        if len(actual_spans) != len(expected_spans) or set(actual_spans) != expected_spans:
            label = "Full" if class_id == full.class_id else "SWA"
            raise RuntimeError(
                f"runtime-session Hybrid release {label} coverage changed"
            )
    return next(iter(reasons)), tuple(identities)
