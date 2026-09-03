from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

from ..ffi.session_types import (
    EngineMaterializedRequest,
    EnginePrefixId,
    EnginePrefixLookup,
    EnginePublishedPrefix,
    EnginePublishedPrefixRelease,
)
from ..runtime import (
    FailStopped,
    PrefixSemanticKey,
    sglang_page_id,
)
from . import session_cache as _session_cache
from . import state as _state
from .private_prefix import install_shared_prefix_metadata
from .state import _config, _request_key, _runtime


@dataclass(frozen=True, slots=True)
class PrefixMaterialization:
    """Validated SGLang mirror plus its semantic SWA eviction frontier."""

    indices: Any
    swa_evicted_seqlen: int


def ensure_session_request_identity(cache: Any, req: Any) -> tuple[Any, Any]:
    key = _request_key(req)
    request_id = getattr(req, "_orbitkv_engine_request_id", None)
    if request_id is not None:
        if getattr(req, "_orbitkv_request_key", None) != key:
            raise RuntimeError("session request key changed before lowering")
        return key, request_id
    runtime = _runtime()
    views = runtime.acquire_unbound((key,))
    if len(views) != 1:
        runtime.fail_stop("session acquire cardinality changed")
        raise FailStopped(runtime.failure_reason or "invalid session acquire")
    request_id = views[0].request_id
    req._orbitkv_request_key = key
    req._orbitkv_engine_request_id = request_id
    cache._register_pending_session_request(req)
    return key, request_id


def semantic_endpoints(cache: Any, tokens: tuple[int, ...]) -> tuple[PrefixSemanticKey, ...]:
    hasher = hashlib.sha256()
    result = []
    for index, token in enumerate(tokens, start=1):
        hasher.update(token.to_bytes(8, "little", signed=False))
        if index % cache.page_size == 0:
            result.append(
                PrefixSemanticKey(cache._namespace, hasher.digest(), index)
            )
    return tuple(result)


def semantic(cache: Any, tokens: tuple[int, ...]) -> PrefixSemanticKey:
    if not tokens or len(tokens) % cache.page_size:
        raise RuntimeError("prefix publication endpoint is not page aligned")
    return semantic_endpoints(cache, tokens)[-1]


def match_prefix_session(
    cache: Any,
    req: Any,
    key: Any,
    semantic_key: PrefixSemanticKey,
    node: Any,
    result_type: Any,
) -> Any:
    runtime = _runtime()
    _key, request_id = ensure_session_request_identity(cache, req)
    current = getattr(req, "_orbitkv_prefix_node", None)
    hint = runtime.prefix_lookup((semantic_key,))
    if len(hint) != 1:
        runtime.fail_stop("session prefix lookup cardinality changed")
        raise FailStopped(runtime.failure_reason or "invalid session prefix lookup")
    lookup = hint[0]
    if (
        type(lookup) is not EnginePrefixLookup
        or lookup.key != semantic_key
        or lookup.candidate != node.prefix
        or lookup.resident_count < 0
    ):
        runtime.fail_stop("session prefix lookup differed from local radix")
        raise FailStopped(runtime.failure_reason or "stale session prefix identity")
    if current is not None:
        entry = _session_cache.require_shared_prefix(cache, req, allow_pending=True)
        if entry.node is not current or entry.semantic != semantic_key:
            raise RuntimeError("attached session prefix identity changed")
        cache._touch(current)
        return cache._finish_match(
            cache._match_result(req.prefix_indices, current, result_type), True
        )
    control_id = None
    committed = False
    waiting_lock = False
    try:
        control_id = runtime.prepare_prefix_attach(((key, lookup),))
        runtime.commit_control(control_id)
        committed = True
        plan = runtime.read_control(control_id)
        if len(plan.requests) != 1:
            raise RuntimeError("session prefix attach plan cardinality changed")
        materialized = plan.requests[0]
        boundary = int(materialized.boundary)
        if (
            materialized.request_id != request_id
            or boundary != int(semantic_key.boundary)
            or boundary != int(node.boundary)
            or int(materialized.resident_count) != int(lookup.resident_count)
            or int(materialized.resident_count) != int(node.resident_count)
            or len(materialized.pages) != int(materialized.resident_count)
        ):
            raise RuntimeError("session prefix attach plan changed identity")
        materialization = validate_prefix_materialization(
            cache, materialized.pages, boundary, int(materialized.resident_count)
        )
        indices = materialization.indices
        cache.inc_lock_ref(node)
        waiting_lock = True
        install_shared_prefix_metadata(
            req,
            key=key,
            request_id=request_id,
            prefix_id=lookup.candidate,
            node=node,
            semantic=semantic_key,
            indices=indices,
            boundary=boundary,
            provisional=True,
        )
        _session_cache.register_pending_shared_prefix(
            cache,
            req=req,
            prefix_id=lookup.candidate,
            semantic=semantic_key,
            node=node,
            boundary=boundary,
            control_id=control_id,
            plan=plan,
            swa_evicted_seqlen=materialization.swa_evicted_seqlen,
        )
    except BaseException as error:
        if committed:
            reason = (
                "committed session prefix attach failed before admission: "
                f"{type(error).__name__}: {error}"
            )
            quarantine_error: BaseException | None = None
            if control_id is not None:
                try:
                    runtime.quarantine_control(control_id)
                except BaseException as caught:
                    quarantine_error = caught
            if waiting_lock:
                try:
                    cache.dec_lock_ref(node)
                except BaseException as caught:
                    quarantine_error = quarantine_error or caught
            if quarantine_error is not None:
                reason += (
                    "; attach quarantine became uncertain: "
                    f"{type(quarantine_error).__name__}: {quarantine_error}"
                )
            runtime.fail_stop(reason)
            raise FailStopped(runtime.failure_reason or reason) from error
        if control_id is not None:
            try:
                runtime.abort_control(control_id)
            except BaseException as caught:
                runtime.fail_stop(
                    "prepared session prefix attach abort became uncertain: "
                    f"{type(caught).__name__}: {caught}"
                )
                raise FailStopped(
                    runtime.failure_reason or "prefix attach abort failed"
                ) from error
        raise
    cache._touch(node)
    return cache._finish_match(
        cache._match_result(indices, node, result_type), True
    )


def publication_prefix_id(
    publication: EnginePublishedPrefix | EnginePublishedPrefixRelease,
) -> EnginePrefixId | None:
    if isinstance(publication, EnginePublishedPrefix):
        return publication.prefix_id
    if isinstance(publication, EnginePublishedPrefixRelease):
        return publication.prefix_id
    return None


def record_publication(
    cache: Any,
    publication: EnginePublishedPrefix | EnginePublishedPrefixRelease,
    tokens: tuple[int, ...],
) -> Any:
    endpoints = semantic_endpoints(cache, tokens)
    semantic_key = endpoints[-1]
    if (
        publication.key != semantic_key
        or publication_prefix_id(publication) is None
        or publication.resident_count <= 0
    ):
        _runtime().fail_stop("manager returned an invalid published prefix")
        raise FailStopped(_runtime().failure_reason or "invalid prefix publication")
    full_pages = semantic_key.boundary // cache.page_size
    swa_pages = publication.resident_count - full_pages
    if (
        swa_pages < 0
        or swa_pages > full_pages
        or (_config().sliding_class is None and swa_pages != 0)
    ):
        _runtime().fail_stop("manager prefix resident census changed")
        raise FailStopped(
            _runtime().failure_reason or "invalid prefix resident census"
        )
    map_key = (semantic_key.boundary, semantic_key.digest)
    parent = cache.root_node
    for endpoint in endpoints:
        endpoint_key = (endpoint.boundary, endpoint.digest)
        edge = tokens[endpoint.boundary - cache.page_size : endpoint.boundary]
        node = cache._nodes.get(endpoint_key)
        if node is None:
            node = type(cache.root_node)(
                endpoint.boundary,
                edge,
                endpoint.digest,
                None,
                0,
                0,
                parent,
            )
            cache._nodes[endpoint_key] = node
            parent.children[endpoint.digest] = node
            edge_tokens = cache._full_edge_tokens(node)
            cache._full_total_tokens += edge_tokens
            cache._full_evictable_tokens += edge_tokens
        elif (
            node.evicted
            or node.parent is not parent
            or node.edge != edge
            or parent.children.get(endpoint.digest) is not node
        ):
            _runtime().fail_stop("semantic radix topology changed")
            raise FailStopped(
                _runtime().failure_reason or "invalid semantic radix topology"
            )
        parent = node
    node = cache._nodes[map_key]
    if node.prefix is not None:
        _runtime().fail_stop("manager published a duplicate semantic prefix")
        raise FailStopped(_runtime().failure_reason or "duplicate prefix")
    node.prefix = publication_prefix_id(publication)
    node.resident_count = publication.resident_count
    current = node
    for _ in range(swa_pages):
        if current.swa_ref_count == 0:
            cache._swa_total_tokens += cache.page_size
            if current.lock_ref == 0:
                cache._swa_evictable_tokens += cache.page_size
            else:
                cache._swa_protected_tokens += cache.page_size
        current.swa_ref_count += 1
        if current.parent is None:
            _runtime().fail_stop("prefix SWA residency exceeds its radix path")
            raise FailStopped(
                _runtime().failure_reason or "invalid prefix SWA residency"
            )
        current = current.parent
    cache._touch(node)
    _state._counter_add("prefix_publishes")
    return node


def validate_prefix_materialization(
    cache: Any, pages: Sequence[Any], boundary: int, resident_count: int
) -> PrefixMaterialization:
    import torch

    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, Integral)
        or int(boundary) <= 0
        or int(boundary) % cache.page_size
    ):
        raise RuntimeError("manager attached a non-page-aligned endpoint")
    pages = tuple(pages)
    if (
        isinstance(resident_count, bool)
        or not isinstance(resident_count, Integral)
        or int(resident_count) != len(pages)
    ):
        raise RuntimeError(
            "manager prefix resident count differs from all returned pages"
        )
    config = _config()
    full = config.full_class
    if full is None:
        raise RuntimeError("OrbitKV prefix cache requires a Full KV class")
    pages_by_class: dict[int, list[Any]] = {item.class_id: [] for item in config.classes}
    seen_physical: set[tuple[int, int]] = set()
    for page in pages:
        try:
            class_config = config.classes_by_id[page.class_id]
        except KeyError as error:
            raise RuntimeError("manager materialized an unknown KV class") from error
        arena = _runtime().arenas_by_class[class_config.class_id]
        physical = (int(page.page.pool_id), int(page.page.page_id))
        if (
            page.backend_domain != class_config.backend_domain
            or page.backend_domain != arena.backend_domain
            or page.page.engine_epoch != arena.engine_epoch
            or page.page.pool_epoch != arena.pool_epoch
            or page.page.pool_id != arena.pool_id
            or page.page.generation <= 0
            or not arena.first_page_id
            <= page.page.page_id
            < arena.first_page_id + arena.page_count
            or page.backend_index
            != arena.backend_base_index + page.page.page_id - arena.first_page_id
            or physical in seen_physical
        ):
            raise RuntimeError(
                "manager prefix page identity differs from its class arena"
            )
        seen_physical.add(physical)
        pages_by_class[class_config.class_id].append(page)
    full_pages = pages_by_class[full.class_id]
    expected_pages = boundary // cache.page_size
    if len(full_pages) != expected_pages:
        raise RuntimeError("manager did not materialize the complete Full prefix")
    primary_parts = []
    full_arena = _runtime().arenas_by_class[full.class_id]
    for ordinal, page in enumerate(full_pages):
        if (
            page.logical_ordinal != ordinal
            or page.temporal_cell_index != ordinal
            or page.temporal_cycle != 0
            or page.valid_token_count != cache.page_size
            or page.visible_token_offset != 0
            or page.visible_token_count != cache.page_size
        ):
            raise RuntimeError("manager Full prefix endpoint is not page exact")
        start = (
            sglang_page_id(page.backend_index, full_arena.backend_base_index)
            * cache.page_size
        )
        primary_parts.append(
            torch.arange(
                start,
                start + cache.page_size,
                dtype=torch.int64,
                device=cache.device,
            )
        )
    primary = torch.cat(primary_parts) if primary_parts else cache._empty
    sliding = config.sliding_class
    if sliding is None:
        if any(
            values
            for class_id, values in pages_by_class.items()
            if class_id != full.class_id
        ):
            raise RuntimeError("manager Full prefix returned an extra KV class")
        return PrefixMaterialization(primary, 0)

    if len(config.classes) != 2 or tuple(
        item.class_id for item in config.classes
    ) != (full.class_id, sliding.class_id):
        raise RuntimeError(
            "shared Prefix materialization requires exact ordered Full+SWA classes"
        )
    window = sliding.window_tokens
    period = sliding.period_blocks
    if (
        isinstance(window, bool)
        or not isinstance(window, Integral)
        or int(window) <= 0
        or isinstance(period, bool)
        or not isinstance(period, Integral)
        or int(period) != 1 + (int(window) - 1 + cache.page_size - 1) // cache.page_size
    ):
        raise RuntimeError("compiled SWA prefix geometry is invalid")
    frontier = max(0, int(boundary) - (int(window) - 1))
    first_ordinal = frontier // cache.page_size
    swa_pages = pages_by_class[sliding.class_id]
    expected_swa_count = expected_pages - first_ordinal
    if len(swa_pages) != expected_swa_count:
        raise RuntimeError("manager did not materialize the exact SWA retained tail")
    sliding_arena = _runtime().arenas_by_class[sliding.class_id]
    full_locations = []
    swa_locations = []
    for offset, page in enumerate(swa_pages):
        ordinal = first_ordinal + offset
        token_begin = ordinal * cache.page_size
        token_end = min(token_begin + cache.page_size, int(boundary))
        visible_begin = max(frontier, token_begin)
        expected_valid = token_end - token_begin
        expected_visible_offset = visible_begin - token_begin
        expected_visible_count = token_end - visible_begin
        expected_cell = ordinal % int(period)
        expected_cycle = ordinal // int(period)
        if (
            page.logical_ordinal != ordinal
            or page.temporal_cell_index != expected_cell
            or page.temporal_cycle != expected_cycle
            or page.valid_token_count != expected_valid
            or page.visible_token_offset != expected_visible_offset
            or page.visible_token_count != expected_visible_count
        ):
            raise RuntimeError(
                "manager did not materialize the exact SWA retained tail"
            )
        full_locations.append(primary[token_begin:token_end])
        swa_start = (
            sglang_page_id(page.backend_index, sliding_arena.backend_base_index)
            * cache.page_size
        )
        swa_locations.append(
            torch.arange(
                swa_start,
                swa_start + expected_valid,
                dtype=torch.int64,
                device=cache.device,
            )
        )
    if int(resident_count) != len(full_pages) + len(swa_pages):
        raise RuntimeError(
            "manager prefix resident count differs from exact Full+SWA pages"
        )
    full_vector = torch.cat(full_locations)
    expected_swa = torch.cat(swa_locations)
    mapping = _state._ALLOCATOR.full_to_swa_index_mapping
    if not torch.equal(mapping[full_vector].to(dtype=torch.int64), expected_swa):
        raise RuntimeError(
            "manager prefix disagrees with the Full-to-SWA mirror"
        )
    return PrefixMaterialization(primary, frontier)


def materialize_prefix_pages(
    cache: Any, pages: Sequence[Any], boundary: int, resident_count: int
) -> Any:
    return validate_prefix_materialization(
        cache, pages, boundary, resident_count
    ).indices


def materialize_prefix(
    cache: Any, materialized: EngineMaterializedRequest, boundary: int
) -> Any:
    if materialized.boundary != boundary:
        raise RuntimeError("manager attached a non-page-aligned endpoint")
    return validate_prefix_materialization(
        cache,
        materialized.pages,
        boundary,
        int(materialized.resident_count),
    ).indices


def adopt_published_node(
    cache: Any,
    req: Any,
    node: Any,
    boundary: int,
    expected_old: Any,
) -> None:
    import torch

    old = cache._preflight_release_node(req, provisional=False)
    if old is not expected_old:
        raise RuntimeError("unfinished prefix changed after publication")
    if old is not None and old is not cache.root_node and old is not node:
        cache.dec_lock_ref(old)
    if old is not node:
        cache.inc_lock_ref(node)
    row = int(req.req_pool_idx)
    indices = cache.req_to_token_pool.req_to_token[row, :boundary].to(
        dtype=torch.int64, copy=True
    )
    semantic_key = PrefixSemanticKey(cache._namespace, node.digest, node.boundary)
    install_shared_prefix_metadata(
        req,
        key=_request_key(req),
        request_id=getattr(req, "_orbitkv_engine_request_id", None),
        prefix_id=node.prefix,
        node=node,
        semantic=semantic_key,
        indices=indices,
        boundary=boundary,
        provisional=False,
    )
    _session_cache.clear_shared_prefix(cache, req)
    _session_cache.register_active_shared_prefix(
        cache,
        req=req,
        prefix_id=node.prefix,
        semantic=semantic_key,
        node=node,
        boundary=boundary,
    )
