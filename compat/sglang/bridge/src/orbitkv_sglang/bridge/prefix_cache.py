from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

from ..ffi.session_types import (
    EngineMaterializedRequest,
    EnginePrefixEvictionPlan,
    EnginePrefixId,
    EnginePublishedPrefix,
    EnginePublishedPrefixRelease,
)
from ..runtime import (
    FailStopped,
    PrefixSemanticKey,
)
from . import state as _state
from .prefix_eviction import PrefixEvictionMixin
from .prefix_runtime_helpers import (
    adopt_published_node as _adopt_published_node_helper,
    ensure_session_request_identity as _ensure_session_request_identity,
    match_prefix_session as _match_prefix_session_helper,
    materialize_prefix as _materialize_prefix_helper,
    materialize_prefix_pages as _materialize_prefix_pages_helper,
    publication_prefix_id as _publication_prefix_id_helper,
    record_publication as _record_publication_helper,
    semantic as _semantic_helper,
    semantic_endpoints as _semantic_endpoints_helper,
)
from .prefix_tokens import request_tokens as _request_tokens
from .prefix_tokens import tokens_from_radix_key as _tokens_from_radix_key
from .prefix_sanity import validate_prefix_cache
from . import session_cache as _session_cache
from .state import _config, _request_key, _runtime


_PREFIX_NAMESPACE_VERSION = b"orbitkv-sglang-prefix-v1\x00"


@dataclass(eq=False, slots=True)
class _PrefixNode:
    """Non-authoritative radix/LRU metadata; KV pages stay manager-owned."""

    boundary: int
    edge: tuple[int, ...]
    digest: bytes
    prefix: EnginePrefixId | None
    resident_count: int
    swa_ref_count: int
    parent: _PrefixNode | None
    children: dict[bytes, _PrefixNode] = field(default_factory=dict)
    lock_ref: int = 0
    last_access: int = 0
    evicted: bool = False
    backuped: bool = False

    @property
    def key(self) -> tuple[int, ...]:
        return self.edge

    def get_last_hash_value(self) -> str | None:
        return self.digest.hex() if self.digest else None

    def get_prefix_hash_values(self, node: _PrefixNode | None) -> list[str]:
        values: list[str] = []
        while node is not None and node.parent is not None:
            if node.digest:
                values.append(node.digest.hex())
            node = node.parent
        values.reverse()
        return values


@dataclass(frozen=True, slots=True)
class _ReleasePublication:
    semantic: PrefixSemanticKey
    tokens: tuple[int, ...]


class OrbitKvPrefixCache(
    PrefixEvictionMixin[EnginePrefixId],
    _session_cache.SessionCacheMixin,
    BasePrefixCache,
):
    """SGLang tree seam backed only by canonical manager prefix leases."""

    def __init__(self, params: Any) -> None:
        import torch

        config = _config()
        no_prefix = _state._requires_disabled_radix_cache()
        if bool(params.disable) != no_prefix:
            raise RuntimeError("OrbitKV radix disable mode differs from its state plan")
        if bool(getattr(params, "is_eagle", False)):
            raise RuntimeError("OrbitKV does not support EAGLE prefix keys")
        if getattr(params, "eviction_policy", "lru") != "lru":
            raise RuntimeError("OrbitKV supports only LRU prefix eviction")
        if getattr(params, "cache_ttl_seconds", None) is not None:
            raise RuntimeError("OrbitKV does not support prefix-cache TTL")
        if bool(getattr(params, "enable_kv_cache_events", False)):
            raise RuntimeError("OrbitKV does not support KV cache events")
        if int(params.page_size) != config.page_tokens:
            raise RuntimeError("OrbitKV prefix page size differs from the manager plan")
        if config.full_class is None and not no_prefix:
            raise RuntimeError(
                "OrbitKV shared Prefix cache requires a Full KV class"
            )
        if params.token_to_kv_pool_allocator is not _state._ALLOCATOR:
            raise RuntimeError("OrbitKV prefix cache received a foreign KV allocator")
        self.disable = False
        self.disable_finished_insert = bool(params.disable_finished_insert)
        self._no_prefix = no_prefix
        if self._no_prefix:
            self.disable_finished_insert = True
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = config.page_tokens
        sliding = config.sliding_class
        # SGLang's cache/scheduler contract stores the number of historical
        # tokens to the left of the current query.  KvPlanInput stores the
        # inclusive attention width, so a W-token window is represented as
        # W - 1 at this engine seam (for example, 128 -> 127).
        expected_window = None if sliding is None else sliding.kernel_window_left
        if getattr(params, "sliding_window_size", None) != expected_window:
            raise RuntimeError("SGLang sliding window differs from the manager plan")
        self.sliding_window_size = expected_window
        self.device = self.req_to_token_pool.device
        self.is_eagle = False
        self._namespace = hashlib.sha256(
            _PREFIX_NAMESPACE_VERSION + config.plan_fingerprint.encode("ascii")
        ).digest()
        self._nodes: dict[tuple[int, bytes], _PrefixNode] = {}
        self._full_total_tokens = 0
        self._full_evictable_tokens = 0
        self._full_protected_tokens = 0
        self._swa_total_tokens = 0
        self._swa_evictable_tokens = 0
        self._swa_protected_tokens = 0
        self._clock = 0
        self._released = False
        self.root_node = _PrefixNode(0, (), b"", None, 0, 0, None, lock_ref=1)
        self._empty = torch.empty((0,), dtype=torch.int64, device=self.device)
        if bool(getattr(params, "enable_metrics", False)):
            self.init_metrics_collector()

    def release_host_resources(self) -> None:
        if self._released:
            return
        runtime = _runtime()
        try:
            if runtime.failure_reason is None:
                try:
                    runtime.poll()
                    before = runtime.stats()
                    live_request_authority = (
                        before.active_requests,
                        before.active_snapshots,
                        before.prepared_steps,
                        before.submitted_steps,
                        before.reserved_pages,
                        before.writing_pages,
                        before.retiring_pages,
                        before.quarantined_pages,
                        before.exhausted_pages,
                        before.pending_reclamations,
                        before.total_request_page_refs,
                        before.total_reader_pins,
                    )
                    if any(live_request_authority):
                        runtime.fail_stop(
                            "shutdown encountered live OrbitKV request ownership"
                        )
                    else:
                        self.reset()
                    if runtime.failure_reason is None:
                        runtime.poll()
                        after = runtime.stats()
                        arena_stats = runtime.arena_stats()
                        live = (
                            after.active_requests,
                            after.active_snapshots,
                            after.active_prefixes,
                            after.evicted_prefixes,
                            after.prepared_steps,
                            after.submitted_steps,
                            after.reserved_pages,
                            after.writing_pages,
                            after.active_pages,
                            after.retiring_pages,
                            after.quarantined_pages,
                            after.exhausted_pages,
                            after.pending_reclamations,
                            after.total_request_page_refs,
                            after.total_prefix_page_refs,
                            after.total_reader_pins,
                        )
                        if any(live) or any(
                            item.free_pages != item.page_count
                            or item.reserved_pages
                            or item.writing_pages
                            or item.active_pages
                            or item.retiring_pages
                            or item.quarantined_pages
                            or item.exhausted_pages
                            or item.request_page_refs
                            or item.prefix_page_refs
                            or item.reader_pins
                            for item in arena_stats
                        ):
                            runtime.fail_stop(
                                "shutdown census retained OrbitKV ownership"
                            )
                except Exception:
                    # Official ShutdownReq does not guarantee quiescence.  Do
                    # not fabricate releases or ACKs for live prefixes; poison
                    # the session so native teardown still destroys the handle
                    # deterministically.
                    runtime.fail_stop(
                        "shutdown encountered live OrbitKV prefix ownership"
                    )
                    raise
        finally:
            try:
                _state._close_owned_runtime(runtime)
            finally:
                _session_cache.clear_session_cache(self)
                self._released = True

    def reset(self) -> None:
        if getattr(self, "_session_requests", {}):
            raise RuntimeError("cannot reset OrbitKV with live session requests")
        if getattr(self, "_session_pending_requests", {}):
            raise RuntimeError("cannot reset OrbitKV with pending session requests")
        if getattr(self, "_session_pending_shared_prefix", {}):
            raise RuntimeError(
                "cannot reset OrbitKV with pending shared Prefix materializations"
            )
        if getattr(self, "_session_active_shared_prefix", {}):
            raise RuntimeError(
                "cannot reset OrbitKV with active shared Prefix materializations"
            )
        if any(not node.evicted and node.lock_ref for node in self._nodes.values()):
            raise RuntimeError("cannot reset OrbitKV while prefixes are protected")
        leases = tuple(self._eviction_plan(0, 0, evict_all=True))
        if leases:
            capacity = self._eviction_batch_capacity()
            for start in range(0, len(leases), capacity):
                self._evict_leases(leases[start : start + capacity])
        if self._nodes:
            raise RuntimeError("OrbitKV prefix tree retained structural nodes")
        self._nodes.clear()
        self.root_node.children.clear()
        if any(
            (
                self._full_total_tokens,
                self._full_evictable_tokens,
                self._full_protected_tokens,
                self._swa_total_tokens,
                self._swa_evictable_tokens,
                self._swa_protected_tokens,
            )
        ):
            raise RuntimeError("OrbitKV prefix size census did not quiesce")
        self._clock = 0

    def supports_fast_match_prefix(self) -> bool:
        return False

    def supports_swa(self) -> bool:
        return _config().sliding_class is not None

    def supports_mamba(self) -> bool:
        # This profile intentionally keeps fixed state request-owned and never
        # enters SGLang's Prefix donation/COW protocol.
        return False

    def mamba_evictable_size(self) -> int:
        return 0

    def mamba_protected_size(self) -> int:
        return 0

    def swa_reprefill_tail_tokens(self) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return self._no_prefix

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def root_node_handle(self, extra_key: str | None = None) -> _PrefixNode:
        if extra_key is not None:
            raise RuntimeError("OrbitKV does not support namespaced prefix roots")
        return self.root_node

    def resolve_node_handle(self, node_handle: Any) -> _PrefixNode:
        self._require_node(node_handle, allow_root=True)
        return node_handle

    def is_backuped(self, node: Any) -> bool:
        self._require_node(node, allow_root=True)
        return False

    def is_root(self, node: Any) -> bool:
        return node is self.root_node

    def get_last_hash_value(self, node: Any) -> str | None:
        self._require_node(node, allow_root=True)
        return node.get_last_hash_value()

    def get_prefix_hash_values(self, node: Any) -> list[str]:
        self._require_node(node, allow_root=True)
        values = node.get_prefix_hash_values(node.parent)
        if node is not self.root_node:
            values.append(node.digest.hex())
        return values

    def match_prefix(self, params: Any) -> Any:
        from sglang.srt.mem_cache.base_prefix_cache import MatchResult

        if self._no_prefix:
            if params.req is not None:
                _ensure_session_request_identity(self, params.req)
            return self._finish_match(
                self._match_result(self._empty, self.root_node, MatchResult), False
            )
        req = params.req
        if req is not None:
            _ensure_session_request_identity(self, req)
        tokens = _tokens_from_radix_key(params.key, self.page_size)
        if not tokens:
            return self._finish_match(
                self._match_result(self._empty, self.root_node, MatchResult), False
            )
        local = tuple(
            (semantic, self._nodes.get((semantic.boundary, semantic.digest)))
            for semantic in self._semantic_endpoints(tokens)
        )
        candidates = tuple(
            (semantic, node)
            for semantic, node in local
            if node is not None and not node.evicted and node.prefix is not None
        )
        if not candidates or req is None:
            return self._finish_match(
                self._match_result(self._empty, self.root_node, MatchResult), False
            )
        semantic, node = candidates[-1]
        assert node is not None
        key = _request_key(req)
        return self._match_prefix_session(
            req, key, semantic, node, MatchResult
        )

    def _match_prefix_session(
        self,
        req: Any,
        key: Any,
        semantic: PrefixSemanticKey,
        node: _PrefixNode,
        result_type: Any,
    ) -> Any:
        return _match_prefix_session_helper(
            self, req, key, semantic, node, result_type
        )

    def cache_unfinished_req(self, req: Any, **kwargs: Any) -> None:
        del kwargs
        if getattr(self, "_no_prefix", False):
            _session_cache.cache_unfinished_request(self, req)
            return
        runtime = _runtime()
        key = _request_key(req)
        entry = _session_cache.require_request(self, req)
        initial_view = runtime.view_for(key)
        old_node = self._preflight_release_node(req, provisional=False)
        runtime.wait_requests((key,))
        binding = runtime.binding_for(key)
        view = runtime.view_for(key)
        boundary = int(view.boundary)
        if (
            binding.request_id != entry.request_id
            or binding.request_row != entry.row
            or getattr(req, "_orbitkv_request_key", None) != key
            or getattr(req, "_orbitkv_engine_request_id", None) != entry.request_id
            or int(initial_view.request_id) != int(entry.request_id)
            or boundary != _session_cache._request_boundary(req, allow_empty_kv=False)
        ):
            raise RuntimeError("unfinished session request identity changed")
        if boundary == 0 or boundary % self.page_size:
            return
        tokens = _request_tokens(req, boundary)
        semantic = self._semantic(tokens)
        node = self._nodes.get((boundary, semantic.digest))
        if self._preflight_release_node(req, provisional=False) is not old_node:
            raise RuntimeError("unfinished session request prefix changed before publish")
        if node is None or node.prefix is None:
            publications = runtime.prefix_publish(((key, semantic),))
            if len(publications) != 1:
                runtime.fail_stop("session prefix publish cardinality changed")
                raise FailStopped(
                    runtime.failure_reason or "invalid session prefix publish"
                )
            try:
                node = self._record_publication(publications[0], tokens)
            except Exception as error:
                if runtime.failure_reason is None:
                    runtime.fail_stop(
                        f"published session prefix installation failed: {error}"
                    )
                if isinstance(error, FailStopped):
                    raise
                raise FailStopped(
                    runtime.failure_reason or "session prefix installation failed"
                ) from error
        try:
            self._adopt_published_node(req, node, boundary, old_node)
        except Exception as error:
            runtime.fail_stop(f"unfinished session prefix adoption failed: {error}")
            raise FailStopped(
                runtime.failure_reason or "unfinished session prefix adoption failed"
            ) from error

    def cache_finished_req(
        self, req: Any, is_insert: bool = True, **kwargs: Any
    ) -> None:
        del kwargs
        from .lowering import _release_kv_cache

        _release_kv_cache(req, self, is_insert=is_insert)

    def publication_for_release(
        self, req: Any, *, is_insert: bool
    ) -> _ReleasePublication | None:
        if getattr(self, "_no_prefix", False):
            return None
        if not is_insert or self.disable_finished_insert:
            return None
        runtime = _runtime()
        key = _request_key(req)
        boundary = int(runtime.view_for(key).boundary)
        if boundary == 0 or boundary % self.page_size:
            return None
        tokens = _request_tokens(req, boundary)
        semantic = self._semantic(tokens)
        existing = self._nodes.get((boundary, semantic.digest))
        if existing is not None and existing.prefix is not None:
            return None
        return _ReleasePublication(semantic, tokens)

    def accept_release_publication(
        self,
        publication: EnginePublishedPrefixRelease,
        tokens: Sequence[int],
    ) -> _PrefixNode:
        return self._record_publication(publication, tuple(tokens))

    def _preflight_release_node(
        self, req: Any, *, provisional: bool
    ) -> _PrefixNode | None:
        if getattr(self, "_no_prefix", False):
            _session_cache.preflight_release_node(self, req)
            return None
        provisional_flag = getattr(req, "_orbitkv_provisional_prefix_lock", False)
        held_flag = getattr(req, "_orbitkv_prefix_lock_held", False)
        if type(provisional_flag) is not bool or type(held_flag) is not bool:
            raise RuntimeError("OrbitKV request prefix-lock marker is invalid")
        node = getattr(req, "_orbitkv_prefix_node", None)
        last_node = getattr(req, "last_node", None)
        if provisional:
            if not provisional_flag or held_flag:
                raise RuntimeError("waiting request lost its provisional prefix lock")
        elif provisional_flag:
            raise RuntimeError("admitted request retained a provisional prefix lock")
        elif not held_flag:
            if node is not None or (
                last_node is not None and last_node is not self.root_node
            ):
                raise RuntimeError("request prefix identity has no matching lock")
            return None
        if node is None or node is self.root_node or last_node is not node:
            raise RuntimeError("request prefix node differs from its held lock")
        self._require_node(node)
        if isinstance(node.lock_ref, bool) or not isinstance(node.lock_ref, int):
            raise RuntimeError("OrbitKV prefix lock count is invalid")
        if node.lock_ref <= 0:
            raise RuntimeError("request prefix lock is not protected")
        return node

    def _commit_release_node(
        self, req: Any, node: _PrefixNode | None, *, provisional: bool
    ) -> None:
        if getattr(self, "_no_prefix", False):
            _session_cache.commit_release_node(
                self, req, node, provisional=provisional
            )
            return
        if self._preflight_release_node(req, provisional=provisional) is not node:
            raise RuntimeError("request prefix identity changed after preflight")
        if node is None:
            return
        self.dec_lock_ref(node)
        marker = (
            "_orbitkv_provisional_prefix_lock"
            if provisional
            else "_orbitkv_prefix_lock_held"
        )
        delattr(req, marker)

    def evict(self, params: Any) -> Any:
        from sglang.srt.mem_cache.base_prefix_cache import EvictResult

        start = time.perf_counter()
        requested_full = max(0, int(params.num_tokens))
        requested_swa = max(0, int(params.swa_num_tokens))
        if requested_full == 0 and requested_swa == 0:
            return EvictResult()
        # Build and sort the complete topological LRU peel exactly once.  Walk
        # that immutable plan in estimated-budget batches; manager
        # last-reference certificates are the only actual capacity truth.  A
        # shared first batch may yield zero, in which case the next slice is
        # consumed without rescanning or reordering the radix.
        plan = self._complete_eviction_plan()
        batch_capacity = self._eviction_batch_capacity()
        cursor = 0
        full_tokens = swa_tokens = 0
        committed_outputs = []
        committed_leases = []
        try:
            while cursor < len(plan) and (
                full_tokens < requested_full or swa_tokens < requested_swa
            ):
                batch_start = cursor
                estimated_full = estimated_swa = 0
                while cursor < len(plan) and (
                    batch_capacity is None
                    or cursor - batch_start < batch_capacity
                ):
                    item = plan[cursor]
                    cursor += 1
                    estimated_full += item.full_tokens
                    estimated_swa += item.swa_tokens
                    if (
                        estimated_full
                        >= max(0, requested_full - full_tokens)
                        and estimated_swa
                        >= max(0, requested_swa - swa_tokens)
                    ):
                        break
                batch_leases = tuple(
                    item.prefix for item in plan[batch_start:cursor]
                )
                outputs, released_full, released_swa = self._evict_native(
                    batch_leases
                )
                committed_outputs.extend(outputs)
                committed_leases.extend(batch_leases)
                full_tokens += released_full
                swa_tokens += released_swa
        except Exception as error:
            if committed_outputs:
                self._commit_evictions(
                    committed_outputs,
                    committed_leases,
                    full_tokens,
                    swa_tokens,
                )
                runtime = _runtime()
                runtime.fail_stop("prefix eviction failed after an earlier commit")
                if isinstance(error, FailStopped):
                    raise
                raise FailStopped(
                    runtime.failure_reason or "partial prefix eviction failed"
                ) from error
            raise
        if committed_outputs:
            self._commit_evictions(
                committed_outputs,
                committed_leases,
                full_tokens,
                swa_tokens,
            )
        self.update_eviction_metrics(max(full_tokens, swa_tokens), start)
        return EvictResult(
            num_tokens_evicted=full_tokens,
            swa_num_tokens_evicted=swa_tokens,
        )

    @staticmethod
    def _eviction_batch_capacity() -> int:
        return _runtime().prefix_eviction_batch_capacity

    def inc_lock_ref(self, node: Any) -> Any:
        from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult

        self._require_node(node, allow_root=True)
        if node is self.root_node:
            return IncLockRefResult(delta=0)
        newly_protected = 0
        current = node
        while current is not self.root_node:
            if current.lock_ref == 0:
                edge_tokens = self._full_edge_tokens(current)
                newly_protected += edge_tokens
                self._full_evictable_tokens -= edge_tokens
                self._full_protected_tokens += edge_tokens
                if current.swa_ref_count > 0:
                    self._swa_evictable_tokens -= self.page_size
                    self._swa_protected_tokens += self.page_size
            current.lock_ref += 1
            assert current.parent is not None
            current = current.parent
        self._touch(node)
        return IncLockRefResult(delta=-newly_protected)

    def dec_lock_ref(self, node: Any, params: Any = None) -> Any:
        from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult

        del params
        self._require_node(node, allow_root=True)
        if node is self.root_node:
            return DecLockRefResult(delta=0)
        path = []
        current = node
        while current is not self.root_node:
            if current.lock_ref <= 0:
                raise RuntimeError("OrbitKV prefix lock underflow")
            path.append(current)
            assert current.parent is not None
            current = current.parent
        newly_evictable = 0
        for current in path:
            current.lock_ref -= 1
            if current.lock_ref == 0:
                edge_tokens = self._full_edge_tokens(current)
                newly_evictable += edge_tokens
                self._full_protected_tokens -= edge_tokens
                self._full_evictable_tokens += edge_tokens
                if current.swa_ref_count > 0:
                    self._swa_protected_tokens -= self.page_size
                    self._swa_evictable_tokens += self.page_size
        self._touch(node)
        return DecLockRefResult(delta=newly_evictable)

    def evictable_size(self) -> int:
        return self._full_evictable_tokens

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def swa_evictable_size(self) -> int:
        if _config().sliding_class is None:
            return 0
        return self._swa_evictable_tokens

    def protected_size(self) -> int:
        return self._full_protected_tokens

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_protected_size(self) -> int:
        if _config().sliding_class is None:
            return 0
        return self._swa_protected_tokens

    def total_size(self) -> int:
        return self._full_total_tokens

    def pretty_print(self) -> None:
        print(f"#OrbitKV prefixes: {len(self._nodes)}; #tokens: {self.total_size()}")

    def sanity_check(self) -> None:
        validate_prefix_cache(self)

    def _semantic_endpoints(
        self, tokens: tuple[int, ...]
    ) -> tuple[PrefixSemanticKey, ...]:
        return _semantic_endpoints_helper(self, tokens)

    def _semantic(self, tokens: tuple[int, ...]) -> PrefixSemanticKey:
        return _semantic_helper(self, tokens)

    def _record_publication(
        self,
        publication: EnginePublishedPrefix | EnginePublishedPrefixRelease,
        tokens: tuple[int, ...],
    ) -> _PrefixNode:
        return _record_publication_helper(self, publication, tokens)

    @staticmethod
    def _publication_prefix_id(
        publication: EnginePublishedPrefix | EnginePublishedPrefixRelease,
    ) -> EnginePrefixId | None:
        return _publication_prefix_id_helper(publication)

    def _accept_evictions(
        self,
        outputs: Sequence[Any],
        requested: Sequence[EnginePrefixId],
        *,
        _work: dict[str, int] | None = None,
    ) -> int:
        values, selected_nodes = self._validate_eviction_outputs(
            outputs, requested
        )
        selected = set(selected_nodes)
        published_parents = self._published_parent_map()
        for node, parent in published_parents.items():
            if parent in selected and node not in selected:
                _runtime().fail_stop("prefix eviction omitted a live descendant")
                raise FailStopped(
                    _runtime().failure_reason or "invalid prefix eviction closure"
                )
        preorder, _top_exclusive, swa_decrements = self._swa_selected_coverage(
            selected_nodes, _work
        )
        if any(
            node.swa_ref_count < swa_decrements[node]
            for node in preorder[1:]
        ):
            _runtime().fail_stop("prefix SWA residency underflowed")
            raise FailStopped(
                _runtime().failure_reason or "prefix SWA residency underflow"
            )
        for item in values:
            node = self._nodes[(item.key.boundary, item.key.digest)]
            node.prefix = None
            node.resident_count = 0
        for node in preorder[1:]:
            count = swa_decrements[node]
            if count == 0:
                continue
            if node.swa_ref_count == count:
                self._swa_total_tokens -= self.page_size
                if node.lock_ref == 0:
                    self._swa_evictable_tokens -= self.page_size
                else:
                    self._swa_protected_tokens -= self.page_size
            node.swa_ref_count -= count
        for node in sorted(selected, key=lambda item: item.boundary, reverse=True):
            if not node.evicted:
                self._prune_structural_leaf(node)
        return len(values)

    def _validate_eviction_outputs(
        self,
        outputs: Sequence[Any],
        requested: Sequence[EnginePrefixId],
    ) -> tuple[tuple[Any, ...], list[_PrefixNode]]:
        values = tuple(outputs)
        expected = tuple(requested)
        identities = tuple(self._evicted_prefix_id(item) for item in values)
        if len(values) != len(expected) or identities != expected:
            _runtime().fail_stop("prefix eviction identity changed")
            raise FailStopped(_runtime().failure_reason or "invalid prefix eviction")
        selected = []
        for item in values:
            node = self._nodes.get((item.key.boundary, item.key.digest))
            if (
                node is None
                or node.prefix != self._evicted_prefix_id(item)
                or node.lock_ref != 0
                or node.parent is None
                or node.parent.children.get(node.digest) is not node
            ):
                _runtime().fail_stop("prefix eviction returned a foreign or locked node")
                raise FailStopped(_runtime().failure_reason or "invalid prefix eviction")
            selected.append(node)
        return values, selected

    @staticmethod
    def _evicted_prefix_id(item: Any) -> EnginePrefixId | None:
        return getattr(item, "prefix_id", None)

    def _node_for_prefix_id(
        self, prefix_id: EnginePrefixId
    ) -> _PrefixNode:
        for node in self._nodes.values():
            if node.prefix == prefix_id:
                return node
        _runtime().fail_stop("prefix identity differs from the local radix")
        raise FailStopped(_runtime().failure_reason or "stale prefix identity")

    def _evict_native(
        self, leases: Sequence[EnginePrefixId]
    ) -> tuple[tuple[Any, ...], int, int]:
        runtime = _runtime()
        values = tuple(leases)
        requested_nodes = tuple(self._node_for_prefix_id(prefix_id) for prefix_id in values)
        control_id = runtime.prepare_prefix_evict(values)
        runtime.commit_control(control_id)
        plan = runtime.read_control(control_id)
        if not isinstance(plan, EnginePrefixEvictionPlan):
            runtime.fail_stop("session prefix eviction returned the wrong plan kind")
            raise FailStopped(
                runtime.failure_reason or "invalid session prefix eviction plan"
            )
        full = _config().full_class
        sliding = _config().sliding_class
        full_tokens = (
            self.page_size
            * sum(item.class_id == full.class_id for item in plan.retirements)
            if full is not None
            else 0
        )
        swa_tokens = (
            self.page_size
            * sum(item.class_id == sliding.class_id for item in plan.retirements)
            if sliding is not None
            else 0
        )
        runtime.confirm_control(control_id)
        outputs = tuple(
            type(
                "_EvictedSessionPrefix",
                (),
                {
                    "prefix_id": prefix_id,
                    "key": PrefixSemanticKey(
                        self._namespace, node.digest, node.boundary
                    ),
                },
            )()
            for prefix_id, node in zip(plan.prefix_ids, requested_nodes, strict=True)
        )
        return outputs, full_tokens, swa_tokens

    def _commit_evictions(
        self,
        outputs: Sequence[Any],
        leases: Sequence[EnginePrefixId],
        full_tokens: int,
        swa_tokens: int,
    ) -> None:
        runtime = _runtime()
        try:
            count = self._accept_evictions(outputs, leases)
        except Exception as error:
            if runtime.failure_reason is None:
                runtime.fail_stop(f"prefix eviction installation failed: {error}")
            if isinstance(error, FailStopped):
                raise
            raise FailStopped(
                runtime.failure_reason or "prefix eviction installation failed"
            ) from error
        _state._counter_add("prefix_evictions", count)
        _state._counter_add("prefix_evicted_full_tokens", full_tokens)
        _state._counter_add("prefix_evicted_swa_tokens", swa_tokens)

    def _evict_leases(
        self, leases: Sequence[EnginePrefixId]
    ) -> tuple[int, int]:
        values = tuple(leases)
        outputs, full_tokens, swa_tokens = self._evict_native(values)
        self._commit_evictions(
            outputs, values, full_tokens, swa_tokens
        )
        return full_tokens, swa_tokens

    def _materialize_prefix_pages(
        self, pages: Sequence[Any], boundary: int, resident_count: int
    ) -> Any:
        return _materialize_prefix_pages_helper(
            self, pages, boundary, resident_count
        )

    def _materialize_prefix(
        self, materialized: EngineMaterializedRequest, boundary: int
    ) -> Any:
        return _materialize_prefix_helper(self, materialized, boundary)

    def _adopt_published_node(
        self,
        req: Any,
        node: _PrefixNode,
        boundary: int,
        expected_old: _PrefixNode | None,
    ) -> None:
        _adopt_published_node_helper(self, req, node, boundary, expected_old)

    def _touch(self, node: _PrefixNode) -> None:
        self._clock += 1
        node.last_access = self._clock

    @staticmethod
    def _full_edge_tokens(node: _PrefixNode) -> int:
        assert node.parent is not None
        return node.boundary - node.parent.boundary

    def _prune_structural_leaf(self, node: _PrefixNode) -> None:
        current = node
        while (
            current is not self.root_node
            and current.prefix is None
            and not current.children
            and current.lock_ref == 0
        ):
            parent = current.parent
            assert parent is not None
            if parent.children.get(current.digest) is not current:
                _runtime().fail_stop("semantic radix prune identity changed")
                raise FailStopped(
                    _runtime().failure_reason or "semantic radix prune failed"
                )
            if current.swa_ref_count != 0:
                _runtime().fail_stop("semantic radix pruned resident SWA state")
                raise FailStopped(
                    _runtime().failure_reason or "semantic radix prune failed"
                )
            edge_tokens = self._full_edge_tokens(current)
            parent.children.pop(current.digest)
            self._nodes.pop((current.boundary, current.digest))
            self._full_total_tokens -= edge_tokens
            self._full_evictable_tokens -= edge_tokens
            current.evicted = True
            current.parent = None
            current = parent

    def _require_node(self, node: Any, *, allow_root: bool = False) -> None:
        if allow_root and node is self.root_node:
            return
        if (
            not isinstance(node, _PrefixNode)
            or node.evicted
            or node.prefix is None
            or self._nodes.get((node.boundary, node.digest)) is not node
        ):
            raise RuntimeError("foreign or evicted OrbitKV prefix node")

    @staticmethod
    def _match_result(indices: Any, node: _PrefixNode, result_type: Any) -> Any:
        return result_type(
            device_indices=indices,
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            cache_protected_len=len(indices),
            full_kv_hit_length=len(indices),
        )

    @staticmethod
    def _finish_match(result: Any, hit: bool) -> Any:
        _state._counter_add("prefix_matches")
        if hit:
            _state._counter_add("prefix_hits")
        return result
def _build_prefix_cache(context: Any) -> OrbitKvPrefixCache:
    no_prefix = _state._requires_disabled_radix_cache()
    if bool(context.disable_radix_cache) != no_prefix:
        required = "true" if no_prefix else "false"
        raise RuntimeError(
            f"--disable-radix-cache must be {required} for this OrbitKV plan"
        )
    if bool(context.is_hybrid_ssm) != bool(_config().fixed_states):
        raise RuntimeError(
            "SGLang hybrid-state storage differs from the attention-state plan"
        )
    if bool(context.enable_hierarchical_cache):
        raise RuntimeError("OrbitKV does not support hierarchical cache")
    cache = OrbitKvPrefixCache(context.params)
    _session_cache.bind_session_cache(cache)
    return cache


__all__ = ["OrbitKvPrefixCache"]
