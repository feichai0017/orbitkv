from __future__ import annotations

import hashlib
import heapq
import time
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

from ..runtime import (
    FailStopped,
    ManagerError,
    MaterializedRequestView,
    PrefixLease,
    PrefixSemanticKey,
    PublishedPrefix,
    RetryableConflict,
    sglang_page_id,
)
from . import state as _state
from .state import _config, _request_key, _runtime


_PREFIX_NAMESPACE_VERSION = b"orbitkv-sglang-prefix-v1\x00"


@dataclass(eq=False, slots=True)
class _PrefixNode:
    """Non-authoritative radix/LRU metadata; KV pages stay manager-owned."""

    boundary: int
    edge: tuple[int, ...]
    digest: bytes
    prefix: PrefixLease | None
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


@dataclass(frozen=True, slots=True)
class _EvictionPlanItem:
    prefix: PrefixLease
    full_tokens: int
    swa_tokens: int


def _tokens_from_radix_key(key: Any, page_size: int) -> tuple[int, ...]:
    if bool(getattr(key, "is_bigram", False)):
        raise RuntimeError("OrbitKV does not support EAGLE/bigram prefix keys")
    if getattr(key, "extra_key", None) is not None:
        raise RuntimeError("OrbitKV does not support LoRA or namespaced prefix keys")
    try:
        values = tuple(key)
    except Exception as error:
        raise RuntimeError("SGLang prefix key is not readable") from error
    result: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) < 2**63
        ):
            raise RuntimeError("SGLang prefix tokens must be nonnegative int64 values")
        result.append(int(value))
    aligned = len(result) // page_size * page_size
    return tuple(result[:aligned])


def _request_tokens(req: Any, boundary: int) -> tuple[int, ...]:
    if getattr(req, "extra_key", None) is not None:
        raise RuntimeError("OrbitKV does not support LoRA or namespaced requests")
    try:
        values = tuple(req.origin_input_ids) + tuple(req.output_ids)
    except Exception as error:
        raise RuntimeError("SGLang request token history is not readable") from error
    if boundary > len(values):
        raise RuntimeError("request KV boundary exceeds its token history")
    result: list[int] = []
    for value in values[:boundary]:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) < 2**63
        ):
            raise RuntimeError("SGLang request tokens must be nonnegative int64 values")
        result.append(int(value))
    return tuple(result)


class OrbitKvPrefixCache(BasePrefixCache):
    """SGLang tree seam backed only by ABI6 manager prefix leases."""

    def __init__(self, params: Any) -> None:
        import torch

        config = _config()
        if bool(params.disable):
            raise RuntimeError("OrbitKV radix backend cannot be disabled")
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
        if config.full_class is None:
            raise RuntimeError("OrbitKV prefix cache requires a Full KV class")
        if params.token_to_kv_pool_allocator is not _state._ALLOCATOR:
            raise RuntimeError("OrbitKV prefix cache received a foreign KV allocator")
        self.disable = False
        self.disable_finished_insert = bool(params.disable_finished_insert)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = config.page_tokens
        sliding = config.sliding_class
        expected_window = None if sliding is None else sliding.window_tokens
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
                        after, arena_stats = runtime.census()
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
                    # the journal so CanonicalRuntime.close can still destroy
                    # the native handle deterministically.
                    runtime.fail_stop(
                        "shutdown encountered live OrbitKV prefix ownership"
                    )
                    raise
        finally:
            try:
                runtime.close()
            finally:
                self._released = True

    def reset(self) -> None:
        if any(not node.evicted and node.lock_ref for node in self._nodes.values()):
            raise RuntimeError("cannot reset OrbitKV while prefixes are protected")
        leases = self._eviction_plan(0, 0, evict_all=True)
        if leases:
            self._evict_leases(leases)
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

    def swa_reprefill_tail_tokens(self) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

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
        req = params.req
        if not candidates or req is None:
            return self._finish_match(
                self._match_result(self._empty, self.root_node, MatchResult), False
            )
        semantic, node = candidates[-1]
        assert node is not None
        key = _request_key(req)
        runtime = _runtime()
        current = None
        if runtime.has_request(key):
            record = runtime.record_for(key)
            current = getattr(req, "_orbitkv_prefix_node", None)
            if current is not None:
                current_semantic = {
                    item.boundary: item for item in self._semantic_endpoints(tokens)
                }.get(record.boundary)
                if (
                    getattr(req, "_orbitkv_request_key", None) != key
                    or getattr(req, "_orbitkv_request_lease", None) != record.lease
                    or getattr(req, "_orbitkv_prefix_semantic", None)
                    != current_semantic
                ):
                    raise RuntimeError("attached request identity changed")
                self._require_node(current)
                if (
                    current_semantic is None
                    or record.boundary != current.boundary
                    or current_semantic.digest != current.digest
                ):
                    raise RuntimeError("request prefix identity changed after attach")
                indices = getattr(req, "prefix_indices", None)
                if indices is None or len(indices) != record.boundary:
                    raise RuntimeError("attached request lost its prefix mirror")
                provisional = getattr(
                    req, "_orbitkv_provisional_prefix_lock", False
                )
                held = getattr(req, "_orbitkv_prefix_lock_held", False)
                if (
                    type(provisional) is not bool
                    or type(held) is not bool
                    or provisional == held
                ):
                    raise RuntimeError("attached request prefix-lock state changed")
                if self._preflight_release_node(
                    req, provisional=provisional
                ) is not current:
                    raise RuntimeError("attached request prefix node changed")
            elif record.boundary != 0:
                raise RuntimeError("nonempty request has no attached prefix node")
        hints = runtime.prefix_lookup_batch((semantic,))
        if len(hints) != 1:
            runtime.fail_stop("prefix lookup cardinality changed")
            raise FailStopped(runtime.failure_reason or "invalid prefix lookup")
        hint = hints[0]
        if (
            hint.key != semantic
            or hint.resident_count < 0
            or hint.candidate != node.prefix
        ):
            runtime.fail_stop("manager prefix generation differs from local radix")
            raise FailStopped(runtime.failure_reason or "stale prefix generation")
        if current is not None:
            self._touch(current)
            return self._finish_match(
                self._match_result(indices, current, MatchResult), True
            )
        if not runtime.has_request(key):
            views = runtime.request_acquire_batch((key,))
            req._orbitkv_request_key = key
            req._orbitkv_request_lease = views[0].request
        attached_committed = False
        try:
            attached = runtime.prefix_attach_batch(((key, hint),))
            attached_committed = True
            if len(attached) != 1 or attached[0].prefix != node.prefix:
                raise RuntimeError("manager prefix attach identity changed")
            indices = self._materialize_prefix(attached[0].target, semantic.boundary)
        except Exception as error:
            if attached_committed:
                runtime.fail_stop(f"attached prefix materialization failed: {error}")
                raise FailStopped(
                    runtime.failure_reason or "attached prefix materialization failed"
                ) from error
            if runtime.failure_reason is None and runtime.has_request(key):
                runtime.release_batch((key,))
                self._clear_request_identity(req)
            raise
        try:
            req._orbitkv_prefix_node = node
            req._orbitkv_prefix_semantic = semantic
            req._orbitkv_request_key = key
            req._orbitkv_request_lease = runtime.record_for(key).lease
            self.inc_lock_ref(node)
            req._orbitkv_provisional_prefix_lock = True
            self._touch(node)
        except Exception as error:
            runtime.fail_stop(f"attached prefix installation failed: {error}")
            raise FailStopped(
                runtime.failure_reason or "attached prefix installation failed"
            ) from error
        return self._finish_match(
            self._match_result(indices, node, MatchResult), True
        )

    def cache_unfinished_req(self, req: Any, **kwargs: Any) -> None:
        del kwargs
        runtime = _runtime()
        key = _request_key(req)
        initial_record = runtime.record_for(key)
        old_node = self._preflight_unfinished_node(
            req, key, initial_record.lease
        )
        runtime.wait_batch((key,))
        record = runtime.record_for(key)
        if record.lease != initial_record.lease:
            raise RuntimeError("unfinished request lease changed while waiting")
        boundary = record.boundary
        if boundary == 0 or boundary % self.page_size:
            return
        tokens = _request_tokens(req, boundary)
        semantic = self._semantic(tokens)
        node = self._nodes.get((boundary, semantic.digest))
        if (
            self._preflight_unfinished_node(req, key, record.lease)
            is not old_node
        ):
            raise RuntimeError("unfinished request prefix changed before publish")
        raw_row = getattr(req, "req_pool_idx", None)
        if (
            isinstance(raw_row, bool)
            or not isinstance(raw_row, Integral)
            or int(raw_row) <= 0
        ):
            raise RuntimeError("unfinished request has an invalid ReqToToken row")
        runtime.bind_request_rows(((key, int(raw_row), False),))
        if node is None or node.prefix is None:
            publications = runtime.prefix_publish_batch(((key, semantic),))
            if len(publications) != 1:
                runtime.fail_stop("prefix publish cardinality changed")
                raise FailStopped(runtime.failure_reason or "invalid prefix publish")
            try:
                node = self._record_publication(publications[0], tokens)
            except Exception as error:
                if runtime.failure_reason is None:
                    runtime.fail_stop(
                        f"published prefix installation failed: {error}"
                    )
                if isinstance(error, FailStopped):
                    raise
                raise FailStopped(
                    runtime.failure_reason or "prefix installation failed"
                ) from error
        try:
            self._adopt_published_node(req, node, boundary, old_node)
        except Exception as error:
            runtime.fail_stop(f"unfinished prefix adoption failed: {error}")
            raise FailStopped(
                runtime.failure_reason or "unfinished prefix adoption failed"
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
        if not is_insert or self.disable_finished_insert:
            return None
        record = _runtime().record_for(_request_key(req))
        boundary = record.boundary
        if boundary == 0 or boundary % self.page_size:
            return None
        tokens = _request_tokens(req, boundary)
        semantic = self._semantic(tokens)
        existing = self._nodes.get((boundary, semantic.digest))
        if existing is not None and existing.prefix is not None:
            return None
        return _ReleasePublication(semantic, tokens)

    def accept_release_publication(
        self, publication: PublishedPrefix, tokens: Sequence[int]
    ) -> _PrefixNode:
        return self._record_publication(publication, tuple(tokens))

    def _preflight_release_node(
        self, req: Any, *, provisional: bool
    ) -> _PrefixNode | None:
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

    def _preflight_unfinished_node(
        self, req: Any, key: Any, lease: Any
    ) -> _PrefixNode | None:
        if (
            getattr(req, "_orbitkv_request_key", None) != key
            or getattr(req, "_orbitkv_request_lease", None) != lease
        ):
            raise RuntimeError("unfinished request identity changed")
        node = self._preflight_release_node(req, provisional=False)
        semantic = getattr(req, "_orbitkv_prefix_semantic", None)
        if node is None:
            if semantic is not None:
                raise RuntimeError("unlocked unfinished request retained a prefix")
        elif semantic != PrefixSemanticKey(
            self._namespace, node.digest, node.boundary
        ):
            raise RuntimeError("unfinished request prefix semantic changed")
        return node

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
                while cursor < len(plan):
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

    def _semantic_endpoints(
        self, tokens: tuple[int, ...]
    ) -> tuple[PrefixSemanticKey, ...]:
        hasher = hashlib.sha256()
        result = []
        for index, token in enumerate(tokens, start=1):
            hasher.update(token.to_bytes(8, "little", signed=False))
            if index % self.page_size == 0:
                result.append(
                    PrefixSemanticKey(self._namespace, hasher.digest(), index)
                )
        return tuple(result)

    def _semantic(self, tokens: tuple[int, ...]) -> PrefixSemanticKey:
        if not tokens or len(tokens) % self.page_size:
            raise RuntimeError("prefix publication endpoint is not page aligned")
        return self._semantic_endpoints(tokens)[-1]

    def _record_publication(
        self, publication: PublishedPrefix, tokens: tuple[int, ...]
    ) -> _PrefixNode:
        endpoints = self._semantic_endpoints(tokens)
        semantic = endpoints[-1]
        if (
            publication.key != semantic
            or publication.prefix is None
            or publication.resident_count <= 0
        ):
            _runtime().fail_stop("manager returned an invalid published prefix")
            raise FailStopped(_runtime().failure_reason or "invalid prefix publication")
        full_pages = semantic.boundary // self.page_size
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
        map_key = (semantic.boundary, semantic.digest)
        parent = self.root_node
        for endpoint in endpoints:
            endpoint_key = (endpoint.boundary, endpoint.digest)
            edge = tokens[endpoint.boundary - self.page_size : endpoint.boundary]
            node = self._nodes.get(endpoint_key)
            if node is None:
                node = _PrefixNode(
                    endpoint.boundary,
                    edge,
                    endpoint.digest,
                    None,
                    0,
                    0,
                    parent,
                )
                self._nodes[endpoint_key] = node
                parent.children[endpoint.digest] = node
                edge_tokens = self._full_edge_tokens(node)
                self._full_total_tokens += edge_tokens
                self._full_evictable_tokens += edge_tokens
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
        node = self._nodes[map_key]
        if node.prefix is not None:
            _runtime().fail_stop("manager published a duplicate semantic prefix")
            raise FailStopped(_runtime().failure_reason or "duplicate prefix")
        node.prefix = publication.prefix
        node.resident_count = publication.resident_count
        current = node
        for _ in range(swa_pages):
            if current.swa_ref_count == 0:
                self._swa_total_tokens += self.page_size
                if current.lock_ref == 0:
                    self._swa_evictable_tokens += self.page_size
                else:
                    self._swa_protected_tokens += self.page_size
            current.swa_ref_count += 1
            if current.parent is None:
                _runtime().fail_stop("prefix SWA residency exceeds its radix path")
                raise FailStopped(
                    _runtime().failure_reason or "invalid prefix SWA residency"
                )
            current = current.parent
        self._touch(node)
        _state._counter_add("prefix_publishes")
        return node

    def _accept_evictions(
        self,
        outputs: Sequence[Any],
        requested: Sequence[PrefixLease],
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
        self, outputs: Sequence[Any], requested: Sequence[PrefixLease]
    ) -> tuple[tuple[Any, ...], list[_PrefixNode]]:
        values = tuple(outputs)
        expected = tuple(requested)
        if len(values) != len(expected) or tuple(
            item.prefix for item in values
        ) != expected:
            _runtime().fail_stop("prefix eviction identity changed")
            raise FailStopped(_runtime().failure_reason or "invalid prefix eviction")
        selected = []
        for item in values:
            node = self._nodes.get((item.key.boundary, item.key.digest))
            if (
                node is None
                or node.prefix != item.prefix
                or node.lock_ref != 0
                or node.parent is None
                or node.parent.children.get(node.digest) is not node
            ):
                _runtime().fail_stop("prefix eviction returned a foreign or locked node")
                raise FailStopped(_runtime().failure_reason or "invalid prefix eviction")
            selected.append(node)
        return values, selected

    def _evict_native(
        self, leases: Sequence[PrefixLease]
    ) -> tuple[tuple[Any, ...], int, int]:
        runtime = _runtime()
        values = tuple(leases)
        output = runtime.prefix_evict_batch(values)
        try:
            runtime.prefix_recycle_batch(values)
        except RetryableConflict:
            # Eviction, mirror cleanup, and reclamation ACK have already
            # committed.  One exact retry closes the only precommit operation
            # that remains; a second refusal cannot be exposed as if the
            # overall cache eviction were still uncommitted.
            try:
                runtime.prefix_recycle_batch(values)
            except (RetryableConflict, ManagerError) as error:
                runtime.fail_stop("prefix slot recycling remained unavailable")
                raise FailStopped(
                    runtime.failure_reason or "prefix recycling failed"
                ) from error
        except ManagerError as error:
            runtime.fail_stop("prefix slot recycling rejected committed evictions")
            raise FailStopped(runtime.failure_reason or "prefix recycling failed") from error
        evicted, _selected = self._validate_eviction_outputs(
            output.evicted, values
        )
        full = _config().full_class
        sliding = _config().sliding_class
        full_tokens = self.page_size * sum(
            item.class_id == full.class_id for item in output.retirements
        )
        swa_tokens = self.page_size * sum(
            item.class_id == sliding.class_id for item in output.retirements
        ) if sliding is not None else 0
        return evicted, full_tokens, swa_tokens

    def _commit_evictions(
        self,
        outputs: Sequence[Any],
        leases: Sequence[PrefixLease],
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
        self, leases: Sequence[PrefixLease]
    ) -> tuple[int, int]:
        values = tuple(leases)
        outputs, full_tokens, swa_tokens = self._evict_native(values)
        self._commit_evictions(
            outputs, values, full_tokens, swa_tokens
        )
        return full_tokens, swa_tokens

    def _published_parent_map(self) -> dict[_PrefixNode, _PrefixNode | None]:
        result: dict[_PrefixNode, _PrefixNode | None] = {}
        stack = [(self.root_node, None)]
        while stack:
            node, published_parent = stack.pop()
            next_parent = published_parent
            if node is not self.root_node and node.prefix is not None:
                result[node] = published_parent
                next_parent = node
            stack.extend(
                (child, next_parent) for child in node.children.values()
            )
        return result

    def _eviction_plan(
        self, requested_full: int, requested_swa: int, *, evict_all: bool = False
    ) -> tuple[PrefixLease, ...]:
        plan = self._complete_eviction_plan()
        if evict_all:
            return tuple(item.prefix for item in plan)
        if requested_full <= 0 and requested_swa <= 0:
            return ()
        full_estimate = swa_estimate = 0
        selected = []
        for item in plan:
            selected.append(item.prefix)
            full_estimate += item.full_tokens
            swa_estimate += item.swa_tokens
            if full_estimate >= requested_full and swa_estimate >= requested_swa:
                break
        return tuple(selected)

    def _complete_eviction_plan(
        self, *, _work: dict[str, int] | None = None
    ) -> tuple[_EvictionPlanItem, ...]:
        published_parents = self._published_parent_map()
        child_counts = {node: 0 for node in published_parents}
        for parent in published_parents.values():
            if parent is not None:
                child_counts[parent] += 1
        own_prefix_live = {
            node: node.prefix is not None for node in self._nodes.values()
        }
        remaining_children = {
            node: len(node.children) for node in self._nodes.values()
        }
        heap: list[tuple[int, int, _PrefixNode]] = []
        for node, count in child_counts.items():
            if count == 0 and node.lock_ref == 0:
                heapq.heappush(heap, (node.last_access, id(node), node))
        selected_nodes: list[_PrefixNode] = []
        selected_prefixes: list[PrefixLease] = []
        full_estimates: list[int] = []
        while heap:
            _clock, _identity, node = heapq.heappop(heap)
            assert node.prefix is not None
            full_estimate = 0
            parent = published_parents[node]
            if not own_prefix_live[node]:
                raise RuntimeError("semantic radix prefix was selected twice")
            own_prefix_live[node] = False
            current = node
            while (
                current is not self.root_node
                and not own_prefix_live[current]
                and remaining_children[current] == 0
            ):
                full_estimate += self._full_edge_tokens(current)
                structural_parent = current.parent
                assert structural_parent is not None
                if structural_parent is self.root_node:
                    break
                remaining_children[structural_parent] -= 1
                if remaining_children[structural_parent] < 0:
                    raise RuntimeError("semantic radix eviction census underflowed")
                current = structural_parent
            if parent is not None:
                child_counts[parent] -= 1
                if child_counts[parent] == 0 and parent.lock_ref == 0:
                    heapq.heappush(
                        heap, (parent.last_access, id(parent), parent)
                    )
            selected_nodes.append(node)
            selected_prefixes.append(node.prefix)
            full_estimates.append(full_estimate)
        swa_estimates = self._swa_eviction_estimates(selected_nodes, _work)
        return tuple(
            _EvictionPlanItem(prefix, full_tokens, swa_tokens)
            for prefix, full_tokens, swa_tokens in zip(
                selected_prefixes, full_estimates, swa_estimates, strict=True
            )
        )

    def _swa_eviction_estimates(
        self,
        selected: Sequence[_PrefixNode],
        work: dict[str, int] | None,
    ) -> tuple[int, ...]:
        """Assign each last-selected SWA page to one eviction-plan item."""

        preorder, top_exclusive, coverage = self._swa_selected_coverage(
            selected, work
        )

        freeable = {
            node
            for node in preorder[1:]
            if coverage[node] > 0 and coverage[node] == node.swa_ref_count
        }
        next_free: dict[_PrefixNode, _PrefixNode | None] = {
            self.root_node: None
        }

        def find(node: _PrefixNode | None) -> _PrefixNode | None:
            trail = []
            current = node
            while current is not None and next_free[current] is not current:
                trail.append(current)
                current = next_free[current]
                if work is not None:
                    work["swa_path_steps"] += 1
            for item in trail:
                next_free[item] = current
            return current

        for node in preorder[1:]:
            parent = node.parent
            assert parent is not None
            next_free[node] = node if node in freeable else find(parent)
            if work is not None:
                work["swa_path_steps"] += 1

        # Later plan items are the last decrements.  Removing each freeable
        # structural node from this parent-chain DSU assigns it exactly once;
        # locked or otherwise unselected references were excluded above.
        estimates = [0] * len(selected)
        assigned = 0
        for index in range(len(selected) - 1, -1, -1):
            node = selected[index]
            boundary = top_exclusive[node].boundary
            current = find(node)
            while current is not None and current.boundary > boundary:
                estimates[index] += self.page_size
                assigned += 1
                parent = current.parent
                assert parent is not None
                next_free[current] = find(parent)
                current = find(current)
                if work is not None:
                    work["swa_path_steps"] += 1
        if assigned != len(freeable):
            raise RuntimeError("semantic SWA eviction ownership was incomplete")
        return tuple(estimates)

    def _swa_selected_coverage(
        self,
        selected: Sequence[_PrefixNode],
        work: dict[str, int] | None,
    ) -> tuple[
        list[_PrefixNode],
        dict[_PrefixNode, _PrefixNode],
        dict[_PrefixNode, int],
    ]:
        """Return exact selected-window coverage with one structural-tree pass."""

        if work is not None:
            work["swa_path_steps"] = 0

        selected_windows = {
            node: node.resident_count - node.boundary // self.page_size
            for node in selected
        }
        preorder: list[_PrefixNode] = []
        top_exclusive: dict[_PrefixNode, _PrefixNode] = {}
        path: list[_PrefixNode] = []
        stack = [(self.root_node, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                path.pop()
                continue
            path.append(node)
            preorder.append(node)
            if work is not None:
                work["swa_path_steps"] += 1
            if node in selected_windows:
                pages = selected_windows[node]
                depth = len(path) - 1
                if pages < 0 or pages > depth:
                    raise RuntimeError("semantic SWA eviction census underflowed")
                top_exclusive[node] = path[depth - pages]
            stack.append((node, True))
            stack.extend(
                (child, False) for child in reversed(tuple(node.children.values()))
            )
        if len(preorder) != len(self._nodes) + 1 or len(top_exclusive) != len(
            selected
        ):
            raise RuntimeError("semantic radix topology changed during eviction")

        # Tree-path difference computes how many selected prefix windows cover
        # each structural page without walking any individual window.
        coverage = {node: 0 for node in preorder}
        for node in selected:
            coverage[node] += 1
            coverage[top_exclusive[node]] -= 1
            if work is not None:
                work["swa_path_steps"] += 1
        for node in reversed(preorder[1:]):
            parent = node.parent
            assert parent is not None
            coverage[parent] += coverage[node]
            if work is not None:
                work["swa_path_steps"] += 1
        if any(
            count < 0 or count > node.swa_ref_count
            for node, count in coverage.items()
            if node is not self.root_node
        ):
            raise RuntimeError("semantic SWA eviction census underflowed")
        return preorder, top_exclusive, coverage

    def _materialize_prefix(
        self, materialized: MaterializedRequestView, boundary: int
    ) -> Any:
        import torch

        if materialized.view.boundary != boundary or boundary % self.page_size:
            raise RuntimeError("manager attached a non-page-aligned endpoint")
        config = _config()
        full = config.full_class
        if full is None:
            raise RuntimeError("OrbitKV prefix cache requires a Full KV class")
        pages_by_class: dict[int, list[Any]] = {
            item.class_id: [] for item in config.classes
        }
        for page in materialized.pages:
            try:
                pages_by_class[page.class_id].append(page)
            except KeyError as error:
                raise RuntimeError("manager materialized an unknown KV class") from error
        full_pages = pages_by_class[full.class_id]
        expected_pages = boundary // self.page_size
        if len(full_pages) != expected_pages:
            raise RuntimeError("manager did not materialize the complete Full prefix")
        primary_parts = []
        full_by_ordinal: dict[int, Any] = {}
        full_arena = _runtime().arenas_by_class[full.class_id]
        for ordinal, page in enumerate(full_pages):
            if (
                page.logical_ordinal != ordinal
                or page.valid_token_count != self.page_size
                or page.visible_token_offset != 0
                or page.visible_token_count != self.page_size
            ):
                raise RuntimeError("manager Full prefix endpoint is not page exact")
            start = (
                sglang_page_id(page.backend_index, full_arena.backend_base_index)
                * self.page_size
            )
            primary_parts.append(
                torch.arange(
                    start,
                    start + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
            full_by_ordinal[ordinal] = page
        primary = torch.cat(primary_parts) if primary_parts else self._empty
        sliding = config.sliding_class
        if sliding is not None:
            sliding_arena = _runtime().arenas_by_class[sliding.class_id]
            full_locations = []
            swa_locations = []
            seen_ordinals: set[int] = set()
            for page in pages_by_class[sliding.class_id]:
                if (
                    page.logical_ordinal in seen_ordinals
                    or page.logical_ordinal not in full_by_ordinal
                    or not 0 <= page.visible_token_offset < self.page_size
                    or page.visible_token_count <= 0
                    or page.visible_token_offset + page.visible_token_count
                    > page.valid_token_count
                    or page.valid_token_count > self.page_size
                ):
                    raise RuntimeError("manager Hybrid prefix visibility is invalid")
                seen_ordinals.add(page.logical_ordinal)
                logical_start = page.logical_ordinal * self.page_size
                visible_start = logical_start + page.visible_token_offset
                full_locations.append(
                    primary[
                        visible_start : visible_start + page.visible_token_count
                    ]
                )
                swa_start = (
                    sglang_page_id(
                        page.backend_index, sliding_arena.backend_base_index
                    )
                    * self.page_size
                    + page.visible_token_offset
                )
                swa_locations.append(
                    torch.arange(
                        swa_start,
                        swa_start + page.visible_token_count,
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
            if full_locations:
                full_vector = torch.cat(full_locations)
                expected_swa = torch.cat(swa_locations)
                mapping = _state._ALLOCATOR.full_to_swa_index_mapping
                if not torch.equal(
                    mapping[full_vector].to(dtype=torch.int64), expected_swa
                ):
                    raise RuntimeError(
                        "manager prefix disagrees with the Full-to-SWA mirror"
                    )
        return primary

    def _adopt_published_node(
        self,
        req: Any,
        node: _PrefixNode,
        boundary: int,
        expected_old: _PrefixNode | None,
    ) -> None:
        import torch

        old = self._preflight_release_node(req, provisional=False)
        if old is not expected_old:
            raise RuntimeError("unfinished prefix changed after publication")
        if old is not None and old is not self.root_node and old is not node:
            self.dec_lock_ref(old)
        if old is not node:
            self.inc_lock_ref(node)
        row = int(req.req_pool_idx)
        indices = self.req_to_token_pool.req_to_token[row, :boundary].to(
            dtype=torch.int64, copy=True
        )
        req.prefix_indices = indices
        req.cache_protected_len = boundary
        req.last_node = node
        req.last_host_node = node
        req.best_match_node = node
        req._orbitkv_prefix_node = node
        req._orbitkv_prefix_semantic = PrefixSemanticKey(
            self._namespace, node.digest, node.boundary
        )
        req._orbitkv_prefix_lock_held = True

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

    @staticmethod
    def _clear_request_identity(req: Any) -> None:
        for name in (
            "_orbitkv_request_key",
            "_orbitkv_request_lease",
            "_orbitkv_prefix_node",
            "_orbitkv_prefix_semantic",
            "_orbitkv_provisional_prefix_lock",
            "_orbitkv_prefix_lock_held",
        ):
            if hasattr(req, name):
                delattr(req, name)


def _build_prefix_cache(context: Any) -> OrbitKvPrefixCache:
    if bool(context.disable_radix_cache):
        raise RuntimeError("--disable-radix-cache must be false for OrbitKV")
    if bool(context.is_hybrid_ssm):
        raise RuntimeError("OrbitKV does not support hybrid SSM/Mamba caches")
    if bool(context.enable_hierarchical_cache):
        raise RuntimeError("OrbitKV does not support hierarchical cache")
    return OrbitKvPrefixCache(context.params)


__all__ = ["OrbitKvPrefixCache"]
