from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, NoReturn, Sequence

from .ffi.session_types import (
    EnginePrefixId,
    EnginePrefixPublishItem,
    EnginePrefixPublishReleasePlan,
    EnginePublishedPrefix,
    EnginePublishedPrefixRelease,
    EngineReleaseId,
    EngineReleasePlan,
    EngineReleasedRequest,
    EngineRequestView,
)
from .runtime import (
    DETACHED_CLEAR,
    DETACHED_PREFIX_TRANSFER,
    DetachedBinding,
    ManagerError,
    PageLease,
    PrefixSemanticKey,
)


_UINT64_LIMIT = 1 << 64


@dataclass(slots=True)
class _ReleaseGroup:
    plan: EngineReleasePlan
    keys: tuple[Hashable, ...]
    publish_release: EnginePrefixPublishReleasePlan | None = None
    cleanup_confirmed: bool = False
    ack_committed: bool = False


class SessionReleaseRuntimeMixin:
    """Atomic prefix publication and request-release coordination."""

    _arenas_by_class: dict[int, Any]
    _bindings: dict[Hashable, Any]
    _controls: dict[Any, Any]
    _engine_epoch: int
    _lock: Any
    _page_tokens: int
    _prefix_occupancy: dict[EnginePrefixId, Any]
    _prefixes: dict[PrefixSemanticKey, EnginePublishedPrefix]
    _releases: dict[EngineReleaseId, _ReleaseGroup]
    _request_occupancy: dict[Hashable, Any]
    _request_keys: dict[Any, Hashable]
    _quarantined_requests: set[Hashable]
    _row_owners: dict[int, Hashable]
    _session: Any
    _views: dict[Hashable, EngineRequestView]

    def _healthy(self) -> None: ...
    def _binding(self, key: Hashable) -> Any: ...
    def _bindings_for_keys(
        self, keys: Sequence[Hashable], operation: str
    ) -> tuple[Any, ...]: ...
    def wait_requests(self, keys: Sequence[Hashable]) -> None: ...
    def _fail_stop(
        self, reason: str, error: BaseException | None = None
    ) -> NoReturn: ...
    def _require_shared_cache_policy(self, operation: str) -> None: ...

    def prepare_prefix_publish_release(
        self, items: Sequence[tuple[Hashable, PrefixSemanticKey]]
    ) -> EnginePrefixPublishReleasePlan:
        """Atomically publish prefixes and begin their request release.

        The returned object describes the newly resident prefixes. The
        ordinary ``pending_release`` lookup returns the matching
        ``EngineReleasePlan`` that must be passed to ``confirm_release``.
        """

        with self._lock:
            self._require_shared_cache_policy(
                "session prefix publish-release"
            )
            self._healthy()
            values = tuple(items)
            if not values:
                raise ManagerError(
                    "session prefix publish-release batch must be nonempty"
                )
            keys: list[Hashable] = []
            semantics: list[PrefixSemanticKey] = []
            bindings = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ManagerError(
                        "session prefix publish-release items must be "
                        "(key, semantic) pairs"
                    )
                key, semantic = item
                binding = self._binding(key)
                self._require_control_free(key)
                self._validate_prefix_semantic_key(semantic)
                keys.append(key)
                semantics.append(semantic)
                bindings.append(binding)
            if len(set(keys)) != len(keys):
                raise ManagerError(
                    "session prefix publish-release contains duplicate request keys"
                )
            if len(set(semantics)) != len(semantics):
                raise ManagerError(
                    "session prefix publish-release contains duplicate semantic keys"
                )
            if not self._prefix_registry_valid():
                self._fail_stop(
                    "session prefix registry is inconsistent before publication"
                )
            if any(semantic in self._prefixes for semantic in semantics):
                raise ManagerError(
                    "session prefix publish-release semantic is already published"
                )

            ordered_keys = tuple(keys)
            self.wait_requests(ordered_keys)
            current_bindings = self._bindings_for_keys(
                ordered_keys, "session prefix publish-release"
            )
            if any(
                before is not current
                for before, current in zip(
                    bindings, current_bindings, strict=True
                )
            ):
                self._fail_stop(
                    "session prefix publish-release binding changed while waiting"
                )
            views = []
            for binding, semantic in zip(
                current_bindings, semantics, strict=True
            ):
                self._require_control_free(binding.key)
                view = self._views.get(binding.key)
                if (
                    type(view) is not EngineRequestView
                    or view.request_id != binding.request_id
                    or type(view.view_version) is not int
                    or not 0 <= view.view_version < _UINT64_LIMIT
                    or type(view.boundary) is not int
                    or not 0 <= view.boundary < _UINT64_LIMIT
                    or type(view.resident_count) is not int
                    or not 0 <= view.resident_count < 1 << 32
                ):
                    self._fail_stop(
                        "session prefix publish-release view mirror is invalid"
                    )
                if semantic.boundary != view.boundary:
                    raise ManagerError(
                        "prefix semantic boundary differs from the request view"
                    )
                views.append(view)
            if (
                not self._same_live_bindings(
                    ordered_keys, current_bindings, tuple(views)
                )
                or not self._participants_free(ordered_keys)
            ):
                self._fail_stop(
                    "session prefix publish-release participants are inconsistent"
                )
            if any(semantic in self._prefixes for semantic in semantics):
                raise ManagerError(
                    "session prefix publish-release semantic is already published"
                )
            raw = tuple(
                EnginePrefixPublishItem(binding.request_id, semantic)
                for binding, semantic in zip(
                    current_bindings, semantics, strict=True
                )
            )
            try:
                plan = self._session.prefix_publish_release_batch(raw)
            except ManagerError:
                raise
            except BaseException as error:
                self._fail_stop(
                    "native prefix publish-release became uncertain", error
                )

            try:
                if (
                    type(plan) is not EnginePrefixPublishReleasePlan
                    or type(plan.outputs) is not tuple
                    or type(plan.release_id) is not EngineReleaseId
                    or type(plan.release_id.session_epoch) is not int
                    or plan.release_id.session_epoch != self._engine_epoch
                    or type(plan.release_id.sequence) is not int
                    or not 0 < plan.release_id.sequence < _UINT64_LIMIT
                    or len(plan.outputs) != len(values)
                ):
                    raise RuntimeError(
                        "prefix publish-release plan identity or cardinality changed"
                    )
                outputs = plan.outputs
                if plan.release_id in self._releases:
                    raise RuntimeError(
                        "prefix publish-release returned a duplicate release id"
                    )
                known_prefix_ids = {
                    published.prefix_id for published in self._prefixes.values()
                }
                returned_prefix_ids: set[EnginePrefixId] = set()
                releases: list[EngineReleasedRequest] = []
                publications: list[EnginePublishedPrefix] = []
                for binding, semantic, view, output in zip(
                    current_bindings, semantics, views, outputs, strict=True
                ):
                    if type(output) is not EnginePublishedPrefixRelease:
                        raise RuntimeError(
                            "prefix publish-release returned an invalid output"
                        )
                    prefix_id = output.prefix_id
                    if (
                        type(prefix_id) is not EnginePrefixId
                        or type(prefix_id.session_epoch) is not int
                        or prefix_id.session_epoch != self._engine_epoch
                        or type(prefix_id.sequence) is not int
                        or not 0 < prefix_id.sequence < _UINT64_LIMIT
                        or prefix_id in returned_prefix_ids
                        or prefix_id in known_prefix_ids
                        or prefix_id in self._prefix_occupancy
                    ):
                        raise RuntimeError(
                            "prefix publish-release returned an invalid prefix id"
                        )
                    returned_prefix_ids.add(prefix_id)
                    if (
                        type(output.key) is not PrefixSemanticKey
                        or output.key != semantic
                    ):
                        raise RuntimeError(
                            "prefix publish-release changed key ordering"
                        )
                    self._validate_prefix_release_output(
                        binding, view, output
                    )
                    releases.append(output.release)
                    publications.append(
                        EnginePublishedPrefix(
                            prefix_id, output.key, output.resident_count
                        )
                    )

                if not self._same_live_bindings(
                    ordered_keys, current_bindings, tuple(views)
                ) or not self._participants_free(ordered_keys):
                    raise RuntimeError(
                        "prefix publish-release participants changed during commit"
                    )
                if any(semantic in self._prefixes for semantic in semantics):
                    raise RuntimeError(
                        "prefix publish-release semantic became locally occupied"
                    )
                if not self._prefix_registry_valid():
                    raise RuntimeError(
                        "session prefix registry changed during publication"
                    )

                release_plan = EngineReleasePlan(
                    plan.release_id, tuple(releases), ()
                )
                next_releases = dict(self._releases)
                next_prefixes = dict(self._prefixes)
                next_releases[plan.release_id] = _ReleaseGroup(
                    release_plan, ordered_keys, plan
                )
                for semantic, publication in zip(
                    semantics, publications, strict=True
                ):
                    next_prefixes[semantic] = publication
                self._releases = next_releases
                self._prefixes = next_prefixes
            except BaseException as error:
                self._fail_stop(
                    "native prefix publish-release returned malformed "
                    "postcommit output",
                    error,
                )
            return plan

    def _participants_free(self, keys: tuple[Hashable, ...]) -> bool:
        return not any(
            key in self._request_occupancy
            or key in self._quarantined_requests
            or any(
                key in group.request_keys or key in group.target_keys
                for group in self._controls.values()
            )
            or any(key in group.keys for group in self._releases.values())
            for key in keys
        )

    def _prefix_registry_valid(self) -> bool:
        prefix_ids: set[EnginePrefixId] = set()
        for semantic, publication in self._prefixes.items():
            prefix_id = getattr(publication, "prefix_id", None)
            if (
                type(semantic) is not PrefixSemanticKey
                or type(publication) is not EnginePublishedPrefix
                or type(publication.key) is not PrefixSemanticKey
                or publication.key != semantic
                or type(prefix_id) is not EnginePrefixId
                or type(prefix_id.session_epoch) is not int
                or prefix_id.session_epoch != self._engine_epoch
                or type(prefix_id.sequence) is not int
                or not 0 < prefix_id.sequence < _UINT64_LIMIT
                or type(publication.resident_count) is not int
                or not 0 <= publication.resident_count < 1 << 32
                or prefix_id in prefix_ids
            ):
                return False
            prefix_ids.add(prefix_id)
        return True

    def _same_live_bindings(
        self,
        keys: tuple[Hashable, ...],
        bindings: tuple[Any, ...],
        views: tuple[EngineRequestView, ...],
    ) -> bool:
        return all(
            self._bindings.get(key) is binding
            and type(binding.request_id) is int
            and 0 < binding.request_id < _UINT64_LIMIT
            and self._request_keys.get(binding.request_id) == key
            and sum(
                owner == key for owner in self._request_keys.values()
            )
            == 1
            and self._views.get(key) is view
            and (
                binding.request_row is None
                and key not in self._row_owners.values()
                or self._row_owners.get(binding.request_row) == key
            )
            for key, binding, view in zip(
                keys, bindings, views, strict=True
            )
        )

    @staticmethod
    def _validate_prefix_semantic_key(semantic: Any) -> None:
        if type(semantic) is not PrefixSemanticKey:
            raise ManagerError(
                "session prefix publish-release semantic must be PrefixSemanticKey"
            )
        if type(semantic.namespace) is not bytes or len(semantic.namespace) != 32:
            raise ManagerError(
                "prefix namespace must contain exactly 32 bytes"
            )
        if type(semantic.digest) is not bytes or len(semantic.digest) != 32:
            raise ManagerError("prefix digest must contain exactly 32 bytes")
        if (
            type(semantic.boundary) is not int
            or not 0 <= semantic.boundary < _UINT64_LIMIT
        ):
            raise ManagerError("prefix boundary must fit in uint64_t")

    def _validate_prefix_release_output(
        self,
        binding: Any,
        view: EngineRequestView,
        output: EnginePublishedPrefixRelease,
    ) -> None:
        release = output.release
        if (
            type(release) is not EngineReleasedRequest
            or type(release.request_id) is not int
            or not 0 < release.request_id < _UINT64_LIMIT
            or release.request_id != binding.request_id
            or type(release.detached) is not tuple
        ):
            raise RuntimeError(
                "prefix publish-release changed release request identity"
            )
        if (
            type(output.resident_count) is not int
            or not 0 <= output.resident_count < 1 << 32
            or output.resident_count != view.resident_count
            or len(release.detached) != output.resident_count
        ):
            raise RuntimeError("prefix publish-release changed resident count")

        zero_page = PageLease(0, 0, 0, 0, 0)
        previous_location: tuple[int, int] | None = None
        seen_pages: set[PageLease] = set()
        for detached in release.detached:
            if type(detached) is not DetachedBinding:
                raise RuntimeError(
                    "prefix publish-release returned an invalid detached binding"
                )
            integer_fields = (
                detached.logical_ordinal,
                detached.old_backend_index,
                detached.replacement_backend_index,
                detached.token_begin,
                detached.token_end_exclusive,
                detached.class_id,
                detached.backend_domain,
                detached.action,
                detached.reason,
                detached.reserved,
            )
            if any(type(value) is not int for value in integer_fields):
                raise RuntimeError(
                    "prefix publish-release detached identity is malformed"
                )
            arena = self._arenas_by_class.get(detached.class_id)
            old = detached.old
            location = (detached.class_id, detached.logical_ordinal)
            if (
                arena is None
                or type(old) is not PageLease
                or any(
                    type(value) is not int
                    for value in (
                        old.engine_epoch,
                        old.pool_epoch,
                        old.generation,
                        old.page_id,
                        old.pool_id,
                    )
                )
                or old.engine_epoch != arena.engine_epoch
                or old.pool_epoch != arena.pool_epoch
                or old.pool_id != arena.pool_id
                or old.generation <= 0
                or not arena.first_page_id
                <= old.page_id
                < arena.first_page_id + arena.page_count
                or detached.backend_domain != arena.backend_domain
                or detached.old_backend_index
                != arena.backend_base_index
                + old.page_id
                - arena.first_page_id
                or detached.logical_ordinal < 0
                or detached.token_begin
                != detached.logical_ordinal * self._page_tokens
                or detached.token_end_exclusive != min(
                    (detached.logical_ordinal + 1) * self._page_tokens,
                    view.boundary,
                )
                or detached.token_begin >= detached.token_end_exclusive
                or type(detached.replacement) is not PageLease
                or detached.replacement != zero_page
                or detached.replacement_backend_index != 0
                or detached.action != DETACHED_CLEAR
                or detached.reason != DETACHED_PREFIX_TRANSFER
                or detached.reserved != 0
                or previous_location is not None
                and location <= previous_location
                or old in seen_pages
            ):
                raise RuntimeError(
                    "prefix publish-release detached identity changed"
                )
            previous_location = location
            seen_pages.add(old)


__all__ = ["SessionReleaseRuntimeMixin"]
