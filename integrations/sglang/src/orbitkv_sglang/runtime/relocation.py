from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

from .completion import BatchCompletionReceipt
from .identity import FailStopped, ManagerError, TokenRelocationManagerProtocol
from .snapshot_shadow import PageShadow, RequestView, sglang_page_id
from .token_relocation import (
    ClassTokenDispositionUpdate,
    PrepareRelocationItem,
    PreparedRelocation,
    RelocationCopyReceipt,
    RelocationPolicy,
    TokenDispositionBatchItem,
    TokenView,
    TokenViewQuery,
)


@dataclass(frozen=True, slots=True)
class RelocationPublication:
    key: Hashable
    old_view: TokenView
    prepared: PreparedRelocation
    publication: RequestView
    retained_locations: tuple[int, ...]
    retirements: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class DispositionPublication:
    key: Hashable
    old_view: TokenView
    publication: RequestView
    retained_locations: tuple[int, ...]


class RelocationRuntimeMixin:
    def _relocation_manager(self) -> TokenRelocationManagerProtocol:
        if not isinstance(self.manager, TokenRelocationManagerProtocol):
            raise ManagerError("manager does not expose ABI7 token relocation")
        return self.manager

    def token_view(self, key: Hashable, class_id: int) -> TokenView:
        with self._lock:
            self._healthy()
            record = self.record_for(key)
            if record.pending is not None:
                raise ManagerError("cannot query token placement for a busy request")
            values = self._relocation_manager().token_views_batch(
                (
                    TokenViewQuery(
                        record.lease, record.head, int(class_id), record.boundary
                    ),
                )
            )
            if len(values) != 1:
                raise ManagerError("manager returned the wrong token-view cardinality")
            return values[0]

    def active_kv_length(self, key: Hashable, class_id: int) -> int:
        with self._lock:
            record = self.record_for(key)
            return record.cursor.active_kv_lengths.get(
                int(class_id), record.boundary
            )

    def mark_token_dispositions(
        self,
        key: Hashable,
        class_id: int,
        updates: Sequence[ClassTokenDispositionUpdate],
    ) -> DispositionPublication:
        with self._lock:
            self._healthy()
            record = self.record_for(key)
            if record.pending is not None:
                raise ManagerError("cannot change dispositions for a busy request")
            manager = self._relocation_manager()
            old_view = manager.token_views_batch(
                (TokenViewQuery(record.lease, record.head, int(class_id), record.boundary),)
            )[0]
            publication = manager.mark_token_dispositions_batch(
                (TokenDispositionBatchItem(record.lease, record.head, tuple(updates)),)
            )[0]
            self._validate_relocation_view(record, publication)
            marked = manager.token_views_batch(
                (TokenViewQuery(record.lease, publication.snapshot, int(class_id), record.boundary),)
            )[0]
            retained_locations = self._retained_locations(
                marked, int(class_id), require_dead_absent=False
            )
            self._replace_head(record.head, publication.snapshot)
            record.cursor.snapshot = publication.snapshot
            record.cursor.view_version = publication.view_version
            record.cursor.active_kv_lengths[int(class_id)] = len(retained_locations)
            return DispositionPublication(
                key, old_view, publication, retained_locations
            )

    def relocate_tokens(
        self,
        key: Hashable,
        class_id: int,
        updates: Sequence[ClassTokenDispositionUpdate],
        policy: RelocationPolicy,
        copy: Any,
        completion_domain: int,
        completion_value: int,
    ) -> RelocationPublication:
        with self._lock:
            self._healthy()
            record = self.record_for(key)
            if record.pending is not None:
                raise ManagerError("cannot relocate a busy request")
            class_config = self.config.classes_by_id.get(int(class_id))
            if (
                len(self.config.classes) != 1
                or class_config is None
                or class_config.retention != "full"
            ):
                raise ManagerError("first engine relocation profile requires Full-only")
            manager = self._relocation_manager()
            old_view = manager.token_views_batch(
                (TokenViewQuery(record.lease, record.head, int(class_id), record.boundary),)
            )[0]
            marked = manager.mark_token_dispositions_batch(
                (
                    TokenDispositionBatchItem(
                        record.lease, record.head, tuple(updates)
                    ),
                )
            )[0]
            self._validate_relocation_view(record, marked)
            self._replace_head(record.head, marked.snapshot)
            record.cursor.snapshot = marked.snapshot
            record.cursor.view_version = marked.view_version
            prepared: PreparedRelocation | None = None
            try:
                prepared = manager.prepare_relocation_batch(
                    (
                        PrepareRelocationItem(
                            record.lease, marked.snapshot, int(class_id), policy
                        ),
                    )
                )[0]
                receipts = tuple(copy(prepared))
                if len(receipts) != len(prepared.moves) or not all(
                    isinstance(item, RelocationCopyReceipt) for item in receipts
                ):
                    raise ManagerError("relocation copy callback returned invalid receipts")
                submitted = manager.submit_relocation_batch(
                    ((prepared.relocation, receipts),)
                )[0]
                output = manager.complete_relocation_batch(
                    BatchCompletionReceipt(
                        self.engine_epoch,
                        int(completion_domain),
                        int(completion_value),
                    ),
                    (submitted.relocation,),
                )
            except Exception as error:
                self.fail_stop(f"token relocation became uncertain: {error}")
                raise FailStopped(self._failure or "token relocation failed") from error
            if len(output.publications) != 1:
                self.fail_stop("relocation publication cardinality changed")
                raise FailStopped(self._failure or "invalid relocation publication")
            publication = output.publications[0]
            try:
                self._validate_relocation_view(record, publication)
                packed = manager.token_views_batch(
                    (TokenViewQuery(record.lease, publication.snapshot, int(class_id), record.boundary),)
                )[0]
                retained_locations = self._retained_locations(packed, int(class_id))
                old_pages = tuple(record.cursor.pages.values())
                new_pages = self._packed_shadows(record, prepared, int(class_id))
                refs = self._page_registry.plan(old_pages, new_pages)
                self._validate_relocation_retirements(
                    prepared, output.retirements, int(class_id),
                    int(completion_domain), int(completion_value)
                )
                self._page_registry.commit(refs)
                record.cursor.pages = {
                    (item.class_id, item.logical_ordinal): item for item in new_pages
                }
                record.cursor.layout_boundaries[int(class_id)] = len(retained_locations)
                record.cursor.active_kv_lengths[int(class_id)] = len(retained_locations)
                self._replace_head(record.head, publication.snapshot)
                record.cursor.snapshot = publication.snapshot
                record.cursor.view_version = publication.view_version
                record.completion_domain = int(completion_domain)
                record.completion_value = int(completion_value)
            except Exception as error:
                self.fail_stop(f"relocation publication consumption failed: {error}")
                raise FailStopped(
                    self._failure or "relocation publication consumption failed"
                ) from error
            return RelocationPublication(
                key,
                old_view,
                prepared,
                publication,
                retained_locations,
                tuple(output.retirements),
            )

    def _validate_relocation_retirements(
        self, prepared: PreparedRelocation, retirements: Sequence[Any],
        class_id: int, completion_domain: int, completion_value: int
    ) -> None:
        arena = self.arenas_by_class[class_id]
        values = tuple(retirements)
        if len(values) != len(prepared.source_pages) or {
            item.page for item in values
        } != set(prepared.source_pages):
            raise ManagerError("relocation retirement set changed")
        previous = None
        for item in values:
            key = (item.logical_ordinal, item.page)
            expected_backend = (
                arena.backend_base_index + item.page.page_id - arena.first_page_id
            )
            if (
                previous is not None and key <= previous
                or item.class_id != class_id
                or item.backend_domain != arena.backend_domain
                or item.backend_index != expected_backend
                or item.token_begin != item.logical_ordinal * self.page_tokens
                or not item.token_begin < item.token_end_exclusive
                    <= item.token_begin + self.page_tokens
                or item.completion_domain != completion_domain
                or item.completion_value != completion_value
            ):
                raise ManagerError("relocation retirement certificate changed")
            previous = key

    def acknowledge_relocation(self, publication: RelocationPublication) -> None:
        with self._lock:
            self._healthy()
            from .reclamation import reclamation_receipts

            if publication.retirements:
                self.manager.acknowledge_reclamations_batch(
                    reclamation_receipts(publication.retirements)
                )

    def _validate_relocation_view(self, record: Any, view: RequestView) -> None:
        if (
            view.request != record.lease
            or view.boundary != record.boundary
            or view.snapshot.engine_epoch != self.engine_epoch
        ):
            raise ManagerError("relocation publication identity changed")

    def _retained_locations(
        self,
        view: TokenView,
        class_id: int,
        *,
        require_dead_absent: bool = True,
    ) -> tuple[int, ...]:
        arena = self.arenas_by_class[class_id]
        result = []
        for placement in view.placements:
            if not placement.disposition.kind.value:
                if placement.location is None:
                    raise ManagerError("retained token lost its physical location")
                location = placement.location
                page = sglang_page_id(location.backend_index, arena.backend_base_index)
                result.append(page * self.page_tokens + location.offset)
            elif require_dead_absent and placement.location is not None:
                raise ManagerError("packed non-retained token kept a location")
        return tuple(result)

    def _packed_shadows(
        self, record: Any, prepared: PreparedRelocation, class_id: int
    ) -> tuple[PageShadow, ...]:
        arena = self.arenas_by_class[class_id]
        return tuple(
            PageShadow(
                record.lease,
                class_id,
                ordinal,
                page,
                arena.backend_base_index + page.page_id - arena.first_page_id,
            )
            for ordinal, page in enumerate(prepared.destination_pages)
        )


__all__ = [
    "DispositionPublication",
    "RelocationPublication",
    "RelocationRuntimeMixin",
]
