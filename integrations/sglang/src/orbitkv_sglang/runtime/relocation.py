from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence

from .completion import BatchCompletionReceipt
from .identity import (
    FailStopped,
    ManagerError,
    PageLease,
    RetryableConflict,
    TokenRelocationManagerProtocol,
)
from .reclamation import ReclamationCertificate
from .snapshot_shadow import PageShadow, RequestView, sglang_page_id
from .token_relocation import (
    ClassTokenDispositionUpdate,
    PrepareRelocationItem,
    PreparedRelocation,
    RelocationBatchItem,
    RelocationCopyBatch,
    RelocationCopyReceipt,
    RelocationCopyUnobserved,
    RelocationPolicy,
    RelocationUnobservedReceipt,
    TokenDisposition,
    TokenDispositionBatchItem,
    TokenDispositionKind,
    TokenLocation,
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
    class_retained_locations: tuple[tuple[int, tuple[int, ...]], ...]
    retirements: tuple[ReclamationCertificate, ...]
    batch_id: int = 0


@dataclass(frozen=True, slots=True)
class RelocationBatchPublication:
    items: tuple[RelocationPublication, ...]
    retirements: tuple[ReclamationCertificate, ...]
    batch_id: int = 0


@dataclass(frozen=True, slots=True)
class DispositionPublication:
    key: Hashable
    old_view: TokenView
    publication: RequestView
    retained_locations: tuple[int, ...]
    class_retained_locations: tuple[tuple[int, tuple[int, ...]], ...]


class RelocationRuntimeMixin:
    def _relocation_manager(self) -> TokenRelocationManagerProtocol:
        if not isinstance(self.manager, TokenRelocationManagerProtocol):
            raise ManagerError("manager does not expose ABI8 token relocation")
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
            record = self._records_for_idle_batch((key,))[0]
            manager = self._relocation_manager()
            values, affected = self._validate_disposition_updates(
                tuple(updates), record.boundary
            )
            if int(class_id) not in affected:
                raise ManagerError("requested class lacks disposition updates")
            old_views = tuple(manager.token_views_batch(
                tuple(
                    TokenViewQuery(record.lease, record.head, item, record.boundary)
                    for item in affected
                )
            ))
            if len(old_views) != len(affected):
                raise ManagerError("manager returned wrong token-view cardinality")
            self._validate_relocation_readback(old_views, affected)
            try:
                publications = manager.mark_token_dispositions_batch(
                    (
                        TokenDispositionBatchItem(
                            record.lease, record.head, values
                        ),
                    )
                )
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"token disposition marking became uncertain: {error}")
                raise FailStopped(
                    self._failure or "token disposition marking failed"
                ) from error
            try:
                if len(publications) != 1:
                    raise ManagerError(
                        "token disposition publication cardinality changed"
                    )
                publication = publications[0]
                self._validate_relocation_view(record, publication)
                marked = manager.token_views_batch(
                    tuple(
                        TokenViewQuery(
                            record.lease, publication.snapshot, item, record.boundary
                        )
                        for item in affected
                    )
                )
                self._validate_relocation_readback(marked, affected)
                class_locations = tuple(
                    (
                        item.class_id,
                        self._retained_locations(
                            item, item.class_id, require_dead_absent=False
                        ),
                    )
                    for item in marked
                )
                self._validate_common_retained_set(marked)
                retained_locations = dict(class_locations)[int(class_id)]
                old_view = old_views[affected.index(int(class_id))]
                self._replace_head(record.head, publication.snapshot)
                record.cursor.snapshot = publication.snapshot
                record.cursor.view_version = publication.view_version
                for affected_class, locations in class_locations:
                    record.cursor.active_kv_lengths[affected_class] = len(locations)
                result = DispositionPublication(
                    key, old_view, publication, retained_locations, class_locations
                )
            except Exception as error:
                self.fail_stop(
                    f"token disposition publication consumption failed: {error}"
                )
                raise FailStopped(
                    self._failure or "token disposition publication consumption failed"
                ) from error
            return result

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
            domain = self._relocation_completion_component(
                "domain", completion_domain
            )
            value = self._relocation_completion_component(
                "value", completion_value
            )
            self._validate_relocation_completion_value(domain, value)

            def copy_one(
                prepared: tuple[PreparedRelocation, ...],
            ) -> RelocationCopyBatch:
                if len(prepared) != 1:
                    raise ManagerError(
                        "scalar relocation received a non-singleton prepared batch"
                    )
                return RelocationCopyBatch(
                    (tuple(copy(prepared[0])),), domain, value
                )

            publication = self.relocate_tokens_batch(
                (
                    RelocationBatchItem(
                        key, int(class_id), tuple(updates), policy
                    ),
                ),
                copy_one,
            )
            if len(publication.items) != 1:
                self.fail_stop("scalar relocation publication cardinality changed")
                raise FailStopped(
                    self._failure or "invalid scalar relocation publication"
                )
            return publication.items[0]

    def relocate_tokens_batch(
        self,
        items: Sequence[RelocationBatchItem],
        copy: Callable[[tuple[PreparedRelocation, ...]], RelocationCopyBatch],
    ) -> RelocationBatchPublication:
        with self._lock:
            self._healthy()
            values = tuple(items)
            if not values:
                raise ManagerError("relocation batch must be nonempty")
            if not all(isinstance(item, RelocationBatchItem) for item in values):
                raise ManagerError("relocation batch contains an invalid item")
            if not callable(copy):
                raise ManagerError("relocation copy callback is not callable")
            keys = tuple(item.key for item in values)
            try:
                if len(set(keys)) != len(keys):
                    raise ManagerError("relocation batch keys must be unique")
            except TypeError as error:
                raise ManagerError("relocation batch keys must be hashable") from error
            records = self._records_for_idle_batch(keys)
            manager = self._relocation_manager()
            if not hasattr(self, "_pending_relocation_batches"):
                self._pending_relocation_batches = {}
                self._pending_relocation_requests = {}
                self._next_relocation_batch_id = 1
            normalized: list[
                tuple[RelocationBatchItem, Any, int, tuple[int, ...]]
            ] = []
            queries = []
            for item, record in zip(values, records, strict=True):
                if isinstance(item.class_id, bool) or not isinstance(
                    item.class_id, int
                ):
                    raise ManagerError("relocation class id is invalid")
                class_id = item.class_id
                class_config = self.config.classes_by_id.get(class_id)
                if class_config is None or class_config.retention != "full":
                    raise ManagerError(
                        "physical token relocation requires a Full class"
                    )
                if not isinstance(item.policy, RelocationPolicy):
                    raise ManagerError("relocation batch policy is invalid")
                self._validate_relocation_policy(item.policy)
                updates, affected = self._validate_disposition_updates(
                    tuple(item.updates), record.boundary
                )
                if class_id not in affected:
                    raise ManagerError("relocation class lacks disposition updates")
                normalized.append((item, record, class_id, affected))
                queries.extend(
                    TokenViewQuery(record.lease, record.head, affected_class, record.boundary)
                    for affected_class in affected
                )
            old_flat = self._relocation_token_views(
                manager, tuple(queries), len(records)
            )
            old_views = self._relocation_view_groups(old_flat, normalized)
            for views, (_item, record, _class_id, affected) in zip(
                old_views, normalized, strict=True
            ):
                self._validate_relocation_readback(views, affected)
                if any(
                    view.view_version != record.cursor.view_version
                    or view.page_tokens != self.page_tokens
                    or len(view.placements) != record.boundary
                    for view in views
                ):
                    raise ManagerError("pre-relocation token view changed")
                for view in views:
                    self._validate_token_view_coordinates(
                        view, view.class_id, record.boundary
                    )
            try:
                marked_publications = manager.mark_token_dispositions_batch(
                    tuple(
                        TokenDispositionBatchItem(
                            record.lease, record.head, tuple(item.updates)
                        )
                        for item, record, _class_id, _affected in normalized
                    )
                )
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(
                    f"token relocation disposition marking became uncertain: {error}"
                )
                raise FailStopped(
                    self._failure or "token relocation disposition marking failed"
                ) from error

            prepared: tuple[PreparedRelocation, ...] = ()
            try:
                marked = tuple(marked_publications)
                if len(marked) != len(records):
                    raise ManagerError(
                        "token relocation disposition publication cardinality changed"
                    )
                self._validate_relocation_heads(
                    records, marked, increment=1, expected_snapshots=None
                )
                prepared = tuple(
                    manager.prepare_relocation_batch(
                        tuple(
                        PrepareRelocationItem(
                                record.lease, publication.snapshot, class_id, item.policy
                            )
                            for (item, record, class_id, _affected), publication in zip(
                                normalized, marked, strict=True
                            )
                        )
                    )
                )
                if len(prepared) != len(records) or not all(
                    isinstance(item, PreparedRelocation) for item in prepared
                ):
                    raise ManagerError(
                        "prepared relocation cardinality changed"
                    )
                old_pages: list[tuple[PageShadow, ...]] = []
                new_pages: list[tuple[PageShadow, ...]] = []
                seen_relocations = set()
                seen_targets = set()
                seen_pages = set()
                destination_slots = set()
                for prepared_item, marked_item, normalized_item, old_view_group in zip(
                    prepared, marked, normalized, old_views, strict=True
                ):
                    item, record, class_id, affected = normalized_item
                    if (
                        prepared_item.request != record.lease
                        or prepared_item.base_snapshot != marked_item.snapshot
                        or prepared_item.base_view_version != marked_item.view_version
                        or prepared_item.target_view_version != marked_item.view_version + 1
                        or prepared_item.class_id != class_id
                        or prepared_item.target_snapshot == marked_item.snapshot
                        or prepared_item.relocation in seen_relocations
                        or prepared_item.target_snapshot in seen_targets
                        or not prepared_item.source_pages
                        or not prepared_item.destination_pages
                        or prepared_item.projected_reclaimed_pages
                        != len(prepared_item.source_pages)
                        - len(prepared_item.destination_pages)
                        or prepared_item.projected_reclaimed_pages <= 0
                    ):
                        raise ManagerError("prepared relocation identity changed")
                    seen_relocations.add(prepared_item.relocation)
                    seen_targets.add(prepared_item.target_snapshot)
                    sources = tuple(
                        page
                        for page in record.cursor.pages.values()
                        if page.class_id == class_id
                    )
                    if set(prepared_item.source_pages) != {page.page for page in sources}:
                        raise ManagerError(
                            "prepared relocation source root changed"
                        )
                    relocation_pages = (
                        *prepared_item.source_pages,
                        *prepared_item.destination_pages,
                    )
                    if any(page in seen_pages for page in relocation_pages):
                        raise ManagerError(
                            "relocation batch aliases a physical page"
                        )
                    seen_pages.update(relocation_pages)
                    source_set = set(prepared_item.source_pages)
                    destination_set = set(prepared_item.destination_pages)
                    if len({move.token_id for move in prepared_item.moves}) != len(
                        prepared_item.moves
                    ) or any(
                        move.source.page not in source_set
                        or move.destination.page not in destination_set
                        for move in prepared_item.moves
                    ):
                        raise ManagerError("prepared relocation moves changed")
                    self._validate_prepared_relocation_moves(
                        item,
                        prepared_item,
                        old_view_group[affected.index(class_id)],
                        destination_slots,
                    )
                    old_pages.append(sources)
                    new_pages.append(
                        self._packed_shadows(record, prepared_item, class_id)
                    )
                page_refs = self._page_registry.plan(
                    tuple(
                        page for group in old_pages for page in group
                    ),
                    tuple(page for group in new_pages for page in group),
                )
                try:
                    copied = copy(prepared)
                except RelocationCopyUnobserved:
                    try:
                        manager.abort_relocations_batch(
                            tuple(
                                RelocationUnobservedReceipt(item.relocation)
                                for item in prepared
                            )
                        )
                    except Exception as abort_error:
                        raise ManagerError(
                            f"unobserved relocation abort became uncertain: {abort_error}"
                        ) from abort_error
                    raise
                if not isinstance(copied, RelocationCopyBatch):
                    raise ManagerError(
                        "relocation copy callback returned an invalid batch"
                    )
                receipt_groups = tuple(tuple(group) for group in copied.receipts)
                if len(receipt_groups) != len(prepared) or any(
                    len(group) != len(prepared_item.moves)
                    or not all(
                        isinstance(receipt, RelocationCopyReceipt)
                        for receipt in group
                    )
                    for prepared_item, group in zip(
                        prepared, receipt_groups, strict=True
                    )
                ):
                    raise ManagerError(
                        "relocation copy callback returned invalid receipts"
                    )
                completion_domain = self._relocation_completion_component(
                    "domain", copied.completion_domain
                )
                completion_value = (
                    self._next_completion_value(completion_domain)
                    if copied.completion_value is None
                    else self._relocation_completion_component(
                        "value", copied.completion_value
                    )
                )
                self._validate_relocation_completion_value(
                    completion_domain, completion_value
                )
                submitted = tuple(
                    manager.submit_relocation_batch(
                        tuple(
                            (prepared_item.relocation, receipts)
                            for prepared_item, receipts in zip(
                                prepared, receipt_groups, strict=True
                            )
                        )
                    )
                )
                if len(submitted) != len(prepared) or any(
                    submitted_item.relocation != prepared_item.relocation
                    or submitted_item.request != prepared_item.request
                    or submitted_item.target_snapshot != prepared_item.target_snapshot
                    for submitted_item, prepared_item in zip(
                        submitted, prepared, strict=True
                    )
                ):
                    raise ManagerError("submitted relocation identity changed")
                receipt = BatchCompletionReceipt(
                    self.engine_epoch, completion_domain, completion_value
                )
                output = manager.complete_relocation_batch(
                    receipt, tuple(item.relocation for item in submitted)
                )
            except Exception as error:
                self.fail_stop(f"token relocation became uncertain: {error}")
                raise FailStopped(
                    self._failure or "token relocation failed"
                ) from error

            try:
                publications = tuple(output.publications)
                if len(publications) != len(records):
                    raise ManagerError("relocation publication cardinality changed")
                self._validate_relocation_heads(
                    records, publications, increment=2,
                    expected_snapshots=tuple(
                        item.target_snapshot for item in prepared
                    ),
                )
                post_queries = tuple(
                    TokenViewQuery(
                        record.lease, publication.snapshot, affected_class,
                        record.boundary,
                    )
                    for publication, (_item, record, _class_id, affected) in zip(
                        publications, normalized, strict=True
                    )
                    for affected_class in affected
                )
                post_flat = self._relocation_token_views(
                    manager, post_queries, len(records)
                )
                view_groups = self._relocation_view_groups(post_flat, normalized)

                source_owner = {}
                for index, prepared_item in enumerate(prepared):
                    for page in prepared_item.source_pages:
                        if page in source_owner:
                            raise ManagerError("relocation source page aliases a request")
                        source_owner[page] = index
                retirement_groups: list[list[Any]] = [
                    [] for _item in prepared
                ]
                seen_retirements = set()
                for certificate in output.retirements:
                    owner = source_owner.get(certificate.page)
                    if owner is None or certificate.page in seen_retirements:
                        raise ManagerError("relocation retirement ownership changed")
                    seen_retirements.add(certificate.page)
                    retirement_groups[owner].append(certificate)
                grouped_retirements = tuple(
                    tuple(group) for group in retirement_groups
                )
                if tuple(
                    certificate
                    for group in grouped_retirements
                    for certificate in group
                ) != tuple(output.retirements):
                    raise ManagerError("relocation retirement order changed")

                batch_id = self._next_relocation_batch_id
                results = []
                cursor_pages = []
                class_location_groups = []
                for (
                    normalized_item,
                    old_view_group,
                    prepared_item,
                    publication,
                    views,
                    packed_pages,
                    retirements,
                ) in zip(
                    normalized, old_views, prepared, publications, view_groups,
                    new_pages, grouped_retirements, strict=True
                ):
                    item, record, class_id, affected = normalized_item
                    self._validate_relocation_readback(views, affected)
                    if any(
                        view.view_version != publication.view_version
                        or view.page_tokens != self.page_tokens
                        or len(view.placements) != record.boundary
                        for view in views
                    ):
                        raise ManagerError("relocation token readback changed")
                    self._validate_common_retained_set(views)
                    class_locations = tuple(
                        (
                            view.class_id,
                            self._retained_locations(
                                view, view.class_id,
                                require_dead_absent=view.class_id == class_id,
                            ),
                        )
                        for view in views
                    )
                    retained_locations = dict(class_locations)[class_id]
                    self._validate_relocation_retirements(
                        record, prepared_item, retirements, class_id,
                        completion_domain, completion_value,
                    )
                    pages = {
                        identity: shadow
                        for identity, shadow in record.cursor.pages.items()
                        if shadow.class_id != class_id
                    }
                    pages.update(
                        {
                            (shadow.class_id, shadow.logical_ordinal): shadow
                            for shadow in packed_pages
                        }
                    )
                    cursor_pages.append(pages)
                    class_location_groups.append(class_locations)
                    results.append(
                        RelocationPublication(
                            item.key,
                            old_view_group[affected.index(class_id)],
                            prepared_item,
                            publication,
                            retained_locations,
                            class_locations,
                            retirements,
                            batch_id,
                        )
                    )

                self._page_registry.commit(page_refs)
                self._replace_heads(
                    tuple(record.head for record in records),
                    tuple(publication.snapshot for publication in publications),
                )
                for (
                    record, normalized_item, publication, pages,
                    class_locations, result,
                ) in zip(
                    records, normalized, publications, cursor_pages,
                    class_location_groups, results, strict=True
                ):
                    _item, _record, class_id, _affected = normalized_item
                    record.cursor.pages = pages
                    record.cursor.layout_boundaries[class_id] = len(
                        result.retained_locations
                    )
                    for affected_class, locations in class_locations:
                        record.cursor.active_kv_lengths[affected_class] = len(locations)
                    record.cursor.snapshot = publication.snapshot
                    record.cursor.view_version = publication.view_version
                    record.completion_domain = completion_domain
                    record.completion_value = completion_value
                self._record_completion_point(
                    completion_domain, completion_value
                )
                self._completion_value = max(
                    self._completion_value, completion_value + 1
                )
                self._runtime_counters["completion_values"] += 1
                batch_publication = RelocationBatchPublication(
                    tuple(results), tuple(output.retirements), batch_id
                )
                self._pending_relocation_batches[batch_id] = batch_publication
                self._pending_relocation_requests.update(
                    (item.key, batch_id) for item in results
                )
                self._next_relocation_batch_id += 1
                return batch_publication
            except Exception as error:
                self.fail_stop(
                    f"token relocation publication consumption failed: {error}"
                )
                raise FailStopped(
                    self._failure
                    or "token relocation publication consumption failed"
                ) from error

    @staticmethod
    def _relocation_completion_component(name: str, value: Any) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value < 1 << 64
        ):
            raise ManagerError(
                f"relocation completion {name} must be a positive uint64"
            )
        return int(value)

    def _validate_disposition_updates(
        self,
        updates: tuple[ClassTokenDispositionUpdate, ...],
        boundary: int,
    ) -> tuple[tuple[ClassTokenDispositionUpdate, ...], tuple[int, ...]]:
        if not updates:
            raise ManagerError("token disposition updates must be nonempty")
        identities = []
        previous = None
        for update in updates:
            if not isinstance(update, ClassTokenDispositionUpdate):
                raise ManagerError("token disposition update is invalid")
            if (
                isinstance(update.class_id, bool)
                or not isinstance(update.class_id, int)
                or isinstance(update.token_id, bool)
                or not isinstance(update.token_id, int)
                or not isinstance(update.disposition, TokenDisposition)
            ):
                raise ManagerError("token disposition update is invalid")
            class_id = update.class_id
            token_id = update.token_id
            identity = (class_id, token_id)
            try:
                kind = TokenDispositionKind(update.disposition.kind)
            except (AttributeError, TypeError, ValueError) as error:
                raise ManagerError("token disposition kind is invalid") from error
            disposition = update.disposition
            if (
                class_id not in self.config.classes_by_id
                or token_id < 0
                or token_id >= boundary
                or previous is not None
                and identity <= previous
                or kind is TokenDispositionKind.RETAINED
                or isinstance(disposition.policy_or_proof_id, bool)
                or not isinstance(disposition.policy_or_proof_id, int)
                or not 0 < disposition.policy_or_proof_id < 1 << 64
                or isinstance(disposition.version, bool)
                or not isinstance(disposition.version, int)
                or not 0 <= disposition.version < 1 << 64
                or isinstance(disposition.quality_contract, bool)
                or not isinstance(disposition.quality_contract, int)
                or not 0 <= disposition.quality_contract < 1 << 64
                or kind is TokenDispositionKind.SEMANTICALLY_DEAD
                and disposition.quality_contract != 0
                or kind is TokenDispositionKind.POLICY_EVICTED
                and disposition.quality_contract == 0
            ):
                raise ManagerError("token disposition updates are invalid")
            identities.append(identity)
            previous = identity
        return updates, tuple(sorted({identity[0] for identity in identities}))

    def _validate_relocation_completion_value(
        self, completion_domain: int, completion_value: int
    ) -> None:
        if completion_value <= self._completion_high_water.get(
            completion_domain, 0
        ):
            raise ManagerError(
                "relocation completion value was already consumed"
            )
        if completion_value >= (1 << 64) - 1:
            raise ManagerError(
                "relocation completion value exhausts the runtime sequence"
            )

    def _validate_relocation_policy(self, policy: RelocationPolicy) -> None:
        integer_values = (
            policy.maximum_source_pages,
            policy.evacuation_headroom_pages,
            policy.fragmentation_threshold_milli,
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values)
            or not 0 < policy.maximum_source_pages <= self.page_count
            or not 0 < policy.evacuation_headroom_pages <= self.page_count
            or not 0 <= policy.fragmentation_threshold_milli <= 1000
            or policy.full_evacuation is not True
        ):
            raise ManagerError("relocation batch policy is invalid")

    def _validate_prepared_relocation_moves(
        self,
        item: RelocationBatchItem,
        prepared: PreparedRelocation,
        old_view: TokenView,
        destination_slots: set[tuple[int, int, int]],
    ) -> None:
        class_id = int(item.class_id)
        arena = self.arenas_by_class[class_id]
        evicted = {
            update.token_id
            for update in item.updates
            if update.class_id == class_id
        }
        retained = tuple(
            placement
            for placement in old_view.placements
            if placement.disposition.kind is TokenDispositionKind.RETAINED
            and placement.token_id not in evicted
        )
        if (
            old_view.class_id != class_id
            or tuple(move.token_id for move in prepared.moves)
            != tuple(placement.token_id for placement in retained)
            or len(prepared.destination_pages)
            != (len(retained) + self.page_tokens - 1) // self.page_tokens
        ):
            raise ManagerError("prepared relocation retained-token coverage changed")
        for ordinal, (movement, placement) in enumerate(
            zip(prepared.moves, retained, strict=True)
        ):
            source = placement.location
            destination_page = prepared.destination_pages[
                ordinal // self.page_tokens
            ]
            if (
                source is None
                or movement.source != source
                or movement.source.page not in prepared.source_pages
                or movement.destination.page != destination_page
                or movement.destination.offset != ordinal % self.page_tokens
                or not 0 <= movement.source.offset < self.page_tokens
                or not 0 <= movement.destination.offset < self.page_tokens
                or movement.source.backend_index
                != self._relocation_backend_index(arena, movement.source.page)
                or movement.destination.backend_index
                != self._relocation_backend_index(arena, movement.destination.page)
            ):
                raise ManagerError("prepared relocation move coordinate changed")
            slot = (
                arena.backend_domain,
                movement.destination.backend_index,
                movement.destination.offset,
            )
            if slot in destination_slots:
                raise ManagerError(
                    "prepared relocation destination slot is not unique"
                )
            destination_slots.add(slot)

    @staticmethod
    def _relocation_backend_index(arena: Any, page: Any) -> int:
        if (
            not isinstance(page, PageLease)
            or page.engine_epoch != arena.engine_epoch
            or page.pool_epoch != arena.pool_epoch
            or page.pool_id != arena.pool_id
            or page.generation <= 0
            or not arena.first_page_id
            <= page.page_id
            < arena.first_page_id + arena.page_count
        ):
            raise ManagerError("prepared relocation page is outside its arena")
        return arena.backend_base_index + page.page_id - arena.first_page_id

    def _validate_token_view_coordinates(
        self, view: TokenView, class_id: int, boundary: int
    ) -> None:
        arena = self.arenas_by_class[class_id]
        if tuple(placement.token_id for placement in view.placements) != tuple(
            range(boundary)
        ):
            raise ManagerError("token view ids are not canonical")
        occupied = set()
        for placement in view.placements:
            location = placement.location
            if placement.disposition.kind is TokenDispositionKind.RETAINED:
                if not isinstance(location, TokenLocation):
                    raise ManagerError("retained token location is invalid")
            elif location is None:
                continue
            elif not isinstance(location, TokenLocation):
                raise ManagerError("token location is invalid")
            assert location is not None
            if (
                location.backend_index
                != self._relocation_backend_index(arena, location.page)
                or not 0 <= location.offset < self.page_tokens
                or (location.page, location.offset) in occupied
            ):
                raise ManagerError("token view location changed")
            occupied.add((location.page, location.offset))

    def _relocation_token_views(
        self,
        manager: TokenRelocationManagerProtocol,
        queries: tuple[TokenViewQuery, ...],
        request_count: int,
    ) -> tuple[TokenView, ...]:
        if not queries:
            raise ManagerError("relocation token-view query must be nonempty")
        # ABI8 bounds token-view query count by operation capacity. Reading in
        # request-sized chunks is side-effect free; native mutations stay collective.
        result = []
        begin = 0
        while begin < len(queries):
            end = min(begin + request_count, len(queries))
            chunk = tuple(manager.token_views_batch(queries[begin:end]))
            if len(chunk) != end - begin:
                raise ManagerError("manager returned wrong token-view cardinality")
            result.extend(chunk)
            begin = end
        return tuple(result)

    @staticmethod
    def _relocation_view_groups(
        views: Sequence[TokenView],
        normalized: Sequence[tuple[RelocationBatchItem, Any, int, tuple[int, ...]]],
    ) -> tuple[tuple[TokenView, ...], ...]:
        result = []
        offset = 0
        values = tuple(views)
        for _item, _record, _class_id, affected in normalized:
            end = offset + len(affected)
            group = values[offset:end]
            if len(group) != len(affected):
                raise ManagerError("relocation token-view spans changed")
            result.append(group)
            offset = end
        if offset != len(values):
            raise ManagerError("relocation token-view spans changed")
        return tuple(result)

    def _validate_relocation_heads(
        self,
        records: Sequence[Any],
        publications: Sequence[RequestView],
        *,
        increment: int,
        expected_snapshots: Sequence[Any] | None,
    ) -> None:
        new_heads = tuple(publication.snapshot for publication in publications)
        if (
            len(set(new_heads)) != len(new_heads)
            or any(head in self._snapshot_leases for head in new_heads)
        ):
            raise ManagerError("relocation publication reused a live snapshot")
        for index, (record, publication) in enumerate(
            zip(records, publications, strict=True)
        ):
            self._validate_relocation_view(record, publication)
            if (
                publication.snapshot == record.head
                or publication.view_version != record.cursor.view_version + increment
                or expected_snapshots is not None
                and publication.snapshot != expected_snapshots[index]
            ):
                raise ManagerError("relocation publication head changed")

    def _validate_relocation_retirements(
        self, record: Any, prepared: PreparedRelocation, retirements: Sequence[Any],
        class_id: int, completion_domain: int, completion_value: int
    ) -> None:
        arena = self.arenas_by_class[class_id]
        values = tuple(retirements)
        old_pages = tuple(
            item for item in record.cursor.pages.values() if item.class_id == class_id
        )
        old_by_page = {item.page: item for item in old_pages}
        if len(old_by_page) != len(old_pages):
            raise ManagerError("relocation source shadow aliases a physical page")
        source_pages = set(prepared.source_pages)
        if len(values) != len(prepared.source_pages) or {
            item.page for item in values
        } != source_pages:
            raise ManagerError("relocation retirement set changed")
        if source_pages != set(old_by_page):
            raise ManagerError("full relocation did not evacuate the published class root")
        old_layout_boundary = record.cursor.layout_boundaries.get(
            class_id, record.boundary
        )
        expected_page_count = (old_layout_boundary + self.page_tokens - 1) // self.page_tokens
        if (
            old_layout_boundary <= 0
            or len(old_pages) != expected_page_count
            or {item.logical_ordinal for item in old_pages}
            != set(range(expected_page_count))
        ):
            raise ManagerError("relocation source shadow has an invalid layout boundary")
        previous = None
        for item in values:
            key = (item.logical_ordinal, item.page)
            shadow = old_by_page.get(item.page)
            expected_backend = (
                arena.backend_base_index + item.page.page_id - arena.first_page_id
            )
            expected_begin = item.logical_ordinal * self.page_tokens
            expected_end = min(expected_begin + self.page_tokens, old_layout_boundary)
            if (
                previous is not None and key <= previous
                or shadow is None
                or shadow.request != record.lease
                or shadow.class_id != class_id
                or shadow.logical_ordinal != item.logical_ordinal
                or shadow.backend_index != item.backend_index
                or item.class_id != class_id
                or item.backend_domain != arena.backend_domain
                or item.backend_index != expected_backend
                or item.token_begin != expected_begin
                or item.token_end_exclusive != expected_end
                or item.completion_domain != completion_domain
                or item.completion_value != completion_value
            ):
                raise ManagerError("relocation retirement certificate changed")
            previous = key

    def acknowledge_relocation_batch(
        self, publication: RelocationBatchPublication
    ) -> None:
        with self._lock:
            self._healthy()
            from .reclamation import reclamation_receipts

            if not isinstance(publication, RelocationBatchPublication):
                raise ManagerError("relocation ACK publication is invalid")
            pending = getattr(self, "_pending_relocation_batches", {}).get(
                publication.batch_id
            )
            if pending is not publication or publication.batch_id <= 0:
                raise ManagerError(
                    "relocation ACK does not name a pending batch"
                )
            flattened = tuple(
                certificate
                for item in publication.items
                for certificate in item.retirements
            )
            if (
                not publication.items
                or any(item.batch_id != publication.batch_id for item in publication.items)
                or flattened != publication.retirements
                or any(
                    getattr(self, "_pending_relocation_requests", {}).get(item.key)
                    != publication.batch_id
                    for item in publication.items
                )
            ):
                raise ManagerError("relocation ACK batch shape changed")
            receipts = reclamation_receipts(publication.retirements)
            if not receipts:
                raise ManagerError("relocation ACK batch has no retirements")
            try:
                self.manager.acknowledge_reclamations_batch(receipts)
            except Exception as error:
                self.fail_stop(f"relocation reclamation ACK became uncertain: {error}")
                raise FailStopped(
                    self._failure or "relocation reclamation ACK failed"
                ) from error
            del self._pending_relocation_batches[publication.batch_id]
            for item in publication.items:
                del self._pending_relocation_requests[item.key]

    def acknowledge_relocation(self, publication: RelocationPublication) -> None:
        if publication.batch_id <= 0:
            raise ManagerError("relocation ACK publication is not runtime-owned")
        pending = getattr(self, "_pending_relocation_batches", {}).get(
            publication.batch_id
        )
        if (
            pending is None
            or len(pending.items) != 1
            or pending.items[0] is not publication
        ):
            raise ManagerError(
                "scalar relocation ACK does not name a singleton batch"
            )
        self.acknowledge_relocation_batch(pending)

    def _validate_relocation_view(self, record: Any, view: RequestView) -> None:
        if (
            view.request != record.lease
            or view.boundary != record.boundary
            or view.snapshot.engine_epoch != self.engine_epoch
        ):
            raise ManagerError("relocation publication identity changed")

    @staticmethod
    def _validate_relocation_readback(
        views: Sequence[TokenView], expected_classes: Sequence[int]
    ) -> None:
        if tuple(item.class_id for item in views) != tuple(expected_classes):
            raise ManagerError("token disposition readback class order changed")

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

    @staticmethod
    def _validate_common_retained_set(views: Sequence[TokenView]) -> None:
        retained = [
            tuple(
                placement.token_id
                for placement in view.placements
                if not placement.disposition.kind.value
            )
            for view in views
        ]
        if not retained or any(value != retained[0] for value in retained[1:]):
            raise ManagerError("attention classes do not share one retained token set")

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
    "RelocationBatchPublication",
    "RelocationPublication",
    "RelocationRuntimeMixin",
]
