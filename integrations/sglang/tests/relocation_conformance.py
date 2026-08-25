from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable, Sequence

if TYPE_CHECKING:
    import torch

from orbitkv_sglang.config import load_config
from orbitkv_sglang.ffi import CtypesManagerFactory
from orbitkv_sglang.ffi.manager import CtypesManager
from orbitkv_sglang.runtime import (
    TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
    TAIL_IN_PLACE,
    TAIL_NONE,
    ArenaRegistration,
    BackendBindReceipt,
    BatchCompletionReceipt,
    CanonicalRuntime,
    ClassTokenDispositionUpdate,
    ManagerCreateSettings,
    ManagerError,
    PageLease,
    PrepareBatchItem,
    PrepareRelocationItem,
    PreparedRelocation,
    RelocationBatchItem,
    RelocationBatchPublication,
    RelocationCopyBatch,
    RelocationCopyReceipt,
    RelocationPolicy,
    RelocationUnobservedReceipt,
    ReleaseBatchItem,
    RequestView,
    SnapshotLease,
    TokenDisposition,
    TokenDispositionBatchItem,
    TokenDispositionKind,
    reclamation_receipts,
)


PAGE_TOKENS = 16
INITIAL_BOUNDARY = 48
APPEND_TOKENS = 17
RETAINED_TOKENS = 24
RETAINED_PER_PAGE = 8
TOKEN_BYTES = 257
APPEND_COMPLETION_DOMAIN = 10
RELOCATION_COMPLETION_DOMAIN = 20
_PAYLOAD_HEADER = struct.Struct("<8sQQQ")
_PAYLOAD_MAGIC = b"ORBITKV1"
_ZERO_PAGE = PageLease(0, 0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    batch_size: int
    cycles: int
    copied_tokens: int
    reused_source_pages: int


def opaque_payload_bytes(cycle: int, request_index: int, token_id: int) -> bytes:
    """Return an injective opaque record for one append operation.

    The complete ``(cycle, request_index, token_id)`` tuple is stored verbatim
    in the record header.  Hashes only fill the opaque tail, so uniqueness does
    not rely on a digest collision assumption.
    """

    values = (cycle, request_index, token_id)
    if any(isinstance(value, bool) or not 0 <= value < 2**64 for value in values):
        raise ValueError("opaque payload coordinates must be unsigned 64-bit integers")
    header = _PAYLOAD_HEADER.pack(_PAYLOAD_MAGIC, *values)
    seed = hashlib.sha256(header).digest()
    body = bytearray()
    counter = 0
    while len(header) + len(seed) + len(body) < TOKEN_BYTES:
        body.extend(hashlib.sha256(seed + counter.to_bytes(4, "little")).digest())
        counter += 1
    raw = header + seed + bytes(body[: TOKEN_BYTES - len(header) - len(seed)])
    assert len(raw) == TOKEN_BYTES
    assert _PAYLOAD_HEADER.unpack(raw[: _PAYLOAD_HEADER.size]) == (
        _PAYLOAD_MAGIC,
        *values,
    )
    return raw


def opaque_payload(cycle: int, request_index: int, token_id: int) -> torch.Tensor:
    import torch

    return torch.tensor(
        tuple(opaque_payload_bytes(cycle, request_index, token_id)),
        dtype=torch.uint8,
    )


@dataclass(slots=True)
class _ReferenceRequest:
    """Manager-independent logical oracle for one long-lived request."""

    request_index: int
    boundary: int = 0
    live_token_ids: list[int] = field(default_factory=list)
    payloads: dict[int, Any] = field(default_factory=dict)
    token_cycles: dict[int, int] = field(default_factory=dict)

    def append(
        self,
        cycle: int,
        target_boundary: int,
        encoded_payloads: set[bytes],
    ) -> tuple[int, ...]:
        if target_boundary <= self.boundary:
            raise AssertionError("reference append boundary did not advance")
        appended = tuple(range(self.boundary, target_boundary))
        new_records = []
        for token_id in appended:
            raw = opaque_payload_bytes(cycle, self.request_index, token_id)
            if raw in encoded_payloads:
                raise AssertionError("opaque payload encoding collided")
            new_records.append((token_id, raw))
        for token_id, raw in new_records:
            encoded_payloads.add(raw)
            self.payloads[token_id] = opaque_payload(
                cycle, self.request_index, token_id
            )
            self.token_cycles[token_id] = cycle
        self.live_token_ids.extend(appended)
        self.boundary = target_boundary
        return appended

    def relocation_partition(
        self, cycle: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if cycle == 0:
            victims = tuple(
                token_id
                for token_id in self.live_token_ids
                if token_id % PAGE_TOKENS >= RETAINED_PER_PAGE
            )
        else:
            victims = tuple(
                token_id
                for token_id in self.live_token_ids
                if self.token_cycles[token_id] == cycle
            )
        victim_set = set(victims)
        retained = tuple(
            token_id
            for token_id in self.live_token_ids
            if token_id not in victim_set
        )
        return victims, retained


class OpaqueCudaTokenArena:
    """Engine-neutral byte records operated on by real CUDA streams."""

    def __init__(self, pages: int, device: torch.device) -> None:
        import torch

        self.device = device
        self.storage = torch.empty(
            ((pages + 1) * PAGE_TOKENS, TOKEN_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        self.append_stream = torch.cuda.Stream(device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.consumer_stream = torch.cuda.Stream(device=device)
        default_stream = torch.cuda.current_stream(device)
        stream_ids = {
            default_stream.cuda_stream,
            self.append_stream.cuda_stream,
            self.copy_stream.cuda_stream,
            self.consumer_stream.cuda_stream,
        }
        assert len(stream_ids) == 4

    @staticmethod
    def slot(backend_index: int, offset: int) -> int:
        if backend_index < 0 or not 0 <= offset < PAGE_TOKENS:
            raise AssertionError("invalid backend token location")
        # Reserve the leading page, matching SGLang's one-based page locations.
        return (backend_index + 1) * PAGE_TOKENS + offset

    def copy_intents(self, intents: Sequence[Any]) -> None:
        import torch

        for intent in intents:
            source = torch.arange(
                self.slot(
                    intent.source_backend_index, intent.source_token_offset
                ),
                self.slot(
                    intent.source_backend_index, intent.source_token_offset
                )
                + intent.token_count,
                dtype=torch.int64,
                device=self.device,
            )
            destination = torch.arange(
                self.slot(
                    intent.destination_backend_index,
                    intent.destination_token_offset,
                ),
                self.slot(
                    intent.destination_backend_index,
                    intent.destination_token_offset,
                )
                + intent.token_count,
                dtype=torch.int64,
                device=self.device,
            )
            copied = self.storage.index_select(0, source)
            self.storage.index_copy_(0, destination, copied)

    def write_records(
        self, slots: Sequence[int], payloads: Sequence[torch.Tensor]
    ) -> None:
        import torch

        indices = torch.tensor(tuple(slots), dtype=torch.int64, device=self.device)
        values = torch.stack(tuple(payloads)).to(self.device)
        self.storage.index_copy_(0, indices, values)

    def enqueue_consume_after(
        self, producer_event: Any, slots: Sequence[int]
    ) -> tuple[Any, Any]:
        """Queue a dependent read before either stream is host-synchronized."""

        import torch

        with torch.cuda.stream(self.consumer_stream):
            self.consumer_stream.wait_event(producer_event)
            indices = torch.tensor(
                tuple(slots), dtype=torch.int64, device=self.device
            )
            observed = self.storage.index_select(0, indices)
            consumer_event = torch.cuda.Event(enable_timing=False)
            consumer_event.record(self.consumer_stream)
        return observed, consumer_event

    @staticmethod
    def finish_consume(observed: Any, consumer_event: Any) -> Any:
        consumer_event.synchronize()
        assert consumer_event.query()
        return observed.cpu()


def _record_ready_event(stream: Any) -> Any:
    import torch

    event = torch.cuda.Event(enable_timing=False)
    event.record(stream)
    return event


def make_manager(
    tmp_path: Path, library: Path, batch_size: int
) -> tuple[Any, CtypesManager]:
    plan = tmp_path / f"cuda-relocation-b{batch_size}.json"
    plan.write_text(
        json.dumps(
            {
                "page_tokens": PAGE_TOKENS,
                "classes": [
                    {
                        "name": "full",
                        "layers": [0],
                        "retention": "full",
                        "bytes_per_token_per_layer": TOKEN_BYTES,
                        "window_tokens": None,
                    }
                ],
            }
        )
    )
    config = load_config(
        {"ORBITKV_PLAN": str(plan), "ORBITKV_LIBRARY": str(library)}
    )
    pages = 5 * batch_size
    class_config = config.classes[0]
    manager = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(
            maximum_requests=2 * batch_size,
            maximum_operations=batch_size,
            maximum_prefixes=2 * batch_size,
            maximum_reclamations=pages,
            maximum_step_tokens=INITIAL_BOUNDARY,
        ),
        (
            ArenaRegistration(
                class_config.class_id,
                class_config.pool_id,
                class_config.backend_domain,
                pages,
                0,
            ),
        ),
    )
    assert isinstance(manager, CtypesManager)
    return config, manager


def assert_batch_prepare_failure_atomic(
    tmp_path: Path, library: Path, *, batch_size: int
) -> None:
    """A stale member cannot partially reserve append or relocation state."""

    _config, manager = make_manager(tmp_path, library, batch_size)
    try:
        views = manager.request_acquire_batch(batch_size)
        before = manager.stats()
        valid_items = tuple(
            PrepareBatchItem(view.request, view.snapshot, INITIAL_BOUNDARY)
            for view in views
        )
        stale_view = views[-1]
        stale_snapshot = SnapshotLease(
            stale_view.snapshot.engine_epoch,
            stale_view.snapshot.slot,
            stale_view.snapshot.generation + 1,
        )
        stale_items = list(valid_items)
        stale_items[-1] = PrepareBatchItem(
            stale_view.request, stale_snapshot, INITIAL_BOUNDARY
        )
        try:
            manager.prepare_batch(tuple(stale_items))
        except ManagerError:
            pass
        else:
            raise AssertionError("stale append batch prepare unexpectedly succeeded")

        assert manager.stats() == before
        prepared = manager.prepare_batch(valid_items)
        assert len(prepared) == batch_size
        arena = manager.arenas_by_class[0]
        submitted = manager.submit_batch(
            tuple(
                (
                    item.step,
                    tuple(
                        BackendBindReceipt(
                            item.step,
                            PageLease(
                                item.request.engine_epoch,
                                arena.pool_epoch,
                                intent.page_generation,
                                intent.page_id,
                                arena.pool_id,
                            ),
                            arena.backend_domain,
                            backend_index=(
                                arena.backend_base_index
                                + intent.page_id
                                - arena.first_page_id
                            ),
                        )
                        for intent in item.write_intents
                    ),
                    (),
                )
                for item in prepared
            )
        )
        completed = manager.complete_batch(
            BatchCompletionReceipt(views[0].request.engine_epoch, 1, 1),
            tuple(item.submission for item in submitted),
        )
        views = tuple(
            RequestView(
                item.request,
                item.published_snapshot,
                item.published_view_version,
                item.published_boundary,
                item.resident_count,
            )
            for item in completed.completions
        )
        assert not completed.retirements

        disposition_updates = tuple(
            ClassTokenDispositionUpdate(
                0,
                token_id,
                TokenDisposition(
                    TokenDispositionKind.POLICY_EVICTED, 7, 1, 99
                ),
            )
            for token_id in range(INITIAL_BOUNDARY)
            if token_id % PAGE_TOKENS >= RETAINED_PER_PAGE
        )
        marked = manager.mark_token_dispositions_batch(
            tuple(
                TokenDispositionBatchItem(
                    view.request, view.snapshot, disposition_updates
                )
                for view in views
            )
        )
        relocation_items = [
            PrepareRelocationItem(
                view.request,
                view.snapshot,
                0,
                RelocationPolicy(3, 2, 250, True),
            )
            for view in marked
        ]
        stale = relocation_items[-1]
        marked_snapshot = stale.expected_snapshot
        relocation_items[-1] = PrepareRelocationItem(
            stale.request,
            SnapshotLease(
                marked_snapshot.engine_epoch,
                marked_snapshot.slot,
                marked_snapshot.generation + 1,
            ),
            stale.class_id,
            stale.policy,
        )
        before_relocation = manager.stats()
        try:
            manager.prepare_relocation_batch(tuple(relocation_items))
        except ManagerError:
            pass
        else:
            raise AssertionError(
                "stale relocation batch prepare unexpectedly succeeded"
            )
        assert manager.stats() == before_relocation

        valid_relocations = manager.prepare_relocation_batch(
            tuple(
                PrepareRelocationItem(
                    view.request,
                    view.snapshot,
                    0,
                    RelocationPolicy(3, 2, 250, True),
                )
                for view in marked
            )
        )
        assert len(valid_relocations) == batch_size
        manager.abort_relocations_batch(
            tuple(
                RelocationUnobservedReceipt(item.relocation)
                for item in valid_relocations
            )
        )
        assert manager.stats() == before_relocation

        released = manager.release_batch(
            tuple(ReleaseBatchItem(view.request, view.snapshot) for view in marked)
        )
        assert len(released.retirements) == 3 * batch_size
        manager.acknowledge_reclamations_batch(
            reclamation_receipts(released.retirements)
        )
        manager.recycle_requests_batch(tuple(view.request for view in views))
        final = manager.stats()
        assert final.free_pages == 5 * batch_size
        assert final.active_requests == final.active_snapshots == 0
        assert final.prepared_steps == final.submitted_steps == 0
    finally:
        manager.destroy()


def _backend_index(runtime: CanonicalRuntime, page: PageLease) -> int:
    arena = runtime.arenas_by_class[0]
    assert page.engine_epoch == arena.engine_epoch
    assert page.pool_epoch == arena.pool_epoch
    assert page.pool_id == arena.pool_id
    assert arena.first_page_id <= page.page_id < arena.first_page_id + arena.page_count
    return arena.backend_base_index + page.page_id - arena.first_page_id


def _location_slot(arena: OpaqueCudaTokenArena, location: Any) -> int:
    return arena.slot(location.backend_index, location.offset)


def _assert_reference_view(
    arena: OpaqueCudaTokenArena, view: Any, reference: _ReferenceRequest
) -> dict[int, int]:
    assert view.class_id == 0
    assert view.page_tokens == PAGE_TOKENS
    assert tuple(item.token_id for item in view.placements) == tuple(
        range(reference.boundary)
    )
    expected_live = tuple(reference.live_token_ids)
    observed_live = tuple(
        item.token_id
        for item in view.placements
        if item.disposition.kind is TokenDispositionKind.RETAINED
    )
    assert observed_live == expected_live

    expected_live_set = set(expected_live)
    mapping: dict[int, int] = {}
    page_by_ordinal: dict[int, tuple[PageLease, int]] = {}
    for physical, token_id in enumerate(expected_live):
        placement = view.placements[token_id]
        assert placement.disposition.kind is TokenDispositionKind.RETAINED
        assert placement.location is not None
        location = placement.location
        assert location.offset == physical % PAGE_TOKENS
        ordinal = physical // PAGE_TOKENS
        identity = (location.page, location.backend_index)
        previous = page_by_ordinal.setdefault(ordinal, identity)
        assert previous == identity
        mapping[token_id] = _location_slot(arena, location)
    assert len(set(page_by_ordinal.values())) == len(page_by_ordinal)
    assert len(set(mapping.values())) == len(mapping)
    for placement in view.placements:
        if placement.token_id not in expected_live_set:
            assert placement.disposition.kind is not TokenDispositionKind.RETAINED
            assert placement.location is None
    return mapping


def _assert_payload_headers(
    observed: Any,
    references: Sequence[_ReferenceRequest],
    token_groups: Sequence[Sequence[int]],
) -> None:
    rows = observed.tolist()
    expected = tuple(
        (
            reference.token_cycles[token_id],
            reference.request_index,
            token_id,
        )
        for reference, token_ids in zip(references, token_groups, strict=True)
        for token_id in token_ids
    )
    assert len(rows) == len(expected)
    for row, coordinates in zip(rows, expected, strict=True):
        assert _PAYLOAD_HEADER.unpack(bytes(row[: _PAYLOAD_HEADER.size])) == (
            _PAYLOAD_MAGIC,
            *coordinates,
        )


def _apply_cow_mapping(
    arena: OpaqueCudaTokenArena,
    mapping: dict[int, int],
    intents: Sequence[Any],
) -> None:
    reverse = {slot: token_id for token_id, slot in mapping.items()}
    assert len(reverse) == len(mapping)
    for intent in intents:
        for offset in range(intent.token_count):
            source = arena.slot(
                intent.source_backend_index, intent.source_token_offset + offset
            )
            destination = arena.slot(
                intent.destination_backend_index,
                intent.destination_token_offset + offset,
            )
            token_id = reverse[source]
            mapping[token_id] = destination


def _append_batch(
    runtime: CanonicalRuntime,
    arena: OpaqueCudaTokenArena,
    keys: Sequence[Hashable],
    references: Sequence[_ReferenceRequest],
    target_boundary: int,
    cycle: int,
    encoded_payloads: set[bytes],
) -> None:
    import torch

    old_maps = tuple(
        _assert_reference_view(arena, runtime.token_view(key, 0), reference)
        for key, reference in zip(keys, references, strict=True)
    )
    batch, plans = runtime.prepare_batch(
        tuple((key, target_boundary) for key in keys)
    )
    assert batch.keys == tuple(keys)
    assert len(plans) == len(keys)
    appended = tuple(
        reference.append(cycle, target_boundary, encoded_payloads)
        for reference in references
    )

    predicted_maps: list[dict[int, int]] = []
    try:
        with torch.cuda.stream(arena.append_stream):
            for reference, new_ids, old_map, pending, plan in zip(
                references, appended, old_maps, batch.records, plans, strict=True
            ):
                spec = plan.by_class[0]
                prepared = pending.prepared
                lowering = prepared.class_lowerings[0]
                action = spec.tail_action
                assert plan.previous_boundary == target_boundary - len(new_ids)
                assert plan.target_boundary == target_boundary
                previous_physical = len(old_map)
                target_physical = previous_physical + len(new_ids)
                assert spec.previous_layout_boundary == previous_physical
                assert spec.target_layout_boundary == target_physical
                copies = spec.copy_intents
                assert copies == prepared.copy_intents[
                    lowering.copy_offset : lowering.copy_offset
                    + lowering.copy_count
                ]

                mapping = dict(old_map)
                arena.copy_intents(copies)
                _apply_cow_mapping(arena, mapping, copies)

                partial = previous_physical % PAGE_TOKENS
                tail_end = (
                    previous_physical + PAGE_TOKENS - partial
                    if partial
                    else previous_physical
                )
                tail_backend: int | None = None
                if partial:
                    if action.kind == TAIL_IN_PLACE:
                        assert action.source == action.destination != _ZERO_PAGE
                        assert not copies
                    elif action.kind == TAIL_COPY_ON_WRITE:
                        assert action.source != action.destination
                        assert action.destination != _ZERO_PAGE
                        assert len(copies) == 1
                        assert copies[0].destination == action.destination
                    elif action.kind == TAIL_FRESH:
                        assert action.source == _ZERO_PAGE
                        assert action.destination != _ZERO_PAGE
                        assert not copies
                    else:
                        raise AssertionError(
                            "partial append has no usable tail action"
                        )
                    tail_backend = _backend_index(runtime, action.destination)
                else:
                    assert action.kind == TAIL_NONE
                    assert action.source == action.destination == _ZERO_PAGE
                    assert not copies

                writes = prepared.write_intents[
                    lowering.write_offset : lowering.write_offset
                    + lowering.write_count
                ]
                first_new_ordinal = (
                    previous_physical + PAGE_TOKENS - 1
                ) // PAGE_TOKENS
                write_slots = []
                for token_id, physical in zip(
                    new_ids,
                    range(previous_physical, target_physical),
                    strict=True,
                ):
                    if physical < tail_end:
                        assert tail_backend is not None
                        backend_index = tail_backend
                    else:
                        intent_index = (
                            physical // PAGE_TOKENS - first_new_ordinal
                        )
                        intent = writes[intent_index]
                        page = PageLease(
                            prepared.request.engine_epoch,
                            runtime.arenas_by_class[0].pool_epoch,
                            intent.page_generation,
                            intent.page_id,
                            runtime.arenas_by_class[0].pool_id,
                        )
                        backend_index = _backend_index(runtime, page)
                    slot = arena.slot(backend_index, physical % PAGE_TOKENS)
                    if slot in mapping.values() or slot in write_slots:
                        raise AssertionError(
                            "append write aliases a live token slot"
                        )
                    mapping[token_id] = slot
                    write_slots.append(slot)
                assert len(writes) == (
                    target_physical + PAGE_TOKENS - 1
                ) // PAGE_TOKENS - first_new_ordinal
                arena.write_records(
                    write_slots,
                    tuple(reference.payloads[token_id] for token_id in new_ids),
                )
                assert tuple(mapping) == tuple(reference.live_token_ids)
                predicted_maps.append(mapping)
            append_event = _record_ready_event(arena.append_stream)
    except Exception as error:
        runtime.lowering_failed(batch, error)
        raise

    expected_slots = tuple(
        mapping[token_id]
        for mapping, reference in zip(predicted_maps, references, strict=True)
        for token_id in reference.live_token_ids
    )
    try:
        observed, consumer_event = arena.enqueue_consume_after(
            append_event, expected_slots
        )
        runtime.mark_lowered(batch)
        runtime.submit_batch(batch)
        runtime.mark_forward(batch)
        runtime.register_event(batch, append_event, APPEND_COMPLETION_DOMAIN)
        runtime.wait_batch(keys)
        actual = arena.finish_consume(observed, consumer_event)
    except Exception as error:
        if runtime.failure_reason is None:
            runtime.lowering_failed(batch, error)
        raise
    wanted = torch.stack(
        tuple(
            reference.payloads[token_id]
            for reference in references
            for token_id in reference.live_token_ids
        )
    )
    assert append_event.query()
    assert torch.equal(actual, wanted)
    _assert_payload_headers(
        actual, references, tuple(reference.live_token_ids for reference in references)
    )

    completion_value = runtime.record_for(keys[0]).completion_value
    assert completion_value > 0
    for key, reference, predicted in zip(
        keys, references, predicted_maps, strict=True
    ):
        record = runtime.record_for(key)
        assert record.boundary == reference.boundary
        assert record.completion_domain == APPEND_COMPLETION_DOMAIN
        assert record.completion_value == completion_value
        actual_mapping = _assert_reference_view(
            arena, runtime.token_view(key, 0), reference
        )
        assert actual_mapping == predicted


def _expected_retirement_spans(
    physical_boundary: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            ordinal,
            ordinal * PAGE_TOKENS,
            min((ordinal + 1) * PAGE_TOKENS, physical_boundary),
        )
        for ordinal in range(
            (physical_boundary + PAGE_TOKENS - 1) // PAGE_TOKENS
        )
    )


def _relocate_batch(
    runtime: CanonicalRuntime,
    arena: OpaqueCudaTokenArena,
    keys: Sequence[Hashable],
    references: Sequence[_ReferenceRequest],
    cycle: int,
) -> tuple[RelocationBatchPublication, dict[tuple[int, int], int], int]:
    """Relocate every request through one real runtime batch transaction.

    The consumer readback is deliberately synchronized before the copy callback
    returns.  This proves eager byte-exact CUDA copying; it does not model or
    claim a pending-event asynchronous publication/ACK barrier.
    """

    import torch

    if not keys or len(keys) != len(references):
        raise AssertionError("relocation batch shape changed")
    source_mappings = tuple(
        _assert_reference_view(arena, runtime.token_view(key, 0), reference)
        for key, reference in zip(keys, references, strict=True)
    )
    physical_boundaries = tuple(
        len(reference.live_token_ids) for reference in references
    )
    partitions = tuple(
        reference.relocation_partition(cycle) for reference in references
    )
    expected_victim_count = 24 if cycle == 0 else APPEND_TOKENS
    assert all(
        len(victims) == expected_victim_count
        and len(retained) == RETAINED_TOKENS
        for victims, retained in partitions
    )
    updates = tuple(
        tuple(
            ClassTokenDispositionUpdate(
                0,
                token_id,
                TokenDisposition(
                    TokenDispositionKind.POLICY_EVICTED,
                    policy_or_proof_id=7,
                    version=cycle + 1,
                    quality_contract=99,
                ),
            )
            for token_id in victims
        )
        for victims, _retained in partitions
    )
    expected_requests = tuple(runtime.record_for(key).lease for key in keys)
    previous_completion_values = {
        runtime.record_for(key).completion_value for key in keys
    }
    assert len(previous_completion_values) == 1
    previous_completion_value = previous_completion_values.pop()
    assert previous_completion_value > 0
    copy_evidence: dict[str, Any] = {"calls": 0}

    def copy(
        prepared: tuple[PreparedRelocation, ...],
    ) -> RelocationCopyBatch:
        copy_evidence["calls"] += 1
        assert copy_evidence["calls"] == 1
        assert isinstance(prepared, tuple)
        assert tuple(item.request for item in prepared) == expected_requests

        source_slot_groups = []
        destination_slot_groups = []
        receipt_groups = []
        for item, source_mapping, partition in zip(
            prepared, source_mappings, partitions, strict=True
        ):
            _victims, retained = partition
            assert len(item.source_pages) == 3
            assert len(item.destination_pages) == 2
            assert item.projected_reclaimed_pages == 1
            assert tuple(move.token_id for move in item.moves) == retained
            source_slots = tuple(
                _location_slot(arena, move.source) for move in item.moves
            )
            destination_slots = tuple(
                _location_slot(arena, move.destination) for move in item.moves
            )
            assert source_slots == tuple(
                source_mapping[token_id] for token_id in retained
            )
            assert len(set(destination_slots)) == len(destination_slots)
            source_slot_groups.append(source_slots)
            destination_slot_groups.append(destination_slots)
            receipt_groups.append(
                tuple(
                    RelocationCopyReceipt(
                        item.relocation,
                        move.token_id,
                        move.source,
                        move.destination,
                    )
                    for move in item.moves
                )
            )

        flat_sources = tuple(
            slot for group in source_slot_groups for slot in group
        )
        flat_destinations = tuple(
            slot for group in destination_slot_groups for slot in group
        )
        assert len(set(flat_sources)) == len(flat_sources)
        assert len(set(flat_destinations)) == len(flat_destinations)
        assert set(flat_sources).isdisjoint(flat_destinations)
        with torch.cuda.stream(arena.copy_stream):
            source_indices = torch.tensor(
                flat_sources, dtype=torch.int64, device=arena.device
            )
            destination_indices = torch.tensor(
                flat_destinations, dtype=torch.int64, device=arena.device
            )
            copied = arena.storage.index_select(0, source_indices)
            arena.storage.index_copy_(0, destination_indices, copied)
            copy_event = _record_ready_event(arena.copy_stream)
        observed, consumer_event = arena.enqueue_consume_after(
            copy_event, flat_destinations
        )
        actual = arena.finish_consume(observed, consumer_event)
        wanted = torch.stack(
            tuple(
                reference.payloads[token_id]
                for reference, (_victims, retained) in zip(
                    references, partitions, strict=True
                )
                for token_id in retained
            )
        )
        assert copy_event.query()
        assert torch.equal(actual, wanted)
        _assert_payload_headers(
            actual, references, tuple(retained for _victims, retained in partitions)
        )
        copy_evidence["event"] = copy_event
        copy_evidence["destination_slot_groups"] = tuple(
            destination_slot_groups
        )
        return RelocationCopyBatch(
            tuple(receipt_groups), RELOCATION_COMPLETION_DOMAIN
        )

    output = runtime.relocate_tokens_batch(
        tuple(
            RelocationBatchItem(
                key,
                0,
                request_updates,
                RelocationPolicy(3, 2, 250, True),
            )
            for key, request_updates in zip(keys, updates, strict=True)
        ),
        copy,
    )
    assert copy_evidence["calls"] == 1
    assert copy_evidence["event"].query()
    assert tuple(item.key for item in output.items) == tuple(keys)
    assert output.batch_id > 0
    assert all(item.batch_id == output.batch_id for item in output.items)
    assert output.retirements == tuple(
        retirement
        for item in output.items
        for retirement in item.retirements
    )
    completion_values = {
        runtime.record_for(key).completion_value for key in keys
    }
    assert len(completion_values) == 1
    completion_value = completion_values.pop()
    assert completion_value > previous_completion_value

    source_generations: dict[tuple[int, int], int] = {}
    moved = 0
    for key, reference, physical_boundary, partition, item, destination_slots in zip(
        keys,
        references,
        physical_boundaries,
        partitions,
        output.items,
        copy_evidence["destination_slot_groups"],
        strict=True,
    ):
        _victims, retained = partition
        assert tuple(move.token_id for move in item.prepared.moves) == retained
        assert item.retained_locations == destination_slots

        expected_spans = _expected_retirement_spans(physical_boundary)
        actual_spans = tuple(
            (
                retirement.logical_ordinal,
                retirement.token_begin,
                retirement.token_end_exclusive,
            )
            for retirement in item.retirements
        )
        assert actual_spans == expected_spans
        if cycle == 1:
            assert actual_spans == ((0, 0, 16), (1, 16, 32), (2, 32, 41))
        assert {retirement.page for retirement in item.retirements} == set(
            item.prepared.source_pages
        )
        assert all(
            retirement.class_id == 0
            and retirement.backend_domain
            == runtime.arenas_by_class[0].backend_domain
            and retirement.completion_domain == RELOCATION_COMPLETION_DOMAIN
            and retirement.completion_value == completion_value
            for retirement in item.retirements
        )

        reference.live_token_ids[:] = retained
        record = runtime.record_for(key)
        assert record.boundary == reference.boundary
        assert record.cursor.layout_boundaries[0] == RETAINED_TOKENS
        assert runtime.active_kv_length(key, 0) == RETAINED_TOKENS
        assert record.completion_domain == RELOCATION_COMPLETION_DOMAIN
        assert record.completion_value == completion_value
        after_mapping = _assert_reference_view(
            arena, runtime.token_view(key, 0), reference
        )
        assert after_mapping == dict(
            zip(retained, destination_slots, strict=True)
        )
        sources = {
            (page.pool_id, page.page_id): page.generation
            for page in item.prepared.source_pages
        }
        assert source_generations.keys().isdisjoint(sources)
        source_generations.update(sources)
        moved += len(item.prepared.moves)
    return output, source_generations, moved


def _probe_reuse(
    runtime: CanonicalRuntime,
    probe_keys: Sequence[Hashable],
    source: dict[tuple[int, int], int],
) -> int:
    batch, _plans = runtime.prepare_batch(
        tuple((key, INITIAL_BOUNDARY) for key in probe_keys)
    )
    generations = {
        (shadow.page.pool_id, shadow.page.page_id): shadow.page.generation
        for pending in batch.records
        for shadow in pending.new_pages
    }
    assert generations.keys() == source.keys()
    assert all(generations[key] > generation for key, generation in source.items())
    runtime.abort_unobserved(batch)
    runtime.release_batch(probe_keys)
    return len(generations)


def run_cuda_relocation_conformance(
    tmp_path: Path,
    library: Path,
    *,
    batch_size: int,
    cycles: int,
    device: torch.device,
) -> ConformanceResult:
    if batch_size not in (1, 4, 32):
        raise ValueError("conformance batch size must be B1, B4, or B32")
    if cycles < 2:
        raise ValueError("conformance cycles must be at least two")
    config, manager = make_manager(tmp_path, library, batch_size)
    runtime = CanonicalRuntime(config, manager)
    arena = OpaqueCudaTokenArena(5 * batch_size, device)
    keys = tuple(("request", index) for index in range(batch_size))
    references = tuple(_ReferenceRequest(index) for index in range(batch_size))
    encoded_payloads: set[bytes] = set()
    acquired = runtime.request_acquire_batch(keys)
    request_ids = tuple(view.request for view in acquired)
    cursors = tuple(runtime.record_for(key).cursor for key in keys)
    copied = reused = 0
    previous_batch_id = 0
    try:
        for cycle in range(cycles):
            previous_boundary = references[0].boundary
            target_boundary = (
                INITIAL_BOUNDARY
                if cycle == 0
                else previous_boundary + APPEND_TOKENS
            )
            assert all(
                reference.boundary == previous_boundary
                for reference in references
            )
            _append_batch(
                runtime,
                arena,
                keys,
                references,
                target_boundary,
                cycle,
                encoded_payloads,
            )
            assert tuple(runtime.record_for(key).lease for key in keys) == request_ids
            assert all(
                runtime.record_for(key).cursor is cursor
                for key, cursor in zip(keys, cursors, strict=True)
            )
            expected_physical_boundary = INITIAL_BOUNDARY if cycle == 0 else 41
            assert all(
                runtime.record_for(key).cursor.layout_boundaries.get(
                    0, runtime.record_for(key).boundary
                )
                == expected_physical_boundary
                for key in keys
            )

            publication, source_generations, moved = _relocate_batch(
                runtime, arena, keys, references, cycle
            )
            assert publication.batch_id > previous_batch_id
            previous_batch_id = publication.batch_id
            copied += moved
            assert tuple(runtime.record_for(key).lease for key in keys) == request_ids
            assert all(
                runtime.record_for(key).cursor is cursor
                for key, cursor in zip(keys, cursors, strict=True)
            )

            stats = runtime.stats()
            assert stats.free_pages == 0
            assert stats.active_pages == 2 * batch_size
            assert stats.retiring_pages == 3 * batch_size
            assert stats.pending_reclamations == 3 * batch_size
            assert len(source_generations) == 3 * batch_size

            probe_keys = tuple(("probe", cycle, index) for index in range(batch_size))
            runtime.request_acquire_batch(probe_keys)
            before_failed_prepare = runtime.stats()
            probe_cursors = tuple(runtime.record_for(key).cursor for key in probe_keys)
            probe_heads = tuple(runtime.record_for(key).head for key in probe_keys)
            try:
                runtime.prepare_batch(
                    tuple((key, INITIAL_BOUNDARY) for key in probe_keys)
                )
            except ManagerError:
                pass
            else:
                raise AssertionError(
                    "retiring source pages were reused before reclamation ACK"
                )
            assert runtime.stats() == before_failed_prepare
            assert all(
                runtime.record_for(key).cursor is cursor
                for key, cursor in zip(probe_keys, probe_cursors, strict=True)
            )
            assert (
                tuple(runtime.record_for(key).head for key in probe_keys)
                == probe_heads
            )
            assert all(runtime.record_for(key).boundary == 0 for key in probe_keys)
            assert all(runtime.record_for(key).pending is None for key in probe_keys)

            ack_calls = manager.performance_counters[
                "acknowledge_reclamations_batch_calls"
            ]
            assert (
                runtime._pending_relocation_batches.get(publication.batch_id)
                is publication
            )
            runtime.acknowledge_relocation_batch(publication)
            assert (
                manager.performance_counters[
                    "acknowledge_reclamations_batch_calls"
                ]
                == ack_calls + 1
            )
            assert (
                publication.batch_id
                not in runtime._pending_relocation_batches
            )
            acknowledged = runtime.stats()
            assert acknowledged.free_pages == 3 * batch_size
            assert acknowledged.active_pages == 2 * batch_size
            assert acknowledged.retiring_pages == 0
            assert acknowledged.pending_reclamations == 0
            reused += _probe_reuse(
                runtime, probe_keys, source_generations
            )
            assert all(
                runtime.record_for(key).cursor is cursor
                for key, cursor in zip(keys, cursors, strict=True)
            )

        assert len(encoded_payloads) == batch_size * (
            INITIAL_BOUNDARY + (cycles - 1) * APPEND_TOKENS
        )
        before_release = runtime.stats()
        assert before_release.active_pages == 2 * batch_size
        assert before_release.retiring_pages == 0
        assert before_release.pending_reclamations == 0
        runtime.release_batch(keys)
        stats = runtime.stats()
        assert stats.free_pages == 5 * batch_size
        assert stats.active_requests == stats.active_snapshots == 0
        assert stats.active_pages == stats.retiring_pages == 0
        assert stats.pending_reclamations == 0
        assert stats.total_request_page_refs == stats.total_prefix_page_refs == 0
        assert stats.total_reader_pins == 0
        runtime.close()
    finally:
        manager.destroy()
    return ConformanceResult(batch_size, cycles, copied, reused)


__all__ = [
    "ConformanceResult",
    "TOKEN_BYTES",
    "assert_batch_prepare_failure_atomic",
    "opaque_payload_bytes",
    "run_cuda_relocation_conformance",
]
