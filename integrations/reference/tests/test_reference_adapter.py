from __future__ import annotations

from dataclasses import replace

import pytest

from orbitkv_reference import (
    AdapterPreflightError,
    ArenaBinding,
    ReferencePagedAdapter,
)
from orbitkv_runtime import (
    ArenaRegistration,
    BackendTokenAddress,
    CompletionEvidence,
    DataPlaneOperation,
    KvDataPlaneAdapter,
    OperationContext,
    PageLease,
    PageMirrorKey,
    PageMirrorUpdate,
    ReclamationCertificate,
    ReclamationLease,
    RelocationLease,
    RequestLease,
    TokenCopy,
    TokenMirrorKey,
    TokenMirrorUpdate,
    TokenWrite,
    StepLease,
)


PAGE_TOKENS = 4
TOKEN_BYTES = 257


def _registration(*, token_bytes: int = TOKEN_BYTES) -> ArenaRegistration:
    return ArenaRegistration(7, 11, 5, 2, 13, 4, PAGE_TOKENS, token_bytes, 8, 101)


def _registration_for(
    *,
    pool_id: int,
    class_id: int,
    backend_domain: int,
    first_page_id: int,
    backend_base_index: int,
) -> ArenaRegistration:
    return ArenaRegistration(
        7,
        11 + pool_id,
        pool_id,
        class_id,
        backend_domain,
        2,
        PAGE_TOKENS,
        TOKEN_BYTES,
        backend_base_index,
        first_page_id,
    )


def _adapter(*, storage=None, token_bytes: int = TOKEN_BYTES):
    registration = _registration(token_bytes=token_bytes)
    if storage is None:
        storage = bytearray(
            registration.page_count
            * registration.page_tokens
            * registration.token_bytes
        )
    adapter = ReferencePagedAdapter(
        (ArenaBinding(registration, storage),), adapter_id="test-reference"
    )
    return adapter, storage


def _page(index: int, generation: int = 1) -> PageLease:
    return PageLease(7, 11, generation, 101 + index, 5)


def _token(page: int, offset: int, generation: int = 1) -> BackendTokenAddress:
    return BackendTokenAddress(
        _page(page, generation), 2, 13, 8 + page, offset
    )


def _payload(tag: int) -> bytes:
    return bytes(((tag + index * 17) % 256 for index in range(TOKEN_BYTES)))


def _context(
    operation: DataPlaneOperation, *, request: int = 0, transaction: int = 0
) -> OperationContext:
    transaction_type = (
        StepLease if operation is DataPlaneOperation.APPEND else RelocationLease
    )
    return OperationContext(
        RequestLease(7, request, 1),
        transaction_type(7, transaction, 1),
        operation,
    )


def _append(adapter, *items: tuple[int, BackendTokenAddress]):
    transaction = getattr(_append, "next_transaction", 0)
    _append.next_transaction = transaction + 1
    context = _context(DataPlaneOperation.APPEND, transaction=transaction)
    writes = tuple(
        TokenWrite(context, token_id, address, _payload(token_id))
        for token_id, address in items
    )
    return adapter.append(writes, completion_domain=3)


def test_reference_adapter_satisfies_public_spi_and_resolves_external_arena() -> None:
    adapter, storage = _adapter()
    assert isinstance(adapter, KvDataPlaneAdapter)

    page = adapter.resolve_pages((_token(2, 0).page_address,))[0]
    token = adapter.resolve_tokens((_token(2, 3),))[0]

    assert page.arena is storage
    assert page.arena_page_index == 2
    assert page.byte_length == PAGE_TOKENS * TOKEN_BYTES
    assert token.arena is storage
    assert token.token_index == 11
    assert token.byte_length == TOKEN_BYTES


def test_append_in_place_and_partial_tail_cow_are_byte_exact() -> None:
    adapter, _storage = _adapter()
    source0, source1 = _token(0, 0), _token(0, 1)
    _append(adapter, (0, source0), (1, source1))
    _append(adapter, (2, _token(0, 2)))
    context = _context(DataPlaneOperation.APPEND, transaction=777)

    evidence = adapter.append(
        (TokenWrite(context, 4, _token(1, 2), _payload(4)),),
        cow_copies=(
            TokenCopy(context, 0, source0, _token(1, 0)),
            TokenCopy(context, 1, source1, _token(1, 1)),
        ),
        completion_domain=3,
    )

    assert evidence.operation is DataPlaneOperation.APPEND
    assert adapter.wait_completion(evidence.completion).fence == evidence.completion
    assert adapter.read_tokens((source0, source1, _token(0, 2))) == (
        _payload(0),
        _payload(1),
        _payload(2),
    )
    assert adapter.read_tokens((_token(1, 0), _token(1, 1), _token(1, 2))) == (
        _payload(0),
        _payload(1),
        _payload(4),
    )


def test_relocation_gathers_all_sources_before_overlapping_scatter() -> None:
    adapter, _storage = _adapter()
    first, second = _token(0, 0), _token(0, 1)
    _append(adapter, (10, first), (11, second))

    evidence = adapter.relocate(
        (
            TokenCopy(_context(DataPlaneOperation.RELOCATE, transaction=901), 10, first, second),
            TokenCopy(_context(DataPlaneOperation.RELOCATE, transaction=901), 11, second, first),
        ),
        completion_domain=4,
    )

    assert evidence.operation is DataPlaneOperation.RELOCATE
    assert adapter.query_completion(evidence.completion) is not None
    assert adapter.read_tokens((first, second)) == (_payload(11), _payload(10))


def test_batch_preflight_fault_leaves_all_external_bytes_unchanged() -> None:
    adapter, storage = _adapter()
    before = bytes(storage)
    context = _context(DataPlaneOperation.APPEND, transaction=611)

    with pytest.raises(AdapterPreflightError, match="payload"):
        adapter.append(
            (
                TokenWrite(context, 0, _token(0, 0), _payload(0)),
                TokenWrite(context, 1, _token(1, 0), b"short"),
            )
        )

    assert bytes(storage) == before
    _append(adapter, (0, _token(0, 0)))
    assert adapter.read_tokens((_token(0, 0),)) == (_payload(0),)


def test_batch_preflight_rejects_mixed_generations_without_consuming_reuse_state() -> None:
    adapter, storage = _adapter()
    before = bytes(storage)
    context = _context(DataPlaneOperation.APPEND, transaction=612)

    with pytest.raises(AdapterPreflightError, match="multiple generations"):
        adapter.append(
            (
                TokenWrite(context, 0, _token(0, 0, generation=1), _payload(0)),
                TokenWrite(context, 1, _token(0, 1, generation=2), _payload(1)),
            )
        )

    assert bytes(storage) == before
    _append(adapter, (0, _token(0, 0, generation=1)))


def test_duplicate_destination_is_rejected_before_any_write() -> None:
    adapter, storage = _adapter()
    before = bytes(storage)
    destination = _token(0, 0)
    context = _context(DataPlaneOperation.APPEND, transaction=613)

    with pytest.raises(AdapterPreflightError, match="duplicate data destination"):
        adapter.append(
            (
                TokenWrite(context, 0, destination, _payload(0)),
                TokenWrite(context, 1, destination, _payload(1)),
            )
        )

    assert bytes(storage) == before


def test_multi_request_batch_allows_repeated_token_ids_but_binds_receipts() -> None:
    adapter, _storage = _adapter()
    first = _context(DataPlaneOperation.APPEND, request=0, transaction=101)
    second = _context(DataPlaneOperation.APPEND, request=1, transaction=102)

    evidence = adapter.append(
        (
            TokenWrite(first, 0, _token(0, 0), _payload(1)),
            TokenWrite(second, 0, _token(1, 0), _payload(2)),
        )
    )

    assert tuple(receipt.context for receipt in evidence.writes) == (first, second)
    assert tuple(receipt.token_id for receipt in evidence.writes) == (0, 0)
    assert adapter.read_tokens((_token(0, 0), _token(1, 0))) == (
        _payload(1),
        _payload(2),
    )


def test_transaction_replay_and_wrong_operation_fail_before_write() -> None:
    adapter, storage = _adapter()
    context = _context(DataPlaneOperation.APPEND, transaction=301)
    adapter.append((TokenWrite(context, 0, _token(0, 0), _payload(0)),))
    before = bytes(storage)

    with pytest.raises(AdapterPreflightError, match="already observed"):
        adapter.append((TokenWrite(context, 1, _token(0, 1), _payload(1)),))
    wrong = _context(DataPlaneOperation.RELOCATE, transaction=302)
    with pytest.raises(AdapterPreflightError, match="context"):
        adapter.append((TokenWrite(wrong, 1, _token(0, 1), _payload(1)),))

    assert bytes(storage) == before


def test_copy_across_pool_class_or_domain_is_rejected_before_write() -> None:
    first_registration = _registration_for(
        pool_id=5, class_id=2, backend_domain=13, first_page_id=101, backend_base_index=8
    )
    second_registration = _registration_for(
        pool_id=6, class_id=3, backend_domain=14, first_page_id=201, backend_base_index=18
    )
    first_storage = bytearray(2 * PAGE_TOKENS * TOKEN_BYTES)
    second_storage = bytearray(2 * PAGE_TOKENS * TOKEN_BYTES)
    adapter = ReferencePagedAdapter(
        (
            ArenaBinding(first_registration, first_storage),
            ArenaBinding(second_registration, second_storage),
        ),
        adapter_id="two-pool",
    )
    source = BackendTokenAddress(PageLease(7, 16, 1, 101, 5), 2, 13, 8, 0)
    destination = BackendTokenAddress(
        PageLease(7, 17, 1, 201, 6), 3, 14, 18, 0
    )
    context = _context(DataPlaneOperation.RELOCATE, transaction=401)

    with pytest.raises(AdapterPreflightError, match="cannot cross"):
        adapter.relocate((TokenCopy(context, 0, source, destination),))

    assert bytes(first_storage) == bytes(len(first_storage))
    assert bytes(second_storage) == bytes(len(second_storage))


def test_mirror_batch_is_compare_and_swap_and_failure_is_atomic() -> None:
    adapter, _storage = _adapter()
    data = _append(adapter, (0, _token(0, 0)))
    ready = adapter.wait_completion(data.completion)
    page_key = PageMirrorKey("request", 2, 0)
    token_key = TokenMirrorKey("request", 2, 0)
    page = _token(0, 0).page_address
    installed = adapter.update_mirrors(
        (PageMirrorUpdate(page_key, None, page),),
        (TokenMirrorUpdate(token_key, None, _token(0, 0)),),
        after=ready,
        completion_domain=3,
    )
    adapter.wait_completion(installed.completion)

    with pytest.raises(AdapterPreflightError, match="covered|compare-and-swap"):
        adapter.update_mirrors(
            (
                PageMirrorUpdate(page_key, None, _token(1, 0).page_address),
                PageMirrorUpdate(PageMirrorKey("request", 2, 1), None, _token(1, 0).page_address),
            ),
            (),
            after=adapter.wait_completion(installed.completion),
        )

    assert adapter.page_mirrors == {page_key: page}
    assert adapter.token_mirrors == {token_key: _token(0, 0)}

    replaced = adapter.update_mirrors(
        (PageMirrorUpdate(page_key, page, None),),
        (TokenMirrorUpdate(token_key, _token(0, 0), None),),
        after=adapter.wait_completion(installed.completion),
    )
    adapter.wait_completion(replaced.completion)
    assert page_key not in adapter.page_mirrors
    assert token_key not in adapter.token_mirrors


def test_completion_domains_are_monotonic_and_foreign_fences_fail_closed() -> None:
    adapter, _storage = _adapter()
    page = _token(0, 0).page_address
    first = adapter.record_completion((page,), completion_domain=9)
    other = adapter.record_completion((page,), completion_domain=10)
    second = adapter.record_completion((page,), completion_domain=9)

    assert (first.completion_value, second.completion_value) == (1, 2)
    assert other.completion_value == 1
    forged = replace(first, adapter_id="foreign")
    with pytest.raises(AdapterPreflightError, match="another adapter"):
        adapter.wait_completion(forged)


def test_same_label_adapters_still_reject_each_others_fences() -> None:
    first, _first_storage = _adapter()
    second, _second_storage = _adapter()
    first_fence = first.record_completion(
        (_token(0, 0).page_address,), completion_domain=9
    )
    second_fence = second.record_completion(
        (_token(0, 0).page_address,), completion_domain=9
    )

    assert first.adapter_id != second.adapter_id
    assert first_fence != second_fence
    with pytest.raises(AdapterPreflightError, match="another adapter"):
        second.wait_completion(first_fence)


def test_publicly_constructed_completion_wrapper_cannot_confirm_a_fence() -> None:
    adapter, _storage = _adapter()
    old = _token(0, 0)
    data = _append(adapter, (0, old))
    forged = CompletionEvidence(data.completion, (old.page_address,))

    with pytest.raises(AdapterPreflightError, match="not confirmed"):
        adapter.update_mirrors((), (), after=forged)

    confirmed = adapter.wait_completion(data.completion)
    forged_coverage = replace(confirmed, covered_pages=())
    with pytest.raises(AdapterPreflightError, match="not confirmed"):
        adapter.update_mirrors((), (), after=forged_coverage)


def test_exact_completion_and_mirror_evidence_gate_higher_generation_reuse() -> None:
    adapter, _storage = _adapter()
    old = _token(0, 0)
    data = _append(adapter, (0, old))
    data_ready = adapter.wait_completion(data.completion)
    key = PageMirrorKey("request", 2, 0)
    installed = adapter.update_mirrors(
        (PageMirrorUpdate(key, None, old.page_address),),
        (),
        after=data_ready,
    )
    adapter.wait_completion(installed.completion)

    last_use_fence = adapter.record_completion(
        (old.page_address,), completion_domain=17
    )
    last_use = adapter.wait_completion(last_use_fence)
    cleanup = adapter.update_mirrors(
        (PageMirrorUpdate(key, old.page_address, None),),
        (),
        after=last_use,
        completion_domain=17,
    )
    cleanup_done = adapter.wait_completion(cleanup.completion)
    certificate = ReclamationCertificate(
        ReclamationLease(7, 1, 1),
        old.page,
        2,
        13,
        0,
        8,
        0,
        PAGE_TOKENS,
        17,
        last_use_fence.completion_value,
    )

    evidence = adapter.prepare_reuse(
        (certificate,), last_use=(last_use,), mirror_cleanup=cleanup_done
    )
    with pytest.raises(AdapterPreflightError, match="before its manager ACK"):
        adapter.resolve_tokens((old,))
    with pytest.raises(AdapterPreflightError, match="before its manager ACK"):
        adapter.resolve_tokens((_token(0, 0, generation=2),))

    adapter.note_reuse_acknowledged(evidence)
    with pytest.raises(AdapterPreflightError, match="higher generation"):
        adapter.resolve_tokens((old,))
    new = adapter.resolve_tokens((_token(0, 0, generation=2),))[0]
    assert new.address.page.generation == 2
    with pytest.raises(AdapterPreflightError, match="already acknowledged"):
        adapter.note_reuse_acknowledged(evidence)


def test_reuse_rejects_access_after_last_use_and_value_equal_cloned_evidence() -> None:
    adapter, _storage = _adapter()
    old = _token(0, 0)
    _append(adapter, (0, old))
    last_use = adapter.wait_completion(
        adapter.record_completion((old.page_address,), completion_domain=19)
    )
    adapter.read_tokens((old,))
    with pytest.raises(AdapterPreflightError, match="stale completion"):
        adapter.update_mirrors((), (), after=last_use, completion_domain=19)

    fresh_last_use = adapter.wait_completion(
        adapter.record_completion((old.page_address,), completion_domain=19)
    )
    cleanup = adapter.update_mirrors(
        (), (), after=fresh_last_use, completion_domain=19
    )
    cleanup_done = adapter.wait_completion(cleanup.completion)
    certificate = ReclamationCertificate(
        ReclamationLease(7, 0, 1),
        old.page,
        2,
        13,
        0,
        8,
        0,
        PAGE_TOKENS,
        19,
        fresh_last_use.fence.completion_value,
    )
    evidence = adapter.prepare_reuse(
        (certificate,),
        last_use=(fresh_last_use,),
        mirror_cleanup=cleanup_done,
    )
    clone = replace(evidence)

    with pytest.raises(AdapterPreflightError, match="forged"):
        adapter.note_reuse_acknowledged(clone)
    adapter.note_reuse_acknowledged(evidence)


class _FakeCudaEvent:
    next_id = 1

    def __init__(self) -> None:
        self.event_id = self.next_id
        _FakeCudaEvent.next_id += 1
        self.recorded_on = None

    def record(self, stream) -> None:
        self.recorded_on = stream

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _FakeCudaStream:
    def __init__(self) -> None:
        self.waited = []

    def wait_event(self, event) -> None:
        self.waited.append(event)


class _FakeCudaModule:
    def __init__(self) -> None:
        self.stream = _FakeCudaStream()

    def current_stream(self, _device):
        return self.stream

    @staticmethod
    def Event():
        return _FakeCudaEvent()


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCudaModule()


class _FakeCudaArena:
    is_cuda = True
    device = "cuda:0"

    def __init__(self) -> None:
        self._torch = _FakeTorch()


def test_cuda_domain_waits_on_predecessor_and_rejects_device_rebinding() -> None:
    adapter, _storage = _adapter()
    arena = _FakeCudaArena()
    first_timeline = adapter._prepare_timeline(31, (arena,))
    first = adapter._record_completion(first_timeline, purpose="test")
    second_timeline = adapter._prepare_timeline(31, (arena,))

    assert arena._torch.cuda.stream.waited == [
        adapter._fences[first.fence_id].event
    ]
    adapter._record_completion(second_timeline, purpose="test")
    with pytest.raises(AdapterPreflightError, match="device timeline"):
        adapter._prepare_timeline(31, ())


def test_torch_device_other_than_cpu_or_cuda_is_rejected(monkeypatch) -> None:
    class FakeTensor:
        pass

    FakeTensor.__module__ = "torch"
    fake = FakeTensor()
    fake.layout = object()
    fake.device = type("Device", (), {"type": "mps"})()
    fake.requires_grad = False
    fake.is_contiguous = lambda: True
    fake.numel = lambda: 4 * PAGE_TOKENS * TOKEN_BYTES
    fake.element_size = lambda: 1
    fake.detach = lambda: fake
    fake.view = lambda _dtype: fake
    fake.reshape = lambda *_shape: fake
    fake_torch = type(
        "FakeTorchModule",
        (),
        {"Tensor": FakeTensor, "strided": fake.layout, "uint8": object()},
    )()
    monkeypatch.setattr(
        "orbitkv_reference.adapter.importlib.import_module",
        lambda name: fake_torch if name == "torch" else None,
    )

    with pytest.raises(AdapterPreflightError, match="CPU or CUDA"):
        _adapter(storage=fake)


def test_reuse_rejects_data_ready_in_place_of_explicit_last_use() -> None:
    adapter, _storage = _adapter()
    old = _token(0, 0)
    data = _append(adapter, (0, old))
    data_ready = adapter.wait_completion(data.completion)
    cleanup = adapter.update_mirrors((), (), after=data_ready, completion_domain=3)
    cleanup_done = adapter.wait_completion(cleanup.completion)
    certificate = ReclamationCertificate(
        ReclamationLease(7, 1, 1),
        old.page,
        2,
        13,
        0,
        8,
        0,
        PAGE_TOKENS,
        data.completion.completion_domain,
        data.completion.completion_value,
    )

    with pytest.raises(AdapterPreflightError, match="last-use"):
        adapter.prepare_reuse(
            (certificate,), last_use=(data_ready,), mirror_cleanup=cleanup_done
        )


def test_optional_torch_cpu_arena_matches_bytearray_contract() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(
        (4, PAGE_TOKENS, TOKEN_BYTES), dtype=torch.uint8, device="cpu"
    )
    adapter, _storage = _adapter(storage=tensor)

    _append(adapter, (0, _token(0, 0)), (1, _token(0, 1)))

    assert adapter.read_tokens((_token(0, 0), _token(0, 1))) == (
        _payload(0),
        _payload(1),
    )
