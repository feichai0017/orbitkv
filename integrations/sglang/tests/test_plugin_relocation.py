from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import orbitkv_sglang.plugin.lowering as lowering
import orbitkv_sglang.plugin.private_prefix as private_prefix
import orbitkv_sglang.plugin.relocation as relocation
import orbitkv_sglang.plugin.state as state
from orbitkv_sglang.runtime import (
    ArenaIdentity,
    FailStopped,
    PageLease,
    RelocationBatchPublication,
    RelocationLease,
    RelocationPublication,
    RequestLease,
    SnapshotLease,
    TokenLocation,
    TokenMove,
    TokenPlacement,
    TokenView,
)
from orbitkv_sglang.runtime.token_relocation import (
    TokenDisposition,
    TokenDispositionKind,
)


class _Pool:
    def __init__(self) -> None:
        self.req_to_token = torch.zeros((2, 64), dtype=torch.int32)

    def write(self, indices: Any, values: Any) -> None:
        self.req_to_token[indices] = values


def _token_view(
    placements: tuple[tuple[int, int | None, TokenDispositionKind], ...],
    *,
    class_id: int = 0,
    view_version: int = 1,
) -> TokenView:
    return TokenView(
        class_id,
        view_version,
        16,
        tuple(
            TokenPlacement(
                token_id,
                TokenDisposition(
                    kind,
                    0 if kind is TokenDispositionKind.RETAINED else 7,
                    0 if kind is TokenDispositionKind.RETAINED else 1,
                    0 if kind is TokenDispositionKind.RETAINED else 99,
                ),
                None
                if location is None
                else TokenLocation(
                    PageLease(1, 2 + class_id, 1, location // 16, class_id + 1),
                    location // 16 - 1,
                    location % 16,
                ),
            )
            for token_id, location, kind in placements
        ),
    )


def _all_retained_view(
    boundary: int, *, class_id: int = 0, view_version: int = 1
) -> TokenView:
    return _token_view(
        tuple(
            (token_id, 16 + token_id, TokenDispositionKind.RETAINED)
            for token_id in range(boundary)
        ),
        class_id=class_id,
        view_version=view_version,
    )


def _hybrid_reclamation_case(monkeypatch):
    classes = (
        SimpleNamespace(class_id=0, retention="full"),
        SimpleNamespace(class_id=1, retention="sliding"),
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=classes,
        full_class=classes[0],
        sliding_class=classes[1],
    )
    arenas = {
        0: ArenaIdentity(1, 2, 1, 0, 10, 8, 16, 0, 1),
        1: ArenaIdentity(1, 3, 2, 1, 11, 8, 16, 100, 9),
    }
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(
        relocation, "_runtime", lambda: SimpleNamespace(arenas_by_class=arenas)
    )
    mapping = torch.zeros(256, dtype=torch.int64)
    monkeypatch.setattr(
        state,
        "_ALLOCATOR",
        SimpleNamespace(full_to_swa_index_mapping=mapping),
    )

    def view(class_id: int, backend_base: int, first_page: int) -> TokenView:
        placements = []
        for token_id in range(48):
            ordinal, offset = divmod(token_id, 16)
            placements.append(
                TokenPlacement(
                    token_id,
                    TokenDisposition(TokenDispositionKind.RETAINED),
                    TokenLocation(
                        PageLease(
                            1, 2 + class_id, 1, first_page + ordinal, class_id + 1
                        ),
                        backend_base + ordinal,
                        offset,
                    ),
                )
            )
        return TokenView(class_id, 1, 16, tuple(placements))

    full_view = view(0, 0, 1)
    swa_view = view(1, 100, 9)
    old_full = relocation._view_locations(full_view)
    old_swa = relocation._view_locations(swa_view)
    mapping[torch.tensor(old_full)] = torch.tensor(old_swa)
    row = torch.tensor(old_full + (0,) * 16, dtype=torch.int32)
    return mapping, full_view, swa_view, old_full, old_swa, row


def test_decode_after_reclamation_writes_active_tail_not_absolute_column(
    monkeypatch,
) -> None:
    req = SimpleNamespace(
        rid="request",
        kv=SimpleNamespace(kv_allocated_len=48),
        _orbitkv_active_kv_len=24,
        _orbitkv_retained_locations=tuple(range(100, 124)),
    )
    pool = _Pool()
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=pool,
        req_pool_indices=torch.tensor([1]),
        device=torch.device("cpu"),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        maybe_evict_swa=lambda: None,
    )
    location = torch.tensor([777], dtype=torch.int64)
    runtime = SimpleNamespace(
        mark_lowered=lambda _batch: None,
        lowering_failed=lambda _batch, error: (_ for _ in ()).throw(error),
        candidate_mirror_failed=lambda _batch, error: (_ for _ in ()).throw(error),
        failure_reason=None,
    )
    monkeypatch.setattr(lowering, "_validate_batch", lambda _batch: None)
    monkeypatch.setattr(
        lowering,
        "_config",
        lambda: SimpleNamespace(
            sliding_class=None,
            full_class=SimpleNamespace(class_id=0),
            token_reclamation=SimpleNamespace(mode="relocate"),
        ),
    )
    monkeypatch.setattr(
        lowering, "_preflight_decode_batch", lambda _batch: ((48,), (1,))
    )
    monkeypatch.setattr(
        lowering, "_prepare_batch", lambda *_args: (object(), (object(),))
    )
    monkeypatch.setattr(lowering, "_lower_all_decode", lambda *_args: {0: location})
    monkeypatch.setattr(lowering, "_primary_locations", lambda values: values[0])
    monkeypatch.setattr(lowering, "_validate_joint_hybrid_tails", lambda _plans: None)
    monkeypatch.setattr(lowering, "_preflight_cow_mirrors", lambda *_args: object())
    monkeypatch.setattr(lowering, "_execute_cow_copies", lambda *_args: (0, 0, 0))
    monkeypatch.setattr(lowering, "_runtime", lambda: runtime)
    monkeypatch.setattr(lowering, "_submit_batch", lambda _batch: ())
    monkeypatch.setattr(lowering, "_write_hybrid_lut", lambda _locations: None)
    monkeypatch.setattr(lowering, "_commit_cow_mirrors", lambda _plan: None)

    lowering._alloc_for_decode(batch, 1)

    assert pool.req_to_token[1, 24].item() == 777
    assert pool.req_to_token[1, 48].item() == 0
    assert req._orbitkv_active_kv_len == 25
    assert req._orbitkv_retained_locations[-1] == 777


def test_victim_updates_use_current_retained_active_ordinals(monkeypatch) -> None:
    full = SimpleNamespace(class_id=0)
    sliding = SimpleNamespace(class_id=1)
    policy = SimpleNamespace(
        retained_per_page=8,
        policy_id=7,
        policy_version=2,
        quality_contract=99,
    )
    monkeypatch.setattr(
        relocation,
        "_config",
        lambda: SimpleNamespace(
            page_tokens=16,
            classes=(full, sliding),
            full_class=full,
            token_reclamation=policy,
        ),
    )
    placements = [(0, None, TokenDispositionKind.POLICY_EVICTED)]
    placements.extend(
        (token_id, 16 + ((token_id * 7) % 32), TokenDispositionKind.RETAINED)
        for token_id in range(1, 18)
    )
    placements.append((18, 40, TokenDispositionKind.POLICY_EVICTED))
    view = _token_view(tuple(placements))

    updates = relocation._victim_updates(view)

    assert tuple((item.class_id, item.token_id) for item in updates) == (
        *((0, token_id) for token_id in range(9, 17)),
        *((1, token_id) for token_id in range(9, 17)),
    )
    assert all(
        item.disposition.kind is TokenDispositionKind.POLICY_EVICTED
        for item in updates
    )


@pytest.mark.parametrize("mode", ["naive", "relocate"])
@pytest.mark.parametrize("private_mirror", [False, True])
def test_reclamation_repeats_when_active_length_reaches_trigger(
    monkeypatch, mode: str, private_mirror: bool
) -> None:
    full = SimpleNamespace(class_id=0)
    policy = SimpleNamespace(
        mode=mode,
        trigger_tokens=48,
        retained_per_page=8,
        policy_id=7,
        policy_version=1,
        quality_contract=99,
        maximum_source_pages=3,
        evacuation_headroom_pages=2,
        fragmentation_threshold_milli=250,
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(full,),
        full_class=full,
        sliding_class=None,
        token_reclamation=policy,
    )
    arena = ArenaIdentity(1, 2, 1, 0, 10, 16, 16, 0, 1)
    lease = RequestLease(1, 7, 1)
    record = SimpleNamespace(boundary=48, lease=lease)
    req = SimpleNamespace(
        rid="request",
        kv=SimpleNamespace(kv_allocated_len=48),
        req_pool_idx=1,
        prefix_indices=torch.empty(0, dtype=torch.int64),
    )
    pool = _Pool()
    pool.req_to_token = torch.zeros((2, 96), dtype=torch.int32)
    pool.req_to_token[1, :48] = torch.arange(16, 64, dtype=torch.int32)
    if private_mirror:
        key = ("str", req.rid)
        req.prefix_indices = pool.req_to_token[1, :48].to(torch.int64).clone()
        req.cache_protected_len = 0
        req._orbitkv_request_key = key
        req._orbitkv_request_lease = lease
        req._orbitkv_private_prefix = private_prefix.PrivatePrefixProvenance(
            req.prefix_indices, key, lease, 48
        )
    batch = SimpleNamespace(
        reqs=[req], req_to_token_pool=pool, device=torch.device("cpu")
    )
    current_view = _all_retained_view(48)
    update_rounds = []
    relocation_batch_calls = []
    acknowledged = []

    class _Runtime:
        arenas_by_class = {0: arena}
        failure_reason = None

        def has_request(self, key):
            return key == ("str", "request")

        def record_for(self, _key):
            return record

        def active_kv_length(self, _key, _class_id):
            return sum(
                placement.disposition.kind is TokenDispositionKind.RETAINED
                for placement in current_view.placements
            )

        def token_view(self, _key, _class_id):
            return current_view

        def mark_token_dispositions(self, _key, class_id, updates):
            nonlocal current_view
            assert class_id == 0
            victim_ids = {item.token_id for item in updates}
            update_rounds.append(tuple(sorted(victim_ids)))
            placements = tuple(
                TokenPlacement(
                    placement.token_id,
                    (
                        TokenDisposition(TokenDispositionKind.POLICY_EVICTED, 7, 1, 99)
                        if placement.token_id in victim_ids
                        else placement.disposition
                    ),
                    placement.location,
                )
                for placement in current_view.placements
            )
            current_view = TokenView(0, current_view.view_version + 1, 16, placements)
            retained_locations = relocation._view_locations(current_view)
            return SimpleNamespace(
                class_retained_locations=((0, retained_locations),)
            )

        def relocate_tokens_batch(self, items, _copy):
            assert len(items) == 1
            relocation_batch_calls.append(tuple(items))
            item = items[0]
            disposition = self.mark_token_dispositions(
                item.key, item.class_id, item.updates
            )
            prepared = SimpleNamespace(moves=(), projected_reclaimed_pages=1)
            publication = RelocationPublication(
                item.key,
                current_view,
                prepared,
                SimpleNamespace(),
                tuple(dict(disposition.class_retained_locations)[item.class_id]),
                disposition.class_retained_locations,
                (),
            )
            return RelocationBatchPublication((publication,), ())

        def acknowledge_relocation_batch(self, publication):
            acknowledged.append(publication)

        def fail_stop(self, reason):
            self.failure_reason = reason

    runtime = _Runtime()
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", lambda: runtime)
    device_module = SimpleNamespace(
        current_stream=lambda _device: SimpleNamespace(synchronize=lambda: None)
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    before = state._activity_counters()

    relocation._maybe_reclaim_decode_batch(batch)

    first_victims = tuple(
        token_id for token_id in range(48) if token_id % 16 >= 8
    )
    assert update_rounds == [first_victims]
    assert req._orbitkv_active_kv_len == 24
    assert req._orbitkv_token_reclamation_next_boundary == 72
    assert not hasattr(req, "_orbitkv_token_reclamation_done")
    assert tuple(pool.req_to_token[1, :24].tolist()) == tuple(
        16 + token_id for token_id in range(48) if token_id % 16 < 8
    )
    assert torch.count_nonzero(pool.req_to_token[1, 24:48]).item() == 0
    if private_mirror:
        assert torch.equal(req.prefix_indices, pool.req_to_token[1, :24].long())
        assert req._orbitkv_private_prefix.tensor is req.prefix_indices

    appended = tuple(range(64, 88))
    pool.req_to_token[1, 24:48] = torch.tensor(appended, dtype=torch.int32)
    req._orbitkv_retained_locations += appended
    req._orbitkv_active_kv_len = 48
    req.kv.kv_allocated_len = 72
    record.boundary = 72
    prior = current_view.placements
    current_view = _token_view(
        tuple(
            (
                placement.token_id,
                (
                    None
                    if placement.disposition.kind is not TokenDispositionKind.RETAINED
                    else 16 + placement.token_id
                ),
                placement.disposition.kind,
            )
            for placement in prior
        )
        + tuple(
            (token_id, location, TokenDispositionKind.RETAINED)
            for token_id, location in zip(range(48, 72), appended, strict=True)
        ),
        view_version=2,
    )

    relocation._maybe_reclaim_decode_batch(batch)

    assert update_rounds == [
        first_victims,
        tuple(range(16, 24)) + tuple(range(48, 56)) + tuple(range(64, 72)),
    ]
    assert req._orbitkv_active_kv_len == 24
    assert req._orbitkv_token_reclamation_next_boundary == 96
    assert not hasattr(req, "_orbitkv_token_reclamation_done")
    assert torch.count_nonzero(pool.req_to_token[1, 24:72]).item() == 0
    if private_mirror:
        assert torch.equal(req.prefix_indices, pool.req_to_token[1, :24].long())
        assert req._orbitkv_private_prefix.tensor is req.prefix_indices
        assert req._orbitkv_private_prefix.boundary == 72
    after = state._activity_counters()
    assert after["token_disposition_batches"] - before["token_disposition_batches"] == 2
    assert after["token_policy_evictions"] - before["token_policy_evictions"] == 48
    assert len(acknowledged) == (2 if mode == "relocate" else 0)
    assert len(relocation_batch_calls) == (2 if mode == "relocate" else 0)
    assert after["relocation_batches"] - before["relocation_batches"] == (
        2 if mode == "relocate" else 0
    )
    assert req.skip_radix_cache_insert is True


def test_reclamation_fails_closed_after_missing_next_boundary(monkeypatch) -> None:
    full = SimpleNamespace(class_id=0)
    config = SimpleNamespace(
        classes=(full,),
        full_class=full,
        token_reclamation=SimpleNamespace(mode="naive", trigger_tokens=48),
    )
    record = SimpleNamespace(boundary=73)
    req = SimpleNamespace(
        rid="request",
        kv=SimpleNamespace(kv_allocated_len=73),
        _orbitkv_active_kv_len=49,
        _orbitkv_token_reclamation_next_boundary=72,
    )
    runtime = SimpleNamespace(
        has_request=lambda _key: True,
        record_for=lambda _key: record,
        active_kv_length=lambda _key, _class_id: 49,
    )
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match="passed the next reclamation boundary"):
        relocation._maybe_reclaim_decode_batch(SimpleNamespace(reqs=[req]))


def test_reclamation_rejects_prefix_before_manager_mutation(monkeypatch) -> None:
    full = SimpleNamespace(class_id=0)
    policy = SimpleNamespace(
        mode="naive",
        trigger_tokens=48,
        retained_per_page=8,
        policy_id=7,
        policy_version=1,
        quality_contract=99,
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(full,),
        full_class=full,
        sliding_class=None,
        token_reclamation=policy,
    )
    arena = ArenaIdentity(1, 2, 1, 0, 10, 16, 16, 0, 1)
    record = SimpleNamespace(boundary=48)
    mutated = []
    runtime = SimpleNamespace(
        arenas_by_class={0: arena},
        has_request=lambda _key: True,
        record_for=lambda _key: record,
        active_kv_length=lambda _key, _class_id: 48,
        token_view=lambda _key, _class_id: _all_retained_view(48),
        mark_token_dispositions=lambda *_args: mutated.append(True),
    )
    req = SimpleNamespace(
        rid="request",
        kv=SimpleNamespace(kv_allocated_len=48),
        req_pool_idx=1,
        prefix_indices=torch.tensor([16], dtype=torch.int64),
    )
    pool = _Pool()
    pool.req_to_token[1, :48] = torch.arange(16, 64, dtype=torch.int32)
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match="forbids Prefix mirrors"):
        relocation._maybe_reclaim_decode_batch(
            SimpleNamespace(
                reqs=[req],
                req_to_token_pool=pool,
                device=torch.device("cpu"),
            )
        )
    assert mutated == []


@pytest.mark.parametrize("mode", ["naive", "relocate"])
@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("mirror", "authority mirror"),
        ("prefix", "forbids Prefix mirrors"),
        ("victim", "configured victim policy produced no updates"),
    ],
)
def test_batch_reclamation_preflights_later_candidate_before_native_mutation(
    monkeypatch, mode: str, fault: str, message: str
) -> None:
    full = SimpleNamespace(class_id=0)
    policy = SimpleNamespace(
        mode=mode,
        trigger_tokens=48,
        retained_per_page=8,
        policy_id=7,
        policy_version=1,
        quality_contract=99,
        maximum_source_pages=3,
        evacuation_headroom_pages=2,
        fragmentation_threshold_milli=250,
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(full,),
        full_class=full,
        sliding_class=None,
        token_reclamation=policy,
    )
    arena = ArenaIdentity(1, 2, 1, 0, 10, 16, 16, 0, 1)
    requests = [
        SimpleNamespace(
            rid=f"request-{index}",
            kv=SimpleNamespace(kv_allocated_len=48),
            req_pool_idx=index + 1,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )
        for index in range(2)
    ]
    if fault == "prefix":
        requests[1].prefix_indices = torch.tensor([16], dtype=torch.int64)
    keys = tuple(("str", req.rid) for req in requests)
    records = {key: SimpleNamespace(boundary=48) for key in keys}
    views = {key: _all_retained_view(48) for key in keys}
    native_mutations = []

    class _Runtime:
        arenas_by_class = {0: arena}
        failure_reason = None

        def has_request(self, key):
            return key in records

        def record_for(self, key):
            return records[key]

        def active_kv_length(self, _key, _class_id):
            return 48

        def token_view(self, key, _class_id):
            return views[key]

        def mark_token_dispositions(self, key, _class_id, _updates):
            native_mutations.append(("naive", key))

        def relocate_tokens(self, key, _class_id, _updates, *_args):
            native_mutations.append(("relocate", key))

    pool = _Pool()
    pool.req_to_token = torch.zeros((3, 64), dtype=torch.int32)
    for row_index in (1, 2):
        pool.req_to_token[row_index, :48] = torch.arange(16, 64, dtype=torch.int32)
    if fault == "mirror":
        pool.req_to_token[2, 47] += 1
    rows_before = pool.req_to_token.clone()
    original_victim_updates = relocation._victim_updates
    victim_calls = 0

    if fault == "victim":

        def victim_updates(view):
            nonlocal victim_calls
            victim_calls += 1
            if victim_calls == 2:
                return ()
            return original_victim_updates(view)

        monkeypatch.setattr(relocation, "_victim_updates", victim_updates)
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", _Runtime)

    with pytest.raises(RuntimeError, match=message):
        relocation._maybe_reclaim_decode_batch(
            SimpleNamespace(
                reqs=requests,
                req_to_token_pool=pool,
                device=torch.device("cpu"),
            )
        )

    assert native_mutations == []
    assert torch.equal(pool.req_to_token, rows_before)
    for req in requests:
        assert req.kv.kv_allocated_len == 48
        assert not hasattr(req, "_orbitkv_active_kv_len")
        assert not hasattr(req, "_orbitkv_token_reclamation_next_boundary")
        assert not hasattr(req, "skip_radix_cache_insert")


@pytest.mark.parametrize("mode", ["naive", "relocate"])
def test_batch_reclamation_rejects_duplicate_rows_before_native_mutation(
    monkeypatch, mode: str
) -> None:
    full = SimpleNamespace(class_id=0)
    policy = SimpleNamespace(
        mode=mode,
        trigger_tokens=48,
        retained_per_page=8,
        policy_id=7,
        policy_version=1,
        quality_contract=99,
        maximum_source_pages=3,
        evacuation_headroom_pages=2,
        fragmentation_threshold_milli=250,
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(full,),
        full_class=full,
        sliding_class=None,
        token_reclamation=policy,
    )
    requests = tuple(
        SimpleNamespace(
            rid=f"request-{index}",
            kv=SimpleNamespace(kv_allocated_len=48),
            req_pool_idx=1,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )
        for index in range(2)
    )
    keys = tuple(("str", req.rid) for req in requests)
    native_mutations = []
    runtime = SimpleNamespace(
        arenas_by_class={0: ArenaIdentity(1, 2, 1, 0, 10, 16, 16, 0, 1)},
        has_request=lambda key: key in keys,
        record_for=lambda _key: SimpleNamespace(boundary=48),
        active_kv_length=lambda _key, _class_id: 48,
        token_view=lambda _key, _class_id: _all_retained_view(48),
        mark_token_dispositions=lambda *_args: native_mutations.append("naive"),
        relocate_tokens_batch=lambda *_args: native_mutations.append("relocate"),
    )
    pool = _Pool()
    pool.req_to_token = torch.zeros((3, 64), dtype=torch.int32)
    pool.req_to_token[1, :48] = torch.arange(16, 64, dtype=torch.int32)
    rows_before = pool.req_to_token.clone()
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match="aliases a ReqToToken row"):
        relocation._maybe_reclaim_decode_batch(
            SimpleNamespace(
                reqs=list(requests),
                req_to_token_pool=pool,
                device=torch.device("cpu"),
            )
        )

    assert native_mutations == []
    assert torch.equal(pool.req_to_token, rows_before)


def _full_relocation_batch_case(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    *,
    invalid_publication_index: int | None = None,
    moves_per_item: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    if moves_per_item is None:
        moves_per_item = (1,) * batch_size
    assert len(moves_per_item) == batch_size
    full = SimpleNamespace(class_id=0, storage="kv", name="full")
    policy = SimpleNamespace(
        mode="relocate",
        trigger_tokens=48,
        retained_per_page=8,
        policy_id=7,
        policy_version=1,
        quality_contract=99,
        maximum_source_pages=3,
        evacuation_headroom_pages=2,
        fragmentation_threshold_milli=250,
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(full,),
        classes_by_id={0: full},
        full_class=full,
        sliding_class=None,
        token_reclamation=policy,
    )
    arena = ArenaIdentity(1, 2, 1, 0, 10, 64, 16, 0, 1)
    requests = tuple(
        SimpleNamespace(
            rid=f"request-{index}",
            kv=SimpleNamespace(kv_allocated_len=48),
            req_pool_idx=index + 1,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )
        for index in range(batch_size)
    )
    keys = tuple(("str", req.rid) for req in requests)
    records = {key: SimpleNamespace(boundary=48) for key in keys}
    views = {key: _all_retained_view(48) for key in keys}
    new_locations = tuple(
        tuple(80 + index * 32 + ordinal for ordinal in range(24))
        for index in range(batch_size)
    )
    relocation_calls = []
    copied_batches = []
    acknowledgements = []
    activity = []

    def location(page: int, offset: int) -> TokenLocation:
        return TokenLocation(
            PageLease(1, 2, 1, page, 1), page - 1, offset
        )

    class _Runtime:
        arenas_by_class = {0: arena}
        failure_reason = None

        def has_request(self, key):
            return key in records

        def record_for(self, key):
            return records[key]

        def active_kv_length(self, _key, _class_id):
            return 48

        def token_view(self, key, _class_id):
            return views[key]

        def relocate_tokens_batch(self, items, copy):
            values = tuple(items)
            relocation_calls.append(values)
            prepared = tuple(
                SimpleNamespace(
                    class_id=0,
                    relocation=RelocationLease(1, index, 1),
                    moves=tuple(
                        TokenMove(
                            index * 10 + move_index,
                            location(1 + index * 4, move_index),
                            location(32 + index * 4, move_index),
                        )
                        for move_index in range(moves_per_item[index])
                    ),
                    projected_reclaimed_pages=1,
                )
                for index in range(batch_size)
            )
            copied = copy(prepared)
            copied_batches.append(copied)
            publications = []
            for index, (item, prepared_item) in enumerate(
                zip(values, prepared, strict=True)
            ):
                locations = new_locations[index]
                if index == invalid_publication_index:
                    locations = locations[:-1] + (-1,)
                publications.append(
                    RelocationPublication(
                        item.key,
                        views[item.key],
                        prepared_item,
                        SimpleNamespace(),
                        locations,
                        ((0, locations),),
                        (),
                    )
                )
            return RelocationBatchPublication(tuple(publications), ())

        def acknowledge_relocation_batch(self, publication):
            activity.append("ack")
            acknowledgements.append(publication)

        def fail_stop(self, reason):
            self.failure_reason = reason

    class _Event:
        def record(self, *, stream):
            assert isinstance(stream, _Stream)
            activity.append(f"{stream.kind}-record")

        def synchronize(self):
            activity.append("copy-sync")

    class _Stream:
        def __init__(self, kind):
            self.kind = kind

        def wait_event(self, _event):
            activity.append(f"{self.kind}-wait")

        def synchronize(self):
            activity.append(f"{self.kind}-sync")

    def move(destination, source):
        activity.append("move")
        moves.append((destination.clone(), source.clone()))

    moves = []
    pool = SimpleNamespace(move_kv_cache=move)
    runtime = _Runtime()
    table = _Pool()
    table.req_to_token = torch.zeros((batch_size + 1, 64), dtype=torch.int32)
    for row in range(1, batch_size + 1):
        table.req_to_token[row, :48] = torch.arange(16, 64, dtype=torch.int32)
    batch = SimpleNamespace(
        reqs=list(requests),
        req_to_token_pool=table,
        # SGLang may expose this as a string (for example, "cuda:0").
        # Completion identity must come from the materialized tensor device,
        # not from ``str.index`` on the scheduler-facing value.
        device="cpu",
    )
    producer_stream = _Stream("producer")
    device_module = SimpleNamespace(
        Stream=lambda **_kwargs: _Stream("copy"),
        Event=_Event,
        stream=lambda _stream: nullcontext(),
        current_stream=lambda _device: producer_stream,
    )
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", lambda: runtime)
    monkeypatch.setattr(torch, "get_device_module", lambda _device=None: device_module)
    monkeypatch.setattr(
        state, "_ALLOCATOR", SimpleNamespace(get_kvcache=lambda: pool)
    )
    return SimpleNamespace(
        batch=batch,
        runtime=runtime,
        relocation_calls=relocation_calls,
        copied_batches=copied_batches,
        acknowledgements=acknowledgements,
        activity=activity,
        moves=moves,
        new_locations=new_locations,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_relocation_uses_one_native_and_cuda_call_per_scheduler_batch(
    monkeypatch, batch_size: int
) -> None:
    move_counts = (2, 1, 3, 2)[:batch_size]
    case = _full_relocation_batch_case(
        monkeypatch, batch_size, moves_per_item=move_counts
    )
    before = state._activity_counters()

    relocation._maybe_reclaim_decode_batch(case.batch)

    assert len(case.relocation_calls) == 1
    assert len(case.relocation_calls[0]) == batch_size
    assert len(case.copied_batches) == 1
    copied = case.copied_batches[0]
    assert len(copied.receipts) == batch_size
    assert tuple(len(group) for group in copied.receipts) == move_counts
    assert tuple(
        receipt.token_id for group in copied.receipts for receipt in group
    ) == tuple(
        index * 10 + move_index
        for index, count in enumerate(move_counts)
        for move_index in range(count)
    )
    assert copied.completion_domain == 65_537
    assert len(case.moves) == 1
    destination, source = case.moves[0]
    assert destination.numel() == source.numel() == sum(move_counts)
    expected_source = tuple(
        (1 + index * 4) * 16 + move_index
        for index, count in enumerate(move_counts)
        for move_index in range(count)
    )
    expected_destination = tuple(
        (32 + index * 4) * 16 + move_index
        for index, count in enumerate(move_counts)
        for move_index in range(count)
    )
    assert tuple(source.tolist()) == expected_source
    assert tuple(destination.tolist()) == expected_destination
    assert case.activity == [
        "producer-record",
        "copy-wait",
        "move",
        "copy-record",
        "copy-sync",
        "producer-sync",
        "ack",
    ]
    assert len(case.acknowledgements) == 1
    for index, req in enumerate(case.batch.reqs):
        row = case.batch.req_to_token_pool.req_to_token[index + 1]
        assert tuple(row[:24].tolist()) == case.new_locations[index]
        assert torch.count_nonzero(row[24:48]).item() == 0
        assert req._orbitkv_active_kv_len == 24
        assert req._orbitkv_token_reclamation_next_boundary == 72
        assert req.skip_radix_cache_insert is True
    after = state._activity_counters()
    assert after["token_disposition_batches"] - before["token_disposition_batches"] == 1
    assert (
        after["token_policy_evictions"] - before["token_policy_evictions"]
        == 24 * batch_size
    )
    assert after["relocation_batches"] - before["relocation_batches"] == 1
    assert after["relocation_moves"] - before["relocation_moves"] == sum(
        move_counts
    )
    assert (
        after["relocation_reclaimed_pages"]
        - before["relocation_reclaimed_pages"]
        == batch_size
    )
    assert after["relocation_copy_events"] - before["relocation_copy_events"] == 1
    assert (
        after["relocation_copy_tokens"] - before["relocation_copy_tokens"]
        == sum(move_counts)
    )


def test_later_relocation_publication_error_writes_no_batch_mirror(
    monkeypatch,
) -> None:
    case = _full_relocation_batch_case(
        monkeypatch, 2, invalid_publication_index=1
    )
    rows_before = case.batch.req_to_token_pool.req_to_token.clone()

    with pytest.raises(FailStopped, match="mirror publication"):
        relocation._maybe_reclaim_decode_batch(case.batch)

    assert len(case.relocation_calls) == 1
    assert torch.equal(case.batch.req_to_token_pool.req_to_token, rows_before)
    assert case.activity == [
        "producer-record",
        "copy-wait",
        "move",
        "copy-record",
        "copy-sync",
    ]
    assert case.acknowledgements == []
    assert case.runtime.failure_reason is not None
    for req in case.batch.reqs:
        assert not hasattr(req, "_orbitkv_active_kv_len")
        assert not hasattr(req, "_orbitkv_retained_locations")
        assert not hasattr(req, "_orbitkv_token_reclamation_next_boundary")
        assert not hasattr(req, "skip_radix_cache_insert")


def test_later_hybrid_publication_error_writes_no_row_or_lut(
    monkeypatch,
) -> None:
    full = SimpleNamespace(class_id=0, storage="kv", name="full")
    sliding = SimpleNamespace(class_id=1, storage="kv", name="swa")
    policy = SimpleNamespace(
        mode="relocate",
        trigger_tokens=48,
        retained_per_page=8,
        policy_id=7,
        policy_version=1,
        quality_contract=99,
        maximum_source_pages=3,
        evacuation_headroom_pages=2,
        fragmentation_threshold_milli=250,
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(full, sliding),
        classes_by_id={0: full, 1: sliding},
        full_class=full,
        sliding_class=sliding,
        token_reclamation=policy,
    )
    arenas = {
        0: ArenaIdentity(1, 2, 1, 0, 10, 32, 16, 0, 1),
        1: ArenaIdentity(1, 3, 2, 1, 11, 32, 16, 100, 100),
    }
    requests = tuple(
        SimpleNamespace(
            rid=f"request-{index}",
            kv=SimpleNamespace(kv_allocated_len=48),
            req_pool_idx=index + 1,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )
        for index in range(4)
    )
    keys = tuple(("str", req.rid) for req in requests)
    records = {key: SimpleNamespace(boundary=48) for key in keys}

    def view(class_id: int, request_index: int) -> TokenView:
        arena = arenas[class_id]
        backend_page = request_index * 3 + (16 if class_id == 1 else 0)
        return TokenView(
            class_id,
            1,
            16,
            tuple(
                TokenPlacement(
                    token_id,
                    TokenDisposition(TokenDispositionKind.RETAINED),
                    TokenLocation(
                        PageLease(
                            1,
                            2 + class_id,
                            1,
                            1 + backend_page + token_id // 16,
                            1 + class_id,
                        ),
                        arena.backend_base_index
                        + backend_page
                        + token_id // 16,
                        token_id % 16,
                    ),
                )
                for token_id in range(48)
            ),
        )

    views = {
        (key, class_id): view(class_id, index)
        for index, key in enumerate(keys)
        for class_id in (0, 1)
    }
    monkeypatch.setattr(
        relocation,
        "_runtime",
        lambda: SimpleNamespace(arenas_by_class=arenas),
    )
    monkeypatch.setattr(relocation, "_config", lambda: config)
    old_locations = {
        identity: relocation._view_locations(token_view)
        for identity, token_view in views.items()
    }
    mapping = torch.zeros(4096, dtype=torch.int64)
    table = _Pool()
    table.req_to_token = torch.zeros((5, 64), dtype=torch.int32)
    for index, key in enumerate(keys):
        full_locations = old_locations[(key, 0)]
        swa_locations = old_locations[(key, 1)]
        table.req_to_token[index + 1, :48] = torch.tensor(
            full_locations, dtype=torch.int32
        )
        mapping[torch.tensor(full_locations)] = torch.tensor(swa_locations)
    batch_calls = []
    acknowledgements = []

    class _Runtime:
        arenas_by_class = arenas
        failure_reason = None

        def has_request(self, key):
            return key in records

        def record_for(self, key):
            return records[key]

        def active_kv_length(self, _key, _class_id):
            return 48

        def token_view(self, key, class_id):
            return views[(key, class_id)]

        def relocate_tokens_batch(self, items, _copy):
            values = tuple(items)
            batch_calls.append(values)
            publications = []
            for index, item in enumerate(values):
                full_locations = tuple(1000 + index * 32 + i for i in range(24))
                swa_locations = tuple(2000 + index * 32 + i for i in range(24))
                if index == 3:
                    swa_locations = swa_locations[:-1]
                prepared = SimpleNamespace(
                    moves=(), projected_reclaimed_pages=1
                )
                publications.append(
                    RelocationPublication(
                        item.key,
                        views[(item.key, 0)],
                        prepared,
                        SimpleNamespace(),
                        full_locations,
                        ((0, full_locations), (1, swa_locations)),
                        (),
                    )
                )
            return RelocationBatchPublication(tuple(publications), ())

        def acknowledge_relocation_batch(self, publication):
            acknowledgements.append(publication)

        def fail_stop(self, reason):
            self.failure_reason = reason

    runtime = _Runtime()
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(relocation, "_runtime", lambda: runtime)
    monkeypatch.setattr(
        state,
        "_ALLOCATOR",
        SimpleNamespace(full_to_swa_index_mapping=mapping),
    )
    rows_before = table.req_to_token.clone()
    mapping_before = mapping.clone()

    with pytest.raises(FailStopped, match="mirror publication"):
        relocation._maybe_reclaim_decode_batch(
            SimpleNamespace(
                reqs=list(requests),
                req_to_token_pool=table,
                device=torch.device("cpu"),
            )
        )

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 4
    assert torch.equal(table.req_to_token, rows_before)
    assert torch.equal(mapping, mapping_before)
    assert acknowledgements == []
    assert runtime.failure_reason is not None
    for req in requests:
        assert not hasattr(req, "_orbitkv_active_kv_len")
        assert not hasattr(req, "_orbitkv_retained_locations")
        assert not hasattr(req, "_orbitkv_retained_swa_locations")
        assert not hasattr(req, "_orbitkv_token_reclamation_next_boundary")
        assert not hasattr(req, "skip_radix_cache_insert")


def test_forward_active_length_override_preserves_absolute_positions() -> None:
    req = SimpleNamespace(_orbitkv_active_kv_len=25)
    batch = SimpleNamespace(
        reqs=[req], seq_lens_cpu=torch.tensor([49], dtype=torch.int64)
    )
    positions = torch.tensor([48], dtype=torch.int32)
    absolute = torch.tensor([49], dtype=torch.int64)
    forward = SimpleNamespace(
        batch_size=1,
        seq_lens=absolute,
        seq_lens_cpu=torch.tensor([49], dtype=torch.int64),
        seq_lens_sum=49,
        positions=positions,
    )

    output = relocation._active_forward_lengths(forward, object(), batch, object())

    assert output.seq_lens.tolist() == [25]
    assert output.absolute_seq_lens is absolute
    assert output.absolute_seq_lens_cpu.tolist() == [49]
    assert output.positions is positions
    assert output.positions.tolist() == [48]


def test_hybrid_compact_publication_rebuilds_full_to_swa_lut(monkeypatch) -> None:
    mapping, full_view, swa_view, old_full, old_swa, row = (
        _hybrid_reclamation_case(monkeypatch)
    )
    req = SimpleNamespace(prefix_indices=torch.empty(0, dtype=torch.int64))
    relocation._preflight_request(req, row, 48, 48, (full_view, swa_view))
    new_full = tuple(range(80, 104))
    retained_swa = tuple(
        location for token_id, location in enumerate(old_swa) if token_id % 16 < 8
    )
    publication = SimpleNamespace(
        class_retained_locations=((0, new_full), (1, retained_swa))
    )
    relocation._publish_compact_row(req, row, publication, old_full, 48)
    assert tuple(row[:24].tolist()) == new_full
    assert torch.count_nonzero(row[24:48]).item() == 0
    assert tuple(mapping[torch.tensor(new_full)].tolist()) == retained_swa
    assert torch.count_nonzero(mapping[torch.tensor(old_full)]).item() == 0
    assert req._orbitkv_retained_locations == new_full
    assert req._orbitkv_retained_swa_locations == retained_swa


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("full_missing", "lost its retained Full locations mirror"),
        ("full_inexact", "retained Full locations differ"),
        ("swa_missing", "lost its retained SWA locations mirror"),
        ("swa_inexact", "retained SWA locations differ"),
    ],
)
def test_compact_hybrid_preflight_requires_exact_retained_location_mirrors(
    monkeypatch, fault: str, message: str
) -> None:
    mapping, full_view, swa_view, old_full, old_swa, row = (
        _hybrid_reclamation_case(monkeypatch)
    )
    req = SimpleNamespace(
        prefix_indices=torch.empty(0, dtype=torch.int64),
        _orbitkv_active_kv_len=48,
        _orbitkv_retained_locations=old_full,
        _orbitkv_retained_swa_locations=old_swa,
    )
    if fault.endswith("missing"):
        delattr(
            req,
            (
                "_orbitkv_retained_locations"
                if fault.startswith("full")
                else "_orbitkv_retained_swa_locations"
            ),
        )
    elif fault == "full_inexact":
        req._orbitkv_retained_locations = old_full[:-1] + (old_full[-1] + 1,)
    else:
        req._orbitkv_retained_swa_locations = old_swa[:-1] + (old_swa[-1] + 1,)
    row_before = row.clone()
    mapping_before = mapping.clone()

    with pytest.raises(RuntimeError, match=message):
        relocation._preflight_request(req, row, 48, 48, (full_view, swa_view))

    assert torch.equal(row, row_before)
    assert torch.equal(mapping, mapping_before)


@pytest.mark.parametrize(
    "fault", ["missing_swa", "wrong_swa_count", "full_out_of_range"]
)
def test_hybrid_publication_is_validated_before_mirror_mutation(
    monkeypatch, fault: str
) -> None:
    mapping, _full_view, _swa_view, old_full, old_swa, row = (
        _hybrid_reclamation_case(monkeypatch)
    )
    req = SimpleNamespace(prefix_indices=torch.empty(0, dtype=torch.int64))
    new_full = tuple(range(80, 104))
    retained_swa = tuple(
        location for token_id, location in enumerate(old_swa) if token_id % 16 < 8
    )
    if fault == "missing_swa":
        class_locations = ((0, new_full),)
    elif fault == "wrong_swa_count":
        class_locations = ((0, new_full), (1, retained_swa[:-1]))
    else:
        out_of_range = new_full[:-1] + (int(mapping.numel()),)
        class_locations = ((0, out_of_range), (1, retained_swa))
    row_before = row.clone()
    mapping_before = mapping.clone()

    with pytest.raises(RuntimeError):
        relocation._publish_compact_row(
            req,
            row,
            SimpleNamespace(class_retained_locations=class_locations),
            old_full,
            48,
        )

    assert torch.equal(row, row_before)
    assert torch.equal(mapping, mapping_before)
    assert not hasattr(req, "_orbitkv_active_kv_len")
    assert not hasattr(req, "_orbitkv_retained_locations")
    assert not hasattr(req, "_orbitkv_retained_swa_locations")


def test_relocation_copy_moves_a_real_mla_latent_and_rope_row(monkeypatch) -> None:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    config = SimpleNamespace(
        page_tokens=16,
        classes=(
            SimpleNamespace(
                class_id=0,
                storage="latent_kv",
                layers=(0, 1),
                components_by_name={"latent": 16, "rope": 8},
            ),
        ),
        classes_by_id={},
        sliding_class=None,
    )
    config.classes_by_id[0] = config.classes[0]
    arena = ArenaIdentity(1, 2, 1, 0, 10, 8, 16, 0, 1)
    pool = MLATokenToKVPool(
        size=128,
        page_size=16,
        dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=2,
    )
    source = 17
    destination = 33
    for layer, tensor in enumerate(pool.kv_buffer):
        tensor[source].copy_(
            torch.arange(12, dtype=torch.bfloat16).reshape(1, 12) + layer * 100
        )
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(
        relocation, "_runtime", lambda: SimpleNamespace(arenas_by_class={0: arena})
    )

    class _Event:
        def record(self, *, stream):
            assert isinstance(stream, _Stream)

        def synchronize(self):
            return None

    class _Stream:
        def wait_event(self, _event):
            return None

    device_module = SimpleNamespace(
        Stream=lambda **_kwargs: _Stream(),
        Event=_Event,
        stream=lambda _stream: nullcontext(),
        current_stream=lambda _device: _Stream(),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    state._ALLOCATOR = SimpleNamespace(get_kvcache=lambda: pool)
    location = lambda page, offset: TokenLocation(
        PageLease(1, 2, 1, page, 1), page - 1, offset
    )
    movement = TokenMove(7, location(1, 1), location(2, 1))
    prepared = SimpleNamespace(
        class_id=0,
        relocation=RelocationLease(1, 0, 1),
        request=RequestLease(1, 0, 1),
        base_snapshot=SnapshotLease(1, 0, 1),
        target_snapshot=SnapshotLease(1, 1, 1),
        moves=(movement,),
    )
    counters_before = state._activity_counters()
    copied = relocation._copy_callback(
        SimpleNamespace(device=torch.device("cpu"))
    )((prepared,))
    assert len(copied.receipts) == 1
    assert len(copied.receipts[0]) == 1
    for tensor in pool.kv_buffer:
        assert torch.equal(tensor[destination], tensor[source])
    counters = state._activity_counters()
    assert (
        counters["relocation_copy_events"]
        - counters_before["relocation_copy_events"]
        == 1
    )
    assert (
        counters["relocation_copy_tokens"]
        - counters_before["relocation_copy_tokens"]
        == 1
    )


def test_relocation_copy_rejects_mla_component_drift_before_move(monkeypatch) -> None:
    component = SimpleNamespace(
        class_id=0,
        storage="latent_kv",
        layers=(0,),
        components_by_name={"latent": 16, "rope": 8},
    )
    config = SimpleNamespace(
        page_tokens=16,
        classes=(component,),
        classes_by_id={0: component},
        sliding_class=None,
    )
    arena = ArenaIdentity(1, 2, 1, 0, 10, 8, 16, 0, 1)
    moves = []
    pool = SimpleNamespace(
        dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=3,
        use_dsa=False,
        dsa_kv_cache_store_fp8=False,
        move_kv_cache=lambda *_args: moves.append("move"),
    )
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(
        relocation, "_runtime", lambda: SimpleNamespace(arenas_by_class={0: arena})
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device=None: SimpleNamespace(
            Stream=lambda **_kwargs: object(),
            Event=lambda: SimpleNamespace(
                record=lambda **_kwargs: None, synchronize=lambda: None
            ),
            stream=lambda _stream: nullcontext(),
        ),
    )
    state._ALLOCATOR = SimpleNamespace(get_kvcache=lambda: pool)
    location = lambda page: TokenLocation(
        PageLease(1, 2, 1, page, 1), page - 1, 0
    )
    prepared = SimpleNamespace(
        class_id=0,
        relocation=RelocationLease(1, 0, 1),
        moves=(TokenMove(0, location(1), location(2)),),
    )
    with pytest.raises(
        relocation.RelocationCopyUnobserved, match="MLA pool geometry"
    ):
        relocation._copy_callback(SimpleNamespace(device=torch.device("cpu")))(
            (prepared,)
        )
    assert moves == []


def test_relocation_copy_flatten_failure_is_unobserved(monkeypatch) -> None:
    component = SimpleNamespace(class_id=0, storage="kv")
    config = SimpleNamespace(
        page_tokens=16,
        classes=(component,),
        classes_by_id={0: component},
        sliding_class=None,
    )
    moves = []
    pool = SimpleNamespace(move_kv_cache=lambda *_args: moves.append("move"))
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(
        relocation,
        "_runtime",
        lambda: SimpleNamespace(arenas_by_class={}),
    )
    monkeypatch.setattr(
        state, "_ALLOCATOR", SimpleNamespace(get_kvcache=lambda: pool)
    )
    prepared = SimpleNamespace(
        class_id=0,
        relocation=RelocationLease(1, 0, 1),
        moves=(object(),),
    )

    with pytest.raises(
        relocation.RelocationCopyUnobserved, match="was not enqueued"
    ):
        relocation._copy_callback(SimpleNamespace(device=torch.device("cpu")))(
            (prepared,)
        )

    assert moves == []


def test_relocation_copy_move_failure_is_not_abortable(monkeypatch) -> None:
    component = SimpleNamespace(class_id=0, storage="kv")
    config = SimpleNamespace(
        page_tokens=16,
        classes=(component,),
        classes_by_id={0: component},
        sliding_class=None,
    )
    arena = ArenaIdentity(1, 2, 1, 0, 10, 8, 16, 0, 1)

    def fail_move(_destination, _source):
        raise OSError("lost move return")

    pool = SimpleNamespace(move_kv_cache=fail_move)
    class _Stream:
        def wait_event(self, _event):
            return None

    device_module = SimpleNamespace(
        Stream=lambda **_kwargs: _Stream(),
        Event=lambda: SimpleNamespace(
            record=lambda **_kwargs: None, synchronize=lambda: None
        ),
        stream=lambda _stream: nullcontext(),
        current_stream=lambda _device: _Stream(),
    )
    monkeypatch.setattr(relocation, "_config", lambda: config)
    monkeypatch.setattr(
        relocation,
        "_runtime",
        lambda: SimpleNamespace(arenas_by_class={0: arena}),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    monkeypatch.setattr(
        state, "_ALLOCATOR", SimpleNamespace(get_kvcache=lambda: pool)
    )
    location = lambda page: TokenLocation(
        PageLease(1, 2, 1, page, 1), page - 1, 0
    )
    prepared = SimpleNamespace(
        class_id=0,
        relocation=RelocationLease(1, 0, 1),
        moves=(TokenMove(0, location(1), location(2)),),
    )

    with pytest.raises(RuntimeError, match="CUDA copy/event failed") as caught:
        relocation._copy_callback(SimpleNamespace(device=torch.device("cpu")))(
            (prepared,)
        )

    assert not isinstance(caught.value, relocation.RelocationCopyUnobserved)
