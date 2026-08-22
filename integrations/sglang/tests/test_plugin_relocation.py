from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

import orbitkv_sglang.plugin.lowering as lowering
import orbitkv_sglang.plugin.relocation as relocation
import orbitkv_sglang.plugin.state as state
from orbitkv_sglang.runtime import ArenaIdentity, PageLease, TokenLocation, TokenPlacement, TokenView
from orbitkv_sglang.runtime.token_relocation import TokenDisposition, TokenDispositionKind


class _Pool:
    def __init__(self) -> None:
        self.req_to_token = torch.zeros((2, 64), dtype=torch.int32)

    def write(self, indices: Any, values: Any) -> None:
        self.req_to_token[indices] = values


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
    monkeypatch.setattr(lowering, "_preflight_decode_batch", lambda _batch: ((48,), (1,)))
    monkeypatch.setattr(lowering, "_prepare_batch", lambda *_args: (object(), (object(),)))
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
    state._ALLOCATOR = SimpleNamespace(full_to_swa_index_mapping=mapping)

    def view(class_id: int, backend_base: int, first_page: int) -> TokenView:
        placements = []
        for token_id in range(48):
            ordinal, offset = divmod(token_id, 16)
            placements.append(
                TokenPlacement(
                    token_id,
                    TokenDisposition(TokenDispositionKind.RETAINED),
                    TokenLocation(
                        PageLease(1, 2 + class_id, 1, first_page + ordinal, class_id + 1),
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
    req = SimpleNamespace(prefix_indices=torch.empty(0, dtype=torch.int64))
    relocation._preflight_request(req, row, 48, (full_view, swa_view))
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
