from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

import orbitkv_sglang.plugin.lowering as lowering
import orbitkv_sglang.plugin.relocation as relocation


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
