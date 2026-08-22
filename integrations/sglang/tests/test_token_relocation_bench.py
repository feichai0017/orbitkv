from __future__ import annotations

from argparse import Namespace

import bench_token_relocation as bench


def test_matched_modes_share_every_policy_field_except_physical_action() -> None:
    naive = bench._policy("naive")
    relocate = bench._policy("relocate")
    assert naive.pop("mode") == "naive"
    assert relocate.pop("mode") == "relocate"
    assert naive == relocate
    assert naive["trigger_tokens"] == 48
    assert naive["retained_per_page"] == 8
    assert bench.VICTIM_COUNT == 24


def test_base_runner_contract_is_full_eager_and_crosses_packed_page() -> None:
    args = Namespace(
        sglang_root="sglang",
        model="model",
        plan="plan",
        library="library",
        requests=4,
        iterations=3,
        max_total_tokens=1024,
        context_length=128,
        seed=7,
        mem_fraction_static=None,
        attention_backend="flashinfer",
    )
    base = bench._base_args(args)
    assert base.mode == "manager"
    assert base.attention_backend == "flashinfer"
    assert base.prompt_tokens == 48
    assert base.decode_tokens == 17
    assert base.prompt_tokens - bench.VICTIM_COUNT + base.decode_tokens - 1 == 40
    assert base.chunked_prefill_size == 4 * 48
