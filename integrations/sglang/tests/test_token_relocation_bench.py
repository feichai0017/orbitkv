from __future__ import annotations

from argparse import Namespace
from typing import Any

import bench_token_relocation as bench
import pytest


def test_matched_modes_share_every_policy_field_except_physical_action() -> None:
    naive = bench._policy("naive")
    relocate = bench._policy("relocate")
    assert naive.pop("mode") == "naive"
    assert relocate.pop("mode") == "relocate"
    assert naive == relocate
    assert naive["trigger_tokens"] == 48
    assert naive["retained_per_page"] == 8
    assert bench.VICTIM_COUNT == 24


def test_base_runner_contract_runs_two_rounds_and_crosses_packed_page() -> None:
    args = Namespace(
        mode="relocate",
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
    assert base.decode_tokens == 41
    assert bench.MATERIALIZED_DECODE_TOKENS == 40
    assert bench.RETAINED_COUNT_AT_TRIGGER == 24
    assert bench.RECLAMATION_INTERVAL_TOKENS == 24
    assert bench.EXPECTED_RECLAMATION_ROUNDS == 2
    assert (
        base.prompt_tokens
        + bench.MATERIALIZED_DECODE_TOKENS
        - bench.EXPECTED_RECLAMATION_ROUNDS * bench.VICTIM_COUNT
        == 40
    )
    materialized_before_last_round = (
        bench.EXPECTED_RECLAMATION_ROUNDS - 1
    ) * bench.RECLAMATION_INTERVAL_TOKENS
    materialized_after_last_round = (
        bench.MATERIALIZED_DECODE_TOKENS - materialized_before_last_round
    )
    assert materialized_after_last_round == bench.common.PAGE_TOKENS
    assert (
        bench.RETAINED_COUNT_AT_TRIGGER + materialized_after_last_round
        == 40
    )
    final_boundary = base.prompt_tokens + bench.MATERIALIZED_DECODE_TOKENS
    last_trigger = (
        bench.TRIGGER_TOKENS
        + (bench.EXPECTED_RECLAMATION_ROUNDS - 1)
        * bench.RECLAMATION_INTERVAL_TOKENS
    )
    next_trigger = last_trigger + bench.RECLAMATION_INTERVAL_TOKENS
    assert (last_trigger, final_boundary, next_trigger) == (72, 88, 96)
    assert base.chunked_prefill_size == 4 * 48

    workload = bench._workload(args, ((1,), (2,), (3,), (4,)))
    assert workload["decode_tokens"] == 41
    assert workload["expected_reclamation_rounds"] == 2
    assert workload["retained_count_at_trigger"] == 24


def test_validate_arguments_supplies_absent_state_plan(tmp_path) -> None:
    sglang_root = tmp_path / "sglang"
    model = tmp_path / "model"
    sglang_root.mkdir()
    model.mkdir()
    plan = tmp_path / "plan.json"
    library = tmp_path / "liborbitkv_ffi.so"
    plan.write_text("{}", encoding="utf-8")
    library.write_bytes(b"ffi")
    args = Namespace(
        mode="relocate",
        sglang_root=str(sglang_root),
        model=str(model),
        plan=str(plan),
        library=str(library),
        requests=1,
        iterations=1,
        max_total_tokens=1024,
        context_length=128,
        mem_fraction_static=None,
        attention_backend="flashinfer",
    )

    paths = bench.validate_arguments(args)

    assert paths["state_plan"] is None


def test_naive_rejects_page_table_backend_as_a_sparse_oracle(tmp_path) -> None:
    args = Namespace(
        mode="naive",
        sglang_root=str(tmp_path),
        model=str(tmp_path),
        plan=str(tmp_path / "plan.json"),
        library=str(tmp_path / "lib.so"),
        requests=1,
        iterations=1,
        max_total_tokens=128,
        context_length=128,
        mem_fraction_static=None,
        attention_backend="fa3",
    )
    with pytest.raises(ValueError, match="cannot represent sparse retained slots"):
        bench.validate_arguments(args)


def test_execution_contract_records_diagnostic_backend_override() -> None:
    contract = {
        "attention_backend": "fa3",
        "backend_profile": {"attention_backend": "fa3"},
    }
    result = bench._execution_contract(contract, "flashinfer")
    assert result["attention_backend"] == "flashinfer"
    assert result["backend_profile"]["attention_backend"] == "flashinfer"
    assert result["workload_profile"] == "fresh_prompt"
    assert result["state_ownership"] == "request_private"
    assert result["qualification_scope"] == "diagnostic_only"
    assert contract["attention_backend"] == "fa3"


def test_manager_state_accepts_one_or_two_compiled_token_classes(monkeypatch) -> None:
    counters = {
        "quarantined_pages": 0,
        "exhausted_pages": 0,
        "prepared_steps": 0,
        "submitted_steps": 0,
        "pending_reclamations": 0,
    }
    state = {
        "abi_version": 8,
        "manager_stats": counters,
        "arena_stats": [{}, {}],
        "batch_counters": {},
    }
    monkeypatch.setattr(
        bench.common, "_state", lambda _info: {"orbitkv_manager": state}
    )
    assert bench._manager_state({}, "test", 2) is state
    try:
        bench._manager_state({}, "test", 1)
    except RuntimeError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("arena cardinality drift was accepted")


def _drained_state(
    counters: dict[str, int], class_count: int
) -> dict[str, Any]:
    return {
        "manager_stats": {
            "active_requests": 0,
            "active_snapshots": 0,
            "active_pages": 0,
            "total_request_page_refs": 0,
            "total_prefix_page_refs": 0,
            "total_reader_pins": 0,
        },
        "arena_stats": [
            {"free_pages": 64, "page_count": 64}
            for _ in range(class_count)
        ],
        "batch_counters": counters,
    }


@pytest.mark.parametrize("requests", (1, 4))
def test_relocation_counters_are_per_scheduler_batch_not_per_request(
    requests: int,
) -> None:
    iterations = 3
    class_count = 2
    scheduler_events = iterations * bench.EXPECTED_RECLAMATION_ROUNDS
    request_rounds = requests * scheduler_events
    counters = {
        "token_disposition_batches": scheduler_events,
        "token_policy_evictions": (
            request_rounds * bench.VICTIM_COUNT * class_count
        ),
        "mark_token_dispositions_batch_calls": scheduler_events,
        "relocation_batches": scheduler_events,
        "relocation_moves": request_rounds * bench.RETAINED_COUNT_AT_TRIGGER,
        "relocation_reclaimed_pages": request_rounds,
        "relocation_copy_events": scheduler_events,
        "relocation_copy_tokens": (
            request_rounds * bench.RETAINED_COUNT_AT_TRIGGER
        ),
        "prepare_relocation_batch_calls": scheduler_events,
        "submit_relocation_batch_calls": scheduler_events,
        "complete_relocation_batch_calls": scheduler_events,
    }

    bench._validate_final_census(
        _drained_state(counters, class_count),
        "relocate",
        requests,
        iterations,
        class_count,
    )


@pytest.mark.parametrize("requests", (1, 4))
def test_naive_counters_keep_scalar_marks_but_batch_scheduler_events(
    requests: int,
) -> None:
    iterations = 3
    class_count = 2
    scheduler_events = iterations * bench.EXPECTED_RECLAMATION_ROUNDS
    request_rounds = requests * scheduler_events
    counters = {
        "token_disposition_batches": scheduler_events,
        "token_policy_evictions": (
            request_rounds * bench.VICTIM_COUNT * class_count
        ),
        "mark_token_dispositions_batch_calls": request_rounds,
        "relocation_batches": 0,
        "relocation_moves": 0,
        "relocation_reclaimed_pages": 0,
        "relocation_copy_events": 0,
        "relocation_copy_tokens": 0,
        "prepare_relocation_batch_calls": 0,
        "submit_relocation_batch_calls": 0,
        "complete_relocation_batch_calls": 0,
    }

    bench._validate_final_census(
        _drained_state(counters, class_count),
        "naive",
        requests,
        iterations,
        class_count,
    )
