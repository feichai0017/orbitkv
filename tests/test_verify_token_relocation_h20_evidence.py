from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


sys.dont_write_bytecode = True
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/verify_token_relocation_h20_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_token_relocation_h20_evidence", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
SMOKE_PATHS = tuple(
    Path(__file__).resolve().parents[1]
    / "results/h20-sglang-v0517-token-relocation-diagnostic-20260825"
    / "records/epoch-001"
    / f"qwen2.5-0.5b-b{batch}-{mode}.json"
    for batch in (1, 4)
    for mode in ("naive", "relocate")
)

EXPECTED_VICTIM_POLICY = {
    "trigger_tokens": 48,
    "retained_per_page": 8,
    "policy_id": 260813263,
    "policy_version": 1,
    "quality_contract": 1,
    "fragmentation_threshold_milli": 500,
    "maximum_source_pages": 3,
    "evacuation_headroom_pages": 2,
}


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _census(mode: str, batch: int, iterations: int) -> dict:
    scheduler_events = iterations * 2
    request_rounds = batch * scheduler_events
    counters = {name: 0 for name in verifier.BATCH_COUNTER_FIELDS}
    counters.update(
        request_acquire_batch_calls=iterations,
        prepare_batch_calls=iterations * 41,
        submit_batch_calls=iterations * 41,
        complete_batch_calls=iterations * 41,
        release_batch_calls=iterations,
        acknowledge_reclamations_batch_calls=(
            iterations + (scheduler_events if mode == "relocate" else 0)
        ),
        recycle_requests_batch_calls=iterations,
        forward_events=iterations * 41,
        completion_values=(
            iterations * 41 + (scheduler_events if mode == "relocate" else 0)
        ),
        event_queries=iterations * 40,
        event_waits=iterations,
        prefix_matches=iterations * batch,
        token_disposition_batches=scheduler_events,
        token_policy_evictions=(
            request_rounds * 24
        ),
        mark_token_dispositions_batch_calls=(
            scheduler_events if mode == "relocate" else request_rounds
        ),
    )
    if mode == "relocate":
        token_view_calls = scheduler_events * (batch + 2)
        counters.update(
            relocation_batches=scheduler_events,
            relocation_moves=(
                request_rounds * 24
            ),
            relocation_reclaimed_pages=request_rounds,
            relocation_copy_events=scheduler_events,
            relocation_copy_tokens=(
                request_rounds * 24
            ),
            prepare_relocation_batch_calls=scheduler_events,
            submit_relocation_batch_calls=scheduler_events,
            complete_relocation_batch_calls=scheduler_events,
            token_views_batch_calls=token_view_calls,
            mirror_validation_calls=iterations * 2,
            mirror_syncs=iterations,
            buffer_too_small_preflights=token_view_calls + iterations,
            cold_workspace_allocations=(
                token_view_calls + iterations + scheduler_events
            ),
        )
    else:
        token_view_calls = request_rounds * 3
        counters.update(
            {name: 0 for name in verifier.RELOCATION_COUNTER_FIELDS}
        )
        counters.update(
            token_views_batch_calls=token_view_calls,
            mirror_validation_calls=iterations * 2,
            mirror_syncs=iterations,
            buffer_too_small_preflights=token_view_calls + iterations,
            cold_workspace_allocations=token_view_calls + iterations,
        )
    stats = {name: 0 for name in verifier.MANAGER_DRAIN_FIELDS}
    stats["free_pages"] = 64 * batch
    identity = {
        "engine_epoch": 1,
        "pool_epoch": 2,
        "pool_id": 1,
        "class_id": 0,
        "backend_domain": 1,
        "page_count": 64 * batch,
        "page_tokens": 16,
        "backend_base_index": 0,
        "first_page_id": 1,
    }
    arena = {name: identity[name] for name in verifier.ARENA_IDENTITY_FIELDS}
    arena.update({name: 0 for name in verifier.ARENA_DRAIN_FIELDS})
    arena["free_pages"] = identity["page_count"]
    return {
        "abi_version": 8,
        "plan_fingerprint": "sha256:" + "9" * 64,
        "state_plan_fingerprint": None,
        "fixed_state_byte_count": 0,
        "fixed_state_descriptors": [],
        "tree_cache_type": {
            "module": "orbitkv_sglang.plugin.prefix_cache",
            "qualname": "OrbitKvPrefixCache",
        },
        "identities": [identity],
        "manager_stats": stats,
        "arena_stats": [arena],
        "swa_activity": {
            "status": "exposed",
            "applicable": False,
            "swa_retirement_certificates": 0,
            "swa_pages_reclaimed": 0,
            "swa_wrap_events": 0,
        },
        "batch_counters": counters,
    }


def _record(epoch: int, batch: int, mode: str) -> dict:
    iterations = 5
    workload = {
        "requests": batch,
        "iterations": iterations,
        "prompt_tokens": 48,
        "decode_tokens": 41,
        "victim_count_per_request": 24,
        "retained_count_at_trigger": 24,
        "expected_reclamation_rounds": 2,
        "seed": 20260825,
        "input_token_digests_by_iteration_sha256": [],
        "input_token_digest_sha256": "1" * 64,
    }
    checkpoint = {
        "config_sha256": verifier.CHECKPOINT_CONFIG_SHA256,
        "index_files": [],
        "indexed_weight_bytes": None,
        "indexed_weight_container_overhead_bytes": None,
        "indexed_weight_files": [],
        "indexed_weights_complete": True,
        "load_format": "auto",
        "missing_indexed_weights": [],
        "observed_indexed_weight_bytes": 0,
        "weight_bytes": verifier.CHECKPOINT_WEIGHT_BYTES,
        "weight_files": [
            {
                "name": "model.safetensors",
                "bytes": verifier.CHECKPOINT_WEIGHT_BYTES,
                "sha256": verifier.CHECKPOINT_WEIGHT_SHA256,
            }
        ],
    }
    engine = {
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": 128,
        "page_size": 16,
        "attention_backend": "flashinfer",
        "disable_hybrid_swa_memory": False,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": batch * 48,
        "max_running_requests": batch,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "enable_dp_attention": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
        "enable_hierarchical_cache": False,
        "enable_streaming_session": False,
        "enable_unified_memory": False,
        "enable_pdmux": False,
        "enable_lmcache": False,
        "enable_flexkv": False,
        "enable_session_radix_cache": False,
        "enable_hisparse": False,
        "enable_page_major_kv_layout": False,
        "random_seed": 20260825,
        "log_level": "error",
        "max_total_tokens": 128 * batch,
        "model_path": "/evidence/qwen2.5-0.5b-instruct",
        "radix_cache_backend": "orbitkv",
    }
    sampling = {
        "temperature": 0,
        "max_new_tokens": 41,
        "min_new_tokens": 41,
        "ignore_eos": True,
        "sampling_seed": 20260825,
    }
    prompts = [
        verifier._fresh_input_ids(
            requests=batch,
            prompt_tokens=48,
            vocab_size=151936,
            seed=sampling["sampling_seed"],
            iteration=iteration,
            token_upper_bound=151936,
        )
        for iteration in range(iterations)
    ]
    workload["input_token_digests_by_iteration_sha256"] = [
        [verifier.canonical_digest(prompt) for prompt in row]
        for row in prompts
    ]
    workload["input_token_digest_sha256"] = verifier._input_digest(
        [prompt for row in prompts for prompt in row]
    )
    contract = {
        "checkpoint_identity_sha256": verifier.canonical_digest(checkpoint),
        "engine_args": {
            name: value
            for name, value in engine.items()
            if name != "radix_cache_backend"
        },
        "sampling_params": sampling,
        "workload": workload,
        "capacity_tokens": engine["max_total_tokens"],
        "victim_policy": copy.deepcopy(EXPECTED_VICTIM_POLICY),
    }
    outputs = [
        [
            [iteration * 1000 + request * 100 + token for token in range(41)]
            for request in range(batch)
        ]
        for iteration in range(iterations)
    ]
    base_day = datetime(2026, 8, 25, tzinfo=timezone.utc)
    epoch_start = base_day + timedelta(hours=(epoch - 1) * 6)
    naive_first = epoch % 2 == 1
    first_mode = "naive" if naive_first else "relocate"
    pair_offset = 0 if batch == 1 else 60
    mode_offset = 0 if mode == first_mode else 15
    offset_minutes = pair_offset + mode_offset
    started = epoch_start + timedelta(minutes=offset_minutes)
    started_ns = int(started.timestamp() * 1_000_000_000)
    timings = [
        float(batch) + epoch * 0.1 + index * 0.01
        + (0.05 if mode == "relocate" else 0.0)
        for index in range(iterations)
    ]
    policy = {"mode": mode, **EXPECTED_VICTIM_POLICY}
    source = {
        "root": "/evidence/sglang-v0.5.17",
        "release": verifier.SGLANG_RELEASE,
        "revision": verifier.SGLANG_REVISION,
        "python_source_contract": verifier.SOURCE_CONTRACT,
        "dirty_paths": [verifier.LOADER_PATH],
        "loader": {
            "path": verifier.LOADER_PATH,
            "head_git_blob": verifier.LOADER_HEAD_GIT_BLOB,
            "worktree_git_blob": verifier.LOADER_WORKTREE_GIT_BLOB,
            "worktree_sha256": verifier.LOADER_WORKTREE_SHA256,
            "patch_sha256": verifier.LOADER_PATCH_SHA256,
        },
        "plugin_selection": {
            **verifier.MANAGER_ENTRYPOINT,
            "module": (
                "/workspace/orbitkv/integrations/sglang/src/"
                "orbitkv_sglang/plugin/__init__.py"
            ),
        },
        "harness_sha256": verifier._sha256_file(verifier.BENCHMARK_PATH),
        "adapter": verifier._current_adapter_identity(),
        "library": {
            "path": "/evidence/liborbitkv_ffi.so",
            "bytes": 4096,
            "sha256": "5" * 64,
        },
        "plan": {
            "path": "/evidence/qwen2.5-0.5b-full-page16-bf16.json",
            "bytes": 512,
            "sha256": "6" * 64,
        },
    }
    census = _census(mode, batch, iterations)
    after_load = copy.deepcopy(census)
    after_load["batch_counters"] = {
        name: 0 for name in verifier.BATCH_COUNTER_FIELDS
    }
    return {
        "schema": verifier.RECORD_SCHEMA,
        "mode": mode,
        "started_at_utc": started.isoformat(),
        "command": [
            "/evidence/python",
            "/evidence/bench_token_relocation.py",
            "--mode", mode,
            "--sglang-root", source["root"],
            "--model", engine["model_path"],
            "--plan", source["plan"]["path"],
            "--library", source["library"]["path"],
            "--requests", str(batch),
            "--iterations", str(iterations),
            "--max-total-tokens", str(engine["max_total_tokens"]),
            "--attention-backend", engine["attention_backend"],
            "--context-length", str(engine["context_length"]),
            "--seed", str(sampling["sampling_seed"]),
            "--output", (
                f"/evidence/records/epoch-{epoch:03d}/"
                f"{verifier.MODEL_SLUG}-b{batch}-{mode}.json"
            ),
        ],
        "environment": {
            "SGLANG_PLUGINS": "orbitkv_manager",
            "SGLANG_USE_HND_KVCACHE": "0",
            "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
            "SGLANG_RADIX_FORCE_MISS": "0",
            "ORBITKV_PLAN": source["plan"]["path"],
            "ORBITKV_LIBRARY": source["library"]["path"],
            "ORBITKV_SGLANG_ROOT": source["root"],
            "ORBITKV_TOKEN_RECLAMATION": json.dumps(
                policy, sort_keys=True, separators=(",", ":")
            ),
            "PATH": "/evidence/bin:/usr/bin",
            "PYTHONPATH": (
                "/evidence/sglang-v0.5.17/python:"
                "/workspace/orbitkv/integrations/sglang/src"
            ),
        },
        "source_identity": source,
        "runtime_identity": {
            "python": "/evidence/python",
            "python_version": "3.11.2",
            "sglang_version": "0.5.17",
            "gpu_profile": "single H20 eager",
        },
        "checkpoint": checkpoint,
        "checkpoint_contract": {
            "architecture": "Qwen2ForCausalLM",
            "attention_backend": "flashinfer",
            "attention_profile": "full",
            "backend_profile": {"attention_backend": "flashinfer"},
            "control_token_ids": {},
            "fixed_states": [],
            "max_position_embeddings": 32768,
            "num_hidden_layers": 24,
            "prompt_token_upper_bound": 151936,
            "qualification_scope": "diagnostic_only",
            "sliding_window": None,
            "state_ownership": "request_private",
            "vocab_size": 151936,
            "workload_profile": "fresh_prompt",
            "classes": [
                {
                    "name": "full",
                    "retention": "full",
                    "layers": list(range(24)),
                    "window_tokens": None,
                }
            ],
        },
        "engine_args": engine,
        "sampling_params": sampling,
        "workload": workload,
        "pairing": {
            "pair_key_sha256": verifier.canonical_digest(contract),
            "contract": contract,
            "only_allowed_difference": verifier.ONLY_ALLOWED_DIFFERENCE,
        },
        "load_seconds": 10.0,
        "iteration_seconds": timings,
        "iteration_total_seconds": sum(timings),
        "total_seconds": 60.0,
        "output_token_digest_sha256": verifier.canonical_digest(outputs),
        "request_output_ids": outputs,
        "manager": {"after_load": after_load, "final_census": census},
        "gpu_snapshots": [
            {
                "label": label,
                "time_ns": started_ns + (index - 1) * 10_000_000_000,
                "gpus": [
                    {
                        "index": "0",
                        "name": verifier.GPU_NAME,
                        "uuid": "GPU-synthetic-h20",
                    }
                ],
            }
            for index, label in enumerate(
                (
                    "before_engine",
                    "after_load",
                    "after_workload",
                    "after_shutdown",
                )
            )
        ],
    }


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    for epoch in verifier.EPOCHS:
        epoch_dir = root / "records" / f"epoch-{epoch:03d}"
        epoch_dir.mkdir(parents=True)
        for batch in verifier.BATCHES:
            for mode in verifier.MODES:
                name = f"{verifier.MODEL_SLUG}-b{batch}-{mode}.json"
                _write(epoch_dir / name, _record(epoch, batch, mode))
    return root


def _path(root: Path, epoch: int, batch: int, mode: str) -> Path:
    return (
        root
        / "records"
        / f"epoch-{epoch:03d}"
        / f"{verifier.MODEL_SLUG}-b{batch}-{mode}.json"
    )


def _mutate(
    root: Path, epoch: int, batch: int, mode: str, mutation
) -> None:
    path = _path(root, epoch, batch, mode)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    _write(path, value)


def test_synthetic_matrix_passes_and_emits_diagnostic_summary(
    evidence: Path,
) -> None:
    summary = verifier.verify_evidence(evidence)

    assert summary["status"] == "diagnostic_pair_verification_passed"
    assert summary["epoch_count"] == 4
    assert summary["batch_sizes"] == [1, 4]
    assert summary["record_count"] == 16
    assert summary["pair_count"] == 8
    for field, expected in (
        ("diagnostic_only", True),
        ("sealed", False),
        ("source_clean", False),
        ("source_dirty", True),
        ("hardware_attested", False),
        ("qualified", False),
        ("performance_go", False),
    ):
        assert summary[field] is expected
    assert summary["execution_order"]["epochs"] == [
        {"epoch": 1, "order": ["naive", "relocate"]},
        {"epoch": 2, "order": ["relocate", "naive"]},
        {"epoch": 3, "order": ["naive", "relocate"]},
        {"epoch": 4, "order": ["relocate", "naive"]},
    ]
    for group in summary["groups"]:
        assert group["excluded_iteration_indices"] == [0]
        assert group["hot_iteration_indices"] == [1, 2, 3, 4]
        assert group["hot_sample_count_per_mode"] == 16
        assert group["naive"]["sample_count"] == 16
        assert group["relocate"]["sample_count"] == 16
        assert set(group["relocate_vs_naive"]) == {
            "mean_latency_percent",
            "median_latency_percent",
            "p95_latency_percent",
            "throughput_percent",
        }


def test_cli_prints_the_summary(evidence: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(evidence)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema"] == verifier.SUMMARY_SCHEMA
    assert summary["performance_go"] is False


def test_rejects_fa3_as_naive_relocation_oracle(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["checkpoint_contract"]["attention_backend"] = "fa3"
        record["checkpoint_contract"]["backend_profile"] = {
            "attention_backend": "fa3"
        }
        record["engine_args"]["attention_backend"] = "fa3"
        record["pairing"]["contract"]["engine_args"][
            "attention_backend"
        ] = "fa3"
        record["pairing"]["pair_key_sha256"] = verifier.canonical_digest(
            record["pairing"]["contract"]
        )
        command = record["command"]
        command[command.index("--attention-backend") + 1] = "fa3"

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="fixed Qwen2.5-0.5B contract"):
        verifier.verify_evidence(evidence)


def test_rejects_missing_b4_record(evidence: Path) -> None:
    _path(evidence, 4, 4, "relocate").unlink()
    with pytest.raises(RuntimeError, match="record matrix differs"):
        verifier.verify_evidence(evidence)


def test_missing_records_directory_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="regular records directory"):
        verifier.verify_evidence(empty)


def test_rejects_pair_key_and_contract_tampering(evidence: Path) -> None:
    _mutate(
        evidence, 1, 1, "relocate",
        lambda record: record["pairing"].__setitem__(
            "pair_key_sha256", "0" * 64
        ),
    )
    with pytest.raises(RuntimeError, match="pair key"):
        verifier.verify_evidence(evidence)


def test_rejects_hash_consistent_pair_contract_tampering(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        contract = record["pairing"]["contract"]
        contract["sampling_params"]["sampling_seed"] += 1
        record["pairing"]["pair_key_sha256"] = verifier.canonical_digest(
            contract
        )

    _mutate(evidence, 1, 4, "relocate", mutate)
    with pytest.raises(RuntimeError, match="does not bind sampling params"):
        verifier.verify_evidence(evidence)


def test_rejects_output_token_tampering_even_with_rehashed_digest(
    evidence: Path,
) -> None:
    def mutate(record: dict) -> None:
        record["request_output_ids"][2][0][3] += 1
        record["output_token_digest_sha256"] = verifier.canonical_digest(
            record["request_output_ids"]
        )

    _mutate(evidence, 2, 4, "relocate", mutate)
    with pytest.raises(RuntimeError, match="output tokens differ"):
        verifier.verify_evidence(evidence)


def test_rejects_same_tampered_outputs_in_both_modes(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["request_output_ids"][1][0][0] += 7
        record["output_token_digest_sha256"] = verifier.canonical_digest(
            record["request_output_ids"]
        )

    _mutate(evidence, 4, 1, "naive", mutate)
    _mutate(evidence, 4, 1, "relocate", mutate)
    with pytest.raises(RuntimeError, match="deterministic output tokens differ"):
        verifier.verify_evidence(evidence)


@pytest.mark.parametrize(
    ("counter", "expected_message"),
    (
        ("relocation_batches", "relocation counters differ"),
        ("quarantine_count", "failure/fail-stop counter is nonzero"),
    ),
)
def test_rejects_counter_tampering(
    evidence: Path, counter: str, expected_message: str
) -> None:
    def mutate(record: dict) -> None:
        record["manager"]["final_census"]["batch_counters"][counter] += 1

    _mutate(evidence, 3, 4, "relocate", mutate)
    with pytest.raises(RuntimeError, match=expected_message):
        verifier.verify_evidence(evidence)


def test_rejects_lifecycle_counter_tampering(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["manager"]["final_census"]["batch_counters"][
            "complete_batch_calls"
        ] += 1

    _mutate(evidence, 2, 4, "relocate", mutate)
    with pytest.raises(RuntimeError, match="append batch call identities disagree"):
        verifier.verify_evidence(evidence)


def test_rejects_final_census_leak(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        final = record["manager"]["final_census"]
        final["manager_stats"]["active_requests"] = 1

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="did not drain"):
        verifier.verify_evidence(evidence)


def test_rejects_after_load_activity(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["manager"]["after_load"]["batch_counters"][
            "token_disposition_batches"
        ] = 1

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="after-load counters are nonzero"):
        verifier.verify_evidence(evidence)


def test_census_fixture_uses_exact_raw_hook_swa_activity_schema() -> None:
    state = _census("naive", 1, 5)
    assert state["swa_activity"] == {
        "status": "exposed",
        "applicable": False,
        "swa_retirement_certificates": 0,
        "swa_pages_reclaimed": 0,
        "swa_wrap_events": 0,
    }


@pytest.mark.parametrize("smoke_path", SMOKE_PATHS, ids=lambda path: path.stem)
def test_real_smokes_pass_strict_single_record_validation(
    smoke_path: Path,
) -> None:
    if not smoke_path.is_file():
        pytest.skip(f"local H20 relocation smoke record is unavailable: {smoke_path}")
    record = json.loads(smoke_path.read_text(encoding="utf-8"))
    batch = record["workload"]["requests"]
    mode = record["mode"]
    assert record["schema"] == verifier.RECORD_SCHEMA
    assert batch in (1, 4)
    assert mode in ("naive", "relocate")
    assert record["engine_args"]["attention_backend"] == "flashinfer"
    assert record["engine_args"]["max_total_tokens"] == 128 * batch
    assert record["checkpoint_contract"]["qualification_scope"] == (
        "diagnostic_only"
    )
    assert record["checkpoint"]["config_sha256"] == (
        verifier.CHECKPOINT_CONFIG_SHA256
    )
    assert record["checkpoint"]["weight_files"] == [
        {
            "name": "model.safetensors",
            "bytes": verifier.CHECKPOINT_WEIGHT_BYTES,
            "sha256": verifier.CHECKPOINT_WEIGHT_SHA256,
        }
    ]
    assert record["manager"]["after_load"]["swa_activity"] == {
        "status": "exposed",
        "applicable": False,
        "swa_retirement_certificates": 0,
        "swa_pages_reclaimed": 0,
        "swa_wrap_events": 0,
    }
    checked = verifier._validate_record(
        record,
        epoch=1,
        batch=batch,
        mode=mode,
        label=smoke_path.name,
        expected_harness_sha256=verifier._sha256_file(verifier.BENCHMARK_PATH),
        expected_adapter=verifier._current_adapter_identity(),
        expected_iterations=5,
    )
    assert checked["iterations"] == 5
    counters = record["manager"]["final_census"]["batch_counters"]
    expected = _census(mode, batch, 5)["batch_counters"]
    assert counters == expected


def test_rejects_normalized_instead_of_raw_swa_activity(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["manager"]["after_load"]["swa_activity"].update(
            status="not_applicable", source="normalized", derived=False
        )

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="SWA census is invalid"):
        verifier.verify_evidence(evidence)


def test_rejects_after_load_census_leak(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["manager"]["after_load"]["manager_stats"][
            "active_requests"
        ] = 1

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="after-load manager did not drain"):
        verifier.verify_evidence(evidence)


def test_rejects_non_qwen25_model_contract(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["checkpoint_contract"]["architecture"] = "OtherModel"

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="fixed Qwen2.5-0.5B contract"):
        verifier.verify_evidence(evidence)


def test_rejects_qualification_scope_tampering(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["checkpoint_contract"]["qualification_scope"] = "qualified"

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="fixed Qwen2.5-0.5B contract"):
        verifier.verify_evidence(evidence)


def test_rejects_wrong_checkpoint_artifact_identity(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["checkpoint"]["weight_files"][0]["bytes"] = 1024
        record["checkpoint"]["weight_bytes"] = 1024
        record["pairing"]["contract"][
            "checkpoint_identity_sha256"
        ] = verifier.canonical_digest(record["checkpoint"])
        record["pairing"]["pair_key_sha256"] = verifier.canonical_digest(
            record["pairing"]["contract"]
        )

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="checkpoint weight identity is invalid"):
        verifier.verify_evidence(evidence)


def test_rejects_recomputed_but_wrong_input_digest(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["workload"]["input_token_digest_sha256"] = "f" * 64
        record["pairing"]["contract"]["workload"] = copy.deepcopy(
            record["workload"]
        )
        record["pairing"]["pair_key_sha256"] = verifier.canonical_digest(
            record["pairing"]["contract"]
        )

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="input-token digest is invalid"):
        verifier.verify_evidence(evidence)


def test_rejects_wrong_runner_command(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["command"][1] = "/evidence/not_the_runner.py"

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="not the token-relocation runner"):
        verifier.verify_evidence(evidence)


def test_rejects_identity_tampering(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["source_identity"]["library"]["sha256"] = "f" * 64

    _mutate(evidence, 4, 1, "relocate", mutate)
    with pytest.raises(RuntimeError, match="source identity differs"):
        verifier.verify_evidence(evidence)


def test_rejects_harness_identity_tampering(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["source_identity"]["harness_sha256"] = "f" * 64

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="harness differs from this checkout"):
        verifier.verify_evidence(evidence)


def test_rejects_unbalanced_started_at_order(evidence: Path) -> None:
    naive = json.loads(_path(evidence, 2, 1, "naive").read_text())

    def mutate(record: dict) -> None:
        changed = (
            datetime.fromisoformat(naive["started_at_utc"])
            + timedelta(minutes=45)
        )
        record["started_at_utc"] = changed.isoformat()
        changed_ns = int(changed.timestamp() * 1_000_000_000)
        for index, snapshot in enumerate(record["gpu_snapshots"]):
            snapshot["time_ns"] = changed_ns + (index - 1) * 10_000_000_000

    _mutate(evidence, 2, 1, "relocate", mutate)
    with pytest.raises(RuntimeError, match="B1/B4 execution orders differ"):
        verifier.verify_evidence(evidence)


def test_rejects_cross_batch_process_overlap(evidence: Path) -> None:
    b1_relocate = json.loads(
        _path(evidence, 1, 1, "relocate").read_text(encoding="utf-8")
    )

    def mutate(record: dict) -> None:
        record["started_at_utc"] = b1_relocate["started_at_utc"]
        for index, snapshot in enumerate(record["gpu_snapshots"]):
            snapshot["time_ns"] = b1_relocate["gpu_snapshots"][index][
                "time_ns"
            ]

    _mutate(evidence, 1, 4, "naive", mutate)
    with pytest.raises(RuntimeError, match="overlaps another record process"):
        verifier.verify_evidence(evidence)


def test_rejects_impossible_total_duration(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["load_seconds"] = 40.0
        record["iteration_seconds"] = [10.0] * 5
        record["iteration_total_seconds"] = 50.0
        record["total_seconds"] = 60.0

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="shorter than measured work"):
        verifier.verify_evidence(evidence)


def test_rejects_impossible_capacity(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["engine_args"]["max_total_tokens"] = 16
        record["pairing"]["contract"]["engine_args"][
            "max_total_tokens"
        ] = 16
        record["pairing"]["contract"]["capacity_tokens"] = 16
        record["pairing"]["pair_key_sha256"] = verifier.canonical_digest(
            record["pairing"]["contract"]
        )
        command = record["command"]
        command[command.index("--max-total-tokens") + 1] = "16"

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="capacity differs from the fixed"):
        verifier.verify_evidence(evidence)


def test_rejects_forbidden_workspace_counter(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["manager"]["final_census"]["batch_counters"][
            "hot_workspace_allocations"
        ] = 1

    _mutate(evidence, 1, 1, "relocate", mutate)
    with pytest.raises(RuntimeError, match="failure/fail-stop counter is nonzero"):
        verifier.verify_evidence(evidence)


def test_rejects_out_of_vocabulary_output(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["request_output_ids"][0][0][0] = 151936
        record["output_token_digest_sha256"] = verifier.canonical_digest(
            record["request_output_ids"]
        )

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="output token vector is invalid"):
        verifier.verify_evidence(evidence)


def test_rejects_plugin_module_outside_recorded_source(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["source_identity"]["plugin_selection"]["module"] = (
            "/tmp/foreign/orbitkv_sglang/plugin/__init__.py"
        )

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="manager plugin identity is invalid"):
        verifier.verify_evidence(evidence)


def test_rejects_pythonpath_without_current_adapter(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["environment"]["PYTHONPATH"] = (
            "/evidence/sglang-v0.5.17/python:/tmp/foreign-adapter"
        )

    _mutate(evidence, 1, 1, "naive", mutate)
    with pytest.raises(RuntimeError, match="PYTHONPATH omits"):
        verifier.verify_evidence(evidence)


def test_rejects_reduced_iteration_count(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["workload"]["iterations"] = 2
        record["pairing"]["contract"]["workload"]["iterations"] = 2
        record["pairing"]["pair_key_sha256"] = verifier.canonical_digest(
            record["pairing"]["contract"]
        )

    _mutate(evidence, 2, 4, "naive", mutate)
    with pytest.raises(RuntimeError, match="iterations must be exactly 5"):
        verifier.verify_evidence(evidence)


def test_statistics_exclude_iteration_zero(evidence: Path) -> None:
    def mutate(record: dict) -> None:
        record["iteration_seconds"][0] = 9999.0
        record["iteration_total_seconds"] = sum(record["iteration_seconds"])
        record["total_seconds"] = 10050.0

    _mutate(evidence, 1, 1, "naive", mutate)
    summary = verifier.verify_evidence(evidence)
    b1 = next(group for group in summary["groups"] if group["batch_size"] == 1)
    assert b1["naive"]["mean_seconds"] < 2.0


def test_rejects_duplicate_json_keys(evidence: Path) -> None:
    path = _path(evidence, 1, 1, "naive")
    path.write_text('{"schema": "one", "schema": "two"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate key"):
        verifier.verify_evidence(evidence)
