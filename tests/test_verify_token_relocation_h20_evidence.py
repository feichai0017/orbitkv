from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
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
SEAL_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/verify_token_relocation_h20_seal.py"
)
SEAL_SPEC = importlib.util.spec_from_file_location(
    "verify_token_relocation_h20_seal", SEAL_MODULE_PATH
)
assert SEAL_SPEC is not None and SEAL_SPEC.loader is not None
seal_verifier = importlib.util.module_from_spec(SEAL_SPEC)
SEAL_SPEC.loader.exec_module(seal_verifier)
SMOKE_PATHS = tuple(
    Path(__file__).resolve().parents[1]
    / "results/h20-sglang-v0517-token-relocation-diagnostic-20260825"
    / "records/epoch-001"
    / f"qwen2.5-0.5b-b{batch}-{mode}.json"
    for batch in (1, 4)
    for mode in ("naive", "relocate")
)
HISTORICAL_RECORD = (
    Path(__file__).resolve().parents[1]
    / "results/h20-sglang-v0517-token-relocation-diagnostic-20260825"
    / "records/epoch-001/qwen2.5-0.5b-b1-naive.json"
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


def _historical_adapter() -> dict:
    return json.loads(HISTORICAL_RECORD.read_text(encoding="utf-8"))[
        "source_identity"
    ]["adapter"]


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
        "harness_sha256": verifier.DIAGNOSTIC_HARNESS_SHA256,
        "adapter": _historical_adapter(),
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
        expected_harness_sha256=verifier.DIAGNOSTIC_HARNESS_SHA256,
        expected_adapter=_historical_adapter(),
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


def test_default_diagnostic_identity_is_historical_not_live_checkout(
    evidence: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verifier, "_current_adapter_identity",
        lambda: pytest.fail("live adapter identity was consulted"),
    )
    monkeypatch.setattr(
        verifier, "BENCHMARK_PATH", Path("/does/not/exist")
    )
    assert verifier.verify_evidence(evidence)["record_count"] == 16


def test_validate_record_binds_an_explicit_adapter_source_root(
    evidence: Path, tmp_path: Path,
) -> None:
    record = json.loads(_path(evidence, 1, 1, "naive").read_text())
    adapter = record["source_identity"]["adapter"]
    archive_source = tmp_path / "qualification/source"
    for item in adapter["files"]:
        source = Path(__file__).resolve().parents[1] / item["path"]
        target = archive_source / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    # Concurrent source drift means the live adapter no longer matches this
    # historical record; restore the historical bytes from the recorded commit.
    for item in adapter["files"]:
        target = archive_source / item["path"]
        if verifier._sha256_file(target) != item["sha256"]:
            target.write_bytes(
                subprocess.run(
                    ["git", "show", f"cd78105:{item['path']}"],
                    cwd=Path(__file__).resolve().parents[1], check=True,
                    capture_output=True,
                ).stdout
            )
    checked = verifier.validate_record(
        record, 1, 1, "naive", HISTORICAL_RECORD.name,
        expected_harness_sha256=verifier.DIAGNOSTIC_HARNESS_SHA256,
        expected_adapter=adapter,
        expected_adapter_source_root=(
            archive_source / "integrations/sglang/src"
        ),
    )
    assert checked["mode"] == "naive"
    changed = (
        archive_source
        / "integrations/sglang/src/orbitkv_sglang/runtime/census.py"
    )
    changed.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="adapter source differs"):
        verifier.validate_record(
            record, 1, 1, "naive", HISTORICAL_RECORD.name,
            expected_harness_sha256=verifier.DIAGNOSTIC_HARNESS_SHA256,
            expected_adapter=adapter,
            expected_adapter_source_root=(
                archive_source / "integrations/sglang/src"
            ),
        )


def _minimal_sealed_manifest() -> dict:
    return {
        "schema": seal_verifier.SEALED_MANIFEST_SCHEMA,
        "qualification_status": seal_verifier.QUALIFICATION_STATUS,
        "qualification_claim": seal_verifier.QUALIFICATION_CLAIM,
        "evidence_class": seal_verifier.SEALED_EVIDENCE_CLASS,
        "sealed": True, "source_clean": True, "source_dirty": False,
        "preflight_bound": True, "qualified": True,
        "hardware_attested": False, "performance_go": False,
        "abi_version": 8, "exact_symbol_count": 40,
        "record_schema": verifier.RECORD_SCHEMA,
        "summary_schema": seal_verifier.SEALED_SUMMARY_SCHEMA,
        "pair_schema": seal_verifier.SEALED_PAIR_SCHEMA,
        "epoch_count": 4, "batch_sizes": [1, 4],
        "record_count": 16, "pair_count": 8,
        "scope": copy.deepcopy(seal_verifier.EXPECTED_SEALED_SCOPE),
        "source_commit": "a" * 40,
        "source_inventory_sha256": "b" * 64,
        "source_provenance": {
            "kind": "git_bundle",
            "path": "qualification/source.bundle",
            "sha256": "c" * 64,
            "commit": "a" * 40, "reference": "HEAD",
        },
        "sglang_release": verifier.SGLANG_RELEASE,
        "sglang_revision": verifier.SGLANG_REVISION,
        "library_sha256": "d" * 64,
        "model_identity_sha256": "e" * 64,
        "input_hashes": {
            "requirements_input_sha256": seal_verifier.REQUIREMENTS_INPUT_SHA256,
            "requirements_sha256": "f" * 64,
            "plan_sha256": seal_verifier.PLAN_SHA256,
            "model": {
                "config_sha256": verifier.CHECKPOINT_CONFIG_SHA256,
                "weight_sha256": verifier.CHECKPOINT_WEIGHT_SHA256,
                "weight_bytes": verifier.CHECKPOINT_WEIGHT_BYTES,
            },
        },
        "observed_hardware": {
            "name": verifier.GPU_NAME, "uuid": "GPU-test",
            "snapshot_count": 64,
            "attestation": "recorded_observation_only",
        },
        "artifacts": {},
    }


def test_sealed_manifest_preserves_narrow_claim_boundary() -> None:
    manifest = _minimal_sealed_manifest()
    seal_verifier._validate_sealed_manifest(manifest, verifier)
    for field, forged in (
        ("hardware_attested", True),
        ("performance_go", True),
        ("qualification_claim", "performance_qualified"),
        ("exact_symbol_count", 39),
    ):
        changed = copy.deepcopy(manifest)
        changed[field] = forged
        with pytest.raises(RuntimeError, match=field):
            seal_verifier._validate_sealed_manifest(changed, verifier)


def test_sha256sums_and_archive_paths_are_strict(tmp_path: Path) -> None:
    for value in ("", "/absolute", "../escape", "a/../b", "a//b", "a\\b"):
        with pytest.raises(RuntimeError):
            seal_verifier._safe_relative(value)
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'a' * 64}  value\n{'b' * 64}  value\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid entry"):
        seal_verifier._read_sha256sums(sums)


def test_pair_and_summary_are_recomputed_from_raw_records(
    evidence: Path, tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(evidence / "records", root / "records")
    diagnostic = verifier.verify_evidence(root)
    for epoch in verifier.EPOCHS:
        for batch in verifier.BATCHES:
            slug = f"{verifier.MODEL_SLUG}-b{batch}"
            record_root = root / "records" / f"epoch-{epoch:03d}"
            naive = json.loads((record_root / f"{slug}-naive.json").read_text())
            relocate = json.loads(
                (record_root / f"{slug}-relocate.json").read_text()
            )
            pair = seal_verifier._expected_pair(
                root, epoch, batch,
                next(
                    item["order"] for item in diagnostic["execution_order"]["epochs"]
                    if item["epoch"] == epoch
                ),
                naive, relocate, verifier,
            )
            path = root / "pairs" / f"epoch-{epoch:03d}/{slug}-pair.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, pair)
    _write(root / "summary.json", seal_verifier._sealed_summary(diagnostic))
    assert seal_verifier._verify_pairs_and_summary(
        root, diagnostic, verifier
    )["qualified"] is True
    pair = root / "pairs/epoch-001/qwen2.5-0.5b-b1-pair.json"
    changed = json.loads(pair.read_text())
    changed["exact_token_equality"] = False
    _write(pair, changed)
    with pytest.raises(RuntimeError, match="stored pair differs"):
        seal_verifier._verify_pairs_and_summary(root, diagnostic, verifier)


def test_component_junit_rejects_failure_despite_zero_suite_counter(
    tmp_path: Path,
) -> None:
    root = ET.Element(
        "testsuite", tests=str(len(seal_verifier.COMPONENT_CASES)),
        errors="0", failures="0", skipped="0",
    )
    properties = ET.SubElement(root, "properties")
    for name, value in {
        "orbitkv.cuda.available": "true",
        "orbitkv.cuda.device_name": verifier.GPU_NAME,
        "orbitkv.cuda.device_uuid": "GPU-test",
        "orbitkv.cuda.runtime_version": "13.0",
        "orbitkv.torch.version": "2.11.0+cu130",
    }.items():
        ET.SubElement(properties, "property", name=name, value=value)
    for index, name in enumerate(sorted(seal_verifier.COMPONENT_CASES)):
        case = ET.SubElement(root, "testcase", name=name)
        if index == 0:
            ET.SubElement(case, "failure")
    path = tmp_path / "component.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    with pytest.raises(RuntimeError, match="non-passing"):
        seal_verifier._verify_component_junit(
            path, {"name": verifier.GPU_NAME, "uuid": "GPU-test"}
        )


def _sealed_junit(path: Path, uuid: str) -> None:
    suite = ET.Element(
        "testsuite", tests=str(len(seal_verifier.COMPONENT_CASES)),
        errors="0", failures="0", skipped="0",
    )
    properties = ET.SubElement(suite, "properties")
    for name, value in {
        "orbitkv.cuda.available": "true",
        "orbitkv.cuda.device_name": verifier.GPU_NAME,
        "orbitkv.cuda.device_uuid": uuid,
        "orbitkv.cuda.runtime_version": "13.0",
        "orbitkv.torch.version": "2.11.0+cu130",
    }.items():
        ET.SubElement(properties, "property", name=name, value=value)
    for name in sorted(seal_verifier.COMPONENT_CASES):
        ET.SubElement(suite, "testcase", name=name)
    ET.ElementTree(suite).write(path, encoding="unicode")


def _git_blob(repository: Path, revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative}"], cwd=repository,
        check=True, capture_output=True,
    ).stdout


@pytest.fixture
def sealed_archive(
    evidence: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "sealed"
    root.mkdir()
    shutil.copytree(evidence / "records", root / "records")
    diagnostic = verifier.verify_evidence(root)
    for epoch in verifier.EPOCHS:
        order = next(
            item["order"] for item in diagnostic["execution_order"]["epochs"]
            if item["epoch"] == epoch
        )
        for batch in verifier.BATCHES:
            slug = f"{verifier.MODEL_SLUG}-b{batch}"
            record_root = root / "records" / f"epoch-{epoch:03d}"
            naive = json.loads((record_root / f"{slug}-naive.json").read_text())
            relocate = json.loads(
                (record_root / f"{slug}-relocate.json").read_text()
            )
            pair = seal_verifier._expected_pair(
                root, epoch, batch, order, naive, relocate, verifier
            )
            pair_path = root / "pairs" / f"epoch-{epoch:03d}/{slug}-pair.json"
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            _write(pair_path, pair)
            for mode in verifier.MODES:
                log = root / "logs" / f"epoch-{epoch:03d}/{slug}-{mode}.stderr.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("", encoding="utf-8")
    _write(root / "summary.json", seal_verifier._sealed_summary(diagnostic))
    (root / "README.md").write_text("sealed test archive\n", encoding="utf-8")
    uuid = diagnostic["hardware"]["observed_uuid"]
    _sealed_junit(root / "component-conformance.xml", uuid)

    repository = Path(__file__).resolve().parents[1]
    historical_revision = "cd78105"
    adapter = _historical_adapter()
    adapter_paths = {item["path"] for item in adapter["files"]}
    closure_paths = set(seal_verifier.SOURCE_REQUIRED_PATHS) | {
        path for path in adapter_paths
        if path.startswith("integrations/sglang/src/orbitkv_sglang/")
    }
    source_root = root / "qualification/source"
    source_inventory = []
    for name in sorted(closure_paths):
        if name in adapter_paths:
            data = _git_blob(repository, historical_revision, name)
        else:
            candidate = repository / name
            data = candidate.read_bytes() if candidate.is_file() else name.encode()
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_inventory.append(
            {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
        )
    closure = copy.deepcopy(source_inventory)
    commit = "a" * 40
    source = {
        "clean": True, "commit": commit,
        "tracked_file_count": len(source_inventory),
        "inventory_sha256": verifier.canonical_digest(source_inventory),
        "inventory": source_inventory,
    }
    baseline = json.loads(
        (root / "records/epoch-001/qwen2.5-0.5b-b1-naive.json").read_text()
    )
    checkpoint = baseline["checkpoint"]
    recorded_library = baseline["source_identity"]["library"]
    for epoch in verifier.EPOCHS:
        for batch in verifier.BATCHES:
            for mode in verifier.MODES:
                path = _path(root, epoch, batch, mode)
                record = json.loads(path.read_text(encoding="utf-8"))
                record["source_identity"]["plan"] = {
                    "path": "/run/qwen2.5-0.5b-full-page16-bf16.json",
                    "bytes": 301,
                    "sha256": seal_verifier.PLAN_SHA256,
                }
                record["environment"]["ORBITKV_PLAN"] = (
                    record["source_identity"]["plan"]["path"]
                )
                plan_index = record["command"].index("--plan") + 1
                record["command"][plan_index] = (
                    record["source_identity"]["plan"]["path"]
                )
                _write(path, record)
    # Recompute all pair files and the summary after rebinding the synthetic
    # records to the archived pinned plan.
    diagnostic = verifier.verify_evidence(root)
    for epoch in verifier.EPOCHS:
        order = next(
            item["order"] for item in diagnostic["execution_order"]["epochs"]
            if item["epoch"] == epoch
        )
        for batch in verifier.BATCHES:
            slug = f"{verifier.MODEL_SLUG}-b{batch}"
            record_root = root / "records" / f"epoch-{epoch:03d}"
            naive = json.loads((record_root / f"{slug}-naive.json").read_text())
            relocate = json.loads(
                (record_root / f"{slug}-relocate.json").read_text()
            )
            _write(
                root / "pairs" / f"epoch-{epoch:03d}/{slug}-pair.json",
                seal_verifier._expected_pair(
                    root, epoch, batch, order, naive, relocate, verifier
                ),
            )
    _write(root / "summary.json", seal_verifier._sealed_summary(diagnostic))
    plan_source = repository / ".qualification/plans/qwen2.5-0.5b-full-page16-bf16.json"
    plan_path = root / f"qualification/plans/{plan_source.name}"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_source, plan_path)
    model_path = root / "qualification/model/config.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path("/workspace/models/qwen2.5-0.5b-instruct/config.json"), model_path)
    model_identity = verifier.canonical_digest(checkpoint)
    _write(
        root / "qualification/model-provenance.json",
        {
            "schema": seal_verifier.MODEL_PROVENANCE_SCHEMA,
            "name": verifier.MODEL_SLUG,
            "directory_name": verifier.MODEL_PATH_BASENAME,
            "config_sha256": verifier.CHECKPOINT_CONFIG_SHA256,
            "weight_files": checkpoint["weight_files"],
            "checkpoint_identity_sha256": model_identity,
        },
    )
    input_lock = repository / ".qualification/requirements-v0.5.17.lock.txt"
    archived_input = root / "qualification/requirements.input.lock.txt"
    shutil.copy2(input_lock, archived_input)
    locked_editable = next(
        line for line in input_lock.read_text().splitlines()
        if line.startswith("-e git+") and "#egg=orbitkv_sglang" in line
    )
    active_editable = seal_verifier.ORBITKV_EDITABLE_TEMPLATE.format(commit=commit)
    materialized = root / "qualification/requirements.lock.txt"
    materialized.write_text(
        seal_verifier._materialize_requirements_lock(archived_input, active_editable),
        encoding="utf-8",
    )
    library_path = root / "qualification/build/liborbitkv_ffi.so"
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_bytes(b"synthetic ELF replaced by trusted test stub")
    bundle = root / "qualification/source.bundle"
    bundle.write_bytes(b"synthetic self-contained bundle")
    preflight = {
        "schema": seal_verifier.SEALED_PREFLIGHT_SCHEMA,
        "qualification_scope": seal_verifier.QUALIFICATION_SCOPE,
        "status": "host_preflight_passed_gpu_not_initialized",
        "source": source, "source_closure": closure,
        "benchmark": {
            "path": "/run/bench_token_relocation.py",
            "sha256": verifier.DIAGNOSTIC_HARNESS_SHA256,
            "record_schema": verifier.RECORD_SCHEMA,
        },
        "verifier": {
            "path": "/run/verify_token_relocation_h20_evidence.py",
            "sha256": next(
                item["sha256"] for item in closure
                if item["path"] == "tools/verify_token_relocation_h20_evidence.py"
            ),
        },
        "library": {
            **recorded_library, "abi_version": 8,
            "symbols": sorted(seal_verifier.EXACT_ABI8_SYMBOLS),
        },
        "build": {
            "command": ["cargo", "build", "--release", "--locked",
                        "--manifest-path", "/repo/crates/orbitkv-ffi/Cargo.toml"],
            "cargo_version": "cargo 1.0", "cargo_target_dir": "/tmp/build",
        },
        "sglang": {
            "release": verifier.SGLANG_RELEASE,
            "revision": verifier.SGLANG_REVISION,
            "manager_root": baseline["source_identity"]["root"],
            "manager": {
                "root": baseline["source_identity"]["root"],
                "release": verifier.SGLANG_RELEASE,
                "revision": verifier.SGLANG_REVISION,
                "python_source_contract": verifier.SOURCE_CONTRACT,
                "dirty_paths": [verifier.LOADER_PATH],
                "loader": baseline["source_identity"]["loader"],
                "tag": verifier.SGLANG_RELEASE,
                "remote": "https://github.com/sgl-project/sglang.git",
            },
            "pinned_contract": {
                "release": verifier.SGLANG_RELEASE,
                "revision": verifier.SGLANG_REVISION,
                "loader_path": verifier.LOADER_PATH,
                "base_source_sha256": "3a975a73f1a7887e68c81ea7a2530250597ac8ae978efc0b0f70f038a99a3164",
                "patched_source_sha256": verifier.LOADER_WORKTREE_SHA256,
                "patch_diff_sha256": verifier.LOADER_PATCH_SHA256,
            },
            "manager_entrypoint": baseline["source_identity"]["plugin_selection"],
        },
        "inputs": {
            "requirements": {
                "path": "/run/requirements-v0.5.17.lock.txt",
                "bytes": input_lock.stat().st_size,
                "sha256": seal_verifier.REQUIREMENTS_INPUT_SHA256,
            },
            "plan": {
                "path": "/run/qwen2.5-0.5b-full-page16-bf16.json",
                "bytes": plan_path.stat().st_size, "sha256": seal_verifier.PLAN_SHA256,
            },
            "model": {
                "name": verifier.MODEL_SLUG,
                "root": f"/models/{verifier.MODEL_PATH_BASENAME}",
                "checkpoint": checkpoint, "identity_sha256": model_identity,
            },
        },
        "python": {
            "executable": "/venv/bin/python", "real_executable": "/usr/bin/python",
            "normalized_freeze_sha256": "b" * 64,
            "active_editable": active_editable, "locked_editable": locked_editable,
        },
    }
    _write(root / "preflight.json", preflight)
    manifest = _minimal_sealed_manifest()
    manifest.update(
        source_commit=commit,
        source_inventory_sha256=source["inventory_sha256"],
        model_identity_sha256=model_identity,
        library_sha256=recorded_library["sha256"],
        input_hashes={
            "requirements_input_sha256": seal_verifier.REQUIREMENTS_INPUT_SHA256,
            "requirements_sha256": seal_verifier._sha256_file(materialized),
            "plan_sha256": seal_verifier.PLAN_SHA256,
            "model": {
                "config_sha256": verifier.CHECKPOINT_CONFIG_SHA256,
                "weight_sha256": verifier.CHECKPOINT_WEIGHT_SHA256,
                "weight_bytes": verifier.CHECKPOINT_WEIGHT_BYTES,
            },
        },
        observed_hardware={
            "name": verifier.GPU_NAME, "uuid": uuid,
            "snapshot_count": 64, "attestation": "recorded_observation_only",
        },
    )
    manifest["source_provenance"] = {
        "kind": "git_bundle", "path": "qualification/source.bundle",
        "sha256": seal_verifier._sha256_file(bundle),
        "commit": commit, "reference": "HEAD",
    }
    manifest["artifacts"] = {
        path.relative_to(root).as_posix(): seal_verifier._sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    _write(root / "manifest.json", manifest)
    sums = dict(manifest["artifacts"])
    sums["manifest.json"] = seal_verifier._sha256_file(root / "manifest.json")
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    monkeypatch.setattr(seal_verifier, "_verify_source_bundle", lambda *_: None)
    fake_library = {
        "sha256": recorded_library["sha256"],
        "bytes": recorded_library["bytes"],
        "abi_version": 8,
        "symbols": sorted(seal_verifier.EXACT_ABI8_SYMBOLS),
    }
    monkeypatch.setattr(
        seal_verifier, "_verify_library",
        lambda *_: fake_library,
    )
    return root


def test_synthetic_sealed_archive_runs_the_complete_pipeline(
    sealed_archive: Path,
) -> None:
    result = seal_verifier.verify_sealed_archive(sealed_archive)
    assert result == {
        "schema": seal_verifier.SEALED_VERIFICATION_SCHEMA,
        "status": "passed",
        "qualification_status": seal_verifier.QUALIFICATION_STATUS,
        "qualification_claim": seal_verifier.QUALIFICATION_CLAIM,
        "sealed": True, "source_clean": True, "preflight_bound": True,
        "hardware_attested": False, "qualified": True,
        "performance_go": False, "epoch_count": 4,
        "record_count": 16, "pair_count": 8, "abi_version": 8,
        "exact_symbol_count": 40, "all_pairs_passed": True,
        "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }


def test_sealed_archive_rejects_unlisted_file_and_stored_summary_tamper(
    sealed_archive: Path,
) -> None:
    (sealed_archive / "unlisted").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        seal_verifier.verify_sealed_archive(sealed_archive)
    (sealed_archive / "unlisted").unlink()
    summary_path = sealed_archive / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["exact_token_equality"] = False
    _write(summary_path, summary)
    manifest_path = sealed_archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["summary.json"] = seal_verifier._sha256_file(summary_path)
    _write(manifest_path, manifest)
    sums = dict(manifest["artifacts"])
    sums["manifest.json"] = seal_verifier._sha256_file(manifest_path)
    (sealed_archive / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stored summary differs"):
        seal_verifier.verify_sealed_archive(sealed_archive)
