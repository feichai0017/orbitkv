from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(INTEGRATION_ROOT / "src"))

import qualify_abi8_h20 as qualification  # noqa: E402


def _refresh_record_derivations(record: dict) -> None:
    workload = record["workload"]
    rows = []
    for iteration in range(workload["iterations"]):
        row = []
        for request in range(workload["requests"]):
            token = (iteration + request + 1) if (
                record.get("workload_profile") == "fresh_prompt"
            ) else request + 1
            output_ids = [token] * workload["decode_tokens"]
            request_id = f"request-{iteration}-{request}"
            row.append({
                "request_index": request,
                "submitted_rid": request_id,
                "returned_rid": request_id,
                "submitted_input_ids_sha256": (
                    f"input-{iteration}-{request}"
                    if record.get("workload_profile") == "fresh_prompt"
                    else "input"
                ),
                "cached_tokens": (
                    0 if record.get("workload_profile") == "fresh_prompt"
                    else record.get("prefix_seed", {}).get(
                        "observed_hit_boundary_tokens", 512
                    )
                ),
                "output_ids": output_ids,
                "output_ids_sha256": qualification.canonical_digest(output_ids),
            })
        rows.append(row)
    record["request_traces"] = rows
    digests = [[item["output_ids_sha256"] for item in row] for row in rows]
    outputs = [[item["output_ids"] for item in row] for row in rows]
    record["output_request_digests_sha256"] = digests
    record["output_token_digest_sha256"] = qualification.canonical_digest(outputs)
    record["completed_requests"] = workload["requests"] * workload["iterations"]
    record["completion_tokens"] = sum(
        len(item["output_ids"]) for row in rows for item in row
    )
    record["iteration_seconds"] = [1.0] * workload["iterations"]
    for name in ("command", "environment", "source_identity"):
        record[f"{name}_sha256"] = qualification.canonical_digest(record[name])


def _build_abi8_stub(path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None or shutil.which("nm") is None:
        pytest.skip("ABI8 seal test requires cc and nm")
    source = path.with_suffix(".c")
    functions = sorted(
        qualification.EXACT_SYMBOL_ALLOWLIST - {"orbitkv_abi_version"}
    )
    source.write_text(
        "#include <stdint.h>\n"
        "uint32_t orbitkv_abi_version(void) { return 8; }\n"
        + "".join(f"int32_t {name}(void) {{ return 0; }}\n" for name in functions),
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-o", str(path), str(source)],
        check=True, capture_output=True, text=True,
    )


def test_python_lock_normalizes_only_the_active_editable_commit(tmp_path, monkeypatch):
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "package==1\n-e git+https://example/orbitkv.git@old#egg=orbitkv_sglang\n",
        encoding="utf-8",
    )

    def fake_run(arguments, **_kwargs):
        if tuple(arguments[-3:]) == ("pip", "freeze", "--all"):
            return subprocess.CompletedProcess(
                arguments, 0,
                "package==1\n-e git+https://example/orbitkv.git@new#egg=orbitkv_sglang\n",
                "",
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(qualification, "_run", fake_run)
    identity = qualification._verify_python_environment(Path(sys.executable), lock)
    assert "@new#egg=orbitkv_sglang" in identity["active_editable"]
    assert "@old#egg=orbitkv_sglang" in identity["locked_editable"]
    assert identity["executable"] == str(Path(sys.executable).absolute())


def _record(
    mode: str, profile: str = "full", *, schema: str | None = None
) -> dict:
    record_schema = qualification.benchmark.RECORD_SCHEMA if schema is None else schema
    engine = {"model_path": "/model", "max_total_tokens": 1024}
    if mode == "manager":
        engine["radix_cache_backend"] = "orbitkv"
    swa = {
        "status": "not_applicable" if profile == "full" else "exposed",
        "swa_retirement_certificates": 0 if profile == "full" else 3,
        "swa_pages_reclaimed": 0 if profile == "full" else 2,
        "swa_wrap_events": 0 if profile == "full" else 1,
    }
    after_load_swa = dict(swa)
    if profile != "full":
        for name in ("swa_retirement_certificates", "swa_pages_reclaimed", "swa_wrap_events"):
            after_load_swa[name] = 0
    counters = {name: 0 for name in qualification.FIXED_STATE_COUNTERS}
    manager = None if mode == "stock" else {
        "after_load": {"swa_activity": after_load_swa},
        "final_census": {
            "abi_version": 8,
            "manager_stats": {
                "active_requests": 0, "active_snapshots": 0,
                "active_prefixes": 0, "active_pages": 0,
                "pending_reclamations": 0, "quarantined_pages": 0,
            },
            "batch_counters": counters, "swa_activity": swa,
        },
    }
    source = {
        "release": "v0.5.17", "revision": qualification.SGLANG_REVISION,
        "harness_sha256": qualification.sha256_file(Path(qualification.benchmark.__file__).resolve()),
        "adapter": qualification.benchmark._adapter_identity(),
        "python_source_contract": (
            "clean_pinned_head" if mode == "stock"
            else "pinned_head_plus_canonical_loader_patch"
        ),
        "library": None, "plan": None,
    }
    manager_record = manager
    if mode == "manager":
        source["library"] = {"path": "/lib", "sha256": "library"}
        source["plan"] = {"path": "/plan", "sha256": "plan"}
        manager_record["library"] = source["library"]
        manager_record["plan"] = {"artifact": source["plan"]}
        manager_record["counter_contract"] = (
            qualification.benchmark._manager_counter_contract(fresh=False)
            if record_schema == qualification.benchmark.RECORD_SCHEMA
            else {"mirror_validation_equals_sync": True}
        )
    record = {
        "schema": record_schema, "mode": mode,
        "command": ["python", "benchmark.py"],
        "environment": {"PYTHONHASHSEED": "0"},
        "manager": manager_record, "engine_args": engine,
        "source_identity": source,
        "checkpoint": {"config_sha256": "config"},
        "checkpoint_identity_sha256": "checkpoint",
        "checkpoint_contract": {
            "attention_profile": profile,
            "workload_profile": "prefix_reuse",
        },
        "sampling_params": {"temperature": 0},
        "workload_profile": "prefix_reuse",
        "prefix_seed": {"prompt_tokens": 512},
        "workload": {
            "requests": 1, "iterations": 1, "profile": "prefix_reuse",
        },
        "capacity_readback": {"full_tokens": 1024},
        "output_token_digest_sha256": "tokens",
        "output_request_digests_sha256": [["request"]],
        "request_traces": [[{"output_ids": [1]}]],
        "completed_requests": 1, "completion_tokens": 33,
        "iteration_seconds": [1.0],
        "pairing": {},
    }
    record["checkpoint_identity_sha256"] = qualification.canonical_digest(
        record["checkpoint"]
    )
    contract = qualification._record_pair_contract(record)
    record["pairing"] = {
        "contract": contract,
        "pair_key_sha256": qualification.canonical_digest(contract),
    }
    for name in ("command", "environment", "source_identity"):
        record[f"{name}_sha256"] = qualification.canonical_digest(record[name])
    return record


def _gdn_record(
    mode: str, *, requests: int = 1, iterations: int = 1,
    schema: str | None = None,
) -> dict:
    record = _record(mode, schema=schema)
    record["checkpoint_contract"] = {
        "architecture": "Qwen3_5ForConditionalGeneration",
        "attention_profile": "hybrid_full_gdn",
        "attention_backend": "fa3",
        "backend_profile": dict(qualification.QWEN35_BACKEND_PROFILE),
        "workload_profile": "fresh_prompt",
        "state_ownership": "request_private",
        "vocab_size": 128,
        "prompt_token_upper_bound": 120,
        "control_token_ids": {
            "image_token_id": 124, "video_token_id": 125,
            "vision_start_token_id": 122, "vision_end_token_id": 123,
        },
    }
    record["workload_profile"] = "fresh_prompt"
    record["prefix_seed"] = None
    record["workload"].update(
        requests=requests, iterations=iterations, prompt_tokens=17,
        decode_tokens=33, seed=20260820,
        profile="fresh_prompt",
    )
    prompts = [
        qualification.benchmark.fresh_input_ids(
            requests=requests, prompt_tokens=17, vocab_size=128,
            seed=20260820, iteration=iteration,
            forbidden_token_ids=(124, 125, 122, 123),
            token_upper_bound=120,
        )
        for iteration in range(iterations)
    ]
    record["workload"]["input_token_digests_by_iteration_sha256"] = [
        [qualification.canonical_digest(prompt) for prompt in row]
        for row in prompts
    ]
    record["workload"]["input_token_digest_sha256"] = (
        qualification.benchmark.input_digest(
            [prompt for row in prompts for prompt in row]
        )
    )
    record["engine_args"]["disable_radix_cache"] = True
    record["runtime_identity"] = {
        "backend_profile": dict(qualification.QWEN35_BACKEND_PROFILE),
    }
    if mode == "manager":
        state_artifact = {
            "path": "/state-plan", "sha256": "state-plan", "bytes": 31,
        }
        record["source_identity"]["state_plan"] = state_artifact
        plan = record["manager"]["plan"]
        plan.update({
            "plan_fingerprint": "sha256:manager",
            "state_plan": {
                "artifact": state_artifact, "plan_fingerprint": "sha256:state",
            },
            "fixed_state_byte_count": 96,
            "fixed_states": [{"name": "gdn"}],
            "classes": [{"name": "full_attention_kv"}],
        })
        record["manager"]["counter_contract"] = (
            qualification.benchmark._manager_counter_contract(fresh=True)
            if record["schema"] == qualification.benchmark.RECORD_SCHEMA
            else {
                "mirror_validation_equals_sync": True,
                "fixed_state_drain_stages": [
                    "after_load", "after_workload",
                    "after_global_cleanup",
                ],
            }
        )
        identity = {
            "engine_epoch": 7, "pool_epoch": 8, "pool_id": 1,
            "page_count": 64,
        }
        fixed = {
            "status": "host_seam",
            "identity": {
                "engine_epoch": 7, "pool_epoch": 9, "pool_id": 2,
                "byte_count": 96, "slot_count": 8,
            },
            "free_slots": 8, "reserved_slots": 0, "relocating_slots": 0,
            "live_slots": 0, "retiring_slots": 0, "quarantined_slots": 0,
            "active_owners": 0, "pending_transitions": 0,
            "pending_retirements": 0,
        }
        counters = qualification.benchmark.expected_batch_counters(
            batch_size=requests, completed_iterations=iterations,
            prompt_tokens=17, decode_tokens=33,
            hybrid=False, prefix_seeded=False, global_cleanup=False, fresh=True,
            legacy_equal_mirror_counters=(
                record["schema"] in qualification.LEGACY_RECORD_SCHEMAS
            ),
        )
        counters.update({
            "fixed_state_prepares": requests * iterations,
            "fixed_state_clears": requests * iterations,
            "fixed_state_copies": 0,
            "fixed_state_events": 33 * iterations,
            "fixed_state_retirements": requests * iterations,
            "fixed_state_acks": requests * iterations,
        })
        census = {
            "abi_version": 8,
            "plan_fingerprint": plan["plan_fingerprint"],
            "state_plan_fingerprint": plan["state_plan"]["plan_fingerprint"],
            "fixed_state_byte_count": plan["fixed_state_byte_count"],
            "fixed_state_descriptors": copy.deepcopy(plan["fixed_states"]),
            "tree_cache_type": {
                "module": "orbitkv_sglang.plugin.prefix_cache",
                "qualname": "OrbitKvPrefixCache",
            },
            "identities": [identity],
            "manager_stats": {
                "active_requests": 0, "active_snapshots": 0,
                "active_prefixes": 0, "active_pages": 0,
                "reserved_pages": 0, "writing_pages": 0,
                "retiring_pages": 0, "quarantined_pages": 0,
                "exhausted_pages": 0, "pending_reclamations": 0,
                "total_request_page_refs": 0, "total_prefix_page_refs": 0,
                "total_reader_pins": 0, "free_pages": 64,
            },
            "batch_counters": counters,
            "swa_activity": {
                "status": "not_applicable",
                "swa_retirement_certificates": 0, "swa_pages_reclaimed": 0,
                "swa_wrap_events": 0,
            },
            "fixed_state": fixed,
        }
        record["manager"]["after_load"] = copy.deepcopy(census)
        record["manager"]["after_workload"] = copy.deepcopy(census)
        record["manager"]["final_census"] = copy.deepcopy(census)
    else:
        record["source_identity"]["state_plan"] = None
    record["checkpoint_identity_sha256"] = qualification.canonical_digest(
        record["checkpoint"]
    )
    record["pairing"]["contract"] = qualification._record_pair_contract(record)
    record["pairing"]["pair_key_sha256"] = qualification.canonical_digest(
        record["pairing"]["contract"]
    )
    _refresh_record_derivations(record)
    for trace_row, digest_row in zip(
        record["request_traces"],
        record["workload"]["input_token_digests_by_iteration_sha256"],
        strict=True,
    ):
        for trace, digest in zip(trace_row, digest_row, strict=True):
            trace["submitted_input_ids_sha256"] = digest
    return record


@pytest.mark.parametrize("profile", ("full", "hybrid_full_swa"))
def test_pair_verifier_accepts_only_declared_radix_backend_difference(profile):
    result = qualification.verify_pair_records(_record("stock", profile), _record("manager", profile))
    assert result["status"] == "passed"
    assert result["profile"] == profile


@pytest.mark.parametrize(
    "schema",
    (
        "orbitkv.sglang-v0517-abi8-single-run.v1",
        "orbitkv.sglang-v0517-abi8-single-run.v2",
    ),
)
def test_pair_verifier_accepts_v1_and_v2_prefix_archives(schema):
    stock = _record("stock", schema=schema)
    manager = _record("manager", schema=schema)
    assert qualification.verify_pair_records(stock, manager)["status"] == "passed"


def test_pair_verifier_rejects_exact_token_drift():
    stock = _record("stock")
    manager = _record("manager")
    manager["request_traces"] = [[{"output_ids": [2]}]]
    with pytest.raises(RuntimeError, match="request_traces"):
        qualification.verify_pair_records(stock, manager)


def test_pair_verifier_recomputes_pair_key():
    stock = _record("stock")
    manager = _record("manager")
    stock["pairing"]["pair_key_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="pair key"):
        qualification.verify_pair_records(stock, manager)


def test_pair_verifier_rejects_fixed_state_activity_for_token_models():
    manager = _record("manager")
    manager["manager"]["final_census"]["batch_counters"]["fixed_state_events"] = 1
    with pytest.raises(RuntimeError, match="fixed-state"):
        qualification.verify_pair_records(_record("stock"), manager)


def test_pair_verifier_rejects_hybrid_without_strict_swa_progress():
    manager = _record("manager", "hybrid_full_swa")
    manager["manager"]["final_census"]["swa_activity"]["swa_wrap_events"] = 0
    with pytest.raises(RuntimeError, match="SWA counters"):
        qualification.verify_pair_records(_record("stock", "hybrid_full_swa"), manager)


def test_pair_verifier_accepts_qwen35_fresh_gdn_pair():
    result = qualification.verify_pair_records(
        _gdn_record("stock", requests=4, iterations=2),
        _gdn_record("manager", requests=4, iterations=2),
    )
    assert result["status"] == "passed"
    assert result["profile"] == "hybrid_full_gdn"
    assert result["workload_profile"] == "fresh_prompt"
    assert result["qualification_claim"] == "pair_verification_only_not_qualified"


def test_pair_verifier_accepts_qwen35_v2_archive_mirror_counters():
    schema = "orbitkv.sglang-v0517-abi8-single-run.v2"
    stock = _gdn_record("stock", requests=4, iterations=5, schema=schema)
    manager = _gdn_record("manager", requests=4, iterations=5, schema=schema)
    counters = manager["manager"]["final_census"]["batch_counters"]
    assert counters["mirror_validation_calls"] == 20
    assert counters["mirror_syncs"] == 20
    assert qualification.verify_pair_records(stock, manager)["status"] == "passed"


@pytest.mark.parametrize(
    ("requests", "iterations", "validations", "syncs"),
    ((1, 1, 4, 1), (4, 5, 20, 5)),
)
def test_pair_verifier_accepts_v3_fresh_mirror_counter_contract(
    requests, iterations, validations, syncs
):
    stock = _gdn_record("stock", requests=requests, iterations=iterations)
    manager = _gdn_record("manager", requests=requests, iterations=iterations)
    counters = manager["manager"]["final_census"]["batch_counters"]
    assert counters["mirror_validation_calls"] == validations
    assert counters["mirror_syncs"] == syncs
    assert qualification.verify_pair_records(stock, manager)["status"] == "passed"


@pytest.mark.parametrize("field", ("mirror_validation_calls", "mirror_syncs"))
def test_pair_verifier_rejects_v3_fresh_mirror_counter_drift(field):
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    manager["manager"]["final_census"]["batch_counters"][field] += 1
    with pytest.raises(RuntimeError, match="GDN final batch counters"):
        qualification.verify_pair_records(stock, manager)


def test_unbound_pair_file_verification_is_explicitly_nonqualifying(tmp_path):
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    stock_path = tmp_path / "stock.json"
    manager_path = tmp_path / "manager.json"
    stock_path.write_text(json.dumps(stock), encoding="utf-8")
    manager_path.write_text(json.dumps(manager), encoding="utf-8")
    result = qualification.verify_pair_files(stock_path, manager_path)
    assert result["status"] == "passed"
    assert result["preflight_bound"] is False
    assert result["hardware_attested"] is False
    assert result["qualified"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: record["manager"]["plan"].update(state_plan=None), "state-plan"),
        (lambda record: record["manager"]["final_census"]["batch_counters"].update(fixed_state_events=32), "fixed-state counters"),
        (lambda record: record["manager"]["final_census"]["batch_counters"].update(fixed_state_copies=1), "fixed-state counters"),
        (lambda record: record["manager"]["final_census"]["fixed_state"].update(live_slots=1, free_slots=7), "fixed-state pool did not drain"),
        (lambda record: record.update(prefix_seed={"prompt_tokens": 512}), "fresh-prompt"),
        (lambda record: record["manager"]["final_census"]["swa_activity"].update(swa_wrap_events=1), "SWA telemetry"),
    ),
)
def test_pair_verifier_rejects_invalid_qwen35_gdn_evidence(mutation, message):
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    mutation(manager)
    with pytest.raises(RuntimeError, match=message):
        qualification.verify_pair_records(stock, manager)


def test_pair_verifier_rejects_stock_state_plan_for_qwen35():
    stock = _gdn_record("stock")
    stock["source_identity"]["state_plan"] = {"sha256": "forged"}
    with pytest.raises(RuntimeError, match="stock record unexpectedly"):
        qualification.verify_pair_records(stock, _gdn_record("manager"))


def test_pair_verifier_rejects_qwen35_workload_and_backend_drift():
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    manager["runtime_identity"]["backend_profile"]["linear_attn_backend"] = "torch"
    with pytest.raises(RuntimeError, match="backend profile"):
        qualification.verify_pair_records(stock, manager)

    manager = _gdn_record("manager")
    manager["workload_profile"] = "prefix_reuse"
    with pytest.raises(RuntimeError, match="workload_profile|workload profile"):
        qualification.verify_pair_records(stock, manager)


def test_pair_verifier_reconstructs_qwen35_fresh_prompt_digests():
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    for record in (stock, manager):
        record["request_traces"][0][0][
            "submitted_input_ids_sha256"
        ] = "forged-shared-prompt"
        record["workload"][
            "input_token_digests_by_iteration_sha256"
        ][0][0] = "forged-shared-prompt"
        record["pairing"]["contract"] = (
            qualification._record_pair_contract(record)
        )
        record["pairing"]["pair_key_sha256"] = qualification.canonical_digest(
            record["pairing"]["contract"]
        )
    with pytest.raises(RuntimeError, match="fresh-prompt input digests"):
        qualification.verify_pair_records(stock, manager)


def test_pair_verifier_rejects_qwen35_cached_tokens():
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    for record in (stock, manager):
        record["request_traces"][0][0]["cached_tokens"] = 1
    with pytest.raises(RuntimeError, match="cached-token evidence"):
        qualification.verify_pair_records(stock, manager)


@pytest.mark.parametrize(
    "stage", ("after_load", "after_workload", "final_census")
)
@pytest.mark.parametrize(
    "field", (
        "plan_fingerprint", "state_plan_fingerprint",
        "fixed_state_descriptors", "tree_cache_type",
    )
)
def test_pair_verifier_rejects_qwen35_worker_plan_readback_drift(stage, field):
    manager = _gdn_record("manager")
    manager["manager"][stage][field] = "forged"
    with pytest.raises(RuntimeError, match="worker plan readback"):
        qualification.verify_pair_records(_gdn_record("stock"), manager)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("completed_requests", 0, "completion totals"),
        ("completion_tokens", 1, "completion totals"),
        ("output_token_digest_sha256", "forged", "output token digest"),
        ("output_request_digests_sha256", [], "request digest matrix"),
    ),
)
def test_pair_verifier_rejects_qwen35_derived_record_drift(
    field, value, message
):
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    for record in (stock, manager):
        record[field] = copy.deepcopy(value)
    with pytest.raises(RuntimeError, match=message):
        qualification.verify_pair_records(stock, manager)


def test_qwen35_pair_preflight_binds_state_plan(tmp_path):
    stock = _gdn_record("stock")
    manager = _gdn_record("manager")
    stock_path = tmp_path / "stock.json"
    manager_path = tmp_path / "manager.json"
    model = {
        "config.json": {"sha256": "config"},
        "model.safetensors.index.json": {"sha256": "index"},
        "weight_shards": [{"filename": "weight.safetensors", "size": 7, "sha256": "weight"}],
        "weight_shards_sha256": qualification.canonical_digest([
            {"filename": "weight.safetensors", "size": 7, "sha256": "weight"}
        ]),
    }
    checkpoint = {
        "config_sha256": "config",
        "index_files": [{"name": "model.safetensors.index.json", "sha256": "index"}],
        "indexed_weight_files": ["weight.safetensors"],
        "weight_files": [{"name": "weight.safetensors", "bytes": 7, "sha256": "weight"}],
    }
    for record in (stock, manager):
        record["checkpoint"] = checkpoint
        record["checkpoint_identity_sha256"] = qualification.canonical_digest(checkpoint)
        record["source_identity"].update(
            root=f"/{record['mode']}", pinned_contract={"pinned": True},
            loader={"mode": record["mode"]},
        )
        record["runtime_identity"]["python_executable"] = sys.executable
        record["pairing"]["contract"] = qualification._record_pair_contract(record)
        record["pairing"]["pair_key_sha256"] = qualification.canonical_digest(
            record["pairing"]["contract"]
        )
    preflight = {
        "qualification_scope": "qwen35",
        "source": {"commit": "a", "inventory_sha256": "source"},
        "library": {"path": "/lib", "sha256": "library", "bytes": None},
        "inputs": {
            "models": {"qwen3.5-0.8b": model},
            "plans": {"qwen3.5-0.8b": manager["source_identity"]["plan"]},
            "state_plans": {"qwen3.5-0.8b": manager["source_identity"]["state_plan"]},
        },
        "sglang": {
            "release": "v0.5.17", "revision": qualification.SGLANG_REVISION,
            "stock_root": "/stock", "manager_root": "/manager",
            "pinned_contract": {"pinned": True},
            "stock": {"loader": {"mode": "stock"}},
            "manager": {"loader": {"mode": "manager"}},
        },
        "python": {"executable": sys.executable},
    }
    manager["source_identity"]["library"] = preflight["library"]
    manager["manager"]["library"] = preflight["library"]
    for record in (stock, manager):
        record["source_identity_sha256"] = qualification.canonical_digest(
            record["source_identity"]
        )
    stock_path.write_text(json.dumps(stock), encoding="utf-8")
    manager_path.write_text(json.dumps(manager), encoding="utf-8")
    result = qualification.verify_pair_files(stock_path, manager_path, preflight)
    assert result["preflight_bound"] is True
    assert result["hardware_attested"] is False
    assert result["qualified"] is False
    assert result["input_identity"]["state_plan_sha256"] == "state-plan"
    preflight["inputs"]["state_plans"]["qwen3.5-0.8b"] = {
        "path": "/other", "sha256": "other", "bytes": 31,
    }
    with pytest.raises(RuntimeError, match="state plan differs"):
        qualification.verify_pair_files(stock_path, manager_path, preflight)


def test_summary_aggregates_epochs_and_never_claims_performance_go():
    pair = qualification.verify_pair_records(_record("stock"), _record("manager"))
    second = copy.deepcopy(pair)
    second["stock_iteration_seconds"] = [2.0]
    second["manager_iteration_seconds"] = [3.0]
    summary = qualification.summarize_pairs((pair, second))
    assert summary["pair_count"] == 2
    assert summary["groups"][0]["epoch_count"] == 2
    assert summary["groups"][0]["sample_count_per_mode"] == 2
    assert summary["performance_go"] is False


def test_default_cli_action_is_host_only_preflight():
    parser = qualification.build_parser()
    assert parser.parse_args(["preflight", "--work-dir", "records"]).action == "preflight"
    run = parser.parse_args(["run", "--work-dir", "records", "--execute", qualification.EXECUTION_TOKEN])
    assert run.action == "run"
    assert run.phase == "all"
    qwen35 = parser.parse_args([
        "run", "--work-dir", "records", "--execute",
        qualification.EXECUTION_TOKEN, "--phase", "qwen35",
    ])
    assert qwen35.phase == "qwen35"
    preflight = parser.parse_args(["preflight", "--work-dir", "records"])
    assert preflight.scope == "sealed"
    assert (
        preflight.qwen35_plan.name
        == "qwen3.5-0.8b-token-manager-page16-bf16.json"
    )
    assert (
        preflight.qwen35_state_plan.name
        == "qwen3.5-0.8b-attention-state-input-page16-bf16.json"
    )
    assert (
        qualification.sha256_file(preflight.qwen35_plan)
        == qualification.QWEN35_TOKEN_PLAN_SHA256
    )
    assert (
        qualification.sha256_file(preflight.qwen35_state_plan)
        == qualification.QWEN35_STATE_INPUT_SHA256
    )
    assert parser.parse_args([
        "preflight", "--work-dir", "records", "--scope", "qwen35",
    ]).scope == "qwen35"
    verify = parser.parse_args(["verify-seal", "sealed"])
    assert verify.action == "verify-seal"
    assert verify.seal_dir == Path("sealed")
    pair = parser.parse_args([
        "verify-pair", "stock.json", "manager.json",
        "--preflight", "preflight.json",
    ])
    assert pair.preflight == Path("preflight.json")


def test_preflight_input_scope_does_not_require_unselected_models(
    tmp_path, monkeypatch
):
    def model(name):
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "model.safetensors"}}),
            encoding="utf-8",
        )
        (root / "model.safetensors").write_bytes(name.encode())
        return root

    selected = {name: model(name) for name in ("qwen", "gpt", "qwen35")}
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    missing = tmp_path / "must-not-be-read"

    def identity(path, expected, _label):
        return {
            "path": str(path), "sha256": expected or "observed",
            "bytes": path.stat().st_size,
        }

    monkeypatch.setattr(qualification, "_require_hash", identity)
    common = {
        "requirements": artifact, "qwen_plan": artifact,
        "gpt_plan": artifact, "qwen35_plan": artifact,
        "qwen35_state_plan": artifact,
    }
    sealed = type("Args", (), {
        **common, "scope": "sealed", "qwen_model": selected["qwen"],
        "gpt_model": selected["gpt"], "qwen35_model": missing,
    })()
    sealed_inputs = qualification._input_identity(sealed)
    assert set(sealed_inputs["models"]) == {"qwen2.5-7b", "gpt-oss-20b"}
    assert "state_plans" not in sealed_inputs

    qwen35 = type("Args", (), {
        **common, "scope": "qwen35", "qwen_model": missing,
        "gpt_model": missing, "qwen35_model": selected["qwen35"],
    })()
    qwen35_inputs = qualification._input_identity(qwen35)
    assert set(qwen35_inputs["models"]) == {"qwen3.5-0.8b"}
    assert set(qwen35_inputs["plans"]) == {"qwen3.5-0.8b"}
    assert set(qwen35_inputs["state_plans"]) == {"qwen3.5-0.8b"}


def test_pair_verifier_does_not_compare_untimed_prefix_seed_duration():
    stock = _record("stock")
    manager = _record("manager")
    stock["prefix_seed"]["seconds"] = 1.0
    manager["prefix_seed"]["seconds"] = 2.0
    assert qualification.verify_pair_records(stock, manager)["status"] == "passed"


def test_execution_order_balances_adjacent_epochs():
    assert qualification.execution_order(1) == ("stock", "manager")
    assert qualification.execution_order(2) == ("manager", "stock")
    assert qualification.execution_order(3) == ("stock", "manager")
    with pytest.raises(ValueError, match="positive"):
        qualification.execution_order(0)


def test_run_matrix_records_actual_balanced_order(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "preflight.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(qualification, "_validate_preflight", lambda _pre: None)
    monkeypatch.setattr(qualification, "_assert_idle_h20", lambda: None)
    monkeypatch.setattr(qualification, "_case_command", lambda _args, _pre, case, mode: [mode, case.slug])
    calls = []

    def fake_run(command, output, stderr):
        calls.append(command[0])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")

    monkeypatch.setattr(qualification, "_run_record", fake_run)
    monkeypatch.setattr(
        qualification, "verify_pair_files",
        lambda _stock, _manager, _pre=None: {
            "schema": qualification.PAIR_SCHEMA, "status": "passed",
            "profile": "full", "batch_size": 1, "iterations": 1,
            "stock_iteration_seconds": [1.0], "manager_iteration_seconds": [1.0],
        },
    )
    args = type("Args", (), {
        "execute": qualification.EXECUTION_TOKEN, "work_dir": work,
        "phase": "b1", "epochs": 2, "seed": 1,
    })()
    qualification.run_matrix(args)
    assert calls == [
        "stock", "manager", "stock", "manager",
        "manager", "stock", "manager", "stock",
    ]
    pair = json.loads(
        (work / "records/epoch-002/qwen2.5-7b-b1-pair.json").read_text()
    )
    assert pair["execution_order"] == ["manager", "stock"]


def test_qwen35_phase_is_isolated_and_manager_command_binds_state_plan(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    work.mkdir()
    preflight = {
        "python": {"executable": sys.executable},
        "benchmark": {"path": "/benchmark.py"},
        "sglang": {"stock_root": "/stock", "manager_root": "/manager"},
        "library": {"path": "/library.so"},
        "inputs": {
            "models": {"qwen3.5-0.8b": {"root": "/qwen35"}},
            "plans": {"qwen3.5-0.8b": {"path": "/token-plan.json"}},
            "state_plans": {"qwen3.5-0.8b": {"path": "/state-plan.json"}},
        },
    }
    preflight["qualification_scope"] = "qwen35"
    (work / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    args = type("Args", (), {
        "execute": qualification.EXECUTION_TOKEN, "work_dir": work,
        "phase": "qwen35", "epochs": 1, "seed": 7,
    })()
    stock_command = qualification._case_command(
        args, preflight, qualification.QWEN35_CASES[0], "stock"
    )
    manager_command = qualification._case_command(
        args, preflight, qualification.QWEN35_CASES[0], "manager"
    )
    assert "--state-plan" not in stock_command
    assert manager_command[manager_command.index("--state-plan") + 1] == "/state-plan.json"
    for command in (stock_command, manager_command):
        assert command[command.index("--requests") + 1] == "1"
        assert command[command.index("--max-running-requests") + 1] == "2"

    monkeypatch.setattr(qualification, "_validate_preflight", lambda _pre: None)
    monkeypatch.setattr(qualification, "_assert_idle_h20", lambda: None)
    monkeypatch.setattr(
        qualification, "_verify_case_records", lambda *_args: None
    )
    calls = []

    def fake_run(command, output, stderr):
        calls.append((command[command.index("--mode") + 1], output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")

    monkeypatch.setattr(qualification, "_run_record", fake_run)
    monkeypatch.setattr(
        qualification, "verify_pair_files",
        lambda _stock, _manager, _pre=None: {
            "schema": qualification.PAIR_SCHEMA, "status": "passed",
            "profile": "hybrid_full_gdn", "workload_profile": "fresh_prompt",
            "qualification_claim": "pair_verification_only_not_qualified",
            "batch_size": int(_stock.name.rsplit("-b", 1)[1].split("-", 1)[0]),
            "iterations": 1, "stock_iteration_seconds": [1.0],
            "manager_iteration_seconds": [1.0],
        },
    )
    summary = qualification.run_matrix(args)
    assert summary["qualified"] is False
    assert summary["preflight_bound"] is True
    assert summary["hardware_attested"] is False
    assert len(calls) == 4
    assert all("qwen35-records" in str(path) for _, path in calls)
    assert not (work / "records").exists()


def test_qwen35_run_validates_case_records_before_pairing(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "preflight.json").write_text(
        json.dumps({"qualification_scope": "qwen35"}), encoding="utf-8"
    )
    args = type("Args", (), {
        "execute": qualification.EXECUTION_TOKEN, "work_dir": work,
        "phase": "qwen35", "epochs": 1, "seed": 7,
    })()
    monkeypatch.setattr(qualification, "_validate_preflight", lambda _pre: None)
    monkeypatch.setattr(qualification, "_assert_idle_h20", lambda: None)
    monkeypatch.setattr(
        qualification, "_case_command",
        lambda _args, _pre, case, mode: [mode, case.slug],
    )

    def fake_run(_command, output, stderr):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")

    monkeypatch.setattr(qualification, "_run_record", fake_run)
    monkeypatch.setattr(
        qualification, "verify_pair_files",
        lambda *_args, **_kwargs: pytest.fail("pairing ran before case validation"),
    )
    with pytest.raises(RuntimeError, match="declared qwen3.5-0.8b-b1 workload"):
        qualification.run_matrix(args)


def test_seal_copies_complete_matrix_and_hashes_without_overwrite(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    library = tmp_path / "liborbitkv_ffi.so"
    _build_abi8_stub(library)
    requirements = tmp_path / "requirements.lock"
    pinned_requirements = (
        qualification.REPOSITORY_ROOT
        / ".qualification/requirements-v0.5.17.lock.txt"
    )
    if not pinned_requirements.is_file():
        pytest.skip("ABI8 sealed lock fixture is unavailable")
    shutil.copy2(pinned_requirements, requirements)
    locked_editable = next(
        line for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.startswith("-e git+") and "#egg=orbitkv_sglang" in line
    )
    source_commit = "a" * 40
    active_editable = qualification.ORBITKV_EDITABLE_TEMPLATE.format(
        commit=source_commit
    )
    plans = {}
    plan_sources = {
        "qwen2.5-7b": "qwen2.5-7b-full-page16-bf16.json",
        "gpt-oss-20b": "gpt-oss-20b-hybrid-page16-bf16.json",
    }
    for name, filename in plan_sources.items():
        source_plan = qualification.REPOSITORY_ROOT / ".qualification/plans" / filename
        if not source_plan.is_file():
            pytest.skip("ABI8 sealed plan fixtures are unavailable")
        plan = tmp_path / f"{name}.json"
        shutil.copy2(source_plan, plan)
        plans[name] = {"path": str(plan), "sha256": qualification.sha256_file(plan)}
    preflight = {
        "schema": qualification.PRECHECK_SCHEMA,
        "source": {
            "commit": source_commit, "clean": True,
        },
        "benchmark": {
            "path": str(Path(qualification.benchmark.__file__).resolve()),
            "sha256": qualification.sha256_file(
                Path(qualification.benchmark.__file__).resolve()
            ),
            "record_schema": qualification.benchmark.RECORD_SCHEMA,
        },
        "library": {
            "path": str(library),
            "sha256": qualification.sha256_file(library),
            "bytes": library.stat().st_size,
            "abi_version": 8,
            "symbols": sorted(qualification.EXACT_SYMBOL_ALLOWLIST),
        },
        "inputs": {
            "plans": plans,
            "requirements": {"path": str(requirements), "sha256": qualification.sha256_file(requirements)},
            "models": {
                name: {
                        "config.json": {
                            "sha256": qualification.MODEL_HASHES[name]["config.json"]
                        },
                        "model.safetensors.index.json": {
                            "sha256": qualification.MODEL_HASHES[name]["model.safetensors.index.json"]
                        },
                    "weight_shards": [{"filename": "model-1.safetensors", "size": 7, "sha256": "shard"}],
                    "weight_shards_sha256": "shards",
                }
                for name in ("qwen2.5-7b", "gpt-oss-20b")
            },
        },
        "python": {
            "executable": sys.executable,
            "real_executable": str(Path(sys.executable).resolve()),
            "normalized_freeze_sha256": "freeze",
            "active_editable": active_editable,
            "locked_editable": locked_editable,
        },
        "sglang": {
            "release": "v0.5.17",
            "revision": qualification.SGLANG_REVISION,
            "stock_root": "/stock",
            "manager_root": "/manager",
            "pinned_contract": qualification.benchmark.verify_pinned_module_constants(),
            "stock": {"loader": {"mode": "stock"}},
            "manager": {"loader": {"mode": "manager"}},
        },
    }
    source_files = [
        Path(qualification.__file__).resolve(),
        Path(qualification.benchmark.__file__).resolve(),
        qualification.INTEGRATION_ROOT / "checkpoint_identity.py",
        qualification.INTEGRATION_ROOT / "prepare_pinned_checkout.py",
        qualification.INTEGRATION_ROOT / "pyproject.toml",
        qualification.INTEGRATION_ROOT / "patches/v0.5.17-orbitkv-fail-closed.patch",
        *sorted((qualification.SOURCE_ROOT / "orbitkv_sglang").rglob("*.py")),
    ]
    source_inventory = [
        {
            "path": path.relative_to(qualification.REPOSITORY_ROOT).as_posix(),
            "sha256": qualification.sha256_file(path),
        }
        for path in source_files
    ]
    preflight["source"].update(
        tracked_file_count=len(source_inventory),
        inventory=source_inventory,
        inventory_sha256=qualification.canonical_digest(source_inventory),
    )
    (work / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(qualification, "_validate_preflight", lambda _pre: None)
    pairs = []
    epoch_dir = work / "records/epoch-001"
    epoch_dir.mkdir(parents=True)
    for case in qualification.CASES:
        stock = _record("stock", case.profile)
        manager = _record("manager", case.profile)
        for mode, record in (("stock", stock), ("manager", manager)):
            record["source_identity"].update(
                root=preflight["sglang"][f"{mode}_root"],
                pinned_contract=preflight["sglang"]["pinned_contract"],
                loader=preflight["sglang"][mode]["loader"],
            )
            record["runtime_identity"] = {"python_executable": sys.executable}
        manager["source_identity"]["library"] = preflight["library"]
        manager["manager"]["library"] = manager["source_identity"]["library"]
        manager["source_identity"]["plan"] = plans[case.model]
        manager["manager"]["plan"]["artifact"] = manager["source_identity"]["plan"]
        manager["checkpoint"].update(
            config_sha256=qualification.MODEL_HASHES[case.model]["config.json"],
            index_files=[{
                "name": "model.safetensors.index.json",
                "sha256": qualification.MODEL_HASHES[case.model]["model.safetensors.index.json"],
            }],
            indexed_weight_files=["model-1.safetensors"],
            weight_files=[{
                "name": "model-1.safetensors",
                "bytes": 7,
                "sha256": "shard",
            }],
        )
        stock["checkpoint"] = copy.deepcopy(manager["checkpoint"])
        for record in (stock, manager):
            record["checkpoint_identity_sha256"] = qualification.canonical_digest(
                record["checkpoint"]
            )
        stock["workload"].update(requests=case.batch, iterations=case.iterations)
        manager["workload"].update(requests=case.batch, iterations=case.iterations)
        for record in (stock, manager):
            record["workload"].update(
                max_running_requests=case.batch, prompt_tokens=513, decode_tokens=33
            )
            record["engine_args"].update(
                attention_backend=case.backend,
                chunked_prefill_size=case.chunk_tokens,
                max_running_requests=case.batch,
                max_total_tokens=case.capacity_tokens,
            )
            record["checkpoint_contract"]["attention_backend"] = case.backend
            record["capacity_readback"].update(
                full_tokens=case.capacity_tokens,
                requested_max_total_tokens=case.capacity_tokens,
            )
            record["runtime_identity"] = {
                "python_executable": sys.executable,
                "attention_backend": case.backend,
                "kv_layout": "nhd",
                "execution": "eager",
            }
            record["source_identity"]["checkpoint_identity_helper_sha256"] = (
                qualification.sha256_file(
                    qualification.INTEGRATION_ROOT / "checkpoint_identity.py"
                )
            )
            record["pairing"]["contract"] = qualification._record_pair_contract(record)
            record["pairing"]["pair_key_sha256"] = qualification.canonical_digest(record["pairing"]["contract"])
            _refresh_record_derivations(record)
        stock_path = epoch_dir / f"{case.slug}-stock.json"
        manager_path = epoch_dir / f"{case.slug}-manager.json"
        stock_path.write_text(json.dumps(stock), encoding="utf-8")
        manager_path.write_text(json.dumps(manager), encoding="utf-8")
        stock_path.with_suffix(".stderr.log").write_text("", encoding="utf-8")
        manager_path.with_suffix(".stderr.log").write_text("", encoding="utf-8")
        pair = qualification.verify_pair_files(stock_path, manager_path, preflight)
        assert pair["stock_record"] == stock_path.name
        assert pair["manager_record"] == manager_path.name
        pair.update(epoch=1, execution_order=["stock", "manager"])
        (epoch_dir / f"{case.slug}-pair.json").write_text(json.dumps(pair), encoding="utf-8")
        pairs.append(pair)
    summary = qualification.summarize_pairs(pairs)
    (work / "summary-all-1-epochs.json").write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "sealed"
    manifest = qualification.seal(type("Args", (), {"work_dir": work, "output_dir": output})())
    assert manifest["qualification_status"] == (
        "abi8_sglang_full_full_swa_prefix_correctness_qualified_performance_pending"
    )
    assert {item["profile"] for item in manifest["scope"]["cases"]} == {
        "full", "hybrid_full_swa"
    }
    assert {item["attention_backend"] for item in manifest["scope"]["cases"]} == {
        "fa3"
    }
    assert "token_relocation" in manifest["scope"]["excluded"]
    assert manifest["pair_count"] == 4
    assert (output / "qualification/build/liborbitkv_ffi.so").read_bytes() == library.read_bytes()
    sealed_lock = output / "qualification/requirements.lock.txt"
    assert sealed_lock.read_text(encoding="utf-8") == (
        qualification._materialize_requirements_lock(requirements, active_editable)
    )
    assert manifest["input_hashes"]["requirements_sha256"] == (
        qualification.sha256_file(sealed_lock)
    )
    assert manifest["input_hashes"]["original_requirements_sha256"] == (
        qualification.sha256_file(requirements)
    )
    assert (output / "qualification/source/checkpoint_identity.py").is_file()
    assert (output / "qualification/source/src/orbitkv_sglang/runtime/records.py").is_file()
    assert not list(output.rglob("__pycache__"))
    assert not list(output.rglob("*.pyc"))
    assert (output / "README.md").is_file()
    assert (output / "SHA256SUMS").is_file()
    readme = (output / "README.md").read_text(encoding="utf-8")
    trusted_command = (
        "PYTHONDONTWRITEBYTECODE=1 python3 tools/verify_manifests.py "
        "/path/to/seal/manifest.json"
    )
    bundled_command = (
        "PYTHONDONTWRITEBYTECODE=1 python3 "
        "qualification/source/qualify_abi8_h20.py verify-seal ."
    )
    assert trusted_command in readme
    assert bundled_command in readme
    assert readme.index(trusted_command) < readme.index(bundled_command)
    assert "Do not use the bundled command as the initial authenticity check." in readme
    verified = qualification.verify_seal(output)
    assert verified["status"] == "passed"
    assert verified["pair_count"] == 4
    with pytest.raises(RuntimeError, match="overwrite"):
        qualification.seal(type("Args", (), {"work_dir": work, "output_dir": output})())

    relocated = tmp_path / "relocated/abi8-seal"
    relocated.parent.mkdir()
    output.rename(relocated)
    shutil.rmtree(work)
    library.unlink()
    library.with_suffix(".c").unlink()
    requirements.unlink()
    for identity in plans.values():
        Path(identity["path"]).unlink()
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    }
    completed = subprocess.run(
        [
            sys.executable, "-S",
            str(relocated / "qualification/source/qualify_abi8_h20.py"),
            "verify-seal", str(relocated),
        ],
        cwd=empty_cwd, env=environment, check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["status"] == "passed"
    assert not list(relocated.rglob("__pycache__"))

    (relocated / "unlisted.txt").write_text("not sealed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted"):
        qualification.verify_seal(relocated)
