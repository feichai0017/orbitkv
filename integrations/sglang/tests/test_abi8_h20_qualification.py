from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(INTEGRATION_ROOT / "src"))

import qualify_abi8_h20 as qualification  # noqa: E402


def _record(mode: str, profile: str = "full") -> dict:
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
    record = {
        "schema": qualification.benchmark.RECORD_SCHEMA, "mode": mode,
        "manager": manager_record, "engine_args": engine,
        "source_identity": source,
        "checkpoint_identity_sha256": "checkpoint",
        "checkpoint_contract": {"attention_profile": profile},
        "sampling_params": {"temperature": 0},
        "prefix_seed": {"prompt_tokens": 512},
        "workload": {"requests": 1, "iterations": 1},
        "capacity_readback": {"full_tokens": 1024},
        "output_token_digest_sha256": "tokens",
        "output_request_digests_sha256": [["request"]],
        "request_traces": [[{"output_ids": [1]}]],
        "completed_requests": 1, "completion_tokens": 33,
        "iteration_seconds": [1.0],
        "pairing": {},
    }
    contract = qualification._record_pair_contract(record)
    record["pairing"] = {
        "contract": contract,
        "pair_key_sha256": qualification.canonical_digest(contract),
    }
    return record


@pytest.mark.parametrize("profile", ("full", "hybrid_full_swa"))
def test_pair_verifier_accepts_only_declared_radix_backend_difference(profile):
    result = qualification.verify_pair_records(_record("stock", profile), _record("manager", profile))
    assert result["status"] == "passed"
    assert result["profile"] == profile


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
        lambda _stock, _manager: {
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


def test_seal_copies_complete_matrix_and_hashes_without_overwrite(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    library = tmp_path / "liborbitkv_ffi.so"
    library.write_bytes(b"abi8")
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("package==1\n", encoding="utf-8")
    plans = {}
    for name in ("qwen2.5-7b", "gpt-oss-20b"):
        plan = tmp_path / f"{name}.json"
        plan.write_text("{}\n", encoding="utf-8")
        plans[name] = {"path": str(plan), "sha256": qualification.sha256_file(plan)}
    preflight = {
        "source": {
            "commit": "a" * 40, "inventory_sha256": "b" * 64,
            "inventory": [{
                "path": "integrations/sglang/qualify_abi8_h20.py",
                "sha256": qualification.sha256_file(Path(qualification.__file__).resolve()),
            }],
        },
        "library": {"path": str(library), "sha256": qualification.sha256_file(library)},
        "inputs": {
            "plans": plans,
            "requirements": {"path": str(requirements), "sha256": qualification.sha256_file(requirements)},
        },
    }
    (work / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(qualification, "_validate_preflight", lambda _pre: None)
    pairs = []
    epoch_dir = work / "records/epoch-001"
    epoch_dir.mkdir(parents=True)
    for case in qualification.CASES:
        stock = _record("stock", case.profile)
        manager = _record("manager", case.profile)
        stock["workload"].update(requests=case.batch, iterations=case.iterations)
        manager["workload"].update(requests=case.batch, iterations=case.iterations)
        for record in (stock, manager):
            record["pairing"]["contract"] = qualification._record_pair_contract(record)
            record["pairing"]["pair_key_sha256"] = qualification.canonical_digest(record["pairing"]["contract"])
        stock_path = epoch_dir / f"{case.slug}-stock.json"
        manager_path = epoch_dir / f"{case.slug}-manager.json"
        stock_path.write_text(json.dumps(stock), encoding="utf-8")
        manager_path.write_text(json.dumps(manager), encoding="utf-8")
        pair = qualification.verify_pair_files(stock_path, manager_path)
        pair.update(epoch=1, execution_order=["stock", "manager"])
        (epoch_dir / f"{case.slug}-pair.json").write_text(json.dumps(pair), encoding="utf-8")
        pairs.append(pair)
    summary = qualification.summarize_pairs(pairs)
    (work / "summary-all-1-epochs.json").write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "sealed"
    manifest = qualification.seal(type("Args", (), {"work_dir": work, "output_dir": output})())
    assert manifest["qualification_status"] == "correctness_qualified_performance_pending"
    assert manifest["pair_count"] == 4
    assert (output / "qualification/build/liborbitkv_ffi.so").read_bytes() == b"abi8"
    assert (output / "README.md").is_file()
    assert (output / "SHA256SUMS").is_file()
    with pytest.raises(RuntimeError, match="overwrite"):
        qualification.seal(type("Args", (), {"work_dir": work, "output_dir": output})())
