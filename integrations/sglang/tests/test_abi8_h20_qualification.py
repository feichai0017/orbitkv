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
            output_ids = [request + 1] * workload["decode_tokens"]
            request_id = f"request-{iteration}-{request}"
            row.append({
                "request_index": request,
                "submitted_rid": request_id,
                "returned_rid": request_id,
                "submitted_input_ids_sha256": "input",
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
        "command": ["python", "benchmark.py"],
        "environment": {"PYTHONHASHSEED": "0"},
        "manager": manager_record, "engine_args": engine,
        "source_identity": source,
        "checkpoint": {"config_sha256": "config"},
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
    verify = parser.parse_args(["verify-seal", "sealed"])
    assert verify.action == "verify-seal"
    assert verify.seal_dir == Path("sealed")


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
    assert "PYTHONDONTWRITEBYTECODE=1 python3 qualification/source/qualify_abi8_h20.py verify-seal ." in (
        output / "README.md"
    ).read_text(encoding="utf-8")
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
