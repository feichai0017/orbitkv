from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(INTEGRATION_ROOT / "src"))

import qualify_token_relocation_h20 as qualification  # noqa: E402


def _args(**values: object) -> object:
    return type("Args", (), values)()


def _minimal_preflight() -> dict:
    return {
        "python": {"executable": sys.executable},
        "benchmark": {"path": "/qualification/bench_token_relocation.py"},
        "sglang": {"manager_root": "/qualification/sglang-v0.5.17"},
        "inputs": {
            "model": {"root": "/models/qwen2.5-0.5b-instruct"},
            "plan": {"path": "/qualification/plan.json"},
        },
        "library": {"path": "/qualification/liborbitkv_ffi.so"},
    }


def test_execution_order_is_balanced_and_rejects_invalid_epochs() -> None:
    assert [qualification.execution_order(epoch) for epoch in range(1, 5)] == [
        ("naive", "relocate"),
        ("relocate", "naive"),
        ("naive", "relocate"),
        ("relocate", "naive"),
    ]
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            qualification.execution_order(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(("batch", "capacity"), ((1, 128), (4, 512)))
@pytest.mark.parametrize("mode", qualification.MODES)
def test_case_command_freezes_the_exact_matrix(
    tmp_path: Path, batch: int, capacity: int, mode: str
) -> None:
    case = next(item for item in qualification.CASES if item.batch == batch)
    output = tmp_path / f"record-{batch}-{mode}.json"
    command = qualification._case_command(
        _minimal_preflight(), case, mode, output, seed=20260821
    )

    def option(name: str) -> str:
        return command[command.index(name) + 1]

    assert command[:2] == [
        sys.executable, "/qualification/bench_token_relocation.py"
    ]
    assert option("--mode") == mode
    assert option("--sglang-root") == "/qualification/sglang-v0.5.17"
    assert option("--requests") == str(batch)
    assert option("--iterations") == "5"
    assert option("--max-total-tokens") == str(capacity)
    assert option("--context-length") == "128"
    assert option("--seed") == "20260821"
    assert option("--attention-backend") == "flashinfer"
    assert option("--output") == str(output)


def test_run_requires_explicit_token_and_exact_four_epochs(tmp_path: Path) -> None:
    invalid_token = _args(
        execute="wrong", epochs=4, work_dir=tmp_path, phase="all", seed=1
    )
    with pytest.raises(RuntimeError, match=qualification.EXECUTION_TOKEN):
        qualification.run_matrix(invalid_token)  # type: ignore[arg-type]

    invalid_epochs = _args(
        execute=qualification.EXECUTION_TOKEN, epochs=3, work_dir=tmp_path,
        phase="all", seed=1,
    )
    with pytest.raises(RuntimeError, match="exactly 4 epochs"):
        qualification.run_matrix(invalid_epochs)  # type: ignore[arg-type]

    split_phase = _args(
        execute=qualification.EXECUTION_TOKEN, epochs=4, work_dir=tmp_path,
        phase="b1", seed=1,
    )
    with pytest.raises(RuntimeError, match="B1 and B4 together"):
        qualification.run_matrix(split_phase)  # type: ignore[arg-type]


def test_run_revalidates_preflight_after_every_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "preflight.json").write_text("{}", encoding="utf-8")
    validations: list[dict] = []
    monkeypatch.setattr(
        qualification, "_validate_preflight", lambda record: validations.append(record)
    )
    monkeypatch.setattr(qualification, "_assert_idle_h20", lambda: None)
    monkeypatch.setattr(
        qualification, "_case_command",
        lambda _pre, _case, mode, output, **_kwargs: [
            sys.executable, "bench.py", "--mode", mode,
            "--output", str(output),
        ],
    )

    def fake_record(_command, output, stderr_path, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"mode": kwargs["mode"]}

    monkeypatch.setattr(qualification, "_run_record", fake_record)
    monkeypatch.setattr(
        qualification, "_pair_records",
        lambda *_args, epoch, case, **_kwargs: {
            "schema": qualification.PAIR_SCHEMA,
            "epoch": epoch, "batch_size": case.batch,
        },
    )
    monkeypatch.setattr(
        qualification, "_verify_work_matrix",
        lambda *_args: ([], {"schema": qualification.SUMMARY_SCHEMA}),
    )
    result = qualification.run_matrix(
        _args(
            execute=qualification.EXECUTION_TOKEN, epochs=4, work_dir=work,
            phase="all", seed=1,
        )  # type: ignore[arg-type]
    )
    assert result["schema"] == qualification.SUMMARY_SCHEMA
    # Initial validation plus one before and one after every one of 16 runs.
    assert len(validations) == 33


def test_component_requires_explicit_token(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=qualification.EXECUTION_TOKEN):
        qualification.run_component(
            _args(execute="wrong", work_dir=tmp_path)  # type: ignore[arg-type]
        )


def test_preflight_refuses_existing_directory_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(
        qualification.abi8, "source_identity",
        lambda: pytest.fail("source inspection ran after overwrite guard"),
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        qualification.preflight(_args(work_dir=work))  # type: ignore[arg-type]


def test_preflight_requires_an_ignored_in_repository_work_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = qualification.REPOSITORY_ROOT / "qualification-output-not-ignored"
    assert not candidate.exists()
    monkeypatch.setattr(
        qualification.abi8, "source_identity",
        lambda: pytest.fail("source inspection ran after work-dir safety gate"),
    )
    with pytest.raises(RuntimeError, match="must be Git-ignored"):
        qualification.preflight(
            _args(work_dir=candidate)  # type: ignore[arg-type]
        )


def test_seal_output_must_not_dirty_the_qualified_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "preflight.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(qualification, "_validate_preflight", lambda _record: None)
    with pytest.raises(RuntimeError, match="outside the clean qualification"):
        qualification.seal(
            _args(
                work_dir=work,
                output_dir=qualification.REPOSITORY_ROOT / "sealed-output",
            )  # type: ignore[arg-type]
        )


def test_preflight_fails_closed_on_dirty_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        qualification.abi8, "source_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("clean Git worktree")),
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        qualification.preflight(
            _args(work_dir=tmp_path / "new")  # type: ignore[arg-type]
        )


def test_editable_requirement_template_binds_the_clean_commit() -> None:
    commit = "a" * 40
    expected = (
        "-e git+https://github.com/feichai0017/orbitkv.git@" + commit
        + "#egg=orbitkv_sglang&subdirectory=integrations/sglang"
    )
    assert qualification._require_active_editable(
        {"active_editable": expected}, commit
    ) == expected
    with pytest.raises(RuntimeError, match="clean source commit"):
        qualification._require_active_editable(
            {"active_editable": expected.replace(commit, "b" * 40)}, commit
        )


def test_source_closure_is_exactly_tracked_and_digest_bound() -> None:
    paths = set(qualification.SOURCE_REQUIRED_PATHS) | {
        "integrations/sglang/src/orbitkv_sglang/runtime/example.py"
    }
    inventory = [
        {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}
        for path in sorted(paths)
    ]
    closure = qualification._source_closure({"inventory": inventory})
    assert {item["path"] for item in closure} == paths
    assert closure == sorted(closure, key=lambda item: item["path"])

    missing = [
        item for item in inventory
        if item["path"] != "integrations/sglang/bench_token_relocation.py"
    ]
    with pytest.raises(RuntimeError, match="omits qualification closure"):
        qualification._source_closure({"inventory": missing})


def test_source_closure_must_exist_in_the_preflight_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"committed source"
    item = {
        "path": "integrations/sglang/example.py",
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    monkeypatch.setattr(
        qualification.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, expected, b""
        ),
    )
    qualification._verify_source_commit_closure(
        {"commit": "a" * 40}, [item]
    )
    item["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="not committed"):
        qualification._verify_source_commit_closure(
            {"commit": "a" * 40}, [item]
        )


def test_require_hash_rejects_symlink_and_wrong_digest(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"identity")
    link = tmp_path / "link"
    link.symlink_to(source)
    digest = hashlib.sha256(b"identity").hexdigest()
    with pytest.raises(RuntimeError, match="not a regular file"):
        qualification._require_hash(link, digest, "input")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        qualification._require_hash(source, "0" * 64, "input")
    assert qualification._require_hash(source, digest, "input") == {
        "path": str(source), "bytes": 8, "sha256": digest
    }


def _junit(path: Path, **overrides: str) -> None:
    properties = {
        "orbitkv.cuda.available": "true",
        "orbitkv.cuda.device_name": "NVIDIA H20",
        "orbitkv.cuda.device_uuid": "GPU-host-test",
        "orbitkv.cuda.runtime_version": "13.0",
        "orbitkv.torch.version": "2.11.0+cu130",
        **overrides,
    }
    root = ET.Element(
        "testsuite", tests=str(len(qualification.COMPONENT_CASES)),
        errors="0", failures="0", skipped="0",
    )
    props = ET.SubElement(root, "properties")
    for name, value in properties.items():
        ET.SubElement(props, "property", name=name, value=value)
    for case in qualification.COMPONENT_CASES:
        ET.SubElement(root, "testcase", name=case)
    ET.ElementTree(root).write(path, encoding="unicode")


def test_component_junit_is_exact_and_h20_bound(tmp_path: Path) -> None:
    path = tmp_path / "component.xml"
    _junit(path)
    properties = qualification._component_properties(path)
    assert properties["orbitkv.cuda.device_name"] == "NVIDIA H20"
    assert properties["orbitkv.cuda.device_uuid"] == "GPU-host-test"

    _junit(path, **{"orbitkv.cuda.available": "false"})
    with pytest.raises(RuntimeError, match="not recorded on one H20"):
        qualification._component_properties(path)


def test_pair_record_preserves_narrow_qualification_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    naive_path = tmp_path / "naive.json"
    relocate_path = tmp_path / "relocate.json"
    common = {
        "pairing": {"pair_key_sha256": "a" * 64},
        "request_output_ids": [[[1]]],
        "output_token_digest_sha256": "b" * 64,
        "checkpoint": {"model": "qwen"},
        "checkpoint_contract": {"profile": "full"},
        "sampling_params": {"temperature": 0},
        "workload": {"requests": 1},
        "runtime_identity": {"sglang_version": "0.5.17"},
    }
    naive = {**common, "mode": "naive"}
    relocate = json.loads(json.dumps({**common, "mode": "relocate"}))
    naive_path.write_text(json.dumps(naive), encoding="utf-8")
    relocate_path.write_text(json.dumps(relocate), encoding="utf-8")

    started = {"naive": 1, "relocate": 2}
    monkeypatch.setattr(
        qualification, "_validate_raw_record",
        lambda record, *_args, **_kwargs: {
            "source": "source", "library": "library",
            "plan": "plan", "gpu_uuid": "GPU",
            "started_at": started[record["mode"]],
            "observed_interval_ns": (
                (0, 1) if record["mode"] == "naive" else (2, 3)
            ),
        },
    )
    pair = qualification._pair_records(
        naive_path, relocate_path, epoch=1, case=qualification.CASES[0],
        order=("naive", "relocate"), preflight_record={},
    )
    assert pair["schema"] == qualification.PAIR_SCHEMA
    assert pair["qualification_claim"] == qualification.QUALIFICATION_CLAIM
    assert pair["qualified"] is True
    assert pair["hardware_attested"] is False
    assert pair["performance_go"] is False

    relocate["request_output_ids"] = [[[2]]]
    relocate_path.write_text(json.dumps(relocate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="request_output_ids"):
        qualification._pair_records(
            naive_path, relocate_path, epoch=1, case=qualification.CASES[0],
            order=("naive", "relocate"), preflight_record={},
        )


def test_sealed_summary_never_opens_hardware_or_performance_gate() -> None:
    original = {
        "schema": qualification.evidence.SUMMARY_SCHEMA,
        "hardware_attested": True, "performance_go": True,
        "qualified": False, "identity": {"value": 1},
    }
    result = qualification._sealed_summary(original)
    assert result["schema"] == qualification.SUMMARY_SCHEMA
    assert result["qualification_claim"] == qualification.QUALIFICATION_CLAIM
    assert result["sealed"] is True
    assert result["source_clean"] is True
    assert result["source_dirty"] is False
    assert result["preflight_bound"] is True
    assert result["qualified"] is True
    assert result["hardware_attested"] is False
    assert result["performance_go"] is False
    assert result["identity"] == {"value": 1}
    assert original["hardware_attested"] is True


def test_source_bundle_is_exactly_preflight_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(arguments, **_kwargs):
        values = tuple(arguments)
        calls.append(values)
        if values[1:3] == ("rev-parse", "HEAD"):
            output = "c" * 40 + "\n"
        elif values[1:3] == ("bundle", "list-heads"):
            output = "c" * 40 + " HEAD\n"
        else:
            output = ""
        return subprocess.CompletedProcess(values, 0, output, "")

    monkeypatch.setattr(qualification, "_run", fake_run)
    qualification._create_source_bundle(tmp_path / "source.bundle", "c" * 40)
    assert calls[1][1:3] == ("bundle", "create")
    assert calls[1][-1] == "HEAD"


def test_verify_seal_requires_exact_conservative_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verify_token_relocation_h20_seal as seal_verifier

    result = {
        "schema": qualification.SEAL_VERIFICATION_SCHEMA,
        "status": "passed",
        "qualification_status": qualification.QUALIFICATION_STATUS,
        "qualification_claim": qualification.QUALIFICATION_CLAIM,
        "sealed": True, "source_clean": True,
        "preflight_bound": True, "qualified": True,
        "hardware_attested": False, "performance_go": False,
        "epoch_count": 4, "record_count": 16, "pair_count": 8,
        "abi_version": 8, "exact_symbol_count": 40,
        "all_pairs_passed": True, "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }
    monkeypatch.setattr(
        seal_verifier, "verify_sealed_archive",
        lambda _path: result, raising=False,
    )
    assert qualification.verify_seal(tmp_path) == result
    forged = dict(result, performance_go=True)
    monkeypatch.setattr(
        seal_verifier, "verify_sealed_archive",
        lambda _path: forged, raising=False,
    )
    with pytest.raises(RuntimeError, match="performance_go"):
        qualification.verify_seal(tmp_path)


# xml.etree is imported late so the qualification module remains the subject
# of import-order checks above.
import xml.etree.ElementTree as ET  # noqa: E402
