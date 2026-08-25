from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


sys.dont_write_bytecode = True
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/verify_qwen35_h20_pair_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen35_h20_pair_evidence", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    shutil.copytree(verifier.DEFAULT_ARCHIVE, root)
    return root


def _reseal(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = _load(manifest_path)
    manifest["artifacts"] = {
        name: verifier.sha256_file(root / name)
        for name in manifest["artifacts"]
    }
    _write(manifest_path, manifest)
    sums = dict(manifest["artifacts"])
    sums["manifest.json"] = verifier.sha256_file(manifest_path)
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(sums.items())
        ),
        encoding="utf-8",
    )


def test_real_archive_passes_complete_offline_verification() -> None:
    result = verifier.verify_archive()
    assert result == {
        "schema": verifier.VERIFICATION_SCHEMA,
        "status": "passed",
        "archive": verifier.ARCHIVE_NAME,
        "source_commit": verifier.SOURCE_COMMIT,
        "epoch_count": 3,
        "pair_count": 6,
        "abi_version": 8,
        "exact_symbol_count": 40,
        "gpu": {"name": "NVIDIA H20", "uuid": verifier.H20_UUID},
        "qualification_claim": verifier.QUALIFICATION_CLAIM,
        "qualified": False,
        "hardware_attested": False,
        "performance_go": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "orbitkv.abi8-h20-qwen35-pair-evidence.v2"),
        ("qualification_status", "qualified"),
        ("qualification_claim", "qualified"),
        ("qualified", True),
        ("hardware_attested", True),
        ("performance_go", True),
        ("abi_version", 7),
        ("exact_symbol_count", 39),
        ("record_schema", "orbitkv.sglang-v0517-abi8-single-run.v1"),
        ("epoch_count", 2),
        ("pair_count", 5),
        ("source_commit", verifier.SOURCE_BASE_COMMIT),
    ),
)
def test_manifest_rejects_policy_or_schema_drift(field: str, value: object) -> None:
    manifest = _load(verifier.DEFAULT_ARCHIVE / "manifest.json")
    manifest[field] = value
    with pytest.raises(RuntimeError):
        verifier._validate_manifest(manifest)


def test_manifest_requires_exact_keys_scope_hardware_and_inputs() -> None:
    original = _load(verifier.DEFAULT_ARCHIVE / "manifest.json")
    mutations = []

    missing = copy.deepcopy(original)
    del missing["scope"]
    mutations.append(missing)

    extra = copy.deepcopy(original)
    extra["unreviewed"] = True
    mutations.append(extra)

    scope = copy.deepcopy(original)
    scope["scope"]["execution"]["gpu_count"] = 2
    mutations.append(scope)

    hardware = copy.deepcopy(original)
    hardware["observed_hardware"]["snapshot_count"] = 59
    mutations.append(hardware)

    inputs = copy.deepcopy(original)
    inputs["input_hashes"]["plans"]["state_input"] = "0" * 64
    mutations.append(inputs)

    for manifest in mutations:
        with pytest.raises(RuntimeError):
            verifier._validate_manifest(manifest)


@pytest.mark.parametrize(
    "value",
    (
        "", "/absolute", "../escape", "a/../b", "a/./b",
        "a//b", "a\\b", "a\x00b", "a\nb",
    ),
)
def test_archive_paths_must_be_canonical_and_relative(value: str) -> None:
    with pytest.raises(RuntimeError):
        verifier._safe_relative(value)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate key"):
        verifier._strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-finite"):
        verifier._strict_json(nonfinite)


def test_archive_rejects_unlisted_file(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    (root / "unlisted.txt").write_text("not sealed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verifier.verify_archive(root)


def test_archive_rejects_artifact_tampering(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    (root / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        verifier.verify_archive(root)


def test_archive_rejects_file_and_root_symlinks(tmp_path: Path) -> None:
    linked_root = tmp_path / "linked-archive"
    linked_root.symlink_to(verifier.DEFAULT_ARCHIVE, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        verifier.verify_archive(linked_root)

    root = _copy_archive(tmp_path)
    readme = root / "README.md"
    readme.unlink()
    readme.symlink_to("manifest.json")
    with pytest.raises(RuntimeError, match="file symlink"):
        verifier.verify_archive(root)


def test_archive_rejects_noncanonical_sha256sums(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    sums_path = root / "SHA256SUMS"
    lines = sums_path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    sums_path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical sorted form"):
        verifier.verify_archive(root)


def test_manifest_artifacts_reject_path_traversal() -> None:
    root = verifier.DEFAULT_ARCHIVE
    manifest = _load(root / "manifest.json")
    digest = manifest["artifacts"].pop("README.md")
    manifest["artifacts"]["../README.md"] = digest
    with pytest.raises(RuntimeError, match="unsafe or non-canonical"):
        verifier._verify_archive_inventory(root, root / "manifest.json", manifest)


def test_preflight_source_inventory_rejects_count_digest_and_path_drift() -> None:
    manifest = _load(verifier.DEFAULT_ARCHIVE / "manifest.json")
    original = _load(verifier.DEFAULT_ARCHIVE / "preflight.json")
    mutations = []

    count = copy.deepcopy(original)
    count["source"]["tracked_file_count"] -= 1
    mutations.append(count)

    digest = copy.deepcopy(original)
    digest["source"]["inventory_sha256"] = "0" * 64
    mutations.append(digest)

    path = copy.deepcopy(original)
    path["source"]["inventory"][0]["path"] = "../escape"
    mutations.append(path)

    duplicate = copy.deepcopy(original)
    duplicate["source"]["inventory"][1] = copy.deepcopy(
        duplicate["source"]["inventory"][0]
    )
    mutations.append(duplicate)

    for preflight in mutations:
        with pytest.raises(RuntimeError):
            verifier._validate_preflight(preflight, manifest)


def test_bundle_rejects_an_inventory_not_matching_the_committed_tree() -> None:
    root = verifier.DEFAULT_ARCHIVE
    manifest = _load(root / "manifest.json")
    preflight = _load(root / "preflight.json")
    inventory = copy.deepcopy(preflight["source"]["inventory"])
    inventory[0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source blob differs"):
        verifier._verify_source_bundle(root, inventory, manifest)


def test_bundle_rejects_manifest_hash_or_prerequisite_drift() -> None:
    root = verifier.DEFAULT_ARCHIVE
    preflight = _load(root / "preflight.json")
    inventory = preflight["source"]["inventory"]

    digest = _load(root / "manifest.json")
    digest["source_provenance"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bundle differs"):
        verifier._verify_source_bundle(root, inventory, digest)

    prerequisite = _load(root / "manifest.json")
    prerequisite["source_provenance"]["prerequisite_commit"] = (
        verifier.SOURCE_COMMIT
    )
    with pytest.raises(RuntimeError, match="prerequisite_commit"):
        verifier._validate_manifest(prerequisite)


def test_source_closure_rejects_extra_file(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    preflight = _load(root / "preflight.json")
    _, indexed = verifier._validate_preflight(
        preflight, _load(root / "manifest.json")
    )
    (root / "qualification/source/unlisted.py").write_text(
        "raise SystemExit\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="incomplete or excessive"):
        verifier._verify_source_closure(root, preflight, indexed)


def test_source_closure_rejects_content_drift(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    preflight = _load(root / "preflight.json")
    manifest = _load(root / "manifest.json")
    _, indexed = verifier._validate_preflight(preflight, manifest)
    (root / "qualification/source/checkpoint_identity.py").write_text(
        "# forged\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="differs from preflight"):
        verifier._verify_source_closure(root, preflight, indexed)


def test_library_requires_exact_preflight_identity_and_40_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = verifier.DEFAULT_ARCHIVE
    manifest = _load(root / "manifest.json")
    preflight = _load(root / "preflight.json")
    forged = copy.deepcopy(preflight["library"])
    forged["path"] = str(root / "qualification/build/liborbitkv_ffi.so")
    forged["symbols"] = forged["symbols"][:-1]
    monkeypatch.setattr(
        verifier.qualification, "sealed_library_identity", lambda _path: forged
    )
    with pytest.raises(RuntimeError, match="differs from preflight|identity"):
        verifier._verify_library(root, preflight, manifest)


def test_input_identity_rejects_model_or_plan_drift() -> None:
    root = verifier.DEFAULT_ARCHIVE
    manifest = _load(root / "manifest.json")
    original = _load(root / "preflight.json")

    model = copy.deepcopy(original)
    model["inputs"]["models"][verifier.MODEL_NAME]["config.json"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="model hashes"):
        verifier._verify_inputs(root, model, manifest)

    plan = copy.deepcopy(original)
    plan["inputs"]["state_plans"][verifier.MODEL_NAME]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="attention-state input"):
        verifier._verify_inputs(root, plan, manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(repo_id="untrusted/model"), "Hugging Face"),
        (lambda value: value.update(revision="0" * 40), "Hugging Face"),
        (lambda value: value.update(resolved_ref="0" * 40), "Hugging Face"),
        (
            lambda value: value["files"]["config.json"].update(
                git_blob="0" * 40
            ),
            "Hugging Face",
        ),
        (
            lambda value: value["files"][
                "model.safetensors.index.json"
            ].update(git_blob="0" * 40),
            "Hugging Face",
        ),
        (
            lambda value: value["files"][
                "model.safetensors-00001-of-00001.safetensors"
            ].update(git_blob="0" * 40),
            "Hugging Face",
        ),
        (
            lambda value: value["files"][
                "model.safetensors-00001-of-00001.safetensors"
            ].update(lfs_sha256="0" * 64),
            "Hugging Face",
        ),
        (
            lambda value: value["files"][
                "model.safetensors-00001-of-00001.safetensors"
            ].update(sha256="0" * 64),
            "Hugging Face",
        ),
        (
            lambda value: value["files"][
                "model.safetensors-00001-of-00001.safetensors"
            ].update(xet_hash="0" * 64),
            "Hugging Face",
        ),
    ),
)
def test_model_provenance_rejects_repo_revision_blob_and_lfs_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    root = _copy_archive(tmp_path)
    provenance_path = root / "qualification/model-provenance.json"
    provenance = _load(provenance_path)
    mutate(provenance)
    _write(provenance_path, provenance)
    manifest = _load(root / "manifest.json")
    manifest["model_provenance_sha256"] = verifier.sha256_file(provenance_path)
    with pytest.raises(RuntimeError, match=message):
        verifier._verify_model_provenance(
            root, _load(root / "preflight.json"), manifest
        )


def test_model_provenance_rejects_hash_binding_and_exact_key_drift(
    tmp_path: Path,
) -> None:
    root = _copy_archive(tmp_path)
    manifest = _load(root / "manifest.json")
    manifest["model_provenance_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest identity"):
        verifier._verify_model_provenance(
            root, _load(root / "preflight.json"), manifest
        )

    provenance_path = root / "qualification/model-provenance.json"
    provenance = _load(provenance_path)
    provenance["files"]["config.json"]["unexpected"] = True
    _write(provenance_path, provenance)
    manifest["model_provenance_sha256"] = verifier.sha256_file(provenance_path)
    with pytest.raises(RuntimeError, match="Hugging Face"):
        verifier._verify_model_provenance(
            root, _load(root / "preflight.json"), manifest
        )


@pytest.mark.parametrize("filename", ("config.json", "model.safetensors.index.json"))
def test_model_provenance_rejects_archived_metadata_content_drift(
    tmp_path: Path, filename: str
) -> None:
    root = _copy_archive(tmp_path)
    model_path = root / "qualification/model" / filename
    model_path.write_bytes(model_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match=f"archived model {filename}"):
        verifier._verify_model_provenance(
            root, _load(root / "preflight.json"),
            _load(root / "manifest.json"),
        )


def test_rejects_hash_consistent_pair_boundary_escalation(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    pair_path = (
        root / "records/epoch-001/qwen3.5-0.8b-b1-pair.json"
    )
    pair = _load(pair_path)
    pair["qualified"] = True
    _write(pair_path, pair)
    _reseal(root)
    with pytest.raises(RuntimeError, match="non-qualification boundary"):
        verifier.verify_archive(root)


def test_rejects_hash_consistent_record_semantic_drift(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    record_path = (
        root / "records/epoch-001/qwen3.5-0.8b-b1-stock.json"
    )
    record = _load(record_path)
    record["request_traces"][0][0]["output_ids"][0] += 1
    _write(record_path, record)
    _reseal(root)
    with pytest.raises(RuntimeError, match="output_ids digest|fields differ"):
        verifier.verify_archive(root)


def test_rejects_hash_consistent_fixed_state_counter_drift(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    record_path = (
        root / "records/epoch-002/qwen3.5-0.8b-b4-manager.json"
    )
    record = _load(record_path)
    record["manager"]["final_census"]["batch_counters"][
        "fixed_state_acks"
    ] -= 1
    _write(record_path, record)
    _reseal(root)
    with pytest.raises(RuntimeError, match="fixed-state counters"):
        verifier.verify_archive(root)


def test_rejects_hash_consistent_summary_drift(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    summary_path = root / "summary.json"
    summary = _load(summary_path)
    summary["groups"][0]["stock_seconds"]["mean"] += 1.0
    _write(summary_path, summary)
    _reseal(root)
    with pytest.raises(RuntimeError, match="summary differs"):
        verifier.verify_archive(root)


def test_rejects_hash_consistent_h20_snapshot_drift(tmp_path: Path) -> None:
    root = _copy_archive(tmp_path)
    record_path = (
        root / "records/epoch-003/qwen3.5-0.8b-b4-manager.json"
    )
    record = _load(record_path)
    record["gpu_snapshots"][0]["gpus"][0]["uuid"] = "GPU-forged"
    _write(record_path, record)
    _reseal(root)
    with pytest.raises(RuntimeError, match="GPU identity"):
        verifier.verify_archive(root)
