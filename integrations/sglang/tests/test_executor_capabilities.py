from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang.config import load_config
from orbitkv_sglang.executor_capabilities import (
    admit_execution_signature,
    admit_runtime_config,
    execution_signature_from_manifest,
    load_executor_capabilities,
    runtime_target_binding_from_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _compile_manifest(tmp_path: Path, state_plan: dict[str, Any]) -> dict[str, Any]:
    plan = tmp_path / "state-plan.json"
    plan.write_text(json.dumps(state_plan), encoding="utf-8")
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "orbitkv",
            "--",
            "compile-runtime-manifest",
            str(plan),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _write_config_inputs(
    tmp_path: Path, manifest: dict[str, Any]
) -> dict[str, str]:
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()
    return {
        "ORBITKV_RUNTIME_MANIFEST": str(manifest_path),
        "ORBITKV_LIBRARY": str(library),
    }


def _sliding_plan() -> dict[str, Any]:
    return {
        "page_tokens": 16,
        "states": [
            {
                "name": "sliding",
                "layers": [0, 1],
                "storage": {
                    "kind": "token_kv",
                    "key_bytes_per_token_per_layer": 64,
                    "value_bytes_per_token_per_layer": 64,
                    "retention": "sliding",
                    "window_tokens": 18,
                },
            }
        ],
    }


def test_packaged_contract_is_strict_and_import_leaf() -> None:
    contract = load_executor_capabilities()
    assert contract["schema"] == "orbitkv.runtime-target-contract"
    assert contract["target"] == {"id": "sglang.abi8", "contract_version": 1}
    assert contract["fingerprint"] == "sha256:" + hashlib.sha256(
        _canonical_json(
            {key: value for key, value in contract.items() if key != "fingerprint"}
        )
    ).hexdigest()
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                "from orbitkv_sglang.executor_capabilities import "
                "load_executor_capabilities; load_executor_capabilities(); "
                "assert not any(n == 'sglang' or n.startswith('sglang.') "
                "for n in sys.modules); "
                "assert not any(n == 'torch' or n.startswith('torch.') "
                "for n in sys.modules); "
                "assert 'orbitkv_sglang.plugin.hooks' not in sys.modules"
            ),
        ],
        cwd=Path("/tmp"),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 and "No module named 'orbitkv_sglang'" in completed.stderr:
        pytest.skip("isolated import requires an installed wheel")
    assert completed.returncode == 0, completed.stderr

    mutated = load_executor_capabilities()
    mutated["target"]["id"] = "poisoned"
    assert load_executor_capabilities()["target"]["id"] == "sglang.abi8"


def test_python_signature_matches_rust_binding(tmp_path: Path) -> None:
    manifest = _compile_manifest(tmp_path, _sliding_plan())
    signature = execution_signature_from_manifest(manifest)
    assert admit_execution_signature(signature) == "whole_domain_sliding_token_kv"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract_path = (
        REPOSITORY_ROOT
        / "integrations/sglang/src/orbitkv_sglang/resources/executor_capabilities.v1.json"
    )
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "orbitkv",
            "--",
            "bind-runtime-manifest",
            str(manifest_path),
            "--executor-capabilities",
            str(contract_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    binding = json.loads(completed.stdout)
    assert binding == runtime_target_binding_from_manifest(manifest)
    assert binding["execution_signature"] == signature
    assert binding["target_contract_fingerprint"] == load_executor_capabilities()[
        "fingerprint"
    ]


def test_manifest_config_carries_admitted_signature_but_legacy_does_not(
    tmp_path: Path,
) -> None:
    manifest = _compile_manifest(tmp_path, _sliding_plan())
    config = load_config(_write_config_inputs(tmp_path, manifest))
    assert admit_runtime_config(config) == "whole_domain_sliding_token_kv"
    assert config.execution_signature == execution_signature_from_manifest(manifest)

    plan_path = tmp_path / "legacy-plan.json"
    legacy_plan = copy.deepcopy(manifest["token_manager_plan"]["input"])
    legacy_plan["classes"][0]["name"] = "swa"
    plan_path.write_text(
        json.dumps(legacy_plan), encoding="utf-8"
    )
    legacy = load_config(
        {
            "ORBITKV_PLAN": str(plan_path),
            "ORBITKV_LIBRARY": str(tmp_path / "liborbitkv_ffi.so"),
        }
    )
    with pytest.raises(ValueError, match="legacy plans"):
        admit_runtime_config(legacy)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(schema="unknown"), "unsupported"),
        (
            lambda value: value["supported_topologies"].append(
                value["supported_topologies"][0]
            ),
            "sorted, unique",
        ),
        (lambda value: value.update(unknown=True), "fields differ"),
    ],
)
def test_contract_mutations_fail_closed(mutation, message) -> None:
    contract = copy.deepcopy(load_executor_capabilities())
    mutation(contract)
    with pytest.raises(ValueError, match=message):
        admit_execution_signature(
            {
                "schema": "orbitkv.execution-signature",
                "version": 1,
                "fingerprint": "sha256:" + "0" * 64,
            },
            contract,
        )


@pytest.mark.parametrize("path", ("target", "admission_profile"))
def test_nested_contract_versions_are_u32(path: str) -> None:
    contract = copy.deepcopy(load_executor_capabilities())
    field = "contract_version" if path == "target" else "version"
    contract[path][field] = 1 << 32
    contract["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(
            {key: item for key, item in contract.items() if key != "fingerprint"}
        )
    ).hexdigest()
    with pytest.raises(ValueError, match="supported integer range"):
        admit_execution_signature(
            {
                "schema": "orbitkv.execution-signature",
                "version": 1,
                "fingerprint": "sha256:" + "0" * 64,
            },
            contract,
        )


@pytest.mark.parametrize("field", ("version", "manifest_version"))
@pytest.mark.parametrize("value", (True, 1.0))
def test_signature_versions_require_exact_json_integers(
    tmp_path: Path, field: str, value: Any
) -> None:
    signature = execution_signature_from_manifest(
        _compile_manifest(tmp_path, _sliding_plan())
    )
    signature[field] = value
    signature["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(
            {key: item for key, item in signature.items() if key != "fingerprint"}
        )
    ).hexdigest()
    with pytest.raises(ValueError, match="must be 1"):
        admit_execution_signature(signature)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda value: value["token_states"][0]["backend"].update(
                page_bytes_per_layer=1
            ),
            "token geometry differs",
        ),
        (
            lambda value: value["token_states"][0]["backend"]["components"][0].update(
                name="value"
            ),
            "component order differs",
        ),
        (
            lambda value: value.update(manifest_schema="forged"),
            "manifest_schema is unsupported",
        ),
    ],
)
def test_signature_geometry_and_envelope_tampering_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    signature = execution_signature_from_manifest(
        _compile_manifest(tmp_path, _sliding_plan())
    )
    mutation(signature)
    signature["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(
            {key: item for key, item in signature.items() if key != "fingerprint"}
        )
    ).hexdigest()
    with pytest.raises(ValueError, match=message):
        admit_execution_signature(signature)


def test_installed_distribution_records_target_resource_when_available() -> None:
    try:
        files = importlib.metadata.distribution("orbitkv-sglang").files
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("source-tree test; wheel metadata is covered by CI")
    assert files is not None
    distribution_root = Path(
        importlib.metadata.distribution("orbitkv-sglang").locate_file("")
    ).resolve()
    source_root = (REPOSITORY_ROOT / "integrations/sglang/src").resolve()
    if distribution_root == source_root:
        pytest.skip("editable source metadata predates the package-data change")
    assert any(
        str(path).endswith(
            "orbitkv_sglang/resources/executor_capabilities.v1.json"
        )
        for path in files
    )


def test_target_mismatch_rejects_before_configurator_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import orbitkv_sglang.plugin.state as plugin_state
    import orbitkv_sglang.plugin.validation as validation

    manifest = _compile_manifest(tmp_path, _sliding_plan())
    config = load_config(_write_config_inputs(tmp_path, manifest))
    signature = copy.deepcopy(config.execution_signature)
    assert signature is not None
    signature["token_classes"][0]["address"] = {"kind": "pinned"}
    signature["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(
            {key: value for key, value in signature.items() if key != "fingerprint"}
        )
    ).hexdigest()
    binding = runtime_target_binding_from_manifest(manifest)
    binding = copy.deepcopy(binding)
    binding["execution_signature"] = signature
    binding["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(
            {key: value for key, value in binding.items() if key != "fingerprint"}
        )
    ).hexdigest()
    config = type(config)(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field not in {"execution_signature", "runtime_target_binding"}
        },
        execution_signature=signature,
        runtime_target_binding=binding,
    )
    plugin_state._install_test_state(config=config)
    original_called = False

    def original_fn(*_args, **_kwargs):
        nonlocal original_called
        original_called = True
        raise AssertionError("configurator allocation must not run")

    with pytest.raises(RuntimeError, match="executor target admission failed"):
        validation._validate_configurator(original_fn, object())
    assert not original_called
