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
from orbitkv_sglang.ffi import WIRE_VERSION
from orbitkv_sglang.runtime_admission import (
    admit_execution_signature,
    admit_runtime_config,
    execution_signature_from_manifest,
    load_product_takeover_profiles,
    load_runtime_target,
    product_takeover_profile,
    runtime_binding_from_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TARGET_PATH = (
    REPOSITORY_ROOT
    / "compat/sglang/bridge/src/orbitkv_sglang/resources/runtime_target.json"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _seal(value: dict[str, Any]) -> None:
    value["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json({key: item for key, item in value.items() if key != "fingerprint"})
    ).hexdigest()


def _compile(tmp_path: Path, command: str, source: dict[str, Any]) -> dict[str, Any]:
    source_path = tmp_path / f"{command}.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    completed = subprocess.run(
        ["cargo", "run", "--quiet", "--bin", "orbitkv", "--", command, str(source_path)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _sliding_manifest(tmp_path: Path) -> dict[str, Any]:
    return _compile(
        tmp_path,
        "compile-runtime-manifest",
        {
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
        },
    )


def _chunked_manifest(tmp_path: Path) -> dict[str, Any]:
    return _compile(
        tmp_path,
        "compile-retention-runtime-manifest",
        {
            "schema": "orbitkv.retention-ir.v1",
            "page_tokens": 16,
            "states": [
                {
                    "name": "chunked",
                    "layers": [0, 1],
                    "bytes_per_token_per_layer": 128,
                    "may_read": {
                        "op": "equal",
                        "lhs": {"op": "floor_div", "value": {"op": "query_position"}, "divisor": 64},
                        "rhs": {"op": "floor_div", "value": {"op": "key_position"}, "divisor": 64},
                    },
                }
            ],
        },
    )


def _latent_manifest(tmp_path: Path) -> dict[str, Any]:
    return _compile(
        tmp_path,
        "compile-runtime-manifest",
        {
            "page_tokens": 16,
            "states": [
                {
                    "name": "latent_mla",
                    "layers": [0, 1],
                    "storage": {
                        "kind": "latent_kv",
                        "latent_bytes_per_token_per_layer": 1024,
                        "rope_bytes_per_token_per_layer": 128,
                        "retention": "full",
                        "window_tokens": None,
                    },
                }
            ],
        },
    )


def _config(tmp_path: Path, manifest: dict[str, Any]):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()
    return load_config(
        {"ORBITKV_RUNTIME_MANIFEST": str(manifest_path), "ORBITKV_LIBRARY": str(library)}
    )


def test_packaged_runtime_target_is_strict_and_import_leaf() -> None:
    target = load_runtime_target()
    assert target["schema"] == "orbitkv.runtime-target"
    assert target["target"] == {"id": "sglang", "contract_version": 4}
    assert target["supported_manifest_versions"] == [3]
    assert target["required_wire_version"] == WIRE_VERSION == 14
    assert target["fingerprint"] == "sha256:" + hashlib.sha256(
        _canonical_json({key: value for key, value in target.items() if key != "fingerprint"})
    ).hexdigest()

    mutated = load_runtime_target()
    mutated["target"]["id"] = "poisoned"
    assert load_runtime_target()["target"]["id"] == "sglang"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; from orbitkv_sglang.runtime_admission import "
                "load_runtime_target; load_runtime_target(); "
                "assert not any(n == 'sglang' or n.startswith('sglang.') for n in sys.modules); "
                "assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)"
            ),
        ],
        cwd=Path("/tmp"),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 and "No module named 'orbitkv_sglang'" in completed.stderr:
        pytest.skip("isolated import requires an installed wheel")
    assert completed.returncode == 0, completed.stderr


def test_product_takeover_profiles_come_from_engine_profile() -> None:
    profiles = load_product_takeover_profiles()

    assert tuple(profile.execution_topology for profile in profiles) == (
        "whole_domain_full_token_kv",
        "whole_domain_full_sliding_token_kv",
        "whole_domain_sliding_token_kv",
        "whole_domain_chunked_token_kv",
        "whole_domain_full_latent_kv",
    )
    assert tuple(profile.ordered_attention_classes for profile in profiles) == (
        ("full:token_kv",),
        ("full:token_kv", "sliding:token_kv"),
        ("sliding:token_kv",),
        ("chunked:token_kv",),
        ("full:latent_kv",),
    )
    assert tuple(profile.cache_policy for profile in profiles) == (
        "shared_prefix",
        "shared_prefix",
        "request_private",
        "request_private",
        "request_private",
    )
    assert tuple(profile.plan_format for profile in profiles) == (
        "kv_plan",
        "kv_plan",
        "kv_plan",
        "retention_ir",
        "kv_plan",
    )
    assert all(profile.token_reclamation == "off" for profile in profiles)
    assert set(load_runtime_target()["supported_topologies"]) > {
        profile.execution_topology for profile in profiles
    }


def test_product_takeover_profile_partition_must_match_static_target(
    tmp_path: Path,
) -> None:
    profile = json.loads(
        (REPOSITORY_ROOT / "compat/sglang/profile.json").read_text(encoding="utf-8")
    )
    profile["manager_supported_profiles"] = [
        item
        for item in profile["manager_supported_profiles"]
        if item["execution_topology"] != "whole_domain_full_latent_kv"
    ]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="do not partition"):
        load_product_takeover_profiles(path)


def test_product_takeover_profile_rejects_schema_v3(tmp_path: Path) -> None:
    profile = json.loads(
        (REPOSITORY_ROOT / "compat/sglang/profile.json").read_text(encoding="utf-8")
    )
    profile["schema_version"] = 3
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 4"):
        load_product_takeover_profiles(path)


def test_product_takeover_profile_rejects_removed_structured_data_plane(
    tmp_path: Path,
) -> None:
    profile = json.loads(
        (REPOSITORY_ROOT / "compat/sglang/profile.json").read_text(encoding="utf-8")
    )
    profile["manager_supported_profiles"][0]["structured_data_plane"] = "off"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="structured_data_plane"):
        load_product_takeover_profiles(path)


@pytest.mark.parametrize("kind", ("sliding", "chunked"))
def test_python_runtime_binding_matches_rust_cli(tmp_path: Path, kind: str) -> None:
    manifest = _sliding_manifest(tmp_path) if kind == "sliding" else _chunked_manifest(tmp_path)
    manifest_path = tmp_path / f"{kind}-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = subprocess.run(
        [
            "cargo", "run", "--quiet", "--bin", "orbitkv", "--",
            "bind-runtime-manifest", str(manifest_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    binding = runtime_binding_from_manifest(manifest)
    assert binding == json.loads(completed.stdout)
    assert binding["schema"] == "orbitkv.runtime-binding"
    assert binding["target"]["id"] == "sglang"
    assert binding["required_wire_version"] == WIRE_VERSION
    assert admit_execution_signature(binding["execution_signature"]) == binding["execution_topology"]


def test_runtime_config_carries_and_revalidates_binding(tmp_path: Path) -> None:
    manifest = _sliding_manifest(tmp_path)
    config = _config(tmp_path, manifest)
    assert config.execution_signature == execution_signature_from_manifest(manifest)
    assert config.runtime_binding == runtime_binding_from_manifest(manifest)
    assert admit_runtime_config(config) == "whole_domain_sliding_token_kv"
    selected = product_takeover_profile(config)
    assert selected is not None
    assert selected.execution_topology == "whole_domain_sliding_token_kv"
    assert selected.ordered_attention_classes == ("sliding:token_kv",)


def test_chunked_runtime_config_selects_retention_ir_product_takeover(
    tmp_path: Path,
) -> None:
    manifest = _chunked_manifest(tmp_path)
    config = _config(tmp_path, manifest)

    assert admit_runtime_config(config) == "whole_domain_chunked_token_kv"
    selected = product_takeover_profile(config)

    assert selected is not None
    assert selected.execution_topology == "whole_domain_chunked_token_kv"
    assert selected.ordered_attention_classes == ("chunked:token_kv",)
    assert selected.plan_format == "retention_ir"
    assert selected.cache_policy == "request_private"


def test_latent_runtime_config_selects_request_private_product_takeover(
    tmp_path: Path,
) -> None:
    manifest = _latent_manifest(tmp_path)
    config = _config(tmp_path, manifest)

    assert admit_runtime_config(config) == "whole_domain_full_latent_kv"
    selected = product_takeover_profile(config)

    assert selected is not None
    assert selected.execution_topology == "whole_domain_full_latent_kv"
    assert selected.ordered_attention_classes == ("full:latent_kv",)
    assert selected.plan_format == "kv_plan"
    assert selected.cache_policy == "request_private"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(schema="unknown"), "unsupported"),
        (lambda value: value["supported_topologies"].append(value["supported_topologies"][0]), "sorted, unique"),
        (lambda value: value.update(unknown=True), "fields differ"),
        (lambda value: value.update(required_wire_version=0), "positive integer"),
    ),
)
def test_runtime_target_mutations_fail_closed(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    target = copy.deepcopy(load_runtime_target())
    mutation(target)
    _seal(target)
    signature = execution_signature_from_manifest(_sliding_manifest(tmp_path))
    with pytest.raises(ValueError, match=message):
        admit_execution_signature(signature, target)


def test_runtime_binding_tampering_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path, _sliding_manifest(tmp_path))
    binding = copy.deepcopy(config.runtime_binding)
    binding["required_wire_version"] += 1
    _seal(binding)
    poisoned = type(config)(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field != "runtime_binding"
        },
        runtime_binding=binding,
    )
    with pytest.raises(ValueError, match="does not match its manifest or target"):
        admit_runtime_config(poisoned)


def test_installed_distribution_records_runtime_target_when_available() -> None:
    try:
        distribution = importlib.metadata.distribution("orbitkv-sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("source-tree test; wheel metadata is covered by CI")
    files = distribution.files
    assert files is not None
    distribution_root = Path(distribution.locate_file("")).resolve()
    source_root = (REPOSITORY_ROOT / "compat/sglang/bridge/src").resolve()
    if distribution_root == source_root:
        pytest.skip("editable source metadata predates the package-data change")
    assert any(
        str(path).endswith("orbitkv_sglang/resources/runtime_target.json")
        for path in files
    )
