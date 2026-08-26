from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang.config import RUNTIME_MANIFEST_MAX_BYTES, load_config
from orbitkv_sglang.ffi import CtypesManagerFactory
from orbitkv_sglang.runtime import ArenaRegistration, ManagerCreateSettings


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def rust_manifest() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "orbitkv",
            "--",
            "compile-runtime-manifest",
            "examples/qwen3.5-0.8b-attention-state-input-page16-bf16.json",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(completed.stdout)
    payload = {
        key: value for key, value in manifest.items() if key != "fingerprint"
    }
    assert manifest["fingerprint"] == "sha256:" + hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    return manifest


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _seal(manifest: dict[str, Any]) -> None:
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()


def test_canonical_fingerprint_interoperability_vector() -> None:
    value = {
        "z": ["状态", 'line\nquote"slash\\', None, True, 42],
        "a": {"é": "值", "\x01": "\b\f\n\r\t"},
    }
    assert (
        "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()
        == "sha256:95762173915ab37233eb16469bdb42ad0c2efcc5aad56052a87ec2ec33819b7f"
    )


def _write_inputs(
    tmp_path: Path, manifest: dict[str, Any]
) -> tuple[Path, Path, dict[str, str]]:
    path = tmp_path / "runtime-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()
    environment = {
        "ORBITKV_RUNTIME_MANIFEST": str(path),
        "ORBITKV_LIBRARY": str(library),
    }
    return path, library, environment


def test_rust_manifest_loads_as_unified_manager_config(
    tmp_path: Path, rust_manifest: dict[str, Any]
) -> None:
    manifest_path, library, environment = _write_inputs(tmp_path, rust_manifest)

    config = load_config(environment)

    assert config.plan_path == manifest_path.resolve()
    assert config.library_path == library.resolve()
    assert config.runtime_manifest_path == manifest_path.resolve()
    assert config.plan_fingerprint == "sha256:" + hashlib.sha256(
        config.plan_json
    ).hexdigest()
    assert config.plan_fingerprint != rust_manifest["fingerprint"]
    assert config.runtime_manifest_fingerprint == rust_manifest["fingerprint"]
    assert config.state_plan_path == manifest_path.resolve()
    assert config.state_plan_fingerprint == rust_manifest["fingerprint"]
    assert config.plan_json == _canonical_json(
        rust_manifest["token_manager_plan"]["input"]
    )
    assert tuple(item.name for item in config.classes) == ("full_attention_kv",)
    assert tuple(item.kind for item in config.fixed_states) == ("gdn", "convolution")
    assert config.num_hidden_layers == 24
    assert config.token_reclamation.mode == "off"
    assert config.capability_requirements == tuple(
        rust_manifest["capability_requirements"]
    )


def test_rust_manifest_creates_the_native_abi8_manager(
    tmp_path: Path, rust_manifest: dict[str, Any]
) -> None:
    manifest_path, _placeholder, environment = _write_inputs(
        tmp_path, rust_manifest
    )
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(REPOSITORY_ROOT / "crates/orbitkv-ffi/Cargo.toml"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    environment["ORBITKV_LIBRARY"] = str(
        REPOSITORY_ROOT
        / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"
    )
    config = load_config(environment)
    registrations = tuple(
        ArenaRegistration(
            item.class_id, item.pool_id, item.backend_domain, 8, 0
        )
        for item in config.classes
    )

    manager = CtypesManagerFactory().create(
        config, ManagerCreateSettings(2, 2, 2, 8, 16), registrations
    )
    try:
        assert tuple(item.class_id for item in manager.arenas) == (0,)
        assert manager.stats().free_pages == 8
        assert config.runtime_manifest_path == manifest_path.resolve()
    finally:
        manager.destroy()


def test_manifest_keeps_token_reclamation_as_an_orthogonal_policy(
    tmp_path: Path, rust_manifest: dict[str, Any]
) -> None:
    _path, _library, environment = _write_inputs(tmp_path, rust_manifest)
    environment["ORBITKV_TOKEN_RECLAMATION"] = json.dumps(
        {
            "mode": "relocate",
            "trigger_tokens": 32,
            "retained_per_page": 1,
            "policy_id": 1,
            "policy_version": 1,
            "quality_contract": 1,
            "fragmentation_threshold_milli": 250,
            "maximum_source_pages": 2,
            "evacuation_headroom_pages": 1,
        }
    )

    config = load_config(environment)

    assert config.token_reclamation.mode == "relocate"
    assert config.token_reclamation.trigger_tokens == 32


@pytest.mark.parametrize("legacy_name", ["ORBITKV_PLAN", "ORBITKV_STATE_PLAN"])
def test_manifest_rejects_mixed_legacy_plan_sources(
    tmp_path: Path, rust_manifest: dict[str, Any], legacy_name: str
) -> None:
    manifest_path, _library, environment = _write_inputs(tmp_path, rust_manifest)
    environment[legacy_name] = str(manifest_path)

    with pytest.raises(ValueError, match="cannot be combined with legacy"):
        load_config(environment)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.__setitem__("fingerprint", "sha256:" + "0" * 64),
            "fingerprint does not match",
        ),
        (
            lambda value: value.__setitem__("version", 2),
            "version must be 1",
        ),
        (
            lambda value: value["token_manager_plan"]["layout"].__setitem__(
                "plan_fingerprint", "sha256:" + "0" * 64
            ),
            "layout fingerprint differs",
        ),
        (
            lambda value: value["capability_requirements"].remove(
                "token_manager"
            ),
            "capability_requirements differ",
        ),
        (
            lambda value: value["attention_state_plan"]["states"][0][
                "backend"
            ].__setitem__("token_relocatable", False),
            "token_relocatable must be true",
        ),
        (
            lambda value: value["token_manager_plan"]["layout"]["classes"][
                0
            ].__setitem__("unknown", 1),
            "has unknown fields: unknown",
        ),
        (
            lambda value: value["token_manager_plan"]["input"]["classes"][
                0
            ].update({"storage": "token_kv", "components": []}),
            "must omit default storage and empty components",
        ),
    ],
)
def test_manifest_rejects_tampered_or_noncanonical_contracts(
    tmp_path: Path,
    rust_manifest: dict[str, Any],
    mutate: Any,
    message: str,
) -> None:
    manifest = copy.deepcopy(rust_manifest)
    mutate(manifest)
    if "fingerprint does not match" not in message:
        _seal(manifest)
    _path, _library, environment = _write_inputs(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        load_config(environment)


def test_manifest_rejects_duplicate_keys_and_non_finite_numbers(
    tmp_path: Path, rust_manifest: dict[str, Any]
) -> None:
    path, _library, environment = _write_inputs(tmp_path, rust_manifest)
    encoded = json.dumps(rust_manifest)
    path.write_text(
        encoded.replace(
            '{"schema":', '{"schema": "duplicate", "schema":', 1
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON key: schema"):
        load_config(environment)

    path.write_text(encoded.replace('"version": 1', '"version": NaN', 1))
    with pytest.raises(ValueError, match="non-finite JSON number: NaN"):
        load_config(environment)


def test_manifest_read_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "runtime-manifest.json"
    path.write_bytes(b" " * (RUNTIME_MANIFEST_MAX_BYTES + 1))
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()

    with pytest.raises(ValueError, match="exceeds the 16777216-byte limit"):
        load_config(
            {
                "ORBITKV_RUNTIME_MANIFEST": str(path),
                "ORBITKV_LIBRARY": str(library),
            }
        )


def test_fixed_only_manifest_is_not_executable_by_sglang(
    tmp_path: Path, rust_manifest: dict[str, Any]
) -> None:
    manifest = copy.deepcopy(rust_manifest)
    manifest["token_manager_plan"] = None
    manifest["attention_state_plan"]["states"] = manifest[
        "attention_state_plan"
    ]["states"][1:]
    manifest["capability_requirements"] = [
        "convolution_state",
        "fixed_state_checkpoints",
        "recurrent_state",
    ]
    _seal(manifest)
    _path, _library, environment = _write_inputs(tmp_path, manifest)

    with pytest.raises(ValueError, match="requires RuntimeManifest.token_manager_plan"):
        load_config(environment)


def test_legacy_plan_loading_is_unchanged(tmp_path: Path) -> None:
    plan = {
        "page_tokens": 16,
        "classes": [
            {
                "name": "full",
                "layers": [0],
                "retention": "full",
                "bytes_per_token_per_layer": 128,
                "window_tokens": None,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()

    config = load_config(
        {"ORBITKV_PLAN": str(plan_path), "ORBITKV_LIBRARY": str(library)}
    )

    canonical = _canonical_json(plan)
    assert config.plan_path == plan_path.resolve()
    assert config.plan_json == canonical
    assert config.plan_fingerprint == (
        "sha256:" + hashlib.sha256(canonical).hexdigest()
    )
    assert config.runtime_manifest_path is None
    assert config.runtime_manifest_fingerprint is None
    assert config.capability_requirements == ()
