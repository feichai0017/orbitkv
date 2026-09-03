from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import orbitkv_sglang.runtime_admission as runtime_admission
from orbitkv_sglang.config import (
    MANAGER_PLAN_JSON_MAX_BYTES,
    RUNTIME_MANIFEST_MAX_BYTES,
    _validate_manager_plan_json,
    load_config,
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


def _compiler_input(source_kind: str, name: str) -> tuple[str, dict[str, Any]]:
    if source_kind == "attention_state":
        return (
            "compile-runtime-manifest",
            {
                "page_tokens": 16,
                "states": [
                    {
                        "name": name,
                        "layers": [0],
                        "storage": {
                            "kind": "token_kv",
                            "key_bytes_per_token_per_layer": 64,
                            "value_bytes_per_token_per_layer": 64,
                            "retention": "full",
                            "window_tokens": None,
                        },
                    }
                ],
            },
        )
    assert source_kind == "retention_ir"
    return (
        "compile-retention-runtime-manifest",
        {
            "schema": "orbitkv.retention-ir.v1",
            "page_tokens": 16,
            "states": [
                {
                    "name": name,
                    "layers": [0],
                    "bytes_per_token_per_layer": 128,
                    "may_read": {
                        "op": "equal",
                        "lhs": {
                            "op": "floor_div",
                            "value": {"op": "query_position"},
                            "divisor": 64,
                        },
                        "rhs": {
                            "op": "floor_div",
                            "value": {"op": "key_position"},
                            "divisor": 64,
                        },
                    },
                }
            ],
        },
    )


def _compile_manifest(
    directory: Path, source_kind: str, name: str, label: str
) -> dict[str, Any]:
    command, compiler_input = _compiler_input(source_kind, name)
    path = directory / f"{source_kind}-{label}.json"
    path.write_text(json.dumps(compiler_input), encoding="utf-8")
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "orbitkv",
            "--",
            command,
            str(path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _embedded_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["source"][
        "input" if manifest["source"]["kind"] == "attention_state" else "program"
    ]


def _native_plan_size(manifest: dict[str, Any]) -> int:
    if manifest["source"]["kind"] == "retention_ir":
        return len(_canonical_json(_embedded_plan(manifest)))
    state = manifest["source"]["input"]["states"][0]
    storage = state["storage"]
    native = {
        "page_tokens": 16,
        "classes": [
            {
                "name": state["name"],
                "layers": state["layers"],
                "retention": storage["retention"],
                "bytes_per_token_per_layer": (
                    storage["key_bytes_per_token_per_layer"]
                    + storage["value_bytes_per_token_per_layer"]
                ),
                "window_tokens": storage["window_tokens"],
                "components": [
                    {"name": "key", "bytes_per_token_per_layer": 64},
                    {"name": "value", "bytes_per_token_per_layer": 64},
                ],
            }
        ],
    }
    return len(_canonical_json(native))


def _native_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["source"]["kind"] == "retention_ir":
        return _embedded_plan(manifest)
    state = manifest["source"]["input"]["states"][0]
    storage = state["storage"]
    return {
        "page_tokens": 16,
        "classes": [
            {
                "name": state["name"],
                "layers": state["layers"],
                "retention": storage["retention"],
                "bytes_per_token_per_layer": (
                    storage["key_bytes_per_token_per_layer"]
                    + storage["value_bytes_per_token_per_layer"]
                ),
                "window_tokens": storage["window_tokens"],
                "components": [
                    {"name": "key", "bytes_per_token_per_layer": 64},
                    {"name": "value", "bytes_per_token_per_layer": 64},
                ],
            }
        ],
    }


@pytest.fixture(scope="module")
def boundary_manifests(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, int], dict[str, Any]]:
    directory = tmp_path_factory.mktemp("manager-plan-boundary")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for source_kind in ("attention_state", "retention_ir"):
        probe = _compile_manifest(directory, source_kind, "x", "probe")
        probe_size = _native_plan_size(probe)
        exact_name_length = 1 + MANAGER_PLAN_JSON_MAX_BYTES - probe_size
        assert exact_name_length > 0
        for extra in (0, 1):
            manifest = _compile_manifest(
                directory,
                source_kind,
                "x" * (exact_name_length + extra),
                f"limit-plus-{extra}",
            )
            assert _native_plan_size(manifest) == (
                MANAGER_PLAN_JSON_MAX_BYTES + extra
            )
            result[(source_kind, extra)] = manifest
    return result


def _environment(
    tmp_path: Path, manifest: dict[str, Any], label: str
) -> dict[str, str]:
    manifest_path = tmp_path / f"{label}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        MANAGER_PLAN_JSON_MAX_BYTES
        < manifest_path.stat().st_size
        <= RUNTIME_MANIFEST_MAX_BYTES
    )
    library_path = tmp_path / "liborbitkv_ffi.so"
    library_path.touch()
    return {
        "ORBITKV_RUNTIME_MANIFEST": str(manifest_path),
        "ORBITKV_LIBRARY": str(library_path),
    }


def test_python_limit_matches_native_inclusive_boundary() -> None:
    exact = b"x" * MANAGER_PLAN_JSON_MAX_BYTES
    assert _validate_manager_plan_json(exact) is exact
    with pytest.raises(ValueError, match=r"1048576-byte \(1 MiB\) native limit"):
        _validate_manager_plan_json(exact + b"x")


@pytest.mark.parametrize("source_kind", ("attention_state", "retention_ir"))
def test_runtime_manifest_accepts_embedded_plan_at_native_limit(
    tmp_path: Path,
    boundary_manifests: dict[tuple[str, int], dict[str, Any]],
    source_kind: str,
) -> None:
    manifest = boundary_manifests[(source_kind, 0)]
    config = load_config(_environment(tmp_path, manifest, f"{source_kind}-limit"))
    assert len(config.plan_json) == MANAGER_PLAN_JSON_MAX_BYTES
    if source_kind == "retention_ir":
        assert config.plan_json == _canonical_json(_embedded_plan(manifest))
    assert config.manager_plan_format == (
        "kv_plan" if source_kind == "attention_state" else "retention_ir"
    )


@pytest.mark.parametrize("source_kind", ("attention_state", "retention_ir"))
def test_runtime_manifest_rejects_limit_plus_one_before_binding_or_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary_manifests: dict[tuple[str, int], dict[str, Any]],
    source_kind: str,
) -> None:
    binding_calls = 0

    def forbidden_binding(_manifest: Any) -> Any:
        nonlocal binding_calls
        binding_calls += 1
        raise AssertionError("target binding must not run for an oversized plan")

    monkeypatch.setattr(
        runtime_admission,
        "runtime_binding_from_manifest",
        forbidden_binding,
    )

    environment = _environment(
        tmp_path,
        boundary_manifests[(source_kind, 1)],
        f"{source_kind}-limit-plus-one",
    )
    with pytest.raises(ValueError, match=r"1048576-byte \(1 MiB\) native limit"):
        load_config(environment)
    assert binding_calls == 0
