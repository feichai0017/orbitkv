from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from orbitkv_sglang._retention_ir import layout_plan_fingerprint
from orbitkv_sglang.config import load_config
from orbitkv_sglang.runtime_manifest import validate_runtime_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _seal(value: dict[str, Any]) -> None:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    value["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()


def _compile_program(tmp_path: Path, program: dict[str, Any]) -> dict[str, Any]:
    source = tmp_path / "retention-ir.json"
    source.write_text(json.dumps(program), encoding="utf-8")
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "orbitkv",
            "--",
            "compile-retention-runtime-manifest",
            str(source),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _chunked_program() -> dict[str, Any]:
    return {
        "schema": "orbitkv.retention-ir.v1",
        "page_tokens": 16,
        "states": [
            {
                "name": "chunked_attention",
                "layers": [0, 1, 2, 3],
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
    }


def _load(tmp_path: Path, manifest: dict[str, Any]):
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()
    return load_config(
        {
            "ORBITKV_RUNTIME_MANIFEST": str(manifest_path),
            "ORBITKV_LIBRARY": str(library),
        }
    )


def test_retention_ir_manifest_validates_and_loads_chunked_runtime(
    tmp_path: Path,
) -> None:
    manifest = _compile_program(tmp_path, _chunked_program())
    assert validate_runtime_manifest(manifest) is manifest
    assert manifest["version"] == 3
    assert manifest["source"]["kind"] == "retention_ir"
    layout = manifest["token_manager_plan"]["layout"]
    assert layout["plan_fingerprint"] == layout_plan_fingerprint(layout)

    config = _load(tmp_path, manifest)
    assert config.manager_plan_format == "retention_ir"
    assert config.plan_json == _canonical_json(manifest["source"]["program"])
    assert config.chunked_class is not None
    assert config.chunked_class.chunk_tokens == 64
    assert config.chunked_class.blocks_per_epoch == 4
    assert config.runtime_binding["execution_topology"] == (
        "whole_domain_chunked_token_kv"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["token_manager_plan"]["layout"]["classes"][0][
                "address"
            ].update(kind="append_only"),
            "unknown fields|inconsistent append_only|retention program",
        ),
        (
            lambda value: value["capability_requirements"].remove(
                "semantic_retirement"
            ),
            "capability_requirements differ",
        ),
        (
            lambda value: value.__setitem__("attention_state_plan", {}),
            "must be null for Retention IR",
        ),
    ),
)
def test_retention_ir_semantic_tampering_fails_after_resealing(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    manifest = _compile_program(tmp_path, _chunked_program())
    mutation(manifest)
    _seal(manifest)
    with pytest.raises(ValueError, match=message):
        validate_runtime_manifest(manifest)


def test_retention_ir_rejects_layout_fingerprint_tampering(tmp_path: Path) -> None:
    manifest = _compile_program(tmp_path, _chunked_program())
    tampered = copy.deepcopy(manifest)
    tampered["token_manager_plan"]["layout"]["plan_fingerprint"] = (
        "sha256:" + "0" * 64
    )
    _seal(tampered)
    with pytest.raises(ValueError, match="plan_fingerprint does not match"):
        validate_runtime_manifest(tampered)


def test_retention_ir_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    manifest = _compile_program(tmp_path, _chunked_program())
    encoded = json.dumps(manifest).replace(
        '"version": 3', '"version": 2, "version": 3', 1
    )
    manifest_path = tmp_path / "duplicate.json"
    manifest_path.write_text(encoded, encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.touch()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_config(
            {
                "ORBITKV_RUNTIME_MANIFEST": str(manifest_path),
                "ORBITKV_LIBRARY": str(library),
            }
        )
