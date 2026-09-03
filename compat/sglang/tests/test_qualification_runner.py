from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
import subprocess
import sys
import uuid
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "bridge/src"
TOOLS_ROOT = INTEGRATION_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import qualification_runner as bench  # noqa: E402
from orbitkv_sglang.runtime_manifest import (  # noqa: E402
    RUNTIME_MANIFEST_VERSION,
)
from orbitkv_sglang._retention_ir import layout_plan_fingerprint  # noqa: E402
from orbitkv_sglang.runtime.pressure import (  # noqa: E402
    ArenaPressureSample,
    PressureClassDescriptor,
    PressureTelemetry,
)
from qualification_chunked_record_support import (  # noqa: E402
    assert_chunked_record_shape as _assert_chunked_record_shape_helper,
)


PAGE_TOKENS = 16
CHUNK_TOKENS = 64
TOPOLOGY = "whole_domain_chunked_token_kv"
CLAIM_KEYS = {
    "correctness_qualified",
    "stream_event_qualified",
    "capacity_qualified",
    "throughput_go",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _seal(value: dict[str, object]) -> None:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    value["fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()


def _chunked_manifest() -> dict[str, object]:
    program = {
        "schema": "orbitkv.retention-ir.v1",
        "page_tokens": PAGE_TOKENS,
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
                        "divisor": CHUNK_TOKENS,
                    },
                    "rhs": {
                        "op": "floor_div",
                        "value": {"op": "key_position"},
                        "divisor": CHUNK_TOKENS,
                    },
                },
            }
        ],
    }
    layout = {
        "schema": "orbitkv.layout-program.v1",
        "plan_fingerprint": "",
        "page_tokens": PAGE_TOKENS,
        "classes": [
            {
                "name": "chunked_attention",
                "layers": [0, 1, 2, 3],
                "bytes_per_token_per_layer": 128,
                "address": {
                    "kind": "resettable_arena",
                    "blocks_per_epoch": CHUNK_TOKENS // PAGE_TOKENS,
                },
                "retirement": {
                    "kind": "epoch_end",
                    "blocks_per_epoch": CHUNK_TOKENS // PAGE_TOKENS,
                },
                "minimum_slots_per_request": CHUNK_TOKENS // PAGE_TOKENS,
            }
        ],
    }
    layout["plan_fingerprint"] = layout_plan_fingerprint(layout)
    manifest: dict[str, object] = {
        "schema": "orbitkv.runtime-manifest",
        "version": RUNTIME_MANIFEST_VERSION,
        "fingerprint": "",
        "source": {"kind": "retention_ir", "program": program},
        "token_manager_plan": {"layout": layout},
        "attention_state_plan": None,
        "capability_requirements": [
            "resettable_arena_addressing",
            "semantic_retirement",
            "token_manager",
        ],
    }
    _seal(manifest)
    return manifest


def _write_inputs(tmp_path: Path) -> dict[str, str]:
    sglang = tmp_path / "sglang"
    (sglang / "python/sglang").mkdir(parents=True)
    (sglang / "python/sglang/__init__.py").write_text("", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 4,
                "vocab_size": 1024,
                "max_position_embeddings": 1024,
                "attention_chunk_size": CHUNK_TOKENS,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "chunked-runtime-manifest.json"
    manifest.write_text(json.dumps(_chunked_manifest()), encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.write_bytes(b"ffi")
    return {
        "sglang_root": str(sglang),
        "model": str(model),
        "manifest": str(manifest),
        "library": str(library),
    }


def _arguments(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "mode": "manager",
        "case": "roomy",
        "sglang_root": "/sglang",
        "model": "/model",
        "manifest": "/chunked-runtime-manifest.json",
        "library": "/liborbitkv_ffi.so",
        "prompt_tokens": 48,
        "decode_tokens": 97,
        "iterations": 3,
        "context_length": 256,
        "max_total_tokens": 256,
        "max_running_requests": 1,
        "mem_fraction_static": None,
        "seed": 20260828,
    }
    values.update(overrides)
    return Namespace(**values)


def _run_arguments(mode: str, inputs: dict[str, str]) -> Namespace:
    common = {
        "case": "roomy" if mode == "manager" else "exact-floor",
        "max_total_tokens": 256 if mode == "manager" else CHUNK_TOKENS,
    }
    if mode == "manager":
        return _arguments(**inputs, **common)
    return _arguments(
        mode="stock",
        manifest=None,
        library=None,
        **common,
        **{key: inputs[key] for key in ("sglang_root", "model")},
    )


def _admission(path: Path) -> dict[str, object]:
    result = bench.load_qualification_admission(path)
    assert set(result) == {
        "runtime_manifest",
        "runtime_binding",
        "chunk_geometry",
        "lifecycle_route",
        "cache_policy",
    }
    return result


def test_benchmark_import_is_host_only() -> None:
    code = (
        "import sys; "
        f"sys.path[:0] = [{str(TOOLS_ROOT)!r}, {str(SOURCE_ROOT)!r}]; "
        "import qualification_runner; "
        "assert not any(name == 'sglang' or name.startswith('sglang.') "
        "for name in sys.modules); "
        "assert not any(name == 'torch' or name.startswith('torch.') "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=Path("/tmp"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_requires_independent_manager_or_stock_artifacts(tmp_path: Path) -> None:
    parser = bench.build_parser()
    assert tuple(parser._option_string_actions["--mode"].choices) == (
        "manager",
        "stock",
    )
    assert parser._option_string_actions["--mode"].required is True
    assert tuple(parser._option_string_actions["--case"].choices) == (
        "roomy",
        "exact-floor",
    )
    assert parser._option_string_actions["--case"].required is True
    assert parser._option_string_actions["--sglang-root"].required is True
    assert parser._option_string_actions["--model"].required is True

    inputs = _write_inputs(tmp_path)
    manager = bench.validate_arguments(_arguments(**inputs))
    assert manager["manifest"] == Path(inputs["manifest"]).resolve()
    assert manager["library"] == Path(inputs["library"]).resolve()

    with pytest.raises(ValueError, match="manager.*manifest.*library|requires"):
        bench.validate_arguments(_arguments(**{**inputs, "manifest": None}))
    with pytest.raises(ValueError, match="stock.*forbid|forbids"):
        bench.validate_arguments(_arguments(mode="stock", **inputs))

    stock = bench.validate_arguments(
        _arguments(
            mode="stock",
            manifest=None,
            library=None,
            **{key: inputs[key] for key in ("sglang_root", "model")},
        )
    )
    assert stock["manifest"] is None
    assert stock["library"] is None


def test_load_qualification_admission_is_strict_and_exact_topology(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = _chunked_manifest()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _admission(path)

    assert bench.WIRE_VERSION == 14
    assert result["runtime_manifest"] == manifest
    assert result["runtime_binding"]["target"] == {
        "id": "sglang",
        "contract_version": 4,
    }
    assert result["runtime_binding"]["required_wire_version"] == 14
    assert result["runtime_binding"]["execution_topology"] == TOPOLOGY
    assert result["runtime_binding"]["manifest_fingerprint"] == (
        manifest["fingerprint"]
    )
    assert result["chunk_geometry"] == {
        "page_tokens": PAGE_TOKENS,
        "chunk_tokens": CHUNK_TOKENS,
        "blocks_per_epoch": CHUNK_TOKENS // PAGE_TOKENS,
    }
    assert result["lifecycle_route"] == "native_session"
    assert result["cache_policy"] == "request_private"


def test_load_qualification_admission_rejects_tampering_and_nonexact_topology(
    tmp_path: Path,
) -> None:
    manifest = _chunked_manifest()
    path = tmp_path / "bad.json"

    manifest["version"] = 1
    _seal(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="version.*3|must be 3"):
        bench.load_qualification_admission(path)

    manifest = _chunked_manifest()
    manifest["fingerprint"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        bench.load_qualification_admission(path)

    manifest = _chunked_manifest()
    state = manifest["source"]["program"]["states"][0]
    item = manifest["token_manager_plan"]["layout"]["classes"][0]
    state["kv_head_range"] = {"start": 0, "end_exclusive": 1}
    item["kv_head_range"] = {"start": 0, "end_exclusive": 1}
    manifest["capability_requirements"].insert(0, "kv_head_partitioning")
    layout = manifest["token_manager_plan"]["layout"]
    layout["plan_fingerprint"] = layout_plan_fingerprint(layout)
    _seal(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=(
            "whole token domain|topology|capability_requirements differ|"
            "unknown fields: kv_head_range"
        ),
    ):
        bench.load_qualification_admission(path)


def test_workload_crosses_chunk_boundary_and_supports_multiple_records() -> None:
    workload = bench.validate_chunked_workload(
        prompt_tokens=48,
        decode_tokens=97,
        chunk_tokens=CHUNK_TOKENS,
        iterations=3,
    )
    assert workload == {
        "prompt_tokens": 48,
        "decode_tokens": 97,
        "final_kv_tokens": 144,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_epoch_count": 3,
        "iterations": 3,
    }
    assert workload["prompt_tokens"] <= workload["chunk_tokens"]
    assert workload["final_kv_tokens"] > workload["chunk_tokens"]
    assert workload["chunk_epoch_count"] >= 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"prompt_tokens": CHUNK_TOKENS + 1}, "prompt"),
        ({"prompt_tokens": 32, "decode_tokens": 33}, "cross|chunk"),
        ({"iterations": 0}, "iteration"),
        ({"chunk_tokens": 0}, "chunk"),
    ),
)
def test_workload_rejects_non_crossing_or_single_epoch_runs(
    overrides: dict[str, int], message: str
) -> None:
    values = {
        "prompt_tokens": 48,
        "decode_tokens": 97,
        "chunk_tokens": CHUNK_TOKENS,
        "iterations": 3,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        bench.validate_chunked_workload(**values)


def test_case_capacity_distinguishes_roomy_and_exact_floor() -> None:
    assert (
        bench.validate_case_capacity(
            case="roomy",
            max_total_tokens=144,
            chunk_tokens=CHUNK_TOKENS,
            final_kv_tokens=144,
        )
        is None
    )
    assert (
        bench.validate_case_capacity(
            case="exact-floor",
            max_total_tokens=CHUNK_TOKENS,
            chunk_tokens=CHUNK_TOKENS,
            final_kv_tokens=144,
        )
        is None
    )
    with pytest.raises(ValueError, match="roomy.*materialized KV"):
        bench.validate_case_capacity(
            case="roomy",
            max_total_tokens=128,
            chunk_tokens=CHUNK_TOKENS,
            final_kv_tokens=144,
        )
    with pytest.raises(ValueError, match="exact-floor.*chunk_tokens"):
        bench.validate_case_capacity(
            case="exact-floor",
            max_total_tokens=128,
            chunk_tokens=CHUNK_TOKENS,
            final_kv_tokens=144,
        )
    with pytest.raises(ValueError, match="physical chunk floor"):
        bench.validate_case_capacity(
            case="exact-floor",
            max_total_tokens=CHUNK_TOKENS,
            chunk_tokens=CHUNK_TOKENS,
            final_kv_tokens=CHUNK_TOKENS,
        )


def test_validate_arguments_allows_exact_floor_logical_workload(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    args = _arguments(case="exact-floor", max_total_tokens=CHUNK_TOKENS, **inputs)
    resolved = bench.validate_arguments(args)
    workload = bench.validate_chunked_workload(
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        chunk_tokens=CHUNK_TOKENS,
        iterations=args.iterations,
    )
    assert workload["final_kv_tokens"] == 144
    assert workload["final_kv_tokens"] > args.max_total_tokens
    assert (
        bench.validate_case_capacity(
            case=args.case,
            max_total_tokens=args.max_total_tokens,
            chunk_tokens=CHUNK_TOKENS,
            final_kv_tokens=workload["final_kv_tokens"],
        )
        is None
    )
    assert resolved["manifest"] == Path(inputs["manifest"]).resolve()


@pytest.mark.parametrize(
    "message",
    (
        "KV cache pool is full",
        "prefill out of memory",
        "decode out of memory",
        "out of memory even after retracting all other requests",
        "Try to allocate 17 tokens",
    ),
)
def test_capacity_failure_classifier_accepts_only_explicit_kv_markers(
    message: str,
) -> None:
    assert (
        bench.classify_capacity_failure(RuntimeError(message))
        == "capacity_exhausted"
    )


@pytest.mark.parametrize(
    "message",
    (
        "CUDA out of memory",
        "prefill out of memory: CUDA out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "HIP out of memory",
        "allocator could not reserve device memory",
        "arbitrary RuntimeError",
        "",
    ),
)
def test_capacity_failure_classifier_reraises_non_kv_failures(
    message: str,
) -> None:
    error = RuntimeError(message)
    with pytest.raises(RuntimeError) as captured:
        bench.classify_capacity_failure(error)
    assert captured.value is error


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_source_helper_calls_direct_pinned_checkout_validator(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = (tmp_path / f"{mode}-checkout").resolve()
    checkout.mkdir()
    calls: list[tuple[str, Path]] = []
    patch = b"reviewed patch"
    patch_path = tmp_path / "reviewed.patch"
    patch_path.write_bytes(patch)
    contract = {
        "release": "v0.test",
        "revision": "abc123",
        "patch_path": str(patch_path.relative_to(tmp_path)),
        "targets": [],
        "patch_diff_sha256": hashlib.sha256(patch).hexdigest(),
    }

    def patched(root: Path) -> Path:
        calls.append(("patched", root))
        return checkout

    def base(root: Path) -> Path:
        calls.append(("base", root))
        return checkout

    monkeypatch.setattr(bench.pinned, "validate_patched_checkout", patched)
    monkeypatch.setattr(bench.pinned, "validate_base_checkout", base)
    monkeypatch.setattr(bench.pinned, "pinned_source_contract", lambda: contract)
    monkeypatch.setattr(bench, "_REPOSITORY_ROOT", tmp_path)
    assert not hasattr(bench, "common")
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )

    result = bench.verify_chunked_source(checkout, mode)

    assert calls == [("patched" if mode == "manager" else "base", checkout)]
    assert result == {
        "root": str(checkout),
        "release": "v0.test",
        "revision": "abc123",
        "contract": contract,
        "contract_sha256": bench.canonical_digest(contract),
        "reviewed_patch": (
            {
                "status": "applied",
                "sha256": hashlib.sha256(patch).hexdigest(),
                "bytes": len(patch),
            }
            if mode == "manager"
            else {"status": "absent", "sha256": None, "bytes": 0}
        ),
    }


def test_direct_source_rejects_legacy_manager_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = SimpleNamespace(name="unrelated_plugin")
    monkeypatch.setattr(
        bench.runtime_support.importlib.metadata,
        "entry_points",
        lambda **_kwargs: [unrelated],
    )
    bench.runtime_support.reject_legacy_manager_entrypoint()

    legacy = SimpleNamespace(name="orbitkv_manager")
    monkeypatch.setattr(
        bench.runtime_support.importlib.metadata,
        "entry_points",
        lambda **_kwargs: [unrelated, legacy],
    )
    with pytest.raises(RuntimeError, match="forbids.*orbitkv_manager"):
        bench.runtime_support.reject_legacy_manager_entrypoint()

    def discovery_failure(**_kwargs: object) -> object:
        raise ValueError("broken entry-point metadata")

    monkeypatch.setattr(
        bench.runtime_support.importlib.metadata,
        "entry_points",
        discovery_failure,
    )
    with pytest.raises(RuntimeError, match="cannot prove.*absence"):
        bench.runtime_support.reject_legacy_manager_entrypoint()


def test_manager_run_rejects_legacy_entrypoint_before_source_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bench,
        "configure_environment",
        lambda _args, _paths: calls.append("environment") or {},
    )

    def reject() -> None:
        calls.append("entrypoint")
        raise RuntimeError("legacy entrypoint installed")

    monkeypatch.setattr(
        bench.runtime_support, "reject_legacy_manager_entrypoint", reject
    )
    monkeypatch.setattr(
        bench,
        "verify_chunked_source",
        lambda *_args: calls.append("source"),
    )

    with pytest.raises(RuntimeError, match="legacy entrypoint installed"):
        bench.run(_arguments(mode="manager"), {})
    assert calls == ["environment", "entrypoint"]


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_engine_arguments_pin_exact_chunked_execution_contract(mode: str) -> None:
    values = bench.engine_arguments(
        _arguments(mode=mode),
        Path("/model"),
        {
            "page_tokens": PAGE_TOKENS,
            "chunk_tokens": CHUNK_TOKENS,
            "blocks_per_epoch": CHUNK_TOKENS // PAGE_TOKENS,
        },
    )
    assert values["attention_backend"] == "fa3"
    assert values["dtype"] == values["kv_cache_dtype"] == "bfloat16"
    assert values["page_size"] == PAGE_TOKENS
    assert values["disable_cuda_graph"] is True
    assert values["enable_torch_compile"] is False
    assert values["tp_size"] == 1
    assert values["pp_size"] == 1
    assert values["dp_size"] == 1
    assert values["dcp_size"] == 1
    assert values["chunked_prefill_size"] == CHUNK_TOKENS
    assert values["prefill_max_requests"] == 1
    assert values["enable_dynamic_chunking"] is False
    assert values["enable_mixed_chunk"] is False
    assert values["max_prefill_tokens"] == CHUNK_TOKENS
    assert values["enable_page_major_kv_layout"] is False
    assert values["disable_overlap_schedule"] is True
    assert values["speculative_algorithm"] is None


def test_environment_selects_direct_source_without_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("ORBITKV_STALE", "must-disappear")
    monkeypatch.setenv("ORBITKV_PLAN", "must-disappear")
    monkeypatch.setenv("SGLANG_PLUGINS", "stale-installed-plugin")

    manager_args = _arguments(**inputs)
    manager_paths = bench.validate_arguments(manager_args)
    manager = bench.configure_environment(manager_args, manager_paths)
    assert "SGLANG_PLUGINS" not in manager
    assert "SGLANG_PLUGINS" not in os.environ
    assert manager["ORBITKV_RUNTIME_MANIFEST"] == str(
        Path(inputs["manifest"]).resolve()
    )
    assert manager["ORBITKV_LIBRARY"] == str(
        Path(inputs["library"]).resolve()
    )
    assert manager["ORBITKV_PRESSURE_TELEMETRY"] == "1"
    assert os.environ["ORBITKV_PRESSURE_TELEMETRY"] == "1"
    assert "ORBITKV_PLAN" not in os.environ
    assert "ORBITKV_STALE" not in os.environ

    monkeypatch.setenv("ORBITKV_RUNTIME_MANIFEST", "stale")
    monkeypatch.setenv("ORBITKV_LIBRARY", "stale")
    monkeypatch.setenv("SGLANG_PLUGINS", "orbitkv_manager")
    stock_args = _arguments(
        mode="stock",
        manifest=None,
        library=None,
        **{key: inputs[key] for key in ("sglang_root", "model")},
    )
    stock_paths = bench.validate_arguments(stock_args)
    stock = bench.configure_environment(stock_args, stock_paths)
    assert "SGLANG_PLUGINS" not in stock
    assert "SGLANG_PLUGINS" not in os.environ
    assert not any(name.startswith("ORBITKV_") for name in os.environ)
    assert not any(name.startswith("ORBITKV_") for name in stock)


def test_request_traces_preserve_exact_ids_output_ids_and_digests() -> None:
    outputs = (
        (
            {
                "output_ids": [11, 12, 13],
                "meta_info": {"id": "chunked-0-0", "cached_tokens": 0},
            },
        ),
        (
            {
                "output_ids": [21, 22, 23],
                "meta_info": {"id": "chunked-1-0", "cached_tokens": 0},
            },
        ),
    )
    traces = bench.request_traces(
        outputs=outputs,
        submitted_rids=(("chunked-0-0",), ("chunked-1-0",)),
        submitted_input_digests=(("input-0",), ("input-1",)),
    )
    assert traces == [
        [
            {
                "request_index": 0,
                "submitted_rid": "chunked-0-0",
                "submitted_input_ids_sha256": "input-0",
                "returned_rid": "chunked-0-0",
                "cached_tokens": 0,
                "output_ids": [11, 12, 13],
                "output_ids_sha256": bench.canonical_digest([11, 12, 13]),
            }
        ],
        [
            {
                "request_index": 0,
                "submitted_rid": "chunked-1-0",
                "submitted_input_ids_sha256": "input-1",
                "returned_rid": "chunked-1-0",
                "cached_tokens": 0,
                "output_ids": [21, 22, 23],
                "output_ids_sha256": bench.canonical_digest([21, 22, 23]),
            }
        ],
    ]
    assert bench.request_token_digests(outputs) == [
        [bench.canonical_digest([11, 12, 13])],
        [bench.canonical_digest([21, 22, 23])],
    ]
    assert bench.token_digest(outputs) == bench.canonical_digest(
        [[[11, 12, 13]], [[21, 22, 23]]]
    )

    foreign = copy.deepcopy(outputs)
    foreign[1][0]["meta_info"]["id"] = "foreign"
    with pytest.raises(RuntimeError, match="foreign request id"):
        bench.request_traces(
            outputs=foreign,
            submitted_rids=(("chunked-0-0",), ("chunked-1-0",)),
            submitted_input_digests=(("input-0",), ("input-1",)),
        )


def _manager_info(
    *, wire_version: int = bench.WIRE_VERSION
) -> dict[str, object]:
    return {
        "internal_states": [
            {
                "orbitkv_manager": {
                    "lifecycle_route": "native_session",
                    "cache_policy": "request_private",
                    "wire_version": wire_version,
                    "plan_fingerprint": "sha256:retention-ir",
                    "runtime_manifest_fingerprint": "sha256:manifest",
                    "runtime_binding_fingerprint": "sha256:binding",
                    "fixed_state_byte_count": 0,
                    "fixed_state_descriptors": [],
                    "tree_cache_type": "ChunkCache",
                    "runtime_proof": _runtime_proof(),
                    "identities": [
                        {
                            "engine_epoch": 1,
                            "pool_epoch": 2,
                            "pool_id": 1,
                            "class_id": 0,
                            "backend_domain": 1,
                            "page_count": 8,
                            "page_tokens": PAGE_TOKENS,
                            "backend_base_index": 0,
                            "first_page_id": 1,
                        }
                    ],
                    "manager_stats": {
                        "active_requests": 0,
                        "active_snapshots": 0,
                        "active_prefixes": 0,
                        "evicted_prefixes": 0,
                        "prepared_steps": 0,
                        "submitted_steps": 0,
                        "free_pages": 8,
                        "reserved_pages": 0,
                        "writing_pages": 0,
                        "active_pages": 0,
                        "retiring_pages": 0,
                        "quarantined_pages": 0,
                        "exhausted_pages": 0,
                        "pending_reclamations": 0,
                        "total_request_page_refs": 0,
                        "total_prefix_page_refs": 0,
                        "total_reader_pins": 0,
                    },
                    "arena_stats": [
                        {
                            "engine_epoch": 1,
                            "pool_epoch": 2,
                            "pool_id": 1,
                            "class_id": 0,
                            "backend_domain": 1,
                            "first_page_id": 1,
                            "page_count": 8,
                            "free_pages": 8,
                            "reserved_pages": 0,
                            "writing_pages": 0,
                            "active_pages": 0,
                            "retiring_pages": 0,
                            "quarantined_pages": 0,
                            "exhausted_pages": 0,
                            "request_page_refs": 0,
                            "prefix_page_refs": 0,
                            "reader_pins": 0,
                        }
                    ],
                    "swa_activity": {
                        "status": "exposed",
                        "applicable": False,
                        "swa_retirement_certificates": 0,
                        "swa_pages_reclaimed": 0,
                        "swa_wrap_events": 0,
                    },
                    "batch_counters": {
                        name: (
                            3
                            if name in (
                                "prepare_batch_calls",
                                "complete_batch_calls",
                            )
                            else 0
                        )
                        for name in bench.runtime_support.BATCH_COUNTER_FIELDS
                    },
                    "pressure": _enabled_pressure("after_workload"),
                }
            }
        ]
    }


def _runtime_proof() -> dict[str, object]:
    return {
        "actual_attention_backend": {
            "backend_class": "FlashAttentionBackend",
            "backend_module": (
                "sglang.srt.layers.attention.flashattention_backend"
            ),
            "prefill_backend": "fa3",
            "decode_backend": "fa3",
            "has_local_attention": True,
            "attention_chunk_size": CHUNK_TOKENS,
            "page_size": PAGE_TOKENS,
            "compiled_layer_ids": [0, 1, 2, 3],
            "use_irope_layer_ids": [0, 1, 2, 3],
        },
        "effective_scheduler": {
            "max_prefill_tokens": CHUNK_TOKENS,
            "max_running_requests": 1,
            "effective_max_running_requests_per_dp": 1,
        },
    }


def _census_expectations() -> dict[str, object]:
    return {
        "expected_manifest_fingerprint": "sha256:manifest",
        "expected_runtime_binding_fingerprint": "sha256:binding",
        "expected_plan_fingerprint": "sha256:retention-ir",
        "expected_lifecycle_route": "native_session",
        "expected_cache_policy": "request_private",
        "expected_chunk_geometry": {
            "page_tokens": PAGE_TOKENS,
            "chunk_tokens": CHUNK_TOKENS,
            "blocks_per_epoch": CHUNK_TOKENS // PAGE_TOKENS,
        },
        "expected_engine_args": {
            "page_size": PAGE_TOKENS,
            "chunked_prefill_size": CHUNK_TOKENS,
            "max_prefill_tokens": CHUNK_TOKENS,
            "max_running_requests": 1,
        },
    }


def _enabled_pressure(stage: str) -> dict[str, object]:
    telemetry = PressureTelemetry(
        (
            PressureClassDescriptor(
                0,
                "chunked_attention",
                PAGE_TOKENS,
                8,
                128 * 4,
            ),
        )
    )
    empty = ArenaPressureSample(0, 8, 8, 0, 0, 0, 0, 0, 0)
    telemetry.sample(
        "runtime_initialized",
        active_requests=0,
        arenas=(empty,),
        request_reachable_unique_pages={0: 0},
        semantic_live_tokens={0: 0},
    )
    if stage != "after_load":
        active = ArenaPressureSample(0, 8, 4, 0, 0, 4, 0, 0, 0)
        telemetry.sample(
            "step_completed",
            active_requests=1,
            arenas=(active,),
            request_reachable_unique_pages={0: 4},
            semantic_live_tokens={0: CHUNK_TOKENS - 1},
        )
        telemetry.sample(
            "request_released",
            active_requests=0,
            arenas=(empty,),
            request_reachable_unique_pages={0: 0},
            semantic_live_tokens={0: 0},
        )
    return telemetry.report()


def test_manager_census_requires_current_wire_and_drained_state() -> None:
    census = bench.manager_census(
        _manager_info(),
        "after_workload",
        **_census_expectations(),
    )
    assert set(census) == {
        "stage",
        "lifecycle_route",
        "cache_policy",
        "identities",
        "manager_stats",
        "arena_stats",
        "batch_counters",
        "pressure",
        "runtime_proof",
    }
    assert census["stage"] == "after_workload"
    assert census["lifecycle_route"] == "native_session"
    assert census["cache_policy"] == "request_private"
    assert census["identities"][0]["page_tokens"] == PAGE_TOKENS
    assert census["manager_stats"]["active_requests"] == 0
    assert census["arena_stats"][0]["free_pages"] == 8
    assert census["pressure"] == _enabled_pressure("after_workload")
    assert census["pressure"]["enabled"] is True
    assert census["pressure"]["sample_count"] == 3
    assert census["pressure"]["max_active_requests"] == 1
    assert census["pressure"]["event_counts"] == {
        "request_released": 1,
        "runtime_initialized": 1,
        "step_completed": 1,
    }
    assert census["runtime_proof"] == _runtime_proof()
    assert "execution_topology" not in census
    assert "chunk_geometry" not in census

    with pytest.raises(RuntimeError, match="wire-version"):
        bench.manager_census(
            _manager_info(wire_version=10),
            "after_workload",
            **_census_expectations(),
        )

    manifest_drift = _manager_info()
    manifest_drift["internal_states"][0]["orbitkv_manager"][
        "runtime_manifest_fingerprint"
    ] = "sha256:other-manifest"
    with pytest.raises(RuntimeError, match="manifest fingerprint"):
        bench.manager_census(
            manifest_drift,
            "after_workload",
            **_census_expectations(),
        )

    binding_drift = _manager_info()
    binding_drift["internal_states"][0]["orbitkv_manager"][
        "runtime_binding_fingerprint"
    ] = "sha256:other-binding"
    with pytest.raises(RuntimeError, match="binding fingerprint"):
        bench.manager_census(
            binding_drift,
            "after_workload",
            **_census_expectations(),
        )

    disabled = _manager_info()
    disabled["internal_states"][0]["orbitkv_manager"]["pressure"] = {
        "schema": "orbitkv.runtime-pressure.v1",
        "enabled": False,
        "mode": "event_driven_high_water",
        "sample_count": 0,
    }
    with pytest.raises(RuntimeError, match="pressure telemetry is not enabled"):
        bench.manager_census(
            disabled,
            "after_workload",
            **_census_expectations(),
        )

    unknown = _manager_info()
    unknown["internal_states"][0]["orbitkv_manager"]["pressure"][
        "unexpected"
    ] = True
    with pytest.raises(RuntimeError, match="pressure schema changed"):
        bench.manager_census(
            unknown,
            "after_workload",
            **_census_expectations(),
        )

    inactive = _manager_info()
    inactive["internal_states"][0]["orbitkv_manager"]["pressure"] = (
        _enabled_pressure("after_load")
    )
    with pytest.raises(RuntimeError, match="pressure request census"):
        bench.manager_census(
            inactive,
            "after_workload",
            **_census_expectations(),
        )

    drifted = _manager_info()
    drifted["internal_states"][0]["orbitkv_manager"][
        "plan_fingerprint"
    ] = "sha256:other-plan"
    with pytest.raises(RuntimeError, match="plan fingerprint|plan_fingerprint"):
        bench.manager_census(
            drifted,
            "after_workload",
            **_census_expectations(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("lifecycle_route", None, "lifecycle route"),
        ("lifecycle_route", "canonical_manager", "lifecycle route"),
        ("cache_policy", None, "cache policy"),
        ("cache_policy", "shared_prefix", "cache policy"),
    ),
)
def test_manager_census_requires_native_request_private_policy(
    field: str, value: str | None, message: str
) -> None:
    info = _manager_info()
    manager = info["internal_states"][0]["orbitkv_manager"]
    if value is None:
        manager.pop(field)
    else:
        manager[field] = value

    with pytest.raises(RuntimeError, match=message):
        bench.manager_census(
            info,
            "after_workload",
            **_census_expectations(),
        )


@pytest.mark.parametrize(
    ("proof_section", "field", "value", "message"),
    (
        (
            "actual_attention_backend",
            "attention_chunk_size",
            CHUNK_TOKENS * 2,
            "attention chunk geometry",
        ),
        (
            "actual_attention_backend",
            "page_size",
            PAGE_TOKENS * 2,
            "attention page geometry",
        ),
        (
            "effective_scheduler",
            "max_prefill_tokens",
            CHUNK_TOKENS * 2,
            "scheduler prefill geometry",
        ),
        (
            "effective_scheduler",
            "max_running_requests",
            2,
            "scheduler concurrency geometry",
        ),
        (
            "effective_scheduler",
            "effective_max_running_requests_per_dp",
            2,
            "scheduler concurrency geometry",
        ),
    ),
)
def test_manager_census_rejects_runtime_proof_geometry_drift(
    proof_section: str, field: str, value: int, message: str
) -> None:
    info = _manager_info()
    proof = info["internal_states"][0]["orbitkv_manager"][
        "runtime_proof"
    ]
    proof[proof_section][field] = value

    with pytest.raises(RuntimeError, match=message):
        bench.manager_census(
            info,
            "after_workload",
            **_census_expectations(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("chunked_prefill_size", CHUNK_TOKENS * 2, "attention chunk geometry"),
        ("page_size", PAGE_TOKENS * 2, "attention page geometry"),
        ("max_prefill_tokens", CHUNK_TOKENS * 2, "scheduler prefill geometry"),
        ("max_running_requests", 2, "scheduler concurrency geometry"),
    ),
)
def test_manager_census_rejects_engine_argument_geometry_drift(
    field: str, value: int, message: str
) -> None:
    expected = _census_expectations()
    engine_args = dict(expected["expected_engine_args"])
    engine_args[field] = value
    expected["expected_engine_args"] = engine_args

    with pytest.raises(RuntimeError, match=message):
        bench.manager_census(
            _manager_info(),
            "after_workload",
            **expected,
        )


@pytest.mark.parametrize(
    ("proof_section", "proof_field", "engine_field", "value", "message"),
    (
        (
            "actual_attention_backend",
            "attention_chunk_size",
            "chunked_prefill_size",
            CHUNK_TOKENS * 2,
            "attention chunk geometry",
        ),
        (
            "actual_attention_backend",
            "page_size",
            "page_size",
            PAGE_TOKENS * 2,
            "attention page geometry",
        ),
        (
            "effective_scheduler",
            "max_prefill_tokens",
            "max_prefill_tokens",
            CHUNK_TOKENS * 2,
            "scheduler prefill geometry",
        ),
    ),
)
def test_manager_census_rejects_matching_proof_and_engine_drift_from_admission(
    proof_section: str,
    proof_field: str,
    engine_field: str,
    value: int,
    message: str,
) -> None:
    info = _manager_info()
    proof = info["internal_states"][0]["orbitkv_manager"][
        "runtime_proof"
    ]
    proof[proof_section][proof_field] = value
    expected = _census_expectations()
    engine_args = dict(expected["expected_engine_args"])
    engine_args[engine_field] = value
    expected["expected_engine_args"] = engine_args

    with pytest.raises(RuntimeError, match=message):
        bench.manager_census(
            info, "after_workload", **expected
        )


def test_manager_census_rejects_arena_page_geometry_drift() -> None:
    info = _manager_info()
    info["internal_states"][0]["orbitkv_manager"]["identities"][0][
        "page_tokens"
    ] = PAGE_TOKENS * 2

    with pytest.raises(RuntimeError, match="arena page geometry"):
        bench.manager_census(
            info,
            "after_workload",
            **_census_expectations(),
        )


def test_stock_census_requires_complete_orbitkv_absence() -> None:
    assert (
        bench.stock_census_absent(
            {"internal_states": [{"scheduler": {}}]}, "after_load"
        )
        is None
    )
    with pytest.raises(RuntimeError, match="stock.*OrbitKV|loaded OrbitKV"):
        bench.stock_census_absent(_manager_info(), "after_load")


@pytest.mark.parametrize(
    "info",
    (
        {},
        {"internal_states": None},
        {"internal_states": {}},
        {"internal_states": []},
        {"internal_states": [{}, {}]},
        {"internal_states": [None]},
    ),
)
def test_stock_census_requires_exactly_one_internal_state(info) -> None:
    with pytest.raises(RuntimeError, match="exactly one scheduler internal state"):
        bench.stock_census_absent(info, "after_load")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda manager: manager.__setitem__("runtime_proof", None),
            "runtime proof is malformed",
        ),
        (
            lambda manager: manager["runtime_proof"].__setitem__(
                "unexpected", True
            ),
            "runtime proof is malformed",
        ),
        (
            lambda manager: manager["runtime_proof"][
                "actual_attention_backend"
            ].pop("backend_module"),
            "backend proof is malformed",
        ),
        (
            lambda manager: manager["runtime_proof"][
                "effective_scheduler"
            ].__setitem__("max_prefill_tokens", None),
            "positive integer",
        ),
        (
            lambda manager: manager["runtime_proof"][
                "effective_scheduler"
            ].__setitem__("max_running_requests", True),
            "positive integer",
        ),
        (
            lambda manager: manager["runtime_proof"][
                "actual_attention_backend"
            ].__setitem__("compiled_layer_ids", [0, 0]),
            "layer execution proof changed",
        ),
        (
            lambda manager: manager["runtime_proof"][
                "actual_attention_backend"
            ].__setitem__("use_irope_layer_ids", [0, 1, 2]),
            "layer execution proof changed",
        ),
    ),
)
def test_manager_census_rejects_noncanonical_runtime_proof(
    mutate, message
) -> None:
    info = _manager_info()
    manager = info["internal_states"][0]["orbitkv_manager"]
    mutate(manager)

    with pytest.raises(RuntimeError, match=message):
        bench.manager_census(
            info,
            "after_workload",
            **_census_expectations(),
        )


def test_all_four_claim_gates_are_fixed_false_with_reasons() -> None:
    claims = bench.claim_gates()
    assert set(claims) == CLAIM_KEYS
    for gate in claims.values():
        assert set(gate) == {"qualified", "reasons"}
        assert gate["qualified"] is False
        assert isinstance(gate["reasons"], list)
        assert gate["reasons"]
        assert all(
            isinstance(reason, str) and reason.strip()
            for reason in gate["reasons"]
        )


def _runtime_state(
    mode: str,
    engine_args: dict[str, object],
    plan_fingerprint: str,
    stage: str = "after_workload",
    *,
    manifest_fingerprint: str = "sha256:manifest",
    runtime_binding_fingerprint: str = "sha256:binding",
) -> dict[str, object]:
    readback_names = (
        "page_size",
        "max_total_tokens",
        "attention_backend",
        "dtype",
        "kv_cache_dtype",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "prefill_max_requests",
        "max_running_requests",
        "enable_dynamic_chunking",
        "enable_mixed_chunk",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "disable_cuda_graph",
        "enable_torch_compile",
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
    )
    state = {name: engine_args[name] for name in readback_names}
    state["effective_max_running_requests_per_dp"] = engine_args[
        "max_running_requests"
    ]
    state["memory_usage"] = {"token_capacity": engine_args["max_total_tokens"]}
    if mode == "manager":
        manager = copy.deepcopy(
            _manager_info()["internal_states"][0]["orbitkv_manager"]
        )
        manager["runtime_proof"]["actual_attention_backend"].update(
            attention_chunk_size=engine_args["chunked_prefill_size"],
            page_size=engine_args["page_size"],
        )
        manager["runtime_proof"]["effective_scheduler"].update(
            max_prefill_tokens=engine_args["max_prefill_tokens"],
            max_running_requests=engine_args["max_running_requests"],
            effective_max_running_requests_per_dp=engine_args[
                "max_running_requests"
            ],
        )
        manager["plan_fingerprint"] = plan_fingerprint
        manager["runtime_manifest_fingerprint"] = manifest_fingerprint
        manager["runtime_binding_fingerprint"] = runtime_binding_fingerprint
        manager["pressure"] = _enabled_pressure(stage)
        state["orbitkv_manager"] = manager
    return state


def _assert_chunked_record_shape(
    record: dict[str, object], mode: str, case: str, *, failed: bool = False
) -> None:
    _assert_chunked_record_shape_helper(bench, record, mode, case, failed=failed)
    return
    assert set(record) == bench.TOP_LEVEL_KEYS
    assert record["schema"] == bench.RECORD_SCHEMA
    assert record["mode"] == mode
    assert isinstance(record["started_at_utc"], str)
    assert isinstance(record["command"], list)
    assert record["command_sha256"] == bench.canonical_digest(record["command"])
    assert record["environment_sha256"] == bench.canonical_digest(
        record["environment"]
    )
    assert record["source_identity_sha256"] == bench.canonical_digest(
        record["source_identity"]
    )
    assert record["checkpoint_identity_sha256"] == bench.canonical_digest(
        record["checkpoint"]
    )

    assert set(record["source_identity"]) == {
        "contract",
        "pinned_contract",
        "direct_source",
        "sglang_python_sha256",
        "harness",
        "checkpoint_identity_helper_sha256",
        "adapter",
        "build_tool",
        "library",
    }
    assert set(record["source_identity"]["direct_source"]) == {
        "kind",
        "owner",
        "takeover_points",
    }
    assert set(record["source_identity"]["harness"]) == {"path", "sha256"}
    assert set(record["checkpoint"]) == {"identity", "config"}
    assert set(record["checkpoint"]["identity"]) == {
        "weight_bytes",
        "indexed_weights_complete",
        "config_sha256",
    }
    assert set(record["checkpoint"]["config"]) == {
        "architectures",
        "num_hidden_layers",
        "vocab_size",
        "max_position_embeddings",
        "attention_chunk_size",
        "control_token_ids",
    }
    assert set(record["runtime_identity"]) == {
        "run_id",
        "runtime_proof",
        "python_executable",
        "python_version",
        "platform",
        "sglang_version",
        "sglang_package",
        "kv_layout",
        "attention_backend",
        "dtype",
        "kv_cache_dtype",
        "execution",
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
        "deterministic_inference",
        "sampling_backend",
    }
    run_id = record["runtime_identity"]["run_id"]
    parsed_run_id = uuid.UUID(run_id)
    assert parsed_run_id.version == 4
    assert str(parsed_run_id) == run_id
    assert record["runtime_identity"]["runtime_proof"] == (
        _runtime_proof() if mode == "manager" else None
    )

    engine_keys = {
        "model_path",
        "load_format",
        "dtype",
        "kv_cache_dtype",
        "skip_tokenizer_init",
        "trust_remote_code",
        "context_length",
        "page_size",
        "attention_backend",
        "disable_hybrid_swa_memory",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "disable_cuda_graph",
        "enable_torch_compile",
        "enable_deterministic_inference",
        "sampling_backend",
        "chunked_prefill_size",
        "prefill_max_requests",
        "max_prefill_tokens",
        "enable_dynamic_chunking",
        "enable_mixed_chunk",
        "max_running_requests",
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
        "enable_dp_attention",
        "speculative_algorithm",
        "disaggregation_mode",
        "enable_hierarchical_cache",
        "enable_streaming_session",
        "enable_unified_memory",
        "enable_pdmux",
        "enable_lmcache",
        "enable_flexkv",
        "enable_session_radix_cache",
        "enable_hisparse",
        "enable_page_major_kv_layout",
        "random_seed",
        "log_level",
        "max_total_tokens",
    }
    if mode == "manager":
        engine_keys.add("radix_cache_backend")
    assert set(record["engine_args"]) == engine_keys
    assert record["sampling_params"] == {
        "temperature": 0,
        "max_new_tokens": 97,
        "min_new_tokens": 97,
        "ignore_eos": True,
        "sampling_seed": 20260828,
    }
    assert set(record["workload"]) == {
        "case",
        "requests",
        "prompt_tokens",
        "decode_tokens",
        "materialized_kv_tokens_per_request",
        "iterations",
        "seed",
        "fresh_prompts",
        "input_token_digest_sha256",
        "input_token_digests_by_iteration_sha256",
        "chunk_geometry",
    }
    assert record["workload"]["case"] == case
    assert set(record["workload"]["chunk_geometry"]) == {
        "page_tokens",
        "chunk_tokens",
        "blocks_per_epoch",
        "chunk_epoch_count_per_request",
        "epoch_end_crossings_per_request",
    }
    assert set(record["timings"]) == {
        "load_seconds",
        "total_seconds",
        "iteration_seconds",
    }
    assert len(record["timings"]["iteration_seconds"]) == (
        0 if failed else 3
    )
    assert set(record["outputs"]) == {"iterations", "aggregate_sha256"}
    for index, iteration in enumerate(record["outputs"]["iterations"]):
        assert iteration["iteration"] == index
        assert set(iteration) == {"iteration", "requests"}
        assert len(iteration["requests"]) == 1
        assert set(iteration["requests"][0]) == {
            "request_id",
            "input_sha256",
            "output_ids",
            "output_sha256",
            "cached_tokens",
        }
    assert record["outputs"]["aggregate_sha256"] == bench.canonical_digest(
        record["outputs"]["iterations"]
    )
    expected_capacity = 256 if case == "roomy" else CHUNK_TOKENS
    if failed:
        assert record["outputs"] == {
            "iterations": [],
            "aggregate_sha256": bench.canonical_digest([]),
        }
        assert set(record["server_capacity"]) == {
            "status",
            "requested_tokens",
            "available_tokens",
            "failure",
        }
        assert record["server_capacity"]["status"] == "failed"
        assert record["server_capacity"]["requested_tokens"] == expected_capacity
        assert set(record["server_capacity"]["failure"]) == {"type", "message"}
        assert record["server_capacity"]["failure"]["type"] == (
            "capacity_exhausted"
        )
    else:
        assert record["server_capacity"] == {
            "status": "observed",
            "requested_tokens": expected_capacity,
            "available_tokens": expected_capacity,
            "failure": None,
        }
    assert [set(item) for item in record["gpu_snapshots"]] == [
        {"stage", "time_ns", "gpus"}
    ] * len(record["gpu_snapshots"])
    stages = [item["stage"] for item in record["gpu_snapshots"]]
    expected_stages = [
        "before_engine",
        "after_load",
        "after_workload",
        "after_shutdown",
    ]
    if failed:
        assert stages[0] == "before_engine"
        assert stages[-1] == "after_shutdown"
        assert len(stages) == len(set(stages))
        assert [expected_stages.index(stage) for stage in stages] == sorted(
            expected_stages.index(stage) for stage in stages
        )
    else:
        assert stages == expected_stages
    assert set(record["claims"]) == CLAIM_KEYS
    assert all(
        set(gate) == {"qualified", "reasons"}
        for gate in record["claims"].values()
    )


@pytest.mark.parametrize(
    ("mode", "generation_error", "returns_record"),
    (
        pytest.param("manager", None, True, id="manager-roomy-success"),
        pytest.param("stock", None, True, id="stock-exact-floor-success"),
        pytest.param(
            "stock",
            "Decode out of memory. Try to allocate 17 tokens",
            True,
            id="stock-exact-floor-capacity-exhausted",
        ),
        pytest.param(
            "manager",
            "Decode out of memory. Try to allocate 17 tokens",
            False,
            id="manager-roomy-capacity-reraised",
        ),
        pytest.param(
            "stock",
            "arbitrary RuntimeError",
            False,
            id="stock-non-kv-error-reraised",
        ),
    ),
)
def test_run_uses_mock_engine_and_emits_frozen_host_observation(
    mode: str,
    generation_error: str | None,
    returns_record: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    args = _run_arguments(mode, inputs)
    paths = bench.validate_arguments(args)
    root = paths["sglang_root"]
    assert isinstance(root, Path)

    manifest = _chunked_manifest()
    retention = manifest["source"]["program"]
    plan_fingerprint = "sha256:" + hashlib.sha256(
        _canonical_json(retention)
    ).hexdigest()
    admission = _admission(Path(inputs["manifest"]))
    manifest_fingerprint = admission["runtime_manifest"]["fingerprint"]
    runtime_binding_fingerprint = admission["runtime_binding"]["fingerprint"]
    engine_events: list[object] = []
    generated: list[dict[str, object]] = []
    engine_kwargs: list[dict[str, object]] = []
    gpu_stages: list[str] = []
    loaded_libraries: list[Path] = []
    source_modes: list[str] = []
    entrypoint_checks: list[None] = []
    configured_environments: list[dict[str, str]] = []
    readback_count = [0]
    generation_exception = (
        RuntimeError(generation_error) if generation_error is not None else None
    )

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            engine_kwargs.append(dict(kwargs))
            engine_events.append("init")

        def __enter__(self):
            engine_events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            engine_events.append("exit")

        def get_server_info(self) -> dict[str, object]:
            engine_events.append("readback")
            stage = "after_load" if readback_count[0] == 0 else "after_workload"
            readback_count[0] += 1
            return {
                "internal_states": [
                    _runtime_state(
                        mode,
                        engine_kwargs[0],
                        plan_fingerprint,
                        stage,
                        manifest_fingerprint=manifest_fingerprint,
                        runtime_binding_fingerprint=runtime_binding_fingerprint,
                    )
                ]
            }

        def generate(self, **kwargs: object) -> dict[str, object]:
            copied = copy.deepcopy(kwargs)
            generated.append(copied)
            rid = copied["rid"]
            engine_events.append(f"generate:{rid[0]}")
            if generation_exception is not None:
                raise generation_exception
            return {
                "output_ids": list(range(args.decode_tokens)),
                "meta_info": {"id": rid[0], "cached_tokens": 0},
            }

    fake_sglang = ModuleType("sglang")
    fake_sglang.Engine = FakeEngine
    fake_sglang.__version__ = bench.pinned.pinned_source_contract()[
        "release"
    ].removeprefix("v")
    fake_sglang.__file__ = str(root / "python/sglang/__init__.py")
    fake_sglang.__path__ = [str(root / "python/sglang")]
    fake_srt = ModuleType("sglang.srt")
    fake_srt.__path__ = []
    fake_environ = ModuleType("sglang.srt.environ")
    fake_environ.envs = SimpleNamespace(
        SGLANG_USE_HND_KVCACHE=SimpleNamespace(get=lambda: False)
    )
    fake_sglang.srt = fake_srt
    fake_srt.environ = fake_environ

    for name in tuple(sys.modules):
        if (
            name == "sglang"
            or name.startswith("sglang.")
            or name == "torch"
            or name.startswith("torch.")
        ):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ORBITKV_STALE", "must-be-cleared")

    original_configure = bench.configure_environment
    canonical_source_identity = bench.direct_source_identity

    def configure_and_install(run_args, resolved):
        environment = original_configure(run_args, resolved)
        configured_environments.append(environment)
        monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
        monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
        monkeypatch.setitem(sys.modules, "sglang.srt.environ", fake_environ)
        return environment

    monkeypatch.setattr(bench, "configure_environment", configure_and_install)
    monkeypatch.setattr(
        bench,
        "verify_chunked_source",
        lambda _root, _selected: {
            "contract": bench.pinned.pinned_source_contract()
        },
    )
    def source_identity(selected_mode: str) -> dict[str, object]:
        source_modes.append(selected_mode)
        return canonical_source_identity(selected_mode)

    monkeypatch.setattr(bench, "direct_source_identity", source_identity)
    monkeypatch.setattr(
        bench.runtime_support,
        "reject_legacy_manager_entrypoint",
        lambda: entrypoint_checks.append(None),
    )
    monkeypatch.setattr(
        bench.runtime_support, "adapter_identity", lambda: {"files": []}
    )
    monkeypatch.setattr(
        bench.runtime_support,
        "build_tool_identity",
        lambda: {"path": "/fake/ninja", "version": "fake"},
    )
    monkeypatch.setattr(
        bench,
        "sha256_file",
        lambda path: hashlib.sha256(str(path).encode()).hexdigest(),
    )
    monkeypatch.setattr(
        bench,
        "_checkpoint_config",
        lambda _model: (
            {
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 4,
                "vocab_size": 1024,
                "max_position_embeddings": 1024,
                "attention_chunk_size": CHUNK_TOKENS,
                "control_token_ids": {},
            },
            {
                "weight_bytes": 1,
                "indexed_weights_complete": True,
                "config_sha256": "config",
            },
        ),
    )

    def fake_gpu_snapshot(stage: str) -> dict[str, object]:
        gpu_stages.append(stage)
        return {
            "stage": stage,
            "time_ns": len(gpu_stages),
            "gpus": [{"name": "host-only fake"}],
        }

    monkeypatch.setattr(bench, "gpu_snapshot", fake_gpu_snapshot)
    monkeypatch.setattr(
        bench,
        "LoadedLibrary",
        lambda path: loaded_libraries.append(Path(path)),
    )

    imported_sglang_names: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *import_args: object, **import_kwargs: object):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("host-only chunked benchmark imported torch")
        if name == "sglang" or name.startswith("sglang."):
            imported_sglang_names.append(name)
            assert name in sys.modules, "attempted to import a real SGLang module"
        return original_import(name, *import_args, **import_kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    if not returns_record:
        with pytest.raises(RuntimeError) as captured:
            bench.run(args, paths)
        assert captured.value is generation_exception
        assert len(generated) == 1
        assert engine_events[-1] == "exit"
        assert not any(
            name == "torch" or name.startswith("torch.")
            for name in sys.modules
        )
        return

    record = bench.run(args, paths)
    capacity_failed = generation_error is not None

    _assert_chunked_record_shape(
        record, mode, args.case, failed=capacity_failed
    )
    assert source_modes == [mode]
    assert entrypoint_checks == [None]
    assert record["source_identity"]["direct_source"] == (
        canonical_source_identity(mode)
    )
    assert len(configured_environments) == 1
    assert record["environment"] == configured_environments[0]
    assert "SGLANG_PLUGINS" not in record["environment"]
    if mode == "manager":
        assert record["environment"]["ORBITKV_PRESSURE_TELEMETRY"] == "1"
    assert imported_sglang_names == ["sglang", "sglang.srt.environ"]
    assert not any(
        name == "torch" or name.startswith("torch.") for name in sys.modules
    )
    assert engine_kwargs == [record["engine_args"]]
    expected_engine_events = (
        [
            "init",
            "enter",
            "readback",
            "generate:orbitkv-chunked-20260828-0-0",
            "exit",
        ]
        if capacity_failed
        else [
            "init",
            "enter",
            "readback",
            "generate:orbitkv-chunked-20260828-0-0",
            "generate:orbitkv-chunked-20260828-1-0",
            "generate:orbitkv-chunked-20260828-2-0",
            "readback",
            "readback",
            "exit",
        ]
    )
    assert engine_events == expected_engine_events
    assert gpu_stages == (
        ["before_engine", "after_load", "after_shutdown"]
        if capacity_failed
        else [
            "before_engine",
            "after_load",
            "after_workload",
            "after_shutdown",
        ]
    )
    assert len(generated) == (1 if capacity_failed else 3)
    assert all(set(call) == {"input_ids", "rid", "sampling_params"} for call in generated)
    input_digests = [
        bench.canonical_digest(call["input_ids"][0]) for call in generated
    ]
    request_ids = [call["rid"][0] for call in generated]
    assert len(set(input_digests)) == 1
    assert len(set(request_ids)) == len(request_ids)
    expected_input_digests = [
        [input_digests[0]],
        [input_digests[0]],
        [input_digests[0]],
    ]
    assert record["workload"]["input_token_digests_by_iteration_sha256"] == (
        expected_input_digests
    )
    requests = [
        iteration["requests"][0] for iteration in record["outputs"]["iterations"]
    ]
    if capacity_failed:
        assert requests == []
        assert record["timings"]["iteration_seconds"] == []
        assert record["server_capacity"]["failure"] == {
            "type": "capacity_exhausted",
            "message": generation_error,
        }
    else:
        assert [item["request_id"] for item in requests] == request_ids
        assert all(item["cached_tokens"] == 0 for item in requests)
        assert len({tuple(item["output_ids"]) for item in requests}) == 1
        assert len({item["output_sha256"] for item in requests}) == 1

    if mode == "manager":
        assert loaded_libraries == [Path(inputs["library"]).resolve()]
        assert set(record["runtime_manifest"]) == {
            "artifact",
            "schema",
            "version",
            "manifest_fingerprint",
            "retention_program_fingerprint",
            "layout_plan_fingerprint",
            "execution_signature_fingerprint",
            "chunk_geometry",
        }
        assert set(record["runtime_binding"]) == {
            "schema",
            "version",
            "fingerprint",
            "manifest_fingerprint",
            "target",
            "admission_profile",
            "target_contract_fingerprint",
            "required_wire_version",
            "execution_topology",
            "execution_signature",
        }
        assert record["runtime_binding"]["execution_topology"] == TOPOLOGY
        assert set(record["manager"]) == {"wire_version", "snapshots"}
        assert record["manager"]["wire_version"] == bench.WIRE_VERSION == 14
        assert [
            snapshot["stage"] for snapshot in record["manager"]["snapshots"]
        ] == ["after_load", "after_workload", "final"]
        assert all(
            set(snapshot)
            == {
                "stage",
                "lifecycle_route",
                "cache_policy",
                "identities",
                "manager_stats",
                "arena_stats",
                "batch_counters",
                "pressure",
                "runtime_proof",
            }
            for snapshot in record["manager"]["snapshots"]
        )
        assert all(
            snapshot["lifecycle_route"] == "native_session"
            and snapshot["cache_policy"] == "request_private"
            for snapshot in record["manager"]["snapshots"]
        )
        assert all(
            snapshot["runtime_proof"] == _runtime_proof()
            for snapshot in record["manager"]["snapshots"]
        )
        after_load_pressure = record["manager"]["snapshots"][0]["pressure"]
        after_workload_pressure = record["manager"]["snapshots"][1][
            "pressure"
        ]
        assert after_load_pressure["enabled"] is True
        assert after_load_pressure["sample_count"] == 1
        assert after_load_pressure["event_counts"] == {
            "runtime_initialized": 1
        }
        assert after_workload_pressure["enabled"] is True
        assert after_workload_pressure["sample_count"] == 3
        assert after_workload_pressure["max_active_requests"] == 1
        assert after_workload_pressure["event_counts"]["step_completed"] == 1
        assert after_workload_pressure["event_counts"]["request_released"] == 1
        assert (
            record["source_identity"]["library"]["wire_version"]
            == bench.WIRE_VERSION
        )
        assert record["environment"]["ORBITKV_RUNTIME_MANIFEST"] == str(
            Path(inputs["manifest"]).resolve()
        )
    else:
        assert loaded_libraries == []
        assert record["runtime_manifest"] is None
        assert record["runtime_binding"] is None
        assert record["manager"] is None
        assert record["source_identity"]["library"] is None
        assert not any(
            name.startswith("ORBITKV_") for name in record["environment"]
        )
