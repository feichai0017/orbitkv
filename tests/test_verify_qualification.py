from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/verify_qualification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_qualification", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
verifier: ModuleType = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
from orbitkv_sglang import pinned as live_pinned
from engine_e2e_verifier_contract import (
    layout_fingerprint_from_manager_input,
    manager_input_from_attention_source,
)
import qualification_native
from qualification_source import direct_source_identity

PAGE_TOKENS = 16
CHUNK_TOKENS = 64
ITERATIONS = 3
ZERO_SHA = "0" * 64
FINGERPRINT = "sha256:" + ZERO_SHA
WIRE_VERSION = verifier.CURRENT_WIRE_VERSION
_PINNED_SOURCE_CONTRACT = live_pinned.pinned_source_contract()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _claims() -> dict[str, dict[str, object]]:
    return {
        name: {"qualified": False, "reasons": ["not derived by runner"]}
        for name in verifier.GATE_KEYS
    }


def _binding(manifest_fingerprint: str) -> dict[str, object]:
    signature: dict[str, object] = {
        "schema": "orbitkv.execution-signature",
        "version": 1,
        "fingerprint": "",
        "manifest_schema": "orbitkv.runtime-manifest",
        "manifest_version": 3,
        "manifest_fingerprint": manifest_fingerprint,
        "page_tokens": PAGE_TOKENS,
        "token_classes": [
            {
                "name": "chunked_attention",
                "layers": [0, 1],
                "bytes_per_token_per_layer": 128,
                "address": {
                    "kind": "resettable_arena",
                    "blocks_per_epoch": 4,
                },
                "retirement": {
                    "kind": "epoch_end",
                    "blocks_per_epoch": 4,
                },
                "minimum_slots_per_request": 4,
            }
        ],
        "token_states": [
            {
                "name": "chunked_attention",
                "layers": [0, 1],
                "backend": {
                    "kind": "token_slots",
                    "storage": "token_kv",
                    "components": [],
                    "bytes_per_token_per_layer": 128,
                    "page_bytes_per_layer": 2048,
                    "retention": "chunked",
                    "window_tokens": None,
                    "token_relocatable": True,
                },
            }
        ],
        "fixed_states": [],
    }
    signature["fingerprint"] = verifier._binding_digest(signature)
    binding: dict[str, object] = {
        "schema": "orbitkv.runtime-binding",
        "version": 1,
        "fingerprint": "",
        "manifest_fingerprint": manifest_fingerprint,
        "target": {
            "id": "sglang",
            "contract_version": verifier.CURRENT_TARGET_CONTRACT_VERSION,
        },
        "admission_profile": {
            "id": "eager-single-device-bf16-nhd",
            "version": 1,
        },
        "target_contract_fingerprint": verifier.CURRENT_TARGET_FINGERPRINT,
        "required_wire_version": WIRE_VERSION,
        "execution_topology": verifier.EXACT_TOPOLOGY,
        "execution_signature": signature,
    }
    binding["fingerprint"] = verifier._binding_digest(binding)
    return binding


def _manager_snapshot(stage: str, counters: dict[str, int]) -> dict[str, object]:
    identity = {
        "engine_epoch": 1,
        "pool_epoch": 2,
        "pool_id": 3,
        "class_id": 0,
        "backend_domain": 4,
        "page_count": 8,
        "page_tokens": PAGE_TOKENS,
        "backend_base_index": 0,
        "first_page_id": 1,
    }
    arena = {name: 0 for name in verifier.ARENA_KEYS}
    arena.update(identity)
    arena.pop("page_tokens")
    arena.pop("backend_base_index")
    arena.update(page_count=8, free_pages=8)
    stats = {name: 0 for name in verifier.MANAGER_STATS_KEYS}
    stats["free_pages"] = 8
    return {
        "stage": stage,
        "lifecycle_route": "native_session",
        "cache_policy": "request_private",
        "identities": [identity],
        "manager_stats": stats,
        "arena_stats": [arena],
        "batch_counters": counters,
        "pressure": _pressure(stage),
    }


def _pressure(stage: str) -> dict[str, object]:
    counts = {"runtime_initialized": 1}
    if stage != "after_load":
        counts.update(step_completed=3, request_released=3)
    samples = sum(counts.values())
    metrics = {
        "resident_data_bytes": 0,
        "semantic_live_bytes": 0,
        "retention_amplification_milli": None,
        "high_water_resident_data_bytes": 8192,
        "high_water_semantic_live_bytes": 4096,
        "high_water_retention_amplification_milli": 2000,
    }
    return {
        "schema": "orbitkv.runtime-pressure.v1",
        "enabled": True,
        "mode": "event_driven_high_water",
        "scope": {"state_ownership": "request_private"},
        "sample_count": samples,
        "last_event": list(counts)[-1],
        "event_counts": counts,
        "active_requests": 0,
        "max_active_requests": 0 if stage == "after_load" else 1,
        "global": dict(metrics),
        "classes": [{"class_id": 0, "name": "chunked", **metrics}],
    }


def _manager_state() -> dict[str, object]:
    initial = {name: 0 for name in verifier.COUNTER_KEYS}
    active = dict(initial)
    workload_batches = ITERATIONS * 97
    for name in (
        "prepare_batch_calls",
        "submit_batch_calls",
        "complete_batch_calls",
        "forward_events",
        "completion_values",
    ):
        active[name] = workload_batches
    active["event_queries"] = workload_batches
    active["acknowledge_reclamations_batch_calls"] = 2 * ITERATIONS
    active["materialized_page_objects"] = 4
    proof = {
        "actual_attention_backend": {
            "backend_class": "FlashAttentionBackend",
            "backend_module": "sglang.srt.layers.attention.flashattention_backend",
            "prefill_backend": "fa3",
            "decode_backend": "fa3",
            "has_local_attention": True,
            "attention_chunk_size": CHUNK_TOKENS,
            "page_size": PAGE_TOKENS,
            "compiled_layer_ids": [0, 1],
            "use_irope_layer_ids": [0, 1],
        },
        "effective_scheduler": {
            "max_prefill_tokens": CHUNK_TOKENS,
            "max_running_requests": 1,
            "effective_max_running_requests_per_dp": 1,
        },
    }
    snapshots = [
        _manager_snapshot("after_load", initial),
        _manager_snapshot("after_workload", active),
        _manager_snapshot("final", dict(active)),
    ]
    for snapshot in snapshots:
        snapshot["runtime_proof"] = copy.deepcopy(proof)
    return {
        "wire_version": WIRE_VERSION,
        "snapshots": snapshots,
    }


def _outputs(mode: str, *, iterations: int = ITERATIONS) -> dict[str, object]:
    rows = []
    for index in range(iterations):
        tokens = list(range(101, 198))
        request = {
            "request_id": f"{mode}-{index}",
            "input_sha256": "1" * 64,
            "output_ids": tokens,
            "output_sha256": verifier._canonical_digest(tokens),
            "cached_tokens": 0,
        }
        rows.append({"iteration": index, "requests": [request]})
    return {
        "iterations": rows,
        "aggregate_sha256": verifier._canonical_digest(rows),
    }


def _gpu_snapshots(*, partial: bool = False) -> list[dict[str, object]]:
    stages = ("before_engine", "after_shutdown") if partial else (
        "before_engine",
        "after_load",
        "after_workload",
        "after_shutdown",
    )
    gpu = {name: "0" for name in verifier.GPU_KEYS}
    gpu.update(index="0", name="Synthetic GPU", uuid="GPU-synthetic")
    return [
        {"stage": stage, "time_ns": index + 1, "gpus": [dict(gpu)]}
        for index, stage in enumerate(stages)
    ]


def _record(
    mode: str,
    case: str = "roomy",
    *,
    seconds: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    capacity_tokens = 256 if case == "roomy" else CHUNK_TOKENS
    failed = case == "exact-floor" and mode == "stock"
    environment = {"PATH": "/usr/bin"}
    pinned = copy.deepcopy(_PINNED_SOURCE_CONTRACT)
    source: dict[str, object] = {
        "contract": {
            "root": f"/synthetic/{mode}-sglang",
            "release": pinned["release"],
            "revision": pinned["revision"],
            "contract": pinned,
            "contract_sha256": verifier._canonical_digest(pinned),
            "reviewed_patch": (
                {
                    "status": "applied",
                    "sha256": pinned["patch_diff_sha256"],
                    "bytes": (
                        Path(
                            __file__
                        ).resolve().parents[1]
                        / pinned["patch_path"]
                    ).stat().st_size,
                }
                if mode == "manager"
                else {"status": "absent", "sha256": None, "bytes": 0}
            ),
        },
        "pinned_contract": pinned,
        "direct_source": copy.deepcopy(
            direct_source_identity(mode)
        ),
        "sglang_python_sha256": ZERO_SHA,
        "harness": {"path": "/synthetic/qualification_runner.py", "sha256": ZERO_SHA},
        "checkpoint_identity_helper_sha256": ZERO_SHA,
        "adapter": {"files": [{"path": "adapter.py", "sha256": ZERO_SHA}]},
        "build_tool": {"path": "/usr/bin/ninja", "version": "1", "sha256": ZERO_SHA},
        "library": None,
    }
    if mode == "manager":
        source["library"] = {
            "path": "/synthetic/liborbitkv_ffi.so",
            "bytes": 1,
            "sha256": ZERO_SHA,
            "wire_version": WIRE_VERSION,
        }
    checkpoint = {
        "identity": {"revision": "synthetic", "weight_bytes": 1},
        "config": {
            "architectures": ["SyntheticChunkedModel"],
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "max_position_embeddings": 4096,
            "attention_chunk_size": CHUNK_TOKENS,
            "control_token_ids": {},
        },
    }
    workload = {
        "case": case,
        "requests": 1,
        "prompt_tokens": 48,
        "decode_tokens": 97,
        "materialized_kv_tokens_per_request": 144,
        "iterations": ITERATIONS,
        "seed": 7,
        "fresh_prompts": True,
        "input_token_digest_sha256": "2" * 64,
        "input_token_digests_by_iteration_sha256": [
            ["1" * 64] for _ in range(ITERATIONS)
        ],
        "chunk_geometry": {
            "page_tokens": PAGE_TOKENS,
            "chunk_tokens": CHUNK_TOKENS,
            "blocks_per_epoch": 4,
            "chunk_epoch_count_per_request": 3,
            "epoch_end_crossings_per_request": 2,
        },
    }
    engine = {
        "attention_backend": "fa3",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "page_size": PAGE_TOKENS,
        "chunked_prefill_size": CHUNK_TOKENS,
        "prefill_max_requests": 1,
        "max_prefill_tokens": CHUNK_TOKENS,
        "max_running_requests": 1,
        "enable_dynamic_chunking": False,
        "enable_mixed_chunk": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "max_total_tokens": capacity_tokens,
    }
    manifest = None
    binding = None
    manager = None
    if mode == "manager":
        binding = _binding(FINGERPRINT)
        manifest = {
            "artifact": {
                "path": "/synthetic/runtime-manifest.json",
                "bytes": 1,
                "sha256": ZERO_SHA,
            },
            "schema": "orbitkv.runtime-manifest",
            "version": 3,
            "manifest_fingerprint": FINGERPRINT,
            "retention_program_fingerprint": FINGERPRINT,
            "layout_plan_fingerprint": FINGERPRINT,
            "execution_signature_fingerprint": binding["execution_signature"]["fingerprint"],
            "chunk_geometry": {
                "page_tokens": PAGE_TOKENS,
                "chunk_tokens": CHUNK_TOKENS,
                "blocks_per_epoch": 4,
            },
        }
        manager = _manager_state()
    output = _outputs(mode, iterations=0 if failed else ITERATIONS)
    command = [
        "python", "qualification_runner.py", "--sglang-root",
        f"/synthetic/{mode}-sglang",
    ]
    record = {
        "schema": verifier.RECORD_SCHEMA,
        "mode": mode,
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "command": command,
        "command_sha256": verifier._canonical_digest(command),
        "environment": environment,
        "environment_sha256": verifier._canonical_digest(environment),
        "source_identity": source,
        "source_identity_sha256": verifier._canonical_digest(source),
        "runtime_identity": {
            "run_id": (
                "00000000-0000-4000-8000-000000000002"
                if mode == "manager"
                else "00000000-0000-4000-8000-000000000001"
            ),
            "python_executable": "/usr/bin/python",
            "python_version": "3.11",
            "platform": "linux",
            "sglang_version": "synthetic",
            "sglang_package": (
                f"/synthetic/{mode}-sglang/python/sglang/__init__.py"
            ),
            "kv_layout": "nhd",
            "attention_backend": "fa3",
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "execution": "eager",
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 1,
            "dcp_size": 1,
            "deterministic_inference": True,
            "sampling_backend": "pytorch",
            "runtime_proof": (
                {
                    "actual_attention_backend": {
                        "backend_class": "FlashAttentionBackend",
                        "backend_module": "sglang.srt.layers.attention.flashattention_backend",
                        "prefill_backend": "fa3",
                        "decode_backend": "fa3",
                        "has_local_attention": True,
                        "attention_chunk_size": CHUNK_TOKENS,
                        "page_size": PAGE_TOKENS,
                        "compiled_layer_ids": [0, 1],
                        "use_irope_layer_ids": [0, 1],
                    },
                    "effective_scheduler": {
                        "max_prefill_tokens": CHUNK_TOKENS,
                        "max_running_requests": 1,
                        "effective_max_running_requests_per_dp": 1,
                    },
                }
                if mode == "manager"
                else None
            ),
        },
        "model": "/synthetic/model",
        "checkpoint": checkpoint,
        "checkpoint_identity_sha256": verifier._canonical_digest(checkpoint),
        "runtime_manifest": manifest,
        "runtime_binding": binding,
        "engine_args": engine,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 97,
            "min_new_tokens": 97,
            "ignore_eos": True,
            "sampling_seed": 7,
        },
        "workload": workload,
        "timings": {
            "load_seconds": 0.1,
            "total_seconds": sum(seconds) if not failed else 0.1,
            "iteration_seconds": list(seconds) if not failed else [],
        },
        "outputs": output,
        "server_capacity": (
            {
                "status": "failed",
                "requested_tokens": capacity_tokens,
                "available_tokens": None,
                "failure": {
                    "type": "capacity_exhausted",
                "message": "KV cache pool is full",
                },
            }
            if failed
            else {
                "status": "observed",
                "requested_tokens": capacity_tokens,
                "available_tokens": capacity_tokens,
                "failure": None,
            }
        ),
        "manager": manager,
        "gpu_snapshots": _gpu_snapshots(partial=failed),
        "claims": _claims(),
    }
    if mode == "manager":
        record["engine_args"]["radix_cache_backend"] = "orbitkv"
        record["environment"]["ORBITKV_SGLANG_ROOT"] = (
            "/synthetic/manager-sglang"
        )
        record["environment_sha256"] = verifier._canonical_digest(
            record["environment"]
        )
    return record


def _reseal_outputs(record: dict[str, Any]) -> None:
    for iteration in record["outputs"]["iterations"]:
        for request in iteration["requests"]:
            request["output_sha256"] = verifier._canonical_digest(
                request["output_ids"]
            )
    record["outputs"]["aggregate_sha256"] = verifier._canonical_digest(
        record["outputs"]["iterations"]
    )


def _write_pair(
    root: Path, name: str, stock: dict[str, Any], manager: dict[str, Any]
) -> Path:
    pair_index = len(list(root.glob("*-pair.json"))) + 1
    for record in (stock, manager):
        suffix = pair_index * 2 + (record["mode"] == "manager")
        record["runtime_identity"]["run_id"] = (
            f"00000000-0000-4000-8000-{suffix:012x}"
        )
        record["started_at_utc"] = (
            f"2026-08-28T00:00:{len(list(root.glob('*-pair.json'))):02d}+00:00"
        )
    stock_path = root / f"{name}-stock.json"
    manager_path = root / f"{name}-manager.json"
    _write(stock_path, stock)
    _write(manager_path, manager)
    references = {
        "stock": {
            "path": stock_path.name,
            "sha256": verifier.sha256_file(stock_path),
        },
        "manager": {
            "path": manager_path.name,
            "sha256": verifier.sha256_file(manager_path),
        },
    }
    seed = {
        "schema": verifier.PAIR_SCHEMA,
        "case": stock["workload"]["case"],
        "contract_sha256": "",
        "suite_contract_sha256": "",
        "records": references,
        "thresholds": dict(verifier.DEFAULT_THRESHOLDS),
        "throughput_statistics": {
            "roomy_epoch_count": 0,
            "paired_sample_count": 0,
            "paired_median_regression_fraction": None,
            "paired_bootstrap_upper_regression_fraction": None,
        },
        "gates": {},
        "overall_qualified": False,
    }
    pair, _, _ = verifier._expected_pair(root, seed)
    pair_path = root / f"{name}-pair.json"
    _write(pair_path, pair)
    return pair_path


def _reseal_binding(record: dict[str, Any]) -> None:
    record["runtime_binding"]["fingerprint"] = verifier._binding_digest(
        record["runtime_binding"]
    )


GENERIC_WINDOW_TOKENS = 32
GENERIC_MINIMUM_RESIDENT_TOKENS = 48
GENERIC_MATERIALIZED_TOKENS = 144
GENERIC_PREFILL_TOKENS = 16
GENERIC_SLIDING_FLOOR_TOKENS = 64
GENERIC_FULL_FLOOR_TOKENS = max(
    (
        (GENERIC_MATERIALIZED_TOKENS + 1 + PAGE_TOKENS - 1)
        // PAGE_TOKENS
        * PAGE_TOKENS
    ),
    GENERIC_SLIDING_FLOOR_TOKENS,
)
GENERIC_MOE = {
    "moe_runner_backend": "triton",
    "moe_a2a_backend": "none",
    "ep_size": 1,
}


def _generic_profile_geometry(profile: str) -> dict[str, Any]:
    sliding = {
        "class_id": 1 if profile == verifier.FULL_SLIDING_TOPOLOGY else 0,
        "retention": "sliding",
        "layers": [1] if profile == verifier.FULL_SLIDING_TOPOLOGY else [0, 1],
        "window_tokens": GENERIC_WINDOW_TOKENS,
        "minimum_resident_tokens": GENERIC_MINIMUM_RESIDENT_TOKENS,
    }
    if profile == verifier.FULL_SLIDING_TOPOLOGY:
        return {
            "page_tokens": PAGE_TOKENS,
            "classes": [
                {
                    "class_id": 0,
                    "retention": "full",
                    "layers": [0],
                    "window_tokens": None,
                    "minimum_resident_tokens": None,
                },
                sliding,
            ],
        }
    if profile == verifier.SLIDING_TOPOLOGY:
        return {"page_tokens": PAGE_TOKENS, "classes": [sliding]}
    raise ValueError(f"unsupported generic profile: {profile}")


def _generic_runtime_admission(
    profile: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    geometry = _generic_profile_geometry(profile)
    states = []
    layout_classes = []
    compiled_states = []
    for item in geometry["classes"]:
        retention = item["retention"]
        if retention == "full":
            address = {"kind": "append_only"}
            retirement = {"kind": "never"}
            minimum_slots = None
        else:
            minimum_slots = GENERIC_MINIMUM_RESIDENT_TOKENS // PAGE_TOKENS
            address = {
                "kind": "periodic",
                "period_blocks": minimum_slots,
            }
            retirement = {
                "kind": "block_end_plus",
                "offset_tokens": GENERIC_WINDOW_TOKENS - 1,
            }
        storage = {
            "kind": "token_kv",
            "key_bytes_per_token_per_layer": 32,
            "value_bytes_per_token_per_layer": 64,
            "retention": retention,
            "window_tokens": item["window_tokens"],
        }
        states.append(
            {
                "name": retention,
                "layers": list(item["layers"]),
                "storage": storage,
            }
        )
        token_class = {
            "name": retention,
            "layers": list(item["layers"]),
            "bytes_per_token_per_layer": 96,
            "address": address,
            "retirement": retirement,
            "minimum_slots_per_request": minimum_slots,
        }
        backend = {
            "kind": "token_slots",
            "storage": "token_kv",
            "components": [
                {"name": "key", "bytes_per_token_per_layer": 32},
                {"name": "value", "bytes_per_token_per_layer": 64},
            ],
            "bytes_per_token_per_layer": 96,
            "page_bytes_per_layer": 96 * PAGE_TOKENS,
            "retention": retention,
            "window_tokens": item["window_tokens"],
            "token_relocatable": True,
        }
        layout_classes.append(token_class)
        compiled_states.append(
            {
                "name": retention,
                "layers": list(item["layers"]),
                "backend": backend,
            }
        )

    attention_input = {"page_tokens": PAGE_TOKENS, "states": states}
    manifest: dict[str, Any] = {
        "schema": "orbitkv.runtime-manifest",
        "version": 3,
        "fingerprint": "",
        "source": {"kind": "attention_state", "input": attention_input},
        "token_manager_plan": {
            "layout": {
                "schema": "orbitkv.layout-program.v1",
                "plan_fingerprint": layout_fingerprint_from_manager_input(
                    manager_input_from_attention_source(
                        {"source": {"kind": "attention_state", "input": attention_input}},
                        "generic fixture",
                    ),
                    "generic fixture",
                ),
                "page_tokens": PAGE_TOKENS,
                "classes": copy.deepcopy(layout_classes),
            }
        },
        "attention_state_plan": {
            "schema": "orbitkv.attention-state-plan.v1",
            "page_tokens": PAGE_TOKENS,
            "states": copy.deepcopy(compiled_states),
        },
        "capability_requirements": sorted(
            {
                "periodic_addressing",
                "semantic_retirement",
                "token_component_geometry",
                "token_manager",
                *(
                    ("append_only_addressing",)
                    if profile == verifier.FULL_SLIDING_TOPOLOGY
                    else ()
                ),
            }
        ),
    }
    manifest["fingerprint"] = verifier._binding_digest(manifest)
    signature: dict[str, Any] = {
        "schema": "orbitkv.execution-signature",
        "version": 1,
        "fingerprint": "",
        "manifest_schema": manifest["schema"],
        "manifest_version": manifest["version"],
        "manifest_fingerprint": manifest["fingerprint"],
        "page_tokens": PAGE_TOKENS,
        "token_classes": copy.deepcopy(layout_classes),
        "token_states": copy.deepcopy(compiled_states),
        "fixed_states": [],
    }
    signature["fingerprint"] = verifier._binding_digest(signature)
    binding: dict[str, Any] = {
        "schema": "orbitkv.runtime-binding",
        "version": 1,
        "fingerprint": "",
        "manifest_fingerprint": manifest["fingerprint"],
        "target": {
            "id": "sglang",
            "contract_version": verifier.CURRENT_TARGET_CONTRACT_VERSION,
        },
        "admission_profile": {
            "id": "eager-single-device-bf16-nhd",
            "version": 1,
        },
        "target_contract_fingerprint": verifier.CURRENT_TARGET_FINGERPRINT,
        "required_wire_version": verifier.CURRENT_WIRE_VERSION,
        "execution_topology": profile,
        "execution_signature": signature,
    }
    binding["fingerprint"] = verifier._binding_digest(binding)
    manifest_record = {
        "artifact": {
            "path": "/synthetic/runtime-manifest.json",
            "bytes": 1,
            "sha256": ZERO_SHA,
        },
        "document": manifest,
        "manifest_fingerprint": manifest["fingerprint"],
        "layout_plan_fingerprint": manifest["token_manager_plan"][
            "layout"
        ]["plan_fingerprint"],
        "execution_signature_fingerprint": signature["fingerprint"],
        "profile_geometry": copy.deepcopy(geometry),
    }
    return manifest_record, binding, geometry


def _generic_engine(
    mode: str, profile: str, capacity_tokens: int
) -> dict[str, Any]:
    engine: dict[str, Any] = {
        "model_path": "/synthetic/model",
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": 4096,
        "page_size": PAGE_TOKENS,
        "attention_backend": "fa3",
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": profile == verifier.SLIDING_TOPOLOGY,
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": GENERIC_PREFILL_TOKENS,
        "prefill_max_requests": 1,
        "max_prefill_tokens": GENERIC_PREFILL_TOKENS,
        "enable_dynamic_chunking": False,
        "enable_mixed_chunk": False,
        "max_running_requests": 1,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "enable_dp_attention": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
        "enable_hierarchical_cache": False,
        "enable_streaming_session": False,
        "enable_unified_memory": False,
        "enable_pdmux": False,
        "enable_lmcache": False,
        "enable_flexkv": False,
        "enable_session_radix_cache": False,
        "enable_hisparse": False,
        "enable_page_major_kv_layout": False,
        "random_seed": 7,
        "log_level": "error",
        "max_total_tokens": capacity_tokens,
        **GENERIC_MOE,
    }
    if profile == verifier.FULL_SLIDING_TOPOLOGY:
        engine["swa_full_tokens_ratio"] = 0.6
    if mode == "manager":
        engine["radix_cache_backend"] = "orbitkv"
    return engine


def _generic_manager_snapshot(
    stage: str, profile: str, *, active: bool
) -> dict[str, Any]:
    class_count = 2 if profile == verifier.FULL_SLIDING_TOPOLOGY else 1
    identities = []
    arenas = []
    for class_id in range(class_count):
        page_count = 16 if class_id == 0 else 8
        first_page_id = 1 + sum(16 if item == 0 else 8 for item in range(class_id))
        identity = {
            "engine_epoch": 1,
            "pool_epoch": 2 + class_id,
            "pool_id": 3 + class_id,
            "class_id": class_id,
            "backend_domain": 4,
            "page_count": page_count,
            "page_tokens": PAGE_TOKENS,
            "backend_base_index": first_page_id - 1,
            "first_page_id": first_page_id,
        }
        arena = {name: 0 for name in verifier.ARENA_KEYS}
        arena.update(
            {
                name: identity[name]
                for name in (
                    "engine_epoch",
                    "pool_epoch",
                    "pool_id",
                    "page_count",
                    "class_id",
                    "backend_domain",
                    "first_page_id",
                )
            }
        )
        arena["free_pages"] = page_count
        identities.append(identity)
        arenas.append(arena)
    stats = {name: 0 for name in verifier.MANAGER_STATS_KEYS}
    stats["free_pages"] = sum(item["page_count"] for item in arenas)
    scale = ITERATIONS * 97 if active else 0
    counters = {name: 0 for name in qualification_native.SESSION_COUNTER_KEYS}
    if active:
        counters.update(
            forward_events=scale,
            completion_values=scale,
            event_queries=scale,
        )
    swa_activity = {
        "status": "exposed",
        "applicable": True,
        "source": "native_runtime_session",
        "derived": False,
        **{
            name: index + 1 if active else 0
            for index, name in enumerate(qualification_native.SWA_COUNTER_KEYS)
        },
    }
    return {
        "stage": stage,
        "lifecycle_route": "native_session",
        "cache_policy": (
            "shared_prefix"
            if profile == verifier.FULL_SLIDING_TOPOLOGY
            else "request_private"
        ),
        "identities": identities,
        "manager_stats": stats,
        "arena_stats": arenas,
        "batch_counters": counters,
        "pressure": {
            "schema": "orbitkv.runtime-pressure.v1",
            "enabled": False,
            "mode": "event_driven_high_water",
            "sample_count": 0,
        },
        "runtime_proof": None,
        "swa_activity": swa_activity,
        "completion_evidence": {
            "event_backend": "cuda_event_current_forward_stream",
            "pending_events": 0,
            "completion_high_water": (
                [{"domain": 4, "value": scale}] if active else []
            ),
        },
    }


def _generic_manager_state(profile: str) -> dict[str, Any]:
    return {
        "wire_version": verifier.CURRENT_WIRE_VERSION,
        "snapshots": [
            _generic_manager_snapshot("after_load", profile, active=False),
            _generic_manager_snapshot("after_workload", profile, active=True),
            _generic_manager_snapshot("final", profile, active=True),
        ],
    }


def _generic_record(
    mode: str,
    profile: str,
    case: str = "roomy",
    *,
    seconds: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    record = _record(mode, case, seconds=seconds)
    record["schema"] = verifier.NATIVE_RECORD_SCHEMA
    record["profile"] = profile
    record["source_identity"]["profile_artifact"] = {
        "path": "/synthetic/runtime-manifest.json",
        "bytes": 1,
        "sha256": ZERO_SHA,
    }
    record["source_identity_sha256"] = verifier._canonical_digest(
        record["source_identity"]
    )
    layer_types = (
        ["full_attention", "sliding_attention"]
        if profile == verifier.FULL_SLIDING_TOPOLOGY
        else ["sliding_attention", "sliding_attention"]
    )
    record["checkpoint"] = {
        "identity": {"revision": "synthetic-gpt-oss", "weight_bytes": 1},
        "config": {
            "architectures": ["GptOssForCausalLM"],
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "max_position_embeddings": 4096,
            "sliding_window": GENERIC_WINDOW_TOKENS,
            "control_token_ids": {},
            "layer_types": layer_types,
        },
        "backend_profile": {"attention_backend": "fa3", **GENERIC_MOE},
    }
    record["checkpoint_identity_sha256"] = verifier._canonical_digest(
        record["checkpoint"]
    )
    record["runtime_identity"]["moe_backend"] = {
        "runner": "triton",
        "a2a": "none",
        "ep_size": 1,
    }
    geometry = _generic_profile_geometry(profile)
    record["workload"].pop("chunk_geometry")
    record["workload"].update(
        profile_geometry=copy.deepcopy(geometry),
        retirement_boundary_crossings_per_request=(
            GENERIC_MATERIALIZED_TOKENS - 1
        ) // GENERIC_MINIMUM_RESIDENT_TOKENS,
    )
    assert (
        record["workload"]["materialized_kv_tokens_per_request"]
        == GENERIC_MATERIALIZED_TOKENS
    )
    if case == "roomy":
        capacity_tokens = 256
    elif profile == verifier.FULL_SLIDING_TOPOLOGY:
        capacity_tokens = GENERIC_FULL_FLOOR_TOKENS
    else:
        capacity_tokens = GENERIC_SLIDING_FLOOR_TOKENS
    record["engine_args"] = _generic_engine(mode, profile, capacity_tokens)
    if case == "exact-floor" and profile == verifier.FULL_SLIDING_TOPOLOGY:
        record["engine_args"]["swa_full_tokens_ratio"] = 0.45
    sliding_tokens = (
        capacity_tokens
        if profile == verifier.SLIDING_TOPOLOGY
        else int(
            capacity_tokens
            * record["engine_args"]["swa_full_tokens_ratio"]
        )
        // PAGE_TOKENS
        * PAGE_TOKENS
    )
    class_capacities = {
        "full_tokens": (
            capacity_tokens
            if profile == verifier.FULL_SLIDING_TOPOLOGY
            else 0
        ),
        "sliding_tokens": sliding_tokens,
    }
    failed = case == "exact-floor" and mode == "stock"
    record["server_capacity"] = {
        "status": "failed" if failed else "observed",
        "requested_tokens": capacity_tokens,
        "available_tokens": None if failed else capacity_tokens,
        "failure": (
            {"type": "capacity_exhausted", "message": "KV cache pool is full"}
            if failed
            else None
        ),
        "class_capacities": class_capacities,
    }
    if not failed:
        record["server_capacity"]["floor"] = {
            "configured_max_total_tokens": capacity_tokens,
            "expected_full_tokens": class_capacities["full_tokens"],
            "expected_sliding_tokens": class_capacities["sliding_tokens"],
            "full_floor_tokens": (
                GENERIC_FULL_FLOOR_TOKENS
                if profile == verifier.FULL_SLIDING_TOPOLOGY
                else 0
            ),
            "sliding_floor_tokens": GENERIC_SLIDING_FLOOR_TOKENS,
        }
    record["runtime_identity"]["runtime_proof"] = None
    if mode == "manager":
        manifest, binding, _ = _generic_runtime_admission(profile)
        record["runtime_manifest"] = manifest
        record["runtime_binding"] = binding
        record["manager"] = _generic_manager_state(profile)
    else:
        record["runtime_manifest"] = None
        record["runtime_binding"] = None
        record["manager"] = None
    return record


def _write_generic_pair(
    root: Path, name: str, stock: dict[str, Any], manager: dict[str, Any]
) -> Path:
    pair_index = len(list(root.glob("*-pair.json"))) + 1
    for offset, record in enumerate((stock, manager), start=1):
        record["runtime_identity"]["run_id"] = (
            f"00000000-0000-4000-8000-{pair_index * 2 + offset:012x}"
        )
        record["started_at_utc"] = (
            f"2026-08-28T00:00:{pair_index:02d}+00:00"
        )
    stock_path = root / f"{name}-stock.json"
    manager_path = root / f"{name}-manager.json"
    _write(stock_path, stock)
    _write(manager_path, manager)
    references = {
        "stock": {
            "path": stock_path.name,
            "sha256": verifier.sha256_file(stock_path),
        },
        "manager": {
            "path": manager_path.name,
            "sha256": verifier.sha256_file(manager_path),
        },
    }
    seed = {
        "schema": verifier.NATIVE_PAIR_SCHEMA,
        "profile": stock["profile"],
        "case": stock["workload"]["case"],
        "contract_sha256": "",
        "suite_contract_sha256": "",
        "records": references,
        "thresholds": dict(verifier.DEFAULT_THRESHOLDS),
        "throughput_statistics": {
            "roomy_epoch_count": 0,
            "paired_sample_count": 0,
            "paired_median_regression_fraction": None,
            "paired_bootstrap_upper_regression_fraction": None,
        },
        "gates": {},
        "overall_qualified": False,
    }
    pair, _, _ = verifier._expected_pair(root, seed)
    path = root / f"{name}-pair.json"
    _write(path, pair)
    return path


def _reseal_generic_manifest(record: dict[str, Any]) -> None:
    manifest_record = record["runtime_manifest"]
    manifest = manifest_record["document"]
    manifest["fingerprint"] = verifier._binding_digest(manifest)
    manifest_record["manifest_fingerprint"] = manifest["fingerprint"]
    signature = record["runtime_binding"]["execution_signature"]
    signature["manifest_fingerprint"] = manifest["fingerprint"]
    signature["fingerprint"] = verifier._binding_digest(signature)
    manifest_record["execution_signature_fingerprint"] = signature["fingerprint"]
    binding = record["runtime_binding"]
    binding["manifest_fingerprint"] = manifest["fingerprint"]
    binding["fingerprint"] = verifier._binding_digest(binding)


@pytest.mark.parametrize(
    "profile",
    (verifier.FULL_SLIDING_TOPOLOGY, verifier.SLIDING_TOPOLOGY),
)
@pytest.mark.parametrize("mode", ("stock", "manager"))
def test_generic_roomy_records_validate(profile: str, mode: str) -> None:
    record = _generic_record(mode, profile)

    metadata = verifier.validate_record(record)

    assert metadata["schema"] == verifier.NATIVE_RECORD_SCHEMA
    assert metadata["profile"] == profile
    assert metadata["mode"] == mode
    assert metadata["failed"] is False
    if mode == "manager":
        expected_classes = 2 if profile == verifier.FULL_SLIDING_TOPOLOGY else 1
        assert len(record["manager"]["snapshots"][0]["identities"]) == expected_classes


@pytest.mark.parametrize(
    "profile",
    (verifier.FULL_SLIDING_TOPOLOGY, verifier.SLIDING_TOPOLOGY),
)
def test_generic_pair_and_summary_use_native_schema(
    tmp_path: Path, profile: str
) -> None:
    pairs = [
        _write_generic_pair(
            tmp_path,
            f"native-{index}",
            _generic_record("stock", profile),
            _generic_record("manager", profile),
        )
        for index in range(3)
    ]

    verified_pair = verifier.verify_pair(pairs[0])
    summary = verifier.build_summary(tmp_path, [path.name for path in pairs])
    summary_path = tmp_path / "native-summary.json"
    _write(summary_path, summary)

    assert verified_pair["schema"] == verifier.NATIVE_PAIR_SCHEMA
    assert verified_pair["profile"] == profile
    assert verified_pair["gates"]["correctness_qualified"]["qualified"]
    assert verified_pair["gates"]["stream_event_qualified"]["qualified"]
    assert verifier.verify_summary(summary_path)["schema"] == (
        verifier.NATIVE_SUMMARY_SCHEMA
    )
    assert summary["profile"] == profile
    assert summary["gates"]["throughput_go"]["qualified"]


def test_generic_pair_rejects_profile_mismatch(tmp_path: Path) -> None:
    stock = _generic_record("stock", verifier.FULL_SLIDING_TOPOLOGY)
    manager = _generic_record("manager", verifier.SLIDING_TOPOLOGY)
    _write(tmp_path / "stock.json", stock)
    _write(tmp_path / "manager.json", manager)

    with pytest.raises(RuntimeError, match="schema or profile differs"):
        verifier.build_pair(tmp_path, "stock.json", "manager.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda record: record["runtime_binding"].update(
                execution_topology=verifier.SLIDING_TOPOLOGY
            ),
            "target, wire, or topology differs",
        ),
        (
            lambda record: record["runtime_manifest"]["profile_geometry"][
                "classes"
            ][1].update(minimum_resident_tokens=64),
            "profile_geometry differs from binding",
        ),
        (
            lambda record: record["runtime_manifest"]["document"][
                "source"
            ]["input"]["states"][1]["storage"].update(
                key_bytes_per_token_per_layer=48
            ),
            "canonical RuntimeManifest",
        ),
    ),
    ids=("topology", "profile-geometry", "attention-state-fingerprint"),
)
def test_generic_profile_topology_geometry_and_fingerprint_are_bound(
    mutation, message: str
) -> None:
    record = _generic_record("manager", verifier.FULL_SLIDING_TOPOLOGY)
    mutation(record)
    if message == "target, wire, or topology differs":
        _reseal_binding(record)

    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_generic_attention_state_geometry_is_exact_after_resealing() -> None:
    record = _generic_record("manager", verifier.FULL_SLIDING_TOPOLOGY)
    manifest = record["runtime_manifest"]["document"]
    manifest["source"]["input"]["states"][1]["storage"][
        "window_tokens"
    ] = 33
    _reseal_generic_manifest(record)

    with pytest.raises(
        RuntimeError, match="token projection differs|Sliding token_kv geometry differs"
    ):
        verifier.validate_record(record)


def test_generic_swa_activity_has_exact_eight_fields() -> None:
    record = _generic_record("manager", verifier.SLIDING_TOPOLOGY)
    activity = record["manager"]["snapshots"][1]["swa_activity"]
    assert set(activity) == set(qualification_native.SWA_KEYS)
    assert len(activity) == 8

    activity["unreviewed"] = 1
    with pytest.raises(RuntimeError, match="swa_activity keys differ"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "counter", qualification_native.SWA_COUNTER_KEYS
)
def test_generic_all_four_swa_counter_deltas_must_be_positive(
    counter: str,
) -> None:
    record = _generic_record("manager", verifier.SLIDING_TOPOLOGY)
    snapshots = record["manager"]["snapshots"]
    snapshots[1]["swa_activity"][counter] = snapshots[0]["swa_activity"][counter]
    snapshots[2]["swa_activity"][counter] = snapshots[0]["swa_activity"][counter]

    with pytest.raises(RuntimeError, match="did not advance all SWA counters"):
        verifier.validate_record(record)


@pytest.mark.parametrize("stage", (0, 1, 2))
def test_generic_completion_pending_events_must_drain(stage: int) -> None:
    record = _generic_record("manager", verifier.SLIDING_TOPOLOGY)
    record["manager"]["snapshots"][stage]["completion_evidence"][
        "pending_events"
    ] = 1

    with pytest.raises(RuntimeError, match="completion backend or drain differs"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda snapshots: snapshots[1]["completion_evidence"].update(
                completion_high_water=[]
            ),
            "completion frontier is empty",
        ),
        (
            lambda snapshots: snapshots[2]["completion_evidence"][
                "completion_high_water"
            ][0].update(value=1),
            "completion frontier regressed",
        ),
        (
            lambda snapshots: snapshots[1]["completion_evidence"].update(
                completion_high_water=[
                    {"domain": 4, "value": 1},
                    {"domain": 4, "value": 2},
                ]
            ),
            "duplicate or unordered",
        ),
    ),
    ids=("empty", "regressed", "duplicate-domain"),
)
def test_generic_completion_frontier_is_live_and_monotonic(
    mutation, message: str
) -> None:
    record = _generic_record("manager", verifier.FULL_SLIDING_TOPOLOGY)
    mutation(record["manager"]["snapshots"])

    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_generic_hybrid_requires_two_arenas_and_aggregate_census() -> None:
    record = _generic_record("manager", verifier.FULL_SLIDING_TOPOLOGY)
    snapshots = record["manager"]["snapshots"]
    assert all(len(snapshot["arena_stats"]) == 2 for snapshot in snapshots)

    snapshots[1]["manager_stats"]["free_pages"] -= 1
    with pytest.raises(RuntimeError, match="manager/arena aggregate differs"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "profile",
    (verifier.FULL_SLIDING_TOPOLOGY, verifier.SLIDING_TOPOLOGY),
)
def test_generic_final_state_must_fully_drain(profile: str) -> None:
    record = _generic_record("manager", profile)
    final = record["manager"]["snapshots"][2]
    final["manager_stats"]["active_pages"] = 1
    final["manager_stats"]["free_pages"] -= 1
    final["arena_stats"][0]["active_pages"] = 1
    final["arena_stats"][0]["free_pages"] -= 1

    with pytest.raises(RuntimeError, match="initial/final state did not fully drain"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "profile",
    (verifier.FULL_SLIDING_TOPOLOGY, verifier.SLIDING_TOPOLOGY),
)
def test_generic_exact_floor_pair_qualifies_capacity(
    tmp_path: Path, profile: str
) -> None:
    path = _write_generic_pair(
        tmp_path,
        "floor",
        _generic_record("stock", profile, "exact-floor"),
        _generic_record("manager", profile, "exact-floor"),
    )

    pair = verifier.verify_pair(path)

    assert pair["gates"]["capacity_qualified"] == {
        "qualified": True,
        "reasons": [],
    }


def test_generic_hybrid_rejects_full_floor_without_next_decode_slot() -> None:
    record = _generic_record(
        "stock", verifier.FULL_SLIDING_TOPOLOGY, "exact-floor"
    )
    record["workload"].update(
        prompt_tokens=560,
        materialized_kv_tokens_per_request=656,
        retirement_boundary_crossings_per_request=(656 - 1)
        // GENERIC_MINIMUM_RESIDENT_TOKENS,
    )
    record["engine_args"].update(
        max_total_tokens=656,
        swa_full_tokens_ratio=0.1,
    )
    record["server_capacity"].update(
        requested_tokens=656,
        class_capacities={
            "full_tokens": 656,
            "sliding_tokens": GENERIC_SLIDING_FLOOR_TOKENS,
        },
    )

    with pytest.raises(
        RuntimeError, match="failed capacity does not use exact class floors"
    ):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("profile", "field", "value"),
    (
        (verifier.FULL_SLIDING_TOPOLOGY, "full_tokens", 128),
        (verifier.FULL_SLIDING_TOPOLOGY, "sliding_tokens", 128),
        (verifier.SLIDING_TOPOLOGY, "full_tokens", 16),
        (verifier.SLIDING_TOPOLOGY, "sliding_tokens", 240),
    ),
)
def test_generic_class_capacities_are_exact(
    profile: str, field: str, value: int
) -> None:
    record = _generic_record("manager", profile)
    record["server_capacity"]["class_capacities"][field] = value

    with pytest.raises(RuntimeError, match="class capacities differ"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "profile",
    (verifier.FULL_SLIDING_TOPOLOGY, verifier.SLIDING_TOPOLOGY),
)
def test_generic_capacity_floor_is_exact(profile: str) -> None:
    record = _generic_record("manager", profile)
    record["server_capacity"]["floor"]["sliding_floor_tokens"] += PAGE_TOKENS

    with pytest.raises(RuntimeError, match="class capacity floor differs"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("checkpoint", "moe_runner_backend"),
        ("checkpoint", "moe_a2a_backend"),
        ("checkpoint", "ep_size"),
        ("engine", "moe_runner_backend"),
        ("engine", "moe_a2a_backend"),
        ("engine", "ep_size"),
        ("runtime", "runner"),
        ("runtime", "a2a"),
        ("runtime", "ep_size"),
    ),
)
def test_generic_gpt_oss_requires_all_three_moe_fields(
    location: str, field: str
) -> None:
    record = _generic_record("manager", verifier.FULL_SLIDING_TOPOLOGY)
    if location == "checkpoint":
        record["checkpoint"]["backend_profile"].pop(field)
        record["checkpoint_identity_sha256"] = verifier._canonical_digest(
            record["checkpoint"]
        )
        message = "backend_profile keys differ"
    elif location == "engine":
        record["engine_args"].pop(field)
        message = "GPT-OSS MoE engine arguments differ"
    else:
        record["runtime_identity"]["moe_backend"].pop(field)
        message = "moe_backend keys differ"

    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_generic_source_profile_artifact_is_required_and_pair_bound(
    tmp_path: Path,
) -> None:
    record = _generic_record("stock", verifier.SLIDING_TOPOLOGY)
    record["source_identity"].pop("profile_artifact")
    record["source_identity_sha256"] = verifier._canonical_digest(
        record["source_identity"]
    )
    with pytest.raises(RuntimeError, match="profile_artifact"):
        verifier.validate_record(record)

    stock = _generic_record("stock", verifier.SLIDING_TOPOLOGY)
    manager = _generic_record("manager", verifier.SLIDING_TOPOLOGY)
    stock["source_identity"]["profile_artifact"]["sha256"] = "f" * 64
    stock["source_identity_sha256"] = verifier._canonical_digest(
        stock["source_identity"]
    )
    _write(tmp_path / "stock.json", stock)
    _write(tmp_path / "manager.json", manager)
    with pytest.raises(RuntimeError, match="source contract differs"):
        verifier.build_pair(tmp_path, "stock.json", "manager.json")


def test_generic_profile_artifact_path_is_normalized_for_pair(
    tmp_path: Path,
) -> None:
    stock = _generic_record("stock", verifier.SLIDING_TOPOLOGY)
    manager = _generic_record("manager", verifier.SLIDING_TOPOLOGY)
    stock["source_identity"]["profile_artifact"]["path"] = (
        "/stock/profile.json"
    )
    manager["source_identity"]["profile_artifact"]["path"] = (
        "/manager/profile.json"
    )
    manager["runtime_manifest"]["artifact"]["path"] = (
        "/manager/profile.json"
    )
    for record in (stock, manager):
        record["source_identity_sha256"] = verifier._canonical_digest(
            record["source_identity"]
        )
    _write(tmp_path / "stock.json", stock)
    _write(tmp_path / "manager.json", manager)

    pair = verifier.build_pair(tmp_path, "stock.json", "manager.json")

    assert pair["profile"] == verifier.SLIDING_TOPOLOGY


def test_generic_runtime_moe_identity_is_pair_bound(tmp_path: Path) -> None:
    stock = _generic_record("stock", verifier.FULL_SLIDING_TOPOLOGY)
    manager = _generic_record("manager", verifier.FULL_SLIDING_TOPOLOGY)
    manager["runtime_identity"]["moe_backend"]["runner"] = "other"
    _write(tmp_path / "stock.json", stock)
    _write(tmp_path / "manager.json", manager)
    with pytest.raises(RuntimeError, match="moe_backend differs"):
        verifier.build_pair(tmp_path, "stock.json", "manager.json")


def test_generic_roomy_pair_never_qualifies_capacity(tmp_path: Path) -> None:
    pair_path = _write_generic_pair(
        tmp_path,
        "roomy-capacity",
        _generic_record("stock", verifier.SLIDING_TOPOLOGY),
        _generic_record("manager", verifier.SLIDING_TOPOLOGY),
    )

    pair = verifier.verify_pair(pair_path)

    assert pair["gates"]["capacity_qualified"] == {
        "qualified": False,
        "reasons": ["roomy records alone do not qualify capacity"],
    }


def test_roomy_pair_qualifies_correctness_but_not_unproven_reuse(tmp_path: Path) -> None:
    path = _write_pair(tmp_path, "roomy", _record("stock"), _record("manager"))

    pair = verifier.verify_pair(path)

    assert pair["gates"] == {
        "correctness_qualified": {"qualified": True, "reasons": []},
        "stream_event_qualified": {
            "qualified": False,
            "reasons": [
                "no per-iteration physical page reuse trace was recorded"
            ],
        },
        "capacity_qualified": {
            "qualified": False,
            "reasons": ["roomy records alone do not qualify capacity"],
        },
        "throughput_go": {
            "qualified": False,
            "reasons": ["throughput requires a multi-epoch summary"],
        },
    }


def test_runtime_contract_is_pinned_to_sglang_v4_and_wire_14() -> None:
    record = _record("manager")

    assert verifier.CURRENT_WIRE_VERSION == 14
    assert record["runtime_binding"]["target"] == {
        "id": "sglang",
        "contract_version": 4,
    }
    assert record["runtime_binding"]["required_wire_version"] == 14
    assert record["manager"]["wire_version"] == 14
    assert record["runtime_binding"]["target_contract_fingerprint"] == (
        "sha256:ac915458195e757e477cf04866dae76147cd71a7472661e0d791e9c4474173ba"
    )


def test_exact_floor_pair_qualifies_capacity(tmp_path: Path) -> None:
    path = _write_pair(
        tmp_path,
        "floor",
        _record("stock", "exact-floor"),
        _record("manager", "exact-floor"),
    )

    pair = verifier.verify_pair(path)

    assert pair["gates"]["capacity_qualified"] == {
        "qualified": True,
        "reasons": [],
    }
    assert pair["gates"]["correctness_qualified"]["qualified"] is False
    assert pair["gates"]["stream_event_qualified"]["qualified"] is False


def test_multi_epoch_summary_qualifies_throughput(tmp_path: Path) -> None:
    paths = [
        _write_pair(
            tmp_path,
            f"epoch-{epoch}",
            _record("stock", seconds=(1.0, 1.0, 1.0)),
            _record("manager", seconds=(1.01, 1.01, 1.01)),
        )
        for epoch in range(3)
    ]
    summary = verifier.build_summary(tmp_path, [path.name for path in paths])
    summary_path = tmp_path / "summary.json"
    _write(summary_path, summary)

    verified = verifier.verify_summary(summary_path)

    assert verified["throughput_statistics"] == {
        "roomy_epoch_count": 3,
        "paired_sample_count": 9,
        "paired_median_regression_fraction": pytest.approx(1 - 1 / 1.01),
        "paired_bootstrap_upper_regression_fraction": pytest.approx(
            1 - 1 / 1.01
        ),
    }
    assert verified["gates"]["throughput_go"] == {
        "qualified": True,
        "reasons": [],
    }


@pytest.mark.parametrize("duplicate_mode", ("stock", "manager"))
def test_summary_rejects_reused_raw_epoch_record(
    tmp_path: Path, duplicate_mode: str
) -> None:
    first = _write_pair(
        tmp_path, "epoch-0", _record("stock"), _record("manager")
    )
    second = _write_pair(
        tmp_path, "epoch-1", _record("stock"), _record("manager")
    )
    first_pair = json.loads(first.read_text(encoding="utf-8"))
    second_pair = json.loads(second.read_text(encoding="utf-8"))
    second_pair["records"][duplicate_mode] = first_pair["records"][duplicate_mode]
    expected, _, _ = verifier._expected_pair(tmp_path, second_pair)
    _write(second, expected)

    with pytest.raises(RuntimeError, match="distinct raw/run/pair epochs"):
        verifier.build_summary(tmp_path, [first.name, second.name])


@pytest.mark.parametrize(
    "message",
    (
        "CUDA out of memory: KV cache pool is full",
        "unrelated failure",
    ),
)
def test_capacity_gate_reclassifies_raw_failure_message(
    tmp_path: Path, message: str
) -> None:
    stock = _record("stock", "exact-floor")
    stock["server_capacity"]["failure"] = {
        "type": "capacity_exhausted",
        "message": message,
    }
    path = _write_pair(
        tmp_path, "floor", stock, _record("manager", "exact-floor")
    )
    pair = verifier.verify_pair(path)
    assert pair["gates"]["capacity_qualified"]["qualified"] is False


def test_failed_exact_floor_rejects_partial_output_or_timing() -> None:
    record = _record("stock", "exact-floor")
    record["timings"]["iteration_seconds"] = [1.0]
    with pytest.raises(RuntimeError, match="timing"):
        verifier.validate_record(record)

    record = _record("stock", "exact-floor")
    record["outputs"] = _outputs("stock", iterations=ITERATIONS)
    with pytest.raises(RuntimeError, match="outputs.*cardinality"):
        verifier.validate_record(record)


def test_stream_gate_needs_single_request_physical_reuse(tmp_path: Path) -> None:
    manager = _record("manager")
    for snapshot in manager["manager"]["snapshots"]:
        snapshot["identities"][0]["page_count"] = 16
        snapshot["arena_stats"][0]["page_count"] = 16
        snapshot["arena_stats"][0]["free_pages"] = 16
        snapshot["manager_stats"]["free_pages"] = 16
    path = _write_pair(tmp_path, "no-reuse", _record("stock"), manager)
    pair = verifier.verify_pair(path)
    gate = pair["gates"]["stream_event_qualified"]
    assert gate["qualified"] is False
    assert any("physical page reuse" in reason for reason in gate["reasons"])


def test_single_record_claims_are_required_to_default_false() -> None:
    record = _record("manager")
    metadata = verifier.validate_record(record)
    assert metadata["failed"] is False
    assert all(not gate["qualified"] and gate["reasons"] for gate in record["claims"].values())

    record["claims"]["correctness_qualified"]["qualified"] = True
    with pytest.raises(RuntimeError, match="must remain false"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lifecycle_route", "canonical_manager"),
        ("cache_policy", "shared_prefix"),
    ),
)
def test_chunked_manager_requires_explicit_request_private_session_policy(
    field: str, value: str
) -> None:
    record = _record("manager")
    record["manager"]["snapshots"][1][field] = value

    with pytest.raises(RuntimeError, match="exact Chunked cache policy"):
        verifier.validate_record(record)


def test_chunked_request_private_manager_rejects_prefix_state() -> None:
    record = _record("manager")
    snapshot = record["manager"]["snapshots"][1]
    snapshot["manager_stats"]["active_prefixes"] = 1

    with pytest.raises(RuntimeError, match="request-private Prefix state"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("mode", "selection"),
    (
        ("stock", ""),
        ("stock", "unrelated_plugin"),
        ("manager", "orbitkv_manager"),
        ("manager", "unrelated_plugin"),
    ),
)
def test_plugin_environment_is_rejected_fail_closed(
    mode: str, selection: str
) -> None:
    record = _record(mode)
    record["environment"]["SGLANG_PLUGINS"] = selection
    record["environment_sha256"] = verifier._canonical_digest(
        record["environment"]
    )
    with pytest.raises(RuntimeError, match="plugin|selected OrbitKV"):
        verifier.validate_record(record)


def test_unknown_nested_key_is_rejected() -> None:
    record = _record("manager")
    record["workload"]["chunk_geometry"]["unreviewed"] = True
    with pytest.raises(RuntimeError, match="extra=.*unreviewed"):
        verifier.validate_record(record)


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_legacy_plugin_source_identity_is_rejected(mode: str) -> None:
    record = _record(mode)
    record["source_identity"].pop("direct_source")
    record["source_identity"]["plugin_selection"] = {
        "name": "orbitkv_manager",
        "value": "orbitkv_sglang.plugin:register",
    }
    record["source_identity_sha256"] = verifier._canonical_digest(
        record["source_identity"]
    )
    with pytest.raises(RuntimeError, match="source_identity.*keys"):
        verifier.validate_record(record)


@pytest.mark.parametrize("literal", ("NaN", "Infinity", "-Infinity"))
def test_raw_json_rejects_nonfinite_numbers(tmp_path: Path, literal: str) -> None:
    path = tmp_path / "record.json"
    path.write_text(f'{{"value": {literal}}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-finite JSON number"):
        verifier._strict_json(path)


def test_record_reference_hash_detects_tamper(tmp_path: Path) -> None:
    path = _write_pair(tmp_path, "pair", _record("stock"), _record("manager"))
    stock_path = tmp_path / "pair-stock.json"
    stock = json.loads(stock_path.read_text(encoding="utf-8"))
    stock["started_at_utc"] = "2026-08-28T00:00:01+00:00"
    _write(stock_path, stock)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verifier.verify_pair(path)


def test_summary_pair_reference_hash_detects_tamper(tmp_path: Path) -> None:
    pair_path = _write_pair(tmp_path, "pair", _record("stock"), _record("manager"))
    summary = verifier.build_summary(tmp_path, [pair_path.name])
    summary_path = tmp_path / "summary.json"
    _write(summary_path, summary)
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    pair["case"] = "exact-floor"
    _write(pair_path, pair)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verifier.verify_summary(summary_path)


def test_pair_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    stock = _record("stock")
    manager = _record("manager")
    manager["checkpoint"]["identity"]["revision"] = "other-revision"
    manager["checkpoint_identity_sha256"] = verifier._canonical_digest(
        manager["checkpoint"]
    )
    with pytest.raises(RuntimeError, match="source contract differs"):
        verifier._validate_pair_contract(stock, manager)


def test_pair_normalizes_only_checkout_root_and_implementation_args() -> None:
    stock = _record("stock")
    manager = _record("manager")
    for record, root in ((stock, "/stock"), (manager, "/manager")):
        record["source_identity"]["contract"]["root"] = root
        record["source_identity_sha256"] = verifier._canonical_digest(
            record["source_identity"]
        )
        record["runtime_identity"]["sglang_package"] = (
            root + "/python/sglang/__init__.py"
        )
        record["environment"]["PYTHONPATH"] = root + "/python"
        if record["mode"] == "manager":
            record["environment"]["ORBITKV_SGLANG_ROOT"] = root
        record["environment_sha256"] = verifier._canonical_digest(
            record["environment"]
        )
    stock["command"][-1] = "/stock"
    manager["command"][-2:] = ["--sglang-root=/manager"]
    stock["command_sha256"] = verifier._canonical_digest(stock["command"])
    manager["command_sha256"] = verifier._canonical_digest(manager["command"])

    assert verifier._validate_pair_contract(stock, manager)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda source: source["contract"].update(revision="0" * 40),
        lambda source: source["pinned_contract"]["targets"][0].update(
            base_sha256="f" * 64
        ),
        lambda source: source["contract"]["reviewed_patch"].update(
            sha256="f" * 64
        ),
        lambda source: source["direct_source"].update(owner=None),
        lambda source: source["direct_source"].update(takeover_points=[]),
        lambda source: source["library"].update(wire_version=10),
    ),
    ids=(
        "revision",
        "target",
        "patch",
        "direct-owner",
        "direct-targets",
        "library",
    ),
)
def test_source_contract_tampering_is_rejected(mutation) -> None:
    record = _record("manager")
    mutation(record["source_identity"])
    record["source_identity_sha256"] = verifier._canonical_digest(
        record["source_identity"]
    )
    with pytest.raises(RuntimeError):
        verifier.validate_record(record)


def test_live_source_contract_matches_frozen_verifier() -> None:
    source = _record("manager")["source_identity"]
    normalized = verifier.validate_source_identity(source, "manager")
    assert normalized["contract"]["contract"] == _PINNED_SOURCE_CONTRACT


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record["runtime_identity"].update(run_id="not-a-uuid"),
        lambda record: record["runtime_identity"].update(runtime_proof=None),
        lambda record: record["runtime_identity"]["runtime_proof"][
            "actual_attention_backend"
        ].update(decode_backend="flashinfer"),
        lambda record: record["runtime_identity"]["runtime_proof"][
            "effective_scheduler"
        ].update(max_running_requests=2),
    ),
    ids=("run-id", "missing-proof", "actual-backend", "scheduler"),
)
def test_runtime_proof_is_required_and_fail_closed(mutation) -> None:
    record = _record("manager")
    mutation(record)
    with pytest.raises(RuntimeError):
        verifier.validate_record(record)


def test_frozen_threshold_policy_rejects_artifact_override() -> None:
    thresholds = dict(verifier.DEFAULT_THRESHOLDS)
    thresholds["minimum_roomy_epochs"] = 1
    with pytest.raises(RuntimeError, match="frozen verifier policy"):
        verifier._validate_thresholds(thresholds)


def test_source_binding_rejects_duplicate_root_and_noncanonical_path() -> None:
    record = _record("stock")
    record["command"] += ["--sglang-root=/other"]
    record["command_sha256"] = verifier._canonical_digest(record["command"])
    with pytest.raises(RuntimeError, match="exactly one"):
        verifier.validate_record(record)

    record = _record("stock")
    record["source_identity"]["contract"]["root"] = "/a/../b"
    record["source_identity_sha256"] = verifier._canonical_digest(
        record["source_identity"]
    )
    with pytest.raises(RuntimeError, match="canonical absolute"):
        verifier.validate_record(record)


def test_exact_floor_partial_snapshot_sequence_is_exact() -> None:
    record = _record("stock", "exact-floor")
    record["gpu_snapshots"].append(
        {**copy.deepcopy(record["gpu_snapshots"][0]), "stage": "after_workload"}
    )
    record["gpu_snapshots"][-1]["time_ns"] = 3
    with pytest.raises(RuntimeError, match="stage sequence"):
        verifier.validate_record(record)


@pytest.mark.parametrize("kind", ("pair-mismatch", "repeat-unstable"))
def test_output_mismatch_or_repeat_instability_fails_correctness(
    tmp_path: Path, kind: str
) -> None:
    stock = _record("stock")
    manager = _record("manager")
    target = manager if kind == "pair-mismatch" else stock
    index = 0 if kind == "pair-mismatch" else 1
    target["outputs"]["iterations"][index]["requests"][0]["output_ids"][0] = 999
    _reseal_outputs(target)

    path = _write_pair(tmp_path, kind, stock, manager)
    pair = verifier.verify_pair(path)

    assert pair["gates"]["correctness_qualified"]["qualified"] is False
    assert pair["gates"]["correctness_qualified"]["reasons"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda record: record["manager"].update(wire_version=10),
            "current wire version 14",
        ),
        (
            lambda record: record["runtime_binding"].update(
                execution_topology="partitioned_chunked_token_kv"
            ),
            "fingerprint does not match|exact chunked topology",
        ),
        (
            lambda record: record["runtime_binding"].update(
                manifest_fingerprint="sha256:" + "f" * 64
            ),
            "fingerprint does not match|does not bind",
        ),
    ),
    ids=("wire", "topology", "binding"),
)
def test_wire_topology_or_binding_mismatch_is_rejected(mutation, message) -> None:
    record = _record("manager")
    mutation(record)
    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_stale_runtime_binding_wire_is_rejected() -> None:
    record = _record("manager")
    record["runtime_binding"]["required_wire_version"] = 10
    _reseal_binding(record)

    with pytest.raises(RuntimeError, match="current wire version 14"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "target",
    (
        {"id": "sglang@4", "contract_version": 4},
        {
            "id": "sglang",
            "contract_version": verifier.CURRENT_TARGET_CONTRACT_VERSION - 1,
        },
    ),
    ids=("version-in-id", "stale-contract-version"),
)
def test_runtime_target_identity_is_exact(target: dict[str, object]) -> None:
    record = _record("manager")
    record["runtime_binding"]["target"] = target
    _reseal_binding(record)

    with pytest.raises(RuntimeError, match="does not target SGLang"):
        verifier.validate_record(record)


@pytest.mark.parametrize("unsafe", ("../stock.json", "/stock.json", "a/../stock.json"))
def test_unsafe_record_path_is_rejected(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        verifier.build_pair(tmp_path, unsafe, "manager.json")
