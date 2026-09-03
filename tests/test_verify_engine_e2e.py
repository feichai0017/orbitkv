from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "tools/verify_engine_e2e.py"
SPEC = importlib.util.spec_from_file_location("verify_engine_e2e", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier: ModuleType = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


def _fingerprint(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint() -> dict[str, Any]:
    return {
        "load_format": "auto",
        "config_sha256": ZERO_SHA,
        "index_files": [
            {"name": "model.safetensors.index.json", "bytes": 12, "sha256": ONE_SHA}
        ],
        "weight_files": [
            {"name": "model.safetensors", "bytes": 100, "sha256": ZERO_SHA}
        ],
        "weight_bytes": 100,
        "indexed_weight_files": ["model.safetensors"],
        "indexed_weight_bytes": 96,
        "observed_indexed_weight_bytes": 100,
        "indexed_weight_container_overhead_bytes": 4,
        "missing_indexed_weights": [],
        "indexed_weights_complete": True,
    }


def _engine(mode: str, *, cache_policy: str = "shared_prefix") -> dict[str, Any]:
    value = {
        "model_path": "/records/model",
        "load_format": "auto",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "skip_tokenizer_init": False,
        "trust_remote_code": False,
        "context_length": 64,
        "page_size": 16,
        "attention_backend": "fa3",
        "disable_hybrid_swa_memory": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": mode == "manager" and cache_policy == "request_private",
        "disable_cuda_graph": True,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "chunked_prefill_size": 16,
        "prefill_max_requests": 1,
        "max_prefill_tokens": 16,
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
        "max_total_tokens": 64,
    }
    if mode == "manager":
        value["radix_cache_backend"] = "orbitkv"
    return value


def _runtime_admission(
    *, hybrid: bool = False, pure_sliding: bool = False, latent: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sum((hybrid, pure_sliding, latent)) > 1:
        raise ValueError("fixture profiles are mutually exclusive")
    components = (
        [
            {"name": "latent", "bytes_per_token_per_layer": 80},
            {"name": "rope", "bytes_per_token_per_layer": 16},
        ]
        if latent
        else [
            {"name": "key", "bytes_per_token_per_layer": 32},
            {"name": "value", "bytes_per_token_per_layer": 64},
        ]
    )
    state_specs = [] if pure_sliding else [
        {
            "name": "full",
            "layers": [0] if hybrid else [0, 1],
            "retention": "full",
            "window_tokens": None,
        }
    ]
    if hybrid or pure_sliding:
        state_specs.append(
            {
                "name": "swa",
                "layers": [1] if hybrid else [0, 1],
                "retention": "sliding",
                "window_tokens": 18,
            }
        )
    source_input = {
        "page_tokens": 16,
        "states": [
            {
                "name": item["name"],
                "layers": item["layers"],
                "storage": {
                    "kind": "latent_kv" if latent else "token_kv",
                    **(
                        {
                            "latent_bytes_per_token_per_layer": 80,
                            "rope_bytes_per_token_per_layer": 16,
                        }
                        if latent
                        else {
                            "key_bytes_per_token_per_layer": 32,
                            "value_bytes_per_token_per_layer": 64,
                        }
                    ),
                    "retention": item["retention"],
                    "window_tokens": item["window_tokens"],
                },
            }
            for item in state_specs
        ],
    }
    manager_input = {
        "page_tokens": 16,
        "classes": [
            {
                "name": item["name"],
                "layers": item["layers"],
                "retention": item["retention"],
                "bytes_per_token_per_layer": 96,
                "window_tokens": item["window_tokens"],
                "components": copy.deepcopy(components),
                **({"storage": "latent_kv"} if latent else {}),
            }
            for item in state_specs
        ],
    }
    layout_classes = []
    compiled_states = []
    for item in state_specs:
        if item["retention"] == "full":
            address = {"kind": "append_only"}
            retirement = {"kind": "never"}
            minimum_slots = None
        else:
            window = item["window_tokens"]
            assert isinstance(window, int)
            minimum_slots = 1 + (window - 1 + 15) // 16
            address = {"kind": "periodic", "period_blocks": minimum_slots}
            retirement = {
                "kind": "block_end_plus",
                "offset_tokens": window - 1,
            }
        layout_classes.append(
            {
                "name": item["name"],
                "layers": item["layers"],
                "bytes_per_token_per_layer": 96,
                "address": address,
                "retirement": retirement,
                "minimum_slots_per_request": minimum_slots,
            }
        )
        compiled_states.append(
            {
                "name": item["name"],
                "layers": item["layers"],
                "backend": {
                    "kind": "token_slots",
                    "storage": "latent_kv" if latent else "token_kv",
                    "components": copy.deepcopy(components),
                    "bytes_per_token_per_layer": 96,
                    "page_bytes_per_layer": 1536,
                    "retention": item["retention"],
                    "window_tokens": item["window_tokens"],
                    "token_relocatable": True,
                },
            }
        )
    layout = {
        "schema": "orbitkv.layout-program.v1",
        "plan_fingerprint": verifier._layout_fingerprint_from_manager_input(
            manager_input, "fixture manager input"
        ),
        "page_tokens": 16,
        "classes": layout_classes,
    }
    manifest: dict[str, Any] = {
        "schema": "orbitkv.runtime-manifest",
        "version": 3,
        "fingerprint": "",
        "source": {"kind": "attention_state", "input": source_input},
        "token_manager_plan": {"layout": layout},
        "attention_state_plan": {
            "schema": "orbitkv.attention-state-plan.v1",
            "page_tokens": 16,
            "states": compiled_states,
        },
        "capability_requirements": sorted(
            {
                "token_component_geometry",
                "token_manager",
                *(("append_only_addressing",) if not pure_sliding else ()),
                *(
                    ("periodic_addressing", "semantic_retirement")
                    if hybrid or pure_sliding
                    else ()
                ),
            }
        ),
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    signature: dict[str, Any] = {
        "schema": "orbitkv.execution-signature",
        "version": 1,
        "fingerprint": "",
        "manifest_schema": manifest["schema"],
        "manifest_version": manifest["version"],
        "manifest_fingerprint": manifest["fingerprint"],
        "page_tokens": 16,
        "token_classes": copy.deepcopy(layout_classes),
        "token_states": copy.deepcopy(compiled_states),
        "fixed_states": [],
    }
    signature["fingerprint"] = _fingerprint(signature)
    binding: dict[str, Any] = {
        "schema": "orbitkv.runtime-binding",
        "version": 1,
        "fingerprint": "",
        "manifest_fingerprint": manifest["fingerprint"],
        "target": {
            "id": "sglang",
            "contract_version": verifier.TARGET_CONTRACT_VERSION,
        },
        "admission_profile": {"id": "single-device", "version": 1},
        "target_contract_fingerprint": verifier.TARGET_CONTRACT_FINGERPRINT,
        "required_wire_version": verifier.WIRE_VERSION,
        "execution_topology": (
            "whole_domain_sliding_token_kv"
            if pure_sliding
            else verifier.FULL_SLIDING_TOKEN_KV_TOPOLOGY
            if hybrid
            else verifier.FULL_LATENT_KV_TOPOLOGY
            if latent
            else verifier.FULL_TOKEN_KV_TOPOLOGY
        ),
        "execution_signature": signature,
    }
    binding["fingerprint"] = _fingerprint(binding)
    return manifest, binding


def _engine_snapshots(mode: str, engine: dict[str, Any]) -> list[dict[str, Any]]:
    resolved = {
        name: engine[name]
        for name in verifier.RESOLVED_ENGINE_KEYS
        if name != "effective_max_running_requests_per_dp"
    }
    resolved["effective_max_running_requests_per_dp"] = 1
    return [
        {
            "stage": stage,
            "resolved_engine": copy.deepcopy(resolved),
            "orbitkv_manager_present": mode == "manager",
        }
        for stage in ("after_load", "after_warmup", "after_workload", "final")
    ]


def _manager_snapshot(
    stage: str,
    scale: int,
    *,
    hybrid: bool = False,
    pure_sliding: bool = False,
    latent: bool = False,
    cache_policy: str | None = None,
    retained_prefix: bool = False,
) -> dict[str, Any]:
    manifest, binding = _runtime_admission(
        hybrid=hybrid, pure_sliding=pure_sliding, latent=latent
    )
    if cache_policy is None:
        cache_policy = (
            "request_private" if latent or pure_sliding else "shared_prefix"
        )
    identities = []
    arenas = []
    for class_id in range(2 if hybrid else 1):
        identity = {
            "engine_epoch": 1,
            "pool_epoch": 2 + class_id,
            "pool_id": 3 + class_id,
            "class_id": class_id,
            "backend_domain": 4,
            "page_count": 8,
            "page_tokens": 16,
            "backend_base_index": class_id * 8,
            "first_page_id": 1 + class_id * 8,
        }
        active_pages = 1 if retained_prefix and class_id == 0 else 0
        arena = {
            name: identity[name]
            for name in (
                "engine_epoch", "pool_epoch", "pool_id", "page_count",
                "class_id", "backend_domain", "first_page_id",
            )
        }
        arena.update(
            free_pages=8 - active_pages,
            reserved_pages=0,
            writing_pages=0,
            active_pages=active_pages,
            retiring_pages=0,
            quarantined_pages=0,
            exhausted_pages=0,
            request_page_refs=0,
            prefix_page_refs=active_pages,
            reader_pins=0,
        )
        identities.append(identity)
        arenas.append(arena)
    stats = {name: 0 for name in verifier.MANAGER_STATS_KEYS}
    stats.update(
        active_prefixes=1 if retained_prefix else 0,
        evicted_prefixes=0,
        free_pages=sum(item["free_pages"] for item in arenas),
        active_pages=sum(item["active_pages"] for item in arenas),
        total_prefix_page_refs=sum(item["prefix_page_refs"] for item in arenas),
    )
    counters = {name: 0 for name in verifier.BATCH_COUNTER_KEYS}
    if scale:
        counters.update(
            forward_events=scale,
            completion_values=scale,
            event_queries=scale,
            event_waits=0,
        )
    return {
        "stage": stage,
        "wire_version": verifier.WIRE_VERSION,
        "manager_input_fingerprint": verifier._manager_input_fingerprint(
            manifest, "fixture manifest"
        ),
        "runtime_manifest_fingerprint": manifest["fingerprint"],
        "runtime_binding_fingerprint": binding["fingerprint"],
        "lifecycle_route": "native_session",
        "cache_policy": cache_policy,
        "direct_source_owner": {
            "module": "orbitkv_sglang.engine",
            "type": "OrbitKvLifecycleOwner",
            "owner_is_process_singleton": True,
            "config_is_canonical": True,
            "allocator_owned": True,
            "tree_cache_owned": True,
        },
        "completion_evidence": {
            "event_backend": "cuda_event_current_forward_stream",
            "pending_events": 0,
            "completion_high_water": [] if not scale else [{"domain": 4, "value": scale}],
        },
        "runtime_proof": None,
        "identities": identities,
        "manager_stats": stats,
        "arena_stats": arenas,
        "batch_counters": counters,
        "swa_activity": {
            "status": (
                "exposed" if hybrid or pure_sliding else "not_applicable"
            ),
            "applicable": hybrid or pure_sliding,
            "source": "native_runtime_session",
            "derived": False,
            **{
                name: scale if hybrid or pure_sliding else 0
                for name in verifier.SWA_COUNTER_FIELDS
            },
        },
        "pressure": {
            "schema": "orbitkv.runtime-pressure.v1",
            "enabled": False,
            "mode": "event_driven_high_water",
            "sample_count": 0,
        },
    }


def _manager_state(
    *,
    hybrid: bool = False,
    pure_sliding: bool = False,
    latent: bool = False,
    cache_policy: str | None = None,
    retained_prefix: bool = False,
) -> dict[str, Any]:
    snapshots = [
        _manager_snapshot(
            "after_load", 0, hybrid=hybrid, pure_sliding=pure_sliding,
            latent=latent, cache_policy=cache_policy,
        ),
        _manager_snapshot(
            "after_warmup", 10, hybrid=hybrid, pure_sliding=pure_sliding,
            latent=latent, cache_policy=cache_policy,
            retained_prefix=retained_prefix,
        ),
        _manager_snapshot(
            "after_workload", 20, hybrid=hybrid, pure_sliding=pure_sliding,
            latent=latent, cache_policy=cache_policy,
            retained_prefix=retained_prefix,
        ),
        _manager_snapshot(
            "final", 20, hybrid=hybrid, pure_sliding=pure_sliding,
            latent=latent, cache_policy=cache_policy,
        ),
    ]
    return {
        "wire_version": verifier.WIRE_VERSION,
        "post_workload_residency": {
            name: snapshots[2]["manager_stats"][name]
            for name in verifier.POST_WORKLOAD_RESIDENCY_KEYS
        },
        "snapshots": snapshots,
    }


def test_manager_input_fingerprint_rebuilds_components_and_default_storage() -> None:
    source_manifest = {
        "source": {
            "kind": "attention_state",
            "input": {
                "page_tokens": 16,
                "states": [
                    {
                        "name": "full",
                        "layers": [0],
                        "storage": {
                            "kind": "token_kv",
                            "key_bytes_per_token_per_layer": 32,
                            "value_bytes_per_token_per_layer": 64,
                            "retention": "full",
                            "window_tokens": None,
                        },
                    },
                    {
                        "name": "latent",
                        "layers": [1],
                        "storage": {
                            "kind": "latent_kv",
                            "latent_bytes_per_token_per_layer": 80,
                            "rope_bytes_per_token_per_layer": 16,
                            "retention": "full",
                            "window_tokens": None,
                        },
                    },
                    {
                        "name": "state",
                        "layers": [2],
                        "storage": {
                            "kind": "recurrent",
                            "family": "gdn",
                            "state_bytes_per_layer": 128,
                            "checkpoint_slots_per_request": 2,
                        },
                    },
                ],
            },
        }
    }
    manager_input = verifier._manager_input_from_attention_source(
        source_manifest, "manifest"
    )
    assert manager_input == {
        "page_tokens": 16,
        "classes": [
            {
                "name": "full",
                "layers": [0],
                "retention": "full",
                "bytes_per_token_per_layer": 96,
                "window_tokens": None,
                "components": [
                    {"name": "key", "bytes_per_token_per_layer": 32},
                    {"name": "value", "bytes_per_token_per_layer": 64},
                ],
            },
            {
                "name": "latent",
                "layers": [1],
                "retention": "full",
                "bytes_per_token_per_layer": 96,
                "window_tokens": None,
                "components": [
                    {"name": "latent", "bytes_per_token_per_layer": 80},
                    {"name": "rope", "bytes_per_token_per_layer": 16},
                ],
                "storage": "latent_kv",
            },
        ],
    }
    assert "storage" not in manager_input["classes"][0]
    expected = "sha256:" + hashlib.sha256(
        json.dumps(
            manager_input,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert verifier._manager_input_fingerprint(source_manifest, "manifest") == expected


def test_manager_input_fingerprint_is_distinct_from_layout_fingerprint() -> None:
    manifest, _binding = _runtime_admission()
    projected = verifier._manager_input_from_attention_source(manifest, "manifest")
    manager_input = verifier._manager_input_fingerprint(manifest, "manifest")
    assert manager_input != manifest["token_manager_plan"]["layout"][
        "plan_fingerprint"
    ]
    assert verifier._layout_fingerprint_from_manager_input(
        projected, "manager input"
    ) == manifest["token_manager_plan"]["layout"]["plan_fingerprint"]


def test_runtime_contract_is_pinned_to_sglang_v4_and_wire_14() -> None:
    _manifest, binding = _runtime_admission()

    assert verifier.WIRE_VERSION == 14
    assert binding["target"] == {"id": "sglang", "contract_version": 4}
    assert binding["required_wire_version"] == 14
    assert binding["target_contract_fingerprint"] == (
        "sha256:ac915458195e757e477cf04866dae76147cd71a7472661e0d791e9c4474173ba"
    )


def _record(
    mode: str,
    seconds: tuple[float, ...] = (2.0, 4.0, 6.0),
    *,
    hybrid: bool = False,
    pure_sliding: bool = False,
    latent: bool = False,
    cache_policy: str | None = None,
    retained_prefix: bool = False,
) -> dict[str, Any]:
    resolved_policy = cache_policy or (
        "request_private" if latent or pure_sliding else "shared_prefix"
    )
    engine = _engine(mode, cache_policy=resolved_policy)
    warmup_prompts = [[7, 8, 9, 10]]
    measured_prompts = [[8, 20, 21, 22], [9, 30, 31, 32], [10, 40, 41, 42]]
    warmup_ids = ["orbitkv-engine-e2e-warmup-7-0"]
    measured_ids = [f"orbitkv-engine-e2e-measured-7-{index}" for index in range(3)]
    workload = {
        "prompt_tokens": 4,
        "decode_tokens": 2,
        "warmups": 1,
        "iterations": 3,
        "seed": 7,
        "input_ids": {"warmup": warmup_prompts, "measured": measured_prompts},
        "input_ids_sha256": {
            "warmup": [verifier._canonical_digest(item) for item in warmup_prompts],
            "measured": [verifier._canonical_digest(item) for item in measured_prompts],
        },
        "request_ids": {"warmup": warmup_ids, "measured": measured_ids},
    }
    warmups = [
        {
            "warmup": 0,
            "request_id": warmup_ids[0],
            "input_ids_sha256": workload["input_ids_sha256"]["warmup"][0],
            "output_ids": [50, 51],
            "output_ids_sha256": verifier._canonical_digest([50, 51]),
            "cached_tokens": 0,
        }
    ]
    iterations = []
    for index, request_id in enumerate(measured_ids):
        tokens = [100 + index, 110 + index]
        iterations.append(
            {
                "iteration": index,
                "request_id": request_id,
                "input_ids_sha256": workload["input_ids_sha256"]["measured"][index],
                "output_ids": tokens,
                "output_ids_sha256": verifier._canonical_digest(tokens),
                "cached_tokens": 0,
            }
        )
    outputs = {"warmups": warmups, "iterations": iterations}
    outputs["aggregate_sha256"] = verifier._canonical_digest(outputs)
    total = sum(seconds)
    timings = {
        "iteration_seconds": list(seconds),
        "median_seconds": sorted(seconds)[1],
        "p95_seconds": verifier._percentile(seconds, 0.95),
        "measured_seconds": total,
        "output_tokens_per_second": 6 / total,
        "total_tokens_per_second": 18 / total,
    }
    manifest, binding = _runtime_admission(
        hybrid=hybrid, pure_sliding=pure_sliding, latent=latent
    )
    root = f"/records/{mode}/sglang"
    environment = {
        "SGLANG_USE_HND_KVCACHE": "0",
        "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
        "SGLANG_RADIX_FORCE_MISS": "0",
        "PYTHONPATH": f"{root}/python:/adapter/source",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if mode == "manager":
        environment.update(
            ORBITKV_RUNTIME_MANIFEST="/records/manager/runtime-manifest.json",
            ORBITKV_LIBRARY="/records/manager/libmanager.so",
            ORBITKV_SGLANG_ROOT=root,
        )
    return {
        "schema": verifier.RECORD_SCHEMA,
        "mode": mode,
        "source": {
            "root": root,
            "release": "v1.2.3",
            "revision": "a" * 40,
            "patch": {
                "status": "absent" if mode == "stock" else "applied",
                "sha256": None if mode == "stock" else "4" * 64,
            },
        },
        "environment": environment,
        "accelerator": {
            "device_type": "cuda",
            "device_name": "Generic Accelerator",
            "compute_capability": {"major": 9, "minor": 0},
            "total_memory_bytes": 1024 * 1024 * 1024,
            "runtime_version": "13.0",
            "driver_version": "13.1",
        },
        "model": engine["model_path"],
        "checkpoint": _checkpoint(),
        "runtime_manifest": None if mode == "stock" else manifest,
        "runtime_binding": None if mode == "stock" else binding,
        "engine_args": engine,
        "server_snapshots": _engine_snapshots(mode, engine),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2,
            "min_new_tokens": 2,
            "ignore_eos": True,
            "sampling_seed": 7,
        },
        "workload": workload,
        "outputs": outputs,
        "timings": timings,
        "manager": (
            None
            if mode == "stock"
            else _manager_state(
                hybrid=hybrid,
                pure_sliding=pure_sliding,
                latent=latent,
                cache_policy=resolved_policy,
                retained_prefix=retained_prefix,
            )
        ),
    }


def _reseal_outputs(record: dict[str, Any]) -> None:
    record["outputs"]["aggregate_sha256"] = verifier._canonical_digest(
        {
            "warmups": record["outputs"]["warmups"],
            "iterations": record["outputs"]["iterations"],
        }
    )


def _reseal_runtime(record: dict[str, Any]) -> None:
    manifest = record["runtime_manifest"]
    binding = record["runtime_binding"]
    manifest["token_manager_plan"]["layout"]["plan_fingerprint"] = (
        verifier._layout_fingerprint_from_manager_input(
            verifier._manager_input_from_attention_source(manifest, "manifest"),
            "manager input",
        )
    )
    manifest["fingerprint"] = _fingerprint(manifest)
    signature = binding["execution_signature"]
    signature["manifest_fingerprint"] = manifest["fingerprint"]
    signature["fingerprint"] = _fingerprint(signature)
    binding["manifest_fingerprint"] = manifest["fingerprint"]
    binding["fingerprint"] = _fingerprint(binding)
    manager_input = verifier._manager_input_fingerprint(manifest, "manifest")
    for snapshot in record["manager"]["snapshots"]:
        snapshot["manager_input_fingerprint"] = manager_input
        snapshot["runtime_manifest_fingerprint"] = manifest["fingerprint"]
        snapshot["runtime_binding_fingerprint"] = binding["fingerprint"]


def test_valid_pair_reports_descriptive_ratios_without_speedup_claim() -> None:
    stock = _record("stock", (2.0, 4.0, 6.0))
    manager = _record("manager", (1.0, 3.0, 3.0))

    result = verifier.verify_pair(stock, manager)

    assert result["manager_to_stock_latency_ratios"] == [0.5, 0.75, 0.5]
    assert result["latency_ratio_median"] == 0.5
    assert result["latency_ratio_p95"] == pytest.approx(0.725)
    assert result["manager_to_stock_output_throughput_ratio"] == pytest.approx(12 / 7)
    assert result["observed_direction"] == "manager_lower_latency"
    assert "speedup" not in result


def test_schema_v3_and_default_shared_full_profile() -> None:
    record = _record("manager")

    assert verifier.RECORD_SCHEMA == "orbitkv.sglang-engine-e2e.v3"
    assert verifier.VERIFICATION_SCHEMA == (
        "orbitkv.sglang-engine-e2e-pairs-verification.v3"
    )
    assert record["manager"]["snapshots"][0]["lifecycle_route"] == (
        "native_session"
    )
    assert record["manager"]["snapshots"][0]["cache_policy"] == (
        "shared_prefix"
    )
    verifier.validate_record(record)
    for snapshot in record["manager"]["snapshots"]:
        assert snapshot["swa_activity"] == {
            "status": "not_applicable",
            "applicable": False,
            "source": "native_runtime_session",
            "derived": False,
            **{name: 0 for name in verifier.SWA_COUNTER_FIELDS},
        }


def test_full_sliding_native_session_profile_with_two_arenas_is_valid() -> None:
    record = _record("manager", hybrid=True)

    validated = verifier.validate_record(record)

    assert validated["mode"] == "manager"
    signature = record["runtime_binding"]["execution_signature"]
    assert record["runtime_binding"]["execution_topology"] == (
        verifier.FULL_SLIDING_TOKEN_KV_TOPOLOGY
    )
    assert [item["backend"]["retention"] for item in signature["token_states"]] == [
        "full",
        "sliding",
    ]
    assert [item["class_id"] for item in record["manager"]["snapshots"][0]["identities"]] == [0, 1]
    for index, snapshot in enumerate(record["manager"]["snapshots"]):
        activity = snapshot["swa_activity"]
        assert set(activity) == set(verifier.SWA_ACTIVITY_KEYS)
        assert len(activity) == 8
        assert activity["status"] == "exposed"
        assert activity["applicable"] is True
        assert activity["source"] == "native_runtime_session"
        assert activity["derived"] is False
        expected_count = (0, 10, 20, 20)[index]
        assert all(
            activity[name] == expected_count
            for name in verifier.SWA_COUNTER_FIELDS
        )


def test_pure_sliding_native_session_profile_is_request_private() -> None:
    record = _record("manager", pure_sliding=True)

    validated = verifier.validate_record(record)

    assert validated["mode"] == "manager"
    signature = record["runtime_binding"]["execution_signature"]
    assert verifier._native_session_contract(signature, "record") == (
        "whole_domain_sliding_token_kv",
        "request_private",
    )
    assert record["runtime_binding"]["execution_topology"] == (
        "whole_domain_sliding_token_kv"
    )
    assert [
        (item["backend"]["retention"], item["backend"]["storage"])
        for item in signature["token_states"]
    ] == [("sliding", "token_kv")]
    assert signature["token_classes"] == [
        {
            "name": "swa",
            "layers": [0, 1],
            "bytes_per_token_per_layer": 96,
            "address": {"kind": "periodic", "period_blocks": 3},
            "retirement": {"kind": "block_end_plus", "offset_tokens": 17},
            "minimum_slots_per_request": 3,
        }
    ]
    assert [
        item["class_id"]
        for item in record["manager"]["snapshots"][0]["identities"]
    ] == [0]
    assert record["manager"]["snapshots"][0]["cache_policy"] == (
        "request_private"
    )
    assert record["engine_args"]["disable_radix_cache"] is True
    assert all(
        snapshot["swa_activity"]["source"] == "native_runtime_session"
        and snapshot["swa_activity"]["derived"] is False
        and snapshot["swa_activity"]["applicable"] is True
        for snapshot in record["manager"]["snapshots"]
    )


@pytest.mark.parametrize(
    "profile",
    ({"hybrid": True}, {"pure_sliding": True}),
    ids=("full-plus-sliding", "pure-sliding"),
)
@pytest.mark.parametrize("counter", verifier.SWA_COUNTER_FIELDS)
def test_sliding_profiles_require_all_four_swa_counters_to_progress(
    profile: dict[str, bool], counter: str
) -> None:
    record = _record("manager", **profile)
    for snapshot in record["manager"]["snapshots"]:
        snapshot["swa_activity"][counter] = 0

    with pytest.raises(RuntimeError, match="Sliding workload did not advance"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "profile",
    ({"hybrid": True}, {"pure_sliding": True}),
    ids=("full-plus-sliding", "pure-sliding"),
)
@pytest.mark.parametrize("counter", verifier.SWA_COUNTER_FIELDS)
def test_sliding_progress_is_measured_after_warmup(
    profile: dict[str, bool], counter: str
) -> None:
    record = _record("manager", **profile)
    snapshots = record["manager"]["snapshots"]
    warmup_value = snapshots[1]["swa_activity"][counter]
    snapshots[2]["swa_activity"][counter] = warmup_value
    snapshots[3]["swa_activity"][counter] = warmup_value

    with pytest.raises(RuntimeError, match="Sliding workload did not advance"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "profile",
    ({"hybrid": True}, {"pure_sliding": True}),
    ids=("full-plus-sliding", "pure-sliding"),
)
@pytest.mark.parametrize("counter", verifier.SWA_COUNTER_FIELDS)
@pytest.mark.parametrize("after_index", (1, 2, 3))
def test_sliding_profiles_require_swa_counters_to_remain_monotonic(
    profile: dict[str, bool], counter: str, after_index: int
) -> None:
    record = _record("manager", **profile)
    snapshots = record["manager"]["snapshots"]
    if after_index == 1:
        snapshots[0]["swa_activity"][counter] = 2
        snapshots[1]["swa_activity"][counter] = 1
    elif after_index == 2:
        snapshots[1]["swa_activity"][counter] = 21
    else:
        snapshots[3]["swa_activity"][counter] = 19

    with pytest.raises(RuntimeError, match="SWA counters decreased"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "not_applicable"),
        ("applicable", False),
        ("applicable", 1),
        ("source", "derived"),
        ("derived", True),
        ("derived", 0),
    ),
)
@pytest.mark.parametrize(
    "profile",
    ({"hybrid": True}, {"pure_sliding": True}),
    ids=("full-plus-sliding", "pure-sliding"),
)
def test_sliding_swa_metadata_must_be_direct_native_session_evidence(
    profile: dict[str, bool], field: str, value: object
) -> None:
    record = _record("manager", **profile)
    record["manager"]["snapshots"][2]["swa_activity"][field] = value

    with pytest.raises(RuntimeError, match="SWA telemetry differs"):
        verifier.validate_record(record)


@pytest.mark.parametrize("field", tuple(verifier.SWA_ACTIVITY_KEYS))
def test_swa_activity_requires_every_native_schema_key(field: str) -> None:
    missing = _record("manager", pure_sliding=True)
    del missing["manager"]["snapshots"][2]["swa_activity"][field]
    with pytest.raises(RuntimeError, match="swa_activity keys differ"):
        verifier.validate_record(missing)


def test_swa_activity_rejects_extra_schema_key() -> None:
    extra = _record("manager", pure_sliding=True)
    extra["manager"]["snapshots"][2]["swa_activity"]["inferred"] = True
    with pytest.raises(RuntimeError, match="swa_activity keys differ"):
        verifier.validate_record(extra)


@pytest.mark.parametrize("counter", verifier.SWA_COUNTER_FIELDS)
@pytest.mark.parametrize("value", (-1, True, 1.0, "1"))
def test_swa_activity_rejects_invalid_counter_values(
    counter: str, value: object
) -> None:
    record = _record("manager", pure_sliding=True)
    record["manager"]["snapshots"][2]["swa_activity"][counter] = value

    with pytest.raises(RuntimeError, match="invalid counter"):
        verifier.validate_record(record)


def test_pure_sliding_rejects_shared_prefix_cache_policy() -> None:
    record = _record(
        "manager", pure_sliding=True, cache_policy="shared_prefix"
    )

    with pytest.raises(RuntimeError, match="admitted native-session topology"):
        verifier.validate_record(record)


def test_pure_sliding_contract_rejects_non_singleton_shape() -> None:
    record = _record("manager", pure_sliding=True)
    signature = record["runtime_binding"]["execution_signature"]
    signature["token_states"].append(copy.deepcopy(signature["token_states"][0]))

    with pytest.raises(RuntimeError, match="exact admitted native-session profile"):
        verifier._native_session_contract(signature, "record")


def test_native_session_contract_rejects_sliding_before_full_order() -> None:
    record = _record("manager", hybrid=True)
    signature = record["runtime_binding"]["execution_signature"]
    signature["token_states"].reverse()

    with pytest.raises(RuntimeError, match="exact admitted native-session profile"):
        verifier._native_session_contract(signature, "record")


@pytest.mark.parametrize(
    "field",
    ("address", "retirement", "minimum_slots_per_request"),
)
def test_pure_sliding_periodic_window_geometry_is_exact(field: str) -> None:
    record = _record("manager", pure_sliding=True)
    manifest_class = record["runtime_manifest"]["token_manager_plan"]["layout"][
        "classes"
    ][0]
    signature_class = record["runtime_binding"]["execution_signature"][
        "token_classes"
    ][0]
    if field == "address":
        manifest_class[field]["period_blocks"] = 2
        signature_class[field]["period_blocks"] = 2
    elif field == "retirement":
        manifest_class[field]["offset_tokens"] = 18
        signature_class[field]["offset_tokens"] = 18
    else:
        manifest_class[field] = 2
        signature_class[field] = 2
    _reseal_runtime(record)

    with pytest.raises(RuntimeError, match="Sliding token_kv geometry differs"):
        verifier.validate_record(record)


def test_pure_sliding_window_must_be_positive() -> None:
    record = _record("manager", pure_sliding=True)
    manifest = record["runtime_manifest"]
    manifest["source"]["input"]["states"][0]["storage"][
        "window_tokens"
    ] = 0
    manifest["fingerprint"] = _fingerprint(manifest)
    signature = record["runtime_binding"]["execution_signature"]
    signature["manifest_fingerprint"] = manifest["fingerprint"]
    signature["fingerprint"] = _fingerprint(signature)
    record["runtime_binding"]["manifest_fingerprint"] = manifest["fingerprint"]
    record["runtime_binding"]["fingerprint"] = _fingerprint(
        record["runtime_binding"]
    )

    with pytest.raises(RuntimeError, match="window_tokens must be positive"):
        verifier.validate_record(record)


def test_full_latent_native_session_profile_is_request_private() -> None:
    record = _record("manager", latent=True)

    verifier.validate_record(record)

    assert record["runtime_binding"]["execution_topology"] == (
        verifier.FULL_LATENT_KV_TOPOLOGY
    )
    assert record["manager"]["snapshots"][0]["cache_policy"] == (
        "request_private"
    )
    assert record["engine_args"]["disable_radix_cache"] is True
    assert all(
        snapshot["swa_activity"]
        == {
            "status": "not_applicable",
            "applicable": False,
            "source": "native_runtime_session",
            "derived": False,
            **{name: 0 for name in verifier.SWA_COUNTER_FIELDS},
        }
        for snapshot in record["manager"]["snapshots"]
    )


@pytest.mark.parametrize(
    "profile", ({}, {"latent": True}), ids=("full", "full-latent")
)
@pytest.mark.parametrize("counter", verifier.SWA_COUNTER_FIELDS)
def test_non_sliding_profiles_reject_nonzero_swa_counters(
    profile: dict[str, bool], counter: str
) -> None:
    record = _record("manager", **profile)
    record["manager"]["snapshots"][2]["swa_activity"][counter] = 1

    with pytest.raises(RuntimeError, match="non-Sliding SWA counters are nonzero"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "exposed"),
        ("applicable", True),
        ("applicable", 0),
        ("source", "derived"),
        ("derived", True),
        ("derived", 0),
    ),
)
@pytest.mark.parametrize(
    "profile", ({}, {"latent": True}), ids=("full", "full-latent")
)
def test_non_sliding_swa_metadata_must_report_native_not_applicable(
    profile: dict[str, bool], field: str, value: object
) -> None:
    record = _record("manager", **profile)
    record["manager"]["snapshots"][2]["swa_activity"][field] = value

    with pytest.raises(RuntimeError, match="SWA telemetry differs"):
        verifier.validate_record(record)


def test_cache_policy_must_match_admitted_native_session_topology() -> None:
    full_private = _record("manager", cache_policy="request_private")
    with pytest.raises(RuntimeError, match="admitted native-session topology"):
        verifier.validate_record(full_private)

    latent_shared = _record("manager", latent=True, cache_policy="shared_prefix")
    with pytest.raises(RuntimeError, match="admitted native-session topology"):
        verifier.validate_record(latent_shared)


def test_shared_prefix_may_survive_workload_but_final_must_drain() -> None:
    record = _record("manager", retained_prefix=True)
    verifier.validate_record(record)

    final = record["manager"]["snapshots"][3]
    final["manager_stats"]["active_prefixes"] = 1
    final["manager_stats"]["active_pages"] = 1
    final["manager_stats"]["free_pages"] -= 1
    final["manager_stats"]["total_prefix_page_refs"] = 1
    final["arena_stats"][0]["active_pages"] = 1
    final["arena_stats"][0]["free_pages"] -= 1
    final["arena_stats"][0]["prefix_page_refs"] = 1
    with pytest.raises(RuntimeError, match="lifecycle did not drain"):
        verifier.validate_record(record)


def test_request_private_rejects_prefix_residency() -> None:
    record = _record(
        "manager", cache_policy="request_private", retained_prefix=True
    )

    with pytest.raises(RuntimeError, match="request-private prefix state"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("total_request_page_refs", "lifecycle did not drain"),
        ("total_reader_pins", "lifecycle did not drain"),
        ("retiring_pages", "lifecycle did not drain"),
        ("quarantined_pages", "lifecycle did not drain"),
    ),
)
def test_shared_prefix_rejects_unsafe_workload_residency(
    field: str, message: str
) -> None:
    record = _record("manager")
    workload = record["manager"]["snapshots"][2]
    workload["manager_stats"][field] = 1
    record["manager"]["post_workload_residency"][field] = 1
    arena_field = {
        "total_request_page_refs": "request_page_refs",
        "total_reader_pins": "reader_pins",
        "retiring_pages": "retiring_pages",
        "quarantined_pages": "quarantined_pages",
    }[field]
    workload["arena_stats"][0][arena_field] = 1
    if field in ("retiring_pages", "quarantined_pages"):
        workload["arena_stats"][0]["free_pages"] -= 1
        workload["manager_stats"]["free_pages"] -= 1

    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("route", "message"),
    (("canonical_manager", "not native_session"), ("unknown", "not native_session")),
)
def test_lifecycle_route_is_validated_and_native_session_is_required(
    route: str, message: str
) -> None:
    record = _record("manager")
    for snapshot in record["manager"]["snapshots"]:
        snapshot["lifecycle_route"] = route

    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_route_and_cache_policy_must_be_stable_across_snapshots() -> None:
    route = _record("manager")
    route["manager"]["snapshots"][2]["lifecycle_route"] = "canonical_manager"
    with pytest.raises(RuntimeError, match="not native_session"):
        verifier.validate_record(route)

    policy = _record("manager")
    policy["manager"]["snapshots"][2]["cache_policy"] = "request_private"
    with pytest.raises(RuntimeError, match="cache_policy changed"):
        verifier.validate_record(policy)


def test_cache_policy_is_checked_against_engine_configuration() -> None:
    record = _record("manager")
    record["engine_args"]["disable_radix_cache"] = True
    for snapshot in record["server_snapshots"]:
        snapshot["resolved_engine"]["disable_radix_cache"] = True

    with pytest.raises(RuntimeError, match="cache policy differs"):
        verifier.validate_record(record)


def test_hybrid_profile_requires_order_and_positive_sliding_window() -> None:
    reversed_profile = _record("manager", hybrid=True)
    signature = reversed_profile["runtime_binding"]["execution_signature"]
    signature["token_classes"].reverse()
    signature["token_states"].reverse()
    signature["fingerprint"] = _fingerprint(signature)
    reversed_profile["runtime_binding"]["fingerprint"] = _fingerprint(
        reversed_profile["runtime_binding"]
    )
    for snapshot in reversed_profile["manager"]["snapshots"]:
        snapshot["runtime_binding_fingerprint"] = reversed_profile["runtime_binding"][
            "fingerprint"
        ]
    with pytest.raises(
        RuntimeError,
        match="token state differs|full token projection|exact admitted native-session profile",
    ):
        verifier.validate_record(reversed_profile)

    zero_window = _record("manager", hybrid=True)
    zero_window["runtime_manifest"]["source"]["input"]["states"][1][
        "storage"
    ]["window_tokens"] = 0
    zero_window["runtime_manifest"]["attention_state_plan"]["states"][1][
        "backend"
    ]["window_tokens"] = 0
    zero_window["runtime_binding"]["execution_signature"]["token_states"][1][
        "backend"
    ]["window_tokens"] = 0
    zero_window["runtime_manifest"]["fingerprint"] = _fingerprint(
        zero_window["runtime_manifest"]
    )
    signature = zero_window["runtime_binding"]["execution_signature"]
    signature["manifest_fingerprint"] = zero_window["runtime_manifest"][
        "fingerprint"
    ]
    signature["fingerprint"] = _fingerprint(signature)
    zero_window["runtime_binding"]["manifest_fingerprint"] = zero_window[
        "runtime_manifest"
    ]["fingerprint"]
    zero_window["runtime_binding"]["fingerprint"] = _fingerprint(
        zero_window["runtime_binding"]
    )
    with pytest.raises(RuntimeError, match="must be positive"):
        verifier.validate_record(zero_window)


def test_two_arena_aggregate_census_is_required() -> None:
    record = _record("manager", hybrid=True)
    record["manager"]["snapshots"][2]["manager_stats"]["free_pages"] -= 1

    with pytest.raises(RuntimeError, match="aggregate free_pages"):
        verifier.validate_record(record)


def test_hybrid_arena_class_ids_and_ranges_are_exact() -> None:
    wrong_class = _record("manager", hybrid=True)
    wrong_class["manager"]["snapshots"][2]["identities"][1]["class_id"] = 0
    wrong_class["manager"]["snapshots"][2]["arena_stats"][1]["class_id"] = 0
    with pytest.raises(RuntimeError, match="class_id must be 1"):
        verifier.validate_record(wrong_class)

    overlapping = _record("manager", hybrid=True)
    for snapshot in overlapping["manager"]["snapshots"]:
        snapshot["identities"][1]["backend_base_index"] = 7
    with pytest.raises(RuntimeError, match="backend index ranges overlap"):
        verifier.validate_record(overlapping)


def test_completion_frontier_must_name_an_admitted_arena_domain() -> None:
    record = _record("manager")
    record["manager"]["snapshots"][2]["completion_evidence"][
        "completion_high_water"
    ] = [{"domain": 99, "value": 20}]

    with pytest.raises(RuntimeError, match="unknown domain"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(schema="unknown"), "schema"),
        (lambda value: value.update(extra=True), "keys differ"),
        (lambda value: value["timings"]["iteration_seconds"].__setitem__(0, 0.0), "positive"),
        (lambda value: value["outputs"]["iterations"][0]["output_ids"].__setitem__(0, 999), "output digest"),
        (lambda value: value["server_snapshots"][0].update(orbitkv_manager_present=True), "presence"),
    ),
)
def test_record_tampering_is_rejected(mutate: Any, message: str) -> None:
    record = _record("stock")
    mutate(record)
    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_output_mismatch_is_rejected_even_with_recomputed_digests() -> None:
    stock = _record("stock")
    manager = _record("manager")
    row = manager["outputs"]["iterations"][1]
    row["output_ids"] = [700, 701]
    row["output_ids_sha256"] = verifier._canonical_digest(row["output_ids"])
    _reseal_outputs(manager)

    with pytest.raises(RuntimeError, match="output token IDs differ at iteration 1"):
        verifier.verify_pair(stock, manager)


def test_request_id_mode_prefix_is_normalized_but_input_ids_are_not() -> None:
    stock = _record("stock")
    manager = _record("manager")
    manager["workload"]["request_ids"]["measured"][0] = (
        "orbitkv-engine-e2e-measured-7-0"
    )
    verifier.verify_pair(stock, manager)

    manager["workload"]["input_ids"]["measured"][0][1] = 999
    manager["workload"]["input_ids_sha256"]["measured"][0] = (
        verifier._canonical_digest(
            manager["workload"]["input_ids"]["measured"][0]
        )
    )
    manager["outputs"]["iterations"][0]["input_ids_sha256"] = (
        manager["workload"]["input_ids_sha256"]["measured"][0]
    )
    _reseal_outputs(manager)
    with pytest.raises(RuntimeError, match="workload and input IDs differ"):
        verifier.verify_pair(stock, manager)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value["source"].update(revision="b" * 40), "release and revision"),
        (lambda value: value["checkpoint"].update(config_sha256="f" * 64), "checkpoint identity"),
        (lambda value: value["engine_args"].update(context_length=80), "engine arguments"),
        (lambda value: value["environment"].update(CUDA_VISIBLE_DEVICES="1"), "environment"),
    ),
)
def test_pair_contract_mismatch_is_rejected(mutate: Any, message: str) -> None:
    stock = _record("stock")
    manager = _record("manager")
    mutate(manager)
    with pytest.raises(RuntimeError, match=message):
        verifier.verify_pair(stock, manager)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.pop("accelerator"), "keys differ"),
        (
            lambda value: value["accelerator"].update(device_name=""),
            "device_name",
        ),
        (
            lambda value: value["accelerator"].update(
                compute_capability={"major": True, "minor": 0}
            ),
            "must be an integer",
        ),
        (
            lambda value: value["accelerator"].update(total_memory_bytes=0),
            "must be positive",
        ),
        (
            lambda value: value["accelerator"].update(runtime_version=13.0),
            "runtime_version",
        ),
        (
            lambda value: value["accelerator"].update(driver_version="unknown"),
            "dotted numeric version",
        ),
    ),
)
def test_accelerator_provenance_is_strict(
    mutate: Any, message: str
) -> None:
    record = _record("manager")
    mutate(record)
    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_pair_requires_the_same_accelerator() -> None:
    stock = _record("stock")
    manager = _record("manager")
    manager["accelerator"]["total_memory_bytes"] += 1
    with pytest.raises(RuntimeError, match="accelerator identity"):
        verifier.verify_pair(stock, manager)


def test_manager_owner_completion_progress_and_final_drain_are_required() -> None:
    mutations = []
    owner = _record("manager")
    owner["manager"]["snapshots"][2]["direct_source_owner"]["allocator_owned"] = False
    mutations.append((owner, "owner proof"))

    completion = _record("manager")
    completion["manager"]["snapshots"][2]["completion_evidence"]["pending_events"] = 1
    mutations.append((completion, "completion_evidence"))

    progress = _record("manager")
    progress["manager"]["snapshots"][2]["batch_counters"] = copy.deepcopy(
        progress["manager"]["snapshots"][1]["batch_counters"]
    )
    mutations.append((progress, "measured lifecycle did not advance"))

    drain = _record("manager")
    drain["manager"]["snapshots"][3]["manager_stats"]["active_requests"] = 1
    mutations.append((drain, "lifecycle did not drain"))

    for record, message in mutations:
        with pytest.raises(RuntimeError, match=message):
            verifier.validate_record(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(
                schema="orbitkv.sglang-engine-e2e.v1"
            ),
            "schema",
        ),
        (
            lambda value: value["environment"].update(
                ORBITKV_STRUCTURED_DATA_PLANE="1"
            ),
            "keys differ",
        ),
        (
            lambda value: value["manager"]["snapshots"][2].update(
                lifecycle_authority="runtime_session"
            ),
            "keys differ",
        ),
        (
            lambda value: value["manager"]["snapshots"][2].update(
                cache_policy="unknown"
            ),
            "cache policy",
        ),
        (
            lambda value: value["manager"]["snapshots"][2][
                "completion_evidence"
            ].update(event_backend="external_last_use_cuda_event"),
            "completion backend",
        ),
        (
            lambda value: value["manager"]["snapshots"][2][
                "swa_activity"
            ].update(status="exposed"),
            "SWA telemetry",
        ),
        (
            lambda value: value["manager"]["snapshots"][2]["pressure"].update(
                enabled=True
            ),
            "pressure",
        ),
        (
            lambda value: value["manager"]["snapshots"][2][
                "batch_counters"
            ].update(prepare_batch_calls=1),
            "counter keys differ",
        ),
        (
            lambda value: value["manager"]["post_workload_residency"].update(
                active_pages=1
            ),
            "residency",
        ),
    ),
)
def test_manager_session_contract_drift_is_rejected(
    mutate: Any, message: str
) -> None:
    record = _record("manager")
    mutate(record)
    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value["manager"]["snapshots"][2][
                "batch_counters"
            ].update(completion_values=19),
            "completion counters differ",
        ),
        (
            lambda value: value["manager"]["snapshots"][2][
                "completion_evidence"
            ]["completion_high_water"].append({"domain": 4, "value": 21}),
            "domain is duplicated",
        ),
        (
            lambda value: value["manager"]["snapshots"][0][
                "batch_counters"
            ].update(event_queries=1),
            "not empty after load",
        ),
        (
            lambda value: value["manager"]["snapshots"][3][
                "batch_counters"
            ].update(event_queries=1),
            "counters decreased",
        ),
        (
            lambda value: value["manager"]["snapshots"][3][
                "completion_evidence"
            ].update(completion_high_water=[{"domain": 4, "value": 19}]),
            "frontier regressed",
        ),
        (
            lambda value: (
                value["manager"]["snapshots"][3]["identities"][0].update(
                    pool_id=7
                ),
                value["manager"]["snapshots"][3]["arena_stats"][0].update(
                    pool_id=7
                ),
            ),
            "identity changed",
        ),
        (
            lambda value: value["manager"]["snapshots"][2].update(
                manager_input_fingerprint="sha256:" + "9" * 64
            ),
            "provenance differs",
        ),
    ),
)
def test_session_evidence_consistency_is_required(
    mutate: Any, message: str
) -> None:
    record = _record("manager")
    mutate(record)
    with pytest.raises(RuntimeError, match=message):
        verifier.validate_record(record)


def test_current_wire_version_is_required() -> None:
    record = _record("manager")
    record["runtime_binding"]["required_wire_version"] = 10
    record["runtime_binding"]["fingerprint"] = _fingerprint(
        record["runtime_binding"]
    )
    record["manager"]["wire_version"] = 10
    for snapshot in record["manager"]["snapshots"]:
        snapshot["wire_version"] = 10
        snapshot["runtime_binding_fingerprint"] = record["runtime_binding"][
            "fingerprint"
        ]
    with pytest.raises(RuntimeError, match="required_wire_version"):
        verifier.validate_record(record)


@pytest.mark.parametrize(
    "target",
    (
        {"id": "sglang@4", "contract_version": 4},
        {
            "id": "sglang",
            "contract_version": verifier.TARGET_CONTRACT_VERSION - 1,
        },
    ),
    ids=("version-in-id", "stale-contract-version"),
)
def test_runtime_target_identity_is_exact(target: dict[str, object]) -> None:
    record = _record("manager")
    record["runtime_binding"]["target"] = target
    record["runtime_binding"]["fingerprint"] = _fingerprint(
        record["runtime_binding"]
    )
    for snapshot in record["manager"]["snapshots"]:
        snapshot["runtime_binding_fingerprint"] = record["runtime_binding"][
            "fingerprint"
        ]

    with pytest.raises(RuntimeError, match="target identity differs"):
        verifier.validate_record(record)


def test_three_pairs_report_only_scoped_consistent_direction(tmp_path: Path) -> None:
    paths = []
    for index, manager_seconds in enumerate(((1.0, 2.0, 3.0), (1.5, 3.0, 4.5), (1.8, 3.6, 5.4))):
        stock_path = tmp_path / f"stock-{index}.json"
        manager_path = tmp_path / f"manager-{index}.json"
        _write(stock_path, _record("stock", (2.0, 4.0, 6.0)))
        _write(manager_path, _record("manager", manager_seconds))
        paths.append((stock_path, manager_path))

    before = {path: path.read_bytes() for pair in paths for path in pair}
    result = verifier.verify_pairs(paths)

    assert result["schema"] == verifier.VERIFICATION_SCHEMA
    assert result["pair_count"] == 3
    assert result["speedup_qualified"] is False
    assert result["aggregate"]["available"] is True
    assert result["aggregate"]["all_pairs_same_direction"] is True
    assert result["aggregate"]["conservative_direction_flag"] == "manager_lower_latency"
    assert result["aggregate"]["scope"] == "descriptive_scoped_evidence_not_a_general_performance_claim"
    assert {path: path.read_bytes() for pair in paths for path in pair} == before


def test_mixed_pair_directions_have_no_conservative_flag(tmp_path: Path) -> None:
    paths = []
    for index, manager_seconds in enumerate(((1.0, 2.0, 3.0), (3.0, 6.0, 9.0), (2.0, 4.0, 6.0))):
        stock_path = tmp_path / f"stock-{index}.json"
        manager_path = tmp_path / f"manager-{index}.json"
        _write(stock_path, _record("stock"))
        _write(manager_path, _record("manager", manager_seconds))
        paths.append((stock_path, manager_path))

    aggregate = verifier.verify_pairs(paths)["aggregate"]
    assert aggregate["all_pairs_same_direction"] is False
    assert aggregate["conservative_direction_flag"] is None


def test_strict_json_and_cli_behavior(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate JSON object key"):
        verifier._strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-finite JSON number"):
        verifier._strict_json(nonfinite)

    stock_path = tmp_path / "stock.json"
    manager_path = tmp_path / "manager.json"
    _write(stock_path, _record("stock"))
    _write(manager_path, _record("manager"))
    assert verifier.main(["--pair", str(stock_path), str(manager_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "passed"
    assert captured.err == ""

    with pytest.raises(SystemExit) as raised:
        verifier.main(["--pair", str(manager_path), str(stock_path)])
    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert captured.out == ""
    assert "requires STOCK then MANAGER" in captured.err
