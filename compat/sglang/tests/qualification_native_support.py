"""Host-only fixtures for native Sliding qualification tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from manifest_test_support import compile_runtime_manifest


PAGE_TOKENS = 16


def write_sliding_manifest(tmp_path: Path, *, hybrid: bool) -> Path:
    classes: list[dict[str, Any]] = []
    if hybrid:
        classes.append({
            "name": "full", "layers": [1, 3], "retention": "full",
            "bytes_per_token_per_layer": 128, "window_tokens": None,
        })
        sliding_layers = [0, 2]
    else:
        sliding_layers = [0, 1, 2, 3]
    classes.append({
        "name": "swa", "layers": sliding_layers,
        "retention": "sliding", "bytes_per_token_per_layer": 128,
        "window_tokens": 32,
    })
    return compile_runtime_manifest(
        tmp_path, manager_plan={"page_tokens": PAGE_TOKENS, "classes": classes},
        stem="hybrid" if hybrid else "sliding",
    )


def native_manager_info(bench: Any, *, active: bool, hybrid: bool = False) -> dict[str, object]:
    class_count = 2 if hybrid else 1
    identities = []
    arenas = []
    for class_id in range(class_count):
        identity = {
            "engine_epoch": 1, "pool_epoch": 2 + class_id,
            "pool_id": 3 + class_id, "class_id": class_id,
            "backend_domain": 4 + class_id, "page_count": 8,
            "page_tokens": PAGE_TOKENS, "backend_base_index": class_id * 8,
            "first_page_id": 1 + class_id * 8,
        }
        arena = {name: 0 for name in bench._ARENA_FIELDS}
        arena.update(identity)
        arena.pop("page_tokens")
        arena.pop("backend_base_index")
        arena.update(page_count=8, free_pages=8)
        identities.append(identity)
        arenas.append(arena)
    stats = {name: 0 for name in bench._MANAGER_STATS_FIELDS}
    stats["free_pages"] = 8 * class_count
    counters = {name: 0 for name in bench.runtime_support.SESSION_COUNTER_FIELDS}
    if active:
        counters.update(forward_events=3, completion_values=3, event_queries=3)
    swa = {
        "status": "exposed", "applicable": True,
        "source": "native_runtime_session", "derived": False,
        "swa_retirement_certificates": 2 if active else 0,
        "swa_pages_reclaimed": 2 if active else 0,
        "swa_wrap_events": 1 if active else 0,
        "swa_page_reuse_events": 1 if active else 0,
    }
    return {"internal_states": [{"orbitkv_manager": {
        "lifecycle_route": "native_session",
        "cache_policy": "shared_prefix" if hybrid else "request_private",
        "wire_version": bench.WIRE_VERSION,
        "manager_input_fingerprint": "sha256:manager-input",
        "runtime_manifest_fingerprint": "sha256:manifest",
        "runtime_binding_fingerprint": "sha256:binding",
        "runtime_proof": None, "identities": identities,
        "manager_stats": stats, "arena_stats": arenas,
        "batch_counters": counters, "swa_activity": swa,
        "completion_evidence": {
            "event_backend": "cuda_event_current_forward_stream",
            "pending_events": 0,
            "completion_high_water": ([{"domain": 4, "value": 3}] if active else []),
        },
        "pressure": {
            "schema": "orbitkv.runtime-pressure.v1", "enabled": False,
            "mode": "event_driven_high_water", "sample_count": 0,
        },
    }}]}
