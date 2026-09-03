"""Host-only helpers shared by the active qualification runner."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
SGLANG_ROOT = TOOLS_ROOT.parent
REPOSITORY_ROOT = SGLANG_ROOT.parents[1]
BRIDGE_ROOT = SGLANG_ROOT / "bridge"
ADAPTER_PACKAGE_ROOT = BRIDGE_ROOT / "src/orbitkv_sglang"
_LEGACY_MANAGER_ENTRYPOINT = "orbitkv_manager"
EXACT_TOPOLOGY = "whole_domain_chunked_token_kv"
FULL_SLIDING_TOPOLOGY = "whole_domain_full_sliding_token_kv"
SLIDING_TOPOLOGY = "whole_domain_sliding_token_kv"
SWA_ACTIVITY_FIELDS = (
    "swa_retirement_certificates",
    "swa_pages_reclaimed",
    "swa_wrap_events",
    "swa_page_reuse_events",
)
SESSION_COUNTER_FIELDS = (
    "forward_events",
    "completion_values",
    "event_queries",
    "event_waits",
    "fail_stop_count",
)

BATCH_COUNTER_FIELDS = (
    "request_acquire_batch_calls",
    "request_fork_batch_calls",
    "prepare_batch_calls",
    "submit_batch_calls",
    "complete_batch_calls",
    "abort_steps_batch_calls",
    "quarantine_steps_batch_calls",
    "quarantine_submissions_batch_calls",
    "release_batch_calls",
    "acknowledge_reclamations_batch_calls",
    "recycle_requests_batch_calls",
    "prefix_lookup_batch_calls",
    "prefix_attach_batch_calls",
    "prefix_publish_batch_calls",
    "prefix_publish_release_batch_calls",
    "prefix_evict_batch_calls",
    "prefix_recycle_batch_calls",
    "token_views_batch_calls",
    "mark_token_dispositions_batch_calls",
    "prepare_relocation_batch_calls",
    "submit_relocation_batch_calls",
    "complete_relocation_batch_calls",
    "abort_relocations_batch_calls",
    "buffer_too_small_preflights",
    "retryable_conflicts",
    "fail_stops",
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "cold_workspace_allocations",
    "materialized_page_objects",
    "forward_events",
    "completion_values",
    "event_queries",
    "event_waits",
    "quarantine_count",
    "fail_stop_count",
    "prefix_matches",
    "prefix_hits",
    "prefix_publishes",
    "prefix_evictions",
    "prefix_evicted_full_tokens",
    "prefix_evicted_swa_tokens",
    "prefix_global_alias_scans",
    "cow_copy_intents",
    "cow_move_calls",
    "cow_copied_tokens",
    "mirror_validation_calls",
    "mirror_syncs",
    "token_disposition_batches",
    "token_policy_evictions",
    "relocation_batches",
    "relocation_moves",
    "relocation_reclaimed_pages",
    "relocation_copy_events",
    "relocation_copy_tokens",
    "fixed_state_prepares",
    "fixed_state_clears",
    "fixed_state_copies",
    "fixed_state_events",
    "fixed_state_retirements",
    "fixed_state_acks",
)


def positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def ceil_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def selected_profile(args: Any, allowed: tuple[str, ...]) -> str:
    value = getattr(args, "profile", EXACT_TOPOLOGY)
    if value not in allowed:
        raise ValueError("--profile is not a supported qualification topology")
    return value


def runtime_identity(
    sglang: Any, package: Path, engine_args: Mapping[str, Any], *,
    run_id: str, runtime_proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "run_id": run_id,
        "runtime_proof": None if runtime_proof is None else dict(runtime_proof),
        "python_executable": sys.executable,
        "python_version": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "sglang_version": sglang.__version__,
        "sglang_package": str(package),
        "kv_layout": "nhd",
        "attention_backend": engine_args["attention_backend"],
        "dtype": engine_args["dtype"],
        "kv_cache_dtype": engine_args["kv_cache_dtype"],
        "execution": "eager",
        "tp_size": engine_args["tp_size"],
        "pp_size": engine_args["pp_size"],
        "dp_size": engine_args["dp_size"],
        "dcp_size": engine_args["dcp_size"],
        "deterministic_inference": engine_args["enable_deterministic_inference"],
        "sampling_backend": engine_args["sampling_backend"],
    }
    if "moe_runner_backend" in engine_args:
        result["moe_backend"] = {
            "runner": engine_args["moe_runner_backend"],
            "a2a": engine_args["moe_a2a_backend"],
            "ep_size": engine_args["ep_size"],
        }
    return result


def sliding_geometry(binding: Mapping[str, Any]) -> dict[str, Any]:
    signature = binding.get("execution_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("target binding omitted its execution signature")
    classes = signature.get("token_classes")
    states = signature.get("token_states")
    topology = binding.get("execution_topology")
    expected = 2 if topology == FULL_SLIDING_TOPOLOGY else 1
    if (not isinstance(classes, list) or not isinstance(states, list)
            or len(classes) != expected or len(states) != expected):
        raise ValueError("Sliding binding has an invalid token-class shape")
    page_tokens = positive_integer(
        "manifest page tokens", signature.get("page_tokens")
    )
    result = []
    retentions = (
        ("full", "sliding")
        if topology == FULL_SLIDING_TOPOLOGY else ("sliding",)
    )
    for class_id, (item, state, retention) in enumerate(
        zip(classes, states, retentions, strict=True)
    ):
        if not isinstance(item, Mapping) or not isinstance(state, Mapping):
            raise ValueError("Sliding binding token class is malformed")
        backend = state.get("backend")
        if (not isinstance(backend, Mapping)
                or backend.get("storage") != "token_kv"
                or backend.get("retention") != retention):
            raise ValueError("Sliding binding token-state shape differs")
        window = None
        floor = None
        if retention == "sliding":
            window = positive_integer(
                "manifest sliding window", backend.get("window_tokens")
            )
            period = 1 + (window - 1 + page_tokens - 1) // page_tokens
            if (item.get("address")
                    != {"kind": "periodic", "period_blocks": period}
                    or item.get("retirement") != {
                        "kind": "block_end_plus",
                        "offset_tokens": window - 1,
                    }
                    or item.get("minimum_slots_per_request") != period):
                raise ValueError("Sliding binding periodic geometry differs")
            floor = period * page_tokens
        result.append({
            "class_id": class_id,
            "retention": retention,
            "layers": item.get("layers"),
            "window_tokens": window,
            "minimum_resident_tokens": floor,
        })
    return {"page_tokens": page_tokens, "classes": result}


def checkpoint_sliding_geometry(
    checkpoint: Mapping[str, Any], profile: str, *, page_tokens: int
) -> dict[str, Any]:
    window = positive_integer(
        "checkpoint sliding_window", checkpoint.get("sliding_window")
    )
    layers = positive_integer(
        "checkpoint num_hidden_layers", checkpoint.get("num_hidden_layers")
    )
    layer_types = checkpoint.get("layer_types")
    if layer_types is None:
        if profile != SLIDING_TOPOLOGY:
            raise RuntimeError(
                "Hybrid qualification requires explicit checkpoint layer_types"
            )
        sliding_layers, full_layers = list(range(layers)), []
    else:
        if (not isinstance(layer_types, list) or len(layer_types) != layers
                or any(item not in {"full_attention", "sliding_attention"}
                           for item in layer_types)):
            raise RuntimeError("checkpoint attention layer types are unsupported")
        full_layers = [i for i, item in enumerate(layer_types)
                       if item == "full_attention"]
        sliding_layers = [i for i, item in enumerate(layer_types)
                          if item == "sliding_attention"]
    if profile == FULL_SLIDING_TOPOLOGY:
        if not full_layers or not sliding_layers:
            raise RuntimeError(
                "Hybrid qualification requires both Full and Sliding layers"
            )
    elif profile == SLIDING_TOPOLOGY:
        if full_layers or sliding_layers != list(range(layers)):
            raise RuntimeError("pure Sliding qualification requires every layer to slide")
    else:
        raise RuntimeError("checkpoint Sliding geometry requires a Sliding profile")
    period = 1 + (window - 1 + page_tokens - 1) // page_tokens
    classes = []
    if full_layers:
        classes.append({
            "class_id": 0, "retention": "full", "layers": full_layers,
            "window_tokens": None, "minimum_resident_tokens": None,
        })
    classes.append({
        "class_id": len(classes), "retention": "sliding",
        "layers": sliding_layers, "window_tokens": window,
        "minimum_resident_tokens": period * page_tokens,
    })
    return {"page_tokens": page_tokens, "classes": classes}


def validate_native_workload(
    *, prompt_tokens: int, decode_tokens: int, iterations: int,
    profile_geometry: Mapping[str, Any],
) -> dict[str, int]:
    prompt = positive_integer("prompt tokens", prompt_tokens)
    decode = positive_integer("decode tokens", decode_tokens)
    count = positive_integer("iterations", iterations)
    positive_integer(
        "profile page tokens", profile_geometry.get("page_tokens")
    )
    classes = profile_geometry.get("classes")
    windows = [item.get("window_tokens") for item in classes
               if isinstance(item, Mapping) and item.get("retention") == "sliding"] \
        if isinstance(classes, list) else []
    if len(windows) != 1:
        raise ValueError("qualification requires exactly one Sliding class")
    window = positive_integer("sliding window", windows[0])
    sliding = next(
        item for item in classes
        if isinstance(item, Mapping) and item.get("retention") == "sliding"
    )
    period_tokens = positive_integer(
        "Sliding temporal period", sliding.get("minimum_resident_tokens")
    )
    materialized = prompt + decode - 1
    if materialized <= period_tokens:
        raise ValueError(
            "Sliding workload must cross at least one compiled temporal cycle"
        )
    return {
        "prompt_tokens": prompt, "decode_tokens": decode,
        "final_kv_tokens": materialized, "sliding_window_tokens": window,
        "retirement_boundary_crossings": (materialized - 1) // period_tokens,
        "iterations": count,
    }


def native_capacity_contract(
    *, case: str, profile: str, max_total_tokens: int,
    chunked_prefill_tokens: int, final_kv_tokens: int,
    profile_geometry: Mapping[str, Any], swa_full_tokens_ratio: float | None,
) -> dict[str, Any]:
    capacity = positive_integer("max total tokens", max_total_tokens)
    prefill = positive_integer("chunked prefill tokens", chunked_prefill_tokens)
    page = positive_integer("profile page tokens", profile_geometry.get("page_tokens"))
    classes = profile_geometry.get("classes")
    sliding = next((item for item in classes
                    if isinstance(item, Mapping) and item.get("retention") == "sliding"), None) \
        if isinstance(classes, list) else None
    if not isinstance(sliding, Mapping):
        raise ValueError("Sliding profile has no Sliding class")
    resident = positive_integer(
        "Sliding resident floor", sliding.get("minimum_resident_tokens")
    )
    # Match ClassConfig.minimum_sliding_pool_tokens for one live request:
    # periodic resident slots plus one page-aligned prefill staging region.
    sliding_floor = resident + ceil_to(prefill, page)
    if final_kv_tokens <= resident:
        raise ValueError(
            "Sliding workload does not cross a compiled temporal cycle"
        )
    if profile == SLIDING_TOPOLOGY:
        if case == "roomy" and capacity <= sliding_floor:
            raise ValueError("roomy pure Sliding requires capacity above the exact floor")
        if case == "exact-floor" and capacity != sliding_floor:
            raise ValueError(
                "exact-floor pure Sliding capacity differs from the compiled resident/staging floor"
            )
        if case not in {"roomy", "exact-floor"}:
            raise ValueError("--case must be roomy or exact-floor")
        return {
            "configured_max_total_tokens": capacity,
            "expected_full_tokens": 0,
            "expected_sliding_tokens": capacity,
            "full_floor_tokens": 0,
            "sliding_floor_tokens": sliding_floor,
        }
    if profile != FULL_SLIDING_TOPOLOGY:
        raise ValueError("native capacity contract requires a Sliding topology")
    if (
        swa_full_tokens_ratio is None
        or isinstance(swa_full_tokens_ratio, bool)
        or not isinstance(swa_full_tokens_ratio, (int, float))
        or not 0.0 < float(swa_full_tokens_ratio) <= 1.0
    ):
        raise ValueError("Hybrid qualification requires --swa-full-tokens-ratio")
    sliding_tokens = int(capacity * float(swa_full_tokens_ratio)) // page * page
    # The scheduler must be able to prepare the next decode write after the
    # currently materialized KV boundary.  A floor that fits only the existing
    # tokens truncates generation one token early at an aligned boundary.
    full_floor = max(ceil_to(final_kv_tokens + 1, page), sliding_floor)
    if sliding_tokens < sliding_floor:
        raise ValueError("Hybrid SWA capacity is below the compiled resident/staging floor")
    if case == "roomy" and (
        capacity < final_kv_tokens or sliding_tokens <= sliding_floor
    ):
        raise ValueError(
            "roomy Hybrid requires Full logical headroom and SWA headroom above the floor"
        )
    if case == "exact-floor" and (
        capacity != full_floor or sliding_tokens != sliding_floor
    ):
        raise ValueError("exact-floor Hybrid requires exact Full and SWA class floors")
    if case not in {"roomy", "exact-floor"}:
        raise ValueError("--case must be roomy or exact-floor")
    return {
        "configured_max_total_tokens": capacity,
        "expected_full_tokens": capacity,
        "expected_sliding_tokens": sliding_tokens,
        "full_floor_tokens": full_floor,
        "sliding_floor_tokens": sliding_floor,
    }


def validate_swa_activity(value: Any, stage: str) -> dict[str, Any]:
    fields = {"status", "applicable", "source", "derived", *SWA_ACTIVITY_FIELDS}
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError(f"OrbitKV native-session SWA activity is malformed at {stage}")
    if (value.get("status") != "exposed" or value.get("applicable") is not True
            or value.get("source") != "native_runtime_session"
            or value.get("derived") is not False):
        raise RuntimeError(
            f"OrbitKV native-session SWA activity is not directly exposed at {stage}"
        )
    counters = {name: value[name] for name in SWA_ACTIVITY_FIELDS}
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in counters.values()
    ):
        raise RuntimeError("swa_activity contains an invalid counter")
    return {
        "status": "exposed", "applicable": True,
        "source": "native_runtime_session", "derived": False, **counters,
    }


def validate_native_runtime_proof(
    value: Any, stage: str, engine_args: Mapping[str, Any]
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(
        value.get("effective_scheduler"), dict
    ):
        raise RuntimeError(f"OrbitKV runtime proof is malformed at {stage}")
    scheduler = value["effective_scheduler"]
    expected = {
        "max_prefill_tokens": engine_args["max_prefill_tokens"],
        "max_running_requests": engine_args["max_running_requests"],
        "effective_max_running_requests_per_dp": engine_args["max_running_requests"],
    }
    if any(scheduler.get(name) != item for name, item in expected.items()):
        raise RuntimeError(f"OrbitKV scheduler proof changed at {stage}")
    return dict(value)


def validate_completion_evidence(
    value: Any, stage: str, *, require_activity: bool
) -> dict[str, Any]:
    fields = {"event_backend", "pending_events", "completion_high_water"}
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError(f"OrbitKV completion evidence is malformed at {stage}")
    if value.get("event_backend") != "cuda_event_current_forward_stream":
        raise RuntimeError(f"OrbitKV completion backend changed at {stage}")
    if value.get("pending_events") != 0:
        raise RuntimeError(f"OrbitKV completion events did not drain at {stage}")
    frontier = value.get("completion_high_water")
    if not isinstance(frontier, list):
        raise RuntimeError(f"OrbitKV completion frontier is malformed at {stage}")
    result = []
    seen = set()
    for point in frontier:
        if not isinstance(point, dict) or set(point) != {"domain", "value"}:
            raise RuntimeError(f"OrbitKV completion point is malformed at {stage}")
        domain = positive_integer("completion domain", point["domain"])
        counter = positive_integer("completion value", point["value"])
        if domain in seen:
            raise RuntimeError(f"OrbitKV completion domain is duplicated at {stage}")
        seen.add(domain)
        result.append({"domain": domain, "value": counter})
    if result != sorted(result, key=lambda item: item["domain"]):
        raise RuntimeError(f"OrbitKV completion frontier is unordered at {stage}")
    if require_activity and not result:
        raise RuntimeError(f"OrbitKV completion frontier is empty at {stage}")
    return {
        "event_backend": value["event_backend"],
        "pending_events": 0,
        "completion_high_water": result,
    }


def require_swa_progress(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    for name in SWA_ACTIVITY_FIELDS:
        if after.get(name, -1) <= before.get(name, -1):
            raise RuntimeError(
                f"Sliding workload did not advance native {name}"
            )


def validate_profile_geometry_match(
    manifest: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    manifest_classes = manifest.get("classes")
    checkpoint_classes = checkpoint.get("classes")
    if (
        not isinstance(manifest_classes, list)
        or not isinstance(checkpoint_classes, list)
        or len(manifest_classes) != len(checkpoint_classes)
    ):
        raise RuntimeError(
            "RuntimeManifest Sliding classes differ from the checkpoint"
        )
    fields = ("class_id", "retention", "layers", "window_tokens")
    for left, right in zip(manifest_classes, checkpoint_classes, strict=True):
        if any(left.get(name) != right.get(name) for name in fields):
            raise RuntimeError(
                "RuntimeManifest Sliding classes differ from the checkpoint"
            )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tool_identity() -> dict[str, str]:
    executable = shutil.which("ninja")
    if executable is None:
        raise RuntimeError("the pinned attention-kernel path requires ninja on PATH")
    path = Path(executable).resolve(strict=True)
    expected_directory = Path(sys.executable).absolute().parent
    if path.parent != expected_directory:
        raise RuntimeError("ninja must come from the active Python environment")
    try:
        version = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot execute the pinned ninja build tool") from error
    if not version:
        raise RuntimeError("ninja returned an empty version")
    return {"path": str(path), "version": version, "sha256": _sha256_file(path)}


def adapter_identity() -> dict[str, Any]:
    files = sorted(
        {
            BRIDGE_ROOT / "pyproject.toml",
            TOOLS_ROOT / "prepare_source.py",
            SGLANG_ROOT / "overlay/adapter.patch",
            *ADAPTER_PACKAGE_ROOT.rglob("*.py"),
            *ADAPTER_PACKAGE_ROOT.rglob("*.json"),
        }
    )
    return {
        "files": [
            {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256_file(path),
            }
            for path in files
        ]
    }


def reject_legacy_manager_entrypoint() -> None:
    """Require direct-source qualification to have no legacy entrypoint."""

    try:
        discovered = importlib.metadata.entry_points(group="sglang.srt.plugins")
        matches = [
            entry
            for entry in discovered
            if entry.name == _LEGACY_MANAGER_ENTRYPOINT
        ]
    except Exception as error:
        raise RuntimeError(
            "cannot prove legacy orbitkv_manager entry-point absence"
        ) from error
    if matches:
        raise RuntimeError(
            "direct-source qualification forbids an installed "
            "orbitkv_manager entry point"
        )
