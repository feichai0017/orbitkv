"""Runtime-policy and census validation for the SGLang Engine E2E runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable


LIFECYCLE_ROUTES = frozenset({"native_session", "canonical_manager"})
CACHE_POLICIES = frozenset({"shared_prefix", "request_private"})
SESSION_COUNTER_FIELDS = (
    "forward_events",
    "completion_values",
    "event_queries",
    "event_waits",
    "fail_stop_count",
)
POST_WORKLOAD_RESIDENCY_FIELDS = (
    "active_pages",
    "retiring_pages",
    "quarantined_pages",
    "total_request_page_refs",
    "total_prefix_page_refs",
    "total_reader_pins",
)
DRAIN_FIELDS = (
    "active_requests",
    "active_snapshots",
    "active_prefixes",
    "prepared_steps",
    "submitted_steps",
    "reserved_pages",
    "writing_pages",
    "active_pages",
    "retiring_pages",
    "quarantined_pages",
    "pending_reclamations",
    "total_request_page_refs",
    "total_prefix_page_refs",
    "total_reader_pins",
)
MANAGER_FIELDS = frozenset(
    {
        "wire_version", "manager_input_fingerprint",
        "runtime_manifest_fingerprint", "runtime_binding_fingerprint",
        "lifecycle_route", "cache_policy", "direct_source_owner",
        "completion_evidence", "runtime_proof", "identities",
        "manager_stats", "arena_stats", "batch_counters",
        "swa_activity", "pressure", "fixed_state_byte_count",
        "fixed_state_descriptors", "tree_cache_type",
    }
)


def scheduler_state(info: Any, stage: str) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        raise RuntimeError(f"SGLang server info is malformed at {stage}")
    states = info.get("internal_states")
    if (
        not isinstance(states, list)
        or len(states) != 1
        or not isinstance(states[0], dict)
    ):
        raise RuntimeError(
            f"SGLang must return exactly one scheduler state at {stage}"
        )
    return states[0]


def nonnegative_mapping(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{label} is missing")
    result: dict[str, int] = {}
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
        ):
            raise RuntimeError(f"{label} contains an invalid counter")
        result[name] = item
    return result


def runtime_policy(
    value: Any, stage: str, *, expected_route: str, expected_cache_policy: str
) -> tuple[str, str]:
    if expected_route not in LIFECYCLE_ROUTES:
        raise ValueError("expected OrbitKV lifecycle route is invalid")
    if expected_cache_policy not in CACHE_POLICIES:
        raise ValueError("expected OrbitKV cache policy is invalid")
    route = value.get("lifecycle_route") if isinstance(value, Mapping) else None
    cache_policy = value.get("cache_policy") if isinstance(value, Mapping) else None
    if route != expected_route:
        raise RuntimeError(f"OrbitKV lifecycle route differs from admission at {stage}")
    if cache_policy != expected_cache_policy:
        raise RuntimeError(f"OrbitKV cache policy differs from admission at {stage}")
    return route, cache_policy


def validate_manager_snapshot(
    info: Mapping[str, Any],
    stage: str,
    *,
    wire_version: int,
    manifest_fingerprint: str,
    binding_fingerprint: str,
    manager_input_fingerprint: str,
    expected_lifecycle_route: str,
    expected_cache_policy: str,
    expected_class_ids: Sequence[int],
    require_activity: bool,
    engine_args: Mapping[str, Any],
    owner_proof: Callable[[Any, str], dict[str, Any]],
    runtime_proof: Callable[[Any, str, Mapping[str, Any]], dict[str, Any] | None],
    completion_evidence: Callable[..., dict[str, Any]],
    session_swa_activity: Callable[[Any, str], dict[str, Any]],
    disabled_pressure: Callable[[Any, str], dict[str, Any]],
) -> dict[str, Any]:
    state = scheduler_state(info, stage)
    raw = state.get("orbitkv_manager")
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"OrbitKV manager census is missing at {stage}")
    if set(raw) != MANAGER_FIELDS:
        raise RuntimeError(f"OrbitKV manager census fields changed at {stage}")
    class_ids = tuple(expected_class_ids)
    if (
        not class_ids
        or len(set(class_ids)) != len(class_ids)
        or any(type(item) is not int or item < 0 for item in class_ids)
    ):
        raise ValueError("expected OrbitKV class identities are invalid")
    expected = {
        "wire_version": wire_version,
        "runtime_manifest_fingerprint": manifest_fingerprint,
        "runtime_binding_fingerprint": binding_fingerprint,
        "manager_input_fingerprint": manager_input_fingerprint,
    }
    mismatch = {
        name: {"expected": value, "actual": raw.get(name)}
        for name, value in expected.items()
        if raw.get(name) != value
    }
    if mismatch:
        raise RuntimeError(f"OrbitKV provenance changed at {stage}: {mismatch}")
    route, policy = runtime_policy(
        raw, stage, expected_route=expected_lifecycle_route,
        expected_cache_policy=expected_cache_policy,
    )
    owner = owner_proof(raw.get("direct_source_owner"), stage)
    proof = runtime_proof(raw.get("runtime_proof"), stage, engine_args)
    completion = completion_evidence(
        raw.get("completion_evidence"), stage, require_activity=require_activity
    )
    stats = nonnegative_mapping(raw.get("manager_stats"), "manager_stats")
    prefix_state = {
        name: stats.get(name)
        for name in ("active_prefixes", "evicted_prefixes", "total_prefix_page_refs")
        if stats.get(name) != 0
    }
    if policy == "request_private" and prefix_state:
        raise RuntimeError(
            f"OrbitKV request-private prefix state is nonzero at {stage}: {prefix_state}"
        )
    counters = nonnegative_mapping(raw.get("batch_counters"), "batch_counters")
    if set(counters) != set(SESSION_COUNTER_FIELDS):
        raise RuntimeError(f"OrbitKV session counter keys differ at {stage}")
    if counters["fail_stop_count"] != 0:
        raise RuntimeError(f"OrbitKV session fail-stop counter is nonzero at {stage}")
    swa = session_swa_activity(raw.get("swa_activity"), stage)
    pressure = disabled_pressure(raw.get("pressure"), stage)
    arenas = raw.get("arena_stats")
    identities = raw.get("identities")
    if (
        not isinstance(arenas, list)
        or len(arenas) != len(class_ids)
        or not isinstance(identities, list)
        or len(identities) != len(arenas)
    ):
        raise RuntimeError(f"OrbitKV arena census is malformed at {stage}")
    normalized_arenas = []
    normalized_identities = []
    page_fields = (
        "free_pages", "reserved_pages", "writing_pages", "active_pages",
        "retiring_pages", "quarantined_pages", "exhausted_pages",
    )
    identity_fields = (
        "engine_epoch", "pool_epoch", "pool_id", "class_id",
        "backend_domain", "page_count", "first_page_id",
    )
    for index, (identity, arena) in enumerate(zip(identities, arenas, strict=True)):
        identity_values = nonnegative_mapping(identity, f"identities[{index}]")
        values = nonnegative_mapping(arena, f"arena_stats[{index}]")
        page_count = values.get("page_count")
        if page_count is None or any(name not in values for name in page_fields):
            raise RuntimeError(f"OrbitKV arena phases are incomplete at {stage}")
        if sum(values[name] for name in page_fields) != page_count:
            raise RuntimeError(f"OrbitKV arena page conservation failed at {stage}")
        if any(
            name not in identity_values
            or name not in values
            or identity_values[name] != values[name]
            for name in identity_fields
        ):
            raise RuntimeError(f"OrbitKV arena identity echo changed at {stage}")
        if (
            identity_values.get("class_id") != class_ids[index]
            or identity_values.get("page_tokens") != state.get("page_size")
        ):
            raise RuntimeError(f"OrbitKV arena identity changed at {stage}")
        normalized_identities.append(identity_values)
        normalized_arenas.append(values)
    for name in page_fields:
        if stats.get(name) != sum(arena[name] for arena in normalized_arenas):
            raise RuntimeError(f"OrbitKV aggregate {name} changed at {stage}")
    references = {
        "total_request_page_refs": "request_page_refs",
        "total_prefix_page_refs": "prefix_page_refs",
        "total_reader_pins": "reader_pins",
    }
    for aggregate, arena_field in references.items():
        if stats.get(aggregate) != sum(
            arena.get(arena_field, -1) for arena in normalized_arenas
        ):
            raise RuntimeError(f"OrbitKV aggregate {aggregate} changed at {stage}")
    if require_activity:
        inactive = [
            name for name in ("forward_events", "completion_values")
            if counters[name] <= 0
        ]
        if inactive:
            raise RuntimeError(
                f"OrbitKV session completion activity is not proven at {stage}: {inactive}"
            )
        if counters["forward_events"] != counters["completion_values"]:
            raise RuntimeError(f"OrbitKV session completion counters differ at {stage}")
        if counters["event_queries"] + counters["event_waits"] <= 0:
            raise RuntimeError(f"OrbitKV completion observation is not proven at {stage}")
    return {
        "stage": stage, "wire_version": raw["wire_version"],
        "manager_input_fingerprint": raw["manager_input_fingerprint"],
        "runtime_manifest_fingerprint": raw["runtime_manifest_fingerprint"],
        "runtime_binding_fingerprint": raw["runtime_binding_fingerprint"],
        "lifecycle_route": route, "cache_policy": policy,
        "direct_source_owner": owner, "completion_evidence": completion,
        "runtime_proof": proof, "identities": normalized_identities,
        "manager_stats": stats, "arena_stats": normalized_arenas,
        "batch_counters": counters, "swa_activity": swa, "pressure": pressure,
    }


def require_manager_drained(snapshot: Mapping[str, Any]) -> None:
    stage = str(snapshot.get("stage", "final"))
    stats = snapshot.get("manager_stats")
    arenas = snapshot.get("arena_stats")
    if not isinstance(stats, Mapping) or not isinstance(arenas, list):
        raise RuntimeError(f"OrbitKV final census is malformed at {stage}")
    missing = [name for name in DRAIN_FIELDS if name not in stats]
    if missing:
        raise RuntimeError(f"OrbitKV final census omits drain counters: {missing}")
    dirty = {name: stats[name] for name in DRAIN_FIELDS if stats[name] != 0}
    for index, arena in enumerate(arenas):
        if not isinstance(arena, Mapping):
            raise RuntimeError(f"OrbitKV final arena {index} is malformed")
        if arena.get("free_pages") != arena.get("page_count"):
            dirty[f"arena_{index}_free_pages"] = {
                "expected": arena.get("page_count"),
                "actual": arena.get("free_pages"),
            }
    if dirty:
        raise RuntimeError(f"OrbitKV manager did not drain at {stage}: {dirty}")


def post_workload_residency(snapshot: Mapping[str, Any]) -> dict[str, int]:
    stats = snapshot.get("manager_stats")
    if not isinstance(stats, Mapping):
        raise RuntimeError("OrbitKV manager stats are unavailable")
    missing = [name for name in POST_WORKLOAD_RESIDENCY_FIELDS if name not in stats]
    if missing:
        raise RuntimeError(f"OrbitKV residency census is incomplete: {missing}")
    residency = {name: stats[name] for name in POST_WORKLOAD_RESIDENCY_FIELDS}
    policy = snapshot.get("cache_policy")
    if policy == "request_private":
        forbidden = POST_WORKLOAD_RESIDENCY_FIELDS
    elif policy == "shared_prefix":
        forbidden = (
            "retiring_pages", "quarantined_pages",
            "total_request_page_refs", "total_reader_pins",
        )
    else:
        raise RuntimeError("OrbitKV cache policy is invalid")
    dirty = {name: residency[name] for name in forbidden if residency[name] != 0}
    if dirty:
        raise RuntimeError(f"OrbitKV {policy} unsafe residency is nonzero: {dirty}")
    return residency


__all__ = [
    "CACHE_POLICIES", "DRAIN_FIELDS", "LIFECYCLE_ROUTES",
    "MANAGER_FIELDS", "POST_WORKLOAD_RESIDENCY_FIELDS",
    "SESSION_COUNTER_FIELDS", "nonnegative_mapping",
    "post_workload_residency", "require_manager_drained",
    "runtime_policy", "scheduler_state", "validate_manager_snapshot",
]
