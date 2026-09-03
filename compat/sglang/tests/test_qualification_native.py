from __future__ import annotations

import copy
import sys
import uuid
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "bridge/src")]
import qualification_runner as bench  # noqa: E402
from qualification_native_support import (  # noqa: E402
    native_manager_info,
    write_sliding_manifest,
)


@pytest.mark.parametrize(
    ("profile", "hybrid", "policy", "class_ids"),
    (
        (bench.FULL_SLIDING_TOPOLOGY, True, "shared_prefix", (0, 1)),
        (bench.SLIDING_TOPOLOGY, False, "request_private", (0,)),
    ),
)
def test_admission_accepts_native_sliding_profiles(
    tmp_path: Path, profile: str, hybrid: bool, policy: str,
    class_ids: tuple[int, ...],
) -> None:
    result = bench.load_qualification_admission(
        write_sliding_manifest(tmp_path, hybrid=hybrid), profile
    )
    assert result["profile"] == profile
    assert result["cache_policy"] == policy
    assert result["class_ids"] == class_ids
    assert result["chunk_geometry"]["classes"][-1] == {
        "class_id": class_ids[-1], "retention": "sliding",
        "layers": [0, 2] if hybrid else [0, 1, 2, 3],
        "window_tokens": 32, "minimum_resident_tokens": 48,
    }


def test_admission_rejects_profile_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selected qualification profile"):
        bench.load_qualification_admission(
            write_sliding_manifest(tmp_path, hybrid=True), bench.SLIDING_TOPOLOGY
        )


def _geometry(*, hybrid: bool) -> dict[str, object]:
    classes = []
    if hybrid:
        classes.append({
            "class_id": 0, "retention": "full", "layers": [1],
            "window_tokens": None, "minimum_resident_tokens": None,
        })
    classes.append({
        "class_id": len(classes), "retention": "sliding",
        "layers": [0] if hybrid else [0, 1], "window_tokens": 32,
        "minimum_resident_tokens": 48,
    })
    return {"page_tokens": 16, "classes": classes}


def test_sliding_workload_and_capacity_boundaries() -> None:
    geometry = _geometry(hybrid=False)
    workload = bench.runtime_support.validate_native_workload(
        prompt_tokens=32, decode_tokens=18, iterations=3,
        profile_geometry=geometry,
    )
    assert workload["retirement_boundary_crossings"] == 1
    with pytest.raises(ValueError, match="temporal cycle"):
        bench.runtime_support.validate_native_workload(
            prompt_tokens=32, decode_tokens=17, iterations=3,
            profile_geometry=geometry,
        )

    non_aligned = {
        "page_tokens": 16,
        "classes": [{
            "class_id": 0, "retention": "sliding", "layers": [0],
            "window_tokens": 18, "minimum_resident_tokens": 48,
        }],
    }
    with pytest.raises(ValueError, match="temporal cycle"):
        bench.runtime_support.validate_native_workload(
            prompt_tokens=18, decode_tokens=18, iterations=1,
            profile_geometry=non_aligned,
        )
    assert bench.runtime_support.validate_native_workload(
        prompt_tokens=18, decode_tokens=32, iterations=1,
        profile_geometry=non_aligned,
    )["retirement_boundary_crossings"] == 1
    floor = bench.runtime_support.native_capacity_contract(
        case="exact-floor", profile=bench.SLIDING_TOPOLOGY,
        max_total_tokens=112, chunked_prefill_tokens=64, final_kv_tokens=49,
        profile_geometry=geometry, swa_full_tokens_ratio=None,
    )
    assert floor["expected_sliding_tokens"] == floor["sliding_floor_tokens"] == 112
    with pytest.raises(ValueError, match="above the exact floor"):
        bench.runtime_support.native_capacity_contract(
            case="roomy", profile=bench.SLIDING_TOPOLOGY,
            max_total_tokens=112, chunked_prefill_tokens=64, final_kv_tokens=49,
            profile_geometry=geometry, swa_full_tokens_ratio=None,
        )


def test_hybrid_exact_floor_is_per_class() -> None:
    exact = bench.runtime_support.native_capacity_contract(
        case="exact-floor", profile=bench.FULL_SLIDING_TOPOLOGY,
        max_total_tokens=112, chunked_prefill_tokens=64, final_kv_tokens=97,
        profile_geometry=_geometry(hybrid=True), swa_full_tokens_ratio=1.0,
    )
    assert exact["full_floor_tokens"] == 112
    assert exact["sliding_floor_tokens"] == 112
    with pytest.raises(ValueError, match="exact Full and SWA class floors"):
        bench.runtime_support.native_capacity_contract(
            case="exact-floor", profile=bench.FULL_SLIDING_TOPOLOGY,
            max_total_tokens=128, chunked_prefill_tokens=64, final_kv_tokens=97,
            profile_geometry=_geometry(hybrid=True), swa_full_tokens_ratio=1.0,
        )


def test_hybrid_full_floor_reserves_next_decode_write_page() -> None:
    geometry = {
        "page_tokens": 16,
        "classes": [
            {"class_id": 0, "retention": "full", "layers": [1],
             "window_tokens": None, "minimum_resident_tokens": None},
            {"class_id": 1, "retention": "sliding", "layers": [0],
             "window_tokens": 128, "minimum_resident_tokens": 144},
        ],
    }
    exact = bench.runtime_support.native_capacity_contract(
        case="exact-floor", profile=bench.FULL_SLIDING_TOPOLOGY,
        max_total_tokens=672, chunked_prefill_tokens=256, final_kv_tokens=656,
        profile_geometry=geometry, swa_full_tokens_ratio=0.61,
    )
    assert exact["full_floor_tokens"] == 672
    assert exact["sliding_floor_tokens"] == 400

    with pytest.raises(ValueError, match="exact Full and SWA class floors"):
        bench.runtime_support.native_capacity_contract(
            case="exact-floor", profile=bench.FULL_SLIDING_TOPOLOGY,
            max_total_tokens=656, chunked_prefill_tokens=256,
            final_kv_tokens=656, profile_geometry=geometry,
            swa_full_tokens_ratio=0.61,
        )


def test_gpt_oss_moe_backend_is_pinned_and_recorded() -> None:
    args = Namespace(
        profile=bench.FULL_SLIDING_TOPOLOGY, context_length=256, seed=7,
        max_total_tokens=256, max_running_requests=1, mem_fraction_static=None,
        chunked_prefill_size=64, swa_full_tokens_ratio=1.0, mode="manager",
    )
    engine = bench.engine_arguments(
        args, Path("/model"), {"page_tokens": 16},
        cache_policy="shared_prefix", architecture="GptOssForCausalLM",
    )
    assert (engine["moe_runner_backend"], engine["moe_a2a_backend"], engine["ep_size"]) == ("triton", "none", 1)
    identity = bench.runtime_support.runtime_identity(
        SimpleNamespace(__version__="0.5.17"), Path("/sglang/__init__.py"),
        engine, run_id=str(uuid.uuid4()), runtime_proof=None,
    )
    assert identity["moe_backend"] == {"runner": "triton", "a2a": "none", "ep_size": 1}


@pytest.mark.parametrize("hybrid", (False, True))
def test_census_preserves_direct_swa_evidence(hybrid: bool) -> None:
    profile = bench.FULL_SLIDING_TOPOLOGY if hybrid else bench.SLIDING_TOPOLOGY
    census = bench.manager_census(
        native_manager_info(bench, active=True, hybrid=hybrid), "after_workload",
        expected_manifest_fingerprint="sha256:manifest",
        expected_runtime_binding_fingerprint="sha256:binding",
        expected_plan_fingerprint="sha256:manager-input",
        expected_lifecycle_route="native_session",
        expected_cache_policy="shared_prefix" if hybrid else "request_private",
        expected_profile=profile, expected_class_ids=(0, 1) if hybrid else (0,),
        expected_chunk_geometry={"page_tokens": 16},
        expected_engine_args={"max_prefill_tokens": 64, "max_running_requests": 1},
    )
    assert census["swa_activity"]["source"] == "native_runtime_session"
    assert census["swa_activity"]["swa_page_reuse_events"] == 1


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(status="not_applicable", applicable=False),
        lambda value: value.update(source="runner_inference", derived=True),
        lambda value: value.update(swa_page_reuse_events=-1),
    ),
)
def test_census_rejects_non_direct_swa_evidence(mutation) -> None:
    info = native_manager_info(bench, active=True)
    mutation(info["internal_states"][0]["orbitkv_manager"]["swa_activity"])
    with pytest.raises(RuntimeError, match="SWA activity|swa_activity"):
        bench.manager_census(
            info, "after_workload",
            expected_manifest_fingerprint="sha256:manifest",
            expected_runtime_binding_fingerprint="sha256:binding",
            expected_plan_fingerprint="sha256:manager-input",
            expected_lifecycle_route="native_session",
            expected_cache_policy="request_private",
            expected_profile=bench.SLIDING_TOPOLOGY, expected_class_ids=(0,),
            expected_chunk_geometry={"page_tokens": 16},
            expected_engine_args={"max_prefill_tokens": 64, "max_running_requests": 1},
        )


def test_progress_rejects_zero_reuse() -> None:
    before = native_manager_info(bench, active=False)["internal_states"][0]["orbitkv_manager"]["swa_activity"]
    after = copy.deepcopy(before)
    for name in ("swa_retirement_certificates", "swa_pages_reclaimed", "swa_wrap_events"):
        after[name] = 1
    with pytest.raises(RuntimeError, match="swa_page_reuse_events"):
        bench.runtime_support.require_swa_progress(before, after)


def test_sliding_final_drain_reads_state_after_cache_flush() -> None:
    events: list[str] = []

    class Engine:
        def flush_cache(self) -> dict[str, bool]:
            events.append("flush")
            return {"success": True}

        def get_server_info(self) -> dict[str, str]:
            events.append("readback")
            return {"stage": "final"}

    engine = Engine()
    final_info = bench._final_server_info(engine, flush_cache=True)

    assert final_info == {"stage": "final"}
    assert events == ["flush", "readback"]
