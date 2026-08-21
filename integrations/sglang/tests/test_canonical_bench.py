from __future__ import annotations

import inspect
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "src"
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import bench_canonical_manager as bench  # noqa: E402
import bench_compact_control as compact  # noqa: E402


@pytest.fixture(scope="session")
def compact_ffi_library() -> Path:
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
        timeout=180,
    )
    library = (
        REPOSITORY_ROOT
        / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"
    )
    assert library.is_file()
    return library


def _arguments(**overrides):
    values = {
        "mode": "manager",
        "sglang_root": "/sglang",
        "model": "/model",
        "plan": "/plan.json",
        "library": "/liborbitkv_ffi.so",
        "requests": 1,
        "max_running_requests": 4,
        "prompt_tokens": 33,
        "decode_tokens": 33,
        "iterations": 1,
        "chunked_prefill_size": 48,
        "context_length": 128,
        "max_total_tokens": 4096,
        "mem_fraction_static": None,
        "attention_backend": "flashinfer",
        "seed": 20260820,
    }
    values.update(overrides)
    return Namespace(**values)


def _attention_contract(architecture="Qwen2ForCausalLM", **values):
    contract = {
        "architecture": architecture,
        "attention_backend": bench.ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture],
    }
    contract.update(values)
    return contract


def _checkpoint_identity(*_args, **_kwargs):
    return {
        "weight_bytes": 1,
        "indexed_weights_complete": True,
        "config_sha256": "config",
    }


def _write_config(tmp_path: Path, value: dict) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(value), encoding="utf-8")
    return model


def _checkout_inputs(tmp_path: Path) -> dict[str, str]:
    sglang = tmp_path / "sglang"
    (sglang / "python/sglang").mkdir(parents=True)
    (sglang / "python/sglang/__init__.py").write_text("", encoding="utf-8")
    model = _write_config(
        tmp_path,
        {
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 2,
            "vocab_size": 128,
            "max_position_embeddings": 256,
            "use_sliding_window": False,
        },
    )
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.write_bytes(b"ffi")
    return {
        "sglang_root": str(sglang),
        "model": str(model),
        "plan": str(plan),
        "library": str(library),
    }


def test_help_exposes_only_independent_manager_and_stock_runs():
    completed = subprocess.run(
        [
            sys.executable,
            str(INTEGRATION_ROOT / "bench_canonical_manager.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--mode {manager,stock}" in completed.stdout
    assert "--max-total-tokens" in completed.stdout
    assert "--attention-backend {flashinfer,fa3}" in completed.stdout


def test_compact_control_help_requires_exact_abi6_matrix_dimensions():
    completed = subprocess.run(
        [
            sys.executable,
            str(INTEGRATION_ROOT / "bench_compact_control.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--profile {full,hybrid}" in completed.stdout
    assert "--batch-size {1,4}" in completed.stdout
    assert "--resident-pages" in completed.stdout
    assert "--iterations" in completed.stdout


def test_compact_control_capacity_covers_initial_and_timed_pages_exactly():
    assert compact._arena_page_count(1, 512, 32) == 514
    assert compact._arena_page_count(4, 512, 20) == 2056
    hybrid = compact._profile_plan("hybrid")
    assert [item["retention"] for item in hybrid["classes"]] == [
        "full",
        "sliding",
    ]
    assert hybrid["classes"][1]["window_tokens"] == 18


@pytest.mark.parametrize(
    ("profile", "batch_size"), (("full", 1), ("hybrid", 4))
)
def test_compact_control_rejects_noncanonical_profile_or_batch(
    profile, batch_size
):
    compact._validate_inputs(profile, batch_size, 512, 32)
    with pytest.raises(ValueError, match="profile"):
        compact._validate_inputs("sliding", batch_size, 512, 32)
    with pytest.raises(ValueError, match="batch-size"):
        compact._validate_inputs(profile, 2, 512, 32)


@pytest.mark.parametrize(
    ("profile", "batch_size", "class_count"),
    (("full", 1, 1), ("hybrid", 4, 2)),
)
def test_compact_control_abi6_runs_real_host_batches_without_root_materialization(
    compact_ffi_library, profile, batch_size, class_count
):
    result = compact.run(
        compact_ffi_library,
        profile=profile,
        batch_size=batch_size,
        resident_pages=32,
        iterations=32,
    )
    assert result["schema"] == "orbitkv.abi6-prefix-control.v1"
    assert result["scope"] == "host_control_only"
    assert result["profile"] == profile
    assert result["batch_size"] == batch_size
    assert result["setup_timed"] is False
    assert len(result["arenas"]) == class_count
    assert {item["page_count"] for item in result["arenas"]} == {
        batch_size * 34
    }
    assert result["manager_limits"] == {
        "maximum_requests": batch_size,
        "maximum_operations": batch_size,
        "maximum_prefixes": batch_size,
        "maximum_reclamations": batch_size * 34 * class_count,
        "maximum_step_tokens": 512,
    }
    assert set(result["phases"]) == {
        "prepare",
        "submit",
        "complete",
        "total",
    }
    assert all(
        set(summary) == {"p50_ms", "p99_ms"}
        for summary in result["phases"].values()
    )
    assert result["compact_counters"] == {
        "hot_workspace_allocations": 0,
        "capacity_memset_bytes": 0,
        "root_entries_crossed": 0,
        "materialized_page_objects": 0,
    }
    assert result["phases"]["total"]["p50_ms"] < 1.25
    assert result["host_gate_passed"] is True
    assert "performance_go" not in result


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_engine_profile_is_identical_and_explicitly_capacity_matched(mode):
    values = bench.engine_arguments(
        _arguments(mode=mode), Path("/model"), _attention_contract()
    )
    assert values["page_size"] == 16
    assert values["max_total_tokens"] == 4096
    assert values["disable_cuda_graph"] is True
    assert values["enable_torch_compile"] is False
    assert values["disable_overlap_schedule"] is True
    assert values["disable_radix_cache"] is False
    if mode == "manager":
        assert values["radix_cache_backend"] == "orbitkv"
    else:
        assert "radix_cache_backend" not in values
    assert values["attention_backend"] == "flashinfer"
    assert values["dtype"] == values["kv_cache_dtype"] == "bfloat16"
    assert values["disable_hybrid_swa_memory"] is False
    assert values["tp_size"] == values["pp_size"] == values["dcp_size"] == 1
    assert values["speculative_algorithm"] is None
    assert values["disaggregation_mode"] == "null"
    assert "moe_runner_backend" not in values


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_gpt_oss_engine_profile_requires_fa3_for_both_modes(mode):
    contract = _attention_contract("GptOssForCausalLM")
    values = bench.engine_arguments(
        _arguments(mode=mode, attention_backend="fa3"), Path("/model"), contract
    )
    assert values["attention_backend"] == "fa3"
    assert values["moe_runner_backend"] == "triton"

    with pytest.raises(RuntimeError, match="requires --attention-backend fa3"):
        bench.engine_arguments(_arguments(mode=mode), Path("/model"), contract)


def test_qwen2_engine_profile_rejects_fa3():
    with pytest.raises(RuntimeError, match="requires --attention-backend flashinfer"):
        bench.engine_arguments(
            _arguments(attention_backend="fa3"),
            Path("/model"),
            _attention_contract(),
        )


def test_engine_profile_enables_batch_invariant_inference():
    values = bench.engine_arguments(
        _arguments(), Path("/model"), _attention_contract()
    )
    assert values["enable_deterministic_inference"] is True
    assert values["sampling_backend"] == "pytorch"


def test_b4_inputs_share_only_the_exact_page_aligned_seed_prefix():
    prompts = bench.deterministic_input_ids(
        requests=4, prompt_tokens=513, vocab_size=1024, seed=20260820
    )
    seed_boundary = (513 - 1) // 16 * 16
    assert seed_boundary == 512
    assert bench.PREFIX_SEED_BATCH_SIZE == 1
    seed_inputs = [list(prompts[0][:seed_boundary])]
    assert len(seed_inputs) == bench.PREFIX_SEED_BATCH_SIZE
    assert len({tuple(prompt[:seed_boundary]) for prompt in prompts}) == 1
    assert all(prompt[:seed_boundary] == seed_inputs[0] for prompt in prompts)
    assert len({tuple(prompt) for prompt in prompts}) == 4


def test_pair_normalization_allows_only_explicit_radix_backend_selection():
    manager = bench.engine_arguments(
        _arguments(mode="manager"), Path("/model"), _attention_contract()
    )
    stock = bench.engine_arguments(
        _arguments(mode="stock"), Path("/model"), _attention_contract()
    )
    assert set(manager) - set(stock) == {"radix_cache_backend"}
    assert bench.pair_engine_arguments("manager", manager) == (
        bench.pair_engine_arguments("stock", stock)
    )
    assert bench.PAIR_IMPLEMENTATION_DIFFERENCE == {
        "field": "radix_cache_backend",
        "manager": {"present": True, "value": "orbitkv"},
        "stock": {"present": False, "value": None},
        "scope": "implementation selection only",
    }

    changed = dict(stock, max_total_tokens=8192)
    assert bench.pair_engine_arguments("stock", changed) != (
        bench.pair_engine_arguments("manager", manager)
    )
    with pytest.raises(RuntimeError, match="must omit"):
        bench.pair_engine_arguments("stock", dict(stock, radix_cache_backend=None))
    missing = dict(manager)
    del missing["radix_cache_backend"]
    with pytest.raises(RuntimeError, match="requires"):
        bench.pair_engine_arguments("manager", missing)
    assert bench.expected_prefix_cache("manager", {"attention_profile": "full"}) == (
        "OrbitKvPrefixCache"
    )
    assert bench.expected_prefix_cache("stock", {"attention_profile": "full"}) == (
        "RadixCache"
    )
    assert bench.expected_prefix_cache(
        "stock", {"attention_profile": "hybrid_full_swa"}
    ) == "UnifiedRadixCache"


def test_manager_accepts_same_explicit_storage_cap(tmp_path):
    paths = _checkout_inputs(tmp_path)
    args = _arguments(**paths)
    resolved = bench.validate_arguments(args)
    assert resolved["plan"] == Path(paths["plan"]).resolve()
    assert resolved["library"] == Path(paths["library"]).resolve()


def test_stock_forbids_manager_artifacts_and_cap_is_always_page_aligned(tmp_path):
    paths = _checkout_inputs(tmp_path)
    with pytest.raises(ValueError, match="stock mode forbids"):
        bench.validate_arguments(_arguments(mode="stock", **paths))

    with pytest.raises(ValueError, match="divisible by 16"):
        bench.validate_arguments(
            _arguments(
                mode="stock",
                plan=None,
                library=None,
                max_total_tokens=4097,
                **{key: paths[key] for key in ("sglang_root", "model")},
            )
        )


def test_abi6_qualification_accepts_only_complete_b1_or_b4_batches(tmp_path):
    paths = _checkout_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly 1 or 4"):
        bench.validate_arguments(_arguments(requests=2, **paths))
    with pytest.raises(ValueError, match="complete B1/B4 prompt batch"):
        bench.validate_arguments(
            _arguments(requests=4, chunked_prefill_size=64, **paths)
        )
    resolved = bench.validate_arguments(
        _arguments(requests=4, chunked_prefill_size=192, **paths)
    )
    assert resolved["plan"] == Path(paths["plan"]).resolve()

    resolved = bench.validate_arguments(
        _arguments(
            requests=4,
            prompt_tokens=513,
            chunked_prefill_size=2112,
            context_length=1024,
            **paths,
        )
    )
    assert resolved["plan"] == Path(paths["plan"]).resolve()
    with pytest.raises(ValueError, match="final KV boundary"):
        bench.validate_arguments(
            _arguments(
                requests=4,
                prompt_tokens=32,
                chunked_prefill_size=128,
                **paths,
            )
        )
    with pytest.raises(ValueError, match="decode-tokens must be exactly 33"):
        bench.validate_arguments(
            _arguments(
                decode_tokens=32,
                **paths,
            )
        )


def test_qwen2_contract_is_strict_full_only(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "checkpoint_identity", _checkpoint_identity)
    model = _write_config(
        tmp_path,
        {
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 3,
            "vocab_size": 128,
            "max_position_embeddings": 1024,
            "use_sliding_window": False,
            "sliding_window": 512,
        },
    )
    contract, _identity = bench.checkpoint_contract(model)
    assert contract["attention_profile"] == "full"
    assert contract["attention_backend"] == "flashinfer"
    assert contract["sliding_window"] is None
    assert contract["classes"] == [
        {
            "name": "full",
            "retention": "full",
            "layers": [0, 1, 2],
            "window_tokens": None,
        }
    ]

    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 3,
                "vocab_size": 128,
                "max_position_embeddings": 1024,
                "use_sliding_window": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="use_sliding_window=false"):
        bench.checkpoint_contract(model)


def test_gpt_oss_contract_matches_ordered_full_plus_swa_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "checkpoint_identity", _checkpoint_identity)
    model = _write_config(
        tmp_path,
        {
            "architectures": ["GptOssForCausalLM"],
            "num_hidden_layers": 4,
            "vocab_size": 256,
            "max_position_embeddings": 4096,
            "sliding_window": 128,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        },
    )
    manager_config = SimpleNamespace(
        page_tokens=16,
        num_hidden_layers=4,
        classes=(
            SimpleNamespace(
                name="full",
                retention="full",
                layers=(1, 3),
                window_tokens=None,
            ),
            SimpleNamespace(
                name="swa",
                retention="sliding",
                layers=(0, 2),
                window_tokens=128,
            ),
        ),
    )
    contract, _identity = bench.checkpoint_contract(model, manager_config)
    assert contract["attention_profile"] == "hybrid_full_swa"
    assert contract["attention_backend"] == "fa3"
    assert contract["classes"][0]["layers"] == [1, 3]
    assert contract["classes"][1]["layers"] == [0, 2]

    manager_config.classes[1].window_tokens = 256
    with pytest.raises(RuntimeError, match="differs from the checkpoint"):
        bench.checkpoint_contract(model, manager_config)


def test_checkpoint_and_source_gates_have_no_legacy_attention_path():
    assert bench.SUPPORTED_ARCHITECTURES == (
        "Qwen2ForCausalLM",
        "GptOssForCausalLM",
    )
    source_gate = inspect.getsource(bench.verify_sglang_source)
    assert "validate_base_checkout" in source_gate
    assert "validate_patched_checkout" in source_gate
    assert bench.SUPPORTED_SGLANG_RELEASE == "v0.5.17"
    assert (
        bench.SUPPORTED_SGLANG_REVISION
        == "29481685462732237d80d86076d6563e1f658102"
    )


def _runtime_info(
    *,
    attention_backend="flashinfer",
    moe_runner_backend="auto",
    radix_cache_backend="orbitkv",
    swa_tokens=None,
    orbitkv_manager=None,
):
    state = {
        "page_size": 16,
        "max_total_tokens": 4096,
        "attention_backend": attention_backend,
        "moe_runner_backend": moe_runner_backend,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "chunked_prefill_size": 48,
        "max_running_requests": 4,
        "disable_overlap_schedule": True,
        "disable_radix_cache": False,
        "radix_cache_backend": radix_cache_backend,
        "disable_cuda_graph": True,
        "disable_hybrid_swa_memory": False,
        "enable_torch_compile": False,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "enable_page_major_kv_layout": False,
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
        "memory_usage": {
            "token_capacity": 4096,
            "token_capacity_swa": swa_tokens,
            "kvcache": 1.25,
        },
    }
    if orbitkv_manager is not None:
        state["orbitkv_manager"] = orbitkv_manager
    return {"internal_states": [state]}


def _manager_config(*, hybrid=True):
    classes = [
        SimpleNamespace(
            class_id=0,
            pool_id=1,
            backend_domain=1,
            retention="full",
        )
    ]
    if hybrid:
        classes.append(
            SimpleNamespace(
                class_id=1,
                pool_id=2,
                backend_domain=2,
                retention="sliding",
            )
        )
    return SimpleNamespace(classes=tuple(classes))


def _hybrid_config():
    return _manager_config(hybrid=True)


def _settled_manager_state(
    *,
    batch_size=4,
    completed_iterations=1,
    hybrid=True,
    prefix_seeded=True,
    global_cleanup=False,
    online_acknowledgements=0,
    swa_retirement_certificates=0,
    swa_pages_reclaimed=0,
    swa_wrap_events=0,
):
    identity_common = {"engine_epoch": 7, "page_tokens": 16}
    identities = [
        {
            **identity_common,
            "pool_epoch": 11,
            "pool_id": 1,
            "class_id": 0,
            "backend_domain": 1,
            "page_count": 4,
            "backend_base_index": 0,
            "first_page_id": 1,
        },
    ]
    if hybrid:
        identities.append(
            {
                **identity_common,
                "pool_epoch": 12,
                "pool_id": 2,
                "class_id": 1,
                "backend_domain": 2,
                "page_count": 2,
                "backend_base_index": 0,
                "first_page_id": 5,
            }
        )
    live_prefix = prefix_seeded and not global_cleanup
    arena_stats = [
        {
            **{key: value for key, value in identity.items() if key in {
                "engine_epoch",
                "pool_epoch",
                "pool_id",
                "page_count",
                "class_id",
                "backend_domain",
                "first_page_id",
            }},
            "free_pages": identity["page_count"] - int(live_prefix),
            "reserved_pages": 0,
            "writing_pages": 0,
            "active_pages": int(live_prefix),
            "retiring_pages": 0,
            "quarantined_pages": 0,
            "exhausted_pages": 0,
            "request_page_refs": 0,
            "prefix_page_refs": int(live_prefix),
            "reader_pins": 0,
        }
        for identity in identities
    ]
    counters = {name: 0 for name in bench._BATCH_COUNTER_FIELDS}
    seed_batches = int(prefix_seeded)
    cleanup_batches = int(global_cleanup)
    forward_batches = seed_batches + completed_iterations * 33
    release_batches = completed_iterations
    warm_request_calls = completed_iterations * batch_size
    cold_calls = warm_request_calls + release_batches + seed_batches + cleanup_batches
    counters.update(
        {
            "request_acquire_batch_calls": seed_batches + warm_request_calls,
            "prepare_batch_calls": forward_batches,
            "submit_batch_calls": forward_batches,
            "complete_batch_calls": forward_batches,
            "release_batch_calls": release_batches,
            "acknowledge_reclamations_batch_calls": (
                completed_iterations
                + cleanup_batches
                + online_acknowledgements
            ),
            "recycle_requests_batch_calls": release_batches + seed_batches,
            "prefix_lookup_batch_calls": warm_request_calls,
            "prefix_attach_batch_calls": warm_request_calls,
            "prefix_publish_release_batch_calls": seed_batches,
            "prefix_evict_batch_calls": cleanup_batches,
            "prefix_recycle_batch_calls": cleanup_batches,
            "buffer_too_small_preflights": cold_calls,
            "cold_workspace_allocations": cold_calls,
            "forward_events": forward_batches,
            "completion_values": forward_batches,
            "event_queries": forward_batches,
            "prefix_matches": seed_batches + completed_iterations * batch_size,
            "prefix_hits": completed_iterations * batch_size,
            "prefix_publishes": seed_batches,
            "prefix_evictions": cleanup_batches,
            "prefix_evicted_full_tokens": 32 * cleanup_batches,
            "prefix_evicted_swa_tokens": 32 * cleanup_batches * int(hybrid),
            "prefix_global_alias_scans": cleanup_batches * int(hybrid),
            "mirror_validation_calls": seed_batches
            * (1 + completed_iterations + cleanup_batches),
            "mirror_syncs": seed_batches
            * (1 + completed_iterations + cleanup_batches),
        }
    )
    page_count = sum(item["page_count"] for item in identities)
    prefix_pages = len(identities) if live_prefix else 0
    return {
        "abi_version": 6,
        "identities": identities,
        "arena_stats": arena_stats,
        "manager_stats": {
            "active_requests": 0,
            "active_snapshots": 0,
            "active_prefixes": int(live_prefix),
            "evicted_prefixes": 0,
            "prepared_steps": 0,
            "submitted_steps": 0,
            "free_pages": page_count - prefix_pages,
            "reserved_pages": 0,
            "writing_pages": 0,
            "active_pages": prefix_pages,
            "retiring_pages": 0,
            "quarantined_pages": 0,
            "exhausted_pages": 0,
            "pending_reclamations": 0,
            "total_request_page_refs": 0,
            "total_prefix_page_refs": prefix_pages,
            "total_reader_pins": 0,
        },
        "swa_activity": {
            "status": "exposed",
            "applicable": hybrid,
            "swa_retirement_certificates": swa_retirement_certificates,
            "swa_pages_reclaimed": swa_pages_reclaimed,
            "swa_wrap_events": swa_wrap_events,
        },
        "batch_counters": counters,
    }


def test_runtime_readback_records_full_and_derived_swa_capacities():
    contract = _attention_contract(
        "GptOssForCausalLM", attention_profile="hybrid_full_swa"
    )
    result = bench.verify_runtime_contract(
        _arguments(mode="manager", attention_backend="fa3"),
        _runtime_info(
            attention_backend="fa3",
            moe_runner_backend="triton",
            swa_tokens=2048,
        ),
        contract,
    )
    assert result["requested_max_total_tokens"] == 4096
    assert result["full_tokens"] == 4096
    assert result["swa_tokens"] == 2048
    assert "no KV compression" in result["interpretation"]

    stock = bench.verify_runtime_contract(
        _arguments(mode="stock", attention_backend="fa3"),
        _runtime_info(
            attention_backend="fa3",
            moe_runner_backend="triton",
            radix_cache_backend=None,
            swa_tokens=2048,
        ),
        contract,
    )
    assert stock == result

    with pytest.raises(RuntimeError, match="moe_runner_backend"):
        bench.verify_runtime_contract(
            _arguments(mode="manager", attention_backend="fa3"),
            _runtime_info(
                attention_backend="fa3",
                moe_runner_backend="triton_kernel",
                swa_tokens=2048,
            ),
            contract,
        )

    missing_backend = _runtime_info(
        attention_backend="fa3",
        moe_runner_backend="triton",
        swa_tokens=2048,
    )
    del missing_backend["internal_states"][0]["radix_cache_backend"]
    with pytest.raises(RuntimeError, match="omitted radix_cache_backend"):
        bench.verify_runtime_contract(
            _arguments(mode="manager", attention_backend="fa3"),
            missing_backend,
            contract,
        )


def test_multi_arena_live_prefix_census_requires_exact_abi6_ref_schema():
    census = bench.manager_census(
        _runtime_info(swa_tokens=32, orbitkv_manager=_settled_manager_state()),
        _hybrid_config(),
        {"full_tokens": 64, "swa_tokens": 32},
        "final",
        batch_size=4,
        completed_iterations=1,
        decode_tokens=33,
        prefix_seeded=True,
    )
    assert [item["class_id"] for item in census["identities"]] == [0, 1]
    assert [item["first_page_id"] for item in census["identities"]] == [1, 5]
    assert census["abi_version"] == 6
    assert census["manager_stats"]["free_pages"] == 4
    assert census["manager_stats"]["active_pages"] == 2
    assert census["manager_stats"]["active_prefixes"] == 1
    assert census["manager_stats"]["total_prefix_page_refs"] == 2
    assert census["swa_activity"] == {
        "status": "exposed",
        "source": "SGLang orbitkv_manager internal state",
        "derived": False,
        "applicable": True,
        "swa_retirement_certificates": 0,
        "swa_pages_reclaimed": 0,
        "swa_wrap_events": 0,
    }


def test_multi_arena_census_rejects_leaks_and_cross_arena_capacity_mismatch():
    state = _settled_manager_state()
    state["manager_stats"]["pending_reclamations"] = 1
    with pytest.raises(RuntimeError, match="did not settle"):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=state),
            _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 32},
            "final",
            batch_size=4,
            completed_iterations=1,
            decode_tokens=33,
            prefix_seeded=True,
        )

    with pytest.raises(RuntimeError, match="capacity differs"):
        bench.manager_census(
            _runtime_info(
                swa_tokens=32, orbitkv_manager=_settled_manager_state()
            ),
            _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 48},
            "final",
            batch_size=4,
            completed_iterations=1,
            decode_tokens=33,
            prefix_seeded=True,
        )

    cleaned = _settled_manager_state(global_cleanup=True)
    census = bench.manager_census(
        _runtime_info(swa_tokens=32, orbitkv_manager=cleaned),
        _hybrid_config(),
        {"full_tokens": 64, "swa_tokens": 32},
        "after_global_cleanup",
        batch_size=4,
        completed_iterations=1,
        decode_tokens=33,
        prefix_seeded=True,
        global_cleanup=True,
    )
    assert census["manager_stats"]["free_pages"] == 6
    assert census["manager_stats"]["active_prefixes"] == 0
    assert census["batch_counters"]["prefix_evict_batch_calls"] == 1
    assert census["batch_counters"]["prefix_recycle_batch_calls"] == 1
    assert census["batch_counters"]["prefix_evictions"] == 1
    assert census["batch_counters"]["prefix_evicted_full_tokens"] == 32
    assert census["batch_counters"]["prefix_evicted_swa_tokens"] == 32
    assert census["batch_counters"]["prefix_global_alias_scans"] == 1
    assert census["batch_counters"]["mirror_validation_calls"] == (
        census["batch_counters"]["mirror_syncs"]
    )

    full_cleaned = _settled_manager_state(hybrid=False, global_cleanup=True)
    full_census = bench.manager_census(
        _runtime_info(orbitkv_manager=full_cleaned),
        _manager_config(hybrid=False),
        {"full_tokens": 64, "swa_tokens": None},
        "after_global_cleanup",
        batch_size=4,
        completed_iterations=1,
        decode_tokens=33,
        prefix_seeded=True,
        global_cleanup=True,
    )
    assert full_census["batch_counters"]["prefix_global_alias_scans"] == 0


def test_exposed_swa_counters_are_collected_only_from_server_and_are_monotonic():
    before_state = _settled_manager_state(
        completed_iterations=1,
        online_acknowledgements=1,
        swa_retirement_certificates=3,
        swa_pages_reclaimed=3,
        swa_wrap_events=2,
    )
    after_state = _settled_manager_state(
        completed_iterations=2,
        online_acknowledgements=2,
        swa_retirement_certificates=5,
        swa_pages_reclaimed=5,
        swa_wrap_events=4,
    )
    args = (_hybrid_config(), {"full_tokens": 64, "swa_tokens": 32})
    before = bench.manager_census(
        _runtime_info(swa_tokens=32, orbitkv_manager=before_state),
        *args,
        "after_load",
        batch_size=4,
        completed_iterations=1,
        decode_tokens=33,
        prefix_seeded=True,
    )
    after = bench.manager_census(
        _runtime_info(swa_tokens=32, orbitkv_manager=after_state),
        *args,
        "final",
        batch_size=4,
        completed_iterations=2,
        decode_tokens=33,
        prefix_seeded=True,
    )
    bench.verify_swa_activity_transition(before, after)
    after["swa_activity"]["swa_retirement_certificates"] = 3
    with pytest.raises(RuntimeError, match="did not advance"):
        bench.verify_swa_activity_transition(before, after)


@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("hybrid", (False, True))
def test_abi6_batch_counter_identities_are_hard_validated(batch_size, hybrid):
    online_acknowledgements = 2 if hybrid else 0
    state = _settled_manager_state(
        batch_size=batch_size,
        completed_iterations=2,
        hybrid=hybrid,
        online_acknowledgements=online_acknowledgements,
        swa_retirement_certificates=5 if hybrid else 0,
        swa_pages_reclaimed=5 if hybrid else 0,
        swa_wrap_events=2 if hybrid else 0,
    )
    census = bench.manager_census(
        _runtime_info(
            swa_tokens=32 if hybrid else None,
            orbitkv_manager=state,
        ),
        _manager_config(hybrid=hybrid),
        {
            "full_tokens": 64,
            "swa_tokens": 32 if hybrid else None,
        },
        "after_workload",
        batch_size=batch_size,
        completed_iterations=2,
        decode_tokens=33,
        prefix_seeded=True,
    )
    counters = census["batch_counters"]
    assert counters["request_acquire_batch_calls"] == 1 + 2 * batch_size
    assert counters["prepare_batch_calls"] == 67
    assert counters["submit_batch_calls"] == 67
    assert counters["complete_batch_calls"] == 67
    assert counters["prefix_lookup_batch_calls"] == 2 * batch_size
    assert counters["prefix_attach_batch_calls"] == 2 * batch_size
    assert counters["prefix_matches"] == 1 + 2 * batch_size
    assert counters["prefix_hits"] == 2 * batch_size
    assert counters["prefix_publishes"] == 1
    assert counters["prefix_global_alias_scans"] == 0
    assert counters["prefix_publish_release_batch_calls"] == 1
    assert counters["release_batch_calls"] == 2
    assert counters["recycle_requests_batch_calls"] == 3
    assert counters["capacity_memset_bytes"] == 0
    assert counters["root_entries_crossed"] == 0
    assert counters["materialized_page_objects"] == 0
    assert counters["cow_copy_intents"] == 0
    assert counters["cow_move_calls"] == 0
    assert counters["cow_copied_tokens"] == 0


def test_b4_counter_contract_rejects_any_extra_hot_or_prefix_call():
    seed_only = _settled_manager_state(
        batch_size=4,
        completed_iterations=0,
        hybrid=False,
    )
    seed_census = bench.manager_census(
        _runtime_info(orbitkv_manager=seed_only),
        _manager_config(hybrid=False),
        {"full_tokens": 64, "swa_tokens": None},
        "after_prefix_seed",
        batch_size=4,
        completed_iterations=0,
        decode_tokens=33,
        prefix_seeded=True,
    )
    seed_counters = seed_census["batch_counters"]
    assert seed_counters["request_acquire_batch_calls"] == 1
    assert seed_counters["prepare_batch_calls"] == 1
    assert seed_counters["prefix_matches"] == 1
    assert seed_counters["prefix_hits"] == 0
    assert seed_counters["prefix_publish_release_batch_calls"] == 1
    assert seed_counters["prefix_publishes"] == 1
    assert seed_counters["release_batch_calls"] == 0
    assert seed_counters["recycle_requests_batch_calls"] == 1

    full_width = _settled_manager_state(
        batch_size=4,
        completed_iterations=5,
        hybrid=False,
    )
    census = bench.manager_census(
        _runtime_info(orbitkv_manager=full_width),
        _manager_config(hybrid=False),
        {"full_tokens": 64, "swa_tokens": None},
        "after_workload",
        batch_size=4,
        completed_iterations=5,
        decode_tokens=33,
        prefix_seeded=True,
    )
    assert census["batch_counters"]["prepare_batch_calls"] == 166
    assert census["batch_counters"]["prefix_attach_batch_calls"] == 20
    assert census["batch_counters"]["prefix_matches"] == 21
    assert census["batch_counters"]["release_batch_calls"] == 5
    assert census["batch_counters"]["recycle_requests_batch_calls"] == 6

    mixed_tail = _settled_manager_state(batch_size=4, completed_iterations=5, hybrid=False)
    mixed_tail["batch_counters"]["prepare_batch_calls"] += 1
    with pytest.raises(RuntimeError, match="B4 batch identities"):
        bench.manager_census(
            _runtime_info(orbitkv_manager=mixed_tail),
            _manager_config(hybrid=False),
            {"full_tokens": 64, "swa_tokens": None},
            "after_workload",
            batch_size=4,
            completed_iterations=5,
            decode_tokens=33,
            prefix_seeded=True,
        )

    illegal_global_scan = _settled_manager_state(
        batch_size=4, completed_iterations=5, hybrid=False
    )
    illegal_global_scan["batch_counters"]["prefix_global_alias_scans"] = 1
    with pytest.raises(RuntimeError, match="B4 batch identities"):
        bench.manager_census(
            _runtime_info(orbitkv_manager=illegal_global_scan),
            _manager_config(hybrid=False),
            {"full_tokens": 64, "swa_tokens": None},
            "after_workload",
            batch_size=4,
            completed_iterations=5,
            decode_tokens=33,
            prefix_seeded=True,
        )


def test_request_token_digests_preserve_iteration_and_request_identity():
    outputs = [
        [{"output_ids": [1, 2]}, {"output_ids": [3]}],
        [{"output_ids": [1, 2]}, {"output_ids": [4]}],
    ]
    digests = bench.request_token_digests(outputs)
    assert len(digests) == 2 and all(len(row) == 2 for row in digests)
    assert digests[0][0] == digests[1][0]
    assert digests[0][1] != digests[1][1]


def test_request_traces_bind_inputs_rids_and_exact_output_tokens():
    outputs = [
        [
            {"output_ids": [1, 2], "meta_info": {"id": "rid-0"}},
            {"output_ids": [3], "meta_info": {"id": "rid-1"}},
        ],
        [
            {"output_ids": [1, 2], "meta_info": {"id": "rid-2"}},
            {"output_ids": [3], "meta_info": {"id": "rid-3"}},
        ],
    ]
    traces = bench.request_traces(
        outputs=outputs,
        submitted_rids=(("rid-0", "rid-1"), ("rid-2", "rid-3")),
        submitted_input_digests=(("input-0", "input-1"),) * 2,
    )
    assert traces[0][0] == {
        "request_index": 0,
        "submitted_rid": "rid-0",
        "submitted_input_ids_sha256": "input-0",
        "returned_rid": "rid-0",
        "output_ids": [1, 2],
        "output_ids_sha256": bench.canonical_digest([1, 2]),
    }
    bench.verify_request_trace_stability(traces)

    outputs[1][1]["output_ids"] = [4]
    unstable = bench.request_traces(
        outputs=outputs,
        submitted_rids=(("rid-0", "rid-1"), ("rid-2", "rid-3")),
        submitted_input_digests=(("input-0", "input-1"),) * 2,
    )
    with pytest.raises(RuntimeError, match="request index 1"):
        bench.verify_request_trace_stability(unstable)


def test_request_traces_reject_foreign_returned_rid():
    with pytest.raises(RuntimeError, match="foreign request id"):
        bench.request_traces(
            outputs=(([{"output_ids": [1], "meta_info": {"id": "wrong"}}]),),
            submitted_rids=(("expected",),),
            submitted_input_digests=(("input",),),
        )


def test_abi6_counter_schema_never_fabricates_missing_internal_state_fields():
    state = _settled_manager_state()
    del state["batch_counters"]["capacity_memset_bytes"]
    with pytest.raises(RuntimeError, match="noncanonical field set"):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=state),
            _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 32},
            "after_workload",
            batch_size=4,
            completed_iterations=1,
            decode_tokens=33,
            prefix_seeded=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("prepare_batch_calls", 35, "B4 batch identities"),
        ("release_batch_calls", 2, "B4 batch identities"),
        ("event_queries", 0, "event observation counters"),
        ("capacity_memset_bytes", 16, "failure counters"),
        ("root_entries_crossed", 1, "failure counters"),
        ("retryable_conflicts", 1, "failure counters"),
        ("prefix_hits", 3, "B4 batch identities"),
        ("cow_copy_intents", 1, "B4 batch identities"),
        ("mirror_syncs", 0, "global mirror cleanup"),
    ),
)
def test_abi6_counter_identity_or_hot_memset_mismatch_fails(field, value, message):
    state = _settled_manager_state()
    state["batch_counters"][field] = value
    with pytest.raises(RuntimeError, match=message):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=state),
            _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 32},
            "after_workload",
            batch_size=4,
            completed_iterations=1,
            decode_tokens=33,
            prefix_seeded=True,
        )


def test_benchmark_record_schema_is_direct_v6_without_compatibility_aliases():
    assert bench.RECORD_SCHEMA == (
        "orbitkv.sglang-v0517-prefix-cow-single-run.v6"
    )
    assert {
        "prefix_matches",
        "prefix_hits",
        "prefix_publishes",
        "prefix_evictions",
        "prefix_global_alias_scans",
        "cow_copy_intents",
        "mirror_validation_calls",
        "mirror_syncs",
    } < set(bench._BATCH_COUNTER_FIELDS)
    assert "retryable_conflicts" in bench._FORBIDDEN_COUNTER_FIELDS
    assert ".v2" not in Path(bench.__file__).read_text(encoding="utf-8")
