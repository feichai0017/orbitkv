from __future__ import annotations

import inspect
import json
import os
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
from orbitkv_sglang.benchmark_profiles import expected_mirror_transactions  # noqa: E402
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
        "state_plan": None,
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
        "attention_backend": "fa3",
        "fp8_gemm_backend": None,
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


def test_checkpoint_identity_hashes_weight_contents(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    weight = model / "model.safetensors"
    weight.write_bytes(b"first")
    first = bench.checkpoint_identity(model, "auto")
    weight.write_bytes(b"other")
    second = bench.checkpoint_identity(model, "auto")
    assert first["weight_files"][0]["bytes"] == second["weight_files"][0]["bytes"]
    assert first["weight_files"][0]["sha256"] != second["weight_files"][0]["sha256"]


def _write_config(tmp_path: Path, value: dict) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(value), encoding="utf-8")
    return model


def _qwen35_config() -> dict:
    return json.loads(
        (REPOSITORY_ROOT / "fixtures/qwen3.5-0.8b/config.json").read_text(
            encoding="utf-8"
        )
    )


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
    assert "--state-plan" in completed.stdout
    assert "--attention-backend {fa3,flashinfer}" in completed.stdout
    assert (
        "--fp8-gemm-backend "
        "{deep_gemm,flashinfer_deepgemm,triton}"
    ) in completed.stdout


def test_fp8_gemm_backend_parser_defaults_to_omitted_and_rejects_auto():
    action = next(
        action
        for action in bench.build_parser()._actions
        if action.dest == "fp8_gemm_backend"
    )
    assert action.default is None
    assert tuple(action.choices) == bench.FP8_GEMM_BACKENDS
    assert "auto" not in action.choices
    assert "aiter" not in action.choices


def test_compact_control_help_requires_exact_abi8_matrix_dimensions():
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
def test_compact_control_abi8_runs_real_host_batches_without_root_materialization(
    compact_ffi_library, profile, batch_size, class_count
):
    result = compact.run(
        compact_ffi_library,
        profile=profile,
        batch_size=batch_size,
        resident_pages=32,
        iterations=32,
    )
    assert result["schema"] == "orbitkv.abi8-compact-control.v1"
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
    assert values["attention_backend"] == "fa3"
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
        bench.engine_arguments(
            _arguments(mode=mode, attention_backend="flashinfer"),
            Path("/model"),
            contract,
        )


def test_qwen2_engine_profile_rejects_flashinfer():
    with pytest.raises(RuntimeError, match="requires --attention-backend fa3"):
        bench.engine_arguments(
            _arguments(attention_backend="flashinfer"),
            Path("/model"),
            _attention_contract(),
        )


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_qwen35_engine_profile_is_exact_and_disables_radix(mode):
    contract = _attention_contract(
        "Qwen3_5ForConditionalGeneration",
        workload_profile="fresh_prompt",
        backend_profile={
            "attention_backend": "fa3",
            "linear_attn_backend": "triton",
            "linear_attn_decode_backend": "triton",
            "linear_attn_prefill_backend": "triton",
            "mamba_ssm_dtype": "float32",
            "mamba_radix_cache_strategy": "no_buffer",
        },
    )
    values = bench.engine_arguments(_arguments(mode=mode), Path("/model"), contract)
    assert values["disable_radix_cache"] is True
    assert values["attention_backend"] == "fa3"
    assert values["linear_attn_backend"] == "triton"
    assert values["linear_attn_decode_backend"] == "triton"
    assert values["linear_attn_prefill_backend"] == "triton"
    assert values["mamba_ssm_dtype"] == "float32"
    assert values["mamba_radix_cache_strategy"] == "no_buffer"
    if mode == "manager":
        assert values["radix_cache_backend"] == "orbitkv"
    else:
        assert "radix_cache_backend" not in values


def test_engine_profile_enables_batch_invariant_inference():
    values = bench.engine_arguments(
        _arguments(), Path("/model"), _attention_contract()
    )
    assert values["enable_deterministic_inference"] is True
    assert values["sampling_backend"] == "pytorch"


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_explicit_fp8_gemm_backend_maps_to_sglang_engine_argument(mode):
    default_values = bench.engine_arguments(
        _arguments(mode=mode), Path("/model"), _attention_contract()
    )
    assert "fp8_gemm_runner_backend" not in default_values

    explicit_values = bench.engine_arguments(
        _arguments(mode=mode, fp8_gemm_backend="triton"),
        Path("/model"),
        _attention_contract(),
    )
    assert explicit_values["fp8_gemm_runner_backend"] == "triton"


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


def test_qwen35_fresh_inputs_share_no_complete_prefix_page_across_iterations():
    forbidden = (1019, 1020, 1021, 1022)
    rows = [
        bench.fresh_input_ids(
            requests=4, prompt_tokens=513, vocab_size=1024,
            seed=20260820, iteration=iteration,
            forbidden_token_ids=forbidden,
            token_upper_bound=1019,
        )
        for iteration in range(5)
    ]
    prompts = [prompt for row in rows for prompt in row]
    assert len({tuple(prompt[:16]) for prompt in prompts}) == 20
    assert len({tuple(prompt) for prompt in prompts}) == 20
    assert not set(forbidden).intersection(
        token for prompt in prompts for token in prompt
    )
    assert all(token < 1019 for prompt in prompts for token in prompt)


def test_fresh_inputs_reject_a_matrix_that_cannot_keep_first_pages_unique():
    with pytest.raises(RuntimeError, match="collision-free token domain"):
        bench.fresh_input_ids(
            requests=2, prompt_tokens=16, vocab_size=6, seed=1, iteration=1,
            forbidden_token_ids=(5,),
        )


def test_fresh_inputs_allow_large_seeds_without_reusing_first_tokens():
    first = bench.fresh_input_ids(
        requests=2, prompt_tokens=16, vocab_size=16, seed=10_000, iteration=0
    )
    second = bench.fresh_input_ids(
        requests=2, prompt_tokens=16, vocab_size=16, seed=10_000, iteration=1
    )
    assert len({row[0] for row in (*first, *second)}) == 4


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


def test_pair_normalization_binds_explicit_fp8_gemm_backend():
    manager = bench.engine_arguments(
        _arguments(mode="manager", fp8_gemm_backend="triton"),
        Path("/model"),
        _attention_contract(),
    )
    stock = bench.engine_arguments(
        _arguments(mode="stock", fp8_gemm_backend="triton"),
        Path("/model"),
        _attention_contract(),
    )
    paired_manager = bench.pair_engine_arguments("manager", manager)
    paired_stock = bench.pair_engine_arguments("stock", stock)
    assert paired_manager == paired_stock
    assert paired_manager["fp8_gemm_runner_backend"] == "triton"

    stock["fp8_gemm_runner_backend"] = "deep_gemm"
    assert bench.pair_engine_arguments("stock", stock) != paired_manager


def test_manager_accepts_same_explicit_storage_cap(tmp_path):
    paths = _checkout_inputs(tmp_path)
    args = _arguments(**paths)
    resolved = bench.validate_arguments(args)
    assert resolved["plan"] == Path(paths["plan"]).resolve()
    assert resolved["library"] == Path(paths["library"]).resolve()


def test_manager_state_plan_is_explicit_and_exported(tmp_path, monkeypatch):
    paths = _checkout_inputs(tmp_path)
    state_plan = tmp_path / "state-plan.json"
    state_plan.write_text("{}", encoding="utf-8")
    args = _arguments(**paths, state_plan=str(state_plan))
    resolved = bench.validate_arguments(args)
    assert resolved["state_plan"] == state_plan.resolve()
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name)
    environment = bench.configure_environment(args, resolved)
    assert environment["ORBITKV_STATE_PLAN"] == str(state_plan.resolve())

    with pytest.raises(ValueError, match="stock mode forbids"):
        bench.validate_arguments(
            _arguments(
                mode="stock", plan=None, library=None,
                state_plan=str(state_plan),
                **{key: paths[key] for key in ("sglang_root", "model")},
            )
        )


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


def test_abi8_qualification_accepts_only_complete_b1_or_b4_batches(tmp_path):
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
    assert contract["attention_backend"] == "fa3"
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


def test_deepseek_v2_contract_matches_explicit_mla_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "checkpoint_identity", _checkpoint_identity)
    model = _write_config(
        tmp_path,
        {
            "architectures": ["DeepseekV2ForCausalLM"],
            "num_hidden_layers": 3,
            "vocab_size": 256,
            "max_position_embeddings": 4096,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
    )
    latent = SimpleNamespace(
        name="latent_mla",
        retention="full",
        layers=(0, 1, 2),
        window_tokens=None,
        storage="latent_kv",
        components_by_name={"latent": 1024, "rope": 128},
    )
    manager_config = SimpleNamespace(
        page_tokens=16, num_hidden_layers=3, classes=(latent,)
    )
    contract, _identity = bench.checkpoint_contract(model, manager_config)
    assert contract["attention_profile"] == "mla"
    assert contract["attention_backend"] == "flashinfer"
    assert contract["classes"][0]["name"] == "latent_mla"

    latent.components_by_name["rope"] = 64
    with pytest.raises(RuntimeError, match="MLA geometry"):
        bench.checkpoint_contract(model, manager_config)


def test_qwen35_contract_uses_nested_text_config_and_exact_state_geometry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bench, "checkpoint_identity", _checkpoint_identity)
    model = _write_config(tmp_path, _qwen35_config())
    config = SimpleNamespace(
        page_tokens=16,
        num_hidden_layers=24,
        classes=(SimpleNamespace(
            name="full_attention_kv", retention="full",
            layers=(3, 7, 11, 15, 19, 23), window_tokens=None,
            storage="token_kv", bytes_per_token_per_layer=2048,
            components=(("key", 1024), ("value", 1024)),
        ),),
        fixed_states=(
            SimpleNamespace(
                name="gdn_recurrent", kind="gdn",
                layers=tuple(i for i in range(24) if i % 4 != 3),
                state_bytes_per_layer=1_048_576,
                checkpoint_slots_per_request=2, kernel_width=None,
            ),
            SimpleNamespace(
                name="gdn_convolution", kind="convolution",
                layers=tuple(i for i in range(24) if i % 4 != 3),
                state_bytes_per_layer=36_864,
                checkpoint_slots_per_request=2, kernel_width=4,
            ),
        ),
    )
    contract, _identity = bench.checkpoint_contract(model, config)
    assert contract["attention_profile"] == "hybrid_full_gdn"
    assert contract["workload_profile"] == "fresh_prompt"
    assert contract["state_ownership"] == "request_private"
    assert contract["vocab_size"] == 248320
    assert contract["prompt_token_upper_bound"] == 248044
    assert contract["control_token_ids"] == {
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
    }
    assert contract["max_position_embeddings"] == 262144
    assert contract["backend_profile"]["linear_attn_backend"] == "triton"
    assert contract["fixed_states"][0]["state_bytes_per_layer"] == 1_048_576
    assert contract["fixed_states"][1]["state_bytes_per_layer"] == 36_864

    broken = _qwen35_config()
    broken["text_config"] = None
    (model / "config.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RuntimeError, match="nested text_config"):
        bench.checkpoint_contract(model)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("model_type",), "other", "model_type=qwen3_5"),
        (
            ("text_config", "model_type"),
            "other",
            "text_config.model_type=qwen3_5_text",
        ),
        (
            ("text_config", "full_attention_interval"),
            3,
            "layer_types differs from full_attention_interval",
        ),
    ),
)
def test_qwen35_checkpoint_contract_rejects_discriminator_or_schedule_drift(
    tmp_path, path, value, message
):
    config = _qwen35_config()
    target = config
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    model = _write_config(tmp_path, config)
    with pytest.raises(RuntimeError, match=message):
        bench.checkpoint_contract(model)


def test_checkpoint_and_source_gates_have_no_legacy_attention_path():
    assert bench.SUPPORTED_ARCHITECTURES == (
        "Qwen2ForCausalLM",
        "GptOssForCausalLM",
        "DeepseekV2ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
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
    attention_backend="fa3",
    moe_runner_backend="auto",
    radix_cache_backend="orbitkv",
    swa_tokens=None,
    orbitkv_manager=None,
    fp8_gemm_runner_backend=None,
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
    if fp8_gemm_runner_backend is not None:
        state["fp8_gemm_runner_backend"] = fp8_gemm_runner_backend
    return {"internal_states": [state]}


def _manager_config(*, hybrid=True, fixed_state=False):
    classes = [
        SimpleNamespace(
            class_id=0,
            pool_id=1,
            backend_domain=1,
            name="full",
            layers=(0,),
            retention="full",
            bytes_per_token_per_layer=128,
            window_tokens=None,
            period_blocks=None,
            storage="token_kv",
            components=(("key", 64), ("value", 64)),
        )
    ]
    if hybrid:
        classes.append(
            SimpleNamespace(
                class_id=1,
                pool_id=2,
                backend_domain=2,
                name="swa",
                layers=(1,),
                retention="sliding",
                bytes_per_token_per_layer=128,
                window_tokens=32,
                period_blocks=3,
                storage="token_kv",
                components=(("key", 64), ("value", 64)),
            )
        )
    fixed_states = (
        (
            SimpleNamespace(
                name="mamba_state",
                kind="mamba",
                layers=(2 if hybrid else 1,),
                state_bytes_per_layer=96,
                checkpoint_slots_per_request=2,
                kernel_width=None,
                byte_count=96,
            ),
        )
        if fixed_state
        else ()
    )
    return SimpleNamespace(
        plan_fingerprint="sha256:manager-test",
        page_tokens=16,
        classes=tuple(classes),
        fixed_states=fixed_states,
        fixed_state_byte_count=96 if fixed_state else 0,
        state_plan_path=None,
        state_plan_fingerprint=None,
    )


def _hybrid_config():
    return _manager_config(hybrid=True)


def _drained_fixed_state(*, slot_count=8):
    return {
        "status": "host_seam",
        "identity": {
            "engine_epoch": 7,
            "pool_epoch": 9,
            "pool_id": 3,
            "byte_count": 96,
            "slot_count": slot_count,
        },
        "free_slots": slot_count,
        "reserved_slots": 0,
        "relocating_slots": 0,
        "live_slots": 0,
        "retiring_slots": 0,
        "quarantined_slots": 0,
        "active_owners": 0,
        "pending_transitions": 0,
        "pending_retirements": 0,
    }


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
    worker_plan = bench._expected_worker_plan_readback(
        _manager_config(hybrid=hybrid)
    )
    return {
        "abi_version": 8,
        **worker_plan,
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


def _fixed_state_info(raw=None):
    state = _settled_manager_state()
    state.update(
        bench._expected_worker_plan_readback(_manager_config(fixed_state=True))
    )
    state["fixed_state"] = _drained_fixed_state() if raw is None else raw
    info = _runtime_info(swa_tokens=32, orbitkv_manager=state)
    info["internal_states"][0]["max_mamba_cache_size"] = 8
    return info


def _attach_fixed_state_runtime_plan(state, *, hybrid=True):
    state.update(
        bench._expected_worker_plan_readback(
            _manager_config(hybrid=hybrid, fixed_state=True)
        )
    )
    return state


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


@pytest.mark.parametrize("mode", ("manager", "stock"))
def test_runtime_readback_verifies_explicit_fp8_gemm_backend(mode):
    radix_cache_backend = "orbitkv" if mode == "manager" else None
    contract = _attention_contract(attention_profile="full")
    result = bench.verify_runtime_contract(
        _arguments(mode=mode, fp8_gemm_backend="triton"),
        _runtime_info(
            radix_cache_backend=radix_cache_backend,
            fp8_gemm_runner_backend="triton",
        ),
        contract,
    )
    assert result["full_tokens"] == 4096

    with pytest.raises(
        RuntimeError, match="fp8_gemm_runner_backend.*deep_gemm"
    ):
        bench.verify_runtime_contract(
            _arguments(mode=mode, fp8_gemm_backend="triton"),
            _runtime_info(
                radix_cache_backend=radix_cache_backend,
                fp8_gemm_runner_backend="deep_gemm",
            ),
            contract,
        )


def test_multi_arena_live_prefix_census_requires_exact_abi8_ref_schema():
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
    assert census["abi_version"] == 8
    expected_worker_plan = bench._expected_worker_plan_readback(_hybrid_config())
    assert {name: census[name] for name in expected_worker_plan} == expected_worker_plan
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
    assert "fixed_state" not in census


def test_manager_plan_identity_records_state_plan_and_fixed_components(tmp_path):
    manager_plan = tmp_path / "manager.json"
    state_plan = tmp_path / "state.json"
    manager_plan.write_text("{}", encoding="utf-8")
    state_plan.write_text("{}", encoding="utf-8")
    config = _manager_config(fixed_state=True)
    config.plan_path = manager_plan
    config.plan_fingerprint = "sha256:manager"
    config.page_tokens = 16
    config.state_plan_path = state_plan
    config.state_plan_fingerprint = "sha256:state"

    identity = bench.manager_plan_identity(config)

    assert identity["state_plan"] == {
        "artifact": bench.artifact_identity(state_plan),
        "plan_fingerprint": "sha256:state",
    }
    assert identity["fixed_state_byte_count"] == 96
    assert identity["fixed_states"] == [
        {
            "name": "mamba_state",
            "kind": "mamba",
            "layers": [2],
            "state_bytes_per_layer": 96,
            "checkpoint_slots_per_request": 2,
            "kernel_width": None,
            "byte_count": 96,
        }
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("plan_fingerprint",), "sha256:foreign-manager"),
        (("state_plan_fingerprint",), "sha256:foreign-state"),
        (("fixed_state_descriptors", 0, "state_bytes_per_layer"), 95),
        (("tree_cache_type", "module"), "sglang.srt.mem_cache.radix_cache"),
        (("tree_cache_type", "qualname"), "RadixCache"),
    ),
)
def test_worker_runtime_plan_rejects_parent_worker_mismatch(path, value):
    config = _manager_config(fixed_state=True)
    reported = bench._expected_worker_plan_readback(config)
    target = reported
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(RuntimeError, match="differs from the parent process"):
        bench.verify_worker_plan_readback(reported, config, "after_load")


def test_manager_census_requires_worker_plan_readback_fields():
    state = _settled_manager_state()
    del state["plan_fingerprint"]

    with pytest.raises(RuntimeError, match="top-level schema"):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=state),
            _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 32},
            "after_load",
            batch_size=4,
            completed_iterations=1,
            decode_tokens=33,
            prefix_seeded=True,
        )


def test_fixed_state_census_presence_matches_plan_and_is_preserved():
    config = _manager_config(fixed_state=True)
    census = bench.manager_census(
        _fixed_state_info(),
        config,
        {"full_tokens": 64, "swa_tokens": 32},
        "after_workload",
        batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
    )
    assert census["fixed_state"] == _drained_fixed_state()

    missing = _settled_manager_state()
    with pytest.raises(RuntimeError, match="top-level schema"):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=missing), config,
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )

    unexpected = _settled_manager_state()
    unexpected["fixed_state"] = _drained_fixed_state()
    with pytest.raises(RuntimeError, match="top-level schema"):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=unexpected), _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update(extra=0),
        lambda value: value.pop("pending_retirements"),
        lambda value: value.update(status="ready"),
        lambda value: value.update(identity=None),
    ),
)
def test_fixed_state_census_rejects_malformed_schema(mutate):
    raw = _drained_fixed_state()
    mutate(raw)
    with pytest.raises(RuntimeError, match="fixed-state"):
        bench.manager_census(
            _fixed_state_info(raw), _manager_config(fixed_state=True),
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("engine_epoch", 8), ("pool_epoch", 8), ("pool_id", 2),
     ("byte_count", 95), ("slot_count", 7)),
)
def test_fixed_state_census_rejects_identity_mismatch(field, value):
    raw = _drained_fixed_state()
    raw["identity"][field] = value
    with pytest.raises(RuntimeError, match="identity differs from plan"):
        bench.manager_census(
            _fixed_state_info(raw), _manager_config(fixed_state=True),
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )


@pytest.mark.parametrize(
    "field",
    ("reserved_slots", "relocating_slots", "live_slots", "retiring_slots",
     "quarantined_slots", "active_owners", "pending_transitions",
     "pending_retirements"),
)
def test_fixed_state_census_rejects_leaks(field):
    raw = _drained_fixed_state()
    raw[field] = 1
    if field.endswith("_slots"):
        raw["free_slots"] -= 1
    with pytest.raises(RuntimeError, match="did not drain"):
        bench.manager_census(
            _fixed_state_info(raw), _manager_config(fixed_state=True),
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )


def test_fixed_state_census_rejects_unaccounted_slot():
    raw = _drained_fixed_state()
    raw["free_slots"] -= 1
    with pytest.raises(RuntimeError, match="slot census is incomplete"):
        bench.manager_census(
            _fixed_state_info(raw), _manager_config(fixed_state=True),
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )


@pytest.mark.parametrize("field", bench._FIXED_STATE_COUNTER_FIELDS)
def test_token_only_census_requires_every_fixed_state_counter_to_be_zero(field):
    token_only = _settled_manager_state()
    token_only["batch_counters"][field] = 1
    with pytest.raises(RuntimeError, match="without a state plan"):
        bench.manager_census(
            _runtime_info(swa_tokens=32, orbitkv_manager=token_only), _hybrid_config(),
            {"full_tokens": 64, "swa_tokens": 32}, "after_workload",
            batch_size=4, completed_iterations=1, decode_tokens=33, prefix_seeded=True,
        )


@pytest.mark.parametrize(
    ("batch_size", "iterations", "expected_validation", "expected_syncs"),
    ((1, 1, 4, 1), (4, 5, 20, 5)),
)
def test_fresh_gdn_census_requires_exact_fixed_state_and_mirror_activity(
    batch_size, iterations, expected_validation, expected_syncs
):
    state = _attach_fixed_state_runtime_plan(
        _settled_manager_state(
            batch_size=batch_size, completed_iterations=iterations, hybrid=False,
            prefix_seeded=False,
        ),
        hybrid=False,
    )
    counters = state["batch_counters"]
    expected = bench.expected_batch_counters(
        batch_size=batch_size, completed_iterations=iterations, prompt_tokens=513,
        decode_tokens=33,
        hybrid=False, prefix_seeded=False, global_cleanup=False, fresh=True,
    )
    counters.update(expected)
    counters.update(
        fixed_state_prepares=iterations * batch_size,
        fixed_state_clears=iterations * batch_size,
        fixed_state_copies=0,
        fixed_state_events=33 * iterations,
        fixed_state_retirements=iterations * batch_size,
        fixed_state_acks=iterations * batch_size,
    )
    fixed = _drained_fixed_state()
    fixed["identity"]["pool_id"] = 2
    state["fixed_state"] = fixed
    info = _runtime_info(orbitkv_manager=state)
    info["internal_states"][0]["max_mamba_cache_size"] = 8
    census = bench.manager_census(
        info, _manager_config(hybrid=False, fixed_state=True),
        {"full_tokens": 64, "swa_tokens": None}, "after_workload",
        batch_size=batch_size, completed_iterations=iterations, prompt_tokens=513,
        decode_tokens=33,
        fresh=True,
    )
    assert census["batch_counters"]["fixed_state_prepares"] == (
        iterations * batch_size
    )
    assert census["batch_counters"]["fixed_state_copies"] == 0
    assert census["batch_counters"]["mirror_validation_calls"] == (
        expected_validation
    )
    assert census["batch_counters"]["mirror_syncs"] == expected_syncs
    assert census["fixed_state"]["free_slots"] == 8

    for field in bench._FIXED_STATE_COUNTER_FIELDS:
        broken = dict(state)
        broken["batch_counters"] = dict(counters)
        broken["batch_counters"][field] += 1
        broken_info = _runtime_info(orbitkv_manager=broken)
        broken_info["internal_states"][0]["max_mamba_cache_size"] = 8
        with pytest.raises(RuntimeError, match="fixed-state lifecycle"):
            bench.manager_census(
                broken_info,
                _manager_config(hybrid=False, fixed_state=True),
                {"full_tokens": 64, "swa_tokens": None}, "after_workload",
                batch_size=batch_size, completed_iterations=iterations,
                prompt_tokens=513,
                decode_tokens=33,
                fresh=True,
            )

    for field in ("mirror_validation_calls", "mirror_syncs"):
        broken = dict(state)
        broken["batch_counters"] = dict(counters)
        broken["batch_counters"][field] += 1
        broken_info = _runtime_info(orbitkv_manager=broken)
        broken_info["internal_states"][0]["max_mamba_cache_size"] = 8
        with pytest.raises(RuntimeError, match="mirror transaction counts"):
            bench.manager_census(
                broken_info,
                _manager_config(hybrid=False, fixed_state=True),
                {"full_tokens": 64, "swa_tokens": None}, "after_workload",
                batch_size=batch_size, completed_iterations=iterations,
                prompt_tokens=513, decode_tokens=33, fresh=True,
            )


@pytest.mark.parametrize(
    ("batch_size", "iterations", "expected_validation", "expected_syncs"),
    (
        (1, 1, 4, 1),
        (4, 5, 20, 5),
    ),
)
def test_v3_fresh_mirror_validation_and_sync_counts_are_distinct(
    batch_size, iterations, expected_validation, expected_syncs
):
    counters = bench.expected_batch_counters(
        batch_size=batch_size, completed_iterations=iterations,
        prompt_tokens=513, decode_tokens=33, hybrid=False,
        prefix_seeded=False,
        global_cleanup=False,
        fresh=True,
    )
    assert counters["mirror_validation_calls"] == expected_validation
    assert counters["mirror_syncs"] == expected_syncs


@pytest.mark.parametrize(
    ("prompt_tokens", "decode_tokens", "iterations", "expected"),
    ((17, 2, 1, 2), (16, 2, 1, 3)),
)
def test_fresh_mirror_validation_follows_page_boundaries(
    prompt_tokens, decode_tokens, iterations, expected
):
    assert expected_mirror_transactions(
        prompt_tokens=prompt_tokens, decode_tokens=decode_tokens,
        completed_iterations=iterations, prefix_seeded=False,
        global_cleanup=False, fresh=True,
    ) == expected


def test_v2_fresh_and_v3_prefix_reuse_keep_equal_mirror_counters():
    legacy_fresh = bench.expected_batch_counters(
        batch_size=4, completed_iterations=5, prompt_tokens=513,
        decode_tokens=33, hybrid=False, prefix_seeded=False,
        global_cleanup=False, fresh=True,
        legacy_equal_mirror_counters=True,
    )
    assert legacy_fresh["mirror_validation_calls"] == 20
    assert legacy_fresh["mirror_syncs"] == 20

    prefix_reuse = bench.expected_batch_counters(
        batch_size=4, completed_iterations=5, prompt_tokens=513,
        decode_tokens=33, hybrid=False, prefix_seeded=True,
        global_cleanup=True, fresh=False,
    )
    assert prefix_reuse["mirror_validation_calls"] == 7
    assert prefix_reuse["mirror_syncs"] == 7


def test_fresh_counter_contract_requires_prompt_length():
    with pytest.raises(RuntimeError, match="requires prompt_tokens"):
        bench.expected_batch_counters(
            batch_size=1, completed_iterations=1, decode_tokens=33,
            hybrid=False, prefix_seeded=False, global_cleanup=False, fresh=True,
        )


def test_fresh_manager_census_requires_prompt_length():
    state = _attach_fixed_state_runtime_plan(
        _settled_manager_state(
            batch_size=1, completed_iterations=1, hybrid=False,
            prefix_seeded=False,
        ),
        hybrid=False,
    )
    info = _runtime_info(orbitkv_manager=state)
    info["internal_states"][0]["max_mamba_cache_size"] = 8

    with pytest.raises(RuntimeError, match="omitted prompt length"):
        bench.manager_census(
            info, _manager_config(hybrid=False, fixed_state=True),
            {"full_tokens": 64, "swa_tokens": None}, "after_workload",
            batch_size=1, completed_iterations=1, decode_tokens=33,
            fresh=True,
        )


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
def test_abi8_batch_counter_identities_are_hard_validated(batch_size, hybrid):
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
            {"output_ids": [1, 2], "meta_info": {"id": "rid-0", "cached_tokens": 16}},
            {"output_ids": [3], "meta_info": {"id": "rid-1", "cached_tokens": 16}},
        ],
        [
            {"output_ids": [1, 2], "meta_info": {"id": "rid-2", "cached_tokens": 16}},
            {"output_ids": [3], "meta_info": {"id": "rid-3", "cached_tokens": 16}},
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
        "cached_tokens": 16,
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


def test_abi8_counter_schema_never_fabricates_missing_internal_state_fields():
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
        ("mirror_syncs", 0, "mirror transaction counts"),
    ),
)
def test_abi8_counter_identity_or_hot_memset_mismatch_fails(field, value, message):
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


def test_benchmark_record_schema_and_mirror_contract_are_explicit_abi8_v3():
    assert bench.RECORD_SCHEMA == "orbitkv.sglang-v0517-abi8-single-run.v3"
    contract = bench._manager_counter_contract(fresh=True)
    assert contract["mirror_validation_scope"] == (
        "all_validated_mirror_transactions"
    )
    assert contract["mirror_sync_scope"] == (
        "device_mutation_transactions_only"
    )
    assert "mirror_validation_equals_sync" not in contract
    assert {
        "prefix_matches",
        "prefix_hits",
        "prefix_publishes",
        "prefix_evictions",
        "prefix_global_alias_scans",
        "cow_copy_intents",
        "mirror_validation_calls",
        "mirror_syncs",
        *bench._FIXED_STATE_COUNTER_FIELDS,
    } < set(bench._BATCH_COUNTER_FIELDS)
    assert "retryable_conflicts" in bench._FORBIDDEN_COUNTER_FIELDS
