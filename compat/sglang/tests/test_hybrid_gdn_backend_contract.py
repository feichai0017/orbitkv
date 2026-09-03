from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.state as state  # noqa: E402
import orbitkv_sglang.bridge.validation as validation  # noqa: E402
import orbitkv_sglang.qualification as qualification  # noqa: E402
import orbitkv_sglang.runtime_policy as runtime_policy  # noqa: E402
from orbitkv_sglang.config import (  # noqa: E402
    ClassConfig,
    FixedStateConfig,
    RuntimeConfig,
)


def _fixed_state_config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:hybrid-gdn-backend-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                name="full_mha",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
        ),
        fixed_states=(
            FixedStateConfig(
                name="gdn_recurrent",
                kind="gdn",
                layers=(1,),
                state_bytes_per_layer=16,
                checkpoint_slots_per_request=2,
            ),
            FixedStateConfig(
                name="gdn_convolution",
                kind="convolution",
                layers=(1,),
                state_bytes_per_layer=8,
                checkpoint_slots_per_request=2,
                kernel_width=4,
            ),
        ),
    )


def _gdn_backend_configurator(**overrides):
    values = {
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "linear_attn_prefill_backend": "triton",
        "mamba_ssm_dtype": "float32",
        "mamba_radix_cache_strategy": "no_buffer",
        "disable_radix_cache": True,
    }
    values.update(overrides)
    attention_backends = values.pop("attention_backends", ("fa3", "fa3"))
    temporal_dtype = values.pop("temporal_dtype", torch.float32)
    mambaish = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            dtype=SimpleNamespace(temporal=temporal_dtype)
        )
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["RenamedGdnForConditionalGeneration"]
            )
        ),
        use_mla_backend=False,
        mambaish_config=mambaish,
        hybrid_gdn_config=mambaish,
        server_args=SimpleNamespace(
            get_attention_backends=lambda: attention_backends, **values
        ),
    )


def _install_fixed_state_config() -> None:
    state._install_test_state(config=_fixed_state_config())


def test_gdn_fixed_state_backend_contract_accepts_renamed_architecture():
    _install_fixed_state_config()
    config = _gdn_backend_configurator()

    validation._validate_attention_backend_contract(config)
    validation._validate_gdn_fixed_state_backend_contract(config)


def test_qualification_profile_mutation_cannot_change_gdn_runtime_policy(
    monkeypatch,
):
    _install_fixed_state_config()
    runtime_profile = runtime_policy.GDN_FIXED_STATE_BACKEND_PROFILE
    assert validation.GDN_FIXED_STATE_BACKEND_PROFILE is runtime_profile
    qualification_profile = qualification.HYBRID_FIXED_STATE_BACKEND_PROFILE
    assert qualification_profile == runtime_profile
    assert qualification_profile is not runtime_profile
    with pytest.raises(TypeError):
        runtime_profile["linear_attn_backend"] = "flashinfer"

    monkeypatch.setitem(
        qualification_profile,
        "linear_attn_backend",
        "flashinfer",
    )
    assert runtime_profile["linear_attn_backend"] == "triton"
    with pytest.raises(RuntimeError, match="linear_attn_backend=triton"):
        validation._validate_gdn_fixed_state_backend_contract(
            _gdn_backend_configurator(linear_attn_backend="flashinfer")
        )


def test_qualification_module_exposes_only_capability_named_profile_data():
    assert set(qualification.__all__) == {
        "HYBRID_FIXED_STATE_BACKEND_PROFILE",
        "PAGE_TOKENS",
        "checkpoint_attention_contract",
    }
    assert all("QWEN" not in name for name in vars(qualification))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("attention_backends", ("flashinfer", "flashinfer"), "Full attention FA3"),
        ("linear_attn_backend", "flashinfer", "linear_attn_backend=triton"),
        (
            "linear_attn_decode_backend",
            "flashinfer",
            "linear_attn_decode_backend=triton",
        ),
        (
            "linear_attn_prefill_backend",
            "flashinfer",
            "linear_attn_prefill_backend=triton",
        ),
        ("mamba_ssm_dtype", "bfloat16", "mamba_ssm_dtype=float32"),
        (
            "temporal_dtype",
            torch.bfloat16,
            "mamba2_cache_params.dtype.temporal=float32",
        ),
        (
            "mamba_radix_cache_strategy",
            "auto",
            "mamba_radix_cache_strategy=no_buffer",
        ),
    ),
)
def test_gdn_fixed_state_backend_contract_rejects_drift_for_renamed_architecture(
    field, value, message
):
    _install_fixed_state_config()
    config = _gdn_backend_configurator(**{field: value})

    with pytest.raises(RuntimeError, match=message):
        validation._validate_gdn_fixed_state_backend_contract(config)


def test_fixed_state_backend_contract_requires_fa3_for_non_gdn_profiles():
    config = _fixed_state_config()
    recurrent = FixedStateConfig(
        name="mamba_recurrent",
        kind="mamba",
        layers=(1,),
        state_bytes_per_layer=24,
        checkpoint_slots_per_request=2,
    )
    state._install_test_state(
        config=RuntimeConfig(
            library_path=config.library_path,
            plan_json=config.plan_json,
            plan_fingerprint=config.plan_fingerprint,
            page_tokens=config.page_tokens,
            classes=config.classes,
            fixed_states=(recurrent,),
        )
    )
    config = _gdn_backend_configurator(
        attention_backends=("flashinfer", "flashinfer"),
        linear_attn_backend="flashinfer",
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
        mamba_ssm_dtype="bfloat16",
        mamba_radix_cache_strategy="auto",
        disable_radix_cache=False,
    )
    config.model_config.hf_config.architectures = ["LegacyMambaForCausalLM"]

    with pytest.raises(RuntimeError, match="fixed-state capability requires SGLang FA3"):
        validation._validate_attention_backend_contract(config)
    validation._validate_gdn_fixed_state_backend_contract(config)


def test_hybrid_gdn_full_pool_validation_checks_wrapper_and_inner_pool(
    monkeypatch,
):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    _install_fixed_state_config()
    inner = SimpleNamespace(
        size=128, page_size=16, dtype=torch.bfloat16, kv_cache_layout="nhd"
    )
    wrapper = object.__new__(HybridLinearKVPool)
    wrapper.size = 128
    wrapper.page_size = 16
    wrapper.dtype = torch.bfloat16
    wrapper.use_mla = False
    wrapper.full_attention_layer_id_mapping = {0: 0}
    wrapper.full_kv_pool = inner
    validation._validate_full_physical_pool(
        wrapper, expected_tokens=128, expected_dtype=torch.bfloat16,
        storage="token_kv",
    )
    inner.kv_cache_layout = "hnd"
    with pytest.raises(RuntimeError, match="not NHD"):
        validation._validate_full_physical_pool(
            wrapper, expected_tokens=128, expected_dtype=torch.bfloat16,
            storage="token_kv",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("size", 144), ("page_size", 8), ("dtype", torch.float32),
        ("use_mla", True), ("full_attention_layer_id_mapping", {1: 0}),
    ),
)
def test_hybrid_gdn_full_pool_validation_rejects_wrapper_drift(
    field, value
):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    _install_fixed_state_config()
    wrapper = object.__new__(HybridLinearKVPool)
    wrapper.size = 128
    wrapper.page_size = 16
    wrapper.dtype = torch.bfloat16
    wrapper.use_mla = False
    wrapper.full_attention_layer_id_mapping = {0: 0}
    wrapper.full_kv_pool = SimpleNamespace(
        size=128, page_size=16, dtype=torch.bfloat16, kv_cache_layout="nhd"
    )
    setattr(wrapper, field, value)
    with pytest.raises(RuntimeError, match="hybrid-linear KV pool envelope"):
        validation._validate_full_physical_pool(
            wrapper, expected_tokens=128, expected_dtype=torch.bfloat16,
            storage="token_kv",
        )


def test_validate_configurator_rejects_hybrid_gdn_backend_drift_before_build():
    _install_fixed_state_config()
    config = _gdn_backend_configurator(linear_attn_backend="flashinfer")
    config.server_args.radix_cache_backend = "orbitkv"
    config.server_args.cuda_graph_config = SimpleNamespace()
    original_called = False

    def original_fn(*_args, **_kwargs):
        nonlocal original_called
        original_called = True
        raise AssertionError("the native configurator must not run")

    with pytest.raises(RuntimeError, match="linear_attn_backend=triton"):
        validation._validate_configurator(original_fn, config)
    assert not original_called
