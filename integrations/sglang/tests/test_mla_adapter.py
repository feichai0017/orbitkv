from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import orbitkv_sglang.plugin.facade as facade
import orbitkv_sglang.plugin.validation as validation
from mla_helpers import drift_components, install_mla, latent_config, mla_configurator
from test_canonical_adapter import FakeFactory, _install


def _pool(*, rope_rank: int = 4):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return MLATokenToKVPool(
        size=128,
        page_size=16,
        dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=rope_rank,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=1,
    )


def test_mla_builder_installs_paged_facade_over_the_real_latent_pool():
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    config = latent_config()
    factory = FakeFactory()
    pool = _pool()
    _install(config, factory)
    allocator = facade._build_token_to_kv_pool_allocator(
        mla_configurator(),
        sizes=SimpleNamespace(max_total_num_tokens=128),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )
    assert isinstance(allocator, PagedTokenToKVPoolAllocator)
    validation._validate_physical_pool(
        pool,
        expected_tokens=128,
        expected_dtype=torch.bfloat16,
        name="Full",
        storage="latent_kv",
    )
    pool.kv_buffer[0][17].copy_(
        torch.arange(12, dtype=torch.bfloat16).reshape(1, 12)
    )
    pool.move_kv_cache(torch.tensor([33]), torch.tensor([17]))
    assert torch.equal(pool.kv_buffer[0][33], pool.kv_buffer[0][17])
    assert tuple(item.page_count for item in factory.calls[0][2]) == (8,)


def test_mla_builder_rejects_a_token_kv_runtime_pool():
    config = latent_config()
    install_mla(config)
    with pytest.raises(RuntimeError, match="attention storage differs"):
        facade._build_token_to_kv_pool_allocator(
            SimpleNamespace(
                page_size=16,
                device="cuda:0",
                kv_cache_dtype=torch.bfloat16,
                is_hybrid_swa=False,
                use_mla_backend=False,
                is_draft_worker=False,
            ),
            sizes=SimpleNamespace(max_total_num_tokens=128),
            token_to_kv_pool=object(),
            is_dsv4_model=False,
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=None,
        )


def test_mla_geometry_gate_rejects_dsa_and_component_drift(monkeypatch):
    config = latent_config(layers=(0, 1), latent_bytes=1024, rope_bytes=128)
    configurator = mla_configurator(kv_lora_rank=512, qk_rope_head_dim=64)
    configurator.model_config.hf_text_config.num_hidden_layers = 2
    install_mla(config)
    validation._validate_checkpoint_geometry(configurator)

    monkeypatch.setattr(
        "sglang.srt.configs.model_config.is_deepseek_dsa", lambda _config: True
    )
    with pytest.raises(RuntimeError, match="first MLA profile"):
        validation._validate_checkpoint_geometry(configurator)

    monkeypatch.setattr(
        "sglang.srt.configs.model_config.is_deepseek_dsa", lambda _config: False
    )
    install_mla(drift_components(config))
    with pytest.raises(RuntimeError, match="latent/RoPE geometry"):
        validation._validate_checkpoint_geometry(configurator)


def test_mla_backend_contract_requires_uniform_supported_backend():
    install_mla(latent_config())
    configurator = mla_configurator()
    assert (
        validation._validate_attention_backend_contract(configurator)
        == "DeepseekV2ForCausalLM"
    )
    configurator.server_args.get_attention_backends = lambda: (
        "flashinfer",
        "trtllm_mla",
    )
    with pytest.raises(RuntimeError, match="uniform MLA backend"):
        validation._validate_attention_backend_contract(configurator)
