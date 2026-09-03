from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import orbitkv_sglang.bridge.facade as facade
import orbitkv_sglang.bridge.state as state
import orbitkv_sglang.bridge.validation as validation
import orbitkv_sglang.runtime_admission as runtime_admission
from mla_helpers import drift_components, install_mla, latent_config, mla_configurator
from orbitkv_sglang.runtime import ArenaRegistration, CacheSharingPolicy


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


def test_mla_physical_pool_geometry_and_copy_are_validated_independently():
    config = latent_config()
    pool = _pool()
    install_mla(config)
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


def test_mla_builder_uses_request_private_native_session_profile(
    monkeypatch,
):
    config = replace(
        latent_config(),
        runtime_manifest_path=Path("runtime-manifest.json"),
        runtime_manifest_fingerprint="sha256:manifest",
        runtime_binding=MappingProxyType(
            {"execution_topology": "whole_domain_full_latent_kv"}
        ),
        manager_plan_format="kv_plan",
    )
    install_mla(config)
    monkeypatch.setattr(
        runtime_admission,
        "admit_runtime_config",
        lambda candidate: "whole_domain_full_latent_kv",
    )

    class Allocator:
        def __init__(
            self,
            size,
            page_size,
            dtype,
            device,
            kvcache,
            need_sort,
            *,
            class_id,
        ):
            self.size = size
            self.page_size = page_size
            self.dtype = dtype
            self.device = device
            self.kvcache = kvcache
            self.need_sort = need_sort
            self.class_id = class_id

    registrations = []
    runtime = object()

    def new_runtime(values):
        registrations.extend(values)
        state._RUNTIME = runtime
        return runtime

    monkeypatch.setattr(facade, "_facade_types", lambda: (Allocator, object, object))
    monkeypatch.setattr(facade, "_new_runtime", new_runtime)
    monkeypatch.setattr(
        facade, "_mirror_cleanup_coordinator", lambda *_args: object()
    )
    pool = _pool()

    allocator = facade._build_token_to_kv_pool_allocator(
        mla_configurator(),
        sizes=SimpleNamespace(max_total_num_tokens=128),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=None,
    )

    assert allocator is state._ALLOCATOR
    assert allocator.size == 128
    assert allocator.page_size == 16
    assert allocator.kvcache is pool
    assert allocator.class_id == 0
    assert registrations == [ArenaRegistration(0, 1, 1, 8)]
    assert state._uses_runtime_session()
    assert state._cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE
    assert state._requires_disabled_radix_cache()


def test_mla_builder_storage_gate_rejects_token_kv_backend_after_mocked_admission(
    monkeypatch,
):
    config = latent_config()
    install_mla(config)
    monkeypatch.setattr(state, "_admit_product_runtime_profile", lambda: object())
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
    assert state._RUNTIME is None
    assert state._ALLOCATOR is None


def test_mla_geometry_gate_rejects_dsa_and_component_drift(monkeypatch):
    config = latent_config(layers=(0, 1), latent_bytes=1024, rope_bytes=128)
    configurator = mla_configurator(kv_lora_rank=512, qk_rope_head_dim=64)
    configurator.model_config.hf_text_config.num_hidden_layers = 2
    install_mla(config)
    validation._validate_checkpoint_geometry(configurator)

    configurator.model_config.hf_config.index_topk = 2048
    with pytest.raises(RuntimeError, match="supported MLA profile"):
        validation._validate_checkpoint_geometry(configurator)

    configurator.model_config.hf_config.index_topk = None
    install_mla(drift_components(config))
    with pytest.raises(RuntimeError, match="latent/RoPE geometry"):
        validation._validate_checkpoint_geometry(configurator)


def test_mla_backend_contract_accepts_capability_backend():
    install_mla(latent_config())
    configurator = mla_configurator()
    assert validation._validate_attention_backend_contract(configurator) is None


@pytest.mark.parametrize(
    "backends", (("flashinfer", "trtllm_mla"), ("triton", "triton"))
)
def test_mla_backend_contract_rejects_nonuniform_or_unsupported_backend(backends):
    install_mla(latent_config())
    configurator = mla_configurator()
    configurator.server_args.get_attention_backends = lambda: backends
    with pytest.raises(RuntimeError, match="latent-KV capability requires"):
        validation._validate_attention_backend_contract(configurator)

    configurator.server_args.get_attention_backends = lambda: ("fa3", "fa3")
    with pytest.raises(RuntimeError, match="latent-KV capability requires SGLang FlashInfer"):
        validation._validate_attention_backend_contract(configurator)
