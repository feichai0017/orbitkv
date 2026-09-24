"""Unit tests for namespace derivation factors.

The namespace isolates storage by KV layout: any config that changes the
on-storage block layout must change the namespace, or two incompatible layouts
collide under one namespace and loads fail the server-side slot-count guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.vllm.config import derive_namespace  # noqa: E402


@pytest.fixture(autouse=True)
def _engine_version(monkeypatch):
    monkeypatch.setattr("orbitkv.vllm.config.version", lambda _: "0.29.0")


def _make_vllm_config(
    *,
    pp_size: int = 1,
    mla_layer_split: bool = False,
    disable_hma: bool = False,
) -> SimpleNamespace:
    model_config = SimpleNamespace(
        model="/data/models/GLM-5.2-FP8",
        dtype="bfloat16",
        revision="a" * 40,
        tokenizer=None,
        tokenizer_revision=None,
        quantization=None,
        hf_config=SimpleNamespace(to_json_string=lambda: '{"hidden_size":576}'),
        get_total_num_kv_heads=lambda: 1,
        get_head_size=lambda: 576,
        get_total_num_hidden_layers=lambda: 78,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            cache_dtype="fp8",
            prefix_caching_hash_algo="sha256",
            kv_cache_layout="LBNHC",
            compute_hash=lambda: "cache-config-v1",
        ),
        lora_config=None,
        attention_config=SimpleNamespace(compute_hash=lambda: "attention-v1"),
        kernel_config=SimpleNamespace(compute_hash=lambda: "kernel-v1"),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hma,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        additional_config={"mla_layer_split_kv_cache": mla_layer_split},
    )


def _ns(**kwargs) -> str:
    return derive_namespace(_make_vllm_config(**kwargs), tp_size=8)


def test_pp_size_isolates_namespace():
    # Same model, different pipeline-parallel degree -> different layer split
    # per server -> must not share storage.
    assert _ns(pp_size=1) != _ns(pp_size=8)


def test_mla_layer_split_isolates_namespace():
    # Layer-split registration shards each block's slots differently from the
    # default full-slot layout (the haitao GLM-5.2 99-vs-156 collision).
    assert _ns(mla_layer_split=False) != _ns(mla_layer_split=True)


def test_hma_enablement_isolates_namespace():
    assert _ns(disable_hma=False) != _ns(disable_hma=True)


def test_namespace_is_stable_for_same_config():
    assert _ns(pp_size=4, mla_layer_split=True) == _ns(pp_size=4, mla_layer_split=True)


def test_resolved_byte_layout_and_cache_config_isolate_identical_geometry():
    cfg = _make_vllm_config()
    original = derive_namespace(cfg, tp_size=8)
    cfg.cache_config.kv_cache_layout = "LBHNC"
    assert derive_namespace(cfg, tp_size=8) != original
    cfg.cache_config.kv_cache_layout = "LBNHC"
    cfg.cache_config.compute_hash = lambda: "different-quantization-or-state-format"
    assert derive_namespace(cfg, tp_size=8) != original


def test_missing_additional_config_defaults_to_no_split():
    cfg = _make_vllm_config(pp_size=4)
    cfg.additional_config = None
    assert derive_namespace(cfg, tp_size=8) == _ns(pp_size=4, mla_layer_split=False)


def test_engine_kv_precision_is_independent_of_weight_quantization():
    cfg = _make_vllm_config()
    original = derive_namespace(cfg, tp_size=1)
    cfg.cache_config.cache_dtype = "auto"
    assert derive_namespace(cfg, tp_size=1) != original
