"""Capability tests for vLLM cache-group layouts supported by OrbitKV."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from orbitkv.vllm.layout import CacheGroupLayout  # noqa: E402


def _group(name, spec):
    return SimpleNamespace(layer_names=(name,), kv_cache_spec=spec)


def _config(*groups):
    return SimpleNamespace(kv_cache_groups=groups)


def _full_attention(block_size=16):
    spec = FullAttentionSpec()
    spec.block_size = block_size
    return spec


def _mamba(block_size=16, mode="align"):
    spec = MambaSpec()
    spec.block_size = block_size
    spec.mamba_cache_mode = mode
    return spec


def _mla(block_size=16, head_size=128):
    spec = MLAAttentionSpec()
    spec.block_size = block_size
    spec.head_size = head_size
    return spec


def _window(block_size=16, window=33):
    spec = SlidingWindowSpec()
    spec.block_size = block_size
    spec.sliding_window = window
    return spec


class SpecializedFullAttentionSpec(FullAttentionSpec):
    pass


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, SpecializedFullAttentionSpec])
def test_accepts_full_attention_with_aligned_mamba(spec_type):
    attention = spec_type()
    attention.block_size = 528
    config = _config(
        _group("attention", attention),
        _group("recurrent", _mamba(block_size=528)),
    )

    layout = CacheGroupLayout.from_config(config)

    assert layout.layer_names == (("attention",), ("recurrent",))
    assert layout.hash_group_index == 0
    assert layout.has_recurrent_state
    assert layout.recurrent_group_indices == frozenset({1})
    assert layout.recurrent_layer_names == frozenset({"recurrent"})
    assert layout.recovery_groups == ((0, "attention", 0), (1, "recurrent", 0))


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, MLAAttentionSpec])
def test_accepts_single_attention_group(spec_type):
    attention = spec_type()
    attention.block_size = 16

    layout = CacheGroupLayout.from_config(_config(_group("attention", attention)))

    assert layout.hash_group_index == 0
    assert not layout.has_recurrent_state


def test_accepts_single_uniform_mla_group():
    group = SimpleNamespace(
        layer_names=("model.layers.0.self_attn.attn", "model.layers.0.self_attn.indexer.k_cache"),
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={
                "model.layers.0.self_attn.attn": _mla(head_size=576),
                "model.layers.0.self_attn.indexer.k_cache": _mla(head_size=128),
            },
        ),
    )

    layout = CacheGroupLayout.from_config(_config(group))

    assert layout.layer_names == (group.layer_names,)
    assert layout.hash_group_index == 0
    assert not layout.has_recurrent_state


@pytest.mark.parametrize("other_spec_type", [FullAttentionSpec, SlidingWindowSpec])
def test_rejects_uniform_group_with_non_mla_layer(other_spec_type):
    other_spec = other_spec_type()
    other_spec.block_size = 16
    spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"attention": _mla(), "other": other_spec},
    )

    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("attention", spec)))


def test_rejects_empty_uniform_mla_group():
    spec = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={})

    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("attention", spec)))


@pytest.mark.parametrize("mode", ["align", "all"])
def test_rejects_single_mamba_group(mode):
    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("recurrent", _mamba(mode=mode))))


def test_rejects_single_sliding_window_group():
    sliding_window = SlidingWindowSpec()
    sliding_window.block_size = 16

    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("sliding_window", sliding_window)))


def test_rejects_misaligned_logical_block_sizes():
    config = _config(
        _group("attention", _full_attention(block_size=16)),
        _group("recurrent", _mamba(block_size=32)),
    )

    with pytest.raises(RuntimeError, match="identical logical block sizes"):
        CacheGroupLayout.from_config(config)


def test_rejects_multiple_full_attention_groups_without_mamba():
    config = _config(
        _group("first", _full_attention()),
        _group("second", _full_attention()),
    )

    with pytest.raises(RuntimeError, match="require window or recurrent"):
        CacheGroupLayout.from_config(config)


def test_rejects_mamba_groups_without_full_attention():
    config = _config(
        _group("recurrent.0", _mamba(block_size=528)),
        _group("recurrent.1", _mamba(block_size=528)),
    )

    with pytest.raises(RuntimeError, match="dense FullAttention"):
        CacheGroupLayout.from_config(config)


@pytest.mark.parametrize("recurrent", [False, True])
def test_window_and_checkpoint_have_independent_storage_and_recovery_rules(recurrent):
    config = _config(
        _group("sliding_window", _window()),
        _group("attention", _full_attention()),
        *([_group("recurrent", _mamba())] if recurrent else []),
    )

    layout = CacheGroupLayout.from_config(config)
    assert layout.hash_group_index == 1
    assert layout.storage_group_ids == ((1, 0, 2) if recurrent else (1, 0))
    assert layout.window_group_indices == frozenset({0})
    assert layout.recovery_groups == (
        (0, "attention", 0),
        (1, "window", 32),
        *(([(2, "recurrent", 0)]) if recurrent else []),
    )


def test_accepts_mla_with_mamba():
    mla = MLAAttentionSpec()
    mla.block_size = 16
    config = _config(
        _group("attention", mla),
        _group("recurrent", _mamba()),
    )

    layout = CacheGroupLayout.from_config(config)

    assert layout.hash_group_index == 0
    assert layout.has_recurrent_state


def test_rejects_non_align_mamba_mode():
    config = _config(
        _group("attention", _full_attention()),
        _group("recurrent", _mamba(mode="all")),
    )

    with pytest.raises(RuntimeError, match="mamba_cache_mode='align'"):
        CacheGroupLayout.from_config(config)


class TestStorageGroupIds:
    def test_attention_first_layout(self):
        config = _config(
            _group("attn", _full_attention()),
            _group("mamba", _mamba()),
        )
        layout = CacheGroupLayout.from_config(config)
        assert layout.storage_group_ids == (0, 1)

    def test_recurrent_group_can_come_first(self):
        # vLLM group order is not guaranteed; attention must always map to
        # storage group 0 regardless of connector position.
        config = _config(
            _group("mamba", _mamba()),
            _group("attn", _full_attention()),
        )
        layout = CacheGroupLayout.from_config(config)
        assert layout.hash_group_index == 1
        assert layout.storage_group_ids == (1, 0)
        assert layout.recovery_groups == ((0, "attention", 0), (1, "recurrent", 0))

    def test_single_group_defaults_to_zero(self):
        config = _config(_group("attn", _full_attention()))
        layout = CacheGroupLayout.from_config(config)
        assert layout.storage_group_ids == (0,)

    def test_multiple_recurrent_groups_get_dense_ids(self):
        config = _config(
            _group("attn", _full_attention()),
            _group("mamba_a", _mamba()),
            _group("mamba_b", _mamba()),
        )
        layout = CacheGroupLayout.from_config(config)
        assert layout.storage_group_ids == (0, 1, 2)
        assert layout.recurrent_group_indices == frozenset({1, 2})
        assert layout.recovery_groups == (
            (0, "attention", 0),
            (1, "recurrent", 0),
            (2, "recurrent", 0),
        )
