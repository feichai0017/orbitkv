from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orbitkv_sglang.config import ClassConfig, RuntimeConfig
from orbitkv_sglang.bridge import state, validation


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:pure-sliding-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                name="local_attention",
                layers=(0,),
                retention="sliding",
                bytes_per_token_per_layer=128,
                window_tokens=32,
                period_blocks=3,
            ),
        ),
    )


def _model(
    *,
    full_layers: list[int] | None = None,
    swa_layers: list[int] | None = None,
    window: int = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LocalAttentionForCausalLM"]),
        hf_text_config=SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=2,
        ),
        head_dim=16,
        v_head_dim=16,
        swa_head_dim=16,
        swa_v_head_dim=16,
        is_hybrid_swa=True,
        full_attention_layer_ids=[] if full_layers is None else full_layers,
        swa_attention_layer_ids=[0] if swa_layers is None else swa_layers,
        sliding_window_size=window,
        disable_hybrid_swa_memory=False,
        is_deepseek_v4_arch=False,
        is_hybrid_swa_compress=False,
        attention_chunk_size=None,
        has_attention_sinks=False,
    )


def _configurator(model: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model,
        kv_cache_dtype=torch.bfloat16,
        use_mla_backend=False,
    )


def test_pure_sliding_geometry_gate_accepts_all_swa_layers() -> None:
    state._install_test_state(config=_config())

    validation._validate_checkpoint_geometry(_configurator(_model()))


def test_pinned_sglang_generic_hybrid_pattern_can_describe_all_swa() -> None:
    from transformers import PretrainedConfig
    from sglang.srt.configs.model_config import (
        get_hybrid_layer_ids,
        is_hybrid_swa_model,
    )

    text = PretrainedConfig(
        num_hidden_layers=3,
        is_hybrid_swa=True,
        hybrid_layer_pattern=[1, 1, 1],
    )
    architecture = ["ExternalLocalAttentionForCausalLM"]

    assert is_hybrid_swa_model(architecture, text)
    sliding, full = get_hybrid_layer_ids(architecture, text)
    assert sliding == [0, 1, 2]
    assert full == []


@pytest.mark.parametrize(
    ("full_layers", "swa_layers", "window", "message"),
    (
        ([0], [], 32, "every model layer"),
        ([], [0], 31, "sliding window differs"),
    ),
)
def test_pure_sliding_geometry_gate_rejects_runtime_drift(
    full_layers: list[int],
    swa_layers: list[int],
    window: int,
    message: str,
) -> None:
    state._install_test_state(config=_config())

    with pytest.raises(RuntimeError, match=message):
        validation._validate_checkpoint_geometry(
            _configurator(
                _model(
                    full_layers=full_layers,
                    swa_layers=swa_layers,
                    window=window,
                )
            )
        )
