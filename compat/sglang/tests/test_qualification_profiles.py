from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(INTEGRATION_ROOT / "bridge/src"))

from orbitkv_sglang.qualification import (  # noqa: E402
    HYBRID_FIXED_STATE_BACKEND_PROFILE,
    checkpoint_attention_contract,
)


def _load_fixture(name: str) -> dict:
    return json.loads(
        (REPOSITORY_ROOT / "core/fixtures" / name / "config.json").read_text(
            encoding="utf-8"
        )
    )


def _manager_config(contract: dict) -> SimpleNamespace:
    classes = tuple(
        SimpleNamespace(
            name=item["name"],
            retention=item["retention"],
            layers=tuple(item["layers"]),
            window_tokens=item["window_tokens"],
            storage=item.get("storage"),
            bytes_per_token_per_layer=item.get("bytes_per_token_per_layer"),
            components=tuple(item.get("components", {}).items()),
        )
        for item in contract["classes"]
    )
    fixed_states = tuple(
        SimpleNamespace(
            name=item["name"],
            kind=item["kind"],
            layers=tuple(item["layers"]),
            state_bytes_per_layer=item["state_bytes_per_layer"],
            checkpoint_slots_per_request=item["checkpoint_slots_per_request"],
            kernel_width=item["kernel_width"],
        )
        for item in contract["fixed_states"]
    )
    return SimpleNamespace(
        page_tokens=16,
        num_hidden_layers=contract["num_hidden_layers"],
        classes=classes,
        fixed_states=fixed_states,
    )


@pytest.mark.parametrize(
    ("fixture_name", "layers", "full_layers", "recurrent_bytes"),
    (
        ("hybrid-fixed-state-small", 24, 6, 1_048_576),
        ("hybrid-fixed-state-large", 64, 16, 3_145_728),
    ),
)
def test_hybrid_fixed_state_fixtures_compile_capability_contracts(
    fixture_name, layers, full_layers, recurrent_bytes
):
    config = _load_fixture(fixture_name)
    contract = checkpoint_attention_contract(config)

    assert contract["attention_profile"] == "hybrid_fixed_state"
    assert contract["backend_profile"] == HYBRID_FIXED_STATE_BACKEND_PROFILE
    assert contract["workload_profile"] == "fresh_prompt"
    assert contract["state_ownership"] == "request_private"
    assert contract["num_hidden_layers"] == layers
    assert len(contract["classes"][0]["layers"]) == full_layers
    assert contract["fixed_states"][0]["state_bytes_per_layer"] == recurrent_bytes

    checkpoint_attention_contract(config, _manager_config(contract))


def test_hybrid_linear_attention_admits_renamed_model_identifiers():
    config = _load_fixture("hybrid-fixed-state-small")
    expected = checkpoint_attention_contract(config)
    config["architectures"] = ["RenamedArchitecture"]
    config["model_type"] = "renamed_envelope"
    config["text_config"]["model_type"] = "renamed_text"

    assert checkpoint_attention_contract(config) == expected


def test_hybrid_linear_attention_admits_missing_identity_fields():
    config = _load_fixture("hybrid-fixed-state-small")
    expected = checkpoint_attention_contract(config)
    del config["architectures"]
    del config["model_type"]
    del config["text_config"]["model_type"]

    assert checkpoint_attention_contract(config) == expected


@pytest.mark.parametrize(
    "marker",
    (
        "full_attention_interval",
        "mamba_ssm_dtype",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ),
)
def test_partial_fixed_state_markers_fail_closed(marker):
    config = {
        "architectures": ["RenamedArchitecture"],
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "layer_types": ["full_attention"],
        "text_config": {marker: 4},
    }

    with pytest.raises(RuntimeError, match="invalid num_hidden_layers"):
        checkpoint_attention_contract(config)


def test_structural_token_and_latent_profiles_ignore_architecture_name():
    base = {
        "architectures": ["RenamedArchitecture"],
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
    }
    full = checkpoint_attention_contract(
        {**base, "use_sliding_window": False}
    )
    sliding = checkpoint_attention_contract(
        {**base, "use_sliding_window": True, "sliding_window": 128}
    )
    hybrid = checkpoint_attention_contract(
        {
            **base,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 128,
        }
    )
    latent = checkpoint_attention_contract(
        {**base, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    )

    assert full["attention_profile"] == "full"
    assert sliding["attention_profile"] == "sliding"
    assert hybrid["attention_profile"] == "hybrid_full_swa"
    assert latent["attention_profile"] == "mla"
    assert full["attention_backend"] == "fa3"
    assert sliding["attention_backend"] == "fa3"
    assert hybrid["attention_backend"] == "fa3"
    assert latent["attention_backend"] == "flashinfer"


@pytest.mark.parametrize(
    "config",
    (
        {"kv_lora_rank": 512},
        {"qk_rope_head_dim": 64},
        {"kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_topk": 8},
    ),
)
def test_partial_or_sparse_latent_markers_fail_closed(config):
    config = {
        "architectures": ["RenamedArchitecture"],
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        **config,
    }

    with pytest.raises(RuntimeError):
        checkpoint_attention_contract(config)


def test_conflicting_structural_markers_fail_closed():
    base = {
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    cases = (
        {**base, "text_config": {"full_attention_interval": 4}},
        {**base, "layer_types": ["full_attention", "full_attention"]},
        {**base, "use_sliding_window": True, "sliding_window": 128},
        {**base, "sliding_window": 128},
    )
    for config in cases:
        with pytest.raises(RuntimeError, match="ambiguous"):
            checkpoint_attention_contract(config)


def test_explicit_layer_types_are_authoritative_over_uniform_flag():
    config = {
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "layer_types": ["sliding_attention", "full_attention"],
        "use_sliding_window": False,
        "sliding_window": 128,
    }

    contract = checkpoint_attention_contract(config)

    assert contract["attention_profile"] == "hybrid_full_swa"
    assert contract["classes"][0]["layers"] == [1]
    assert contract["classes"][1]["layers"] == [0]


def test_missing_structural_discriminator_fails_closed():
    config = {
        "architectures": ["RenamedArchitecture"],
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "sliding_window": 128,
    }

    with pytest.raises(RuntimeError, match="does not explicitly prove"):
        checkpoint_attention_contract(config)
