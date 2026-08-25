from __future__ import annotations

from typing import Any, Sequence

from .runtime_policy import (
    GDN_FIXED_STATE_BACKEND_PROFILE as _RUNTIME_GDN_FIXED_STATE_BACKEND_PROFILE,
)


ATTENTION_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": "fa3",
    "GptOssForCausalLM": "fa3",
    "DeepseekV2ForCausalLM": "flashinfer",
    "Qwen3_5ForConditionalGeneration": "fa3",
}
MOE_RUNNER_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": None,
    "GptOssForCausalLM": "triton",
    "DeepseekV2ForCausalLM": None,
    "Qwen3_5ForConditionalGeneration": None,
}
SUPPORTED_ARCHITECTURES = tuple(ATTENTION_BACKENDS_BY_ARCHITECTURE)
PAGE_TOKENS = 16
QWEN_HYBRID_GDN_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
# Historical qualification code imports this name. Keep it as a value alias,
# while current code names the config family rather than one model release.
QWEN35_ARCHITECTURE = QWEN_HYBRID_GDN_ARCHITECTURE
GDN_FIXED_STATE_BACKEND_PROFILE = dict(
    _RUNTIME_GDN_FIXED_STATE_BACKEND_PROFILE
)
# Historical qualification evidence imports and compares this plain dict. Keep
# it independent from both the current qualification and immutable runtime views.
QWEN35_BACKEND_PROFILE = dict(_RUNTIME_GDN_FIXED_STATE_BACKEND_PROFILE)


def _positive_checkpoint_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"checkpoint has invalid {name}")
    return value


def _qwen_hybrid_gdn_geometry(text: dict[str, Any]) -> dict[str, Any]:
    if text.get("dtype") != "bfloat16":
        raise RuntimeError(
            "Qwen qwen3_5 dense config family requires "
            "text_config.dtype=bfloat16"
        )
    if text.get("mamba_ssm_dtype") != "float32":
        raise RuntimeError(
            "Qwen qwen3_5 dense config family requires "
            "text_config.mamba_ssm_dtype=float32"
        )
    layers = _positive_checkpoint_int(text, "num_hidden_layers")
    interval = _positive_checkpoint_int(text, "full_attention_interval")
    raw = text.get("layer_types")
    if not isinstance(raw, list) or len(raw) != layers:
        raise RuntimeError(
            "Qwen qwen3_5 dense config family requires one "
            "text_config.layer_type per layer"
        )
    if any(value not in {"full_attention", "linear_attention"} for value in raw):
        raise RuntimeError(
            "Qwen qwen3_5 dense config family has an unsupported layer_type"
        )
    expected = [
        "full_attention" if (index + 1) % interval == 0
        else "linear_attention"
        for index in range(layers)
    ]
    if raw != expected:
        raise RuntimeError(
            "Qwen qwen3_5 dense config family text_config.layer_types differs from "
            "full_attention_interval"
        )
    full = [index for index, value in enumerate(raw) if value == "full_attention"]
    linear = [index for index, value in enumerate(raw) if value == "linear_attention"]
    if not full or not linear:
        raise RuntimeError(
            "Qwen qwen3_5 dense config family requires both Full and GDN layers"
        )
    values = {
        name: _positive_checkpoint_int(text, name)
        for name in (
            "num_key_value_heads", "head_dim", "linear_num_key_heads",
            "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim",
        )
    }
    if values["linear_conv_kernel_dim"] < 2:
        raise RuntimeError(
            "Qwen qwen3_5 dense config family linear_conv_kernel_dim "
            "must be at least two"
        )
    key_bytes = values["num_key_value_heads"] * values["head_dim"] * 2
    recurrent_bytes = (
        values["linear_num_value_heads"]
        * values["linear_key_head_dim"]
        * values["linear_value_head_dim"]
        * 4
    )
    convolution_channels = (
        2 * values["linear_num_key_heads"] * values["linear_key_head_dim"]
        + values["linear_num_value_heads"] * values["linear_value_head_dim"]
    )
    convolution_bytes = (
        convolution_channels * (values["linear_conv_kernel_dim"] - 1) * 2
    )
    return {
        "layers": layers, "full_layers": full, "linear_layers": linear,
        "key_bytes": key_bytes, "recurrent_bytes": recurrent_bytes,
        "convolution_bytes": convolution_bytes,
        "kernel_width": values["linear_conv_kernel_dim"],
    }


def _qwen_hybrid_gdn_control_token_ids(
    config: dict[str, Any], vocab_size: int
) -> dict[str, int]:
    names = (
        "image_token_id", "video_token_id",
        "vision_start_token_id", "vision_end_token_id",
    )
    values = {name: _positive_checkpoint_int(config, name) for name in names}
    if len(set(values.values())) != len(values) or any(
        value >= vocab_size for value in values.values()
    ):
        raise RuntimeError(
            "Qwen qwen3_5 dense config family requires distinct in-vocabulary "
            "multimodal control token ids"
        )
    return values


def _validate_plan_attention(
    manager_config: Any, expected_classes: Sequence[dict[str, Any]], layers: int,
    expected_fixed_states: Sequence[dict[str, Any]] = (),
) -> None:
    if manager_config.page_tokens != PAGE_TOKENS:
        raise RuntimeError("manager plan does not use page_tokens=16")
    if manager_config.num_hidden_layers != layers:
        raise RuntimeError("manager plan layer count differs from the checkpoint")
    if len(manager_config.classes) != len(expected_classes):
        raise RuntimeError("manager plan attention classes differ from the checkpoint")
    for actual, expected in zip(
        manager_config.classes, expected_classes, strict=True
    ):
        fields: dict[str, Any] = {
            "name": actual.name,
            "retention": actual.retention,
            "layers": list(actual.layers),
            "window_tokens": actual.window_tokens,
        }
        if "storage" in expected:
            fields.update(
                storage=actual.storage,
                bytes_per_token_per_layer=actual.bytes_per_token_per_layer,
                components=dict(actual.components),
            )
        if fields != expected:
            raise RuntimeError(
                "manager plan attention class differs from the checkpoint: "
                f"expected={expected} actual={fields}"
            )
    fixed = [
        {
            "name": item.name, "kind": item.kind, "layers": list(item.layers),
            "state_bytes_per_layer": item.state_bytes_per_layer,
            "checkpoint_slots_per_request": item.checkpoint_slots_per_request,
            "kernel_width": item.kernel_width,
        }
        for item in getattr(manager_config, "fixed_states", ())
    ]
    if fixed != list(expected_fixed_states):
        raise RuntimeError(
            "manager plan fixed-state components differ from the checkpoint: "
            f"expected={list(expected_fixed_states)} actual={fixed}"
        )


def checkpoint_attention_contract(
    config: dict[str, Any], manager_config: Any | None = None
) -> dict[str, Any]:
    architectures = config.get("architectures")
    if not any(architectures == [item] for item in SUPPORTED_ARCHITECTURES):
        raise RuntimeError(
            "qualification supports only Qwen2, GPT-OSS, DeepSeek-V2, or Qwen3.5"
        )
    architecture = architectures[0]
    text = config
    if architecture == QWEN_HYBRID_GDN_ARCHITECTURE:
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise RuntimeError(
                "Qwen qwen3_5 dense config family requires nested text_config"
            )
        if config.get("model_type") != "qwen3_5":
            raise RuntimeError(
                "Qwen qwen3_5 dense config family requires model_type=qwen3_5"
            )
        if text.get("model_type") != "qwen3_5_text":
            raise RuntimeError(
                "Qwen qwen3_5 dense config family requires "
                "text_config.model_type=qwen3_5_text"
            )
    layers = _positive_checkpoint_int(text, "num_hidden_layers")
    vocab_size = _positive_checkpoint_int(text, "vocab_size")
    max_positions = _positive_checkpoint_int(text, "max_position_embeddings")
    fixed_states: list[dict[str, Any]] = []
    backend_profile: dict[str, Any] = {
        "attention_backend": ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture]
    }
    control_token_ids: dict[str, int] = {}
    prompt_token_upper_bound = vocab_size
    workload_profile = "prefix_reuse"
    state_ownership = "token_prefix_shareable"

    if architecture == "Qwen2ForCausalLM":
        if config.get("use_sliding_window") is not False:
            raise RuntimeError("Qwen2 qualification requires use_sliding_window=false")
        if "layer_types" in config:
            raise RuntimeError("Qwen2 Full qualification forbids layer_types")
        classes = [_class("full", "full", range(layers), None)]
        profile = "full"
        sliding_window = None
    elif architecture == "GptOssForCausalLM":
        raw = config.get("layer_types")
        if not isinstance(raw, list) or len(raw) != layers:
            raise RuntimeError(
                "GptOss qualification requires one explicit layer_type per layer"
            )
        if any(value not in {"full_attention", "sliding_attention"} for value in raw):
            raise RuntimeError("GptOss checkpoint has an unsupported layer_type")
        full = [index for index, value in enumerate(raw) if value == "full_attention"]
        sliding = [
            index for index, value in enumerate(raw) if value == "sliding_attention"
        ]
        if not full or not sliding:
            raise RuntimeError("GptOss qualification requires both Full and SWA layers")
        sliding_window = _positive_checkpoint_int(config, "sliding_window")
        classes = [
            _class("full", "full", full, None),
            _class("swa", "sliding", sliding, sliding_window),
        ]
        profile = "hybrid_full_swa"
    elif architecture == "DeepseekV2ForCausalLM":
        latent = _positive_checkpoint_int(config, "kv_lora_rank")
        rope = _positive_checkpoint_int(config, "qk_rope_head_dim")
        if config.get("index_topk") is not None:
            raise RuntimeError("DeepSeek-V2 MLA qualification forbids DSA indexing")
        classes = [_class("latent_mla", "full", range(layers), None)]
        if manager_config is not None:
            actual = manager_config.classes[0]
            if actual.storage != "latent_kv" or actual.components_by_name != {
                "latent": latent * 2,
                "rope": rope * 2,
            }:
                raise RuntimeError(
                    "manager plan MLA geometry differs from the checkpoint"
                )
        profile = "mla"
        sliding_window = None
    else:
        geometry = _qwen_hybrid_gdn_geometry(text)
        control_token_ids = _qwen_hybrid_gdn_control_token_ids(config, vocab_size)
        prompt_token_upper_bound = _positive_checkpoint_int(text, "eos_token_id")
        if prompt_token_upper_bound <= 3 or any(
            value < prompt_token_upper_bound
            for value in control_token_ids.values()
        ):
            raise RuntimeError(
                "Qwen qwen3_5 dense config family requires control tokens at or above "
                "the EOS prompt boundary"
            )
        key_bytes = geometry["key_bytes"]
        classes = [{
            **_class(
                "full_attention_kv", "full", geometry["full_layers"], None
            ),
            "storage": "token_kv",
            "bytes_per_token_per_layer": key_bytes * 2,
            "components": {"key": key_bytes, "value": key_bytes},
        }]
        fixed_states = [
            {
                "name": "gdn_recurrent", "kind": "gdn",
                "layers": geometry["linear_layers"],
                "state_bytes_per_layer": geometry["recurrent_bytes"],
                "checkpoint_slots_per_request": 2, "kernel_width": None,
            },
            {
                "name": "gdn_convolution", "kind": "convolution",
                "layers": geometry["linear_layers"],
                "state_bytes_per_layer": geometry["convolution_bytes"],
                "checkpoint_slots_per_request": 2,
                "kernel_width": geometry["kernel_width"],
            },
        ]
        profile = "hybrid_full_gdn"
        sliding_window = None
        backend_profile = dict(_RUNTIME_GDN_FIXED_STATE_BACKEND_PROFILE)
        workload_profile = "fresh_prompt"
        state_ownership = "request_private"

    if manager_config is not None:
        _validate_plan_attention(manager_config, classes, layers, fixed_states)
    return {
        "architecture": architecture,
        "attention_profile": profile,
        "attention_backend": ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture],
        "backend_profile": backend_profile,
        "workload_profile": workload_profile,
        "state_ownership": state_ownership,
        "num_hidden_layers": layers,
        "vocab_size": vocab_size,
        "prompt_token_upper_bound": prompt_token_upper_bound,
        "control_token_ids": control_token_ids,
        "max_position_embeddings": max_positions,
        "sliding_window": sliding_window,
        "classes": classes,
        "fixed_states": fixed_states,
    }


def _class(
    name: str, retention: str, layers: Any, window_tokens: int | None
) -> dict[str, Any]:
    return {
        "name": name,
        "retention": retention,
        "layers": list(layers),
        "window_tokens": window_tokens,
    }


__all__ = [
    "ATTENTION_BACKENDS_BY_ARCHITECTURE",
    "GDN_FIXED_STATE_BACKEND_PROFILE",
    "MOE_RUNNER_BACKENDS_BY_ARCHITECTURE",
    "PAGE_TOKENS",
    "QWEN_HYBRID_GDN_ARCHITECTURE",
    "QWEN35_ARCHITECTURE",
    "QWEN35_BACKEND_PROFILE",
    "SUPPORTED_ARCHITECTURES",
    "checkpoint_attention_contract",
]
