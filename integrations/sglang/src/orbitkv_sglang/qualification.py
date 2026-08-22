from __future__ import annotations

from typing import Any, Sequence


ATTENTION_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": "flashinfer",
    "GptOssForCausalLM": "fa3",
    "DeepseekV2ForCausalLM": "flashinfer",
}
MOE_RUNNER_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": None,
    "GptOssForCausalLM": "triton",
    "DeepseekV2ForCausalLM": None,
}
SUPPORTED_ARCHITECTURES = tuple(ATTENTION_BACKENDS_BY_ARCHITECTURE)
PAGE_TOKENS = 16


def _positive_checkpoint_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"checkpoint has invalid {name}")
    return value


def _validate_plan_attention(
    manager_config: Any, expected_classes: Sequence[dict[str, Any]], layers: int
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
        fields = {
            "name": actual.name,
            "retention": actual.retention,
            "layers": list(actual.layers),
            "window_tokens": actual.window_tokens,
        }
        if fields != expected:
            raise RuntimeError(
                "manager plan attention class differs from the checkpoint: "
                f"expected={expected} actual={fields}"
            )


def checkpoint_attention_contract(
    config: dict[str, Any], manager_config: Any | None = None
) -> dict[str, Any]:
    architectures = config.get("architectures")
    if not any(architectures == [item] for item in SUPPORTED_ARCHITECTURES):
        raise RuntimeError(
            "qualification supports only Qwen2, GPT-OSS, or DeepSeek-V2"
        )
    architecture = architectures[0]
    layers = _positive_checkpoint_int(config, "num_hidden_layers")
    vocab_size = _positive_checkpoint_int(config, "vocab_size")
    max_positions = _positive_checkpoint_int(config, "max_position_embeddings")

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
    else:
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

    if manager_config is not None:
        _validate_plan_attention(manager_config, classes, layers)
    return {
        "architecture": architecture,
        "attention_profile": profile,
        "attention_backend": ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture],
        "num_hidden_layers": layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_positions,
        "sliding_window": sliding_window,
        "classes": classes,
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
    "MOE_RUNNER_BACKENDS_BY_ARCHITECTURE",
    "PAGE_TOKENS",
    "SUPPORTED_ARCHITECTURES",
    "checkpoint_attention_contract",
]
