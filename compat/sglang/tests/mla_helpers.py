from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

import orbitkv_sglang.bridge.state as state
from orbitkv_sglang.config import ClassConfig, RuntimeConfig, load_config
from manifest_test_support import runtime_manifest_environment


def latent_class(
    *,
    layers: tuple[int, ...] = (0,),
    latent_bytes: int = 16,
    rope_bytes: int = 8,
) -> ClassConfig:
    return ClassConfig(
        class_id=0,
        pool_id=1,
        backend_domain=1,
        name="latent_mla",
        layers=layers,
        retention="full",
        bytes_per_token_per_layer=latent_bytes + rope_bytes,
        window_tokens=None,
        period_blocks=None,
        storage="latent_kv",
        components=(("latent", latent_bytes), ("rope", rope_bytes)),
    )


def latent_config(
    *,
    layers: tuple[int, ...] = (0,),
    latent_bytes: int = 16,
    rope_bytes: int = 8,
) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:mla-test",
        page_tokens=16,
        classes=(
            latent_class(
                layers=layers,
                latent_bytes=latent_bytes,
                rope_bytes=rope_bytes,
            ),
        ),
    )


def install_mla(config: RuntimeConfig) -> None:
    state._install_test_state(
        config=config,
        limits=state.RuntimeLimits(
            maximum_running_requests=4,
            chunked_prefill_tokens=64,
            maximum_context_tokens=256,
        ),
        runtime=None,
    )


def mla_configurator(
    *, kv_lora_rank: int = 8, qk_rope_head_dim: int = 4
) -> SimpleNamespace:
    text = SimpleNamespace(
        architectures=["RenamedLatentArchitecture"],
        num_hidden_layers=1,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        index_topk=None,
    )
    model = SimpleNamespace(
        hf_config=text,
        hf_text_config=text,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_hybrid_swa=False,
        is_deepseek_v4_arch=False,
        is_hybrid_swa_compress=False,
        attention_chunk_size=None,
    )
    return SimpleNamespace(
        page_size=16,
        device="cuda:0",
        kv_cache_dtype=torch.bfloat16,
        is_hybrid_swa=False,
        use_mla_backend=True,
        is_draft_worker=False,
        model_config=model,
        server_args=SimpleNamespace(
            get_attention_backends=lambda: ("flashinfer", "flashinfer")
        ),
    )


def write_mla_plan(
    path: Path, *, layers: tuple[int, ...] = (0, 1), latent: int = 1024, rope: int = 128
) -> None:
    path.write_text(
        json.dumps(
            {
                "page_tokens": 16,
                "classes": [
                    {
                        "name": "latent_mla",
                        "layers": list(layers),
                        "retention": "full",
                        "bytes_per_token_per_layer": latent + rope,
                        "window_tokens": None,
                        "storage": "latent_kv",
                        "components": [
                            {
                                "name": "latent",
                                "bytes_per_token_per_layer": latent,
                            },
                            {
                                "name": "rope",
                                "bytes_per_token_per_layer": rope,
                            },
                        ],
                    }
                ],
            }
        )
    )


def load_mla_plan(path: Path, library: Path) -> RuntimeConfig:
    environment = runtime_manifest_environment(
        path.parent,
        library,
        manager_plan=path,
        stem=path.stem,
    )
    return load_config(environment)


def drift_components(config: RuntimeConfig) -> RuntimeConfig:
    bad = replace(config.classes[0], components=(("latent", 1024), ("rope", 64)))
    return replace(config, classes=(bad,))


__all__ = [
    "drift_components",
    "install_mla",
    "latent_class",
    "latent_config",
    "load_mla_plan",
    "mla_configurator",
    "write_mla_plan",
]
