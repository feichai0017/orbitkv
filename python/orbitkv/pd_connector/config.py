"""Connector config helpers."""

from __future__ import annotations

from typing import Any


def extra_config_value(vllm_config: Any, key: str, default: Any = None) -> Any:
    config = getattr(vllm_config, "kv_transfer_config", None)
    getter = getattr(config, "get_from_extra_config", None)
    if getter is not None:
        return getter(key, default)
    extra_config = getattr(config, "extra_config", None)
    if isinstance(extra_config, dict):
        return extra_config.get(key, default)
    return default
