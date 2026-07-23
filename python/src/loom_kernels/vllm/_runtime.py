"""Shared vLLM version and environment policy."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

DEFAULT_PROVIDER = "loom_cuda"
SUPPORTED_VLLM_SERIES = ((0, 24), (0, 25))

def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def installed_vllm_version() -> str | None:
    """Return the installed vLLM release, if the package is present."""
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def supports_installed_vllm() -> bool:
    """Return whether the installed vLLM series is qualified by Loom."""
    release = installed_vllm_version()
    if release is None:
        return False
    components = release.split(".", 2)
    if len(components) < 2:
        return False
    try:
        series = (int(components[0]), int(components[1]))
    except ValueError:
        return False
    return series in SUPPORTED_VLLM_SERIES
