"""Backward-compatible alias for :mod:`orbitkv.vllm`.

New integrations should import ``orbitkv.vllm``. This alias preserves the
historical vLLM connector module path while deployments migrate.
"""

from __future__ import annotations

import sys

from orbitkv import vllm as _vllm
from orbitkv.vllm import *  # noqa: F403

for _module in (
    "common",
    "connector_metrics",
    "scheduler",
    "state_manager",
    "tp_shards",
    "worker",
):
    _imported = __import__(f"orbitkv.vllm.{_module}", fromlist=[_module])
    sys.modules[f"{__name__}.{_module}"] = _imported
    globals()[_module] = _imported

__all__ = _vllm.__all__
