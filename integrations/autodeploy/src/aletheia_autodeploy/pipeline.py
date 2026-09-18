"""Safe AletheiaRT entrypoints for source-pinned AutoDeploy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .transforms import register


def inventory_transform_config(output: str, *, stage: str = "visualize") -> dict[str, Any]:
    return {
        "stage": stage,
        "output": output,
        "run_per_gm": True,
        "run_graph_cleanup": False,
        "requires_clean_graph": False,
    }


def registered_optimizer(factory: Any, config: Mapping[str, Any]):
    """Construct upstream InferenceOptimizer after local registration.

    AutoDeploy does not discover third-party transforms through Python entry
    points. All local tools use this wrapper so registration cannot be omitted.
    """

    register()
    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

    return InferenceOptimizer(factory=factory, config=config)
