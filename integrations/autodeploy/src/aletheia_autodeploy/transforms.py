"""AutoDeploy transforms that export evidence for AletheiaRT."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Type

_REGISTERED = False


def _node_target(node: Any) -> str:
    target = getattr(node, "target", None)
    return getattr(target, "__qualname__", None) or str(target)


def inventory(graph_module: Any) -> dict[str, Any]:
    """Return a deterministic, non-mutating inventory of an FX graph."""

    nodes = list(graph_module.graph.nodes)
    operations = Counter(
        f"{node.op}:{_node_target(node)}"
        for node in nodes
        if node.op not in {"placeholder", "output"}
    )
    readable = graph_module.print_readable(print_output=False)
    return {
        "schema_version": 1,
        "graph_sha256": hashlib.sha256(readable.encode()).hexdigest(),
        "nodes": len(nodes),
        "operations": dict(sorted(operations.items())),
    }


def register() -> None:
    """Register the inventory transform with AutoDeploy exactly once."""

    global _REGISTERED
    if _REGISTERED:
        return

    from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactory
    from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
    from tensorrt_llm._torch.auto_deploy.transform.interface import (
        BaseTransform,
        SharedConfig,
        TransformConfig,
        TransformInfo,
        TransformRegistry,
    )

    class QualifiedGraphInventoryConfig(TransformConfig):
        output: str

    @TransformRegistry.register("qualified_graph_inventory")
    class QualifiedGraphInventory(BaseTransform):
        config: QualifiedGraphInventoryConfig

        @classmethod
        def get_config_class(cls) -> Type[TransformConfig]:
            return QualifiedGraphInventoryConfig

        def _apply(
            self,
            gm: Any,
            cm: CachedSequenceInterface,
            factory: ModelFactory,
            shared_config: SharedConfig,
        ) -> tuple[Any, TransformInfo]:
            del cm, factory
            report = inventory(gm)
            report["local_rank"] = shared_config.local_rank
            report["world_size"] = shared_config.world_size
            output = Path(self.config.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return gm, TransformInfo(
                skipped=False,
                num_matches=report["nodes"],
                is_clean=True,
                has_valid_shapes=True,
            )

    _REGISTERED = True
