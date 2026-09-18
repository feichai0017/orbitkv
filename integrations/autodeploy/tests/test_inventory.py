import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aletheia_autodeploy.transforms import inventory  # noqa: E402
import aletheia_autodeploy.transforms as transforms  # noqa: E402
from aletheia_autodeploy.pipeline import inventory_transform_config  # noqa: E402


class Node:
    def __init__(self, op, target):
        self.op = op
        self.target = target


class Graph:
    nodes = [Node("placeholder", "x"), Node("call_function", "aten.add"), Node("output", "output")]


class GraphModule:
    graph = Graph()

    @staticmethod
    def print_readable(*, print_output):
        assert not print_output
        return "x -> aten.add -> output"


class InventoryTests(unittest.TestCase):
    def test_inventory_config_uses_visualize_stage_without_cleanup(self):
        config = inventory_transform_config("inventory.json")
        self.assertEqual(config["stage"], "visualize")
        self.assertEqual(config["output"], "inventory.json")
        self.assertFalse(config["run_graph_cleanup"])

    def test_inventory_is_stable_and_ignores_io_nodes(self):
        report = inventory(GraphModule())
        self.assertEqual(report["nodes"], 3)
        self.assertEqual(report["operations"], {"call_function:aten.add": 1})
        self.assertEqual(len(report["graph_sha256"]), 64)

    def test_register_uses_autodeploy_transform_registry(self):
        registered = {}

        class TransformRegistry:
            @classmethod
            def register(cls, name):
                def decorator(transform):
                    registered[name] = transform
                    return transform

                return decorator

        class BaseTransform:
            pass

        class TransformConfig:
            pass

        class TransformInfo:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        interface = types.ModuleType(
            "tensorrt_llm._torch.auto_deploy.transform.interface"
        )
        interface.BaseTransform = BaseTransform
        interface.SharedConfig = object
        interface.TransformConfig = TransformConfig
        interface.TransformInfo = TransformInfo
        interface.TransformRegistry = TransformRegistry
        factory = types.ModuleType(
            "tensorrt_llm._torch.auto_deploy.models.factory"
        )
        factory.ModelFactory = object
        shim = types.ModuleType(
            "tensorrt_llm._torch.auto_deploy.shim.interface"
        )
        shim.CachedSequenceInterface = object
        modules = {
            "tensorrt_llm": types.ModuleType("tensorrt_llm"),
            "tensorrt_llm._torch": types.ModuleType("tensorrt_llm._torch"),
            "tensorrt_llm._torch.auto_deploy": types.ModuleType("tensorrt_llm._torch.auto_deploy"),
            "tensorrt_llm._torch.auto_deploy.transform": types.ModuleType("tensorrt_llm._torch.auto_deploy.transform"),
            "tensorrt_llm._torch.auto_deploy.transform.interface": interface,
            "tensorrt_llm._torch.auto_deploy.models": types.ModuleType("tensorrt_llm._torch.auto_deploy.models"),
            "tensorrt_llm._torch.auto_deploy.models.factory": factory,
            "tensorrt_llm._torch.auto_deploy.shim": types.ModuleType("tensorrt_llm._torch.auto_deploy.shim"),
            "tensorrt_llm._torch.auto_deploy.shim.interface": shim,
        }
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        transforms._REGISTERED = False
        try:
            transforms.register()
        finally:
            transforms._REGISTERED = False
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.assertIn("qualified_graph_inventory", registered)


if __name__ == "__main__":
    unittest.main()
