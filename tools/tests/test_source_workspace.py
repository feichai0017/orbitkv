import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "source_workspace.py"
SPEC = importlib.util.spec_from_file_location("source_workspace", MODULE_PATH)
source_workspace = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(source_workspace)


def flattened(plan):
    return " ".join(argument for _, command in plan for argument in command)


class SourceWorkspaceTests(unittest.TestCase):
    def test_all_stacks_have_an_explicit_source_path(self):
        expected = {
            "sglang": "third-party/sglang",
            "autodeploy": "third-party/tensorrt-llm",
            "tensorrt-source": "third-party/tensorrt-llm",
            "kernels": "third-party/flashinfer",
        }
        environment = Path("/tmp/aletheia-test-env")
        for stack, source in expected.items():
            self.assertIn(source, flattened(source_workspace.commands(stack, environment)))

    def test_sglang_builds_its_kernel_from_the_submodule(self):
        environment = Path("/tmp/aletheia-test-env")
        plan = flattened(source_workspace.commands("sglang", environment))
        self.assertIn("tools/build_sglang_kernel.py", plan)
        self.assertIn("sglang_kernel-*.whl", plan)
        self.assertIn("tools/build_deepgemm.py", plan)

    def test_full_tensorrt_build_is_not_precompiled_mode(self):
        environment = Path("/tmp/aletheia-test-env")
        fast = flattened(source_workspace.commands("autodeploy", environment))
        full = flattened(source_workspace.commands("tensorrt-source", environment))
        self.assertIn("TRTLLM_USE_PRECOMPILED=1", fast)
        self.assertIn("scripts/build_wheel.py", full)
        self.assertNotIn("TRTLLM_USE_PRECOMPILED=1", full)

    def test_doctor_is_machine_readable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            source_workspace.doctor()
        report = json.loads(output.getvalue())
        self.assertIn("source-only", report["stacks"])
        self.assertTrue(report["stacks"]["source-only"]["build_ready"])
        self.assertTrue(report["stacks"]["source-only"]["run_ready"])

    def test_build_environment_adds_detected_cuda_home(self):
        environment = Path("/tmp/aletheia-test-env")
        with mock.patch.object(source_workspace, "_find_nvcc", return_value="/opt/cuda/bin/nvcc"):
            prefix = source_workspace._build_environment(environment)
        self.assertIn("CUDA_HOME=/opt/cuda", prefix)
        self.assertTrue(any("/opt/cuda/bin" in item for item in prefix))
        self.assertTrue(any(item.startswith("FLASHINFER_WORKSPACE_BASE=") for item in prefix))
        self.assertTrue(any(".aletheia/cache" in item for item in prefix))


if __name__ == "__main__":
    unittest.main()
