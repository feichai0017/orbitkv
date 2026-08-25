from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
REFERENCE_SOURCE = INTEGRATION_ROOT / "src"
RUNTIME_SOURCE = REPOSITORY_ROOT / "python" / "orbitkv-runtime" / "src"


def _top_level_imports(root: Path) -> set[str]:
    result = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.add(node.module.split(".", 1)[0])
    return result


def test_neutral_and_reference_sources_never_import_sglang() -> None:
    runtime_imports = _top_level_imports(RUNTIME_SOURCE / "orbitkv_runtime")
    reference_imports = _top_level_imports(REFERENCE_SOURCE / "orbitkv_reference")

    assert "sglang" not in runtime_imports | reference_imports
    assert "torch" not in runtime_imports


def test_reference_package_import_is_torch_optional() -> None:
    script = """
import builtins
import sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('reference import attempted to import torch')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import orbitkv_reference
assert 'torch' not in sys.modules
assert orbitkv_reference.ReferencePagedAdapter
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(RUNTIME_SOURCE), str(REFERENCE_SOURCE))
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
