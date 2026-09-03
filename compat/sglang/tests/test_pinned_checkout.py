from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = INTEGRATION_ROOT / "bridge/src"
BRIDGE_ROOT = INTEGRATION_ROOT / "bridge"
PREPARE_SCRIPT = INTEGRATION_ROOT / "tools/prepare_source.py"
UPSTREAM_SGLANG_ROOT = Path(
    os.environ.get("ORBITKV_TEST_SGLANG_ROOT", INTEGRATION_ROOT / "source")
)
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang import pinned  # noqa: E402
from manifest_test_support import compile_runtime_manifest  # noqa: E402


PINNED_SOURCE_CONTRACT = pinned.pinned_source_contract()
HOST_SGLANG_PYTHON = Path(
    os.environ.get(
        "ORBITKV_TEST_SGLANG_PYTHON",
        REPOSITORY_ROOT / ".venv-sglang/bin/python",
    )
)
HOST_SGLANG_DEPENDENCIES = ("numpy", "torch", "sgl_kernel")


def _run(*command: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


@pytest.fixture(scope="module")
def patched_sglang_checkout(tmp_path_factory):
    if not (UPSTREAM_SGLANG_ROOT / ".git").exists():
        pytest.skip(f"pinned SGLang checkout is unavailable: {UPSTREAM_SGLANG_ROOT}")
    try:
        revision = _run(
            "git",
            "-C",
            str(UPSTREAM_SGLANG_ROOT),
            "rev-parse",
            "--verify",
            f"{PINNED_SOURCE_CONTRACT['release']}^{{commit}}",
        ).stdout.strip()
    except subprocess.CalledProcessError:
        pytest.skip("the pinned SGLang release tag is unavailable")
    assert revision == PINNED_SOURCE_CONTRACT["revision"]

    checkout = tmp_path_factory.mktemp("orbitkv-sglang-worktree") / "sglang"
    try:
        _run(
            "git",
            "-C",
            str(UPSTREAM_SGLANG_ROOT),
            "worktree",
            "add",
            "--detach",
            str(checkout),
            PINNED_SOURCE_CONTRACT["revision"],
        )
    except subprocess.CalledProcessError as error:
        pytest.skip(f"cannot create an isolated pinned worktree: {error.stderr.strip()}")
    try:
        _run(
            sys.executable,
            str(PREPARE_SCRIPT),
            "check-base",
            "--sglang-root",
            str(checkout),
        )
        _run(
            sys.executable,
            str(PREPARE_SCRIPT),
            "apply",
            "--sglang-root",
            str(checkout),
        )
        _run(
            sys.executable,
            str(PREPARE_SCRIPT),
            "verify",
            "--sglang-root",
            str(checkout),
        )
        yield checkout
    finally:
        subprocess.run(
            [
                "git",
                "-C",
                str(UPSTREAM_SGLANG_ROOT),
                "worktree",
                "remove",
                "--force",
                str(checkout),
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )


def test_adapter_metadata_has_no_second_sglang_source_or_legacy_package(
    tmp_path: Path,
):
    with (BRIDGE_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["dependencies"] == []
    assert project.get("optional-dependencies", {}) == {}
    directory = tmp_path / "installed"
    _run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--target",
        str(directory),
        str(BRIDGE_ROOT),
    )
    distributions = tuple(
        item
        for item in importlib.metadata.distributions(path=[str(directory)])
        if item.metadata.get("Name") == project["name"]
    )
    assert len(distributions) == 1
    metadata = distributions[0]
    requirements = [requirement.lower() for requirement in metadata.requires or ()]
    assert all(not requirement.startswith("sglang") for requirement in requirements)
    assert requirements == []
    installed_files = {
        str(path).replace("\\", "/") for path in metadata.files or ()
    }
    assert any(path.startswith("orbitkv_sglang/bridge/") for path in installed_files)
    assert not any(path.startswith("orbitkv_sglang/plugin/") for path in installed_files)

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(directory),
        }
    )
    _run(
        sys.executable,
        "-S",
        "-c",
        (
            "import importlib.util; "
            "assert importlib.util.find_spec('orbitkv_sglang.bridge') is not None; "
            "assert importlib.util.find_spec('orbitkv_sglang.plugin') is None"
        ),
        env=environment,
    )


def test_default_bridge_lifecycle_import_does_not_require_optional_runtime():
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(SOURCE_ROOT),
        }
    )
    environment.pop("SGLANG_PLUGINS", None)
    _run(
        sys.executable,
        "-S",
        "-c",
        "import orbitkv_sglang.bridge.session_lifecycle",
        env=environment,
    )


def _pinned_env(checkout: Path) -> dict[str, str]:
    runtime_manifest = compile_runtime_manifest(
        checkout.parent,
        manager_plan=REPOSITORY_ROOT / "core/examples/sliding-token-manager-plan.json",
        stem="pinned-loader",
    )
    env = dict(os.environ)
    python_path = [str(SOURCE_ROOT), str(checkout / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "ORBITKV_SGLANG_ROOT": str(checkout),
            "ORBITKV_RUNTIME_MANIFEST": str(runtime_manifest),
            # Hook activation only validates that the configured manager artifact
            # is a regular file; it does not load the ABI until arena creation.
            "ORBITKV_LIBRARY": str(PREPARE_SCRIPT),
        }
    )
    env.pop("SGLANG_PLUGINS", None)
    return env


def _pinned_host_env(checkout: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "FLASHINFER_WORKSPACE_BASE": str(checkout.parent / "flashinfer-cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(checkout / "python"), str(SOURCE_ROOT))
            ),
        }
    )
    env.pop("SGLANG_PLUGINS", None)
    return env


def _pinned_python() -> str:
    return str(HOST_SGLANG_PYTHON) if HOST_SGLANG_PYTHON.is_file() else sys.executable


def _run_pinned_host(checkout: Path, code: str, env: dict[str, str]):
    imports = "; ".join(f"import {name}" for name in HOST_SGLANG_DEPENDENCIES)
    try:
        _run(_pinned_python(), "-c", imports, env=env)
    except subprocess.CalledProcessError:
        pytest.skip("the pinned SGLang host-test dependencies are unavailable")
    return _run(_pinned_python(), "-c", code, env=env)


def test_reviewed_overlay_is_the_only_full_source_tree_mutation(
    patched_sglang_checkout,
):
    checkout = patched_sglang_checkout
    assert pinned.validate_patched_checkout(checkout) == checkout.resolve()

    extra = checkout / "python/sglang/_orbitkv_unreviewed.py"
    extra.write_text("unreviewed = True\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="exactly the reviewed"):
            pinned.validate_patched_checkout(checkout)
    finally:
        extra.unlink()

    upstream_readme = checkout / "README.md"
    original_readme = upstream_readme.read_bytes()
    upstream_readme.write_bytes(original_readme + b"\nunreviewed product edit\n")
    try:
        with pytest.raises(RuntimeError, match="exactly the reviewed"):
            pinned.validate_patched_checkout(checkout)
    finally:
        upstream_readme.write_bytes(original_readme)

    root_extra = checkout / "orbitkv-unreviewed.txt"
    root_extra.write_text("unreviewed\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="exactly the reviewed"):
            pinned.validate_patched_checkout(checkout)
    finally:
        root_extra.unlink()

    for target_identity in PINNED_SOURCE_CONTRACT["targets"]:
        relative = target_identity["path"]
        target = checkout / relative
        reviewed = target.read_bytes()
        target.write_bytes(reviewed + b"# unreviewed\n")
        try:
            with pytest.raises(RuntimeError, match="unexpected hash"):
                pinned.validate_patched_checkout(checkout)
        finally:
            target.write_bytes(reviewed)
    pinned.validate_patched_checkout(checkout)


@pytest.mark.parametrize(
    ("seq_len", "expected_query_starts", "expected_k_lens", "expected_pages"),
    (
        (17, [0, 17], [17], [[10, 20]]),
        (31, [0, 31], [31], [[10, 20]]),
        (32, [0, 32], [32], [[10, 20]]),
        (33, [0, 32, 33], [32, 1], [[10, 20], [30, 30]]),
    ),
)
def test_fa3_local_attention_keeps_chunk_geometry_and_uses_page_ids(
    patched_sglang_checkout,
    seq_len,
    expected_query_starts,
    expected_k_lens,
    expected_pages,
):
    code = r'''
import json
import os
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)

page_size = 16
seq_len = int(os.environ["ORBITKV_TEST_SEQ_LEN"])
token_table = torch.cat(
    [torch.arange(page * page_size, (page + 1) * page_size) for page in (10, 20, 30)]
).to(torch.int32).reshape(1, -1)
backend = SimpleNamespace(
    has_local_attention=True,
    use_sliding_window_kv_pool=False,
    _unified_dense=False,
    page_size=page_size,
    attention_chunk_size=32,
)
metadata = FlashAttentionMetadata(
    cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32),
    cache_seqlens_int32=torch.tensor([seq_len], dtype=torch.int32),
    page_table=token_table,
)
FlashAttentionBackend._maybe_init_local_attn_metadata(
    backend, None, metadata, torch.device("cpu")
)
local = metadata.local_attn_metadata
print(json.dumps({
    "query_starts": local.local_query_start_loc.tolist(),
    "k_lens": local.local_seqused_k.tolist(),
    "pages": local.local_block_table.tolist(),
}))
'''
    env = _pinned_host_env(patched_sglang_checkout)
    env["ORBITKV_TEST_SEQ_LEN"] = str(seq_len)
    completed = _run_pinned_host(patched_sglang_checkout, code, env)
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result == {
        "query_starts": expected_query_starts,
        "k_lens": expected_k_lens,
        "pages": expected_pages,
    }


def test_fa3_local_attention_prefill_crosses_chunk_and_batch_is_invariant(
    patched_sglang_checkout,
):
    code = r'''
import json

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    make_local_attention_virtual_batches,
)

single = make_local_attention_virtual_batches(
    32,
    __import__("numpy").array([0, 4], dtype="int32"),
    __import__("numpy").array([33], dtype="int32"),
    torch.tensor([[10, 20, 30]], dtype=torch.int32),
    16,
)
batch = make_local_attention_virtual_batches(
    32,
    __import__("numpy").array([0, 4, 21], dtype="int32"),
    __import__("numpy").array([33, 17], dtype="int32"),
    torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int32),
    16,
)
def serial(value):
    return [value[0].tolist(), value[1].tolist(), value[2].tolist(), value[3].tolist()]
print(json.dumps({"single": serial(single), "batch": serial(batch)}))
'''
    completed = _run_pinned_host(
        patched_sglang_checkout, code, _pinned_host_env(patched_sglang_checkout)
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["single"] == [
        [3, 1],
        [0, 3, 4],
        [32, 1],
        [[10, 20], [30, 30]],
    ]
    assert result["batch"] == [
        [3, 1, 17],
        [0, 3, 4, 21],
        [32, 1, 17],
        [[10, 20], [30, 30], [40, 50]],
    ]
    assert result["batch"][0][:2] == result["single"][0]
    assert result["batch"][2][:2] == result["single"][2]
    assert result["batch"][3][:2] == result["single"][3]


@pytest.mark.parametrize(
    ("chunk_size", "page_size", "table_width", "message"),
    (
        (0, 16, 1, "attn_chunk_size must be a positive integer"),
        (32, 0, 1, "page_size must be a positive integer"),
        (31, 16, 2, "is not divisible by page_size"),
        (32, 16, 1, "block_table is too narrow"),
    ),
)
def test_fa3_local_attention_rejects_invalid_geometry(
    patched_sglang_checkout, chunk_size, page_size, table_width, message
):
    code = r'''
import os

import numpy as np
import torch

from sglang.srt.layers.attention.flashattention_backend import (
    make_local_attention_virtual_batches,
)

try:
    make_local_attention_virtual_batches(
        int(os.environ["ORBITKV_TEST_CHUNK"]),
        np.array([0, 17], dtype=np.int32),
        np.array([17], dtype=np.int32),
        torch.arange(int(os.environ["ORBITKV_TEST_WIDTH"]), dtype=torch.int32).reshape(1, -1),
        int(os.environ["ORBITKV_TEST_PAGE"]),
    )
except ValueError as error:
    print(error)
else:
    raise AssertionError("invalid local-attention geometry was accepted")
'''
    env = _pinned_host_env(patched_sglang_checkout)
    env.update(
        ORBITKV_TEST_CHUNK=str(chunk_size),
        ORBITKV_TEST_PAGE=str(page_size),
        ORBITKV_TEST_WIDTH=str(table_width),
    )
    completed = _run_pinned_host(patched_sglang_checkout, code, env)
    assert message in completed.stdout


def test_patched_sources_dispatch_directly_without_hook_registry(
    patched_sglang_checkout,
):
    code = r'''
import ast
import os
import pathlib
import sys

assert "SGLANG_PLUGINS" not in os.environ
points = (
    ("sglang/srt/mem_cache/kv_cache_configurator.py", "KVCacheConfigurator.configure", "configure"),
    ("sglang/srt/mem_cache/kv_cache_configurator.py", "KVCacheConfigurator._build_token_to_kv_pool_allocator", "build_allocator"),
    ("sglang/srt/mem_cache/allocation.py", "alloc_for_extend", "prepare_extend"),
    ("sglang/srt/mem_cache/allocation.py", "alloc_for_decode", "prepare_decode"),
    ("sglang/srt/managers/schedule_batch.py", "ScheduleBatch.maybe_evict_swa", "maybe_evict_swa"),
    ("sglang/srt/managers/scheduler.py", "Scheduler._prepare_waiting_request_removal", "prepare_waiting_request_removal"),
    ("sglang/srt/managers/scheduler.py", "Scheduler.get_next_batch_to_run", "next_batch"),
    ("sglang/srt/managers/scheduler.py", "Scheduler.run_batch", "run_batch"),
    ("sglang/srt/managers/scheduler.py", "Scheduler.get_internal_state", "internal_state"),
    ("sglang/srt/mem_cache/common.py", "release_kv_cache", "release_request"),
)
root = pathlib.Path(os.environ["ORBITKV_SGLANG_ROOT"]) / "python"
for relative, symbol, owner_method in points:
    module = ast.parse((root / relative).read_text())
    nodes = module.body
    for part in symbol.split("."):
        node = next(item for item in nodes if getattr(item, "name", None) == part)
        nodes = node.body
    assert any(
        isinstance(item, ast.ImportFrom)
        and item.module == "orbitkv_sglang.engine"
        and any(alias.name == "get_owner" for alias in item.names)
        for item in ast.walk(node)
    )
    assert any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == owner_method
        for item in ast.walk(node)
    )
import orbitkv_sglang.engine
assert "sglang.srt.plugins.hook_registry" not in sys.modules
print("direct_seams=10 hook_registry=absent")
'''
    completed = _run(
        sys.executable,
        "-c",
        code,
        env=_pinned_env(patched_sglang_checkout),
    )
    assert "direct_seams=10 hook_registry=absent" in completed.stdout


def test_distribution_does_not_publish_legacy_general_plugin_entrypoint():
    project_text = (BRIDGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.entry-points."sglang.srt.plugins"]' not in project_text
    assert "orbitkv_manager =" not in project_text
