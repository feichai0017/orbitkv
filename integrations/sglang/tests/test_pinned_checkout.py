from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "src"
PREPARE_SCRIPT = INTEGRATION_ROOT / "prepare_pinned_checkout.py"
UPSTREAM_SGLANG_ROOT = Path("/workspace/sglang")
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang import pinned  # noqa: E402


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
        pytest.skip("pinned /workspace/sglang checkout is unavailable")
    try:
        revision = _run(
            "git",
            "-C",
            str(UPSTREAM_SGLANG_ROOT),
            "rev-parse",
            "--verify",
            f"{pinned.SUPPORTED_SGLANG_RELEASE}^{{commit}}",
        ).stdout.strip()
    except subprocess.CalledProcessError:
        pytest.skip("the pinned SGLang release tag is unavailable")
    assert revision == pinned.SUPPORTED_SGLANG_REVISION

    checkout = tmp_path_factory.mktemp("orbitkv-sglang-worktree") / "sglang"
    _run(
        "git",
        "-C",
        str(UPSTREAM_SGLANG_ROOT),
        "worktree",
        "add",
        "--detach",
        str(checkout),
        pinned.SUPPORTED_SGLANG_REVISION,
    )
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


def test_python_dependency_matches_pinned_release():
    project_lines = (INTEGRATION_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    ).splitlines()
    expected = (
        'dependencies = ["sglang=='
        f'{pinned.SUPPORTED_SGLANG_RELEASE.removeprefix("v")}"]'
    )
    assert project_lines.count(expected) == 1


def _pinned_env(checkout: Path) -> dict[str, str]:
    env = dict(os.environ)
    python_path = [str(SOURCE_ROOT), str(checkout / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "SGLANG_PLUGINS": "orbitkv_manager",
            "ORBITKV_SGLANG_ROOT": str(checkout),
            "ORBITKV_PLAN": str(REPOSITORY_ROOT / "examples/mistral_uniform_swa.json"),
            # Hook activation only validates that the configured manager artifact
            # is a regular file; it does not load the ABI until arena creation.
            "ORBITKV_LIBRARY": str(PREPARE_SCRIPT),
        }
    )
    return env


def test_reviewed_patch_is_the_only_python_tree_mutation(patched_sglang_checkout):
    checkout = patched_sglang_checkout
    assert pinned.validate_patched_checkout(checkout) == checkout.resolve()

    extra = checkout / "python/sglang/_orbitkv_unreviewed.py"
    extra.write_text("unreviewed = True\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="exactly the reviewed"):
            pinned.validate_patched_checkout(checkout)
    finally:
        extra.unlink()

    target = checkout / pinned.PATCHED_SOURCE_PATH
    reviewed = target.read_bytes()
    target.write_bytes(reviewed + b"# unreviewed\n")
    try:
        with pytest.raises(RuntimeError, match="unexpected hash"):
            pinned.validate_patched_checkout(checkout)
    finally:
        target.write_bytes(reviewed)
    pinned.validate_patched_checkout(checkout)


@pytest.mark.parametrize("failure", ("selection", "missing", "load"))
def test_patched_loader_fails_before_stock_fallback(
    patched_sglang_checkout, failure
):
    env = _pinned_env(patched_sglang_checkout)
    env["ORBITKV_LOADER_FAILURE"] = failure
    code = r'''
import os
import sglang.srt.plugins as plugins

failure = os.environ["ORBITKV_LOADER_FAILURE"]
if failure == "selection":
    pass
elif failure == "missing":
    plugins.entry_points = lambda **_kwargs: []
else:
    class BrokenEntryPoint:
        name = "orbitkv_manager"
        value = "broken:register"
        dist = None

        def load(self):
            raise RuntimeError("injected ep.load failure")

    plugins.entry_points = lambda **_kwargs: [BrokenEntryPoint()]

try:
    plugins.load_plugins()
except SystemExit as error:
    assert not plugins._plugins_loaded
    print(error)
else:
    raise AssertionError("patched loader continued to the stock runtime")
'''
    if failure == "selection":
        env.pop("SGLANG_PLUGINS", None)
    completed = _run(sys.executable, "-c", code, env=env)
    expected = (
        "SGLANG_PLUGINS=orbitkv_manager"
        if failure == "selection"
        else "required orbitkv_manager entrypoint"
    )
    assert expected in completed.stdout


def test_patched_loader_registers_all_hooks_and_propagated_aliases(
    patched_sglang_checkout,
):
    code = r'''
from sglang.srt.plugins import load_plugins

load_plugins()

from orbitkv_sglang.plugin.validation import HOOK_TARGETS
from sglang.srt.mem_cache import allocation, common
from sglang.srt.managers import schedule_batch, scheduler
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.mem_cache.registry import (
    get_radix_cache_factory,
    registered_radix_cache_backends,
)
from sglang.srt.plugins.hook_registry import HookRegistry

assert HOOK_TARGETS == (
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool_allocator",
    "sglang.srt.mem_cache.allocation.alloc_for_extend",
    "sglang.srt.mem_cache.allocation.alloc_for_decode",
    "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
    "sglang.srt.mem_cache.common.release_kv_cache",
    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    "sglang.srt.managers.scheduler.Scheduler.run_batch",
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure",
    "sglang.srt.managers.scheduler.Scheduler.get_internal_state",
)
assert all(target in HookRegistry._patched for target in HOOK_TARGETS)
assert schedule_batch.alloc_for_extend is allocation.alloc_for_extend
assert schedule_batch.alloc_for_decode is allocation.alloc_for_decode
assert schedule_batch.release_kv_cache is common.release_kv_cache
assert scheduler.release_kv_cache is common.release_kv_cache
assert batch_result_processor.release_kv_cache is common.release_kv_cache
assert get_radix_cache_factory("orbitkv") is not None
assert "orbitkv" in registered_radix_cache_backends()
print(f"hooks={len(HOOK_TARGETS)} aliases=5 radix=orbitkv")
'''
    completed = _run(
        sys.executable,
        "-c",
        code,
        env=_pinned_env(patched_sglang_checkout),
    )
    assert "hooks=9 aliases=5 radix=orbitkv" in completed.stdout
