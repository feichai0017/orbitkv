from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


sys.dont_write_bytecode = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "tools/verify_engine_profile.py"
SPEC = importlib.util.spec_from_file_location("verify_engine_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _function(name: str) -> str:
    return f"def {name}(*args, **kwargs):\n    return None\n"


def _direct_function(name: str, owner_method: str) -> str:
    return (
        f"def {name}(*args, **kwargs):\n"
        "    from orbitkv_sglang.engine import get_owner\n"
        f"    return get_owner().{owner_method}(*args, **kwargs)\n"
    )


def _direct_class(
    name: str, methods: tuple[tuple[str, str, str | None], ...]
) -> str:
    bodies = []
    for method, owner_method, native in methods:
        arguments = (
            f"{name}.{native}, self, *args, **kwargs"
            if native is not None
            else "self, *args, **kwargs"
        )
        bodies.append(
            f"    def {method}(self, *args, **kwargs):\n"
            "        from orbitkv_sglang.engine import get_owner\n"
            f"        return get_owner().{owner_method}({arguments})"
        )
        if native is not None:
            bodies.append(
                f"    def {native}(self, *args, **kwargs):\n"
                "        return None"
            )
    body = "\n".join(bodies)
    return f"class {name}:\n{body}\n"


def _direct_scheduler() -> str:
    return (
        "class Scheduler:\n"
        "    def _prepare_waiting_request_removal(self, req):\n"
        "        from orbitkv_sglang.engine import get_owner\n"
        "        return get_owner().prepare_waiting_request_removal(req, self.tree_cache)\n"
        "    def _retry_waiting_request_removals(self):\n"
        "        return None\n"
        "    def _defer_waiting_request_removal(self, req, output):\n"
        "        req._orbitkv_waiting_removal_pending = True\n"
        "        return None\n"
        "    def _abort_on_queued_limit(self, candidate_req, idx):\n"
        "        if self._prepare_waiting_request_removal(candidate_req):\n"
        "            self.waiting_queue.pop(idx)\n"
        "        else:\n"
        "            self._defer_waiting_request_removal(candidate_req, output)\n"
        "    def _abort_on_waiting_timeout(self, req, index):\n"
        "        if not self._prepare_waiting_request_removal(req):\n"
        "            self._defer_waiting_request_removal(req, output)\n"
        "        else:\n"
        "            self.waiting_queue.pop(index)\n"
        "    def abort_request(self, req, i):\n"
        "        if not self._prepare_waiting_request_removal(req):\n"
        "            self._defer_waiting_request_removal(req, output)\n"
        "        else:\n"
        "            self.waiting_queue.pop(i)\n"
        "    def get_next_batch_to_run(self, *args, **kwargs):\n"
        "        from orbitkv_sglang.engine import get_owner\n"
        "        return get_owner().next_batch(Scheduler._orbitkv_native_get_next_batch_to_run, self, *args, **kwargs)\n"
        "    def _orbitkv_native_get_next_batch_to_run(self, *args, **kwargs):\n"
        "        self._retry_waiting_request_removals()\n"
        "        for req in self.waiting_queue:\n"
        '            if getattr(req, "_orbitkv_waiting_removal_pending", False):\n'
        "                continue\n"
        "        return self.get_new_batch_prefill()\n"
        "    def run_batch(self, *args, **kwargs):\n"
        "        from orbitkv_sglang.engine import get_owner\n"
        "        return get_owner().run_batch(Scheduler._orbitkv_native_run_batch, self, *args, **kwargs)\n"
        "    def _orbitkv_native_run_batch(self, *args, **kwargs):\n"
        "        return None\n"
        "    def get_internal_state(self, *args, **kwargs):\n"
        "        from orbitkv_sglang.engine import get_owner\n"
        "        return get_owner().internal_state(Scheduler._orbitkv_native_get_internal_state, self, *args, **kwargs)\n"
        "    def _orbitkv_native_get_internal_state(self, *args, **kwargs):\n"
        "        return None\n"
    )


def _write_direct_overlay(root: Path) -> None:
    sources = {
        "python/sglang/srt/layers/attention/flashattention_backend.py": (
            _function("make_local_attention_virtual_batches")
            + "# reviewed page-ID compatibility overlay\n"
        ),
        "python/sglang/srt/mem_cache/kv_cache_configurator.py": _direct_class(
            "KVCacheConfigurator",
            (
                ("configure", "configure", "_orbitkv_native_configure"),
                ("_build_token_to_kv_pool_allocator", "build_allocator", None),
            ),
        ),
        "python/sglang/srt/mem_cache/allocation.py": (
            _direct_function("alloc_for_extend", "prepare_extend")
            + _direct_function("alloc_for_decode", "prepare_decode")
        ),
        "python/sglang/srt/managers/schedule_batch.py": _direct_class(
            "ScheduleBatch", (("maybe_evict_swa", "maybe_evict_swa", None),)
        ),
        "python/sglang/srt/managers/scheduler.py": _direct_scheduler(),
        "python/sglang/srt/mem_cache/common.py": _direct_function(
            "release_kv_cache", "release_request"
        ),
    }
    for relative, source in sources.items():
        (root / relative).write_text(source, encoding="utf-8")


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    base_sources = {
        "LICENSE": "Apache License\nVersion 2.0, January 2004\n",
        "README.md": "full upstream source fixture\n",
        "scripts/upstream-tool.sh": "#!/bin/sh\nexit 0\n",
        "python/sglang/__init__.py": "",
        "python/sglang/srt/layers/attention/flashattention_backend.py": (
            _function("make_local_attention_virtual_batches")
        ),
        "python/sglang/srt/mem_cache/kv_cache_configurator.py": (
            "class KVCacheConfigurator:\n"
            "    def configure(self):\n        return None\n"
            "    def _build_token_to_kv_pool_allocator(self):\n        return None\n"
        ),
        "python/sglang/srt/mem_cache/allocation.py": (
            _function("alloc_for_extend") + _function("alloc_for_decode")
        ),
        "python/sglang/srt/managers/schedule_batch.py": (
            "class ScheduleBatch:\n"
            "    def maybe_evict_swa(self):\n        return None\n"
        ),
        "python/sglang/srt/mem_cache/common.py": _function("release_kv_cache"),
        "python/sglang/srt/managers/scheduler.py": "class Scheduler:\n    pass\n",
    }
    for relative, text in base_sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    executable = tmp_path / "scripts/upstream-tool.sh"
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "user.name", "OrbitKV fixture")
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-qm", "pinned upstream")
    _write_direct_overlay(tmp_path)
    return tmp_path


def _fixture_contract(root: Path) -> dict:
    contract = copy.deepcopy(
        verifier._load_source_contract(
            "orbitkv_sglang.pinned:pinned_source_contract"
        )
    )
    contract["revision"] = _git(root, "rev-parse", "HEAD")
    for target in contract["targets"]:
        relative = target["path"]
        target["base_sha256"] = hashlib.sha256(
            subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"],
                check=True,
                capture_output=True,
                timeout=10,
            ).stdout
        ).hexdigest()
        target["patched_sha256"] = hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
    return contract


@pytest.fixture(autouse=True)
def fixture_source_contract(
    source_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    contract = _fixture_contract(source_tree)
    monkeypatch.setattr(
        verifier, "_load_source_contract", lambda _provider: copy.deepcopy(contract)
    )
    return contract


@pytest.fixture
def profile() -> dict:
    return json.loads(
        (REPOSITORY_ROOT / "compat/sglang/profile.json").read_text(encoding="utf-8")
    )


def test_repository_profile_declares_full_source_overlay(
    profile: dict, source_tree: Path
) -> None:
    summary = verifier.validate_profile(profile, source_tree)

    assert summary == {
        "product_id": "orbitkv-engine",
        "capabilities": 8,
        "overlay_targets": 6,
        "manager_supported_profiles": 5,
        "manager_unsupported_profiles": 2,
        "manager_takeover_points": 6,
        "lifecycle_notification_points": 3,
        "observability_extensions": 1,
        "compatibility_fixes": 1,
    }
    assert profile["schema_version"] == 4
    assert profile["capabilities"] == [
        "continuous-batched-text-generation",
        "full-attention",
        "http-generation-api",
        "multi-head-latent-attention",
        "paged-token-kv",
        "prefix-cache",
        "runtime-observability",
        "sliding-window-attention",
    ]
    assert profile["mode"] == "pinned-source-overlay"
    assert profile["source_preservation"] == "all-upstream-tracked-paths"
    for removed_field in ("retained_paths", "excluded_paths", "excluded_features"):
        assert removed_field not in profile


def test_overlay_targets_are_the_exact_pinned_contract(
    profile: dict, source_tree: Path, fixture_source_contract: dict
) -> None:
    assert profile["overlay_targets"] == [
        target["path"] for target in fixture_source_contract["targets"]
    ]
    verifier.validate_profile(profile, source_tree)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("product_id", "engine"),
        ("mode", "trimmed-fork"),
        ("product_scope", "text-only-subtree"),
        ("source_preservation", "selected-paths"),
    ),
)
def test_product_identity_and_preservation_are_fixed(
    profile: dict, source_tree: Path, field: str, value: str
) -> None:
    profile[field] = value
    with pytest.raises(RuntimeError, match=field):
        verifier.validate_profile(profile, source_tree)


def test_schema_rejects_old_version_and_unknown_fields(
    profile: dict, source_tree: Path
) -> None:
    old_profile = copy.deepcopy(profile)
    old_profile["schema_version"] = 3
    with pytest.raises(RuntimeError, match="unsupported schema version"):
        verifier.validate_profile(old_profile, source_tree)

    profile["retained_paths"] = ["python/sglang/**"]
    with pytest.raises(RuntimeError, match="invalid schema"):
        verifier.validate_profile(profile, source_tree)


def test_supported_profiles_are_exact_and_ordered(
    profile: dict, source_tree: Path
) -> None:
    assert [
        (item["execution_topology"], item["cache_policy"])
        for item in profile["manager_supported_profiles"]
    ] == [
        ("whole_domain_full_token_kv", "shared_prefix"),
        ("whole_domain_full_sliding_token_kv", "shared_prefix"),
        ("whole_domain_sliding_token_kv", "request_private"),
        ("whole_domain_chunked_token_kv", "request_private"),
        ("whole_domain_full_latent_kv", "request_private"),
    ]
    assert profile["manager_supported_profiles"][4] == {
        "execution_topology": "whole_domain_full_latent_kv",
        "ordered_attention_classes": ["full:latent_kv"],
        "cache_policy": "request_private",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "kv_plan",
        "token_reclamation": "off",
        "fixed_state": "none",
    }
    assert profile["manager_supported_profiles"][3] == {
        "execution_topology": "whole_domain_chunked_token_kv",
        "ordered_attention_classes": ["chunked:token_kv"],
        "cache_policy": "request_private",
        "execution": "eager-non-overlap",
        "device_scope": "single-device",
        "plan_format": "retention_ir",
        "token_reclamation": "off",
        "fixed_state": "none",
    }
    profile["manager_supported_profiles"].reverse()
    with pytest.raises(RuntimeError, match="qualified RuntimeSession boundary"):
        verifier.validate_profile(profile, source_tree)


def test_supported_profiles_require_explicit_cache_policy(
    profile: dict, source_tree: Path
) -> None:
    profile["manager_supported_profiles"][0].pop("cache_policy")
    with pytest.raises(RuntimeError, match="cache_policy"):
        verifier.validate_profile(profile, source_tree)


def test_supported_profiles_reject_removed_structured_data_plane(
    profile: dict, source_tree: Path
) -> None:
    profile["manager_supported_profiles"][0]["structured_data_plane"] = "off"

    with pytest.raises(RuntimeError, match="structured_data_plane"):
        verifier.validate_profile(profile, source_tree)


def test_supported_profiles_reject_unknown_cache_policy(
    profile: dict, source_tree: Path
) -> None:
    profile["manager_supported_profiles"][0]["cache_policy"] = "unknown"
    with pytest.raises(RuntimeError, match="cache_policy is unsupported"):
        verifier.validate_profile(profile, source_tree)


def test_unsupported_profiles_are_manager_boundaries(
    profile: dict, source_tree: Path
) -> None:
    assert profile["manager_unsupported_profiles"] == [
        "whole_domain_full_token_kv_gdn_convolution",
        "whole_domain_full_token_kv_mamba",
    ]
    profile["manager_unsupported_profiles"].append("training")
    profile["manager_unsupported_profiles"].sort()
    with pytest.raises(RuntimeError, match="current RuntimeSession boundary"):
        verifier.validate_profile(profile, source_tree)


def test_takeover_categories_are_exact(
    profile: dict, source_tree: Path
) -> None:
    profile["manager_takeover_points"].pop()
    with pytest.raises(RuntimeError, match="manager_takeover_points"):
        verifier.validate_profile(profile, source_tree)


def test_category_cannot_mislabel_observability_as_takeover(
    profile: dict, source_tree: Path
) -> None:
    profile["manager_takeover_points"].append(
        profile["observability_extensions"][0]
    )
    with pytest.raises(RuntimeError, match="manager_takeover_points"):
        verifier.validate_profile(profile, source_tree)


def test_compatibility_fix_is_not_a_takeover(
    profile: dict, source_tree: Path
) -> None:
    assert profile["compatibility_fixes"] == [
        {
            "source_path": (
                "python/sglang/srt/layers/attention/flashattention_backend.py"
            ),
            "symbol": "make_local_attention_virtual_batches",
            "purpose": "preserve-page-id-addressing-for-local-attention",
        }
    ]
    verifier.validate_profile(profile, source_tree)


def test_overlay_symbol_must_dispatch_through_owner(
    profile: dict, source_tree: Path
) -> None:
    source = source_tree / "python/sglang/srt/mem_cache/allocation.py"
    source.write_text(
        _function("alloc_for_extend")
        + _direct_function("alloc_for_decode", "prepare_decode"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not dispatch through get_owner"):
        verifier.validate_profile(profile, source_tree)


def test_around_wrapper_must_name_preserved_native_method(
    profile: dict, source_tree: Path
) -> None:
    source = source_tree / "python/sglang/srt/managers/scheduler.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "Scheduler._orbitkv_native_run_batch, self", "Scheduler.wrong_native, self"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not dispatch through get_owner"):
        verifier.validate_profile(profile, source_tree)


@pytest.mark.parametrize(
    ("cleanup", "removal"),
    (
        (
            "self._prepare_waiting_request_removal(candidate_req)",
            "self.waiting_queue.pop(idx)",
        ),
        ("self._prepare_waiting_request_removal(req)", "self.waiting_queue.pop(index)"),
        ("self._prepare_waiting_request_removal(req)", "self.waiting_queue.pop(i)"),
    ),
)
def test_waiting_cleanup_must_precede_queue_removal(
    profile: dict, source_tree: Path, cleanup: str, removal: str
) -> None:
    source = source_tree / "python/sglang/srt/managers/scheduler.py"
    text = source.read_text(encoding="utf-8")
    old = f"        if {cleanup}:\n            {removal}"
    if old not in text:
        old = (
            f"        if not {cleanup}:\n"
            "            self._defer_waiting_request_removal(req, output)\n"
            f"        else:\n            {removal}"
        )
    assert old in text
    source.write_text(text.replace(old, f"        {removal}", 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="before queue removal|gate pop|cannot parse"):
        verifier.validate_profile(profile, source_tree)


def test_requires_a_git_checkout(profile: dict, source_tree: Path) -> None:
    nested = source_tree / "python"
    with pytest.raises(RuntimeError, match="complete Git checkout root"):
        verifier.validate_profile(profile, nested)


def test_non_git_source_directory_is_rejected(
    profile: dict, source_tree: Path, tmp_path_factory
) -> None:
    non_git = tmp_path_factory.mktemp("non-git-source")
    (non_git / "python/sglang").mkdir(parents=True)
    (non_git / "python/sglang/__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot verify pinned SGLang checkout"):
        verifier.validate_profile(profile, non_git)


def test_checkout_must_match_pinned_revision(
    profile: dict, source_tree: Path, fixture_source_contract: dict, monkeypatch
) -> None:
    contract = copy.deepcopy(fixture_source_contract)
    contract["revision"] = "0" * 40
    monkeypatch.setattr(verifier, "_load_source_contract", lambda _provider: contract)
    with pytest.raises(RuntimeError, match="checkout revision.*pinned_source_contract"):
        verifier.validate_profile(profile, source_tree)


def test_deleted_upstream_tracked_path_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    (source_tree / "README.md").unlink()
    with pytest.raises(RuntimeError, match="tracked path is not materialized"):
        verifier.validate_profile(profile, source_tree)


def test_renamed_upstream_tracked_path_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    (source_tree / "README.md").rename(source_tree / "RENAMED.md")
    with pytest.raises(RuntimeError, match="tracked path is not materialized"):
        verifier.validate_profile(profile, source_tree)


def test_tracked_path_type_change_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    source = source_tree / "README.md"
    source.unlink()
    source.mkdir()
    with pytest.raises(RuntimeError, match="type or mode changed"):
        verifier.validate_profile(profile, source_tree)


def test_tracked_executable_mode_change_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    executable = source_tree / "scripts/upstream-tool.sh"
    executable.chmod(executable.stat().st_mode & ~0o111)
    with pytest.raises(RuntimeError, match="type or mode changed"):
        verifier.validate_profile(profile, source_tree)


def test_non_overlay_tracked_edit_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    (source_tree / "README.md").write_text("trimmed product\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified tracked paths must equal"):
        verifier.validate_profile(profile, source_tree)


def test_missing_overlay_edit_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    target = profile["overlay_targets"][0]
    subprocess.run(
        ["git", "-C", str(source_tree), "checkout", "HEAD", "--", target],
        check=True,
        timeout=10,
    )
    with pytest.raises(RuntimeError, match="modified tracked paths must equal"):
        verifier.validate_profile(profile, source_tree)


def test_unexpected_untracked_file_is_rejected(
    profile: dict, source_tree: Path
) -> None:
    extra = source_tree / "unreviewed-product-file.txt"
    extra.write_text("unexpected = True\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected untracked paths"):
        verifier.validate_profile(profile, source_tree)


def test_git_ignored_file_is_allowed(
    profile: dict, source_tree: Path
) -> None:
    ignore = source_tree / ".git/info/exclude"
    ignore.write_text("*.pyc\n", encoding="utf-8")
    artifact = source_tree / "python/sglang/cache.pyc"
    artifact.write_bytes(b"ignored")
    verifier.validate_profile(profile, source_tree)


def test_overlay_target_symlink_is_rejected(
    profile: dict, source_tree: Path, tmp_path_factory
) -> None:
    source = source_tree / "python/sglang/srt/mem_cache/allocation.py"
    external_root = tmp_path_factory.mktemp("external")
    external = external_root / "allocation.py"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(RuntimeError, match="type or mode changed"):
        verifier.validate_profile(profile, source_tree)


def test_source_contract_schema_is_checked(
    profile: dict, source_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verifier, "_load_source_contract", lambda _provider: {"targets": []})
    with pytest.raises(RuntimeError, match="pinned_source_contract.*invalid schema"):
        verifier.validate_profile(profile, source_tree)


def test_overlay_target_list_must_equal_source_contract(
    profile: dict, source_tree: Path
) -> None:
    profile["overlay_targets"].pop()
    with pytest.raises(RuntimeError, match="exactly equal"):
        verifier.validate_profile(profile, source_tree)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("upstream_license", "unknown"),
        ("license_action", "omit"),
        ("notice_action_when_present", "omit"),
        ("notice_action_when_absent", "synthesize"),
    ),
)
def test_license_and_notice_policy_is_fixed(
    profile: dict, source_tree: Path, field: str, value: str
) -> None:
    profile["license_notice_policy"][field] = value
    with pytest.raises(RuntimeError, match="license_notice_policy"):
        verifier.validate_profile(profile, source_tree)


def test_verifier_is_read_only(profile: dict, source_tree: Path) -> None:
    before_status = _git(source_tree, "status", "--porcelain=v1", "--untracked-files=all")
    verifier.validate_profile(copy.deepcopy(profile), source_tree)
    after_status = _git(source_tree, "status", "--porcelain=v1", "--untracked-files=all")
    assert after_status == before_status


def test_cli_prints_full_overlay_summary(
    profile: dict, source_tree: Path, tmp_path_factory, capsys
) -> None:
    output_root = tmp_path_factory.mktemp("profile")
    profile_path = output_root / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    assert verifier.main(["--profile", str(profile_path), "--sglang-root", str(source_tree)]) == 0
    output = capsys.readouterr().out
    assert "product=orbitkv-engine" in output
    assert "overlay_targets=6" in output
    assert "integration_points=10" in output
    assert "compatibility_fixes=1" in output


def test_profile_loader_rejects_duplicate_keys(tmp_path_factory) -> None:
    root = tmp_path_factory.mktemp("duplicate")
    profile_path = root / "profile.json"
    profile_path.write_text('{"schema": "first", "schema": "second"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        verifier.load_profile(profile_path)
