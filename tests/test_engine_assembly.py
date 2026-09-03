from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


sys.dont_write_bytecode = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "compat/sglang/assemble.py"
SPEC = importlib.util.spec_from_file_location("engine_assemble", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    root = tmp_path / "source"
    sources = {
        "LICENSE": "Apache License\nVersion 2.0, January 2004\n",
        "README.md": "complete upstream source\n",
        "python/pyproject.toml": "[project]\nname = 'sglang'\nversion = '0'\n",
        "python/sglang/__init__.py": "VALUE = 'base'\n",
        "scripts/run.sh": "#!/bin/sh\nexit 0\n",
    }
    for relative, content in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    executable = root / "scripts/run.sh"
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    (root / "README-link").symlink_to("README.md")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "OrbitKV fixture")
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "pinned source")
    revision = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    patch = tmp_path / "overlay.patch"
    (root / "python/sglang/__init__.py").write_text(
        "VALUE = 'patched'\n", encoding="utf-8"
    )
    patch.write_bytes(
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--full-index",
                "--binary",
                "--unified=3",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "--",
                "python/sglang/__init__.py",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    )
    _git(root, "checkout", "--", "python/sglang/__init__.py")
    contract = {
        "revision": revision,
        "patch_path": patch.relative_to(tmp_path).as_posix(),
        "patch_diff_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "targets": [
            {
                "path": "python/sglang/__init__.py",
                "patched_sha256": hashlib.sha256(
                    b"VALUE = 'patched'\n"
                ).hexdigest(),
            }
        ],
    }

    def validate_base(candidate: Path | str) -> Path:
        candidate = Path(candidate).resolve(strict=True)
        if _git(candidate, "rev-parse", "HEAD") != revision:
            raise RuntimeError("wrong revision")
        if _git(candidate, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError("not pristine")
        return candidate

    def validate_patched(candidate: Path | str) -> Path:
        candidate = Path(candidate).resolve(strict=True)
        if _git(candidate, "rev-parse", "HEAD") != revision:
            raise RuntimeError("wrong revision")
        if (candidate / "python/sglang/__init__.py").read_text() != "VALUE = 'patched'\n":
            raise RuntimeError("not reviewed patch")
        return candidate

    monkeypatch.setattr(assembly, "ROOT", tmp_path)
    product = tmp_path / "compat/sglang"
    product.mkdir(parents=True)
    profile_payload = b'{"fixture": "engine-product-profile"}\n'
    (product / "profile.json").write_bytes(profile_payload)
    manager = tmp_path / "manager"
    (manager / "src").mkdir(parents=True)
    (manager / "Cargo.toml").write_text(
        "[package]\nname = 'fixture-manager'\nversion = '0.0.0'\nedition = '2021'\n",
        encoding="utf-8",
    )
    (manager / "Cargo.lock").write_text("version = 3\n", encoding="utf-8")
    (manager / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    (manager / "src/lib.rs").write_text("pub fn fixture() {}\n", encoding="utf-8")
    monkeypatch.setattr(assembly, "MANAGER_PROJECT", manager)
    monkeypatch.setattr(assembly, "ROOT", tmp_path)
    monkeypatch.setattr(
        assembly, "MANAGER_INCLUDED", ("Cargo.lock", "Cargo.toml", "LICENSE", "src")
    )
    monkeypatch.setattr(assembly, "pinned_source_contract", lambda: dict(contract))
    monkeypatch.setattr(assembly, "validate_base_checkout", validate_base)
    monkeypatch.setattr(assembly, "validate_patched_checkout", validate_patched)
    monkeypatch.setattr(assembly, "_verify_product_profile", lambda _root: None)
    return root, {
        "contract": contract,
        "profile_payload": profile_payload,
        "tree": tree,
    }


def _install_adapter_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    integration = tmp_path / "compat/sglang/bridge"
    package = integration / "src/orbitkv_sglang"
    package.mkdir(parents=True)
    (integration / "pyproject.toml").write_text(
        "[project]\nname = 'orbitkv-sglang'\nversion = '0'\ndependencies = []\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("VALUE = 'adapter'\n", encoding="utf-8")
    monkeypatch.setattr(assembly, "INTEGRATION_SOURCE", integration / "src")


def test_assemble_preserves_every_tracked_path_and_adds_components(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = tmp_path / "product"

    assert assembly.assemble(source, output) == output
    tracked = set(_git(source, "ls-files").splitlines())
    assert all((output / relative).exists() or (output / relative).is_symlink() for relative in tracked)
    assert (output / "README-link").is_symlink()
    assert os.readlink(output / "README-link") == "README.md"
    assert os.access(output / "scripts/run.sh", os.X_OK)
    assert (output / "python/sglang/__init__.py").read_text() == "VALUE = 'patched'\n"
    assert (output / "orbitkv/adapter/src/orbitkv_sglang/__init__.py").is_file()
    assert (output / "orbitkv/profile.json").read_bytes() == identity[
        "profile_payload"
    ]
    assert (output / "orbitkv/manager/Cargo.toml").is_file()
    assert (output / "orbitkv/manager/src/lib.rs").is_file()
    assert not (output / "orbitkv/runtime").exists()

    manifest = json.loads((output / assembly.MANIFEST_NAME).read_text())
    assert manifest["source_revision"] == identity["contract"]["revision"]
    assert manifest["source_tree"] == identity["tree"]
    assert manifest["reviewed_patch_sha256"] == identity["contract"]["patch_diff_sha256"]
    assert manifest["components"]["sglang"]["file_count"] == len(tracked)
    assert set(manifest["components"]) == {
        "sglang",
        "orbitkv-product",
        "orbitkv-adapter",
        "orbitkv-manager",
    }
    profile_digest = hashlib.sha256(identity["profile_payload"]).hexdigest()
    assert manifest["components"]["orbitkv-product"] == {
        "root": "orbitkv",
        "source": manifest["orbitkv_source"],
        "file_count": 1,
        "inventory_sha256": assembly._inventory_digest(
            [
                {
                    "path": "profile.json",
                    "kind": "file",
                    "sha256": profile_digest,
                }
            ]
        ),
        "files": [
            {
                "path": "profile.json",
                "kind": "file",
                "sha256": profile_digest,
            }
        ],
    }
    assert manifest["components"]["orbitkv-manager"]["file_count"] == 4
    assert manifest["schema_version"] == 2
    assert manifest["components"]["orbitkv-manager"]["files"] == sorted(
        manifest["components"]["orbitkv-manager"]["files"],
        key=lambda entry: entry["path"],
    )
    assert manifest["file_count"] == sum(
        component["file_count"] for component in manifest["components"].values()
    )
    assert assembly.verify_assembly(output) == output
    metadata = subprocess.run(
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(output / "orbitkv/manager/Cargo.toml"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert {
        item["name"] for item in json.loads(metadata.stdout)["packages"]
    } == {"fixture-manager"}


def test_assemble_accepts_reviewed_patched_source_without_reapplying(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    (source / "python/sglang/__init__.py").write_text(
        "VALUE = 'patched'\n", encoding="utf-8"
    )
    output = tmp_path / "product"

    assembly.assemble(source, output)

    assert (output / "python/sglang/__init__.py").read_text() == "VALUE = 'patched'\n"


def test_assemble_rejects_missing_product_profile(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    (tmp_path / "compat/sglang/profile.json").unlink()
    output = tmp_path / "product"

    with pytest.raises(RuntimeError, match="component source is missing.*profile.json"):
        assembly.assemble(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".product.assembling-*"))


def test_assemble_rejects_wrong_revision_or_source_mutation(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    _git(source, "commit", "--allow-empty", "-qm", "wrong revision")
    with pytest.raises(RuntimeError, match="neither the exact pinned"):
        assembly.assemble(source, tmp_path / "wrong-revision")

    subprocess.run(
        ["git", "-C", str(source), "checkout", "-q", "HEAD^"],
        check=True,
        capture_output=True,
        timeout=10,
    )
    (source / "README.md").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="neither the exact pinned"):
        assembly.assemble(source, tmp_path / "mutated")


def test_assemble_rejects_existing_output_without_modifying_it(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("unchanged\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output already exists"):
        assembly.assemble(source, output)

    assert sentinel.read_text() == "unchanged\n"


def test_assemble_rejects_output_inside_source_checkout(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="outside the source checkout"):
        assembly.assemble(source, source / "product")

    assert not (source / "product").exists()


def test_failed_assembly_removes_only_its_private_temporary_directory(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        assembly,
        "_copy_component",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    output = tmp_path / "failed-product"

    with pytest.raises(RuntimeError, match="injected"):
        assembly.assemble(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-product.assembling-*"))


def test_real_manager_source_closure_has_valid_path_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "manager"
    paths = assembly._copy_manager_component(REPOSITORY_ROOT / "core", root)

    assert "Cargo.lock" in paths
    assert (root / "Cargo.lock").read_bytes() == (
        REPOSITORY_ROOT / "Cargo.lock"
    ).read_bytes()
    assert "Cargo.toml" in paths
    assert "src/runtime_session.rs" in paths
    assert "ffi/include/orbitkv.h" in paths
    assert "ffi/src/lib.rs" in paths
    assert "fixtures/hybrid-fixed-state-small/config.json" in paths
    assert "fixtures/hybrid-fixed-state-large/config.json" in paths
    assert "fixtures/hybrid-fixed-state-large/PROVENANCE.md" in paths
    metadata = subprocess.run(
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(root / "Cargo.toml"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    packages = {item["name"] for item in json.loads(metadata.stdout)["packages"]}
    assert "orbitkv" in packages
    ffi_metadata = subprocess.run(
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(root / "ffi/Cargo.toml"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ffi_packages = {
        item["name"] for item in json.loads(ffi_metadata.stdout)["packages"]
    }
    assert "orbitkv-ffi" in ffi_packages


def test_real_product_profile_is_the_only_product_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "orbitkv"

    paths = assembly._copy_component(
        REPOSITORY_ROOT / "compat/sglang", destination, assembly.PRODUCT_INCLUDED
    )

    assert paths == ("profile.json",)
    assert (destination / "profile.json").read_bytes() == (
        REPOSITORY_ROOT / "compat/sglang/profile.json"
    ).read_bytes()


def test_patched_input_closure_is_rechecked_when_validator_is_weak(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    (source / "python/sglang/__init__.py").write_text(
        "VALUE = 'patched'\n", encoding="utf-8"
    )
    (source / "README.md").write_text("unreviewed edit\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly match overlay targets"):
        assembly.assemble(source, tmp_path / "product")


def test_contract_target_must_be_canonically_contained(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    contract = dict(identity["contract"])
    contract["targets"] = [
        {
            "path": "python/sglang/../../../outside",
            "patched_sha256": "0" * 64,
        }
    ]
    monkeypatch.setattr(assembly, "pinned_source_contract", lambda: contract)

    with pytest.raises(RuntimeError, match="target path is unsafe"):
        assembly.assemble(source, tmp_path / "product")


def test_component_copy_rejects_symlink_race_after_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "component"
    source.mkdir()
    candidate = source / "payload"
    candidate.write_text("safe\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("secret\n", encoding="utf-8")
    original = assembly._component_files

    def raced(root: Path, included: tuple[str, ...]) -> tuple[str, ...]:
        files = original(root, included)
        candidate.unlink()
        candidate.symlink_to(outside)
        return files

    monkeypatch.setattr(assembly, "_component_files", raced)

    with pytest.raises(RuntimeError, match="contained regular file"):
        assembly._copy_component(source, tmp_path / "output", ("payload",))


def test_upstream_copy_rejects_regular_file_symlink_race(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    outside = tmp_path / "outside"
    outside.write_text("secret\n", encoding="utf-8")
    original = assembly._tracked_modes

    def raced(root: Path) -> dict[str, str]:
        modes = original(root)
        victim = root / "README.md"
        victim.unlink()
        victim.symlink_to(outside)
        return modes

    monkeypatch.setattr(assembly, "_tracked_modes", raced)

    with pytest.raises(RuntimeError, match="contained regular file"):
        assembly._copy_upstream(source, tmp_path / "output")


def test_upstream_symlink_must_not_escape_source(
    source_tree: tuple[Path, dict], tmp_path: Path
) -> None:
    source, _identity = source_tree
    link = source / "README-link"
    link.unlink()
    link.symlink_to("../outside")

    with pytest.raises(RuntimeError, match="target escapes"):
        assembly._copy_upstream(source, tmp_path / "output")


def test_publish_does_not_replace_concurrent_destination(tmp_path: Path) -> None:
    temporary = tmp_path / "temporary"
    destination = tmp_path / "product"
    temporary.mkdir()
    destination.mkdir()
    (temporary / "ours").write_text("ours\n", encoding="utf-8")
    sentinel = destination / "theirs"
    sentinel.write_text("theirs\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output already exists"):
        assembly._publish_no_replace(temporary, destination)

    assert temporary.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "theirs\n"


@pytest.mark.parametrize("mutation", ["content", "mode", "remove", "add"])
def test_verify_assembly_rejects_tampering(
    source_tree: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = assembly.assemble(source, tmp_path / "product")
    target = output / "orbitkv/manager/src/lib.rs"
    if mutation == "content":
        target.write_text("tampered\n", encoding="utf-8")
    elif mutation == "mode":
        target.chmod(0o755)
    elif mutation == "remove":
        target.unlink()
    else:
        (output / "unexpected").write_text("extra\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid assembly manifest"):
        assembly.verify_assembly(output)


def test_verify_assembly_rejects_manifest_inventory_tampering(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = assembly.assemble(source, tmp_path / "product")
    manifest_path = output / assembly.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"]["orbitkv-manager"]["files"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid assembly manifest"):
        assembly.verify_assembly(output)


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_verify_assembly_rejects_missing_or_tampered_product_profile(
    source_tree: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = assembly.assemble(source, tmp_path / "product")
    profile = output / "orbitkv/profile.json"
    if mutation == "missing":
        profile.unlink()
    else:
        profile.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="component orbitkv-product file.*profile.json"
    ):
        assembly.verify_assembly(output)


def test_failed_publish_cleanup_never_removes_concurrent_destination(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = tmp_path / "product"

    real_publish = assembly._publish_no_replace

    def concurrent_publish(temporary: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "theirs").write_text("theirs\n", encoding="utf-8")
        real_publish(temporary, destination)

    monkeypatch.setattr(assembly, "_publish_no_replace", concurrent_publish)

    with pytest.raises(RuntimeError, match="output already exists"):
        assembly.assemble(source, output)

    assert (output / "theirs").read_text(encoding="utf-8") == "theirs\n"
    assert not list(tmp_path.glob(".product.assembling-*"))


def test_verify_cli_accepts_intact_product(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = assembly.assemble(source, tmp_path / "product")

    assert assembly.main(["verify", "--output", str(output)]) == 0


def test_legacy_assemble_cli_shape_remains_valid(
    source_tree: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _identity = source_tree
    _install_adapter_fixture(tmp_path, monkeypatch)
    output = tmp_path / "product"

    assert (
        assembly.main(
            ["--sglang-root", str(source), "--output", str(output)]
        )
        == 0
    )
    assert assembly.verify_assembly(output) == output
