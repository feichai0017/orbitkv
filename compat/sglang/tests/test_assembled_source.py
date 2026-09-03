from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "bridge/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from orbitkv_sglang import pinned  # noqa: E402


ASSEMBLED = Path("/tmp/orbitkv-engine-wire14-final")


def test_current_assembled_product_is_runtime_valid_without_git() -> None:
    if not ASSEMBLED.is_dir():
        pytest.skip("current assembled product is unavailable")
    assert not (ASSEMBLED / ".git").exists()
    assert pinned.validate_patched_source(ASSEMBLED) == ASSEMBLED.resolve()


def test_assembled_tampering_fails_without_git_fallback(tmp_path: Path) -> None:
    if not ASSEMBLED.is_dir():
        pytest.skip("current assembled product is unavailable")
    product = tmp_path / "product"
    shutil.copytree(ASSEMBLED, product, symlinks=True)
    target = product / "orbitkv/profile.json"
    target.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory differs"):
        pinned.validate_patched_source(product)


def test_present_invalid_assembly_manifest_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    (root / "python/sglang").mkdir(parents=True)
    (root / "python/sglang/__init__.py").write_text("", encoding="utf-8")
    (root / "orbitkv-engine-manifest.json").write_text(
        json.dumps({"schema": "wrong"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        pinned,
        "validate_patched_checkout",
        lambda _root: pytest.fail("invalid assembly must not fall back to Git"),
    )
    with pytest.raises(RuntimeError, match="contract differs"):
        pinned.validate_patched_source(root)


def test_manifest_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "python/sglang").mkdir(parents=True)
    (root / "python/sglang/__init__.py").write_text("", encoding="utf-8")
    target = tmp_path / "manifest.json"
    target.write_text("{}", encoding="utf-8")
    (root / "orbitkv-engine-manifest.json").symlink_to(target)
    with pytest.raises(RuntimeError, match="regular file"):
        pinned.validate_patched_source(root)


def test_retired_runtime_component_is_rejected(tmp_path: Path) -> None:
    if not ASSEMBLED.is_dir():
        pytest.skip("current assembled product is unavailable")
    product = tmp_path / "product"
    shutil.copytree(ASSEMBLED, product, symlinks=True)
    manifest_path = product / "orbitkv-engine-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"]["orbitkv-runtime"] = {
        "root": "orbitkv/runtime",
        "source": manifest["orbitkv_source"],
        "file_count": 1,
        "inventory_sha256": "0" * 64,
        "files": [
            {"path": "placeholder", "kind": "file", "sha256": "0" * 64}
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="component set differs"):
        pinned.validate_patched_source(product)
