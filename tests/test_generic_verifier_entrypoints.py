from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
ENTRYPOINTS = (
    "verify_token_relocation_evidence.py",
    "verify_token_relocation_seal.py",
    "verify_fixed_state_pair_evidence.py",
)


def _load(filename: str) -> ModuleType:
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(
        f"generic_entrypoint_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relocation_evidence = _load("verify_token_relocation_evidence.py")
relocation_seal = _load("verify_token_relocation_seal.py")
fixed_state = _load("verify_fixed_state_pair_evidence.py")


def test_relocation_evidence_api_forwards_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp/token-relocation-evidence")
    source_root = Path("/tmp/adapter-source")
    adapter = {"files": [{"path": "adapter.py"}]}
    expected = {"status": "passed"}
    calls: list[tuple[Path, dict[str, object]]] = []

    def verify(received: Path, **kwargs: object) -> dict[str, object]:
        calls.append((received, kwargs))
        return expected

    monkeypatch.setattr(relocation_evidence._legacy, "verify_evidence", verify)

    result = relocation_evidence.verify_evidence(
        root,
        expected_harness_sha256="a" * 64,
        expected_adapter=adapter,
        expected_adapter_source_root=source_root,
    )

    assert result is expected
    assert calls == [
        (
            root,
            {
                "expected_harness_sha256": "a" * 64,
                "expected_adapter": adapter,
                "expected_adapter_source_root": source_root,
            },
        )
    ]


def test_relocation_evidence_sealed_api_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp/token-relocation-seal")
    expected = {"sealed": True}
    calls: list[Path] = []

    monkeypatch.setattr(
        relocation_evidence._legacy,
        "verify_sealed_archive",
        lambda received: calls.append(received) or expected,
    )
    assert relocation_evidence.verify_sealed_archive(root) is expected
    assert calls == [root]


def test_relocation_evidence_main_prints_json_and_reports_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path("/tmp/token-relocation-evidence")
    expected = {"status": "passed"}
    calls: list[Path] = []
    monkeypatch.setattr(
        relocation_evidence._legacy,
        "verify_evidence",
        lambda received, **_kwargs: calls.append(received) or expected,
    )

    assert relocation_evidence.main([str(root)]) == 0
    output = capsys.readouterr()
    assert calls == [root]
    assert json.loads(output.out) == expected
    assert output.err == ""

    def reject(_received: Path, **_kwargs: object) -> None:
        raise RuntimeError("invalid relocation evidence")

    monkeypatch.setattr(
        relocation_evidence._legacy, "verify_evidence", reject
    )
    with pytest.raises(SystemExit) as raised:
        relocation_evidence.main([str(root)])

    output = capsys.readouterr()
    assert raised.value.code == 1
    assert output.out == ""
    assert output.err == "error: invalid relocation evidence\n"


def test_relocation_seal_main_uses_path_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["/tmp/custom-relocation-seal"]
    expected_path = Path(argv[0])
    expected = {"status": "passed", "sealed": True}
    calls: list[Path] = []
    monkeypatch.setattr(
        relocation_seal._legacy,
        "verify_sealed_archive",
        lambda received: calls.append(received) or expected,
    )

    assert relocation_seal.main(argv) == 0

    output = capsys.readouterr()
    assert calls == [expected_path]
    assert json.loads(output.out) == expected
    assert output.err == ""


def test_relocation_seal_api_forwards_and_cli_reports_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path("/tmp/token-relocation-seal")
    expected = {"sealed": True}
    calls: list[Path] = []
    monkeypatch.setattr(
        relocation_seal._legacy,
        "verify_sealed_archive",
        lambda received: calls.append(received) or expected,
    )

    assert relocation_seal.verify_sealed_archive(root) is expected
    assert calls == [root]

    def reject(_received: Path) -> None:
        raise RuntimeError("invalid sealed archive")

    monkeypatch.setattr(
        relocation_seal._legacy, "verify_sealed_archive", reject
    )
    with pytest.raises(SystemExit) as raised:
        relocation_seal.main([str(root)])

    output = capsys.readouterr()
    assert raised.value.code == 1
    assert output.out == ""
    assert output.err == "error: invalid sealed archive\n"


def test_fixed_state_api_forwards_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = Path("/tmp/fixed-state-evidence")
    expected = {"status": "passed"}
    calls: list[Path] = []
    monkeypatch.setattr(
        fixed_state._legacy,
        "verify_archive",
        lambda received: calls.append(received) or expected,
    )

    assert fixed_state.verify_archive(explicit) is expected
    assert calls == [explicit]


def test_fixed_state_main_uses_path_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["/tmp/custom-fixed-state"]
    expected_path = Path(argv[0])
    expected = {"status": "passed", "qualified": False}
    calls: list[Path] = []
    monkeypatch.setattr(
        fixed_state._legacy,
        "verify_archive",
        lambda received: calls.append(received) or expected,
    )

    assert fixed_state.main(argv) == 0

    output = capsys.readouterr()
    assert calls == [expected_path]
    assert json.loads(output.out) == expected
    assert output.err == ""


def test_fixed_state_main_reports_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject(_received: Path) -> None:
        raise RuntimeError("invalid fixed-state evidence")

    monkeypatch.setattr(fixed_state._legacy, "verify_archive", reject)
    with pytest.raises(SystemExit) as raised:
        fixed_state.main(["/tmp/fixed-state-evidence"])

    output = capsys.readouterr()
    assert raised.value.code == 1
    assert output.out == ""
    assert output.err == "error: invalid fixed-state evidence\n"


@pytest.mark.parametrize(
    "module", (relocation_seal, fixed_state), ids=("relocation", "fixed-state")
)
def test_archive_verifier_cli_requires_an_explicit_path(module: ModuleType) -> None:
    with pytest.raises(SystemExit) as raised:
        module.main([])

    assert raised.value.code == 2


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_help_runs_without_site_packages(filename: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-S", str(TOOLS / filename), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "usage:" in completed.stdout
    assert completed.stderr == ""
    assert all(
        marker not in completed.stdout.lower()
        for marker in ("h20", "qwen", "gpt-oss", "v0.5.17", "v0517")
    )


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_import_runs_without_site_packages(filename: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    path = TOOLS / filename
    code = (
        "import importlib.util; "
        f"path = {str(path)!r}; "
        "spec = importlib.util.spec_from_file_location('facade', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "assert callable(module.main)"
    )
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
