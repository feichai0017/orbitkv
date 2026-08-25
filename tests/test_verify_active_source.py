from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


sys.dont_write_bytecode = True
MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/verify_active_source.py"
)
SPEC = importlib.util.spec_from_file_location("verify_active_source", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write_lines(path: Path, count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("line\n" * count, encoding="utf-8")


@pytest.fixture
def source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    return tmp_path


def test_discovers_top_level_sglang_and_tool_scripts(
    source_root: Path,
) -> None:
    expected = {
        source_root / "integrations/sglang/new_qualifier.py",
        source_root / "tools/new_verifier.py",
    }
    for path in expected:
        write_lines(path)

    failures: list[str] = []
    counts = verifier.check_source_sizes(failures)

    assert set(verifier.active_script_files()) == expected
    assert counts == (0, 0, 2)
    assert failures == []


def test_active_script_line_limit_is_enforced(source_root: Path) -> None:
    oversized = source_root / "tools/oversized.py"
    write_lines(oversized, verifier.ACTIVE_SCRIPT_LIMIT + 1)

    failures: list[str] = []
    assert verifier.check_source_sizes(failures) == (0, 0, 1)

    assert failures == [
        "tools/oversized.py: "
        f"{verifier.ACTIVE_SCRIPT_LIMIT + 1} lines > "
        f"{verifier.ACTIVE_SCRIPT_LIMIT}"
    ]


def test_missing_canonical_generic_entrypoint_fails(source_root: Path) -> None:
    missing = verifier.CANONICAL_GENERIC_ENTRYPOINTS[-1]
    for relative_path in verifier.CANONICAL_GENERIC_ENTRYPOINTS[:-1]:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            verifier.GENERIC_ENTRYPOINT_MARKERS[relative_path] + "\n",
            encoding="utf-8",
        )
    for relative_path in verifier.LEGACY_COMPATIBILITY_ENTRYPOINTS:
        write_lines(source_root / relative_path)
    for relative_path, marker in verifier.ABI8_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=re.escape(f"missing required qualification entrypoint: {missing}"),
    ):
        verifier.main()


def test_invalid_canonical_generic_entrypoint_fails(source_root: Path) -> None:
    for relative_path in verifier.CANONICAL_GENERIC_ENTRYPOINTS:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            verifier.GENERIC_ENTRYPOINT_MARKERS[relative_path] + "\n",
            encoding="utf-8",
        )
    for relative_path, marker in verifier.ABI8_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")

    invalid = verifier.CANONICAL_GENERIC_ENTRYPOINTS[0]
    (source_root / invalid).write_text("pass\n", encoding="utf-8")
    failures: list[str] = []
    verifier.check_generic_entrypoints(failures)

    assert f"invalid canonical generic entrypoint: {invalid}" in failures


def test_excludes_results_targets_and_cache_directories(
    source_root: Path,
) -> None:
    live = source_root / "src/live.py"
    write_lines(live)
    for relative_path in (
        "src/results/frozen.py",
        "src/target/generated.rs",
        "src/__pycache__/generated.py",
        "src/.pytest_cache/generated.py",
        "results/qualification/source/tools/frozen.py",
        "tools/results/frozen.py",
        "tools/__pycache__/generated.py",
    ):
        write_lines(source_root / relative_path)

    assert verifier.source_files(source_root / "src") == [live]
    assert verifier.active_script_files() == []


def test_overlapping_roots_count_and_check_each_file_once(
    source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = source_root / "tools/shared.py"
    write_lines(shared)
    monkeypatch.setattr(verifier, "PRODUCTION_ROOTS", (Path("tools"),))
    monkeypatch.setattr(verifier, "TEST_BENCH_ROOTS", (Path("tools"),))
    monkeypatch.setattr(
        verifier, "ACTIVE_SCRIPT_ROOTS", (Path("tools"), Path("tools"))
    )
    checked: list[Path] = []
    original = verifier.check_line_limit

    def record_check(path: Path, limit: int, failures: list[str]) -> None:
        checked.append(path)
        original(path, limit, failures)

    monkeypatch.setattr(verifier, "check_line_limit", record_check)
    failures: list[str] = []

    assert verifier.check_source_sizes(failures) == (0, 0, 1)
    assert checked == [shared]
    assert failures == []


def test_main_reports_separate_category_counts(
    source_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for relative_path in verifier.CANONICAL_GENERIC_ENTRYPOINTS:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            verifier.GENERIC_ENTRYPOINT_MARKERS[relative_path] + "\n",
            encoding="utf-8",
        )
    for relative_path in verifier.LEGACY_COMPATIBILITY_ENTRYPOINTS:
        write_lines(source_root / relative_path)
    for relative_path, marker in verifier.ABI8_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")
    write_lines(source_root / "tests/test_gate.py")

    verifier.main()

    output = capsys.readouterr().out
    assert "2 production files" in output
    assert "1 test/bench files" in output
    assert "9 active scripts" in output
    assert "generic and compatibility entrypoints present" in output
