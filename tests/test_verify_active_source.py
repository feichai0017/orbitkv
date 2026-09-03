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
        source_root / "compat/sglang/tools/new_qualifier.py",
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


def test_retired_python_product_roots_are_not_scanned(source_root: Path) -> None:
    for relative_path in (
        "core/python/src/removed.py",
        "core/python/tests/test_removed.py",
        "core/reference/src/removed.py",
        "core/reference/tests/test_removed.py",
    ):
        write_lines(source_root / relative_path)

    failures: list[str] = []
    assert verifier.check_source_sizes(failures) == (0, 0, 0)
    assert failures == []


def test_missing_current_qualification_entrypoint_fails(source_root: Path) -> None:
    missing = tuple(verifier.CURRENT_ENTRYPOINT_MARKERS)[-1]
    for relative_path, marker in list(
        verifier.CURRENT_ENTRYPOINT_MARKERS.items()
    )[:-1]:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")
    for relative_path, markers in verifier.WIRE_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        values = (markers,) if isinstance(markers, str) else markers
        path.write_text("\n".join(values) + "\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=re.escape(f"missing required qualification entrypoint: {missing}"),
    ):
        verifier.main()


def test_invalid_current_qualification_entrypoint_fails(source_root: Path) -> None:
    for relative_path, marker in verifier.CURRENT_ENTRYPOINT_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")

    invalid = tuple(verifier.CURRENT_ENTRYPOINT_MARKERS)[0]
    (source_root / invalid).write_text("pass\n", encoding="utf-8")
    failures: list[str] = []
    verifier.check_current_entrypoints(failures)

    assert f"invalid current qualification entrypoint: {invalid}" in failures


def test_wire_marker_gate_rejects_stale_surface(source_root: Path) -> None:
    for relative_path, markers in verifier.WIRE_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        values = (markers,) if isinstance(markers, str) else markers
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    for relative_path, markers in verifier.WIRE_POLICY_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as target:
            target.write("\n".join(markers) + "\n")

    stale_path = Path("core/ffi/include/orbitkv.h")
    source = (source_root / stale_path).read_text(encoding="utf-8")
    (source_root / stale_path).write_text(
        source.replace(
            verifier.WIRE_MARKERS[stale_path],
            "#define ORBITKV_WIRE_VERSION 8u",
        ),
        encoding="utf-8",
    )
    failures: list[str] = []
    verifier.check_wire_markers(failures)

    assert failures == [
        f"{stale_path}: missing exact wire marker "
        f"{verifier.WIRE_MARKERS[stale_path]!r}"
    ]


def test_wire_marker_gate_rejects_missing_explicit_cache_policy(
    source_root: Path,
) -> None:
    for relative_path, markers in verifier.WIRE_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        values = (markers,) if isinstance(markers, str) else markers
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    for relative_path, markers in verifier.WIRE_POLICY_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as target:
            target.write("\n".join(markers) + "\n")

    stale_path = Path("core/ffi/include/orbitkv.h")
    source = (source_root / stale_path).read_text(encoding="utf-8")
    missing = verifier.WIRE_POLICY_MARKERS[stale_path][1]
    (source_root / stale_path).write_text(
        source.replace(missing, "REMOVED_POLICY"), encoding="utf-8"
    )
    failures: list[str] = []
    verifier.check_wire_markers(failures)

    assert failures == [
        f"{stale_path}: missing exact cache-policy marker {missing!r}"
    ]


def test_wire_marker_gate_rejects_rust_core_policy_value_drift(
    source_root: Path,
) -> None:
    for relative_path, markers in verifier.WIRE_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        values = (markers,) if isinstance(markers, str) else markers
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    for relative_path, markers in verifier.WIRE_POLICY_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as target:
            target.write("\n".join(markers) + "\n")

    stale_path = Path("core/src/runtime_session.rs")
    source = (source_root / stale_path).read_text(encoding="utf-8")
    missing = verifier.WIRE_POLICY_MARKERS[stale_path][1]
    (source_root / stale_path).write_text(
        source.replace(missing, "RequestPrivate = 7"), encoding="utf-8"
    )
    failures: list[str] = []
    verifier.check_wire_markers(failures)

    assert failures == [
        f"{stale_path}: missing exact cache-policy marker {missing!r}"
    ]


def test_canonical_api_gate_rejects_missing_marker_and_retired_spelling(
    source_root: Path,
) -> None:
    for relative_path, marker in verifier.CANONICAL_API_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")

    missing = tuple(verifier.CANONICAL_API_MARKERS)[0]
    (source_root / missing).write_text("class ManagerPlanConfig:\n    pass\n")
    scalar = source_root / "compat/sglang/bridge/src/runtime.py"
    scalar.parent.mkdir(parents=True, exist_ok=True)
    scalar.write_text(
        "def relocate_tokens(self):\n    pass\n", encoding="utf-8"
    )

    failures: list[str] = []
    verifier.check_canonical_api(failures)

    assert any("missing canonical API marker" in item for item in failures)
    assert any("retired production spelling ManagerPlanConfig" in item for item in failures)
    assert any("retired scalar method relocate_tokens" in item for item in failures)


@pytest.mark.parametrize(
    "legacy_relative", verifier.REMOVED_PRODUCTION_PATHS
)
def test_canonical_api_gate_rejects_retired_production_paths(
    source_root: Path,
    legacy_relative: Path,
) -> None:
    for relative_path, marker in verifier.CANONICAL_API_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")
    legacy = source_root / legacy_relative
    if legacy.suffix:
        write_lines(legacy)
    else:
        legacy.mkdir(parents=True)

    failures: list[str] = []
    verifier.check_canonical_api(failures)

    assert failures == [
        f"retired production path remains: {legacy_relative}"
    ]


def test_canonical_api_gate_ignores_archives_and_negative_tests(
    source_root: Path,
) -> None:
    for relative_path, marker in verifier.CANONICAL_API_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")
    write_lines(source_root / "results/archive/executor_capabilities.py")
    negative = source_root / "compat/sglang/tests/test_removed.py"
    negative.parent.mkdir(parents=True, exist_ok=True)
    negative.write_text(
        "assert 'ORBITKV_PLAN' not in environ\n", encoding="utf-8"
    )

    failures: list[str] = []
    verifier.check_canonical_api(failures)
    assert failures == []


@pytest.mark.parametrize(
    "relative_path",
    (
        "tools/verify_abi7.py",
        "compat/sglang/tools/qualify_h20.py",
        "core/examples/qwen-profile.json",
        "core/examples/mistral-plan.json",
        "core/src/runtime_v2.rs",
    ),
)
def test_generic_filename_gate_rejects_branding(
    source_root: Path, relative_path: str
) -> None:
    write_lines(source_root / relative_path)
    failures: list[str] = []
    verifier.check_generic_filenames(failures)
    assert len(failures) == 1
    assert relative_path in failures[0]


def test_generic_filename_gate_excludes_archives_and_generated_trees(
    source_root: Path,
) -> None:
    for relative_path in (
        "results/h20-qwen-v2/source.py",
        ".qualification/abi8/archive.py",
        "vendor/mistral/source.py",
        "build/qwen/generated.py",
        "core/src/__pycache__/h20.py",
    ):
        write_lines(source_root / relative_path)
    failures: list[str] = []
    verifier.check_generic_filenames(failures)
    assert failures == []


def test_public_identifier_gate_rejects_cli_and_target_branding(
    source_root: Path,
) -> None:
    cli = source_root / "tools/runner.py"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "parser.add_parser('qualify-h20')\n"
        "target = {'id': 'sglang.abi9'}\n",
        encoding="utf-8",
    )
    failures: list[str] = []
    verifier.check_public_identifiers(failures)
    assert len(failures) == 2
    assert all("public CLI/target identifier" in item for item in failures)


def test_public_identifier_gate_does_not_flag_versioned_schema_or_test_text(
    source_root: Path,
) -> None:
    source = source_root / "tests/contracts.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "schema = 'orbitkv.runtime-pressure.v1'\n"
        "assert target['id'] == 'sglang.abi9'\n",
        encoding="utf-8",
    )
    failures: list[str] = []
    verifier.check_public_identifiers(failures)
    assert failures == []


def test_public_identifier_gate_allows_wire_and_source_contract_data(
    source_root: Path,
) -> None:
    source = source_root / "core/src/contract.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "WIRE_VERSION = 9\n"
        "SCHEMA = 'orbitkv.runtime-manifest.v3'\n"
        "UPSTREAM_RELEASE = 'v0.5.17'\n"
        "target = {'id': 'sglang'}\n",
        encoding="utf-8",
    )
    failures: list[str] = []
    verifier.check_public_identifiers(failures)
    assert failures == []


def test_excludes_results_targets_and_cache_directories(
    source_root: Path,
) -> None:
    live = source_root / "core/src/live.py"
    write_lines(live)
    for relative_path in (
        "core/src/results/frozen.py",
        "core/src/target/generated.rs",
        "core/src/__pycache__/generated.py",
        "core/src/.pytest_cache/generated.py",
        "results/qualification/source/tools/frozen.py",
        "tools/results/frozen.py",
        "tools/__pycache__/generated.py",
        "compat/sglang/source/python/sglang/models/qwen.py",
    ):
        write_lines(source_root / relative_path)

    assert verifier.source_files(source_root / "core/src") == [live]
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
    for relative_path, marker in verifier.CURRENT_ENTRYPOINT_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")
    for relative_path, markers in verifier.WIRE_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        values = (markers,) if isinstance(markers, str) else markers
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    for relative_path, markers in verifier.WIRE_POLICY_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as target:
            target.write("\n".join(markers) + "\n")
    for relative_path, marker in verifier.CANONICAL_API_MARKERS.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n", encoding="utf-8")
    write_lines(source_root / "tests/test_gate.py")

    verifier.main()

    output = capsys.readouterr().out
    assert "10 production files" in output
    assert "1 test/bench files" in output
    assert "5 active scripts" in output
    assert "current qualification entrypoints present" in output
    assert "wire markers and canonical APIs present" in output
