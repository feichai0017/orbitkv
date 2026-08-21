from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_LIMIT = 1_500
TEST_BENCH_LIMIT = 2_000

PRODUCTION_ROOTS = (
    ROOT / "src",
    ROOT / "crates",
    ROOT / "integrations/sglang/src",
)
TEST_BENCH_ROOTS = (
    ROOT / "tests",
    ROOT / "integrations/sglang/tests",
)
TEST_BENCH_FILES = (
    ROOT / "integrations/sglang/bench_canonical_manager.py",
    ROOT / "integrations/sglang/bench_compact_control.py",
)

ABI6_MARKERS = {
    ROOT / "crates/orbitkv-ffi/include/orbitkv.h": "#define ORBITKV_ABI_VERSION 6u",
    ROOT / "crates/orbitkv-ffi/src/lib.rs": "pub const ORBITKV_ABI_VERSION: u32 = 6;",
    ROOT / "integrations/sglang/src/orbitkv_sglang/ffi/library.py": "ABI_VERSION = 6",
}

# ABI5 exposed these scalar-shaped names even though their arguments were
# arrays. ABI6 is consistently batch-named. Historical result closures are not
# scanned, so their exact archived headers remain untouched.
REMOVED_ABI5_LIFECYCLE_ALIASES = (
    "orbitkv_manager_abort_steps",
    "orbitkv_manager_quarantine_steps",
    "orbitkv_manager_quarantine_submissions",
    "orbitkv_manager_acknowledge_reclamations",
    "orbitkv_manager_recycle_requests",
)
REMOVED_ABI5_PYTHON_ALIASES = (
    "abort_steps",
    "quarantine_steps",
    "quarantine_submissions",
    "acknowledge_reclamations",
    "recycle_requests",
)


def source_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".rs", ".py"}
        and "target" not in path.parts
        and "__pycache__" not in path.parts
    )


def is_test_module(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return "tests" in relative.parts or path.name.startswith("test_")


def check_line_limit(path: Path, limit: int, failures: list[str]) -> None:
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if line_count > limit:
        failures.append(f"{path.relative_to(ROOT)}: {line_count} lines > {limit}")


def check_source_sizes(failures: list[str]) -> tuple[int, int]:
    production: set[Path] = set()
    tests_and_benches: set[Path] = set()

    for root in PRODUCTION_ROOTS:
        for path in source_files(root):
            if is_test_module(path):
                tests_and_benches.add(path)
            else:
                production.add(path)

    for root in TEST_BENCH_ROOTS:
        tests_and_benches.update(source_files(root))
    tests_and_benches.update(path for path in TEST_BENCH_FILES if path.is_file())

    for path in sorted(production):
        check_line_limit(path, PRODUCTION_LIMIT, failures)
    for path in sorted(tests_and_benches):
        check_line_limit(path, TEST_BENCH_LIMIT, failures)
    return len(production), len(tests_and_benches)


def check_abi6_markers(failures: list[str]) -> None:
    for path, marker in ABI6_MARKERS.items():
        if not path.is_file():
            failures.append(f"missing ABI6 surface: {path.relative_to(ROOT)}")
            continue
        if marker not in path.read_text(encoding="utf-8"):
            failures.append(
                f"{path.relative_to(ROOT)}: missing exact ABI6 marker {marker!r}"
            )


def check_removed_aliases(failures: list[str]) -> None:
    c_surfaces = (
        ROOT / "crates/orbitkv-ffi/include/orbitkv.h",
        ROOT / "crates/orbitkv-ffi/src",
    )
    python_surfaces = (
        ROOT / "integrations/sglang/src",
        ROOT / "integrations/sglang/tests",
    )
    c_alias_pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, REMOVED_ABI5_LIFECYCLE_ALIASES))})\s*\("
    )
    python_alias_pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, REMOVED_ABI5_PYTHON_ALIASES))})\s*\("
    )
    for surface in c_surfaces:
        paths = [surface] if surface.is_file() else source_files(surface)
        for path in paths:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                match = c_alias_pattern.search(line)
                if match is not None:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line_number}: "
                        f"removed ABI5 lifecycle alias {match.group(1)}"
                    )
    for surface in python_surfaces:
        for path in source_files(surface):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                match = python_alias_pattern.search(line)
                if match is not None:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line_number}: "
                        f"removed ABI5 Python lifecycle alias {match.group(1)}"
                    )


def main() -> None:
    failures: list[str] = []
    production_count, test_bench_count = check_source_sizes(failures)
    check_abi6_markers(failures)
    check_removed_aliases(failures)

    if failures:
        detail = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"active-source architecture gate failed:\n{detail}")

    print(
        "verified active source: "
        f"{production_count} production files <= {PRODUCTION_LIMIT} lines, "
        f"{test_bench_count} test/bench files <= {TEST_BENCH_LIMIT} lines, "
        "ABI6 markers present, no ABI5 lifecycle aliases"
    )


if __name__ == "__main__":
    main()
