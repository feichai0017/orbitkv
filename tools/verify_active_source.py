from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_LIMIT = 1_500
TEST_BENCH_LIMIT = 2_000
ACTIVE_SCRIPT_LIMIT = TEST_BENCH_LIMIT

PRODUCTION_ROOTS = (
    Path("src"),
    Path("crates"),
    Path("python/orbitkv-runtime/src"),
    Path("integrations/reference/src"),
    Path("integrations/sglang/src"),
)
TEST_BENCH_ROOTS = (
    Path("tests"),
    Path("python/orbitkv-runtime/tests"),
    Path("integrations/reference/tests"),
    Path("integrations/sglang/tests"),
)
ACTIVE_SCRIPT_ROOTS = (
    Path("integrations/sglang"),
    Path("tools"),
)
CANONICAL_GENERIC_ENTRYPOINTS = (
    Path("integrations/sglang/qualify_token_relocation.py"),
    Path("tools/verify_token_relocation_evidence.py"),
    Path("tools/verify_token_relocation_seal.py"),
    Path("tools/verify_fixed_state_pair_evidence.py"),
)
GENERIC_ENTRYPOINT_MARKERS = {
    Path("integrations/sglang/qualify_token_relocation.py"): "def main(",
    Path("tools/verify_token_relocation_evidence.py"): "def verify_evidence(",
    Path("tools/verify_token_relocation_seal.py"): "def verify_sealed_archive(",
    Path("tools/verify_fixed_state_pair_evidence.py"): "def verify_archive(",
}
LEGACY_COMPATIBILITY_ENTRYPOINTS = (
    Path("integrations/sglang/qualify_abi8_h20.py"),
    Path("integrations/sglang/qualify_token_relocation_h20.py"),
    Path("tools/verify_qwen35_h20_pair_evidence.py"),
    Path("tools/verify_token_relocation_h20_evidence.py"),
    Path("tools/verify_token_relocation_h20_seal.py"),
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "results", "target"}
)

ABI8_MARKERS = {
    Path("crates/orbitkv-ffi/include/orbitkv.h"): "#define ORBITKV_ABI_VERSION 8u",
    Path("crates/orbitkv-ffi/src/lib.rs"): "pub const ORBITKV_ABI_VERSION: u32 = 8;",
    Path("integrations/sglang/src/orbitkv_sglang/ffi/library.py"): "ABI_VERSION = 8",
}

# ABI5 exposed these scalar-shaped names even though their arguments were
# arrays. The current ABI8 surface is consistently batch-named. Historical
# result closures are not scanned, so their exact archived headers remain
# untouched.
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
        and not EXCLUDED_DIRECTORY_NAMES.intersection(
            path.relative_to(ROOT).parts[:-1]
        )
    )


def active_script_files() -> list[Path]:
    return sorted(
        {
            path
            for relative_root in ACTIVE_SCRIPT_ROOTS
            for path in (ROOT / relative_root).glob("*.py")
            if path.is_file()
        }
    )


def is_test_module(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return "tests" in relative.parts or path.name.startswith("test_")


def check_line_limit(path: Path, limit: int, failures: list[str]) -> None:
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if line_count > limit:
        failures.append(f"{path.relative_to(ROOT)}: {line_count} lines > {limit}")


def check_source_sizes(failures: list[str]) -> tuple[int, int, int]:
    production: set[Path] = set()
    tests_and_benches: set[Path] = set()
    active_scripts = set(active_script_files())

    for relative_root in PRODUCTION_ROOTS:
        for path in source_files(ROOT / relative_root):
            if is_test_module(path):
                tests_and_benches.add(path)
            else:
                production.add(path)

    for relative_root in TEST_BENCH_ROOTS:
        tests_and_benches.update(source_files(ROOT / relative_root))

    # Category precedence makes the counts disjoint even if roots overlap.
    tests_and_benches.difference_update(active_scripts)
    production.difference_update(active_scripts)
    production.difference_update(tests_and_benches)

    for path in sorted(production):
        check_line_limit(path, PRODUCTION_LIMIT, failures)
    for path in sorted(tests_and_benches):
        check_line_limit(path, TEST_BENCH_LIMIT, failures)
    for path in sorted(active_scripts):
        check_line_limit(path, ACTIVE_SCRIPT_LIMIT, failures)
    return len(production), len(tests_and_benches), len(active_scripts)


def check_generic_entrypoints(failures: list[str]) -> None:
    for relative_path in (
        *CANONICAL_GENERIC_ENTRYPOINTS,
        *LEGACY_COMPATIBILITY_ENTRYPOINTS,
    ):
        if not (ROOT / relative_path).is_file():
            failures.append(
                f"missing required qualification entrypoint: {relative_path}"
            )
    for relative_path, marker in GENERIC_ENTRYPOINT_MARKERS.items():
        path = ROOT / relative_path
        if path.is_file() and marker not in path.read_text(encoding="utf-8"):
            failures.append(
                f"invalid canonical generic entrypoint: {relative_path}"
            )


def check_abi8_markers(failures: list[str]) -> None:
    for relative_path, marker in ABI8_MARKERS.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing ABI8 surface: {path.relative_to(ROOT)}")
            continue
        if marker not in path.read_text(encoding="utf-8"):
            failures.append(
                f"{path.relative_to(ROOT)}: missing exact ABI8 marker {marker!r}"
            )


def check_removed_aliases(failures: list[str]) -> None:
    c_surfaces = (
        Path("crates/orbitkv-ffi/include/orbitkv.h"),
        Path("crates/orbitkv-ffi/src"),
    )
    python_surfaces = (
        Path("integrations/sglang/src"),
        Path("integrations/sglang/tests"),
    )
    c_alias_pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, REMOVED_ABI5_LIFECYCLE_ALIASES))})\s*\("
    )
    python_alias_pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, REMOVED_ABI5_PYTHON_ALIASES))})\s*\("
    )
    for relative_surface in c_surfaces:
        surface = ROOT / relative_surface
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
    for relative_surface in python_surfaces:
        surface = ROOT / relative_surface
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
    production_count, test_bench_count, active_script_count = check_source_sizes(
        failures
    )
    check_generic_entrypoints(failures)
    check_abi8_markers(failures)
    check_removed_aliases(failures)

    if failures:
        detail = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"active-source architecture gate failed:\n{detail}")

    print(
        "verified active source: "
        f"{production_count} production files <= {PRODUCTION_LIMIT} lines, "
        f"{test_bench_count} test/bench files <= {TEST_BENCH_LIMIT} lines, "
        f"{active_script_count} active scripts <= {ACTIVE_SCRIPT_LIMIT} lines, "
        "generic and compatibility entrypoints present, "
        "ABI8 markers present, no ABI5 lifecycle aliases"
    )


if __name__ == "__main__":
    main()
