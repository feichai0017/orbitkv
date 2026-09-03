from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_LIMIT = 1_500
TEST_BENCH_LIMIT = 2_000
ACTIVE_SCRIPT_LIMIT = TEST_BENCH_LIMIT

PRODUCTION_ROOTS = (
    Path("core/src"),
    Path("core/ffi/src"),
    Path("compat/sglang/bridge/src"),
)
TEST_BENCH_ROOTS = (
    Path("tests"),
    Path("core/tests"),
    Path("core/ffi/tests"),
    Path("compat/sglang/tests"),
)
ACTIVE_SCRIPT_ROOTS = (
    Path("compat/sglang/tools"),
    Path("tools"),
)
CURRENT_ENTRYPOINT_MARKERS = {
    Path("compat/sglang/tools/qualification_runner.py"): "def main(",
    Path("compat/sglang/tools/qualification_runtime.py"): "def adapter_identity(",
    Path("tools/verify_qualification.py"): "def verify_pair(",
    Path("tools/qualification_source.py"): "def validate_source_identity(",
    Path("tools/qualification_gates.py"): "def pair_gates(",
}
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".qualification",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "results",
        "target",
        "vendor",
    }
)
EXCLUDED_REPOSITORY_ROOTS = (
    Path("compat/sglang/source"),
    Path("executor/luminal"),
)

WIRE_MARKERS = {
    Path("core/ffi/include/orbitkv.h"): "#define ORBITKV_WIRE_VERSION 14u",
    Path("core/ffi/src/lib.rs"): "pub const ORBITKV_WIRE_VERSION: u32 = 14;",
    Path("compat/sglang/bridge/src/orbitkv_sglang/ffi/library.py"): "WIRE_VERSION = 14",
    Path("core/src/runtime_target.rs"): (
        "const SGLANG_TARGET_CONTRACT_VERSION: u32 = 4;",
        "const REQUIRED_WIRE_VERSION: u32 = 14;",
    ),
}
WIRE_POLICY_MARKERS = {
    Path("core/src/runtime_session.rs"): (
        "pub enum CacheSharingPolicy",
        "RequestPrivate = 1",
        "SharedPrefix = 2",
    ),
    Path("core/ffi/include/orbitkv.h"): (
        "ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE 1u",
        "ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX 2u",
        "OrbitKvSessionCreateConfig",
    ),
    Path("core/ffi/src/session/layouts.rs"): (
        "ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE: u32 = 1;",
        "ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX: u32 = 2;",
        "OrbitKvSessionCreateConfig",
    ),
    Path("compat/sglang/bridge/src/orbitkv_sglang/ffi/layouts.py"): (
        "class SessionCreateConfigLayout(ctypes.Structure):",
        '("cache_sharing_policy", U32)',
    ),
    Path("compat/sglang/bridge/src/orbitkv_sglang/runtime/identity.py"): (
        "class CacheSharingPolicy(IntEnum):",
        "REQUEST_PRIVATE = 1",
        "SHARED_PREFIX = 2",
    ),
}
CANONICAL_API_MARKERS = {
    Path("compat/sglang/bridge/src/orbitkv_sglang/config.py"): "class RuntimeConfig:",
    Path("compat/sglang/bridge/src/orbitkv_sglang/ffi/session_relocation.py"): (
        "class SessionRelocationMixin:"
    ),
    Path("compat/sglang/bridge/src/orbitkv_sglang/bridge/__init__.py"): (
        "direct-source SGLang adapter"
    ),
}
REMOVED_PRODUCTION_PATHS = (
    Path("core/python"),
    Path("core/reference"),
    Path("compat/sglang/bridge/src/orbitkv_sglang/bridge/external_append.py"),
    Path("compat/sglang/bridge/src/orbitkv_sglang/bridge/external_lifecycle.py"),
    Path("compat/sglang/bridge/src/orbitkv_sglang/bridge/external_validation.py"),
    Path("compat/sglang/bridge/src/orbitkv_sglang/plugin"),
    Path("compat/sglang/bridge/src/orbitkv_sglang/bridge/structured_arena.py"),
)
REMOVED_PRODUCTION_SPELLINGS = (
    "ManagerPlanConfig",
    "state_plan_path",
    "state_plan_fingerprint",
    "build_structured_arenas",
    "ORBITKV_PLAN",
    "ORBITKV_STATE_PLAN",
    "compile-hf-manager-plan",
    "executor_capabilities",
    "runtime_manifest_v2",
    "sglang-v0517",
    "CtypesManager",
    "CanonicalRuntime",
    "orbitkv_runtime",
    "orbitkv_manager_",
)
REMOVED_PRODUCTION_METHODS = (
    "acknowledge_relocation",
    "record_completion",
    "relocate_tokens",
)

_FILENAME_BRANDING = (
    ("ABI version", re.compile(r"abi[0-9]+", re.IGNORECASE)),
    ("device model", re.compile(r"(?<![a-z0-9])h20(?![a-z0-9])", re.IGNORECASE)),
    (
        "model family",
        re.compile(
            r"qwen|mistral|gpt[-_]?oss|deepseek|llama|gemma",
            re.IGNORECASE,
        ),
    ),
    (
        "version suffix",
        re.compile(r"(?:^|[._-])v[0-9]+(?=$|[._-])", re.IGNORECASE),
    ),
)
_PUBLIC_IDENTIFIER_SUFFIXES = frozenset({".json", ".py", ".rs"})
_PUBLIC_IDENTIFIER_CONTEXT = re.compile(
    r"(?:add_argument|add_parser|subcommand|command|target(?:[-_ ]?id)?)",
    re.IGNORECASE,
)
_STRING_LITERAL = re.compile(r"[\"']([^\"']+)[\"']")

# A retired wire exposed these scalar-shaped names even though their arguments
# were arrays. The current surface is consistently batch-named. Historical
# result closures are not scanned, so their exact archived headers remain
# untouched.
REMOVED_SCALAR_LIFECYCLE_ALIASES = (
    "orbitkv_manager_abort_steps",
    "orbitkv_manager_quarantine_steps",
    "orbitkv_manager_quarantine_submissions",
    "orbitkv_manager_acknowledge_reclamations",
    "orbitkv_manager_recycle_requests",
)
REMOVED_SCALAR_PYTHON_ALIASES = (
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


def _is_excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return any(
        part in EXCLUDED_DIRECTORY_NAMES
        or part.endswith(".egg-info")
        or part.startswith(".venv")
        for part in relative.parts
    ) or any(relative.is_relative_to(root) for root in EXCLUDED_REPOSITORY_ROOTS)


def active_repository_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not _is_excluded(path)
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


def check_current_entrypoints(failures: list[str]) -> None:
    for relative_path, marker in CURRENT_ENTRYPOINT_MARKERS.items():
        if not (ROOT / relative_path).is_file():
            failures.append(
                f"missing required qualification entrypoint: {relative_path}"
            )
            continue
        path = ROOT / relative_path
        if marker not in path.read_text(encoding="utf-8"):
            failures.append(
                f"invalid current qualification entrypoint: {relative_path}"
            )


def check_wire_markers(failures: list[str]) -> None:
    for relative_path, markers in WIRE_MARKERS.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing wire-version surface: {path.relative_to(ROOT)}")
            continue
        source = path.read_text(encoding="utf-8")
        for marker in (markers,) if isinstance(markers, str) else markers:
            if marker not in source:
                failures.append(
                    f"{path.relative_to(ROOT)}: missing exact wire marker {marker!r}"
                )
    for relative_path, markers in WIRE_POLICY_MARKERS.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(
                f"missing cache-policy wire surface: {path.relative_to(ROOT)}"
            )
            continue
        source = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in source:
                failures.append(
                    f"{path.relative_to(ROOT)}: missing exact cache-policy "
                    f"marker {marker!r}"
                )


def check_canonical_api(failures: list[str]) -> None:
    """Require the breaking-only API and reject retired production spellings."""

    for relative_path in REMOVED_PRODUCTION_PATHS:
        if (ROOT / relative_path).exists():
            failures.append(f"retired production path remains: {relative_path}")

    for relative_path, marker in CANONICAL_API_MARKERS.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing canonical API surface: {relative_path}")
            continue
        if marker not in path.read_text(encoding="utf-8"):
            failures.append(
                f"{relative_path}: missing canonical API marker {marker!r}"
            )

    spelling_pattern = re.compile(
        "|".join(
            rf"\b{re.escape(value)}\b"
            for value in REMOVED_PRODUCTION_SPELLINGS
        )
    )
    method_pattern = re.compile(
        rf"^\s*def\s+({'|'.join(map(re.escape, REMOVED_PRODUCTION_METHODS))})\s*\("
    )
    production = {
        path
        for relative_root in PRODUCTION_ROOTS
        for path in source_files(ROOT / relative_root)
        if not is_test_module(path)
    }
    for path in sorted(production):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            spelling = spelling_pattern.search(line)
            if spelling is not None:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: retired production "
                    f"spelling {spelling.group(0)}"
                )
            method = method_pattern.search(line)
            if method is not None:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: retired scalar method "
                    f"{method.group(1)}"
                )


def _branding(value: str) -> str | None:
    for label, pattern in _FILENAME_BRANDING:
        if pattern.search(value):
            return label
    return None


def check_generic_filenames(failures: list[str]) -> None:
    """Reject hardware, model, ABI, and version branding in active paths."""

    reported: set[tuple[Path, str]] = set()
    for path in active_repository_files():
        relative = path.relative_to(ROOT)
        for component in relative.parts:
            label = _branding(component)
            key = (relative, label or "")
            if label is not None and key not in reported:
                failures.append(
                    f"{relative}: active filename contains {label} branding"
                )
                reported.add(key)


def _branded_public_value(value: str) -> str | None:
    """Return branding only when the whole public spelling carries it."""

    # Wire/schema versions and upstream release/revision data are deliberately
    # allowed. This gate is about CLI names and target identities, not arbitrary
    # string literals embedded in source contracts.
    return _branding(value)


def _public_identifier_candidates(path: Path) -> list[tuple[int, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    candidates = []
    for number, line in enumerate(lines, start=1):
        if _PUBLIC_IDENTIFIER_CONTEXT.search(line) is None:
            continue
        candidates.extend(
            (number, match.group(1)) for match in _STRING_LITERAL.finditer(line)
        )
    return candidates


def check_public_identifiers(failures: list[str]) -> None:
    """Reject branding in public CLI spellings and runtime target IDs."""

    for path in active_repository_files():
        if path.suffix.casefold() not in _PUBLIC_IDENTIFIER_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if "tests" in relative.parts or path.name.startswith("test_"):
            continue
        candidates = _public_identifier_candidates(path)
        for line_number, value in candidates:
            label = _branded_public_value(value)
            if label is not None:
                failures.append(
                    f"{relative}:{line_number}: public CLI/target identifier "
                    f"contains {label} branding"
                )


def check_removed_aliases(failures: list[str]) -> None:
    c_surfaces = (
        Path("core/ffi/include/orbitkv.h"),
        Path("core/ffi/src"),
    )
    python_surfaces = (
        Path("compat/sglang/bridge/src"),
        Path("compat/sglang/tests"),
    )
    c_alias_pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, REMOVED_SCALAR_LIFECYCLE_ALIASES))})\s*\("
    )
    python_alias_pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, REMOVED_SCALAR_PYTHON_ALIASES))})\s*\("
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
                        f"removed scalar lifecycle alias {match.group(1)}"
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
                        f"removed scalar Python lifecycle alias {match.group(1)}"
                    )


def main() -> None:
    failures: list[str] = []
    production_count, test_bench_count, active_script_count = check_source_sizes(
        failures
    )
    check_current_entrypoints(failures)
    check_wire_markers(failures)
    check_canonical_api(failures)
    check_generic_filenames(failures)
    check_public_identifiers(failures)
    check_removed_aliases(failures)

    if failures:
        detail = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"active-source architecture gate failed:\n{detail}")

    print(
        "verified active source: "
        f"{production_count} production files <= {PRODUCTION_LIMIT} lines, "
        f"{test_bench_count} test/bench files <= {TEST_BENCH_LIMIT} lines, "
        f"{active_script_count} active scripts <= {ACTIVE_SCRIPT_LIMIT} lines, "
        "current qualification entrypoints present, "
        "wire markers and canonical APIs present, active names are generic, "
        "no retired scalar lifecycle aliases"
    )


if __name__ == "__main__":
    main()
