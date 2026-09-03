from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_LIMIT = 1_500
TEST_LIMIT = 2_000
SOURCE_ROOTS = (Path("core/src"), Path("executor/src"), Path("server/src"))
TEST_ROOTS = (Path("core/tests"), Path("executor/tests"))
REMOVED_PATHS = (Path("compat"), Path("core/ffi"), Path("tests"))
EXCLUDED = frozenset({".git", "target", "results", "node_modules", "luminal"})
SPECIFIC_FILENAME = re.compile(
    r"(?:^|[._-])(?:abi\d+|wire\d+|h\d+|v\d+)(?:$|[._-])"
    r"|qwen|mistral|deepseek|llama|gemma|sglang",
    re.IGNORECASE,
)


def source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*.rs")
        if not EXCLUDED.intersection(path.relative_to(ROOT).parts)
    )


def active_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED.intersection(path.relative_to(ROOT).parts)
        and not path.relative_to(ROOT).is_relative_to(Path(".qualification"))
    )


def main() -> int:
    failures: list[str] = []
    for path in REMOVED_PATHS:
        if (ROOT / path).exists():
            failures.append(f"removed product path still exists: {path}")

    production_count = 0
    test_count = 0
    for root in SOURCE_ROOTS:
        for path in source_files(ROOT / root):
            lines = len(path.read_text(encoding="utf-8").splitlines())
            is_test = "tests" in path.parts or path.name.endswith("_tests.rs")
            limit = TEST_LIMIT if is_test else PRODUCTION_LIMIT
            test_count += int(is_test)
            production_count += int(not is_test)
            if lines > limit:
                failures.append(
                    f"{path.relative_to(ROOT)}: {lines} lines exceeds {limit}"
                )
    for root in TEST_ROOTS:
        for path in source_files(ROOT / root):
            test_count += 1
            lines = len(path.read_text(encoding="utf-8").splitlines())
            if lines > TEST_LIMIT:
                failures.append(
                    f"{path.relative_to(ROOT)}: {lines} lines exceeds {TEST_LIMIT}"
                )

    for path in active_files():
        relative = path.relative_to(ROOT)
        if SPECIFIC_FILENAME.search(relative.name):
            failures.append(f"specific active filename: {relative}")

    executor_manifest = (ROOT / "executor/Cargo.toml").read_text(encoding="utf-8")
    revisions = set(
        re.findall(
            r'orbitkv-luminal\.git", rev = "([0-9a-f]{40})"',
            executor_manifest,
        )
    )
    submodule = ROOT / "executor/luminal"
    if len(revisions) != 1 or not submodule.is_dir():
        failures.append("executor dependencies do not pin one Luminal fork revision")
    else:
        actual = subprocess.run(
            ["git", "-C", str(submodule), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual not in revisions:
            failures.append(
                f"Luminal dependency revision {next(iter(revisions))} "
                f"differs from submodule {actual}"
            )

    if failures:
        for failure in failures:
            print(f"active-source failure: {failure}")
        return 1

    print(
        "active-source verified: "
        f"{production_count} production Rust files, {test_count} Rust test files; "
        "removed integration paths absent and filenames generic"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
