from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_LIMIT = 1_500
TEST_LIMIT = 2_000
SOURCE_ROOTS = (
    Path("crates/orbitkv/src"),
    Path("crates/orbitkv-engine/src"),
    Path("crates/orbitkv-executor/src"),
    Path("crates/orbitkv-server/src"),
)
TEST_ROOTS = (
    Path("crates/orbitkv/tests"),
    Path("crates/orbitkv-engine/tests"),
    Path("crates/orbitkv-executor/tests"),
)
REMOVED_PATHS = (
    Path("compat"),
    Path("core"),
    Path("engine"),
    Path("executor"),
    Path("server"),
    Path("tests"),
    Path("crates/orbitkv/ffi"),
)
EXCLUDED = frozenset({".git", "target", "results", "node_modules", "luminal"})
SPECIFIC_FILENAME = re.compile(
    r"(?:^|[._-])(?:abi\d+|wire\d+|h\d+|v\d+)(?:$|[._-])"
    r"|qwen|mistral|deepseek|llama|gemma|sglang",
    re.IGNORECASE,
)
LAYER_MANIFESTS = {
    "orbitkv": Path("crates/orbitkv/Cargo.toml"),
    "executor": Path("crates/orbitkv-executor/Cargo.toml"),
    "engine": Path("crates/orbitkv-engine/Cargo.toml"),
    "server": Path("crates/orbitkv-server/Cargo.toml"),
}
WORKSPACE_MEMBERS = (
    "crates/orbitkv",
    "crates/orbitkv-engine",
    "crates/orbitkv-executor",
    "crates/orbitkv-server",
)
FORBIDDEN_LAYER_DEPENDENCIES = {
    "orbitkv": frozenset({"orbitkv-executor", "orbitkv-server", "luminal", "luminal_cuda_lite", "luminal_nn"}),
    "executor": frozenset({"orbitkv-server"}),
    "engine": frozenset(),
    "server": frozenset({"orbitkv", "orbitkv-executor", "luminal", "luminal_cuda_lite", "luminal_nn"}),
}
SERVER_PHYSICAL_TYPES = re.compile(
    r"\b(?:BackendArenaRegistration|CanonicalKvManager|ExecutorArena|ExecutorPlan|"
    r"PageLease|RuntimeSession)\b"
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


def direct_dependencies(manifest: Path) -> set[str]:
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    return set(document.get("dependencies", {}))


def main() -> int:
    failures: list[str] = []
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]
    if tuple(workspace.get("members", ())) != WORKSPACE_MEMBERS:
        failures.append("root workspace members are not the four owned OrbitKV crates")
    if tuple(workspace.get("default-members", ())) != WORKSPACE_MEMBERS:
        failures.append("root default members are not the four owned OrbitKV crates")
    if "third_party/luminal" not in workspace.get("exclude", ()):
        failures.append("third-party Luminal must be excluded from the OrbitKV workspace")
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

    for layer, relative in LAYER_MANIFESTS.items():
        forbidden = direct_dependencies(ROOT / relative) & FORBIDDEN_LAYER_DEPENDENCIES[layer]
        if forbidden:
            failures.append(
                f"{layer} has forbidden inward dependency: {', '.join(sorted(forbidden))}"
            )

    for path in source_files(ROOT / "crates/orbitkv-server/src"):
        if SERVER_PHYSICAL_TYPES.search(path.read_text(encoding="utf-8")):
            failures.append(
                f"server source names a physical KV ownership type: {path.relative_to(ROOT)}"
            )

    executor_manifest = (ROOT / "crates/orbitkv-executor/Cargo.toml").read_text(encoding="utf-8")
    submodule = ROOT / "third_party/luminal"
    expected_paths = {
        "luminal": "../../third_party/luminal",
        "luminal_cuda_lite": "../../third_party/luminal/crates/luminal_cuda_lite",
        "luminal_nn": "../../third_party/luminal/crates/luminal_nn",
    }
    if not submodule.is_dir():
        failures.append("Luminal submodule is missing")
    for dependency, path in expected_paths.items():
        pattern = rf'{dependency} = \{{ path = "{re.escape(path)}", optional = true \}}'
        if re.search(pattern, executor_manifest) is None:
            failures.append(f"{dependency} does not use the visible Luminal submodule")
    if "orbitkv-luminal.git" in executor_manifest:
        failures.append("executor manifest still has a second remote Luminal source")

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
