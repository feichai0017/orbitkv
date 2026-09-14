"""Check the owned workspace, dependency direction, and source/test boundaries."""
from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_LIMIT = 1_500
TEST_LIMIT = 2_000
# Allowed production dependencies among owned crates. External packages and
# test-only references do not acquire authority across these boundaries.
LAYER_DEPENDENCIES = {
    "orbitkv": frozenset(),
    "orbitkv-compiler": frozenset(),
    "orbitkv-cuda": frozenset({"orbitkv-compiler", "orbitkv-ops", "orbitkv-tracing"}),
    "orbitkv-engine": frozenset({"orbitkv", "orbitkv-executor"}),
    "orbitkv-executor": frozenset({"orbitkv", "orbitkv-compiler", "orbitkv-cuda", "orbitkv-ops", "orbitkv-tracing"}),
    "orbitkv-ops": frozenset({"orbitkv-compiler"}),
    "orbitkv-tracing": frozenset(),
}
WORKSPACE_MEMBERS = tuple(f"crates/{name}" for name in LAYER_DEPENDENCIES)
DEFAULT_MEMBERS = tuple(path for path in WORKSPACE_MEMBERS if path != "crates/orbitkv-cuda")
REMOVED_PATHS = (
    "compat", "core", "engine", "executor", "server", "tests", "third_party",
    ".gitmodules", "crates/orbitkv/ffi", "crates/orbitkv-server",
)
EXCLUDED = frozenset({
    ".git", "target", "results", "node_modules", ".qualification", ".venv",
    ".astro", "__pycache__", "dist", "build",
})
SPECIFIC_FILENAME = re.compile(
    r"(?:^|[._-])(?:abi\d+|wire\d+|h\d+|v\d+)(?:$|[._-])"
    r"|qwen|mistral|deepseek|llama|gemma|sglang", re.IGNORECASE,
)
PROTOCOL_IMPLEMENTATION_TYPES = re.compile(
    r"\b(?:BackendArenaRegistration|CanonicalKvManager|ExecutorArena|ExecutorPlan|"
    r"PageLease|RuntimeSession|ModelEngine|CompiledDecoder)\b"
)
PROTOCOL_IMPLEMENTATION_IMPORTS = re.compile(
    r"\b(?:orbitkv|orbitkv_executor|orbitkv_compiler|orbitkv_cuda|orbitkv_ops|orbitkv_tracing)\s*::"
)
PROTOCOL_ROOTS = (
    "crates/orbitkv-engine/src/protocol.rs", "crates/orbitkv-engine/src/protocol",
    "crates/orbitkv-engine/src/frontend.rs", "crates/orbitkv-engine/src/frontend",
)
UNIT_BRIDGE = re.compile(
    r'#\[cfg\(test\)\]\s*#\[path = "([^"]+)"\]\s*'
    r'(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+;'
)


def source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".rs" else []
    return sorted(root.rglob("*.rs")) if root.is_dir() else []


def active_files() -> list[Path]:
    files = []
    for directory, children, names in os.walk(ROOT):
        children[:] = sorted(name for name in children if name not in EXCLUDED)
        files.extend(Path(directory) / name for name in sorted(names))
    return files


def direct_dependencies(manifest: dict) -> set[str]:
    tables = [manifest.get("dependencies", {})]
    tables.extend(target.get("dependencies", {}) for target in manifest.get("target", {}).values())
    return {value.get("package", name) if isinstance(value, dict) else name
            for table in tables for name, value in table.items()}


def check_rust_layout(path: Path, source: str, *, production: bool) -> list[str]:
    relative = path.relative_to(ROOT)
    failures = []
    if production and path.name == "mod.rs":
        failures.append(f"{relative}: use a sibling module.rs entry, not mod.rs")
    if production and (
        path.name == "tests.rs" or path.name.endswith("_tests.rs") or "tests" in relative.parts
        or re.search(r"\bmod\s+tests\s*\{", source)
        or re.search(r"#\[\s*(?:\w+::)*test\s*(?:\]|\()", source)
    ):
        failures.append(f"{relative}: test code belongs in the crate's tests/ directory")
    bridges = list(UNIT_BRIDGE.finditer(source))
    for attribute in re.finditer(r"#\s*\[\s*path\s*=", source):
        # Test suites may themselves compose private test modules by path.
        if not production:
            continue
        bridge = next((item for item in bridges if item.start() <= attribute.start() < item.end()), None)
        if bridge is None:
            failures.append(f"{relative}: explicit module paths are reserved for cfg(test) unit bridges")
            continue
        crate_root = next(parent for parent in path.parents if (parent / "Cargo.toml").is_file())
        target = (path.parent / bridge.group(1)).resolve()
        if not target.is_relative_to(crate_root / "tests/unit") or not target.is_file():
            failures.append(f"{relative}: unit-test bridge must resolve inside this crate's tests/unit/")
    if re.search(r"\binclude!\s*\(", source):
        failures.append(f"{relative}: use modules instead of flattening Rust source with include!")
    return failures


def main() -> int:
    failures = []
    document = tomllib.loads((ROOT / "Cargo.toml").read_text())
    workspace = document["workspace"]
    if tuple(workspace.get("members", ())) != WORKSPACE_MEMBERS:
        failures.append("root workspace members do not match the owned OrbitKV crates")
    if tuple(workspace.get("default-members", ())) != DEFAULT_MEMBERS:
        failures.append("default members must include every host crate and exclude CUDA device tests")
    if workspace.get("exclude"):
        failures.append("owned crates must not be hidden in excluded workspaces")
    for path in REMOVED_PATHS:
        if (ROOT / path).exists():
            failures.append(f"removed product path still exists: {path}")
    for name, allowed in LAYER_DEPENDENCIES.items():
        crate_root = ROOT / "crates" / name
        manifest = tomllib.loads((crate_root / "Cargo.toml").read_text())
        if manifest["package"]["name"] != name or "workspace" in manifest:
            failures.append(f"{name}: package name or workspace ownership mismatch")
        if (crate_root / "Cargo.lock").exists() or (crate_root / ".git").exists():
            failures.append(f"{name}: use the root lockfile and repository")
        dependencies = direct_dependencies(manifest)
        forbidden = dependencies.intersection(LAYER_DEPENDENCIES) - allowed
        if forbidden:
            failures.append(f"{name} has forbidden inward dependency: {', '.join(sorted(forbidden))}")
        if any(dependency.startswith("luminal") for dependency in dependencies):
            failures.append(f"{name}: compiler dependencies must use the owned OrbitKV packages")
        binding = workspace.get("dependencies", {}).get(name, {})
        if binding.get("path") != f"crates/{name}":
            failures.append(f"{name}: workspace dependency must name its local source")
    # Explicit, non-growing migration debt is audited for every owned crate.
    # Generated protobuf source is identified by its generator declaration.
    budgets = json.loads((ROOT / "tools/source-size-baseline.json").read_text())["files"]
    observed_budgets = set()
    counts = {"production": 0, "test": 0}
    for member in WORKSPACE_MEMBERS:
        for directory, production in (("src", True), ("tests", False)):
            for path in source_files(ROOT / member / directory):
                source = path.read_text()
                relative = str(path.relative_to(ROOT))
                lines = len(source.splitlines())
                failures.extend(check_rust_layout(path, source, production=production))
                counts["production" if production else "test"] += 1
                generated = source.startswith("// This file is @generated by prost-build.")
                limit = PRODUCTION_LIMIT if production else TEST_LIMIT
                if relative in budgets:
                    observed_budgets.add(relative)
                    if generated or lines <= limit:
                        failures.append(f"{relative}: remove the obsolete source-size baseline entry")
                    limit = budgets[relative]["max_lines"]
                if lines > limit and not generated:
                    failures.append(f"{relative}: {lines} lines exceeds {limit}; extract a cohesive module")
    for stale in budgets.keys() - observed_budgets:
        failures.append(f"stale source-size baseline: {stale}")
    for path in active_files():
        if SPECIFIC_FILENAME.search(path.name):
            failures.append(f"specific active filename: {path.relative_to(ROOT)}")
    for boundary in PROTOCOL_ROOTS:
        for path in source_files(ROOT / boundary):
            source = path.read_text()
            if PROTOCOL_IMPLEMENTATION_TYPES.search(source) or PROTOCOL_IMPLEMENTATION_IMPORTS.search(source):
                failures.append(f"logical protocol/frontend source crosses the execution boundary: {path.relative_to(ROOT)}")
    for failure in failures:
        print(f"active-source failure: {failure}")
    if failures:
        return 1
    print(f"active-source verified: {len(WORKSPACE_MEMBERS)} owned crates; "
          f"{counts['production']} production and {counts['test']} test Rust files; "
          "one workspace, local dependencies, and consistent test/module boundaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
