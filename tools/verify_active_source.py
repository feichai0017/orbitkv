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
    Path("crates/orbitkv-server"),
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
}
WORKSPACE_MEMBERS = (
    "crates/orbitkv",
    "crates/orbitkv-engine",
    "crates/orbitkv-executor",
)
FORBIDDEN_LAYER_DEPENDENCIES = {
    "orbitkv": frozenset({"orbitkv-executor", "orbitkv-engine", "luminal", "luminal_cuda_lite", "luminal_nn", "luminal_tracing"}),
    "executor": frozenset({"orbitkv-engine"}),
    "engine": frozenset({"luminal", "luminal_cuda_lite", "luminal_nn", "luminal_tracing"}),
}
PROTOCOL_IMPLEMENTATION_TYPES = re.compile(
    r"\b(?:BackendArenaRegistration|CanonicalKvManager|ExecutorArena|ExecutorPlan|"
    r"PageLease|RuntimeSession|ModelEngine|CompiledDecoder)\b"
)
PROTOCOL_IMPLEMENTATION_IMPORTS = re.compile(
    r"\b(?:orbitkv|orbitkv_executor|luminal|luminal_cuda_lite|luminal_nn|luminal_tracing)\s*::"
)
PROTOCOL_ROOTS = (
    Path("crates/orbitkv-engine/src/protocol.rs"),
    Path("crates/orbitkv-engine/src/protocol"),
    Path("crates/orbitkv-engine/src/frontend.rs"),
    Path("crates/orbitkv-engine/src/frontend"),
)


def source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".rs" else []
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


def check_rust_layout(path: Path, source: str, *, production: bool) -> list[str]:
    relative = path.relative_to(ROOT)
    failures: list[str] = []
    if production and path.name == "mod.rs":
        failures.append(f"{relative}: use a sibling module.rs entry, not mod.rs")
    if production and (
        path.name == "tests.rs"
        or path.name.endswith("_tests.rs")
        or "tests" in relative.parts
        or re.search(r"\bmod\s+tests\s*\{", source)
        or re.search(r"#\[\s*(?:\w+::)*test\s*(?:\]|\()", source)
    ):
        failures.append(f"{relative}: test code belongs in the crate's tests/ directory")

    # Unit tests keep their private module namespace while their source lives
    # entirely under tests/unit/. This bridge must remain test-only.
    bridges = list(re.finditer(
        r'#\[cfg\(test\)\]\s*#\[path = "([^"]+)"\]\s*mod tests;', source
    ))
    for attribute in re.finditer(r"#\s*\[\s*path\s*=", source):
        bridge = next((item for item in bridges if item.start() <= attribute.start() < item.end()), None)
        if not production or bridge is None:
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
    failures: list[str] = []
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]
    if tuple(workspace.get("members", ())) != WORKSPACE_MEMBERS:
        failures.append("root workspace members do not match the owned OrbitKV crates")
    if tuple(workspace.get("default-members", ())) != WORKSPACE_MEMBERS:
        failures.append("root default members do not match the owned OrbitKV crates")
    if "third_party/luminal" not in workspace.get("exclude", ()):
        failures.append("third-party Luminal must be excluded from the OrbitKV workspace")
    fork = ROOT / "third_party/luminal"
    retained = {"luminal", "luminal_nn", "luminal_cuda_lite", "luminal_tracing"}
    fork_manifest = tomllib.loads((fork / "Cargo.toml").read_text())
    expected_members = {"crates/" + name for name in retained if name != "luminal"}
    if set(fork_manifest["workspace"]["members"]) != expected_members:
        failures.append("Luminal fork workspace must contain only the inference dependency closure")
    for manifest in [fork / "Cargo.toml", *sorted((fork / "crates").glob("*/Cargo.toml"))]:
        package = tomllib.loads(manifest.read_text())["package"]["name"]
        if package not in retained:
            failures.append(f"unused Luminal fork package: {package}")
        dependencies = direct_dependencies(manifest)
        if package in {"luminal", "luminal_nn", "luminal_tracing"} and "luminal_cuda_lite" in dependencies:
            failures.append(f"{package} must not depend on the CUDA backend")
    for path in REMOVED_PATHS:
        if (ROOT / path).exists():
            failures.append(f"removed product path still exists: {path}")

    production_count = 0
    test_count = 0
    for root in SOURCE_ROOTS:
        for path in source_files(ROOT / root):
            source = path.read_text(encoding="utf-8")
            lines = len(source.splitlines())
            failures.extend(check_rust_layout(path, source, production=True))
            limit = PRODUCTION_LIMIT
            production_count += 1
            if lines > limit:
                failures.append(
                    f"{path.relative_to(ROOT)}: {lines} lines exceeds {limit}"
                )
    for root in TEST_ROOTS:
        for path in source_files(ROOT / root):
            test_count += 1
            source = path.read_text(encoding="utf-8")
            lines = len(source.splitlines())
            failures.extend(check_rust_layout(path, source, production=False))
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

    for boundary in PROTOCOL_ROOTS:
        for path in source_files(ROOT / boundary):
            source = path.read_text(encoding="utf-8")
            if (PROTOCOL_IMPLEMENTATION_TYPES.search(source)
                    or PROTOCOL_IMPLEMENTATION_IMPORTS.search(source)):
                failures.append(
                    f"logical protocol/frontend source crosses the execution boundary: {path.relative_to(ROOT)}"
                )

    executor_manifest = (ROOT / "crates/orbitkv-executor/Cargo.toml").read_text(encoding="utf-8")
    submodule = ROOT / "third_party/luminal"
    expected_paths = {
        "luminal": "../../third_party/luminal",
        "luminal_cuda_lite": "../../third_party/luminal/crates/luminal_cuda_lite",
        "luminal_nn": "../../third_party/luminal/crates/luminal_nn",
        "luminal_tracing": "../../third_party/luminal/crates/luminal_tracing",
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
        "module/test layout consistent, removed integration paths absent and filenames generic"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
