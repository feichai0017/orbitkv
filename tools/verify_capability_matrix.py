from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/capability-matrix.md"
WEBSITE_EVIDENCE = ROOT / "website/src/pages/evidence.astro"
WIRE_HEADER = ROOT / "core/ffi/include/orbitkv.h"
WIRE_LIBRARY = ROOT / "compat/sglang/bridge/src/orbitkv_sglang/ffi/library.py"
WIRE_LAYOUTS = ROOT / "compat/sglang/bridge/src/orbitkv_sglang/ffi/layouts.py"
RUST_RUNTIME_TARGET = ROOT / "core/src/runtime_target.rs"
RUNTIME_TARGET = (
    ROOT
    / "compat/sglang/bridge/src/orbitkv_sglang/resources/runtime_target.json"
)

CURRENT_WIRE_VERSION = 14
CURRENT_SYMBOL_COUNT = 48
CURRENT_LAYOUT_COUNT = 78
CURRENT_TARGET_ID = "sglang"
CURRENT_TARGET_CONTRACT_VERSION = 4
CURRENT_TARGET_FINGERPRINT = (
    "sha256:ac915458195e757e477cf04866dae76147cd71a7472661e0d791e9c4474173ba"
)
SESSION_CREATE_LAYOUT = ("SessionCreateConfigLayout", (40, 8))
SESSION_CREATE_FIELDS = ("manager", "cache_sharing_policy", "reserved")
REQUIRED_CURRENT_WIRE_HEADER_MARKERS = (
    "#define ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE 1u",
    "#define ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX 2u",
    "const OrbitKvSessionCreateConfig *config",
)
REQUIRED_SESSION_SYMBOLS = frozenset(
    {
        "orbitkv_session_abort_control",
        "orbitkv_session_abort_prepared_relocation",
        "orbitkv_session_commit_control",
        "orbitkv_session_complete_relocation",
        "orbitkv_session_confirm_control",
        "orbitkv_session_confirm_relocation_publication",
        "orbitkv_session_mark_token_dispositions_batch",
        "orbitkv_session_prefix_lookup_batch",
        "orbitkv_session_prefix_publish_batch",
        "orbitkv_session_prefix_publish_release_batch",
        "orbitkv_session_prepare_prefix_attach",
        "orbitkv_session_prepare_prefix_evict",
        "orbitkv_session_prepare_relocation_batch",
        "orbitkv_session_prepare_request_fork",
        "orbitkv_session_quarantine_control",
        "orbitkv_session_quarantine_relocation",
        "orbitkv_session_read_control_plan",
        "orbitkv_session_submit_relocation",
        "orbitkv_session_token_views_batch",
    }
)

MATRIX_LEVELS = (
    "L1 Compiler",
    "L2 Host/ABI",
    "L3 GPU Primitive",
    "L4 Engine E2E",
    "L5 Production",
)
MATRIX_REQUIRED_CLAIMS = (
    "Current typed C wire",
    "Current Python FFI/runtime",
    "Exactly 48 typed symbols",
    "RuntimeManifest",
    "RuntimeTarget",
    "RuntimeBinding",
    "WIRE_VERSION = 14",
    "exactly 78 ctypes layouts",
    'id = "sglang"',
    "contract_version = 4",
    "required_wire_version = 14",
    "source contract",
    "qualification_runner.py",
    "Results Index",
)

WEBSITE_TEXT_SUFFIXES = frozenset(
    {".astro", ".css", ".js", ".jsx", ".md", ".mdx", ".ts", ".tsx"}
)

# These identities belong in the append-only Results Index, not in the active
# public description of the current capability surface. Generic terms such as
# "typed wire", "runtime", "latent KV", and "fixed state" remain valid.
BANNED_PUBLIC_IDENTITY_PATTERNS = (
    (
        "numbered ABI generation",
        re.compile(
            r"(?<![a-z0-9])abi[\s._-]*v?[\s._-]*(?:5|8|9)(?![0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "device identity H20",
        re.compile(r"(?<![a-z0-9])h20(?![a-z0-9])", re.IGNORECASE),
    ),
    (
        "pinned engine version v0.5.17/v0517",
        re.compile(
            r"(?<![a-z0-9])v0[._-]?5[._-]?17(?![0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "model brand or model-specific path",
        re.compile(
            r"qwen|gpt[\s._-]*oss|deepseek|mistral|mixtral|llama|gemma",
            re.IGNORECASE,
        ),
    ),
)

BANNED_PUBLIC_CONTRACT_PATTERNS = (
    (
        "retired RuntimeManifest version branch",
        re.compile(
            r"\bruntimemanifest\s+v(?:1|2)\b|"
            r"\bruntimemanifestv(?:1|2)\b|"
            r"\bruntime_manifest_v(?:1|2)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "retired version-suffixed runtime type",
        re.compile(
            r"\b(?:RuntimeTargetContractV1|RuntimeTargetBindingV1|"
            r"ExecutionTopologyV1|RuntimeAdmissionProfileV1)\b"
        ),
    ),
    (
        "retired runtime-target schema or artifact name",
        re.compile(
            r"orbitkv\.runtime-target-(?:contract|binding)|"
            r"\bruntime-target-binding(?:\.json)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "retired runtime-target CLI flag",
        re.compile(r"--executor-capabilities\b", re.IGNORECASE),
    ),
    (
        "retired runtime-target resource",
        re.compile(r"\bexecutor_capabilities\.v1\.json\b", re.IGNORECASE),
    ),
)

RESULTS_PATH_PATTERN = re.compile(
    r"(?<![a-z0-9_.-])results/(?P<target>[a-z0-9][a-z0-9._/-]*)",
    re.IGNORECASE,
)
RESULTS_INDEX_LINK_PATTERN = re.compile(
    r"\[Results Index\]\((?:\.\./)?results/README\.md(?:#[^)]+)?\)",
    re.IGNORECASE,
)
ALLOWED_ACTIVE_RESULTS_TARGETS = frozenset(
    {
        "readme.md",
        "full-engine-e2e-alternating-20260901/readme.md",
    }
)
WIRE_SYMBOL_PATTERN = re.compile(r"\b(orbitkv_[a-z0-9_]+)\s*\(")
RETIRED_RAW_MANAGER_SYMBOL_PREFIX = "orbitkv_manager_"
WIRE_FENCE_PATTERN = re.compile(
    r"### Exact current C wire surface.*?```text\s*(?P<symbols>.*?)```",
    re.DOTALL,
)
HEADER_WIRE_VERSION_PATTERN = re.compile(
    r"^#define\s+ORBITKV_WIRE_VERSION\s+(?P<version>[0-9]+)u?\s*$",
    re.MULTILINE,
)
RUST_TARGET_CONSTANT_PATTERNS = {
    "contract_version": re.compile(
        r"^const\s+SGLANG_TARGET_CONTRACT_VERSION:\s*u32\s*=\s*(?P<value>[0-9]+);$",
        re.MULTILINE,
    ),
    "required_wire_version": re.compile(
        r"^const\s+REQUIRED_WIRE_VERSION:\s*u32\s*=\s*(?P<value>[0-9]+);$",
        re.MULTILINE,
    ),
}
CHUNKED_FAIL_CLOSED_PATTERN = re.compile(
    r"\bfail(?:s|ing)?[- ]closed\b", re.IGNORECASE
)
CHUNKED_HOST_OBSERVATION_PATTERN = re.compile(
    r"\bhost checks?\b.{0,160}\bdo(?:es)? not observe\b"
    r".{0,160}\b(?:kernel|scheduler)\b",
    re.IGNORECASE,
)
CHUNKED_EXCLUSION_PATTERNS = (
    ("real-accelerator exclusion", re.compile(r"\breal[- ]accelerator\b", re.I)),
    ("released-model exclusion", re.compile(r"\breleased[- ]model\b", re.I)),
    ("performance exclusion", re.compile(r"\bperformance\b", re.I)),
    (
        "complete-engine exclusion",
        re.compile(r"\bcomplete[- ]engine qualification\b", re.I),
    ),
)

# Keep an explicit migration guard in addition to the generic path-name scan.
# These paths were public identity-specific entry points and must not reappear.
RETIRED_PUBLIC_PATHS = (
    "docs/abi5-sglang-batch-adapter.md",
    "core/examples/deepseek-v2-lite-mla-state-plan.json",
    "core/examples/deepseek-v2-lite-mla.json",
    "core/examples/gpt_oss_20b_retention.json",
    "core/examples/gpt_oss_hybrid_62l.json",
    "core/examples/gpt_oss_hybrid_tiny.json",
    "core/examples/mistral_uniform_swa.json",
    "core/examples/qwen3.5-0.8b-attention-state-input-page16-bf16.json",
    "core/examples/qwen3.5-0.8b-token-manager-page16-bf16.json",
    "core/fixtures/gpt-oss-hybrid-62l/config.json",
    "core/fixtures/gpt-oss-hybrid-tiny/config.json",
    "core/fixtures/mistral-uniform-swa-tiny/config.json",
    "core/fixtures/qwen2.5-full-tiny/config.json",
    "core/fixtures/qwen3.5-0.8b/config.json",
    "core/fixtures/qwen3.8-27b/PROVENANCE.md",
    "core/fixtures/qwen3.8-27b/config.json",
)

# These replacements are stable capability entry points, not aliases for one
# model, device, engine release, or wire generation. Keeping the inventory in
# the gate prevents a rename-only migration from silently dropping an example.
REQUIRED_CAPABILITY_PATHS = (
    "core/examples/full-token-manager-plan.json",
    "core/examples/hybrid-fixed-state-attention-state-plan.json",
    "core/examples/latent-kv-attention-state-plan.json",
    "core/examples/latent-kv-token-manager-plan.json",
    "core/examples/sliding-token-manager-plan.json",
    "core/fixtures/hybrid-fixed-state-large/PROVENANCE.md",
    "core/fixtures/hybrid-fixed-state-large/config.json",
    "core/fixtures/hybrid-fixed-state-small/config.json",
)


def _files_below(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.rglob("*") if path.is_file()))


def discover_public_prose_surfaces(root: Path = ROOT) -> tuple[Path, ...]:
    docs = tuple(sorted((root / "docs").rglob("*.md")))
    website = tuple(
        path
        for path in _files_below(root / "website/src")
        if path.suffix.lower() in WEBSITE_TEXT_SUFFIXES
    )
    return (root / "README.md", *docs, *website)


def discover_example_fixture_surfaces(root: Path = ROOT) -> tuple[Path, ...]:
    return (
        *_files_below(root / "core/examples"),
        *_files_below(root / "core/fixtures"),
    )


def discover_public_active_surfaces(root: Path = ROOT) -> tuple[Path, ...]:
    return (
        *discover_public_prose_surfaces(root),
        *discover_example_fixture_surfaces(root),
    )


PUBLIC_PROSE_SURFACES = discover_public_prose_surfaces()
EXAMPLE_FIXTURE_SURFACES = discover_example_fixture_surfaces()
PUBLIC_ACTIVE_SURFACES = (*PUBLIC_PROSE_SURFACES, *EXAMPLE_FIXTURE_SURFACES)


def _is_results_index(path: Path) -> bool:
    return len(path.parts) >= 2 and path.parts[-2:] == ("results", "README.md")


def _results_targets(text: str) -> tuple[str, ...]:
    return tuple(match.group("target") for match in RESULTS_PATH_PATTERN.finditer(text))


def _normalized_results_target(target: str) -> str:
    return target.rstrip("/.,;:!?").casefold()


def verify_results_links(text: str, path: Path) -> tuple[str, ...]:
    targets = _results_targets(text)
    for target in targets:
        if _normalized_results_target(target) not in ALLOWED_ACTIVE_RESULTS_TARGETS:
            raise RuntimeError(
                "active public surface links a non-allowlisted results archive: "
                f"{path}: results/{target}"
            )
    return targets


def verify_public_prose(text: str, path: Path) -> None:
    if _is_results_index(path):
        return

    for identity, pattern in (
        *BANNED_PUBLIC_IDENTITY_PATTERNS,
        *BANNED_PUBLIC_CONTRACT_PATTERNS,
    ):
        match = pattern.search(text)
        if match is not None:
            raise RuntimeError(
                f"active public surface exposes {identity}: {path}: {match.group(0)}"
            )
    verify_results_links(text, path)


def verify_public_surface(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    verify_public_prose(text, path)
    return text


def _chunked_executor_rows(matrix: str) -> list[str]:
    rows: list[str] = []
    for line in matrix.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        first_cell = stripped.strip("|").split("|", maxsplit=1)[0]
        lowered = first_cell.casefold()
        if "chunked" in lowered and "executor" in lowered:
            rows.append(stripped)
    return rows


def _matrix_wire_symbols(matrix: str) -> tuple[str, ...]:
    match = WIRE_FENCE_PATTERN.search(matrix)
    if match is None:
        raise RuntimeError(
            "Capability Matrix is missing the exact current C wire surface"
        )
    symbols = tuple(
        line.strip()
        for line in match.group("symbols").splitlines()
        if line.strip()
    )
    malformed = tuple(
        symbol
        for symbol in symbols
        if re.fullmatch(r"orbitkv_[a-z0-9_]+", symbol) is None
    )
    if malformed:
        raise RuntimeError(
            "Capability Matrix current C wire contains malformed symbols: "
            + ", ".join(malformed)
        )
    if len(symbols) != len(set(symbols)):
        raise RuntimeError(
            "Capability Matrix current C wire contains duplicate symbols"
        )
    retired = tuple(
        symbol
        for symbol in symbols
        if symbol.startswith(RETIRED_RAW_MANAGER_SYMBOL_PREFIX)
    )
    if retired:
        raise RuntimeError(
            "Capability Matrix current C wire exposes retired raw manager symbols: "
            + ", ".join(retired)
        )
    if len(symbols) != CURRENT_SYMBOL_COUNT:
        raise RuntimeError(
            "Capability Matrix current C wire must list exactly "
            f"{CURRENT_SYMBOL_COUNT} symbols, found {len(symbols)}"
        )
    missing = REQUIRED_SESSION_SYMBOLS.difference(symbols)
    if missing:
        raise RuntimeError(
            "Capability Matrix current C wire is missing required session wire symbols: "
            + ", ".join(sorted(missing))
        )
    return symbols


def _ast_assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return node.value
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node.value
    raise RuntimeError(f"wire source is missing assignment {name}")


def _integer_assignment(source: str, name: str, label: str) -> int:
    value = _ast_assignment(ast.parse(source), name)
    if not isinstance(value, ast.Constant) or type(value.value) is not int:
        raise RuntimeError(f"{label} {name} is not a literal integer")
    return value.value


def _library_wire_symbols(source: str) -> frozenset[str]:
    value = _ast_assignment(ast.parse(source), "FUNCTION_SPECS")
    if not isinstance(value, ast.Dict):
        raise RuntimeError("Python wire FUNCTION_SPECS is not a literal mapping")
    symbols: list[str] = []
    for key in value.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise RuntimeError(
                "Python wire FUNCTION_SPECS contains a non-literal symbol"
            )
        symbols.append(key.value)
    if len(symbols) != len(set(symbols)):
        raise RuntimeError("Python wire FUNCTION_SPECS contains duplicate symbols")
    return frozenset({"orbitkv_wire_version", *symbols})


def _frozen_layouts(tree: ast.Module) -> dict[str, tuple[int, int]]:
    value = _ast_assignment(tree, "FROZEN_LAYOUTS")
    if not isinstance(value, ast.Dict):
        raise RuntimeError("Python wire FROZEN_LAYOUTS is not a literal mapping")
    layouts: dict[str, tuple[int, int]] = {}
    for key, raw_shape in zip(value.keys, value.values, strict=True):
        if (
            not isinstance(key, ast.Name)
            or not isinstance(raw_shape, ast.Tuple)
            or len(raw_shape.elts) != 2
            or any(
                not isinstance(item, ast.Constant)
                or type(item.value) is not int
                for item in raw_shape.elts
            )
        ):
            raise RuntimeError(
                "Python wire FROZEN_LAYOUTS contains a non-literal layout"
            )
        if key.id in layouts:
            raise RuntimeError(
                "Python wire FROZEN_LAYOUTS contains a duplicate layout"
            )
        layouts[key.id] = tuple(item.value for item in raw_shape.elts)  # type: ignore[misc]
    return layouts


def _ctypes_layout_fields(tree: ast.Module, name: str) -> tuple[str, ...]:
    layout = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        ),
        None,
    )
    if layout is None:
        raise RuntimeError(f"Python wire is missing ctypes layout {name}")
    fields = _ast_assignment(ast.Module(body=layout.body, type_ignores=[]), "_fields_")
    if not isinstance(fields, (ast.List, ast.Tuple)):
        raise RuntimeError(f"Python wire {name}._fields_ is not a literal sequence")
    names: list[str] = []
    for field in fields.elts:
        if (
            not isinstance(field, ast.Tuple)
            or len(field.elts) != 2
            or not isinstance(field.elts[0], ast.Constant)
            or not isinstance(field.elts[0].value, str)
        ):
            raise RuntimeError(f"Python wire {name} contains a non-literal field")
        names.append(field.elts[0].value)
    return tuple(names)


def _canonical_fingerprint(value: dict[str, object]) -> str:
    payload = {name: item for name, item in value.items() if name != "fingerprint"}
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def verify_current_wire_contract(
    matrix: str,
    *,
    header_path: Path = WIRE_HEADER,
    library_path: Path = WIRE_LIBRARY,
    layouts_path: Path = WIRE_LAYOUTS,
    target_path: Path = RUNTIME_TARGET,
    rust_target_path: Path = RUST_RUNTIME_TARGET,
    expected_target_fingerprint: str | None = CURRENT_TARGET_FINGERPRINT,
) -> None:
    matrix_symbols = frozenset(_matrix_wire_symbols(matrix))

    header = header_path.read_text(encoding="utf-8")
    version_match = HEADER_WIRE_VERSION_PATTERN.search(header)
    if version_match is None:
        raise RuntimeError("C wire header is missing ORBITKV_WIRE_VERSION")
    header_version = int(version_match.group("version"))
    header_symbols = frozenset(WIRE_SYMBOL_PATTERN.findall(header))
    for marker in REQUIRED_CURRENT_WIRE_HEADER_MARKERS:
        if marker not in header:
            raise RuntimeError(
                f"C wire header is missing explicit cache-policy marker {marker!r}"
            )

    library = library_path.read_text(encoding="utf-8")
    library_version = _integer_assignment(library, "WIRE_VERSION", "Python wire")
    library_symbols = _library_wire_symbols(library)

    for label, version in (
        ("C header", header_version),
        ("Python loader", library_version),
    ):
        if version != CURRENT_WIRE_VERSION:
            raise RuntimeError(
                f"{label} wire version must be exactly {CURRENT_WIRE_VERSION}, "
                f"found {version}"
            )
    for label, symbols in (
        ("C header", header_symbols),
        ("Python loader", library_symbols),
        ("Capability Matrix", matrix_symbols),
    ):
        if len(symbols) != CURRENT_SYMBOL_COUNT:
            raise RuntimeError(
                f"{label} must expose exactly {CURRENT_SYMBOL_COUNT} wire symbols, "
                f"found {len(symbols)}"
            )
        missing = REQUIRED_SESSION_SYMBOLS.difference(symbols)
        if missing:
            raise RuntimeError(
                f"{label} is missing required session wire symbols: "
                + ", ".join(sorted(missing))
            )
    if header_symbols != library_symbols or header_symbols != matrix_symbols:
        raise RuntimeError(
            "C header, Python loader, and Capability Matrix wire symbols differ"
        )

    layouts_tree = ast.parse(layouts_path.read_text(encoding="utf-8"))
    layouts = _frozen_layouts(layouts_tree)
    if len(layouts) != CURRENT_LAYOUT_COUNT:
        raise RuntimeError(
            "Python wire must freeze exactly "
            f"{CURRENT_LAYOUT_COUNT} layouts, found {len(layouts)}"
        )
    layout_name, expected_shape = SESSION_CREATE_LAYOUT
    if layouts.get(layout_name) != expected_shape:
        raise RuntimeError(
            f"Python wire {layout_name} must freeze layout {expected_shape}"
        )
    if _ctypes_layout_fields(layouts_tree, layout_name) != SESSION_CREATE_FIELDS:
        raise RuntimeError(
            f"Python wire {layout_name} fields must be {SESSION_CREATE_FIELDS}"
        )

    try:
        target = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read current runtime target: {error}") from error
    expected_identity = {
        "id": CURRENT_TARGET_ID,
        "contract_version": CURRENT_TARGET_CONTRACT_VERSION,
    }
    if target.get("target") != expected_identity:
        raise RuntimeError(
            "current runtime target identity must remain exactly "
            f"{expected_identity!r}"
        )
    if target.get("required_wire_version") != CURRENT_WIRE_VERSION:
        raise RuntimeError(
            "current runtime target must require wire version "
            f"{CURRENT_WIRE_VERSION}"
        )
    rust_target = rust_target_path.read_text(encoding="utf-8")
    rust_values: dict[str, int] = {}
    for name, pattern in RUST_TARGET_CONSTANT_PATTERNS.items():
        match = pattern.search(rust_target)
        if match is None:
            raise RuntimeError(f"Rust runtime target is missing {name}")
        rust_values[name] = int(match.group("value"))
    expected_rust_values = {
        "contract_version": CURRENT_TARGET_CONTRACT_VERSION,
        "required_wire_version": CURRENT_WIRE_VERSION,
    }
    if rust_values != expected_rust_values:
        raise RuntimeError(
            "Rust and packaged runtime target versions differ: "
            f"rust={rust_values!r}, expected={expected_rust_values!r}"
        )
    if target.get("fingerprint") != _canonical_fingerprint(target):
        raise RuntimeError(
            "current runtime target fingerprint does not match its canonical payload"
        )
    if (
        expected_target_fingerprint is not None
        and target["fingerprint"] != expected_target_fingerprint
    ):
        raise RuntimeError(
            "current runtime target fingerprint differs from the frozen contract"
        )


def verify_matrix(matrix: str) -> None:
    verify_public_prose(matrix, MATRIX)

    for level in MATRIX_LEVELS:
        if level not in matrix:
            raise RuntimeError(f"Capability Matrix is missing {level}")

    folded_matrix = matrix.casefold()
    for claim in MATRIX_REQUIRED_CLAIMS:
        if claim == f"WIRE_VERSION = {CURRENT_WIRE_VERSION}":
            present = re.search(
                rf"(?<![a-z0-9_])WIRE_VERSION\s*=\s*{CURRENT_WIRE_VERSION}\b",
                matrix,
                re.IGNORECASE,
            ) is not None
        else:
            present = claim.casefold() in folded_matrix
        if not present:
            raise RuntimeError(
                f"Capability Matrix is missing stable boundary: {claim}"
            )

    _matrix_wire_symbols(matrix)

    if RESULTS_INDEX_LINK_PATTERN.search(matrix) is None:
        raise RuntimeError(
            "Capability Matrix must route historical identities through the "
            "Results Index"
        )

    chunked_rows = _chunked_executor_rows(matrix)
    if len(chunked_rows) != 1:
        raise RuntimeError(
            "Capability Matrix must contain exactly one chunked executor row"
        )
    chunked_row = chunked_rows[0]
    if CHUNKED_FAIL_CLOSED_PATTERN.search(chunked_row) is None:
        raise RuntimeError(
            "Capability Matrix chunked executor must state its fail-closed boundary"
        )
    if CHUNKED_HOST_OBSERVATION_PATTERN.search(chunked_row) is None:
        raise RuntimeError(
            "Capability Matrix chunked executor must state its host-only "
            "observation boundary"
        )
    for boundary, pattern in CHUNKED_EXCLUSION_PATTERNS:
        if pattern.search(chunked_row) is None:
            raise RuntimeError(
                f"Capability Matrix chunked executor is missing {boundary}"
            )


def _relative_surface(path: Path, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        return candidate.relative_to(root)
    except ValueError:
        return path


def verify_no_retired_public_paths(root: Path = ROOT) -> None:
    for relative in RETIRED_PUBLIC_PATHS:
        if (root / relative).exists():
            raise RuntimeError(f"retired identity-specific public path exists: {relative}")


def verify_required_capability_paths(root: Path = ROOT) -> None:
    for relative in REQUIRED_CAPABILITY_PATHS:
        if not (root / relative).is_file():
            raise RuntimeError(f"required capability-generic path is missing: {relative}")


def verify_example_fixture_paths(
    paths: Iterable[Path], root: Path = ROOT
) -> None:
    verify_no_retired_public_paths(root)
    for path in paths:
        relative = _relative_surface(path, root).as_posix()
        for identity, pattern in BANNED_PUBLIC_IDENTITY_PATTERNS:
            match = pattern.search(relative)
            if match is not None:
                raise RuntimeError(
                    "example/fixture path is not capability-generic "
                    f"({identity}): {relative}"
                )


def verify_website_evidence(path: Path = WEBSITE_EVIDENCE) -> None:
    text = verify_public_surface(path)
    targets = _results_targets(text)
    if not targets:
        raise RuntimeError(
            "website evidence must link the Results Index at results/README.md"
        )
    if any(
        _normalized_results_target(target) not in ALLOWED_ACTIVE_RESULTS_TARGETS
        for target in targets
    ):
        raise RuntimeError(
            "website evidence links a non-allowlisted results archive"
        )


def main() -> None:
    verify_no_retired_public_paths()
    verify_required_capability_paths()
    matrix = MATRIX.read_text(encoding="utf-8")
    verify_matrix(matrix)
    verify_current_wire_contract(matrix)

    prose_surfaces = discover_public_prose_surfaces()
    for path in prose_surfaces:
        verify_public_surface(path)

    example_fixture_surfaces = discover_example_fixture_surfaces()
    verify_example_fixture_paths(example_fixture_surfaces)
    verify_website_evidence()

    print(
        "verified generic Capability Matrix boundary: "
        f"5 levels, {len(prose_surfaces)} active prose surfaces, "
        f"{len(example_fixture_surfaces)} capability-generic example/fixture paths"
    )


if __name__ == "__main__":
    main()
