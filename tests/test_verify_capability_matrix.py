from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


sys.dont_write_bytecode = True
MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/verify_capability_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_capability_matrix", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

WIRE14_SYMBOLS = (
    "orbitkv_session_abort_control",
    "orbitkv_session_abort_prepared",
    "orbitkv_session_abort_prepared_relocation",
    "orbitkv_session_acquire_requests",
    "orbitkv_session_arena_identities",
    "orbitkv_session_arena_stats",
    "orbitkv_session_cancel_pending_attach",
    "orbitkv_session_commit_control",
    "orbitkv_session_complete_execution",
    "orbitkv_session_complete_relocation",
    "orbitkv_session_confirm_control",
    "orbitkv_session_confirm_publication",
    "orbitkv_session_confirm_release",
    "orbitkv_session_confirm_relocation_publication",
    "orbitkv_session_create",
    "orbitkv_session_destroy",
    "orbitkv_session_finalize_pending_attach_cancel",
    "orbitkv_session_mark_token_dispositions_batch",
    "orbitkv_session_prefix_lookup_batch",
    "orbitkv_session_prefix_publish_batch",
    "orbitkv_session_prefix_publish_release_batch",
    "orbitkv_session_prepare_append",
    "orbitkv_session_prepare_prefix_attach",
    "orbitkv_session_prepare_prefix_evict",
    "orbitkv_session_prepare_release",
    "orbitkv_session_prepare_relocation_batch",
    "orbitkv_session_prepare_request_fork",
    "orbitkv_session_quarantine_control",
    "orbitkv_session_quarantine_prepared",
    "orbitkv_session_quarantine_relocation",
    "orbitkv_session_quarantine_submitted",
    "orbitkv_session_read_control_plan",
    "orbitkv_session_stats",
    "orbitkv_session_submit_execution",
    "orbitkv_session_submit_relocation",
    "orbitkv_session_token_views_batch",
    "orbitkv_state_pool_abort_batch",
    "orbitkv_state_pool_acknowledge_batch",
    "orbitkv_state_pool_complete_batch",
    "orbitkv_state_pool_create",
    "orbitkv_state_pool_current_batch",
    "orbitkv_state_pool_destroy",
    "orbitkv_state_pool_identity",
    "orbitkv_state_pool_prepare_batch",
    "orbitkv_state_pool_retire_owners_batch",
    "orbitkv_state_pool_stats",
    "orbitkv_state_pool_submit_batch",
    "orbitkv_wire_version",
)


def valid_matrix() -> str:
    # Intentionally literal: this fixture must not inherit the verifier's own
    # constants and accidentally make a broken requirement self-validating.
    return """# Capability Matrix

| Level | Meaning |
| --- | --- |
| L1 Compiler | checked semantics |
| L2 Host/ABI | checked host boundary |
| L3 GPU Primitive | isolated primitive |
| L4 Engine E2E | scoped engine execution |
| L5 Production | production matrix |

| Capability | Level | Boundary |
| --- | --- | --- |
| Current typed C wire | L2 GO | Exactly 48 typed symbols at WIRE_VERSION = 14. |
| Current Python FFI/runtime | L2 GO | Typed current runtime freezes exactly 78 ctypes layouts. |
| Runtime contracts | L2 GO | RuntimeManifest is admitted by RuntimeTarget and produces RuntimeBinding. The target stores id = "sglang", contract_version = 4, and required_wire_version = 14. |
| Qualification | L4 pending | The source contract is checked by qualification_runner.py. |
| Exact whole-domain chunked executor | P2c host L2 | The validator is failing closed if it cannot prove the contract. These host checks do not observe real kernel or scheduler execution. No real-accelerator, released-model, performance, or complete-engine qualification. |

Historical identities and immutable records are listed only in the
[Results Index](../results/README.md).

### Exact current C wire surface

```text
""" + "\n".join(WIRE14_SYMBOLS) + """
```
"""


def write_file(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_wire_contract(
    tmp_path: Path,
    *,
    wire_version: int = 14,
    symbols: tuple[str, ...] = WIRE14_SYMBOLS,
    layout_count: int = 78,
    target_id: str = "sglang",
    contract_version: int = 4,
) -> tuple[Path, Path, Path, Path, Path]:
    header = write_file(
        tmp_path / "orbitkv.h",
        f"#define ORBITKV_WIRE_VERSION {wire_version}u\n"
        "#define ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE 1u\n"
        "#define ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX 2u\n"
        "const OrbitKvSessionCreateConfig *config;\n"
        + "\n".join(f"int {symbol}(void);" for symbol in symbols),
    )
    library_symbols = tuple(
        symbol for symbol in symbols if symbol != "orbitkv_wire_version"
    )
    library = write_file(
        tmp_path / "library.py",
        f"WIRE_VERSION = {wire_version}\nFUNCTION_SPECS = "
        + repr({symbol: None for symbol in library_symbols})
        + "\n",
    )
    layouts = write_file(
        tmp_path / "layouts.py",
        "class SessionCreateConfigLayout:\n"
        "    _fields_ = [('manager', object), ('cache_sharing_policy', object), ('reserved', object)]\n"
        "FROZEN_LAYOUTS = {\n"
        "    SessionCreateConfigLayout: (40, 8),\n"
        + "".join(
            f"    Layout{index}: (1, 1),\n"
            for index in range(layout_count - 1)
        )
        + "}\n",
    )
    target = write_file(
        tmp_path / "runtime_target.json",
        "",
    )
    target_value: dict[str, object] = {
        "fingerprint": "",
        "target": {
            "id": target_id,
            "contract_version": contract_version,
        },
        "required_wire_version": wire_version,
    }
    target_value["fingerprint"] = verifier._canonical_fingerprint(target_value)
    target.write_text(json.dumps(target_value), encoding="utf-8")
    rust_target = write_file(
        tmp_path / "runtime_target.rs",
        f"const SGLANG_TARGET_CONTRACT_VERSION: u32 = {contract_version};\n"
        f"const REQUIRED_WIRE_VERSION: u32 = {wire_version};\n",
    )
    return header, library, layouts, target, rust_target


def test_matrix_accepts_stable_generation_independent_semantics() -> None:
    verifier.verify_matrix(valid_matrix())


def test_current_wire_contract_is_exactly_14_48_78_and_sglang_v4(
    tmp_path: Path,
) -> None:
    paths = write_wire_contract(tmp_path)

    verifier.verify_current_wire_contract(
        valid_matrix(),
        header_path=paths[0],
        library_path=paths[1],
        layouts_path=paths[2],
        target_path=paths[3],
        rust_target_path=paths[4],
        expected_target_fingerprint=None,
    )
    assert verifier.CURRENT_WIRE_VERSION == 14
    assert verifier.CURRENT_SYMBOL_COUNT == 48
    assert verifier.CURRENT_LAYOUT_COUNT == 78
    assert verifier.CURRENT_TARGET_ID == "sglang"
    assert verifier.CURRENT_TARGET_CONTRACT_VERSION == 4


@pytest.mark.parametrize(
    "wire_version,layout_count,target_id,contract_version,error",
    (
        (13, 78, "sglang", 4, "wire version must be exactly 14"),
        (14, 77, "sglang", 4, "freeze exactly 78 layouts"),
        (14, 78, "sglang@4", 4, "identity must remain exactly"),
        (14, 78, "sglang", 3, "identity must remain exactly"),
    ),
)
def test_current_wire_contract_rejects_stale_counts_or_target_identity(
    tmp_path: Path,
    wire_version: int,
    layout_count: int,
    target_id: str,
    contract_version: int,
    error: str,
) -> None:
    paths = write_wire_contract(
        tmp_path,
        wire_version=wire_version,
        layout_count=layout_count,
        target_id=target_id,
        contract_version=contract_version,
    )

    with pytest.raises(RuntimeError, match=error):
        verifier.verify_current_wire_contract(
            valid_matrix(),
            header_path=paths[0],
            library_path=paths[1],
            layouts_path=paths[2],
            target_path=paths[3],
            rust_target_path=paths[4],
            expected_target_fingerprint=None,
        )


def test_current_wire_contract_rejects_rust_target_drift(tmp_path: Path) -> None:
    paths = write_wire_contract(tmp_path)
    paths[4].write_text(
        "const SGLANG_TARGET_CONTRACT_VERSION: u32 = 3;\n"
        "const REQUIRED_WIRE_VERSION: u32 = 12;\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Rust and packaged runtime target versions differ"):
        verifier.verify_current_wire_contract(
            valid_matrix(),
            header_path=paths[0],
            library_path=paths[1],
            layouts_path=paths[2],
            target_path=paths[3],
            rust_target_path=paths[4],
            expected_target_fingerprint=None,
        )


def test_matrix_requires_all_guarded_session_symbols() -> None:
    assert verifier.REQUIRED_SESSION_SYMBOLS.issubset(WIRE14_SYMBOLS)
    for symbol in verifier.REQUIRED_SESSION_SYMBOLS:
        matrix = valid_matrix().replace(f"{symbol}\n", "", 1)
        with pytest.raises(RuntimeError, match="exactly 48 symbols"):
            verifier.verify_matrix(matrix)


def test_matrix_rejects_a_retired_raw_manager_symbol_at_the_exact_count() -> None:
    matrix = valid_matrix().replace(
        "orbitkv_session_abort_prepared\n",
        "orbitkv_manager_prepare_batch\n",
        1,
    )

    with pytest.raises(RuntimeError, match="retired raw manager symbols"):
        verifier.verify_matrix(matrix)


@pytest.mark.parametrize(
    "claim",
    (
        "L1 Compiler",
        "L2 Host/ABI",
        "L3 GPU Primitive",
        "L4 Engine E2E",
        "L5 Production",
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
    ),
)
def test_matrix_requires_each_stable_semantic_claim(claim: str) -> None:
    matrix = valid_matrix().replace(claim, "removed claim", 1)

    with pytest.raises(RuntimeError, match="Capability Matrix"):
        verifier.verify_matrix(matrix)


@pytest.mark.parametrize(
    "claim, expected",
    (
        ("failing closed", "fail-closed boundary"),
        (
            "These host checks do not observe real kernel or scheduler execution.",
            "host-only observation boundary",
        ),
        ("real-accelerator", "real-accelerator exclusion"),
        ("released-model", "released-model exclusion"),
        ("performance", "performance exclusion"),
        ("complete-engine qualification", "complete-engine exclusion"),
    ),
)
def test_matrix_requires_chunked_fail_closed_boundary_in_its_row(
    claim: str, expected: str
) -> None:
    matrix = valid_matrix().replace(claim, "removed boundary", 1)
    matrix += f"\nUnrelated prose says {claim}.\n"

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        verifier.verify_matrix(matrix)


def test_matrix_requires_exactly_one_chunked_executor_row() -> None:
    row = next(
        line
        for line in valid_matrix().splitlines()
        if line.startswith("| Exact whole-domain chunked executor")
    )

    with pytest.raises(RuntimeError, match="exactly one chunked executor row"):
        verifier.verify_matrix(f"{valid_matrix()}\n{row}\n")


def test_matrix_rejects_a_non_allowlisted_archive_instead_of_requiring_it() -> None:
    matrix = valid_matrix().replace(
        "../results/README.md",
        "../results/scoped-prefix-record/README.md",
    )

    with pytest.raises(RuntimeError, match="non-allowlisted results archive"):
        verifier.verify_matrix(matrix)


@pytest.mark.parametrize(
    "identity",
    (
        "ABI5",
        "ABI 8",
        "ABI-v9",
        "H20",
        "v0.5.17",
        "v0517",
        "Qwen3.5",
        "GPT-OSS",
        "DeepSeek-V2",
        "Mistral",
    ),
)
def test_active_public_prose_rejects_historical_or_model_identity(
    tmp_path: Path, identity: str
) -> None:
    path = tmp_path / "README.md"

    with pytest.raises(RuntimeError, match="active public surface exposes"):
        verifier.verify_public_prose(
            f"Current generic capability description: {identity}.", path
        )


@pytest.mark.parametrize(
    "retired_contract",
    (
        "RuntimeManifest v1",
        "RuntimeManifestV2",
        "runtime_manifest_v2.py",
        "RuntimeTargetContractV1",
        "RuntimeTargetBindingV1",
        "ExecutionTopologyV1",
        "RuntimeAdmissionProfileV1",
        "orbitkv.runtime-target-contract",
        "orbitkv.runtime-target-binding",
        "runtime-target-binding.json",
        "--executor-capabilities",
        "executor_capabilities.v1.json",
    ),
)
def test_active_public_prose_rejects_retired_runtime_contract_vocabulary(
    tmp_path: Path, retired_contract: str
) -> None:
    with pytest.raises(RuntimeError, match="active public surface exposes"):
        verifier.verify_public_prose(
            f"Retired public contract: {retired_contract}.",
            tmp_path / "README.md",
        )


def test_active_public_prose_allows_generic_capability_vocabulary(
    tmp_path: Path,
) -> None:
    verifier.verify_public_prose(
        (
            "The current typed C wire and Python runtime expose latent KV, "
            "fixed-state, and chunked-retention capabilities. RuntimeManifest, "
            "RuntimeTarget, and RuntimeBinding require WIRE_VERSION agreement. Historical "
            "identities are in the Results Index at results/README.md."
        ),
        tmp_path / "README.md",
    )


def test_results_index_is_the_unscanned_historical_identity_entrypoint(
    tmp_path: Path,
) -> None:
    results_index = tmp_path / "results/README.md"
    verifier.verify_public_prose(
        "Historical ABI9, H20, v0.5.17, Qwen, and GPT-OSS records.",
        results_index,
    )


def test_public_prose_allows_the_current_generic_evidence_bundle(
    tmp_path: Path,
) -> None:
    verifier.verify_public_prose(
        "See results/full-engine-e2e-alternating-20260901/README.md.",
        tmp_path / "README.md",
    )


def test_public_prose_rejects_a_non_allowlisted_results_archive(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="non-allowlisted results archive"):
        verifier.verify_public_prose(
            "See results/scoped-hardware-record/README.md.",
            tmp_path / "README.md",
        )


def test_website_evidence_accepts_index_and_current_evidence(tmp_path: Path) -> None:
    evidence = write_file(
        tmp_path / "website/src/pages/evidence.astro",
        '<a href="/repository/blob/main/results/README.md">Results Index</a>'
        '<a href="/repository/blob/main/results/'
        'full-engine-e2e-alternating-20260901/README.md">Current evidence</a>',
    )

    verifier.verify_website_evidence(evidence)


def test_website_evidence_rejects_a_concrete_archive(tmp_path: Path) -> None:
    evidence = write_file(
        tmp_path / "website/src/pages/evidence.astro",
        '<a href="/repository/tree/main/results/scoped-record">record</a>',
    )

    with pytest.raises(RuntimeError, match="non-allowlisted results archive"):
        verifier.verify_website_evidence(evidence)


def test_website_evidence_requires_a_results_index_link(tmp_path: Path) -> None:
    evidence = write_file(
        tmp_path / "website/src/pages/evidence.astro",
        "<p>Historical evidence is indexed separately.</p>",
    )

    with pytest.raises(RuntimeError, match="must link the Results Index"):
        verifier.verify_website_evidence(evidence)


def test_example_and_fixture_paths_must_be_capability_generic(
    tmp_path: Path,
) -> None:
    generic_example = write_file(
        tmp_path / "core/examples/latent-kv-attention-state-plan.json",
        '{"model_type": "deepseek_v2"}',
    )
    generic_fixture = write_file(
        tmp_path / "core/fixtures/hybrid-fixed-state-large/PROVENANCE.md",
        "Source model: Qwen3.8-27B on H20 with the historical ABI8 wire.",
    )

    # Fixture contents retain real frontend discriminators and provenance; the
    # generic-public rule applies only to example/fixture path names.
    verifier.verify_example_fixture_paths(
        (generic_example, generic_fixture), root=tmp_path
    )


@pytest.mark.parametrize(
    "relative",
    (
        "core/examples/qwen-attention-plan.json",
        "core/examples/gpt_oss_retention.json",
        "core/fixtures/h20-hybrid/config.json",
        "core/fixtures/abi9-runtime/config.json",
        "core/fixtures/v0517-engine/config.json",
        "core/fixtures/deepseek-latent-kv/config.json",
    ),
)
def test_example_and_fixture_paths_reject_specific_identities(
    tmp_path: Path, relative: str
) -> None:
    path = write_file(tmp_path / relative, "{}")

    with pytest.raises(RuntimeError, match="path is not capability-generic"):
        verifier.verify_example_fixture_paths((path,), root=tmp_path)


def test_retired_identity_specific_paths_must_not_exist(tmp_path: Path) -> None:
    write_file(tmp_path / "core/fixtures/qwen3.8-27b/config.json", "{}")

    with pytest.raises(RuntimeError, match="retired identity-specific public path exists"):
        verifier.verify_no_retired_public_paths(tmp_path)


def test_required_capability_paths_must_all_exist(tmp_path: Path) -> None:
    for relative in verifier.REQUIRED_CAPABILITY_PATHS:
        write_file(tmp_path / relative, "{}")

    verifier.verify_required_capability_paths(tmp_path)

    missing = tmp_path / verifier.REQUIRED_CAPABILITY_PATHS[-1]
    missing.unlink()
    with pytest.raises(RuntimeError, match="required capability-generic path is missing"):
        verifier.verify_required_capability_paths(tmp_path)


def test_public_active_surface_discovery_covers_the_generic_boundary(
    tmp_path: Path,
) -> None:
    expected = {
        write_file(tmp_path / "README.md", "root"),
        write_file(tmp_path / "docs/capability-matrix.md", "matrix"),
        write_file(tmp_path / "docs/architecture.md", "architecture"),
        write_file(tmp_path / "docs/runtime/admission.md", "admission"),
        write_file(tmp_path / "website/src/pages/index.astro", "page"),
        write_file(tmp_path / "website/src/styles/global.css", "style"),
        write_file(tmp_path / "core/examples/full-token-manager-plan.json", "{}"),
        write_file(tmp_path / "core/examples/full-swa.json", "{}"),
        write_file(tmp_path / "core/fixtures/full-attention/config.json", "{}"),
    }
    results_index = write_file(tmp_path / "results/README.md", "archive")

    discovered = set(verifier.discover_public_active_surfaces(tmp_path))

    assert discovered == expected
    assert results_index not in discovered


def test_discovery_does_not_hide_a_resurrected_retired_document(
    tmp_path: Path,
) -> None:
    retired = write_file(
        tmp_path / "docs/abi5-sglang-batch-adapter.md", "legacy"
    )

    assert retired in verifier.discover_public_prose_surfaces(tmp_path)
    with pytest.raises(RuntimeError, match="retired identity-specific public path exists"):
        verifier.verify_no_retired_public_paths(tmp_path)
