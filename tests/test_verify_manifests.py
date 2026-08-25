from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/verify_manifests.py"
SPEC = importlib.util.spec_from_file_location("verify_manifests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Abi8ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "evidence.txt").write_bytes(b"sealed evidence\n")
        self.manifest = {
            "schema": verifier.ABI8_H20_SCHEMA,
            "qualification_status": verifier.ABI8_H20_QUALIFICATION_STATUS,
            "scope": {
                "cases": deepcopy(verifier.ABI8_H20_CASES),
                "excluded": list(verifier.ABI8_H20_EXCLUSIONS),
            },
            "abi_version": 8,
            "exact_symbol_count": 40,
            "pair_count": 12,
            "epoch_count": 3,
            "performance_go": False,
            "artifacts": {
                "evidence.txt": digest(b"sealed evidence\n"),
            },
        }
        self.reseal()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def reseal(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sums = dict(self.manifest["artifacts"])
        sums["manifest.json"] = digest(self.manifest_path.read_bytes())
        (self.root / "SHA256SUMS").write_text(
            "".join(
                f"{file_digest}  {relative_path}\n"
                for relative_path, file_digest in sorted(sums.items())
            ),
            encoding="utf-8",
        )

    def test_accepts_self_contained_seal(self) -> None:
        with patch.object(
            verifier, "verify_abi8_h20_semantics", return_value=1
        ) as semantics:
            self.assertEqual(verifier.verify_manifest(self.manifest_path), 3)
        semantics.assert_called_once_with(self.root, self.manifest)

    def test_rejects_a_hash_consistent_but_semantically_fake_seal(self) -> None:
        with patch.object(
            verifier,
            "verify_abi8_h20_semantics",
            side_effect=RuntimeError("fabricated evidence"),
        ):
            with self.assertRaisesRegex(RuntimeError, "fabricated evidence"):
                verifier.verify_manifest(self.manifest_path)

    def test_rejects_changed_missing_and_unlisted_artifacts(self) -> None:
        mutations = (
            lambda: (self.root / "evidence.txt").write_bytes(b"changed"),
            lambda: (self.root / "evidence.txt").unlink(),
            lambda: (self.root / "extra.txt").write_bytes(b"extra"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                with tempfile.TemporaryDirectory() as temporary:
                    original_root = self.root
                    self.root = Path(temporary)
                    (self.root / "evidence.txt").write_bytes(b"sealed evidence\n")
                    self.reseal()
                    mutate()
                    with self.assertRaises(RuntimeError):
                        verifier.verify_manifest(self.manifest_path)
                    self.root = original_root

    def test_rejects_unsafe_artifact_path(self) -> None:
        self.manifest["artifacts"] = {"../evidence.txt": digest(b"sealed evidence\n")}
        self.reseal()
        with self.assertRaises(RuntimeError):
            verifier.verify_manifest(self.manifest_path)

    def test_rejects_duplicate_artifact_path(self) -> None:
        manifest_text = self.manifest_path.read_text(encoding="utf-8")
        evidence_digest = digest(b"sealed evidence\n")
        artifact_line = f'    "evidence.txt": "{evidence_digest}"'
        self.assertIn(artifact_line, manifest_text)
        self.manifest_path.write_text(
            manifest_text.replace(artifact_line, f"{artifact_line},\n{artifact_line}"),
            encoding="utf-8",
        )
        with self.assertRaises(RuntimeError):
            verifier.verify_manifest(self.manifest_path)

    def test_rejects_symlink_artifact(self) -> None:
        (self.root / "evidence.txt").unlink()
        (self.root / "target.txt").write_bytes(b"sealed evidence\n")
        (self.root / "evidence.txt").symlink_to("target.txt")
        with self.assertRaises(RuntimeError):
            verifier.verify_manifest(self.manifest_path)

    def test_rejects_duplicate_or_noncanonical_sha256sums(self) -> None:
        sums_path = self.root / "SHA256SUMS"
        sums_path.write_text(
            sums_path.read_text(encoding="utf-8")
            + f"{digest(b'duplicate')}  evidence.txt\n",
            encoding="utf-8",
        )
        with self.assertRaises(RuntimeError):
            verifier.verify_manifest(self.manifest_path)

    def test_rejects_unknown_sealed_schema(self) -> None:
        self.manifest["schema"] = "orbitkv.abi8-h20-sealed-manifest.v2"
        del self.manifest["artifacts"]
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(RuntimeError):
            verifier.verify_manifest(self.manifest_path)

    def test_default_manifest_set_requires_the_abi8_seal(self) -> None:
        self.assertIn(verifier.ABI8_H20_MANIFEST, verifier.DEFAULT_MANIFESTS)

    def test_rejects_wrong_qualification_policy(self) -> None:
        invalid_values = {
            "abi_version": True,
            "exact_symbol_count": 39,
            "pair_count": 11,
            "epoch_count": 2,
            "performance_go": True,
            "qualification_status": "unqualified",
        }
        for field, invalid in invalid_values.items():
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = invalid
                self.reseal()
                with self.assertRaises(RuntimeError):
                    verifier.verify_manifest(self.manifest_path)
                self.manifest[field] = original

    def test_rejects_changed_cases_or_exclusions(self) -> None:
        for field in ("cases", "excluded"):
            with self.subTest(field=field):
                original = self.manifest["scope"][field]
                self.manifest["scope"][field] = original[:-1]
                self.reseal()
                with self.assertRaises(RuntimeError):
                    verifier.verify_manifest(self.manifest_path)
                self.manifest["scope"][field] = original


class Qwen35PairEvidenceManifestTest(unittest.TestCase):
    def test_default_manifest_routes_to_trusted_verifier(self) -> None:
        self.assertIn(
            verifier.QWEN35_H20_PAIR_EVIDENCE_MANIFEST,
            verifier.DEFAULT_MANIFESTS,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"schema": verifier.QWEN35_H20_PAIR_EVIDENCE_SCHEMA}
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier,
                "verify_qwen35_h20_pair_evidence",
                return_value=6,
            ) as trusted_verifier:
                self.assertEqual(verifier.verify_manifest(manifest_path), 6)
            trusted_verifier.assert_called_once_with(root)

    def test_qwen35_router_rejects_noncanonical_manifest_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forged.json"
            path.write_text(
                json.dumps(
                    {"schema": verifier.QWEN35_H20_PAIR_EVIDENCE_SCHEMA}
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier, "verify_qwen35_h20_pair_evidence"
            ) as trusted_verifier:
                with self.assertRaisesRegex(
                    RuntimeError, "must be named manifest.json"
                ):
                    verifier.verify_manifest(path)
            trusted_verifier.assert_not_called()


class Qwen38DiagnosticManifestTest(unittest.TestCase):
    def test_default_manifest_routes_to_pinned_diagnostic_verifier(self) -> None:
        self.assertIn(
            verifier.QWEN38_H20_DIAGNOSTIC_MANIFEST,
            verifier.DEFAULT_MANIFESTS,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"schema": verifier.QWEN38_H20_DIAGNOSTIC_SCHEMA}
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier,
                "verify_qwen38_h20_diagnostic",
                return_value=43,
            ) as trusted_verifier:
                self.assertEqual(verifier.verify_manifest(manifest_path), 43)
            trusted_verifier.assert_called_once_with(root)

    def test_qwen38_router_rejects_noncanonical_manifest_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forged.json"
            path.write_text(
                json.dumps(
                    {"schema": verifier.QWEN38_H20_DIAGNOSTIC_SCHEMA}
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier, "verify_qwen38_h20_diagnostic"
            ) as trusted_verifier:
                with self.assertRaisesRegex(
                    RuntimeError, "must be named manifest.json"
                ):
                    verifier.verify_manifest(path)
            trusted_verifier.assert_not_called()


class TokenRelocationDiagnosticManifestTest(unittest.TestCase):
    def test_default_manifest_routes_to_trusted_verifier(self) -> None:
        self.assertIn(
            verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST,
            verifier.DEFAULT_MANIFESTS,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_SCHEMA
                        )
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier,
                "verify_token_relocation_h20_diagnostic",
                return_value=19,
            ) as trusted_verifier:
                self.assertEqual(verifier.verify_manifest(manifest_path), 19)
            trusted_verifier.assert_called_once_with(root)

    def test_router_rejects_noncanonical_manifest_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forged.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": (
                            verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_SCHEMA
                        )
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier, "verify_token_relocation_h20_diagnostic"
            ) as trusted_verifier:
                with self.assertRaisesRegex(
                    RuntimeError, "must be named manifest.json"
                ):
                    verifier.verify_manifest(path)
            trusted_verifier.assert_not_called()

    def test_archive_manifest_retains_diagnostic_boundary(self) -> None:
        manifest = verifier.load_json(
            verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST
        )
        self.assertIsInstance(manifest, dict)
        for field, expected in (
            ("evidence_class", "diagnostic_only"),
            ("diagnostic_only", True),
            ("sealed", False),
            ("source_dirty", True),
            ("hardware_attested", False),
            ("qualified", False),
            ("performance_go", False),
        ):
            with self.subTest(field=field):
                self.assertIs(type(manifest[field]), type(expected))
                self.assertEqual(manifest[field], expected)
        self.assertEqual(manifest["artifact_count"], 19)
        self.assertEqual(len(manifest["artifacts"]), 19)
        self.assertEqual(
            sum(name.startswith("records/") for name in manifest["artifacts"]),
            16,
        )
        self.assertEqual(
            manifest["integrity_scope"],
            "all_payload_artifacts_except_manifest_and_checksum_index",
        )

    def test_component_conformance_binds_recorded_device(self) -> None:
        manifest = verifier.load_json(
            verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST
        )
        verifier.verify_token_relocation_component_conformance(
            verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST.parent
            / "component-conformance.xml",
            manifest["observed_hardware"],
        )
        with self.assertRaisesRegex(RuntimeError, "device identity differs"):
            verifier.verify_token_relocation_component_conformance(
                verifier.TOKEN_RELOCATION_H20_DIAGNOSTIC_MANIFEST.parent
                / "component-conformance.xml",
                {"name": "different GPU", "uuid": "GPU-forged"},
            )


class TokenRelocationSealedManifestTest(unittest.TestCase):
    def test_default_manifest_includes_sealed_relocation_archive(self) -> None:
        self.assertIn(
            verifier.TOKEN_RELOCATION_H20_SEALED_MANIFEST,
            verifier.DEFAULT_MANIFESTS,
        )

    def sealed_result(self) -> dict[str, object]:
        return {
            "schema": (
                "orbitkv.abi8-h20-token-relocation-seal-verification.v1"
            ),
            "status": "passed",
            "qualification_status": (
                "abi8_sglang_full_token_relocation_correctness_lifecycle_"
                "qualified_performance_pending"
            ),
            "qualification_claim": (
                "scoped_correctness_and_lifecycle_only"
            ),
            "sealed": True,
            "source_clean": True,
            "preflight_bound": True,
            "hardware_attested": False,
            "qualified": True,
            "performance_go": False,
            "epoch_count": 4,
            "record_count": 16,
            "pair_count": 8,
            "abi_version": 8,
            "exact_symbol_count": 40,
            "all_pairs_passed": True,
            "exact_token_equality": True,
            "manager_census_fully_drained": True,
            "failure_and_quarantine_counters_zero": True,
        }

    def test_any_archive_manifest_routes_to_trusted_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "portable-archive"
            root.mkdir()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"schema": verifier.TOKEN_RELOCATION_H20_SEALED_SCHEMA}
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier,
                "verify_token_relocation_h20_sealed",
                return_value=8,
            ) as trusted_verifier:
                self.assertEqual(verifier.verify_manifest(manifest_path), 8)
            trusted_verifier.assert_called_once_with(root)

    def test_router_rejects_noncanonical_manifest_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forged.json"
            path.write_text(
                json.dumps(
                    {"schema": verifier.TOKEN_RELOCATION_H20_SEALED_SCHEMA}
                ),
                encoding="utf-8",
            )
            with patch.object(
                verifier, "verify_token_relocation_h20_sealed"
            ) as trusted_verifier:
                with self.assertRaisesRegex(
                    RuntimeError, "must be named manifest.json"
                ):
                    verifier.verify_manifest(path)
            trusted_verifier.assert_not_called()

    def test_unknown_sealed_relocation_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "orbitkv.abi8-h20-token-relocation-"
                            "sealed-manifest.v2"
                        )
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "unsupported sealed manifest schema"
            ):
                verifier.verify_manifest(path)

    def test_trusted_sealed_policy_is_conservative_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "portable-archive"
            root.mkdir()
            trusted_path = Path(temporary) / "trusted_verifier.py"
            trusted_path.write_text(
                "import json\n"
                "def verify_sealed_archive(root):\n"
                "    return json.loads("
                "(root / 'trusted-result.json').read_text())\n",
                encoding="utf-8",
            )

            def verify_result(result: object) -> int:
                (root / "trusted-result.json").write_text(
                    json.dumps(result),
                    encoding="utf-8",
                )
                with patch.object(
                    verifier,
                    "TOKEN_RELOCATION_H20_VERIFIER",
                    trusted_path,
                ):
                    return verifier.verify_token_relocation_h20_sealed(root)

            self.assertEqual(verify_result(self.sealed_result()), 8)
            invalid_values = {
                "schema": "untrusted",
                "status": "diagnostic_pair_verification_passed",
                "qualification_status": "unqualified",
                "qualification_claim": "performance_qualified",
                "sealed": False,
                "source_clean": False,
                "preflight_bound": False,
                "hardware_attested": True,
                "qualified": False,
                "performance_go": True,
                "epoch_count": True,
                "record_count": 15,
                "pair_count": 7,
                "abi_version": 9,
                "exact_symbol_count": 41,
                "all_pairs_passed": False,
                "exact_token_equality": False,
                "manager_census_fully_drained": False,
                "failure_and_quarantine_counters_zero": False,
            }
            for field, invalid in invalid_values.items():
                with self.subTest(field=field):
                    result = self.sealed_result()
                    result[field] = invalid
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "trusted token-relocation sealed verification is incomplete",
                    ):
                        verify_result(result)


if __name__ == "__main__":
    unittest.main()
