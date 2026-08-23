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


if __name__ == "__main__":
    unittest.main()
