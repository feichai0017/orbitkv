import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "stage_bundle.py"
SPEC = importlib.util.spec_from_file_location("stage_bundle", MODULE_PATH)
stage_bundle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage_bundle)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "examples" / "staging"
ALETHEIA = ROOT / "target" / "debug" / "aletheia-rt"


class StageBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["cargo", "build", "--locked", "-q", "-p", "aletheia-cli"],
            cwd=ROOT,
            check=True,
        )

    def test_fixture_stages_and_is_rust_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = stage_bundle.stage(
                FIXTURES / "reference-plan.json",
                FIXTURES / "reference-evidence.json",
                FIXTURES / "reference-trace.jsonl",
                Path(directory),
                ALETHEIA,
            )
            self.assertTrue((destination / "plan.json").is_file())
            self.assertTrue((destination / "certificate.json").is_file())
            certificate = json.loads((destination / "certificate.json").read_text())
            self.assertEqual(certificate["status"], "qualified")
            self.assertEqual(certificate["performance"]["samples"], 12)
            digest = json.loads((destination / "plan.json").read_text())["steps"][0]["artifact_sha256"]
            self.assertTrue((destination / "artifacts" / digest).is_file())
            staged_evidence = json.loads((destination / "evidence.json").read_text())
            self.assertEqual(staged_evidence["artifacts"][0]["path"], f"artifacts/{digest}")
            stage_bundle.command([str(ALETHEIA)], ["validate-registry", str(Path(directory))])

    def test_bad_artifact_hash_leaves_no_bundle(self):
        plan = json.loads((FIXTURES / "reference-plan.json").read_text())
        evidence = json.loads((FIXTURES / "reference-evidence.json").read_text())
        evidence["artifacts"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            evidence_path = root / "evidence.json"
            plan_path.write_text(json.dumps(plan))
            evidence_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "no file for artifact"):
                stage_bundle.stage(
                    plan_path,
                    evidence_path,
                    FIXTURES / "reference-trace.jsonl",
                    root / "registry",
                    ALETHEIA,
                )
            self.assertFalse((root / "registry" / plan["id"]).exists())

    def test_registry_rejects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = stage_bundle.stage(
                FIXTURES / "reference-plan.json",
                FIXTURES / "reference-evidence.json",
                FIXTURES / "reference-trace.jsonl",
                root,
                ALETHEIA,
            )
            digest = json.loads((destination / "plan.json").read_text())["steps"][0]["artifact_sha256"]
            (destination / "artifacts" / digest).write_bytes(b"tampered")
            with self.assertRaises(Exception):
                stage_bundle.command([str(ALETHEIA)], ["validate-registry", str(root)])

    def test_trace_must_cover_exact_bucket(self):
        plan = json.loads((FIXTURES / "reference-plan.json").read_text())
        evidence = json.loads((FIXTURES / "reference-evidence.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            evidence_path = root / "evidence.json"
            trace_path = root / "trace.jsonl"
            artifact_path = root / "reference.artifact"
            plan_path.write_text(json.dumps(plan))
            artifact_path.write_bytes((FIXTURES / "reference.artifact").read_bytes())
            evidence_path.write_text(json.dumps(evidence))
            event = json.loads((FIXTURES / "reference-trace.jsonl").read_text())
            event["workload"]["batch"] = 2
            trace_path.write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "does not contain"):
                stage_bundle.stage(plan_path, evidence_path, trace_path, root / "registry", ALETHEIA)

    def test_rejects_too_few_timing_samples(self):
        plan = json.loads((FIXTURES / "reference-plan.json").read_text())
        evidence = json.loads((FIXTURES / "reference-evidence.json").read_text())
        evidence["timing"]["samples_micros"] = [10.0]
        with self.assertRaisesRegex(ValueError, "at least 12"):
            stage_bundle.build_certificate(plan, evidence, FIXTURES / "reference-trace.jsonl")

    def test_rejects_path_traversal_before_registry_write(self):
        plan = json.loads((FIXTURES / "reference-plan.json").read_text())
        plan["id"] = "../../escape"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(Exception):
                stage_bundle.stage(
                    plan_path,
                    FIXTURES / "reference-evidence.json",
                    FIXTURES / "reference-trace.jsonl",
                    root / "registry",
                    ALETHEIA,
                )
            self.assertFalse((root / "registry").exists())


if __name__ == "__main__":
    unittest.main()
