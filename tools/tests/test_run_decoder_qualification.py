import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_decoder_qualification.py"
SPEC = importlib.util.spec_from_file_location("run_decoder_qualification", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(MODULE)


class DecoderQualificationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 2}}))
        (self.model / "model.safetensors.index.json").write_text("{}")
        self.reference = self.root / "oracle"
        self.reference.mkdir()
        for phase in MODULE.parity_phases(8):
            name = "prefill-last" if phase == "prefill" else phase
            (self.reference / f"{name}.f32").write_bytes(b"\0" * 8)
        self.binary = self.root / "decoder-test"
        self.output = self.root / "result"

    def fixture(self, behavior="pass"):
        # A real subprocess exercises argument/env transport, failure persistence,
        # timeout handling and artifact sequencing without a GPU or CUDA import.
        self.binary.write_text(f'''#!{sys.executable}
import json, os, sys, time
from pathlib import Path
behavior = {behavior!r}
artifact = Path(os.environ["ORBITKV_DECODER_ARTIFACT"])
mode = "replay" if artifact.exists() else "search"
if behavior.endswith("replay_only"):
    assert mode == "replay", "search was not allowed"
assert sys.argv[1:] == ["decoder_contract", "--ignored", "--exact", "--nocapture", "--test-threads=1"]
assert Path(os.environ["ORBITKV_REFERENCE_DIR"]).is_dir()
profile = "ORBITKV_CUDA_PROFILE_GRAPH_STEPS" in os.environ
print(json.dumps({{"mode": mode, "profile": profile,
                 "tuning": os.environ.get("ORBITKV_TUNING_PROFILE")}}), flush=True)
if behavior == "timeout":
    print("waiting in fixture", file=sys.stderr, flush=True)
    time.sleep(20)
print("ORBITKV_MODULE_ARTIFACT " + json.dumps({{"loaded_image_count": None}}), file=sys.stderr)
if mode == "search":
    artifact.write_text('{{"schema":6,"schedule":"fixture"}}')
if behavior == "nonzero":
    print("fixture failed after writing artifact", file=sys.stderr)
    sys.exit(9)
if behavior == "mutate_artifact" and mode == "replay":
    artifact.write_text("changed")
if behavior == "mutate_oracle":
    (Path(os.environ["ORBITKV_REFERENCE_DIR"]) / "decode.f32").write_bytes(b"1" * 8)
if behavior == "mutate_model_card":
    (Path(os.environ["ORBITKV_MODEL_DIR"]) / "README.md").write_text("changed model release claim")
if behavior == "mutate_replay_source":
    Path(os.environ["FIXTURE_REPLAY_SOURCE"]).write_text("changed external artifact")
batched = behavior.startswith("batched")
batch_size = int(os.environ.get("ORBITKV_QUALIFICATION_BATCH_SIZE", "4"))
for phase in ["prefill", "decode", *[f"decode-{{i}}" for i in range(2, 8)]]:
    if behavior == "missing_parity" and phase == "decode-4":
        continue
    error = "nan" if behavior == "nonfinite" else "0.25"
    requests = list(range(batch_size)) if batched else [None]
    if behavior == "batched_out_of_order" and phase == "decode-4":
        requests.reverse()
    for request in requests:
        if behavior == "batched_missing_row" and phase == "decode-4" and request == 2:
            continue
        label = f"{{phase}}-request-{{request}}" if batched else phase
        line = f"Any decoder {{label}} parity: actual_top=[] reference_top=[] max_abs={{error}}"
        print(line, file=sys.stderr)
        if behavior == "batched_duplicate_row" and phase == "decode-4" and request == 2:
            print(line, file=sys.stderr)
print('ORBITKV_DECODER_STEP ' + json.dumps({{"phase":"decode-7", "seconds":0.04, "diagnostic_logits":True}}), file=sys.stderr)
if behavior != "no_marker":
    marker = {{
        "schema": 1, "reference_enabled": behavior != "no_reference",
        "parity_steps": 8, "drain_passed": behavior != "no_drain", "artifact_mode": mode
    }}
    if batched:
        marker.update(batch_size=batch_size, parity_request_steps=8 * batch_size,
                      batch_capacity=int(os.environ.get("ORBITKV_QUALIFICATION_BATCH_CAPACITY", str(batch_size))),
                      ragged_prefill=os.environ.get("ORBITKV_QUALIFICATION_RAGGED") == "1")
    print("ORBITKV_DECODER_QUALIFICATION " + json.dumps(marker), file=sys.stderr)
if profile:
    print("CUDA_GRAPH_STEP_PROFILE dyn={{s: 1}} total_ms=2.5 Kernel[1]=2.5ms", file=sys.stderr)
stage_path = os.environ.get("ORBITKV_STAGE_TRACE")
if stage_path and behavior != "missing_stage_trace":
    rows = [{{"event": "trace_started", "schema": 2}},
            {{"event": "stage", "id": 0, "parent": None, "start_ns": 0,
              "wall_duration_ns": 10, "thread": "main", "name": "fixture",
              "fields": {{}}, "panicking": False}}]
    if profile and behavior != "missing_gpu_stage_metrics":
        rows += [
            {{"event": "stage", "id": 1, "parent": 0, "start_ns": 1,
              "wall_duration_ns": 8, "thread": "main", "name": "cuda.execute",
              "fields": {{"program": "fixture-program", "profiling": False}}, "panicking": False}},
            {{"event": "stage", "id": 2, "parent": 1, "start_ns": 2,
              "wall_duration_ns": 6, "thread": "main", "name": "cuda.graph.profile",
              "fields": {{"graph_node": 0, "detailed": True, "status": "measured",
                         "step_count": 1, "total_device_ms": 2.5}}, "panicking": False}},
            {{"event": "metric", "parent": 2, "at_ns": 3,
              "thread": "main", "name": "cuda.graph.step", "panicking": False,
              "fields": {{"index": 0, "operator": "Kernel", "implementation": "fixture",
                         "duration_ms": 2.5}}}},
        ]
    rows.append({{"event": "trace_incomplete" if behavior == "incomplete_stage_trace" else "trace_completed",
                 "open_spans": 0}})
    Path(stage_path).write_text("\\n".join(json.dumps(row) for row in rows))
print("test result: ok. 1 passed; 0 failed; 0 ignored; 3 filtered out")
''')
        self.binary.chmod(0o755)

    def arguments(self, *extra):
        return MODULE.parse_args([
            "--test-binary", str(self.binary), "--test-name", "decoder_contract",
            "--model-dir", str(self.model), "--reference-dir", str(self.reference),
            "--output-dir", str(self.output), *extra,
        ])

    def run_qualification(self, *extra):
        with patch.object(MODULE, "source_identity", return_value={"commit": "fixture"}):
            return MODULE.qualify(self.arguments(*extra))

    def test_source_snapshot_does_not_inherit_parent_checkout_identity(self):
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                        "commit", "--quiet", "--allow-empty", "-m", "fixture"], check=True)
        self.assertIn("commit", MODULE.source_identity(self.root))
        self.assertEqual(MODULE.source_identity(self.model), {"available": False})
        self.assertEqual(MODULE.source_identity(self.root / "missing"), {"available": False})

    def test_three_processes_preserve_identity_and_separate_profiling(self):
        self.fixture()
        tuning = self.root / "tuning.json"
        tuning.write_text('{"keep_best": 3}')
        with patch.dict(os.environ, dict.fromkeys(MODULE.PROFILE_FLAGS, "inherited")):
            report = self.run_qualification("--tuning-profile", str(tuning))
        self.assertEqual(report["status"], "passed")
        self.assertEqual([p["phase"] for p in report["phases"]], ["cold-search", "strict-replay", "profile"])
        cold, replay, profile = report["phases"]
        self.assertIsNone(cold["artifact_before"])
        self.assertEqual(cold["artifact_after"], replay["artifact_before"])
        self.assertEqual(replay["artifact_after"], profile["artifact_after"])
        self.assertEqual(replay["input_identity"], cold["input_identity"])
        self.assertEqual(len(profile["evidence"]["parity"]), 8)
        self.assertEqual(len(profile["evidence"]["cuda_graph_profiles"]), 1)
        self.assertEqual(profile["evidence"]["diagnostic_wall_timings"][0]["phase"], "decode-7")
        for entry in (cold, replay):
            self.assertFalse(entry["instrumented"])
            self.assertNotIn(MODULE.PROFILE_FLAGS[0], entry["selected_environment"])
            self.assertEqual(entry["evidence"]["cuda_graph_profiles"], [])
        self.assertEqual(profile["selected_environment"][MODULE.PROFILE_FLAGS[0]], "1")
        self.assertEqual(cold["input_identity"]["tuning_profile"]["sha256"], MODULE.sha256(tuning))

    def test_batched_oracle_rows_and_environment_survive_all_three_phases(self):
        self.fixture("batched_pass")
        environment = {"ORBITKV_QUALIFICATION_BATCH_SIZE": "4", "ORBITKV_QUALIFICATION_RAGGED": "1",
                       "ORBITKV_QUALIFICATION_BATCH_CAPACITY": "8"}
        with patch.dict(os.environ, environment):
            report = self.run_qualification()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["phases"]), 3)
        for phase in report["phases"]:
            evidence = phase["evidence"]
            self.assertEqual(len(evidence["parity"]), 32)
            self.assertEqual(evidence["completion"]["batch_size"], 4)
            self.assertEqual(evidence["completion"]["batch_capacity"], 8)
            self.assertEqual(evidence["completion"]["parity_request_steps"], 32)
            self.assertTrue(evidence["completion"]["ragged_prefill"])
            self.assertEqual(evidence["parity"][-1]["phase"], "decode-7")
            self.assertEqual(evidence["parity"][-1]["request"], 3)
            for key, value in environment.items():
                self.assertEqual(phase["selected_environment"][key], value)

    def test_requested_stage_traces_are_separate_complete_and_identified(self):
        self.fixture()
        inherited = self.root / "inherited.jsonl"
        with patch.dict(os.environ, {"ORBITKV_STAGE_TRACE": str(inherited)}):
            report = self.run_qualification("--stage-trace")
        self.assertEqual(report["status"], "passed")
        self.assertFalse(inherited.exists())
        self.assertIsNotNone(report["input_identity"]["stage_summarizer"])
        for phase in report["phases"]:
            self.assertTrue(phase["instrumented"])
            self.assertTrue(phase["stage_instrumented"])
            self.assertEqual(phase["cuda_graph_instrumented"], phase["phase"] == "profile")
            trace = self.output / phase["phase"] / "stages.jsonl"
            self.assertEqual(phase["stage_trace"], MODULE.file_identity(trace))
            summary = json.loads((trace.parent / "stages-summary.json").read_text())
            self.assertEqual(summary["span_count"], 3 if phase["phase"] == "profile" else 1)

    def test_requested_stage_trace_cannot_omit_device_measurements(self):
        self.fixture("missing_gpu_stage_metrics")
        report = self.run_qualification("--stage-trace")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phases"][-1]["phase"], "profile")
        self.assertIn("structured CUDA measurements", report["phases"][-1]["error"])

    def test_missing_or_incomplete_requested_stage_trace_fails_before_replay(self):
        for behavior in ("missing_stage_trace", "incomplete_stage_trace"):
            with self.subTest(behavior=behavior):
                self.output = self.root / behavior
                self.fixture(behavior)
                report = self.run_qualification("--stage-trace")
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["phases"]), 1)
                self.assertFalse((self.output / "strict-replay").exists())

    def test_stage_trace_is_opt_in_even_with_an_ambient_environment(self):
        self.fixture()
        inherited = self.root / "inherited.jsonl"
        with patch.dict(os.environ, {"ORBITKV_STAGE_TRACE": str(inherited)}):
            report = self.run_qualification()
        self.assertEqual(report["status"], "passed")
        self.assertFalse(inherited.exists())
        for phase in report["phases"]:
            self.assertFalse(phase["stage_instrumented"])
            self.assertNotIn("ORBITKV_STAGE_TRACE", phase["selected_environment"])

    def test_batched_marker_cannot_hide_missing_duplicate_or_out_of_order_request_rows(self):
        for behavior in ("batched_missing_row", "batched_duplicate_row", "batched_out_of_order"):
            with self.subTest(behavior=behavior):
                self.output = self.root / behavior
                self.fixture(behavior)
                report = self.run_qualification()
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["phases"]), 1)
                self.assertIn("request rows", report["phases"][0]["error"])
                self.assertFalse((self.output / "strict-replay").exists())

    def test_external_artifact_is_copied_for_two_replay_processes_without_search(self):
        self.fixture("batched_replay_only")
        source = self.root / "external.json"
        content = b'{"schema":5,"schedule":"existing selected schedule"}'
        source.write_bytes(content)
        with patch.dict(os.environ, {"ORBITKV_QUALIFICATION_BATCH_SIZE": "4",
                                     "ORBITKV_QUALIFICATION_BATCH_CAPACITY": "8"}):
            report = self.run_qualification("--replay-artifact", str(source))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["execution_plan"], ["strict-replay", "profile"])
        self.assertEqual([phase["phase"] for phase in report["phases"]], ["strict-replay", "profile"])
        self.assertFalse((self.output / "cold-search").exists())
        self.assertEqual(source.read_bytes(), content)
        copied = self.output / "decoder.json"
        self.assertEqual(copied.read_bytes(), content)
        self.assertFalse(os.path.samefile(source, copied))
        self.assertEqual(report["input_identity"]["replay_artifact"]["sha256"], MODULE.sha256(source))
        self.assertEqual(report["replay_artifact_copy"]["sha256"], MODULE.sha256(source))
        for phase in report["phases"]:
            self.assertEqual(phase["evidence"]["completion"]["artifact_mode"], "replay")
            self.assertEqual(phase["artifact_before"], phase["artifact_after"])

    def test_mutating_replay_copy_fails_and_preserves_external_source(self):
        self.fixture("mutate_artifact")
        source = self.root / "external.json"
        source.write_text("external artifact remains immutable")
        report = self.run_qualification("--replay-artifact", str(source))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["phases"]), 1)
        self.assertIn("changed the decoder artifact", report["phases"][0]["error"])
        self.assertEqual(source.read_text(), "external artifact remains immutable")
        self.assertFalse((self.output / "profile").exists())

    def test_mutating_external_replay_source_invalidates_otherwise_successful_test(self):
        self.fixture("mutate_replay_source")
        source = self.root / "external.json"
        source.write_text("external artifact")
        with patch.dict(os.environ, {"FIXTURE_REPLAY_SOURCE": str(source)}):
            report = self.run_qualification("--replay-artifact", str(source))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["phases"]), 1)
        self.assertIn("inputs changed", report["phases"][0]["error"])
        self.assertEqual((self.output / "decoder.json").read_text(), "external artifact")
        self.assertFalse((self.output / "profile").exists())

    def test_artifact_and_exit_zero_are_insufficient_without_parity_or_drain(self):
        for behavior in ("no_marker", "no_reference", "no_drain", "missing_parity", "nonfinite"):
            with self.subTest(behavior=behavior):
                self.output = self.root / behavior
                self.fixture(behavior)
                report = self.run_qualification()
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["phases"]), 1)
                self.assertTrue((self.output / "decoder.json").exists())
                self.assertFalse((self.output / "strict-replay").exists())

    def test_nonzero_exit_preserves_logs_and_stops(self):
        self.fixture("nonzero")
        report = self.run_qualification()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phases"][0]["exit_code"], 9)
        self.assertIn("failed after writing", (self.output / "cold-search/stderr.log").read_text())
        self.assertFalse((self.output / "strict-replay").exists())
        self.assertEqual(json.loads((self.output / "result.json").read_text())["status"], "failed")

    def test_timeout_preserves_partial_output_and_stops(self):
        self.fixture("timeout")
        report = self.run_qualification("--phase-timeout-seconds", "0.2")
        self.assertEqual(report["status"], "failed")
        phase = report["phases"][0]
        self.assertTrue(phase["timed_out"])
        self.assertIsNotNone(phase["exit_code"])
        self.assertIn("waiting in fixture", (self.output / "cold-search/stderr.log").read_text())
        self.assertFalse((self.output / "strict-replay").exists())

    def test_replay_mutating_artifact_stops_before_profile(self):
        self.fixture("mutate_artifact")
        report = self.run_qualification()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["phases"]), 2)
        self.assertIn("changed the decoder artifact", report["phases"][1]["error"])
        self.assertFalse((self.output / "profile").exists())

    def test_mutating_oracle_invalidates_an_otherwise_successful_phase(self):
        self.fixture("mutate_oracle")
        report = self.run_qualification()
        self.assertEqual(report["status"], "failed")
        self.assertIn("inputs changed", report["phases"][0]["error"])

    def test_optional_model_card_is_fingerprinted_and_changes_invalidate_run(self):
        self.fixture("mutate_model_card")
        card = self.model / "README.md"
        card.write_text("# Model release name\nBuilt on a different architecture name.\n")
        original = MODULE.file_identity(card)
        report = self.run_qualification()
        self.assertIn(original, report["input_identity"]["model"])
        self.assertEqual(report["status"], "failed")
        self.assertIn("inputs changed", report["phases"][0]["error"])
        self.assertFalse((self.output / "strict-replay").exists())

    def test_preflight_rejects_missing_oracle_and_existing_output(self):
        self.fixture()
        oracle = self.reference / "decode-7.f32"
        oracle.unlink()
        with self.assertRaises(OSError):
            self.run_qualification()
        self.assertFalse(self.output.exists())
        oracle.write_bytes(b"\0" * 8)
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.run_qualification()
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
