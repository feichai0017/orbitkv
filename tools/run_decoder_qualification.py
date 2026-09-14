#!/usr/bin/env python3
"""Qualify a prebuilt decoder test in separate search, replay and profile processes.

This records diagnostic execution with logits, not serving TPOT. It never builds,
downloads inputs, clears shared caches, or promotes results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from summarize_stage_trace import summarize as summarize_stages


ROOT = Path(__file__).resolve().parents[1]
MARKER = "ORBITKV_DECODER_QUALIFICATION "
PROFILE_FLAGS = (
    "LUMINAL_CUDA_PROFILE_GRAPH_STEPS",
    "LUMINAL_CUDA_PROFILE_GRAPH_STEP_DETAILS",
)
ENVIRONMENT_KEYS = (
    "PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "FLASHINFER_CUDA_ARCH", "LUMINAL_DEEPGEMM_DIR", "LUMINAL_FLASHINFER_DIR",
    "LUMINAL_DEEPGEMM_CACHE_DIR", "LUMINAL_MAX_ROLLED_REGIONS",
    "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH",
    "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
    "ORBITKV_MODEL_DIR", "ORBITKV_REFERENCE_DIR", "ORBITKV_SEARCH_GRAPHS",
    "ORBITKV_GRAPH_CACHE_CAPACITY",
    "ORBITKV_QUALIFICATION_BATCH_SIZE", "ORBITKV_QUALIFICATION_BATCH_CAPACITY",
    "ORBITKV_QUALIFICATION_RAGGED",
    "ORBITKV_DECODER_ARTIFACT", "ORBITKV_TUNING_PROFILE", *PROFILE_FLAGS,
    "LUMINAL_SEARCH_TRACE", "LUMINAL_STAGE_TRACE",
)


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", required=True, type=Path)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=ROOT,
                        help="Source checkout/snapshot observed for provenance; binary SHA remains authoritative.")
    parser.add_argument("--search-graphs", type=int, default=16)
    parser.add_argument("--tuning-profile", type=Path)
    parser.add_argument("--stage-trace", action="store_true",
                        help="Record compiler/runtime CPU spans separately for each phase; requires an instrumented harness.")
    parser.add_argument("--replay-artifact", type=Path,
                        help="Copy this existing artifact into the fresh output and run only strict replay/profile.")
    parser.add_argument("--expected-parity-steps", type=int, default=8)
    parser.add_argument("--phase-timeout-seconds", type=float, default=3600)
    return parser.parse_args(argv)


def parity_phases(steps: int) -> list[str]:
    return ["prefill", "decode", *(f"decode-{i}" for i in range(2, steps))]


def reference_files(args: argparse.Namespace) -> list[Path]:
    return [args.reference_dir / ("prefill-last.f32" if phase == "prefill" else f"{phase}.f32")
            for phase in parity_phases(args.expected_parity_steps)]


def validate_inputs(args: argparse.Namespace) -> None:
    for name in ("test_binary", "model_dir", "reference_dir", "output_dir", "source_dir", "tuning_profile", "replay_artifact"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not args.test_binary.is_file() or not os.access(args.test_binary, os.X_OK):
        raise ValueError("--test-binary must be a prebuilt executable")
    if not args.source_dir.is_dir():
        raise ValueError("--source-dir must be an existing checkout or source snapshot")
    if not args.test_name or args.test_name.startswith("-"):
        raise ValueError("--test-name must be a nonempty exact test name")
    if args.output_dir.exists():
        raise ValueError("--output-dir must be fresh; existing results are never overwritten")
    if args.replay_artifact is not None and (not args.replay_artifact.is_file()
                                            or args.replay_artifact.stat().st_size == 0):
        raise ValueError("--replay-artifact must be an existing nonempty artifact")
    if args.search_graphs <= 0 or args.expected_parity_steps < 2:
        raise ValueError("search graphs must be positive and parity steps must be at least two")
    if not math.isfinite(args.phase_timeout_seconds) or args.phase_timeout_seconds <= 0:
        raise ValueError("phase timeout must be finite and positive")
    config = json.loads((args.model_dir / "config.json").read_text())
    vocabulary = config.get("text_config", config).get("vocab_size")
    sizes = {path.stat().st_size for path in reference_files(args)}
    if len(sizes) != 1 or next(iter(sizes)) == 0 or next(iter(sizes)) % 4:
        raise ValueError("oracle files must contain equally sized, nonempty little-endian f32 rows")
    if vocabulary is not None and sizes != {vocabulary * 4}:
        raise ValueError("oracle row size does not match the model vocabulary")
    if args.tuning_profile is not None:
        if not isinstance(json.loads(args.tuning_profile.read_text()), dict):
            raise ValueError("tuning profile must be a JSON object")


def input_identity(args: argparse.Namespace) -> dict:
    model_files = [args.model_dir / "config.json"]
    for name in ("model.safetensors.index.json", "README.md"):
        path = args.model_dir / name
        if path.is_file():
            model_files.append(path)
    oracle_files = reference_files(args)
    metadata = args.reference_dir / "metadata.json"
    if metadata.is_file():
        oracle_files.append(metadata)
    return {
        "binary": file_identity(args.test_binary),
        "model": [file_identity(path) for path in model_files],
        "oracle": [file_identity(path) for path in oracle_files],
        "tuning_profile": file_identity(args.tuning_profile) if args.tuning_profile else None,
        "replay_artifact": file_identity(args.replay_artifact) if args.replay_artifact else None,
        "stage_summarizer": (file_identity(Path(__file__).with_name("summarize_stage_trace.py"))
                             if getattr(args, "stage_trace", False) else None),
    }


def source_identity(directory: Path) -> dict:
    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(directory), *arguments],
                              capture_output=True, check=False, timeout=20)
    try:
        checkout = git("rev-parse", "--show-toplevel")
        if checkout.returncode or Path(checkout.stdout.decode().strip()).resolve() != directory.resolve():
            # A source snapshot inside another checkout does not inherit that
            # parent's Git identity. Its build manifest supplies provenance.
            return {"available": False}
        head = git("rev-parse", "HEAD")
        diff = git("diff", "--binary", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no")
        if any(result.returncode for result in (head, diff, status)):
            return {"available": False}
        return {"commit": head.stdout.decode().strip(),
                "tracked_worktree_dirty": bool(status.stdout),
                "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest()}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}


def phase_environment(args: argparse.Namespace, artifact: Path, phase: str) -> dict[str, str]:
    environment = os.environ.copy()
    diagnostics = [key for key in environment if key.startswith("LUMINAL_CUDA_PROFILE_")]
    for name in (*diagnostics, "LUMINAL_CUDA_ARENA_PROFILE", "LUMINAL_CUDA_SYNC_EACH_EXEC_OP",
                 "LUMINAL_CUDA_CHECK_NONFINITE_INTERNAL", "ORBITKV_TUNING_PROFILE", "LUMINAL_STAGE_TRACE"):
        environment.pop(name, None)
    environment.update({
        "ORBITKV_MODEL_DIR": str(args.model_dir),
        "ORBITKV_REFERENCE_DIR": str(args.reference_dir),
        "ORBITKV_SEARCH_GRAPHS": str(args.search_graphs),
        "ORBITKV_DECODER_ARTIFACT": str(artifact),
    })
    if args.tuning_profile:
        environment["ORBITKV_TUNING_PROFILE"] = str(args.tuning_profile)
    if getattr(args, "stage_trace", False):
        environment["LUMINAL_STAGE_TRACE"] = str(args.output_dir / phase / "stages.jsonl")
    if phase == "profile":
        environment.update(dict.fromkeys(PROFILE_FLAGS, "1"))
    return environment


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def parse_evidence(text: str, expected_steps: int, mode: str, profile: bool) -> dict:
    markers = []
    module_markers = []
    parities = []
    profiles = []
    timings = []
    for line in text.splitlines():
        if "ORBITKV_MODULE_ARTIFACT " in line:
            module_markers.append(json.loads(line.split("ORBITKV_MODULE_ARTIFACT ", 1)[1]))
        if MARKER in line:
            markers.append(json.loads(line.split(MARKER, 1)[1]))
        if "ORBITKV_DECODER_STEP " in line:
            step = json.loads(line.split("ORBITKV_DECODER_STEP ", 1)[1])
            seconds = step.get("seconds")
            if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid diagnostic step timing")
            timings.append(step)
        match = re.search(r"\b(prefill|decode(?:-\d+)?)(?:-request-(\d+))? parity:.*\bmax_abs=(\S+)", line)
        if match:
            value = float(match[3])
            if not math.isfinite(value) or not 0 <= value <= 1.0:
                raise ValueError("nonfinite or out-of-contract oracle error")
            entry = {"phase": match[1], "max_abs": value}
            if match[2] is not None:
                entry["request"] = int(match[2])
            parities.append(entry)
        match = re.search(r"CUDA_GRAPH_STEP_PROFILE dyn=(.*?) total_ms=([0-9.eE+-]+)", line)
        if match:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid CUDA Graph profile timing")
            profiles.append({"dynamic_dimensions": match[1], "total_ms": value})
        match = re.search(r"(decoder schedule ready|bounded prefill completed|bounded decode completed) after ([0-9.eE+-]+)s", line)
        if match:
            timings.append({"operation": match[1], "seconds": float(match[2])})
    if not re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", text):
        raise ValueError("exact test did not report one successful, non-ignored execution")
    if len(markers) != 1:
        raise ValueError("expected one structured qualification completion marker")
    marker = markers[0]
    expected = {"schema": 1, "reference_enabled": True, "parity_steps": expected_steps,
                "drain_passed": True, "artifact_mode": mode}
    if not isinstance(marker, dict) or any(type(marker.get(key)) is not type(value) or marker.get(key) != value
                                          for key, value in expected.items()):
        raise ValueError("qualification marker did not confirm oracle, drain, and artifact mode")
    batch_fields = ("batch_size", "parity_request_steps", "ragged_prefill")
    if any(key in marker for key in batch_fields):
        batch_size = marker.get("batch_size")
        request_steps = marker.get("parity_request_steps")
        if (type(batch_size) is not int or batch_size < 1
                or type(request_steps) is not int or request_steps != expected_steps * batch_size
                or type(marker.get("ragged_prefill")) is not bool):
            raise ValueError("qualification marker did not confirm the batched oracle contract")
        capacity = marker.get("batch_capacity", batch_size)
        if type(capacity) is not int or capacity < batch_size:
            raise ValueError("qualification marker reported an insufficient batch capacity")
        expected_parities = [(phase, request) for phase in parity_phases(expected_steps)
                            for request in range(batch_size)]
    else:
        expected_parities = [(phase, None) for phase in parity_phases(expected_steps)]
    if [(entry["phase"], entry.get("request")) for entry in parities] != expected_parities:
        raise ValueError("missing, duplicate, or out-of-order oracle parity steps or request rows")
    if profile and not profiles:
        raise ValueError("profile phase emitted no CUDA Graph step profile")
    if not profile and profiles:
        raise ValueError("timing phase unexpectedly included instrumented CUDA Graph profiles")
    if len(module_markers) > 1:
        raise ValueError("duplicate module artifact marker")
    return {"completion": marker, "parity": parities, "cuda_graph_profiles": profiles,
            "module_artifact": module_markers[0] if module_markers else None,
            "diagnostic_wall_timings": timings}


def run_phase(args: argparse.Namespace, phase: str, artifact: Path, initial_inputs: dict) -> dict:
    directory = args.output_dir / phase
    directory.mkdir()
    before = file_identity(artifact) if artifact.is_file() else None
    environment = phase_environment(args, artifact, phase)
    command = [str(args.test_binary), args.test_name, "--ignored", "--exact", "--nocapture",
               "--test-threads=1"]
    result = {"phase": phase, "status": "running", "started_at_utc": timestamp(),
              "command": command, "artifact_before": before,
              "selected_environment": {key: environment[key] for key in ENVIRONMENT_KEYS if key in environment},
              "instrumented": phase == "profile" or getattr(args, "stage_trace", False),
              "cuda_graph_instrumented": phase == "profile",
              "stage_instrumented": getattr(args, "stage_trace", False),
              "input_identity": initial_inputs}
    write_json(directory / "run.json", result)
    started = time.monotonic()
    try:
        if input_identity(args) != initial_inputs:
            raise ValueError("qualification inputs changed between phases")
        if (phase == "cold-search") != (before is None):
            raise ValueError("search requires a fresh artifact; replay requires the produced artifact")
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process_started = time.monotonic()
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                result["exit_code"] = process.wait(timeout=args.phase_timeout_seconds)
            except subprocess.TimeoutExpired:
                stop_process(process)
                result.update(exit_code=process.returncode, timed_out=True)
                raise ValueError("phase timed out") from None
            except BaseException:
                stop_process(process)
                raise
            finally:
                result["elapsed_process_seconds"] = time.monotonic() - process_started
        if result["exit_code"] != 0:
            raise ValueError(f"test process exited with {result['exit_code']}")
        text = (directory / "stdout.log").read_text(errors="replace") + "\n" + (directory / "stderr.log").read_text(errors="replace")
        result["evidence"] = parse_evidence(text, args.expected_parity_steps,
                                           "search" if phase == "cold-search" else "replay",
                                           phase == "profile")
        if getattr(args, "stage_trace", False):
            stage_path = directory / "stages.jsonl"
            summary = summarize_stages(stage_path)
            write_json(directory / "stages-summary.json", summary)
            result["stage_trace"] = file_identity(stage_path)
            result["stage_summary"] = file_identity(directory / "stages-summary.json")
        if not artifact.is_file():
            raise ValueError("successful test did not produce a decoder artifact")
        after = file_identity(artifact)
        if before is not None and before != after:
            raise ValueError("strict replay changed the decoder artifact")
        if input_identity(args) != initial_inputs:
            raise ValueError("qualification inputs changed during execution")
        result["status"] = "passed"
    except (OSError, ValueError, KeyboardInterrupt) as error:
        result.update(status="failed", error=str(error) or "interrupted")
    finally:
        result["elapsed_phase_seconds"] = time.monotonic() - started
        result["finished_at_utc"] = timestamp()
        result["artifact_after"] = file_identity(artifact) if artifact.is_file() else None
        result["logs"] = {name: file_identity(directory / name) for name in ("stdout.log", "stderr.log")
                          if (directory / name).is_file()}
        write_json(directory / "result.json", result)
    return result


def qualify(args: argparse.Namespace) -> dict:
    validate_inputs(args)
    inputs = input_identity(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact = args.output_dir / "decoder.json"
    phases = ("strict-replay", "profile") if args.replay_artifact else ("cold-search", "strict-replay", "profile")
    report = {
        "schema": "orbitkv.decoder-qualification.v1", "status": "running",
        "created_at_utc": timestamp(), "test_name": args.test_name,
        "execution_plan": list(phases),
        "input_identity": inputs, "harness": file_identity(Path(__file__).resolve()),
        "source_observation_directory": str(args.source_dir),
        "source": source_identity(args.source_dir),
        "luminal_source": source_identity(args.source_dir / "third_party/luminal"),
        "provenance_note": "Source checkout is observed, not proof of binary build inputs; retain the prebuilt binary and its build record.",
        "cache_policy": {"selected_schedule": ("immutable external artifact copied into fresh output; no search"
                                               if args.replay_artifact else
                                               "fresh artifact for cold-search, same artifact for replay/profile"),
                         "provider_and_cuda_caches": "existing caches reused; not cleared or claimed cold",
                         "build": "prebuilt binary; build time excluded",
                         "measurement": "diagnostic execution with logits; not serving TPOT",
                         "stage_trace": ("buffered CPU spans in every phase; diagnostic overhead included, no added device synchronization"
                                         if getattr(args, "stage_trace", False) else "disabled"),
                         "profile": "separate process with timing events; not comparable to uninstrumented wall time"},
        "phases": [],
    }
    write_json(args.output_dir / "run.json", report)
    if args.replay_artifact:
        try:
            shutil.copyfile(args.replay_artifact, artifact)
            copied = file_identity(artifact)
            source = inputs["replay_artifact"]
            if any(copied[key] != source[key] for key in ("bytes", "sha256")):
                raise ValueError("replay artifact copy differs from the recorded source")
            if input_identity(args) != inputs:
                raise ValueError("qualification inputs changed while copying the replay artifact")
            report["replay_artifact_copy"] = copied
            write_json(args.output_dir / "run.json", report)
        except (OSError, ValueError) as error:
            report.update(status="failed", error=str(error), finished_at_utc=timestamp())
            write_json(args.output_dir / "result.json", report)
            return report
    for phase in phases:
        result = run_phase(args, phase, artifact, inputs)
        report["phases"].append(result)
        write_json(args.output_dir / "result.json", report)
        if result["status"] != "passed":
            report["status"] = "failed"
            break
    else:
        report["status"] = "passed"
    report["finished_at_utc"] = timestamp()
    write_json(args.output_dir / "result.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = qualify(args)
    except (OSError, ValueError) as error:
        print(f"decoder qualification: {error}", file=sys.stderr)
        return 1
    print(f"decoder qualification {report['status']}: {args.output_dir / 'result.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
