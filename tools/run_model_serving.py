#!/usr/bin/env python3
"""Measure one OrbitKV model through the real HTTP frontend and common client."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from statistics import median

from run_matched_serving import (
    METRICS, ServerSpec, add_workload_arguments, bench_command,
    environment_snapshot, git_revision, load_workload, normalize_url,
    output_directory, parse_command, run_one,
)
from serving_metrics import engine_report, validate_drain


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def summarize(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("no completed serving runs")
    digests = {run["generated_texts_sha256"] for run in runs}
    return {
        "runs": len(runs),
        "aggregation": "median of per-run metrics; percentiles are not pooled",
        "metrics": {name: median(run["metrics"][name] for run in runs)
                    for name in METRICS},
        "input_lengths": sorted({length for run in runs for length in run["input_lengths"]}),
        "output_repeatable": None not in digests and len(digests) == 1,
        "sampled_max_gpu_bytes": max((run["memory"]["sampled_max_gpu_bytes"]
            for run in runs if run["memory"]["sampled_max_gpu_bytes"] is not None), default=None),
        "sampled_max_host_rss_bytes": max((run["memory"]["sampled_max_host_rss_bytes"]
            for run in runs if run["memory"]["sampled_max_host_rss_bytes"] is not None), default=None),
        "sampled_max_device_gpu_bytes": max((run["memory"]["sampled_max_device_gpu_bytes"]
            for run in runs if run["memory"].get("sampled_max_device_gpu_bytes") is not None), default=None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--memory-sample-seconds", type=float, default=1.0)
    parser.add_argument("--memory-device", type=int,
                        help="Also sample total usage on this GPU; it is distinct from process memory.")
    add_workload_arguments(parser)
    args = parser.parse_args()
    try:
        if args.runs <= 0 or not math.isfinite(args.memory_sample_seconds) or args.memory_sample_seconds <= 0:
            raise ValueError("runs and memory sampling interval must be positive")
        if args.startup_timeout_seconds <= 0 or args.shutdown_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if args.memory_device is not None and args.memory_device < 0:
            raise ValueError("memory device must be nonnegative")
        server = ServerSpec("orbitkv", parse_command(args.server_command, "server",
            require_executable=not args.dry_run), normalize_url(args.base_url))
        client = parse_command(args.vllm_command, "client", require_executable=not args.dry_run)
        diagnostic_variables = ("ORBITKV_STAGE_TRACE", "ORBITKV_CUDA_PROFILE_GRAPH_STEPS", "ORBITKV_LLIR_PROFILE")
        if any(os.environ.get(name) for name in diagnostic_variables):
            raise ValueError("disable compiler/device diagnostic tracing for serving measurements")
        workload = load_workload(args.profiles_file, args.profile)
        root = output_directory(args.output_dir, args.profile)
        plan = {
            "schema": "orbitkv.model-serving.v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": git_revision(), "model": args.model, "tokenizer": args.tokenizer,
            "profile": args.profile, "workload": workload.__dict__, "runs": args.runs,
            "server_command": list(server.command), "client_command": list(client),
            "seed": args.seed, "environment": environment_snapshot(client, args.dry_run),
            "measurement": "HTTP serving after readiness; startup excluded; periodic process memory sampling",
            "memory_device": args.memory_device,
        }
        if args.dry_run:
            plan["example_client_command"] = bench_command(client, server, args, workload, root, "benchmark.json")
            print(json.dumps(plan, indent=2))
            return 0
        binary_path = Path(shutil.which(server.command[0]) or server.command[0]).resolve()
        plan["binary"] = identity(binary_path)
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "run.json", plan)
        runs = []
        for number in range(1, args.runs + 1):
            print(f"{args.profile}: run {number}/{args.runs}", flush=True)
            run = run_one(server, number, client, args, workload, root,
                          memory_interval_seconds=args.memory_sample_seconds,
                          memory_device_index=args.memory_device)
            if run["shutdown"]["returncode"] != 0:
                raise RuntimeError(f"OrbitKV exited with status {run['shutdown']['returncode']}")
            result_path = root / run["result"]
            raw = json.loads(result_path.read_text())
            log = (result_path.parent / "server.log").read_text()
            run["startup"] = engine_report(log, "ORBITKV_ENGINE_STARTUP")
            run["state_drain"] = engine_report(log, "ORBITKV_ENGINE_SHUTDOWN")
            validate_drain(run["state_drain"], workload.requests)
            run["input_lengths"] = sorted(set(raw["input_lens"]))
            runs.append(run)
            write_json(root / "observations.json", runs)
        if plan["binary"] != identity(binary_path):
            raise ValueError("server binary changed during measurements")
        write_json(root / "completed.json", {"status": "completed", "runs": runs, "summary": summarize(runs)})
        print(root)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as error:
        print(f"model serving benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
