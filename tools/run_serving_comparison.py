#!/usr/bin/env python3
"""Compare serving engines sequentially using one vLLM benchmark client."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from run_matched_serving import (
    REPOSITORY_ROOT, ServerSpec, add_workload_arguments, environment_snapshot,
    git_revision, load_workload, normalize_url, output_directory, parse_command,
    run_client, stop_server, wait_until_ready,
)
from run_model_serving import identity, summarize, write_json
from serving_metrics import ProcessMemorySampler, engine_report, validate_drain


def load_servers(path: Path, *, dry_run: bool) -> list[tuple[ServerSpec, str | None]]:
    payload = json.loads(path.read_text())
    entries = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("servers file must contain at least two servers")
    servers = []
    names = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each server must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError("server names must be lowercase path-safe identifiers")
        if name in names:
            raise ValueError(f"duplicate server: {name}")
        names.add(name)
        command = entry.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(arg, str) and arg for arg in command
        ):
            raise ValueError(f"{name}: command must be a nonempty argument array")
        lifecycle = entry.get("lifecycle")
        if lifecycle not in (None, "orbitkv"):
            raise ValueError(f"{name}: unsupported lifecycle audit {lifecycle}")
        base_url = entry.get("base_url")
        if not isinstance(base_url, str):
            raise ValueError(f"{name}: base_url must be a URL")
        servers.append((ServerSpec(name, parse_command(shlex.join(command), name,
            require_executable=not dry_run), normalize_url(base_url)), lifecycle))
    return servers


def balanced_schedule(names: list[str], epochs: int) -> list[list[str]]:
    if not names or epochs <= 0 or epochs % len(names):
        raise ValueError("epochs must be a positive multiple of the engine count")
    return [names[offset:] + names[:offset]
            for epoch in range(epochs) for offset in [epoch % len(names)]]


def load_traces(directory: Path, workloads: dict) -> dict[str, Path]:
    """Validate fixed-load vLLM timed traces: one hash expands to one token."""
    paths = {}
    for name, workload in workloads.items():
        path = (directory / f"{name}.jsonl").resolve()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != workload.requests:
            raise ValueError(f"{name}: trace request count differs from the workload")
        for row in rows:
            if not isinstance(row, dict) or any(row.get(key) != expected for key, expected in (
                ("input_length", workload.input_tokens), ("output_length", workload.output_tokens),
                ("timestamp", 0),
            )):
                raise ValueError(f"{name}: trace lengths/timestamp differ from the workload")
            hashes = row.get("hash_ids")
            if not isinstance(hashes, list) or len(hashes) != workload.input_tokens or any(
                type(value) is not int for value in hashes
            ):
                raise ValueError(f"{name}: trace requires one integer hash per input token")
        paths[name] = path
    return paths


def comparison_summary(observations: list[dict], names: list[str], profiles: list[str], epochs: int) -> list[dict]:
    rows = []
    for profile in profiles:
        selected = [run for run in observations if run["profile"] == profile]
        # Check server-reported lengths as well as the frozen client inputs.
        if len({tuple(run["input_lengths"]) for run in selected}) != 1:
            raise RuntimeError(f"{profile}: request input lengths differ between runs")
        digests = {run["generated_texts_sha256"] for run in selected}
        for name in names:
            runs = [run for run in selected if run["server"] == name]
            if len(runs) != epochs or {run["epoch"] for run in runs} != set(range(1, epochs + 1)):
                raise RuntimeError(f"{profile}/{name}: incomplete or duplicated epochs")
            summary = summarize(runs)
            summary["metric_ranges"] = {
                metric: {"minimum": min(run["metrics"][metric] for run in runs),
                         "maximum": max(run["metrics"][metric] for run in runs)}
                for metric in summary["metrics"]
            }
            rows.append({"profile": profile, "engine": name, **summary,
                "outputs_match_across_engines": None not in digests and len(digests) == 1})
    return rows


def run_session(server: ServerSpec, lifecycle: str | None, epoch: int,
                client: tuple[str, ...], args: argparse.Namespace,
                workloads: dict, traces: dict[str, Path], root: Path) -> dict:
    session_dir = root / f"epoch-{epoch:03d}" / server.name
    session_dir.mkdir(parents=True, exist_ok=False)
    observations = []
    with (session_dir / "server.log").open("wb") as log:
        process = subprocess.Popen(server.command, cwd=REPOSITORY_ROOT,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            wait_until_ready(process, f"{server.base_url}{args.health_path}", args.startup_timeout_seconds)
            ready_seconds = time.monotonic() - started
            for profile, workload in workloads.items():
                print(f"epoch {epoch}: {server.name}/{profile}", flush=True)
                run_dir = session_dir / profile
                run_dir.mkdir()
                sampler = ProcessMemorySampler(process.pid, args.memory_sample_seconds, args.memory_device)
                sampler.start()
                try:
                    measured = run_client(server, client, args, workload, run_dir,
                                          trace_path=traces[profile])
                finally:
                    sampler.stop()
                if process.poll() is not None:
                    raise RuntimeError(f"{server.name}: server exited during measurement")
                raw = json.loads((run_dir / "benchmark.json").read_text())
                lengths = raw.get("input_lens")
                if not isinstance(lengths, list) or len(lengths) != workload.requests or any(
                    type(length) is not int or length != workload.input_tokens for length in lengths
                ):
                    raise RuntimeError(f"{server.name}/{profile}: missing valid input lengths")
                observations.append({**measured, "server": server.name, "epoch": epoch,
                    "profile": profile, "input_lengths": lengths, "memory": sampler.report(),
                    "result": str((run_dir / "benchmark.json").relative_to(root))})
                write_json(session_dir / "observations.json", observations)
        finally:
            shutdown = stop_server(process, args.shutdown_timeout_seconds)
    if shutdown.forced:
        raise RuntimeError(f"{server.name}: shutdown required SIGKILL")
    session = {"server": server.name, "epoch": epoch, "ready_seconds": ready_seconds,
               "shutdown": shutdown.__dict__, "observations": observations}
    if lifecycle == "orbitkv":
        if shutdown.returncode != 0:
            raise RuntimeError(f"{server.name}: unexpected exit {shutdown.returncode}")
        log = (session_dir / "server.log").read_text()
        session["startup"] = engine_report(log, "ORBITKV_ENGINE_STARTUP")
        session["state_drain"] = engine_report(log, "ORBITKV_ENGINE_SHUTDOWN")
        validate_drain(session["state_drain"], sum(workload.requests for workload in workloads.values()))
    write_json(session_dir / "completed.json", session)
    return session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servers-file", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True,
                        help="Frozen vLLM timed traces, <profile>.jsonl; one hash per input token.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--memory-sample-seconds", type=float, default=1.0)
    parser.add_argument("--memory-device", type=int)
    parser.add_argument("--identity-file", type=Path, action="append", default=[],
                        help="Pin an executable, artifact or other input before and after the run.")
    add_workload_arguments(parser, multiple_profiles=True)
    args = parser.parse_args()
    try:
        if any(not math.isfinite(value) or value <= 0 for value in (
            args.memory_sample_seconds, args.startup_timeout_seconds, args.shutdown_timeout_seconds
        )):
            raise ValueError("timeouts and sampling interval must be positive and finite")
        if args.memory_device is not None and args.memory_device < 0:
            raise ValueError("memory device must be nonnegative")
        if any(os.environ.get(name) for name in (
            "ORBITKV_STAGE_TRACE", "ORBITKV_CUDA_PROFILE_GRAPH_STEPS", "ORBITKV_LLIR_PROFILE"
        )):
            raise ValueError("disable diagnostic tracing for serving measurements")
        if len(set(args.profile)) != len(args.profile):
            raise ValueError("profiles must not repeat")
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", profile) for profile in args.profile):
            raise ValueError("profiles must be path-safe identifiers")
        servers = load_servers(args.servers_file, dry_run=args.dry_run)
        by_name = {server.name: (server, lifecycle) for server, lifecycle in servers}
        schedule = balanced_schedule(list(by_name), args.epochs)
        workloads = {name: load_workload(args.profiles_file, name) for name in args.profile}
        traces = load_traces(args.trace_dir, workloads)
        client = parse_command(args.vllm_command, "client", require_executable=not args.dry_run)
        root = output_directory(args.output_dir, "comparison")
        files = list(dict.fromkeys(path.resolve() for path in [
            args.servers_file, args.profiles_file, *traces.values(), *args.identity_file
        ]))
        identities = [identity(path) for path in files]
        plan = {"schema": "orbitkv.serving-comparison.v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": git_revision(), "model": args.model, "tokenizer": args.tokenizer,
            "servers": [{**server.__dict__, "lifecycle": lifecycle} for server, lifecycle in servers],
            "workloads": {name: workload.__dict__ for name, workload in workloads.items()},
            "schedule": schedule, "client": list(client), "seed": args.seed,
            "client_style": args.client_style, "bench_args": args.bench_arg,
            "dataset": {"name": "timed_trace", "chunk_hash_size": 1,
                        "python_hash_seed": args.seed, "self_timed": False,
                        "traces": {name: str(path) for name, path in traces.items()}},
            "environment": environment_snapshot(client, args.dry_run), "files": identities,
            "measurement": "one fresh process per engine/epoch; workloads run in declared order after readiness",
            "cache_policy": "provider and compiler disk caches retained; no benchmark observations discarded"}
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "run.json", plan)
        sessions = []
        for epoch, names in enumerate(schedule, start=1):
            for name in names:
                server, lifecycle = by_name[name]
                print(f"epoch {epoch}: starting {name}", flush=True)
                sessions.append(run_session(server, lifecycle, epoch, client, args, workloads, traces, root))
                write_json(root / "sessions.json", sessions)
        if identities != [identity(path) for path in files]:
            raise RuntimeError("benchmark inputs changed during measurement")
        observations = [run for session in sessions for run in session["observations"]]
        write_json(root / "completed.json", {"status": "completed", "sessions": sessions,
            "summary": comparison_summary(observations, list(by_name), args.profile, args.epochs)})
        print(root)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as error:
        print(f"serving comparison failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
