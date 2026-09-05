#!/usr/bin/env python3
"""Run alternating matched serving benchmarks with one vLLM client."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = REPOSITORY_ROOT / "benchmarks" / "serving-profiles.json"
METRICS = (
    "request_throughput",
    "output_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
)


@dataclass(frozen=True)
class Workload:
    input_tokens: int
    output_tokens: int
    requests: int
    max_concurrency: int


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: tuple[str, ...]
    base_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run candidate and baseline servers sequentially with identical "
            "vllm bench serve workloads."
        )
    )
    parser.add_argument("--candidate-command", required=True)
    parser.add_argument("--baseline-command", required=True)
    parser.add_argument("--candidate-url", default="http://127.0.0.1:8000")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--health-path", default="/v1/models")
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-command", default="vllm")
    parser.add_argument(
        "--client-style",
        choices=("auto", "python", "rust"),
        default="auto",
        help="Use `vllm bench serve` or the standalone Rust `vllm-bench` CLI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Unreviewed output directory; defaults below .qualification/.",
    )
    parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
        help="Additional single argument passed to vllm bench serve; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_workload(path: Path, name: str) -> Workload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown serving profile {name!r} in {path}")
    workload = Workload(
        input_tokens=positive_int(profile, "input_tokens"),
        output_tokens=positive_int(profile, "output_tokens"),
        requests=positive_int(profile, "requests"),
        max_concurrency=positive_int(profile, "max_concurrency"),
    )
    if workload.max_concurrency > workload.requests:
        raise ValueError("max_concurrency must not exceed requests")
    return workload


def positive_int(source: dict[str, Any], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def parse_command(
    value: str, label: str, *, require_executable: bool = True
) -> tuple[str, ...]:
    command = tuple(shlex.split(value))
    if not command:
        raise ValueError(f"{label} command is empty")
    executable = Path(command[0])
    if require_executable and (executable.is_absolute() or executable.parent != Path(".")):
        resolved = executable if executable.is_absolute() else REPOSITORY_ROOT / executable
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError(f"{label} executable does not exist or is not executable: {resolved}")
    elif require_executable and shutil.which(str(executable)) is None:
        raise ValueError(f"{label} executable is not on PATH: {executable}")
    return command


def normalize_url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ValueError(f"server URL must use http or https: {value}")
    return value.rstrip("/")


def output_directory(requested: Path | None, profile: str) -> Path:
    if requested is not None:
        return requested.resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (REPOSITORY_ROOT / ".qualification" / "matched-serving" / f"{profile}-{timestamp}").resolve()


def bench_command(
    client: tuple[str, ...],
    server: ServerSpec,
    args: argparse.Namespace,
    workload: Workload,
    result_dir: Path,
    result_name: str,
) -> list[str]:
    client_style = args.client_style
    if client_style == "auto":
        executable = Path(client[0]).name
        client_style = "rust" if executable == "vllm-bench" else "python"
    prefix = [*client] if client_style == "rust" else [*client, "bench", "serve"]
    return [
        *prefix,
        "--backend",
        args.backend,
        "--base-url",
        server.base_url,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.input_tokens),
        "--random-output-len",
        str(workload.output_tokens),
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        str(workload.requests),
        "--max-concurrency",
        str(workload.max_concurrency),
        "--request-rate",
        args.request_rate,
        "--seed",
        str(args.seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "95,99",
        "--ignore-eos",
        "--extra-body",
        '{"temperature":0}',
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_name,
        *args.bench_arg,
    ]


def benchmark_result(
    payload: Any, expected_requests: int
) -> tuple[dict[str, float], str | None]:
    if not isinstance(payload, dict):
        raise ValueError("benchmark result must be a JSON object")
    completed = payload.get("completed")
    failed = payload.get("failed")
    if completed != expected_requests or failed != 0:
        raise RuntimeError(
            f"benchmark request gate failed: completed={completed}, failed={failed}, "
            f"expected={expected_requests}"
        )
    metrics = {}
    for name in METRICS:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"benchmark result is missing numeric {name}")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"benchmark result has invalid {name}: {value}")
        metrics[name] = value
    generated = payload.get("generated_texts")
    output_digest = None
    if generated is not None:
        if not isinstance(generated, list) or len(generated) != expected_requests:
            raise ValueError("generated_texts does not match the request count")
        encoded = json.dumps(
            generated, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        output_digest = hashlib.sha256(encoded).hexdigest()
    return metrics, output_digest


def wait_until_ready(process: subprocess.Popen[bytes], url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"server exited before readiness with status {status}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def stop_server(process: subprocess.Popen[bytes], timeout: float) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def run_one(
    server: ServerSpec,
    epoch: int,
    client: tuple[str, ...],
    args: argparse.Namespace,
    workload: Workload,
    root: Path,
) -> dict[str, Any]:
    run_dir = root / f"epoch-{epoch:03d}" / server.name
    run_dir.mkdir(parents=True, exist_ok=False)
    result_name = "benchmark.json"
    command = bench_command(client, server, args, workload, run_dir, result_name)
    server_log = (run_dir / "server.log").open("wb")
    process = subprocess.Popen(
        server.command,
        cwd=REPOSITORY_ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        wait_until_ready(
            process,
            f"{server.base_url}{args.health_path}",
            args.startup_timeout_seconds,
        )
        ready_seconds = time.monotonic() - started
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "client.log").write_bytes(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(
                f"vllm bench serve failed for {server.name} with status {completed.returncode}"
            )
        result_path = run_dir / result_name
        if not result_path.is_file():
            raise RuntimeError(f"benchmark did not create {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics, output_digest = benchmark_result(result, workload.requests)
        return {
            "server": server.name,
            "epoch": epoch,
            "ready_seconds": ready_seconds,
            "result": str(result_path.relative_to(root)),
            "client_command": command,
            "reported_completed": workload.requests,
            "reported_failed": 0,
            "metrics": metrics,
            "generated_texts_sha256": output_digest,
        }
    finally:
        stop_server(process, args.shutdown_timeout_seconds)
        server_log.close()


def git_revision() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "worktree_dirty": dirty}


def command_version(command: tuple[str, ...]) -> str | None:
    completed = subprocess.run(
        [*command, "--version"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    version = (completed.stdout or completed.stderr).strip()
    return version or None


def optional_command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def environment_snapshot(client: tuple[str, ...], dry_run: bool) -> dict[str, Any]:
    luminal = REPOSITORY_ROOT / "executor" / "luminal"
    luminal_commit = None
    if (luminal / ".git").exists():
        luminal_commit = optional_command_output(
            ["git", "-C", str(luminal), "rev-parse", "HEAD"]
        )
    return {
        "platform": platform.platform(),
        "python": sys.version.splitlines()[0],
        "rustc": optional_command_output(["rustc", "--version"]),
        "cargo": optional_command_output(["cargo", "--version"]),
        "benchmark_client": None if dry_run else command_version(client),
        "gpu": (
            None
            if dry_run
            else optional_command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            )
        ),
        "luminal_commit": luminal_commit,
    }


def paired_summary(runs: list[dict[str, Any]], epochs: int) -> dict[str, Any]:
    pairs = []
    for epoch in range(1, epochs + 1):
        by_server = {run["server"]: run for run in runs if run["epoch"] == epoch}
        if set(by_server) != {"candidate", "baseline"}:
            raise RuntimeError(f"epoch {epoch} does not contain one candidate and baseline")
        candidate = by_server["candidate"]
        baseline = by_server["baseline"]
        ratios = {
            name: candidate["metrics"][name] / baseline["metrics"][name]
            for name in METRICS
        }
        candidate_digest = candidate["generated_texts_sha256"]
        baseline_digest = baseline["generated_texts_sha256"]
        pairs.append(
            {
                "epoch": epoch,
                "candidate_over_baseline": ratios,
                "output_equivalence": (
                    None
                    if candidate_digest is None or baseline_digest is None
                    else candidate_digest == baseline_digest
                ),
            }
        )
    aggregates = {
        name: median(pair["candidate_over_baseline"][name] for pair in pairs)
        for name in METRICS
    }
    equivalence = [pair["output_equivalence"] for pair in pairs]
    return {
        "schema": "orbitkv.matched-serving-summary",
        "pairs": pairs,
        "median_candidate_over_baseline": aggregates,
        "ratio_interpretation": {
            "request_throughput": "greater_than_one_favors_candidate",
            "output_throughput": "greater_than_one_favors_candidate",
            "latency_metrics": "less_than_one_favors_candidate",
        },
        "request_completion_gate_passed": True,
        "output_equivalence_evaluated": all(value is not None for value in equivalence),
        "output_equivalence_passed": all(value is True for value in equivalence),
        "performance_qualified": False,
        "qualification_note": (
            "Raw paired metrics only. Apply predeclared statistical and lifecycle gates "
            "before promoting a benefit claim."
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        if args.epochs <= 0 or args.epochs % 2 != 0:
            raise ValueError("--epochs must be a positive even number")
        if args.startup_timeout_seconds <= 0 or args.shutdown_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        workload = load_workload(args.profiles_file.resolve(), args.profile)
        candidate = ServerSpec(
            "candidate",
            parse_command(
                args.candidate_command,
                "candidate",
                require_executable=not args.dry_run,
            ),
            normalize_url(args.candidate_url),
        )
        baseline = ServerSpec(
            "baseline",
            parse_command(
                args.baseline_command,
                "baseline",
                require_executable=not args.dry_run,
            ),
            normalize_url(args.baseline_url),
        )
        client = parse_command(
            args.vllm_command,
            "vllm client",
            require_executable=not args.dry_run,
        )
        root = output_directory(args.output_dir, args.profile)
        schedule = [
            [baseline, candidate] if epoch % 2 else [candidate, baseline]
            for epoch in range(1, args.epochs + 1)
        ]
        plan = {
            "schema": "orbitkv.matched-serving-run",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": git_revision(),
            "profile": args.profile,
            "workload": workload.__dict__,
            "model": args.model,
            "tokenizer": args.tokenizer,
            "backend": args.backend,
            "endpoint": args.endpoint,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "schedule": [[server.name for server in epoch] for epoch in schedule],
            "candidate_command": list(candidate.command),
            "baseline_command": list(baseline.command),
            "client_command": list(client),
            "client_style": args.client_style,
            "environment": environment_snapshot(client, args.dry_run),
        }
        if args.dry_run:
            plan["example_client_command"] = bench_command(
                client, candidate, args, workload, root / "epoch-001" / "candidate", "benchmark.json"
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        root.mkdir(parents=True, exist_ok=False)
        (root / "run.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        runs = []
        for epoch, servers in enumerate(schedule, start=1):
            for server in servers:
                print(f"epoch {epoch}: starting {server.name}", flush=True)
                runs.append(run_one(server, epoch, client, args, workload, root))
        summary = paired_summary(runs, args.epochs)
        (root / "completed.json").write_text(
            json.dumps(
                {"status": "completed", "runs": runs, "summary": summary},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(root)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as error:
        print(f"matched serving benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
