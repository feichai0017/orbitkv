"""Measure cold prefill, HBM hits, and external-cache reuse after HBM pressure.

Run from the repository root with the selected engine's Python environment:
python -m benches.single_node --engine sglang --backend orbitkv --model /path/to/model
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .launch import configure
from .metrics import summarize
from .runtime import ROOT, manifest, server, storage_manifest
from .workload import run_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument(
        "--backend",
        choices=["native", "cpu", "orbitkv", "lmcache", "flexkv"],
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Empty result directory (default: benches/results/runs/<timestamp>-<engine>-<backend>)",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--workload", choices=["serial", "concurrent", "sustained"], default="serial"
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60,
        help="Sustained admission window per concurrency; admitted requests drain afterward",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10000,
        help="Maximum sustained requests per window, bounding retained samples",
    )
    parser.add_argument(
        "--working-set",
        type=int,
        default=12,
        help="Number of independently prepared prefixes in a sustained window",
    )
    parser.add_argument(
        "--reuse-ratio",
        type=float,
        default=0.75,
        help="Probability of requesting a prepared prefix; other requests are cold",
    )
    parser.add_argument(
        "--query-budget-gib", type=float, help="OrbitKV global query ownership budget"
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--gpu-tokens", type=int, default=16384)
    parser.add_argument("--host-gib", type=int, default=16)
    parser.add_argument(
        "--ssd-gib",
        type=int,
        default=0,
        help="OrbitKV SSD capacity; sustained traffic naturally evicts DRAM, other workloads add a forced-eviction phase",
    )
    parser.add_argument("--orbitkv-transfer-backend", choices=["direct", "kernel"])
    parser.add_argument("--queue-warmup", choices=["on", "off"], default="off")
    parser.add_argument("--prepare-requests", choices=["on", "off"], default="off")
    parser.add_argument("--read-batch-mib", type=int, default=0)
    parser.add_argument("--read-timeout-ms", type=int, default=0)
    parser.add_argument("--read-max-batches", type=int, default=0)
    parser.add_argument("--trace-transfers", action="store_true")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--settle-seconds", type=float, default=1.2)
    args = parser.parse_args()
    if min(args.read_batch_mib, args.read_timeout_ms, args.read_max_batches) < 0:
        parser.error("read controls must be nonnegative")
    if args.prepare_requests == "on" and args.queue_warmup == "on":
        parser.error("compare prepared ownership and unowned warming in separate runs")
    if args.backend != "orbitkv" and (
        args.prepare_requests == "on"
        or args.read_batch_mib
        or args.read_timeout_ms
        or args.read_max_batches
    ):
        parser.error("request preparation and read controls require --backend orbitkv")
    if args.read_max_batches and not args.read_batch_mib:
        parser.error("--read-max-batches requires --read-batch-mib")
    if args.backend != "orbitkv" and (args.queue_warmup == "on" or args.trace_transfers):
        parser.error("--queue-warmup on and --trace-transfers require --backend orbitkv")
    args.model = args.model.resolve()
    args.output = (
        args.output
        or ROOT
        / "benches/results/runs"
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{args.engine}-{args.backend}"
    ).resolve()
    if args.orbitkv_transfer_backend and (args.engine != "vllm" or args.backend != "orbitkv"):
        parser.error("--orbitkv-transfer-backend requires --engine vllm --backend orbitkv")
    if args.ssd_gib < 0 or (args.ssd_gib and args.backend != "orbitkv"):
        parser.error("--ssd-gib must be nonnegative and requires --backend orbitkv")
    if args.query_budget_gib is not None and (
        args.backend != "orbitkv" or not 0 < args.query_budget_gib <= args.host_gib
    ):
        parser.error("--query-budget-gib requires OrbitKV and 0 < budget <= host capacity")
    if (
        not args.concurrencies
        or len(set(args.concurrencies)) != len(args.concurrencies)
        or any(c not in (1, 4, 8) for c in args.concurrencies)
    ):
        parser.error("--concurrencies must be distinct values from 1, 4, 8")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty so measurements cannot mix across runs")
    if (
        args.repeats < 1
        or len(set(args.lengths)) != len(args.lengths)
        or min(args.lengths) < 64
        or max(args.lengths) >= args.gpu_tokens * 3 // 4
    ):
        parser.error(
            "use positive repeats and distinct lengths between 64 and 3/4 of GPU token capacity"
        )
    if args.gpu_tokens % 64 or args.host_gib <= 0 or not 1 <= args.output_tokens <= 64:
        parser.error(
            "use page-aligned GPU capacity, positive host capacity, and 1–64 output tokens"
        )
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("--settle-seconds must be finite and nonnegative")
    if args.workload == "sustained":
        if not math.isfinite(args.duration_seconds) or args.duration_seconds < 1:
            parser.error("--duration-seconds must be finite and at least 1")
        if not 1 <= args.max_requests <= 100000 or not 1 <= args.working_set <= 1024:
            parser.error("use 1–100000 max requests and 1–1024 working-set prefixes")
        if not math.isfinite(args.reuse_ratio) or not 0 <= args.reuse_ratio <= 1:
            parser.error("--reuse-ratio must be finite and between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        parser.error("this fixed-capacity benchmark currently targets dense Qwen3")
    bytes_per_token = (
        2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * 2
    )

    launch = configure(args, bytes_per_token)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest(args, launch, bytes_per_token), indent=2, allow_nan=False) + "\n"
    )
    try:
        with contextlib.ExitStack() as stack:
            if launch.manager_command:
                manager = stack.enter_context(
                    server(
                        launch.manager_command,
                        launch.env,
                        launch.manager_url,
                        args.output / "manager.log",
                        launch.manager_health_path,
                    )
                )
                if args.ssd_gib:
                    (args.output / "storage.json").write_text(
                        json.dumps(
                            storage_manifest(manager.pid, args.output / "cache.bin"), indent=2
                        )
                        + "\n"
                    )
            stack.enter_context(
                server(launch.command, launch.env, launch.base_url, args.output / "engine.log")
            )
            if args.workload in ("concurrent", "sustained"):
                from . import concurrent, sustained

                workload = sustained if args.workload == "sustained" else concurrent
                samples, batches = workload.run_workload(args, launch.base_url, launch.manager_url)
                workload.validate(vars(args), samples, batches)
                summary = workload.summarize(samples, batches)
            else:
                samples = run_workload(args, launch.base_url, launch.manager_url)
                summary = summarize(samples, args.lengths)
            if args.trace_transfers and args.backend == "orbitkv":
                from .timeline import collect

                collect(args.output, samples)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n"
        )
        raise
    finally:
        if args.ssd_gib:
            (args.output / "cache.bin").unlink(missing_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
