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
    parser.add_argument("--workload", choices=["serial", "concurrent"], default="serial")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8])
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
        help="OrbitKV SSD capacity; adds a measured phase after evicting manager DRAM",
    )
    parser.add_argument("--orbitkv-transfer-backend", choices=["direct", "kernel"])
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--settle-seconds", type=float, default=1.2)
    args = parser.parse_args()
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
            if args.workload == "concurrent":
                from . import concurrent

                samples, batches = concurrent.run_workload(
                    args, launch.base_url, launch.manager_url
                )
                concurrent.validate(vars(args), samples, batches)
                summary = concurrent.summarize(samples, batches)
            else:
                samples = run_workload(args, launch.base_url, launch.manager_url)
                summary = summarize(samples, args.lengths)
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
