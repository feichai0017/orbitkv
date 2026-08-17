from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "integrations/sglang/bench_hybrid_shadow.py"


def run_once(args, transport: str) -> dict:
    command = [
        args.python,
        str(BENCH),
        "--mode",
        "owner",
        "--owner-transport",
        transport,
        "--orbitkv-bin",
        args.orbitkv_bin,
        "--orbitkv-owner-lib",
        args.owner_library,
        "--model",
        args.model,
        "--plan",
        args.plan,
        "--requests",
        str(args.requests),
        "--max-running-requests",
        str(args.max_running_requests),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--iterations",
        "1",
        "--max-total-tokens",
        "0",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--eviction-interval",
        str(args.eviction_interval),
        "--no-trace-allocations",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'integrations/sglang/src'}:/workspace/sglang/python"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    if not records:
        raise RuntimeError(
            f"no JSON result for {transport}:\n{completed.stdout}\n{completed.stderr}"
        )
    result = records[-1]
    result["process_wall_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", default=str(ROOT / ".venv-sglang-h20/bin/python")
    )
    parser.add_argument(
        "--orbitkv-bin", default=str(ROOT / "target/release/orbitkv")
    )
    parser.add_argument(
        "--owner-library",
        default=str(
            ROOT / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"
        ),
    )
    parser.add_argument(
        "--model", default=str(ROOT / "fixtures/gpt-oss-hybrid-62l")
    )
    parser.add_argument(
        "--plan", default=str(ROOT / "examples/gpt_oss_hybrid_62l.json")
    )
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=6000)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--mem-fraction-static", type=float, default=0.05)
    parser.add_argument("--eviction-interval", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")

    pairs = []
    ratios = []
    for pair_index in range(args.pairs):
        order = (
            ("sidecar", "ffi")
            if pair_index % 2 == 0
            else ("ffi", "sidecar")
        )
        runs_by_transport = {}
        for transport in order:
            print(
                f"pair={pair_index} transport={transport} starting",
                flush=True,
            )
            result = run_once(args, transport)
            runs_by_transport[transport] = result
            print(
                f"pair={pair_index} transport={transport} "
                f"workload={result['iteration_seconds'][0]:.6f}s "
                f"digest={result['output_digest'][:12]} "
                f"capacity={result['server_memory']['token_capacity']}",
                flush=True,
            )
        sidecar = runs_by_transport["sidecar"]
        ffi = runs_by_transport["ffi"]
        if sidecar["output_digest"] != ffi["output_digest"]:
            raise RuntimeError("Sidecar and FFI output digests differ")
        if sidecar["server_memory"] != ffi["server_memory"]:
            raise RuntimeError("Sidecar and FFI memory reports differ")
        ratio = ffi["iteration_seconds"][0] / sidecar["iteration_seconds"][0]
        ratios.append(ratio)
        pairs.append(
            {
                "pair": pair_index,
                "execution_order": list(order),
                "sidecar": sidecar,
                "ffi": ffi,
                "ffi_over_sidecar_ratio": ratio,
            }
        )

    report = {
        "schema": "orbitkv.h20-owner-ffi-ab.v1",
        "pairs": pairs,
        "ffi_over_sidecar_ratios": ratios,
        "median_ffi_over_sidecar_ratio": statistics.median(ratios),
        "median_ffi_over_sidecar_percent": (
            statistics.median(ratios) - 1
        )
        * 100,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
