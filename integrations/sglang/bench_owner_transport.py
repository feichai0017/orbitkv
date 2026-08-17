from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from orbitkv_sglang.plugin import FfiOwnerClient, SidecarOwnerClient


ROOT = Path(__file__).resolve().parents[2]


def load_policy(orbitkv_bin: Path, plan: Path) -> dict:
    completed = subprocess.run(
        [
            str(orbitkv_bin),
            "emit-sglang-policy",
            str(plan),
            "--eviction-interval",
            "32",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_trial(
    transport: str,
    *,
    orbitkv_bin: Path,
    owner_library: Path,
    plan: Path,
    policy: dict,
    cycles: int,
) -> dict:
    started = time.perf_counter_ns()
    if transport == "ffi":
        owner = FfiOwnerClient(str(owner_library), str(plan), policy)
    else:
        owner = SidecarOwnerClient(str(orbitkv_bin), str(plan))
    startup_ns = time.perf_counter_ns() - started

    page_tokens = int(policy["page_tokens"])
    window_tokens = int(policy["bounded_classes"][0]["window_tokens"])
    latencies = []
    request_id = f"transport-{transport}"
    try:
        observed = 0
        for cycle in range(cycles):
            command_started = time.perf_counter_ns()
            response = owner.command(
                {
                    "op": "plan_reclamation",
                    "request_id": request_id,
                    "observed_evicted_seqlen": observed,
                    "semantic_frontier": observed + window_tokens + page_tokens,
                    "execution_epoch": cycle + 1,
                    "cache_kind": "chunk",
                }
            )
            certificate = response["certificate"]
            if certificate is None:
                raise RuntimeError("transport benchmark expected a certificate")
            owner.command(
                {
                    "op": "commit_reclamations",
                    "certificate_ids": [int(certificate["certificate_id"])],
                }
            )
            latencies.append(time.perf_counter_ns() - command_started)
            observed = int(certificate["token_end_exclusive"])
        stats = owner.command({"op": "stats"})["stats"]
        owner.command({"op": "release_request", "request_id": request_id})
    finally:
        owner.close()

    return {
        "transport": transport,
        "cycles": cycles,
        "startup_ns": startup_ns,
        "plan_commit_ns_median": statistics.median(latencies),
        "plan_commit_ns_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "plan_commit_ns_mean": statistics.fmean(latencies),
        "committed_reclamations": stats["committed_reclamations"],
        "committed_tokens": stats["committed_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(ROOT / "examples/full_swa.json"))
    parser.add_argument("--orbitkv-bin", default=str(ROOT / "target/debug/orbitkv"))
    parser.add_argument(
        "--owner-library",
        default=str(
            ROOT / "crates/orbitkv-ffi/target/debug/liborbitkv_ffi.so"
        ),
    )
    parser.add_argument("--cycles", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.cycles <= 0 or args.trials <= 0:
        raise ValueError("cycles and trials must be positive")

    plan = Path(args.plan).resolve()
    orbitkv_bin = Path(args.orbitkv_bin).resolve()
    owner_library = Path(args.owner_library).resolve()
    policy = load_policy(orbitkv_bin, plan)
    trials = []
    for trial in range(args.trials):
        for transport in ("sidecar", "ffi"):
            result = run_trial(
                transport,
                orbitkv_bin=orbitkv_bin,
                owner_library=owner_library,
                plan=plan,
                policy=policy,
                cycles=args.cycles,
            )
            result["trial"] = trial
            trials.append(result)

    sidecar = [
        result["plan_commit_ns_median"]
        for result in trials
        if result["transport"] == "sidecar"
    ]
    ffi = [
        result["plan_commit_ns_median"]
        for result in trials
        if result["transport"] == "ffi"
    ]
    summary = {
        "schema": "orbitkv.owner-transport-benchmark.v1",
        "plan_fingerprint": policy["plan_fingerprint"],
        "cycles_per_trial": args.cycles,
        "trials_per_transport": args.trials,
        "sidecar_plan_commit_ns_median": statistics.median(sidecar),
        "ffi_plan_commit_ns_median": statistics.median(ffi),
        "median_speedup": statistics.median(sidecar) / statistics.median(ffi),
        "trials": trials,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
