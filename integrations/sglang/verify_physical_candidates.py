from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "integrations/sglang/bench_hybrid_shadow.py"


def compile_physical_plan(args, path: Path) -> dict:
    command = [
        args.orbitkv_bin,
        "compile-hf-physical-plan",
        str(Path(args.model).resolve() / "config.json"),
        "--page-tokens",
        str(args.page_tokens),
        "--kv-dtype-bytes",
        str(args.kv_dtype_bytes),
        "--available-kv-bytes",
        str(args.available_kv_bytes),
        "--max-running-requests",
        str(args.max_running_requests),
        "--attention-dp-size",
        str(args.attention_dp_size),
        "--chunked-prefill-tokens",
        str(args.chunked_prefill_tokens),
        "--workload-requests",
        str(args.workload_requests),
        "--prompt-tokens",
        str(args.workload_prompt_tokens),
        "--decode-tokens",
        str(args.workload_decode_tokens),
        "--candidate-intervals",
        args.candidate_intervals,
        "--max-reclamation-calls",
        str(args.max_reclamation_calls),
        "--min-admitted-requests",
        str(args.min_admitted_requests),
        "--objective",
        args.objective,
    ]
    artifact = json.loads(subprocess.check_output(command, text=True))
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def run_candidate(args, interval: int, plan_path: Path) -> dict:
    command = [
        args.python,
        str(BENCH),
        "--mode",
        "native_policy",
        "--load-format",
        "auto",
        "--model",
        args.model,
        "--plan",
        str(plan_path),
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
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--eviction-interval",
        str(interval),
        "--attention-backend",
        args.attention_backend,
        "--moe-runner-backend",
        args.moe_runner_backend,
        "--no-trace-allocations",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'integrations/sglang/src'}:/workspace/sglang/python"
    )
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
            f"no candidate result for interval {interval}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return records[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", default=str(ROOT / ".venv-sglang-h20/bin/python")
    )
    parser.add_argument(
        "--orbitkv-bin", default=str(ROOT / "target/release/orbitkv")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--available-kv-bytes", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--attention-dp-size", type=int, default=1)
    parser.add_argument("--chunked-prefill-tokens", type=int, default=2048)
    parser.add_argument("--workload-requests", type=int, default=8)
    parser.add_argument("--workload-prompt-tokens", type=int, default=6000)
    parser.add_argument("--workload-decode-tokens", type=int, default=32)
    parser.add_argument("--candidate-intervals", default="16,32,64,128")
    parser.add_argument("--max-reclamation-calls", type=int, default=4)
    parser.add_argument("--min-admitted-requests", type=int, default=8)
    parser.add_argument(
        "--objective", choices=("capacity", "reclamation"), default="capacity"
    )
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.18)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--moe-runner-backend", default="auto")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="orbitkv-candidates-") as directory:
        artifact_path = Path(directory) / "physical-plan.json"
        artifact = compile_physical_plan(args, artifact_path)
        retention_path = Path(directory) / "retention.json"
        retention_path.write_text(
            json.dumps(
                artifact["compilation"]["program"], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        candidates = {
            int(candidate["eviction_interval_tokens"]): candidate
            for candidate in artifact["physical_plan"]["candidates"]
        }
        observations = []
        checkpoint = None
        digest = None
        runtime = None
        for interval in sorted(candidates):
            print(f"interval={interval} starting", flush=True)
            result = run_candidate(args, interval, retention_path)
            predicted = candidates[interval]["cost"]
            actual = result["server_memory"]
            if int(actual["token_capacity"]) != int(
                predicted["full_token_capacity"]
            ):
                raise RuntimeError(
                    f"interval {interval}: Full capacity mismatch "
                    f"{actual['token_capacity']} != {predicted['full_token_capacity']}"
                )
            if int(actual["token_capacity_swa"]) != int(
                predicted["physical_swa_token_slots"]
            ):
                raise RuntimeError(
                    f"interval {interval}: SWA capacity mismatch "
                    f"{actual['token_capacity_swa']} != "
                    f"{predicted['physical_swa_token_slots']}"
                )
            if checkpoint is None:
                checkpoint = result["checkpoint"]
                digest = result["output_digest"]
                runtime = result["resolved_runtime"]
            if result["checkpoint"] != checkpoint:
                raise RuntimeError(f"interval {interval}: checkpoint differs")
            if result["output_digest"] != digest:
                raise RuntimeError(f"interval {interval}: output digest differs")
            if result["resolved_runtime"] != runtime:
                raise RuntimeError(f"interval {interval}: runtime differs")
            observations.append(
                {
                    "interval": interval,
                    "predicted": predicted,
                    "actual_server_memory": actual,
                    "prediction_matches": True,
                    "output_digest": result["output_digest"],
                    "num_retractions": result["num_retractions"],
                }
            )
            print(
                f"interval={interval} "
                f"full={actual['token_capacity']} "
                f"swa={actual['token_capacity_swa']} matched",
                flush=True,
            )

    report = {
        "schema": "orbitkv.sglang-physical-candidate-validation.v1",
        "artifact": artifact,
        "observations": observations,
        "checkpoint": checkpoint,
        "resolved_runtime": runtime,
        "output_digest": digest,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
