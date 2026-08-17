from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "integrations/sglang/bench_hybrid_shadow.py"
MODES = ("stock128", "stock32", "policy32", "owner32")
EXECUTION_ORDERS = (
    ("stock128", "stock32", "policy32", "owner32"),
    ("owner32", "policy32", "stock32", "stock128"),
    ("stock32", "owner32", "stock128", "policy32"),
    ("policy32", "stock128", "owner32", "stock32"),
)


def compile_model_plan(
    args, retention_path: Path, physical_plan_path: Path
) -> dict:
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
        str(args.requests),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--candidate-intervals",
        args.candidate_intervals,
        "--max-reclamation-calls",
        str(args.max_reclamation_calls),
        "--min-admitted-requests",
        str(args.min_admitted_requests),
        "--objective",
        args.objective,
    ]
    report = json.loads(subprocess.check_output(command, text=True))
    program = report["compilation"]["program"]
    retention_path.write_text(
        json.dumps(program, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    physical_plan_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report["compilation"]["layer_inference"] != "explicit_layer_types":
        raise RuntimeError(
            "real hybrid ablation requires explicit checkpoint layer_types"
        )
    if not any(state["name"] == "swa" for state in program["states"]):
        raise RuntimeError("compiled checkpoint does not contain an SWA state")
    if report["physical_plan"]["selected_eviction_interval_tokens"] != 32:
        raise RuntimeError("four-way ablation requires the optimizer to select 32")
    return report


def bench_mode(mode: str) -> tuple[str, int]:
    if mode == "stock128":
        return "stock", 128
    if mode == "stock32":
        return "native_policy", 32
    if mode == "policy32":
        return "policy", 32
    if mode == "owner32":
        return "owner", 32
    raise ValueError(f"unknown ablation mode: {mode}")


def run_once(
    args,
    mode: str,
    plan_path: Path,
    physical_plan_path: Path,
    round_index: int,
) -> dict:
    bench, interval = bench_mode(mode)
    command = [
        args.python,
        str(BENCH),
        "--mode",
        bench,
        "--load-format",
        "auto",
        "--orbitkv-bin",
        args.orbitkv_bin,
        "--orbitkv-owner-lib",
        args.owner_library,
        "--owner-transport",
        "ffi",
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
        str(args.max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--eviction-interval",
        str(interval),
        "--attention-backend",
        args.attention_backend,
        "--moe-runner-backend",
        args.moe_runner_backend,
        "--no-trace-allocations",
    ]
    if mode in ("policy32", "owner32"):
        command.extend(["--physical-plan", str(physical_plan_path)])
    if args.mem_fraction_static is not None:
        command.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
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
            f"no JSON result for round={round_index} mode={mode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    result = records[-1]
    result["ablation_mode"] = mode
    result["process_wall_seconds"] = time.perf_counter() - started
    if result["checkpoint"]["load_format"] != "auto":
        raise RuntimeError(f"{mode} did not use load_format=auto")
    if not result["checkpoint"]["indexed_weights_complete"]:
        raise RuntimeError(f"{mode} observed an incomplete checkpoint")
    if result["eviction_interval"] != interval:
        raise RuntimeError(f"{mode} used the wrong eviction interval")
    if mode in ("stock128", "stock32") and result["owner_transport"] is not None:
        raise RuntimeError(f"{mode} unexpectedly loaded an OrbitKV owner")
    return result


def validate_round(runs: dict[str, dict]) -> None:
    checkpoint = runs[MODES[0]]["checkpoint"]
    digest = runs[MODES[0]]["output_digest"]
    completion_tokens = runs[MODES[0]]["completion_tokens"]
    runtime = runs[MODES[0]]["resolved_runtime"]
    for mode in MODES:
        result = runs[mode]
        if result["checkpoint"] != checkpoint:
            raise RuntimeError(f"{mode} checkpoint identity differs")
        if result["output_digest"] != digest:
            raise RuntimeError(f"{mode} output token digest differs")
        if result["completion_tokens"] != completion_tokens:
            raise RuntimeError(f"{mode} completion count differs")
        if result["resolved_runtime"] != runtime:
            raise RuntimeError(f"{mode} resolved SGLang runtime differs")
        if any(result["num_retractions"]):
            raise RuntimeError(f"{mode} retracted a request")


def ratio(numerator: dict, denominator: dict) -> float:
    return numerator["iteration_seconds"][0] / denominator["iteration_seconds"][0]


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
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=6000)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--available-kv-bytes", type=int, default=2_123_759_616)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.18)
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--attention-dp-size", type=int, default=1)
    parser.add_argument("--chunked-prefill-tokens", type=int, default=2048)
    parser.add_argument("--candidate-intervals", default="16,32,64,128")
    parser.add_argument("--max-reclamation-calls", type=int, default=4)
    parser.add_argument("--min-admitted-requests", type=int, default=8)
    parser.add_argument(
        "--objective", choices=("capacity", "reclamation"), default="capacity"
    )
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--moe-runner-backend", default="auto")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")

    with tempfile.TemporaryDirectory(prefix="orbitkv-hf-plan-") as directory:
        plan_path = Path(directory) / "retention.json"
        physical_plan_path = Path(directory) / "physical-plan.json"
        compiler_report = compile_model_plan(
            args, plan_path, physical_plan_path
        )
        rounds = []
        contributions = {
            "physical_policy_stock32_over_stock128": [],
            "compiler_policy_policy32_over_stock32": [],
            "ownership_owner32_over_policy32": [],
        }
        for round_index in range(args.rounds):
            order = EXECUTION_ORDERS[round_index % len(EXECUTION_ORDERS)]
            runs = {}
            for mode in order:
                print(f"round={round_index} mode={mode} starting", flush=True)
                result = run_once(
                    args,
                    mode,
                    plan_path,
                    physical_plan_path,
                    round_index,
                )
                runs[mode] = result
                print(
                    f"round={round_index} mode={mode} "
                    f"workload={result['iteration_seconds'][0]:.6f}s "
                    f"capacity={result['server_memory'].get('token_capacity')} "
                    f"digest={result['output_digest'][:12]}",
                    flush=True,
                )
            validate_round(runs)
            contributions[
                "physical_policy_stock32_over_stock128"
            ].append(ratio(runs["stock32"], runs["stock128"]))
            contributions[
                "compiler_policy_policy32_over_stock32"
            ].append(ratio(runs["policy32"], runs["stock32"]))
            contributions[
                "ownership_owner32_over_policy32"
            ].append(ratio(runs["owner32"], runs["policy32"]))
            rounds.append(
                {
                    "round": round_index,
                    "execution_order": list(order),
                    "runs": runs,
                }
            )

    report = {
        "schema": "orbitkv.real-model-four-way-ablation.v1",
        "model": str(Path(args.model).resolve()),
        "compiler_report": compiler_report,
        "rounds": rounds,
        "contribution_ratios": contributions,
        "median_contribution_ratios": {
            name: statistics.median(values)
            for name, values in contributions.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
