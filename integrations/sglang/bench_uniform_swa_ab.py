from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "integrations/sglang/bench_hybrid_shadow.py"


def compile_state_plan(args, path: Path) -> dict:
    command = [
        args.orbitkv_bin,
        "compile-hf-state-plan",
        str(Path(args.model) / "config.json"),
        "--page-tokens",
        str(args.page_size),
        "--kv-dtype-bytes",
        str(args.kv_dtype_bytes),
        "--boundary",
        str(args.context_length),
        "--max-running-requests",
        str(args.requests),
        "--chunked-prefill-tokens",
        str(args.chunked_prefill_tokens),
        "--eviction-interval",
        str(args.eviction_interval),
        "--decode-headroom-tokens",
        str(args.decode_tokens),
    ]
    artifact = json.loads(subprocess.check_output(command, text=True))
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_once(args, mode: str, state_plan: Path) -> dict:
    max_total_tokens = (
        args.execute_pool_tokens
        if mode == "state_plan"
        else args.reference_pool_tokens
    )
    command = [
        args.python,
        str(BENCH),
        "--mode",
        mode,
        "--load-format",
        "auto",
        "--model",
        args.model,
        "--plan",
        str(ROOT / "examples/full_swa.json"),
        "--state-plan",
        str(state_plan),
        "--requests",
        str(args.requests),
        "--max-running-requests",
        str(args.requests),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--iterations",
        "1",
        "--max-total-tokens",
        str(max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--page-size",
        str(args.page_size),
        "--attention-backend",
        args.attention_backend,
        "--moe-runner-backend",
        "auto",
        "--eviction-interval",
        str(args.eviction_interval),
        "--no-trace-allocations",
        "--orbitkv-bin",
        args.orbitkv_bin,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'integrations/sglang/src'}:/workspace/sglang/python"
    )
    environment["ORBITKV_SGLANG_REVISION"] = args.sglang_revision
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
        raise RuntimeError(f"no JSON output for {mode}: {completed.stderr}")
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
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=12000)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--chunked-prefill-tokens", type=int, default=2048)
    parser.add_argument("--eviction-interval", type=int, default=128)
    parser.add_argument("--execute-pool-tokens", type=int, default=19077)
    parser.add_argument("--reference-pool-tokens", type=int, default=50000)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--sglang-revision",
        default="095ec6c997bfdd25d3864cb0ce77a6562a934b96",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")

    with tempfile.TemporaryDirectory() as directory:
        state_plan_path = Path(directory) / "state-plan.json"
        state_plan = compile_state_plan(args, state_plan_path)
        minimum_pool = int(
            state_plan["sglang_lowering"]["contract"]["minimum_pool_tokens"]
        )
        if args.execute_pool_tokens != minimum_pool:
            raise ValueError(
                "execute pool must equal the compiled minimum: "
                f"{args.execute_pool_tokens} != {minimum_pool}"
            )

        pairs = []
        ratios = []
        for pair_index in range(args.pairs):
            order = (
                ("state_plan", "kernel_reference")
                if pair_index % 2 == 0
                else ("kernel_reference", "state_plan")
            )
            runs = {}
            for mode in order:
                print(f"pair={pair_index} mode={mode} starting", flush=True)
                runs[mode] = run_once(args, mode, state_plan_path)
            execute = runs["state_plan"]
            reference = runs["kernel_reference"]
            if execute["checkpoint"] != reference["checkpoint"]:
                raise RuntimeError("checkpoint identities differ")
            if execute["output_digest"] != reference["output_digest"]:
                raise RuntimeError("output token digests differ")
            if execute["completion_tokens"] != reference["completion_tokens"]:
                raise RuntimeError("completion token counts differ")
            if any(execute["num_retractions"]) or any(
                reference["num_retractions"]
            ):
                raise RuntimeError("a run retracted a request")
            ratio = (
                execute["iteration_seconds"][0]
                / reference["iteration_seconds"][0]
            )
            ratios.append(ratio)
            pairs.append(
                {
                    "pair": pair_index,
                    "execution_order": list(order),
                    "state_plan": execute,
                    "kernel_reference": reference,
                    "execute_over_reference_ratio": ratio,
                }
            )

    report = {
        "schema": "orbitkv.uniform-swa-sglang-ab.v1",
        "state_plan": state_plan,
        "pairs": pairs,
        "execute_over_reference_ratios": ratios,
        "median_execute_over_reference_ratio": statistics.median(ratios),
        "median_execute_over_reference_percent": (
            statistics.median(ratios) - 1
        )
        * 100,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
