from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import bench_canonical_manager as common


RECORD_SCHEMA = "orbitkv.sglang-v0517-token-relocation-single-run.v1"
TRIGGER_TOKENS = 48
RETAINED_PER_PAGE = 8
DECODE_TOKENS = 17
VICTIM_COUNT = 24


def _stage(name: str) -> None:
    print(f"[orbitkv-relocation] {name}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one matched Full-only Naive-Evict or Relocate H20 sample."
    )
    parser.add_argument("--mode", choices=("naive", "relocate"), required=True)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--requests", type=int, choices=(1, 4), required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument(
        "--attention-backend", choices=("flashinfer", "fa3"), required=True
    )
    parser.add_argument("--output")
    return parser


def validate_arguments(args: argparse.Namespace) -> dict[str, Path]:
    if args.iterations <= 0 or args.max_total_tokens <= 0:
        raise ValueError("iterations and max-total-tokens must be positive")
    if args.max_total_tokens % common.PAGE_TOKENS:
        raise ValueError("max-total-tokens must be page aligned")
    if args.context_length <= TRIGGER_TOKENS + DECODE_TOKENS:
        raise ValueError("context-length must exceed the complete workload")
    if args.requests * (TRIGGER_TOKENS + DECODE_TOKENS) > args.max_total_tokens:
        raise ValueError("max-total-tokens cannot hold the complete batch")
    if args.mem_fraction_static is not None and not 0 < args.mem_fraction_static <= 1:
        raise ValueError("mem-fraction-static must be in (0, 1]")
    return {
        "sglang_root": common._directory(args.sglang_root, "--sglang-root"),
        "model": common._directory(args.model, "--model"),
        "plan": common._regular_file(args.plan, "--plan"),
        "library": common._regular_file(args.library, "--library"),
    }


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mode="manager",
        sglang_root=args.sglang_root,
        model=args.model,
        plan=args.plan,
        library=args.library,
        requests=args.requests,
        max_running_requests=args.requests,
        prompt_tokens=TRIGGER_TOKENS,
        decode_tokens=DECODE_TOKENS,
        iterations=args.iterations,
        chunked_prefill_size=args.requests * TRIGGER_TOKENS,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        seed=args.seed,
    )


def _policy(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "trigger_tokens": TRIGGER_TOKENS,
        "retained_per_page": RETAINED_PER_PAGE,
        "policy_id": 260813263,
        "policy_version": 1,
        "quality_contract": 1,
        "fragmentation_threshold_milli": 500,
        "maximum_source_pages": 3,
        "evacuation_headroom_pages": 2,
    }


def _manager_state(info: dict[str, Any], stage: str) -> dict[str, Any]:
    state = common._state(info).get("orbitkv_manager")
    if not isinstance(state, dict) or state.get("abi_version") != 7:
        raise RuntimeError(f"ABI7 manager state is missing at {stage}")
    stats = state.get("manager_stats")
    arenas = state.get("arena_stats")
    counters = state.get("batch_counters")
    if (
        not isinstance(stats, dict)
        or not isinstance(arenas, list)
        or len(arenas) != 1
        or not isinstance(counters, dict)
    ):
        raise RuntimeError(f"manager census is malformed at {stage}")
    forbidden = (
        "quarantined_pages", "exhausted_pages", "prepared_steps",
        "submitted_steps", "pending_reclamations", "fail_stops",
        "fail_stop_count", "quarantine_count",
    )
    if any(int(stats.get(name, counters.get(name, 0))) for name in forbidden):
        raise RuntimeError(f"manager fail-stop census is nonzero at {stage}")
    return state


def _validate_final_census(
    state: dict[str, Any], mode: str, requests: int, iterations: int, class_count: int
) -> None:
    stats = state["manager_stats"]
    arena = state["arena_stats"][0]
    counters = state["batch_counters"]
    if (
        stats["active_requests"]
        or stats["active_snapshots"]
        or stats["active_pages"]
        or stats["total_request_page_refs"]
        or stats["total_prefix_page_refs"]
        or stats["total_reader_pins"]
        or arena["free_pages"] != arena["page_count"]
    ):
        raise RuntimeError("manager did not drain after the relocation workload")
    operations = requests * iterations
    expected = {
        "token_disposition_batches": operations,
        "token_policy_evictions": operations * VICTIM_COUNT * class_count,
        "mark_token_dispositions_batch_calls": operations,
    }
    if mode == "relocate":
        expected.update(
            relocation_batches=operations,
            relocation_moves=operations * (TRIGGER_TOKENS - VICTIM_COUNT),
            relocation_reclaimed_pages=operations,
            relocation_copy_events=operations,
            relocation_copy_tokens=operations * (TRIGGER_TOKENS - VICTIM_COUNT),
            prepare_relocation_batch_calls=operations,
            submit_relocation_batch_calls=operations,
            complete_relocation_batch_calls=operations,
        )
    else:
        expected.update(
            relocation_batches=0, relocation_moves=0, relocation_reclaimed_pages=0,
            relocation_copy_events=0, relocation_copy_tokens=0,
            prepare_relocation_batch_calls=0, submit_relocation_batch_calls=0,
            complete_relocation_batch_calls=0,
        )
    mismatch = {
        name: {"expected": value, "actual": counters.get(name)}
        for name, value in expected.items()
        if counters.get(name) != value
    }
    if mismatch:
        raise RuntimeError(f"relocation counters differ from the workload: {mismatch}")


def run(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    started = time.perf_counter()
    _stage("freeze-environment")
    base = _base_args(args)
    environment = common.configure_environment(base, paths)
    environment["ORBITKV_TOKEN_RECLAMATION"] = json.dumps(
        _policy(args.mode), sort_keys=True, separators=(",", ":")
    )
    os.environ["ORBITKV_TOKEN_RECLAMATION"] = environment[
        "ORBITKV_TOKEN_RECLAMATION"
    ]
    _stage("verify-source")
    source = common.verify_sglang_source(paths["sglang_root"], "manager")
    source["plugin_selection"] = common.verify_manager_entrypoint()
    source["harness_sha256"] = common.sha256_file(Path(__file__).resolve())
    source["adapter"] = common._adapter_identity()
    source["library"] = common.artifact_identity(paths["library"])
    source["plan"] = common.artifact_identity(paths["plan"])

    from orbitkv_sglang.config import load_config

    _stage("hash-checkpoint")
    manager_config = load_config()
    contract, checkpoint = common.checkpoint_contract(paths["model"], manager_config)
    if contract["attention_profile"] not in ("full", "hybrid_full_swa"):
        raise RuntimeError("token-relocation qualification requires Full or Full+SWA")
    prompts = tuple(
        tuple(value)
        for value in common.deterministic_input_ids(
            requests=args.requests,
            prompt_tokens=TRIGGER_TOKENS,
            vocab_size=contract["vocab_size"],
            seed=args.seed,
        )
    )
    engine_args = common.engine_arguments(base, paths["model"], contract)
    sampling = {
        "temperature": 0, "max_new_tokens": DECODE_TOKENS,
        "min_new_tokens": DECODE_TOKENS, "ignore_eos": True,
        "sampling_seed": args.seed,
    }
    _stage("snapshot-before-engine")
    before = common.gpu_snapshot("before_engine")
    import sglang as sgl

    outputs: list[list[dict[str, Any]]] = []
    seconds: list[float] = []
    started_at = datetime.now(timezone.utc).isoformat()
    load_started = time.perf_counter()
    _stage("load-engine")
    with sgl.Engine(**engine_args) as engine:
        load_seconds = time.perf_counter() - load_started
        after_load = common.gpu_snapshot("after_load")
        info_load = engine.get_server_info()
        common.verify_runtime_contract(base, info_load, contract)
        state_load = _manager_state(info_load, "after_load")
        _stage("run-workload")
        for iteration in range(args.iterations):
            batch_inputs = [list(item) for item in prompts]
            rids = [
                f"orbitkv-relocation-{args.mode}-{args.seed}-{iteration}-{index}"
                for index in range(args.requests)
            ]
            begin = time.perf_counter()
            raw = engine.generate(
                input_ids=batch_inputs, rid=rids, sampling_params=sampling
            )
            seconds.append(time.perf_counter() - begin)
            normalized = common._normalize_outputs(raw, args.requests)
            if any(len(item["output_ids"]) != DECODE_TOKENS for item in normalized):
                raise RuntimeError("relocation workload returned the wrong token count")
            if any(item.get("meta_info", {}).get("cached_tokens") != 0 for item in normalized):
                raise RuntimeError("relocation workload unexpectedly reused Prefix KV")
            outputs.append(normalized)
        _stage("validate-drain")
        info_final = engine.get_server_info()
        common.verify_runtime_contract(base, info_final, contract)
        state_final = _manager_state(info_final, "final")
        _validate_final_census(
            state_final, args.mode, args.requests, args.iterations,
            len(contract["classes"])
        )
        after_workload = common.gpu_snapshot("after_workload")
    _stage("snapshot-after-shutdown")
    after_shutdown = common.gpu_snapshot("after_shutdown")

    workload = {
        "requests": args.requests, "iterations": args.iterations,
        "prompt_tokens": TRIGGER_TOKENS, "decode_tokens": DECODE_TOKENS,
        "victim_count_per_request": VICTIM_COUNT,
        "retained_count_at_trigger": TRIGGER_TOKENS - VICTIM_COUNT,
        "input_token_digest_sha256": common.input_digest(prompts),
    }
    pair = {
        "checkpoint_identity_sha256": common.canonical_digest(checkpoint),
        "engine_args": common.pair_engine_arguments("manager", engine_args),
        "sampling_params": sampling, "workload": workload,
        "capacity_tokens": args.max_total_tokens,
        "victim_policy": {key: value for key, value in _policy(args.mode).items() if key != "mode"},
    }
    _stage("complete")
    return {
        "schema": RECORD_SCHEMA, "mode": args.mode,
        "started_at_utc": started_at,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "environment": environment, "source_identity": source,
        "runtime_identity": {
            "python": sys.executable, "python_version": platform.python_version(),
            "sglang_version": sgl.__version__, "gpu_profile": "single H20 eager",
        },
        "checkpoint": checkpoint, "checkpoint_contract": contract,
        "engine_args": engine_args, "sampling_params": sampling,
        "workload": workload, "pairing": {
            "pair_key_sha256": common.canonical_digest(pair), "contract": pair,
            "only_allowed_difference": "naive versus byte-exact relocation",
        },
        "load_seconds": load_seconds, "iteration_seconds": seconds,
        "iteration_total_seconds": sum(seconds),
        "total_seconds": time.perf_counter() - started,
        "output_token_digest_sha256": common.token_digest(outputs),
        "request_output_ids": [[item["output_ids"] for item in row] for row in outputs],
        "manager": {"after_load": state_load, "final_census": state_final},
        "gpu_snapshots": [before, after_load, after_workload, after_shutdown],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args, validate_arguments(args))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    encoded = json.dumps(result, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if not output.parent.is_dir():
            parser.error("--output parent directory does not exist")
        output.write_text(encoded + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"output": str(output), "record_sha256": common.sha256_file(output)},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(encoded, flush=True)


if __name__ == "__main__":
    main()
