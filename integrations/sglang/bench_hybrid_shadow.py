from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import sglang as sgl

from checkpoint_identity import checkpoint_identity


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures/gpt-oss-hybrid-tiny"
)
DEFAULT_PLAN_PATH = (
    Path(__file__).resolve().parents[2] / "examples/gpt_oss_hybrid_tiny.json"
)


def gpu_snapshot() -> dict:
    fields = [
        "name",
        "uuid",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    ]
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    return dict(zip(fields, values, strict=True))


def extract_memory(info: dict) -> dict:
    states = info.get("internal_states", [])
    state = states[0] if states else {}
    return state.get("memory_usage", {})


def extract_runtime(info: dict) -> dict:
    states = info.get("internal_states", [])
    state = states[0] if states else {}
    fields = (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
        "moe_runner_backend",
        "quantization",
        "dtype",
        "kv_cache_dtype",
        "page_size",
        "chunked_prefill_size",
        "disable_hybrid_swa_memory",
        "disable_radix_cache",
        "disable_overlap_schedule",
        "max_running_requests",
        "effective_max_running_requests_per_dp",
    )
    return {
        field: state.get(field, info.get(field))
        for field in fields
        if state.get(field, info.get(field)) is not None
    }


def token_digest(outputs: list[dict]) -> str:
    digest = hashlib.sha256()
    for output in outputs:
        digest.update(json.dumps(output["output_ids"], separators=(",", ":")).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("stock", "native_policy", "shadow", "policy", "owner"),
        required=True,
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--physical-plan")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--load-format", choices=("auto", "dummy"), default="dummy")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--moe-runner-backend", default="triton")
    parser.add_argument("--trace", default="/tmp/orbitkv-hybrid-shadow.jsonl")
    parser.add_argument("--orbitkv-bin", default=str(Path(__file__).resolve().parents[2] / "target/debug/orbitkv"))
    parser.add_argument(
        "--orbitkv-owner-lib",
        default=str(
            Path(__file__).resolve().parents[2]
            / "crates/orbitkv-ffi/target/debug/liborbitkv_ffi.so"
        ),
    )
    parser.add_argument(
        "--owner-transport", choices=("ffi", "sidecar"), default="ffi"
    )
    parser.add_argument("--eviction-interval", type=int, default=16)
    parser.add_argument("--no-trace-allocations", action="store_true")
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    plan_path = Path(args.plan).resolve()
    max_running_requests = args.max_running_requests or args.requests

    if args.mode in ("stock", "native_policy"):
        os.environ.pop("SGLANG_PLUGINS", None)
        if args.mode == "native_policy":
            os.environ["SGLANG_SWA_EVICTION_INTERVAL"] = str(
                args.eviction_interval
            )
        else:
            os.environ.pop("SGLANG_SWA_EVICTION_INTERVAL", None)
        os.environ.pop("ORBITKV_TRACE_PATH", None)
        os.environ.pop("ORBITKV_SGLANG_POLICY", None)
        os.environ.pop("ORBITKV_SGLANG_PHYSICAL_PLAN", None)
        os.environ.pop("ORBITKV_SGLANG_EVICTION_INTERVAL", None)
        os.environ.pop("ORBITKV_SGLANG_OWNING", None)
        os.environ.pop("ORBITKV_OWNER_TRANSPORT", None)
        os.environ.pop("ORBITKV_OWNER_LIB", None)
        os.environ.pop("ORBITKV_BIN", None)
        os.environ.pop("ORBITKV_TRACE_ALLOCATIONS", None)
        os.environ.pop("ORBITKV_SGLANG_REVISION", None)
    else:
        os.environ["SGLANG_PLUGINS"] = "orbitkv_shadow"
        os.environ["ORBITKV_TRACE_PATH"] = args.trace
        os.environ["ORBITKV_BIN"] = args.orbitkv_bin
        os.environ["ORBITKV_TRACE_ALLOCATIONS"] = (
            "0" if args.no_trace_allocations else "1"
        )
        if args.mode in ("policy", "owner"):
            os.environ["ORBITKV_SGLANG_POLICY"] = str(plan_path)
            if args.physical_plan:
                os.environ["ORBITKV_SGLANG_PHYSICAL_PLAN"] = str(
                    Path(args.physical_plan).resolve()
                )
                os.environ.pop("ORBITKV_SGLANG_EVICTION_INTERVAL", None)
            else:
                os.environ.pop("ORBITKV_SGLANG_PHYSICAL_PLAN", None)
                os.environ["ORBITKV_SGLANG_EVICTION_INTERVAL"] = str(
                    args.eviction_interval
                )
            os.environ["ORBITKV_SGLANG_OWNING"] = (
                "1" if args.mode == "owner" else "0"
            )
            os.environ["ORBITKV_OWNER_TRANSPORT"] = args.owner_transport
            if args.owner_transport == "ffi":
                os.environ["ORBITKV_OWNER_LIB"] = args.orbitkv_owner_lib
            else:
                os.environ.pop("ORBITKV_OWNER_LIB", None)
        else:
            os.environ.pop("ORBITKV_SGLANG_POLICY", None)
            os.environ.pop("ORBITKV_SGLANG_PHYSICAL_PLAN", None)
            os.environ.pop("ORBITKV_SGLANG_EVICTION_INTERVAL", None)
            os.environ.pop("ORBITKV_SGLANG_OWNING", None)
            os.environ.pop("ORBITKV_OWNER_TRANSPORT", None)
            os.environ.pop("ORBITKV_OWNER_LIB", None)
        Path(args.trace).unlink(missing_ok=True)

    engine_args = dict(
        model_path=str(model_path),
        load_format=args.load_format,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        context_length=args.context_length,
        page_size=16,
        moe_runner_backend=args.moe_runner_backend,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=2048,
        max_running_requests=max_running_requests,
        random_seed=20260816,
        log_level="error",
    )
    if args.max_total_tokens > 0:
        engine_args["max_total_tokens"] = args.max_total_tokens
    if args.mem_fraction_static is not None:
        engine_args["mem_fraction_static"] = args.mem_fraction_static
    if args.attention_backend != "auto":
        engine_args["attention_backend"] = args.attention_backend
    prompts = [
        [((request * 131 + position) % 1000) + 3 for position in range(args.prompt_tokens)]
        for request in range(args.requests)
    ]
    sampling = {"temperature": 0, "max_new_tokens": args.decode_tokens}
    physical_plan = None
    if args.physical_plan:
        physical_plan = json.loads(
            Path(args.physical_plan).read_text(encoding="utf-8")
        )["physical_plan"]

    before = gpu_snapshot()
    started = time.perf_counter()
    with sgl.Engine(**engine_args) as engine:
        loaded = time.perf_counter()
        info = engine.get_server_info()
        after_load = gpu_snapshot()
        iteration_seconds = []
        outputs = []
        for _ in range(args.iterations):
            iteration_started = time.perf_counter()
            outputs = engine.generate(input_ids=prompts, sampling_params=sampling)
            iteration_seconds.append(time.perf_counter() - iteration_started)
        after_workload = gpu_snapshot()
        result = {
            "schema": "orbitkv.sglang-ab.v1",
            "mode": args.mode,
            "sglang_revision": "095ec6c997bfdd25d3864cb0ce77a6562a934b96",
            "model": str(model_path),
            "checkpoint": checkpoint_identity(model_path, args.load_format),
            "plan": str(plan_path),
            "physical_plan": physical_plan,
            "engine_args": engine_args,
            "requests": args.requests,
            "max_running_requests": max_running_requests,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "iterations": args.iterations,
            "eviction_interval": (
                args.eviction_interval
                if args.mode in ("native_policy", "policy", "owner")
                else 128
            ),
            "owner_transport": (
                args.owner_transport if args.mode == "owner" else None
            ),
            "load_seconds": loaded - started,
            "iteration_seconds": iteration_seconds,
            "output_digest": token_digest(outputs),
            "completion_tokens": sum(
                output["meta_info"]["completion_tokens"] for output in outputs
            ),
            "num_retractions": [
                output["meta_info"].get("num_retractions", 0) for output in outputs
            ],
            "request_e2e_seconds": [
                output["meta_info"].get("e2e_latency") for output in outputs
            ],
            "resolved_runtime": extract_runtime(info),
            "server_memory": extract_memory(info),
            "gpu_before": before,
            "gpu_after_load": after_load,
            "gpu_after_workload": after_workload,
        }

    if args.mode in ("shadow", "policy", "owner") and not args.no_trace_allocations:
        summary = subprocess.check_output(
            [
                args.orbitkv_bin,
                "analyze-sglang",
                str(plan_path),
                args.trace,
                "--max-active-requests",
                str(args.requests),
            ],
            text=True,
        )
        result["orbitkv_trace"] = json.loads(summary)
        result["trace_path"] = args.trace

    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
