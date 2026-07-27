#!/usr/bin/env python3
"""Run a real-vLLM seeded categorical-sampling A/B."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


PROVIDERS = ("vllm", "loom")


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    input_len: int
    output_len: int

    @property
    def label(self) -> str:
        return f"b{self.batch_size}-in{self.input_len}-out{self.output_len}"

    @property
    def argument(self) -> str:
        return f"{self.batch_size}x{self.input_len}x{self.output_len}"


def parse_case(value: str) -> BenchmarkCase:
    try:
        dimensions = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be BATCHxINPUTxOUTPUT"
        ) from error
    if len(dimensions) != 3 or min(dimensions) <= 0:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT")
    return BenchmarkCase(*dimensions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=461)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--provider-order",
        choices=("baseline-first", "loom-first"),
        default="baseline-first",
    )
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-provider",
        choices=PROVIDERS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-cache-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 32, 32),
            BenchmarkCase(2, 32, 32),
            BenchmarkCase(4, 32, 32),
            BenchmarkCase(8, 32, 32),
            BenchmarkCase(32, 32, 32),
        ]
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("top-p must be in (0, 1]")
    if len(args.repository_revision) != 40:
        parser.error("repository-revision must be a full Git SHA")
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error("internal runs require result and cache paths")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def make_prompts(case: BenchmarkCase) -> list[dict[str, list[int]]]:
    return [
        {
            "prompt_token_ids": [
                3 + ((row * 17 + position * 13) % 1000)
                for position in range(case.input_len)
            ]
        }
        for row in range(case.batch_size)
    ]


def request_metrics(outputs: list[Any]) -> tuple[list[float], list[float]]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            continue
        if metrics.first_token_latency > 0.0:
            ttft_ms.append(metrics.first_token_latency * 1000.0)
        generated = metrics.num_generation_tokens
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        if generated > 1 and decode_seconds >= 0.0:
            tpot_ms.append(decode_seconds * 1000.0 / (generated - 1))
    return ttft_ms, tpot_ms


def run_case(
    engine: Any,
    sampling_type: Any,
    operator: Any,
    launch_count: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    prompts = make_prompts(case)
    sampling = [
        sampling_type(
            temperature=1.0,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=case.output_len,
            ignore_eos=True,
            seed=args.seed + row,
        )
        for row in range(case.batch_size)
    ]
    for _ in range(args.warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    latencies_ms: list[float] = []
    throughputs: list[float] = []
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    launch_deltas: list[int] = []
    replay_tokens: list[list[int]] | None = None
    for _ in range(args.repeats):
        before_launches = launch_count(operator)
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        launch_deltas.append(launch_count(operator) - before_launches)
        latencies_ms.append(elapsed_ms)
        throughputs.append(
            case.batch_size * case.output_len / (elapsed_ms / 1000.0)
        )
        request_ttft_ms, request_tpot_ms = request_metrics(outputs)
        ttft_ms.extend(request_ttft_ms)
        tpot_ms.extend(request_tpot_ms)
        token_ids = [
            list(request.outputs[0].token_ids) for request in outputs
        ]
        if any(len(tokens) != case.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")
        if replay_tokens is None:
            replay_tokens = token_ids
        elif replay_tokens != token_ids:
            raise RuntimeError(
                f"{args.internal_provider} failed exact seeded replay for "
                f"{case.label}"
            )

    expected_launches = (
        case.output_len if args.internal_provider == "loom" else 0
    )
    if any(delta != expected_launches for delta in launch_deltas):
        raise RuntimeError(
            f"unexpected categorical launch deltas for {case.label}: "
            f"{launch_deltas}, expected {expected_launches}"
        )
    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "batch_latency_ms": summary(latencies_ms),
        "request_ttft_ms": summary(ttft_ms),
        "request_tpot_ms": summary(tpot_ms),
        "output_tokens_per_second": summary(throughputs),
        "categorical_launches_per_generation": launch_deltas,
        "exact_seeded_replay": True,
        "token_ids": replay_tokens,
    }


def prepare_environment(cache_root: Path) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    required = [str(Path(sys.executable).absolute().parent), str(cuda_home / "bin")]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required if entry not in current_entries]
        + current_entries
    )


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None and args.internal_cache_root is not None
    prepare_environment(args.internal_cache_root.resolve())

    import torch

    from loom_kernels.torch_ops import (
        Operator,
        bridge_abi_version,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import (
        provider_metadata,
        register_vllm_categorical_sample,
    )

    explicit_registration = None
    if provider == "loom":
        explicit_registration = register_vllm_categorical_sample()
        if explicit_registration is None:
            raise RuntimeError("Loom categorical registration failed")

    import vllm
    from vllm import LLM, SamplingParams

    operator = Operator.CATEGORICAL_SAMPLE
    reset_launch_count(operator)
    model_path = Path(args.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else args.model
    max_model_len = max(
        case.input_len + case.output_len for case in args.cases
    )
    engine = LLM(
        model=model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=max(case.batch_size for case in args.cases),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        disable_log_stats=False,
    )
    launches_after_engine_init = launch_count(operator)
    cases = [
        run_case(
            engine,
            SamplingParams,
            operator,
            launch_count,
            case,
            args,
        )
        for case in args.cases
    ]
    report = {
        "provider": provider,
        "model": model,
        "dtype": "bfloat16 model; float32 sampling probabilities",
        "sampling": {
            "temperature": 1.0,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed_policy": "explicit seed + row index",
            "ignore_eos": True,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
        "loom_path": {
            "explicit_registration": explicit_registration,
            "launches_after_engine_init": launches_after_engine_init,
            "host_launch_count": launch_count(operator),
            "provider_metadata": provider_metadata(),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi_version": bridge_abi_version(),
        },
    }
    assert args.internal_result is not None
    args.internal_result.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def comparison(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_cases = {
        case["case"]: case for case in reports["vllm"]["cases"]
    }
    loom_cases = {
        case["case"]: case for case in reports["loom"]["cases"]
    }
    compared: list[dict[str, Any]] = []
    for label, baseline in baseline_cases.items():
        loom = loom_cases[label]
        baseline_latency = baseline["batch_latency_ms"]["median"]
        loom_latency = loom["batch_latency_ms"]["median"]
        baseline_throughput = baseline["output_tokens_per_second"]["median"]
        loom_throughput = loom["output_tokens_per_second"]["median"]
        compared.append(
            {
                "case": label,
                "batch_latency_speedup": baseline_latency / loom_latency,
                "output_throughput_speedup": (
                    loom_throughput / baseline_throughput
                ),
                "baseline_exact_seeded_replay": baseline[
                    "exact_seeded_replay"
                ],
                "loom_exact_seeded_replay": loom["exact_seeded_replay"],
                "loom_launches_per_decode_step": 1,
            }
        )
    return compared


def child_command(
    args: argparse.Namespace,
    provider: str,
    result_path: Path,
    cache_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--repository-revision",
        args.repository_revision,
        "--internal-provider",
        provider,
        "--internal-result",
        str(result_path),
        "--internal-cache-root",
        str(cache_root),
    ]
    for case in args.cases:
        command.extend(("--case", case.argument))
    return command


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    order = (
        ("vllm", "loom")
        if args.provider_order == "baseline-first"
        else ("loom", "vllm")
    )
    repository = Path(__file__).resolve().parents[1]
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix="loom-vllm-categorical-"
    ) as temporary:
        temporary_root = Path(temporary)
        for provider in order:
            result_path = temporary_root / f"{provider}.json"
            subprocess.run(
                child_command(
                    args,
                    provider,
                    result_path,
                    temporary_root / f"{provider}-cache",
                ),
                check=True,
                cwd=repository,
            )
            reports[provider] = json.loads(
                result_path.read_text(encoding="utf-8")
            )

    sources = (
        "benchmarks/vllm_engine_categorical_sample.py",
        "python/src/loom_kernels/vllm/categorical.py",
        "crates/loom-cuda-sys/cuda/src/categorical_sample.cu",
    )
    report = {
        "schema_version": 1,
        "benchmark": "vllm_engine_categorical_sample",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_base_revision": args.repository_revision,
        "source_sha256": {
            source: sha256_file(repository / source) for source in sources
        },
        "provider_order": list(order),
        "cases": [case.argument for case in args.cases],
        "providers": reports,
        "comparison": comparison(reports),
        "scope": (
            "real vLLM engine A/B for explicitly seeded, non-speculative "
            "sampling; provider token streams intentionally differ"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is None:
        run_parent(args)
    else:
        run_provider(args)


if __name__ == "__main__":
    main()
