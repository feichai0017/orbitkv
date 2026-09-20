"""Measure cold prefill, HBM hits, and reuse after HBM pressure on one GPU.

Run with the selected engine's Python environment. Each run owns its engine
and, for OrbitKV, a fresh Cache Manager. Results include raw streaming timings,
cache-source counters, exact launch commands, and package/hardware versions.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def server(command: list[str], env: dict[str, str], url: str, log: Path):
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {log}\n{log.read_text()[-6000:]}")
            try:
                if requests.get(f"{url}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError(f"Server startup timed out: {log}")
        yield
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def metrics(url: str | None) -> dict[str, float]:
    if url is None:
        return {}
    response = requests.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value, *_ = line.rsplit(" ", 1)
        name = name.split("{", 1)[0]
        if name.endswith("_created") or "bucket" in name:
            continue
        values[name] = values.get(name, 0) + float(value)
    return values


def generate(
    url: str, engine: str, model: str, tokens: list[int], output_len: int
) -> dict:
    if engine == "vllm":
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        endpoint = "/generate"
        payload = {
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
        }
    started = time.perf_counter()
    first = None
    text = ""
    usage = {}
    with requests.post(
        url + endpoint, json=payload, stream=True, timeout=(10, 300)
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if engine == "vllm":
                chunk = "".join(
                    choice.get("text", "") for choice in event.get("choices", [])
                )
                text += chunk
                usage = event.get("usage") or usage
            else:
                chunk = event.get("text", "")
                text = chunk
                usage = event.get("meta_info", usage)
            if first is None and chunk:
                first = time.perf_counter()
    ended = time.perf_counter()
    if first is None:
        raise RuntimeError("No generated text in streaming response")
    if usage.get("prompt_tokens") != len(tokens):
        raise RuntimeError(f"Input length changed: expected {len(tokens)}, got {usage}")
    if usage.get("completion_tokens") != output_len:
        raise RuntimeError(f"Output length changed: expected {output_len}, got {usage}")
    return {
        "ttft_ms": (first - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "text": text,
        "usage": usage,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cache_source(engine: str, result: dict) -> str:
    if engine == "sglang":
        details = result["usage"].get("cached_tokens_details") or {}
        device = details.get("device", 0)
        external = details.get("host", 0) + details.get("storage", 0)
    else:
        delta = result["metrics_delta"]
        device = delta.get("vllm:prefix_cache_hits_total", 0)
        external = delta.get("vllm:external_prefix_cache_hits_total", 0)
    if device > 0:
        return "mixed" if external > 0 else "hbm"
    if external > 0 or result["manager_delta"].get("orbitkv_load_bytes_total", 0) > 0:
        return "external"
    cached = (
        result["usage"].get("cached_tokens", 0)
        if engine == "sglang"
        else (result["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
    )
    return "unverified" if cached else "miss"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument(
        "--backend", choices=["native", "cpu", "orbitkv"], required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--gpu-tokens", type=int, default=16384)
    parser.add_argument("--host-gib", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--settle-seconds", type=float, default=1.2)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty so measurements cannot mix across runs")
    if (
        args.repeats < 1
        or min(args.lengths) < 64
        or max(args.lengths) >= args.gpu_tokens * 3 // 4
    ):
        parser.error(
            "use positive repeats and lengths between 64 and 3/4 of GPU token capacity"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        parser.error("this fixed-capacity benchmark currently targets dense Qwen3")
    bytes_per_token = (
        2
        * config["num_hidden_layers"]
        * config["num_key_value_heads"]
        * config["head_dim"]
        * 2
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocabulary = tokenizer.encode(
        "The river flows past green trees and quiet houses. A researcher measures memory "
        "transfer latency and checks that repeated requests produce accurate results. ",
        add_special_tokens=False,
    )
    rng = random.Random(args.seed)

    def prompt(length: int) -> list[int]:
        # Distinct first blocks prevent accidental reuse across samples, while
        # repeated calls below deliberately reuse the exact same token IDs.
        return [rng.choice(vocabulary) for _ in range(length)]

    env = dict(os.environ)
    env.update(PYTHONHASHSEED="0", VLLM_LOG_STATS_INTERVAL="1")
    env.pop("VLLM_BATCH_INVARIANT", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), str(args.output), sysconfig.get_path("purelib")]
    )
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    pressure_tokens = args.gpu_tokens * 3 // 4
    manager_url = None
    manager_command = None
    if args.backend == "orbitkv":
        manager_port = free_port()
        manager_http = free_port()
        manager_url = f"http://127.0.0.1:{manager_http}"
        env.update(
            ORBITKV_PORT=str(manager_port),
            ORBITKV_SGLANG_ENDPOINT=f"unix:///tmp/orbitkv-{manager_port}.sock",
            PYO3_PYTHON=sys.executable,
            PYTHONHOME=sys.base_prefix,
        )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [sysconfig.get_config_var("LIBDIR"), env.get("LD_LIBRARY_PATH", "")]
        )
        manager_command = [
            str(ROOT / "python/orbitkv/orbitkv-cache-manager-py"),
            "--addr",
            f"127.0.0.1:{manager_port}",
            "--http-addr",
            f"127.0.0.1:{manager_http}",
            "--pool-size",
            f"{args.host_gib}gb",
            "--enable-prometheus",
        ]
        plugin = args.output / "orbitkv_benchmark-0.0.dist-info"
        plugin.mkdir()
        (plugin / "METADATA").write_text("Name: orbitkv-benchmark\nVersion: 0.0\n")
        (plugin / "entry_points.txt").write_text(
            "[sglang.srt.plugins]\norbitkv = orbitkv.sglang.plugin:register\n"
        )
    if args.engine == "vllm":
        command = [
            str(Path(sys.executable).parent / "vllm"),
            "serve",
            str(args.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "auto",
            "--block-size",
            "64",
            "--enable-prefix-caching",
            "--kv-cache-memory-bytes",
            str(bytes_per_token * args.gpu_tokens),
            "--max-model-len",
            str(pressure_tokens + 64),
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "8192",
            "--generation-config",
            "vllm",
            "--seed",
            "42",
            "--enable-prompt-tokens-details",
        ]
        if args.backend == "cpu":
            connector = {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": args.host_gib * 1024**3,
                    "block_size": 64,
                },
            }
        elif args.backend == "orbitkv":
            connector = {
                "kv_connector": "OrbitKVConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "orbitkv.vllm",
            }
        if args.backend != "native":
            command += ["--kv-transfer-config", json.dumps(connector)]
    else:
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(args.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--page-size",
            "64",
            "--max-total-tokens",
            str(args.gpu_tokens),
            "--context-length",
            str(pressure_tokens + 64),
            "--max-running-requests",
            "8",
            "--cuda-graph-max-bs-decode",
            "8",
            "--cuda-graph-max-bs-prefill",
            "8",
            "--chunked-prefill-size",
            "8192",
            "--random-seed",
            "42",
            "--enable-cache-report",
            "--enable-metrics",
        ]
        if args.backend == "cpu":
            command += [
                "--enable-hierarchical-cache",
                "--hicache-size",
                str(args.host_gib),
                "--hicache-write-policy",
                "write_through",
            ]
        elif args.backend == "orbitkv":
            command += [
                "--radix-cache-backend",
                "orbitkv",
                "--enable-unified-cache-external-linker",
            ]

    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "engine_command": command,
        "manager_command": manager_command,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff"], cwd=ROOT, text=True),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (args.engine, "torch", "transformers")
        },
        "python": sys.version,
        "kv_bytes_per_token": bytes_per_token,
        "model_revision": (args.model / ".revision").read_text().strip()
        if (args.model / ".revision").exists()
        else None,
        "concurrency": 1,
        "notes": "Serial latency experiment; pressure traffic and startup are excluded from request timings.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    samples = []
    with contextlib.ExitStack() as stack:
        if manager_command:
            stack.enter_context(
                server(manager_command, env, manager_url, args.output / "manager.log")
            )
        stack.enter_context(server(command, env, base_url, args.output / "engine.log"))
        for _ in range(3):
            generate(
                base_url, args.engine, str(args.model), prompt(256), args.output_tokens
            )
        with (args.output / "samples.jsonl").open("w") as output:
            for length in args.lengths:
                generate(
                    base_url,
                    args.engine,
                    str(args.model),
                    prompt(length),
                    args.output_tokens,
                )
                for repeat in range(args.repeats):
                    tokens = prompt(length)
                    texts = []
                    for phase in ("cold", "hbm_hit", "after_pressure"):
                        if phase == "after_pressure":
                            for _ in range(2):
                                generate(
                                    base_url,
                                    args.engine,
                                    str(args.model),
                                    prompt(pressure_tokens),
                                    1,
                                )
                        time.sleep(args.settle_seconds)
                        before = metrics(base_url)
                        manager_before = metrics(manager_url)
                        result = generate(
                            base_url,
                            args.engine,
                            str(args.model),
                            tokens,
                            args.output_tokens,
                        )
                        time.sleep(args.settle_seconds)
                        after = metrics(base_url)
                        manager_after = metrics(manager_url)
                        result.update(
                            length=length,
                            repeat=repeat,
                            phase=phase,
                            metrics_delta={
                                key: value - before.get(key, 0)
                                for key, value in after.items()
                                if value != before.get(key, 0)
                            },
                            manager_delta={
                                key: value - manager_before.get(key, 0)
                                for key, value in manager_after.items()
                                if value != manager_before.get(key, 0)
                            },
                        )
                        texts.append(result["text"])
                        result["cache_source"] = cache_source(args.engine, result)
                        result["matches_cold_output"] = result["text"] == texts[0]
                        samples.append(result)
                        output.write(json.dumps(result) + "\n")
                        output.flush()
                        print(
                            f"{args.engine}/{args.backend} len={length} repeat={repeat} {phase}: TTFT={result['ttft_ms']:.2f}ms usage={result['usage']}",
                            flush=True,
                        )
    summary = []
    for length in args.lengths:
        for phase in ("cold", "hbm_hit", "after_pressure"):
            group = [
                sample
                for sample in samples
                if sample["length"] == length and sample["phase"] == phase
            ]
            times = [sample["ttft_ms"] for sample in group]
            summary.append(
                {
                    "length": length,
                    "phase": phase,
                    "n": len(group),
                    "ttft_p50_ms": statistics.median(times),
                    "ttft_p95_ms": percentile(times, 0.95),
                    "e2e_p50_ms": statistics.median(
                        sample["e2e_ms"] for sample in group
                    ),
                    "output_mismatches": sum(
                        not sample["matches_cold_output"] for sample in group
                    ),
                    "cache_sources": {
                        source: sum(
                            sample["cache_source"] == source for sample in group
                        )
                        for source in sorted(
                            {sample["cache_source"] for sample in group}
                        )
                    },
                    "orbitkv_load_bytes": sum(
                        sample["manager_delta"].get("orbitkv_load_bytes_total", 0)
                        for sample in group
                    ),
                }
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
