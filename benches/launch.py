"""Explicit engine/backend launch configurations for matched cache budgets."""

from __future__ import annotations

import json
import os
import sys
import sysconfig
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from .runtime import ROOT, free_port


@dataclass(frozen=True)
class Launch:
    command: list[str]
    env: dict[str, str]
    base_url: str
    manager_command: list[str] | None
    manager_url: str | None
    manager_health_path: str
    backend_configuration: dict


def configure(args: Namespace, bytes_per_token: int) -> Launch:
    env = dict(os.environ)
    env.update(PYTHONHASHSEED="0", VLLM_LOG_STATS_INTERVAL="1")
    env.pop("VLLM_BATCH_INVARIANT", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), str(args.output)]
        + [path for path in sys.path if Path(path).name in {"site-packages", "dist-packages"}]
    )
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    manager_url = None
    manager_health_path = "/health"
    manager_command = None
    backend_configuration = {}
    cache_port = None
    cache_config = None
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
    elif args.backend == "lmcache":
        env["LMCACHE_TRACK_USAGE"] = "false"
        cache_port, cache_http = free_port(), free_port()
        manager_url = f"http://127.0.0.1:{cache_http}"
        manager_health_path = "/healthcheck"
        manager_command = [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(cache_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(cache_http),
            "--l1-size-gb",
            str(args.host_gib),
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            "64",
        ]
        if args.engine == "sglang":
            backend_configuration = {
                "chunk_size": 64,
                "mp_host": "127.0.0.1",
                "mp_port": cache_port,
            }
            cache_config = args.output / "lmcache.json"
            cache_config.write_text(json.dumps(backend_configuration, indent=2))
    elif args.backend == "flexkv":
        backend_configuration = {
            "FLEXKV_CPU_CACHE_GB": str(args.host_gib),
            "FLEXKV_SSD_CACHE_GB": "0",
            "FLEXKV_ENABLE_GDS": "0",
            "FLEXKV_ENABLE_MPS": "0",
            "FLEXKV_ENABLE_METRICS": "0",
            "FLEXKV_SERVER_RECV_PORT": f"ipc://{args.output}/flexkv.sock",
        }
        env.pop("FLEXKV_CONFIG_PATH", None)
        env.update(backend_configuration)
    command = (
        vllm_command(args, port, bytes_per_token, cache_port)
        if args.engine == "vllm"
        else sglang_command(args, port, cache_config)
    )
    return Launch(
        command,
        env,
        base_url,
        manager_command,
        manager_url,
        manager_health_path,
        backend_configuration,
    )


def vllm_command(
    args: Namespace, port: int, bytes_per_token: int, cache_port: int | None
) -> list[str]:
    pressure_tokens = args.gpu_tokens * 3 // 4
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
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
        if args.orbitkv_transfer_backend:
            connector["kv_connector_extra_config"] = {
                "orbitkv.transfer_backend": args.orbitkv_transfer_backend
            }
    elif args.backend == "lmcache":
        connector = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "127.0.0.1",
                "lmcache.mp.port": cache_port,
            },
        }
    elif args.backend == "flexkv":
        connector = {"kv_connector": "FlexKVConnectorV1", "kv_role": "kv_both"}
    if args.backend != "native":
        command += ["--kv-transfer-config", json.dumps(connector)]
    return command


def sglang_command(args: Namespace, port: int, cache_config: Path | None) -> list[str]:
    pressure_tokens = args.gpu_tokens * 3 // 4
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
    elif args.backend == "lmcache":
        command += ["--enable-lmcache", "--lmcache-config-file", str(cache_config)]
    elif args.backend == "flexkv":
        command += ["--enable-flexkv"]
    return command
