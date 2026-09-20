#!/usr/bin/env python3
"""
Start vLLM P/D disaggregation setup with OrbitKV.

Launches:
- Router on port 8000
- P (prefill) nodes starting from port 8100
- D (decode) nodes starting from port 8200

Usage:
    python run_vllm_pd_with_pega.py --model <model_path>
    python run_vllm_pd_with_pega.py --model Qwen/Qwen3-8B --num-p 1 --num-d 1

Note: Start OrbitKV Cache Manager separately before running this script.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def get_gpu_count() -> int:
    """Detect number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start vLLM P/D disaggregation with OrbitKV"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--num-p",
        type=int,
        default=1,
        help="Number of prefill nodes (default: 1)",
    )
    parser.add_argument(
        "--num-d",
        type=int,
        default=1,
        help="Number of decode nodes (default: 1)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size per node (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization per node (default: 0.9)",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=8000,
        help="Router port (default: 8000)",
    )
    parser.add_argument(
        "--p-base-port",
        type=int,
        default=8100,
        help="Base port for prefill nodes (default: 8100)",
    )
    parser.add_argument(
        "--d-base-port",
        type=int,
        default=8200,
        help="Base port for decode nodes (default: 8200)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="examples/pd_logs",
        help="Directory to save logs (default: examples/pd_logs)",
    )
    parser.add_argument(
        "--no-log-files",
        action="store_true",
        help="Disable log files and output to terminal instead",
    )
    return parser.parse_args()


def build_vllm_cmd(
    model: str,
    port: int,
    tp_size: int,
    gpu_util: float,
    is_prefill: bool,
    router_endpoint: str,
) -> list[str]:
    """Build vllm serve command."""
    kv_transfer_config = {
        "kv_connector": "OrbitKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "orbitkv.vllm",
    }

    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]

    return cmd


def main():
    args = parse_args()

    # Detect GPUs and validate configuration
    gpu_count = get_gpu_count()
    tp_size = args.tensor_parallel_size
    gpus_per_node = tp_size
    total_nodes = args.num_p + args.num_d
    required_gpus = total_nodes * gpus_per_node

    print(f"Detected {gpu_count} GPUs")

    if gpu_count == 0:
        print("Error: No GPUs detected", file=sys.stderr)
        sys.exit(1)

    if required_gpus > gpu_count:
        print(
            f"Error: Need {required_gpus} GPUs ({args.num_p}P + {args.num_d}D, TP={tp_size}), but only {gpu_count} available",
            file=sys.stderr,
        )
        sys.exit(1)

    # Allocate GPUs: P nodes first, then D nodes
    # Each node gets `tp_size` consecutive GPUs
    p_gpu_assignments: list[str] = []
    d_gpu_assignments: list[str] = []
    gpu_idx = 0

    for i in range(args.num_p):
        gpus = ",".join(str(gpu_idx + j) for j in range(tp_size))
        p_gpu_assignments.append(gpus)
        gpu_idx += tp_size

    for i in range(args.num_d):
        gpus = ",".join(str(gpu_idx + j) for j in range(tp_size))
        d_gpu_assignments.append(gpus)
        gpu_idx += tp_size

    processes: list[subprocess.Popen] = []
    log_handles: list = []  # Track open log file handles
    run_dir: Optional[Path] = None

    # Create log directory if log files are enabled
    if not args.no_log_files:
        log_base = Path(args.log_dir)
        log_base.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = log_base / f"pd_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(signum=None, frame=None):
        print("\nShutting down all processes...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        # Close all log file handles
        for handle in log_handles:
            handle.close()
        if run_dir:
            print(f"Logs saved to: {run_dir}")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Compute endpoints
    p_endpoints = [
        f"http://localhost:{args.p_base_port + i}" for i in range(args.num_p)
    ]
    d_endpoints = [
        f"http://localhost:{args.d_base_port + i}" for i in range(args.num_d)
    ]
    router_endpoint = f"http://localhost:{args.router_port}"

    print("=" * 60)
    print("OrbitKV P/D Disaggregation Setup")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"GPUs: {gpu_count} available, {required_gpus} required")
    print(f"Router: {router_endpoint}")
    print(f"Prefill nodes: {p_endpoints}")
    for i, gpus in enumerate(p_gpu_assignments):
        print(f"  P{i}: GPU {gpus}")
    print(f"Decode nodes: {d_endpoints}")
    for i, gpus in enumerate(d_gpu_assignments):
        print(f"  D{i}: GPU {gpus}")
    if run_dir:
        print(f"Log directory: {run_dir}")
    else:
        print("Log files: disabled (output to terminal)")
    print("=" * 60)

    # Base environment
    base_env = os.environ.copy()
    base_env["PYTHONHASHSEED"] = "0"

    # Start P nodes
    for i in range(args.num_p):
        port = args.p_base_port + i
        cmd = build_vllm_cmd(
            args.model,
            port,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            is_prefill=True,
            router_endpoint=router_endpoint,
        )

        env = base_env.copy()
        env["ORBITKV_ROUTER_ENDPOINT"] = router_endpoint
        env["ORBITKV_INSTANCE_ID"] = f"p{i}"
        env["CUDA_VISIBLE_DEVICES"] = p_gpu_assignments[i]

        print(f"\n[P{i}] Starting prefill node on port {port}")
        print(f"[P{i}] CUDA_VISIBLE_DEVICES={p_gpu_assignments[i]}")
        print(f"[P{i}] Command: {' '.join(cmd)}")

        if run_dir:
            log_file = run_dir / f"prefill_{i}.log"
            print(f"[P{i}] Log file: {log_file}")
            log_handle = open(log_file, "w")
            log_handles.append(log_handle)
            p = subprocess.Popen(
                cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT
            )
        else:
            p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Start D nodes
    for i in range(args.num_d):
        port = args.d_base_port + i
        cmd = build_vllm_cmd(
            args.model,
            port,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            is_prefill=False,
            router_endpoint=router_endpoint,
        )

        env = base_env.copy()
        env["ORBITKV_INSTANCE_ID"] = f"d{i}"
        env["CUDA_VISIBLE_DEVICES"] = d_gpu_assignments[i]

        print(f"\n[D{i}] Starting decode node on port {port}")
        print(f"[D{i}] CUDA_VISIBLE_DEVICES={d_gpu_assignments[i]}")
        print(f"[D{i}] Command: {' '.join(cmd)}")

        if run_dir:
            log_file = run_dir / f"decode_{i}.log"
            print(f"[D{i}] Log file: {log_file}")
            log_handle = open(log_file, "w")
            log_handles.append(log_handle)
            p = subprocess.Popen(
                cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT
            )
        else:
            p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for vLLM nodes to start
    print("\nWaiting for vLLM nodes to initialize...")
    time.sleep(10)

    # Start Router (Rust version)
    # Get project root directory (assuming this script is in examples/)
    project_root = Path(__file__).parent.parent.resolve()

    router_cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "orbitkv-router",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.router_port),
        "--prefill",
        *p_endpoints,
        "--decode",
        *d_endpoints,
    ]

    print(f"\n[Router] Starting Rust orbitkv-router on port {args.router_port}")
    print(f"[Router] Working directory: {project_root}")
    print(f"[Router] Command: {' '.join(router_cmd)}")

    if run_dir:
        log_file = run_dir / "router.log"
        print(f"[Router] Log file: {log_file}")
        log_handle = open(log_file, "w")
        log_handles.append(log_handle)
        router_p = subprocess.Popen(
            router_cmd,
            cwd=project_root,
            env=base_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    else:
        router_p = subprocess.Popen(router_cmd, cwd=project_root, env=base_env)
    processes.append(router_p)

    print("\n" + "=" * 60)
    print("All processes started. Press Ctrl+C to stop.")
    if run_dir:
        print(f"Logs: {run_dir}")
        print(f"  - tail -f {run_dir}/prefill_*.log")
        print(f"  - tail -f {run_dir}/decode_*.log")
        print(f"  - tail -f {run_dir}/router.log")
    print("=" * 60)

    # Wait for any process to exit
    while True:
        for i, p in enumerate(processes):
            ret = p.poll()
            if ret is not None:
                print(f"\nProcess {i} exited with code {ret}")
                cleanup()
        time.sleep(1)


if __name__ == "__main__":
    main()
