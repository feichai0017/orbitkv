"""Qualify native GPU storage on a bare-metal NVMe host; never accepts CPU fallback.

Uses prebuilt artifacts so Cargo cannot restage libraries under running services.
Run from the repository root: python -m benches.gds --help.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def native_io_stats(output: str) -> dict:
    """Require explicit per-GPU read/write evidence; unknown formats fail closed."""
    operations = {"read": 0, "write": 0}
    rows = re.findall(r"GPU\s+\d+\s+Read:\s*(.*?)\s+Write:\s*(.*?)(?:BufRegister:|$)", output, re.M)
    if not rows:
        raise ValueError("gds_stats did not report per-GPU counters; enable profile.cufile_stats=3")
    for row in rows:
        for direction, section in zip(operations, row, strict=True):
            counters = dict(re.findall(r"(\w+)=([\d.]+)", section))
            if not {"n", "posix", "err", "unalign"} <= counters.keys():
                raise ValueError(f"Incomplete {direction} counters: {section}")
            if any(
                float(counters.get(name, 0)) != 0
                for name in ("posix", "err", "unalign", "r_sparse", "r_inline")
            ):
                raise ValueError(
                    f"Native qualification rejected fallback/unaligned/failed I/O: {section}"
                )
            operations[direction] += int(counters["n"])
    if not all(operations.values()):
        raise ValueError(f"Both native reads and writes must execute: {operations}")
    return {"operations": operations, "posix_operations": 0, "errors": 0}


def require_bare_metal(ssd_dir: Path) -> dict:
    cgroup = Path("/proc/1/cgroup").read_text()
    root_fs = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "/"], text=True).strip()
    if (
        root_fs == "overlay"
        or any(Path(p).exists() for p in ("/.dockerenv", "/run/.containerenv"))
        or re.search(r"kubepods|docker|containerd|lxc", cgroup)
    ):
        raise RuntimeError(
            "This gate requires a bare-metal host; container correctness uses docs/gds.md"
        )
    detector = shutil.which("systemd-detect-virt")
    if (
        detector
        and subprocess.run([detector, "--container", "--quiet"], check=False).returncode == 0
    ):
        raise RuntimeError("Container detected; native host qualification was not run")
    mount = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(ssd_dir), "--output", "SOURCE,FSTYPE,TARGET"],
            text=True,
        )
    )["filesystems"][0]
    if mount["fstype"] not in ("ext4", "xfs"):
        raise RuntimeError(f"This local-NVMe gate requires ext4 or xfs: {mount}")
    source = mount["source"].split("[", 1)[0]
    devices = json.loads(
        subprocess.check_output(
            ["lsblk", "--inverse", "--json", "--output", "NAME,TYPE,TRAN", source], text=True
        )
    )
    pending = list(devices["blockdevices"])
    disks = []
    while pending:
        device = pending.pop()
        pending.extend(device.get("children", []))
        if device["type"] == "disk":
            disks.append(device)
    if not disks or any(disk.get("tran") != "nvme" for disk in disks):
        raise RuntimeError(f"Could not establish NVMe backing: {devices}")
    return {"mount": mount, "devices": devices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssd-dir",
        type=Path,
        required=True,
        help="Writable NVMe directory; only a new private subdirectory is used",
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Local dense Qwen3 model, e.g. Qwen3-8B"
    )
    parser.add_argument(
        "--manager", type=Path, required=True, help="Prebuilt release orbitkv-cache-manager-py"
    )
    parser.add_argument(
        "--fault-manager",
        type=Path,
        required=True,
        help="Separate prebuilt Manager with test-hooks",
    )
    parser.add_argument(
        "--core-test",
        type=Path,
        required=True,
        help="Prebuilt Rust cufile integration-test executable",
    )
    parser.add_argument("--vllm-python", type=Path, default=ROOT / ".venv/vllm-release/bin/python")
    parser.add_argument(
        "--sglang-python", type=Path, default=ROOT / ".venv/sglang-release/bin/python"
    )
    parser.add_argument(
        "--gds-tools",
        type=Path,
        required=True,
        help="NVIDIA directory containing gdscheck.py, gdsio and gds_stats",
    )
    parser.add_argument(
        "--cufile-library", type=Path, default=Path("/usr/local/cuda/lib64/libcufile.so.0")
    )
    parser.add_argument("--host-gib", type=int, default=1)
    parser.add_argument("--ssd-gib", type=int, default=16)
    parser.add_argument("--working-set", type=int, default=16)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--duration-seconds", type=int, default=60)
    args = parser.parse_args()
    args.ssd_dir = args.ssd_dir.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    if (
        min(args.host_gib, args.duration_seconds, args.working_set) < 1
        or args.ssd_gib <= args.host_gib
    ):
        parser.error("require positive host/time/working-set and SSD capacity greater than DRAM")
    if not 64 <= args.length < 12288:
        parser.error("--length must be between 64 and 12287 tokens")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        parser.error("the matched-capacity benchmark requires dense Qwen3")
    bytes_per_token = (
        4 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    )
    working_bytes = args.working_set * args.length * bytes_per_token
    if working_bytes <= max(args.host_gib * 1024**3, 16384 * bytes_per_token):
        parser.error("working set must exceed both Manager DRAM and engine HBM KV capacities")
    for name in ("manager", "fault_manager", "core_test", "vllm_python", "sglang_python"):
        # Keep venv interpreter symlinks intact so Python finds its pyvenv.cfg.
        value = Path(os.path.abspath(getattr(args, name)))
        if not value.is_file() or not os.access(value, os.X_OK):
            parser.error(f"{name} is not an executable file: {value}")
        setattr(args, name, value)
    args.gds_tools = args.gds_tools.resolve(strict=True)
    for tool in ("gdscheck.py", "gdsio", "gds_stats"):
        if not os.access(args.gds_tools / tool, os.X_OK):
            parser.error(f"missing executable GDS tool: {args.gds_tools / tool}")
    args.cufile_library = args.cufile_library.resolve(strict=True)
    host = require_bare_metal(args.ssd_dir)
    if shutil.disk_usage(args.ssd_dir).free < (args.ssd_gib + 4) * 1024**3:
        parser.error("NVMe directory needs SSD capacity plus 4 GiB of free space")
    output = Path(tempfile.mkdtemp(prefix="orbitkv-gds-", dir=args.ssd_dir))
    print(f"Qualification artifacts: {output}", flush=True)
    temporary = output / "tmp"
    temporary.mkdir()
    configuration = output / "cufile.json"
    configuration.write_text(
        json.dumps(
            {
                "profile": {"cufile_stats": 3},
                "properties": {"allow_compat_mode": False},
                "fs": {"generic": {"posix_unaligned_writes": False}},
                "logging": {"dir": str(output), "level": "ERROR"},
            },
            indent=2,
        )
        + "\n"
    )
    env = {
        **os.environ,
        "CUFILE_ALLOW_COMPAT_MODE": "false",
        "CUFILE_FORCE_COMPAT_MODE": "false",
        "CUFILE_ENV_PATH_JSON": str(configuration),
        "TMPDIR": str(temporary),
        "LD_PRELOAD": str(args.cufile_library),
        "ORBITKV_CACHE_MANAGER_BINARY": str(args.manager),
        "TRITON_CACHE_DIR": str(output / "kernels/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(output / "kernels/inductor"),
        "VLLM_CACHE_ROOT": str(output / "kernels/vllm"),
    }
    report = {"status": "running", "host": host, "working_set_bytes": working_bytes, "stages": []}

    def run(name, command, *, cwd=ROOT, overrides=None, timeout=3600):
        print(f"Running {name}", flush=True)
        started = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                [str(v) for v in command],
                cwd=cwd,
                env=env | (overrides or {}),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            finally:
                if process.poll() is None:
                    # Let pytest fixtures / benchmark ExitStack stop their services.
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                report["stages"].append(
                    {
                        "name": name,
                        "exit_code": process.returncode,
                        "seconds": round(time.monotonic() - started, 3),
                    }
                )
        if returncode:
            raise RuntimeError(f"{name} failed; see {output / f'{name}.log'}")

    try:
        run("topology", ["nvidia-smi", "topo", "-m"], timeout=30)
        run("gdscheck", [args.gds_tools / "gdscheck.py", "-p"], timeout=120)
        probe = output / "probe.bin"
        try:
            for direction, operation in (("write", 1), ("read", 0)):
                run(
                    f"gdsio-{direction}",
                    [
                        args.gds_tools / "gdsio",
                        "-f",
                        probe,
                        "-d",
                        "0",
                        "-w",
                        "1",
                        "-s",
                        "64M",
                        "-i",
                        "1M",
                        "-x",
                        "0",
                        "-I",
                        str(operation),
                    ],
                    timeout=120,
                )
        finally:
            probe.unlink(missing_ok=True)
        run("core", [args.core_test, "--ignored", "--test-threads=1"], timeout=300)
        run(
            "faults",
            [
                args.vllm_python,
                "-m",
                "pytest",
                "-m",
                "integration",
                "tests/integration/test_cache_faults.py",
                "--ssd-backend",
                "cufile",
                f"--basetemp={temporary / 'faults'}",
            ],
            cwd=ROOT / "python",
            overrides={
                "ORBITKV_CACHE_MANAGER_BINARY": str(args.fault_manager),
                "ORBITKV_FAULT_TESTS": "1",
            },
            timeout=900,
        )
        for engine in ("vllm", "sglang"):
            interpreter = getattr(args, f"{engine}_python")
            e2e = [
                interpreter,
                "-m",
                "pytest",
                "-m",
                "e2e",
                "--model",
                args.model,
                "--ssd-backend",
                "cufile",
                f"--basetemp={temporary / engine}",
            ]
            if engine == "vllm":
                e2e += [
                    "tests/e2e/test_vllm_e2e_correctness.py",
                    "--vllm-cache-tier",
                    "ssd",
                    "--max-model-len",
                    "4096",
                    "--orbitkv-pool-size",
                    f"{args.host_gib}gb",
                ]
            else:
                e2e += ["tests/e2e/test_sglang_direct_e2e.py", "-k", "ssd"]
            run(f"{engine}-correctness", e2e, cwd=ROOT / "python")
            for workload in ("serial", "sustained"):
                for backend in ("uring", "cufile"):
                    name = f"{engine}-{workload}-{backend}"
                    command = [
                        interpreter,
                        "-m",
                        "benches.single_node",
                        "--engine",
                        engine,
                        "--backend",
                        "orbitkv",
                        "--model",
                        args.model,
                        "--output",
                        output / name,
                        "--host-gib",
                        str(args.host_gib),
                        "--ssd-gib",
                        str(args.ssd_gib),
                        "--ssd-backend",
                        backend,
                        "--workload",
                        workload,
                        "--working-set",
                        str(args.working_set),
                        "--lengths",
                        str(args.length),
                        "--duration-seconds",
                        str(args.duration_seconds),
                        "--repeats",
                        "3",
                        "--concurrencies",
                        "1",
                        "4",
                    ]
                    if backend == "cufile":
                        command += ["--gds-stats", args.gds_tools / "gds_stats"]
                    run(name, command)
                    summary = json.loads((output / name / "summary.json").read_text())
                    read_key = (
                        "orbitkv_ssd_read_bytes"
                        if workload == "serial"
                        else (
                            "orbitkv_ssd_cufile_read_bytes_total"
                            if backend == "cufile"
                            else "orbitkv_ssd_prefetch_bytes_total"
                        )
                    )
                    load_key = (
                        "orbitkv_load_bytes" if workload == "serial" else "orbitkv_load_bytes_total"
                    )
                    if not all(
                        sum(row.get(key, 0) for row in summary) > 0 for key in (read_key, load_key)
                    ):
                        raise RuntimeError(
                            f"{name}: measured workload lacks SSD read/GPU restore evidence"
                        )
                    report.setdefault("benchmarks", {})[name] = {
                        "summary": summary,
                        "manager_usage": json.loads(
                            (output / name / "manager-usage.json").read_text()
                        ),
                    }
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status']}: {output / 'qualification.json'}", flush=True)


if __name__ == "__main__":
    main()
