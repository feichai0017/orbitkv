"""Serving result gates and explicitly sampled process memory observations."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any


def engine_report(log: str, marker: str) -> dict[str, Any]:
    prefix = marker + " "
    records = [json.loads(line[len(prefix):]) for line in log.splitlines()
               if line.startswith(prefix)]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"expected one {marker} report")
    return records[0]


def validate_drain(report: dict[str, Any], requests: int) -> None:
    stats = report["stats"]
    for name in ("admitted_requests", "completed_requests"):
        if stats[name] != requests:
            raise ValueError(f"unexpected {name}: {stats[name]}")
    for name in ("active_requests", "queued_requests", "cancelled_requests"):
        if stats[name] != 0:
            raise ValueError(f"nonzero {name}: {stats[name]}")
    cumulative = {"free_pages", "evicted_prefixes", "exhausted_pages"}
    for name, value in stats["manager"].items():
        if name not in cumulative and value != 0:
            raise ValueError(f"undrained KV state {name}: {value}")
    for _, state in report["fixed_states"]:
        if state["free_slots"] != state["identity"]["slot_count"]:
            raise ValueError("fixed-state slots did not drain")
        if any(value != 0 for name, value in state.items()
               if name not in ("identity", "free_slots")):
            raise ValueError("fixed-state ownership did not drain")


class ProcessMemorySampler:
    """Sample serving-process RSS and NVML-reported GPU memory after readiness.

    No CUDA calls or synchronization are added to the server. Values include
    retained allocations and are sampled observations, not allocation peaks.
    The first sample and thread shutdown stay outside client timing.
    """

    def __init__(self, pid: int, interval_seconds: float, device_index: int | None = None):
        self.pid = pid
        self.interval = interval_seconds
        self.device_index = device_index
        self.samples: list[dict[str, int]] = []
        self.errors: set[str] = set()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        sample = {}
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    sample["host_rss_bytes"] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError) as error:
            self.errors.add(f"host memory: {error}")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=3, check=True,
            )
            values = [int(memory.strip()) * 1024 * 1024
                      for pid, memory in (line.split(",", 1)
                                          for line in result.stdout.splitlines())
                      if int(pid.strip()) == self.pid]
            if values:
                sample["gpu_bytes"] = sum(values)
            else:
                self.errors.add("GPU process memory unavailable")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.errors.add(f"GPU memory: {error}")
        if self.device_index is not None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--id={self.device_index}", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, timeout=3, check=True,
                )
                sample["device_gpu_bytes"] = int(result.stdout.strip()) * 1024 * 1024
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors.add(f"device memory: {error}")
        self.samples.append(sample)

    def _run(self) -> None:
        while not self._done.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        self._thread.join()

    def report(self) -> dict[str, Any]:
        return {
            "method": "periodic process RSS and nvidia-smi after server readiness",
            "interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "sampled_max_gpu_bytes": max(
                (s["gpu_bytes"] for s in self.samples if "gpu_bytes" in s), default=None),
            "sampled_max_host_rss_bytes": max(
                (s["host_rss_bytes"] for s in self.samples if "host_rss_bytes" in s), default=None),
            "device_index": self.device_index,
            "sampled_max_device_gpu_bytes": max(
                (s["device_gpu_bytes"] for s in self.samples if "device_gpu_bytes" in s), default=None),
            "errors": sorted(self.errors),
        }
