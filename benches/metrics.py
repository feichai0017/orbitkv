"""Cache-source evidence and latency summaries shared by benchmark reports."""

from __future__ import annotations

import math
import statistics
import threading
import time
from contextlib import contextmanager

import requests

REMOTE_STAGES = ("discovery_rpc", "authorization", "allocation", "read", "rebuild", "release")


def delta(before: dict, after: dict) -> dict:
    return {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if value != before.get(key, 0)
    }


def cached_tokens(engine: str, sample: dict) -> int:
    usage = sample["usage"]
    if engine == "sglang":
        return usage.get("cached_tokens", 0)
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


@contextmanager
def measure(base_url: str, manager_url: str | None, settle_seconds: float):
    before, manager_before = metrics(base_url), metrics(manager_url)
    stop = threading.Event()
    peaks, errors, result = {}, [], {}

    def observe():
        while not stop.is_set():
            try:
                for key, value in metrics(manager_url).items():
                    if key in (
                        "orbitkv_pool_used_bytes",
                        "orbitkv_warmup_pending_bytes",
                    ) or key.startswith("orbitkv_query_reserved_bytes"):
                        peaks[key] = max(value, peaks.get(key, 0))
            except Exception as error:
                errors.append(str(error))
                return
            stop.wait(0.025)

    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        yield result
        result["wall_seconds"] = time.perf_counter() - started
        drain_started = time.perf_counter()
        time.sleep(settle_seconds)
        deadline = time.monotonic() + 30
        quiet = 0
        while True:
            manager_after = metrics(manager_url)
            busy = any(
                manager_after.get(name, 0) != 0
                for name in (
                    "orbitkv_query_reserved_bytes",
                    "orbitkv_inflight_bytes",
                    "orbitkv_ssd_write_queue_pending",
                    "orbitkv_ssd_write_inflight",
                    "orbitkv_ssd_prefetch_inflight",
                )
            )
            quiet = 0 if busy else quiet + 1
            if quiet >= 3:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Cache work survived completed requests: {manager_after}")
            time.sleep(0.1)
        result.update(
            drain_seconds=time.perf_counter() - drain_started,
            metrics_delta=delta(before, metrics(base_url)),
            manager_delta=delta(manager_before, manager_after),
            sampled_peak_bytes=peaks,
            manager_before=manager_before,
            manager_after=manager_after,
        )
    finally:
        stop.set()
        monitor.join(timeout=12)
    if monitor.is_alive() or errors:
        raise RuntimeError(f"Memory observation failed: {errors}")


def workload_phases(ssd: bool = False) -> tuple[str, ...]:
    phases = ("cold", "hbm_hit", "after_pressure")
    return (*phases, "after_host_eviction") if ssd else phases


def metrics(url: str | None) -> dict[str, float]:
    if url is None:
        return {}
    response = requests.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, value = line.rsplit(" ", 1)
        name = series.split("{", 1)[0]
        if name.endswith("_created") or "bucket" in name:
            continue
        number = float(value)
        if math.isfinite(number):
            values[name] = values.get(name, 0) + number
            if name.startswith("orbitkv_remote_stage_duration_seconds_"):
                for stage in REMOTE_STAGES:
                    if f'stage="{stage}"' in series:
                        key = f"{name}_{stage}"
                        values[key] = values.get(key, 0) + number
            if name == "orbitkv_query_reserved_bytes" and 'phase="warming"' in series:
                key = "orbitkv_query_reserved_bytes_warming"
                values[key] = values.get(key, 0) + number
            if name == "orbitkv_query_reserved_bytes" and any(
                f'phase="{phase}"' in series for phase in ("warming", "preloading", "prepared")
            ):
                key = "orbitkv_query_reserved_bytes_speculative"
                values[key] = values.get(key, 0) + number
            if name == "orbitkv_warmup_wait_byte_seconds_total":
                for outcome in ("restored", "unused"):
                    if f'outcome="{outcome}"' in series:
                        key = f"{name}_{outcome}"
                        values[key] = values.get(key, 0) + number
    return values


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
    external = max(external, result["manager_delta"].get("orbitkv_load_bytes_total", 0))
    if device > 0:
        return "mixed" if external > 0 else "hbm"
    if external > 0:
        return "external"
    cached = (
        result["usage"].get("cached_tokens", 0)
        if engine == "sglang"
        else (result["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    )
    return "unverified" if cached else "miss"


def summarize(samples: list[dict], lengths: list[int]) -> list[dict]:
    summary = []
    phases = tuple(dict.fromkeys(sample["phase"] for sample in samples))
    for length in lengths:
        for phase in phases:
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
                    "e2e_p50_ms": statistics.median(sample["e2e_ms"] for sample in group),
                    "output_mismatches": sum(not sample["matches_cold_output"] for sample in group),
                    "cache_sources": {
                        source: sum(sample["cache_source"] == source for sample in group)
                        for source in sorted({sample["cache_source"] for sample in group})
                    },
                    "orbitkv_load_bytes": sum(
                        sample["manager_delta"].get("orbitkv_load_bytes_total", 0)
                        for sample in group
                    ),
                    "orbitkv_ssd_read_bytes": sum(
                        sample["manager_delta"].get("orbitkv_ssd_prefetch_bytes_total", 0)
                        for sample in group
                    ),
                    "ssd_reads_without_gpu_restore": sum(
                        sample["manager_delta"].get("orbitkv_ssd_prefetch_bytes_total", 0) > 0
                        and sample["manager_delta"].get("orbitkv_load_bytes_total", 0) == 0
                        for sample in group
                    ),
                    "load_task_p50_ms": statistics.median(
                        sample["manager_delta"].get("orbitkv_load_duration_seconds_sum", 0) * 1000
                        for sample in group
                    ),
                    "ssd_prefetch_p50_ms": statistics.median(
                        sample["manager_delta"].get("orbitkv_ssd_prefetch_duration_seconds_sum", 0)
                        * 1000
                        for sample in group
                    ),
                }
            )
    return summary
