"""Cache-source evidence and latency summaries shared by benchmark reports."""

from __future__ import annotations

import math
import statistics

import requests


def metrics(url: str | None) -> dict[str, float]:
    if url is None:
        return {}
    response = requests.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        name = name.split("{", 1)[0]
        if name.endswith("_created") or "bucket" in name:
            continue
        number = float(value)
        if math.isfinite(number):
            values[name] = values.get(name, 0) + number
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
    for length in lengths:
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
                }
            )
    return summary
