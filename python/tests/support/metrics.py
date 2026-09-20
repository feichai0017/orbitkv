"""Read engine and Cache Manager counters for correctness gates."""

import re

import requests


def fetch_orbitkv_metrics(metrics_port: int) -> dict[str, float]:
    """Fetch and parse Prometheus metrics from OrbitKV server.

    Args:
        metrics_port: Port where OrbitKV exposes /metrics endpoint.

    Returns:
        Dict mapping metric name to value (for counters/gauges).
    """
    url = f"http://localhost:{metrics_port}/metrics"
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    metrics = {}
    for line in response.text.splitlines():
        # Skip comments and empty lines
        if line.startswith("#") or not line.strip():
            continue
        # Parse: metric_name{labels} value or metric_name value
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\{[^}]*\})?\s+([\d.eE+-]+)$", line)
        if match:
            name, value = match.groups()
            # Accumulate values for metrics with labels (e.g., sum across all labels)
            metrics[name] = metrics.get(name, 0) + float(value)
    return metrics


def fetch_orbitkv_rpc_failures(metrics_port: int, method: str | None = None) -> dict[str, float]:
    """Return non-ok RPC counts keyed by ``"method/status"``.

    ``fetch_orbitkv_metrics`` collapses label sets, so it cannot tell a failed
    RPC from a successful one. This keeps the ``method``/``status`` labels so a
    test can assert that no connector<->server RPC failed. Pass ``method`` to
    restrict to one RPC; ``None`` (default) reports failures across all methods.
    """
    url = f"http://localhost:{metrics_port}/metrics"
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    failures: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line.startswith("orbitkv_rpc_requests"):
            continue
        match = re.match(r"^orbitkv_rpc_requests(?:_total)?\{([^}]*)\}\s+([\d.eE+-]+)$", line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        if labels.get("status") == "ok":
            continue
        rpc = labels.get("method", "")
        if method is not None and rpc != method:
            continue
        key = f"{rpc}/{labels.get('status', '')}"
        failures[key] = failures.get(key, 0.0) + float(match.group(2))
    return failures


def fetch_vllm_prefix_cache_hits(port: int) -> float:
    """Read vLLM's native (not external-connector) prefix-hit token counter."""
    response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
    response.raise_for_status()
    hits = 0.0
    found = False
    for line in response.text.splitlines():
        match = re.match(
            r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([\d.eE+-]+)$",
            line,
        )
        if match:
            found = True
            hits += float(match.group(1))
    if not found:
        raise AssertionError("vLLM native prefix-cache hit counter is absent from /metrics")
    return hits
