"""Qualify independent matching replicas through their existing serving APIs.

The Managers own discovery, authorization, transfer and source reservations in
Rust. This driver submits token-exact requests and checks their exported evidence.
Run only against idle, dedicated qualification deployments with fresh prefixes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from .metrics import delta, metrics
from .workload import generate

IDLE_METRICS = (
    "orbitkv_query_reserved_bytes",
    "orbitkv_inflight_bytes",
    "orbitkv_transfer_lock_active",
    "orbitkv_transfer_reserved_bytes",
    "orbitkv_transfer_completion_outstanding",
    "orbitkv_ssd_prefetch_inflight",
    "orbitkv_ssd_write_queue_pending",
    "orbitkv_ssd_write_inflight",
)


def drain(*manager_urls: str, timeout: float = 30) -> list[dict]:
    deadline = time.monotonic() + timeout
    quiet = 0
    while True:
        snapshots = [metrics(url) for url in manager_urls]
        busy = [
            {name: snapshot[name] for name in IDLE_METRICS if snapshot.get(name, 0)}
            for snapshot in snapshots
        ]
        quiet = 0 if any(busy) else quiet + 1
        if quiet == 3:
            return snapshots
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Shared-cache resources did not drain: {busy}")
        time.sleep(0.1)


def synchronize(manager_url: str) -> None:
    response = requests.post(f"{manager_url}/cache/sync", timeout=35)
    response.raise_for_status()


def verify_restore(before: dict, after: dict, expected: str, actual: dict) -> dict:
    changes = delta(before, after)
    remote = changes.get("orbitkv_remote_fetch_bytes_total", 0)
    restored = changes.get("orbitkv_load_bytes_total", 0)
    if remote <= 0 or restored <= 0:
        raise AssertionError(f"Expected both Mooncake READ and GPU restore bytes: {changes}")
    if actual["text"] != expected:
        raise AssertionError("Shared-cache output differs from the cold source control")
    return {
        "ttft_ms": actual["ttft_ms"],
        "e2e_ms": actual["e2e_ms"],
        "remote_bytes": int(remote),
        "h2d_bytes": int(restored),
        "discovery_rpcs": int(changes.get("orbitkv_candidate_lookup_rpcs_total", 0)),
        "remote_stages": {
            stage: {
                "calls": int(
                    changes.get(f"orbitkv_remote_stage_duration_seconds_count_{stage}", 0)
                ),
                "total_ms": changes.get(f"orbitkv_remote_stage_duration_seconds_sum_{stage}", 0)
                * 1000,
            }
            for stage in ("discovery_rpc", "authorization", "allocation", "read", "rebuild")
        },
        "output_match": True,
    }


def qualify(
    *,
    engine: str,
    model: str,
    source_url: str,
    target_url: str,
    source_manager: str,
    target_manager: str,
    prompts: list[list[int]],
    output_tokens: int = 8,
) -> list[dict]:
    if source_url.rstrip("/") == target_url.rstrip("/") or source_manager.rstrip(
        "/"
    ) == target_manager.rstrip("/"):
        raise ValueError("Shared-cache qualification requires two engines and two Managers")
    results = []
    for index, prompt in enumerate(prompts):
        source_before, _ = drain(source_manager, target_manager)
        cold = generate(source_url, engine, model, prompt, output_tokens)
        synchronize(source_manager)
        source_after, target_before = drain(source_manager, target_manager)
        saved = source_after.get("orbitkv_save_bytes_total", 0) - source_before.get(
            "orbitkv_save_bytes_total", 0
        )
        if saved <= 0:
            raise AssertionError("Source did not publish new pages; use fresh prefixes")
        if source_after.get("orbitkv_load_bytes_total", 0) != source_before.get(
            "orbitkv_load_bytes_total", 0
        ):
            raise AssertionError("Source control unexpectedly restored cached pages")
        restored = generate(target_url, engine, model, prompt, output_tokens)
        _, target_after = drain(source_manager, target_manager)
        row = verify_restore(target_before, target_after, cold["text"], restored)
        row.update(
            prompt=index,
            input_tokens=len(prompt),
            source_ttft_ms=cold["ttft_ms"],
            save_bytes=int(saved),
            resources_drained=True,
        )
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument(
        "--model", required=True, help="Identical served model name on both replicas"
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--source-manager", required=True)
    parser.add_argument("--target-manager", required=True)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON array of fresh token-ID arrays"
    )
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument(
        "--deployment",
        choices=("same-host-tcp", "two-host-tcp", "two-host-rdma"),
        required=True,
        help="Operator-declared environment; this driver does not detect physical hosts or RDMA",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(
            not isinstance(prompt, list)
            or len(prompt) < 128
            or any(type(token) is not int or token < 0 for token in prompt)
            for prompt in prompts
        )
        or args.output_tokens < 1
    ):
        parser.error(
            "provide nonempty prompts of at least 128 token IDs and positive output tokens"
        )
    results = qualify(
        engine=args.engine,
        model=args.model,
        source_url=args.source_url,
        target_url=args.target_url,
        source_manager=args.source_manager,
        target_manager=args.target_manager,
        prompts=prompts,
        output_tokens=args.output_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "engine": args.engine,
                "model": args.model,
                "deployment_declared": args.deployment,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(results)} remote restores with drained resources: {args.output}")


if __name__ == "__main__":
    main()
