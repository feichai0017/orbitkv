#!/usr/bin/env python3
"""Measure the ABI6 compact host control path without starting a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.config import load_config  # noqa: E402
from orbitkv_sglang.ffi import CtypesManagerFactory  # noqa: E402
from orbitkv_sglang.runtime import (  # noqa: E402
    ArenaRegistration,
    CanonicalRuntime,
    ManagerCreateSettings,
    MirrorCleanupBinding,
)


PAGE_TOKENS = 16
HYBRID_WINDOW_TOKENS = 18
HOST_GATE_TOTAL_P50_LIMIT_MS = 1.25
Profile = Literal["full", "hybrid"]


class _ReadyEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _NoopCleanup:
    def preflight(self, items, _retirements):
        return tuple(items)

    def commit(self, _plan) -> None:
        return None

    def synchronize(self, _plan) -> None:
        return None

    def finalize(self, _plan) -> None:
        return None


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_inputs(
    profile: str, batch_size: int, resident_pages: int, iterations: int
) -> Profile:
    if profile not in ("full", "hybrid"):
        raise ValueError("profile must be exactly 'full' or 'hybrid'")
    if isinstance(batch_size, bool) or batch_size not in (1, 4):
        raise ValueError("batch-size must be exactly 1 or 4")
    _positive("resident-pages", resident_pages)
    _positive("iterations", iterations)

    resident_tokens = resident_pages * PAGE_TOKENS
    if resident_tokens >= 1 << 32:
        raise ValueError("resident token count does not fit maximum_step_tokens")
    arena_page_count = batch_size * (
        resident_pages + _ceil_div(iterations, PAGE_TOKENS)
    )
    class_count = 2 if profile == "hybrid" else 1
    if arena_page_count >= 1 << 32:
        raise ValueError("per-arena page count does not fit uint32_t")
    if arena_page_count * class_count >= 1 << 32:
        raise ValueError("total reclamation capacity does not fit uint32_t")
    return profile


def _profile_plan(profile: Profile) -> dict:
    classes = [
        {
            "name": "full",
            "layers": [0],
            "retention": "full",
            "bytes_per_token_per_layer": 128,
            "window_tokens": None,
        }
    ]
    if profile == "hybrid":
        classes.append(
            {
                "name": "swa",
                "layers": [1],
                "retention": "sliding",
                "bytes_per_token_per_layer": 128,
                "window_tokens": HYBRID_WINDOW_TOKENS,
            }
        )
    return {"page_tokens": PAGE_TOKENS, "classes": classes}


def _arena_page_count(batch_size: int, resident_pages: int, iterations: int) -> int:
    # The untimed initial step reserves every logical page, including pages a
    # sliding class retires at publication. Each request then needs exactly one
    # new page for every page boundary crossed by the timed token steps.
    return batch_size * (
        resident_pages + _ceil_div(iterations, PAGE_TOKENS)
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values),
        "p99_ms": _percentile(values, 0.99),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    library: Path,
    *,
    profile: str,
    batch_size: int,
    resident_pages: int,
    iterations: int,
) -> dict:
    selected_profile = _validate_inputs(
        profile, batch_size, resident_pages, iterations
    )
    initial_boundary = resident_pages * PAGE_TOKENS
    page_count = _arena_page_count(batch_size, resident_pages, iterations)
    plan = _profile_plan(selected_profile)
    class_count = len(plan["classes"])
    limits = {
        "maximum_requests": batch_size,
        "maximum_operations": batch_size,
        "maximum_prefixes": batch_size,
        "maximum_reclamations": page_count * class_count,
        "maximum_step_tokens": initial_boundary,
    }
    arena_layout = tuple(
        {
            "class_id": class_id,
            "pool_id": class_id + 1,
            "backend_domain": class_id + 1,
            "page_count": page_count,
            "backend_base_index": 0,
        }
        for class_id in range(class_count)
    )

    with tempfile.TemporaryDirectory(prefix="orbitkv-compact-control-") as directory:
        plan_path = Path(directory) / "plan.json"
        plan_path.write_text(
            json.dumps(plan, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        config = load_config(
            {"ORBITKV_PLAN": str(plan_path), "ORBITKV_LIBRARY": str(library)}
        )
        manager = CtypesManagerFactory().create(
            config,
            ManagerCreateSettings(**limits),
            tuple(ArenaRegistration(**item) for item in arena_layout),
        )
        runtime = CanonicalRuntime(config, manager)
        keys = tuple(f"request-{index}" for index in range(batch_size))
        cleanup = _NoopCleanup()

        # Setup is intentionally outside every timer. The same cleanup
        # coordinator is bound to the full B1/B4 EventGroup so Hybrid
        # reclamation remains one collective transaction.
        initial, _ = runtime.prepare_batch(
            tuple((key, initial_boundary) for key in keys)
        )
        for key in keys:
            runtime.bind_reclamation_cleanup(
                key, MirrorCleanupBinding(cleanup, key)
            )
        runtime.mark_lowered(initial)
        runtime.submit_batch(initial)
        runtime.mark_forward(initial)
        runtime.register_event(initial, _ReadyEvent(), 1)
        runtime.poll()

        timings = {
            "prepare": [],
            "submit": [],
            "complete": [],
            "total": [],
        }
        for target in range(
            initial_boundary + 1, initial_boundary + iterations + 1
        ):
            total_start = time.perf_counter_ns()
            phase_start = total_start
            batch, _ = runtime.prepare_batch(tuple((key, target) for key in keys))
            timings["prepare"].append(
                (time.perf_counter_ns() - phase_start) / 1_000_000
            )

            phase_start = time.perf_counter_ns()
            runtime.mark_lowered(batch)
            runtime.submit_batch(batch)
            timings["submit"].append(
                (time.perf_counter_ns() - phase_start) / 1_000_000
            )

            runtime.mark_forward(batch)
            runtime.register_event(batch, _ReadyEvent(), 1)
            phase_start = time.perf_counter_ns()
            runtime.poll()
            timings["complete"].append(
                (time.perf_counter_ns() - phase_start) / 1_000_000
            )
            timings["total"].append(
                (time.perf_counter_ns() - total_start) / 1_000_000
            )

        counters = runtime.performance_counters()
        compact_counters = {
            name: counters[name]
            for name in (
                "hot_workspace_allocations",
                "capacity_memset_bytes",
                "root_entries_crossed",
                "materialized_page_objects",
            )
        }
        if any(compact_counters.values()):
            raise RuntimeError("compact control path crossed a forbidden hot surface")
        runtime.release_batch(keys)
        runtime.close()

    phase_summaries = {
        name: _summary(values) for name, values in timings.items()
    }
    host_gate_passed = (
        not any(compact_counters.values())
        and phase_summaries["total"]["p50_ms"]
        < HOST_GATE_TOTAL_P50_LIMIT_MS
    )
    return {
        "schema": "orbitkv.abi6-prefix-control.v1",
        "scope": "host_control_only",
        "python": platform.python_version(),
        "library": {"path": str(library), "sha256": _sha256(library)},
        "profile": selected_profile,
        "batch_size": batch_size,
        "page_tokens": PAGE_TOKENS,
        "hybrid_window_tokens": (
            HYBRID_WINDOW_TOKENS if selected_profile == "hybrid" else None
        ),
        "resident_pages_per_request": resident_pages,
        "resident_tokens_per_request": initial_boundary,
        "iterations": iterations,
        "setup_timed": False,
        "manager_limits": limits,
        "arenas": list(arena_layout),
        "phases": phase_summaries,
        "compact_counters": compact_counters,
        "host_gate_total_p50_limit_ms": HOST_GATE_TOTAL_P50_LIMIT_MS,
        "host_gate_passed": host_gate_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "hybrid"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--resident-pages", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    arguments = parser.parse_args()
    library = arguments.library.resolve(strict=True)
    print(
        json.dumps(
            run(
                library,
                profile=arguments.profile,
                batch_size=arguments.batch_size,
                resident_pages=arguments.resident_pages,
                iterations=arguments.iterations,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
