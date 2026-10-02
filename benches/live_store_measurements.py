"""Source-only timing and process observations for live-store experiments."""

from __future__ import annotations

import os
import statistics
from pathlib import Path

MEASUREMENT_CONTRACT = "s2.10-performance-v2"


def _summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction):
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "samples": len(ordered),
        "p50": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _process_sample(pids):
    result = {
        "thread_context_switches": {},
        "context_switch_scope": "snapshot sum of live threads; exited threads excluded",
        "cpu_ticks": 0,
        "rss_kib": 0,
        "hwm_kib": 0,
        "voluntary_context_switches": 0,
        "nonvoluntary_context_switches": 0,
        "cpu_affinity": [],
        "memory_affinity": [],
        "processes": 0,
    }
    cpu_affinity = set()
    memory_affinity = set()
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, IndexError):
            continue
        values = {}
        for line in status:
            key, _, value = line.partition(":")
            if key in {
                "VmRSS",
                "VmHWM",
                "voluntary_ctxt_switches",
                "nonvoluntary_ctxt_switches",
            }:
                values[key] = int(value.split()[0])
            elif key == "Cpus_allowed_list":
                cpu_affinity.add(value.strip())
            elif key == "Mems_allowed_list":
                memory_affinity.add(value.strip())
        result["cpu_ticks"] += int(stat[11]) + int(stat[12])
        result["rss_kib"] += values.get("VmRSS", 0)
        result["hwm_kib"] += values.get("VmHWM", 0)
        for thread in Path(f"/proc/{pid}/task").iterdir():
            try:
                fields = dict(
                    line.split(":", 1) for line in (thread / "status").read_text().splitlines()
                )
                thread_stat = (thread / "stat").read_text().rsplit(") ", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            identity = f"{pid}:{thread.name}:{thread_stat[19]}"
            counters = [
                int(fields[name])
                for name in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
            ]
            result["thread_context_switches"][identity] = counters
            result["voluntary_context_switches"] += counters[0]
            result["nonvoluntary_context_switches"] += counters[1]
        result["processes"] += 1
    result["cpu_affinity"] = sorted(cpu_affinity)
    result["memory_affinity"] = sorted(memory_affinity)
    return result


def _clock_domain(pids):
    pids = [os.getpid(), *pids]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    clocks = {}
    for pid in pids:
        clocks[str(pid)] = {
            "time_namespace": os.readlink(f"/proc/{pid}/ns/time"),
            "time_offsets": Path(f"/proc/{pid}/timens_offsets").read_text(),
        }
    assert len({(row["time_namespace"], row["time_offsets"]) for row in clocks.values()}) == 1, (
        clocks
    )
    return {"clock": "CLOCK_MONOTONIC", "boot_id": boot_id, "processes": clocks}


def _installation_sample(targets, installed, harness_completed_ns):
    owners = []
    for node, target in targets.items():
        row = installed[node]
        install_ns = row["installed_mono_ns"]
        assert row["applied_sequence"] == target["sequence"], (target, row)
        assert row["view_id"] == target["view_id"], (target, row)
        assert target["published_mono_ns"] > 0
        assert target["published_mono_ns"] <= install_ns <= harness_completed_ns
        owners.append(
            {
                **target,
                "installed_sequence": row["applied_sequence"],
                "installed_view_id": row["view_id"],
                "installed_mono_ns": install_ns,
                "publication_to_install_ms": (install_ns - target["published_mono_ns"]) / 1_000_000,
                "save_start_to_install_ms": (install_ns - target["save_start_mono_ns"]) / 1_000_000,
                "install_to_harness_ms": (harness_completed_ns - install_ns) / 1_000_000,
            }
        )
    return {
        "owners": owners,
        "publication_to_install_ms": max(row["publication_to_install_ms"] for row in owners),
        "save_start_to_install_ms": max(row["save_start_to_install_ms"] for row in owners),
        "install_to_harness_ms": max(row["install_to_harness_ms"] for row in owners),
    }


def _context_switch_delta(before, after):
    initial = before["thread_context_switches"]
    final = after["thread_context_switches"]
    surviving = initial.keys() & final.keys()
    return {
        "scope": "surviving thread identities only; exited threads are not a measured zero",
        "surviving_threads": len(surviving),
        "exited_threads": len(initial.keys() - final.keys()),
        "new_threads": len(final.keys() - initial.keys()),
        "voluntary": sum(final[key][0] - initial[key][0] for key in surviving),
        "nonvoluntary": sum(final[key][1] - initial[key][1] for key in surviving),
    }
