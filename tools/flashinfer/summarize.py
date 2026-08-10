#!/usr/bin/env python3
"""Summarize matched JSON Lines samples without merging different contracts."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTRACT_FIELDS = (
    "schema_version",
    "operator",
    "case",
    "measurement",
    "dtype",
    "layout",
    "shape",
    "fixture_id",
    "fixture_digests",
    "warmup_launches",
    "launches_per_sample",
)
RUN_FIELDS = (
    "provider",
    "provider_version",
    "provider_commit",
    "run_label",
)
REQUIRED_FIELDS = (*CONTRACT_FIELDS, *RUN_FIELDS, "execution", "samples_us")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def frozen(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def field_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(frozen(record[field]) for field in fields)


def execution_identity(record: dict[str, Any]) -> dict[str, Any]:
    execution = record["execution"]
    if not isinstance(execution, dict):
        raise ValueError("execution must be a JSON object")
    return {
        field: value
        for field, value in execution.items()
        if field != "correctness"
    }


def correctness_record(record: dict[str, Any]) -> Any:
    return record["execution"].get("correctness")


def validate_record(record: dict[str, Any], source: str) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{source}: record is missing required fields: {names}")
    values = record["samples_us"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{source}: samples_us must be a nonempty list")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
        for value in values
    ):
        raise ValueError(f"{source}: samples_us must contain positive finite numbers")


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_string in paths:
        path = Path(path_string)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = f"{path}:{line_number}"
            if not isinstance(record, dict):
                raise ValueError(f"{source}: record must be a JSON object")
            validate_record(record, source)
            records.append(record)
    return records


def require_one_contract(records: list[dict[str, Any]]) -> None:
    contracts: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {}
    for index, record in enumerate(records, start=1):
        logical_key = (record["operator"], record["case"])
        contract_key = field_key(record, CONTRACT_FIELDS)
        previous = contracts.get(logical_key)
        if previous is None:
            contracts[logical_key] = (contract_key, f"record {index}")
            continue
        previous_key, previous_source = previous
        if previous_key == contract_key:
            continue
        changed = [
            field
            for field, left, right in zip(
                CONTRACT_FIELDS, previous_key, contract_key
            )
            if left != right
        ]
        names = ", ".join(changed)
        raise ValueError(
            f"operator={logical_key[0]} case={logical_key[1]} has incompatible "
            f"metadata between {previous_source} and record {index}: {names}"
        )


def sample_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.10),
        "p90_us": percentile(values, 0.90),
        "min_us": min(values),
        "max_us": max(values),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    for index, record in enumerate(records, start=1):
        validate_record(record, f"record {index}")
    require_one_contract(records)
    grouped_samples: dict[tuple[str, ...], list[float]] = defaultdict(list)
    grouped_metadata: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            *field_key(record, (*CONTRACT_FIELDS, *RUN_FIELDS)),
            frozen(execution_identity(record)),
        )
        previous = grouped_metadata.get(key)
        if previous is not None and frozen(correctness_record(previous)) != frozen(
            correctness_record(record)
        ):
            raise ValueError(
                f"provider={record['provider']} case={record['case']} "
                f"run_label={record['run_label']} has inconsistent correctness records"
            )
        grouped_samples[key].extend(float(value) for value in record["samples_us"])
        grouped_metadata[key] = record

    logical_cases: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for key, record in grouped_metadata.items():
        logical_cases[(record["operator"], record["case"])].append(key)

    summaries = []
    for logical_key in sorted(logical_cases):
        keys = logical_cases[logical_key]
        contract_record = grouped_metadata[keys[0]]
        contract = {field: contract_record[field] for field in CONTRACT_FIELDS}
        runs = []
        for key in sorted(
            keys,
            key=lambda item: (
                grouped_metadata[item]["provider"],
                grouped_metadata[item]["run_label"],
                frozen(execution_identity(grouped_metadata[item])),
            ),
        ):
            record = grouped_metadata[key]
            run = {field: record[field] for field in RUN_FIELDS}
            run["execution"] = record["execution"]
            run.update(sample_stats(grouped_samples[key]))
            runs.append(run)
        summaries.append({"contract": contract, "runs": runs})

    return {"schema_version": 2, "cases": summaries}


def main(paths: list[str]) -> None:
    print(json.dumps(summarize_records(load_records(paths)), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize.py RESULTS.jsonl [RESULTS.jsonl ...]")
    main(sys.argv[1:])
