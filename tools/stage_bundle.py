#!/usr/bin/env python3
"""Build and atomically publish a qualified plan bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_SRC = ROOT / "integrations" / "providers" / "src"
sys.path.insert(0, str(PROVIDER_SRC))

from aletheia_providers.qualification import (  # noqa: E402
    canonical_sha256,
    load_jsonl,
    summarize_samples,
    trace_covers,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def singleton_workload(domain: dict[str, Any]) -> dict[str, Any]:
    phases = domain.get("phases", [])
    if len(phases) != 1:
        raise ValueError("qualification currently requires exactly one phase")
    point = {"phase": phases[0]}
    for key in ("batch", "rows_per_sequence", "context_tokens"):
        bounds = domain.get(key, {})
        if bounds.get("min") != bounds.get("max"):
            raise ValueError(f"qualification currently requires singleton {key}")
        point[key] = bounds.get("min")
    return point


def build_certificate(plan: dict[str, Any], evidence: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    if evidence.get("schema_version") != 1:
        raise ValueError("qualification evidence schema_version must be 1")
    point = singleton_workload(plan["workload"])
    events = load_jsonl(trace_path)
    if not trace_covers(events, point):
        raise ValueError("SGLang trace does not contain a successful event for the plan workload point")

    numerical = evidence.get("numerical", {})
    resources = evidence.get("resources", {})
    resilience = evidence.get("resilience", {})
    timing = evidence.get("timing", {})
    required = {
        "numerical.oracle": numerical.get("oracle"),
        "numerical.cases": numerical.get("cases"),
        "numerical.atol": numerical.get("atol"),
        "numerical.rtol": numerical.get("rtol"),
        "resources.peak_device_memory_bytes": resources.get("peak_device_memory_bytes"),
        "resources.peak_workspace_bytes": resources.get("peak_workspace_bytes"),
        "timing.samples_micros": timing.get("samples_micros"),
        "timing.tokens_per_sample": timing.get("tokens_per_sample"),
        "hardware": evidence.get("hardware"),
        "software": evidence.get("software"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("qualification evidence is incomplete: " + ", ".join(missing))
    if len(timing["samples_micros"]) < 12:
        raise ValueError("qualification requires at least 12 raw timing samples")
    if not evidence["software"]:
        raise ValueError("qualification requires a non-empty software fingerprint")

    performance = summarize_samples(timing["samples_micros"], int(timing["tokens_per_sample"]))
    samples = performance.pop("samples_micros")
    performance["workload_sha256"] = canonical_sha256({"point": point, "samples_micros": samples})

    return {
        "schema_version": 1,
        "plan_id": plan["id"],
        "plan_sha256": "",
        "status": "qualified",
        "hardware": evidence["hardware"],
        "workload": plan["workload"],
        "numerical": {
            "oracle": numerical["oracle"],
            "cases": numerical["cases"],
            "atol": numerical["atol"],
            "rtol": numerical["rtol"],
            "max_abs_error": numerical.get("max_abs_error", 0.0),
            "min_output_agreement": numerical.get("min_output_agreement", 1.0),
        },
        "performance": performance,
        "resources": resources,
        "resilience": {
            "soak_seconds": resilience.get("soak_seconds", 0),
            "passed_faults": resilience.get("passed_faults", []),
        },
        "software": evidence["software"],
    }


def verified_artifacts(plan: dict[str, Any], evidence: dict[str, Any], evidence_path: Path) -> list[tuple[Path, str]]:
    declared = evidence.get("artifacts", [])
    by_digest = {item.get("sha256"): item for item in declared if isinstance(item, dict)}
    required = {step["artifact_sha256"] for step in plan.get("steps", [])}
    missing = sorted(required - by_digest.keys())
    if missing:
        raise ValueError("qualification evidence has no file for artifact(s): " + ", ".join(missing))
    verified = []
    for digest in sorted(required):
        item = by_digest[digest]
        source = Path(item["path"])
        if not source.is_absolute():
            source = evidence_path.parent / source
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"artifact digest mismatch for {source}: expected {digest}, got {actual}")
        verified.append((source, digest))
    return verified


def aletheia_binary(path: Path | None) -> list[str]:
    if path is not None:
        return [str(path)]
    return ["cargo", "run", "--locked", "-q", "-p", "aletheia-cli", "--"]


def command(prefix: list[str], arguments: list[str]) -> str:
    result = subprocess.run(
        [*prefix, *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def stage(plan_path: Path, evidence_path: Path, trace_path: Path, registry: Path, binary: Path | None) -> Path:
    plan = load(plan_path)
    runner = aletheia_binary(binary)
    # Validate the identifier and the complete plan before using its ID as a
    # directory name or trusting any of its artifact declarations.
    command(runner, ["validate-plan", str(plan_path)])
    evidence = load(evidence_path)
    certificate = build_certificate(plan, evidence, trace_path)
    artifacts = verified_artifacts(plan, evidence, evidence_path)

    registry.mkdir(parents=True, exist_ok=True)
    destination = registry / plan["id"]
    if destination.exists():
        raise ValueError(f"plan bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{plan['id']}.", dir=registry))
    try:
        staged_plan = temporary / "plan.json"
        staged_certificate = temporary / "certificate.json"
        staged_trace = temporary / "trace.jsonl"
        staged_evidence = temporary / "evidence.json"
        artifact_dir = temporary / "artifacts"
        artifact_dir.mkdir()
        staged_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        plan_digest = command(runner, ["validate-plan", str(staged_plan)])
        certificate["plan_sha256"] = plan_digest
        staged_certificate.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
        shutil.copy2(trace_path, staged_trace)
        staged_evidence_value = dict(evidence)
        staged_evidence_value["artifacts"] = [
            {"path": f"artifacts/{digest}", "sha256": digest} for _, digest in artifacts
        ]
        staged_evidence.write_text(json.dumps(staged_evidence_value, indent=2, sort_keys=True) + "\n")
        for source, digest in artifacts:
            shutil.copy2(source, artifact_dir / digest)
        command(runner, ["validate-certificate", str(staged_plan), str(staged_certificate)])
        os.rename(temporary, destination)
        try:
            command(runner, ["validate-registry", str(registry)])
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--aletheia-bin", type=Path)
    args = parser.parse_args()
    destination = stage(args.plan, args.evidence, args.trace, args.registry, args.aletheia_bin)
    print(destination)


if __name__ == "__main__":
    main()
