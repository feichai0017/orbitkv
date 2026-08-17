from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/h20-gpt-oss-20b-real-20260817"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"expected {expected}, got {actual}")


def main() -> None:
    checkpoint = load("checkpoint.json")
    capacity = load("capacity.json")
    admission = load("admission.json")
    ablation = load("ablation.json")
    overhead = load("overhead.json")
    owner_trace = load("owner-trace.json")
    physical_plan = load("physical-plan.json")
    candidate_validation = load("candidate-validation.json")
    physical_owner_smoke = load("physical-owner-smoke.json")

    if not checkpoint["indexed_weights_complete"]:
        raise RuntimeError("checkpoint shards are incomplete")
    if checkpoint["missing_indexed_weights"]:
        raise RuntimeError("checkpoint reports missing indexed weights")
    if checkpoint["load_format"] != "auto":
        raise RuntimeError("checkpoint was not loaded with load_format=auto")

    full_increase = (
        capacity["orbitkv"]["token_capacity"]
        - capacity["stock"]["token_capacity"]
    )
    if full_increase != capacity["full_token_capacity_increase"]:
        raise RuntimeError("Full token-capacity delta is inconsistent")
    close(
        capacity["full_token_capacity_increase_percent"],
        full_increase / capacity["stock"]["token_capacity"] * 100,
    )
    if not capacity["outputs_equal"]:
        raise RuntimeError("capacity output digests differ")
    if any(capacity["retractions"]["stock"] + capacity["retractions"]["orbitkv"]):
        raise RuntimeError("capacity run retracted a request")

    admission_reductions = []
    for pair in admission["pairs"]:
        if not pair["outputs_equal"]:
            raise RuntimeError("admission output digests differ")
        if any(
            pair["stock_num_retractions"]
            + pair["orbitkv_num_retractions"]
        ):
            raise RuntimeError("admission run retracted a request")
        admission_reductions.append(
            (1 - pair["orbitkv_seconds"] / pair["stock_seconds"]) * 100
        )
    close(
        admission["median_makespan_reduction_percent"],
        statistics.median(admission_reductions),
    )
    close(
        admission["stock_median_seconds"],
        statistics.median(pair["stock_seconds"] for pair in admission["pairs"]),
    )
    close(
        admission["orbitkv_median_seconds"],
        statistics.median(
            pair["orbitkv_seconds"] for pair in admission["pairs"]
        ),
    )

    overhead_ratios = []
    for pair in overhead["pairs"]:
        if not pair["outputs_equal"]:
            raise RuntimeError("fixed-capacity output digests differ")
        if any(
            pair["stock_num_retractions"]
            + pair["orbitkv_num_retractions"]
        ):
            raise RuntimeError("fixed-capacity run retracted a request")
        overhead_ratios.append(
            pair["orbitkv_seconds"] / pair["stock_seconds"]
        )
    close(
        overhead["median_orbitkv_over_stock_ratio"],
        statistics.median(overhead_ratios),
    )
    close(
        overhead["median_orbitkv_over_stock_percent"],
        (statistics.median(overhead_ratios) - 1) * 100,
    )
    expected_saved_bytes = (
        (
            overhead["stock_swa_token_capacity"]
            - overhead["orbitkv_swa_token_capacity"]
        )
        * 12
        * 2048
    )
    if expected_saved_bytes != overhead["fixed_capacity_kv_bytes_saved"]:
        raise RuntimeError("fixed-capacity byte reduction is inconsistent")
    if expected_saved_bytes != overhead["fixed_capacity_kv_mib_saved"] << 20:
        raise RuntimeError("fixed-capacity MiB reduction is inconsistent")

    certificate_ids = sorted(
        certificate["certificate_id"]
        for certificate in owner_trace["certificates"]
    )
    committed_ids = sorted(owner_trace["committed_certificate_ids"])
    if certificate_ids != committed_ids:
        raise RuntimeError("not every retirement certificate was committed")
    if not owner_trace["all_certificates_committed"]:
        raise RuntimeError("owner trace reports an incomplete commit set")

    if physical_plan["schema"] != "orbitkv.hf-physical-compilation.v1":
        raise RuntimeError("unsupported physical-plan artifact")
    optimized = physical_plan["physical_plan"]
    if optimized["selected_eviction_interval_tokens"] != 32:
        raise RuntimeError("physical optimizer did not select interval 32")
    if not optimized["physical_plan_fingerprint"].startswith("sha256:"):
        raise RuntimeError("physical plan fingerprint is missing")
    if optimized["physical_plan_fingerprint"] != candidate_validation[
        "physical_plan_fingerprint"
    ]:
        raise RuntimeError("candidate validation physical fingerprint differs")
    if optimized["physical_plan_fingerprint"] != physical_owner_smoke[
        "physical_plan_fingerprint"
    ]:
        raise RuntimeError("owner smoke physical fingerprint differs")
    if optimized["plan_fingerprint"] != optimized["selected"]["policy"][
        "plan_fingerprint"
    ]:
        raise RuntimeError("physical plan semantic fingerprint differs")
    if candidate_validation["selected_eviction_interval_tokens"] != 32:
        raise RuntimeError("candidate validation selected the wrong interval")
    expected_capacity = {
        16: (61952, 24464),
        32: (59904, 26512),
        64: (55808, 30608),
        128: (47616, 38800),
    }
    for candidate in candidate_validation["candidates"]:
        interval = candidate["interval"]
        expected_full, expected_swa = expected_capacity[interval]
        if candidate["predicted_full_token_capacity"] != expected_full:
            raise RuntimeError(f"interval {interval} predicted Full capacity differs")
        if candidate["actual_full_token_capacity"] != expected_full:
            raise RuntimeError(f"interval {interval} actual Full capacity differs")
        if candidate["predicted_swa_token_capacity"] != expected_swa:
            raise RuntimeError(f"interval {interval} predicted SWA capacity differs")
        if candidate["actual_swa_token_capacity"] != expected_swa:
            raise RuntimeError(f"interval {interval} actual SWA capacity differs")
        if not candidate["prediction_matches"]:
            raise RuntimeError(f"interval {interval} prediction did not match")
        if any(candidate["num_retractions"]):
            raise RuntimeError(f"interval {interval} retracted a request")
    candidates = {
        candidate["interval"]: candidate
        for candidate in candidate_validation["candidates"]
    }
    if candidates[16]["rejection_reasons"] != [
        "estimated reclamation calls 5 exceed maximum 4"
    ]:
        raise RuntimeError("interval 16 rejection reason differs")
    if candidates[128]["rejection_reasons"] != [
        "admitted requests 7 below minimum 8"
    ]:
        raise RuntimeError("interval 128 rejection reason differs")
    smoke_predicted = physical_owner_smoke["predicted"]
    smoke_actual = physical_owner_smoke["actual_server_memory"]
    if smoke_predicted["full_token_capacity"] != smoke_actual["token_capacity"]:
        raise RuntimeError("physical owner Full capacity contract differs")
    if (
        smoke_predicted["physical_swa_token_slots"]
        != smoke_actual["token_capacity_swa"]
    ):
        raise RuntimeError("physical owner SWA capacity contract differs")
    if physical_owner_smoke["contract_validation"] != "passed":
        raise RuntimeError("physical owner contract did not pass")
    if any(physical_owner_smoke["num_retractions"]):
        raise RuntimeError("physical owner smoke retracted a request")

    positions = {
        mode: [] for mode in ("stock128", "stock32", "policy32", "owner32")
    }
    contribution_ratios = {
        "physical_policy_stock32_over_stock128": [],
        "compiler_policy_policy32_over_stock32": [],
        "ownership_owner32_over_policy32": [],
    }
    owner_over_stock = []
    for round_record in ablation["rounds"]:
        order = round_record["execution_order"]
        runs = round_record["runs"]
        for position, mode in enumerate(order):
            positions[mode].append(position)
        reference = runs["stock128"]
        for mode, run in runs.items():
            if run["output_digest"] != reference["output_digest"]:
                raise RuntimeError(f"{mode} output digest differs in ablation")
            if run["completion_tokens"] != reference["completion_tokens"]:
                raise RuntimeError(f"{mode} completion count differs in ablation")
            if any(run["num_retractions"]):
                raise RuntimeError(f"{mode} retracted a request in ablation")
            expected_capacity = 47616 if mode == "stock128" else 59904
            if run["full_token_capacity"] != expected_capacity:
                raise RuntimeError(f"{mode} Full capacity differs in ablation")
        if runs["stock128"]["owner_transport"] is not None:
            raise RuntimeError("Stock128 unexpectedly used an OrbitKV owner")
        if runs["stock32"]["owner_transport"] is not None:
            raise RuntimeError("Stock32 unexpectedly used an OrbitKV owner")
        contribution_ratios["physical_policy_stock32_over_stock128"].append(
            runs["stock32"]["seconds"] / runs["stock128"]["seconds"]
        )
        contribution_ratios["compiler_policy_policy32_over_stock32"].append(
            runs["policy32"]["seconds"] / runs["stock32"]["seconds"]
        )
        contribution_ratios["ownership_owner32_over_policy32"].append(
            runs["owner32"]["seconds"] / runs["policy32"]["seconds"]
        )
        owner_over_stock.append(
            runs["owner32"]["seconds"] / runs["stock128"]["seconds"]
        )
    if any(sorted(values) != [0, 1, 2, 3] for values in positions.values()):
        raise RuntimeError("four-way ablation execution positions are not balanced")
    for name, values in contribution_ratios.items():
        recorded = ablation["contribution_ratios"][name]
        if len(recorded) != len(values):
            raise RuntimeError(f"{name} sample count differs")
        for actual, expected in zip(recorded, values, strict=True):
            close(actual, expected)
        close(
            ablation["median_contribution_ratios"][name],
            statistics.median(values),
        )
        close(
            ablation["median_contribution_percent"][name],
            (statistics.median(values) - 1) * 100,
        )
    close(
        ablation["median_owner32_over_stock128_ratio"],
        statistics.median(owner_over_stock),
    )
    close(
        ablation["median_owner32_over_stock128_percent"],
        (statistics.median(owner_over_stock) - 1) * 100,
    )

    print(
        "verified real-model records: "
        f"{checkpoint['observed_indexed_weight_bytes']} checkpoint bytes, "
        f"{capacity['full_token_capacity_increase_percent']:.2f}% capacity, "
        f"{-ablation['median_owner32_over_stock128_percent']:.2f}% balanced makespan, "
        f"{ablation['median_contribution_percent']['ownership_owner32_over_policy32']:.2f}% owner overhead, "
        f"{len(candidate_validation['candidates'])}/4 physical candidates matched, "
        f"{overhead['fixed_capacity_kv_mib_saved']} MiB fixed-capacity KV"
    )


if __name__ == "__main__":
    main()
