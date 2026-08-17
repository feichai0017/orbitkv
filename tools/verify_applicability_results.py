from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/applicability-h20-20260817"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"expected {expected}, got {actual}")


def main() -> None:
    applicability = load("applicability.json")
    mistral = load("mistral-e2e.json")
    qwen = load("qwen-noop.json")
    state_plan = load("mistral-state-plan.json")
    multireq = load("mistral-multireq.json")
    multireq_state_plan = load("mistral-state-plan-multireq.json")

    models = {
        model["architecture"]: model for model in applicability["models"]
    }
    expected = {
        "Qwen2ForCausalLM": ("safe_fallback", 0),
        "MistralForCausalLM": ("uniform_bounded", 87_500),
        "GptOssForCausalLM": ("hybrid_lifetimes", 49_780),
    }
    for architecture, (classification, reduction) in expected.items():
        model = models[architecture]
        if model["applicability"] != classification:
            raise RuntimeError(f"{architecture} applicability differs")
        if model["static_reduction_percent_milli"] != reduction:
            raise RuntimeError(f"{architecture} static reduction differs")

    lowering = state_plan["sglang_lowering"]
    if lowering["status"] != "enabled" or lowering["kind"] != "uniform_swa":
        raise RuntimeError("Mistral state plan is not executable")
    contract = lowering["contract"]
    if contract["page_tokens"] != 1 or contract["maximum_running_requests"] != 1:
        raise RuntimeError("Mistral SGLang execution contract differs")
    if contract["plan_fingerprint"] != state_plan["layout"]["plan_fingerprint"]:
        raise RuntimeError("Mistral state-plan fingerprint differs")

    if not mistral["checkpoint"]["indexed_weights_complete"]:
        raise RuntimeError("Mistral checkpoint shards are incomplete")
    if not mistral["digest_match"] or not mistral["checkpoint_match"]:
        raise RuntimeError("Mistral execution and reference differ")
    if any(mistral["execute"]["num_retractions"]):
        raise RuntimeError("Mistral execution retracted a request")
    if any(mistral["kernel_reference"]["num_retractions"]):
        raise RuntimeError("Mistral reference retracted a request")
    expected_saved = (
        mistral["kernel_reference"]["kv_gib"] - mistral["execute"]["kv_gib"]
    )
    close(mistral["kv_gib_saved"], expected_saved)
    close(
        mistral["kv_reduction_percent"],
        (1 - mistral["execute"]["kv_gib"] / mistral["kernel_reference"]["kv_gib"])
        * 100,
    )
    close(
        mistral["warm_execute_over_reference_ratio"],
        mistral["execute"]["iteration_seconds"]
        / mistral["kernel_reference"]["iteration_seconds"],
    )
    if (
        mistral["workload"]["prompt_tokens"]
        <= mistral["execute"]["physical_kv_token_slots"]
    ):
        raise RuntimeError("Mistral workload does not exceed physical KV slots")
    if mistral["stock_same_pool_long_prompt"]["status"] != "rejected":
        raise RuntimeError("Stock long-prompt boundary differs")

    if not qwen["checkpoint"]["indexed_weights_complete"]:
        raise RuntimeError("Qwen checkpoint shards are incomplete")
    if not qwen["digest_match"] or not qwen["checkpoint_match"]:
        raise RuntimeError("Qwen Stock and shadow differ")
    if not qwen["capacity_match"]:
        raise RuntimeError("Qwen shadow changed Full-attention capacity")
    close(
        qwen["shadow_over_stock_ratio"],
        qwen["shadow"]["iteration_seconds"] / qwen["stock"]["iteration_seconds"],
    )

    contract = multireq["compiled_contract"]
    expected_per_request = (
        contract["window_tokens"]
        + contract["eviction_interval_tokens"]
        + contract["page_tokens"]
        + contract["decode_headroom_tokens"]
    )
    if contract["per_request_resident_tokens"] != expected_per_request:
        raise RuntimeError("multi-request per-request budget differs")
    expected_staging = (
        contract["chunked_prefill_tokens"] + contract["page_tokens"]
    )
    if contract["global_staging_tokens"] != expected_staging:
        raise RuntimeError("multi-request staging budget differs")
    expected_minimum = (
        expected_per_request * contract["maximum_running_requests"]
        + expected_staging
    )
    if contract["minimum_pool_tokens"] != expected_minimum:
        raise RuntimeError("multi-request minimum pool differs")
    state_contract = multireq_state_plan["sglang_lowering"]["contract"]
    if state_contract["minimum_pool_tokens"] != expected_minimum:
        raise RuntimeError("multi-request state plan minimum differs")
    if not state_contract["contract_fingerprint"].startswith("sha256:"):
        raise RuntimeError("multi-request contract fingerprint is missing")
    if (
        state_contract["plan_fingerprint"]
        != multireq_state_plan["layout"]["plan_fingerprint"]
    ):
        raise RuntimeError("multi-request plan fingerprint differs")
    close(
        multireq["slot_reduction_percent"],
        (1 - expected_minimum / multireq["reference_pool_tokens"]) * 100,
    )
    ratios = []
    for pair in multireq["pairs"]:
        if pair["execute_token_slots"] != expected_minimum:
            raise RuntimeError("multi-request execute pool differs")
        if pair["reference_token_slots"] != multireq["reference_pool_tokens"]:
            raise RuntimeError("multi-request reference pool differs")
        if any(pair["execute_retractions"] + pair["reference_retractions"]):
            raise RuntimeError("multi-request run retracted a request")
        if pair["execute_output_digest"] != pair["reference_output_digest"]:
            raise RuntimeError("multi-request output digest differs")
        if (
            pair["execute_config_sha256"]
            != pair["reference_config_sha256"]
            or pair["execute_config_sha256"] != multireq["checkpoint"]["config_sha256"]
        ):
            raise RuntimeError("multi-request checkpoint config differs")
        if pair["execute_completion_tokens"] != pair["reference_completion_tokens"]:
            raise RuntimeError("multi-request completion count differs")
        if pair["execute_completion_tokens"] != 128:
            raise RuntimeError("multi-request completion count differs")
        ratio = pair["execute_seconds"] / pair["reference_seconds"]
        close(pair["execute_over_reference_ratio"], ratio)
        ratios.append(ratio)
    close(
        multireq["median_execute_over_reference_ratio"],
        statistics.median(ratios),
    )
    close(
        multireq["median_execute_over_reference_percent"],
        (statistics.median(ratios) - 1) * 100,
    )
    if not multireq["checkpoint"]["indexed_weights_complete"]:
        raise RuntimeError("multi-request checkpoint shards are incomplete")
    if multireq["below_minimum"]["pool_tokens"] != expected_minimum - 1:
        raise RuntimeError("below-minimum boundary differs")
    if multireq["below_minimum"]["status"] != "rejected_at_startup":
        raise RuntimeError("below-minimum plan did not fail closed")
    print(
        "verified applicability records: "
        "Qwen safe fallback, Mistral single and multi-request bounded execution, "
        "GPT-OSS hybrid plan"
    )


if __name__ == "__main__":
    main()
