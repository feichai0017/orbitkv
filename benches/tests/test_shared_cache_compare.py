"""Prevent incomplete, mismatched or unconsumed settings from qualifying a comparison."""

import hashlib
import json
import math

import pytest

from benches.shared_cache_compare import summarize


def comparison(tmp_path):
    design = {
        "pairs": 3,
        "warmup": 1,
        "measured": 20,
        "prompt_tokens": [1025],
        "block_tokens": 64,
        "block_bytes": 9 * 1024**2,
        "engines": ["vllm"],
        "bootstrap_seed": 42,
        "bootstrap_draws": 100,
    }
    contract = {
        "design": design,
        "guards": {
            "ttft_ratio": 0.95,
            "ttft_ci_upper": 1,
            "maximum_pair_ratio": 1.05,
        },
        "cells": [],
    }
    campaign = {"status": "PASS_COMPONENT_COLLECTION", "cells": []}
    for pair in range(3):
        variants = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        for order, variant in enumerate(variants):
            name = f"{pair}-{variant}"
            batch = 32 if variant == "baseline" else 128
            cell = {
                "name": name,
                "engine": "vllm",
                "pair": pair,
                "variant": variant,
                "batch_mib": batch,
                "order_index": pair * 2 + order,
            }
            contract["cells"].append(cell)
            calls = math.ceil(16 / (batch // 9))
            rows = [
                {
                    "prompt": 0,
                    "repeat": repeat,
                    "status": "PASS",
                    "output_match": True,
                    "sample_kind": "warmup" if repeat == 0 else "measured",
                    "remote_bytes": 144 * 1024**2,
                    "h2d_bytes": 144 * 1024**2,
                    "source_ssd_read_bytes": 144 * 1024**2,
                    "source_response": {
                        "text": "same",
                        "usage": {"prompt_tokens": 1025, "completion_tokens": 8},
                    },
                    "ttft_ms": 8 if variant == "candidate" else 10,
                    "e2e_ms": 12,
                    "remote_stages": {
                        "authorization": {"calls": calls, "total_ms": 2},
                        "read": {"calls": calls, "total_ms": 1},
                    },
                }
                for repeat in range(21)
            ]
            profile = {
                "engine": "vllm",
                "model": "frozen-model",
                "prompt_token_ids_sha256": "frozen-token-ids",
                "bytes_per_token": design["block_bytes"] // 64,
                "block_tokens": 64,
                "source_medium": "ssd",
                "status": "VALID_DESCRIPTIVE_PROFILE",
                "samples": rows,
                "warmup_repeats_per_prompt": 1,
                "measured_repeats_per_prompt": 20,
            }
            path = tmp_path / f"{name}-profile.json"
            path.write_text(json.dumps(profile))
            campaign["cells"].append(
                {
                    "name": name,
                    "spec": cell,
                    "status": "PASS_COMPONENT_CELL",
                    "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (tmp_path / "CAMPAIGN.json").write_text(json.dumps(campaign))
    return contract, campaign


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "invalid_cell",
        "duplicate_cell",
        "changed_profile",
        "missing_sample",
        "warmup",
        "partial_bytes",
        "batch_not_used",
        "output",
        "order",
    ],
)
def test_only_complete_matching_pairs_establish_component_performance(tmp_path, failure):
    contract, campaign = comparison(tmp_path)
    run = campaign["cells"][0]
    path = tmp_path / f"{run['name']}-profile.json"
    profile = json.loads(path.read_text())
    if failure == "invalid_cell":
        run["status"] = "INVALID_FAIL_STOP"
    elif failure == "duplicate_cell":
        campaign["cells"].append(run)
    elif failure == "missing_sample":
        profile["samples"].pop()
    elif failure == "warmup":
        profile["samples"][0]["sample_kind"] = "measured"
    elif failure == "partial_bytes":
        profile["samples"][1]["remote_bytes"] //= 2
    elif failure == "batch_not_used":
        profile["samples"][1]["remote_stages"]["authorization"]["calls"] = 1
    elif failure == "output":
        for row in profile["samples"]:
            row["source_response"]["text"] = "different"
    elif failure == "order":
        contract["cells"][0]["order_index"] = 1
        contract["cells"][1]["order_index"] = 0
    if failure and failure not in {"invalid_cell", "duplicate_cell", "order"}:
        path.write_text(json.dumps(profile) + "\n")
        if failure != "changed_profile":
            run["profile_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "CAMPAIGN.json").write_text(json.dumps(campaign))
    if failure:
        with pytest.raises(ValueError):
            summarize(tmp_path, contract)
    else:
        result = summarize(tmp_path, contract)
        assert result["state"] == "VALID_COMPONENT_PASS"
        assert result["summary"][0]["metrics"]["ttft_ms"]["bootstrap_95_ci"] == [0.8, 0.8]
        assert not result["production_qualified"] and not result["p99_qualified"]


def test_valid_regression_is_retained_as_component_failure(tmp_path):
    contract, campaign = comparison(tmp_path)
    for run in campaign["cells"]:
        if run["spec"]["variant"] == "candidate":
            path = tmp_path / f"{run['name']}-profile.json"
            profile = json.loads(path.read_text())
            for row in profile["samples"]:
                row["ttft_ms"] = 11
            path.write_text(json.dumps(profile))
            run["profile_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "CAMPAIGN.json").write_text(json.dumps(campaign))
    assert summarize(tmp_path, contract)["state"] == "VALID_COMPONENT_FAIL"


@pytest.mark.parametrize("changed_count", [None, "prompt_tokens", "completion_tokens"])
def test_sglang_request_metadata_is_preserved_but_only_native_counts_must_match(
    tmp_path, changed_count
):
    contract, campaign = comparison(tmp_path)
    contract["design"]["engines"] = ["sglang"]
    for index, run in enumerate(campaign["cells"]):
        run["spec"]["engine"] = "sglang"
        path = tmp_path / f"{run['name']}-profile.json"
        profile = json.loads(path.read_text())
        profile["engine"] = "sglang"
        for row in profile["samples"]:
            usage = row["source_response"]["usage"]
            usage.update(
                id=f"request-{index}",
                finish_reason={"type": "length", "length": 8},
                request_received_ts=1000 + index,
                forward_entry_time=1000.1 + index,
                prefill_finished_time=1000.2 + index,
                queue_time=0.01 * index,
                e2e_latency=0.2 + index,
                first_token_latency=0.1 + index,
                decode_throughput=10 + index,
            )
            if changed_count and index == 0:
                usage[changed_count] += 1
        path.write_text(json.dumps(profile))
        run["profile_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "CAMPAIGN.json").write_text(json.dumps(campaign))
    if changed_count:
        with pytest.raises(ValueError):
            summarize(tmp_path, contract)
    else:
        result = summarize(tmp_path, contract)
        assert result["state"] == "VALID_COMPONENT_PASS"
        pair = result["summary"][0]["pairs"][0]
        assert pair["baseline"]["output"]["usage"]["id"] == "request-0"
        assert pair["candidate"]["output"]["usage"]["id"] == "request-1"
        assert not result["production_qualified"]
