"""Protect cache-source evidence and incomplete-run rejection in comparisons."""

import json
from types import SimpleNamespace

import pytest

from benchs.metrics import cache_source, metrics
from benchs.report import collect_run


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize(
    ("device", "external", "bytes_loaded", "expected"),
    [
        (0, 0, 0, "miss"),
        (1, 0, 0, "hbm"),
        (0, 0, 512, "external"),
        (1, 0, 512, "mixed"),
        (0, 1, 0, "external"),
        (1, 1, 0, "mixed"),
    ],
)
def test_source_requires_evidence(engine, device, external, bytes_loaded, expected):
    sample = {
        "usage": {"cached_tokens_details": {"device": device, "host": external}},
        "metrics_delta": {
            "vllm:prefix_cache_hits_total": device,
            "vllm:external_prefix_cache_hits_total": external,
        },
        "manager_delta": {"orbitkv_load_bytes_total": bytes_loaded},
    }
    assert cache_source(engine, sample) == expected
    if expected == "miss":
        sample["usage"].update(cached_tokens=1, prompt_tokens_details={"cached_tokens": 1})
        assert cache_source(engine, sample) == "unverified"


def test_unused_nan_metrics_do_not_poison_json(monkeypatch):
    monkeypatch.setattr(
        "benchs.metrics.requests.get",
        lambda *a, **kw: SimpleNamespace(
            text='unused NaN\ncounter{worker="a"} 10\ncounter{worker="b"} 5\nidle +Inf\n',
            raise_for_status=lambda: None,
        ),
    )
    observed = metrics("http://localhost")
    assert observed == {"counter": 15}
    json.dumps(observed, allow_nan=False)


def test_report_keeps_output_mismatches_and_rejects_partial_runs(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": {"engine": "vllm", "lengths": [64], "repeats": 1},
            }
        )
    )
    samples = [
        {
            "length": 64,
            "repeat": 0,
            "phase": phase,
            "ttft_ms": 5,
            "e2e_ms": 10,
            "matches_cold_output": phase == "cold",
            "usage": {},
            "metrics_delta": {},
            "manager_delta": {},
        }
        for phase in ("cold", "hbm_hit", "after_pressure")
    ]
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text("\n".join(map(json.dumps, samples)))
    assert sum(s["output_mismatches"] for s in collect_run(tmp_path)["summary"]) == 2
    for invalid in (samples[:2], [*samples, samples[0]]):
        sample_file.write_text("\n".join(map(json.dumps, invalid)))
        with pytest.raises(ValueError, match="Incomplete or duplicated"):
            collect_run(tmp_path)
