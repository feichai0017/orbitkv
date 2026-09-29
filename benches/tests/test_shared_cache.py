import pytest

from benches.shared_cache import drain, qualify, verify_restore, verify_source_ssd


@pytest.mark.parametrize("remote,restored,text", [(0, 64, "ok"), (64, 0, "ok"), (64, 64, "wrong")])
def test_shared_cache_gate_requires_payload_gpu_copy_and_output(remote, restored, text):
    with pytest.raises(AssertionError):
        verify_restore(
            {},
            {
                "orbitkv_remote_fetch_bytes_total": remote,
                "orbitkv_load_bytes_total": restored,
                "orbitkv_remote_stage_duration_seconds_count_release": 1,
            },
            "ok",
            {"text": text},
        )


def test_shared_cache_gate_uses_counter_deltas():
    before = {"orbitkv_remote_fetch_bytes_total": 1024, "orbitkv_load_bytes_total": 2048}
    after = {
        "orbitkv_remote_fetch_bytes_total": 1088,
        "orbitkv_load_bytes_total": 2112,
        "orbitkv_remote_stage_duration_seconds_count_release": 1,
    }
    result = verify_restore(before, after, "ok", {"text": "ok", "ttft_ms": 1, "e2e_ms": 2})
    assert result["remote_bytes"] == result["h2d_bytes"] == 64
    without_ack = {**after, "orbitkv_remote_stage_duration_seconds_count_release": 0}
    with pytest.raises(AssertionError, match="completion evidence"):
        verify_restore(before, without_ack, "ok", {"text": "ok"})
    with pytest.raises(AssertionError):
        verify_restore(before, before, "ok", {"text": "ok"})


def test_source_ssd_gate_requires_successful_physical_reads():
    before = {
        "orbitkv_ssd_prefetch_bytes_total": 1024,
        "orbitkv_ssd_prefetch_success_total": 3,
    }
    after = {
        "orbitkv_ssd_prefetch_bytes_total": 5120,
        "orbitkv_ssd_prefetch_success_total": 5,
    }
    assert verify_source_ssd(before, after) == {
        "source_ssd_read_bytes": 4096,
        "source_ssd_reads": 2,
    }
    for missing in [
        {**after, "orbitkv_ssd_prefetch_bytes_total": 1024},
        {**after, "orbitkv_ssd_prefetch_success_total": 3},
    ]:
        with pytest.raises(AssertionError, match="SSD materialization"):
            verify_source_ssd(before, missing)


def test_forced_source_ssd_qualification_evicts_dram_and_checks_both_nodes(monkeypatch):
    source_empty = {"orbitkv_save_bytes_total": 0, "orbitkv_ssd_write_bytes_total": 0}
    source_saved = {"orbitkv_save_bytes_total": 4096, "orbitkv_ssd_write_bytes_total": 4096}
    source_before_read = {
        **source_saved,
        "orbitkv_ssd_prefetch_bytes_total": 0,
        "orbitkv_ssd_prefetch_success_total": 0,
    }
    source_after_read = {
        **source_saved,
        "orbitkv_ssd_prefetch_bytes_total": 4096,
        "orbitkv_ssd_prefetch_success_total": 1,
    }
    target_before = {
        "orbitkv_remote_fetch_bytes_total": 0,
        "orbitkv_load_bytes_total": 0,
        "orbitkv_remote_stage_duration_seconds_count_release": 0,
    }
    target_after = {
        "orbitkv_remote_fetch_bytes_total": 4096,
        "orbitkv_load_bytes_total": 4096,
        "orbitkv_remote_stage_duration_seconds_count_release": 1,
    }
    snapshots = iter(
        [
            [source_empty, {}],
            [source_saved, target_before],
            [source_before_read, target_before],
            [source_after_read, target_after],
        ]
    )
    monkeypatch.setattr("benches.shared_cache.drain", lambda *_: next(snapshots))
    generated = iter(
        [
            {"text": "same", "ttft_ms": 2, "e2e_ms": 3},
            {"text": "same", "ttft_ms": 1, "e2e_ms": 2},
        ]
    )
    monkeypatch.setattr("benches.shared_cache.generate", lambda *_: next(generated))
    syncs = []
    monkeypatch.setattr("benches.shared_cache.synchronize", lambda url: syncs.append(url))
    cleanup = {"cleanup": {"evicted_blocks": 1, "still_referenced_blocks": 0}}
    monkeypatch.setattr("benches.shared_cache.evict_host_cache", lambda _: cleanup)

    result = qualify(
        engine="vllm",
        model="model",
        source_url="http://source-engine",
        target_url="http://target-engine",
        source_manager="http://source-manager",
        target_manager="http://target-manager",
        prompts=[[1] * 128],
        source_medium="ssd",
    )

    assert syncs == ["http://source-manager", "http://source-manager"]
    assert result[0]["source_medium"] == "ssd"
    assert result[0]["source_preparation"] == cleanup
    assert result[0]["source_ssd_read_bytes"] == 4096
    assert result[0]["remote_bytes"] == result[0]["h2d_bytes"] == 4096


@pytest.mark.parametrize(
    "metric",
    [
        "orbitkv_transfer_reserved_bytes",
        "orbitkv_transfer_completion_outstanding",
        "orbitkv_ssd_read_pinned_bytes",
    ],
)
def test_finished_requests_do_not_hide_unreleased_transfer_resources(monkeypatch, metric):
    monkeypatch.setattr("benches.shared_cache.metrics", lambda _: {metric: 1})
    with pytest.raises(TimeoutError, match=metric):
        drain("source", "consumer", timeout=0)
