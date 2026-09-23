import pytest

from benches.shared_cache import drain, verify_restore


@pytest.mark.parametrize("remote,restored,text", [(0, 64, "ok"), (64, 0, "ok"), (64, 64, "wrong")])
def test_shared_cache_gate_requires_payload_gpu_copy_and_output(remote, restored, text):
    with pytest.raises(AssertionError):
        verify_restore(
            {},
            {"orbitkv_remote_fetch_bytes_total": remote, "orbitkv_load_bytes_total": restored},
            "ok",
            {"text": text},
        )


def test_shared_cache_gate_uses_counter_deltas():
    before = {"orbitkv_remote_fetch_bytes_total": 1024, "orbitkv_load_bytes_total": 2048}
    after = {"orbitkv_remote_fetch_bytes_total": 1088, "orbitkv_load_bytes_total": 2112}
    result = verify_restore(before, after, "ok", {"text": "ok", "ttft_ms": 1, "e2e_ms": 2})
    assert result["remote_bytes"] == result["h2d_bytes"] == 64
    with pytest.raises(AssertionError):
        verify_restore(before, before, "ok", {"text": "ok"})


def test_finished_requests_do_not_hide_unreleased_source_allocations(monkeypatch):
    monkeypatch.setattr(
        "benches.shared_cache.metrics", lambda _: {"orbitkv_transfer_reserved_bytes": 4096}
    )
    with pytest.raises(TimeoutError, match="orbitkv_transfer_reserved_bytes"):
        drain("source", "consumer", timeout=0)
