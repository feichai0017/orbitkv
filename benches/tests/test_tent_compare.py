"""Reject prelaunch failures without authorizing an endpoint or performance result."""

import json

import pytest

from benches import tent_compare


@pytest.mark.parametrize("failure", ["runtime_changed", "probe_missing"])
def test_prelaunch_failure_never_enters_native_cell_or_qualifies(tmp_path, monkeypatch, failure):
    contract = {
        "hosts": [{"alias": "test-host"}],
        "libraries": {"baseline": {}},
        "design": {},
        "guards": {},
        "remote_probe": "/frozen/probe.py",
        "input_hashes": {"probe": "expected"},
    }
    monkeypatch.setattr(
        tent_compare,
        "snapshot",
        lambda *_: {
            "gpu": "",
            "libraries": {} if failure == "runtime_changed" else {"baseline": {}},
        },
    )

    def missing(*_):
        raise RuntimeError("probe missing")

    monkeypatch.setattr(tent_compare, "ssh", missing)
    with pytest.raises(RuntimeError):
        tent_compare.run(contract, tmp_path)
    result = json.loads((tmp_path / "RESULT.json").read_text())
    assert result["state"] == "INVALID_PRELAUNCH"
    assert not result["cell_process_started"] and not result["independent_acceptance"]
    assert result["cells"] == 0 and result["held_owners"] == []
    assert "summary" not in result and "production_qualified" not in result
