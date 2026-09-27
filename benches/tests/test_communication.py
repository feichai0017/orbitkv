"""Protect timing boundaries and source ownership in the communication harness."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benches.communication import distribution, manager_environment, sample


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1]])
def test_invalid_measurements_cannot_become_successful_percentiles(values):
    with pytest.raises(ValueError, match="finite, nonnegative, and nonempty"):
        distribution(values)


@pytest.mark.parametrize("operation", ["restore", "query_hit"])
def test_lease_setup_and_idle_are_outside_restore_timer_and_query_release_is_outside_timer(
    monkeypatch, operation
):
    events = []

    def clock():
        events.append("clock")
        return len(events)

    for name in ("thread_time_ns", "process_time_ns", "perf_counter_ns"):
        monkeypatch.setattr(f"benches.communication.time.{name}", clock)
    monkeypatch.setattr("benches.communication.time.sleep", lambda _: events.append("idle"))

    def ready(*args):
        events.append("query")
        return SimpleNamespace(lease=b"held"), 2

    monkeypatch.setattr("benches.communication.query_ready", ready)

    def submit(*args):
        events.append("submit")
        assert args[-1] == [(b"held", [[7]])]
        return "handle"

    def wait(handle, *, timeout):
        events.append("ready")
        return SimpleNamespace(done=True, success=True)

    client = SimpleNamespace(
        start_restore=submit,
        wait_restore=wait,
        release=lambda _: events.append("release"),
    )
    measured = sample(
        client, None, "instance", 0, ["layer"], None, [7], operation, "request", 0.001, 1
    )
    assert measured["query_calls"] == 2
    if operation == "restore":
        assert events[:2] == ["query", "idle"]
        assert events.index("clock") < events.index("submit") < events.index("ready")
        assert "release" not in events
    else:
        assert events[0] == "idle"
        assert events.index("clock") < events.index("query")
        assert events[-1] == "release"


def test_ambiguous_restore_never_releases_its_source_or_returns_a_timing(monkeypatch):
    monkeypatch.setattr(
        "benches.communication.query_ready",
        lambda *args: (SimpleNamespace(lease=b"held"), 1),
    )
    released = []

    def wait(*args, **kwargs):
        raise TimeoutError("GPU completion is unknown")

    client = SimpleNamespace(
        start_restore=lambda *args: "accepted",
        wait_restore=wait,
        release=released.append,
    )
    with pytest.raises(TimeoutError, match="completion is unknown"):
        sample(client, None, "instance", 0, ["layer"], None, [7], "restore", "request", 0, 1)
    assert not released


def test_publish_uses_new_keys_per_sample_and_complete_matching_layer_sets():
    publications = []

    def save(*args):
        publications.append(args[-1])
        return True, ""

    client = SimpleNamespace(save=save)
    for request in ("first", "second"):
        sample(client, None, "instance", 0, ["a", "b"], None, [7, 8], "publish", request, 0, 1)
    first, second = publications
    assert first[0][:2] == ("a", [0, 1])
    assert first[1][:2] == ("b", [0, 1])
    assert first[0][2] == first[1][2]
    assert set(first[0][2]).isdisjoint(second[0][2])


def test_manager_python_imports_follow_the_selected_external_artifact(monkeypatch, tmp_path):
    snapshot = tmp_path / "baseline" / "python"
    monkeypatch.setenv("PYTHONPATH", str(snapshot))
    env = manager_environment(snapshot)
    paths = env["PYTHONPATH"].split(":")
    assert paths[0] == str(snapshot)
    assert paths.count(str(snapshot)) == 1
    assert str(Path(__file__).resolve().parents[2] / "python") not in paths
