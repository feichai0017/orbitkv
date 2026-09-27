"""Protect timing boundaries and source ownership in the communication harness."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benches.communication import arguments, distribution, manager_environment, sample


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1]])
def test_invalid_measurements_cannot_become_successful_percentiles(values):
    with pytest.raises(ValueError, match="finite, nonnegative, and nonempty"):
        distribution(values)


@pytest.mark.parametrize(
    ("operation", "batch_size"), [("restore", 1), ("restore", 4), ("query_hit", 1)]
)
def test_lease_setup_and_idle_are_outside_restore_timer_and_query_release_is_outside_timer(
    monkeypatch, operation, batch_size
):
    events = []
    targets = list(range(7, 7 + 2 * batch_size))
    restore_batches = [
        (object(), targets[start : start + 2]) for start in range(0, len(targets), 2)
    ]
    acquired = []

    def clock():
        events.append("clock")
        return len(events)

    for name in ("thread_time_ns", "process_time_ns", "perf_counter_ns"):
        monkeypatch.setattr(f"benches.communication.time.{name}", clock)
    monkeypatch.setattr("benches.communication.time.sleep", lambda _: events.append("idle"))

    def ready(*args):
        events.append("query")
        if operation == "restore":
            index = len(acquired)
            assert args[3] is restore_batches[index][0]
            assert args[4] == f"request-lease-{index}"
            assert args[5] == 2
        lease = bytes([len(acquired) + 1])
        acquired.append(lease)
        return SimpleNamespace(lease=lease), 2

    monkeypatch.setattr("benches.communication.query_ready", ready)

    def submit(*args):
        events.append("submit")
        assert args[-1] == [
            (lease, [batch_targets])
            for lease, (_, batch_targets) in zip(acquired, restore_batches, strict=True)
        ]
        assert [target for _, groups in args[-1] for target in groups[0]] == targets
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
        client,
        None,
        "instance",
        0,
        ["layer"],
        None,
        targets,
        restore_batches,
        operation,
        "request",
        0.001,
        1,
    )
    assert measured["query_calls"] == 2 * batch_size
    if operation == "restore":
        assert events[: batch_size + 1] == ["query"] * batch_size + ["idle"]
        assert events.index("clock") < events.index("submit") < events.index("ready")
        assert events.count("submit") == 1
        assert "release" not in events
    else:
        assert events[0] == "idle"
        assert events.index("clock") < events.index("query")
        assert events[-1] == "release"


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_ambiguous_restore_never_releases_its_source_or_returns_a_timing(monkeypatch, failure):
    monkeypatch.setattr(
        "benches.communication.query_ready",
        lambda *args: (SimpleNamespace(lease=b"held"), 1),
    )
    released = []

    def wait(*args, **kwargs):
        raise TimeoutError("GPU completion is unknown")

    client = SimpleNamespace(
        start_restore=wait if failure == "submit" else lambda *args: "accepted",
        wait_restore=wait,
        release=released.append,
    )
    with pytest.raises(TimeoutError, match="completion is unknown"):
        sample(
            client,
            None,
            "instance",
            0,
            ["layer"],
            None,
            [7, 8],
            [(None, [7]), (None, [8])],
            "restore",
            "request",
            0,
            1,
        )
    assert not released


def test_failed_batch_acquisition_releases_prior_leases_before_any_restore(monkeypatch):
    acquired = []
    released = []

    def ready(*args):
        if len(acquired) == 2:
            raise TimeoutError("third query failed")
        lease = bytes([len(acquired) + 1])
        acquired.append(lease)
        return SimpleNamespace(lease=lease), 1

    def release(lease):
        released.append(lease)
        if len(released) == 1:
            raise OSError("first cleanup failed")

    monkeypatch.setattr("benches.communication.query_ready", ready)
    client = SimpleNamespace(release=release)
    with pytest.raises(TimeoutError, match="third query failed"):
        sample(
            client,
            None,
            "instance",
            0,
            ["layer"],
            None,
            [7, 8, 9],
            [(None, [7]), (None, [8]), (None, [9])],
            "restore",
            "request",
            0,
            1,
        )
    assert released == acquired


@pytest.mark.parametrize(
    ("batch_size", "payloads", "valid"),
    [
        (1, [4096, 262144], True),
        (4, [16384, 262144], True),
        (0, [4096], False),
        (-1, [4096], False),
        (2, [4096], False),
        (3, [16384], False),
        (4, [16384, 20480], False),
    ],
)
def test_restore_batch_requires_equal_nonempty_parts_of_every_payload(batch_size, payloads, valid):
    argv = [
        "--manager",
        "/manager",
        "--label",
        "batch",
        "--output",
        "/results",
        "--restore-batch-size",
        str(batch_size),
        "--payload-bytes",
        *map(str, payloads),
    ]
    if valid:
        assert arguments(argv).restore_batch_size == batch_size
    else:
        with pytest.raises(SystemExit):
            arguments(argv)


def test_restore_batch_defaults_to_one():
    assert (
        arguments(
            ["--manager", "/manager", "--label", "batch", "--output", "/results"]
        ).restore_batch_size
        == 1
    )


def test_publish_uses_new_keys_per_sample_and_complete_matching_layer_sets():
    publications = []

    def save(*args):
        publications.append(args[-1])
        return True, ""

    client = SimpleNamespace(save=save)
    for request in ("first", "second"):
        sample(
            client,
            None,
            "instance",
            0,
            ["a", "b"],
            None,
            [7, 8],
            [],
            "publish",
            request,
            0,
            1,
        )
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
