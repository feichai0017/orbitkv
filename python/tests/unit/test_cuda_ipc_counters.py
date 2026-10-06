"""Counter diagnostics must distinguish MapInfo from actual exported credits."""

import struct

import pytest

from tests.support.trace_sglang_ipc import read_counter


@pytest.mark.parametrize("live_offset", [None, 0, 71], ids=["drained", "first-live", "last-live"])
def test_counter_slots_exclude_file_header(tmp_path, live_offset):
    path = tmp_path / "torch-counter"
    counts = [int(offset == live_offset) for offset in range(72)]
    path.write_bytes(struct.pack("=i", 1) + bytes(60) + struct.pack("=72q", *counts))

    assert [read_counter(path, offset) for offset in range(72)] == counts


@pytest.mark.parametrize("offset", [-1, 0, 1], ids=["negative", "truncated-first", "missing-next"])
def test_invalid_counter_slot_is_not_a_zero(tmp_path, offset):
    path = tmp_path / "torch-counter"
    path.write_bytes(struct.pack("=i", 1) + bytes(60) + bytes(7))

    with pytest.raises(ValueError):
        read_counter(path, offset)
