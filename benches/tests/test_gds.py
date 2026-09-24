import pytest

from benches.gds import native_io_stats


def stats(read=2, write=3, posix=0, extra=""):
    return (
        f"GPU 0 Read: bw=1 n={read} posix={posix} unalign=0 r_sparse=0 err=0 MiB=8 {extra} "
        f"Write: bw=1 n={write} posix=0 unalign=0 err=0 MiB=8 BufRegister: n=1 err=0\n"
        "GPU 1 Read: bw=0 n=0 posix=0 unalign=0 err=0 MiB=0 "
        "Write: bw=0 n=0 posix=0 unalign=0 err=0 MiB=0 BufRegister: n=0 err=0\n"
    )


def test_native_evidence_requires_both_directions_and_rejects_unknown_or_fallback_counters():
    assert native_io_stats(stats())["operations"] == {"read": 2, "write": 3}
    for output in (
        "no statistics",
        stats(read=0),
        stats(write=0),
        stats(posix=1),
        stats(extra="err=1"),
        stats(extra="r_sparse=1"),
        stats(extra="r_inline=1"),
        stats().replace("posix=0", "unknown=0"),
    ):
        with pytest.raises(ValueError):
            native_io_stats(output)
