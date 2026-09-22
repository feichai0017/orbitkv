"""Integration tests for the local Cache Manager client.

These tests verify the UDS/iceoryx2 client can communicate with
a running Cache Manager instance. The server is automatically started
by the `orbitkv_server` fixture.

Requirements:
- Rust extension built: maturin develop --release
- GPU available for Cache Manager

Run with:
    cd python && pytest -m integration tests/integration/test_cache_manager_client.py -v
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_native_hash_batch_is_an_immutable_snapshot_with_shared_views():
    from orbitkv import BlockHashes

    source = [b"first", b"second", b"third"]
    batch = BlockHashes(source)
    source[0] = b"changed"
    assert batch == BlockHashes([b"first", b"second", b"third"])
    assert batch[1:][:1] == BlockHashes([b"second"])
    assert len(batch[3:]) == 0
    assert len(batch[2:1]) == 0
    assert batch[-2:] == batch[1:]
    with pytest.raises(ValueError, match="unit slice step"):
        batch[::2]


@pytest.mark.parametrize(
    ("case", "hash_count"),
    [
        pytest.param("empty_query", 0, id="empty_query"),
        pytest.param("unknown_hashes", 5, id="unknown_hashes"),
    ],
)
def test_query_prefetch_ready_zero_contract(
    case: str,
    hash_count: int,
    client,
    registered_instance: str,
    block_hashes: list[bytes],
):
    """A fresh server query returns Ready(0), not a dict or miss sentinel."""
    from orbitkv import BlockHashes

    requested_hashes = block_hashes[:hash_count]

    result = client.query_prefetch(
        registered_instance,
        BlockHashes(requested_hashes),
        req_id=f"query-contract-{case}",
    )

    assert result.num_hit_blocks == 0
    assert result.lease == b""
