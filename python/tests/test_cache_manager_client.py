"""Integration tests for the local Cache Manager client.

These tests verify the UDS/iceoryx2 client can communicate with
a running Cache Manager instance. The server is automatically started
by the `orbitkv_server` fixture.

Requirements:
- Rust extension built: maturin develop --release
- GPU available for Cache Manager

Run with:
    cd python && pytest -m integration tests/test_cache_manager_client.py -v
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


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
    engine_client,
    registered_instance: str,
    block_hashes: list[bytes],
):
    """A fresh server query returns Ready(0), not a dict or miss sentinel."""
    requested_hashes = block_hashes[:hash_count]

    result = engine_client.query_prefetch(
        registered_instance,
        requested_hashes,
        req_id=f"query-contract-{case}",
    )

    assert result.num_hit_blocks == 0
    assert result.lease == b""
