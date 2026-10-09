"""Installed query-target export over an authenticated local Manager channel.

Trigger: registered-query protocol or native binding changes; requires CUDA and
an installed wheel. Source-only collection does not import Torch or the extension.
"""

import os
import sys
import time

import pytest

from tests.support.cache_manager import ClientContext, find_available_port
from tests.support.installed_serving import manager_command, service, wait_for_drain

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_installed_query_target_export_preserves_registration_and_channel(enabled, tmp_path):
    from orbitkv import BlockHashes, CacheManagerClient, QueryLoading, QueryReady

    port, http_port = find_available_port(), find_available_port()
    command = manager_command(sys.executable, port, http_port, "dram", tmp_path)
    if enabled:
        command.append("--enable-query-control")
    environment = dict(os.environ)
    with service(command, f"http://127.0.0.1:{http_port}", environment, tmp_path, "manager"):
        client = CacheManagerClient(f"/tmp/orbitkv-{port}.sock")
        context = ClientContext(
            client, "registered", "query-control-binding", num_blocks=4, num_heads=1, head_size=16
        )
        try:
            client.start_session_watcher("registered", "query-control-binding", 1, 1)
            context.register_kv_caches()
            if enabled:
                target = client.export_query_target("registered", "query-control-binding", 1, 1)
                assert isinstance(target, bytes) and 0 < len(target) <= 64 * 1024
                assert (
                    client.export_query_target("registered", "query-control-binding", 1, 1)
                    == target
                )
                for namespace, tp_size, world_size in [
                    ("wrong", 1, 1),
                    ("query-control-binding", 2, 1),
                    ("query-control-binding", 1, 2),
                ]:
                    with pytest.raises(ValueError, match="registration differs"):
                        client.export_query_target("registered", namespace, tp_size, world_size)
                    assert client.health()[0]
                context.unregister_context()
                with pytest.raises(ValueError):
                    client.export_query_target("registered", "query-control-binding", 1, 1)
                assert client.health()[0]
                context.register_kv_caches()
                assert (
                    client.export_query_target("registered", "query-control-binding", 1, 1)
                    != target
                )
            else:
                with pytest.raises(ValueError, match="query control is not enabled"):
                    client.export_query_target("registered", "query-control-binding", 1, 1)
            assert client.health()[0]
            hashes = BlockHashes([bytes([19]) * 32])
            deadline = time.monotonic() + 10
            while True:
                result = client.query_prefetch("registered", hashes, req_id="local-control")
                if not isinstance(result, QueryLoading):
                    assert isinstance(result, QueryReady)
                    assert result.num_hit_blocks == 0 and not result.lease
                    break
                assert time.monotonic() < deadline, "registered local query did not drain"
                time.sleep(0.001)
        finally:
            context.unregister_context()
            client.close()
        wait_for_drain(http_port)
