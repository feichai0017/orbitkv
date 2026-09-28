"""Exec-process owner used by the local Restore crash integration gate."""

import sys

import torch

from orbitkv import CacheManagerClient
from tests.support.cache_manager import ClientContext


def main():
    endpoint, namespace = sys.argv[1:]
    client = CacheManagerClient(endpoint)
    client.start_session_watcher("crashing-engine", namespace, 1, 1)
    context = ClientContext(client, "crashing-engine", namespace, num_blocks=4)
    context.register_kv_caches()
    ready = context.query([b"engine-crash-source"])
    # The parent publishes and seals this hash before starting this process.
    assert ready.num_hit_blocks == 1
    handle = client.start_restore(
        context.instance_id,
        0,
        0,
        [context._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    client.wait_restore(handle, timeout=60)
    raise AssertionError("parent should kill this process while its grant is Active")


if __name__ == "__main__":
    main()
