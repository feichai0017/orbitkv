"""Same-host multi-Manager real-Publish inventory capacity qualification."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import requests

from tests.integration.test_distributed_cache import (
    _await_fence,
    _cleanup_dram,
    _discover_storage_namespaces,
    _etcd_keys,
    _metadata,
    _payload,
    _restore,
    _sync,
    _until,
    _wait_for_remote_drain,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import etcd_server
from tests.support.metrics import fetch_orbitkv_metrics

from .artifacts import external_path
from .scoped_metadata import _etcd_revision, _process_sample, _summary


def run(
    output: Path,
    owners: int,
    duration_seconds: int,
    index_budget: str,
    expect_degraded: bool,
    seed: str,
):
    import torch

    import orbitkv.orbitkv as native
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    assert torch.cuda.is_available()
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes = 8, 4096
    payload_bytes = pages * block_bytes
    identity = f"s2.10:capacity:{owners}:{seed}"
    (storage_namespace,) = _discover_storage_namespaces(output, [identity], pages, block_bytes)
    cluster = f"s210-capacity-{owners}-{uuid.uuid4().hex[:8]}"
    result = {
        "owners": owners,
        "seed": seed,
        "duration_seconds": duration_seconds,
        "index_budget": index_budget,
        "expect_degraded": expect_degraded,
        "storage_namespace": storage_namespace,
        "cluster": cluster,
        "artifacts": {
            name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for name, path in {
                "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
                "etcd": os.environ["ETCD_BIN"],
                "extension": native.__file__,
                "tent": str(Path(os.environ["ORBITKV_MOONCAKE_LIB_DIR"]) / "libtent_shared.so"),
            }.items()
        },
    }

    raw = (output / "samples.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(output))
        managers = {}
        clients = {}
        tensors = {}

        def start_manager(node):
            port = find_available_port()
            manager = CacheManagerProcess(
                port,
                pool_size="32mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s210-capacity-{port}.sock",
                query_budget="16mb",
                log_path=output / f"{node}-manager.log",
                extra_args=(
                    "--etcd-endpoints",
                    endpoint,
                    "--node-id",
                    node,
                    "--cluster-name",
                    cluster,
                    "--membership-ttl-secs",
                    "120",
                    "--inventory-journal-bytes",
                    str(256 * 1024),
                    "--inventory-stream-coalesce-ms",
                    "2",
                    "--index-budget",
                    index_budget,
                    "--metadata-namespace",
                    storage_namespace,
                    "--enable-prometheus",
                ),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers[node] = manager
            client = CacheManagerClient(manager.bootstrap_socket)
            stack.callback(client.close)
            clients[node] = client
            tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
            tensors[node] = tensor
            client.start_session_watcher(node, identity, 1, 1)
            ok, message = client.register_context_batch(
                node,
                identity,
                0,
                0,
                1,
                1,
                resolve_device_id(),
                ["kv:0"],
                [serialize_gpu_buffer(tensor)],
                [pages],
                [block_bytes],
                [0],
                [1],
                "direct",
                False,
                tensors=[tensor],
            )
            assert ok, message

        start_manager("observer")
        for owner in range(owners):
            start_manager(f"source-{owner}")
        manager_list = list(managers.values())
        source_nodes = [f"source-{owner}" for owner in range(owners)]
        device = resolve_device_id()
        hashes = {
            node: [
                hashlib.sha256(f"s2.10:capacity:{node}:{block}".encode()).digest()
                for block in range(pages)
            ]
            for node in source_nodes
        }

        for owner, node in enumerate(source_nodes):
            payload = _payload(torch, pages, block_bytes, owner)
            tensors[node].copy_(payload)
            torch.cuda.synchronize()
            ok, message = clients[node].save(
                node, 0, 0, device, [("kv:0", list(range(pages)), hashes[node])]
            )
            assert ok, message

        bootstrap_started = time.monotonic()
        fences = {node: _sync(managers[node]) for node in source_nodes}
        if not expect_degraded:
            for node in source_nodes:
                _await_fence(managers["observer"], fences[node], manager_list, timeout=60)
            bootstrap_ms = (time.monotonic() - bootstrap_started) * 1000
            assert _metadata(managers["observer"])["index"]["coverage"] == (
                "complete_at_watermarks"
            )
        else:
            _until(
                lambda: _metadata(managers["observer"])["index"]["coverage"]
                != "complete_at_watermarks",
                manager_list,
                timeout=30,
            )
            bootstrap_ms = None

        before_revision = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        process_before = _process_sample(manager.process.pid for manager in manager_list)
        visibility = []
        save_latency = []
        started = time.monotonic()
        cycle = 0
        remote_bytes = 0
        if not expect_degraded:
            while time.monotonic() - started < duration_seconds:
                cycle_started = time.monotonic()
                save_started = time.monotonic()
                for owner, node in enumerate(source_nodes):
                    cleaned = _cleanup_dram(managers[node])
                    assert cleaned["evicted_blocks"] == pages
                    payload = _payload(torch, pages, block_bytes, cycle * owners + owner + 1)
                    tensors[node].copy_(payload)
                    torch.cuda.synchronize()
                    ok, message = clients[node].save(
                        node,
                        0,
                        0,
                        device,
                        [("kv:0", list(range(pages)), hashes[node])],
                    )
                    assert ok, message
                save_latency.append((time.monotonic() - save_started) * 1000)
                fences = {node: _sync(managers[node]) for node in source_nodes}
                visibility_started = time.monotonic()
                for node in source_nodes:
                    _await_fence(managers["observer"], fences[node], manager_list, timeout=60)
                visibility.append((time.monotonic() - visibility_started) * 1000)
                selected = source_nodes[cycle % owners]
                expected = _payload(
                    torch, pages, block_bytes, cycle * owners + (cycle % owners) + 1
                )
                _cleanup_dram(managers["observer"])
                before = fetch_orbitkv_metrics(managers["observer"].http_port).get(
                    "orbitkv_remote_fetch_bytes_total", 0
                )
                _restore(
                    clients["observer"],
                    "observer",
                    tensors["observer"],
                    hashes[selected],
                    f"capacity-{cycle}",
                    expected,
                    manager_list,
                )
                _wait_for_remote_drain(managers[selected], managers["observer"], manager_list)
                after = fetch_orbitkv_metrics(managers["observer"].http_port)[
                    "orbitkv_remote_fetch_bytes_total"
                ]
                assert after >= before + payload_bytes
                remote_bytes += after - before
                observer = _metadata(managers["observer"])
                raw.write(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "elapsed": time.monotonic() - started,
                            "visibility_ms": visibility[-1],
                            "save_ms": save_latency[-1],
                            "index": observer["index"],
                            "stream": observer["stream"],
                        }
                    )
                    + "\n"
                )
                cycle += 1
                remaining = 1 - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
            assert time.monotonic() - started >= duration_seconds
        else:
            while time.monotonic() - started < duration_seconds:
                cycle_started = time.monotonic()
                save_started = time.monotonic()
                for owner, node in enumerate(source_nodes):
                    cleaned = _cleanup_dram(managers[node])
                    assert cleaned["evicted_blocks"] == pages
                    payload = _payload(torch, pages, block_bytes, cycle * owners + owner + 1)
                    tensors[node].copy_(payload)
                    torch.cuda.synchronize()
                    ok, message = clients[node].save(
                        node,
                        0,
                        0,
                        device,
                        [("kv:0", list(range(pages)), hashes[node])],
                    )
                    assert ok, message
                save_latency.append((time.monotonic() - save_started) * 1000)
                observer = _metadata(managers["observer"])
                assert observer["index"]["coverage"] != "complete_at_watermarks"
                raw.write(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "elapsed": time.monotonic() - started,
                            "save_ms": save_latency[-1],
                            "index": observer["index"],
                            "stream": observer["stream"],
                        }
                    )
                    + "\n"
                )
                cycle += 1
                remaining = 1 - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
            assert time.monotonic() - started >= duration_seconds

        observer_status = _metadata(managers["observer"])
        owner_rows = requests.get(
            f"http://127.0.0.1:{managers['observer'].http_port}/cache/metadata/owners",
            params={"limit": 128},
            timeout=5,
        ).json()
        if expect_degraded:
            assert observer_status["index"]["coverage"] != "complete_at_watermarks"
            assert (
                observer_status["index"]["accounted_bytes"]
                <= int(index_budget.removesuffix("kb")) * 1024
            )
            assert observer_status["index"]["expected_owner_views"] >= owners
            assert (
                observer_status["index"]["installed_owner_views"]
                < observer_status["index"]["expected_owner_views"]
            )
        else:
            assert len([row for row in owner_rows if row["records"] == pages]) >= owners
        after_revision = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert after_revision == before_revision
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        process_after = _process_sample(manager.process.pid for manager in manager_list)
        result.update(
            {
                "status": "bounded_degradation" if expect_degraded else "passed",
                "cycles": cycle,
                "wall_seconds": time.monotonic() - started,
                "bootstrap_ms": bootstrap_ms,
                "visibility_ms": _summary(visibility),
                "save_latency_ms": _summary(save_latency),
                "remote_fetch_bytes": remote_bytes,
                "observer": observer_status,
                "owner_rows": owner_rows,
                "process_before": process_before,
                "process_after": process_after,
                "etcd_revision_delta": after_revision - before_revision,
                "etcd_keys": keys,
            }
        )
        if owners == 16 and not expect_degraded:
            assert result["visibility_ms"]["p99"] <= 50
        (output / "capacity-result.json").write_text(json.dumps(result, indent=2) + "\n")

        for node, client in clients.items():
            with contextlib.suppress(Exception):
                client.unregister_context(node)
            client.close()
        tensors.clear()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        for manager in managers.values():
            exit_code, seconds = manager.terminate_gracefully(timeout=10)
            assert exit_code == 0 and seconds <= 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owners", type=int, choices=(1, 4, 16), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--index-budget", default="16mb")
    parser.add_argument("--expect-degraded", action="store_true")
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("duration must be positive")
    if args.expect_degraded and not args.index_budget.endswith("kb"):
        parser.error("degraded index budget must use kb")
    for variable in ("ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY", "ORBITKV_MOONCAKE_LIB_DIR"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(
            args.output,
            args.owners,
            args.duration_seconds,
            args.index_budget,
            args.expect_degraded,
            args.seed,
        )
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
