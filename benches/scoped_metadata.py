"""Matched full-Manager all-domain versus scoped inventory-stream experiment."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import statistics
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
    _owner_status,
    _payload,
    _query_ready,
    _restore,
    _sync,
    _until,
    _wait_for_remote_drain,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import TcpGate, etcd_server

from .artifacts import external_path


def _summary(values):
    ordered = sorted(values)

    def percentile(fraction):
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "samples": len(ordered),
        "p50": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _process_sample(pids):
    result = {"cpu_ticks": 0, "rss_kib": 0, "hwm_kib": 0, "processes": 0}
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, IndexError):
            continue
        values = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in status
            if line.startswith(("VmRSS:", "VmHWM:"))
        }
        result["cpu_ticks"] += int(stat[11]) + int(stat[12])
        result["rss_kib"] += values.get("VmRSS", 0)
        result["hwm_kib"] += values.get("VmHWM", 0)
        result["processes"] += 1
    return result


def _etcd_revision(endpoint, prefix):
    key = prefix.encode()
    end = bytearray(key)
    end[-1] += 1
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(bytes(end)).decode(),
            "limit": "1",
        },
        timeout=5,
    )
    response.raise_for_status()
    return int(response.json()["header"]["revision"])


def run(output: Path, subscription: str, coalesce_ms: int):
    import torch

    import orbitkv.orbitkv as native
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    assert torch.cuda.is_available()
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes, namespace_count, cycles = 8, 4096, 4, 50
    payload_bytes = pages * block_bytes
    identities = [f"s2.9:bench:{index}:{uuid.uuid4().hex}" for index in range(namespace_count)]
    storage_namespaces = _discover_storage_namespaces(output, identities, pages, block_bytes)
    selected_scope = storage_namespaces[0]
    hashes = [hashlib.sha256(f"s2.9:matched:{block}".encode()).digest() for block in range(pages)]
    payloads = [
        _payload(torch, pages, block_bytes, 120 + index) for index in range(namespace_count)
    ]
    cluster = f"s29-{subscription}-{uuid.uuid4().hex[:12]}"
    result = {
        "subscription": subscription,
        "coalesce_ms": coalesce_ms,
        "namespace_count": namespace_count,
        "selected_scope": selected_scope,
        "storage_namespaces": storage_namespaces,
        "cycles": cycles,
        "pages": pages,
        "block_bytes": block_bytes,
        "inference_latency": None,
        "scope": "same-host A100, two full Managers, forced TCP, metadata/client workload",
    }
    artifacts = {
        "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
        "etcd": os.environ["ETCD_BIN"],
        "extension": native.__file__,
        "tent": str(Path(os.environ["ORBITKV_MOONCAKE_LIB_DIR"]) / "libtent_shared.so"),
    }
    result["artifacts"] = {
        name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in artifacts.items()
    }
    raw = (output / "samples.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(output))
        source_backend = find_available_port()
        gate = TcpGate(f"http://127.0.0.1:{source_backend}")
        stack.callback(gate.close)
        source_advertise = gate.endpoint.rsplit("//", 1)[1]
        managers = []
        clients = {}
        tensors = {}
        for node in ("source", "observer"):
            source = node == "source"
            port = source_backend if source else find_available_port()
            args = [
                "--etcd-endpoints",
                endpoint,
                "--node-id",
                node,
                "--cluster-name",
                cluster,
                "--membership-ttl-secs",
                "120",
                "--inventory-journal-bytes",
                str(8 * 1024),
                "--inventory-stream-coalesce-ms",
                str(coalesce_ms),
                "--enable-prometheus",
            ]
            if subscription == "scoped":
                args.extend(["--metadata-namespace", selected_scope])
            if source:
                args.extend(["--peer-advertise-addr", source_advertise])
            manager = CacheManagerProcess(
                port,
                pool_size="96mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s29-bench-{port}.sock",
                log_path=output / f"{node}-manager.log",
                extra_args=tuple(args),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)
            for index, identity in enumerate(identities):
                instance = f"{node}-{index}"
                client = CacheManagerClient(manager.bootstrap_socket)
                stack.callback(client.close)
                clients[(node, index)] = client
                tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
                tensors[(node, index)] = tensor
                client.start_session_watcher(instance, identity, 1, 1)
                ok, message = client.register_context_batch(
                    instance,
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

        source_manager, observer_manager = managers
        device = resolve_device_id()

        def save_all(round_id):
            for index, payload in enumerate(payloads):
                tensor = tensors[("source", index)]
                tensor.copy_((payload + round_id) % 251)
                torch.cuda.synchronize()
                ok, message = clients[("source", index)].save(
                    f"source-{index}", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
                )
                assert ok, message

        save_all(0)
        bootstrap_started = time.monotonic()
        initial_fence = _sync(source_manager)
        _await_fence(observer_manager, initial_fence, managers)
        bootstrap_ms = (time.monotonic() - bootstrap_started) * 1000
        before_source = _metadata(source_manager)
        before_observer = _metadata(observer_manager)
        process_before = _process_sample(manager.process.pid for manager in managers)
        revision_before = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        async_visibility = []
        barrier_visibility = []
        for cycle in range(cycles):
            cleaned = _cleanup_dram(source_manager)
            assert cleaned["evicted_blocks"] == pages * namespace_count
            sequence_before = _metadata(source_manager)["inventory_sequence"]
            started = time.monotonic()
            save_all(cycle + 1)
            if cycle % 2 == 0:
                expected = sequence_before + pages * namespace_count
                _until(
                    lambda expected=expected: _metadata(source_manager)["inventory_sequence"]
                    >= expected,
                    managers,
                )
                target = _metadata(source_manager)["inventory_sequence"]
                _until(
                    lambda target=target: (
                        (
                            status := _owner_status(
                                observer_manager, initial_fence["source_incarnation"]
                            )
                        )
                        and status["fresh"]
                        and status["applied_sequence"] >= target
                    ),
                    managers,
                )
                visibility = (time.monotonic() - started) * 1000
                async_visibility.append(visibility)
                mode = "async"
            else:
                fence = _sync(source_manager)
                _await_fence(observer_manager, fence, managers)
                visibility = (time.monotonic() - started) * 1000
                barrier_visibility.append(visibility)
                mode = "barrier"
            _cleanup_dram(observer_manager)
            expected_payload = (payloads[0] + cycle + 1) % 251
            _restore(
                clients[("observer", 0)],
                "observer-0",
                tensors[("observer", 0)],
                hashes,
                f"selected-{cycle}",
                expected_payload,
                managers,
            )
            _wait_for_remote_drain(source_manager, observer_manager, managers)
            if subscription == "scoped":
                miss = _query_ready(
                    clients[("observer", 1)],
                    "observer-1",
                    hashes,
                    f"outside-{cycle}",
                    0,
                    managers,
                )
                assert not miss.lease
            raw.write(
                json.dumps({"cycle": cycle, "mode": mode, "visibility_ms": visibility}) + "\n"
            )

        gate.partition()
        for repair in range(10):
            _cleanup_dram(source_manager)
            save_all(cycles + repair + 1)
        repair_fence = _sync(source_manager)
        repair_started = time.monotonic()
        gate.heal()
        _await_fence(observer_manager, repair_fence, managers, timeout=20)
        repair_ms = (time.monotonic() - repair_started) * 1000
        _cleanup_dram(observer_manager)
        _restore(
            clients[("observer", 0)],
            "observer-0",
            tensors[("observer", 0)],
            hashes,
            "repair-selected",
            (payloads[0] + cycles + 10) % 251,
            managers,
        )
        _wait_for_remote_drain(source_manager, observer_manager, managers)

        after_source = _metadata(source_manager)
        after_observer = _metadata(observer_manager)
        process_after = _process_sample(manager.process.pid for manager in managers)
        revision_after = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        owner = _owner_status(observer_manager, initial_fence["source_incarnation"])
        expected_records = pages * (namespace_count if subscription == "all" else 1)
        assert owner and owner["records"] == expected_records
        result.update(
            {
                "cluster": cluster,
                "manager_commands": [manager.command for manager in managers],
                "bootstrap_ms": bootstrap_ms,
                "repair_ms": repair_ms,
                "async_visibility_ms": _summary(async_visibility),
                "barrier_visibility_ms": _summary(barrier_visibility),
                "owner_records": owner["records"],
                "expected_owner_records": expected_records,
                "stream_bytes_sent": after_source["stream"]["encoded_bytes_sent"]
                - before_source["stream"]["encoded_bytes_sent"],
                "stream_bytes_received": after_observer["stream"]["encoded_bytes_received"]
                - before_observer["stream"]["encoded_bytes_received"],
                "delta_input_records": after_source["inventory_delta_input_records"]
                - before_source["inventory_delta_input_records"],
                "delta_output_records": after_source["inventory_delta_output_records"]
                - before_source["inventory_delta_output_records"],
                "delta_frames": after_source["inventory_delta_frames"]
                - before_source["inventory_delta_frames"],
                "filter_input_records": after_source["inventory_scope_filter_input_records"]
                - before_source["inventory_scope_filter_input_records"],
                "filter_output_records": after_source["inventory_scope_filter_output_records"]
                - before_source["inventory_scope_filter_output_records"],
                "filter_micros": after_source["inventory_scope_filter_micros"]
                - before_source["inventory_scope_filter_micros"],
                "index": after_observer["index"],
                "queue_peak_bytes": after_source["stream"]["outbound_queue_bytes_peak"],
                "source_sessions_peak": after_source["stream"]["source_sessions_peak"],
                "receiver_sessions_peak": after_observer["stream"]["receiver_sessions_peak"],
                "process_before": process_before,
                "process_after": process_after,
                "etcd_revision_delta": revision_after - revision_before,
                "etcd_keys": keys,
            }
        )
        assert result["filter_input_records"] >= result["filter_output_records"]
        assert after_observer["index"]["coverage"] == "complete_at_watermarks"
        assert after_source["stream"]["source_sessions_peak"] >= 1
        assert after_observer["stream"]["receiver_sessions_peak"] >= 1
        (output / "scoped-result.json").write_text(json.dumps(result, indent=2) + "\n")
        for (node, index), client in clients.items():
            with contextlib.suppress(Exception):
                client.unregister_context(f"{node}-{index}")
            client.close()
        tensor = None
        tensors.clear()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", choices=("all", "scoped"), required=True)
    parser.add_argument("--coalesce-ms", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    for variable in ("ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY", "ORBITKV_MOONCAKE_LIB_DIR"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.output, args.subscription, args.coalesce_ms)
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
