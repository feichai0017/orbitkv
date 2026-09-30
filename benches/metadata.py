"""Explicit full-Manager metadata coalescing experiment; see distributed-cache.md.

Run with the frozen package and test support on PYTHONPATH. No native build is
performed. Failed runs and raw measurements remain in the required external output.
"""

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
    _cleanup_dram,
    _expected_records,
    _group_hash,
    _member_owner,
    _metadata,
    _payload,
    _prefix_end,
    _query_ready,
    _restore,
    _source_records,
    _sync,
    _until,
    _wait_for_revision,
    _wait_source_records,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import etcd_server
from tests.support.metrics import fetch_orbitkv_metrics

from .artifacts import external_path


def _process_sample(pids):
    result = {"cpu_ticks": 0, "rss_kib": 0, "hwm_kib": 0, "processes": 0}
    for pid in pids:
        try:
            stat = (Path(f"/proc/{pid}/stat")).read_text().rsplit(") ", 1)[1].split()
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


def _etcd_pid(endpoint):
    encoded = endpoint.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        if Path(os.fsdecode(command.split(b"\0", 1)[0])).name == "etcd" and encoded in command:
            return int(entry.name)
    raise AssertionError(f"cannot find test-owned etcd for {endpoint}")


def _etcd_metrics(endpoint):
    response = requests.get(f"{endpoint}/metrics", timeout=5)
    response.raise_for_status()
    wanted = {
        "etcd_network_client_grpc_received_bytes_total",
        "etcd_network_client_grpc_sent_bytes_total",
    }
    result = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(maxsplit=1)
        name = name.split("{", 1)[0]
        if name in wanted:
            result[name] = result.get(name, 0.0) + float(value)
    assert set(result) == wanted, "etcd byte metrics missing"
    return result


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


def _etcd_revision(endpoint: str, prefix: str) -> int:
    key = prefix.encode()
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(_prefix_end(key)).decode(),
            "limit": "1",
        },
        timeout=5,
    )
    response.raise_for_status()
    return int(response.json()["header"]["revision"])


def run(tmp_path: Path, profile: str):
    import torch

    assert torch.cuda.is_available()
    import orbitkv.orbitkv as native
    from orbitkv import BlockHashes, CacheManagerClient, QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    artifact_paths = {
        "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
        "etcd": os.environ["ETCD_BIN"],
        "extension": native.__file__,
    }
    if library_dir := os.environ.get("ORBITKV_MOONCAKE_LIB_DIR"):
        artifact_paths["tent"] = str(Path(library_dir) / "libtent_shared.so")
    manifest = {}
    for name, value in artifact_paths.items():
        path = Path(value).resolve()
        manifest[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes, cycles, low_traffic_samples = 8, 4096, 200, 10
    payload_bytes = pages * block_bytes
    cluster = f"s27-{profile}-{uuid.uuid4().hex[:12]}"
    namespace = f"s2.7:{profile}:{uuid.uuid4().hex}"
    hashes = [hashlib.sha256(f"s2.7:{block}".encode()).digest() for block in range(pages)]
    hash_batch = BlockHashes(hashes)
    expected_records = _expected_records(hashes, "dram")
    payload = _payload(torch, pages, block_bytes, 27)
    result = {
        "inference_latency": None,
        "inference_scope": "client metadata experiment; no model-serving workload",
        "profile": profile,
        "pages": pages,
        "block_bytes": block_bytes,
        "burst_cycles": cycles,
        "low_traffic_samples": low_traffic_samples,
        "expected_input_mutations": (cycles + low_traffic_samples) * pages * 2,
    }

    raw = (tmp_path / "samples.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(tmp_path))
        managers = []
        for node in ("source", "observer"):
            port = find_available_port()
            http_port = find_available_port()
            manager_args = [
                "--etcd-endpoints",
                endpoint,
                "--node-id",
                node,
                "--cluster-name",
                cluster,
                "--membership-ttl-secs",
                "120",
                "--inventory-journal-bytes",
                str(16 * 1024 * 1024),
                "--enable-prometheus",
            ]
            if profile != "legacy":
                manager_args.extend(["--inventory-publish-coalesce-ms", profile])
            manager = CacheManagerProcess(
                port,
                pool_size="64mb",
                http_port=http_port,
                bootstrap_socket=f"/tmp/orbitkv-s27-{port}.sock",
                log_path=tmp_path / f"{node}-manager.log",
                extra_args=tuple(manager_args),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)

        source_manager, observer_manager = managers
        source_client = CacheManagerClient(source_manager.bootstrap_socket)
        stack.callback(source_client.close)
        tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
        observer_client = CacheManagerClient(observer_manager.bootstrap_socket)
        stack.callback(observer_client.close)
        observer_tensor = torch.empty_like(tensor)
        tensor.copy_(payload)
        torch.cuda.synchronize()
        device = resolve_device_id()
        source_client.start_session_watcher("source", namespace, 1, 1)
        ok, message = source_client.register_context_batch(
            "source",
            namespace,
            0,
            0,
            1,
            1,
            device,
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
        observer_client.start_session_watcher("observer", namespace, 1, 1)
        ok, message = observer_client.register_context_batch(
            "observer",
            namespace,
            0,
            0,
            1,
            1,
            device,
            ["kv:0"],
            [serialize_gpu_buffer(observer_tensor)],
            [pages],
            [block_bytes],
            [0],
            [1],
            "direct",
            False,
            tensors=[observer_tensor],
        )
        assert ok, message

        def unregister_on_exit():
            for client, instance in [(observer_client, "observer"), (source_client, "source")]:
                with contextlib.suppress(Exception):
                    client.unregister_context(instance)

        stack.callback(unregister_on_exit)
        _until(
            lambda: (
                _metadata(source_manager)["index"]["available"]
                and _metadata(observer_manager)["index"]["available"]
            ),
            managers,
        )

        ok, message = source_client.save(
            "source", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
        )
        assert ok, message
        initial_revision = _sync(source_manager)
        _wait_for_revision(observer_manager, initial_revision, managers)
        owner = _until(lambda: _member_owner(endpoint, cluster, "source"), managers)
        incarnation = owner["incarnation"]
        _wait_source_records(endpoint, cluster, incarnation, expected_records, managers)

        source_before = _metadata(source_manager)
        watch_metric = "orbitkv_metadata_watch_key_value_bytes_total"
        watch_before = [fetch_orbitkv_metrics(m.http_port).get(watch_metric) for m in managers]
        etcd_before = _etcd_metrics(endpoint)
        revision_before = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        manager_before = _process_sample(manager.process.pid for manager in managers)
        etcd_process_before = _process_sample([_etcd_pid(endpoint)])
        (tmp_path / "launch.json").write_text(
            json.dumps(
                {
                    "commands": [m.command for m in managers],
                    "pids": [m.process.pid for m in managers],
                    "etcd_pid": _etcd_pid(endpoint),
                    "source_owner": owner,
                    "profile": profile,
                },
                indent=2,
            )
        )
        immediate_local_hits = []
        save_ms = []
        query_ms = []
        cleanup_ms = []
        workload_started = time.monotonic()
        for cycle in range(cycles):
            started = time.monotonic()
            cleaned = _cleanup_dram(source_manager)
            cleanup_ms.append((time.monotonic() - started) * 1000)
            assert cleaned["evicted_blocks"] == pages

            started = time.monotonic()
            ok, message = source_client.save(
                "source", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
            )
            save_ms.append((time.monotonic() - started) * 1000)
            assert ok, message

            started = time.monotonic()
            query_calls = 1
            ready = source_client.query_prefetch("source", hash_batch, f"s2.7-burst-{cycle}")
            immediate_local_hits.append(
                ready.num_hit_blocks if isinstance(ready, QueryReady) else 0
            )
            deadline = started + 5
            while not isinstance(ready, QueryReady) or ready.num_hit_blocks != pages:
                if isinstance(ready, QueryReady) and ready.lease:
                    source_client.release(ready.lease)
                assert time.monotonic() < deadline, (cycle, ready)
                time.sleep(0.0005)
                query_calls += 1
                ready = source_client.query_prefetch("source", hash_batch, f"s2.7-burst-{cycle}")
            query_ms.append((time.monotonic() - started) * 1000)
            source_client.release(ready.lease)
            raw.write(
                json.dumps(
                    {
                        "phase": "burst",
                        "cycle": cycle,
                        "query_calls": query_calls,
                        "immediate_hits": immediate_local_hits[-1],
                        "save_ms": save_ms[-1],
                        "query_ms": query_ms[-1],
                        "cleanup_ms": cleanup_ms[-1],
                        "hits": ready.num_hit_blocks,
                    }
                )
                + "\n"
            )

        burst_revision = _sync(source_manager)
        burst_sync_ms = (time.monotonic() - workload_started) * 1000
        watch_started = time.monotonic()
        _wait_for_revision(observer_manager, burst_revision, managers)
        burst_watch_ms = (time.monotonic() - watch_started) * 1000
        _wait_source_records(endpoint, cluster, incarnation, expected_records, managers)

        low_visibility_ms = []
        immediate_remote_hits = []
        for sample in range(low_traffic_samples):
            _cleanup_dram(observer_manager)
            cleaned = _cleanup_dram(source_manager)
            assert cleaned["evicted_blocks"] == pages
            delete_revision = _sync(source_manager)
            _wait_for_revision(observer_manager, delete_revision, managers)
            assert not _source_records(endpoint, cluster, incarnation)
            target_sequence = _metadata(source_manager)["inventory_sequence"] + pages

            started = time.monotonic()
            ok, message = source_client.save(
                "source", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
            )
            assert ok, message
            immediate = observer_client.query_prefetch(
                "observer", hash_batch, f"immediate-{sample}"
            )
            deadline = time.monotonic() + 5
            while not isinstance(immediate, QueryReady):
                assert time.monotonic() < deadline, immediate
                time.sleep(0.0005)
                immediate = observer_client.query_prefetch(
                    "observer", hash_batch, f"immediate-{sample}"
                )
            immediate_remote_hits.append(immediate.num_hit_blocks)
            if immediate.lease:
                observer_client.release(immediate.lease)
            deadline = time.monotonic() + 5
            while True:
                progress = _metadata(source_manager)["published"]
                observed = _metadata(observer_manager)["index"]
                if (
                    progress["ready"]
                    and progress["sequence"] >= target_sequence
                    and observed["available"]
                    and observed["revision"] >= progress["revision"]
                ):
                    break
                assert time.monotonic() < deadline, (progress, observed)
                time.sleep(0.0005)
            low_visibility_ms.append((time.monotonic() - started) * 1000)
            ready = _query_ready(
                source_client,
                "source",
                hashes,
                f"s2.7-low-{sample}",
                pages,
                managers,
            )
            source_client.release(ready.lease)
            records = _wait_source_records(
                endpoint, cluster, incarnation, expected_records, managers
            )
            assert {record["sequence"] for record in records.values()} == set(
                range(target_sequence - pages + 1, target_sequence + 1)
            )
            for block, block_hash in enumerate(hashes):
                assert (
                    records[(_group_hash(block_hash), "dram")]["sequence"]
                    == target_sequence - pages + block + 1
                )
            _restore(
                observer_client,
                "observer",
                observer_tensor,
                hashes,
                f"observed-{sample}",
                payload,
                managers,
            )
            raw.write(
                json.dumps(
                    {
                        "phase": "low",
                        "sample": sample,
                        "visibility_ms": low_visibility_ms[-1],
                        "target_sequence": target_sequence,
                        "key_generations": {
                            identity[0].hex(): record["sequence"]
                            for identity, record in records.items()
                        },
                        "source_published": progress,
                        "observer_index": observed,
                        "immediate_remote_hits": immediate_remote_hits[-1],
                    }
                )
                + "\n"
            )
            time.sleep(0.02)

        final_revision = _sync(source_manager)
        _wait_for_revision(observer_manager, final_revision, managers)
        final_records = _wait_source_records(
            endpoint, cluster, incarnation, expected_records, managers
        )
        assert _member_owner(endpoint, cluster, "source") == owner
        assert all(record["stored_bytes"] == block_bytes for record in final_records.values())
        source_after = _metadata(source_manager)
        watch_after = [fetch_orbitkv_metrics(m.http_port).get(watch_metric) for m in managers]
        observer_after = _metadata(observer_manager)
        etcd_after = _etcd_metrics(endpoint)
        revision_after = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        manager_after = _process_sample(manager.process.pid for manager in managers)
        etcd_process_after = _process_sample([_etcd_pid(endpoint)])
        elapsed = time.monotonic() - workload_started

        counters = {}
        for name in (
            "inventory_delta_input_records",
            "inventory_delta_input_bytes",
            "inventory_delta_output_records",
            "inventory_delta_transactions",
            "inventory_delta_encoded_bytes",
            "inventory_coalescing_windows",
            "inventory_coalescing_wait_micros",
        ):
            before = source_before.get(name)
            after = source_after.get(name)
            counters[name] = None if before is None or after is None else after - before

        result.update(
            {
                "cluster": cluster,
                "source_incarnation": incarnation,
                "manager_commands": [manager.command for manager in managers],
                "manager_pids": [manager.process.pid for manager in managers],
                "etcd_pid": _etcd_pid(endpoint),
                "workload_seconds": elapsed,
                "burst_sync_ms": burst_sync_ms,
                "burst_watch_ms": burst_watch_ms,
                "save_ms": _summary(save_ms),
                "local_query_ms": _summary(query_ms),
                "cleanup_ms": _summary(cleanup_ms),
                "low_traffic_visibility_ms": _summary(low_visibility_ms),
                "effective_local_hit_rate": 1.0,
                "immediate_local_hit_rate": sum(immediate_local_hits) / (pages * cycles),
                "immediate_remote_hit_rate": sum(immediate_remote_hits)
                / (pages * low_traffic_samples),
                "after_visibility_remote_hit_rate": 1.0,
                "final_records": len(final_records),
                "final_revision": final_revision,
                "observer_revision": observer_after["index"]["revision"],
                "publisher_counters": counters,
                "watch_key_value_bytes": [
                    None if before is None else after - before
                    for before, after in zip(watch_before, watch_after, strict=True)
                ],
                "etcd_metrics_delta": {
                    name: etcd_after[name] - etcd_before[name] for name in etcd_before
                },
                "manager_process_before": manager_before,
                "manager_process_after": manager_after,
                "etcd_process_before": etcd_process_before,
                "etcd_process_after": etcd_process_after,
                "etcd_revision_before": revision_before,
                "etcd_revision_after": revision_after,
                "etcd_revision_delta": revision_after - revision_before,
            }
        )
        if profile != "legacy":
            assert all(value is not None and value > 0 for value in result["watch_key_value_bytes"])
            assert counters["inventory_delta_input_records"] == result["expected_input_mutations"]
            assert (
                counters["inventory_delta_output_records"]
                <= counters["inventory_delta_input_records"]
            )
            assert (
                source_after["inventory_journal_bytes"]
                <= source_after["inventory_journal_capacity_bytes"]
            )
        _restore(source_client, "source", tensor, hashes, "final-local-bytes", payload, managers)
        for label, before, after in [
            ("managers", manager_before, manager_after),
            ("etcd", etcd_process_before, etcd_process_after),
        ]:
            assert before["processes"] == after["processes"] and before["processes"] > 0
            assert after["hwm_kib"] - before["hwm_kib"] <= 256 * 1024, label
        (tmp_path / "coalescing-result.json").write_text(json.dumps(result, indent=2) + "\n")
        ok, message = observer_client.unregister_context("observer")
        assert ok, message
        ok, message = source_client.unregister_context("source")
        assert ok, message
        tensor = observer_tensor = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        (tmp_path / "coalescing-result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("legacy", "0", "2", "5"), required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    for variable in ["ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY"]:
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.output, args.profile)
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
