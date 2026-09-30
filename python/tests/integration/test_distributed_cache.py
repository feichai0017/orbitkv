"""Real etcd + Manager processes: metadata faults and exact GPU recovery.

Requires ETCD_BIN, a frozen Cache Manager/native extension, CUDA and
MC_FORCE_TCP=1 on one host. This is not a cross-host qualification.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from contextlib import ExitStack

import pytest
import requests

from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import TcpGate, etcd_server
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _until(predicate, managers, *, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        assert time.monotonic() < deadline, [manager.read_logs() for manager in managers]
        time.sleep(0.02)


def _metadata(manager):
    response = requests.get(f"http://127.0.0.1:{manager.http_port}/cache/metadata", timeout=5)
    response.raise_for_status()
    return response.json()


def _sync(manager):
    response = requests.post(f"http://127.0.0.1:{manager.http_port}/cache/sync", timeout=35)
    response.raise_for_status()
    return response.json()["published_revision"]


def _wait_for_revision(manager, revision, managers):
    return _until(
        lambda: (status := _metadata(manager))["index"]["available"]
        and status["index"]["revision"] >= revision,
        managers,
    )


def _query_ready(client, instance, hashes, request, expected, managers):
    from orbitkv import BlockHashes, QueryReady

    batch = BlockHashes(hashes)

    def poll():
        result = client.query_prefetch(instance, batch, request)
        if not isinstance(result, QueryReady):
            return None
        if result.num_hit_blocks == expected:
            return result
        if result.lease:
            client.release(result.lease)
        if expected == 0:
            raise AssertionError(
                f"expected no hits for {request}, observed {result.num_hit_blocks}"
            )
        return None

    return _until(poll, managers)


def _restore(client, instance, tensor, hashes, request, expected, managers):
    import torch

    tensor.zero_()
    torch.cuda.synchronize()
    result = _query_ready(client, instance, hashes, request, len(hashes), managers)
    operation = client.start_restore(
        instance,
        0,
        tensor.device.index or 0,
        [["kv:0"]],
        [(result.lease, [list(range(len(hashes)))])],
        ready_stream=torch.cuda.current_stream(tensor.device).cuda_stream,
    )
    status = client.wait_restore(operation, timeout=20)
    assert status.success, status
    torch.cuda.synchronize()
    assert torch.equal(tensor, expected)


def _cleanup_dram(manager):
    response = requests.post(
        f"http://127.0.0.1:{manager.http_port}/cache/memory/cleanup", timeout=10
    )
    response.raise_for_status()
    result = response.json()
    assert result["still_referenced_blocks"] == 0
    return result


def _wait_for_ssd_write(manager, minimum_bytes, managers):
    return _until(
        lambda: (
            metrics
            if (metrics := fetch_orbitkv_metrics(manager.http_port)).get(
                "orbitkv_ssd_write_bytes_total", 0
            )
            >= minimum_bytes
            and not any(
                metrics.get(name, 0)
                for name in (
                    "orbitkv_ssd_write_inflight",
                    "orbitkv_ssd_write_queue_pending",
                    "orbitkv_inflight_bytes",
                )
            )
            else None
        ),
        managers,
    )


def _prefix_end(prefix: bytes) -> bytes:
    end = bytearray(prefix)
    end[-1] += 1
    return bytes(end)


def _etcd_range(endpoint: str, prefix: str):
    key = prefix.encode()
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(_prefix_end(key)).decode(),
        },
        timeout=5,
    )
    response.raise_for_status()
    return [
        (base64.b64decode(row["key"]).decode(), base64.b64decode(row.get("value", "")))
        for row in response.json().get("kvs", [])
    ]


def _member_incarnation(endpoint: str, cluster: str, node: str) -> str | None:
    key = f"/orbitkv/v2/{cluster}/members/{node}"
    rows = [row for row in _etcd_range(endpoint, key) if row[0] == key]
    if not rows:
        return None
    assert len(rows) == 1
    return json.loads(rows[0][1])["owner"]["incarnation"]


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")


def _protobuf_fields(data: bytes) -> dict[int, list[int | bytes]]:
    fields: dict[int, list[int | bytes]] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 2:
            size, offset = _varint(data, offset)
            value = data[offset : offset + size]
            offset += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        fields.setdefault(field, []).append(value)
    return fields


def _inventory_record(data: bytes):
    fields = _protobuf_fields(data)
    metadata = _protobuf_fields(fields[5][0])
    return {
        "namespace": fields[1][0].decode(),
        "block_hash": fields[2][0],
        "sequence": fields[3][0],
        "present": bool(fields.get(4, [0])[0]),
        "medium": metadata[1][0],
        "representation": metadata[2][0],
        "stored_bytes": metadata.get(3, [None])[0],
    }


def _source_records(endpoint: str, cluster: str, incarnation: str):
    prefix = f"/orbitkv/v2/{cluster}/blocks/{incarnation}/"
    records = {}
    for key, value in _etcd_range(endpoint, prefix):
        record = _inventory_record(value)
        medium = key.rsplit("/", 1)[-1]
        assert medium in {"dram", "ssd"}
        assert record["medium"] == {"dram": 1, "ssd": 2}[medium]
        assert record["present"]
        assert record["representation"] == 1
        digest = hashlib.sha256()
        digest.update(b"orbitkv/state-location/v2\0")
        namespace = record["namespace"].encode()
        digest.update(len(namespace).to_bytes(8, "little"))
        digest.update(namespace)
        digest.update(record["block_hash"])
        assert key == f"{prefix}{digest.hexdigest()}/{medium}"
        records[(record["block_hash"], medium)] = record
    return records


def _group_hash(block_hash: bytes) -> bytes:
    return (
        b"OKS\x01" + (0).to_bytes(4, "little") + len(block_hash).to_bytes(8, "little") + block_hash
    )


def _expected_records(hashes, medium):
    return {(_group_hash(block_hash), medium) for block_hash in hashes}


def _wait_source_records(endpoint, cluster, incarnation, expected, managers):
    return _until(
        lambda: (
            records
            if set(records := _source_records(endpoint, cluster, incarnation)) == expected
            else None
        ),
        managers,
    )


def _payload(torch, pages, block_bytes, round_id):
    values = torch.arange(pages * block_bytes, device="cuda", dtype=torch.int64).reshape(
        pages, block_bytes
    )
    offsets = torch.arange(pages, device="cuda", dtype=torch.int64).unsqueeze(1) * 17
    return ((values + offsets + round_id * 31) % 251).to(torch.uint8).flatten()


def _hashes(medium, round_id, pages):
    return [
        hashlib.sha256(f"s2.5:{medium}:{round_id}:{block}".encode()).digest()
        for block in range(pages)
    ]


@pytest.mark.parametrize("medium", ["dram", "ssd"])
def test_manager_process_metadata_faults_preserve_exact_dram_and_ssd(tmp_path, monkeypatch, medium):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    monkeypatch.setenv("MC_FORCE_TCP", "1")
    pages, block_bytes = 8, 4096
    payload_bytes = pages * block_bytes
    cluster = f"s25-{medium}-{uuid.uuid4().hex[:12]}"
    namespace = f"s2.5:{medium}:{uuid.uuid4().hex}"
    result = {
        "medium": medium,
        "client_pid": os.getpid(),
        "cluster": cluster,
        "pages": pages,
        "block_bytes": block_bytes,
        "rounds": [],
    }

    with ExitStack() as stack:
        endpoint, _ = stack.enter_context(etcd_server(tmp_path))
        gate = TcpGate(endpoint)
        stack.callback(gate.close)
        managers = []
        clients = []
        tensors = []
        for node in ("source", "consumer"):
            port = find_available_port()
            http_port = find_available_port()
            source = node == "source"
            ssd_enabled = source and medium == "ssd"
            manager_args = [
                "--etcd-endpoints",
                gate.endpoint if source else endpoint,
                "--node-id",
                node,
                "--cluster-name",
                cluster,
                "--membership-ttl-secs",
                "12",
                "--inventory-journal-bytes",
                "1024",
            ]
            if not ssd_enabled:
                manager_args.append("--enable-prometheus")
            manager = CacheManagerProcess(
                port,
                pool_size="64mb",
                http_port=http_port,
                bootstrap_socket=f"/tmp/orbitkv-s25-{port}.sock",
                ssd_cache_path=tmp_path / "source-ssd" if ssd_enabled else None,
                ssd_cache_capacity="32kb",
                ssd_backend="uring",
                ssd_read_path="uring" if ssd_enabled else None,
                log_path=tmp_path / f"{node}-manager.log",
                extra_args=tuple(manager_args),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)
            client = CacheManagerClient(manager.bootstrap_socket)
            stack.callback(client.close)
            clients.append(client)
            tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
            tensors.append(tensor)
            client.start_session_watcher(node, namespace, 1, 1)
            ok, message = client.register_context_batch(
                node,
                namespace,
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

        source_manager, consumer_manager = managers
        source_client, consumer_client = clients
        source_tensor, consumer_tensor = tensors
        device = resolve_device_id()
        result["manager_pids"] = [manager.process.pid for manager in managers]
        result["manager_commands"] = [manager.command for manager in managers]
        ssd_written = 0

        def save_round(round_id):
            nonlocal ssd_written
            expected = _payload(torch, pages, block_bytes, round_id)
            hashes = _hashes(medium, round_id, pages)
            source_tensor.copy_(expected)
            torch.cuda.synchronize()
            ok, message = source_client.save(
                "source", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
            )
            assert ok, message
            if medium == "ssd":
                ssd_written += payload_bytes
                _wait_for_ssd_write(source_manager, ssd_written, managers)
                cleaned = _cleanup_dram(source_manager)
                assert cleaned["evicted_blocks"] == pages
                before = fetch_orbitkv_metrics(source_manager.http_port).get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                )
                _restore(
                    source_client,
                    "source",
                    source_tensor,
                    hashes,
                    f"source-ssd-{round_id}",
                    expected,
                    managers,
                )
                after = fetch_orbitkv_metrics(source_manager.http_port)[
                    "orbitkv_ssd_prefetch_bytes_total"
                ]
                assert after >= before + payload_bytes
                cleaned = _cleanup_dram(source_manager)
                assert cleaned["evicted_blocks"] == 0
            else:
                _restore(
                    source_client,
                    "source",
                    source_tensor,
                    hashes,
                    f"source-dram-{round_id}",
                    expected,
                    managers,
                )
            result["rounds"].append(
                {
                    "round": round_id,
                    "hashes": [block_hash.hex() for block_hash in hashes],
                    "payload_sha256": [
                        hashlib.sha256(block.cpu().numpy().tobytes()).hexdigest()
                        for block in expected.reshape(pages, block_bytes)
                    ],
                }
            )
            return hashes, expected

        control_hashes, control_payload = save_round(0)
        control_revision = _sync(source_manager)
        _wait_for_revision(consumer_manager, control_revision, managers)
        source_incarnation = _until(
            lambda: _member_incarnation(endpoint, cluster, "source"), managers
        )
        expected_medium = "ssd" if medium == "ssd" else "dram"
        control_records = _wait_source_records(
            endpoint,
            cluster,
            source_incarnation,
            _expected_records(control_hashes, expected_medium),
            managers,
        )
        assert all(record["stored_bytes"] == block_bytes for record in control_records.values())
        remote_before = fetch_orbitkv_metrics(consumer_manager.http_port).get(
            "orbitkv_remote_fetch_bytes_total", 0
        )
        _restore(
            consumer_client,
            "consumer",
            consumer_tensor,
            control_hashes,
            "control-remote",
            control_payload,
            managers,
        )
        remote_after = fetch_orbitkv_metrics(consumer_manager.http_port)[
            "orbitkv_remote_fetch_bytes_total"
        ]
        assert remote_after >= remote_before + payload_bytes
        result["control_revision"] = control_revision

        gate.partition()
        partition_started = time.monotonic()
        transient_rounds = []
        for round_id in range(1, 4):
            if medium == "dram":
                cleaned = _cleanup_dram(source_manager)
                assert cleaned["evicted_blocks"] == pages
            hashes, expected = save_round(round_id)
            miss = _query_ready(
                consumer_client,
                "consumer",
                hashes,
                f"partition-miss-{round_id}",
                0,
                managers,
            )
            assert not miss.lease
            transient_rounds.append((hashes, expected))

        source_status = _metadata(source_manager)
        assert source_status["index"]["registration_valid"]
        assert source_status["inventory_sequence"] > source_status["published"]["sequence"]
        assert set(_source_records(endpoint, cluster, source_incarnation)) == _expected_records(
            control_hashes, expected_medium
        )
        gate.heal()
        transient_revision = _sync(source_manager)
        _wait_for_revision(consumer_manager, transient_revision, managers)
        _until(
            lambda: "Inventory publication history unavailable" in source_manager.read_logs(),
            managers,
        )
        final_hashes, final_payload = transient_rounds[-1]
        final_records = _wait_source_records(
            endpoint,
            cluster,
            source_incarnation,
            _expected_records(final_hashes, expected_medium),
            managers,
        )
        assert all(record["stored_bytes"] == block_bytes for record in final_records.values())
        cleaned = _cleanup_dram(consumer_manager)
        assert cleaned["evicted_blocks"] >= pages
        deleted_rounds = [control_hashes, *(item[0] for item in transient_rounds[:-1])]
        for deleted_round, round_hashes in enumerate(deleted_rounds):
            for index, block_hash in enumerate(round_hashes):
                missing = _query_ready(
                    consumer_client,
                    "consumer",
                    [block_hash],
                    f"deleted-{deleted_round}-{index}",
                    0,
                    managers,
                )
                assert not missing.lease
        _restore(
            consumer_client,
            "consumer",
            consumer_tensor,
            final_hashes,
            "transient-final-remote",
            final_payload,
            managers,
        )
        result["transient"] = {
            "duration_ms": (time.monotonic() - partition_started) * 1000,
            "revision": transient_revision,
            "inventory_sequence": source_status["inventory_sequence"],
            "published_sequence_before_heal": source_status["published"]["sequence"],
        }

        if medium == "dram":
            cleaned = _cleanup_dram(source_manager)
            assert cleaned["evicted_blocks"] == pages
        expiry_hashes, expiry_payload = save_round(4)
        expiry_revision = _sync(source_manager)
        _wait_for_revision(consumer_manager, expiry_revision, managers)
        _wait_source_records(
            endpoint,
            cluster,
            source_incarnation,
            _expected_records(expiry_hashes, expected_medium),
            managers,
        )
        cleaned = _cleanup_dram(consumer_manager)
        assert cleaned["evicted_blocks"] >= pages
        remote_before_expiry = fetch_orbitkv_metrics(consumer_manager.http_port)[
            "orbitkv_remote_fetch_bytes_total"
        ]

        gate.partition()
        expiry_started = time.monotonic()
        _until(lambda: not _metadata(source_manager)["index"]["registration_valid"], managers)
        _until(
            lambda: _member_incarnation(endpoint, cluster, "source") is None
            and _metadata(consumer_manager)["index"]["available"],
            managers,
            timeout=25,
        )
        assert not _source_records(endpoint, cluster, source_incarnation)
        ssd_read_before_expiry = fetch_orbitkv_metrics(source_manager.http_port).get(
            "orbitkv_ssd_prefetch_bytes_total", 0
        )
        _restore(
            source_client,
            "source",
            source_tensor,
            expiry_hashes,
            "expired-source-local",
            expiry_payload,
            managers,
        )
        if medium == "ssd":
            assert (
                fetch_orbitkv_metrics(source_manager.http_port)["orbitkv_ssd_prefetch_bytes_total"]
                >= ssd_read_before_expiry + payload_bytes
            )
        miss = _query_ready(
            consumer_client,
            "consumer",
            expiry_hashes,
            "expired-source-remote",
            0,
            managers,
        )
        assert not miss.lease
        assert (
            fetch_orbitkv_metrics(consumer_manager.http_port)["orbitkv_remote_fetch_bytes_total"]
            == remote_before_expiry
        )

        gate.heal()
        time.sleep(1)
        source_after_heal = _metadata(source_manager)
        assert not source_after_heal["index"]["registration_valid"]
        assert not source_after_heal["published"]["ready"]
        assert _member_incarnation(endpoint, cluster, "source") is None
        assert _metadata(consumer_manager)["index"]["available"]
        miss = _query_ready(
            consumer_client,
            "consumer",
            expiry_hashes,
            "healed-expired-source-remote",
            0,
            managers,
        )
        assert not miss.lease
        assert (
            fetch_orbitkv_metrics(consumer_manager.http_port)["orbitkv_remote_fetch_bytes_total"]
            == remote_before_expiry
        )
        result["expiry"] = {
            "duration_ms": (time.monotonic() - expiry_started) * 1000,
            "source_incarnation": source_incarnation,
            "observer_available": True,
            "source_member_absent": True,
            "source_records_absent": True,
            "old_runtime_remained_fenced_after_heal": True,
            "requester_dram_evicted_blocks": cleaned["evicted_blocks"],
            "remote_fetch_bytes_unchanged": remote_before_expiry,
        }
        result["final_source_metrics"] = fetch_orbitkv_metrics(source_manager.http_port)
        result["final_consumer_metrics"] = fetch_orbitkv_metrics(consumer_manager.http_port)
        for node, client in zip(("source", "consumer"), clients, strict=True):
            ok, message = client.unregister_context(node)
            assert ok, message
        tensors.clear()
        source_tensor = consumer_tensor = tensor = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        (tmp_path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
