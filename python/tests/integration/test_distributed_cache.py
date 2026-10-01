"""Real Manager processes: inventory-stream faults and exact GPU recovery.

Requires ETCD_BIN, a frozen Cache Manager/native extension, CUDA and
MC_FORCE_TCP=1 on one host. This is not cross-host qualification.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import ExitStack
from urllib.parse import urlsplit

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


def _owner_status(manager, incarnation):
    response = requests.get(
        f"http://127.0.0.1:{manager.http_port}/cache/metadata/owners",
        params={"limit": 128},
        timeout=5,
    )
    response.raise_for_status()
    return next((row for row in response.json() if row["owner"] == incarnation), None)


def _sync(manager):
    response = requests.post(f"http://127.0.0.1:{manager.http_port}/cache/sync", timeout=35)
    response.raise_for_status()
    return response.json()["inventory_fence"]


def _await_fence(manager, fence, managers, *, timeout=30):
    scope = _metadata(manager)["stream"]["scope_digest"]
    response = requests.post(
        f"http://127.0.0.1:{manager.http_port}/cache/metadata/await",
        json={
            "inventory_fence": fence,
            "scope_digest": scope,
            "timeout_ms": int(timeout * 1000),
        },
        timeout=timeout + 5,
    )
    assert response.status_code == 200, (
        response.status_code,
        response.text,
        [m.read_logs() for m in managers],
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


def _wait_for_remote_drain(source_manager, consumer_manager, managers):
    def drained():
        source_metrics = fetch_orbitkv_metrics(source_manager.http_port)
        consumer_metrics = fetch_orbitkv_metrics(consumer_manager.http_port)
        if source_metrics.get("orbitkv_transfer_lock_active", 0):
            return None
        if consumer_metrics.get("orbitkv_transfer_completion_outstanding", 0):
            return None
        return source_metrics, consumer_metrics

    return _until(drained, managers)


def _prefix_end(prefix: bytes) -> bytes:
    end = bytearray(prefix)
    end[-1] += 1
    return bytes(end)


def _etcd_keys(endpoint: str, prefix: str):
    key = prefix.encode()
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(_prefix_end(key)).decode(),
            "keys_only": True,
        },
        timeout=5,
    )
    response.raise_for_status()
    return [base64.b64decode(row["key"]).decode() for row in response.json().get("kvs", [])]


def _payload(torch, pages, block_bytes, round_id):
    values = torch.arange(pages * block_bytes, device="cuda", dtype=torch.int64).reshape(
        pages, block_bytes
    )
    offsets = torch.arange(pages, device="cuda", dtype=torch.int64).unsqueeze(1) * 17
    return ((values + offsets + round_id * 31) % 251).to(torch.uint8).flatten()


def _hashes(medium, round_id, pages):
    return [
        hashlib.sha256(f"s2.8:{medium}:{round_id}:{block}".encode()).digest()
        for block in range(pages)
    ]


def _discover_storage_namespaces(tmp_path, identities, pages, block_bytes):
    import torch

    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    port = find_available_port()
    manager = CacheManagerProcess(
        port,
        pool_size="64mb",
        http_port=find_available_port(),
        bootstrap_socket=f"/tmp/orbitkv-s29-probe-{port}.sock",
        log_path=tmp_path / "namespace-probe-manager.log",
    )
    clients = []
    tensors = []
    try:
        assert manager.start(), manager.read_logs()
        client = CacheManagerClient(manager.bootstrap_socket)
        clients.append(client)
        for index, identity in enumerate(identities):
            instance = f"namespace-probe-{index}"
            tensor = torch.empty(pages * block_bytes, dtype=torch.uint8, device="cuda")
            tensors.append(tensor)
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
        pattern = re.compile(
            r"instance=(namespace-probe-\d+).*storage_namespace=(orbitkv:v2:[0-9a-f]{64})"
        )

        def discovered():
            found = dict(pattern.findall(manager.read_logs()))
            return found if len(found) == len(identities) else None

        found = _until(discovered, [manager])
        return [found[f"namespace-probe-{index}"] for index in range(len(identities))]
    finally:
        for client in clients:
            for instance in [f"namespace-probe-{item}" for item in range(len(identities))]:
                with contextlib.suppress(Exception):
                    client.unregister_context(instance)
            client.close()
        tensors.clear()
        manager.stop()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


@pytest.mark.parametrize("medium", ["dram", "ssd"])
def test_manager_inventory_stream_faults_preserve_exact_dram_and_ssd(tmp_path, monkeypatch, medium):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    monkeypatch.setenv("MC_FORCE_TCP", "1")
    pages, block_bytes = 8, 4096
    payload_bytes = pages * block_bytes
    cluster = f"s28-{medium}-{uuid.uuid4().hex[:12]}"
    namespace = f"s2.8:{medium}:{uuid.uuid4().hex}"
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
        etcd_gate = TcpGate(endpoint)
        stack.callback(etcd_gate.close)
        source_port = find_available_port()
        peer_gate = TcpGate(f"http://127.0.0.1:{source_port}")
        stack.callback(peer_gate.close)
        advertised_source = urlsplit(peer_gate.endpoint).netloc
        managers = []
        clients = []
        tensors = []
        for node in ("source", "consumer"):
            port = source_port if node == "source" else find_available_port()
            http_port = find_available_port()
            source = node == "source"
            ssd_enabled = source and medium == "ssd"
            manager_args = [
                "--etcd-endpoints",
                etcd_gate.endpoint if source else endpoint,
                "--node-id",
                node,
                "--cluster-name",
                cluster,
                "--membership-ttl-secs",
                "12",
                "--inventory-journal-bytes",
                "1024",
                "--inventory-stream-coalesce-ms",
                "2",
            ]
            if source:
                manager_args.extend(["--peer-advertise-addr", advertised_source])
            manager = CacheManagerProcess(
                port,
                pool_size="64mb",
                http_port=http_port,
                bootstrap_socket=f"/tmp/orbitkv-s28-{port}.sock",
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
        control_fence = _sync(source_manager)
        _await_fence(consumer_manager, control_fence, managers)
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
        _wait_for_remote_drain(source_manager, consumer_manager, managers)

        peer_gate.partition()
        partition_started = time.monotonic()
        final_hashes = final_payload = final_fence = None
        for round_id in range(1, 5):
            if medium == "dram":
                _cleanup_dram(source_manager)
            final_hashes, final_payload = save_round(round_id)
            final_fence = _sync(source_manager)
            miss = _query_ready(
                consumer_client,
                "consumer",
                final_hashes,
                f"partition-miss-{round_id}",
                0,
                managers,
            )
            assert not miss.lease
        source_status = _metadata(source_manager)
        assert source_status["index"]["registration_valid"]
        assert source_status["inventory_journal_bytes"] <= 1024
        assert source_status["inventory_journal_bytes_peak"] <= 1024
        peer_gate.heal()
        _await_fence(consumer_manager, final_fence, managers, timeout=20)
        _cleanup_dram(consumer_manager)
        _restore(
            consumer_client,
            "consumer",
            consumer_tensor,
            final_hashes,
            "recovered-after-overflow",
            final_payload,
            managers,
        )
        _wait_for_remote_drain(source_manager, consumer_manager, managers)
        recovered = _metadata(consumer_manager)
        assert recovered["index"]["coverage"] == "complete_at_watermarks"
        result["stream_repair"] = {
            "duration_ms": (time.monotonic() - partition_started) * 1000,
            "fence": final_fence,
            "source_history_gaps": _metadata(source_manager)["inventory_history_gaps"],
            "consumer_resets": recovered["stream"]["resets"],
        }

        if medium == "dram":
            _cleanup_dram(source_manager)
        expiry_hashes, expiry_payload = save_round(5)
        expiry_fence = _sync(source_manager)
        _await_fence(consumer_manager, expiry_fence, managers)
        _cleanup_dram(consumer_manager)
        remote_before_expiry = fetch_orbitkv_metrics(consumer_manager.http_port)[
            "orbitkv_remote_fetch_bytes_total"
        ]

        etcd_gate.partition()
        expiry_started = time.monotonic()
        _until(lambda: not _metadata(source_manager)["index"]["registration_valid"], managers)
        source_member_key = f"/orbitkv/v2/{cluster}/members/source"
        _until(
            lambda: source_member_key not in _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
            and _metadata(consumer_manager)["index"]["registration_valid"]
            and _metadata(consumer_manager)["index"]["coverage"] == "complete_at_watermarks",
            managers,
            timeout=25,
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
        etcd_gate.heal()
        time.sleep(1)
        assert not _metadata(source_manager)["index"]["registration_valid"]
        result["expiry"] = {
            "duration_ms": (time.monotonic() - expiry_started) * 1000,
            "source_incarnation": expiry_fence["source_incarnation"],
            "source_member_absent": True,
            "old_runtime_remained_fenced_after_heal": True,
            "remote_fetch_bytes_unchanged": remote_before_expiry,
        }

        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        result["etcd_keys"] = keys
        result["final_source_metrics"] = fetch_orbitkv_metrics(source_manager.http_port)
        result["final_consumer_metrics"] = fetch_orbitkv_metrics(consumer_manager.http_port)
        for node, client in zip(("source", "consumer"), clients, strict=True):
            ok, message = client.unregister_context(node)
            assert ok, message
        for client in clients:
            client.close()
        tensors.clear()
        del tensor
        source_tensor = consumer_tensor = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        (tmp_path / "result.json").write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("medium", ["dram", "ssd"])
def test_manager_scoped_inventory_keeps_outside_local_cache_and_exact_remote_bytes(
    tmp_path, monkeypatch, medium
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    monkeypatch.setenv("MC_FORCE_TCP", "1")
    pages, block_bytes = 8, 4096
    payload_bytes = pages * block_bytes
    selected_identity = f"s2.9:selected:{uuid.uuid4().hex}"
    outside_identity = f"s2.9:outside:{uuid.uuid4().hex}"
    selected_scope, outside_scope = _discover_storage_namespaces(
        tmp_path, [selected_identity, outside_identity], pages, block_bytes
    )
    assert selected_scope != outside_scope
    cluster = f"s29-{medium}-{uuid.uuid4().hex[:12]}"
    hashes = [hashlib.sha256(f"s2.9:shared:{block}".encode()).digest() for block in range(pages)]
    selected_payload = _payload(torch, pages, block_bytes, 90)
    outside_payload = _payload(torch, pages, block_bytes, 91)
    result = {
        "medium": medium,
        "cluster": cluster,
        "selected_scope": selected_scope,
        "outside_scope": outside_scope,
        "hashes": [value.hex() for value in hashes],
    }

    with ExitStack() as stack:
        endpoint, _ = stack.enter_context(etcd_server(tmp_path))
        managers = []
        clients = {}
        tensors = {}
        for node in ("source", "consumer"):
            port = find_available_port()
            source = node == "source"
            ssd_enabled = source and medium == "ssd"
            manager = CacheManagerProcess(
                port,
                pool_size="64mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s29-{port}.sock",
                ssd_cache_path=tmp_path / "scoped-source-ssd" if ssd_enabled else None,
                ssd_cache_capacity="64kb",
                ssd_backend="uring",
                ssd_read_path="uring" if ssd_enabled else None,
                log_path=tmp_path / f"scoped-{node}-manager.log",
                extra_args=(
                    "--etcd-endpoints",
                    endpoint,
                    "--node-id",
                    node,
                    "--cluster-name",
                    cluster,
                    "--membership-ttl-secs",
                    "60",
                    "--metadata-namespace",
                    selected_scope,
                    "--enable-prometheus",
                ),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)
            client = CacheManagerClient(manager.bootstrap_socket)
            stack.callback(client.close)
            clients[node] = client
            for label, identity in (("selected", selected_identity), ("outside", outside_identity)):
                instance = f"{node}-{label}"
                tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
                tensors[(node, label)] = tensor
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

        source_manager, consumer_manager = managers
        source_client = clients["source"]
        consumer_client = clients["consumer"]
        device = resolve_device_id()
        for label, payload in (("selected", selected_payload), ("outside", outside_payload)):
            tensor = tensors[("source", label)]
            tensor.copy_(payload)
            torch.cuda.synchronize()
            ok, message = source_client.save(
                f"source-{label}", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
            )
            assert ok, message
        if medium == "ssd":
            _wait_for_ssd_write(source_manager, payload_bytes * 2, managers)
            cleaned = _cleanup_dram(source_manager)
            assert cleaned["evicted_blocks"] == pages * 2
            assert cleaned["still_referenced_blocks"] == 0

        fence = _sync(source_manager)
        _await_fence(consumer_manager, fence, managers)
        source_owner = fence["source_incarnation"]
        owner = _owner_status(consumer_manager, source_owner)
        assert owner and owner["records"] == pages
        consumer_metadata = _metadata(consumer_manager)
        assert consumer_metadata["stream"]["scope_kind"] == "exact_namespaces"
        assert consumer_metadata["stream"]["scope_namespaces"] == 1
        assert consumer_metadata["index"]["scope_kind"] == "exact_namespaces"

        miss = _query_ready(
            consumer_client,
            "consumer-outside",
            hashes,
            "outside-remote-miss",
            0,
            managers,
        )
        assert not miss.lease
        selected_before = fetch_orbitkv_metrics(consumer_manager.http_port).get(
            "orbitkv_remote_fetch_bytes_total", 0
        )
        _restore(
            consumer_client,
            "consumer-selected",
            tensors[("consumer", "selected")],
            hashes,
            "selected-remote",
            selected_payload,
            managers,
        )
        selected_after = fetch_orbitkv_metrics(consumer_manager.http_port)[
            "orbitkv_remote_fetch_bytes_total"
        ]
        assert selected_after >= selected_before + payload_bytes
        _wait_for_remote_drain(source_manager, consumer_manager, managers)

        local = tensors[("consumer", "outside")]
        local.copy_(outside_payload)
        torch.cuda.synchronize()
        ok, message = consumer_client.save(
            "consumer-outside", 0, 0, device, [("kv:0", list(range(pages)), hashes)]
        )
        assert ok, message
        _restore(
            consumer_client,
            "consumer-outside",
            local,
            hashes,
            "outside-local",
            outside_payload,
            managers,
        )
        _cleanup_dram(consumer_manager)
        miss = _query_ready(
            consumer_client,
            "consumer-outside",
            hashes,
            "outside-after-local-eviction",
            0,
            managers,
        )
        assert not miss.lease
        assert (
            fetch_orbitkv_metrics(consumer_manager.http_port)["orbitkv_remote_fetch_bytes_total"]
            == selected_after
        )

        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        result.update(
            {
                "fence": fence,
                "owner_status": owner,
                "consumer_metadata": consumer_metadata,
                "etcd_keys": keys,
                "remote_fetch_bytes": selected_after,
                "outside_remote_rejected": True,
                "outside_local_exact": True,
            }
        )
        for node, client in clients.items():
            for label in ("selected", "outside"):
                ok, message = client.unregister_context(f"{node}-{label}")
                assert ok, message
        for client in clients.values():
            client.close()
        tensors.clear()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        (tmp_path / f"scoped-{medium}-result.json").write_text(json.dumps(result, indent=2) + "\n")
