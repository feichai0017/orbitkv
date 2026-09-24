"""Same-host TP=1 serving qualification; not a physical two-host/RDMA benchmark.

Trigger: peer transfer, catalog recovery, source ownership or adapter changes.
Requires ETCD_BIN, a prebuilt Manager, one GPU and the selected engine environment.
"""

import contextlib
import json
import os
import random
from argparse import Namespace
from pathlib import Path

import pytest

from tests.support.cluster import etcd_server

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_shared_cache_serving_and_restart(engine, request, tmp_path, monkeypatch):
    pytest.importorskip(engine)
    if not os.environ.get("ETCD_BIN") or not os.environ.get("ORBITKV_CACHE_MANAGER_BINARY"):
        pytest.skip("set ETCD_BIN and ORBITKV_CACHE_MANAGER_BINARY; no builds during runtime gates")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    from transformers import AutoTokenizer

    from benches.launch import configure
    from benches.metrics import metrics
    from benches.runtime import server
    from benches.shared_cache import drain, qualify, synchronize, verify_restore
    from benches.workload import evict_host_cache, generate

    model = Path(request.config.getoption("--model"))
    config = json.loads((model / "config.json").read_text())
    assert config["model_type"] == "qwen3", "this gate qualifies dense Qwen3"
    bpt = 4 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    vocabulary = AutoTokenizer.from_pretrained(model, local_files_only=True).encode(
        "A shared cache keeps completed model state available to independent replicas. ",
        add_special_tokens=False,
    )
    rng = random.Random(20260923)
    prompts = [[rng.choice(vocabulary) for _ in range(length)] for length in (513, 1025)]
    report = []
    with contextlib.ExitStack() as resources:
        endpoint, _ = resources.enter_context(etcd_server(tmp_path))
        launches, managers, engines = {}, {}, {}
        starts = {"source": 0, "consumer": 0}
        engine_starts = {"source": 0, "consumer": 0}
        for node in starts:
            output = tmp_path / node
            output.mkdir()
            args = Namespace(
                engine=engine,
                backend="orbitkv",
                model=model,
                output=output,
                host_gib=2,
                gpu_tokens=4096,
                prefill_tokens=4096,
                workload="serial",
                ssd_gib=0,
                storage_codec="none",
                storage_codec_budget=64 * 1024**2,
                cache_protected_percent=0,
                ssd_write_policy="all",
                query_budget_gib=1,
                queue_warmup="off",
                prepare_requests="off",
                trace_transfers=False,
                read_batch_mib=32,
                read_timeout_ms=0,
                read_max_batches=0,
                orbitkv_transfer_backend=None,
            )
            launch = configure(args, bpt)
            launch.env["MC_FORCE_TCP"] = "1"
            launch.manager_command.extend(
                [
                    "--etcd-endpoints",
                    endpoint,
                    "--node-id",
                    node,
                    "--catalog-nodes",
                    "consumer",
                    "--membership-ttl-secs",
                    "12",
                ]
            )
            if engine == "vllm":
                launch.env["VLLM_BATCH_INVARIANT"] = "1"
                launch.command.extend(["--gpu-memory-utilization", "0.35", "--enforce-eager"])
            else:
                launch.command.extend(
                    [
                        "--mem-fraction-static",
                        "0.35",
                        "--enable-deterministic-inference",
                    ]
                )
            launches[node] = launch
            managers[node] = resources.enter_context(contextlib.ExitStack())
            engines[node] = resources.enter_context(contextlib.ExitStack())

        def start(node, *, with_engine=True):
            launch = launches[node]
            run = starts[node]
            starts[node] += 1
            managers[node].enter_context(
                server(
                    launch.manager_command,
                    launch.env,
                    launch.manager_url,
                    tmp_path / f"{node}-manager-{run}.log",
                )
            )
            if with_engine:
                start_engine(node)

        def start_engine(node):
            launch = launches[node]
            run = engine_starts[node]
            engine_starts[node] += 1
            engines[node].enter_context(
                server(
                    launch.command,
                    launch.env,
                    launch.base_url,
                    tmp_path / f"{node}-engine-{run}.log",
                )
            )

        # Catalog must be reachable before the source completes registration.
        start("consumer")
        start("source")
        source, target = launches["source"], launches["consumer"]
        report.extend(
            qualify(
                engine=engine,
                model=str(model),
                source_url=source.base_url,
                target_url=target.base_url,
                source_manager=source.manager_url,
                target_manager=target.manager_url,
                prompts=prompts,
            )
        )
        expected = generate(source.base_url, engine, str(model), prompts[0], 8)["text"]
        drain(source.manager_url, target.manager_url)

        # Restart the sole catalog host and its empty consumer cache. Source KV survives.
        engines["consumer"].close()
        managers["consumer"].close()
        start("consumer")
        synchronize(source.manager_url)
        before = metrics(target.manager_url)
        recovered = generate(target.base_url, engine, str(model), prompts[0], 8)
        _, after = drain(source.manager_url, target.manager_url)
        row = verify_restore(before, after, expected, recovered)
        report.append({"case": "catalog_restart", **row, "resources_drained": True})

        # Remove all consumer copies and restart the source with no payload.
        engines["consumer"].close()
        evict_host_cache(target.manager_url)
        synchronize(target.manager_url)
        engines["source"].close()
        managers["source"].close()
        start("source", with_engine=False)
        start_engine("consumer")
        before = metrics(target.manager_url)
        fallback = generate(target.base_url, engine, str(model), prompts[0], 8)
        _, after = drain(source.manager_url, target.manager_url)
        assert fallback["text"] == expected
        assert after.get("orbitkv_remote_fetch_bytes_total", 0) == before.get(
            "orbitkv_remote_fetch_bytes_total", 0
        )
        assert after.get("orbitkv_load_bytes_total", 0) == before.get("orbitkv_load_bytes_total", 0)
        report.append(
            {
                "case": "source_restart_miss",
                "output_match": True,
                "remote_bytes": 0,
                "h2d_bytes": 0,
                "resources_drained": True,
                "ttft_ms": fallback["ttft_ms"],
            }
        )
    (tmp_path / "summary.json").write_text(
        json.dumps({"engine": engine, "deployment": "same-host-tcp", "results": report}, indent=2)
        + "\n"
    )
