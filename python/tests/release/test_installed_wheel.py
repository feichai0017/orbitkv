"""Release gate: use installed plugins and binaries without source-tree imports."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import find_available_port
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [pytest.mark.release_smoke, pytest.mark.gpu]


@contextlib.contextmanager
def service(command, url, env, directory, name):
    log_path = directory / f"{name}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"{name} exited: {log_path.read_text()[-8000:]}")
            try:
                if requests.get(f"{url}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            pytest.fail(f"{name} startup timed out: {log_path.read_text()[-8000:]}")
        yield
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_installed_wheel_recovers_after_engine_restart(engine, model, tmp_path):
    assert Path(model).is_dir(), "release smoke requires --model with a local dense model"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "ORBITKV_"))
    }
    # Do not let the source checkout's staged libraries satisfy wheel dependencies.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        part
        for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if part and "/.orbitkv/" not in part and not part.endswith("/python/orbitkv")
    )
    env.update(
        ORBITKV_CACHE_SCOPE=tmp_path.name,
        VLLM_BATCH_INVARIANT="1",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        VLLM_CACHE_ROOT=str(tmp_path / "vllm"),
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import importlib.metadata as m, json, pathlib, sysconfig, orbitkv; "
            "p=pathlib.Path(orbitkv.__file__).resolve(); "
            "assert p.is_relative_to(pathlib.Path(sysconfig.get_path('purelib')).resolve()), p; "
            "d=m.distribution('orbitkv-llm-cu13' if m.packages_distributions()"
            "['orbitkv']==['orbitkv-llm-cu13'] else 'orbitkv-llm'); "
            "assert not json.loads(d.read_text('direct_url.json') or '{}')"
            ".get('dir_info',{}).get('editable',False); "
            "assert orbitkv.__version__==d.version; "
            "print(json.dumps({'package':str(p),'version':d.version}))",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    manager = str(Path(sys.executable).parent / "orbitkv-cache-manager")
    subprocess.run([manager, "--help"], cwd=tmp_path, env=env, check=True, capture_output=True)
    port, http_port, engine_port = (find_available_port() for _ in range(3))
    env["ORBITKV_PORT"] = str(port)
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
    manager_url = f"http://127.0.0.1:{http_port}"
    engine_url = f"http://127.0.0.1:{engine_port}"
    manager_command = [
        manager,
        "--addr",
        f"127.0.0.1:{port}",
        "--http-addr",
        f"127.0.0.1:{http_port}",
        "--pool-size",
        "1gb",
    ]
    if engine == "vllm":
        command = [
            sys.executable,
            "-I",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.4",
            "--enforce-eager",
            "--enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "OrbitKVConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "orbitkv.vllm",
                }
            ),
        ]
    else:
        command = [
            sys.executable,
            "-I",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--nccl-port",
            str(find_available_port()),
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--mem-fraction-static",
            "0.4",
            "--page-size",
            "64",
            "--disable-cuda-graph",
            "--enable-deterministic-inference",
            "--enable-unified-cache-external-linker",
            "--radix-cache-backend",
            "orbitkv",
        ]
    prompt = "The cache retains reusable prefixes for later inference requests. " * 64
    outputs = []
    with service(manager_command, manager_url, env, tmp_path, "manager"):
        for incarnation in range(2):
            with service(command, engine_url, env, tmp_path, f"{engine}-{incarnation}"):
                requests.get(f"{engine_url}/v1/models", timeout=10).raise_for_status()
                if engine == "vllm":
                    response = requests.post(
                        f"{engine_url}/v1/completions",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": 8,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                        timeout=90,
                    )
                else:
                    response = requests.post(
                        f"{engine_url}/generate",
                        json={
                            "text": prompt,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8,
                                "ignore_eos": True,
                            },
                        },
                        timeout=90,
                    )
                response.raise_for_status()
                result = response.json()
                outputs.append(result["choices"][0]["text"] if engine == "vllm" else result["text"])
                deadline = time.monotonic() + 10
                while True:
                    metrics = fetch_orbitkv_metrics(http_port)
                    counter = (
                        "orbitkv_save_bytes_total"
                        if incarnation == 0
                        else "orbitkv_load_bytes_total"
                    )
                    if metrics.get(counter, 0) > 0:
                        break
                    assert time.monotonic() < deadline, metrics
                    time.sleep(0.1)
        assert outputs[0] == outputs[1], outputs
        assert metrics.get("orbitkv_cache_block_hits_total", 0) > 0, metrics
        assert metrics.get("orbitkv_hll_total_requests", 0) > 0, metrics
        deadline = time.monotonic() + 10
        while True:
            metrics = fetch_orbitkv_metrics(http_port)
            if not metrics.get("orbitkv_query_reserved_bytes", 0) and not metrics.get(
                "orbitkv_inflight_bytes", 0
            ):
                break
            assert time.monotonic() < deadline, metrics
            time.sleep(0.1)
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    **json.loads(probe.stdout),
                    "engine": engine,
                    "model": model,
                    "save_bytes": metrics["orbitkv_save_bytes_total"],
                    "load_bytes": metrics["orbitkv_load_bytes_total"],
                    "final_query_bytes": metrics.get("orbitkv_query_reserved_bytes", 0),
                },
                indent=2,
            )
            + "\n"
        )
