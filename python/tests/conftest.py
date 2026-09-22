"""Shared command-line options for engine correctness gates."""

import os
import uuid

import pytest


@pytest.fixture(scope="module")
def model(request) -> str:
    return request.config.getoption("--model")


@pytest.fixture(scope="module")
def base_port(request) -> int:
    return request.config.getoption("--e2e-port")


@pytest.fixture(scope="module")
def orbitkv_metrics_port(request) -> int:
    return request.config.getoption("--orbitkv-metrics-port")


@pytest.fixture(scope="module")
def tensor_parallel_size(request) -> int:
    return request.config.getoption("--tensor-parallel-size")


@pytest.fixture(scope="module")
def pipeline_parallel_size(request) -> int:
    return request.config.getoption("--pipeline-parallel-size")


@pytest.fixture(scope="module")
def max_model_len(request) -> int | None:
    return request.config.getoption("--max-model-len")


@pytest.fixture(scope="module")
def orbitkv_transfer_backend(request) -> str:
    return request.config.getoption("--orbitkv-transfer-backend")


@pytest.fixture(scope="module")
def orbitkv_use_hugepages(request) -> bool:
    return request.config.getoption("--orbitkv-use-hugepages")


@pytest.fixture(scope="module")
def orbitkv_pool_size(request) -> str:
    return request.config.getoption("--orbitkv-pool-size")


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options for E2E tests."""
    parser.addoption(
        "--model",
        action="store",
        default="Qwen/Qwen3-0.6B",
        help="Model to use for E2E testing",
    )
    parser.addoption(
        "--e2e-port",
        action="store",
        default=8100,
        type=int,
        help="Base port for vLLM servers in E2E tests",
    )
    parser.addoption(
        "--orbitkv-metrics-port",
        action="store",
        default=9091,
        type=int,
        help="OrbitKV server metrics port for E2E tests",
    )
    parser.addoption(
        "--tensor-parallel-size",
        action="store",
        default=1,
        type=int,
        help="Tensor parallel size for vLLM servers in E2E tests (tp * pp <= 4)",
    )
    parser.addoption(
        "--pipeline-parallel-size",
        action="store",
        default=1,
        type=int,
        help="Pipeline parallel size for vLLM servers in E2E tests (tp * pp <= 4)",
    )
    parser.addoption(
        "--max-model-len",
        action="store",
        default=None,
        type=int,
        help="Max model length for vLLM servers (e.g. 16384 for large models on small GPUs)",
    )
    parser.addoption(
        "--orbitkv-transfer-backend",
        action="store",
        default="direct",
        choices=("direct", "kernel"),
        help="OrbitKV server H2D/D2H transfer backend for E2E tests",
    )
    parser.addoption(
        "--orbitkv-use-hugepages",
        action="store_true",
        default=False,
        help="Start Cache Manager with --use-hugepages for E2E tests",
    )
    parser.addoption(
        "--orbitkv-pool-size",
        action="store",
        default="30gb",
        help="OrbitKV server pinned memory pool size for E2E tests",
    )


@pytest.fixture
def channel_server(request, tmp_path):
    """Start an isolated Cache Manager with a known channel identity."""
    from tests.support.cache_manager import CacheManagerProcess, find_available_port

    service_name = f"orbitkv/test/python/{os.getpid()}/{uuid.uuid4().hex}"
    bootstrap_socket = f"/tmp/orbitkv-python-{os.getpid()}-{uuid.uuid4().hex}.sock"
    configuration = getattr(request, "param", "dram")
    mode = configuration["tier"] if isinstance(configuration, dict) else configuration
    pool_size = "256mb" if mode == "ssd" else "100mb"
    if isinstance(configuration, dict):
        pool_size = configuration.get("pool_size", pool_size)
    server = CacheManagerProcess(
        port=find_available_port(),
        pool_size=pool_size,
        query_budget="128kb" if mode == "budget" else None,
        query_instance_budget="64kb" if mode == "budget" else None,
        http_port=find_available_port(),
        channel_service=service_name,
        channel_session_epoch=0x0B17_17C0,
        bootstrap_socket=bootstrap_socket,
        ssd_cache_path=tmp_path / "cache.bin" if mode == "ssd" else None,
    )

    if not server._binary_path:
        pytest.skip("Cache Manager binary not found")
    if not server.start():
        logs = server.read_logs()
        server.stop()
        pytest.fail(f"Cache Manager failed to start:\n{logs}")

    yield server
    server.stop()
