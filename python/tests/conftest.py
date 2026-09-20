"""Pytest fixtures for OrbitKV connector integration tests.

Provides fixtures for automatically starting/stopping the Cache Manager
and test helpers for connector testing against a running server.
"""

import contextlib
import hashlib
import logging
import os
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch

# Import the GPU registration helper for integration tests.
try:
    from orbitkv.client.gpu import serialize_gpu_buffer
except ImportError:
    serialize_gpu_buffer = None

logger = logging.getLogger(__name__)


# =============================================================================
# Test Constants
# =============================================================================

DEFAULT_POOL_SIZE = "100mb"
SERVER_STARTUP_TIMEOUT = 60  # seconds (increased for slow GPU init)
SERVER_READY_CHECK_INTERVAL = 1.0  # seconds


# =============================================================================
# Utility Functions
# =============================================================================


def _torch():
    return pytest.importorskip("torch", reason="torch is required for GPU integration tests")


def find_available_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_cache_manager_binary() -> str | None:
    """
    Locate the Cache Manager binary.

    Search order:
    1. Installed orbitkv-cache-manager-py in package directory
    2. cargo target/release/orbitkv-cache-manager
    3. cargo target/debug/orbitkv-cache-manager
    """
    if configured := os.environ.get("ORBITKV_CACHE_MANAGER_BINARY"):
        binary = Path(configured).resolve()
        if not binary.is_file():
            raise FileNotFoundError(binary)
        return str(binary)

    # 1. Check installed package binary
    try:
        from orbitkv._cache_manager import get_cache_manager_binary

        binary = get_cache_manager_binary()
        if Path(binary).exists():
            return binary
    except ImportError:
        pass

    # 2. Check cargo build outputs
    project_root = Path(__file__).parent.parent.parent  # python/tests -> orbitkv
    for build_type in ["release", "debug"]:
        cargo_binary = project_root / "target" / build_type / "orbitkv-cache-manager"
        if cargo_binary.exists():
            return str(cargo_binary)

    return None


def wait_for_server_ready(
    bootstrap_socket: str,
    timeout: float = SERVER_STARTUP_TIMEOUT,
    process: subprocess.Popen | None = None,
) -> bool:
    """Wait for the local Cache Manager to accept lifecycle connections."""
    # Import directly from submodule to avoid triggering __init__.py imports (vllm dependency)
    import importlib

    orbitkv_module = importlib.import_module("orbitkv.orbitkv")

    start_time = time.time()
    last_error = None
    while time.time() - start_time < timeout:
        if process is not None and process.poll() is not None:
            return False
        try:
            client = orbitkv_module.LocalQueryClient(bootstrap_socket)
            ok, _ = client.health()
            client.close()
            if ok:
                return True
        except Exception as e:
            last_error = str(e)
        time.sleep(SERVER_READY_CHECK_INTERVAL)

    if last_error:
        print(f"Last health check error: {last_error}", file=sys.stderr)
    return False


# =============================================================================
# Test Helpers: ClientContext
# =============================================================================


def initialize_kv_cache(
    device: "torch.device",
    num_blocks: int = 64,
    num_layers: int = 1,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: "torch.dtype | None" = None,
) -> list["torch.Tensor"]:
    """
    Initialize KV cache tensors on GPU for testing.

    Creates tensors in KV-first layout: (2, num_blocks, block_size, num_heads, head_size)
    where the first dimension is [K, V].
    """
    torch = _torch()
    if dtype is None:
        dtype = torch.bfloat16

    torch.random.manual_seed(42)

    gpu_tensors = [
        torch.rand(
            (2, num_blocks, block_size, num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    return gpu_tensors


class ClientContext:
    """
    Client context that represents a vLLM instance.

    This class abstracts a vLLM instance by managing:
    - GPU KV cache tensors (like WorkerConnector)
    - Query operations (like SchedulerConnector)
    - Context registration/unregistration
    """

    def __init__(
        self,
        engine_client,
        instance_id: str,
        namespace: str,
        device_id: int = 0,
        num_blocks: int = 64,
        num_layers: int = 1,
        block_size: int = 16,
        num_heads: int = 8,
        head_size: int = 128,
        dtype: "torch.dtype | None" = None,
    ):
        torch = _torch()
        if dtype is None:
            dtype = torch.bfloat16
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"device_id {device_id} >= available GPUs ({torch.cuda.device_count()})"
            )

        self.engine_client = engine_client
        self.instance_id = instance_id
        self.namespace = namespace
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        # Initialize KV cache tensors
        self.gpu_kv_caches = initialize_kv_cache(
            self.device, num_blocks, num_layers, block_size, num_heads, head_size, dtype
        )

        # Map layer index to layer name (for compatibility with vLLM)
        self._layer_names = [f"layer_{i}" for i in range(num_layers)]
        self._registered = False

    def register_kv_caches(self) -> None:
        """Register KV cache tensors with the engine server (like WorkerConnector.register_kv_caches)."""
        if serialize_gpu_buffer is None:
            raise RuntimeError("GPU registration helper not available")

        if self._registered:
            return

        kv_caches = {name: self.gpu_kv_caches[i] for i, name in enumerate(self._layer_names)}

        layer_names: list[str] = []
        wrapper_bytes_list: list[bytes] = []
        num_blocks_list: list[int] = []
        bytes_per_block_list: list[int] = []
        kv_stride_bytes_list: list[int] = []
        segments_list: list[int] = []

        for layer_name, kv_cache in kv_caches.items():
            if not kv_cache.is_contiguous():
                kv_cache = kv_cache.contiguous()
            wrapper_bytes = serialize_gpu_buffer(kv_cache)

            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            element_size = kv_cache.element_size()

            if len(shape) >= 2 and shape[0] == 2:
                num_blocks = shape[1]
                bytes_per_block = stride[1] * element_size
                kv_stride_bytes = stride[0] * element_size
                segments = 2
            else:
                num_blocks = shape[0]
                bytes_per_block = stride[0] * element_size
                kv_stride_bytes = 0
                segments = 1

            layer_names.append(layer_name)
            wrapper_bytes_list.append(wrapper_bytes)
            num_blocks_list.append(num_blocks)
            bytes_per_block_list.append(bytes_per_block)
            kv_stride_bytes_list.append(kv_stride_bytes)
            segments_list.append(segments)

        # One batch per device, like the production worker: the engine seals
        # the instance topology once all world_size devices have registered.
        ok, message = self.engine_client.register_context_batch(
            self.instance_id,
            self.namespace,
            0,  # tp_rank
            0,  # pp_rank
            1,  # tp_size
            1,  # world_size
            self.device_id,
            layer_names,
            wrapper_bytes_list,
            num_blocks_list,
            bytes_per_block_list,
            kv_stride_bytes_list,
            segments_list,
            "direct",
            False,
        )

        if not ok:
            raise RuntimeError(f"Register context failed for {layer_names}: {message}")

        self._registered = True

    def unregister_context(self) -> None:
        """Unregister context from server (like WorkerConnector.unregister_context)."""
        if not self._registered:
            return

        try:
            ok, message = self.engine_client.unregister_context(self.instance_id)
            if not ok:
                logger.warning(f"Unregister context failed: {message}")
        except Exception as e:
            logger.warning(f"Unregister context exception: {e}")

        self._registered = False

    def query(self, block_hashes: list[bytes]) -> dict:
        """Query available blocks (like SchedulerConnector._count_available_block_prefix).

        Args:
            block_hashes: List of block hashes to query

        Returns:
            Query result dict
        """
        return self.engine_client.query_prefetch(self.instance_id, block_hashes, req_id="test")

    def get_kv_cache(self, layer: int = 0) -> "torch.Tensor":
        """Get KV cache tensor for a specific layer."""
        return self.gpu_kv_caches[layer]

    def get_tensor_slice(self, layer: int, start_block: int, num_blocks: int) -> "torch.Tensor":
        """Get a slice of the KV cache tensor for a specific layer."""
        return self.gpu_kv_caches[layer][:, start_block : start_block + num_blocks]


# =============================================================================
# Fixtures: Test Data
# =============================================================================


@pytest.fixture
def instance_id() -> str:
    """Generate a unique instance ID for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def namespace() -> str:
    """Generate a test namespace."""
    return "test_namespace"


@pytest.fixture
def block_hashes() -> list[bytes]:
    """Generate deterministic block hashes for testing."""
    hashes = []
    for i in range(20):
        content = f"test_block_{i}".encode()
        hash_bytes = hashlib.sha256(content).digest()
        hashes.append(hash_bytes)
    return hashes


# =============================================================================
# Fixtures: Server Management
# =============================================================================


class CacheManagerProcess:
    """Manages a Cache Manager subprocess for testing."""

    def __init__(
        self,
        port: int,
        pool_size: str = DEFAULT_POOL_SIZE,
        devices: str = "0",
        *,
        http_port: int | None = None,
        local_control_service: str | None = None,
        local_control_session_epoch: int | None = None,
        local_bootstrap_socket: str | None = None,
    ):
        self.port = port
        self.pool_size = pool_size
        self.devices = devices
        self.http_port = http_port
        self.local_control_service = local_control_service
        self.local_control_session_epoch = local_control_session_epoch
        self.local_bootstrap_socket = local_bootstrap_socket or f"/tmp/orbitkv-{port}.sock"
        self.process: subprocess.Popen | None = None
        self._binary_path = find_cache_manager_binary()
        self._log_path: Path | None = None
        self._log_file = None

    def start(self) -> bool:
        """Start the server process. Returns True if successful."""
        if not self._binary_path:
            return False

        env = os.environ.copy()
        env["PYO3_PYTHON"] = sys.executable
        env["PYTHONHOME"] = sys.base_prefix

        # Add libpython to LD_LIBRARY_PATH if available
        if libdir := sysconfig.get_config_var("LIBDIR"):
            env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"

        # Set PYTHONPATH to include python package and venv site-packages
        python_dir = Path(__file__).parent.parent
        site_packages = [
            path
            for path in dict.fromkeys(
                [
                    sysconfig.get_path("purelib"),
                    *(path for path in sys.path if "site-packages" in path),
                ]
            )
            if path
        ]
        env["PYTHONPATH"] = ":".join([str(python_dir), *site_packages])

        cmd = [
            self._binary_path,
            "--addr",
            f"127.0.0.1:{self.port}",
            "--pool-size",
            self.pool_size,
            "--devices",
            self.devices,
        ]
        if self.http_port is not None:
            cmd.extend(["--http-addr", f"127.0.0.1:{self.http_port}"])
        if self.local_control_service is not None:
            cmd.extend(["--local-control-service", self.local_control_service])
        if self.local_control_session_epoch is not None:
            cmd.extend(
                [
                    "--local-control-session-epoch",
                    str(self.local_control_session_epoch),
                ]
            )
        cmd.extend(["--local-bootstrap-socket", self.local_bootstrap_socket])

        # Route logs to a tempfile so the pipe buffer cannot fill up and
        # block the server mid-startup, and so tests can read the log
        # contents via read_logs() (used by integration tests that assert
        # on server-side log signals).
        fd, path = tempfile.mkstemp(prefix=f"orbitkv-cache-manager-{self.port}-", suffix=".log")
        self._log_path = Path(path)
        self._log_file = os.fdopen(fd, "wb")

        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                cwd="/tmp",
                preexec_fn=os.setsid,
            )
        except (FileNotFoundError, PermissionError):
            self._close_log()
            return False

        return wait_for_server_ready(self.local_bootstrap_socket, process=self.process)

    def stop(self) -> None:
        """Stop the server process."""
        if self.process is None:
            self._close_log()
            return
        try:
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=2)
        except (ProcessLookupError, OSError):
            self.process.wait(timeout=2)
        finally:
            self.process = None
            self._close_log()
            if self.local_bootstrap_socket is not None:
                with contextlib.suppress(OSError):
                    Path(self.local_bootstrap_socket).unlink()

    def is_running(self) -> bool:
        """Check if server process is still running."""
        return self.process is not None and self.process.poll() is None

    def read_logs(self) -> str:
        """Return current server log contents (stdout + stderr merged)."""
        if not self._log_path or not self._log_path.exists():
            return ""
        return self._log_path.read_text(errors="replace")

    def _close_log(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None
        if self._log_path and self._log_path.exists():
            with contextlib.suppress(OSError):
                self._log_path.unlink()


@pytest.fixture(scope="session")
def orbitkv_server() -> Generator[CacheManagerProcess, None, None]:
    """Session-scoped fixture that starts a Cache Manager for integration tests."""
    port = find_available_port()
    server = CacheManagerProcess(port=port)

    if not server.start() or not server._binary_path:
        pytest.skip("Cache Manager binary not found or failed to start")

    yield server
    server.stop()


@pytest.fixture
def local_control_server() -> Generator[CacheManagerProcess, None, None]:
    """Start an isolated server with a known local-control identity."""
    service_name = f"orbitkv/test/python/{os.getpid()}/{uuid.uuid4().hex}"
    bootstrap_socket = f"/tmp/orbitkv-python-{os.getpid()}-{uuid.uuid4().hex}.sock"
    server = CacheManagerProcess(
        port=find_available_port(),
        http_port=find_available_port(),
        local_control_service=service_name,
        local_control_session_epoch=0x0B17_17C0,
        local_bootstrap_socket=bootstrap_socket,
    )

    if not server._binary_path:
        pytest.skip("Cache Manager binary not found")
    if not server.start():
        logs = server.read_logs()
        server.stop()
        pytest.fail(f"Cache Manager failed to start:\n{logs}")

    yield server
    server.stop()


@pytest.fixture
def local_control_client_context(
    local_control_server: CacheManagerProcess, instance_id: str, namespace: str
) -> Generator[ClientContext, None, None]:
    """Register a minimal GPU context on the isolated local-control server."""
    import importlib

    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    ctx = ClientContext(
        engine_client=orbitkv_native.LocalQueryClient(local_control_server.local_bootstrap_socket),
        instance_id=instance_id,
        namespace=namespace,
        device_id=0,
        num_blocks=4,
        num_layers=1,
    )
    ctx.register_kv_caches()
    yield ctx
    if local_control_server.is_running():
        ctx.unregister_context()


@pytest.fixture
def engine_client(orbitkv_server: CacheManagerProcess):
    """Create a local Cache Manager client for integration tests."""
    from orbitkv.client.data_plane import LocalDataClient

    client = LocalDataClient(orbitkv_server.local_bootstrap_socket)
    yield client
    client.close()


@pytest.fixture
def client_context(
    engine_client, instance_id: str, namespace: str
) -> Generator[ClientContext, None, None]:
    """Fixture that provides a ClientContext representing a vLLM instance.

    Args:
        engine_client: LocalDataClient connected to the Cache Manager
        instance_id: Unique instance identifier
        namespace: Namespace for the instance

    Returns:
        ClientContext instance (automatically registered)
    """
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    ctx = ClientContext(
        engine_client=engine_client,
        instance_id=instance_id,
        namespace=namespace,
        device_id=0,
    )

    # Auto-register on creation
    ctx.register_kv_caches()

    yield ctx

    # Cleanup
    ctx.unregister_context()
    del ctx.gpu_kv_caches
    torch.cuda.empty_cache()


@pytest.fixture
def registered_instance(client_context: ClientContext) -> Generator[str, None, None]:
    """Fixture that returns the instance_id of a registered ClientContext.

    The client_context fixture already handles registration/unregistration.
    This fixture just provides the instance_id for convenience.
    """
    yield client_context.instance_id


# =============================================================================
# Fixtures: E2E shared options
# =============================================================================


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


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require Cache Manager with GPU)",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (require vLLM + OrbitKV)",
    )
