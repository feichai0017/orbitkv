"""Type stubs for the orbitkv Rust extension module (PyO3 bindings).

This module provides local Cache Manager bindings for LLM inference.
"""

__version__: str

# Custom exceptions for error classification

class OrbitKVError(Exception):
    """Base exception for all OrbitKV errors."""

    ...

class OrbitKVInternal(OrbitKVError):
    """Internal server error."""

    ...

class MooncakeTransferEngine:
    def __init__(self, *, bind_host: str, nics: list[str] = ...) -> None: ...
    @property
    def endpoint(self) -> str: ...
    def register_memory(self, regions: list[dict[str, int | str]]) -> None: ...
    def unregister_memory(self, addresses: list[int]) -> None: ...
    def write(
        self,
        remote_endpoint: str,
        slices: list[tuple[int, int, int]],
        timeout_s: float = 30.0,
        notify_name: str | None = None,
        notify_message: str | None = None,
    ) -> int: ...
    def read(
        self,
        remote_endpoint: str,
        slices: list[tuple[int, int, int]],
        timeout_s: float = 30.0,
    ) -> int: ...
    def send_notification(self, remote_endpoint: str, name: str, message: str) -> None: ...
    def take_notifications(self) -> list[tuple[str, str]]: ...
    def invalidate_segment(self, remote_endpoint: str) -> None: ...

class QueryLoading:
    def __init__(self) -> None: ...

class QueryReady:
    num_hit_blocks: int
    lease: bytes
    hit_positions: list[int]
    def __init__(
        self,
        num_hit_blocks: int,
        lease: bytes,
        hit_positions: list[int] = ...,
    ) -> None: ...

class ChannelProbeClient:
    """Low-level iceoryx2 probe for Cache Manager channel diagnostics."""

    def __init__(
        self,
        service_name: str,
        session_epoch: int,
        timeout_ms: int = 5000,
        spin_iterations: int = 64,
    ) -> None: ...
    @property
    def service_name(self) -> str: ...
    @property
    def session_epoch(self) -> int: ...
    def ping(self, value: int = 0, request_id: int = 1) -> int: ...
    def shutdown(self, request_id: int = 1) -> None: ...

class ChannelClient:
    """UDS-bootstrapped iceoryx2 client for framework-neutral cache operations."""

    def __init__(
        self,
        bootstrap_socket: str,
        timeout_ms: int = 5000,
        spin_iterations: int = 64,
    ) -> None: ...
    def close(self) -> None: ...
    def health(self) -> tuple[bool, str]: ...
    def unregister_context(self, instance_id: str) -> tuple[bool, str]: ...
    def start_session_watcher(
        self, instance_id: str, namespace: str, tp_size: int, world_size: int
    ) -> None: ...
    def register_context_batch(
        self,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        pp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        layer_names: list[str],
        wrapper_bytes_list: list[bytes],
        num_blocks_list: list[int],
        bytes_per_block_list: list[int],
        kv_stride_bytes_list: list[int],
        segments_list: list[int],
        transfer_backend: str,
        page_first: bool,
        layer_group_ids: list[int] | None = None,
    ) -> tuple[bool, str]: ...
    @property
    def service_name(self) -> str: ...
    @property
    def session_epoch(self) -> int: ...
    @property
    def notification_fd(self) -> int: ...
    def query_bundle(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
        wait_for_full_prefix: bool = False,
        group_id: int = 0,
        request_id: int = 1,
    ) -> QueryLoading | QueryReady: ...
    def release(self, lease: bytes, request_id: int = 1) -> None: ...
    def cancel_query(
        self, instance_id: str, req_id: str, group_id: int = 0, request_id: int = 1
    ) -> None: ...
    def publish(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
        request_id: int = 1,
    ) -> None: ...
    def restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups: list[list[str]],
        loads: list[tuple[bytes, list[list[int | None]]]],
        timeout_ms: int = 5000,
        request_id: int = 1,
    ) -> None: ...
    def restore_submit(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups: list[list[str]],
        loads: list[tuple[bytes, list[list[int | None]]]],
        request_id: int = 1,
    ) -> int: ...
    def restore_poll(self, operation_id: int, request_id: int = 1) -> tuple[str, str]: ...
