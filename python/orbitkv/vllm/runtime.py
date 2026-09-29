"""Restore boundary for the pinned vLLM 0.29 GPU model runner."""

from functools import wraps


def install_restore_boundary() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
        MultiKVConnectorMetadata,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from orbitkv.vllm.connector import OrbitKVConnector

    update_requests = GPUModelRunner.update_requests
    if getattr(update_requests, "_orbitkv_restore_boundary", False):
        return

    def cache_workers(connector, metadata):
        if isinstance(connector, OrbitKVConnector):
            assert connector._worker is not None and metadata is not None
            yield connector._worker, metadata
        elif isinstance(connector, MultiConnector):
            if not isinstance(metadata, MultiKVConnectorMetadata):
                raise TypeError("MultiConnector requires its ordered child metadata")
            for child, child_metadata in zip(connector._connectors, metadata.metadata, strict=True):
                yield from cache_workers(child, child_metadata)

    @wraps(update_requests)
    def update_with_restore(runner, scheduler_output):
        active = runner.kv_connector
        connector = getattr(active, "kv_connector", None)
        if not isinstance(connector, (OrbitKVConnector, MultiConnector)) or active._disabled:
            return update_requests(runner, scheduler_output)

        workers = tuple(cache_workers(connector, scheduler_output.kv_connector_metadata))
        # Preempted saves must stop reading before the runner zeroes reused pages.
        for worker, metadata in workers:
            worker.handle_preemptions(metadata.preempted_req_ids)
        result = update_requests(runner, scheduler_output)
        if scheduler_output.has_sync_kv_loads and scheduler_output.total_num_scheduled_tokens:
            # update_requests has enqueued zeroing/COW. Restore must precede the
            # recurrent pre-copy in preprocess_state, which runs before pre_forward.
            for worker, metadata in workers:
                worker.start_load_kv(metadata)
        return result

    update_with_restore._orbitkv_restore_boundary = True
    GPUModelRunner.update_requests = update_with_restore
