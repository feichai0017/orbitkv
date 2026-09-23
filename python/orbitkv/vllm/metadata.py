"""Scheduler/worker transfer intents and completion metadata."""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""

    block_ids_by_group: tuple[tuple[int | None, ...], ...]
    leases: tuple[bytes, ...]
    num_tokens: int
    # Hybrid loads carry the exact window/checkpoint leases separately from
    # the full-attention prefix lease.
    recovery_hold: "RecoveryLoadHold | None" = None


@dataclass(frozen=True)
class RecoveryLoadHold:
    """Pinned auxiliary groups for one hybrid external load.

    Indexed by nonzero storage-group order on the outside and TP
    shard on the inside: ``leases[g][shard]`` is the membership lease over
    group ``g``'s hit blocks; ``hit_positions[g][shard]`` lists each leased
    block's position in the scheduler's query hash list (lease order).
    ``last_position`` is the last position at the chosen recovery boundary.
    A window lease contains its trailing pages; recurrent state needs only
    that final position. Its absolute end
    is ``(computed_blocks + last_position + 1) * block_size``; the scheduler
    validates that boundary, while the worker addresses the lease by its
    query-relative position. The externally restored span is ``last_position + 1`` blocks.
    """

    leases: tuple[tuple[bytes, ...], ...]
    hit_positions: tuple[tuple[tuple[int, ...], ...], ...]
    last_position: int


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[bytes, ...]


class OrbitKVConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
        boundary_save_intents: dict[int, SaveIntent] | None = None,
        preempted_req_ids: set[str] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}
        # HMA: recurrent boundary states handed off by vLLM this step, keyed
        # by a scheduler-issued job id. Their blocks are pinned by the
        # scheduler until every worker reports the job through
        # OrbitKVWorkerMetadata, so they are decoupled from request lifetimes.
        self.boundary_save_intents: dict[int, SaveIntent] = boundary_save_intents or {}
        self.preempted_req_ids: set[str] = preempted_req_ids or set()

    def __repr__(self) -> str:
        return (
            f"OrbitKVConnectorMetadata(loads={len(self.load_intents)}, "
            f"saves={len(self.save_intents)}, "
            f"boundary_saves={len(self.boundary_save_intents)})"
        )


@dataclass
class OrbitKVWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> scheduler completion report for boundary-state save jobs.

    ``completed_boundary_jobs`` maps a job id to the number of workers that
    finished it (successfully or not). vLLM aggregates one instance per
    worker before the scheduler sees it.
    """

    completed_boundary_jobs: dict[int, int]

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "OrbitKVWorkerMetadata":
        if not isinstance(other, OrbitKVWorkerMetadata):
            raise TypeError(f"cannot aggregate {type(other).__name__} into OrbitKVWorkerMetadata")
        for job_id, count in other.completed_boundary_jobs.items():
            self.completed_boundary_jobs[job_id] = (
                self.completed_boundary_jobs.get(job_id, 0) + count
            )
        return self
