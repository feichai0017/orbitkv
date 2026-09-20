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
    # Hybrid-cache loads carry one membership lease per recurrent storage
    # group (pinned checkpoints in hit-positions order) on top of the
    # attention prefix leases. See RecurrentLoadHold.
    recurrent_hold: "RecurrentLoadHold | None" = None


@dataclass(frozen=True)
class RecurrentLoadHold:
    """Pinned recurrent checkpoints for one hybrid external load.

    Indexed by ``sorted(recurrent_group_indices)`` on the outside and TP
    shard on the inside: ``leases[g][shard]`` is the membership lease over
    group ``g``'s hit blocks; ``hit_positions[g][shard]`` lists each leased
    block's position in the scheduler's query hash list (lease order).
    ``checkpoint`` is the chosen query position — the mamba state stored
    there covers all tokens through the end of that block (vLLM convention:
    state block ``i`` ends at token ``(i + 1) * block_size``), so the
    resumable prefix is ``checkpoint + 1`` blocks.
    """

    leases: tuple[tuple[bytes, ...], ...]
    hit_positions: tuple[tuple[tuple[int, ...], ...], ...]
    checkpoint: int


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
