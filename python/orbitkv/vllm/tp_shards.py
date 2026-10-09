"""Query and lease coordination for node-local TP shards."""

from dataclasses import dataclass

from orbitkv import BlockHashes, CacheManagerClient, QueryCandidates, RecoveryContract
from orbitkv.logging_utils import get_connector_logger
from orbitkv.orbitkv import QueryLoading, QueryReady
from orbitkv.vllm.metadata import RecoveryLoadHold

logger = get_connector_logger()


@dataclass(frozen=True, slots=True)
class ShardedQueryReady:
    num_hit_blocks: int
    leases: tuple[bytes, ...]
    control_id: bytes = b""
    # HMA only: per auxiliary group, per shard membership leases and their
    # hit positions (see RecoveryLoadHold for the wire/load contract).
    recovery_hold: RecoveryLoadHold | None = None
    # HMA only: the selected absolute boundary with complete leased state.
    boundary: int = 0
    # HMA only: the attention-only prefix hit before recovery validation
    # shrank it. Tells the scheduler where a shared prefix ends without a
    # usable recurrent checkpoint (see SchedulerAdapter's junction hint).
    attention_hit_blocks: int = 0


class TpShardQueryClient:
    def __init__(self, clients: tuple[CacheManagerClient, ...]):
        self._clients = clients

    def candidates(
        self, instance_id: str, hashes: BlockHashes, req_id: str, group_id: int
    ) -> list[tuple[int, ...]] | None:
        results = []
        for client in self._clients:
            result = client.query_candidates(instance_id, hashes, req_id, group_id=group_id)
            if isinstance(result, QueryLoading):
                return None
            if not isinstance(result, QueryCandidates):
                raise TypeError("candidate discovery returned a payload result")
            results.append(tuple(result.hit_positions))
        return results

    def read_recovery(
        self,
        instance_id: str,
        hashes: BlockHashes,
        req_id: str,
        contract: RecoveryContract,
        namespace: str,
        start: int,
        end: int,
        group_id: int,
    ) -> list[tuple[tuple[int, ...], bytes]] | None:
        results = []
        try:
            for client in self._clients:
                result = client.read_recovery(
                    instance_id, hashes, req_id, contract, namespace, start, end, group_id
                )
                if isinstance(result, QueryLoading):
                    self.release(tuple(lease for _, lease in results), req_id)
                    return None
                if not isinstance(result, QueryReady):
                    raise TypeError("recovery read returned an unleased candidate")
                results.append((tuple(result.hit_positions), result.lease))
                if not result.lease:
                    self.release(tuple(lease for _, lease in results), req_id)
                    return []
        except Exception:
            self.release(tuple(lease for _, lease in results), req_id)
            raise
        return results

    def release(self, leases: tuple[bytes, ...], req_id: str) -> bool:
        released = True
        for client, lease in zip(self._clients, leases, strict=False):
            released = self._release_one(client, lease, req_id) and released
        return released

    def cancel(self, instance_id: str, req_id: str, group_id: int = 0) -> None:
        for client in self._clients:
            try:
                if group_id:
                    client.cancel_query(instance_id, req_id, group_id=group_id)
                else:
                    client.cancel_query(instance_id, req_id)
            except Exception:
                logger.exception("Could not cancel cache query: req=%s", req_id)

    @staticmethod
    def _validate_ready(result: QueryReady, queried_blocks: int, shard_index: int) -> None:
        if result.num_hit_blocks > queried_blocks:
            raise RuntimeError(
                f"TP shard {shard_index} reported {result.num_hit_blocks} hits for "
                f"a {queried_blocks}-block query"
            )
        if result.num_hit_blocks and not result.lease:
            raise RuntimeError(
                f"TP shard {shard_index} returned {result.num_hit_blocks} hits without a lease"
            )

    @staticmethod
    def _release_one(client: CacheManagerClient, lease: bytes, req_id: str) -> bool:
        if not lease:
            return True
        try:
            client.release(lease)
        except Exception:
            logger.exception(
                "[OrbitKVConnector] query lease release exception: req=%s",
                req_id,
            )
            return False
        return True


__all__ = ["ShardedQueryReady", "TpShardQueryClient"]
