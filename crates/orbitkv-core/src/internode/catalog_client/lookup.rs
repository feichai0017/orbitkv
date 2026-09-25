use std::collections::{BTreeMap, HashMap};

use futures::{StreamExt, stream};
use opentelemetry::KeyValue;
use orbitkv_proto::proto::engine::LocateBlocksRequest;
use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, ReplicaLocation,
    StateKey, catalog_shard, validate_discovery_query,
};
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore};

use super::*;

pub(super) const MAX_LOOKUP_HOSTS: usize = 4;

#[derive(Eq, Hash, PartialEq)]
struct LookupKey {
    owner: CacheOwner,
    placement: String,
    namespace: String,
    hashes: Vec<Vec<u8>>,
}

impl LookupKey {
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.owner.endpoint.capacity()
            + self.placement.capacity()
            + self.namespace.capacity()
            + self.hashes.capacity() * std::mem::size_of::<Vec<u8>>()
            + self.hashes.iter().map(Vec::capacity).sum::<usize>()
            + std::mem::size_of::<SharedLookup>()
            + std::mem::size_of::<(Arc<Self>, Weak<SharedLookup>, usize)>()
    }
}

#[derive(Default)]
pub(super) struct PendingLookups {
    entries: HashMap<Arc<LookupKey>, (Weak<SharedLookup>, usize)>,
    bytes: usize,
}

struct LookupReply {
    rows: Option<Vec<BlockCandidates>>,
    // Completed responses remain bounded even if a coalesced waiter is not polled.
    _slot: Option<OwnedSemaphorePermit>,
}

struct SharedLookup {
    key: Arc<LookupKey>,
    reply: OnceCell<LookupReply>,
    pending: Weak<parking_lot::Mutex<PendingLookups>>,
}

impl SharedLookup {
    fn acquire(
        pending: &Arc<parking_lot::Mutex<PendingLookups>>,
        key: LookupKey,
    ) -> Option<Arc<Self>> {
        let mut entries = pending.lock();
        if let Some(shared) = entries
            .entries
            .get(&key)
            .and_then(|(weak, _)| weak.upgrade())
        {
            return Some(shared);
        }
        if let Some((_, bytes)) = entries.entries.remove(&key) {
            entries.bytes -= bytes;
        }
        let bytes = key.retained_bytes();
        if bytes > CANDIDATE_CACHE_BYTES.saturating_sub(entries.bytes) {
            return None;
        }
        let key = Arc::new(key);
        let shared = Arc::new(Self {
            key: Arc::clone(&key),
            reply: OnceCell::new(),
            pending: Arc::downgrade(pending),
        });
        entries.bytes += bytes;
        entries
            .entries
            .insert(key, (Arc::downgrade(&shared), bytes));
        Some(shared)
    }

    fn remove(&self) {
        let Some(pending) = self.pending.upgrade() else {
            return;
        };
        let mut entries = pending.lock();
        if entries
            .entries
            .get(&self.key)
            .is_some_and(|(weak, _)| std::ptr::eq(weak.as_ptr(), self))
            && let Some((_, bytes)) = entries.entries.remove(&self.key)
        {
            entries.bytes -= bytes;
        }
    }
}

impl Drop for SharedLookup {
    fn drop(&mut self) {
        self.remove();
    }
}

impl CatalogClient {
    /// Fresh positive hints only; this lookup never issues a catalog RPC.
    pub(crate) fn cached_blocks(&self, keys: &[StateKey]) -> Vec<Option<BlockCandidates>> {
        let rows: Vec<_> = {
            let mut index = self.candidates.lock();
            let now = std::time::Instant::now();
            keys.iter().map(|key| index.get(key, now)).collect()
        };
        let hits = rows.iter().filter(|row| row.is_some()).count();
        for (result, count) in [("hit", hits), ("miss", rows.len() - hits)] {
            core_metrics()
                .candidate_cache_lookups
                .add(count as u64, &[KeyValue::new("result", result)]);
        }
        rows
    }

    pub(crate) async fn locate_blocks(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<BlockCandidates>, String> {
        let deadline = Instant::now() + crate::storage::DISCOVERY_TIMEOUT;
        let keys: Vec<_> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.into(), hash.clone()))
            .collect();
        let mut rows = self.cached_blocks(&keys);
        if rows.iter().all(Option::is_some) {
            return Ok(rows.into_iter().flatten().collect());
        }
        let owners: [_; CATALOG_SHARDS] =
            std::array::from_fn(|shard| self.membership.catalog_owner(shard));
        let mut missing = BTreeMap::<CacheOwner, Vec<usize>>::new();
        for (i, row) in rows.iter().enumerate() {
            if row.is_none()
                && let Some(owner) = &owners[catalog_shard(&keys[i])]
            {
                missing.entry(owner.clone()).or_default().push(i);
            }
        }
        let mut requests = Vec::with_capacity(missing.len());
        {
            let mut clients = self.query_clients.lock();
            clients.retain(|owner, (_, admission)| {
                Arc::strong_count(admission) > 1
                    || owners.iter().any(|current| current.as_ref() == Some(owner))
            });
            for (owner, indices) in missing {
                let (client, admission) = match clients.entry(owner.clone()) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                    std::collections::hash_map::Entry::Vacant(entry) => entry
                        .insert((connect(&owner)?, Arc::new(Semaphore::new(1))))
                        .clone(),
                };
                let mut batches = Vec::new();
                let mut cursor = 0;
                while cursor < indices.len() {
                    let start = cursor;
                    let mut bytes = namespace.len();
                    while cursor < indices.len() && cursor - start < DISCOVERY_MAX_KEYS {
                        let next = hashes[indices[cursor]].len();
                        if bytes.saturating_add(next) > DISCOVERY_MAX_BYTES {
                            break;
                        }
                        bytes += next;
                        cursor += 1;
                    }
                    if cursor == start {
                        return Err("discovery key exceeds byte budget".into());
                    }
                    let positions = indices[start..cursor].to_vec();
                    let block_hashes: Vec<_> =
                        positions.iter().map(|&i| hashes[i].clone()).collect();
                    validate_discovery_query(namespace, &block_hashes)?;
                    batches.push((positions, block_hashes));
                }
                requests.push((owner, client, admission, batches));
            }
        }
        let mut results = stream::iter(requests)
            .map(|(owner, client, admission, batches)| async move {
                let mut found = Vec::new();
                for (positions, hashes) in batches {
                    if Instant::now() >= deadline {
                        break;
                    }
                    let key = LookupKey {
                        owner: owner.clone(),
                        placement: self.membership.placement_id().into(),
                        namespace: namespace.into(),
                        hashes,
                    };
                    let Some(batch) = self
                        .lookup_batch(key, client.clone(), Arc::clone(&admission), deadline)
                        .await
                    else {
                        break;
                    };
                    found.extend(positions.into_iter().zip(batch));
                }
                found
            })
            .buffer_unordered(MAX_LOOKUP_HOSTS);
        while let Some(found) = results.next().await {
            for (i, row) in found {
                rows[i] = Some(row);
            }
        }
        Ok(rows
            .into_iter()
            .zip(keys)
            .map(|(row, key)| {
                row.unwrap_or(BlockCandidates {
                    key,
                    replicas: Vec::new(),
                })
            })
            .collect())
    }

    async fn lookup_batch(
        &self,
        key: LookupKey,
        mut client: GrpcClient<Channel>,
        admission: Arc<Semaphore>,
        deadline: Instant,
    ) -> Option<Vec<BlockCandidates>> {
        let shared = SharedLookup::acquire(&self.pending_lookups, key)?;
        let reply = tokio::time::timeout_at(
            deadline,
            shared.reply.get_or_init(|| async {
                let mut slot = None;
                let rows = async {
                    // A busy peer must not consume slots while other peers can progress.
                    let _peer = admission.acquire().await.ok()?;
                    slot = Some(Arc::clone(&self.lookup_slots).acquire_owned().await.ok()?);
                    let key = &shared.key;
                    let keys: Vec<_> = key
                        .hashes
                        .iter()
                        .map(|hash| StateKey::new(key.namespace.clone(), hash.clone()))
                        .collect();
                    let cached = {
                        let mut index = self.candidates.lock();
                        let now = std::time::Instant::now();
                        keys.iter()
                            .map(|key| index.get(key, now))
                            .collect::<Option<Vec<_>>>()
                    };
                    if cached.is_some() {
                        return cached;
                    }
                    let mut shards: Vec<_> = keys.iter().map(catalog_shard).collect();
                    shards.sort_unstable();
                    shards.dedup();
                    if shards.iter().any(|&shard| {
                        self.membership.catalog_owner(shard).as_ref() != Some(&key.owner)
                    }) {
                        return None;
                    }
                    let request = LocateBlocksRequest {
                        routes: shards
                            .iter()
                            .map(|&shard| route(&self.membership, shard, &key.owner))
                            .collect(),
                        namespace: key.namespace.clone(),
                        block_hashes: key.hashes.clone(),
                        exclude_node: self.advertise_addr.clone(),
                    };
                    let started = Instant::now();
                    let response =
                        tokio::time::timeout_at(deadline, client.locate_blocks(timed(request)))
                            .await;
                    let result = match &response {
                        Ok(Ok(_)) => "ok",
                        Ok(Err(_)) => "error",
                        Err(_) => "timeout",
                    };
                    core_metrics()
                        .candidate_lookup_rpcs
                        .add(1, &[KeyValue::new("result", result)]);
                    core_metrics().remote_stage_duration_seconds.record(
                        started.elapsed().as_secs_f64(),
                        &[
                            KeyValue::new("stage", "discovery_rpc"),
                            KeyValue::new("status", result),
                        ],
                    );
                    let response = match response {
                        Ok(Ok(response)) => response.into_inner(),
                        Ok(Err(error)) => {
                            warn!(
                                "Candidate lookup failed; retaining known prefix evidence: {error}"
                            );
                            return None;
                        }
                        Err(_) => return None,
                    };
                    if response.blocks.len() != keys.len() {
                        warn!("Discovery response count mismatch");
                        return None;
                    }
                    let validated = response
                        .blocks
                        .into_iter()
                        .zip(keys)
                        .map(|(row, key)| row.into_candidates(key, &self.advertise_addr))
                        .collect::<Result<Vec<_>, _>>();
                    let batch = match validated {
                        Ok(batch) => batch,
                        Err(error) => {
                            warn!("Invalid discovery response: {error}");
                            return None;
                        }
                    };
                    if shards.iter().any(|&shard| {
                        self.membership.catalog_owner(shard).as_ref() != Some(&key.owner)
                    }) {
                        return None;
                    }
                    let mut index = self.candidates.lock();
                    let now = std::time::Instant::now();
                    for row in &batch {
                        index.insert(row.clone(), now);
                    }
                    Some(batch)
                }
                .await;
                // Later queries must retry negatives and must observe source rejection.
                shared.remove();
                LookupReply { rows, _slot: slot }
            }),
        )
        .await
        .ok()?;
        reply.rows.clone()
    }

    pub(crate) fn reject_candidate(&self, key: &StateKey, replica: &ReplicaLocation) {
        self.candidates.lock().reject(key, replica);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/internode/catalog_client/lookup.rs"]
mod tests;
