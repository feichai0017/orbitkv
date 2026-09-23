use std::collections::BTreeMap;

use futures::{StreamExt, stream};
use opentelemetry::KeyValue;
use orbitkv_proto::proto::engine::LocateBlocksRequest;
use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, ReplicaLocation,
    StateKey, catalog_shard, validate_discovery_query,
};

use super::*;

const MAX_LOOKUP_HOSTS: usize = 4;

impl CatalogClient {
    pub(crate) async fn locate_blocks(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<BlockCandidates>, String> {
        let deadline = Instant::now() + RPC_TIMEOUT;
        let keys: Vec<_> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.into(), hash.clone()))
            .collect();
        let cached = || {
            let mut index = self.candidates.lock();
            let now = std::time::Instant::now();
            keys.iter()
                .map(|key| index.get(key, now))
                .collect::<Vec<_>>()
        };
        let mut rows = cached();
        let hits = rows.iter().filter(|row| row.is_some()).count();
        for (result, count) in [("hit", hits), ("miss", rows.len() - hits)] {
            core_metrics()
                .candidate_cache_lookups
                .add(count as u64, &[KeyValue::new("result", result)]);
        }
        if hits == rows.len() {
            return Ok(rows.into_iter().flatten().collect());
        }
        // Coalescing wait is part of the same deadline. Positive hits bypass it.
        let lookup = tokio::time::timeout_at(deadline, self.discovery_gate.lock()).await;
        if lookup.is_ok() {
            rows = cached();
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
                clients.retain(|owner, _| {
                    owners.iter().any(|current| current.as_ref() == Some(owner))
                });
                for (owner, indices) in missing {
                    let client = match clients.entry(owner.clone()) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(connect(&owner)?).clone()
                        }
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
                        let mut shards: Vec<_> =
                            positions.iter().map(|&i| catalog_shard(&keys[i])).collect();
                        shards.sort_unstable();
                        shards.dedup();
                        batches.push((
                            positions,
                            LocateBlocksRequest {
                                routes: shards
                                    .into_iter()
                                    .map(|shard| route(&self.membership, shard, &owner))
                                    .collect(),
                                namespace: namespace.into(),
                                block_hashes,
                                exclude_node: self.advertise_addr.clone(),
                            },
                        ));
                    }
                    requests.push((client, batches));
                }
            }
            let mut results = stream::iter(requests).map(|(mut client, batches)| {
                let keys = &keys;
                async move {
                    let mut found = Vec::new();
                    for (positions, request) in batches {
                        let started = Instant::now();
                        if started >= deadline { break; }
                        let response = tokio::time::timeout_at(deadline, client.locate_blocks(timed(request))).await;
                        let result = match &response {
                            Ok(Ok(_)) => "ok",
                            Ok(Err(_)) => "error",
                            Err(_) => "timeout",
                        };
                        core_metrics().candidate_lookup_rpcs.add(1, &[KeyValue::new("result", result)]);
                        core_metrics().remote_stage_duration_seconds.record(
                            started.elapsed().as_secs_f64(),
                            &[KeyValue::new("stage", "discovery_rpc"), KeyValue::new("status", result)],
                        );
                        let response = match response {
                            Ok(Ok(response)) => response.into_inner(),
                            Ok(Err(error)) => {
                                warn!("Candidate lookup failed; retaining known prefix evidence: {error}");
                                break;
                            }
                            Err(_) => break,
                        };
                        if response.blocks.len() != positions.len() {
                            warn!("Discovery response count mismatch");
                            break;
                        }
                        let validated = response.blocks.into_iter().zip(&positions)
                            .map(|(row, &i)| row.into_candidates(keys[i].clone(), &self.advertise_addr))
                            .collect::<Result<Vec<_>, _>>();
                        match validated {
                            Ok(batch) => found.extend(positions.into_iter().zip(batch)),
                            Err(error) => {
                                warn!("Invalid discovery response: {error}");
                                break;
                            }
                        }
                    }
                    found
                }
            }).buffer_unordered(MAX_LOOKUP_HOSTS);
            while let Some(found) = results.next().await {
                let mut index = self.candidates.lock();
                for (i, row) in found {
                    index.insert(row.clone(), std::time::Instant::now());
                    rows[i] = Some(row);
                }
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

    pub(crate) fn reject_candidate(&self, key: &StateKey, replica: &ReplicaLocation) {
        self.candidates.lock().reject(key, replica);
    }
}
