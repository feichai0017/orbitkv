use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "mooncake")]
use super::discovery::{CANDIDATE_CACHE_BYTES, CandidateIndex};
use log::warn;
use orbitkv_common::grpc::{GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL, GRPC_CONNECT_TIMEOUT};
#[cfg(feature = "mooncake")]
use orbitkv_proto::proto::engine::LocateBlocksRequest;
use orbitkv_proto::proto::engine::catalog_client::CatalogClient as GrpcClient;
use orbitkv_proto::proto::engine::{
    HeartbeatNodeRequest, SyncInventoryRequest, UnregisterNodeRequest,
};
#[cfg(feature = "mooncake")]
use orbitkv_state::{BlockCandidates, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, ReplicaLocation};
use orbitkv_state::{InventoryOperation, InventoryStatus, StateKey};
use tokio::sync::{Notify, watch};
use tokio::time::{Duration, Instant};
use tonic::{
    Request, Status,
    transport::{Channel, Endpoint},
};
use uuid::Uuid;

use crate::metrics::core_metrics;
use crate::storage::{ReadCache, inventory::InventoryReadError};
use orbitkv_catalog::MembershipView;
use orbitkv_proto::proto::engine::CatalogRoute;
#[cfg(any(feature = "mooncake", test))]
use orbitkv_state::catalog_shard;
use orbitkv_state::{CATALOG_SHARDS, CacheOwner};

mod sync;

const RPC_TIMEOUT: Duration = Duration::from_secs(3);
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);
const MIN_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);

type CatalogConnection = (CacheOwner, GrpcClient<Channel>);

#[derive(Default)]
struct Control {
    flush_requests: AtomicU64,
    wake: Notify,
}

#[derive(Clone, Default)]
struct Acknowledgement {
    inventory: InventoryStatus,
    verified_flush: u64,
    stopped: bool,
    catalog: Option<CacheOwner>,
}

pub(crate) struct CatalogClient {
    pub(crate) node_id: Uuid,
    #[cfg(feature = "mooncake")]
    advertise_addr: String,
    #[cfg(feature = "mooncake")]
    candidates: parking_lot::Mutex<CandidateIndex>,
    #[cfg(feature = "mooncake")]
    discovery_gate: tokio::sync::Mutex<()>,
    read_cache: Weak<ReadCache>,
    membership: Arc<MembershipView>,
    streams: Vec<(Arc<Control>, watch::Receiver<Acknowledgement>)>,
    shutdown: watch::Sender<bool>,
    #[cfg(feature = "mooncake")]
    query_clients: parking_lot::Mutex<[Option<CatalogConnection>; CATALOG_SHARDS]>,
}

impl CatalogClient {
    pub(crate) fn new(
        membership: Arc<MembershipView>,
        read_cache: Weak<ReadCache>,
    ) -> Result<Self, String> {
        let cache = read_cache.upgrade().ok_or("cache has stopped")?;
        let (shutdown, _) = watch::channel(false);
        let mut streams = Vec::with_capacity(CATALOG_SHARDS);
        for shard in 0..CATALOG_SHARDS {
            let control = Arc::new(Control::default());
            let (progress_tx, progress) = watch::channel(Acknowledgement::default());
            let worker = sync::InventorySync::new(shard, membership.clone());
            tokio::spawn(worker.run(
                read_cache.clone(),
                cache.inventory_changed(shard),
                Arc::clone(&control),
                shutdown.subscribe(),
                progress_tx,
            ));
            streams.push((control, progress));
        }
        Ok(Self {
            node_id: membership.owner().incarnation,
            #[cfg(feature = "mooncake")]
            advertise_addr: membership.owner().endpoint.clone(),
            #[cfg(feature = "mooncake")]
            candidates: parking_lot::Mutex::new(CandidateIndex::new(CANDIDATE_CACHE_BYTES)),
            #[cfg(feature = "mooncake")]
            discovery_gate: tokio::sync::Mutex::new(()),
            read_cache,
            membership,
            streams,
            shutdown,
            #[cfg(feature = "mooncake")]
            query_clients: parking_lot::Mutex::new(std::array::from_fn(|_| None)),
        })
    }

    /// Wait for a fresh heartbeat and acknowledgement through the current local
    /// inventory sequence. Concurrent eviction can legitimately remove a block.
    pub(crate) async fn flush(&self) -> Result<(), String> {
        self.flush_with_timeout(FLUSH_TIMEOUT).await
    }

    async fn flush_with_timeout(&self, timeout: Duration) -> Result<(), String> {
        let cache = self.read_cache.upgrade().ok_or("cache has stopped")?;
        let waits = self
            .streams
            .iter()
            .enumerate()
            .map(|(shard, (control, progress))| {
                let target = cache.inventory_sequence(shard);
                let ticket = control.flush_requests.fetch_add(1, Ordering::AcqRel) + 1;
                control.wake.notify_one();
                let mut progress = progress.clone();
                async move {
                    let ack = progress
                        .wait_for(|ack| {
                            ack.stopped
                                || (ack.verified_flush >= ticket
                                    && ack.inventory.ready
                                    && ack.inventory.sequence >= target
                                    && ack.catalog.is_some()
                                    && ack.catalog == self.membership.catalog_owner(shard))
                        })
                        .await
                        .map_err(|_| "inventory synchronization stopped".to_string())?;
                    if ack.stopped {
                        Err("inventory synchronization stopped".to_string())
                    } else {
                        Ok(())
                    }
                }
            });
        tokio::time::timeout(timeout, futures::future::try_join_all(waits))
            .await
            .map_err(|_| "inventory acknowledgement timed out".to_string())??;
        Ok(())
    }

    pub(crate) async fn shutdown(&self) {
        let _ = self.shutdown.send(true);
        let waits = self.streams.iter().map(|(_, progress)| {
            let mut progress = progress.clone();
            async move {
                let _ = progress.wait_for(|ack| ack.stopped).await;
            }
        });
        let _ = tokio::time::timeout(
            RPC_TIMEOUT * 2 + Duration::from_secs(1),
            futures::future::join_all(waits),
        )
        .await;
    }

    #[cfg(feature = "mooncake")]
    pub(crate) async fn locate_blocks(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<BlockCandidates>, String> {
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
        let rows = cached();
        let hits = rows.iter().filter(|row| row.is_some()).count();
        for (result, count) in [("hit", hits), ("miss", rows.len() - hits)] {
            core_metrics().candidate_cache_lookups.add(
                count as u64,
                &[opentelemetry::KeyValue::new("result", result)],
            );
        }
        if rows.iter().all(Option::is_some) {
            return Ok(rows.into_iter().flatten().collect());
        }
        // Coalesce concurrent misses; a waiter rechecks the cache after the lookup.
        // Hits never wait for an unrelated directory RPC.
        let _lookup = self.discovery_gate.lock().await;
        let mut rows = cached();
        let deadline = Instant::now() + RPC_TIMEOUT;
        'shards: for shard in 0..CATALOG_SHARDS {
            let missing: Vec<_> = rows
                .iter()
                .enumerate()
                .filter_map(|(i, row)| {
                    (row.is_none() && catalog_shard(&keys[i]) == shard).then_some(i)
                })
                .collect();
            if missing.is_empty() {
                continue;
            }
            let Some(owner) = self.membership.catalog_owner(shard) else {
                continue;
            };
            let mut client = {
                let mut clients = self.query_clients.lock();
                let slot = &mut clients[shard];
                if slot.as_ref().is_none_or(|(cached, _)| cached != &owner) {
                    *slot = Some((owner.clone(), connect(&owner)?));
                }
                slot.as_ref().expect("installed channel").1.clone()
            };
            let mut cursor = 0;
            while cursor < missing.len() {
                let start = cursor;
                let mut bytes = namespace.len();
                while cursor < missing.len() && cursor - start < DISCOVERY_MAX_KEYS {
                    let next = hashes[missing[cursor]].len();
                    if bytes.saturating_add(next) > DISCOVERY_MAX_BYTES {
                        break;
                    }
                    bytes += next;
                    cursor += 1;
                }
                if cursor == start {
                    return Err("discovery key exceeds byte budget".into());
                }
                let batch: Vec<_> = missing[start..cursor]
                    .iter()
                    .map(|&i| hashes[i].clone())
                    .collect();
                orbitkv_state::validate_discovery_query(namespace, &batch)?;
                let response = match tokio::time::timeout_at(
                    deadline,
                    client.locate_blocks(timed(LocateBlocksRequest {
                        route: Some(route(&self.membership, shard, &owner)),
                        namespace: namespace.into(),
                        block_hashes: batch,
                        exclude_node: self.advertise_addr.clone(),
                    })),
                )
                .await
                {
                    Ok(response) => response,
                    Err(_) => break 'shards,
                };
                core_metrics().candidate_lookup_rpcs.add(
                    1,
                    &[opentelemetry::KeyValue::new(
                        "result",
                        if response.is_ok() { "ok" } else { "error" },
                    )],
                );
                let response = match response {
                    Ok(response) => response.into_inner(),
                    Err(error) => {
                        warn!("Candidate lookup failed; retaining known prefix evidence: {error}");
                        break;
                    }
                };
                if response.blocks.len() != cursor - start {
                    return Err("discovery response count mismatch".into());
                }
                // Validate the whole batch before populating the index.
                let validated: Vec<_> = response
                    .blocks
                    .into_iter()
                    .zip(&missing[start..cursor])
                    .map(|(row, &i)| row.into_candidates(keys[i].clone(), &self.advertise_addr))
                    .collect::<Result<_, _>>()?;
                let mut index = self.candidates.lock();
                for (row, &i) in validated.into_iter().zip(&missing[start..cursor]) {
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

    #[cfg(feature = "mooncake")]
    pub(crate) fn reject_candidate(&self, key: &StateKey, replica: &ReplicaLocation) {
        self.candidates.lock().reject(key, replica);
    }
}

fn connect(owner: &CacheOwner) -> Result<GrpcClient<Channel>, String> {
    let endpoint = Endpoint::from_shared(format!("http://{}", owner.endpoint))
        .map_err(|error| error.to_string())?
        .connect_timeout(GRPC_CONNECT_TIMEOUT)
        .timeout(RPC_TIMEOUT)
        .http2_keep_alive_interval(GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL)
        .keep_alive_while_idle(true);
    Ok(GrpcClient::new(endpoint.connect_lazy()).max_decoding_message_size(4 * 1024 * 1024))
}

fn route(view: &MembershipView, shard: usize, owner: &CacheOwner) -> CatalogRoute {
    CatalogRoute {
        shard: shard as u32,
        placement_id: view.placement_id().into(),
        incarnation: owner.incarnation.to_string(),
    }
}

fn timed<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.set_timeout(RPC_TIMEOUT);
    request
}

#[cfg(test)]
#[path = "../../tests/unit/internode/catalog_client.rs"]
mod tests;
