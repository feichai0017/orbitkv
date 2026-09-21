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
use orbitkv_proto::proto::engine::meta_server_client::MetaServerClient as GrpcClient;
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

const RPC_TIMEOUT: Duration = Duration::from_secs(3);
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);
const MIN_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);

#[derive(Default)]
struct Control {
    flush_requests: AtomicU64,
    wake: Notify,
}

#[derive(Clone, Copy, Default)]
struct Acknowledgement {
    inventory: InventoryStatus,
    verified_flush: u64,
    stopped: bool,
}

pub(crate) struct MetaServerClient {
    pub(crate) node_id: Uuid,
    #[cfg(feature = "mooncake")]
    advertise_addr: String,
    #[cfg(feature = "mooncake")]
    candidates: parking_lot::Mutex<CandidateIndex>,
    #[cfg(feature = "mooncake")]
    discovery_gate: tokio::sync::Mutex<()>,
    read_cache: Weak<ReadCache>,
    control: Arc<Control>,
    shutdown: watch::Sender<bool>,
    progress: watch::Receiver<Acknowledgement>,
    #[cfg(feature = "mooncake")]
    query_client: GrpcClient<Channel>,
}

impl MetaServerClient {
    pub(crate) fn new(
        metaserver_addr: String,
        advertise_addr: String,
        read_cache: Weak<ReadCache>,
    ) -> Result<Self, String> {
        let endpoint = Endpoint::from_shared(metaserver_addr)
            .map_err(|e| e.to_string())?
            .connect_timeout(GRPC_CONNECT_TIMEOUT)
            .timeout(RPC_TIMEOUT)
            .http2_keep_alive_interval(GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL)
            .keep_alive_while_idle(true);
        let client = GrpcClient::new(endpoint.connect_lazy());
        let changed = read_cache
            .upgrade()
            .ok_or("cache has stopped")?
            .inventory_changed();
        let control = Arc::new(Control::default());
        let (shutdown, shutdown_rx) = watch::channel(false);
        let (progress_tx, progress) = watch::channel(Acknowledgement::default());
        let node_id = Uuid::new_v4();
        let worker = InventorySync {
            client: client.clone(),
            node: advertise_addr.clone(),
            node_id: node_id.to_string(),
            epoch: String::new(),
            progress: InventoryStatus::default(),
            generation: 0,
            phase: Phase::Restart,
            verified_flush: 0,
            heartbeat_at: Instant::now(),
            retry_at: Instant::now(),
            retry_delay: MIN_RETRY,
        };
        tokio::spawn(worker.run(
            read_cache.clone(),
            changed,
            Arc::clone(&control),
            shutdown_rx,
            progress_tx,
        ));
        Ok(Self {
            node_id,
            #[cfg(feature = "mooncake")]
            advertise_addr,
            #[cfg(feature = "mooncake")]
            candidates: parking_lot::Mutex::new(CandidateIndex::new(CANDIDATE_CACHE_BYTES)),
            #[cfg(feature = "mooncake")]
            discovery_gate: tokio::sync::Mutex::new(()),
            read_cache,
            control,
            shutdown,
            progress,
            #[cfg(feature = "mooncake")]
            query_client: client,
        })
    }

    /// Wait for a fresh heartbeat and acknowledgement through the current local
    /// inventory sequence. Concurrent eviction can legitimately remove a block.
    pub(crate) async fn flush(&self) -> Result<(), String> {
        self.flush_with_timeout(FLUSH_TIMEOUT).await
    }

    async fn flush_with_timeout(&self, timeout: Duration) -> Result<(), String> {
        let target = self
            .read_cache
            .upgrade()
            .ok_or("cache has stopped")?
            .inventory_sequence();
        let ticket = self.control.flush_requests.fetch_add(1, Ordering::AcqRel) + 1;
        self.control.wake.notify_one();
        let mut progress = self.progress.clone();
        let ack = tokio::time::timeout(
            timeout,
            progress.wait_for(|ack| {
                ack.stopped
                    || (ack.verified_flush >= ticket
                        && ack.inventory.ready
                        && ack.inventory.sequence >= target)
            }),
        )
        .await
        .map_err(|_| "inventory acknowledgement timed out".to_string())?
        .map_err(|_| "inventory synchronization stopped".to_string())?;
        if ack.stopped {
            Err("inventory synchronization stopped".into())
        } else {
            Ok(())
        }
    }

    pub(crate) async fn shutdown(&self) {
        let _ = self.shutdown.send(true);
        let mut progress = self.progress.clone();
        let _ = tokio::time::timeout(
            RPC_TIMEOUT * 2 + Duration::from_secs(1),
            progress.wait_for(|ack| ack.stopped),
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
        let missing: Vec<_> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| row.is_none().then_some(i))
            .collect();
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
            let response = self
                .query_client
                .clone()
                .locate_blocks(timed(LocateBlocksRequest {
                    namespace: namespace.into(),
                    block_hashes: batch,
                    exclude_node: self.advertise_addr.clone(),
                }))
                .await;
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

#[derive(Clone)]
enum Phase {
    Restart,
    Snapshot { cursor: Option<StateKey>, page: u64 },
    Replay { through: u64 },
    Live,
}

struct InventorySync {
    client: GrpcClient<Channel>,
    node: String,
    node_id: String,
    epoch: String,
    progress: InventoryStatus,
    generation: u64,
    phase: Phase,
    verified_flush: u64,
    heartbeat_at: Instant,
    retry_at: Instant,
    retry_delay: Duration,
}

impl InventorySync {
    async fn run(
        mut self,
        cache: Weak<ReadCache>,
        changed: Arc<Notify>,
        control: Arc<Control>,
        mut shutdown: watch::Receiver<bool>,
        acknowledgements: watch::Sender<Acknowledgement>,
    ) {
        loop {
            if *shutdown.borrow() || cache.strong_count() == 0 {
                break;
            }
            if Instant::now() < self.retry_at {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = tokio::time::sleep_until(self.retry_at) => {},
                }
            }
            let ticket = control.flush_requests.load(Ordering::Acquire);
            if Instant::now() >= self.heartbeat_at || ticket > self.verified_flush {
                match self.heartbeat(ticket).await {
                    Ok(()) => self.publish(&acknowledgements),
                    Err(err) => {
                        core_metrics().metaserver_heartbeat_failures.add(1, &[]);
                        self.failed(&err);
                        continue;
                    }
                }
            }
            let Some(source) = cache.upgrade() else {
                break;
            };
            let step = self.next_operation(&source);
            drop(source);
            match step {
                Ok(Some((operation, next))) => {
                    if let Err(err) = self.send(operation, next, &cache).await {
                        core_metrics().inventory_sync_failures.add(1, &[]);
                        // A lost reply during a scan makes its cursor ambiguous.
                        // Live deltas can resume from the next heartbeat's ACK.
                        if !matches!(self.phase, Phase::Live) {
                            self.phase = Phase::Restart;
                        }
                        self.failed(&err);
                    }
                    self.publish(&acknowledgements);
                    tokio::task::yield_now().await;
                }
                Ok(None) => {
                    tokio::select! {
                        _ = shutdown.changed() => break,
                        _ = changed.notified() => {},
                        _ = control.wake.notified() => {},
                        _ = tokio::time::sleep_until(self.heartbeat_at) => {},
                    }
                }
                Err(err) => {
                    if err.code() == tonic::Code::OutOfRange {
                        core_metrics().inventory_history_gaps.add(1, &[]);
                    }
                    self.phase = Phase::Restart;
                    self.failed(&err);
                }
            }
        }
        if let Err(err) = self
            .client
            .unregister_node(timed(UnregisterNodeRequest {
                node: self.node.clone(),
                node_id: self.node_id.clone(),
            }))
            .await
        {
            warn!("Inventory unregister failed: {err}");
            core_metrics().metaserver_unregister_failures.add(1, &[]);
        }
        acknowledgements.send_replace(Acknowledgement {
            stopped: true,
            ..Acknowledgement::default()
        });
    }

    fn publish(&self, tx: &watch::Sender<Acknowledgement>) {
        tx.send_replace(Acknowledgement {
            inventory: self.progress,
            verified_flush: self.verified_flush,
            stopped: false,
        });
    }

    fn failed(&mut self, error: &Status) {
        warn!("Inventory synchronization will retry: {error}");
        let jitter = Duration::from_millis(rand::random_range(
            0..=self.retry_delay.as_millis() as u64 / 4,
        ));
        self.retry_at = Instant::now() + self.retry_delay + jitter;
        self.heartbeat_at = self.retry_at;
        self.retry_delay = (self.retry_delay * 2).min(MAX_RETRY);
    }

    async fn heartbeat(&mut self, ticket: u64) -> Result<(), Status> {
        let response = self
            .client
            .heartbeat_node(timed(HeartbeatNodeRequest {
                node: self.node.clone(),
                node_id: self.node_id.clone(),
            }))
            .await?
            .into_inner();
        let remote: InventoryStatus = response
            .progress
            .ok_or_else(|| Status::data_loss("missing inventory progress"))?
            .into();
        let same_epoch = self.epoch == response.catalog_epoch;
        let same_generation = remote.generation == self.progress.generation;
        let resumable = matches!(self.phase, Phase::Live) && remote.ready && same_generation;
        if !same_epoch || (!resumable && remote != self.progress) || remote.generation == 0 {
            self.phase = Phase::Restart;
        }
        self.generation = self.generation.max(remote.generation);
        self.epoch = response.catalog_epoch;
        self.progress = remote;
        self.verified_flush = ticket;
        self.heartbeat_at = Instant::now()
            + Duration::from_millis((response.stale_after_secs.saturating_mul(1000) / 3).max(100));
        Ok(())
    }

    fn next_operation(
        &mut self,
        cache: &ReadCache,
    ) -> Result<Option<(InventoryOperation, Phase)>, Status> {
        loop {
            match &self.phase {
                Phase::Restart => {
                    self.generation = self
                        .generation
                        .checked_add(1)
                        .ok_or_else(|| Status::out_of_range("inventory generation exhausted"))?;
                    core_metrics().inventory_snapshots_started.add(1, &[]);
                    return Ok(Some((
                        InventoryOperation::Begin {
                            sequence: cache.inventory_sequence(),
                        },
                        Phase::Snapshot {
                            cursor: None,
                            page: 0,
                        },
                    )));
                }
                Phase::Snapshot { cursor, page } => {
                    if !cache.inventory_covers(self.progress.sequence) {
                        return Err(Status::out_of_range(
                            "inventory journal expired during snapshot",
                        ));
                    }
                    let records = cache
                        .inventory_page(cursor.as_ref())
                        .map_err(inventory_error)?;
                    if let Some(last) = records.last() {
                        let next = Phase::Snapshot {
                            cursor: Some(last.key.clone()),
                            page: page + 1,
                        };
                        return Ok(Some((
                            InventoryOperation::Snapshot {
                                page: *page,
                                records,
                            },
                            next,
                        )));
                    }
                    self.phase = Phase::Replay {
                        through: cache.inventory_sequence(),
                    };
                }
                Phase::Replay { through } => {
                    let records = cache
                        .inventory_changes(self.progress.sequence, *through)
                        .map_err(inventory_error)?;
                    if records.is_empty() {
                        return Ok(Some((
                            InventoryOperation::Commit { sequence: *through },
                            Phase::Live,
                        )));
                    }
                    return Ok(Some((
                        InventoryOperation::Delta {
                            after: self.progress.sequence,
                            records,
                        },
                        self.phase.clone(),
                    )));
                }
                Phase::Live => {
                    let records = cache
                        .inventory_changes(self.progress.sequence, cache.inventory_sequence())
                        .map_err(inventory_error)?;
                    return Ok((!records.is_empty()).then_some((
                        InventoryOperation::Delta {
                            after: self.progress.sequence,
                            records,
                        },
                        Phase::Live,
                    )));
                }
            }
        }
    }

    async fn send(
        &mut self,
        operation: InventoryOperation,
        next: Phase,
        cache: &Weak<ReadCache>,
    ) -> Result<(), Status> {
        let mut expected = self.progress;
        match &operation {
            InventoryOperation::Begin { sequence } => {
                expected = InventoryStatus {
                    generation: self.generation,
                    sequence: *sequence,
                    next_page: 0,
                    ready: false,
                }
            }
            InventoryOperation::Snapshot { page, .. } => expected.next_page = page + 1,
            InventoryOperation::Delta { records, .. } => {
                expected.sequence = records
                    .last()
                    .ok_or_else(|| Status::internal("empty delta"))?
                    .sequence;
            }
            InventoryOperation::Commit { .. } => expected.ready = true,
        }
        let count = match &operation {
            InventoryOperation::Snapshot { records, .. }
            | InventoryOperation::Delta { records, .. } => records.len(),
            _ => 0,
        };
        let commit = matches!(operation, InventoryOperation::Commit { .. });
        let response = self
            .client
            .sync_inventory(timed(SyncInventoryRequest {
                node: self.node.clone(),
                node_id: self.node_id.clone(),
                catalog_epoch: self.epoch.clone(),
                generation: self.generation,
                operation: Some(operation.into()),
            }))
            .await?
            .into_inner();
        let actual: InventoryStatus = response
            .progress
            .ok_or_else(|| Status::data_loss("missing inventory acknowledgement"))?
            .into();
        if actual != expected {
            return Err(Status::data_loss("unexpected inventory acknowledgement"));
        }
        self.progress = actual;
        self.phase = next;
        core_metrics().inventory_records_sent.add(count as u64, &[]);
        if commit {
            core_metrics().inventory_snapshots_completed.add(1, &[]);
        }
        if self.progress.ready {
            self.retry_delay = MIN_RETRY;
        }
        if !response.reclaimable.is_empty()
            && let Some(cache) = cache.upgrade()
        {
            cache.mark_reclaimable_records(
                &response
                    .reclaimable
                    .into_iter()
                    .map(Into::into)
                    .collect::<Vec<_>>(),
            );
        }
        Ok(())
    }
}

fn inventory_error(error: InventoryReadError) -> Status {
    match error {
        InventoryReadError::HistoryGap => {
            Status::out_of_range("inventory journal no longer covers directory progress")
        }
        InventoryReadError::RecordTooLarge => {
            Status::resource_exhausted("inventory record exceeds batch byte limit")
        }
    }
}

fn timed<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.set_timeout(RPC_TIMEOUT);
    request
}

#[cfg(test)]
#[path = "../../tests/unit/internode/metaserver_client.rs"]
mod tests;
