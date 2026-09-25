use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "mooncake")]
use super::discovery::{CANDIDATE_CACHE_BYTES, CandidateIndex};
use log::warn;
use orbitkv_common::grpc::{GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL, GRPC_CONNECT_TIMEOUT};
use orbitkv_proto::proto::engine::catalog_client::CatalogClient as GrpcClient;
use orbitkv_proto::proto::engine::{
    HeartbeatNodeRequest, SyncInventoryRequest, UnregisterNodeRequest,
};
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
#[cfg(test)]
use orbitkv_state::catalog_shard;
use orbitkv_state::{CATALOG_SHARDS, CacheOwner};

#[cfg(feature = "mooncake")]
mod lookup;
mod sync;

const RPC_TIMEOUT: Duration = Duration::from_secs(3);
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);
const MIN_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);

type CatalogConnection = (CacheOwner, GrpcClient<Channel>);
#[cfg(feature = "mooncake")]
type QueryConnection = (GrpcClient<Channel>, Arc<tokio::sync::Semaphore>);

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
    pending_lookups: Arc<parking_lot::Mutex<lookup::PendingLookups>>,
    #[cfg(feature = "mooncake")]
    lookup_slots: Arc<tokio::sync::Semaphore>,
    read_cache: Weak<ReadCache>,
    membership: Arc<MembershipView>,
    streams: Vec<(Arc<Control>, watch::Receiver<Acknowledgement>)>,
    shutdown: watch::Sender<bool>,
    #[cfg(feature = "mooncake")]
    query_clients: parking_lot::Mutex<std::collections::HashMap<CacheOwner, QueryConnection>>,
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
            pending_lookups: Arc::new(parking_lot::Mutex::new(lookup::PendingLookups::default())),
            #[cfg(feature = "mooncake")]
            lookup_slots: Arc::new(tokio::sync::Semaphore::new(lookup::MAX_LOOKUP_HOSTS)),
            read_cache,
            membership,
            streams,
            shutdown,
            #[cfg(feature = "mooncake")]
            query_clients: parking_lot::Mutex::new(std::collections::HashMap::new()),
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
