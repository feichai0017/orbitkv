use std::sync::Arc;
use std::time::Duration;

use hashlink::LinkedHashMap;
use log::warn;
use orbitkv_proto::proto::engine::engine_client::EngineClient;
use orbitkv_proto::proto::engine::{
    OpenTransferWindowRequest, QueryBlocksForTransferRequest, QueryBlocksForTransferResponse,
    ReleaseTransferLockRequest, TransferTicket,
};
use orbitkv_state::CacheOwner;
use parking_lot::Mutex;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore};
use tonic::Status;
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

use super::fetch_plan::FetchSegment;
use crate::metrics::core_metrics;
use crate::storage::transfer_lock::TRANSFER_WINDOW_SLOTS;

const MAX_COMPLETIONS: usize = 1024;
const CACHED_PEERS: usize = 64;
const RPC_TIMEOUT: Duration = Duration::from_secs(3);
const MIN_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);

#[derive(Default)]
struct Slot {
    generation: u64,
    busy: bool,
}

struct PeerState {
    window: Arc<OnceCell<String>>,
    slots: [Slot; TRANSFER_WINDOW_SLOTS],
}

struct Peer {
    client: EngineClient<Channel>,
    state: Mutex<PeerState>,
}

/// Owns peer channels, authorization tickets and completion retries. Capacity
/// covers window setup, authorization, active READs and unacknowledged releases.
pub(super) struct TransferCompletions {
    slots: Arc<Semaphore>,
    peers: Mutex<LinkedHashMap<CacheOwner, Arc<Peer>>>,
}

impl Default for TransferCompletions {
    fn default() -> Self {
        Self {
            slots: Arc::new(Semaphore::new(MAX_COMPLETIONS)),
            peers: Mutex::new(LinkedHashMap::new()),
        }
    }
}

impl TransferCompletions {
    fn peer(&self, owner: &CacheOwner) -> Result<Arc<Peer>, Status> {
        let mut peers = self.peers.lock();
        if let Some(peer) = peers.to_back(owner) {
            return Ok(Arc::clone(peer));
        }
        // Busy peers stay in the map so reconnects cannot bypass their budget.
        // The global permit bounds them even when all cached peers are busy.
        while peers.len() >= CACHED_PEERS {
            let idle = peers
                .iter()
                .find_map(|(owner, peer)| (Arc::strong_count(peer) == 1).then(|| owner.clone()));
            let Some(idle) = idle else { break };
            peers.remove(&idle);
        }
        let addr = &owner.endpoint;
        let url = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.clone()
        } else {
            format!("http://{addr}")
        };
        let channel = Endpoint::from_shared(url)
            .map_err(|e| Status::invalid_argument(e.to_string()))?
            .connect_timeout(RPC_TIMEOUT)
            .timeout(RPC_TIMEOUT)
            .connect_lazy();
        const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
        let peer = Arc::new(Peer {
            client: EngineClient::new(channel)
                .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE),
            state: Mutex::new(PeerState {
                window: Arc::new(OnceCell::new()),
                slots: std::array::from_fn(|_| Slot::default()),
            }),
        });
        peers.insert(owner.clone(), Arc::clone(&peer));
        Ok(peer)
    }

    async fn reserve(
        &self,
        owner: &CacheOwner,
        requester: Uuid,
    ) -> Result<TransferLockGuard, Status> {
        let reject = || {
            core_metrics().transfer_completion_rejections.add(1, &[]);
            Status::resource_exhausted("transfer completion budget exhausted")
        };
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| reject())?;
        let peer = self.peer(owner)?;
        let (index, generation, window) = {
            let mut state = peer.state.lock();
            let (index, slot) = state
                .slots
                .iter_mut()
                .enumerate()
                .find(|(_, slot)| !slot.busy && slot.generation != u64::MAX)
                .ok_or_else(reject)?;
            slot.busy = true;
            slot.generation += 1;
            (index, slot.generation, Arc::clone(&state.window))
        };
        core_metrics().transfer_completion_outstanding.add(1, &[]);
        let guard = TransferLockGuard {
            completion: Some(Completion {
                peer: Arc::clone(&peer),
                window: Arc::clone(&window),
                index,
                generation,
                remote: owner.endpoint.clone(),
                _permit: permit,
            }),
            handle: tokio::runtime::Handle::current(),
        };
        // Single-flight setup with a bounded wait. Lost setup replies retain no
        // payload, and source idle-window metadata has its own bounded cache.
        tokio::time::timeout(
            RPC_TIMEOUT,
            window.get_or_try_init(|| async {
                let response = peer
                    .client
                    .clone()
                    .open_transfer_window(OpenTransferWindowRequest {
                        owner_incarnation: owner.incarnation.to_string(),
                        requester_incarnation: requester.to_string(),
                    })
                    .await?
                    .into_inner();
                let id = response
                    .window_id
                    .parse::<Uuid>()
                    .map_err(|_| Status::data_loss("invalid transfer window"))?;
                if id.is_nil() {
                    return Err(Status::data_loss("nil transfer window"));
                }
                Ok(response.window_id)
            }),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("transfer window setup timed out"))??;
        Ok(guard)
    }

    pub(super) async fn authorize(
        &self,
        segment: &FetchSegment,
        requester: Uuid,
    ) -> Result<(TransferLockGuard, QueryBlocksForTransferResponse), Status> {
        let guard = self.reserve(&segment.owner, requester).await?;
        let completion = guard
            .completion
            .as_ref()
            .ok_or_else(|| Status::internal("missing transfer guard"))?;
        let request = QueryBlocksForTransferRequest {
            namespace: segment.records[0].key.namespace.clone(),
            block_hashes: segment.records.iter().map(|r| r.key.hash.clone()).collect(),
            owner_incarnation: segment.owner.incarnation.to_string(),
            residency_sequences: segment.records.iter().map(|r| r.sequence).collect(),
            ticket: completion.ticket(),
        };
        // The ticket is already known: dropping this future can safely close it
        // even if the RPC is still queued or its successful reply never arrives.
        let response = tokio::time::timeout(
            RPC_TIMEOUT,
            completion
                .peer
                .client
                .clone()
                .query_blocks_for_transfer(request),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("transfer authorization timed out"))?;
        let response = match response {
            Ok(response) => response.into_inner(),
            Err(error) => {
                if error.code() == tonic::Code::NotFound {
                    let mut state = completion.peer.state.lock();
                    if Arc::ptr_eq(&state.window, &completion.window) {
                        state.window = Arc::new(OnceCell::new());
                    }
                }
                return Err(error);
            }
        };
        if !response.status.as_ref().is_some_and(|status| status.ok) {
            return Err(Status::data_loss("missing transfer authorization"));
        }
        Ok((guard, response))
    }
}

struct Completion {
    peer: Arc<Peer>,
    window: Arc<OnceCell<String>>,
    index: usize,
    generation: u64,
    remote: String,
    _permit: OwnedSemaphorePermit,
}

impl Completion {
    fn ticket(&self) -> Option<TransferTicket> {
        self.window.get().map(|id| TransferTicket {
            window_id: id.clone(),
            slot: self.index as u32,
            generation: self.generation,
        })
    }

    async fn acknowledge(self) {
        let started = tokio::time::Instant::now();
        let mut backoff = MIN_RETRY;
        loop {
            let request = ReleaseTransferLockRequest {
                ticket: self.ticket(),
            };
            let response = tokio::time::timeout(
                RPC_TIMEOUT,
                self.peer.client.clone().release_transfer_lock(request),
            )
            .await;
            if matches!(response, Ok(Ok(ref reply)) if reply.get_ref().status.as_ref().is_some_and(|s| s.ok))
            {
                core_metrics().remote_stage_duration_seconds.record(
                    started.elapsed().as_secs_f64(),
                    &[
                        opentelemetry::KeyValue::new("stage", "release"),
                        opentelemetry::KeyValue::new("status", "ok"),
                    ],
                );
                return;
            }
            core_metrics().transfer_completion_retries.add(1, &[]);
            if backoff == MIN_RETRY {
                warn!(
                    "Retaining transfer completion until acknowledged: remote={} ticket={:?}",
                    self.remote,
                    self.ticket()
                );
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(MAX_RETRY);
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        self.peer.state.lock().slots[self.index].busy = false;
        core_metrics().transfer_completion_outstanding.add(-1, &[]);
    }
}

/// Moves with destination buffers into the native READ task. Cancellation of
/// authorization closes the ticket; cancellation of READ waits for native drain.
pub(super) struct TransferLockGuard {
    completion: Option<Completion>,
    handle: tokio::runtime::Handle,
}

impl TransferLockGuard {
    pub(super) async fn run_with_buffers<B: Send + 'static, R: Send + 'static>(
        self,
        buffers: B,
        transfer: impl FnOnce() -> R + Send + 'static,
    ) -> Result<(B, R), tokio::task::JoinError> {
        tokio::task::spawn_blocking(move || {
            let result = transfer();
            drop(self);
            (buffers, result)
        })
        .await
    }
}

impl Drop for TransferLockGuard {
    fn drop(&mut self) {
        if let Some(completion) = self.completion.take()
            && completion.window.get().is_some()
        {
            self.handle.spawn(completion.acknowledge());
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/backing/transfer_lock_guard.rs"]
mod tests;
