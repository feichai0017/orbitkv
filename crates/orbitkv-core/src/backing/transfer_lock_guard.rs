use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use log::warn;
use orbitkv_proto::proto::engine::ReleaseTransferLockRequest;
use orbitkv_proto::proto::engine::engine_client::EngineClient;
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::transport::Channel;

use crate::metrics::core_metrics;

const MAX_COMPLETIONS: usize = 1024;
const MAX_PEER_COMPLETIONS: usize = 64;
const RPC_TIMEOUT: Duration = Duration::from_secs(3);
const MIN_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);

/// Capacity covers authorization, active READs and completed but unacknowledged
/// releases. One unreachable peer cannot consume the entire requester budget.
pub(super) struct TransferCompletions {
    slots: Arc<Semaphore>,
    peers: Mutex<HashMap<String, usize>>,
}

impl Default for TransferCompletions {
    fn default() -> Self {
        Self {
            slots: Arc::new(Semaphore::new(MAX_COMPLETIONS)),
            peers: Mutex::new(HashMap::new()),
        }
    }
}

impl TransferCompletions {
    pub(super) fn reserve(
        self: &Arc<Self>,
        client: EngineClient<Channel>,
        remote: &str,
    ) -> Option<TransferLockGuard> {
        let mut peers = self.peers.lock();
        let permit = (peers.get(remote).copied().unwrap_or(0) < MAX_PEER_COMPLETIONS)
            .then(|| self.slots.clone().try_acquire_owned().ok())
            .flatten();
        let Some(permit) = permit else {
            core_metrics().transfer_completion_rejections.add(1, &[]);
            return None;
        };
        *peers.entry(remote.into()).or_default() += 1;
        core_metrics().transfer_completion_outstanding.add(1, &[]);
        Some(TransferLockGuard {
            completion: Some(Completion {
                client,
                session_id: String::new(),
                remote: remote.into(),
                budget: Arc::clone(self),
                _permit: permit,
            }),
            handle: tokio::runtime::Handle::current(),
        })
    }
}

struct Completion {
    client: EngineClient<Channel>,
    session_id: String,
    remote: String,
    budget: Arc<TransferCompletions>,
    _permit: OwnedSemaphorePermit,
}

impl Completion {
    async fn acknowledge(mut self) {
        let started = tokio::time::Instant::now();
        let mut backoff = MIN_RETRY;
        loop {
            let request = ReleaseTransferLockRequest {
                transfer_session_id: self.session_id.clone(),
            };
            let response =
                tokio::time::timeout(RPC_TIMEOUT, self.client.release_transfer_lock(request)).await;
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
                    "Retaining transfer completion until acknowledged: remote={} session={}",
                    self.remote, self.session_id
                );
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(MAX_RETRY);
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        let mut peers = self.budget.peers.lock();
        if let Some(count) = peers.get_mut(&self.remote) {
            *count -= 1;
            if *count == 0 {
                peers.remove(&self.remote);
            }
        }
        core_metrics().transfer_completion_outstanding.add(-1, &[]);
    }
}

/// The authorization task owns this guard before sending its RPC. After a
/// successful reply it moves with the destination buffers into the READ task.
pub(super) struct TransferLockGuard {
    completion: Option<Completion>,
    handle: tokio::runtime::Handle,
}

impl TransferLockGuard {
    pub(super) fn authorize(&mut self, session_id: String) {
        if let Some(completion) = self.completion.as_mut() {
            completion.session_id = session_id;
        }
    }

    /// A deadline or caller cancellation does not establish terminal completion.
    /// The blocking operation must drain submitted transfers before returning.
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
            && !completion.session_id.is_empty()
        {
            self.handle.spawn(completion.acknowledge());
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/backing/transfer_lock_guard.rs"]
mod tests;
