// RAII release of a remote transfer-lock session acquired via
// QueryBlocksForTransfer. See `mooncake_fetch` for the fetch flow that owns it.

use log::warn;
use orbitkv_proto::proto::engine::ReleaseTransferLockRequest;
use orbitkv_proto::proto::engine::engine_client::EngineClient;
use tonic::transport::Channel;

/// Releases a transfer session exactly once, on whichever exit runs first:
/// `release()` on the coded completion paths, or `Drop` when the fetch task
/// panics or its future is dropped mid-await. A session that is never
/// released pins the remote blocks until the holder's GC expires it.
pub(super) struct TransferLockGuard {
    client: EngineClient<Channel>,
    session_id: String,
    remote_addr: String,
    req_id: String,
    // Captured at construction (always inside the runtime) so Drop can spawn
    // even from a panic unwind; spawning on a shut-down runtime is a no-op.
    handle: tokio::runtime::Handle,
}

impl TransferLockGuard {
    pub(super) fn new(
        client: EngineClient<Channel>,
        session_id: String,
        remote_addr: &str,
        req_id: &str,
    ) -> Self {
        Self {
            client,
            session_id,
            remote_addr: remote_addr.to_string(),
            req_id: req_id.to_string(),
            handle: tokio::runtime::Handle::current(),
        }
    }

    /// Release on a completed fetch (success or handled error). Fire-and-forget.
    pub(super) fn release(mut self) {
        self.spawn_release();
    }

    /// Keep source and destination memory alive even if the async caller is
    /// cancelled. The blocking operation must drain submitted transfers before
    /// returning; an elapsed deadline alone is not terminal completion.
    pub(super) async fn run_with_buffers<B: Send + 'static, R: Send + 'static>(
        self,
        buffers: B,
        transfer: impl FnOnce() -> R + Send + 'static,
    ) -> Result<(B, R), tokio::task::JoinError> {
        tokio::task::spawn_blocking(move || {
            let result = transfer();
            self.release();
            (buffers, result)
        })
        .await
    }

    fn spawn_release(&mut self) {
        let session_id = std::mem::take(&mut self.session_id);
        if session_id.is_empty() {
            return;
        }
        let mut client = self.client.clone();
        self.handle.spawn(async move {
            let req = ReleaseTransferLockRequest {
                transfer_session_id: session_id.clone(),
            };
            if let Err(e) = client.release_transfer_lock(req).await {
                warn!("ReleaseTransferLock failed for session {session_id}: {e}");
            }
        });
    }
}

impl Drop for TransferLockGuard {
    fn drop(&mut self) {
        if self.session_id.is_empty() {
            return;
        }
        // Only panic/cancellation reaches here — the coded paths call release().
        warn!(
            "Mooncake fetch aborted without releasing transfer lock; releasing via drop guard: session={} remote={} req_id={}",
            self.session_id, self.remote_addr, self.req_id
        );
        self.spawn_release();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/backing/transfer_lock_guard.rs"]
mod tests;
