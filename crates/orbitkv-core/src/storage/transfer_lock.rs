// Transfer lock manager: prevents LRU eviction of blocks during cross-node
// remote transfer by holding Arc<SealedBlock> references. When the TinyLFU cache
// evicts a key, the pinned memory stays allocated as long as this lock holds an Arc.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, info, warn};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::block::{SealedBlock, StateKey};
use crate::metrics::core_metrics;

struct TransferSession {
    blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    created_at: Instant,
    requester_id: String,
}

pub(crate) struct TransferLockManager {
    inner: Mutex<HashMap<String, TransferSession>>,
    lock_timeout: Duration,
}

impl TransferLockManager {
    pub(crate) fn new(lock_timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            lock_timeout,
        }
    }

    pub(crate) fn lock_timeout(&self) -> Duration {
        self.lock_timeout
    }

    /// Lock blocks for a transfer session. Returns the session ID.
    ///
    /// The caller must later call `release()` to free the locks. If the caller
    /// crashes, `gc_expired()` will auto-release after `lock_timeout`.
    pub(crate) fn lock_blocks(
        &self,
        requester_id: &str,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    ) -> String {
        let session_id = Uuid::new_v4().to_string();
        let block_count = blocks.len();

        let mut inner = self.inner.lock();
        inner.insert(
            session_id.clone(),
            TransferSession {
                blocks,
                created_at: Instant::now(),
                requester_id: requester_id.to_string(),
            },
        );

        core_metrics()
            .transfer_lock_active
            .add(block_count as i64, &[]);
        debug!(
            "Transfer lock acquired: session={} requester={} blocks={}",
            session_id, requester_id, block_count
        );

        session_id
    }

    /// Release a transfer session's locks. Returns the number of blocks released.
    ///
    /// # Security model
    ///
    /// Any caller with the session ID can release the lock. This relies on:
    /// 1. Session IDs are UUIDv4 (cryptographically random, unguessable)
    /// 2. The gRPC port is network-isolated (internal cluster only)
    pub(crate) fn release(&self, session_id: &str) -> usize {
        let mut inner = self.inner.lock();
        if let Some(session) = inner.remove(session_id) {
            let count = session.blocks.len();
            core_metrics()
                .transfer_lock_active
                .add(-(count as i64), &[]);
            debug!(
                "Transfer lock released: session={} requester={} blocks={}",
                session_id, session.requester_id, count
            );
            count
        } else {
            warn!("Transfer lock release: session not found: {}", session_id);
            0
        }
    }

    /// Garbage-collect expired sessions. Returns the number of sessions removed.
    pub(crate) fn gc_expired(&self) -> usize {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let timeout = self.lock_timeout;

        let expired: Vec<String> = inner
            .iter()
            .filter(|(_, session)| now.duration_since(session.created_at) > timeout)
            .map(|(id, _)| id.clone())
            .collect();

        let mut expired_count = 0;
        let mut expired_blocks = 0usize;
        for id in &expired {
            if let Some(session) = inner.remove(id) {
                expired_blocks += session.blocks.len();
                expired_count += 1;
                warn!(
                    "Transfer lock expired: session={} requester={} blocks={} age={:?}",
                    id,
                    session.requester_id,
                    session.blocks.len(),
                    now.duration_since(session.created_at),
                );
            }
        }

        if expired_count > 0 {
            core_metrics()
                .transfer_lock_active
                .add(-(expired_blocks as i64), &[]);
            core_metrics()
                .transfer_lock_timeouts_total
                .add(expired_count as u64, &[]);
            info!(
                "Transfer lock GC: expired {} sessions ({} blocks)",
                expired_count, expired_blocks
            );
        }

        expired_count
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/transfer_lock.rs"]
mod tests;
