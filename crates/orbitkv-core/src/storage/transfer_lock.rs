// Transfer lock manager: prevents LRU eviction of blocks during cross-node
// remote transfer by holding Arc<SealedBlock> references. When the TinyLFU cache
// evicts a key, the pinned memory stays allocated as long as this lock holds an Arc.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, warn};
use opentelemetry::KeyValue;
use parking_lot::Mutex;
use uuid::Uuid;

use crate::block::{SealedBlock, StateKey};
use crate::metrics::core_metrics;

struct TransferSession {
    blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    created_at: Instant,
    requester_id: String,
    reserved_bytes: u64,
    expired: bool,
}

#[derive(Default)]
struct Transfers {
    sessions: HashMap<String, TransferSession>,
    reserved_bytes: u64,
}

const MAX_TRANSFER_SESSIONS: usize = 1024;

pub(crate) struct TransferLockManager {
    inner: Mutex<Transfers>,
    lock_timeout: Duration,
    budget_bytes: u64,
}

impl TransferLockManager {
    pub(crate) fn new(lock_timeout: Duration, budget_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(Transfers::default()),
            lock_timeout,
            budget_bytes,
        }
    }

    pub(crate) fn lock_timeout(&self) -> Duration {
        self.lock_timeout
    }

    /// Lock blocks for a transfer session. Returns the session ID.
    ///
    /// Expiry cannot prove that a remote READ has stopped. Only a release after
    /// terminal transport completion permits these allocations to be reused.
    pub(crate) fn lock_blocks(
        &self,
        requester_id: &str,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    ) -> Option<String> {
        if blocks.is_empty() {
            return None;
        }
        let allocations: HashMap<_, _> = blocks
            .iter()
            .flat_map(|(_, block)| block.pinned_allocations())
            .collect();
        let bytes = allocations
            .values()
            .try_fold(0u64, |sum, size| sum.checked_add(*size))?;
        let block_count = blocks.len();
        let mut inner = self.inner.lock();
        let rejection = if inner.sessions.len() >= MAX_TRANSFER_SESSIONS {
            Some("sessions")
        } else if bytes > self.budget_bytes.saturating_sub(inner.reserved_bytes) {
            Some("bytes")
        } else {
            None
        };
        if let Some(reason) = rejection {
            core_metrics()
                .transfer_lock_rejections
                .add(1, &[KeyValue::new("reason", reason)]);
            return None;
        }
        let session_id = Uuid::new_v4().to_string();
        inner.sessions.insert(
            session_id.clone(),
            TransferSession {
                blocks,
                created_at: Instant::now(),
                requester_id: requester_id.to_string(),
                reserved_bytes: bytes,
                expired: false,
            },
        );
        inner.reserved_bytes += bytes;
        core_metrics()
            .transfer_reserved_bytes
            .add(bytes as i64, &[]);
        core_metrics()
            .transfer_lock_active
            .add(block_count as i64, &[]);
        debug!(
            "Transfer lock acquired: session={} requester={} blocks={}",
            session_id, requester_id, block_count
        );

        Some(session_id)
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
        if let Some(session) = inner.sessions.remove(session_id) {
            let count = session.blocks.len();
            inner.reserved_bytes -= session.reserved_bytes;
            core_metrics()
                .transfer_reserved_bytes
                .add(-(session.reserved_bytes as i64), &[]);
            if session.expired {
                core_metrics().transfer_expired_sessions.add(-1, &[]);
            }
            core_metrics()
                .transfer_lock_active
                .add(-(count as i64), &[]);
            debug!(
                "Transfer lock released: session={} requester={} blocks={}",
                session_id, session.requester_id, count
            );
            count
        } else {
            debug!("Transfer lock release already acknowledged: {}", session_id);
            0
        }
    }

    /// Mark overdue sessions once; keep their pins and byte reservations.
    pub(crate) fn expire(&self) -> usize {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let mut expired_count = 0;
        for (id, session) in &mut inner.sessions {
            if !session.expired && now.duration_since(session.created_at) >= self.lock_timeout {
                session.expired = true;
                expired_count += 1;
                warn!(
                    "Transfer overdue, retaining source memory: session={} requester={} blocks={} age={:?}",
                    id,
                    session.requester_id,
                    session.blocks.len(),
                    now.duration_since(session.created_at),
                );
            }
        }

        if expired_count > 0 {
            core_metrics()
                .transfer_expired_sessions
                .add(expired_count as i64, &[]);
            core_metrics()
                .transfer_lock_timeouts_total
                .add(expired_count as u64, &[]);
        }

        expired_count
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/transfer_lock.rs"]
mod tests;
