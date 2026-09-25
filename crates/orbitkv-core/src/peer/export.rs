// Source allocation ownership and replay protection for peer READs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hashlink::LinkedHashMap;
use log::warn;
use opentelemetry::KeyValue;
use parking_lot::Mutex;
use uuid::Uuid;

use crate::block::{SealedBlock, StateKey};
use crate::metrics::core_metrics;

pub(crate) const TRANSFER_WINDOW_SLOTS: usize = 64;
const MAX_TRANSFER_SESSIONS: usize = 1024;
const MAX_TRANSFER_WINDOWS: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct TransferTicket {
    pub(crate) window: Uuid,
    pub(crate) slot: usize,
    pub(crate) generation: u64,
}

impl TransferTicket {
    pub fn new(window: Uuid, slot: usize, generation: u64) -> Result<Self, PeerError> {
        if window.is_nil() || slot >= TRANSFER_WINDOW_SLOTS || generation == 0 {
            return Err(PeerError::InvalidRequest("invalid transfer ticket".into()));
        }
        Ok(Self {
            window,
            slot,
            generation,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PeerError {
    Unavailable,
    StaleReplica,
    InvalidRequest(String),
    UnknownWindow,
    StaleTicket,
    BudgetExhausted,
}

/// Authoritative DRAM export admission and the lifetime of peer source grants.
/// Directory candidates cannot bypass this owner's incarnation/version checks.
pub struct PeerExports {
    dram: Arc<crate::storage::dram::DramStore>,
    membership: Option<Arc<orbitkv_catalog::MembershipView>>,
    endpoint: Option<String>,
    locks: TransferLockManager,
}

impl PeerExports {
    pub(crate) fn new(
        dram: Arc<crate::storage::dram::DramStore>,
        membership: Option<Arc<orbitkv_catalog::MembershipView>>,
        endpoint: Option<String>,
        timeout: Duration,
        budget_bytes: u64,
    ) -> Self {
        Self {
            dram,
            membership,
            endpoint,
            locks: TransferLockManager::new(timeout, budget_bytes),
        }
    }

    fn validate_owner(&self, owner: Uuid) -> Result<(), PeerError> {
        if self.endpoint.is_none() {
            return Err(PeerError::Unavailable);
        }
        if self
            .membership
            .as_ref()
            .is_none_or(|view| view.owner().incarnation != owner || !view.permits(view.owner()))
        {
            return Err(PeerError::StaleReplica);
        }
        Ok(())
    }

    pub fn open(&self, owner: Uuid, requester: Uuid) -> Result<Uuid, PeerError> {
        self.validate_owner(owner)?;
        if requester.is_nil() {
            return Err(PeerError::InvalidRequest(
                "nil requester incarnation".into(),
            ));
        }
        self.locks.open(requester).ok_or(PeerError::BudgetExhausted)
    }

    pub fn authorize(
        &self,
        owner: Uuid,
        ticket: TransferTicket,
        records: &[orbitkv_state::InventoryRecord],
    ) -> Result<Vec<(StateKey, Arc<SealedBlock>)>, PeerError> {
        self.validate_owner(owner)?;
        let first = records
            .first()
            .ok_or_else(|| PeerError::InvalidRequest("empty transfer".into()))?;
        if records.iter().any(|record| {
            record.key.namespace != first.key.namespace || !record.present || record.sequence == 0
        }) {
            return Err(PeerError::InvalidRequest(
                "invalid residency evidence".into(),
            ));
        }
        let hashes: Vec<_> = records
            .iter()
            .map(|record| record.key.hash.clone())
            .collect();
        orbitkv_state::validate_discovery_query(&first.key.namespace, &hashes)
            .map_err(|reason| PeerError::InvalidRequest(reason.into()))?;
        let blocks = self
            .dram
            .pin_residencies(records)
            .ok_or(PeerError::StaleReplica)?;
        self.locks.lock_blocks(ticket, blocks.clone())?;
        Ok(blocks)
    }

    pub fn release(&self, ticket: TransferTicket) -> Result<usize, PeerError> {
        // Fencing prevents new grants, but terminal completion must still drain.
        self.locks.release(ticket)
    }

    pub fn lock_timeout(&self) -> Duration {
        self.locks.lock_timeout()
    }

    pub(crate) fn expire(&self) -> usize {
        self.locks.expire()
    }
}

struct TransferSession {
    blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    created_at: Instant,
    reserved_bytes: u64,
    expired: bool,
}

#[derive(Default)]
struct TransferSlot {
    generation: u64,
    transfer: Option<TransferSession>,
}

struct TransferWindow {
    requester: Uuid,
    slots: [TransferSlot; TRANSFER_WINDOW_SLOTS],
}

#[derive(Default)]
struct Transfers {
    windows: LinkedHashMap<Uuid, TransferWindow>,
    active: usize,
    reserved_bytes: u64,
}

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

    /// Opening a window pins no data. Only idle windows may be evicted; an
    /// evicted UUID is never recreated by authorization or completion.
    pub(crate) fn open(&self, requester: Uuid) -> Option<Uuid> {
        let mut inner = self.inner.lock();
        if inner.windows.len() == MAX_TRANSFER_WINDOWS {
            let idle = inner.windows.iter().find_map(|(id, window)| {
                window
                    .slots
                    .iter()
                    .all(|slot| slot.transfer.is_none())
                    .then_some(*id)
            })?;
            inner.windows.remove(&idle);
        }
        let id = Uuid::new_v4();
        inner.windows.insert(
            id,
            TransferWindow {
                requester,
                slots: std::array::from_fn(|_| TransferSlot::default()),
            },
        );
        Some(id)
    }

    /// A ticket is single use. A completion arriving before authorization
    /// closes its generation, so delayed authorization cannot resurrect a hold.
    pub(crate) fn lock_blocks(
        &self,
        ticket: TransferTicket,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    ) -> Result<(), PeerError> {
        let allocations: HashMap<_, _> = blocks
            .iter()
            .flat_map(|(_, block)| block.pinned_allocations())
            .collect();
        let bytes = allocations
            .values()
            .try_fold(0u64, |sum, size| sum.checked_add(*size))
            .ok_or(PeerError::BudgetExhausted)?;
        let block_count = blocks.len();
        let mut inner = self.inner.lock();
        let window = inner
            .windows
            .to_back(&ticket.window)
            .ok_or(PeerError::UnknownWindow)?;
        let slot = window
            .slots
            .get_mut(ticket.slot)
            .ok_or(PeerError::StaleTicket)?;
        if blocks.is_empty() || ticket.generation <= slot.generation || slot.transfer.is_some() {
            return Err(PeerError::StaleTicket);
        }
        // Consume the generation even when admission fails.
        slot.generation = ticket.generation;
        let rejection = if inner.active >= MAX_TRANSFER_SESSIONS {
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
            return Err(PeerError::BudgetExhausted);
        }
        inner.windows[&ticket.window].slots[ticket.slot].transfer = Some(TransferSession {
            blocks,
            created_at: Instant::now(),
            reserved_bytes: bytes,
            expired: false,
        });
        inner.active += 1;
        inner.reserved_bytes += bytes;
        core_metrics()
            .transfer_reserved_bytes
            .add(bytes as i64, &[]);
        core_metrics()
            .transfer_lock_active
            .add(block_count as i64, &[]);
        Ok(())
    }

    /// Completion is idempotent, including unknown windows. Never close an
    /// active different generation. The peer endpoint must be cluster-isolated;
    /// possession of a window UUID is not a replacement for authentication.
    pub(crate) fn release(&self, ticket: TransferTicket) -> Result<usize, PeerError> {
        let mut inner = self.inner.lock();
        let Some(window) = inner.windows.to_back(&ticket.window) else {
            return Ok(0);
        };
        let slot = window
            .slots
            .get_mut(ticket.slot)
            .ok_or(PeerError::StaleTicket)?;
        if ticket.generation == 0 {
            return Err(PeerError::StaleTicket);
        }
        if ticket.generation < slot.generation {
            return Ok(0);
        }
        if ticket.generation > slot.generation && slot.transfer.is_some() {
            return Err(PeerError::StaleTicket);
        }
        slot.generation = ticket.generation;
        let Some(session) = slot.transfer.take() else {
            return Ok(0);
        };
        let count = session.blocks.len();
        inner.active -= 1;
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
        Ok(count)
    }

    /// Overdue is observational: only terminal completion permits memory reuse.
    pub(crate) fn expire(&self) -> usize {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let mut expired_count = 0;
        for (id, window) in &mut inner.windows {
            for (index, slot) in window.slots.iter_mut().enumerate() {
                if let Some(session) = &mut slot.transfer
                    && !session.expired
                    && now.duration_since(session.created_at) >= self.lock_timeout
                {
                    session.expired = true;
                    expired_count += 1;
                    warn!(
                        "Transfer overdue, retaining source memory: window={} slot={} generation={} requester={} blocks={} age={:?}",
                        id,
                        index,
                        slot.generation,
                        window.requester,
                        session.blocks.len(),
                        now.duration_since(session.created_at)
                    );
                }
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
#[path = "../../tests/unit/peer/export.rs"]
mod tests;
