use std::num::NonZeroU64;
use std::ptr::NonNull;

use super::*;
use crate::block::{RawBlock, Segment};
use crate::memory::numa::NumaNode;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};

fn shared_slab() -> (Vec<(StateKey, Arc<SealedBlock>)>, Arc<PinnedAllocation>) {
    let pool = PinnedAllocator::new_global(1024 * 1024, 1, false, false, None);
    let allocation = pool
        .allocate(NonZeroU64::new(4096).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let blocks = (0..2)
        .map(|index| {
            let ptr = unsafe {
                NonNull::new_unchecked(allocation.as_non_null().as_ptr().add(index * 64))
            };
            let raw = RawBlock::single_segment(Segment::new(ptr, 64, Arc::clone(&allocation)));
            (
                StateKey::new("ns".into(), vec![index as u8]),
                Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)])),
            )
        })
        .collect();
    (blocks, allocation)
}

fn ticket(manager: &TransferLockManager) -> TransferTicket {
    TransferTicket {
        window: manager.open(Uuid::new_v4()).unwrap(),
        slot: 0,
        generation: 1,
    }
}

#[test]
fn overdue_transfer_keeps_the_entire_slab_until_completion() {
    let (blocks, allocation) = shared_slab();
    let bytes = allocation.size_bytes();
    let memory = Arc::downgrade(&allocation);
    let manager = TransferLockManager::new(Duration::ZERO, bytes);
    let session = ticket(&manager);
    manager.lock_blocks(session, blocks.clone()).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, bytes);
    assert_eq!(manager.expire(), 1);
    assert_eq!(manager.expire(), 0);
    assert_eq!(
        manager.lock_blocks(ticket(&manager), blocks.clone()),
        Err(TransferLockError::BudgetExhausted)
    );
    drop(blocks);
    drop(allocation);
    assert!(
        memory.upgrade().is_some(),
        "eviction and timeout must not free a DMA source"
    );
    assert_eq!(manager.release(session), Ok(2));
    assert!(memory.upgrade().is_none());
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.release(session), Ok(0));
}

#[test]
fn concurrent_readers_share_one_admission_budget() {
    let (blocks, allocation) = shared_slab();
    let bytes = allocation.size_bytes();
    let manager = Arc::new(TransferLockManager::new(Duration::from_secs(30), 2 * bytes));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let manager = Arc::clone(&manager);
            let barrier = Arc::clone(&barrier);
            let blocks = blocks.clone();
            std::thread::spawn(move || {
                let ticket = ticket(&manager);
                barrier.wait();
                manager.lock_blocks(ticket, blocks).ok().map(|()| ticket)
            })
        })
        .collect();
    let sessions: Vec<_> = handles
        .into_iter()
        .filter_map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(sessions.len(), 2);
    assert_eq!(manager.inner.lock().reserved_bytes, 2 * bytes);
    manager.release(sessions[0]).unwrap();
    let replacement = ticket(&manager);
    manager.lock_blocks(replacement, blocks).unwrap();
    manager.release(sessions[1]).unwrap();
    manager.release(replacement).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
}

#[test]
fn a_small_slice_cannot_bypass_the_allocation_budget() {
    let (mut blocks, allocation) = shared_slab();
    let manager = TransferLockManager::new(Duration::ZERO, allocation.size_bytes() - 1);
    blocks.truncate(1);
    let ticket = ticket(&manager);
    assert_eq!(
        manager.lock_blocks(ticket, blocks.clone()),
        Err(TransferLockError::BudgetExhausted)
    );
    assert_eq!(
        manager.lock_blocks(ticket, blocks),
        Err(TransferLockError::StaleTicket)
    );
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.inner.lock().active, 0);
}

#[test]
fn closed_generations_cannot_resurrect_or_release_a_different_read() {
    let (blocks, _) = shared_slab();
    let manager = TransferLockManager::new(Duration::ZERO, u64::MAX);
    let first = ticket(&manager);
    // Completion wins the race against queued authorization.
    assert_eq!(manager.release(first), Ok(0));
    assert_eq!(
        manager.lock_blocks(first, blocks.clone()),
        Err(TransferLockError::StaleTicket)
    );
    let second = TransferTicket {
        generation: 2,
        ..first
    };
    manager.lock_blocks(second, blocks.clone()).unwrap();
    assert_eq!(
        manager.lock_blocks(second, blocks.clone()),
        Err(TransferLockError::StaleTicket)
    );
    assert_eq!(manager.release(first), Ok(0));
    let future = TransferTicket {
        generation: 3,
        ..first
    };
    assert_eq!(manager.release(future), Err(TransferLockError::StaleTicket));
    assert_eq!(
        manager.lock_blocks(future, blocks.clone()),
        Err(TransferLockError::StaleTicket)
    );
    assert_eq!(manager.inner.lock().active, 1);
    assert_eq!(manager.release(second), Ok(2));
    manager.lock_blocks(future, blocks).unwrap();
    assert_eq!(manager.release(second), Ok(0));
    assert_eq!(manager.release(future), Ok(2));
}

#[test]
fn idle_window_eviction_is_bounded_and_rejects_late_authorization() {
    let (blocks, _) = shared_slab();
    let manager = TransferLockManager::new(Duration::ZERO, u64::MAX);
    let active = ticket(&manager);
    manager.lock_blocks(active, blocks.clone()).unwrap();
    let idle = ticket(&manager);
    manager.release(idle).unwrap();
    // Simulates lost setup replies and requester churn; none pins payload.
    for _ in 0..MAX_TRANSFER_WINDOWS * 2 {
        manager.open(Uuid::new_v4()).unwrap();
    }
    assert_eq!(manager.inner.lock().windows.len(), MAX_TRANSFER_WINDOWS);
    assert_eq!(manager.inner.lock().active, 1);
    assert_eq!(
        manager.lock_blocks(idle, blocks),
        Err(TransferLockError::UnknownWindow)
    );
    assert_eq!(manager.release(idle), Ok(0));
    assert_eq!(manager.release(active), Ok(2));
}

#[test]
fn metadata_is_bounded_even_for_zero_byte_blocks() {
    let manager = TransferLockManager::new(Duration::ZERO, 1);
    let blocks = vec![(
        StateKey::new("ns".into(), vec![0]),
        Arc::new(SealedBlock::from_slots(Vec::new())),
    )];
    let mut first = None;
    for _ in 0..MAX_TRANSFER_SESSIONS {
        let ticket = ticket(&manager);
        first.get_or_insert(ticket);
        manager.lock_blocks(ticket, blocks.clone()).unwrap();
    }
    assert_eq!(manager.expire(), MAX_TRANSFER_SESSIONS);
    assert!(
        manager.open(Uuid::new_v4()).is_none(),
        "busy windows are never evicted"
    );
    let first = first.unwrap();
    assert_eq!(
        manager.lock_blocks(TransferTicket { slot: 1, ..first }, blocks),
        Err(TransferLockError::BudgetExhausted)
    );
    manager.release(first).unwrap();
    assert!(manager.open(Uuid::new_v4()).is_some());
}
