use std::num::NonZeroU64;
use std::ptr::NonNull;

use super::*;
use crate::block::{RawBlock, Segment};
use crate::numa::NumaNode;
use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};

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

#[test]
fn overdue_transfer_keeps_the_entire_slab_until_completion() {
    let (blocks, allocation) = shared_slab();
    let bytes = allocation.size_bytes();
    let memory = Arc::downgrade(&allocation);
    let manager = TransferLockManager::new(Duration::ZERO, bytes);
    let session = manager.lock_blocks("reader", blocks.clone()).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, bytes);
    assert_eq!(manager.expire(), 1);
    assert_eq!(manager.expire(), 0);
    assert!(
        manager
            .lock_blocks("another-reader", blocks.clone())
            .is_none()
    );
    drop(blocks);
    drop(allocation);
    assert!(
        memory.upgrade().is_some(),
        "eviction and timeout must not free a DMA source"
    );
    assert_eq!(manager.release(&session), 2);
    assert!(memory.upgrade().is_none());
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.release(&session), 0);
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
                barrier.wait();
                manager.lock_blocks("reader", blocks)
            })
        })
        .collect();
    let sessions: Vec<_> = handles
        .into_iter()
        .filter_map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(sessions.len(), 2);
    assert_eq!(manager.inner.lock().reserved_bytes, 2 * bytes);
    manager.release(&sessions[0]);
    let replacement = manager.lock_blocks("replacement", blocks).unwrap();
    manager.release(&sessions[1]);
    manager.release(&replacement);
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
}

#[test]
fn a_small_slice_cannot_bypass_the_allocation_budget() {
    let (mut blocks, allocation) = shared_slab();
    let bytes = allocation.size_bytes();
    let manager = TransferLockManager::new(Duration::ZERO, bytes - 1);
    blocks.truncate(1);
    assert!(manager.lock_blocks("reader", blocks).is_none());
    assert!(manager.lock_blocks("reader", Vec::new()).is_none());
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert!(manager.inner.lock().sessions.is_empty());
}

#[test]
fn metadata_is_bounded_even_for_zero_byte_blocks() {
    let manager = TransferLockManager::new(Duration::ZERO, 1);
    let blocks = vec![(
        StateKey::new("ns".into(), vec![0]),
        Arc::new(SealedBlock::from_slots(Vec::new())),
    )];
    for _ in 0..MAX_TRANSFER_SESSIONS {
        assert!(manager.lock_blocks("reader", blocks.clone()).is_some());
    }
    assert_eq!(manager.expire(), MAX_TRANSFER_SESSIONS);
    assert!(manager.lock_blocks("reader", blocks).is_none());
}
