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
        Err(PeerError::BudgetExhausted)
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
        Err(PeerError::BudgetExhausted)
    );
    assert_eq!(
        manager.lock_blocks(ticket, blocks),
        Err(PeerError::StaleTicket)
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
        Err(PeerError::StaleTicket)
    );
    let second = TransferTicket {
        generation: 2,
        ..first
    };
    manager.lock_blocks(second, blocks.clone()).unwrap();
    assert_eq!(
        manager.lock_blocks(second, blocks.clone()),
        Err(PeerError::StaleTicket)
    );
    assert_eq!(manager.release(first), Ok(0));
    let future = TransferTicket {
        generation: 3,
        ..first
    };
    assert_eq!(manager.release(future), Err(PeerError::StaleTicket));
    assert_eq!(
        manager.lock_blocks(future, blocks.clone()),
        Err(PeerError::StaleTicket)
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
        Err(PeerError::UnknownWindow)
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
        Err(PeerError::BudgetExhausted)
    );
    manager.release(first).unwrap();
    assert!(manager.open(Uuid::new_v4()).is_some());
}

fn exports() -> (PeerExports, Arc<orbitkv_catalog::MembershipView>) {
    use orbitkv_catalog::{MembershipView, Placement};
    use orbitkv_state::CacheOwner;

    let owner = CacheOwner {
        endpoint: "127.0.0.1:50055".into(),
        incarnation: Uuid::new_v4(),
    };
    let membership = Arc::new(MembershipView::new(
        owner.clone(),
        Placement::new(vec!["source".into()]).unwrap(),
    ));
    assert!(membership.renew(Instant::now(), Duration::from_secs(3600)));
    membership.replace_members([("source".into(), owner)]);
    let dram = Arc::new(crate::storage::dram::DramStore::new(
        1 << 20,
        false,
        None,
        Some(16 * 1024),
        0,
    ));
    (
        PeerExports::new(
            dram,
            Some(membership.clone()),
            Some("127.0.0.1:12345".into()),
            Duration::ZERO,
            1 << 20,
        ),
        membership,
    )
}

#[test]
fn export_revalidates_residency_and_drains_after_fencing() {
    let (exports, membership) = exports();
    let owner = membership.owner().incarnation;
    let (blocks, allocation) = shared_slab();
    let key = blocks[0].0.clone();
    exports.dram.batch_insert(blocks.clone());
    let records = exports
        .dram
        .inventory_page(orbitkv_state::catalog_shard(&key), None)
        .unwrap();
    let ticket = TransferTicket::new(exports.open(owner, Uuid::new_v4()).unwrap(), 0, 1).unwrap();
    // Replacing the same key invalidates previously discovered evidence.
    drop(exports.dram.remove_all());
    exports.dram.batch_insert(blocks);
    assert!(matches!(
        exports.authorize(owner, ticket, &records),
        Err(PeerError::StaleReplica)
    ));
    assert_eq!(exports.locks.inner.lock().reserved_bytes, 0);
    let records = exports
        .dram
        .inventory_page(orbitkv_state::catalog_shard(&key), None)
        .unwrap();
    let authorized = exports.authorize(owner, ticket, &records).unwrap();
    assert_eq!(authorized.len(), records.len());
    let bytes = allocation.size_bytes();
    assert_eq!(exports.locks.inner.lock().reserved_bytes, bytes);
    let memory = Arc::downgrade(&allocation);
    drop(authorized);
    drop(allocation);
    drop(exports.dram.remove_all());
    membership.fence();
    assert_eq!(
        exports.open(owner, Uuid::new_v4()),
        Err(PeerError::StaleReplica)
    );
    assert!(matches!(
        exports.authorize(owner, ticket, &records),
        Err(PeerError::StaleReplica)
    ));
    assert_eq!(exports.expire(), 1);
    assert!(
        memory.upgrade().is_some(),
        "fencing/expiry must preserve the DMA source"
    );
    assert_eq!(exports.release(ticket), Ok(records.len()));
    assert_eq!(exports.release(ticket), Ok(0));
    assert!(memory.upgrade().is_none());
    assert_eq!(exports.locks.inner.lock().reserved_bytes, 0);
}

#[test]
fn export_rejects_invalid_evidence_before_reserving_resources() {
    use orbitkv_state::InventoryRecord;

    let (mut exports, membership) = exports();
    let owner = membership.owner().incarnation;
    assert_eq!(
        exports.open(Uuid::new_v4(), Uuid::new_v4()),
        Err(PeerError::StaleReplica)
    );
    assert!(matches!(
        exports.open(owner, Uuid::nil()),
        Err(PeerError::InvalidRequest(_))
    ));
    let window = exports.open(owner, Uuid::new_v4()).unwrap();
    for (window, slot, generation) in [
        (Uuid::nil(), 0, 1),
        (window, TRANSFER_WINDOW_SLOTS, 1),
        (window, 0, 0),
    ] {
        assert!(matches!(
            TransferTicket::new(window, slot, generation),
            Err(PeerError::InvalidRequest(_))
        ));
    }
    let ticket = TransferTicket::new(window, 0, 1).unwrap();
    let valid = InventoryRecord {
        key: StateKey::new("ns".into(), vec![1]),
        sequence: 1,
        present: true,
    };
    let mut mixed = valid.clone();
    mixed.key.namespace = "other".into();
    for records in [
        vec![],
        vec![valid.clone(), mixed],
        vec![InventoryRecord {
            sequence: 0,
            ..valid.clone()
        }],
        vec![InventoryRecord {
            present: false,
            ..valid.clone()
        }],
        vec![InventoryRecord {
            key: StateKey::new("ns".into(), vec![]),
            ..valid.clone()
        }],
    ] {
        assert!(matches!(
            exports.authorize(owner, ticket, &records),
            Err(PeerError::InvalidRequest(_))
        ));
    }
    assert_eq!(exports.locks.inner.lock().active, 0);
    exports.endpoint = None;
    assert_eq!(
        exports.open(owner, Uuid::new_v4()),
        Err(PeerError::Unavailable)
    );
    assert!(matches!(
        exports.authorize(owner, ticket, &[valid]),
        Err(PeerError::Unavailable)
    ));
}
