use std::num::NonZeroU64;

use super::*;
use crate::block::{RawBlock, Segment};
use crate::memory::pool::PinnedAllocator;
use crate::transfer::layout::KVCacheGeometry;
use crate::{NumaNode, QueryAdmission, QueryMode, QueryOwner};

#[test]
fn raw_grant_retains_allocation_and_query_charge_until_explicit_finish() {
    let pool = PinnedAllocator::new_global(512, 1, false, None);
    let allocation = pool
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let allocation_owner = Arc::downgrade(&allocation);
    let slot = RawBlock::single_segment(Segment::new(allocation.as_non_null(), 512, allocation));
    let block = Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)]));
    let source_owner = Arc::downgrade(&block);
    let budget = crate::query::QueryBudget::new(512, 512).unwrap();
    let QueryAdmission::Admitted(reservation) =
        budget.reserve("engine", "ns", 512, QueryMode::Demand)
    else {
        panic!("budget available")
    };
    reservation.ready(512).unwrap();
    let owner = QueryOwner {
        session: 1,
        operation: 1,
        revision: 1,
    };
    let leases = QueryLeaseManager::default();
    let lease = leases.create(
        "engine",
        vec![RestoreSource::Memory(block)],
        1,
        Some((owner, reservation)),
    );
    let groups = vec![RestoreGroup {
        layers: vec![RestoreLayer {
            name: "layer_99".into(),
            slot_id: 0,
            host_offset: 64,
        }],
        storage_slots: Some((0, 1)),
        targets: vec![],
    }];
    let layout = KVCacheLayout::bind(
        0x1000,
        512,
        KVCacheGeometry::new(4, 32, 256, 2, None, 1).unwrap(),
    )
    .unwrap();
    let prepared = PreparedRestore::prepare(
        &leases,
        "engine",
        0,
        groups,
        &[(lease, vec![vec![Some(2)]])],
        &[layout],
    )
    .unwrap();
    let (encoded, bytes) = prepared.raw.unwrap();
    let plan = RawRestorePlan::decode(&encoded).unwrap();
    assert_eq!(plan.copies.len(), 2);
    assert_eq!(plan.copies[0].layer, "layer_99");
    assert_eq!(plan.copies[0].destination_offset, 64);
    assert_eq!(plan.copies[1].destination_offset, 320);
    assert_eq!(plan.copies[0].source.offset, 64);
    assert_eq!(plan.copies[1].source.offset, 96);
    assert_eq!(
        plan.copies[0].source.allocation_id,
        plan.copies[1].source.allocation_id
    );
    assert_eq!(bytes, 64);
    let sources = prepared
        .sources
        .into_iter()
        .map(|source| match source {
            RestoreSource::Memory(source) => source,
            _ => unreachable!(),
        })
        .collect();
    let grant = RawRestoreGrant {
        plan: encoded,
        sources,
        reservations: prepared.reservations,
        bytes,
        started: std::time::Instant::now(),
    };
    leases.release_owner(|candidate| candidate.session == 1);
    assert!(source_owner.upgrade().is_some());
    assert!(allocation_owner.upgrade().is_some());
    assert!(
        pool.allocate(NonZeroU64::new(1).unwrap(), NumaNode::UNKNOWN)
            .is_none()
    );
    assert!(matches!(
        budget.reserve("engine", "ns", 1, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    grant.finish(true);
    assert!(source_owner.upgrade().is_none());
    assert!(allocation_owner.upgrade().is_none());
    assert!(
        pool.allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
            .is_some()
    );
    assert!(matches!(
        budget.reserve("engine", "ns", 512, QueryMode::Demand),
        QueryAdmission::Admitted(_)
    ));
}

#[test]
fn raw_plan_rejection_preserves_lease_before_source_bounds_or_plan_limit_failure() {
    let pool = PinnedAllocator::new_global(512, 1, false, None);
    let allocation = pool
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let slot = RawBlock::single_segment(Segment::new(allocation.as_non_null(), 16, allocation));
    let block = Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)]));
    let leases = QueryLeaseManager::default();
    let lease = leases.create("engine", vec![RestoreSource::Memory(block)], 1, None);
    let layout = KVCacheLayout::bind(
        0x1000,
        32,
        KVCacheGeometry::new(1, 32, 0, 1, None, 1).unwrap(),
    )
    .unwrap();
    for name in ["range".to_string(), "x".repeat(MAX_PLAN_BYTES)] {
        let groups = vec![RestoreGroup {
            layers: vec![RestoreLayer {
                name,
                slot_id: 0,
                host_offset: 0,
            }],
            storage_slots: Some((0, 1)),
            targets: vec![],
        }];
        assert!(
            PreparedRestore::prepare(
                &leases,
                "engine",
                0,
                groups,
                &[(lease, vec![vec![Some(0)]])],
                std::slice::from_ref(&layout)
            )
            .is_err()
        );
    }
    assert!(
        leases.release(&lease),
        "rejected plan must leave its lease available"
    );
}
