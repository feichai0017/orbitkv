use std::sync::Arc;

use super::*;
use crate::block::SealedBlock;

#[test]
fn memory_sources_bind_engine_target_and_deduplicate_identity() {
    let block = RestoreSource::Memory(Arc::new(SealedBlock::from_slots(Vec::new())));
    let plan = RestorePlan::new(3, [(7, &block), (7, &block)]).unwrap();
    assert_eq!(plan.device_id(), 3);
    assert_eq!(plan.ssd_path(), None);
    assert_eq!(plan.ssd_source_bytes(), 0);
    assert_eq!(plan.ssd_source_fragments(), 0);
    assert_eq!(plan.source_bytes(), 0);
    assert_eq!(plan.source_fragments(), 0);
    assert!(plan.has_memory());
    assert!(plan.decode_pages().is_none());
    assert!(RestorePlan::new(-1, [(0, &block)]).is_err());
}

#[test]
fn decode_page_grant_is_device_bound_nonempty_and_single_use() {
    let mut plan = RestorePlan::new(
        3,
        std::iter::empty::<(usize, &crate::block::RestoreSource)>(),
    )
    .unwrap();
    assert!(plan.admit_decode_pages(0, 1).is_err());
    assert!(plan.admit_decode_pages(4096, 0).is_err());
    plan.admit_decode_pages(4096, 2).unwrap();
    let grant = plan.decode_pages().unwrap();
    assert_eq!(grant.device_id(), 3);
    assert_eq!(grant.bytes(), 4096);
    assert_eq!(grant.fragments(), 2);
    assert!(plan.admit_decode_pages(4096, 2).is_err());
}

#[test]
fn automatic_cufile_can_fall_back_but_explicit_route_cannot() {
    let mut automatic = RestorePlan {
        device_id: 0,
        ssd_path: Some(SsdReadPath::Cufile),
        allow_uring_fallback: true,
        ssd_source_bytes: 4096,
        ssd_source_fragments: 1,
        source_bytes: 4096,
        source_fragments: 1,
        source_set_hash: 7,
        has_memory: false,
        decode_pages: None,
    };
    assert_eq!(automatic.fallback_from_cufile(), Ok(true));
    assert_eq!(automatic.ssd_path(), Some(SsdReadPath::Uring));
    assert_eq!(automatic.fallback_from_cufile(), Ok(false));

    let mut explicit = RestorePlan {
        device_id: 0,
        ssd_path: Some(SsdReadPath::Cufile),
        allow_uring_fallback: false,
        ssd_source_bytes: 4096,
        ssd_source_fragments: 1,
        source_bytes: 4096,
        source_fragments: 1,
        source_set_hash: 7,
        has_memory: false,
        decode_pages: None,
    };
    assert!(explicit.fallback_from_cufile().is_err());
    assert_eq!(explicit.ssd_path(), Some(SsdReadPath::Cufile));
}
