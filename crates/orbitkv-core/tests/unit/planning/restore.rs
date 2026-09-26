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
    assert!(plan.has_memory());
    assert!(RestorePlan::new(-1, [(0, &block)]).is_err());
}

#[test]
fn automatic_cufile_can_fall_back_but_explicit_route_cannot() {
    let mut automatic = RestorePlan {
        device_id: 0,
        ssd_path: Some(SsdReadPath::Cufile),
        allow_uring_fallback: true,
        ssd_source_bytes: 4096,
        ssd_source_fragments: 1,
        has_memory: false,
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
        has_memory: false,
    };
    assert!(explicit.fallback_from_cufile().is_err());
    assert_eq!(explicit.ssd_path(), Some(SsdReadPath::Cufile));
}
