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
