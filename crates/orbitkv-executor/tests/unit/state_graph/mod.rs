use super::*;
use crate::FixedStateDeviceRange;

#[test]
fn destination_slots_preserve_request_order_and_reject_bad_classes() {
    let range = |request_id: u64, state_id: u16, slot_id: u32| FixedStateDeviceBatch {
        request_id,
        sources: Box::default(),
        destinations: vec![FixedStateDeviceRange {
            state_id,
            lease: orbitkv::StateSlotLease {
                engine_epoch: 1,
                pool_epoch: 2,
                generation: 1,
                slot_id,
                pool_id: 3,
            },
            device_ptr: 64,
            byte_offset: 0,
            byte_count: 16,
        }]
        .into_boxed_slice(),
    };
    let ordered = [range(9, 4, 2), range(3, 4, 0)];
    assert_eq!(destination_slot_ids(4, &ordered).unwrap(), vec![2, 0]);
    assert!(matches!(
        destination_slot_ids(5, &ordered),
        Err(FixedStateGraphError::InvalidBatch)
    ));
    let duplicate = FixedStateDeviceBatch {
        destinations: vec![ordered[0].destinations[0], ordered[0].destinations[0]]
            .into_boxed_slice(),
        ..ordered[0].clone()
    };
    assert!(matches!(
        destination_slot_ids(4, &[duplicate]),
        Err(FixedStateGraphError::InvalidBatch)
    ));
}
