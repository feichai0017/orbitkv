use orbitkv::RecurrentFamily;

use super::*;

fn class() -> FixedStateClass {
    FixedStateClass {
        state_id: 2,
        name: "recurrent".into(),
        layers: vec![0, 1, 2].into_boxed_slice(),
        storage: FixedStateStorage::Recurrent {
            family: RecurrentFamily::Gdn,
            bytes_per_layer: 64,
            slots_per_request: 2,
            bytes_per_request: 384,
        },
    }
}

#[test]
fn registration_and_slot_ranges_are_exact() {
    let registration = FixedStateArenaRegistration::bind(
        &class(),
        2,
        StatePoolIdentity {
            engine_epoch: 1,
            pool_epoch: 2,
            byte_count: 192,
            pool_id: 3,
            slot_count: 4,
        },
    )
    .unwrap();
    assert_eq!(registration.arena_bytes().unwrap(), 768);
    let range = registration
        .slot_range(StateSlotLease {
            engine_epoch: 1,
            pool_epoch: 2,
            generation: 7,
            slot_id: 3,
            pool_id: 3,
        })
        .unwrap();
    assert_eq!(range.byte_offset, 576);
    assert_eq!(range.byte_count, 192);
}

#[test]
fn registration_rejects_mismatched_geometry() {
    let mut identity = StatePoolIdentity {
        engine_epoch: 1,
        pool_epoch: 2,
        byte_count: 192,
        pool_id: 3,
        slot_count: 4,
    };
    identity.byte_count -= 1;
    assert!(matches!(
        FixedStateArenaRegistration::bind(&class(), 2, identity),
        Err(ExecutorError::FixedStateRegistrationMismatch)
    ));
}
