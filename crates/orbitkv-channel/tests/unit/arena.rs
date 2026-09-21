use super::*;

#[test]
fn arena_round_trip_rejects_stale_generation() {
    let arena = DescriptorArena::create(7, 16 * 1024, 1024).unwrap();
    let descriptor = arena.write_slot(0, 11, b"query").unwrap();
    assert_eq!(arena.read(descriptor).unwrap(), b"query");

    let response = arena.write_response(descriptor, b"ready").unwrap();
    assert_eq!(response.generation, 12);
    assert_eq!(arena.read(response).unwrap(), b"ready");
    assert!(matches!(
        arena.read(descriptor),
        Err(ArenaError::StaleGeneration {
            expected: 11,
            actual: 12
        })
    ));
}

#[test]
fn arena_rejects_unaligned_and_oversized_references() {
    let arena = DescriptorArena::create(9, 16 * 1024, 128).unwrap();
    assert!(matches!(
        arena.write_slot(0, 1, &[0; 129]),
        Err(ArenaError::PayloadTooLarge { .. })
    ));
    assert!(matches!(
        arena.read(DescriptorRef {
            offset: 4100,
            len: 0,
            generation: 1
        }),
        Err(ArenaError::InvalidOffset { .. })
    ));
}

#[test]
fn odd_slot_capacity_keeps_every_generation_atomic_aligned() {
    let arena = DescriptorArena::create(13, 16 * 1024, 129).unwrap();
    for slot in 0..arena.slot_count() {
        let offset = arena.slot_offset(slot).unwrap() as usize;
        assert_eq!(
            (offset - SLOT_HEADER_BYTES) % std::mem::align_of::<AtomicU64>(),
            0
        );
    }
}
