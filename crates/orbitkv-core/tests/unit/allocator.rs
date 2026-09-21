use super::*;

#[test]
fn creates_allocator_with_scaled_capacity() {
    let allocator = ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(
        10 * 1024 * 1024 * 1024,
        64,
        128 * 1024,
    )
    .unwrap();
    assert_eq!(allocator.unit_size.get(), 64);
    assert_eq!(
        allocator.total_units,
        (10 * 1024 * 1024 * 1024u64 / 64) as u32
    );
}

#[test]
fn allocate_rounds_up_to_unit_size() {
    let mut allocator =
        ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(1024, 64, 128 * 1024).unwrap();
    let allocation = allocator.allocate(1).unwrap().unwrap();
    assert_eq!(allocation.offset_bytes, 0);
    assert_eq!(allocation.size_bytes.get(), 64);

    let storage = allocator.storage_report();
    assert_eq!(storage.total_free_bytes, 960);
    assert_eq!(storage.largest_free_allocation_bytes, 960);
}

#[test]
fn capacity_is_rounded_down_to_unit_size() {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    const UNIT_SIZE: u64 = 256 * MIB;
    let total_bytes = GIB + 123 * MIB;

    let allocator = ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(
        total_bytes,
        UNIT_SIZE,
        128 * 1024,
    )
    .unwrap();
    assert_eq!(allocator.total_units, 4);
    assert_eq!(allocator.total_bytes(), GIB);

    let storage = allocator.storage_report();
    assert_eq!(storage.total_free_bytes, GIB);
    assert_eq!(storage.largest_free_allocation_bytes, GIB);
}

#[test]
fn does_not_allocate_past_unaligned_capacity() {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    const UNIT_SIZE: u64 = 256 * MIB;
    let total_bytes = GIB + 123 * MIB;

    let mut allocator = ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(
        total_bytes,
        UNIT_SIZE,
        128 * 1024,
    )
    .unwrap();

    for idx in 0..4 {
        let allocation = allocator.allocate(UNIT_SIZE).unwrap().unwrap();
        assert_eq!(allocation.offset_bytes, idx * UNIT_SIZE);
        assert_eq!(allocation.size_bytes.get(), UNIT_SIZE);
    }
    assert!(allocator.allocate(1).unwrap().is_none());
}

#[test]
fn reuse_after_free_merges_neighboring_regions() {
    let mut allocator =
        ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(256, 64, 128 * 1024).unwrap();
    let a = allocator.allocate(64).unwrap().unwrap();
    let b = allocator.allocate(64).unwrap().unwrap();

    allocator.free(&a);
    allocator.free(&b);

    let merged = allocator.allocate(128).unwrap().unwrap();
    assert_eq!(merged.offset_bytes, 0);
    assert_eq!(merged.size_bytes.get(), 128);
}

#[test]
fn rejects_unreasonably_large_requests() {
    let mut allocator =
        ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(1024 * 1024, 1, 128 * 1024)
            .unwrap();
    let too_large = u64::from(u32::MAX) * 2;
    let err = allocator.allocate(too_large).unwrap_err();
    assert_eq!(
        err,
        AllocatorError::RequestTooLarge {
            requested_bytes: too_large,
            unit_size: 1
        }
    );
}

#[test]
fn rejects_invalid_unit_size() {
    let err =
        ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(1024, 0, 128 * 1024).unwrap_err();
    assert_eq!(err, AllocatorError::InvalidUnitSize);
}
