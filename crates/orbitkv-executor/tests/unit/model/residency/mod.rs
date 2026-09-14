use super::preparation_order;

#[test]
fn preparation_preserves_existing_bucket_before_filling_unused_slots() {
    // Loading can leave a later artifact bucket materialized. A single-slot
    // startup must not evict it to prepare the first artifact bucket instead.
    assert_eq!(preparation_order(3, &[2], 1), [2]);
    assert_eq!(preparation_order(3, &[2], 2), [2, 0]);
    assert_eq!(preparation_order(3, &[1, 2], 2), [1, 2]);
    assert_eq!(preparation_order(3, &[], 2), [0, 1]);
    assert_eq!(preparation_order(2, &[1], usize::MAX), [1, 0]);
}
