use super::{checked_byte_range, ranges_overlap};

#[test]
fn byte_ranges_reject_overflow_and_detect_overlap() {
    let first = checked_byte_range(0x1000_usize as *const f32, 8, "first").unwrap();
    let overlapping = checked_byte_range(0x1010_usize as *const f32, 8, "overlap").unwrap();
    let disjoint = checked_byte_range(0x1020_usize as *const f32, 8, "disjoint").unwrap();
    assert!(ranges_overlap(first, overlapping));
    assert!(!ranges_overlap(first, disjoint));
    assert!(checked_byte_range::<f32>(std::ptr::null(), 8, "null").is_err());
    assert!(checked_byte_range(0x1000_usize as *const f32, 0, "empty").is_err());
}
