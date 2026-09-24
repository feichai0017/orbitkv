use super::*;
use smallvec::smallvec;

#[test]
fn segmented_and_page_first_sources_check_exact_component_bounds() {
    let split = SlotMeta::new(smallvec![4096, 8192], crate::NumaNode::UNKNOWN);
    assert_eq!(
        segment_offset(&split, 16384, 1, 512, 700).unwrap(),
        16384 + 4096 + 512
    );
    assert!(segment_offset(&split, 0, 0, 4000, 100).is_err());
    assert!(segment_offset(&split, 0, 2, 0, 1).is_err());
    assert!(segment_offset(&split, 0, 0, usize::MAX, 2).is_err());
    let page = SlotMeta::new(smallvec![16384], crate::NumaNode::UNKNOWN);
    assert_eq!(segment_offset(&page, 4096, 0, 8192, 4096).unwrap(), 12288);
}
