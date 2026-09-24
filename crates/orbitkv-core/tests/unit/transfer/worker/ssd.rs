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

#[test]
fn encoded_reads_use_physical_offsets_and_full_logical_destinations() {
    use crate::transfer::layout::BlockCopy;
    let mut slot = SlotMeta::new(smallvec![1024, 2048], crate::NumaNode::UNKNOWN);
    slot.encoding = Some(vec![
        EncodedSegment {
            version: 1,
            format: StorageFormat::Fp8FromBf16,
            logical_bytes: 1400,
            stored_bytes: 700,
            checksum: 1,
        },
        EncodedSegment {
            version: 1,
            format: StorageFormat::Exact,
            logical_bytes: 1400,
            stored_bytes: 1400,
            checksum: 2,
        },
    ]);
    let copies = || BlockCopies::Split {
        k: BlockCopy {
            addr: 0x10000,
            bytes: 1400,
        },
        v: BlockCopy {
            addr: 0x20000,
            bytes: 1400,
        },
    };
    validate_slot(&slot).unwrap();
    let planned = encoded_copies(&slot, 4096, 0, copies(), StorageFormat::Fp8FromBf16).unwrap();
    assert_eq!(
        planned[0].0,
        CopyRange {
            file_offset: 4096,
            device: 0x10000,
            bytes: 700
        }
    );
    assert_eq!(
        planned[1].0,
        CopyRange {
            file_offset: 5120,
            device: 0x20000,
            bytes: 1400
        }
    );
    assert_eq!(planned[0].1.logical_bytes, 1400);
    assert!(encoded_copies(&slot, 4096, 1, copies(), StorageFormat::Fp8FromBf16).is_err());
    assert!(encoded_copies(&slot, 4096, 0, copies(), StorageFormat::Ans).is_err());
    assert!(
        encoded_copies(
            &slot,
            4096,
            0,
            BlockCopies::Contiguous(BlockCopy {
                addr: 1,
                bytes: 2800
            }),
            StorageFormat::Fp8FromBf16
        )
        .is_err()
    );
    slot.encoding.as_mut().unwrap()[0].stored_bytes = 1500;
    assert!(validate_slot(&slot).is_err());
    slot.encoding.as_mut().unwrap().pop();
    assert!(validate_slot(&slot).is_err());
}

#[test]
fn invalid_slot_sizes_and_offset_overflow_are_rejected_before_gpu_io() {
    let mut slot = SlotMeta::new(smallvec![4096], crate::NumaNode::UNKNOWN);
    slot.total_size = 8192;
    assert!(validate_slot(&slot).is_err());
    slot.segment_sizes = smallvec![u64::MAX, 1];
    assert!(validate_slot(&slot).is_err());
    assert!(segment_offset(&slot, 1, 1, 0, 1).is_err());
}
