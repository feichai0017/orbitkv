use super::*;

fn contiguous_addr(layout: &KVCacheLayout, block_idx: usize) -> u64 {
    match layout.block_copies(block_idx).unwrap() {
        BlockCopies::Contiguous(c) => c.addr,
        BlockCopies::Split { .. } => panic!("expected contiguous copies"),
    }
}

#[test]
fn dense_layout_addresses() {
    let layout = KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
    assert!(!layout.is_split());
    assert_eq!(layout.padded_block_bytes(), 1024);
    assert_eq!(contiguous_addr(&layout, 0), 0x1000);
    assert_eq!(contiguous_addr(&layout, 5), 0x1000 + 5 * 1024);
    assert!(layout.block_copies(100).is_err());
}

#[test]
fn split_layout_addresses() {
    // 100 K blocks contiguous, then 100 V blocks: kv_stride = region size.
    let layout = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 100 * 1024, 2).unwrap();
    assert!(layout.is_split());
    match layout.block_copies(3).unwrap() {
        BlockCopies::Split { k, v } => {
            assert_eq!(k.addr, 0x1000 + 3 * 1024);
            assert_eq!(v.addr, 0x1000 + (100 + 3) * 1024);
            assert_eq!(k.bytes, 1024);
            assert_eq!(v.bytes, 1024);
        }
        BlockCopies::Contiguous(_) => panic!("expected split copies"),
    }
}

#[test]
fn adjacent_kv_collapses_to_contiguous() {
    // kv_stride == segment_bytes: one copy of both segments.
    let layout = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 1024, 2).unwrap();
    assert!(!layout.is_split());
    match layout.block_copies(0).unwrap() {
        BlockCopies::Contiguous(c) => assert_eq!(c.bytes, 2048),
        BlockCopies::Split { .. } => panic!("expected contiguous copies"),
    }
    assert_eq!(contiguous_addr(&layout, 1), 0x1000 + 2048);
}

#[test]
fn overlapping_kv_stride_rejected() {
    let err = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 512, 2).unwrap_err();
    assert!(err.contains("overlap"), "{err}");
}

#[test]
fn split_kv_regions_must_not_overlap() {
    let err = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 2048, 2).unwrap_err();
    assert!(err.contains("overlaps blocks 2 apart"), "{err}");
}

#[test]
fn sparse_split_layout_can_have_interleaved_disjoint_segments() {
    let layout = KVCacheLayout::new(0x1000, 1_600, 2, 100, 500, 2)
        .unwrap()
        .with_block_stride(1_000)
        .unwrap();

    match layout.block_copies(0).unwrap() {
        BlockCopies::Split { k, v } => {
            assert_eq!(k.addr, 0x1000);
            assert_eq!(v.addr, 0x1000 + 500);
        }
        BlockCopies::Contiguous(_) => panic!("expected split copies"),
    }
    match layout.block_copies(1).unwrap() {
        BlockCopies::Split { k, v } => {
            assert_eq!(k.addr, 0x1000 + 1_000);
            assert_eq!(v.addr, 0x1000 + 1_500);
        }
        BlockCopies::Contiguous(_) => panic!("expected split copies"),
    }
}

#[test]
fn adjacent_kv_default_stride_keeps_blocks_disjoint() {
    let layout = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 1024, 2).unwrap();

    let block0 = match layout.block_copies(0).unwrap() {
        BlockCopies::Contiguous(c) => c,
        BlockCopies::Split { .. } => panic!("expected contiguous copies"),
    };
    let block1 = match layout.block_copies(1).unwrap() {
        BlockCopies::Contiguous(c) => c,
        BlockCopies::Split { .. } => panic!("expected contiguous copies"),
    };

    assert_eq!(block0.addr + block0.bytes as u64, block1.addr);
}

#[test]
fn block_stride_decouples_step_from_copy_size() {
    let segment_bytes = 4096 * 2;
    let page_stride_bytes = 8192 * 2;
    let num_blocks = 8;
    let size_bytes = num_blocks * page_stride_bytes;

    let layout = KVCacheLayout::new(0x10000, size_bytes, num_blocks, segment_bytes, 0, 1)
        .unwrap()
        .with_block_stride(page_stride_bytes)
        .unwrap();

    assert_eq!(
        contiguous_addr(&layout, 3),
        0x10000 + 3 * page_stride_bytes as u64
    );
    assert_eq!(
        contiguous_addr(&layout, 7),
        0x10000 + 7 * page_stride_bytes as u64
    );
}

#[test]
fn block_stride_defaults_to_dense_and_validates() {
    let layout = KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
    assert_eq!(contiguous_addr(&layout, 5), 0x1000 + 5 * 1024);

    // Stride smaller than the block: overlap.
    assert!(
        KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1)
            .unwrap()
            .with_block_stride(512)
            .is_err()
    );

    // Strided layout exceeds the registered region.
    assert!(
        KVCacheLayout::new(0x1000, 8 * 1024, 8, 1024, 0, 1)
            .unwrap()
            .with_block_stride(4096)
            .is_err()
    );

    // Split K/V regions that are disjoint with the default dense stride can
    // overlap again if the explicit per-block stride grows too far.
    let err = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 100 * 1024, 2)
        .unwrap()
        .with_block_stride(2048)
        .unwrap_err();
    assert!(err.contains("overlaps blocks 50 apart"), "{err}");
}

#[test]
fn null_pointer_rejected() {
    assert!(KVCacheLayout::new(0, 1024, 10, 64, 0, 1).is_err());
}

#[test]
fn memory_too_small_rejected() {
    let err = KVCacheLayout::new(0x1000, 5120, 10, 1024, 0, 1).unwrap_err();
    assert!(err.contains("too small"), "{err}");
}

#[test]
fn ssd_padding() {
    // Unaligned: 8848 % 512 = 144, padded to 9216.
    let layout = KVCacheLayout::new(0x1000, 10_000_000, 100, 8848, 0, 1)
        .unwrap()
        .with_ssd_padding(512);
    assert_eq!(layout.padded_segment_bytes(), 9216);
    assert_eq!(layout.padded_block_bytes(), 9216);

    // Already aligned: no change.
    let layout = KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1)
        .unwrap()
        .with_ssd_padding(512);
    assert_eq!(layout.padded_segment_bytes(), 1024);
    assert_eq!(layout.padded_block_bytes(), 1024);

    // Split layout: padded per segment, total = padded * segments.
    let layout = KVCacheLayout::new(0x1000, 10_000_000, 100, 8848, 900_000, 2)
        .unwrap()
        .with_ssd_padding(512);
    assert_eq!(layout.padded_segment_bytes(), 9216);
    assert_eq!(layout.padded_block_bytes(), 9216 * 2);
}
