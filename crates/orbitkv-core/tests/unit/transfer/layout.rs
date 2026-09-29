use super::*;

fn contiguous_addr(layout: &KVCacheLayout, block_idx: usize) -> u64 {
    match layout.block_copies(block_idx).unwrap() {
        BlockCopies::Contiguous(c) => c.addr,
        BlockCopies::Split { .. } => panic!("expected contiguous copies"),
    }
}

#[test]
fn dense_geometry_binds_independently_to_local_addresses() {
    let geometry = KVCacheGeometry::new(100, 1024, 0, 1, None, 1).unwrap();
    assert!(!geometry.is_split());
    assert_eq!(geometry.padded_block_bytes(), 1024);
    assert_eq!(
        geometry.block_ranges(5).unwrap(),
        BlockRanges::Contiguous(5 * 1024..6 * 1024)
    );
    assert!(geometry.block_ranges(100).is_err());
    for base in [0x1000, 0x90000] {
        let layout = KVCacheLayout::bind(base, 100 * 1024, geometry.clone()).unwrap();
        assert_eq!(layout.geometry().num_blocks(), 100);
        assert_eq!(contiguous_addr(&layout, 0), base);
        assert_eq!(contiguous_addr(&layout, 5), base + 5 * 1024);
        assert!(layout.block_copies(100).is_err());
    }
}

#[test]
fn split_geometry_preserves_separate_kv_ranges() {
    let geometry = KVCacheGeometry::new(100, 1024, 100 * 1024, 2, None, 1).unwrap();
    assert!(geometry.is_split());
    assert_eq!(
        geometry.block_ranges(3).unwrap(),
        BlockRanges::Split {
            k: 3 * 1024..4 * 1024,
            v: 103 * 1024..104 * 1024,
        }
    );
    let layout = KVCacheLayout::bind(0x1000, 200 * 1024, geometry).unwrap();
    match layout.block_copies(3).unwrap() {
        BlockCopies::Split { k, v } => {
            assert_eq!(k.addr, 0x1000 + 3 * 1024);
            assert_eq!(v.addr, 0x1000 + 103 * 1024);
            assert_eq!((k.bytes, v.bytes), (1024, 1024));
        }
        BlockCopies::Contiguous(_) => panic!("expected split copies"),
    }
}

#[test]
fn adjacent_kv_collapses_without_overlapping_blocks() {
    let geometry = KVCacheGeometry::new(100, 1024, 1024, 2, None, 1).unwrap();
    assert!(!geometry.is_split());
    let layout = KVCacheLayout::bind(0x1000, 200 * 1024, geometry).unwrap();
    match layout.block_copies(0).unwrap() {
        BlockCopies::Contiguous(c) => assert_eq!(c.bytes, 2048),
        BlockCopies::Split { .. } => panic!("expected contiguous copies"),
    }
    assert_eq!(contiguous_addr(&layout, 1), 0x1000 + 2048);
}

#[test]
fn geometry_rejects_overlapping_segments_and_blocks() {
    for (segments, kv_stride, block_stride, expected) in [
        (2, 512, None, "segments would overlap"),
        (2, 2048, None, "overlaps blocks 2 apart"),
        (2, 100 * 1024, Some(2048), "overlaps blocks 50 apart"),
        (1, 0, Some(512), "blocks would overlap"),
    ] {
        let error =
            KVCacheGeometry::new(100, 1024, kv_stride, segments, block_stride, 1).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn final_stride_is_validated_without_an_invalid_dense_intermediate() {
    // Dense K blocks overlap V blocks, but the registered sparse layout is
    // disjoint. Validate the actual stride without constructing the dense one.
    assert!(KVCacheGeometry::new(4, 100, 200, 2, None, 1).is_err());
    let geometry = KVCacheGeometry::new(4, 100, 200, 2, Some(1000), 1).unwrap();
    assert_eq!(
        geometry.block_ranges(3).unwrap(),
        BlockRanges::Split {
            k: 3000..3100,
            v: 3200..3300,
        }
    );
    let layout = KVCacheLayout::bind(0x1000, 3300, geometry).unwrap();
    match layout.block_copies(3).unwrap() {
        BlockCopies::Split { k, v } => {
            assert_eq!(k.addr, 0x1000 + 3000);
            assert_eq!(v.addr, 0x1000 + 3200);
        }
        BlockCopies::Contiguous(_) => panic!("expected split copies"),
    }
}

#[test]
fn block_stride_decouples_step_from_copy_size() {
    let segment_bytes = 4096 * 2;
    let page_stride_bytes = 8192 * 2;
    let geometry =
        KVCacheGeometry::new(8, segment_bytes, 0, 1, Some(page_stride_bytes), 1).unwrap();
    assert!(KVCacheLayout::bind(0x10000, 8 * segment_bytes, geometry.clone()).is_err());
    let layout = KVCacheLayout::bind(0x10000, 8 * page_stride_bytes, geometry).unwrap();
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
fn binding_rejects_invalid_local_address_ranges() {
    let geometry = KVCacheGeometry::new(10, 1024, 0, 1, None, 1).unwrap();
    for (base, bytes, expected) in [
        (0, 10240, "null"),
        (0x1000, 0, "size_bytes"),
        (0x1000, 5120, "too small"),
        (u64::MAX - 100, 10240, "overflows"),
    ] {
        let error = KVCacheLayout::bind(base, bytes, geometry.clone()).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn ssd_padding_changes_host_footprint_without_changing_gpu_ranges() {
    for (segment_bytes, kv_stride, segments, expected) in [
        (8848, 0, 1, 9216),
        (1024, 0, 1, 1024),
        (8848, 900_000, 2, 9216),
    ] {
        let raw = KVCacheGeometry::new(100, segment_bytes, kv_stride, segments, None, 1).unwrap();
        let padded =
            KVCacheGeometry::new(100, segment_bytes, kv_stride, segments, None, 512).unwrap();
        assert_eq!(padded.padded_segment_bytes(), expected);
        assert_eq!(padded.padded_block_bytes(), expected * segments);
        assert_eq!(
            raw.block_ranges(99).unwrap(),
            padded.block_ranges(99).unwrap()
        );
    }
}

#[test]
fn geometry_rejects_offset_and_padding_overflow() {
    for (num_blocks, segment_bytes, kv_stride, segments, stride, alignment, expected) in [
        (1, usize::MAX, 1, 2, None, 1, "block size overflow"),
        (usize::MAX, 2, 0, 1, None, 1, "memory layout overflow"),
        (3, 1, 0, 1, Some(usize::MAX), 1, "memory layout overflow"),
        (1, 2, usize::MAX, 2, None, 1, "memory layout overflow"),
        (1, usize::MAX, 0, 1, None, 2, "padded segment size overflow"),
        (
            1,
            usize::MAX / 2,
            usize::MAX / 2,
            2,
            None,
            2,
            "padded block size overflow",
        ),
        (1, 1, 0, 1, None, 0, "alignment must be > 0"),
    ] {
        let error = KVCacheGeometry::new(
            num_blocks,
            segment_bytes,
            kv_stride,
            segments,
            stride,
            alignment,
        )
        .unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}
