use super::*;

fn copy(device: u64, host: *mut u8, device_allocation: usize, host_allocation: usize) -> CopyDesc {
    CopyDesc {
        device,
        host,
        host_device: 0,
        size: 4,
        device_allocation,
        host_allocation,
    }
}

#[test]
fn merge_requires_contiguous_ranges_from_the_same_allocations() {
    let mut host = [0_u8; 12];
    let first_host = host.as_mut_ptr();
    // SAFETY: `host` contains eight bytes, so this points to its second half.
    let second_host = unsafe { first_host.add(4) };

    let merged: Vec<_> =
        merged_ranges(&[copy(100, first_host, 1, 2), copy(104, second_host, 1, 2)]).collect();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].size, 8);

    for second in [
        copy(104, second_host, 3, 2),
        copy(104, second_host, 1, 4),
        copy(108, second_host, 1, 2),
        copy(104, first_host.wrapping_add(8), 1, 2),
    ] {
        assert_eq!(
            merged_ranges(&[copy(100, first_host, 1, 2), second]).count(),
            2
        );
    }
    assert_eq!(merged_ranges(&[]).count(), 0);
}

#[test]
fn coalescing_preserves_execution_order_and_each_merged_address() {
    let mut host = [0_u8; 16];
    let first = host.as_mut_ptr();
    let copies = [
        copy(100, first, 1, 2),
        copy(104, first.wrapping_add(4), 1, 2),
        copy(112, first.wrapping_add(12), 1, 2),
        copy(108, first.wrapping_add(8), 1, 2),
    ];
    let merged: Vec<_> = merged_ranges(&copies)
        .map(|range| (range.device, range.host, range.size))
        .collect();
    assert_eq!(
        merged,
        vec![
            (100, first, 8),
            (112, first.wrapping_add(12), 4),
            (108, first.wrapping_add(8), 4),
        ]
    );
}
