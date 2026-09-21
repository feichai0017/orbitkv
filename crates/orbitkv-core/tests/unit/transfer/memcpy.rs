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
    let mut host = [0_u8; 8];
    let first_host = host.as_mut_ptr();
    // SAFETY: `host` contains eight bytes, so this points to its second half.
    let second_host = unsafe { first_host.add(4) };

    let merged = merge(&[copy(100, first_host, 1, 2), copy(104, second_host, 1, 2)]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].size, 8);

    for second in [copy(104, second_host, 3, 2), copy(104, second_host, 1, 4)] {
        assert_eq!(merge(&[copy(100, first_host, 1, 2), second]).len(), 2);
    }
}
