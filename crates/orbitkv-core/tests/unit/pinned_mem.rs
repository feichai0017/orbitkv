use super::*;

#[test]
fn test_allocate_cuda_host_alloc() {
    // Skip if no CUDA context available
    if cudarc::driver::CudaContext::new(0).is_err() {
        return;
    }

    let mem = PinnedMemory::allocate_cuda_host_alloc(4096).unwrap();
    assert!(mem.size() >= 4096);
}

#[test]
fn test_allocate_regular() {
    if cudarc::driver::CudaContext::new(0).is_err() {
        return;
    }

    let mem = PinnedMemory::allocate_regular(4096, NumaNode::UNKNOWN).unwrap();
    assert!(mem.size() >= 4096);
}

#[test]
fn test_zero_size_fails() {
    assert!(matches!(
        PinnedMemory::allocate_cuda_host_alloc(0),
        Err(PinnedMemError::ZeroSize)
    ));
    assert!(matches!(
        PinnedMemory::allocate_regular(0, NumaNode::UNKNOWN),
        Err(PinnedMemError::ZeroSize)
    ));
}

#[test]
fn test_read_hugepage_size() {
    // Hugepagesize is always present in /proc/meminfo on Linux
    let size = read_hugepage_size_from_proc();
    assert!(size.is_some(), "Hugepagesize should exist in /proc/meminfo");

    let size = size.unwrap();
    // Common sizes: 2MB (default), 1GB
    assert!(
        size >= 2 * 1024 * 1024,
        "Hugepage size should be at least 2MB"
    );
    assert!(
        size.is_power_of_two(),
        "Hugepage size should be power of two"
    );
}
