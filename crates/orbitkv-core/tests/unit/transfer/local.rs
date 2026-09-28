use super::*;

#[test]
fn raw_plan_round_trip_and_framing() {
    let plan = RawRestorePlan {
        copies: vec![RawCopy {
            source: SourceRange {
                arena_id: 1,
                allocation_id: 4,
                allocation_offset: 4096,
                allocation_size: 8192,
                offset: 6144,
                size: 512,
            },
            layer: "decoder.0.key".into(),
            destination_offset: 2048,
        }],
    };
    let bytes = plan.encode().unwrap();
    assert_eq!(RawRestorePlan::decode(&bytes).unwrap(), plan);
    for end in 0..bytes.len() {
        assert!(RawRestorePlan::decode(&bytes[..end]).is_err());
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(RawRestorePlan::decode(&extra).is_err());
    let mut invalid_count = bytes;
    invalid_count[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(RawRestorePlan::decode(&invalid_count).is_err());
}

#[test]
fn raw_plan_encoding_obeys_session_budget() {
    let copy = RawCopy {
        source: SourceRange {
            arena_id: 1,
            allocation_id: 1,
            allocation_offset: 0,
            allocation_size: 1,
            offset: 0,
            size: 1,
        },
        layer: "layer".into(),
        destination_offset: 0,
    };
    let plan = RawRestorePlan {
        copies: vec![copy; MAX_PLAN_BYTES / 63 + 1],
    };
    assert!(plan.encode().is_err());
    assert!(RawRestorePlan::decode(&vec![0; MAX_PLAN_BYTES + 1]).is_err());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn caller_context_survives_registration_and_readiness_success_and_failure() {
    let primary = CudaContext::new(0).unwrap();
    let destination_stream = primary.new_stream().unwrap();
    // Match the CUDA-IPC-capable allocations exported by framework tensors.
    // Cudarc's alloc_zeros uses cuMemAllocAsync, whose context attribute is null
    // and whose allocation cannot use the legacy CUDA IPC registration path.
    // SAFETY: the primary context is current; the allocation remains owned by
    // this test until the executor has been dropped below.
    let address = unsafe { result::malloc_sync(64) }.unwrap();
    let caller = CudaContext::new_non_primary(0, 0).unwrap();
    let caller_stream = caller.new_stream().unwrap();
    let current = || result::ctx::get_current().unwrap();
    let binding = |pointer| {
        LocalTensor::new("layer".into(), pointer, 64, pointer as usize, 1, 64, 0, 1).unwrap()
    };

    let mut executor =
        LocalRestoreExecutor::new(0, vec![binding(address)], Vec::new(), TransferMode::Direct)
            .unwrap();
    assert_eq!(current(), Some(caller.cu_ctx()));
    executor
        .wait_for_destination(destination_stream.cu_stream() as u64)
        .unwrap();
    assert_eq!(current(), Some(caller.cu_ctx()));
    assert!(
        executor
            .wait_for_destination(caller_stream.cu_stream() as u64)
            .is_err()
    );
    assert_eq!(current(), Some(caller.cu_ctx()));

    // A real allocation in a custom context must still be rejected. Accepting
    // another current context does not permit unrelated destination pointers.
    // SAFETY: this thread owns the live caller context and frees this allocation
    // after the constructor has rejected it without submitting GPU work.
    let custom_pointer = unsafe { result::malloc_sync(64) }.unwrap();
    let rejected = LocalRestoreExecutor::new(
        0,
        vec![binding(custom_pointer)],
        Vec::new(),
        TransferMode::Direct,
    );
    assert!(rejected.is_err());
    assert_eq!(current(), Some(caller.cu_ctx()));
    unsafe { result::free_sync(custom_pointer) }.unwrap();
    drop(executor);
    primary.bind_to_thread().unwrap();
    unsafe { result::free_sync(address) }.unwrap();
}
