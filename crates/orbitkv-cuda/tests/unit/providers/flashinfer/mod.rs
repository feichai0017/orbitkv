mod jit;
use super::*;
use itertools::Itertools;

mod buckets;
mod capture;
mod geometry;

#[test]
fn resource_signature_depends_only_on_kv_lengths() {
    let attention = FlashInferAttention::default();
    let inputs = (0..6).map(NodeIndex::new).collect_vec();

    assert_eq!(
        attention.resource_buffer_nodes(&inputs),
        vec![inputs[1], inputs[2]]
    );
    assert!(attention.resource_buffer_nodes(&inputs[..2]).is_empty());
}

#[test]
fn device_resource_spec_is_pointer_free_and_capacity_adjusted() {
    let attention = FlashInferAttention {
        algorithm: crate::providers::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
        num_qo_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        page_size: 1,
        query_tokens: 1.into(),
        requests: 1.into(),
        context_dim: Expression::from('c'),
        dtype: DType::F16,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };
    let inputs = (0..4).map(NodeIndex::new).collect_vec();
    let kv_row_bytes = 2 * 64 * 2;
    let lengths = FxHashMap::from_iter([
        (inputs[1], 4096 * kv_row_bytes),
        (inputs[2], 4096 * kv_row_bytes),
    ]);
    let dyn_map = FxHashMap::from_iter([(Symbol::from('c'), 100)]);
    let resident_before = resident_shared_device_memory_allocations();

    let spec = attention
        .device_resource_spec(&inputs, &lengths, &dyn_map, true)
        .unwrap();
    let mut other_metadata_inputs = inputs.clone();
    other_metadata_inputs[3] = NodeIndex::new(99);
    let other_spec = attention
        .device_resource_spec(&other_metadata_inputs, &lengths, &dyn_map, true)
        .unwrap();

    assert_eq!(resident_shared_device_memory_allocations(), resident_before);
    assert!(spec.cache_key.is_some());
    assert_ne!(spec.cache_key, other_spec.cache_key);
    // Capacity tier 256: kv indptr (8), current-c (4), indices
    // (1024), last-page length (4), and temporary output (512).
    assert_eq!(
        spec.prepared_device_bytes().unwrap(),
        INT_WORKSPACE_SIZE + 1_552
    );
    assert_eq!(
        shared_device_memory_allocation().bytes,
        FLOAT_WORKSPACE_SIZE
    );
}

#[test]
fn device_resource_spec_uses_context_for_uninstalled_cache_sentinels() {
    let attention = FlashInferAttention {
        algorithm: crate::providers::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
        num_qo_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        page_size: 1,
        query_tokens: 1.into(),
        requests: 1.into(),
        context_dim: Expression::from('c'),
        dtype: DType::F16,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };
    let inputs = (0..4).map(NodeIndex::new).collect_vec();
    let dyn_map = FxHashMap::from_iter([(Symbol::from('c'), 100)]);
    let uninstalled_lengths = FxHashMap::from_iter([(inputs[1], 0), (inputs[2], 0)]);

    let spec = attention
        .device_resource_spec(&inputs, &uninstalled_lengths, &dyn_map, true)
        .unwrap();

    assert_eq!(spec.spec.max_kv_pages, 100);
    assert_eq!(spec.spec.c, 100);
    assert_eq!(
        spec.prepared_device_bytes().unwrap(),
        INT_WORKSPACE_SIZE + 928
    );
    assert_eq!(
        attention
            .device_resource_spec(&inputs, &FxHashMap::default(), &dyn_map, true)
            .unwrap_err(),
        ResourceViolation::HostResourcePlanning {
            name: "FlashInfer K-cache length"
        }
    );
}

#[test]
fn explicit_indptr_resource_specs_do_not_assume_cache_sharing() {
    let attention = FlashInferAttention {
        algorithm: crate::providers::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
        num_qo_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        page_size: 1,
        query_tokens: 2.into(),
        requests: 2.into(),
        context_dim: Expression::from('c'),
        dtype: DType::F16,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };
    let inputs = (0..6).map(NodeIndex::new).collect_vec();
    let kv_row_bytes = 2 * 64 * 2;
    let lengths = FxHashMap::from_iter([
        (inputs[1], 4096 * kv_row_bytes),
        (inputs[2], 4096 * kv_row_bytes),
        // Installed CSR length is checked at execution, not used to size
        // resources for other retained buckets.
        (inputs[5], 3 * std::mem::size_of::<i32>()),
    ]);
    let dyn_map = FxHashMap::from_iter([(Symbol::from('c'), 100)]);

    let spec = attention
        .device_resource_spec(&inputs, &lengths, &dyn_map, true)
        .unwrap();

    assert!(spec.cache_key.is_none());
    // No owned indptrs: indices (400), two last-page lengths (8),
    // and temporary output (1024).
    assert_eq!(
        spec.prepared_device_bytes().unwrap(),
        INT_WORKSPACE_SIZE + 1_432
    );
}

#[test]
fn external_page_plan_supports_block_pages_without_private_metadata() {
    let attention = FlashInferAttention::paged(
        crate::providers::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
        32,
        8,
        128,
        16,
        2.into(),
        3.into(),
        2.into(),
        DType::Bf16,
        0.0,
        Some(4095),
    );
    let inputs = (0..7).map(NodeIndex::new).collect_vec();
    let kv_page_bytes = 16 * 8 * 128 * 2;
    let lengths = FxHashMap::from_iter([
        (inputs[1], 64 * kv_page_bytes),
        (inputs[2], 64 * kv_page_bytes),
        (inputs[4], 3 * std::mem::size_of::<i32>()),
        (inputs[5], 3 * std::mem::size_of::<i32>()),
        (inputs[6], 2 * std::mem::size_of::<i32>()),
    ]);

    let spec = attention
        .device_resource_spec(&inputs, &lengths, &FxHashMap::default(), true)
        .unwrap();

    assert_eq!(spec.spec.page_size, 16);
    assert_eq!(spec.spec.batch_size, 2);
    assert_eq!(spec.spec.max_kv_pages, 64);
    assert!(spec.explicit_qo_indptr);
    assert!(spec.explicit_kv_indptr);
    assert!(spec.explicit_last_page_len);
    assert!(spec.cache_key.is_none());
    assert_eq!(
        spec.prepared_device_bytes().unwrap(),
        INT_WORKSPACE_SIZE + 2 * 32 * 128 * 2
    );
}

#[test]
fn external_page_plan_accepts_non_power_of_two_gqa_geometry() {
    let attention = FlashInferAttention::paged(
        crate::providers::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
        14,
        2,
        64,
        16,
        1.into(),
        1.into(),
        1.into(),
        DType::Bf16,
        0.0,
        None,
    );
    let inputs = (0..7).map(NodeIndex::new).collect_vec();
    let page_bytes = 16 * 2 * 64 * 2;
    let lengths = FxHashMap::from_iter([
        (inputs[1], 4 * page_bytes),
        (inputs[2], 4 * page_bytes),
        (inputs[4], 2 * std::mem::size_of::<i32>()),
        (inputs[5], 2 * std::mem::size_of::<i32>()),
        (inputs[6], std::mem::size_of::<i32>()),
    ]);

    let spec = attention
        .device_resource_spec(&inputs, &lengths, &FxHashMap::default(), false)
        .unwrap();
    assert_eq!(spec.spec.num_qo_heads / spec.spec.num_kv_heads, 7);
}

#[test]
fn semantic_debug_ignores_attention_plan_cache() {
    let mut attention = FlashInferAttention::paged(
        crate::providers::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
        2,
        1,
        64,
        16,
        's'.into(),
        'c'.into(),
        'b'.into(),
        DType::Bf16,
        0.0,
        None,
    );
    let before = format!("{attention:?}");
    *attention.plan_info.lock().unwrap() = vec![3, 5, 8];
    assert_eq!(before, format!("{attention:?}"));
    attention.head_dim *= 2;
    assert_ne!(before, format!("{attention:?}"));
}
