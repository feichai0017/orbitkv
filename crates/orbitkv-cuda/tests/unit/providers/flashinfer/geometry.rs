use super::*;

fn attention() -> FlashInferAttention {
    FlashInferAttention::paged(
        FlashInferAlgorithm::TensorCore,
        4,
        2,
        64,
        16,
        Symbol::new("query_rows").into(),
        Symbol::new("context_pages").into(),
        Symbol::new("request_groups").into(),
        DType::Bf16,
        0.0,
        None,
    )
}

fn dimensions(queries: usize, requests: usize) -> DynMap {
    DynMap::from_iter([
        (Symbol::new("query_rows"), queries),
        (Symbol::new("context_pages"), 8),
        (Symbol::new("request_groups"), requests),
    ])
}

#[test]
fn retained_bucket_geometry_is_independent_of_installed_csr_lengths() {
    let op = attention();
    let inputs = (0..7).map(NodeIndex::new).collect_vec();
    let page_bytes = op.page_size * op.num_kv_heads * op.head_dim * 2;
    let mut lengths =
        FxHashMap::from_iter([(inputs[1], 64 * page_bytes), (inputs[2], 64 * page_bytes)]);
    for (queries, requests) in [(1, 1), (8, 8), (32, 8)] {
        let expected = op
            .device_resource_spec(&inputs, &lengths, &dimensions(queries, requests), true)
            .unwrap();
        assert_eq!(expected.spec.total_q_tokens, queries);
        assert_eq!(expected.spec.batch_size, requests);
        // The last profiled bucket can leave any of these logical lengths in
        // stable metadata allocations. Planning must use its own dimensions.
        for installed_requests in [1, 2, 8] {
            for input in [inputs[4], inputs[5]] {
                lengths.insert(input, (installed_requests + 1) * size_of::<i32>());
            }
            lengths.insert(inputs[6], installed_requests * size_of::<i32>());
            let actual = op
                .device_resource_spec(&inputs, &lengths, &dimensions(queries, requests), true)
                .unwrap();
            assert_eq!(actual.spec, expected.spec);
            assert_eq!(
                actual.prepared_device_bytes().unwrap(),
                expected.prepared_device_bytes().unwrap()
            );
        }
    }
}

#[test]
fn execution_checks_each_metadata_length_against_the_compiled_requests() {
    let op = attention();
    let nodes = (0..8).map(NodeIndex::new).collect_vec();
    let context_pages = 8;
    let requests = 3;
    let queries = 6;
    let page_bytes = op.page_size * op.num_kv_heads * op.head_dim * 2;
    let output_bytes = queries * op.num_qo_heads * op.head_dim * 2;
    let lengths = [
        output_bytes,
        64 * page_bytes,
        64 * page_bytes,
        context_pages * size_of::<i32>(),
        (requests + 1) * size_of::<i32>(),
        (requests + 1) * size_of::<i32>(),
        requests * size_of::<i32>(),
        output_bytes,
    ];
    // Resolution inspects metadata and returns pointer bindings; no device
    // access or provider preparation occurs in this contract test.
    let mut buffers = nodes
        .iter()
        .copied()
        .zip(lengths.map(|length| DeviceBuffer::new(0, length)))
        .collect::<FxHashMap<_, _>>();
    let dims = dimensions(queries, requests);
    assert_eq!(
        op.resolve_for_graph(nodes[7], &nodes[..7], &buffers, &dims)
            .unwrap()
            .spec
            .batch_size,
        requests
    );
    for (index, name) in [(4, "qo_indptr"), (5, "kv_indptr"), (6, "kv_last_page_len")] {
        for invalid_length in [0, lengths[index] - 1, lengths[index] + size_of::<i32>()] {
            buffers.insert(nodes[index], DeviceBuffer::new(0, invalid_length));
            let error = op
                .resolve_for_graph(nodes[7], &nodes[..7], &buffers, &dims)
                .unwrap_err();
            assert!(error.to_string().contains(name), "{error}");
        }
        buffers.insert(nodes[index], DeviceBuffer::new(0, lengths[index]));
    }
}

#[test]
fn implicit_metadata_cannot_describe_packed_multi_request_prefill() {
    let op = attention();
    let inputs = (0..4).map(NodeIndex::new).collect_vec();
    let lengths = FxHashMap::from_iter([(inputs[1], 0), (inputs[2], 0)]);
    for (queries, requests, valid) in [(8, 1, true), (8, 8, true), (8, 2, false)] {
        assert_eq!(
            op.device_resource_spec(&inputs, &lengths, &dimensions(queries, requests), false)
                .is_ok(),
            valid
        );
    }
    let mut missing = dimensions(1, 1);
    missing.remove(&Symbol::new("request_groups"));
    assert_eq!(
        op.device_resource_spec(&inputs, &lengths, &missing, false)
            .unwrap_err(),
        ResourceViolation::UnresolvedExpression {
            resource: "FlashInfer request count"
        }
    );
}
