use super::*;

#[test]
fn builds_external_page_plan_node() {
    let mut graph = Graph::default();
    let query_tokens = Expression::from('s');
    let context_pages = Expression::from('c');
    let q = graph
        .named_tensor("q", (query_tokens, 4, 64))
        .as_dtype(DType::Bf16);
    let k = graph
        .named_tensor("k", (8, 16, 1, 64))
        .as_dtype(DType::Bf16);
    let v = graph
        .named_tensor("v", (8, 16, 1, 64))
        .as_dtype(DType::Bf16);
    let metadata = PagedAttentionMetadata::new(&mut graph, 0, 2.into(), context_pages);
    let inputs = PagedAttentionInputs {
        q,
        k_cache: k,
        v_cache: v,
        query_tokens,
        context_pages,
    };
    let class = AttentionClass {
        class_id: 0,
        name: "attention".into(),
        layers: vec![0].into_boxed_slice(),
        page_tokens: 16,
        key_bytes_per_token_per_layer: 128,
        value_bytes_per_token_per_layer: 128,
        visibility: AttentionVisibility::Sliding { window_tokens: 64 },
    };
    let kernel = AttentionGeometry {
        query_heads: 4,
        kv_heads: 1,
        head_dim: 64,
        dtype: DType::Bf16,
        softmax_scale: 64_f64.sqrt().recip(),
    };
    for softmax_scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            paged_attention(
                inputs,
                metadata,
                &class,
                AttentionGeometry {
                    softmax_scale,
                    ..kernel
                }
            ),
            Err(ExecutorError::InvalidKernelGeometry)
        ));
    }
    let output = paged_attention(inputs, metadata, &class, kernel).unwrap();
    assert_eq!(output.dims(), &[4.into(), query_tokens, 64.into()]);
    assert_eq!(graph.get_sources(output.id).len(), 7);
}

#[test]
fn orbitkv_layout_facts_bind_to_paged_attention_during_search_build() {
    use orbitkv::{
        AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage,
        compile_runtime_manifest, plan::RetentionKind,
    };
    use orbitkv_compiler::graph::CompileOptions;
    use orbitkv_cuda::runtime::CudaRuntime;

    let manifest = compile_runtime_manifest(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![AttentionStateSpec {
            name: "attention".into(),
            layers: vec![0],
            storage: AttentionStateStorage::TokenKv {
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                retention: RetentionKind::Full,
                window_tokens: None,
            },
        }],
    })
    .unwrap();
    let plan = crate::ExecutorPlan::compile(&manifest).unwrap();
    let facts = plan
        .compiler_facts(&[crate::ExecutorArena {
            engine_epoch: 1,
            pool_epoch: 1,
            pool_id: 1,
            class_id: 0,
            backend_domain: 1,
            first_page_id: 1,
            page_count: 8,
            backend_base_index: 0,
        }])
        .unwrap();

    let mut graph = Graph::default();
    let query_tokens = Expression::from('s');
    let context_pages = Expression::from('c');
    let q = graph
        .named_tensor("q", (query_tokens, 4, 64))
        .as_dtype(DType::Bf16);
    let k = graph
        .named_tensor("k", (8, 16, 1, 64))
        .as_dtype(DType::Bf16);
    let v = graph
        .named_tensor("v", (8, 16, 1, 64))
        .as_dtype(DType::Bf16);
    let metadata = PagedAttentionMetadata::new(&mut graph, 0, 1.into(), context_pages);
    paged_attention(
        PagedAttentionInputs {
            q,
            k_cache: k,
            v_cache: v,
            query_tokens,
            context_pages,
        },
        metadata,
        &plan.classes[0],
        AttentionGeometry {
            query_heads: 4,
            kv_heads: 1,
            head_dim: 64,
            dtype: DType::Bf16,
            softmax_scale: 64_f64.sqrt().recip(),
        },
    )
    .unwrap()
    .output();
    graph.set_dim('s', 1);
    graph.set_dim('c', 1);
    graph.build_search_space::<CudaRuntime>(
        CompileOptions::default().compiler_facts(facts.egglog().to_owned()),
    );
    let egraph = graph.egraph().unwrap();
    assert!(
        egraph
            .enodes
            .values()
            .any(|(op, _)| op == "persistent-state-attention-op")
    );
}
