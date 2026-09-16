use crate::{providers::flashinfer::FlashInferAttention, runtime::CudaRuntime};
use half::bf16;
use orbitkv_compiler::{dtype::DType, prelude::Expression};
use orbitkv_compiler::{
    graph::CompileOptions,
    op::{EgglogOp, IntoEgglogOp, Runtime},
    prelude::Graph,
};
use orbitkv_ops::ops::attention::*;
use rand::SeedableRng;

#[test]
fn unsupported_attention_combinations_never_become_provider_candidates() {
    let base = AttentionSpec {
        query_heads: 4,
        kv_heads: 1,
        query_key_dim: 64,
        value_dim: 64,
        dtype: DType::Bf16,
        scale: 0.125,
        mask: AttentionMask::Causal,
    };
    for (spec, layout, query_tokens, requests, admitted) in [
        (base, PagedKvLayout::TokenMajor, 2, 1, true),
        (
            AttentionSpec {
                dtype: DType::F32,
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            2,
            true,
        ),
        (
            AttentionSpec {
                dtype: DType::F32,
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            1,
            false,
        ),
        (
            AttentionSpec {
                query_key_dim: 96,
                value_dim: 96,
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            1,
            false,
        ),
        (
            AttentionSpec {
                value_dim: 32,
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            1,
            false,
        ),
        (
            AttentionSpec {
                dtype: DType::F64,
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            1,
            false,
        ),
        (
            AttentionSpec {
                dtype: DType::F32,
                query_key_dim: 512,
                value_dim: 512,
                ..base
            },
            PagedKvLayout::TokenMajor,
            1,
            1,
            false,
        ),
        (
            AttentionSpec {
                mask: AttentionMask::Unmasked,
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            1,
            false,
        ),
        (
            AttentionSpec {
                mask: AttentionMask::Sliding {
                    window_left: i32::MAX as usize + 1,
                },
                ..base
            },
            PagedKvLayout::TokenMajor,
            2,
            1,
            false,
        ),
        (base, PagedKvLayout::HeadMajor, 2, 1, false),
    ] {
        let mut graph = Graph::default();
        let inputs = AttentionInputs {
            query: graph
                .tensor((query_tokens, spec.query_heads, spec.query_key_dim))
                .as_dtype(spec.dtype),
            query_indptr: graph.tensor(requests + 1).as_dtype(DType::Int),
            kv: KvView::Paged(PagedKvView {
                state_class_id: 0,
                key: graph
                    .tensor((8, 16, spec.kv_heads, spec.query_key_dim))
                    .as_dtype(spec.dtype),
                value: graph
                    .tensor((8, 16, spec.kv_heads, spec.value_dim))
                    .as_dtype(spec.dtype),
                page_size: 16,
                layout,
                page_indices: graph.tensor(2).as_dtype(DType::Int),
                page_indptr: graph.tensor(requests + 1).as_dtype(DType::Int),
                last_page_len: graph.tensor(requests).as_dtype(DType::Int),
            }),
        };
        attention(inputs, spec).unwrap();
        graph.build_search_space::<CudaRuntime>(
            CompileOptions::default()
                .compiler_facts(crate::target::CudaTarget { major: 9, minor: 0 }.compiler_facts()),
        );
        let present = graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(label, _)| label == "FlashInferAttention");
        assert_eq!(
            present, admitted,
            "{spec:?} {layout:?}, q={query_tokens}, b={requests}"
        );
        assert!(
            graph
                .custom_ops
                .iter()
                .all(|op| !op.to_llir_op().is_lowered())
        );
    }
}

#[test]
fn flashinfer_is_a_provider_for_paged_attention() {
    let mut graph = Graph::default();
    let query_tokens = Expression::from('s');
    let context_pages = Expression::from('c');
    let request_count = Expression::from('b');
    let output = attention(
        AttentionInputs {
            query: graph.tensor((query_tokens, 4, 64)).as_dtype(DType::Bf16),
            query_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
            kv: KvView::Paged(PagedKvView {
                state_class_id: 3,
                key: graph.tensor((8, 16, 1, 64)).as_dtype(DType::Bf16),
                value: graph.tensor((8, 16, 1, 64)).as_dtype(DType::Bf16),
                page_size: 16,
                layout: PagedKvLayout::TokenMajor,
                page_indices: graph.tensor(context_pages).as_dtype(DType::Int),
                page_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
                last_page_len: graph.tensor(request_count).as_dtype(DType::Int),
            }),
        },
        AttentionSpec {
            query_heads: 4,
            kv_heads: 1,
            query_key_dim: 64,
            value_dim: 64,
            dtype: DType::Bf16,
            scale: 0.125,
            mask: AttentionMask::Causal,
        },
    )
    .unwrap();
    output.output();
    graph.set_dim('s', 1);
    graph.set_dim('c', 1);
    graph.set_dim('b', 1);
    // Static search has no device instance. Missing or unsupported targets
    // must not admit this provider; changing the explicit target must rebuild
    // eligibility without any process-global cached GPU probe.
    for (target, admitted) in [
        (None, false),
        (
            Some(crate::target::CudaTarget { major: 7, minor: 5 }),
            false,
        ),
        (Some(crate::target::CudaTarget { major: 8, minor: 0 }), true),
        (Some(crate::target::CudaTarget { major: 9, minor: 0 }), true),
    ] {
        graph.build_search_space::<CudaRuntime>(CompileOptions::default().compiler_facts(
            target.map_or_else(String::new, crate::target::CudaTarget::compiler_facts),
        ));
        let present = graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(label, _)| label == "FlashInferAttention");
        assert_eq!(present, admitted, "execution target {target:?}");
    }
    assert!(
        <CudaRuntime as Runtime>::Ops::into_vec()
            .iter()
            .any(|op| op.sort().name == FlashInferAttention::default().sort().name)
    );
}

#[test]
fn compiler_search_admits_both_sixteen_bit_decode_families() {
    let mut graph = Graph::default();
    let query_tokens = Expression::from('s');
    let request_count = Expression::from('b');
    attention(
        AttentionInputs {
            query: graph.tensor((query_tokens, 24, 256)).as_dtype(DType::Bf16),
            query_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
            kv: KvView::Paged(PagedKvView {
                state_class_id: 3,
                key: graph.tensor((8, 16, 4, 256)).as_dtype(DType::Bf16),
                value: graph.tensor((8, 16, 4, 256)).as_dtype(DType::Bf16),
                page_size: 16,
                layout: PagedKvLayout::TokenMajor,
                page_indices: graph.tensor('c').as_dtype(DType::Int),
                page_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
                last_page_len: graph.tensor(request_count).as_dtype(DType::Int),
            }),
        },
        AttentionSpec {
            query_heads: 24,
            kv_heads: 4,
            query_key_dim: 256,
            value_dim: 256,
            dtype: DType::Bf16,
            scale: 0.0625,
            mask: AttentionMask::Causal,
        },
    )
    .unwrap();
    graph.set_dim('s', 1);
    graph.set_dim('b', 1);
    graph.set_dim('c', 1);
    graph.build_search_space::<CudaRuntime>(
        CompileOptions::default()
            .compiler_facts(crate::target::CudaTarget { major: 9, minor: 0 }.compiler_facts()),
    );
    let egraph = graph.egraph().unwrap();
    let algorithms = egraph
        .enodes
        .values()
        .filter(|(label, _)| label.starts_with('\"'))
        .map(|(label, _)| label.as_str())
        .collect::<Vec<_>>();
    assert!(algorithms.contains(&"\"tensor-core\""));
    assert!(algorithms.contains(&"\"cuda-core-decode\""));
}

#[test]
fn compiler_facts_can_restrict_attention_provider_candidates() {
    let candidates = |policy: &str| {
        let mut graph = Graph::default();
        let query_tokens = Expression::from('s');
        let request_count = Expression::from('b');
        attention(
            AttentionInputs {
                query: graph.tensor((query_tokens, 24, 256)).as_dtype(DType::Bf16),
                query_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
                kv: KvView::Paged(PagedKvView {
                    state_class_id: 3,
                    key: graph.tensor((8, 16, 4, 256)).as_dtype(DType::Bf16),
                    value: graph.tensor((8, 16, 4, 256)).as_dtype(DType::Bf16),
                    page_size: 16,
                    layout: PagedKvLayout::TokenMajor,
                    page_indices: graph.tensor('c').as_dtype(DType::Int),
                    page_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
                    last_page_len: graph.tensor(request_count).as_dtype(DType::Int),
                }),
            },
            AttentionSpec {
                query_heads: 24,
                kv_heads: 4,
                query_key_dim: 256,
                value_dim: 256,
                dtype: DType::Bf16,
                scale: 0.0625,
                mask: AttentionMask::Causal,
            },
        )
        .unwrap();
        graph.set_dim('s', 1);
        graph.set_dim('b', 1);
        graph.set_dim('c', 1);
        graph.build_search_space::<CudaRuntime>(
            CompileOptions::default()
                .dim_buckets(
                    'c',
                    &[orbitkv_compiler::prelude::DimBucket::new(1, 8).representative(1)],
                )
                .compiler_facts(format!(
                    "{}\n(set (cuda-attention-policy) \"{policy}\")",
                    crate::target::CudaTarget { major: 9, minor: 0 }.compiler_facts(),
                )),
        );
        graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .filter(|(label, _)| matches!(label.as_str(), "FlashAttention" | "FlashInferAttention"))
            .map(|(label, _)| label.clone())
            .collect::<Vec<_>>()
    };
    let flashattention = candidates("flashattention");
    assert!(flashattention.iter().any(|name| name == "FlashAttention"));
    assert!(
        !flashattention
            .iter()
            .any(|name| name == "FlashInferAttention")
    );
    let flashinfer = candidates("flashinfer");
    assert!(flashinfer.iter().any(|name| name == "FlashInferAttention"));
    assert!(!flashinfer.iter().any(|name| name == "FlashAttention"));
}

#[test]
fn search_profiles_available_attention_algorithms() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let mut graph = Graph::default();
    let query = graph
        .named_tensor("query", (1, 24, 256))
        .as_dtype(DType::Bf16);
    let key_cache = graph
        .named_tensor("key_cache", (2, 16, 4, 256))
        .as_dtype(DType::Bf16);
    let value_cache = graph
        .named_tensor("value_cache", (2, 16, 4, 256))
        .as_dtype(DType::Bf16);
    let page_indices = graph.named_tensor("page_indices", 2).as_dtype(DType::Int);
    let query_indptr = graph.named_tensor("query_indptr", 2).as_dtype(DType::Int);
    let page_indptr = graph.named_tensor("page_indptr", 2).as_dtype(DType::Int);
    let last_page_len = graph.named_tensor("last_page_len", 1).as_dtype(DType::Int);
    let output = attention(
        AttentionInputs {
            query,
            query_indptr,
            kv: KvView::Paged(PagedKvView {
                state_class_id: 3,
                key: key_cache,
                value: value_cache,
                page_size: 16,
                layout: PagedKvLayout::TokenMajor,
                page_indices,
                page_indptr,
                last_page_len,
            }),
        },
        AttentionSpec {
            query_heads: 24,
            kv_heads: 4,
            query_key_dim: 256,
            value_dim: 256,
            dtype: DType::Bf16,
            scale: 0.0625,
            mask: AttentionMask::Causal,
        },
    )
    .unwrap()
    .output();
    let stream = crate::cudarc::driver::CudaContext::new(0)
        .unwrap()
        .new_stream()
        .unwrap();
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.set_data(query, vec![bf16::from_f32(0.125); 24 * 256]);
    runtime.set_data(key_cache, vec![bf16::from_f32(0.25); 2 * 16 * 4 * 256]);
    runtime.set_data(value_cache, vec![bf16::from_f32(0.5); 2 * 16 * 4 * 256]);
    runtime.set_data(page_indices, vec![1_i32, 0]);
    runtime.set_data(query_indptr, vec![0_i32, 1]);
    runtime.set_data(page_indptr, vec![0_i32, 2]);
    runtime.set_data(last_page_len, vec![3_i32]);
    graph.build_search_space::<CudaRuntime>(
        CompileOptions::default().compiler_facts(runtime.compilation_facts()),
    );
    let egraph = graph.egraph().unwrap();
    assert!(
        egraph
            .enodes
            .values()
            .any(|(label, _)| label == "\"tensor-core\"")
    );
    assert!(
        egraph
            .enodes
            .values()
            .any(|(label, _)| label == "FlashInferAttention")
    );
    let mut rng = orbitkv_compiler::prelude::rand::rngs::SmallRng::seed_from_u64(7);
    runtime = graph.search_with_rng(
        runtime,
        CompileOptions::default().search_graph_limit(8),
        &mut rng,
    );
    runtime.set_data(query, vec![bf16::from_f32(0.125); 24 * 256]);
    runtime.set_data(key_cache, vec![bf16::from_f32(0.25); 2 * 16 * 4 * 256]);
    runtime.set_data(value_cache, vec![bf16::from_f32(0.5); 2 * 16 * 4 * 256]);
    runtime.set_data(page_indices, vec![1_i32, 0]);
    runtime.set_data(query_indptr, vec![0_i32, 1]);
    runtime.set_data(page_indptr, vec![0_i32, 2]);
    runtime.set_data(last_page_len, vec![3_i32]);
    runtime.execute(&graph.dyn_map);
    let actual = runtime.get_bf16(output);
    assert_eq!(actual.len(), 24 * 256);
    assert!(actual.iter().all(|value| value.to_f32() == 0.5));
    let schedule = serde_json::to_string(graph.selected_schedule().unwrap()).unwrap();
    assert!(schedule.contains("FlashInferAttention") || schedule.contains("FlashAttention"));
}

#[test]
fn paged_attention_semantics_are_provider_neutral() {
    type SemanticOnlyOps = (crate::kernel::Ops, AttentionSemantics);
    type SemanticOnlyRuntime = crate::runtime::CudaRuntimeImpl<SemanticOnlyOps>;

    let mut graph = Graph::default();
    let query_tokens = Expression::from('s');
    let context_pages = Expression::from('c');
    let request_count = Expression::from('b');
    attention(
        AttentionInputs {
            query: graph.tensor((query_tokens, 4, 64)).as_dtype(DType::Bf16),
            query_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
            kv: KvView::Paged(PagedKvView {
                state_class_id: 3,
                key: graph.tensor((8, 16, 1, 64)).as_dtype(DType::Bf16),
                value: graph.tensor((8, 16, 1, 64)).as_dtype(DType::Bf16),
                page_size: 16,
                layout: PagedKvLayout::TokenMajor,
                page_indices: graph.tensor(context_pages).as_dtype(DType::Int),
                page_indptr: graph.tensor(request_count + 1).as_dtype(DType::Int),
                last_page_len: graph.tensor(request_count).as_dtype(DType::Int),
            }),
        },
        AttentionSpec {
            query_heads: 4,
            kv_heads: 1,
            query_key_dim: 64,
            value_dim: 64,
            dtype: DType::Bf16,
            scale: 0.125,
            mask: AttentionMask::Causal,
        },
    )
    .unwrap()
    .output();
    graph.set_dim('s', 1);
    graph.set_dim('c', 1);
    graph.set_dim('b', 1);
    graph.build_search_space::<SemanticOnlyRuntime>(CompileOptions::default());
    let egraph = graph.egraph().unwrap();
    assert!(
        !egraph
            .enodes
            .values()
            .any(|(label, _)| label == "FlashInferAttention")
    );
    assert!(
        graph
            .custom_ops
            .iter()
            .all(|op| !op.to_llir_op().is_lowered())
    );
}
