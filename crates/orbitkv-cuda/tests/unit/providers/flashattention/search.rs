use super::*;
use crate::runtime::CudaRuntimeImpl;
use half::bf16;
use orbitkv_compiler::{
    graph::CompileOptions,
    op::Runtime,
    prelude::{Graph, GraphTensor},
};
use orbitkv_ops::ops::attention::*;
use rand::SeedableRng;

type AttentionRuntime = CudaRuntimeImpl<(crate::kernel::Ops, AttentionSemantics, FlashAttention)>;

fn graph(case: &Case) -> (Graph, [GraphTensor; 7], GraphTensor) {
    let mut graph = Graph::default();
    let dtype = DType::Bf16;
    let pages = case.page_indices.len();
    let requests = case.last_page_len.len();
    let tensors = [
        graph
            .tensor((case.query_tokens, case.query_heads, case.head_dim))
            .as_dtype(dtype),
        graph
            .tensor((pages, case.page_size, case.kv_heads, case.head_dim))
            .as_dtype(dtype),
        graph
            .tensor((pages, case.page_size, case.kv_heads, case.head_dim))
            .as_dtype(dtype),
        graph.tensor(pages).as_dtype(DType::Int),
        graph.tensor(requests + 1).as_dtype(DType::Int),
        graph.tensor(requests + 1).as_dtype(DType::Int),
        graph.tensor(requests).as_dtype(DType::Int),
    ]
    .map(|tensor| tensor.persist());
    let [
        query,
        key,
        value,
        page_indices,
        query_indptr,
        page_indptr,
        last_page_len,
    ] = tensors;
    let output = attention(
        AttentionInputs {
            query,
            query_indptr,
            kv: KvView::Paged(PagedKvView {
                state_class_id: 0,
                key,
                value,
                page_indices,
                page_indptr,
                last_page_len,
                page_size: case.page_size,
                layout: PagedKvLayout::TokenMajor,
            }),
        },
        AttentionSpec {
            query_heads: case.query_heads,
            kv_heads: case.kv_heads,
            query_key_dim: case.head_dim,
            value_dim: case.head_dim,
            dtype,
            scale: case.scale(),
            mask: AttentionMask::Causal,
        },
    )
    .unwrap()
    .output();
    (graph, tensors, output)
}

fn populate(runtime: &mut AttentionRuntime, tensors: &[GraphTensor; 7], case: &Case) {
    for (tensor, values) in tensors[..3]
        .iter()
        .zip([&case.query, &case.key, &case.value])
    {
        runtime.set_data(
            *tensor,
            values
                .iter()
                .map(|&value| bf16::from_f32(value))
                .collect::<Vec<_>>(),
        );
    }
    for (tensor, values) in tensors[3..].iter().zip([
        &case.page_indices,
        &case.query_indptr,
        &case.page_indptr,
        &case.last_page_len,
    ]) {
        runtime.set_data(*tensor, values.clone());
    }
}

#[test]
fn semantic_search_extracts_and_replays_flashattention_schedule() {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let case = Case::new(256, 16, &[2, 1], &[35, 7]);
    let (mut searched, tensors, _) = graph(&case);
    let options = CompileOptions::default().search_graph_limit(2);
    let mut runtime = AttentionRuntime::initialize(stream.clone());
    populate(&mut runtime, &tensors, &case);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    runtime = searched.compile_with_rng(runtime, options.clone(), &mut rng);
    let serialized = serde_json::to_vec(searched.selected_schedule().unwrap()).unwrap();
    assert!(String::from_utf8_lossy(&serialized).contains("FlashAttention"));
    let modules = runtime.capture_module_artifact(&searched).unwrap();
    drop(runtime);
    drop(searched);

    let (mut loaded, tensors, output) = graph(&case);
    loaded.prepare_selected_schedule(&options);
    loaded.install_selected_schedule(serde_json::from_slice(&serialized).unwrap());
    let mut replay = AttentionRuntime::initialize(stream);
    populate(&mut replay, &tensors, &case);
    replay
        .load_selected_schedule_with_modules(&loaded, &modules)
        .unwrap();
    populate(&mut replay, &tensors, &case);
    for _ in 0..3 {
        replay.execute(&loaded.dyn_map);
        let actual = replay.get_bf16(output);
        let expected = case.expected(None);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && (actual.to_f32() - expected).abs() < 0.016);
        }
    }
}
