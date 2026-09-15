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
    graph_with_context(case, case.page_indices.len().into())
}

fn graph_with_context(
    case: &Case,
    context_pages: Expression,
) -> (Graph, [GraphTensor; 7], GraphTensor) {
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
        graph.tensor(context_pages).as_dtype(DType::Int),
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

#[test]
fn dynamic_context_requires_a_bucket_capacity_proof() {
    use orbitkv_compiler::graph::DimBucket;

    let case = Case::new(256, 16, &[1, 1], &[35, 19]);
    let (mut graph, _, _) = graph_with_context(&case, 'c'.into());
    graph.set_dim('c', case.page_indices.len());
    let options = CompileOptions::default()
        .compiler_facts(crate::target::CudaTarget { major: 9, minor: 0 }.compiler_facts());
    graph.build_search_space::<AttentionRuntime>(options.clone());
    assert!(
        !graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(name, _)| name == "FlashAttention")
    );
    graph.build_search_space::<AttentionRuntime>(options.dim_buckets(
        'c',
        &[
            DimBucket::new(2, 8).representative(5),
            DimBucket::new(9, 32).representative(16),
        ],
    ));
    let space = graph.search_space().unwrap();
    assert_eq!(space.buckets.len(), 2);
    for (bucket, capacity) in space.buckets.iter().zip([8, 32]) {
        let egraph = &bucket.egraph;
        let choices = egraph
            .enodes
            .values()
            .filter(|(name, _)| name == "FlashAttention");
        let mut count = 0;
        for (_, fields) in choices {
            let expression = &egraph.eclasses[&fields[5]].1[0];
            assert_eq!(
                extract_expr(egraph, expression, &mut FxHashMap::default())
                    .unwrap()
                    .to_usize(),
                Some(capacity),
            );
            count += 1;
        }
        assert_eq!(count, 1);
    }
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
    check_search_replay(false);
}

#[test]
fn bounded_context_search_and_replay_keep_reserved_input_capacity() {
    check_search_replay(true);
}

fn check_search_replay(bounded: bool) {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let mut case = Case::new(256, 16, &[2, 1], &[35, 7]);
    let capacity = case.page_indices.len() * 2;
    let build = || {
        if bounded {
            let (mut graph, tensors, output) = graph_with_context(&case, 'c'.into());
            graph.set_dim('c', case.page_indices.len());
            (graph, tensors, output)
        } else {
            graph(&case)
        }
    };
    let populate_reserved = |runtime: &mut AttentionRuntime, tensors: &[GraphTensor; 7]| {
        populate(runtime, tensors, &case);
        runtime.set_data_with_capacity(
            tensors[3],
            case.page_indices.clone(),
            capacity * size_of::<i32>(),
        );
    };
    let (mut searched, tensors, _) = build();
    let mut options = CompileOptions::default().search_graph_limit(2);
    if bounded {
        options = options.dim_buckets(
            'c',
            &[
                orbitkv_compiler::graph::DimBucket::new(case.last_page_len.len(), capacity)
                    .representative(case.page_indices.len()),
            ],
        );
    }
    let mut runtime = AttentionRuntime::initialize(stream.clone());
    populate_reserved(&mut runtime, &tensors);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    runtime = searched.compile_with_rng(runtime, options.clone(), &mut rng);
    let serialized = serde_json::to_vec(searched.selected_schedule().unwrap()).unwrap();
    assert!(String::from_utf8_lossy(&serialized).contains("FlashAttention"));
    let modules = runtime.capture_module_artifact(&searched).unwrap();
    drop(runtime);
    drop(searched);

    let (mut loaded, tensors, output) = build();
    loaded.prepare_selected_schedule(&options);
    loaded.install_selected_schedule(serde_json::from_slice(&serialized).unwrap());
    let mut replay = AttentionRuntime::initialize(stream);
    populate_reserved(&mut replay, &tensors);
    replay
        .load_selected_schedule_with_modules(&loaded, &modules)
        .unwrap();
    let pages = case.page_indices.clone();
    let histories = if bounded {
        vec![vec![0, 3, 4], vec![0, 1, 2], vec![0, 2, 3], vec![0, 3, 4]]
    } else {
        vec![case.page_indptr.clone(); 3]
    };
    let mut captured = None;
    for indptr in histories {
        case.page_indices = pages[..*indptr.last().unwrap() as usize].to_vec();
        case.page_indptr = indptr;
        if bounded {
            loaded.set_dim('c', case.page_indices.len());
        }
        populate(&mut replay, &tensors, &case);
        replay.execute(&loaded.dyn_map);
        let residency = replay.cuda_graph_residency_stats();
        assert_eq!(*captured.get_or_insert(residency), residency);
        let actual = replay.get_bf16(output);
        let expected = case.expected(None);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && (actual.to_f32() - expected).abs() < 0.016);
        }
    }
}
