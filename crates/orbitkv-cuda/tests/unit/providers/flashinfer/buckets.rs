use super::*;
use crate::runtime::CudaRuntimeImpl;
use half::bf16;
use orbitkv_ops::ops::attention::*;
use rand::{SeedableRng, rngs::SmallRng};

#[test]
#[ignore = "requires CUDA and FlashInfer; compiles and replays different request-count buckets"]
fn retained_request_buckets_plan_and_execute_their_own_geometry() {
    type Runtime = CudaRuntimeImpl<(crate::kernel::Ops, AttentionSemantics, FlashInferAttention)>;
    const CAPACITY: usize = 8;
    const HEAD_DIM: usize = 64;
    const PAGE_SIZE: usize = 16;
    const QUERY_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    let requests = Symbol::new("request_groups");
    let rows = Expression::from(requests);
    let mut graph = Graph::default();
    let query = graph
        .tensor((rows, QUERY_HEADS, HEAD_DIM))
        .as_dtype(DType::Bf16)
        .persist();
    let key = graph
        .tensor((CAPACITY, PAGE_SIZE, KV_HEADS, HEAD_DIM))
        .as_dtype(DType::Bf16)
        .persist();
    let value = graph
        .tensor((CAPACITY, PAGE_SIZE, KV_HEADS, HEAD_DIM))
        .as_dtype(DType::Bf16)
        .persist();
    let indices = graph.tensor(rows).as_dtype(DType::Int).persist();
    let qptr = graph.tensor(rows + 1).as_dtype(DType::Int).persist();
    let kptr = graph.tensor(rows + 1).as_dtype(DType::Int).persist();
    let last = graph.tensor(rows).as_dtype(DType::Int).persist();
    let output = attention(
        AttentionInputs {
            query,
            query_indptr: qptr,
            kv: KvView::Paged(PagedKvView {
                state_class_id: 0,
                key,
                value,
                page_size: PAGE_SIZE,
                layout: PagedKvLayout::TokenMajor,
                page_indices: indices,
                page_indptr: kptr,
                last_page_len: last,
            }),
        },
        AttentionSpec {
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            query_key_dim: HEAD_DIM,
            value_dim: HEAD_DIM,
            dtype: DType::Bf16,
            scale: 1.0 / (HEAD_DIM as f64).sqrt(),
            mask: AttentionMask::Causal,
        },
    )
    .unwrap()
    .output();
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let mut runtime = Runtime::initialize(context.default_stream());
    runtime.set_data(query, vec![bf16::ZERO; CAPACITY * QUERY_HEADS * HEAD_DIM]);
    runtime.set_data(
        key,
        vec![bf16::ZERO; CAPACITY * PAGE_SIZE * KV_HEADS * HEAD_DIM],
    );
    runtime.set_data(
        value,
        vec![bf16::from_f32(0.5); CAPACITY * PAGE_SIZE * KV_HEADS * HEAD_DIM],
    );
    let metadata = move |dimensions: &DynMap| {
        let count = dimensions[&requests] as i32;
        vec![
            (indices.id, (0..count).collect()),
            (qptr.id, (0..=count).collect()),
            (kptr.id, (0..=count).collect()),
            (last.id, vec![1; count as usize]),
        ]
    };
    graph.set_dim(requests, CAPACITY);
    for (input, values) in metadata(&graph.dyn_map) {
        runtime.set_data(input, values);
    }
    runtime.register_profile_input_generator(metadata);
    // Search profiles the larger bucket last. Both must survive aggregate
    // resource planning with those larger CSR buffers still installed.
    runtime = graph.compile_with_rng(
        runtime,
        CompileOptions::default().search_graph_limit(1).dim_buckets(
            requests,
            &[
                DimBucket::new(1, 1),
                DimBucket::new(2, CAPACITY).representative(CAPACITY),
            ],
        ),
        &mut SmallRng::seed_from_u64(47),
    );
    for count in [1, CAPACITY, 3, 1] {
        graph.set_dim(requests, count);
        runtime.set_data(query, vec![bf16::ZERO; count * QUERY_HEADS * HEAD_DIM]);
        for (input, values) in metadata(&graph.dyn_map) {
            runtime.set_data(input, values);
        }
        runtime.execute(&graph.dyn_map);
        let actual = runtime.get_bf16(output);
        assert_eq!(actual.len(), count * QUERY_HEADS * HEAD_DIM);
        // One live KV token per request: softmax is exactly one, independent
        // of the selected decode algorithm, bucket, or output head layout.
        assert!(actual.iter().all(|value| value.to_f32() == 0.5));
    }
}
