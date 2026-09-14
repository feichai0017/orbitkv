//! These contracts must build and saturate on a machine without CUDA.
use orbitkv_compiler::{
    prelude::*,
    search::{GeneticSearch, extract_one},
};
use orbitkv_ops::ops::{attention::*, linear::*};
use rand::SeedableRng;

fn linear_graph() -> Graph {
    let mut graph = Graph::default();
    let input = graph.tensor((2, 128)).as_dtype(DType::Bf16);
    let weight = graph.tensor((128, 128)).as_dtype(DType::F8E4M3);
    let scale = graph.tensor((1, 1));
    block_scaled_linear(
        input,
        weight,
        scale,
        BlockScaledLinearSpec {
            rows: 2.into(),
            output_features: 128,
            input_features: 128,
            weight_block_rows: FP8_SCALE_BLOCK,
            weight_block_columns: FP8_SCALE_BLOCK,
        },
    )
    .output();
    graph
}

#[test]
fn linear_semantics_build_without_a_backend_and_cannot_be_extracted_as_executable() {
    let mut graph = linear_graph();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    assert!(space.custom_ops.iter().all(|op| !op.is_lowered()));
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extract_one(space, &contexts[0], &mut rng);
    }));
    assert!(error.is_err());
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GeneticSearch::<std::time::Duration>::new(
            space,
            &contexts[0],
            &CompileOptions::default(),
            std::time::Instant::now(),
        );
    }));
    assert!(error.is_err());
}

#[test]
fn attention_semantics_build_without_a_backend() {
    let mut graph = Graph::default();
    let spec = AttentionSpec {
        query_heads: 4,
        kv_heads: 1,
        query_key_dim: 64,
        value_dim: 64,
        dtype: DType::Bf16,
        scale: 0.125,
        mask: AttentionMask::Causal,
    };
    let inputs = AttentionInputs {
        query: graph.tensor((2, 4, 64)).as_dtype(DType::Bf16),
        query_indptr: graph.tensor(2).as_dtype(DType::Int),
        kv: KvView::Paged(PagedKvView {
            state_class_id: 3,
            page_size: 16,
            layout: PagedKvLayout::TokenMajor,
            key: graph.tensor((8, 16, 1, 64)).as_dtype(DType::Bf16),
            value: graph.tensor((8, 16, 1, 64)).as_dtype(DType::Bf16),
            page_indices: graph.tensor(1).as_dtype(DType::Int),
            page_indptr: graph.tensor(2).as_dtype(DType::Int),
            last_page_len: graph.tensor(1).as_dtype(DType::Int),
        }),
    };
    let output = attention(inputs, spec).unwrap();
    assert_eq!(output.dims(), [4, 2, 64].map(Expression::from));
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    assert!(
        graph
            .search_space()
            .unwrap()
            .custom_ops
            .iter()
            .all(|op| !op.is_lowered())
    );
}

#[test]
#[should_panic(expected = "must belong to one graph")]
fn rejects_cross_graph_linear_inputs() {
    let mut left = Graph::default();
    let mut right = Graph::default();
    let input = left.tensor((1, 128)).as_dtype(DType::Bf16);
    let weight = right.tensor((128, 128)).as_dtype(DType::F8E4M3);
    let scale = left.tensor((1, 1));
    block_scaled_linear(
        input,
        weight,
        scale,
        BlockScaledLinearSpec {
            rows: 1.into(),
            output_features: 128,
            input_features: 128,
            weight_block_rows: FP8_SCALE_BLOCK,
            weight_block_columns: FP8_SCALE_BLOCK,
        },
    );
}
