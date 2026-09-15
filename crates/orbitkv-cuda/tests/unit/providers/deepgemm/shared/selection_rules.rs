use super::*;
use orbitkv_compiler::graph::DimBucket;

#[test]
fn dynamic_rules_use_each_bucket_bound_and_reject_unbounded_rows() {
    let mut graph = Graph::default();
    graph.set_dim('s', 4);
    let x = graph.tensor(('s', 128)).as_dtype(DType::Bf16);
    let w = graph.tensor((128, 128)).as_dtype(DType::F8E4M3);
    let ws = graph.tensor((1, 1));
    block_scaled_linear(
        x,
        w,
        ws,
        BlockScaledLinearSpec {
            rows: 's'.into(),
            output_features: 128,
            input_features: 128,
            weight_block_rows: BLOCK,
            weight_block_columns: BLOCK,
        },
    )
    .output();
    graph.build_search_space::<RewriteRuntime>(CompileOptions::default());
    assert_eq!(count_operations(graph.egraph().unwrap(), "DeepGemm"), 0);
    graph.build_search_space::<RewriteRuntime>(CompileOptions::default().dim_buckets(
        's',
        &[
            DimBucket::new(1, 16).representative(4),
            DimBucket::new(17, 256).representative(32),
        ],
    ));
    let space = graph.search_space().unwrap();
    assert_eq!(space.buckets.len(), 2);
    for (bucket, limit) in space.buckets.iter().zip([16, 256]) {
        let egraph = &bucket.egraph;
        let choices = egraph
            .enodes
            .values()
            .filter(|(label, _)| label == "DeepGemm");
        let mut count = 0;
        for (_, children) in choices {
            let descriptor = &egraph.eclasses[&children[1]].1[0];
            let encoded: String = serde_json::from_str(&egraph.enodes[descriptor].0).unwrap();
            let choice: Selection = serde_json::from_str(&encoded).unwrap();
            choice.validate().unwrap();
            assert_eq!(choice.row_limit, limit);
            assert_eq!(choice.config.num_sms, 78);
            count += 1;
        }
        assert_eq!(count, selection::SEARCH_VARIANTS);
    }
}
