use super::*;

fn options() -> CompileOptions {
    CompileOptions::default()
        .dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 8)])
        .dim_buckets('b', &[DimBucket::new(1, 1), DimBucket::new(2, 4)])
}

fn shape(tokens: usize, batch: usize) -> DynMap {
    [(Symbol::from('s'), tokens), (Symbol::from('b'), batch)]
        .into_iter()
        .collect()
}

#[test]
fn explicit_joint_buckets_preserve_correlated_representatives() {
    let options = options().bucket_representatives(vec![shape(1, 1), shape(4, 3)]);
    let combinations = joint_bucket_combinations(&options);
    assert_eq!(combinations.len(), 2);
    assert_eq!(combinations[1].0[&Symbol::from('s')], 1);
    assert_eq!(combinations[1].0[&Symbol::from('b')], 1);
    assert_eq!(combinations[1].1.as_ref().unwrap()[&Symbol::from('b')], 3);
    let mut graph = Graph::default();
    graph.tensor(('s', 'b')).output();
    graph.build_search_space::<crate::prelude::ReferenceRuntime>(options);
    let contexts = graph
        .search_space()
        .unwrap()
        .bucket_contexts(&DynMap::default());
    assert_eq!(contexts[1].representative_dyn_map, shape(4, 3));
}

#[test]
fn explicit_joint_buckets_reject_duplicate_or_invalid_maps() {
    for representatives in [
        vec![],
        vec![shape(2, 2), shape(4, 3)],
        vec![shape(9, 1)],
        vec![[(Symbol::from('s'), 2)].into_iter().collect()],
    ] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                joint_bucket_combinations(&options().bucket_representatives(representatives))
            }))
            .is_err()
        );
    }
    assert_eq!(joint_bucket_combinations(&options()).len(), 4);
}
