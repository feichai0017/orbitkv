use super::*;
use crate::{
    egglog_utils::{ClassId, NodeId},
    hlir::ReferenceRuntime,
    prelude::Graph,
    search::BucketSearchSpace,
};
use rand::{SeedableRng, rngs::StdRng};

fn space_with_unused_choices() -> SearchSpace {
    let mut graph = Graph::default();
    graph.tensor(1).output();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let source = graph.search_space().unwrap();
    let bucket = &source.buckets[0];
    let mut egraph = bucket.egraph.clone();
    for site in 0..8 {
        let class = ClassId::from(format!("unused-{site}"));
        let nodes = (0..4)
            .map(|n| NodeId::from(format!("unused-{site}-{n}")))
            .collect::<Vec<_>>();
        for node in &nodes {
            egraph.enodes.insert(node.clone(), ("Input".into(), vec![]));
            egraph.node_to_class.insert(node.clone(), class.clone());
        }
        egraph.eclasses.insert(class, ("IR".into(), nodes));
    }
    SearchSpace {
        buckets: vec![BucketSearchSpace {
            egraph,
            bucket_indices: bucket.bucket_indices.clone(),
            representative_override: bucket.representative_override.clone(),
            intervals: bucket.intervals.clone(),
        }],
        ops: source.ops.clone(),
        custom_ops: source.custom_ops.clone(),
        dim_buckets: source.dim_buckets.clone(),
    }
}

#[test]
fn duplicate_programs_consume_the_bounded_coverage_allowance() {
    let space = space_with_unused_choices();
    let contexts = space.bucket_contexts(&DynMap::default());
    let options = CompileOptions::default()
        .search_graph_limit(8)
        .initial_population(4)
        .search_log(false);
    let mut search = GeneticSearch::new(&space, &contexts[0], &options, Instant::now());
    let mut rng = StdRng::seed_from_u64(71);
    let first = search.next_candidate(&mut rng).unwrap();
    assert_eq!(first.sampling, SamplingOrigin::Coverage);
    search.report(first, Outcome::Measured(1_usize, "fixture".into()));
    search.breed(&mut rng);
    assert_eq!(search.coverage_generated, 4);
    // All unused bindings extract to the already measured program.
    while let Some(genome) = search.pending.pop_front() {
        assert!(search.extract(&genome).unwrap().is_none());
    }
    search.breed(&mut rng);
    assert_eq!(search.generation_sampling, SamplingOrigin::Mutation);
    assert!(search.pending.is_empty());
    assert_eq!(search.measured(), 1);
}

#[test]
fn initial_population_respects_the_total_graph_budget() {
    let space = space_with_unused_choices();
    let contexts = space.bucket_contexts(&DynMap::default());
    let options = CompileOptions::default()
        .search_graph_limit(2)
        .initial_population(32)
        .search_log(false);
    let mut search = GeneticSearch::new(&space, &contexts[0], &options, Instant::now());
    let mut rng = StdRng::seed_from_u64(71);
    let first = search.next_candidate(&mut rng).unwrap();
    search.report(first, Outcome::Measured(1_usize, "fixture".into()));
    search.breed(&mut rng);
    assert_eq!(search.coverage_generated, 2);
    assert_eq!(search.pending.len(), 1);
}
