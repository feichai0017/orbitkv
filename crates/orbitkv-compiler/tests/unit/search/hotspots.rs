use super::*;
use crate::search::{GeneticSearch, ProfiledRegion, SamplingOrigin};
use petgraph::Direction;

fn producer(llir: &LLIRGraph, output: NodeIndex) -> NodeIndex {
    let output = llir
        .node_indices()
        .find(|&node| {
            llir[node]
                .to_op::<crate::hlir::Output>()
                .is_some_and(|op| op.node == output.index())
        })
        .unwrap();
    llir.neighbors_directed(output, Direction::Incoming)
        .next()
        .unwrap()
}

fn run_snapshot(
    space: &SearchSpace,
    hot_output: NodeIndex,
    other_output: NodeIndex,
    reject: bool,
) -> Vec<(String, String, String)> {
    let contexts = space.bucket_contexts(&DynMap::default());
    let options = CompileOptions::default()
        .search_graph_limit(3)
        .hotspot_candidates(2)
        .search_log(false);
    let mut search = GeneticSearch::new(space, &contexts[0], &options, std::time::Instant::now());
    let mut rng = rand::rngs::StdRng::seed_from_u64(71);
    let mut seed = search.next_candidate(&mut rng).unwrap();
    let fixed = format!("{:?}", seed.llir[producer(&seed.llir, other_output)]);
    let hot = producer(&seed.llir, hot_output);
    seed.profile = vec![ProfiledRegion {
        nodes: vec![hot],
        cost: 10.0,
    }];
    search.report(seed, Outcome::Measured(10_usize, "seed".into()));
    let mut sequence = Vec::new();
    while let Some(mut candidate) = search.next_candidate(&mut rng) {
        if candidate.sampling != SamplingOrigin::Hotspot {
            search.report(
                candidate,
                Outcome::Rejected("end fixture exploration".into()),
            );
            break;
        }
        assert_eq!(
            format!(
                "{:?}",
                candidate.llir[producer(&candidate.llir, other_output)]
            ),
            fixed,
            "a local change disturbed the independently constrained branch"
        );
        let choice = candidate.targeted_choice.as_ref().unwrap();
        sequence.push((choice.class.clone(), choice.from.clone(), choice.to.clone()));
        let hot = producer(&candidate.llir, hot_output);
        candidate.profile = vec![ProfiledRegion {
            nodes: vec![hot],
            cost: 1.0,
        }];
        // A rejection must not install its attractive feedback or displace the
        // valid seed. Measured candidates deliberately improve whole-graph cost.
        search.report(
            candidate,
            if reject {
                Outcome::Rejected("state contract".into())
            } else {
                Outcome::Measured(1, "improved".into())
            },
        );
    }
    assert!(!sequence.is_empty());
    assert!(sequence.len() <= 2);
    let ranked = search.into_ranked();
    assert_eq!(ranked[0].0, if reject { 10 } else { 1 });
    sequence
}

#[test]
fn fixed_snapshot_feedback_changes_only_the_hot_branch_and_keeps_a_valid_fallback() {
    let mut graph = Graph::default();
    let hot = graph.tensor(1).sin().output();
    let other = graph.tensor(2).sin().output();
    graph.build_search_space::<ExplicitLoopRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    for reject in [false, true] {
        assert_eq!(
            run_snapshot(space, hot.id, other.id, reject),
            run_snapshot(space, hot.id, other.id, reject)
        );
    }
}
