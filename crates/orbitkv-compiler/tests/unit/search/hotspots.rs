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
        let mutation = candidate.targeted_mutation.as_ref().unwrap();
        assert_eq!(mutation.changes.len(), 1);
        let choice = &mutation.changes[0];
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

#[test]
fn measured_dependency_pair_can_win_without_an_improving_intermediate() {
    let mut graph = Graph::default();
    let inner = graph.tensor(1).sin().output();
    let outer = inner.sin().output();
    let other = graph.tensor(2).sin().output();
    graph.build_search_space::<ExplicitLoopRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&DynMap::default());
    for width in [1, 2] {
        let options = CompileOptions::default()
            .search_graph_limit(16)
            .hotspot_candidates(64)
            .hotspot_max_changes(std::num::NonZeroUsize::new(width).unwrap())
            .search_log(false);
        let mut search =
            GeneticSearch::new(space, &contexts[0], &options, std::time::Instant::now());
        let mut rng = rand::rngs::StdRng::seed_from_u64(71);
        let mut seed = search.next_candidate(&mut rng).unwrap();
        let signature = |llir: &LLIRGraph, output| format!("{:?}", llir[producer(llir, output)]);
        let originals = [inner.id, outer.id, other.id].map(|output| signature(&seed.llir, output));
        seed.profile = vec![ProfiledRegion {
            nodes: vec![producer(&seed.llir, outer.id)],
            cost: 10.0,
        }];
        search.report(seed, Outcome::Measured(10_usize, "seed".into()));
        let mut pair_measured = false;
        while let Some(candidate) = search.next_candidate(&mut rng) {
            if candidate.sampling != SamplingOrigin::Hotspot {
                search.report(
                    candidate,
                    Outcome::Rejected("fixture stops before random fallback".into()),
                );
                break;
            }
            assert_eq!(signature(&candidate.llir, other.id), originals[2]);
            let paired = signature(&candidate.llir, inner.id) != originals[0]
                && signature(&candidate.llir, outer.id) != originals[1];
            if paired {
                assert_eq!(
                    candidate.targeted_mutation.as_ref().unwrap().changes.len(),
                    2
                );
                pair_measured = true;
            }
            search.report(
                candidate,
                Outcome::Measured(
                    if paired { 1 } else { 11 },
                    "complete-program metric".into(),
                ),
            );
        }
        assert_eq!(pair_measured, width == 2);
        assert_eq!(search.into_ranked()[0].0, if width == 2 { 1 } else { 10 });
    }
}
