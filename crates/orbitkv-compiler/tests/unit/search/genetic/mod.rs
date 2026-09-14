use super::*;
use crate::{hlir::ReferenceRuntime, prelude::Graph};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn report_observer_sees_the_timeout_used_for_ranking() {
    let mut graph = Graph::default();
    graph.tensor(1).output();
    let options = CompileOptions::default()
        .candidate_timeout(std::time::Duration::ZERO)
        .search_log(false);
    graph.build_search_space::<ReferenceRuntime>(options.clone());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let mut search = GeneticSearch::new(space, &contexts[0], &options, Instant::now());
    let mut rng = StdRng::seed_from_u64(59);
    let candidate = search.next_candidate(&mut rng).unwrap();
    let id = candidate.id;
    let mut observed = false;
    search.report_with_observer(
        candidate,
        Outcome::Measured(1_usize, "fixture".into()),
        |candidate, outcome, timed_out| {
            assert_eq!(candidate.id, id);
            assert!(matches!(outcome, Outcome::Measured(1, _)));
            assert!(timed_out);
            observed = true;
        },
    );
    assert!(observed);
    assert!(search.into_ranked().is_empty());
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
        })
        .unwrap_or_default()
}

#[test]
fn initial_runtime_rejections_observe_time_budget_and_report_reason() {
    let mut graph = Graph::default();
    graph.tensor(1).output();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let options = CompileOptions::default().search_time_limit(std::time::Duration::ZERO);
    let mut search = GeneticSearch::<usize>::new(space, &contexts[0], &options, Instant::now());
    let mut rng = StdRng::seed_from_u64(59);
    let first = search.next_candidate(&mut rng).unwrap();
    search.report(
        first,
        Outcome::Rejected("fixture has no valid deployment".to_owned()),
    );
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        search.next_candidate(&mut rng)
    }));
    let message = panic_text(
        failure
            .err()
            .expect("expired initial search must terminate"),
    );
    assert!(message.contains("Search time limit expired"));
    assert!(message.contains("after 1 attempts"));
    assert!(message.contains("fixture has no valid deployment"));
}

#[test]
fn exhausted_initial_space_reports_last_rejection() {
    let mut graph = Graph::default();
    graph.tensor(1).output();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let options = CompileOptions::default();
    let mut search = GeneticSearch::<usize>::new(space, &contexts[0], &options, Instant::now());
    let mut rng = StdRng::seed_from_u64(127);
    let first = search.next_candidate(&mut rng).unwrap();
    search.report(
        first,
        Outcome::Rejected("fixture resource contract".to_owned()),
    );
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        search.next_candidate(&mut rng)
    }));
    let message = panic_text(failure.err().expect("finite rejected space must terminate"));
    assert!(message.contains("Failed to find a viable initial genome"));
    assert!(message.contains("fixture resource contract"));
}
