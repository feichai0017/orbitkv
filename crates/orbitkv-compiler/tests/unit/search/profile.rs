use super::*;

#[test]
fn loop_instances_accumulate_and_fused_regions_share_their_measured_cost() {
    let a = ChoiceSite(0);
    let b = ChoiceSite(1);
    let fixed = ChoiceSite(2);
    let origins: Vec<Arc<[ChoiceSite]>> =
        vec![vec![a, fixed].into(), vec![a].into(), vec![b].into()];
    let regions = vec![
        ProfiledRegion {
            nodes: vec![NodeIndex::new(0)],
            cost: 4.0,
        },
        ProfiledRegion {
            nodes: vec![NodeIndex::new(1), NodeIndex::new(2)],
            cost: 6.0,
        },
    ];
    assert_eq!(
        aggregate_costs(&origins, &regions, |s| s != fixed),
        vec![(a, 7.0), (b, 3.0)]
    );
}

#[test]
fn invalid_feedback_does_not_create_a_search_priority() {
    let origins: Vec<Arc<[ChoiceSite]>> = vec![vec![ChoiceSite(0)].into()];
    let regions = [f64::NAN, f64::INFINITY, -1.0, 0.0]
        .into_iter()
        .map(|cost| ProfiledRegion {
            nodes: vec![NodeIndex::new(0)],
            cost,
        })
        .collect::<Vec<_>>();
    assert!(aggregate_costs(&origins, &regions, |_| true).is_empty());
}

#[test]
fn duplicate_origins_do_not_multiply_cost_and_ties_are_snapshot_ordered() {
    let origins: Vec<Arc<[ChoiceSite]>> =
        vec![vec![ChoiceSite(1), ChoiceSite(0), ChoiceSite(1)].into()];
    let regions = vec![ProfiledRegion {
        nodes: vec![NodeIndex::new(0), NodeIndex::new(0)],
        cost: 8.0,
    }];
    assert_eq!(
        aggregate_costs(&origins, &regions, |_| true),
        vec![(ChoiceSite(0), 4.0), (ChoiceSite(1), 4.0)]
    );
}
