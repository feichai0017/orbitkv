use super::*;

fn reserve(budget: &Arc<QueryBudget>, instance: &str, bytes: u64) -> QueryReservation {
    match budget.reserve(instance, "state", bytes, QueryMode::Demand) {
        QueryAdmission::Admitted(reservation) => reservation,
        _ => panic!("expected admission"),
    }
}

#[test]
fn bytes_remain_charged_until_all_consumers_finish() {
    let budget = QueryBudget::new(100, 80).unwrap();
    let first = reserve(&budget, "a", 70);
    assert!(matches!(
        budget.reserve("a", "state", 11, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    let second = reserve(&budget, "b", 30);
    assert!(matches!(
        budget.reserve("c", "state", 1, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "state", 81, QueryMode::Demand),
        QueryAdmission::TooLarge
    ));
    assert!(first.ready(71).is_err());
    first.ready(40).unwrap();
    let gpu = first.clone();
    gpu.restoring();
    drop(first);
    assert_eq!(budget.usage.lock().total, 70);
    drop(gpu);
    assert_eq!(budget.usage.lock().total, 30);
    drop(second);
    assert_eq!(budget.usage.lock().total, 0);
    assert!(budget.usage.lock().instances.is_empty());
}

#[test]
fn warmups_leave_foreground_headroom_and_hold_bytes_until_io_drains() {
    let budget = QueryBudget::new(100, 80).unwrap();
    let QueryAdmission::Admitted(first) = budget.reserve("a", "ns", 20, QueryMode::Warmup) else {
        panic!("warmup should fit");
    };
    assert!(
        first.ready(20).is_err(),
        "warmup cannot become a restore lease"
    );
    assert!(matches!(
        budget.reserve("a", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "ns", 21, QueryMode::Warmup),
        QueryAdmission::TooLarge
    ));
    let QueryAdmission::Admitted(second) = budget.reserve("b", "ns", 5, QueryMode::Warmup) else {
        panic!("global warmup budget should fit");
    };
    assert!(matches!(
        budget.reserve("c", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Busy
    ));
    let foreground = reserve(&budget, "a", 60);
    let other_foreground = reserve(&budget, "b", 15);
    let submitted_io = first.clone();
    drop(first);
    assert_eq!(budget.usage.lock().warming, 25);
    drop(submitted_io);
    assert_eq!(budget.usage.lock().warming, 5);
    drop((second, foreground, other_foreground));
    let usage = budget.usage.lock();
    assert_eq!(usage.total, 0);
    assert_eq!(usage.warming, 0);
    assert!(usage.instances.is_empty());
    assert!(usage.warming_instances.is_empty());
}

#[test]
fn foreground_ownership_suppresses_new_warmups_until_the_last_gpu_owner_releases() {
    let budget = QueryBudget::new(100, 100).unwrap();
    let foreground = reserve(&budget, "a", 10);
    for phase in [Phase::Preparing, Phase::Ready, Phase::Restoring] {
        match phase {
            Phase::Ready => foreground.ready(10).unwrap(),
            Phase::Restoring => foreground.restoring(),
            _ => {}
        }
        assert!(matches!(
            budget.reserve("b", "ns", 1, QueryMode::Warmup),
            QueryAdmission::Busy
        ));
    }
    let gpu = foreground.clone();
    drop(foreground);
    assert!(matches!(
        budget.reserve("b", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Busy
    ));
    drop(gpu);
    assert!(matches!(
        budget.reserve("b", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Admitted(_)
    ));
}

#[test]
fn prepared_pages_keep_headroom_until_claim_and_total_bytes_until_gpu_completion() {
    let budget = QueryBudget::new(400, 400).unwrap();
    let QueryAdmission::Admitted(prepared) = budget.reserve("a", "ns", 100, QueryMode::Prepare)
    else {
        panic!("preparation should fit");
    };
    assert_eq!(prepared.batch_blocks(10, 24), 2);
    assert_eq!(prepared.batch_blocks(10, 1), 1);
    prepared.ready(80).unwrap();
    assert_eq!(budget.usage.lock().warming, 80);
    assert!(matches!(
        budget.reserve("b", "ns", 21, QueryMode::Prepare),
        QueryAdmission::Busy
    ));
    let io = prepared.clone();
    prepared.claim();
    prepared.claim();
    assert_eq!(budget.usage.lock().warming, 0);
    assert_eq!(budget.usage.lock().total, 80);
    prepared.restoring();
    drop(prepared);
    assert_eq!(budget.usage.lock().total, 80);
    drop(io);
    assert_eq!(budget.usage.lock().total, 0);
    assert!(budget.usage.lock().warming_instances.is_empty());
}

#[test]
fn owned_lookahead_can_overlap_foreground_only_within_both_budgets() {
    let budget = QueryBudget::new(100, 100).unwrap();
    let foreground = reserve(&budget, "a", 70);
    let QueryAdmission::Admitted(prepared) = budget.reserve("a", "ns", 25, QueryMode::Prepare)
    else {
        panic!("owned lookahead should use available speculative headroom");
    };
    assert!(matches!(
        budget.reserve("b", "ns", 1, QueryMode::Prepare),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "ns", 6, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    let remaining_foreground = reserve(&budget, "b", 5);
    prepared.ready(25).unwrap();
    assert_eq!(budget.usage.lock().total, 100);
    drop((foreground, prepared, remaining_foreground));
    assert_eq!(budget.usage.lock().total, 0);
}
