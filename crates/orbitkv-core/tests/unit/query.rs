use super::*;

fn reserve(budget: &Arc<QueryBudget>, instance: &str, bytes: u64) -> QueryReservation {
    match budget.reserve(instance, "state", bytes, false) {
        QueryAdmission::Admitted(reservation) => reservation,
        _ => panic!("expected admission"),
    }
}

#[test]
fn bytes_remain_charged_until_all_consumers_finish() {
    let budget = QueryBudget::new(100, 80).unwrap();
    let first = reserve(&budget, "a", 70);
    assert!(matches!(
        budget.reserve("a", "state", 11, false),
        QueryAdmission::Busy
    ));
    let second = reserve(&budget, "b", 30);
    assert!(matches!(
        budget.reserve("c", "state", 1, false),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "state", 81, false),
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
    let QueryAdmission::Admitted(first) = budget.reserve("a", "ns", 20, true) else {
        panic!("warmup should fit");
    };
    assert!(
        first.ready(20).is_err(),
        "warmup cannot become a restore lease"
    );
    assert!(matches!(
        budget.reserve("a", "ns", 1, true),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "ns", 21, true),
        QueryAdmission::TooLarge
    ));
    let QueryAdmission::Admitted(second) = budget.reserve("b", "ns", 5, true) else {
        panic!("global warmup budget should fit");
    };
    assert!(matches!(
        budget.reserve("c", "ns", 1, true),
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
