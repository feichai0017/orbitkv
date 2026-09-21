use super::*;

fn reserve(budget: &Arc<QueryBudget>, instance: &str, bytes: u64) -> QueryReservation {
    match budget.reserve(instance, "state", bytes) {
        QueryAdmission::Admitted(reservation) => reservation,
        _ => panic!("expected admission"),
    }
}

#[test]
fn bytes_remain_charged_until_all_consumers_finish() {
    let budget = QueryBudget::new(100, 80).unwrap();
    let first = reserve(&budget, "a", 70);
    assert!(matches!(
        budget.reserve("a", "state", 11),
        QueryAdmission::Busy
    ));
    let second = reserve(&budget, "b", 30);
    assert!(matches!(
        budget.reserve("c", "state", 1),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "state", 81),
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
