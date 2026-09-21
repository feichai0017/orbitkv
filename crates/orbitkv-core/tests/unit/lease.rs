use super::*;

#[test]
fn consume_rejects_wrong_instance_without_removing_lease() {
    let manager = QueryLeaseManager::default();
    let lease_id = QueryLeaseId::fresh();
    manager
        .inner
        .leases
        .lock()
        .expect("query leases lock poisoned")
        .insert(
            lease_id,
            QueryLease {
                instance_id: "inst-a".to_string(),
                blocks: Vec::new(),
                remaining_consumers: 1,
                expires_at: Instant::now() + DEFAULT_LEASE_TTL,
                ownership: None,
            },
        );

    let err = match manager.consume("inst-b", &lease_id) {
        Ok(_) => panic!("wrong instance consumed lease"),
        Err(err) => err,
    };
    assert!(err.contains("belongs to instance inst-a"));

    manager
        .consume("inst-a", &lease_id)
        .expect("original instance can still consume lease");
}

#[test]
fn consume_allows_configured_number_of_consumers() {
    let manager = QueryLeaseManager::default();
    let blocks = vec![Arc::new(SealedBlock::from_slots(Vec::new()))];
    let lease_id = manager.create("inst-a", blocks, 2, None);

    assert_eq!(manager.consume("inst-a", &lease_id).unwrap().0.len(), 1);
    assert_eq!(manager.consume("inst-a", &lease_id).unwrap().0.len(), 1);

    let err = manager
        .consume("inst-a", &lease_id)
        .err()
        .expect("lease should be exhausted");
    assert!(err.contains("query lease is unknown or expired"));
}

#[test]
fn disconnect_releases_ready_interest_but_not_a_gpu_consumers_reservation() {
    let manager = QueryLeaseManager::default();
    let budget = crate::query::QueryBudget::new(100, 100).unwrap();
    let crate::QueryAdmission::Admitted(reservation) = budget.reserve("a", "ns", 100, false) else {
        panic!("budget available");
    };
    reservation.ready(100).unwrap();
    let owner = QueryOwner {
        session: 7,
        operation: 1,
        revision: 2,
    };
    let id = manager.create(
        "a",
        vec![Arc::new(SealedBlock::from_slots(Vec::new()))],
        2,
        Some((owner, reservation)),
    );
    let (_, gpu) = manager.consume("a", &id).unwrap();
    gpu.as_ref().unwrap().restoring();
    manager.release_owner(|candidate| candidate.session == 7);
    assert!(matches!(
        budget.reserve("a", "ns", 1, false),
        crate::QueryAdmission::Busy
    ));
    assert!(manager.consume("a", &id).is_err());
    drop(gpu);
    assert!(matches!(
        budget.reserve("a", "ns", 100, false),
        crate::QueryAdmission::Admitted(_)
    ));
}
