use super::*;
use orbitkv_state::{ReplicaLocation, StateKey};

fn row(hash: u8, owners: &[&str]) -> ReplicaSet {
    let mut row = ReplicaSet::new(StateKey::new("ns".into(), vec![hash]));
    row.set_peer_dram(
        owners
            .iter()
            .map(|owner| ReplicaLocation {
                owner: CacheOwner {
                    endpoint: (*owner).into(),
                    incarnation: uuid::Uuid::from_u128(1),
                },
                sequence: u64::from(hash),
            })
            .collect(),
    );
    row
}

#[test]
fn planner_selects_longest_cover_then_stable_owner_and_stops_at_gap() {
    let mut rows = vec![
        row(1, &["c", "b", "a"]),
        row(2, &["c", "b"]),
        row(3, &["d"]),
        row(4, &[]),
        row(5, &["a"]),
    ];
    let plan = FetchPlan::new(&mut rows, 1).unwrap();
    assert_eq!(plan.block_count(), 3);
    let first = plan.next_segment(0).unwrap();
    assert_eq!(first.owner.endpoint, "b");
    assert_eq!(
        first.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(plan.next_segment(2).unwrap().owner.endpoint, "d");
    let mut rows = (0..=DISCOVERY_MAX_KEYS)
        .map(|_| row(1, &["a"]))
        .collect::<Vec<_>>();
    assert_eq!(
        FetchPlan::new(&mut rows, 1)
            .unwrap()
            .next_segment(0)
            .unwrap()
            .records
            .len(),
        DISCOVERY_MAX_KEYS
    );
    rows[0].key.hash = vec![1; DISCOVERY_MAX_BYTES - 2];
    assert_eq!(
        FetchPlan::new(&mut rows, 1)
            .unwrap()
            .next_segment(0)
            .unwrap()
            .records
            .len(),
        1
    );

    rows[0].key.hash = vec![1; DISCOVERY_MAX_BYTES];
    assert!(
        FetchPlan::new(&mut rows, 1)
            .unwrap()
            .next_segment(0)
            .is_none()
    );
}

#[test]
fn selected_source_records_keep_its_versions_and_rejection_preserves_alternatives() {
    let mut rows = vec![row(1, &["a", "b"]), row(2, &["a", "b"])];
    rows[0].set_peer_dram(vec![
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "a".into(),
                incarnation: uuid::Uuid::from_u128(1),
            },
            sequence: 11,
        },
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "b".into(),
                incarnation: uuid::Uuid::from_u128(2),
            },
            sequence: 21,
        },
    ]);
    rows[1].set_peer_dram(vec![
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "a".into(),
                incarnation: uuid::Uuid::from_u128(1),
            },
            sequence: 12,
        },
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "b".into(),
                incarnation: uuid::Uuid::from_u128(2),
            },
            sequence: 22,
        },
    ]);
    let mut plan = FetchPlan::new(&mut rows, 2).unwrap();
    let first = plan.next_segment(0).unwrap();
    assert_eq!(
        first.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [11, 12]
    );
    plan.reject(0, &first);
    let next = plan.next_segment(0).unwrap();
    assert_eq!(next.owner.endpoint, "b");
    assert_eq!(next.owner.incarnation, uuid::Uuid::from_u128(2));
    assert_eq!(
        next.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [21, 22]
    );
}
