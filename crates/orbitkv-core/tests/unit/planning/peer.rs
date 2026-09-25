use super::*;
use orbitkv_state::{ReplicaLocation, StateKey};

fn row(hash: u8, owners: &[&str]) -> BlockCandidates {
    BlockCandidates {
        key: StateKey::new("ns".into(), vec![hash]),
        replicas: owners
            .iter()
            .map(|owner| ReplicaLocation {
                owner: CacheOwner {
                    endpoint: (*owner).into(),
                    incarnation: uuid::Uuid::from_u128(1),
                },
                sequence: u64::from(hash),
            })
            .collect(),
    }
}

#[test]
fn planner_selects_longest_cover_then_stable_owner_and_stops_at_gap() {
    let rows = vec![
        row(1, &["c", "b", "a"]),
        row(2, &["c", "b"]),
        row(3, &["d"]),
        row(4, &[]),
        row(5, &["a"]),
    ];
    let plan = FetchPlan::new(rows).unwrap();
    assert_eq!(plan.block_count(), 3);
    let first = plan.next_segment(0).unwrap();
    assert_eq!(first.owner.endpoint, "b");
    assert_eq!(
        first.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(plan.next_segment(2).unwrap().owner.endpoint, "d");
    let mut rows = vec![row(1, &["a"]); DISCOVERY_MAX_KEYS + 1];
    assert_eq!(
        FetchPlan::new(rows.clone())
            .unwrap()
            .next_segment(0)
            .unwrap()
            .records
            .len(),
        DISCOVERY_MAX_KEYS
    );
    rows[0].key.hash = vec![1; DISCOVERY_MAX_BYTES - 2];
    assert_eq!(
        FetchPlan::new(rows)
            .unwrap()
            .next_segment(0)
            .unwrap()
            .records
            .len(),
        1
    );
}
