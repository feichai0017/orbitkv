use super::*;
use orbitkv_state::CacheOwner;

fn row(hash: u8, sequence: u64) -> BlockCandidates {
    BlockCandidates {
        key: StateKey::new("ns".into(), vec![hash]),
        replicas: vec![ReplicaLocation {
            owner: CacheOwner {
                endpoint: "owner:50055".into(),
                incarnation: uuid::Uuid::from_u128(1),
            },
            sequence,
        }],
    }
}

#[test]
fn bounded_lru_expiry_and_version_specific_rejection() {
    let now = Instant::now();
    let first = row(1, 1);
    let mut index = CandidateIndex::new(4096);
    index.insert(first.clone(), now);
    let one_entry = index.bytes;
    index.capacity = one_entry * 2;
    index.insert(row(2, 2), now);
    assert!(index.get(&first.key, now).is_some());
    index.insert(row(3, 3), now);
    assert!(index.get(&row(2, 2).key, now).is_none());
    let replacement = row(1, 4);
    index.insert(replacement.clone(), now);
    index.reject(&first.key, &first.replicas[0]);
    assert_eq!(index.get(&first.key, now), Some(replacement.clone()));
    index.reject(&first.key, &replacement.replicas[0]);
    assert!(index.get(&first.key, now).is_none());
    assert!(index.get(&row(3, 3).key, now + CANDIDATE_TTL).is_none());
    assert_eq!(index.bytes, 0);
    let mut negative = row(1, 1);
    negative.replicas.clear();
    index.insert(negative, now);
    index.capacity = 1;
    index.insert(first, now);
    assert_eq!(index.bytes, 0);
}
