use super::*;

#[test]
fn discovery_rejects_ambiguous_or_malformed_source_evidence() {
    let key = StateKey::new("ns".into(), vec![1]);
    let replica = wire::ReplicaLocation {
        endpoint: "owner:50055".into(),
        incarnation: "00000000-0000-0000-0000-000000000001".into(),
        sequence: 1,
    };
    let row = wire::BlockCandidates {
        block_hash: vec![1],
        replicas: vec![replica.clone()],
    };
    let converted = row
        .clone()
        .into_candidates(key.clone(), "requester")
        .unwrap();
    assert_eq!(wire::BlockCandidates::from(converted), row);
    assert!(
        row.clone()
            .into_candidates(StateKey::new("ns".into(), vec![2]), "requester")
            .is_err()
    );
    assert!(
        row.clone()
            .into_candidates(key.clone(), "owner:50055")
            .is_err()
    );
    for invalid in [
        wire::ReplicaLocation {
            endpoint: String::new(),
            ..replica.clone()
        },
        wire::ReplicaLocation {
            endpoint: "x".repeat(DISCOVERY_MAX_ENDPOINT_BYTES + 1),
            ..replica.clone()
        },
        wire::ReplicaLocation {
            incarnation: String::new(),
            ..replica.clone()
        },
        wire::ReplicaLocation {
            incarnation: "00000000-0000-0000-0000-000000000000".into(),
            ..replica.clone()
        },
        wire::ReplicaLocation {
            sequence: 0,
            ..replica.clone()
        },
    ] {
        assert!(
            wire::BlockCandidates {
                block_hash: vec![1],
                replicas: vec![invalid]
            }
            .into_candidates(key.clone(), "requester")
            .is_err()
        );
    }
    for count in [2, DISCOVERY_MAX_REPLICAS + 1] {
        assert!(
            wire::BlockCandidates {
                block_hash: vec![1],
                replicas: vec![replica.clone(); count]
            }
            .into_candidates(key.clone(), "requester")
            .is_err()
        );
    }
    assert!(
        orbitkv_state::validate_discovery_query(
            "ns",
            &vec![vec![1]; orbitkv_state::DISCOVERY_MAX_KEYS + 1]
        )
        .is_err()
    );
    assert!(
        orbitkv_state::validate_discovery_query(
            "ns",
            &[vec![1; orbitkv_state::DISCOVERY_MAX_BYTES]]
        )
        .is_err()
    );
}
