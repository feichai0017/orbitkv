use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_ENDPOINT_BYTES, DISCOVERY_MAX_REPLICAS,
    ReplicaLocation, StateKey,
};

use crate::proto::engine as wire;

impl From<BlockCandidates> for wire::BlockCandidates {
    fn from(value: BlockCandidates) -> Self {
        Self {
            block_hash: value.key.hash,
            replicas: value
                .replicas
                .into_iter()
                .map(|r| wire::ReplicaLocation {
                    endpoint: r.owner.endpoint,
                    incarnation: r.owner.incarnation.to_string(),
                    sequence: r.sequence,
                })
                .collect(),
        }
    }
}

impl wire::BlockCandidates {
    pub fn into_candidates(
        self,
        key: StateKey,
        exclude: &str,
    ) -> Result<BlockCandidates, &'static str> {
        if self.block_hash != key.hash || self.replicas.len() > DISCOVERY_MAX_REPLICAS {
            return Err("invalid discovery row or replica count");
        }
        let mut replicas: Vec<ReplicaLocation> = Vec::with_capacity(self.replicas.len());
        for replica in self.replicas {
            if replica.endpoint.is_empty()
                || replica.endpoint == exclude
                || replica.endpoint.len() > DISCOVERY_MAX_ENDPOINT_BYTES
                || replica.sequence == 0
            {
                return Err("invalid discovery replica");
            }
            let owner = CacheOwner {
                endpoint: replica.endpoint,
                incarnation: replica
                    .incarnation
                    .parse()
                    .map_err(|_| "invalid owner incarnation")?,
            };
            if owner.incarnation.is_nil() || replicas.iter().any(|r| r.owner == owner) {
                return Err("duplicate or invalid discovery owner");
            }
            replicas.push(ReplicaLocation {
                owner,
                sequence: replica.sequence,
            });
        }
        Ok(BlockCandidates { key, replicas })
    }
}

#[cfg(test)]
mod tests {
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
}
