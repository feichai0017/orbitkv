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
#[path = "../tests/unit/discovery.rs"]
mod tests;
