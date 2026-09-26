use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_ENDPOINT_BYTES, DISCOVERY_MAX_REPLICAS,
    ReplicaLocation, ReplicaMedium, StateKey,
};

use crate::inventory::{metadata_from_wire, metadata_to_wire};
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
                    metadata: Some(metadata_to_wire(r.metadata)),
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
            let metadata = replica
                .metadata
                .map(metadata_from_wire)
                .ok_or("missing discovery replica metadata")?;
            if metadata.medium == ReplicaMedium::Unknown {
                return Err("invalid discovery replica medium");
            }
            replicas.push(ReplicaLocation {
                owner,
                sequence: replica.sequence,
                metadata,
            });
        }
        Ok(BlockCandidates { key, replicas })
    }
}

#[cfg(test)]
#[path = "../tests/unit/discovery.rs"]
mod tests;
