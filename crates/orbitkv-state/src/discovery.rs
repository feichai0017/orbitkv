use crate::StateKey;

/// Discovery is bounded, positive evidence, never a complete cluster inventory.
pub const DISCOVERY_MAX_KEYS: usize = 128;
pub const DISCOVERY_MAX_BYTES: usize = 64 * 1024;
pub const DISCOVERY_MAX_REPLICAS: usize = 4;
pub const DISCOVERY_MAX_ENDPOINT_BYTES: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheOwner {
    pub endpoint: String,
    pub incarnation: uuid::Uuid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaLocation {
    pub owner: CacheOwner,
    /// Sequence of the insertion in this owner's current runtime inventory.
    pub sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockCandidates {
    pub key: StateKey,
    pub replicas: Vec<ReplicaLocation>,
}

impl BlockCandidates {
    /// Logical retained bytes, including owned capacities but not allocator overhead.
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.key.namespace.capacity()
            + self.key.hash.capacity()
            + self.replicas.capacity() * std::mem::size_of::<ReplicaLocation>()
            + self
                .replicas
                .iter()
                .map(|r| r.owner.endpoint.capacity())
                .sum::<usize>()
    }
}

pub fn validate_discovery_query(namespace: &str, hashes: &[Vec<u8>]) -> Result<(), &'static str> {
    if namespace.is_empty() || hashes.is_empty() || hashes.len() > DISCOVERY_MAX_KEYS {
        return Err("invalid discovery key count or namespace");
    }
    let mut bytes = namespace.len();
    for hash in hashes {
        if hash.is_empty() {
            return Err("empty discovery hash");
        }
        bytes = bytes
            .checked_add(hash.len())
            .ok_or("discovery size overflow")?;
    }
    if bytes > DISCOVERY_MAX_BYTES {
        return Err("discovery query byte limit exceeded");
    }
    Ok(())
}
