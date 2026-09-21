use orbitkv_state::CATALOG_SHARDS;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Immutable v1 placement. Liveness resolves endpoints but never changes owners.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    version: u32,
    nodes: Vec<String>,
}

impl Placement {
    pub fn new(mut nodes: Vec<String>) -> Result<Self, String> {
        nodes.sort_unstable();
        if nodes.is_empty()
            || nodes.len() > CATALOG_SHARDS
            || nodes.windows(2).any(|pair| pair[0] == pair[1])
            || nodes.iter().any(|node| {
                node.is_empty()
                    || node.len() > 128
                    || !node
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            })
        {
            return Err(format!(
                "catalog placement requires 1..={CATALOG_SHARDS} distinct valid node IDs"
            ));
        }
        Ok(Self { version: 1, nodes })
    }

    pub fn id(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"orbitkv/catalog/placement/v1\0");
        hash.update(self.version.to_be_bytes());
        for node in &self.nodes {
            hash.update((node.len() as u64).to_be_bytes());
            hash.update(node.as_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    pub fn host(&self, shard: usize) -> Option<&str> {
        if shard >= CATALOG_SHARDS {
            return None;
        }
        self.nodes
            .iter()
            .max_by_key(|node| {
                let mut hash = Sha256::new();
                hash.update(b"orbitkv/catalog/host/v1\0");
                hash.update((shard as u32).to_be_bytes());
                hash.update(node.as_bytes());
                (hash.finalize(), node.as_str())
            })
            .map(String::as_str)
    }
}

#[cfg(test)]
#[path = "../tests/unit/placement.rs"]
mod tests;
