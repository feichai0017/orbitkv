use crate::StateKey;
use sha2::{Digest, Sha256};

/// Part of the v1 inventory wire contract; changes require a new placement protocol.
pub const CATALOG_SHARDS: usize = 16;

pub fn catalog_shard(key: &StateKey) -> usize {
    let mut hash = Sha256::new();
    hash.update(b"orbitkv/catalog/key/v1\0");
    hash.update((key.namespace.len() as u64).to_be_bytes());
    hash.update(key.namespace.as_bytes());
    hash.update(&key.hash);
    usize::from(hash.finalize()[0]) % CATALOG_SHARDS
}

pub const INVENTORY_BATCH_BYTES: usize = 512 * 1024;
pub const INVENTORY_BATCH_RECORDS: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryRecord {
    pub key: StateKey,
    pub sequence: u64,
    pub present: bool,
}

impl InventoryRecord {
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.key.namespace.len() + self.key.hash.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InventoryOperation {
    Begin {
        sequence: u64,
    },
    Snapshot {
        page: u64,
        records: Vec<InventoryRecord>,
    },
    Delta {
        after: u64,
        records: Vec<InventoryRecord>,
    },
    Commit {
        sequence: u64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InventoryStatus {
    pub generation: u64,
    pub sequence: u64,
    pub next_page: u64,
    pub ready: bool,
}
