use crate::StateKey;
use uuid::Uuid;

pub const INVENTORY_BATCH_BYTES: usize = 512 * 1024;
pub const INVENTORY_BATCH_RECORDS: usize = 1024;
pub const INVENTORY_STREAM_PROTOCOL: &str = "orbitkv/inventory-stream/v3";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryFence {
    pub protocol: String,
    pub cluster_uuid: Uuid,
    pub source_node_epoch: u64,
    pub source_incarnation: Uuid,
    pub inventory_sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryRecord {
    pub key: StateKey,
    pub sequence: u64,
    pub present: bool,
    /// Required for residency advertisements; authorization-only records may
    /// omit it because the authoritative owner checks its live inventory.
    pub metadata: Option<crate::ReplicaMetadata>,
}

impl InventoryRecord {
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.key.namespace.len() + self.key.hash.len()
    }
}
