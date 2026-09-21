use orbitkv_state::{InventoryOperation, InventoryRecord, InventoryStatus, StateKey};

use crate::proto::engine::{self as wire, sync_inventory_request::Operation};

impl From<InventoryRecord> for wire::InventoryRecord {
    fn from(record: InventoryRecord) -> Self {
        Self {
            namespace: record.key.namespace,
            block_hash: record.key.hash,
            sequence: record.sequence,
            present: record.present,
        }
    }
}

impl From<wire::InventoryRecord> for InventoryRecord {
    fn from(record: wire::InventoryRecord) -> Self {
        Self {
            key: StateKey::new(record.namespace, record.block_hash),
            sequence: record.sequence,
            present: record.present,
        }
    }
}

impl From<InventoryStatus> for wire::InventoryProgress {
    fn from(status: InventoryStatus) -> Self {
        Self {
            generation: status.generation,
            sequence: status.sequence,
            next_page: status.next_page,
            ready: status.ready,
        }
    }
}

impl From<wire::InventoryProgress> for InventoryStatus {
    fn from(status: wire::InventoryProgress) -> Self {
        Self {
            generation: status.generation,
            sequence: status.sequence,
            next_page: status.next_page,
            ready: status.ready,
        }
    }
}

impl From<InventoryOperation> for Operation {
    fn from(operation: InventoryOperation) -> Self {
        match operation {
            InventoryOperation::Begin { sequence } => {
                Self::Begin(wire::InventoryBegin { sequence })
            }
            InventoryOperation::Snapshot { page, records } => Self::Page(wire::InventoryPage {
                page,
                records: records.into_iter().map(Into::into).collect(),
            }),
            InventoryOperation::Delta { after, records } => Self::Delta(wire::InventoryDelta {
                after_sequence: after,
                records: records.into_iter().map(Into::into).collect(),
            }),
            InventoryOperation::Commit { sequence } => {
                Self::Commit(wire::InventoryCommit { sequence })
            }
        }
    }
}

impl From<Operation> for InventoryOperation {
    fn from(operation: Operation) -> Self {
        match operation {
            Operation::Begin(begin) => Self::Begin {
                sequence: begin.sequence,
            },
            Operation::Page(page) => Self::Snapshot {
                page: page.page,
                records: page.records.into_iter().map(Into::into).collect(),
            },
            Operation::Delta(delta) => Self::Delta {
                after: delta.after_sequence,
                records: delta.records.into_iter().map(Into::into).collect(),
            },
            Operation::Commit(commit) => Self::Commit {
                sequence: commit.sequence,
            },
        }
    }
}
