use orbitkv_state::{
    InventoryOperation, InventoryRecord, InventoryStatus, ReplicaMedium, ReplicaMetadata,
    ReplicaRepresentation, StateKey,
};

use crate::proto::engine::{self as wire, sync_inventory_request::Operation};

impl From<InventoryRecord> for wire::InventoryRecord {
    fn from(record: InventoryRecord) -> Self {
        Self {
            namespace: record.key.namespace,
            block_hash: record.key.hash,
            sequence: record.sequence,
            present: record.present,
            metadata: record.metadata.map(metadata_to_wire),
        }
    }
}

impl From<wire::InventoryRecord> for InventoryRecord {
    fn from(record: wire::InventoryRecord) -> Self {
        Self {
            key: StateKey::new(record.namespace, record.block_hash),
            sequence: record.sequence,
            present: record.present,
            metadata: record.metadata.map(metadata_from_wire),
        }
    }
}

pub(crate) fn metadata_to_wire(metadata: ReplicaMetadata) -> wire::ReplicaMetadata {
    wire::ReplicaMetadata {
        medium: match metadata.medium {
            ReplicaMedium::Unknown => wire::ReplicaMedium::Unknown,
            ReplicaMedium::Dram => wire::ReplicaMedium::Dram,
            ReplicaMedium::Ssd => wire::ReplicaMedium::Ssd,
            ReplicaMedium::Hbm => wire::ReplicaMedium::Hbm,
        } as i32,
        representation: match metadata.representation {
            ReplicaRepresentation::Unknown => wire::ReplicaRepresentation::Unknown,
            ReplicaRepresentation::Raw => wire::ReplicaRepresentation::Raw,
            ReplicaRepresentation::Ans => wire::ReplicaRepresentation::Ans,
            ReplicaRepresentation::Fp8 => wire::ReplicaRepresentation::Fp8,
            ReplicaRepresentation::TurboQuant => wire::ReplicaRepresentation::TurboQuant,
            ReplicaRepresentation::Mixed => wire::ReplicaRepresentation::Mixed,
        } as i32,
        stored_bytes: metadata.stored_bytes,
    }
}

pub(crate) fn metadata_from_wire(metadata: wire::ReplicaMetadata) -> ReplicaMetadata {
    let medium = match wire::ReplicaMedium::try_from(metadata.medium) {
        Ok(wire::ReplicaMedium::Dram) => ReplicaMedium::Dram,
        Ok(wire::ReplicaMedium::Ssd) => ReplicaMedium::Ssd,
        Ok(wire::ReplicaMedium::Hbm) => ReplicaMedium::Hbm,
        Ok(wire::ReplicaMedium::Unknown) | Err(_) => ReplicaMedium::Unknown,
    };
    let representation = match wire::ReplicaRepresentation::try_from(metadata.representation) {
        Ok(wire::ReplicaRepresentation::Raw) => ReplicaRepresentation::Raw,
        Ok(wire::ReplicaRepresentation::Ans) => ReplicaRepresentation::Ans,
        Ok(wire::ReplicaRepresentation::Fp8) => ReplicaRepresentation::Fp8,
        Ok(wire::ReplicaRepresentation::TurboQuant) => ReplicaRepresentation::TurboQuant,
        Ok(wire::ReplicaRepresentation::Mixed) => ReplicaRepresentation::Mixed,
        Ok(wire::ReplicaRepresentation::Unknown) | Err(_) => ReplicaRepresentation::Unknown,
    };
    ReplicaMetadata {
        medium,
        representation,
        stored_bytes: metadata.stored_bytes,
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

#[cfg(test)]
#[path = "../tests/unit/inventory.rs"]
mod tests;
