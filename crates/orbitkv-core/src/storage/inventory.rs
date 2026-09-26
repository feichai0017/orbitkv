use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Arc;

use orbitkv_state::{
    CATALOG_SHARDS, INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, InventoryRecord, ReplicaMedium,
    ReplicaMetadata, StateKey, catalog_shard,
};
use parking_lot::Mutex;
use tokio::sync::Notify;

pub const DEFAULT_INVENTORY_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

pub(crate) struct ResidencyInventory {
    state: Mutex<ResidencyState>,
}

struct ResidencyState {
    shards: [Inventory; CATALOG_SHARDS],
    media: HashMap<StateKey, Residences>,
}

#[derive(Default)]
struct Residences {
    dram: Option<ReplicaMetadata>,
    ssd: Option<ReplicaMetadata>,
}

impl ResidencyInventory {
    pub(crate) fn new(byte_limit: usize) -> Self {
        Self {
            state: Mutex::new(ResidencyState {
                shards: std::array::from_fn(|_| Inventory::new(byte_limit / CATALOG_SHARDS)),
                media: HashMap::new(),
            }),
        }
    }

    pub(crate) fn change(
        &self,
        key: &StateKey,
        medium: ReplicaMedium,
        metadata: Option<ReplicaMetadata>,
    ) {
        debug_assert!(
            metadata.is_none_or(|metadata| metadata.medium == medium),
            "residency metadata medium differs from its owner"
        );
        let mut state = self.state.lock();
        let residences = state.media.entry(key.clone()).or_default();
        match medium {
            ReplicaMedium::Dram => residences.dram = metadata,
            ReplicaMedium::Ssd => residences.ssd = metadata,
            ReplicaMedium::Unknown | ReplicaMedium::Hbm => {
                debug_assert!(false, "unsupported local residency transition");
                return;
            }
        }
        let advertised = residences.dram.or(residences.ssd);
        if residences.dram.is_none() && residences.ssd.is_none() {
            state.media.remove(key);
        }
        state.shards[catalog_shard(key)].change(key, advertised);
    }

    pub(crate) fn sequence(&self, shard: usize) -> u64 {
        self.state.lock().shards[shard].sequence()
    }

    pub(crate) fn changed(&self, shard: usize) -> Arc<Notify> {
        self.state.lock().shards[shard].changed()
    }

    pub(crate) fn page(
        &self,
        shard: usize,
        after: Option<&StateKey>,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        self.state.lock().shards[shard].snapshot_page(after)
    }

    pub(crate) fn changes(
        &self,
        shard: usize,
        after: u64,
        through: u64,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        self.state.lock().shards[shard].changes(after, through)
    }

    pub(crate) fn covers(&self, shard: usize, after: u64) -> bool {
        self.state.lock().shards[shard].covers(after)
    }

    pub(crate) fn contains_record(&self, record: &InventoryRecord) -> bool {
        self.state.lock().shards[catalog_shard(&record.key)].contains_record(record)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InventoryReadError {
    HistoryGap,
    RecordTooLarge,
}

pub(super) struct Inventory {
    residents: BTreeMap<StateKey, ResidentEvidence>,
    journal: VecDeque<InventoryRecord>,
    sequence: u64,
    journal_bytes: usize,
    byte_limit: usize,
    changed: Arc<Notify>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentEvidence {
    sequence: u64,
    metadata: ReplicaMetadata,
}

impl Inventory {
    pub(super) fn new(byte_limit: usize) -> Self {
        Self {
            residents: BTreeMap::new(),
            journal: VecDeque::new(),
            sequence: 0,
            journal_bytes: 0,
            byte_limit,
            changed: Arc::new(Notify::new()),
        }
    }

    pub(super) fn change(&mut self, key: &StateKey, metadata: Option<ReplicaMetadata>) {
        let present = metadata.is_some();
        if self.residents.get(key).map(|resident| resident.metadata) == metadata {
            return;
        }
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("inventory sequence exhausted");
        if present {
            self.residents.insert(
                key.clone(),
                ResidentEvidence {
                    sequence: self.sequence,
                    metadata: metadata.expect("present residency has metadata"),
                },
            );
        } else {
            self.residents.remove(key);
        }
        let record = InventoryRecord {
            key: key.clone(),
            sequence: self.sequence,
            present,
            metadata,
        };
        self.journal_bytes += record.estimated_size();
        self.journal.push_back(record);
        while self.journal_bytes > self.byte_limit {
            if let Some(record) = self.journal.pop_front() {
                self.journal_bytes -= record.estimated_size();
            }
        }
        self.changed.notify_one();
    }

    pub(super) fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) fn changed(&self) -> Arc<Notify> {
        self.changed.clone()
    }

    pub(super) fn contains_record(&self, record: &InventoryRecord) -> bool {
        record.present
            && self
                .residents
                .get(&record.key)
                .is_some_and(|resident| resident.sequence == record.sequence)
    }

    pub(super) fn covers(&self, after: u64) -> bool {
        after <= self.sequence
            && (after == self.sequence
                || self
                    .journal
                    .front()
                    .is_some_and(|record| record.sequence <= after + 1))
    }

    pub(super) fn snapshot_page(
        &self,
        after: Option<&StateKey>,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        let bounds = (after.map_or(Unbounded, Excluded), Unbounded);
        bounded_records(
            self.residents
                .range::<StateKey, _>(bounds)
                .map(|(key, resident)| InventoryRecord {
                    key: key.clone(),
                    sequence: resident.sequence,
                    present: true,
                    metadata: Some(resident.metadata),
                }),
        )
    }

    pub(super) fn changes(
        &self,
        after: u64,
        through: u64,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        if after > through || through > self.sequence || !self.covers(after) {
            return Err(InventoryReadError::HistoryGap);
        }
        if after == through {
            return Ok(Vec::new());
        }
        let first = self
            .journal
            .front()
            .ok_or(InventoryReadError::HistoryGap)?
            .sequence;
        let start = (after + 1 - first) as usize;
        let end = (through - first + 1) as usize;
        bounded_records(self.journal.range(start..end).cloned())
    }
}

fn bounded_records(
    records: impl Iterator<Item = InventoryRecord>,
) -> Result<Vec<InventoryRecord>, InventoryReadError> {
    let mut batch = Vec::new();
    let mut bytes = 0;
    for record in records {
        let size = record.estimated_size();
        if size > INVENTORY_BATCH_BYTES {
            return Err(InventoryReadError::RecordTooLarge);
        }
        if bytes + size > INVENTORY_BATCH_BYTES || batch.len() == INVENTORY_BATCH_RECORDS {
            break;
        }
        bytes += size;
        batch.push(record);
    }
    Ok(batch)
}

#[cfg(test)]
#[path = "../../tests/unit/storage/dram/inventory.rs"]
mod tests;
