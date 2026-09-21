use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Arc;

use orbitkv_state::{INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, InventoryRecord, StateKey};
use tokio::sync::Notify;

pub const DEFAULT_INVENTORY_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InventoryReadError {
    HistoryGap,
    RecordTooLarge,
}

pub(super) struct Inventory {
    residents: BTreeMap<StateKey, u64>,
    journal: VecDeque<InventoryRecord>,
    sequence: u64,
    journal_bytes: usize,
    byte_limit: usize,
    changed: Arc<Notify>,
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

    pub(super) fn change(&mut self, key: &StateKey, present: bool) {
        if self.residents.contains_key(key) == present {
            return;
        }
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("inventory sequence exhausted");
        if present {
            self.residents.insert(key.clone(), self.sequence);
        } else {
            self.residents.remove(key);
        }
        let record = InventoryRecord {
            key: key.clone(),
            sequence: self.sequence,
            present,
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
        record.present && self.residents.get(&record.key) == Some(&record.sequence)
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
                .map(|(key, sequence)| InventoryRecord {
                    key: key.clone(),
                    sequence: *sequence,
                    present: true,
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
#[path = "../../tests/unit/storage/inventory.rs"]
mod tests;
