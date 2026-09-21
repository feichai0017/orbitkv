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
mod tests {
    use super::*;

    fn key(n: u32) -> StateKey {
        StateKey::new("model".into(), n.to_be_bytes().to_vec())
    }

    #[test]
    fn snapshot_with_concurrent_mutations_converges_at_the_cut() {
        let mut inventory = Inventory::new(DEFAULT_INVENTORY_JOURNAL_BYTES);
        for n in 0..1500 {
            inventory.change(&key(n), true);
        }
        let start = inventory.sequence();
        let mut snapshot = inventory.snapshot_page(None).unwrap();
        let cursor = snapshot.last().unwrap().key.clone();
        inventory.change(&key(0), false);
        inventory.change(&key(1400), false);
        inventory.change(&key(1400), true);
        let behind_cursor = StateKey::new("a-model".into(), vec![1]);
        inventory.change(&behind_cursor, true);
        snapshot.extend(inventory.snapshot_page(Some(&cursor)).unwrap());
        let end = inventory.sequence();
        let mut reconstructed: BTreeMap<_, _> =
            snapshot.into_iter().map(|r| (r.key, r.sequence)).collect();
        for update in inventory.changes(start, end).unwrap() {
            if reconstructed
                .get(&update.key)
                .is_some_and(|s| *s > update.sequence)
            {
                continue;
            }
            if update.present {
                reconstructed.insert(update.key, update.sequence);
            } else {
                reconstructed.remove(&update.key);
            }
        }
        assert_eq!(reconstructed, inventory.residents);
        assert_eq!(inventory.sequence(), end);
    }

    #[test]
    fn bounded_history_exposes_gaps_and_duplicates_do_not_advance_sequence() {
        let mut inventory = Inventory::new(300);
        inventory.change(&key(0), true);
        inventory.change(&key(0), true);
        assert_eq!(inventory.sequence(), 1);
        for n in 1..30 {
            inventory.change(&key(n), true);
        }
        assert!(inventory.journal_bytes <= 300);
        assert!(inventory.changes(0, 30).is_err());
        assert_eq!(inventory.changes(29, 30).unwrap()[0].key, key(29));
        assert_eq!(inventory.snapshot_page(None).unwrap().len(), 30);
        assert!(inventory.changes(30, 30).unwrap().is_empty());
        assert!(inventory.changes(31, 30).is_err());
    }
    #[test]
    fn oversized_records_cannot_escape_batch_bounds_or_be_silently_skipped() {
        let mut inventory = Inventory::new(DEFAULT_INVENTORY_JOURNAL_BYTES);
        inventory.change(
            &StateKey::new("ns".into(), vec![1; INVENTORY_BATCH_BYTES]),
            true,
        );
        assert_eq!(
            inventory.snapshot_page(None),
            Err(InventoryReadError::RecordTooLarge)
        );
        assert_eq!(
            inventory.changes(0, 1),
            Err(InventoryReadError::RecordTooLarge)
        );
    }
}
