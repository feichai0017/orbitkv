use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Arc;
use std::time::Duration;

use orbitkv_state::{
    INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, InventoryRecord, ReplicaMedium,
    ReplicaMetadata, StateKey,
};
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};

pub const DEFAULT_INVENTORY_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct PublishedInventory {
    pub sequence: u64,
    pub revision: i64,
    pub ready: bool,
}

pub struct ResidencyInventory {
    state: Mutex<Inventory>,
    changed: Arc<Notify>,
    published: watch::Sender<PublishedInventory>,
}

struct Inventory {
    residents: BTreeMap<(StateKey, ReplicaMedium), InventoryRecord>,
    journal: VecDeque<InventoryRecord>,
    sequence: u64,
    journal_bytes: usize,
    byte_limit: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryReadError {
    HistoryGap,
    RecordTooLarge,
}

impl ResidencyInventory {
    pub fn new(byte_limit: usize) -> Self {
        Self {
            state: Mutex::new(Inventory {
                residents: BTreeMap::new(),
                journal: VecDeque::new(),
                sequence: 0,
                journal_bytes: 0,
                byte_limit,
            }),
            changed: Arc::new(Notify::new()),
            published: watch::channel(PublishedInventory::default()).0,
        }
    }

    pub(crate) fn change(
        &self,
        key: &StateKey,
        medium: ReplicaMedium,
        metadata: Option<ReplicaMetadata>,
    ) {
        assert!(matches!(medium, ReplicaMedium::Dram | ReplicaMedium::Ssd));
        assert!(metadata.is_none_or(|value| value.medium == medium));
        let mut state = self.state.lock();
        let identity = (key.clone(), medium);
        let old = state
            .residents
            .get(&identity)
            .and_then(|record| record.metadata);
        if old == metadata {
            return;
        }
        state.sequence = state
            .sequence
            .checked_add(1)
            .expect("inventory sequence exhausted");
        let record = InventoryRecord {
            key: key.clone(),
            sequence: state.sequence,
            present: metadata.is_some(),
            metadata: metadata.or(old),
        };
        if record.present {
            state.residents.insert(identity, record.clone());
        } else {
            state.residents.remove(&identity);
        }
        state.journal_bytes += record.estimated_size();
        state.journal.push_back(record);
        while state.journal_bytes > state.byte_limit {
            if let Some(record) = state.journal.pop_front() {
                state.journal_bytes -= record.estimated_size();
            }
        }
        self.changed.notify_one();
    }

    pub fn sequence(&self) -> u64 {
        self.state.lock().sequence
    }
    pub fn changed(&self) -> Arc<Notify> {
        self.changed.clone()
    }

    pub fn page(
        &self,
        after: Option<&(StateKey, ReplicaMedium)>,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        let state = self.state.lock();
        bounded_records(
            state
                .residents
                .range((after.map_or(Unbounded, Excluded), Unbounded))
                .map(|(_, record)| record.clone()),
        )
    }

    pub fn changes(
        &self,
        after: u64,
        through: u64,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        let state = self.state.lock();
        if after > through
            || through > state.sequence
            || (after != state.sequence
                && state
                    .journal
                    .front()
                    .is_none_or(|record| record.sequence > after + 1))
        {
            return Err(InventoryReadError::HistoryGap);
        }
        if after == through {
            return Ok(Vec::new());
        }
        let first = state
            .journal
            .front()
            .ok_or(InventoryReadError::HistoryGap)?
            .sequence;
        let start =
            usize::try_from(after + 1 - first).map_err(|_| InventoryReadError::HistoryGap)?;
        let end =
            usize::try_from(through - first + 1).map_err(|_| InventoryReadError::HistoryGap)?;
        bounded_records(state.journal.range(start..end).cloned())
    }

    pub(crate) fn contains_record(&self, record: &InventoryRecord, medium: ReplicaMedium) -> bool {
        record.present
            && self
                .state
                .lock()
                .residents
                .get(&(record.key.clone(), medium))
                .is_some_and(|resident| resident.sequence == record.sequence)
    }

    pub fn published(&self) -> PublishedInventory {
        *self.published.borrow()
    }

    pub fn acknowledge(&self, progress: PublishedInventory) {
        self.published.send_replace(progress);
    }

    pub async fn flush(&self) -> Result<i64, String> {
        let target = self.sequence();
        let mut progress = self.published.subscribe();
        self.changed.notify_one();
        tokio::time::timeout(Duration::from_secs(30), async {
            let ack = progress
                .wait_for(|ack| ack.ready && ack.sequence >= target)
                .await
                .map_err(|_| "inventory publisher stopped".to_string())?;
            Ok(ack.revision)
        })
        .await
        .map_err(|_| "inventory publication timed out".to_string())?
    }
}

fn bounded_records(
    records: impl Iterator<Item = InventoryRecord>,
) -> Result<Vec<InventoryRecord>, InventoryReadError> {
    let mut result = Vec::new();
    let mut bytes = 0;
    for record in records {
        let size = record.estimated_size();
        if size > INVENTORY_BATCH_BYTES {
            return Err(InventoryReadError::RecordTooLarge);
        }
        if result.len() == INVENTORY_BATCH_RECORDS || bytes + size > INVENTORY_BATCH_BYTES {
            break;
        }
        bytes += size;
        result.push(record);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../../tests/unit/storage/inventory.rs"]
mod tests;
