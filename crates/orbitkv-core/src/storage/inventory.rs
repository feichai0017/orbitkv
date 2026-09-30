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
    flush_requested: Arc<Notify>,
    publish_coalesce_window: Duration,
    published: watch::Sender<PublishedInventory>,
}

struct Inventory {
    residents: BTreeMap<(StateKey, ReplicaMedium), InventoryRecord>,
    journal: VecDeque<InventoryRecord>,
    sequence: u64,
    journal_bytes: usize,
    journal_bytes_peak: usize,
    byte_limit: usize,
    history_gaps: u64,
    flush_through_sequence: u64,
    delta_input_records: u64,
    delta_input_bytes: u64,
    delta_output_records: u64,
    delta_transactions: u64,
    delta_encoded_bytes: u64,
    coalescing_windows: u64,
    coalescing_wait_micros: u64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct InventoryStatus {
    pub sequence: u64,
    pub resident_records: usize,
    pub journal_records: usize,
    pub journal_bytes: usize,
    pub journal_bytes_peak: usize,
    pub journal_capacity_bytes: usize,
    pub history_gaps: u64,
    pub flush_through_sequence: u64,
    pub delta_input_records: u64,
    pub delta_input_bytes: u64,
    pub delta_output_records: u64,
    pub delta_transactions: u64,
    pub delta_encoded_bytes: u64,
    pub coalescing_windows: u64,
    pub coalescing_wait_micros: u64,
}

#[derive(Debug)]
pub struct InventoryDelta {
    pub through: u64,
    pub records: Vec<InventoryRecord>,
    pub input_records: usize,
    pub input_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryReadError {
    HistoryGap,
    RecordTooLarge,
    InvalidRecord,
}

impl ResidencyInventory {
    pub fn new(byte_limit: usize) -> Self {
        Self::with_publish_coalescing(byte_limit, Duration::ZERO)
            .expect("zero inventory coalescing window is valid")
    }

    pub fn with_publish_coalescing(
        byte_limit: usize,
        publish_coalesce_window: Duration,
    ) -> Result<Self, String> {
        if publish_coalesce_window > Duration::from_millis(5) {
            return Err("inventory publication coalescing cannot exceed 5 ms".into());
        }
        Ok(Self {
            state: Mutex::new(Inventory {
                residents: BTreeMap::new(),
                journal: VecDeque::new(),
                sequence: 0,
                journal_bytes: 0,
                journal_bytes_peak: 0,
                byte_limit,
                history_gaps: 0,
                flush_through_sequence: 0,
                delta_input_records: 0,
                delta_input_bytes: 0,
                delta_output_records: 0,
                delta_transactions: 0,
                delta_encoded_bytes: 0,
                coalescing_windows: 0,
                coalescing_wait_micros: 0,
            }),
            changed: Arc::new(Notify::new()),
            flush_requested: Arc::new(Notify::new()),
            publish_coalesce_window,
            published: watch::channel(PublishedInventory::default()).0,
        })
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
        state.journal_bytes_peak = state.journal_bytes_peak.max(state.journal_bytes);
        self.changed.notify_one();
    }

    pub fn sequence(&self) -> u64 {
        self.state.lock().sequence
    }

    pub fn status(&self) -> InventoryStatus {
        let state = self.state.lock();
        InventoryStatus {
            sequence: state.sequence,
            resident_records: state.residents.len(),
            journal_records: state.journal.len(),
            journal_bytes: state.journal_bytes,
            journal_bytes_peak: state.journal_bytes_peak,
            journal_capacity_bytes: state.byte_limit,
            history_gaps: state.history_gaps,
            flush_through_sequence: state.flush_through_sequence,
            delta_input_records: state.delta_input_records,
            delta_input_bytes: state.delta_input_bytes,
            delta_output_records: state.delta_output_records,
            delta_transactions: state.delta_transactions,
            delta_encoded_bytes: state.delta_encoded_bytes,
            coalescing_windows: state.coalescing_windows,
            coalescing_wait_micros: state.coalescing_wait_micros,
        }
    }

    pub fn changed(&self) -> Arc<Notify> {
        self.changed.clone()
    }

    pub async fn wait_to_publish(&self, after: u64) -> Duration {
        let window = self.publish_coalesce_window;
        if window.is_zero() || self.state.lock().flush_through_sequence > after {
            return Duration::ZERO;
        }
        let started = tokio::time::Instant::now();
        let maximum = started + Duration::from_millis(5);
        let mut quiet = (started + window).min(maximum);
        loop {
            let change = self.changed.notified();
            let flush = self.flush_requested.notified();
            tokio::pin!(change, flush);
            change.as_mut().enable();
            flush.as_mut().enable();
            if self.state.lock().flush_through_sequence > after {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep_until(quiet) => break,
                _ = tokio::time::sleep_until(maximum) => break,
                _ = &mut flush => {
                    if self.state.lock().flush_through_sequence > after {
                        break;
                    }
                },
                _ = &mut change => quiet = (tokio::time::Instant::now() + window).min(maximum),
            }
        }
        started.elapsed()
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
        let mut state = self.state.lock();
        if after > through
            || through > state.sequence
            || (after != state.sequence
                && state
                    .journal
                    .front()
                    .is_none_or(|record| record.sequence > after + 1))
        {
            state.history_gaps = state.history_gaps.saturating_add(1);
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

    pub fn coalesced_changes(
        &self,
        after: u64,
        through: u64,
    ) -> Result<InventoryDelta, InventoryReadError> {
        let input = self.changes(after, through)?;
        let through = input.last().map_or(after, |record| record.sequence);
        let input_records = input.len();
        let input_bytes = input.iter().map(InventoryRecord::estimated_size).sum();
        let mut records = BTreeMap::new();
        for record in input {
            let Some(metadata) = record.metadata else {
                return Err(InventoryReadError::InvalidRecord);
            };
            records.insert((record.key.clone(), metadata.medium), record);
        }
        Ok(InventoryDelta {
            through,
            records: records.into_values().collect(),
            input_records,
            input_bytes,
        })
    }

    pub fn record_delta_publication(
        &self,
        input_records: usize,
        input_bytes: usize,
        output_records: usize,
        transactions: usize,
        encoded_bytes: usize,
        coalescing_wait: Duration,
    ) {
        let mut state = self.state.lock();
        let input_records = u64::try_from(input_records).unwrap_or(u64::MAX);
        let input_bytes = u64::try_from(input_bytes).unwrap_or(u64::MAX);
        let output_records = u64::try_from(output_records).unwrap_or(u64::MAX);
        let transactions = u64::try_from(transactions).unwrap_or(u64::MAX);
        let encoded_bytes = u64::try_from(encoded_bytes).unwrap_or(u64::MAX);
        let coalescing_wait = u64::try_from(coalescing_wait.as_micros()).unwrap_or(u64::MAX);
        state.delta_input_records = state.delta_input_records.saturating_add(input_records);
        state.delta_input_bytes = state.delta_input_bytes.saturating_add(input_bytes);
        state.delta_output_records = state.delta_output_records.saturating_add(output_records);
        state.delta_transactions = state.delta_transactions.saturating_add(transactions);
        state.delta_encoded_bytes = state.delta_encoded_bytes.saturating_add(encoded_bytes);
        state.coalescing_windows = state.coalescing_windows.saturating_add(1);
        state.coalescing_wait_micros = state.coalescing_wait_micros.saturating_add(coalescing_wait);
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
        let mut progress = self.published.subscribe();
        let target = {
            let mut state = self.state.lock();
            state.flush_through_sequence = state.flush_through_sequence.max(state.sequence);
            state.flush_through_sequence
        };
        self.flush_requested.notify_one();
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
