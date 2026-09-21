use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use dashmap::{DashMap, mapref::entry::Entry};
use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_REPLICAS, INVENTORY_BATCH_BYTES,
    INVENTORY_BATCH_RECORDS, InventoryOperation, InventoryRecord, InventoryStatus, ReplicaLocation,
    StateKey,
};
use parking_lot::Mutex;
use uuid::Uuid;

pub const DEFAULT_NODE_STALE_SECS: u64 = 30;
pub const DEFAULT_TTL_MINUTES: u64 = 120;
pub const DEFAULT_INVENTORY_BYTES_PER_NODE: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct StoreConfig {
    pub node_stale_after: Duration,
    pub ttl: Duration,
    /// Accounted key/index bytes per owner, excluding one bounded retry batch.
    pub inventory_bytes_per_node: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            node_stale_after: Duration::from_secs(DEFAULT_NODE_STALE_SECS),
            ttl: Duration::from_secs(DEFAULT_TTL_MINUTES * 60),
            inventory_bytes_per_node: DEFAULT_INVENTORY_BYTES_PER_NODE,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepStats {
    pub removed_owners: usize,
    pub removed_keys: usize,
    pub removed_nodes: usize,
}

impl SweepStats {
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// Stored copies, including inventories that have not committed yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RedundancySnapshot {
    pub keys_1: u64,
    pub keys_2: u64,
    pub keys_3: u64,
    pub keys_4plus: u64,
    pub copies: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    UnknownNode,
    StaleSession,
    CatalogRestarted,
    OutOfOrder,
    InvalidInventory,
    Capacity,
}

struct NodeInventory {
    node_id: Uuid,
    last_seen: Instant,
    retired: bool,
    progress: InventoryStatus,
    entries: BTreeMap<StateKey, u64>,
    bytes: usize,
    snapshot_max_sequence: u64,
    last_key: Option<StateKey>,
    replaying: bool,
    /// Only the immediately preceding operation may be retried verbatim.
    last_operation: Option<InventoryOperation>,
}

impl NodeInventory {
    fn new(node_id: Uuid) -> Self {
        Self {
            node_id,
            last_seen: Instant::now(),
            retired: false,
            progress: InventoryStatus::default(),
            entries: BTreeMap::new(),
            bytes: 0,
            snapshot_max_sequence: 0,
            last_key: None,
            replaying: false,
            last_operation: None,
        }
    }
}

/// Each owner's mutations are serialized. Never hold a DashMap guard while
/// acquiring a node lock, or a block guard while checking node visibility.
pub struct BlockHashStore {
    blocks: DashMap<StateKey, HashSet<Arc<str>>>,
    nodes: DashMap<Arc<str>, Arc<Mutex<NodeInventory>>>,
    epoch: Uuid,
    config: StoreConfig,
    redundancy: RedundancyCounters,
}

impl BlockHashStore {
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    pub fn with_config(config: StoreConfig) -> Self {
        Self {
            blocks: DashMap::new(),
            nodes: DashMap::new(),
            epoch: Uuid::new_v4(),
            config,
            redundancy: RedundancyCounters::default(),
        }
    }

    pub fn config(&self) -> StoreConfig {
        self.config
    }
    pub fn catalog_epoch(&self) -> Uuid {
        self.epoch
    }

    fn node(&self, node: &str) -> Option<Arc<Mutex<NodeInventory>>> {
        self.nodes.get(node).map(|entry| Arc::clone(entry.value()))
    }

    pub fn heartbeat_node(&self, node: &str, node_id: Uuid) -> Result<InventoryStatus, StoreError> {
        if node.is_empty() || node.len() > 4096 {
            return Err(StoreError::InvalidInventory);
        }
        loop {
            let inventory = Arc::clone(
                self.nodes
                    .entry(Arc::from(node))
                    .or_insert_with(|| Arc::new(Mutex::new(NodeInventory::new(node_id))))
                    .value(),
            );
            let mut state = inventory.lock();
            if state.retired {
                continue;
            }
            if state.node_id != node_id {
                if state.last_seen.elapsed() <= self.config.node_stale_after {
                    return Err(StoreError::StaleSession);
                }
                self.clear_owner(node, &mut state);
                *state = NodeInventory::new(node_id);
            }
            state.last_seen = Instant::now();
            return Ok(state.progress);
        }
    }

    pub fn unregister_node(&self, node: &str, node_id: Uuid) -> Result<usize, StoreError> {
        let inventory = self.node(node).ok_or(StoreError::UnknownNode)?;
        let mut state = inventory.lock();
        Self::check_session(&state, node_id)?;
        state.retired = true;
        let stats = self.clear_owner(node, &mut state);
        self.nodes
            .remove_if(node, |_, current| Arc::ptr_eq(current, &inventory));
        Ok(stats.removed_owners)
    }

    pub fn sync_inventory(
        &self,
        node: &str,
        node_id: Uuid,
        epoch: Uuid,
        generation: u64,
        operation: InventoryOperation,
    ) -> Result<(InventoryStatus, Vec<InventoryRecord>), StoreError> {
        if epoch != self.epoch {
            return Err(StoreError::CatalogRestarted);
        }
        let inventory = self.node(node).ok_or(StoreError::UnknownNode)?;
        let mut state = inventory.lock();
        Self::check_session(&state, node_id)?;
        if generation == 0 {
            return Err(StoreError::InvalidInventory);
        }
        let retry = generation == state.progress.generation
            && state.last_operation.as_ref() == Some(&operation);
        if !retry {
            match &operation {
                InventoryOperation::Begin { sequence } => {
                    if generation <= state.progress.generation {
                        return Err(StoreError::OutOfOrder);
                    }
                    self.clear_owner(node, &mut state);
                    state.progress = InventoryStatus {
                        generation,
                        sequence: *sequence,
                        next_page: 0,
                        ready: false,
                    };
                    state.snapshot_max_sequence = *sequence;
                    state.last_key = None;
                    state.replaying = false;
                }
                InventoryOperation::Snapshot { page, records } => {
                    Self::check_generation(&state, generation)?;
                    validate_records(records)?;
                    if state.progress.ready
                        || state.replaying
                        || *page != state.progress.next_page
                        || *page == u64::MAX
                        || records.iter().any(|r| !r.present)
                        || records.windows(2).any(|w| w[0].key >= w[1].key)
                        || state
                            .last_key
                            .as_ref()
                            .is_some_and(|k| *k >= records[0].key)
                    {
                        return Err(StoreError::OutOfOrder);
                    }
                    self.check_capacity(&state, records)?;
                    for record in records {
                        self.apply(node, &mut state, record);
                        state.snapshot_max_sequence =
                            state.snapshot_max_sequence.max(record.sequence);
                    }
                    state.last_key = records.last().map(|r| r.key.clone());
                    state.progress.next_page += 1;
                }
                InventoryOperation::Delta { after, records } => {
                    Self::check_generation(&state, generation)?;
                    validate_records(records)?;
                    if *after != state.progress.sequence
                        || records
                            .iter()
                            .enumerate()
                            .any(|(i, r)| after.checked_add(i as u64 + 1) != Some(r.sequence))
                    {
                        return Err(StoreError::OutOfOrder);
                    }
                    self.check_capacity(&state, records)?;
                    for record in records {
                        self.apply(node, &mut state, record);
                    }
                    state.progress.sequence =
                        records.last().ok_or(StoreError::InvalidInventory)?.sequence;
                    state.replaying = true;
                }
                InventoryOperation::Commit { sequence } => {
                    Self::check_generation(&state, generation)?;
                    if state.progress.ready
                        || *sequence != state.progress.sequence
                        || *sequence < state.snapshot_max_sequence
                    {
                        return Err(StoreError::OutOfOrder);
                    }
                    state.progress.ready = true;
                }
            }
        }
        state.last_seen = Instant::now();
        let progress = state.progress;
        let candidates = match &operation {
            InventoryOperation::Snapshot { records, .. }
            | InventoryOperation::Delta { records, .. } => records
                .iter()
                .filter(|r| r.present)
                .cloned()
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        state.last_operation = Some(operation);
        drop(state);
        // Advisory replacement hints never lock two owners at once.
        let reclaimable = candidates
            .into_iter()
            .filter(|r| {
                self.visible_replicas(&r.key, node)
                    .into_iter()
                    .take(2)
                    .count()
                    == 2
            })
            .collect();
        Ok((progress, reclaimable))
    }

    fn check_session(state: &NodeInventory, node_id: Uuid) -> Result<(), StoreError> {
        if state.retired {
            Err(StoreError::UnknownNode)
        } else if state.node_id != node_id {
            Err(StoreError::StaleSession)
        } else {
            Ok(())
        }
    }

    fn check_generation(state: &NodeInventory, generation: u64) -> Result<(), StoreError> {
        if state.progress.generation == generation {
            Ok(())
        } else {
            Err(StoreError::OutOfOrder)
        }
    }

    fn check_capacity(
        &self,
        state: &NodeInventory,
        records: &[InventoryRecord],
    ) -> Result<(), StoreError> {
        let mut bytes = state.bytes;
        let mut projected = HashMap::new();
        for record in records {
            let current = projected
                .entry(&record.key)
                .or_insert_with(|| state.entries.get(&record.key).copied());
            if current.is_some_and(|sequence| sequence > record.sequence) {
                continue;
            }
            if *current == Some(record.sequence) && !record.present {
                return Err(StoreError::InvalidInventory);
            }
            match (current.is_some(), record.present) {
                (false, true) => bytes += key_bytes(&record.key),
                (true, false) => bytes -= key_bytes(&record.key),
                _ => {}
            }
            *current = record.present.then_some(record.sequence);
            if bytes > self.config.inventory_bytes_per_node {
                return Err(StoreError::Capacity);
            }
        }
        Ok(())
    }

    fn apply(&self, node: &str, state: &mut NodeInventory, record: &InventoryRecord) {
        // A paginated snapshot can sample a key after an earlier replayed delta.
        if state
            .entries
            .get(&record.key)
            .is_some_and(|s| *s > record.sequence)
        {
            return;
        }
        if record.present {
            if state
                .entries
                .insert(record.key.clone(), record.sequence)
                .is_none()
            {
                state.bytes += key_bytes(&record.key);
                let mut owners = self.blocks.entry(record.key.clone()).or_default();
                let before = owners.len();
                owners.insert(Arc::from(node));
                self.redundancy.adjust(before as u64, owners.len() as u64);
            }
        } else if state.entries.remove(&record.key).is_some() {
            state.bytes -= key_bytes(&record.key);
            self.remove_owner(node, &record.key);
        }
    }

    fn remove_owner(&self, node: &str, key: &StateKey) -> bool {
        if let Entry::Occupied(mut entry) = self.blocks.entry(key.clone()) {
            let before = entry.get().len();
            entry.get_mut().remove(node);
            self.redundancy
                .adjust(before as u64, entry.get().len() as u64);
            if entry.get().is_empty() {
                entry.remove();
                return true;
            }
        }
        false
    }

    fn clear_owner(&self, node: &str, state: &mut NodeInventory) -> SweepStats {
        state.progress.ready = false;
        let mut stats = SweepStats {
            removed_owners: state.entries.len(),
            ..SweepStats::default()
        };
        for (key, _) in std::mem::take(&mut state.entries) {
            stats.removed_keys += usize::from(self.remove_owner(node, &key));
        }
        state.bytes = 0;
        stats
    }

    fn visible_replicas(&self, key: &StateKey, exclude: &str) -> Vec<ReplicaLocation> {
        let mut candidates: Vec<_> = self
            .blocks
            .get(key)
            .map(|owners| {
                owners
                    .iter()
                    .filter(|owner| owner.as_ref() != exclude)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        candidates.sort_unstable();
        candidates
            .into_iter()
            .filter_map(|endpoint| {
                let inventory = self.node(&endpoint)?;
                let state = inventory.lock();
                if state.retired
                    || !state.progress.ready
                    || state.last_seen.elapsed() > self.config.node_stale_after
                {
                    return None;
                }
                Some(ReplicaLocation {
                    owner: CacheOwner {
                        endpoint: endpoint.to_string(),
                        incarnation: state.node_id,
                    },
                    sequence: *state.entries.get(key)?,
                })
            })
            .take(DISCOVERY_MAX_REPLICAS)
            .collect()
    }

    /// Return bounded owner evidence for every position, including misses.
    /// Candidates confer no right to read memory; the owner must validate and pin.
    pub fn locate_blocks(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        exclude: &str,
    ) -> Vec<BlockCandidates> {
        hashes
            .iter()
            .map(|hash| {
                let key = StateKey::new(namespace.to_owned(), hash.clone());
                let replicas = self.visible_replicas(&key, exclude);
                BlockCandidates { key, replicas }
            })
            .collect()
    }

    /// Cleanup is proportional to the expired owners' inventories, not all keys.
    pub fn sweep_expired(&self) -> SweepStats {
        let nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|e| (Arc::clone(e.key()), Arc::clone(e.value())))
            .collect();
        let mut stats = SweepStats::default();
        for (node, inventory) in nodes {
            let mut state = inventory.lock();
            if state.retired || state.last_seen.elapsed() <= self.config.ttl {
                continue;
            }
            state.retired = true;
            let removed = self.clear_owner(&node, &mut state);
            stats.removed_owners += removed.removed_owners;
            stats.removed_keys += removed.removed_keys;
            stats.removed_nodes += 1;
            self.nodes
                .remove_if(node.as_ref(), |_, current| Arc::ptr_eq(current, &inventory));
        }
        stats
    }

    pub fn redundancy_snapshot(&self) -> RedundancySnapshot {
        self.redundancy.snapshot()
    }
    pub fn entry_count(&self) -> u64 {
        let snap = self.redundancy.snapshot();
        snap.keys_1 + snap.keys_2 + snap.keys_3 + snap.keys_4plus
    }
    pub fn owner_count(&self) -> u64 {
        self.redundancy.copies.load(Ordering::Relaxed)
    }
    pub fn node_counts(&self) -> (u64, u64) {
        let nodes: Vec<_> = self.nodes.iter().map(|e| Arc::clone(e.value())).collect();
        let (mut active, mut stale) = (0, 0);
        for inventory in nodes {
            let state = inventory.lock();
            if state.retired {
                continue;
            }
            if state.last_seen.elapsed() <= self.config.node_stale_after {
                active += 1;
            } else {
                stale += 1;
            }
        }
        (active, stale)
    }
}

fn key_bytes(key: &StateKey) -> usize {
    // Both the forward index and the owner index hold an owned key.
    192 + 2 * (key.namespace.len() + key.hash.len())
}

fn validate_records(records: &[InventoryRecord]) -> Result<(), StoreError> {
    if records.is_empty()
        || records.len() > INVENTORY_BATCH_RECORDS
        || records
            .iter()
            .map(InventoryRecord::estimated_size)
            .sum::<usize>()
            > INVENTORY_BATCH_BYTES
        || records
            .iter()
            .any(|r| r.sequence == 0 || r.key.namespace.is_empty() || r.key.hash.is_empty())
    {
        Err(StoreError::InvalidInventory)
    } else {
        Ok(())
    }
}

impl Default for BlockHashStore {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Default)]
struct RedundancyCounters {
    keys_1: AtomicU64,
    keys_2: AtomicU64,
    keys_3: AtomicU64,
    keys_4plus: AtomicU64,
    copies: AtomicU64,
}

impl RedundancyCounters {
    fn snapshot(&self) -> RedundancySnapshot {
        RedundancySnapshot {
            keys_1: self.keys_1.load(Ordering::Relaxed),
            keys_2: self.keys_2.load(Ordering::Relaxed),
            keys_3: self.keys_3.load(Ordering::Relaxed),
            keys_4plus: self.keys_4plus.load(Ordering::Relaxed),
            copies: self.copies.load(Ordering::Relaxed),
        }
    }

    fn adjust_bucket(&self, count: u64, delta: i64) {
        let counter = match count {
            1 => &self.keys_1,
            2 => &self.keys_2,
            3 => &self.keys_3,
            _ if count >= 4 => &self.keys_4plus,
            _ => return,
        };
        if delta > 0 {
            counter.fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            counter.fetch_sub((-delta) as u64, Ordering::Relaxed);
        }
    }

    fn adjust(&self, before: u64, after: u64) {
        if before == after {
            return;
        }
        self.adjust_bucket(before, -1);
        self.adjust_bucket(after, 1);
        if after > before {
            self.copies.fetch_add(after - before, Ordering::Relaxed);
        } else {
            self.copies.fetch_sub(before - after, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/store.rs"]
mod tests;
