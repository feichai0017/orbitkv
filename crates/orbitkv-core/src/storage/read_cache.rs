use std::{collections::HashMap, sync::Arc, time::Instant};

use hashlink::LruCache;
use orbitkv_state::{CATALOG_SHARDS, InventoryRecord, catalog_shard};
use parking_lot::Mutex;
use tokio::sync::Notify;

use super::inventory::{Inventory, InventoryReadError};

use crate::block::{SealedBlock, StateKey};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::{
    CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED, CACHE_RESIDENCE_REASON_CLEANUP,
    CACHE_RESIDENCE_REASON_PRESSURE, core_metrics,
};

pub(crate) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    inventory: Option<[Inventory; CATALOG_SHARDS]>,
    cache: TinyLfuCache<StateKey, Arc<SealedBlock>>,
    reclaimable: LruCache<StateKey, ResidentMetadata>,
    retained: LruCache<StateKey, ResidentMetadata>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResidentMetadata {
    inserted_at: Instant,
}

struct RemovedResident {
    key: StateKey,
    block: Arc<SealedBlock>,
    /// Insertion time taken from the block's replacement-class metadata.
    inserted_at: Instant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResidentClass {
    Reclaimable,
    Retained,
}

impl ReadCache {
    pub(crate) fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
        inventory_journal_bytes: Option<usize>,
    ) -> Self {
        let cache =
            TinyLfuCache::new_unbounded(capacity_bytes, enable_lfu_admission, value_size_hint);
        Self {
            inner: Mutex::new(ReadCacheInner {
                inventory: inventory_journal_bytes
                    .map(|bytes| std::array::from_fn(|_| Inventory::new(bytes / CATALOG_SHARDS))),
                cache,
                reclaimable: LruCache::new_unbounded(),
                retained: LruCache::new_unbounded(),
            }),
        }
    }

    pub(crate) fn inventory_sequence(&self, shard: usize) -> u64 {
        self.inner
            .lock()
            .inventory
            .as_ref()
            .expect("inventory enabled")[shard]
            .sequence()
    }

    pub(crate) fn inventory_changed(&self, shard: usize) -> Arc<Notify> {
        self.inner
            .lock()
            .inventory
            .as_ref()
            .expect("inventory enabled")[shard]
            .changed()
    }

    pub(crate) fn inventory_page(
        &self,
        shard: usize,
        after: Option<&StateKey>,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        self.inner
            .lock()
            .inventory
            .as_ref()
            .expect("inventory enabled")[shard]
            .snapshot_page(after)
    }

    pub(crate) fn inventory_changes(
        &self,
        shard: usize,
        after: u64,
        through: u64,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        self.inner
            .lock()
            .inventory
            .as_ref()
            .expect("inventory enabled")[shard]
            .changes(after, through)
    }

    pub(crate) fn inventory_covers(&self, shard: usize, after: u64) -> bool {
        self.inner
            .lock()
            .inventory
            .as_ref()
            .expect("inventory enabled")[shard]
            .covers(after)
    }

    pub(super) fn contains_keys(&self, keys: &[StateKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    pub(super) fn get_prefix_blocks(
        &self,
        keys: &[StateKey],
        warming: bool,
    ) -> (usize, Vec<Arc<SealedBlock>>) {
        let mut hit = 0usize;
        let mut blocks = Vec::with_capacity(keys.len());
        {
            let mut inner = self.inner.lock();
            for key in keys {
                let block = if warming {
                    inner.cache.peek(key)
                } else {
                    inner.cache.get(key)
                };
                if let Some(block) = block {
                    if !warming {
                        retain_warmed(&mut inner, key, &block);
                        refresh_recency(&mut inner, key);
                    }
                    hit += 1;
                    blocks.push(block);
                } else {
                    break;
                }
            }
        }
        (hit, blocks)
    }

    pub(super) fn retain_warmed(&self, keys: &[StateKey], blocks: &[Arc<SealedBlock>]) {
        if !blocks.iter().any(|block| block.was_warmed()) {
            return;
        }
        let mut inner = self.inner.lock();
        for (key, block) in keys.iter().zip(blocks) {
            if block.was_warmed()
                && inner
                    .cache
                    .peek(key)
                    .is_some_and(|resident| Arc::ptr_eq(&resident, block))
            {
                retain_warmed(&mut inner, key, block);
            }
        }
    }

    pub(super) fn batch_insert(&self, blocks: Vec<(StateKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(&mut inner, key, block, ResidentClass::Retained);
        }
    }

    pub(super) fn batch_insert_reclaimable(&self, blocks: Vec<(StateKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(&mut inner, key, block, ResidentClass::Reclaimable);
        }
    }

    pub(super) fn batch_insert_refs(&self, blocks: &[(StateKey, Arc<SealedBlock>)]) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(
                &mut inner,
                key.clone(),
                Arc::clone(block),
                ResidentClass::Retained,
            );
        }
    }

    /// Check every insertion version and acquire payload Arcs under the same
    /// cache lock. A stale batch exposes no addresses and creates no session.
    pub(super) fn pin_residencies(
        &self,
        records: &[InventoryRecord],
    ) -> Option<Vec<(StateKey, Arc<SealedBlock>)>> {
        let mut inner = self.inner.lock();
        let inventory = inner.inventory.as_ref()?;
        if records.is_empty()
            || !records
                .iter()
                .all(|r| inventory[catalog_shard(&r.key)].contains_record(r))
        {
            return None;
        }
        let mut found = Vec::with_capacity(records.len());
        for record in records {
            let block = inner.cache.get(&record.key)?;
            refresh_recency(&mut inner, &record.key);
            found.push((record.key.clone(), block));
        }
        Some(found)
    }

    /// Position-aligned membership: entry `i` is the block for `keys[i]`, or
    /// `None` on miss. Unlike [`Self::get_prefix_blocks`] this never stops at
    /// the first gap — hybrid-cache checkpoint groups (recurrent state) have
    /// sparse hit patterns by design, where the caller picks the rightmost
    /// hit instead of a prefix.
    pub(super) fn get_blocks_aligned(&self, keys: &[StateKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let mut inner = self.inner.lock();
        keys.iter()
            .map(|key| {
                inner.cache.get(key).inspect(|_| {
                    refresh_recency(&mut inner, key);
                })
            })
            .collect()
    }

    pub(super) fn remove_lru_batch(
        &self,
        batch_size: usize,
        target_bytes: u64,
    ) -> Vec<(StateKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let mut removed = Vec::with_capacity(batch_size);
            let mut removed_bytes = 0;
            remove_lru_batch_from_class(
                &mut inner,
                ResidentClass::Reclaimable,
                batch_size,
                target_bytes,
                &mut removed,
                &mut removed_bytes,
            );
            if removed.len() < batch_size && removed_bytes < target_bytes {
                remove_lru_batch_from_class(
                    &mut inner,
                    ResidentClass::Retained,
                    batch_size,
                    target_bytes,
                    &mut removed,
                    &mut removed_bytes,
                );
            }
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_PRESSURE)
    }

    pub(super) fn remove_all(&self) -> Vec<(StateKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let reclaimable_blocks = inner.reclaimable.len() as i64;
            let retained_blocks = inner.retained.len() as i64;
            let mut metadata = HashMap::with_capacity(
                inner.reclaimable.len().saturating_add(inner.retained.len()),
            );
            metadata.extend(inner.reclaimable.drain());
            metadata.extend(inner.retained.drain());
            let removed = inner
                .cache
                .remove_all()
                .into_iter()
                .map(|(key, block)| {
                    let inserted_at = metadata.remove(&key).map(|entry| entry.inserted_at);
                    debug_assert!(
                        inserted_at.is_some(),
                        "resident block is missing its replacement metadata"
                    );
                    RemovedResident {
                        inserted_at: inserted_at.unwrap_or_else(Instant::now),
                        key,
                        block,
                    }
                })
                .collect::<Vec<_>>();
            if let Some(inventory) = &mut inner.inventory {
                for entry in &removed {
                    inventory[catalog_shard(&entry.key)].change(&entry.key, false);
                }
            }
            debug_assert_eq!(
                removed.len() as i64,
                reclaimable_blocks + retained_blocks,
                "resident cache and replacement classes diverged"
            );
            debug_assert!(
                metadata.is_empty(),
                "replacement metadata outlives its resident block"
            );
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-reclaimable_blocks, &*CACHE_CLASS_RECLAIMABLE);
            metrics
                .cache_resident_blocks
                .add(-retained_blocks, &*CACHE_CLASS_RETAINED);
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_CLEANUP)
    }

    pub(crate) fn mark_reclaimable_records(&self, records: &[InventoryRecord]) {
        let mut inner = self.inner.lock();
        let mut moved = 0;
        for record in records {
            if inner.inventory.as_ref().is_some_and(|inventory| {
                inventory[catalog_shard(&record.key)].contains_record(record)
            }) && mark_reclaimable(&mut inner, &record.key)
            {
                moved += 1;
            }
        }
        if moved > 0 {
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-moved, &*CACHE_CLASS_RETAINED);
            metrics
                .cache_resident_blocks
                .add(moved, &*CACHE_CLASS_RECLAIMABLE);
        }
    }
}

impl ResidentClass {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::Reclaimable => &*CACHE_CLASS_RECLAIMABLE,
            Self::Retained => &*CACHE_CLASS_RETAINED,
        }
    }
}

fn insert_block(
    inner: &mut ReadCacheInner,
    key: StateKey,
    block: Arc<SealedBlock>,
    class: ResidentClass,
) -> CacheInsertOutcome {
    if block.was_warmed() && inner.cache.contains_key(&key) {
        return CacheInsertOutcome::AlreadyExists;
    }
    let footprint_bytes = block.memory_footprint();
    let outcome = inner.cache.insert(key.clone(), block);
    match outcome {
        CacheInsertOutcome::InsertedNew => {
            if let Some(inventory) = &mut inner.inventory {
                inventory[catalog_shard(&key)].change(&key, true);
            }
            class_lru(inner, class).insert(
                key,
                ResidentMetadata {
                    inserted_at: Instant::now(),
                },
            );
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
        }
        CacheInsertOutcome::AlreadyExists => refresh_recency(inner, &key),
        CacheInsertOutcome::Rejected => {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }
    outcome
}

fn class_lru(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
) -> &mut LruCache<StateKey, ResidentMetadata> {
    match class {
        ResidentClass::Reclaimable => &mut inner.reclaimable,
        ResidentClass::Retained => &mut inner.retained,
    }
}

fn retain_warmed(inner: &mut ReadCacheInner, key: &StateKey, block: &SealedBlock) {
    if block.was_warmed()
        && let Some(metadata) = inner.reclaimable.remove(key)
    {
        inner.retained.insert(key.clone(), metadata);
        let metrics = core_metrics();
        metrics
            .cache_resident_blocks
            .add(-1, &*CACHE_CLASS_RECLAIMABLE);
        metrics.cache_resident_blocks.add(1, &*CACHE_CLASS_RETAINED);
    }
}

fn refresh_recency(inner: &mut ReadCacheInner, key: &StateKey) {
    let classified = inner.reclaimable.get(key).is_some() || inner.retained.get(key).is_some();
    debug_assert!(
        classified || !inner.cache.contains_key(key),
        "resident block is missing its replacement class"
    );
}

fn mark_reclaimable(inner: &mut ReadCacheInner, key: &StateKey) -> bool {
    if !inner.cache.contains_key(key) {
        return false;
    }
    if let Some(metadata) = inner.retained.remove(key) {
        inner.reclaimable.insert(key.clone(), metadata);
        true
    } else {
        debug_assert!(
            inner.reclaimable.contains_key(key),
            "resident block is missing its replacement class"
        );
        false
    }
}

fn remove_lru(inner: &mut ReadCacheInner, class: ResidentClass) -> Option<RemovedResident> {
    while let Some((key, metadata)) = class_lru(inner, class).remove_lru() {
        let block = inner.cache.remove(&key);
        debug_assert!(
            block.is_some(),
            "replacement class contains a non-resident block"
        );
        let Some(block) = block else {
            continue;
        };
        if let Some(inventory) = &mut inner.inventory {
            inventory[catalog_shard(&key)].change(&key, false);
        }
        let metrics = core_metrics();
        metrics.cache_resident_blocks.add(-1, class.attributes());
        metrics
            .cache_block_evictions_by_class
            .add(1, class.attributes());
        return Some(RemovedResident {
            key,
            block,
            inserted_at: metadata.inserted_at,
        });
    }
    None
}

fn remove_lru_batch_from_class(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
    batch_size: usize,
    target_bytes: u64,
    removed: &mut Vec<RemovedResident>,
    removed_bytes: &mut u64,
) {
    let candidates = class_lru(inner, class).len();
    for _ in 0..candidates {
        if removed.len() == batch_size || *removed_bytes >= target_bytes {
            break;
        }

        let Some(key) = class_lru(inner, class)
            .iter()
            .next()
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        if inner.cache.is_cache_owned_only(&key) {
            let block = remove_lru(inner, class)
                .expect("cache-owned LRU candidate must remain resident while locked");
            *removed_bytes = removed_bytes.saturating_add(block.block.memory_footprint());
            removed.push(block);
        } else {
            class_lru(inner, class).get(&key);
        }
    }
}

fn record_residence_durations(
    removed: Vec<RemovedResident>,
    attributes: &[opentelemetry::KeyValue],
) -> Vec<(StateKey, Arc<SealedBlock>)> {
    let removed_at = Instant::now();
    let metrics = core_metrics();
    removed
        .into_iter()
        .map(|entry| {
            metrics.cache_residence_duration.record(
                residence_duration_seconds(entry.inserted_at, removed_at),
                attributes,
            );
            (entry.key, entry.block)
        })
        .collect()
}

fn residence_duration_seconds(inserted_at: Instant, removed_at: Instant) -> f64 {
    removed_at
        .saturating_duration_since(inserted_at)
        .as_secs_f64()
}

#[cfg(test)]
#[path = "../../tests/unit/storage/read_cache.rs"]
mod tests;
