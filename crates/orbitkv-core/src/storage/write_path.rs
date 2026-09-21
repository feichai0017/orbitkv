use std::collections::{HashMap, hash_map::Entry};
use std::sync::{Arc, Weak};

use log::{debug, error, info, warn};
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;

use crate::backing::SsdBackingStore;
use crate::block::{InflightBlock, SealedBlock, SlotInsertResult, StateKey};
use crate::metrics::core_metrics;
use crate::offload::InsertEntries;
use orbitkv_common::NumaNode;

use super::read_cache::ReadCache;

pub(super) enum InsertWorkerCommand {
    RawInsert(crate::offload::RawSaveBatch),
    Flush(oneshot::Sender<()>),
    Gc {
        max_age: std::time::Duration,
        reply: oneshot::Sender<usize>,
    },
}

pub(super) struct WritePipeline {
    insert_tx: Sender<InsertWorkerCommand>,
}

impl WritePipeline {
    pub(super) fn new() -> (Self, Receiver<InsertWorkerCommand>) {
        let (insert_tx, insert_rx) = std::sync::mpsc::channel();
        (Self { insert_tx }, insert_rx)
    }

    pub(super) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
        let _ = self.insert_tx.send(InsertWorkerCommand::RawInsert(batch));
    }

    /// Send a flush barrier through the insert channel.
    ///
    /// The returned receiver resolves once the worker has processed all commands
    /// that were enqueued before this flush.
    pub(super) fn flush(&self) -> Option<oneshot::Receiver<()>> {
        let (tx, rx) = oneshot::channel();
        self.insert_tx
            .send(InsertWorkerCommand::Flush(tx))
            .ok()
            .map(|()| rx)
    }

    pub(super) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .insert_tx
            .send(InsertWorkerCommand::Gc {
                max_age,
                reply: reply_tx,
            })
            .is_err()
        {
            return 0;
        }
        reply_rx.await.unwrap_or(0)
    }
}

pub(super) struct InsertDeps {
    pub(super) read_cache: Arc<ReadCache>,
    pub(super) ssd_store: Option<Arc<SsdBackingStore>>,
}

pub(super) fn insert_worker_loop(rx: Receiver<InsertWorkerCommand>, deps: Weak<InsertDeps>) {
    let mut inflight: HashMap<StateKey, InflightBlock> = HashMap::new();

    while let Ok(cmd) = rx.recv() {
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        for cmd in cmds {
            match cmd {
                InsertWorkerCommand::RawInsert(batch) => {
                    process_raw_save_batch(&mut inflight, &deps, batch);
                }
                InsertWorkerCommand::Flush(tx) => {
                    let _ = tx.send(());
                }
                InsertWorkerCommand::Gc { max_age, reply } => {
                    let cleaned = gc_inflight(&mut inflight, max_age);
                    let _ = reply.send(cleaned);
                }
            }
        }
    }

    info!(
        "Insert worker shutting down, {} inflight blocks remaining",
        inflight.len()
    );
}

fn process_raw_save_batch(
    inflight: &mut HashMap<StateKey, InflightBlock>,
    deps: &Weak<InsertDeps>,
    batch: crate::offload::RawSaveBatch,
) {
    let start = std::time::Instant::now();
    let namespace = batch.namespace.clone();
    let numa_node = batch.numa_node;
    let total_slots = batch.total_slots;

    let (entries, total_bytes, total_blocks) = crate::offload::build_insert_entries(batch);

    process_insert_batch(inflight, deps, entries, total_slots, numa_node, &namespace);

    debug!(
        "insert_worker: batch sealed blocks={} bytes={} ms={:.2}",
        total_blocks,
        total_bytes,
        start.elapsed().as_secs_f64() * 1000.0,
    );
}

fn process_insert_batch(
    inflight: &mut HashMap<StateKey, InflightBlock>,
    deps: &Weak<InsertDeps>,
    entries: InsertEntries,
    total_slots: usize,
    numa_node: NumaNode,
    namespace: &str,
) -> usize {
    let mut sealed_blocks: Vec<(StateKey, Arc<SealedBlock>)> = Vec::new();
    let mut inflight_bytes_added: u64 = 0;
    let mut inflight_bytes_removed: u64 = 0;
    let mut ordered_fast_path_seals = 0usize;

    // Upgrade once: dedup against resident blocks below + publish seals at the end.
    let deps = deps.upgrade();

    for (key, slots) in entries {
        // Drop a late duplicate save of an already-resident block.
        if let Some(deps) = &deps
            && deps.read_cache.contains_keys(std::slice::from_ref(&key))[0]
        {
            continue;
        }

        if !inflight.contains_key(&key) {
            match SealedBlock::from_ordered_slot_inserts(slots, total_slots, numa_node) {
                Ok(sealed) => {
                    ordered_fast_path_seals += 1;
                    sealed_blocks.push((key, Arc::new(sealed)));
                    continue;
                }
                Err(slots) => {
                    insert_partial_slots(
                        inflight,
                        key,
                        slots,
                        total_slots,
                        numa_node,
                        namespace,
                        &mut sealed_blocks,
                        &mut inflight_bytes_added,
                        &mut inflight_bytes_removed,
                    );
                    continue;
                }
            }
        }

        insert_partial_slots(
            inflight,
            key,
            slots,
            total_slots,
            numa_node,
            namespace,
            &mut sealed_blocks,
            &mut inflight_bytes_added,
            &mut inflight_bytes_removed,
        );
    }

    if inflight_bytes_added > 0 {
        core_metrics()
            .inflight_bytes
            .add(inflight_bytes_added as i64, &[]);
    }
    if inflight_bytes_removed > 0 {
        core_metrics()
            .inflight_bytes
            .add(-(inflight_bytes_removed as i64), &[]);
    }

    if !sealed_blocks.is_empty()
        && let Some(deps) = &deps
    {
        deps.read_cache.batch_insert_refs(&sealed_blocks);
        if let Some(ssd) = &deps.ssd_store {
            ssd.ingest_batch(
                sealed_blocks
                    .iter()
                    .map(|(key, block)| (key.clone(), Arc::downgrade(block)))
                    .collect(),
            );
        }
    }

    ordered_fast_path_seals
}

#[allow(
    clippy::too_many_arguments,
    reason = "insert worker threads batch-local accounting through the fallback path"
)]
fn insert_partial_slots(
    inflight: &mut HashMap<StateKey, InflightBlock>,
    key: StateKey,
    slots: Vec<(usize, crate::block::RawBlock)>,
    total_slots: usize,
    numa_node: NumaNode,
    namespace: &str,
    sealed_blocks: &mut Vec<(StateKey, Arc<SealedBlock>)>,
    inflight_bytes_added: &mut u64,
    inflight_bytes_removed: &mut u64,
) {
    let inflight_block = match inflight.entry(key.clone()) {
        Entry::Vacant(v) => v.insert(InflightBlock::new(total_slots)),
        Entry::Occupied(o) => {
            let ib = o.into_mut();
            if ib.total_slots() != total_slots {
                error!(
                    "insert worker: slot count mismatch: key namespace={} expected={} got={}",
                    namespace,
                    ib.total_slots(),
                    total_slots
                );
                return;
            }
            ib
        }
    };

    let mut completed = false;
    for (slot_id, block) in slots {
        match inflight_block.insert_slot(slot_id, block, numa_node) {
            SlotInsertResult::Inserted {
                completed: c,
                footprint_added,
            } => {
                *inflight_bytes_added = inflight_bytes_added.saturating_add(footprint_added);
                completed = c;
                if completed {
                    break;
                }
            }
            SlotInsertResult::Duplicate => {}
        }
    }

    if completed {
        let inflight_block = inflight.remove(&key).expect("just inserted");
        let total_footprint = inflight_block.footprint();
        *inflight_bytes_removed = inflight_bytes_removed.saturating_add(total_footprint);
        let sealed = Arc::new(inflight_block.seal());

        sealed_blocks.push((key, sealed));
    }
}

fn gc_inflight(
    inflight: &mut HashMap<StateKey, InflightBlock>,
    max_age: std::time::Duration,
) -> usize {
    let before = inflight.len();

    inflight.retain(|key, block| {
        let age = block.age();
        if age > max_age {
            warn!(
                "GC: removing stale inflight block: namespace={} hash_len={} filled={} total={} age_secs={}",
                key.namespace,
                key.hash.len(),
                block.filled_count(),
                block.total_slots(),
                age.as_secs()
            );
            core_metrics().inflight_bytes.add(-(block.footprint() as i64), &[]);
            false
        } else {
            true
        }
    });

    let cleaned = before - inflight.len();
    if cleaned > 0 {
        core_metrics().inflight_gc_cleaned.add(cleaned as u64, &[]);
        info!("GC cleaned stale inflight blocks: cleaned={}", cleaned);
    }
    cleaned
}

#[cfg(test)]
#[path = "../../tests/unit/storage/write_path.rs"]
mod tests;
