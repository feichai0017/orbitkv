use futures::stream::{FuturesOrdered, FuturesUnordered, StreamExt};
use log::{debug, warn};
use mea::oneshot;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use super::ssd::SsdBackingStore;
use super::uring::UringIoEngine;
use crate::block::{RawBlock, SealedBlock, StateKey};
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
use crate::pinned_pool::PinnedAllocation;
use crate::seal_offload::{self, SlotMeta};
use smallvec::SmallVec;

/// SSD I/O alignment requirement (O_DIRECT requires 512-byte aligned I/O)
pub(crate) const SSD_ALIGNMENT: usize = 512;

/// Default write queue depth for SSD writer thread (blocks dropped if full)
pub const DEFAULT_SSD_WRITE_QUEUE_DEPTH: usize = 8;

/// Default prefetch queue depth (limits read tail latency)
pub const DEFAULT_SSD_PREFETCH_QUEUE_DEPTH: usize = 2;

/// Default max concurrent writes (not critical path, keep low)
pub const DEFAULT_SSD_WRITE_INFLIGHT: usize = 2;

/// Default max concurrent prefetches
pub const DEFAULT_SSD_PREFETCH_INFLIGHT: usize = 16;

/// Upper bound for one pinned-pool allocation while staging an SSD prefetch.
/// Large contiguous requests can otherwise force disproportionate LRU reclaim
/// from a fragmented pool.
const SSD_PREFETCH_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// Result of a single prefetch I/O.
type SinglePrefetchResult = (
    StateKey,
    SsdIndexEntry,
    Option<Arc<SealedBlock>>,
    f64,
    u64,
    Arc<BatchContext>,
);

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the SSD cache (logical ring).
///
/// Supports one or more cache directories. When multiple paths are provided,
/// cache shards are distributed across them in round-robin order so that I/O
/// is balanced across independent devices.
#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
    /// Cache data directories. Each path receives a subset of the total shards.
    pub cache_paths: Vec<PathBuf>,
    /// Total logical capacity of the cache (bytes).
    pub capacity_bytes: u64,
    /// Number of cache files per path. 1 keeps the existing single-file SSD layout
    /// when only one path is configured. With multiple paths each path receives this
    /// many shards so that every device is utilised.
    pub shards: NonZeroUsize,
    /// Max pending write batches. New sealed blocks are dropped if the queue is full.
    pub write_queue_depth: usize,
    /// Max pending prefetch batches (limits read tail latency).
    pub prefetch_queue_depth: usize,
    /// Max concurrent block writes (not critical path, keep low).
    pub write_inflight: usize,
    /// Max concurrent block prefetches.
    pub prefetch_inflight: usize,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_paths: vec![PathBuf::from("/tmp/orbitkv-ssd-cache/cache.bin")],
            capacity_bytes: 512 * 1024 * 1024 * 1024, // 512GB
            shards: NonZeroUsize::new(1).unwrap(),
            write_queue_depth: DEFAULT_SSD_WRITE_QUEUE_DEPTH,
            prefetch_queue_depth: DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
            write_inflight: DEFAULT_SSD_WRITE_INFLIGHT,
            prefetch_inflight: DEFAULT_SSD_PREFETCH_INFLIGHT,
        }
    }
}

// ============================================================================
// Types for SSD operations
// ============================================================================

/// Metadata for a block stored in SSD cache
#[derive(Clone)]
pub(super) struct SsdIndexEntry {
    /// Cache file shard containing this entry.
    pub shard_id: usize,
    /// Logical offset in the ring buffer (monotonically increasing)
    pub begin: u64,
    /// Block size in bytes
    pub len: u64,
    /// Physical file offset for IO
    pub file_offset: u64,
    /// Per-slot metadata for rebuilding SealedBlock
    pub slots: Vec<SlotMeta>,
}

/// State of an SSD index entry (two-phase commit)
#[derive(Clone)]
pub(super) enum SsdEntryState {
    /// IO in progress, not yet readable
    Writing(SsdIndexEntry),
    /// IO completed, readable
    Committed(SsdIndexEntry),
}

impl SsdEntryState {
    #[inline]
    fn entry(&self) -> &SsdIndexEntry {
        match self {
            Self::Writing(e) | Self::Committed(e) => e,
        }
    }
}

struct SsdShardRing {
    capacity: u64,
    head: u64,
    tail: u64,
    order: VecDeque<StateKey>,
}

/// SSD ring buffer: unified state for space allocation + block index.
///
/// Combines head/tail pointers with FIFO index. Maintains insertion order
/// for O(k) tail pruning while preserving O(1) lookup via HashMap.
///
/// Two-phase commit: prepare_batch inserts Writing state, commit transitions
/// to Committed (or removes on failure). Only Committed entries are readable.
pub(super) struct SsdRingBuffer {
    /// Per-file ring state.
    shards: Vec<SsdShardRing>,
    /// Round-robin cursor for selecting the next write shard.
    next_shard: usize,
    /// Fast lookup: key -> state (Writing or Committed)
    entries: HashMap<StateKey, SsdEntryState>,
}

impl SsdRingBuffer {
    /// Create a new ring buffer with given capacity.
    pub(super) fn new(capacity: u64) -> Self {
        Self::new_sharded(vec![capacity])
    }

    pub(super) fn new_sharded(shard_capacities: Vec<u64>) -> Self {
        assert!(
            !shard_capacities.is_empty(),
            "SSD cache needs at least one shard"
        );
        Self {
            shards: shard_capacities
                .into_iter()
                .map(|capacity| SsdShardRing {
                    capacity,
                    head: 0,
                    tail: 0,
                    order: VecDeque::new(),
                })
                .collect(),
            next_shard: 0,
            entries: HashMap::new(),
        }
    }

    /// Lookup a Committed entry by key, returning None if Writing or expired.
    pub(super) fn get(&self, key: &StateKey) -> Option<&SsdIndexEntry> {
        match self.entries.get(key) {
            Some(SsdEntryState::Committed(e)) if self.is_offset_valid(e) => Some(e),
            _ => None,
        }
    }

    /// Check if a logical offset is still valid (not yet overwritten).
    #[inline]
    pub(super) fn is_offset_valid(&self, entry: &SsdIndexEntry) -> bool {
        self.shards
            .get(entry.shard_id)
            .is_some_and(|shard| entry.begin >= shard.tail)
    }

    /// Allocate contiguous space for a batch and advance tail.
    /// Returns (begin, file_offset). Skips wrap-around gap if needed.
    fn allocate_contiguous(&mut self, shard_id: usize, size: u64) -> Option<(u64, u64)> {
        let shard = &mut self.shards[shard_id];
        if size > shard.capacity {
            return None;
        }
        let phys = shard.head % shard.capacity;
        let space_until_end = shard.capacity - phys;
        if size > space_until_end {
            // Skip to next wrap point
            shard.head += space_until_end;
        }
        let begin = shard.head;
        shard.head += size;

        // Advance tail to maintain invariant: head - tail <= capacity
        let new_tail = shard.head.saturating_sub(shard.capacity);
        let capacity = shard.capacity;
        self.advance_tail(shard_id, new_tail);

        Some((begin, begin % capacity))
    }

    /// Advance tail and prune expired entries (FIFO order).
    /// Handles both Writing and Committed states uniformly.
    fn advance_tail(&mut self, shard_id: usize, new_tail: u64) {
        if new_tail <= self.shards[shard_id].tail {
            return;
        }
        self.shards[shard_id].tail = new_tail;

        while let Some(key) = self.shards[shard_id].order.front() {
            match self.entries.get(key) {
                Some(state) if state.entry().begin >= new_tail => break,
                _ => {
                    let key = self.shards[shard_id]
                        .order
                        .pop_front()
                        .expect("front key exists");
                    self.entries.remove(&key);
                }
            }
        }
    }

    /// Commit a write: success=true transitions Writing→Committed, success=false removes.
    /// Returns false if entry was already expired or missing.
    pub(super) fn commit(&mut self, key: &StateKey, success: bool) -> bool {
        let Some(state) = self.entries.get(key) else {
            // Already removed by advance_tail or previous abort
            return false;
        };

        // Only process Writing state
        let entry = match state {
            SsdEntryState::Writing(e) => e,
            SsdEntryState::Committed(_) => {
                warn!("SSD commit: key already committed, ignoring");
                return true;
            }
        };

        // Check if expired (eviction faster than write)
        if !self.is_offset_valid(entry) {
            warn!("SSD commit: entry expired before IO completed");
            self.entries.remove(key);
            return false;
        }

        if success {
            // Writing → Committed
            let entry = entry.clone();
            self.entries
                .insert(key.clone(), SsdEntryState::Committed(entry));
            true
        } else {
            // Write failed, remove entry (order will be cleaned by advance_tail)
            self.entries.remove(key);
            false
        }
    }

    /// Prepare a batch for writing: filter, allocate space, advance tail, insert Writing.
    /// Returns list of blocks to write with their allocated offsets.
    pub(super) fn prepare_batch(
        &mut self,
        candidates: Vec<(StateKey, Arc<SealedBlock>)>,
    ) -> PreparedBatch {
        // 1. Filter: skip keys that already exist (Writing or Committed)
        let to_write: Vec<_> = candidates
            .into_iter()
            .filter(|(k, _)| !self.entries.contains_key(k))
            .collect();

        if to_write.is_empty() {
            return PreparedBatch::empty();
        }

        // 2. Insert Writing state and build WriteInfo
        let mut writes = Vec::with_capacity(to_write.len());
        for (key, block) in to_write {
            let size = block.memory_footprint();
            let shard_id = self.next_shard;
            let Some((begin, file_offset)) = self.allocate_contiguous(shard_id, size) else {
                warn!(
                    "SSD cache: dropping block {:?}, size {} exceeds shard capacity {}",
                    key, size, self.shards[shard_id].capacity
                );
                continue;
            };
            self.next_shard = (self.next_shard + 1) % self.shards.len();
            let slot_numas = block.slot_numas();
            assert_eq!(
                slot_numas.len(),
                block.slots().len(),
                "slot_numas must cover every slot",
            );
            let slots: Vec<SlotMeta> = block
                .slots()
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let segment_sizes: SmallVec<[u64; 2]> = (0..s.num_segments())
                        .map(|idx| s.segment_size(idx).unwrap() as u64)
                        .collect();
                    SlotMeta::new(segment_sizes, slot_numas[i])
                })
                .collect();
            let entry = SsdIndexEntry {
                shard_id,
                begin,
                len: size,
                file_offset,
                slots,
            };

            // Insert Writing state
            self.entries
                .insert(key.clone(), SsdEntryState::Writing(entry.clone()));
            self.shards[shard_id].order.push_back(key.clone());

            writes.push(WriteInfo { key, block, entry });
        }

        PreparedBatch { writes }
    }
}

impl Default for SsdRingBuffer {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Info for a single block write within a batch.
pub(super) struct WriteInfo {
    pub key: StateKey,
    pub block: Arc<SealedBlock>,
    pub entry: SsdIndexEntry,
}

/// Prepared batch ready for IO.
pub(super) struct PreparedBatch {
    pub writes: Vec<WriteInfo>,
}

impl PreparedBatch {
    pub(super) fn empty() -> Self {
        Self { writes: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

/// Batch of sealed blocks to write to SSD
pub(super) struct SsdWriteBatch {
    pub blocks: Vec<(StateKey, Weak<SealedBlock>)>,
}

/// Commands sent to the SSD writer task.
pub(super) enum SsdWriteCommand {
    Write(SsdWriteBatch),
    Flush(tokio::sync::oneshot::Sender<()>),
}

/// Request to prefetch a block from SSD (metadata only, allocation done in worker)
pub(super) struct PrefetchRequest {
    pub key: StateKey,
    pub entry: SsdIndexEntry,
}

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub(super) struct PrefetchBatch {
    pub requests: Vec<PrefetchRequest>,
    pub done_tx: oneshot::Sender<super::PrefetchResult>,
}

/// Shared context for a batch of prefetch operations.
/// Collects successful blocks and delivers them as a batch when all reads finish.
pub(super) struct BatchContext {
    results: Mutex<super::PrefetchResult>,
    remaining: AtomicUsize,
    done_tx: Mutex<Option<oneshot::Sender<super::PrefetchResult>>>,
}

impl BatchContext {
    fn new(count: usize, done_tx: oneshot::Sender<super::PrefetchResult>) -> Self {
        Self {
            results: Mutex::new(Vec::with_capacity(count)),
            remaining: AtomicUsize::new(count),
            done_tx: Mutex::new(Some(done_tx)),
        }
    }

    fn complete_one(&self, key: StateKey, block: Option<Arc<SealedBlock>>) {
        if let Some(block) = block {
            self.results.lock().push((key, block));
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(tx) = self.done_tx.lock().take()
        {
            let results = std::mem::take(&mut *self.results.lock());
            let _ = tx.send(results);
        }
    }
}

/// Per-slot allocation reference: which allocation and offset within it.
struct SlotAlloc {
    allocation: Arc<PinnedAllocation>,
    offset: usize,
}

/// Internal: single block prefetch task with per-slot allocated memory.
struct PrefetchTask {
    key: StateKey,
    entry: SsdIndexEntry,
    /// One per slot (parallel to `entry.slots`), each from the correct NUMA pool.
    slot_allocs: Vec<SlotAlloc>,
    /// Shared batch context: per-block callback + completion counter.
    ctx: Arc<BatchContext>,
}

/// Internal: single block write task
struct WriteTask {
    key: StateKey,
    block: Arc<SealedBlock>,
    entry: SsdIndexEntry,
}

/// Result of a single write operation: (key, success, duration_secs, block_size)
type WriteResult = (StateKey, bool, f64, u64);

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// SSD writer task: receives batches of sealed blocks and writes them.
pub(super) async fn ssd_writer_loop(
    store: Weak<SsdBackingStore>,
    mut rx: tokio::sync::mpsc::Receiver<SsdWriteCommand>,
    io: Arc<UringIoEngine>,
    write_inflight: usize,
) {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;

    type WriteFuture = Pin<Box<dyn Future<Output = WriteResult> + Send>>;

    let metrics = core_metrics();
    let max_inflight = write_inflight.max(1);

    let mut pending: VecDeque<WriteTask> = VecDeque::new();
    let mut inflight: FuturesOrdered<WriteFuture> = FuturesOrdered::new();
    let mut flush_waiters: Vec<tokio::sync::oneshot::Sender<()>> = Vec::new();

    loop {
        // If a flush is pending and all work is drained, fire it.
        if !flush_waiters.is_empty() && pending.is_empty() && inflight.is_empty() {
            for tx in flush_waiters.drain(..) {
                let _ = tx.send(());
            }
        }

        tokio::select! {
            biased;

            // Priority 1: Complete writes
            Some((key, success, duration_secs, block_size)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_write_inflight.add(-1, &[]);

                // Commit result to ring buffer (Writing→Committed or remove)
                if let Some(s) = store.upgrade() {
                    s.commit_write(&key, success);
                }

                if success {
                    metrics.ssd_write_bytes.add(block_size, &[]);
                    let throughput = block_size as f64 / duration_secs;
                    metrics.ssd_write_throughput_bytes_per_second.record(throughput, &[]);
                } else {
                    metrics.ssd_write_failures.add(1, &[]);
                    warn!("SSD cache write failed for {:?}", key);
                }
            }

            // Priority 2: Submit pending writes if inflight has room
            _ = std::future::ready(()), if inflight.len() < max_inflight && !pending.is_empty() => {
                let task = pending.pop_front().unwrap();
                metrics.ssd_write_inflight.add(1, &[]);
                inflight.push_back(Box::pin(execute_write(task, io.clone())));
            }

            // Priority 3: Receive new command
            cmd = rx.recv(), if pending.is_empty() && flush_waiters.is_empty() => {
                match cmd {
                    Some(SsdWriteCommand::Write(b)) => {
                        // Dequeue metric
                        metrics.ssd_write_queue_pending.add(-(b.blocks.len() as i64), &[]);

                        // Upgrade weak refs (per-block, not prefix semantics)
                        let candidates: Vec<_> = b.blocks
                            .into_iter()
                            .filter_map(|(k, w)| w.upgrade().map(|b| (k, b)))
                            .collect();

                        if candidates.is_empty() {
                            continue;
                        }

                        // Prepare batch: filter + allocate + insert Writing
                        let Some(s) = store.upgrade() else { continue };
                        let prepared = s.prepare_batch(candidates);

                        if prepared.is_empty() {
                            continue;
                        }

                        // Convert to WriteTask
                        for w in prepared.writes {
                            pending.push_back(WriteTask {
                                key: w.key,
                                block: w.block,
                                entry: w.entry,
                            });
                        }
                    }
                    Some(SsdWriteCommand::Flush(tx)) => {
                        flush_waiters.push(tx);
                    }
                    None => break,
                }
            }
        }
    }

    // Drain remaining inflight writes
    drain_inflight(&store, metrics, &mut inflight).await;

    // Fire any remaining flush waiters
    for tx in flush_waiters.drain(..) {
        let _ = tx.send(());
    }

    debug!("SSD writer task exiting");
}

async fn drain_inflight(
    store: &Weak<SsdBackingStore>,
    metrics: &crate::metrics::CoreMetrics,
    inflight: &mut FuturesOrdered<
        std::pin::Pin<Box<dyn std::future::Future<Output = WriteResult> + Send>>,
    >,
) {
    while let Some((key, success, duration_secs, block_size)) = inflight.next().await {
        metrics.ssd_write_inflight.add(-1, &[]);

        if let Some(s) = store.upgrade() {
            s.commit_write(&key, success);
        }

        if success {
            metrics.ssd_write_bytes.add(block_size, &[]);
            let throughput = block_size as f64 / duration_secs;
            metrics
                .ssd_write_throughput_bytes_per_second
                .record(throughput, &[]);
        } else {
            metrics.ssd_write_failures.add(1, &[]);
            warn!("SSD cache write failed for {:?}", key);
        }
    }
}

/// Execute a single block write to SSD.
async fn execute_write(task: WriteTask, io: Arc<UringIoEngine>) -> WriteResult {
    let start = Instant::now();
    let key = task.key;
    let block_size = task.block.memory_footprint();

    let result = write_block_to_ssd(
        &io,
        task.entry.shard_id,
        task.entry.file_offset,
        &task.block,
    )
    .await;

    let duration_secs = start.elapsed().as_secs_f64();
    core_metrics()
        .ssd_write_duration_seconds
        .record(duration_secs, &[]);
    (key, result.is_ok(), duration_secs, block_size)
}

/// Write a sealed block to SSD file using writev.
///
/// Uses vectorized I/O to write all slots in a single syscall, reducing overhead
/// compared to writing each slot separately.
async fn write_block_to_ssd(
    io: &UringIoEngine,
    shard_id: usize,
    offset: u64,
    block: &SealedBlock,
) -> std::io::Result<()> {
    // Build iovecs from RawBlock segments (layout-agnostic)
    let rx = {
        let iovecs: Vec<_> = block
            .slots()
            .iter()
            .flat_map(seal_offload::write_iovecs)
            .collect();

        io.writev_at_async(shard_id, iovecs, offset)?
    };

    rx.await
        .map_err(|_| std::io::Error::other("writev recv failed"))??;

    Ok(())
}

// ============================================================================
// SSD Prefetch Pipeline (Dispatcher + Worker)
// ============================================================================

/// SSD prefetch entry point. Spawns dispatcher + worker pipeline internally.
pub(super) async fn ssd_prefetch_loop(
    store: Weak<SsdBackingStore>,
    rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    io: Arc<UringIoEngine>,
    prefetch_inflight: usize,
) {
    let prefetch_inflight = prefetch_inflight.max(1);

    // Bounded channel: capacity = max inflight tasks
    let (task_tx, task_rx) = tokio::sync::mpsc::channel(prefetch_inflight);

    // Spawn dispatcher and worker
    let dispatcher = tokio::spawn(ssd_prefetch_dispatcher(store.clone(), rx, task_tx));
    let worker = tokio::spawn(ssd_prefetch_worker(store, task_rx, io, prefetch_inflight));

    // Wait for both to complete
    let _ = dispatcher.await;
    let _ = worker.await;

    debug!("SSD prefetch pipeline exiting");
}

/// Dispatcher: receives batches, allocates per-slot memory grouped by NUMA,
/// then splits into block-level tasks.
async fn ssd_prefetch_dispatcher(
    store: Weak<SsdBackingStore>,
    mut batch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    task_tx: tokio::sync::mpsc::Sender<PrefetchTask>,
) {
    while let Some(batch) = batch_rx.recv().await {
        if batch.requests.is_empty() {
            let _ = batch.done_tx.send(Vec::new());
            continue;
        }

        let Some(s) = store.upgrade() else { break };
        if !dispatch_prefetch_batch(&s, &task_tx, batch).await {
            return;
        }
    }

    debug!("SSD prefetch dispatcher exiting");
}

/// Slot reference for NUMA grouping during allocation.
struct SlotRef {
    block_idx: usize,
    slot_idx: usize,
    size: u64,
}

struct SlotPlacement {
    block_idx: usize,
    slot_idx: usize,
    offset: u64,
}

struct PrefetchChunk {
    capacity: u64,
    size: u64,
    slots: Vec<SlotPlacement>,
}

/// Group all slots across all blocks by NUMA node.
///
/// Returns a map from NUMA key (None = global/unknown) to its slot references.
fn group_slots_by_numa(
    is_numa: bool,
    requests: &[PrefetchRequest],
) -> HashMap<Option<NumaNode>, Vec<SlotRef>> {
    let mut groups: HashMap<Option<NumaNode>, Vec<SlotRef>> = HashMap::new();
    for (block_idx, req) in requests.iter().enumerate() {
        for (slot_idx, meta) in req.entry.slots.iter().enumerate() {
            let numa_key = if is_numa {
                let numa = meta.numa_node;
                if numa.is_unknown() { None } else { Some(numa) }
            } else {
                None
            };
            groups.entry(numa_key).or_default().push(SlotRef {
                block_idx,
                slot_idx,
                size: meta.total_size(),
            });
        }
    }
    groups
}

fn chunk_slot_refs(refs: &[SlotRef], chunk_bytes: u64) -> Result<Vec<PrefetchChunk>, String> {
    assert!(chunk_bytes > 0, "SSD prefetch chunk size must be non-zero");

    let mut remaining = refs.iter().try_fold(0u64, |total, slot| {
        total
            .checked_add(slot.size)
            .ok_or_else(|| "SSD prefetch allocation size overflow".to_string())
    })?;
    let mut chunks = Vec::new();
    let mut current: Option<PrefetchChunk> = None;

    for slot in refs {
        let needs_new_chunk = current.as_ref().is_none_or(|chunk| {
            chunk
                .size
                .checked_add(slot.size)
                .is_none_or(|end| end > chunk.capacity)
        });

        if needs_new_chunk {
            if let Some(chunk) = current.take() {
                chunks.push(chunk);
            }
            current = Some(PrefetchChunk {
                capacity: remaining.min(chunk_bytes).max(slot.size),
                size: 0,
                slots: Vec::new(),
            });
        }

        let chunk = current
            .as_mut()
            .expect("non-empty slot list must have an active prefetch chunk");
        chunk.slots.push(SlotPlacement {
            block_idx: slot.block_idx,
            slot_idx: slot.slot_idx,
            offset: chunk.size,
        });
        chunk.size = chunk
            .size
            .checked_add(slot.size)
            .expect("new SSD prefetch chunk must fit its first slot");
        remaining = remaining
            .checked_sub(slot.size)
            .expect("SSD prefetch remaining bytes must cover every slot");
    }

    if let Some(chunk) = current {
        chunks.push(chunk);
    }
    Ok(chunks)
}

/// Allocate per-slot memory grouped by NUMA, build PrefetchTasks, and enqueue.
/// Returns false if the task channel is closed (should exit).
async fn dispatch_prefetch_batch(
    store: &SsdBackingStore,
    task_tx: &tokio::sync::mpsc::Sender<PrefetchTask>,
    batch: PrefetchBatch,
) -> bool {
    let PrefetchBatch { requests, done_tx } = batch;

    // 1. Group all slots across all blocks by NUMA node
    let numa_groups = group_slots_by_numa(store.is_numa(), &requests);

    // 2. Allocate bounded chunks per NUMA group, assign per-slot offsets.
    //    A failure still fails the whole batch because every requested block
    //    must have all of its slots before it can be rebuilt.
    let mut slot_allocs: Vec<Vec<Option<SlotAlloc>>> = requests
        .iter()
        .map(|r| (0..r.entry.slots.len()).map(|_| None).collect())
        .collect();

    for (numa_node, refs) in &numa_groups {
        let chunks = match chunk_slot_refs(refs, SSD_PREFETCH_CHUNK_BYTES) {
            Ok(chunks) => chunks,
            Err(err) => {
                warn!("SSD prefetch dispatcher: {err}, failing entire batch");
                let _ = done_tx.send(Vec::new());
                return true;
            }
        };
        for chunk in chunks {
            let allocation = match store.allocate_prefetch(chunk.size, *numa_node) {
                Some(alloc) => alloc,
                None => {
                    warn!(
                        "SSD prefetch dispatcher: alloc failed for {} bytes numa={:?}, failing entire batch",
                        chunk.size, numa_node
                    );
                    let _ = done_tx.send(Vec::new());
                    return true;
                }
            };
            for slot in chunk.slots {
                let offset =
                    usize::try_from(slot.offset).expect("SSD prefetch chunk offset must fit usize");
                slot_allocs[slot.block_idx][slot.slot_idx] = Some(SlotAlloc {
                    allocation: Arc::clone(&allocation),
                    offset,
                });
            }
        }
    }

    // 3. Build BatchContext + PrefetchTasks and enqueue
    let ctx = Arc::new(BatchContext::new(requests.len(), done_tx));

    let mut iter = requests.into_iter().enumerate();
    while let Some((block_idx, req)) = iter.next() {
        let allocs: Vec<SlotAlloc> = slot_allocs[block_idx]
            .drain(..)
            .map(|opt| opt.expect("all slots must have allocations"))
            .collect();

        let task = PrefetchTask {
            key: req.key,
            entry: req.entry,
            slot_allocs: allocs,
            ctx: Arc::clone(&ctx),
        };

        if let Err(err) = task_tx.send(task).await {
            debug!("SSD prefetch dispatcher: worker channel closed");
            let task = err.0;
            ctx.complete_one(task.key, None);
            for (_, req) in iter {
                ctx.complete_one(req.key, None);
            }
            return false;
        }
    }

    true
}

/// Worker: maintains FuturesUnordered with max_inflight concurrent I/O operations.
async fn ssd_prefetch_worker(
    store: Weak<SsdBackingStore>,
    mut task_rx: tokio::sync::mpsc::Receiver<PrefetchTask>,
    io: Arc<UringIoEngine>,
    max_inflight: usize,
) {
    use std::future::Future;
    use std::pin::Pin;

    type PrefetchFuture = Pin<Box<dyn Future<Output = SinglePrefetchResult> + Send>>;

    let metrics = core_metrics();
    let mut inflight: FuturesUnordered<PrefetchFuture> = FuturesUnordered::new();

    loop {
        tokio::select! {
            biased;

            // Complete finished tasks first (priority)
            Some((key, entry, result, duration_secs, block_size, ctx)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_prefetch_inflight.add(-1, &[]);

                // Validate data wasn't overwritten during read
                let valid = store.upgrade().is_some_and(|s| s.is_offset_valid(&entry));
                let result = if result.is_some() && !valid {
                    warn!("SSD prefetch: data overwritten during read, discarding");
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                } else if result.is_some() {
                    metrics.ssd_prefetch_success.add(1, &[]);
                    metrics.ssd_prefetch_bytes.add(block_size, &[]);
                    let throughput = block_size as f64 / duration_secs;
                    metrics.ssd_prefetch_throughput_bytes_per_second.record(throughput, &[]);
                    result
                } else {
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                };
                ctx.complete_one(key, result);
            }

            // Accept new task if below limit
            task = task_rx.recv(), if inflight.len() < max_inflight => {
                match task {
                    Some(t) => {
                        metrics.ssd_prefetch_inflight.add(1, &[]);
                        inflight.push(Box::pin(execute_prefetch(t, io.clone())));
                    }
                    None => {
                        // Channel closed, drain remaining
                        break;
                    }
                }
            }
        }
    }

    // Drain remaining inflight tasks
    while let Some((key, entry, result, duration_secs, block_size, ctx)) = inflight.next().await {
        metrics.ssd_prefetch_inflight.add(-1, &[]);

        let valid = store.upgrade().is_some_and(|s| s.is_offset_valid(&entry));
        let result = if result.is_some() && !valid {
            metrics.ssd_prefetch_failures.add(1, &[]);
            None
        } else if result.is_some() {
            metrics.ssd_prefetch_success.add(1, &[]);
            metrics.ssd_prefetch_bytes.add(block_size, &[]);
            let throughput = block_size as f64 / duration_secs;
            metrics
                .ssd_prefetch_throughput_bytes_per_second
                .record(throughput, &[]);
            result
        } else {
            metrics.ssd_prefetch_failures.add(1, &[]);
            None
        };
        ctx.complete_one(key, result);
    }

    debug!("SSD prefetch worker exiting");
}

/// Execute a single prefetch operation.
async fn execute_prefetch(task: PrefetchTask, io: Arc<UringIoEngine>) -> SinglePrefetchResult {
    let start = Instant::now();
    let duration_secs = || start.elapsed().as_secs_f64();

    let key = task.key;
    let block_size = task.entry.len;
    let ctx = task.ctx;

    // Build iovecs from per-slot allocations
    let read_result = {
        let iovecs: Vec<_> = task
            .entry
            .slots
            .iter()
            .zip(&task.slot_allocs)
            .flat_map(|(meta, alloc)| {
                let base_ptr = alloc.allocation.as_ptr() as *mut u8;
                // SAFETY: each allocation is sized to fit its NUMA group's slots
                unsafe { seal_offload::read_iovecs(meta, base_ptr, alloc.offset) }
            })
            .collect();

        io.readv_at_async(task.entry.shard_id, iovecs, task.entry.file_offset)
    };

    #[cfg(feature = "test-hooks")]
    crate::test_faults::pause("ssd").await;

    // Await IO result and rebuild block
    let expected_len = task.entry.len as usize;
    let block = match read_result {
        Ok(rx) => match rx.await {
            Ok(Ok(bytes_read)) if bytes_read == expected_len => Some(Arc::new(
                rebuild_sealed_block_per_slot(task.slot_allocs, &task.entry.slots),
            )),
            Ok(Ok(n)) => {
                warn!("SSD prefetch: short read {} of {} bytes", n, expected_len);
                None
            }
            Ok(Err(e)) => {
                warn!("SSD prefetch: read error: {}", e);
                None
            }
            Err(_) => {
                warn!("SSD prefetch: read channel closed");
                None
            }
        },
        Err(e) => {
            warn!("SSD prefetch: failed to submit read: {}", e);
            None
        }
    };

    (key, task.entry, block, duration_secs(), block_size, ctx)
}

// ============================================================================
// Block Rebuilding
// ============================================================================

/// Rebuild a SealedBlock from per-slot allocations (consumed).
///
/// Each slot may reside in a different NUMA-local allocation. Takes ownership
/// of `SlotAlloc`s to move (not clone) the `Arc<PinnedAllocation>` references.
fn rebuild_sealed_block_per_slot(
    slot_allocs: Vec<SlotAlloc>,
    slot_metas: &[SlotMeta],
) -> SealedBlock {
    let slots: Vec<(RawBlock, NumaNode)> = slot_metas
        .iter()
        .zip(slot_allocs)
        .map(|(meta, alloc)| {
            let raw = unsafe {
                seal_offload::reconstruct_raw_block(meta, alloc.allocation, alloc.offset)
            };
            (raw, meta.numa_node)
        })
        .collect();

    SealedBlock::from_slots(slots)
}

#[cfg(test)]
#[path = "../../tests/unit/backing/ssd_cache.rs"]
mod tests;
