use super::{
    SsdBackingStore,
    codec::{self, Encoding},
    index::SsdIndexEntry,
    uring::UringIoEngine,
};
use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
use crate::metrics::core_metrics;
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, warn};
use mea::oneshot;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Result of a single prefetch I/O.
type SinglePrefetchResult = (
    StateKey,
    SsdIndexEntry,
    Option<Arc<SealedBlock>>,
    f64,
    u64,
    Arc<BatchContext>,
);

/// Request to prefetch a block from SSD (metadata only, allocation done in worker)
pub(super) struct PrefetchRequest {
    pub key: StateKey,
    pub entry: SsdIndexEntry,
}

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub(super) struct PrefetchBatch {
    pub requests: Vec<PrefetchRequest>,
    pub done_tx: oneshot::Sender<crate::backing::PrefetchResult>,
}

/// Shared context for a batch of prefetch operations.
/// Collects successful blocks and delivers them as a batch when all reads finish.
pub(super) struct BatchContext {
    results: Mutex<crate::backing::PrefetchResult>,
    remaining: AtomicUsize,
    done_tx: Mutex<Option<oneshot::Sender<crate::backing::PrefetchResult>>>,
}

impl BatchContext {
    fn new(count: usize, done_tx: oneshot::Sender<crate::backing::PrefetchResult>) -> Self {
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

/// Internal: single block prefetch task with per-slot allocated memory.
struct PrefetchTask {
    key: StateKey,
    entry: SsdIndexEntry,
    /// One per slot (parallel to `entry.slots`), each from the correct NUMA pool.
    slots: Vec<RawBlock>,
    /// Shared batch context: per-block callback + completion counter.
    ctx: Arc<BatchContext>,
    store: Arc<SsdBackingStore>,
}

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

/// Dispatcher: receives batches, allocates page segments on their NUMA node,
/// then submits block-level read tasks.
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

/// Allocate each stored segment independently, then enqueue block reads.
/// Read and write allocations use the same page/segment lifetime and sizes;
/// a surviving prefix cannot pin unrelated pages from a larger batch.
async fn dispatch_prefetch_batch(
    store: &Arc<SsdBackingStore>,
    task_tx: &tokio::sync::mpsc::Sender<PrefetchTask>,
    batch: PrefetchBatch,
) -> bool {
    let PrefetchBatch { requests, done_tx } = batch;
    let mut block_slots = Vec::with_capacity(requests.len());
    for req in &requests {
        let mut slots = Vec::with_capacity(req.entry.slots.len());
        for meta in &req.entry.slots {
            let numa_node =
                (store.is_numa() && !meta.numa_node.is_unknown()).then_some(meta.numa_node);
            let mut segments = Vec::with_capacity(meta.segment_sizes.len());
            for &size in &meta.segment_sizes {
                let Some(allocation) = store.allocate_prefetch(size, numa_node) else {
                    warn!(
                        "SSD prefetch dispatcher: alloc failed for {size} bytes numa={numa_node:?}, failing entire batch"
                    );
                    let _ = done_tx.send(Vec::new());
                    return true;
                };
                segments.push(Segment::new(
                    allocation.mapped_ptr().host(),
                    size as usize,
                    allocation,
                ));
            }
            slots.push(RawBlock::new(segments));
        }
        block_slots.push(slots);
    }

    let ctx = Arc::new(BatchContext::new(requests.len(), done_tx));
    let mut iter = requests.into_iter().zip(block_slots);
    while let Some((req, slots)) = iter.next() {
        let task = PrefetchTask {
            key: req.key,
            entry: req.entry,
            slots,
            ctx: Arc::clone(&ctx),
            store: Arc::clone(store),
        };

        if let Err(err) = task_tx.send(task).await {
            debug!("SSD prefetch dispatcher: worker channel closed");
            let task = err.0;
            ctx.complete_one(task.key, None);
            for (req, _) in iter {
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
    let block_size: u64 = task
        .entry
        .slots
        .iter()
        .map(crate::SlotMeta::total_size)
        .sum();
    let ctx = task.ctx;

    let mut encoded = match &task.entry.encoding {
        Encoding::Raw => None,
        Encoding::Lz4V1(_) => {
            let buffer = match &task.store.codec {
                Some(codec) => codec.read_buffer(task.entry.len as usize).await.ok(),
                None => None,
            };
            let Some(buffer) = buffer else {
                return (key, task.entry, None, duration_secs(), block_size, ctx);
            };
            Some(buffer)
        }
    };
    let expected_len = encoded
        .as_ref()
        .map_or(block_size as usize, |buffer| buffer.len);
    let read_result = {
        let iovecs = match &encoded {
            Some(buffer) => vec![(buffer.ptr(), buffer.len)],
            None => task
                .slots
                .iter()
                .flat_map(|slot| {
                    slot.segment_iovecs()
                        .map(|(ptr, size)| (ptr.as_ptr(), size))
                })
                .collect(),
        };
        io.readv_at_async(task.entry.shard_id, iovecs, task.entry.file_offset)
    };

    #[cfg(feature = "test-hooks")]
    crate::test_faults::pause("ssd").await;

    let read_ok = match read_result {
        Ok(rx) => matches!(rx.await, Ok(Ok(bytes)) if bytes == expected_len),
        Err(_) => false,
    };
    let slots = if !read_ok {
        warn!("SSD prefetch: failed or short read for {key:?}");
        None
    } else if let Some(mut buffer) = encoded.take() {
        let encoding = task.entry.encoding.clone();
        let slots = task.slots;
        let decoded = tokio::task::spawn_blocking(move || {
            let mut segments: Vec<_> = slots
                .iter()
                .flat_map(|slot| slot.segment_iovecs())
                .map(|(ptr, len)| {
                    // SAFETY: fresh, disjoint allocations belong only to this job.
                    unsafe {
                        ptr.as_ptr().write_bytes(0, len);
                        std::slice::from_raw_parts_mut(ptr.as_ptr(), len)
                    }
                })
                .collect();
            codec::decode(&encoding, &mut buffer, &mut segments)?;
            Ok::<_, std::io::Error>(slots)
        })
        .await;
        match decoded {
            Ok(Ok(slots)) => Some(slots),
            Ok(Err(error)) => {
                task.store
                    .inner
                    .lock()
                    .ring
                    .invalidate_encoded(&key, &task.entry);
                warn!("SSD prefetch: invalid encoded object {key:?}: {error}");
                None
            }
            Err(_) => None,
        }
    } else {
        Some(task.slots)
    };
    let block = slots.map(|slots| {
        Arc::new(SealedBlock::from_slots(
            slots
                .into_iter()
                .zip(&task.entry.slots)
                .map(|(slot, meta)| (slot, meta.numa_node))
                .collect(),
        ))
    });
    (key, task.entry, block, duration_secs(), block_size, ctx)
}
