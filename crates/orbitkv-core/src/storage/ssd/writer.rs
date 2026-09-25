use super::{SsdStore, index::Encoding, uring::UringIoEngine};
use crate::block::{SealedBlock, StateKey};
use crate::cost::{Observation, Outcome};
use crate::metrics::core_metrics;
use futures::stream::{FuturesOrdered, StreamExt};
use log::{debug, warn};
use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Batch of sealed blocks to write to SSD
pub(super) struct SsdWriteBatch {
    pub blocks: Vec<(StateKey, Weak<SealedBlock>)>,
    pub observation: Observation,
}

struct WriteBatchObservation {
    remaining: usize,
    outcome: Outcome,
    observation: Observation,
}

fn complete_write_batch(batches: &mut VecDeque<WriteBatchObservation>, success: bool, bytes: u64) {
    let Some(batch) = batches.front_mut() else {
        return;
    };
    batch.remaining -= 1;
    if !success {
        if bytes != 0 {
            batch.outcome = Outcome::Failed;
        } else if batch.outcome == Outcome::Completed {
            batch.outcome = Outcome::Cancelled;
        }
    }
    if batch.remaining == 0
        && let Some(batch) = batches.pop_front()
    {
        // FuturesOrdered returns writes in submission order. Completion follows
        // every ring commit; the physical io_uring owner counts actual bytes.
        batch.observation.finish(batch.outcome, None);
    }
}

/// Commands sent to the SSD writer task.
pub(super) enum SsdWriteCommand {
    Write(SsdWriteBatch),
    Flush(tokio::sync::oneshot::Sender<()>),
}

/// Internal: single block write task
struct WriteTask {
    key: StateKey,
    block: Arc<SealedBlock>,
}

/// Result: key, success, elapsed seconds, physical bytes (zero for unreserved skips).
type WriteResult = (StateKey, bool, f64, u64);

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// SSD writer task: receives batches of sealed blocks and writes them.
pub(super) async fn ssd_writer_loop(
    store: Weak<SsdStore>,
    mut rx: tokio::sync::mpsc::Receiver<SsdWriteCommand>,
    io: Arc<UringIoEngine>,
    write_inflight: usize,
) {
    use std::future::Future;
    use std::pin::Pin;

    type WriteFuture = Pin<Box<dyn Future<Output = WriteResult> + Send>>;

    let metrics = core_metrics();
    let max_inflight = write_inflight.max(1);

    let mut pending: VecDeque<WriteTask> = VecDeque::new();
    let mut inflight: FuturesOrdered<WriteFuture> = FuturesOrdered::new();
    let mut batches: VecDeque<WriteBatchObservation> = VecDeque::new();
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
                } else if block_size != 0 {
                    metrics.ssd_write_failures.add(1, &[]);
                    warn!("SSD cache write failed for {:?}", key);
                }
                complete_write_batch(&mut batches, success, block_size);
            }

            // Priority 2: Submit pending writes if inflight has room
            _ = std::future::ready(()), if inflight.len() < max_inflight && !pending.is_empty() => {
                let task = pending.pop_front().unwrap();
                // Only the newest dequeued batch can still have pending tasks.
                // This composite starts service at worker admission; low-level
                // submission and CQE service are measured separately by uring.
                if let Some(batch) = batches.back_mut() {
                    batch.observation.submitted();
                }
                metrics.ssd_write_inflight.add(1, &[]);
                inflight.push_back(Box::pin(execute_write(task, store.clone(), io.clone())));
            }

            // Priority 3: Receive new command
            cmd = rx.recv(), if pending.is_empty() && flush_waiters.is_empty() => {
                match cmd {
                    Some(SsdWriteCommand::Write(mut b)) => {
                        // Dequeue metric
                        metrics.ssd_write_queue_pending.add(-(b.blocks.len() as i64), &[]);
                        b.observation.admitted();

                        let Some(s) = store.upgrade() else {
                            b.observation.finish(Outcome::Cancelled, None);
                            continue;
                        };
                        let mut outcome = Outcome::Completed;
                        for (key, weak) in b.blocks {
                            if let Some(block) = weak.upgrade() {
                                pending.push_back(WriteTask { key, block });
                            } else {
                                s.inner.lock().pending_writes.remove(&key);
                                outcome = Outcome::Cancelled;
                            }
                        }
                        if pending.is_empty() {
                            b.observation.finish(Outcome::Cancelled, None);
                        } else if crate::cost::enabled() {
                            batches.push_back(WriteBatchObservation {
                                remaining: pending.len(),
                                outcome,
                                observation: b.observation,
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
    drain_inflight(&store, metrics, &mut inflight, &mut batches).await;

    // Fire any remaining flush waiters
    for tx in flush_waiters.drain(..) {
        let _ = tx.send(());
    }

    debug!("SSD writer task exiting");
}

async fn drain_inflight(
    store: &Weak<SsdStore>,
    metrics: &crate::metrics::CoreMetrics,
    inflight: &mut FuturesOrdered<
        std::pin::Pin<Box<dyn std::future::Future<Output = WriteResult> + Send>>,
    >,
    batches: &mut VecDeque<WriteBatchObservation>,
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
        } else if block_size != 0 {
            metrics.ssd_write_failures.add(1, &[]);
            warn!("SSD cache write failed for {:?}", key);
        }
        complete_write_batch(batches, success, block_size);
    }
}

/// Reserve extents at the stored size; retain sealed buffers through I/O completion.
async fn execute_write(
    task: WriteTask,
    store: Weak<SsdStore>,
    io: Arc<UringIoEngine>,
) -> WriteResult {
    let start = Instant::now();
    let key = task.key;
    let Some(store) = store.upgrade() else {
        return (key, false, 0.0, 0);
    };
    let encoded = task.block.slots().iter().any(|s| s.encoding.is_some());
    let slots = task
        .block
        .slots()
        .iter()
        .zip(task.block.slot_numas())
        .map(|(slot, numa)| {
            let mut meta = crate::SlotMeta::new(
                slot.segment_iovecs().map(|(_, size)| size as u64).collect(),
                *numa,
            );
            meta.encoding = slot.encoding.clone();
            meta
        })
        .collect();
    let encoding = if encoded {
        Encoding::Encoded
    } else {
        Encoding::Raw
    };
    let entry = store.inner.lock().ring.reserve(&key, slots, encoding);
    let Some(entry) = entry else {
        return (key, false, start.elapsed().as_secs_f64(), 0);
    };
    let bytes = task.block.memory_footprint();
    let rx = {
        let iovecs = task
            .block
            .slots()
            .iter()
            .flat_map(|slot| {
                slot.segment_iovecs()
                    .map(|(ptr, len)| (ptr.as_ptr() as *const u8, len))
            })
            .collect();
        io.writev_at_async(entry.shard_id, iovecs, entry.file_offset)
    };
    let success = match rx {
        Ok(rx) => matches!(rx.await, Ok(Ok(written)) if written as u64 == bytes),
        Err(_) => false,
    };
    let duration = start.elapsed().as_secs_f64();
    core_metrics()
        .ssd_write_duration_seconds
        .record(duration, &[]);
    (key, success, duration, bytes)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/writer.rs"]
mod tests;
