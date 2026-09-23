use super::{SsdBackingStore, index::SsdIndexEntry, uring::UringIoEngine};
use crate::block::{SealedBlock, StateKey};
use crate::metrics::core_metrics;
use futures::stream::{FuturesOrdered, StreamExt};
use log::{debug, warn};
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Batch of sealed blocks to write to SSD
pub(super) struct SsdWriteBatch {
    pub blocks: Vec<(StateKey, Weak<SealedBlock>)>,
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

                        // Prepare batch: filter + allocate + insert Writing
                        let Some(s) = store.upgrade() else { continue };
                        let prepared = s.prepare_batch(b.blocks);

                        if prepared.is_empty() {
                            continue;
                        }

                        // Convert to WriteTask
                        for w in prepared {
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
            .flat_map(|slot| {
                slot.segment_iovecs()
                    .map(|(ptr, size)| (ptr.as_ptr() as *const u8, size))
            })
            .collect();

        io.writev_at_async(shard_id, iovecs, offset)?
    };

    rx.await
        .map_err(|_| std::io::Error::other("writev recv failed"))??;

    Ok(())
}
