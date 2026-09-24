use super::{SsdBackingStore, codec::Encoding, uring::UringIoEngine};
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
}

/// Result: key, success, elapsed seconds, physical bytes (zero for unreserved skips).
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
                } else if block_size != 0 {
                    metrics.ssd_write_failures.add(1, &[]);
                    warn!("SSD cache write failed for {:?}", key);
                }
            }

            // Priority 2: Submit pending writes if inflight has room
            _ = std::future::ready(()), if inflight.len() < max_inflight && !pending.is_empty() => {
                let task = pending.pop_front().unwrap();
                metrics.ssd_write_inflight.add(1, &[]);
                inflight.push_back(Box::pin(execute_write(task, store.clone(), io.clone())));
            }

            // Priority 3: Receive new command
            cmd = rx.recv(), if pending.is_empty() && flush_waiters.is_empty() => {
                match cmd {
                    Some(SsdWriteCommand::Write(b)) => {
                        // Dequeue metric
                        metrics.ssd_write_queue_pending.add(-(b.blocks.len() as i64), &[]);

                        let Some(s) = store.upgrade() else { continue };
                        for (key, weak) in b.blocks {
                            if let Some(block) = weak.upgrade() {
                                pending.push_back(WriteTask { key, block });
                            } else {
                                s.inner.lock().pending_writes.remove(&key);
                            }
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
        } else if block_size != 0 {
            metrics.ssd_write_failures.add(1, &[]);
            warn!("SSD cache write failed for {:?}", key);
        }
    }
}

/// Encoding runs outside the ring lock and Tokio's I/O executor. Extents are
/// reserved at their physical size, and all buffers survive I/O completion.
async fn execute_write(
    task: WriteTask,
    store: Weak<SsdBackingStore>,
    io: Arc<UringIoEngine>,
) -> WriteResult {
    let start = Instant::now();
    let key = task.key;
    let Some(store) = store.upgrade() else {
        return (key, false, 0.0, 0);
    };
    let logical = task.block.memory_footprint();
    let encoded = if let Some(codec) = &store.codec {
        let codec = Arc::clone(codec);
        let block = Arc::clone(&task.block);
        let alignment = store.alignment;
        tokio::task::spawn_blocking(move || {
            let segments: Vec<_> = block
                .slots()
                .iter()
                .flat_map(|slot| {
                    slot.segment_iovecs()
                        .map(move |(ptr, len)| (ptr, len, slot.storage_format))
                })
                .map(|(ptr, len, format)| {
                    // SAFETY: sealed host segments have completed DMA, initialized padding,
                    // and remain immutable under this block's ownership.
                    (
                        unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) },
                        format,
                    )
                })
                .collect();
            codec.encode(&segments, alignment)
        })
        .await
        .ok()
        .flatten()
    } else {
        None
    };
    let slots = task
        .block
        .slots()
        .iter()
        .zip(task.block.slot_numas())
        .map(|(slot, numa)| {
            crate::SlotMeta::new(
                slot.segment_iovecs().map(|(_, size)| size as u64).collect(),
                *numa,
            )
        })
        .collect();
    let encoding = encoded
        .as_ref()
        .map_or(Encoding::Raw, |(encoding, _)| encoding.clone());
    let entry = store.inner.lock().ring.reserve(&key, slots, encoding);
    let Some(entry) = entry else {
        return (key, false, start.elapsed().as_secs_f64(), 0);
    };
    let bytes = encoded
        .as_ref()
        .map_or(logical, |(_, buffer)| buffer.len as u64);
    let rx = {
        let iovecs = match &encoded {
            Some((_, buffer)) => vec![(buffer.ptr() as *const u8, buffer.len)],
            None => task
                .block
                .slots()
                .iter()
                .flat_map(|slot| {
                    slot.segment_iovecs()
                        .map(|(ptr, len)| (ptr.as_ptr() as *const u8, len))
                })
                .collect(),
        };
        io.writev_at_async(entry.shard_id, iovecs, entry.file_offset)
    };
    let success = match rx {
        Ok(rx) => matches!(rx.await, Ok(Ok(written)) if written as u64 == bytes),
        Err(_) => false,
    };
    if success && encoded.is_some() {
        let metrics = core_metrics();
        metrics.ssd_codec_bytes.add(
            logical,
            &[opentelemetry::KeyValue::new("representation", "logical")],
        );
        metrics.ssd_codec_bytes.add(
            bytes,
            &[opentelemetry::KeyValue::new("representation", "stored")],
        );
    }
    let duration = start.elapsed().as_secs_f64();
    core_metrics()
        .ssd_write_duration_seconds
        .record(duration, &[]);
    (key, success, duration, bytes)
}
