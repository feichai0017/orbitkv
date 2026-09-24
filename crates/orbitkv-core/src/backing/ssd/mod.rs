use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytesize::ByteSize;
use hashlink::LruCache;
use log::{debug, info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::block::{SealedBlock, StateKey};
use crate::memory::numa::NumaNode;
use crate::memory::pool::PinnedAllocation;
use crate::metrics::core_metrics;

mod config;
pub(crate) mod cufile;
mod files;
pub(crate) mod index;
mod reader;
mod uring;
mod writer;

use super::{AllocateFn, PrefetchResult};
pub(crate) use config::SSD_ALIGNMENT;
pub use config::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdBackend, SsdCacheConfig, SsdWritePolicy,
};
use cufile::CufileFile;
use index::{SsdIndexEntry, SsdRingBuffer};
use reader::{PrefetchBatch, PrefetchRequest, ssd_prefetch_loop};
use uring::{UringConfig, UringIoEngine};
use writer::{SsdWriteBatch, SsdWriteCommand, ssd_writer_loop};

/// Owns an immutable SSD source until the last query/GPU consumer releases it.
pub struct SsdReadLease {
    pub(crate) entry: SsdIndexEntry,
    key: StateKey,
    store: Arc<SsdBackingStore>,
}

impl SsdReadLease {
    pub(crate) fn file(&self) -> &Arc<CufileFile> {
        &self.store.cufile_files[self.entry.shard_id]
    }

    pub(crate) fn invalidate_encoded(&self) {
        self.store
            .inner
            .lock()
            .ring
            .invalidate_encoded(&self.key, &self.entry);
    }
}

impl Drop for SsdReadLease {
    fn drop(&mut self) {
        if self.entry.readers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.store
                .inner
                .lock()
                .ring
                .release_invalid(&self.key, &self.entry);
        }
        core_metrics()
            .ssd_read_pinned_bytes
            .add(-(self.entry.len as i64), &[]);
    }
}

/// An unpublished extent. Dropping a failed/unsubmitted write rolls it back;
/// the GPU worker commits only after every cuFile chunk has completed.
pub(crate) struct GpuWriteLease {
    pub(crate) entry: SsdIndexEntry,
    key: Option<StateKey>,
    store: Arc<SsdBackingStore>,
}

impl GpuWriteLease {
    pub(crate) fn file(&self) -> &Arc<CufileFile> {
        &self.store.cufile_files[self.entry.shard_id]
    }

    pub(crate) fn commit(mut self) {
        if let Some(key) = self.key.take() {
            self.store.commit_write(&key, true);
            core_metrics().ssd_write_bytes.add(self.entry.len, &[]);
        }
    }
}

impl Drop for GpuWriteLease {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            self.store.commit_write(&key, false);
            core_metrics().ssd_write_failures.add(1, &[]);
        }
        core_metrics().ssd_write_inflight.add(-1, &[]);
    }
}

struct SsdInner {
    ring: SsdRingBuffer,
    pending_writes: HashSet<StateKey>,
    reuse_history: LruCache<StateKey, ()>,
}

const REUSE_HISTORY_BLOCKS: usize = 16_384;

/// Shared admission state. Disabling new GPU I/O never revokes existing leases.
pub(crate) struct GpuIo {
    automatic: bool,
    enabled: AtomicBool,
}

impl GpuIo {
    pub(crate) fn available(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub(crate) fn failed(&self, error: &str) {
        if self.automatic && self.enabled.swap(false, Ordering::AcqRel) {
            core_metrics().ssd_backend_fallbacks.add(1, &[]);
            warn!("SSD auto backend selected uring for new operations: {error}");
        }
    }
}

impl SsdInner {
    fn admission_skip(
        &mut self,
        key: &StateKey,
        reused: bool,
        policy: SsdWritePolicy,
    ) -> Option<&'static str> {
        if self.ring.get(key).is_some() {
            return Some("resident");
        }
        if self.pending_writes.contains(key) {
            return Some("pending");
        }
        if policy == SsdWritePolicy::Reuse {
            let seen = self.reuse_history.get(key).is_some();
            self.reuse_history.insert(key.clone(), ());
            if !reused && !seen {
                return Some("cold");
            }
        }
        None
    }
}

pub(crate) struct SsdBackingStore {
    pub(crate) gpu_io: Arc<GpuIo>,
    /// Keeps file descriptors alive for io_uring operations.
    _files: Vec<std::fs::File>,
    cufile_files: Vec<Arc<CufileFile>>,
    io: Arc<UringIoEngine>,
    write_tx: tokio::sync::mpsc::Sender<SsdWriteCommand>,
    write_policy: SsdWritePolicy,
    prefetch_tx: tokio::sync::mpsc::Sender<PrefetchBatch>,
    inner: Mutex<SsdInner>,
    allocate_fn: AllocateFn,
    is_numa: bool,
}

impl SsdBackingStore {
    pub(crate) fn reserve_gpu(
        self: &Arc<Self>,
        key: StateKey,
        slots: Vec<crate::SlotMeta>,
    ) -> Option<GpuWriteLease> {
        if !self.gpu_io.available() {
            return None;
        }
        let mut inner = self.inner.lock();
        if self.write_policy == SsdWritePolicy::Reuse && !inner.reuse_history.contains_key(&key) {
            return None;
        }
        if inner
            .admission_skip(&key, false, self.write_policy)
            .is_some()
        {
            return None;
        }
        let encoding = if slots.iter().any(|slot| slot.encoding.is_some()) {
            index::Encoding::Encoded
        } else {
            index::Encoding::Raw
        };
        let entry = inner.ring.reserve(&key, slots, encoding)?;
        inner.pending_writes.insert(key.clone());
        core_metrics().ssd_write_inflight.add(1, &[]);
        Some(GpuWriteLease {
            entry,
            key: Some(key),
            store: Arc::clone(self),
        })
    }

    pub(super) fn new(
        config: SsdCacheConfig,
        allocate_fn: AllocateFn,
        is_numa: bool,
    ) -> std::io::Result<Arc<Self>> {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        let shards_per_path = config.shards.get();
        let total_shards = config.cache_paths.len() * shards_per_path;
        let try_gpu = config.backend != SsdBackend::Uring
            && (config.backend == SsdBackend::Cufile
                || config.capacity_bytes / total_shards.max(1) as u64 >= cufile::ALIGNMENT as u64);
        let gpu_io = Arc::new(GpuIo {
            automatic: config.backend == SsdBackend::Auto,
            enabled: AtomicBool::new(try_gpu),
        });
        let alignment = if try_gpu {
            cufile::ALIGNMENT
        } else {
            SSD_ALIGNMENT
        };
        let shard_capacity =
            files::aligned_shard_capacity(config.capacity_bytes, total_shards, alignment)?;
        let files = files::open_cache_files(
            &config.cache_paths,
            shards_per_path,
            shard_capacity,
            &mut OpenOptions::new(),
        )?;
        let fds: Vec<_> = files.iter().map(|file| file.as_raw_fd()).collect();
        let cufile_files = if try_gpu {
            let registered = files
                .iter()
                .map(|file| {
                    CufileFile::new(file.try_clone()?, Arc::clone(&gpu_io))
                        .map(Arc::new)
                        .map_err(std::io::Error::other)
                })
                .collect::<std::io::Result<Vec<_>>>();
            match registered {
                Ok(files) => files,
                Err(error) if gpu_io.automatic => {
                    gpu_io.failed(&error.to_string());
                    Vec::new()
                }
                Err(error) => return Err(error),
            }
        } else {
            Vec::new()
        };
        if !cufile_files.is_empty() {
            files::reserve_cache_space(&files, shard_capacity)?;
        }
        let ring_alignment = if gpu_io.available() {
            cufile::ALIGNMENT
        } else {
            SSD_ALIGNMENT
        };

        let io = Arc::new(UringIoEngine::new_multi(fds, UringConfig::default())?);

        let (write_tx, write_rx) = tokio::sync::mpsc::channel(config.write_queue_depth);
        let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(config.prefetch_queue_depth);

        let capacity = shard_capacity * total_shards as u64;
        let write_inflight = config.write_inflight;
        let prefetch_inflight = config.prefetch_inflight;

        info!(
            "SSD cache initialized at {} (capacity {}, shards {}, shard capacity {})",
            config
                .cache_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            ByteSize(capacity),
            total_shards,
            ByteSize(shard_capacity)
        );
        info!(
            "SSD backend requested={:?} selected={}",
            config.backend,
            if gpu_io.available() {
                "cufile"
            } else {
                "uring"
            },
        );

        let store = Arc::new(Self {
            gpu_io,
            _files: files,
            cufile_files,
            io: Arc::clone(&io),
            write_tx,
            write_policy: config.write_policy,
            prefetch_tx,
            inner: Mutex::new(SsdInner {
                ring: SsdRingBuffer::new_sharded(
                    vec![shard_capacity; total_shards],
                    ring_alignment as u64,
                ),
                pending_writes: HashSet::new(),
                reuse_history: LruCache::new(REUSE_HISTORY_BLOCKS),
            }),
            allocate_fn,
            is_numa,
        });

        Self::spawn_workers(
            &store,
            write_rx,
            prefetch_rx,
            write_inflight,
            prefetch_inflight,
        );

        Ok(store)
    }

    pub(crate) fn pin_prefix(
        self: &Arc<Self>,
        keys: &[StateKey],
        codec_budget: usize,
    ) -> Option<Vec<Arc<SsdReadLease>>> {
        if !self.gpu_io.available() {
            return None;
        }
        let inner = self.inner.lock();
        // A tiny-budget CPU codec can persist a valid representation that
        // cannot fit GPU staging. Preserve the complete prefix via io_uring.
        if keys
            .iter()
            .map_while(|key| inner.ring.get(key))
            .any(|entry| !entry.fits_gpu_decode(codec_budget))
        {
            return None;
        }
        Some(
            keys.iter()
                .map_while(|key| {
                    let entry = inner.ring.get(key)?.clone();
                    entry.readers.fetch_add(1, Ordering::Relaxed);
                    core_metrics()
                        .ssd_read_pinned_bytes
                        .add(entry.len as i64, &[]);
                    Some(Arc::new(SsdReadLease {
                        entry,
                        key: key.clone(),
                        store: Arc::clone(self),
                    }))
                })
                .collect(),
        )
    }

    pub(super) fn is_offset_valid(&self, entry: &SsdIndexEntry) -> bool {
        self.inner.lock().ring.is_offset_valid(entry)
    }

    pub(super) fn allocate_prefetch(
        &self,
        size: u64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        (self.allocate_fn)(size, numa_node)
    }

    pub(super) fn commit_write(&self, key: &StateKey, success: bool) {
        let mut inner = self.inner.lock();
        inner.ring.commit(key, success);
        inner.pending_writes.remove(key);
    }

    pub(super) fn is_numa(&self) -> bool {
        self.is_numa
    }

    fn spawn_workers(
        store: &Arc<Self>,
        write_rx: tokio::sync::mpsc::Receiver<SsdWriteCommand>,
        prefetch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
        write_inflight: usize,
        prefetch_inflight: usize,
    ) {
        let io = Arc::clone(&store.io);

        let writer_weak = Arc::downgrade(store);
        let writer_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_writer_loop(writer_weak, write_rx, writer_io, write_inflight).await;
        });

        let prefetch_weak = Arc::downgrade(store);
        let prefetch_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_prefetch_loop(prefetch_weak, prefetch_rx, prefetch_io, prefetch_inflight).await;
        });

        debug!("SSD backing store workers spawned");
    }

    /// Fire-and-forget write.
    ///
    /// Queue weak sources so waiting writes cannot prevent pressure eviction.
    /// Foreground demand may retry a selectively admitted page; speculative
    /// warming must not call this with `reused = true`.
    pub(crate) fn ingest_batch<'a>(
        &self,
        blocks: impl IntoIterator<Item = (&'a StateKey, &'a Arc<SealedBlock>)>,
        reused: bool,
    ) {
        if reused && self.write_policy == SsdWritePolicy::All {
            return;
        }
        let mut inner = self.inner.lock();
        let metrics = core_metrics();
        let mut admitted = Vec::new();
        let mut seen_batch = HashSet::new();
        for (key, block) in blocks {
            let skip = if seen_batch.insert(key) {
                inner.admission_skip(key, reused, self.write_policy)
            } else {
                Some("duplicate")
            };
            if let Some(reason) = skip {
                metrics
                    .ssd_write_admission_skips
                    .add(1, &[opentelemetry::KeyValue::new("reason", reason)]);
            } else {
                admitted.push((key.clone(), Arc::downgrade(block)));
            }
        }
        if admitted.is_empty() {
            return;
        }
        let len = admitted.len();
        match self.write_tx.try_reserve() {
            Ok(permit) => {
                inner
                    .pending_writes
                    .extend(admitted.iter().map(|(key, _)| key.clone()));
                metrics.ssd_write_queue_pending.add(len as i64, &[]);
                permit.send(SsdWriteCommand::Write(SsdWriteBatch { blocks: admitted }));
            }
            Err(_) => {
                warn!("SSD write queue full, dropping {len} blocks");
                metrics.ssd_write_queue_full.add(len as u64, &[]);
            }
        }
    }

    /// Flush the SSD writer: waits until all enqueued writes complete.
    pub(crate) async fn flush(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.write_tx.send(SsdWriteCommand::Flush(tx)).await.is_ok() {
            let _ = rx.await;
        }
    }

    pub(crate) fn contains_keys(&self, keys: &[StateKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter()
            .map(|key| inner.ring.get(key).is_some())
            .collect()
    }

    /// Count consecutive SSD-resident keys from the start of `keys`.
    pub(crate) fn prefix_len(&self, keys: &[StateKey]) -> usize {
        let inner = self.inner.lock();
        keys.iter()
            .map_while(|key| inner.ring.get(key).map(|_| ()))
            .count()
    }

    /// Submit prefix reads: scan `keys` in order, submit reads for consecutive hits, stop at first miss.
    ///
    /// Returns `(submitted, done_rx)` where `done_rx` delivers completed blocks.
    async fn submit_prefix(
        &self,
        keys: Vec<StateKey>,
    ) -> (usize, oneshot::Receiver<PrefetchResult>) {
        let (done_tx, done_rx) = oneshot::channel();

        // Prefix-scan the ring buffer: stop at first miss.
        let requests: Vec<PrefetchRequest> = {
            let inner = self.inner.lock();
            keys.into_iter()
                .map_while(|key| {
                    let entry = inner.ring.get(&key)?.clone();
                    Some(PrefetchRequest { key, entry })
                })
                .collect()
        };

        let found = requests.len();
        if found == 0 {
            let _ = done_tx.send(Vec::new());
            return (0, done_rx);
        }

        let batch = PrefetchBatch { requests, done_tx };

        if let Err(e) = self.prefetch_tx.send(batch).await {
            let batch = e.0;
            let count = batch.requests.len();
            warn!("SSD prefetch queue closed, dropping {} reads", count);
            core_metrics()
                .ssd_prefetch_queue_closed
                .add(count as u64, &[]);
            let _ = batch.done_tx.send(Vec::new());
        }

        (found, done_rx)
    }

    /// Prefetch prefix reads and await completion.
    pub(crate) async fn prefetch_prefix(&self, keys: Vec<StateKey>) -> (usize, PrefetchResult) {
        let started = std::time::Instant::now();
        let (found, done_rx) = self.submit_prefix(keys).await;
        if found == 0 {
            return (0, Vec::new());
        }

        let result = match done_rx.await {
            Ok(blocks) => (found, blocks),
            Err(_) => {
                warn!("SSD prefetch completion channel closed");
                (found, Vec::new())
            }
        };
        core_metrics()
            .ssd_prefetch_duration_seconds
            .record(started.elapsed().as_secs_f64(), &[]);
        result
    }
}

/// Creates the SSD backing store, failing startup if it cannot be initialised.
pub(crate) fn new_ssd(
    config: SsdCacheConfig,
    allocate_fn: AllocateFn,
    is_numa: bool,
) -> Arc<SsdBackingStore> {
    SsdBackingStore::new(config, allocate_fn, is_numa)
        .unwrap_or_else(|e| panic!("failed to initialise SSD backing store: {e}"))
}

#[cfg(test)]
#[path = "../../../tests/unit/backing/ssd/mod.rs"]
mod tests;
