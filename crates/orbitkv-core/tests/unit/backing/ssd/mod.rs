use super::*;

#[test]
fn only_automatic_failures_disable_gpu_admission() {
    for automatic in [true, false] {
        let state = GpuIo {
            automatic,
            enabled: AtomicBool::new(true),
        };
        state.failed("storage buffer registration failed");
        assert_eq!(state.available(), !automatic);
        state.failed("another submitted operation also failed");
        assert_eq!(state.available(), !automatic);
    }
}

#[test]
fn selective_writes_track_republication_without_pinning_payloads() {
    let mut inner = SsdInner {
        ring: SsdRingBuffer::new_sharded(vec![4096], 512),
        pending_writes: HashSet::new(),
        reuse_history: LruCache::new(2),
    };
    let key = StateKey::new("ns".into(), vec![1]);
    let other = StateKey::new("other".into(), vec![1]);
    assert_eq!(
        inner.admission_skip(&key, false, SsdWritePolicy::Reuse),
        Some("cold")
    );
    assert_eq!(
        inner.admission_skip(&key, false, SsdWritePolicy::Reuse),
        None
    );
    assert_eq!(
        inner.admission_skip(&other, false, SsdWritePolicy::Reuse),
        Some("cold")
    );
    let fresh = StateKey::new("ns".into(), vec![2]);
    assert_eq!(
        inner.admission_skip(&fresh, true, SsdWritePolicy::Reuse),
        None
    );
    assert_eq!(inner.reuse_history.len(), 2);
    assert_eq!(
        inner.admission_skip(&key, false, SsdWritePolicy::Reuse),
        Some("cold")
    );
    inner.pending_writes.insert(key.clone());
    assert_eq!(
        inner.admission_skip(&key, true, SsdWritePolicy::Reuse),
        Some("pending")
    );
    // A failed/drained write clears pending ownership and can be retried.
    inner.pending_writes.remove(&key);
    assert_eq!(
        inner.admission_skip(&key, true, SsdWritePolicy::Reuse),
        None
    );
    assert_eq!(
        inner.admission_skip(&other, false, SsdWritePolicy::All),
        None
    );
}

#[tokio::test]
async fn neutral_leases_read_raw_and_encoded_generations_through_uring() {
    use std::num::NonZeroU64;

    use crate::block::{RawBlock, Segment};
    use crate::memory::pool::PinnedAllocator;

    let allocator = Arc::new(PinnedAllocator::new_global(
        16 * 1024,
        1,
        false,
        true,
        NonZeroU64::new(SSD_ALIGNMENT as u64),
    ));
    for encoded in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let pool = Arc::clone(&allocator);
        let store = SsdBackingStore::new(
            SsdCacheConfig {
                cache_paths: vec![directory.path().join("cache.bin")],
                capacity_bytes: SSD_ALIGNMENT as u64,
                backend: SsdBackend::Uring,
                ..Default::default()
            },
            Arc::new(move |bytes, node| {
                pool.allocate(NonZeroU64::new(bytes)?, node.unwrap_or(NumaNode::UNKNOWN))
            }),
            false,
        )
        .unwrap();
        let data = [if encoded { 0x71 } else { 0x32 }; SSD_ALIGNMENT];
        let allocation = allocator
            .allocate(
                NonZeroU64::new(SSD_ALIGNMENT as u64).unwrap(),
                NumaNode::UNKNOWN,
            )
            .unwrap();
        // SAFETY: this unshared allocation owns the complete target range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                allocation.mapped_ptr().host().as_ptr(),
                data.len(),
            );
        }
        let mut raw = RawBlock::single_segment(Segment::new(
            allocation.mapped_ptr().host(),
            data.len(),
            allocation,
        ));
        if encoded {
            raw.encoding = Some(vec![crate::codec::EncodedSegment {
                version: 1,
                format: orbitkv_state::StorageFormat::Exact,
                logical_bytes: data.len(),
                stored_bytes: data.len(),
                checksum: crc32fast::hash(&data),
            }]);
        }
        let block = Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)]));
        let key = StateKey::new("lease-uring".into(), vec![u8::from(encoded)]);
        store.ingest_batch(std::iter::once((&key, &block)), false);
        store.flush().await;
        drop(block);

        let lease = store
            .discover(std::slice::from_ref(&key))
            .into_iter()
            .map_while(|candidate| candidate?.pin())
            .collect::<Vec<_>>()
            .pop()
            .unwrap();
        assert!(!lease.cufile_eligible(0));
        assert!(lease.file().is_err());
        assert_eq!(lease.cost_resource(), store.io.cost_resource);
        if encoded {
            assert!(!lease.entry.fits_gpu_decode(0));
        }
        let generation = lease.entry.begin;
        let readers = Arc::clone(&lease.entry.readers);
        let restored = lease.read_host().await.unwrap();
        let slot = restored.get_slot(0).unwrap();
        // SAFETY: the successful read initialized the owned segment above.
        let bytes = unsafe {
            std::slice::from_raw_parts(slot.segment_ptr(0).unwrap().as_ptr(), data.len())
        };
        assert_eq!(bytes, data);
        assert_eq!(slot.encoding.is_some(), encoded);

        let replacement = StateKey::new("lease-uring".into(), vec![2]);
        let slots = || {
            vec![crate::SlotMeta::new(
                smallvec::smallvec![SSD_ALIGNMENT as u64],
                NumaNode::UNKNOWN,
            )]
        };
        assert!(
            store
                .inner
                .lock()
                .ring
                .reserve(&replacement, slots(), index::Encoding::Raw)
                .is_none(),
            "the selected generation must remain pinned after host materialization"
        );
        drop(lease);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while readers.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed host reader released its generation");
        let next = store
            .inner
            .lock()
            .ring
            .reserve(&replacement, slots(), index::Encoding::Raw)
            .unwrap();
        assert_ne!(next.begin, generation);
    }
}

pub(super) fn queued_read_store() -> (
    Arc<SsdBackingStore>,
    tokio::sync::mpsc::Receiver<PrefetchBatch>,
) {
    use std::os::fd::AsRawFd;

    let file = tempfile::tempfile().unwrap();
    file.set_len(SSD_ALIGNMENT as u64).unwrap();
    let io = Arc::new(
        UringIoEngine::new_multi(
            vec![file.as_raw_fd()],
            UringConfig {
                threads: 2,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let (write_tx, _) = tokio::sync::mpsc::channel(1);
    let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(1);
    let store = Arc::new(SsdBackingStore {
        gpu_io: Arc::new(GpuIo {
            automatic: false,
            enabled: AtomicBool::new(false),
        }),
        read_path: None,
        _files: vec![file],
        cufile_files: Vec::new(),
        io,
        write_tx,
        write_policy: SsdWritePolicy::All,
        prefetch_tx,
        inner: Mutex::new(SsdInner {
            ring: SsdRingBuffer::new_sharded(vec![SSD_ALIGNMENT as u64], SSD_ALIGNMENT as u64),
            pending_writes: HashSet::new(),
            reuse_history: LruCache::new(1),
        }),
        allocate_fn: Arc::new(|_, _| None),
        is_numa: false,
    });
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let mut inner = store.inner.lock();
    inner
        .ring
        .reserve(
            &key,
            vec![crate::SlotMeta::new(
                smallvec::smallvec![SSD_ALIGNMENT as u64],
                NumaNode::UNKNOWN,
            )],
            index::Encoding::Raw,
        )
        .unwrap();
    assert!(inner.ring.commit(&key, true));
    drop(inner);
    (store, prefetch_rx)
}

#[tokio::test]
async fn candidates_do_not_pin_and_cannot_authorize_a_replaced_generation() {
    let (store, queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let mut candidates = store.discover(&[key.clone(), StateKey::new("ns".into(), vec![1])]);
    assert!(candidates.pop().unwrap().is_none());
    let candidate = candidates.pop().unwrap().unwrap();
    assert_eq!(candidate.entry.len, SSD_ALIGNMENT as u64);
    assert_eq!(candidate.entry.slots[0].total_size(), SSD_ALIGNMENT as u64);
    assert_eq!(candidate.entry.readers.load(Ordering::Acquire), 0);
    assert!(!candidate.cufile_eligible(64 * 1024));
    assert_eq!(queued.len(), 0, "discovery must not enqueue payload reads");
    let missing = StateKey::new("queued-lease".into(), vec![1]);
    let prefix = store.discover_prefix(&[key.clone(), missing.clone(), key.clone()]);
    assert_eq!(prefix.len(), 1);
    assert_eq!(prefix[0].entry.readers.load(Ordering::Acquire), 0);
    assert!(store.discover_prefix(&[missing, key.clone()]).is_empty());
    let lease = candidate.pin().unwrap();
    assert!(Arc::ptr_eq(&lease.entry.readers, &candidate.entry.readers));
    assert_eq!(candidate.entry.readers.load(Ordering::Acquire), 1);
    drop(lease);

    // A replacement may reuse the same key and physical offset. The old
    // snapshot must neither pin that generation nor prevent its allocation.
    let mut inner = store.inner.lock();
    let replacement = StateKey::new("queued-lease".into(), vec![2]);
    for next in [&replacement, &key] {
        assert!(
            inner
                .ring
                .reserve(next, candidate.entry.slots.clone(), index::Encoding::Raw)
                .is_some()
        );
        assert!(inner.ring.commit(next, true));
    }
    assert_eq!(
        inner.ring.get(&key).unwrap().file_offset,
        candidate.entry.file_offset
    );
    drop(inner);
    assert!(candidate.pin().is_none());
    assert_eq!(candidate.entry.readers.load(Ordering::Acquire), 0);
    let fresh = store
        .discover(std::slice::from_ref(&key))
        .pop()
        .unwrap()
        .unwrap();
    assert!(fresh.pin().is_some());
    drop(store);
    assert!(fresh.pin().is_none());
}

#[tokio::test]
async fn cancelled_host_read_keeps_the_same_generation_owned_by_its_queue() {
    let (store, mut queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let lease = store
        .discover(std::slice::from_ref(&key))
        .into_iter()
        .map_while(|candidate| candidate?.pin())
        .collect::<Vec<_>>()
        .pop()
        .unwrap();
    let readers = Arc::clone(&lease.entry.readers);
    let source = Arc::downgrade(&lease);
    let mut read = Box::pin(lease.read_host());
    assert!(futures::poll!(read.as_mut()).is_pending());
    drop(read);
    drop(lease);

    let batch = queued.recv().await.unwrap();
    assert!(batch.done_tx.is_closed());
    assert_eq!(readers.load(Ordering::Acquire), 1);
    let request = &batch.requests[0];
    assert!(Arc::ptr_eq(&request.entry.readers, &readers));
    assert!(Arc::ptr_eq(
        request.lease.as_ref().unwrap(),
        &source.upgrade().unwrap()
    ));
    drop(batch);
    assert_eq!(readers.load(Ordering::Acquire), 0);
    assert!(source.upgrade().is_none());

    queued.close();
    let lease = store
        .discover(std::slice::from_ref(&key))
        .into_iter()
        .map_while(|candidate| candidate?.pin())
        .collect::<Vec<_>>()
        .pop()
        .unwrap();
    assert!(lease.read_host().await.is_err());
    assert_eq!(readers.load(Ordering::Acquire), 1);
    drop(lease);
    assert_eq!(readers.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn ssd_planning_preserves_default_preparation_and_explicit_route_rules() {
    use crate::planning::ssd::SsdReadPlan;
    use crate::{QueryMode, RestoreSource};

    for (path, mode, should_plan) in [
        (None, QueryMode::Demand, false),
        (Some(SsdReadPath::Uring), QueryMode::Demand, true),
        (Some(SsdReadPath::Cufile), QueryMode::Demand, false),
        (Some(SsdReadPath::Uring), QueryMode::Prepare, false),
        (Some(SsdReadPath::Uring), QueryMode::Warmup, false),
    ] {
        let (mut store, queued) = queued_read_store();
        Arc::get_mut(&mut store).unwrap().read_path = path;
        let key = StateKey::new("queued-lease".into(), vec![0]);
        let version = store.inner.lock().ring.get(&key).unwrap().readers.clone();
        let plan = SsdReadPlan::discover(&store, std::slice::from_ref(&key), mode, 64 * 1024);
        assert_eq!(plan.is_some(), should_plan, "path={path:?}, mode={mode:?}");
        assert_eq!(version.load(Ordering::Acquire), 0);
        assert_eq!(queued.len(), 0, "planning cannot read payloads");
        if let Some(plan) = plan {
            let sources = plan.acquire(64 * 1024).unwrap();
            assert!(matches!(
                &sources[0],
                RestoreSource::Ssd {
                    path: SsdReadPath::Uring,
                    ..
                }
            ));
            assert_eq!(version.load(Ordering::Acquire), 1);
            drop(sources);
            assert_eq!(version.load(Ordering::Acquire), 0);
        }
    }
}

#[tokio::test]
async fn ssd_plan_revalidates_versions_and_requires_complete_selected_prefixes() {
    use crate::QueryMode;
    use crate::planning::ssd::SsdReadPlan;

    let (mut store, queued) = queued_read_store();
    Arc::get_mut(&mut store).unwrap().read_path = Some(SsdReadPath::Uring);
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let missing = StateKey::new("queued-lease".into(), vec![1]);
    let keys = [key.clone(), missing.clone()];
    assert!(
        SsdReadPlan::discover(&store, &keys, QueryMode::WaitForFullPrefix, 64 * 1024).is_none()
    );
    let partial = SsdReadPlan::discover(&store, &keys, QueryMode::Demand, 64 * 1024).unwrap();
    assert_eq!(partial.acquire(64 * 1024).unwrap().len(), 1);

    let plan = SsdReadPlan::discover(&store, &keys[..1], QueryMode::Demand, 64 * 1024).unwrap();
    let mut inner = store.inner.lock();
    let old = inner.ring.get(&key).unwrap().clone();
    for next in [&missing, &key] {
        inner
            .ring
            .reserve(next, old.slots.clone(), index::Encoding::Raw)
            .unwrap();
        assert!(inner.ring.commit(next, true));
    }
    drop(inner);
    assert!(plan.acquire(64 * 1024).is_none());
    assert_eq!(old.readers.load(Ordering::Acquire), 0);

    // A later source can disappear after discovery. Strict acquisition must
    // release earlier leases; ordinary demand may still use that prefix.
    store._files[0].set_len(2 * SSD_ALIGNMENT as u64).unwrap();
    let mut inner = store.inner.lock();
    inner.ring = SsdRingBuffer::new_sharded(vec![2 * SSD_ALIGNMENT as u64], SSD_ALIGNMENT as u64);
    for (key, encoding) in [
        (&key, index::Encoding::Raw),
        (&missing, index::Encoding::Encoded),
    ] {
        inner
            .ring
            .reserve(key, old.slots.clone(), encoding)
            .unwrap();
        assert!(inner.ring.commit(key, true));
    }
    let first = inner.ring.get(&key).unwrap().clone();
    let last = inner.ring.get(&missing).unwrap().clone();
    drop(inner);
    let full =
        SsdReadPlan::discover(&store, &keys, QueryMode::WaitForFullPrefix, 64 * 1024).unwrap();
    let partial = SsdReadPlan::discover(&store, &keys, QueryMode::Demand, 64 * 1024).unwrap();
    store.inner.lock().ring.invalidate_encoded(&missing, &last);
    assert!(full.acquire(64 * 1024).is_none());
    assert_eq!(first.readers.load(Ordering::Acquire), 0);
    let prefix = partial.acquire(64 * 1024).unwrap();
    assert_eq!(prefix.len(), 1);
    assert_eq!(first.readers.load(Ordering::Acquire), 1);
    drop(prefix);
    assert_eq!(first.readers.load(Ordering::Acquire), 0);
    assert_eq!(queued.len(), 0);
}
