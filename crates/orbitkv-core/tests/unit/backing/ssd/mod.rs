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
    assert!(Arc::ptr_eq(request, &source.upgrade().unwrap()));
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
    use crate::QueryMode;
    use crate::planning::read::ReadPlan;

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
        let read = ReadPlan::new(std::slice::from_ref(&key), mode, Some(&store));
        let plan = read.deferred_ssd(&store, 64 * 1024);
        assert!(read.ssd(SsdReadPath::Uring, 64 * 1024).is_some());
        assert!(read.ssd(SsdReadPath::Cufile, 64 * 1024).is_none());
        assert_eq!(plan.is_some(), should_plan, "path={path:?}, mode={mode:?}");
        assert_eq!(version.load(Ordering::Acquire), 0);
        assert_eq!(queued.len(), 0, "planning cannot read payloads");
        if let Some(plan) = plan {
            assert_eq!(plan.path, SsdReadPath::Uring);
            let sources = plan.acquire(64 * 1024).unwrap();
            assert_eq!(version.load(Ordering::Acquire), 1);
            drop(sources);
            assert_eq!(version.load(Ordering::Acquire), 0);
        }
    }
}

#[tokio::test]
async fn ssd_plan_revalidates_versions_and_requires_complete_selected_prefixes() {
    use crate::QueryMode;
    use crate::planning::read::ReadPlan;

    let (mut store, queued) = queued_read_store();
    Arc::get_mut(&mut store).unwrap().read_path = Some(SsdReadPath::Uring);
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let missing = StateKey::new("queued-lease".into(), vec![1]);
    let keys = [key.clone(), missing.clone()];
    assert!(
        ReadPlan::new(&keys, QueryMode::WaitForFullPrefix, Some(&store))
            .deferred_ssd(&store, 64 * 1024)
            .is_none()
    );
    let partial_read = ReadPlan::new(&keys, QueryMode::Demand, Some(&store));
    let partial = partial_read.deferred_ssd(&store, 64 * 1024).unwrap();
    assert_eq!(partial.acquire(64 * 1024).unwrap().len(), 1);

    let plan_read = ReadPlan::new(&keys[..1], QueryMode::Demand, Some(&store));
    let plan = plan_read.deferred_ssd(&store, 64 * 1024).unwrap();
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
    let full_read = ReadPlan::new(&keys, QueryMode::WaitForFullPrefix, Some(&store));
    let full = full_read.deferred_ssd(&store, 64 * 1024).unwrap();
    let partial_read = ReadPlan::new(&keys, QueryMode::Demand, Some(&store));
    let partial = partial_read.deferred_ssd(&store, 64 * 1024).unwrap();
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

#[tokio::test]
async fn batched_host_reads_hold_selected_generations_and_reject_foreign_stores() {
    use crate::QueryMode;
    use crate::planning::read::ReadPlan;

    let (store, mut queued) = queued_read_store();
    let (other, other_queue) = queued_read_store();
    let keys = [
        StateKey::new("queued-lease".into(), vec![0]),
        StateKey::new("queued-lease".into(), vec![1]),
    ];
    let (slots, versions) = {
        let mut inner = store.inner.lock();
        let slots = inner.ring.get(&keys[0]).unwrap().slots.clone();
        inner.ring =
            SsdRingBuffer::new_sharded(vec![2 * SSD_ALIGNMENT as u64], SSD_ALIGNMENT as u64);
        for key in &keys {
            inner
                .ring
                .reserve(key, slots.clone(), index::Encoding::Raw)
                .unwrap();
            assert!(inner.ring.commit(key, true));
        }
        let versions: Vec<_> = keys
            .iter()
            .map(|key| inner.ring.get(key).unwrap().readers.clone())
            .collect();
        (slots, versions)
    };
    let plan = ReadPlan::new(&keys, QueryMode::Prepare, Some(&store));
    let leases = plan.ssd(SsdReadPath::Uring, 0).unwrap().acquire(0).unwrap();
    assert_eq!(leases.len(), 2);
    assert!(other.read_host_batch(leases.clone()).await.is_err());
    assert!(other_queue.is_empty());
    let mut read = Box::pin(store.read_host_batch(leases));
    assert!(futures::poll!(read.as_mut()).is_pending());
    drop(read);
    let batch = queued.recv().await.unwrap();
    assert!(batch.done_tx.is_closed());
    assert_eq!(batch.requests.len(), 2);
    for (lease, version) in batch.requests.iter().zip(&versions) {
        assert!(Arc::ptr_eq(&lease.entry.readers, version));
        assert_eq!(version.load(Ordering::Acquire), 1);
    }
    assert!(
        store
            .inner
            .lock()
            .ring
            .reserve(
                &StateKey::new("queued-lease".into(), vec![2]),
                slots,
                index::Encoding::Raw
            )
            .is_none()
    );
    drop(batch);
    assert!(
        versions
            .iter()
            .all(|version| version.load(Ordering::Acquire) == 0)
    );
}

#[cfg(feature = "mooncake")]
#[tokio::test]
async fn peer_rejection_updates_request_evidence_without_discarding_ssd_versions() {
    use crate::QueryMode;
    use crate::planning::{peer::FetchPlan, read::ReadPlan};
    use orbitkv_state::{CacheOwner, ReplicaLocation};

    let (store, queued) = queued_read_store();
    let keys = [
        StateKey::new("queued-lease".into(), vec![0]),
        StateKey::new("queued-lease".into(), vec![1]),
    ];
    let mut plan = ReadPlan::new(&keys, QueryMode::Demand, Some(&store));
    let location = ReplicaLocation {
        owner: CacheOwner {
            endpoint: "peer".into(),
            incarnation: uuid::Uuid::from_u128(1),
        },
        sequence: 7,
    };
    plan.rows[0].set_peer_dram(vec![location]);
    assert!(FetchPlan::new(&mut plan.rows, 2).is_none());
    let mut route = FetchPlan::new(&mut plan.rows, 1).unwrap();
    let segment = route.next_segment(0).unwrap();
    route.reject(0, &segment);
    assert!(route.next_segment(0).is_none());
    assert!(plan.rows[0].peer_dram().next().is_none());
    assert_eq!(
        plan.rows.len(),
        2,
        "a short peer plan must not truncate other evidence"
    );
    let leases = plan.ssd(SsdReadPath::Uring, 0).unwrap().acquire(0).unwrap();
    assert_eq!(leases.len(), 1);
    assert!(
        queued.is_empty(),
        "selection and rejection must not read payloads"
    );
}
