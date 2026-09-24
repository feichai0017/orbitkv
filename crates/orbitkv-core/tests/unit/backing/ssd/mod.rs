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
