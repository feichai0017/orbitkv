use super::*;

fn make_engine() -> Arc<StorageEngine> {
    StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]).unwrap()
}

#[tokio::test]
async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
    let storage = make_engine();
    let mut hashes: HashSet<Vec<u8>> = HashSet::new();

    storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
    assert!(hashes.is_empty());
}

#[tokio::test]
async fn cleanup_memory_cache_evicts_all_resident_blocks() {
    let storage = make_engine();
    let key1 = StateKey::new("ns".into(), vec![1]);
    let key2 = StateKey::new("ns".into(), vec![2]);
    let block1 = Arc::new(SealedBlock::from_slots(Vec::new()));
    let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));

    storage
        .read_cache
        .batch_insert(vec![(key1, Arc::clone(&block1))]);
    storage.read_cache.batch_insert(vec![(key2, block2)]);

    let stats = storage.cleanup_memory_cache();
    assert_eq!(stats.evicted_blocks, 2);
    assert_eq!(stats.still_referenced_blocks, 1);
    assert_eq!(stats.reclaimed_bytes, 0);

    let stats = storage.cleanup_memory_cache();
    assert_eq!(stats, MemoryCacheCleanupStats::default());
}

#[tokio::test]
async fn allocate_bounded_reclaim_terminates() {
    // With a tiny pool, allocation of a huge block should fail fast
    // (not loop forever) thanks to MAX_RECLAIM_ROUNDS.
    let storage =
        StorageEngine::new_with_config(4096, false, StorageConfig::default(), &[]).unwrap();

    // Try to allocate more than the entire pool
    let result = storage.allocate(NonZeroU64::new(1 << 30).unwrap(), None);
    assert!(result.is_none(), "should fail, not loop forever");
}

#[tokio::test]
async fn gc_stale_inflight_returns_zero_when_empty() {
    let storage = make_engine();
    let cleaned = storage
        .gc_stale_inflight(std::time::Duration::from_secs(60))
        .await;
    assert_eq!(cleaned, 0);
}

#[tokio::test]
async fn filter_hashes_not_in_cache_removes_cached() {
    let storage = make_engine();
    let key1 = StateKey::new("ns".into(), vec![1]);
    let key2 = StateKey::new("ns".into(), vec![2]);
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));

    storage.read_cache.batch_insert(vec![(key1, block.clone())]);
    storage.read_cache.batch_insert(vec![(key2, block)]);

    let mut hashes: HashSet<Vec<u8>> = [vec![1], vec![2], vec![3]].into_iter().collect();

    storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);

    assert_eq!(hashes.len(), 1);
    assert!(hashes.contains(&vec![3]));
}

// ---- Cross-node transfer: serving side tests ----

#[tokio::test]
async fn pinned_memory_regions_returns_non_empty() {
    let storage = make_engine();
    let regions = storage.pinned_memory_regions();
    // With a 1 MiB pool, there should be at least one region
    assert!(
        !regions.is_empty(),
        "pinned_memory_regions should return at least one region"
    );
    // Each region should have a non-zero size
    for (_ptr, size) in &regions {
        assert!(*size > 0, "region size should be non-zero");
    }
}
