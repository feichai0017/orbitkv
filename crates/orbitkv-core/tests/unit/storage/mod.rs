use super::*;

fn make_engine() -> Arc<StorageEngine> {
    StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]).unwrap()
}

#[cfg(feature = "mooncake")]
impl StorageEngine {
    pub(crate) fn with_discovery_catalog_for_test(
        read_cache: Arc<ReadCache>,
        catalog: Arc<CatalogClient>,
    ) -> Self {
        let mut storage = Arc::try_unwrap(make_engine()).ok().unwrap();
        storage.read_cache = read_cache;
        storage.catalog_client = Some(catalog);
        storage
    }
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
async fn reclaim_preserves_residents_and_rechecks_real_contiguous_capacity() {
    use crate::block::{RawBlock, Segment};

    const PAGE: usize = 64 * 1024;
    for fragmented in [false, true] {
        let storage = StorageEngine::new_with_config(
            16 * PAGE,
            false,
            StorageConfig {
                hint_value_size_bytes: Some(PAGE),
                ..StorageConfig::default()
            },
            &[],
        )
        .unwrap();
        let mut blocks = Vec::new();
        for i in 0..16 {
            let allocation = storage
                .allocate(NonZeroU64::new(PAGE as u64).unwrap(), None)
                .unwrap();
            let slot =
                RawBlock::single_segment(Segment::new(allocation.as_non_null(), PAGE, allocation));
            blocks.push((
                StateKey::new("ns".into(), vec![i]),
                Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)])),
            ));
        }
        let mut keys = blocks
            .iter()
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if fragmented {
            // First victims leave separate holes; footprint alone is not
            // proof that the allocator can satisfy a contiguous request.
            blocks.sort_by_key(|(key, _)| match key.hash[0] {
                0 => 0,
                2 => 1,
                4 => 2,
                6 => 3,
                1 => 4,
                3 => 5,
                other => other as usize + 6,
            });
            storage.read_cache.batch_insert(blocks);
        } else {
            let warm = blocks.pop().unwrap();
            warm.1.mark_warmed();
            storage.read_cache.batch_insert(blocks);
            storage.read_cache.batch_insert_reclaimable(vec![warm]);
            keys.rotate_right(1);
        }
        assert_eq!(storage.allocator.usage().0, (16 * PAGE) as u64);
        let required = if fragmented { 2 * PAGE } else { PAGE };
        let allocated = storage.allocate(NonZeroU64::new(required as u64).unwrap(), None);
        assert!(allocated.is_some(), "fragmented={fragmented}");
        let remaining = storage.read_cache.contains_keys(&keys);
        if fragmented {
            assert!(remaining.iter().filter(|present| **present).count() >= 8);
        } else {
            assert!(!remaining[0], "reclaimable warmup goes first");
            assert!(
                remaining[1..].iter().all(|present| *present),
                "one-page demand must not evict the retained cache"
            );
        }
    }
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
