use super::*;
use std::fs;

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

#[test]
fn test_open_cache_files_single_path_single_shard() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cache_path = temp_dir.path().join("cache.bin");
    let mut options = std::fs::OpenOptions::new();
    let files = open_cache_files(std::slice::from_ref(&cache_path), 1, 4096, &mut options).unwrap();
    assert_eq!(files.len(), 1);
    assert!(cache_path.is_file());
    assert_eq!(fs::metadata(&cache_path).unwrap().len(), 4096);
}

#[test]
fn test_open_cache_files_single_path_multi_shard() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cache_path = temp_dir.path().join("cache");
    let mut options = std::fs::OpenOptions::new();
    let files = open_cache_files(std::slice::from_ref(&cache_path), 4, 4096, &mut options).unwrap();
    assert_eq!(files.len(), 4);
    assert!(cache_path.is_dir());
    for shard_id in 0..4 {
        let shard = cache_path.join(format!("shard-{shard_id:06}.dat"));
        assert!(shard.is_file());
        assert_eq!(fs::metadata(&shard).unwrap().len(), 4096);
    }
}

#[test]
fn test_open_cache_files_multi_path_per_path_shards() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path0 = temp_dir.path().join("ssd0");
    let path1 = temp_dir.path().join("ssd1");
    let mut options = std::fs::OpenOptions::new();
    // 2 paths * 2 shards_per_path = 4 total shards
    let files = open_cache_files(&[path0.clone(), path1.clone()], 2, 4096, &mut options).unwrap();
    assert_eq!(files.len(), 4);
    assert!(path0.is_dir());
    assert!(path1.is_dir());
    // path0 gets shards 0,1 ; path1 gets shards 2,3
    for path_id in 0..2 {
        for local_shard in 0..2 {
            let global_shard_id = path_id * 2 + local_shard;
            let expected_path = if path_id == 0 { &path0 } else { &path1 };
            let shard = expected_path.join(format!("shard-{global_shard_id:06}.dat"));
            assert!(
                shard.is_file(),
                "shard {global_shard_id} should be at {}",
                shard.display()
            );
            assert_eq!(fs::metadata(&shard).unwrap().len(), 4096);
        }
    }
}

#[test]
fn test_open_cache_files_multi_path_single_shard_uses_all_paths() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path0 = temp_dir.path().join("ssd0");
    let path1 = temp_dir.path().join("ssd1");
    let mut options = std::fs::OpenOptions::new();
    // 2 paths * 1 shard_per_path = 2 total shards, one on each path
    let files = open_cache_files(&[path0.clone(), path1.clone()], 1, 4096, &mut options).unwrap();
    assert_eq!(files.len(), 2);
    assert!(path0.is_dir());
    assert!(path1.is_dir());
    assert!(path0.join("shard-000000.dat").is_file());
    assert!(path1.join("shard-000001.dat").is_file());
}

#[test]
fn test_open_cache_files_empty_paths_fails() {
    let mut options = std::fs::OpenOptions::new();
    let result = open_cache_files(&[], 1, 4096, &mut options);
    assert!(result.is_err());
}
