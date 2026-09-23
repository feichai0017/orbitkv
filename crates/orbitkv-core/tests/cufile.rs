//! GPU storage qualification. Run explicitly with cuFile's native or forced
//! compatibility configuration; these tests never label compatibility as GDS.
mod common;

use common::TestEnvBuilder;
use orbitkv_core::{QueryMode, RestoreSource, SsdBackend, SsdCacheConfig, StorageConfig};

fn storage(file: std::path::PathBuf, capacity_bytes: u64) -> StorageConfig {
    StorageConfig {
        ssd_cache_config: Some(SsdCacheConfig {
            cache_paths: vec![file],
            capacity_bytes,
            backend: SsdBackend::Cufile,
            ..SsdCacheConfig::default()
        }),
        ..StorageConfig::default()
    }
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn automatic_fallback_preserves_host_layout_and_small_block_capacity() {
    let dir = tempfile::tempdir().unwrap();
    // A driver initialized by an explicit cuFile owner has not established the
    // native-only policy required by auto, even on a native-capable mount.
    let _explicit = TestEnvBuilder::new("explicit-owner", "explicit-owner")
        .layer("attention", 1, 512)
        .pool_size(64 * 1024)
        .storage(storage(dir.path().join("explicit.bin"), 4096))
        .build();
    let env = TestEnvBuilder::new("auto-fallback", "auto-fallback")
        .layer("attention", 4, 512)
        .pool_size(64 * 1024)
        .storage(StorageConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                cache_paths: vec![dir.path().join("cache.bin")],
                capacity_bytes: 4096,
                ..SsdCacheConfig::default()
            }),
            ..StorageConfig::default()
        })
        .build();
    let hashes = common::make_block_hashes(4, 71);
    env.save_all_layers_one_batch(&hashes).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    env.layers[0].data.zero_gpu();
    let result = env.query(&hashes).await;
    assert_eq!(result.blocks.len(), 4);
    assert!(
        result
            .blocks
            .iter()
            .all(|source| matches!(source, RestoreSource::Memory(_)))
    );
    let lease = env
        .engine
        .create_query_lease(&env.instance_id, result.blocks)
        .unwrap();
    env.load_to_gpu(lease, 4).await;
    env.layers[0].data.assert_gpu_matches_expected();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn fragmented_publications_seal_before_ssd_visibility_and_restore_through_cufile() {
    let dir = tempfile::tempdir().unwrap();
    let env = TestEnvBuilder::new("cufile-partial", "gds-partial")
        .layer("attention", 3, 700)
        .split_layer("split", 3, 700, 3 * 700)
        .storage(storage(dir.path().join("cache.bin"), 1 << 20))
        .build();
    let hashes = common::make_block_hashes(3, 11);
    env.save_layer(0, &hashes).await;
    env.engine.flush_all().await;
    assert!(env.query(&hashes).await.blocks.is_empty());
    env.save_layer(1, &hashes).await;
    env.engine.flush_all().await;
    assert_eq!(
        env.engine.cleanup_memory_cache().evicted_blocks,
        hashes.len()
    );
    let result = env.query(&hashes).await;
    assert_eq!(result.blocks.len(), hashes.len());
    assert!(
        result
            .blocks
            .iter()
            .all(|b| matches!(b, RestoreSource::Ssd(_)))
    );
    let lease = env
        .engine
        .create_query_lease(&env.instance_id, result.blocks)
        .unwrap();
    for layer in &env.layers {
        layer.data.zero_gpu();
    }
    env.load_to_gpu(lease, hashes.len()).await;
    for layer in &env.layers {
        layer.data.assert_gpu_matches_expected();
    }
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn ssd_sources_restore_split_and_page_first_layouts_without_host_materialization() {
    for page_first in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut builder = TestEnvBuilder::new("cufile-layout", "gds-test")
            .layer("attention", 4, 8192)
            .storage(storage(dir.path().join("cache.bin"), 1 << 20));
        if page_first {
            builder = builder.layer("compact", 4, 700).page_first();
        } else {
            builder = builder.split_layer("split", 4, 700, 4 * 700);
        }
        let env = builder.build();
        let hashes = common::make_block_hashes(4, 21);
        env.save_all_layers_one_batch(&hashes).await;
        env.engine.flush_all().await;
        assert!(env.engine.cleanup_memory_cache().evicted_blocks > 0);
        for layer in &env.layers {
            layer.data.zero_gpu();
        }
        let result = env.query(&hashes).await;
        assert_eq!(result.blocks.len(), hashes.len());
        assert!(
            result
                .blocks
                .iter()
                .all(|block| matches!(block, RestoreSource::Ssd(_)))
        );
        assert_eq!(
            env.engine.cleanup_memory_cache().evicted_blocks,
            0,
            "query must not populate DRAM"
        );
        let lease = env
            .engine
            .create_query_lease(&env.instance_id, result.blocks)
            .unwrap();
        env.load_to_gpu(lease, hashes.len()).await;
        for layer in &env.layers {
            layer.data.assert_gpu_matches_expected();
        }

        // Mix one republished DRAM block with leased SSD sources in one restore.
        env.save_all_layers_one_batch(&hashes[..1]).await;
        let mixed = env.query(&hashes).await;
        assert!(matches!(mixed.blocks[0], RestoreSource::Memory(_)));
        assert!(
            mixed.blocks[1..]
                .iter()
                .all(|block| matches!(block, RestoreSource::Ssd(_)))
        );
        let lease = env
            .engine
            .create_query_lease(&env.instance_id, mixed.blocks)
            .unwrap();
        for layer in &env.layers {
            layer.data.zero_gpu();
        }
        env.load_to_gpu(lease, hashes.len()).await;
        for layer in &env.layers {
            layer.data.assert_gpu_matches_expected();
        }

        let prepared = env
            .engine
            .count_prefix_hit_blocks_with_prefetch(
                &env.instance_id,
                "prepare",
                &hashes,
                QueryMode::Prepare,
            )
            .await
            .unwrap();
        assert_eq!(prepared.blocks.len(), hashes.len());
        assert!(
            prepared
                .blocks
                .iter()
                .all(|block| matches!(block, RestoreSource::Memory(_)))
        );
        drop(prepared);
        env.engine
            .unregister_instance_and_wait(&env.instance_id)
            .await
            .unwrap();
    }
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn canceled_disk_lease_releases_ring_capacity_and_large_checkpoint_restores() {
    let dir = tempfile::tempdir().unwrap();
    let size = 10 * 1024 * 1024;
    let env = TestEnvBuilder::new("cufile-pins", "gds-pins")
        .layer("checkpoint", 1, size)
        .pool_size(64 << 20)
        .storage(storage(dir.path().join("cache.bin"), size as u64))
        .build();
    let original = env.hashes(17);
    env.save_and_wait(&original).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let pinned = env.query(&original).await;
    assert!(matches!(pinned.blocks.as_slice(), [RestoreSource::Ssd(_)]));
    let lease = env
        .engine
        .create_query_lease(&env.instance_id, pinned.blocks)
        .unwrap();
    let replacement = env.hashes(18);
    env.save_and_wait(&replacement).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    assert_eq!(
        env.query(&original).await.blocks.len(),
        1,
        "pinned source survived attempted overwrite"
    );
    env.layers[0].data.zero_gpu();
    env.load_to_gpu(lease, 1).await;
    env.layers[0].data.assert_gpu_matches_expected();

    let canceled = env.assert_all_hit_lease(&original).await;
    env.release(&canceled);
    env.save_and_wait(&replacement).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    assert!(env.query(&original).await.blocks.is_empty());
    let ready = env.assert_all_hit_lease(&replacement).await;
    env.release(&ready);
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn short_disk_read_fails_restore_and_releases_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cache.bin");
    let env = TestEnvBuilder::new("cufile-short", "gds-short")
        .layer("layer", 1, 4096)
        .storage(storage(file.clone(), 4096))
        .build();
    let hashes = env.hashes(22);
    env.save_and_wait(&hashes).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let lease = env.assert_all_hit_lease(&hashes).await;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&file)
        .unwrap()
        .set_len(0)
        .unwrap();
    let outcome = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &[vec!["layer"]],
            &[(lease, vec![vec![Some(0)]])],
        )
        .unwrap()
        .await
        .unwrap();
    assert!(
        outcome.result.is_err(),
        "short reads must not become successful restores"
    );
    let replacement = env.hashes(23);
    env.save_and_wait(&replacement).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let next = env.assert_all_hit_lease(&replacement).await;
    env.layers[0].data.zero_gpu();
    env.load_to_gpu(next, 1).await;
    env.layers[0].data.assert_gpu_matches_expected();
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn pinned_shard_does_not_prevent_writes_to_another_shard() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = storage(dir.path().join("cache.bin"), 8192);
    config.ssd_cache_config.as_mut().unwrap().shards = std::num::NonZeroUsize::new(2).unwrap();
    let env = TestEnvBuilder::new("cufile-shards", "gds-shards")
        .layer("layer", 1, 4096)
        .storage(config)
        .build();
    let original = env.hashes(31);
    env.save_and_wait(&original).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let held = env.assert_all_hit_lease(&original).await;
    // The third write revisits shard zero, whose source is still pinned.
    for salt in [32, 33] {
        let hashes = env.hashes(salt);
        env.save_and_wait(&hashes).await;
        env.engine.flush_all().await;
        env.engine.cleanup_memory_cache();
        let found = env.assert_all_hit_lease(&hashes).await;
        env.release(&found);
    }
    env.layers[0].data.zero_gpu();
    env.load_to_gpu(held, 1).await;
    env.layers[0].data.assert_gpu_matches_expected();
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}
