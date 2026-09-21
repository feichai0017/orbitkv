//! Measure owner cleanup with a million keys across the directory.

use criterion::{Criterion, criterion_group, criterion_main};
use orbitkv_catalog::store::{BlockHashStore, StoreConfig};
use orbitkv_state::{InventoryOperation, InventoryRecord, StateKey};
use std::time::Duration;
use uuid::Uuid;

const TOTAL_KEYS: usize = 1_000_000;
const TARGET_OWNED_KEYS: usize = 10_000;

fn populate_store() -> (BlockHashStore, String, Uuid) {
    let store = BlockHashStore::with_config(StoreConfig {
        node_stale_after: Duration::from_secs(30),
        ttl: Duration::from_secs(7_200),
        ..StoreConfig::default()
    });
    let target_node = "target-node:50055".to_string();
    let other_node = "other-node:50055".to_string();
    let target_id = Uuid::new_v4();
    let other_id = Uuid::new_v4();
    store.heartbeat_node(&target_node, target_id).unwrap();
    store.heartbeat_node(&other_node, other_id).unwrap();

    for (node, id, start, end) in [
        (&target_node, target_id, 0, TARGET_OWNED_KEYS),
        (&other_node, other_id, TARGET_OWNED_KEYS, TOTAL_KEYS),
    ] {
        let epoch = store.catalog_epoch();
        store
            .sync_inventory(
                node,
                id,
                epoch,
                1,
                InventoryOperation::Begin { sequence: 0 },
            )
            .unwrap();
        store
            .sync_inventory(
                node,
                id,
                epoch,
                1,
                InventoryOperation::Commit { sequence: 0 },
            )
            .unwrap();
        for chunk_start in (start..end).step_by(1000) {
            let after = (chunk_start - start) as u64;
            let records = (chunk_start..(chunk_start + 1000).min(end))
                .enumerate()
                .map(|(i, key)| InventoryRecord {
                    key: StateKey::new("bench".into(), (key as u64).to_le_bytes().to_vec()),
                    sequence: after + i as u64 + 1,
                    present: true,
                })
                .collect();
            store
                .sync_inventory(
                    node,
                    id,
                    epoch,
                    1,
                    InventoryOperation::Delta { after, records },
                )
                .unwrap();
        }
    }

    (store, target_node, target_id)
}

fn bench_unregister_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("unregister_node");
    group.sample_size(10);
    group.bench_function("1m_keys_10k_owned", |b| {
        b.iter_batched(
            populate_store,
            |(store, target_node, target_id)| {
                let removed = store.unregister_node(&target_node, target_id).unwrap();
                assert_eq!(removed, TARGET_OWNED_KEYS);
                store
            },
            criterion::BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_unregister_node);
criterion_main!(benches);
