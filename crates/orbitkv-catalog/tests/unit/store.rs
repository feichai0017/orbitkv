use super::*;

fn record(key: u8, sequence: u64, present: bool) -> InventoryRecord {
    InventoryRecord {
        key: StateKey::new("ns".into(), vec![key]),
        sequence,
        present,
    }
}

fn sync(
    store: &BlockHashStore,
    node: &str,
    id: Uuid,
    generation: u64,
    op: InventoryOperation,
) -> Result<(InventoryStatus, Vec<InventoryRecord>), StoreError> {
    store.sync_inventory(node, id, store.catalog_epoch(), generation, op)
}

fn seed(store: &BlockHashStore, node: &str, keys: &[u8]) -> Uuid {
    let id = Uuid::new_v4();
    store.heartbeat_node(node, id).unwrap();
    sync(
        store,
        node,
        id,
        1,
        InventoryOperation::Begin {
            sequence: keys.len() as u64,
        },
    )
    .unwrap();
    if !keys.is_empty() {
        sync(
            store,
            node,
            id,
            1,
            InventoryOperation::Snapshot {
                page: 0,
                records: keys
                    .iter()
                    .enumerate()
                    .map(|(i, key)| record(*key, i as u64 + 1, true))
                    .collect(),
            },
        )
        .unwrap();
    }
    sync(
        store,
        node,
        id,
        1,
        InventoryOperation::Commit {
            sequence: keys.len() as u64,
        },
    )
    .unwrap();
    id
}

fn assert_counts(store: &BlockHashStore) {
    let mut expected = RedundancySnapshot::default();
    for owners in &store.blocks {
        match owners.len() {
            0 => panic!("empty key retained"),
            1 => expected.keys_1 += 1,
            2 => expected.keys_2 += 1,
            3 => expected.keys_3 += 1,
            _ => expected.keys_4plus += 1,
        }
        expected.copies += owners.len() as u64;
    }
    assert_eq!(store.redundancy_snapshot(), expected);
    assert_eq!(store.owner_count(), expected.copies);
    assert_eq!(store.entry_count(), store.blocks.len() as u64);
    let bytes: usize = store
        .nodes
        .iter()
        .map(|entry| {
            let state = entry.value().lock();
            node_bytes(entry.key()) + state.bytes + operation_bytes(state.last_operation.as_ref())
        })
        .sum();
    assert_eq!(store.metadata_bytes.load(Ordering::Acquire), bytes);
    assert!(bytes <= store.config.metadata_bytes);
}

#[test]
fn all_owner_streams_share_one_budget_and_cleanup_releases_it() {
    let item = record(1, 1, true);
    let store = BlockHashStore::with_config(StoreConfig {
        metadata_bytes: node_bytes("a")
            + node_bytes("b")
            + key_bytes(&item.key)
            + 128
            + item.estimated_size(),
        ..Default::default()
    });
    let a = seed(&store, "a", &[1]);
    let b = seed(&store, "b", &[]);
    let delta = InventoryOperation::Delta {
        after: 0,
        records: vec![item],
    };
    let before = store.metadata_bytes();
    assert_eq!(
        sync(&store, "b", b, 1, delta.clone()),
        Err(StoreError::Capacity)
    );
    assert_eq!(store.metadata_bytes(), before);
    assert_counts(&store);
    store.unregister_node("a", a).unwrap();
    sync(&store, "b", b, 1, delta).unwrap();
    assert_counts(&store);
    store.unregister_node("b", b).unwrap();
    assert_eq!(store.metadata_bytes(), 0);
}

#[test]
fn snapshot_cut_preserves_newer_samples_and_replays_evictions() {
    let store = BlockHashStore::new();
    let id = Uuid::new_v4();
    store.heartbeat_node("a", id).unwrap();
    sync(
        &store,
        "a",
        id,
        1,
        InventoryOperation::Begin { sequence: 2 },
    )
    .unwrap();
    let page = InventoryOperation::Snapshot {
        page: 0,
        records: vec![record(1, 1, true), record(2, 5, true)],
    };
    let progress = sync(&store, "a", id, 1, page.clone()).unwrap().0;
    assert_eq!(sync(&store, "a", id, 1, page.clone()).unwrap().0, progress);
    assert!(prefix(&store, "ns", &[vec![1]]).is_empty());
    assert_eq!(
        sync(
            &store,
            "a",
            id,
            1,
            InventoryOperation::Commit { sequence: 2 }
        ),
        Err(StoreError::OutOfOrder)
    );
    sync(
        &store,
        "a",
        id,
        1,
        InventoryOperation::Delta {
            after: 2,
            records: vec![
                record(1, 3, false),
                record(2, 4, false),
                record(2, 5, true),
                record(3, 6, true),
            ],
        },
    )
    .unwrap();
    assert_eq!(sync(&store, "a", id, 1, page), Err(StoreError::OutOfOrder));
    sync(
        &store,
        "a",
        id,
        1,
        InventoryOperation::Commit { sequence: 6 },
    )
    .unwrap();
    assert!(prefix(&store, "ns", &[vec![1]]).is_empty());
    assert_eq!(prefix(&store, "ns", &[vec![2], vec![3]]).len(), 2);
    let remove = InventoryOperation::Delta {
        after: 6,
        records: vec![record(2, 7, false)],
    };
    sync(&store, "a", id, 1, remove.clone()).unwrap();
    sync(&store, "a", id, 1, remove).unwrap();
    assert!(prefix(&store, "ns", &[vec![2]]).is_empty());
    assert_eq!(
        sync(
            &store,
            "a",
            id,
            1,
            InventoryOperation::Begin { sequence: 2 }
        ),
        Err(StoreError::OutOfOrder)
    );
    assert_eq!(
        sync(
            &store,
            "a",
            id,
            1,
            InventoryOperation::Delta {
                after: 4,
                records: vec![record(2, 5, true)]
            }
        ),
        Err(StoreError::OutOfOrder)
    );
    // A replacement view is hidden until committed, and fences old generations.
    sync(
        &store,
        "a",
        id,
        2,
        InventoryOperation::Begin { sequence: 7 },
    )
    .unwrap();
    assert!(prefix(&store, "ns", &[vec![3]]).is_empty());
    assert_eq!(
        sync(
            &store,
            "a",
            id,
            1,
            InventoryOperation::Commit { sequence: 7 }
        ),
        Err(StoreError::OutOfOrder)
    );
    assert_counts(&store);
}

#[test]
fn malformed_batches_are_atomic_and_shard_budget_is_enforced() {
    let store = BlockHashStore::with_config(StoreConfig {
        metadata_bytes: node_bytes("a")
            + key_bytes(&record(1, 1, true).key)
            + 64
            + 2 * record(1, 1, true).estimated_size(),
        ..StoreConfig::default()
    });
    let id = seed(&store, "a", &[1]);
    let before = store.heartbeat_node("a", id).unwrap();
    for (op, error) in [
        (
            InventoryOperation::Delta {
                after: 1,
                records: vec![],
            },
            StoreError::InvalidInventory,
        ),
        (
            InventoryOperation::Delta {
                after: 1,
                records: vec![record(1, 2, false), record(2, 4, true)],
            },
            StoreError::OutOfOrder,
        ),
        (
            InventoryOperation::Delta {
                after: 1,
                records: vec![record(2, 2, true)],
            },
            StoreError::Capacity,
        ),
        (
            InventoryOperation::Delta {
                after: 1,
                records: vec![record(2, 2, true); INVENTORY_BATCH_RECORDS + 1],
            },
            StoreError::InvalidInventory,
        ),
        (
            InventoryOperation::Snapshot {
                page: 1,
                records: vec![record(2, 2, true)],
            },
            StoreError::OutOfOrder,
        ),
    ] {
        assert_eq!(sync(&store, "a", id, 1, op), Err(error));
        assert_eq!(store.heartbeat_node("a", id).unwrap(), before);
        assert_eq!(prefix(&store, "ns", &[vec![1]]).len(), 1);
    }
    sync(
        &store,
        "a",
        id,
        1,
        InventoryOperation::Delta {
            after: 1,
            records: vec![record(1, 2, false), record(2, 3, true)],
        },
    )
    .unwrap();
    assert!(prefix(&store, "ns", &[vec![1]]).is_empty());
    assert_eq!(prefix(&store, "ns", &[vec![2]]).len(), 1);
    sync(
        &store,
        "a",
        id,
        2,
        InventoryOperation::Begin { sequence: 3 },
    )
    .unwrap();
    for records in [
        vec![record(2, 2, true), record(1, 1, true)],
        vec![record(1, 0, true)],
        vec![record(1, 1, false)],
        vec![InventoryRecord {
            key: StateKey::new("ns".into(), vec![1; INVENTORY_BATCH_BYTES]),
            sequence: 1,
            present: true,
        }],
    ] {
        assert!(
            sync(
                &store,
                "a",
                id,
                2,
                InventoryOperation::Snapshot { page: 0, records }
            )
            .is_err()
        );
        assert_eq!(store.owner_count(), 0);
    }
    assert_counts(&store);
}

#[test]
fn sessions_epochs_liveness_and_cleanup_fence_old_evidence() {
    let store = BlockHashStore::new();
    let old = seed(&store, "a", &[1, 2]);
    let b = seed(&store, "b", &[2]);
    assert_eq!(
        store.heartbeat_node("a", Uuid::new_v4()),
        Err(StoreError::StaleSession)
    );
    store.node("a").unwrap().lock().last_seen -= Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
    assert!(prefix(&store, "ns", &[vec![1]]).is_empty());
    assert_eq!(
        prefix(&store, "ns", &[vec![2]])[0]
            .replicas
            .iter()
            .map(|r| r.owner.endpoint.as_str())
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    let current = Uuid::new_v4();
    assert_eq!(
        store.heartbeat_node("a", current).unwrap(),
        InventoryStatus::default()
    );
    assert_eq!(
        store.unregister_node("a", old),
        Err(StoreError::StaleSession)
    );
    assert_eq!(
        sync(
            &store,
            "a",
            old,
            1,
            InventoryOperation::Commit { sequence: 2 }
        ),
        Err(StoreError::StaleSession)
    );
    assert_eq!(
        store.sync_inventory(
            "a",
            current,
            Uuid::new_v4(),
            1,
            InventoryOperation::Begin { sequence: 0 }
        ),
        Err(StoreError::CatalogRestarted)
    );
    assert_eq!(store.owner_count(), 1);
    // Registration age is irrelevant while the owner continues heartbeating.
    assert!(store.sweep_expired().is_empty());
    store.node("b").unwrap().lock().last_seen -= store.config.ttl + Duration::from_secs(1);
    assert_eq!(
        store.sweep_expired(),
        SweepStats {
            removed_owners: 1,
            removed_keys: 1,
            removed_nodes: 1
        }
    );
    assert_eq!(store.unregister_node("b", b), Err(StoreError::UnknownNode));
    assert_eq!(store.unregister_node("a", current), Ok(0));
    assert_eq!(store.node_counts(), (0, 0));
    assert_counts(&store);
}

#[test]
fn reclaim_hints_require_two_other_live_committed_owners() {
    let store = BlockHashStore::new();
    seed(&store, "a", &[1]);
    seed(&store, "b", &[1]);
    let c = seed(&store, "c", &[]);
    let (_, hints) = sync(
        &store,
        "c",
        c,
        1,
        InventoryOperation::Delta {
            after: 0,
            records: vec![record(1, 1, true)],
        },
    )
    .unwrap();
    assert_eq!(hints, vec![record(1, 1, true)]);
    store.node("a").unwrap().lock().last_seen -= Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
    store.node("b").unwrap().lock().last_seen -= Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
    let d = seed(&store, "d", &[]);
    assert!(
        sync(
            &store,
            "d",
            d,
            1,
            InventoryOperation::Delta {
                after: 0,
                records: vec![record(1, 1, true)]
            }
        )
        .unwrap()
        .1
        .is_empty()
    );
    assert_counts(&store);
}

#[test]
fn concurrent_owner_updates_and_queries_preserve_accounting() {
    let store = BlockHashStore::new();
    let owners: Vec<_> = (0..4)
        .map(|i| {
            let node = format!("node-{i}");
            let id = seed(&store, &node, &[1]);
            (node, id)
        })
        .collect();
    std::thread::scope(|scope| {
        for (node, id) in &owners {
            let store = &store;
            scope.spawn(move || {
                for i in 0..100 {
                    let after = 1 + i * 2;
                    sync(
                        store,
                        node,
                        *id,
                        1,
                        InventoryOperation::Delta {
                            after,
                            records: vec![record(1, after + 1, false), record(1, after + 2, true)],
                        },
                    )
                    .unwrap();
                    prefix(store, "ns", &[vec![1]]);
                }
            });
        }
    });
    assert_eq!(prefix(&store, "ns", &[vec![1]])[0].replicas.len(), 4);
    assert_counts(&store);
    for (node, id) in owners {
        assert_eq!(store.unregister_node(&node, id), Ok(1));
    }
    assert_counts(&store);
    assert_eq!(store.entry_count(), 0);
}

fn prefix(store: &BlockHashStore, namespace: &str, hashes: &[Vec<u8>]) -> Vec<BlockCandidates> {
    store
        .locate_blocks(namespace, hashes, "")
        .into_iter()
        .take_while(|row| !row.replicas.is_empty())
        .collect()
}

#[test]
fn discovery_is_position_aligned_bounded_and_preserves_source_versions() {
    let store = BlockHashStore::new();
    let id = seed(&store, "a", &[1, 3]);
    for node in ["b", "c", "d", "e", "f"] {
        seed(&store, node, &[1]);
    }
    let rows = store.locate_blocks("ns", &[vec![1], vec![2], vec![3]], "b");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].replicas.len(), DISCOVERY_MAX_REPLICAS);
    assert!(rows[0].replicas.iter().all(|r| r.owner.endpoint != "b"));
    assert!(rows[1].replicas.is_empty());
    assert_eq!(
        rows[2].replicas,
        vec![ReplicaLocation {
            owner: CacheOwner {
                endpoint: "a".into(),
                incarnation: id
            },
            sequence: 2,
        }]
    );
}
