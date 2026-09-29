use super::*;

fn key(n: u32) -> StateKey {
    StateKey::new("model".into(), n.to_be_bytes().to_vec())
}

fn metadata(bytes: u64) -> orbitkv_state::ReplicaMetadata {
    orbitkv_state::ReplicaMetadata {
        medium: orbitkv_state::ReplicaMedium::Dram,
        representation: orbitkv_state::ReplicaRepresentation::Raw,
        stored_bytes: Some(bytes),
    }
}

fn ssd_metadata(bytes: u64) -> orbitkv_state::ReplicaMetadata {
    orbitkv_state::ReplicaMetadata {
        medium: orbitkv_state::ReplicaMedium::Ssd,
        representation: orbitkv_state::ReplicaRepresentation::Raw,
        stored_bytes: Some(bytes),
    }
}

#[test]
fn snapshot_with_concurrent_mutations_converges_at_the_cut() {
    let mut inventory = Inventory::new(DEFAULT_INVENTORY_JOURNAL_BYTES);
    for n in 0..1500 {
        inventory.change(&key(n), Some(metadata(u64::from(n) + 1)));
    }
    let start = inventory.sequence();
    let mut snapshot = inventory.snapshot_page(None).unwrap();
    assert_eq!(
        snapshot[1].metadata.unwrap().stored_bytes,
        Some(2),
        "resident payload evidence must survive snapshots"
    );
    let cursor = snapshot.last().unwrap().key.clone();
    inventory.change(&key(0), None);
    inventory.change(&key(1400), None);
    inventory.change(&key(1400), Some(metadata(1401)));
    let behind_cursor = StateKey::new("a-model".into(), vec![1]);
    inventory.change(&behind_cursor, Some(metadata(1)));
    snapshot.extend(inventory.snapshot_page(Some(&cursor)).unwrap());
    let end = inventory.sequence();
    let mut reconstructed: BTreeMap<_, _> =
        snapshot.into_iter().map(|r| (r.key, r.sequence)).collect();
    for update in inventory.changes(start, end).unwrap() {
        if reconstructed
            .get(&update.key)
            .is_some_and(|s| *s > update.sequence)
        {
            continue;
        }
        if update.present {
            reconstructed.insert(update.key, update.sequence);
        } else {
            reconstructed.remove(&update.key);
        }
    }
    assert_eq!(
        reconstructed,
        inventory
            .residents
            .iter()
            .map(|(key, resident)| (key.clone(), resident.sequence))
            .collect()
    );
    assert_eq!(inventory.sequence(), end);
}

#[test]
fn bounded_history_exposes_gaps_and_duplicates_do_not_advance_sequence() {
    let mut inventory = Inventory::new(300);
    inventory.change(&key(0), Some(metadata(1)));
    inventory.change(&key(0), Some(metadata(1)));
    assert_eq!(inventory.sequence(), 1);
    for n in 1..30 {
        inventory.change(&key(n), Some(metadata(u64::from(n) + 1)));
    }
    assert!(inventory.journal_bytes <= 300);
    assert!(inventory.changes(0, 30).is_err());
    assert_eq!(inventory.changes(29, 30).unwrap()[0].key, key(29));
    assert_eq!(inventory.snapshot_page(None).unwrap().len(), 30);
    assert!(inventory.changes(30, 30).unwrap().is_empty());
    assert!(inventory.changes(31, 30).is_err());
}
#[test]
fn oversized_records_cannot_escape_batch_bounds_or_be_silently_skipped() {
    let mut inventory = Inventory::new(DEFAULT_INVENTORY_JOURNAL_BYTES);
    inventory.change(
        &StateKey::new("ns".into(), vec![1; INVENTORY_BATCH_BYTES]),
        Some(metadata(1)),
    );
    assert_eq!(
        inventory.snapshot_page(None),
        Err(InventoryReadError::RecordTooLarge)
    );
    assert_eq!(
        inventory.changes(0, 1),
        Err(InventoryReadError::RecordTooLarge)
    );
}

#[test]
fn shared_residency_prefers_dram_and_falls_back_to_surviving_ssd() {
    let inventory = ResidencyInventory::new(DEFAULT_INVENTORY_JOURNAL_BYTES);
    let key = key(1);
    let shard = catalog_shard(&key);

    inventory.change(
        &key,
        orbitkv_state::ReplicaMedium::Ssd,
        Some(ssd_metadata(2048)),
    );
    assert_eq!(inventory.sequence(shard), 1);
    assert_eq!(
        inventory.page(shard, None).unwrap()[0].metadata,
        Some(ssd_metadata(2048))
    );

    inventory.change(
        &key,
        orbitkv_state::ReplicaMedium::Dram,
        Some(metadata(4096)),
    );
    assert_eq!(inventory.sequence(shard), 2);
    assert_eq!(
        inventory.page(shard, None).unwrap()[0].metadata,
        Some(metadata(4096))
    );

    inventory.change(&key, orbitkv_state::ReplicaMedium::Ssd, None);
    assert_eq!(
        inventory.sequence(shard),
        2,
        "removing a hidden SSD residence must not publish a false transition"
    );
    inventory.change(
        &key,
        orbitkv_state::ReplicaMedium::Ssd,
        Some(ssd_metadata(2048)),
    );
    assert_eq!(inventory.sequence(shard), 2);

    inventory.change(&key, orbitkv_state::ReplicaMedium::Dram, None);
    assert_eq!(inventory.sequence(shard), 3);
    assert_eq!(
        inventory.page(shard, None).unwrap()[0].metadata,
        Some(ssd_metadata(2048))
    );

    inventory.change(&key, orbitkv_state::ReplicaMedium::Ssd, None);
    assert_eq!(inventory.sequence(shard), 4);
    assert!(inventory.page(shard, None).unwrap().is_empty());
}
