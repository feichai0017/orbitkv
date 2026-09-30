use super::*;
use orbitkv_state::ReplicaRepresentation;

fn key(n: u32) -> StateKey {
    StateKey::new("model".into(), n.to_le_bytes().to_vec())
}
fn metadata(medium: ReplicaMedium) -> ReplicaMetadata {
    ReplicaMetadata {
        medium,
        representation: ReplicaRepresentation::Raw,
        stored_bytes: Some(1024),
    }
}

#[test]
fn media_have_independent_generations_and_deletions_keep_identity() {
    let inventory = ResidencyInventory::new(4096);
    let k = key(1);
    inventory.change(&k, ReplicaMedium::Dram, Some(metadata(ReplicaMedium::Dram)));
    inventory.change(&k, ReplicaMedium::Ssd, Some(metadata(ReplicaMedium::Ssd)));
    let records = inventory.page(None).unwrap();
    assert_eq!(records.len(), 2);
    let dram = records
        .iter()
        .find(|r| r.metadata.unwrap().medium == ReplicaMedium::Dram)
        .unwrap();
    let ssd = records
        .iter()
        .find(|r| r.metadata.unwrap().medium == ReplicaMedium::Ssd)
        .unwrap();
    assert!(inventory.contains_record(dram, ReplicaMedium::Dram));
    assert!(!inventory.contains_record(ssd, ReplicaMedium::Dram));
    inventory.change(&k, ReplicaMedium::Dram, None);
    assert!(inventory.contains_record(ssd, ReplicaMedium::Ssd));
    let removed = inventory.changes(2, 3).unwrap();
    assert!(!removed[0].present);
    assert_eq!(removed[0].metadata.unwrap().medium, ReplicaMedium::Dram);
    inventory.change(&k, ReplicaMedium::Dram, Some(metadata(ReplicaMedium::Dram)));
    assert!(!inventory.contains_record(dram, ReplicaMedium::Dram));
}

#[test]
fn bounded_journal_reports_snapshot_gaps_and_batch_overflow() {
    let inventory = ResidencyInventory::new(300);
    for n in 0..30 {
        inventory.change(
            &key(n),
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        );
    }
    assert_eq!(
        inventory.changes(0, 30),
        Err(InventoryReadError::HistoryGap)
    );
    let status = inventory.status();
    assert_eq!(status.sequence, 30);
    assert_eq!(status.resident_records, 30);
    assert!(status.journal_records < 30);
    assert!(status.journal_bytes <= status.journal_capacity_bytes);
    assert!(status.journal_bytes_peak <= status.journal_capacity_bytes);
    assert_eq!(status.history_gaps, 1);
    assert_eq!(inventory.changes(29, 30).unwrap().len(), 1);
    assert_eq!(inventory.page(None).unwrap().len(), 30);
    inventory.change(
        &StateKey::new("m".into(), vec![1; INVENTORY_BATCH_BYTES]),
        ReplicaMedium::Ssd,
        Some(metadata(ReplicaMedium::Ssd)),
    );
    assert_eq!(
        inventory.page(None),
        Err(InventoryReadError::RecordTooLarge)
    );
}

#[tokio::test]
async fn flush_requires_a_complete_published_inventory() {
    let inventory = Arc::new(ResidencyInventory::new(4096));
    inventory.change(
        &key(1),
        ReplicaMedium::Dram,
        Some(metadata(ReplicaMedium::Dram)),
    );
    inventory.acknowledge(PublishedInventory {
        sequence: 1,
        revision: 4,
        ready: false,
    });
    let pending = tokio::spawn({
        let inventory = inventory.clone();
        async move { inventory.flush().await }
    });
    tokio::task::yield_now().await;
    assert!(!pending.is_finished());
    inventory.acknowledge(PublishedInventory {
        sequence: 1,
        revision: 5,
        ready: true,
    });
    assert_eq!(pending.await.unwrap().unwrap(), 5);
}
