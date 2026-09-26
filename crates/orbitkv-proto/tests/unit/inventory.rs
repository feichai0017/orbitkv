use super::*;

#[test]
fn inventory_wire_roundtrip_preserves_replica_evidence_and_removals() {
    for (medium, representation) in [
        (ReplicaMedium::Dram, ReplicaRepresentation::Raw),
        (ReplicaMedium::Ssd, ReplicaRepresentation::Ans),
        (ReplicaMedium::Hbm, ReplicaRepresentation::Fp8),
        (ReplicaMedium::Dram, ReplicaRepresentation::TurboQuant),
        (ReplicaMedium::Dram, ReplicaRepresentation::Mixed),
        (ReplicaMedium::Dram, ReplicaRepresentation::Unknown),
    ] {
        let record = InventoryRecord {
            key: StateKey::new("ns".into(), vec![1]),
            sequence: 7,
            present: true,
            metadata: Some(ReplicaMetadata {
                medium,
                representation,
                stored_bytes: Some(4096),
            }),
        };
        assert_eq!(
            InventoryRecord::from(wire::InventoryRecord::from(record.clone())),
            record
        );
    }
    let removal = InventoryRecord {
        key: StateKey::new("ns".into(), vec![1]),
        sequence: 8,
        present: false,
        metadata: None,
    };
    assert_eq!(
        InventoryRecord::from(wire::InventoryRecord::from(removal.clone())),
        removal
    );
}
