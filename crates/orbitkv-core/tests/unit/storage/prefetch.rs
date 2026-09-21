use super::*;

fn key(n: u8) -> StateKey {
    StateKey::new("ns".to_string(), vec![n])
}

fn block() -> Arc<SealedBlock> {
    Arc::new(SealedBlock::from_slots(Vec::new()))
}

#[test]
fn ready_result_rebuilds_prefix_in_requested_key_order() {
    let local = block();
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let b1 = block();
    let b2 = block();
    let b3 = block();

    let result = build_ready_result(
        vec![Arc::clone(&local)],
        4,
        Some(PrefetchSource::Ssd),
        &[k1.clone(), k2.clone(), k3.clone()],
        vec![
            (k2, Arc::clone(&b2)),
            (k1, Arc::clone(&b1)),
            (k3, Arc::clone(&b3)),
        ],
    );

    assert_eq!(result.ready_blocks.len(), 4);
    assert!(Arc::ptr_eq(&result.ready_blocks[0], &local));
    assert!(Arc::ptr_eq(&result.ready_blocks[1], &b1));
    assert!(Arc::ptr_eq(&result.ready_blocks[2], &b2));
    assert!(Arc::ptr_eq(&result.ready_blocks[3], &b3));
    assert_eq!(result.missing, 0);
    assert_eq!(result.cache_inserts.len(), 3);
}

#[test]
fn ready_result_stops_at_first_missing_prefetch_key() {
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let b1 = block();
    let b3 = block();

    let result = build_ready_result(
        Vec::new(),
        3,
        Some(PrefetchSource::Ssd),
        &[k1.clone(), k2, k3.clone()],
        vec![(k3, b3), (k1, Arc::clone(&b1))],
    );

    assert_eq!(result.ready_blocks.len(), 1);
    assert!(Arc::ptr_eq(&result.ready_blocks[0], &b1));
    assert_eq!(result.missing, 2);
    assert_eq!(result.cache_inserts.len(), 2);
}
