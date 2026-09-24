use super::*;

fn key(value: u64) -> LookupKey {
    LookupKey {
        owner: CacheOwner {
            endpoint: "127.0.0.1:50055".into(),
            incarnation: Uuid::from_u128(1),
        },
        placement: "placement".into(),
        namespace: "ns".into(),
        hashes: vec![value.to_be_bytes().to_vec()],
    }
}

#[test]
fn lookup_identity_and_last_waiter_bound_retained_discovery_keys() {
    let pending = Arc::new(parking_lot::Mutex::new(PendingLookups::default()));
    let first = SharedLookup::acquire(&pending, key(1)).unwrap();
    let duplicate = SharedLookup::acquire(&pending, key(1)).unwrap();
    assert!(Arc::ptr_eq(&first, &duplicate));
    let bytes = pending.lock().bytes;
    assert!(bytes > 0);
    for distinct in [
        LookupKey {
            namespace: "other".into(),
            ..key(1)
        },
        LookupKey {
            placement: "new-placement".into(),
            ..key(1)
        },
        LookupKey {
            owner: CacheOwner {
                incarnation: Uuid::from_u128(2),
                ..key(1).owner
            },
            ..key(1)
        },
        key(2),
    ] {
        let different = SharedLookup::acquire(&pending, distinct).unwrap();
        assert!(!Arc::ptr_eq(&first, &different));
        drop(different);
        assert_eq!(pending.lock().bytes, bytes);
    }
    drop(first);
    assert_eq!(pending.lock().entries.len(), 1);
    drop(duplicate);
    assert!(pending.lock().entries.is_empty());
    assert_eq!(pending.lock().bytes, 0);

    let mut held = Vec::new();
    let mut value = 0;
    loop {
        let next = LookupKey {
            namespace: "n".repeat(DISCOVERY_MAX_BYTES - 8),
            ..key(value)
        };
        validate_discovery_query(&next.namespace, &next.hashes).unwrap();
        let Some(shared) = SharedLookup::acquire(&pending, next) else {
            break;
        };
        held.push(shared);
        value += 1;
        assert!(pending.lock().bytes <= CANDIDATE_CACHE_BYTES);
    }
    assert!(!held.is_empty());
    assert!(held.len() <= CANDIDATE_CACHE_BYTES / DISCOVERY_MAX_BYTES);
    drop(held.pop());
    let reopened = SharedLookup::acquire(
        &pending,
        LookupKey {
            namespace: "n".repeat(DISCOVERY_MAX_BYTES - 8),
            ..key(value)
        },
    );
    assert!(reopened.is_some());
    drop(reopened);
    drop(held);
    assert!(pending.lock().entries.is_empty());
    assert_eq!(pending.lock().bytes, 0);
}

#[test]
fn completed_lookup_cleanup_cannot_remove_a_new_generation_of_the_same_batch() {
    let pending = Arc::new(parking_lot::Mutex::new(PendingLookups::default()));
    let finished = SharedLookup::acquire(&pending, key(1)).unwrap();
    finished.remove();
    let next = SharedLookup::acquire(&pending, key(1)).unwrap();
    assert!(!Arc::ptr_eq(&finished, &next));
    drop(finished);
    assert_eq!(pending.lock().entries.len(), 1);
    assert!(Arc::ptr_eq(
        &next,
        &SharedLookup::acquire(&pending, key(1)).unwrap()
    ));
    drop(next);
    assert!(pending.lock().entries.is_empty());
    assert_eq!(pending.lock().bytes, 0);
}
