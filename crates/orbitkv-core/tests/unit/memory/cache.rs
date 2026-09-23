use super::*;

#[test]
fn contains_key_does_not_bump_frequency() {
    let mut cache = TinyLfuCache::new_unbounded(1024, true, Some(1));
    let key = StateKey::new("ns".to_string(), vec![1, 2, 3, 4]);
    let value = Arc::new(SealedBlock::from_slots(Vec::new()));

    let _ = cache.insert(key.clone(), value);
    let before = cache.freq.as_ref().expect("lfu enabled").get(&key);

    assert!(cache.contains_key(&key));
    let after_contains = cache.freq.as_ref().expect("lfu enabled").get(&key);
    assert_eq!(before, after_contains);

    let _ = cache.get(&key);
    let after_get = cache.freq.as_ref().expect("lfu enabled").get(&key);
    assert!(after_get > after_contains);
}
