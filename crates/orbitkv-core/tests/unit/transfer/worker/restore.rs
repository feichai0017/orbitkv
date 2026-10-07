use super::*;

#[test]
fn host_batches_preserve_store_key_and_bounded_generation_groups() {
    let keys: Vec<_> = (0..34)
        .map(|index| StateKey::new("one".into(), vec![index]))
        .collect();
    let batches = host_read_batches(keys.iter().map(|key| (1, key)));
    assert_eq!(
        batches.iter().map(Vec::len).collect::<Vec<_>>(),
        [16, 16, 2]
    );
    assert_eq!(
        batches.into_iter().flatten().collect::<Vec<_>>(),
        (0..34).collect::<Vec<_>>()
    );

    let other_namespace = StateKey::new("two".into(), keys[0].hash.clone());
    let sources = [
        (1, &keys[0]),
        (2, &keys[0]),
        (1, &keys[1]),
        (1, &keys[0]),
        (1, &other_namespace),
    ];
    assert_eq!(
        host_read_batches(sources),
        [vec![0, 2], vec![1], vec![3, 4]]
    );
    assert!(host_read_batches([]).is_empty());
}

#[test]
fn host_results_bind_reordered_keys_to_original_generation_identities() {
    let first = StateKey::new("first".into(), vec![0]);
    let second = StateKey::new("second".into(), vec![0]);
    let one = Arc::new(SealedBlock::from_slots(Vec::new()));
    let two = Arc::new(SealedBlock::from_slots(Vec::new()));
    let bound = bind_host_results(
        [(11, &first), (22, &second)],
        vec![
            (second.clone(), Arc::clone(&two)),
            (first.clone(), Arc::clone(&one)),
        ],
    )
    .unwrap();
    assert_eq!(
        bound
            .iter()
            .map(|(identity, _)| *identity)
            .collect::<Vec<_>>(),
        [11, 22]
    );
    assert!(Arc::ptr_eq(&bound[0].1, &one));
    assert!(Arc::ptr_eq(&bound[1].1, &two));
}

#[test]
fn host_results_reject_missing_duplicate_and_unexpected_blocks() {
    let key = StateKey::new("expected".into(), vec![0]);
    let other = StateKey::new("unexpected".into(), vec![0]);
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    for (name, rows) in [
        ("missing", vec![]),
        (
            "duplicate",
            vec![
                (key.clone(), Arc::clone(&block)),
                (key.clone(), Arc::clone(&block)),
            ],
        ),
        ("wrong_key", vec![(other.clone(), Arc::clone(&block))]),
        (
            "unexpected",
            vec![
                (key.clone(), Arc::clone(&block)),
                (other.clone(), Arc::clone(&block)),
            ],
        ),
    ] {
        assert!(bind_host_results([(1, &key)], rows).is_err(), "{name}");
    }
    assert!(bind_host_results([], vec![]).unwrap().is_empty());
    assert_eq!(Arc::strong_count(&block), 1);
}
