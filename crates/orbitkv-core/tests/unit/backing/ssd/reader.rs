use super::*;

#[test]
fn cancelled_batch_releases_results_and_finishes_after_every_reader() {
    let (done_tx, done_rx) = oneshot::channel();
    let observation = Observation::new(
        CostKey::new(CostPath::SsdPrefetch, 99, Representation::Raw, 4096, 3),
        None,
    );
    let context = Arc::new(BatchContext::new(3, done_tx, observation, Vec::new()));
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    drop(done_rx);

    std::thread::scope(|scope| {
        for index in 0..3 {
            let context = Arc::clone(&context);
            let result = (index != 1).then(|| Arc::clone(&block));
            scope.spawn(move || {
                context.complete_one(StateKey::new("cancelled".into(), vec![index]), result);
            });
        }
    });

    assert_eq!(context.remaining.load(Ordering::Acquire), 0);
    assert!(context.failed.load(Ordering::Acquire));
    assert!(context.observation.lock().is_none());
    assert!(context.done_tx.lock().is_none());
    assert!(context.results.lock().is_empty());
    assert_eq!(Arc::strong_count(&block), 1);
}

#[tokio::test]
async fn cancelled_batch_retains_extent_until_the_last_reader_releases_its_context() {
    let (store, _queued) = super::super::tests::queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let lease = store.pin_prefix(std::slice::from_ref(&key)).pop().unwrap();
    let readers = Arc::clone(&lease.entry.readers);
    let (done_tx, done_rx) = oneshot::channel();
    let context = Arc::new(BatchContext::new(
        2,
        done_tx,
        Observation::disabled(),
        vec![lease],
    ));
    let submitted = Arc::clone(&context);
    drop(done_rx);
    context.complete_one(key.clone(), None);
    drop(context);
    assert_eq!(readers.load(Ordering::Acquire), 1);
    submitted.complete_one(key, None);
    assert_eq!(readers.load(Ordering::Acquire), 1);
    drop(submitted);
    assert_eq!(readers.load(Ordering::Acquire), 0);
}
