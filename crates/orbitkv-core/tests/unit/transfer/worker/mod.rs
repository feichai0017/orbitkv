use super::*;

#[tokio::test]
async fn drain_rejects_new_transfers_and_waits_for_all_workers() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let (ssd_tx, mut ssd_rx) = mpsc::unbounded_channel();
    let pool = Arc::new(GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(Some(ssd_tx)),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    });
    let (reply, _result) = oneshot::channel();
    pool.submit_load(LoadTask {
        layers: vec![],
        completion: reply,
        reservations: vec![],
        codec_budget: 64 * 1024 * 1024,
    })
    .unwrap();
    let draining = Arc::clone(&pool);
    let waiter = tokio::spawn(async move { draining.drain().await });
    assert!(matches!(load_rx.recv().await, Some(WorkerCommand::Load(_))));
    let Some(WorkerCommand::Drain(load_ack)) = load_rx.recv().await else {
        panic!("missing load barrier")
    };
    let Some(WorkerCommand::Drain(save_ack)) = save_rx.recv().await else {
        panic!("missing save barrier")
    };
    let Some(WorkerCommand::Drain(ssd_ack)) = ssd_rx.recv().await else {
        panic!("missing SSD barrier")
    };
    let (reply, _) = oneshot::channel();
    assert!(
        pool.submit_load(LoadTask {
            layers: vec![],
            completion: reply,
            reservations: vec![],
            codec_budget: 64 * 1024 * 1024,
        })
        .is_err()
    );
    assert!(pool.batch_save(vec![], vec![], None).await.is_err());
    load_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "load completion alone must not release mappings"
    );
    save_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "SSD completion must precede mapping release"
    );
    ssd_ack.send(Ok(())).unwrap();
    waiter.await.unwrap().unwrap();
    pool.drain().await.unwrap();
}
