use super::*;

#[test]
fn overlapping_restore_targets_are_rejected_before_worker_or_codec_dispatch() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, _save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let mut layout = KVCacheLayout::new(0x10000, 257 * 4096, 257, 4096, 0, 1).unwrap();
    layout.storage_format = orbitkv_state::StorageFormat::Fp8FromBf16;
    for codec_budget in [4096, 64 * 1024 * 1024] {
        for indices in [vec![0, 0], (0..257).chain([0]).collect()] {
            let (completion, _) = oneshot::channel();
            let task = LoadTask {
                layers: vec![LayerTransferData {
                    layer_name: "attention".into(),
                    layout: layout.clone(),
                    blocks: indices
                        .into_iter()
                        .map(|block_idx| TransferBlock {
                            block_idx,
                            // Rejection must precede source access or GPU work.
                            block: TransferPayload::Pending,
                        })
                        .collect(),
                }],
                completion,
                reservations: vec![],
                codec_budget,
            };
            let error = pool.submit_load(task).unwrap_err();
            assert!(error.to_string().contains("overlap"));
            assert!(matches!(
                load_rx.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }
}

#[tokio::test]
async fn drain_rejects_new_transfers_and_waits_for_all_workers() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let (ssd_tx, mut ssd_rx) = mpsc::unbounded_channel();
    let (ssd_host_tx, mut ssd_host_rx) = mpsc::unbounded_channel();
    let (codec_write_tx, mut codec_write_rx) = mpsc::unbounded_channel();
    let pool = Arc::new(GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(Some(ssd_tx)),
        ssd_host_tx: Mutex::new(Some(ssd_host_tx)),
        codec_write_tx: Mutex::new(Some(codec_write_tx)),
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
    assert!(matches!(
        load_rx.recv().await,
        Some(WorkerCommand::Load(..))
    ));
    let Some(WorkerCommand::Drain(load_ack)) = load_rx.recv().await else {
        panic!("missing load barrier")
    };
    let Some(WorkerCommand::Drain(save_ack)) = save_rx.recv().await else {
        panic!("missing save barrier")
    };
    let Some(WorkerCommand::Drain(ssd_ack)) = ssd_rx.recv().await else {
        panic!("missing SSD barrier")
    };
    let Some(WorkerCommand::Drain(codec_write_ack)) = codec_write_rx.recv().await else {
        panic!("missing encoded writeback barrier")
    };
    let Some(WorkerCommand::Drain(ssd_host_ack)) = ssd_host_rx.recv().await else {
        panic!("missing SSD host barrier")
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
    assert!(pool.batch_save(vec![], vec![], vec![], None).await.is_err());
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
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished());
    codec_write_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "host reads must drain before mapping release"
    );
    ssd_host_ack.send(Ok(())).unwrap();
    waiter.await.unwrap().unwrap();
    pool.drain().await.unwrap();
}

#[test]
fn transfer_cost_shape_uses_logical_ranges_and_actual_encoding() {
    use crate::codec::EncodedSegment;
    use orbitkv_state::StorageFormat;

    let mut encoded = RawBlock::new(vec![]);
    encoded.encoding = Some(vec![EncodedSegment {
        version: 1,
        format: StorageFormat::Ans,
        logical_bytes: 600,
        stored_bytes: 100,
        checksum: 0,
    }]);
    let layers = vec![LayerTransferData {
        layer_name: "split".into(),
        layout: KVCacheLayout::new(0x10000, 4096, 2, 300, 2048, 2)
            .unwrap()
            .with_ssd_padding(512),
        blocks: vec![
            TransferBlock {
                block_idx: 0,
                block: TransferPayload::Owned(encoded),
            },
            TransferBlock {
                block_idx: 1,
                block: TransferPayload::Owned(RawBlock::new(vec![])),
            },
        ],
    }];
    assert_eq!(transfer_shape(&layers), (1200, 4));
    let (key, bytes) = transfer_key(&layers, 2, TransferMode::Direct, false, false);
    assert_eq!(bytes, 1200);
    assert_eq!(
        key,
        CostKey::new(CostPath::GpuDecode, 2, Representation::Mixed, 1200, 4)
    );
    let (key, _) = transfer_key(&layers, 2, TransferMode::Direct, false, true);
    assert_eq!(
        key,
        CostKey::new(CostPath::GpuSsdLoad, 2, Representation::Mixed, 1200, 4)
    );
}

#[test]
fn failed_gpu_work_is_not_reclassified_by_consumer_loss() {
    for (completed, closed, expected) in [
        (true, false, Outcome::Completed),
        (true, true, Outcome::Cancelled),
        (false, false, Outcome::Failed),
        (false, true, Outcome::Failed),
    ] {
        assert_eq!(terminal_outcome(completed, closed), expected);
    }
}
