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
    use std::num::NonZeroU64;

    use crate::block::Segment;
    use crate::codec::EncodedSegment;
    use crate::memory::pool::PinnedAllocator;
    use orbitkv_state::StorageFormat;

    let pool = PinnedAllocator::new_global(4096, 1, false, false, None);
    let block = |bytes: usize| {
        let allocation = pool
            .allocate(NonZeroU64::new(bytes as u64).unwrap(), NumaNode::UNKNOWN)
            .unwrap();
        RawBlock::single_segment(Segment::new(allocation.as_non_null(), bytes, allocation))
    };
    let mut encoded = block(512);
    encoded.storage_format = StorageFormat::Ans;
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
                block: TransferPayload::Owned(block(1024)),
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

#[test]
fn raw_copy_candidates_distinguish_dma_coalescing_and_direction() {
    let mut host = [0u8; 16];
    let contiguous: Vec<_> = (0..4)
        .map(|index| CopyDesc {
            device: 0x1000 + (index * 4) as u64,
            host: host.as_mut_ptr().wrapping_add(index * 4),
            host_device: 0x2000 + (index * 4) as u64,
            size: 4,
            device_allocation: 1,
            host_allocation: 2,
        })
        .collect();
    let mut fragmented = contiguous.clone();
    fragmented.swap(1, 2);
    let mut allocations = contiguous.clone();
    for (index, copy) in allocations.iter_mut().enumerate() {
        copy.host_allocation = index;
    }
    for write in [false, true] {
        let (merged_keys, bytes) = raw_copy_keys(&contiguous, 3, write);
        assert_eq!(bytes, 16);
        let paths = if write {
            [CostPath::GpuSaveDirect, CostPath::GpuSaveKernel]
        } else {
            [CostPath::GpuLoadDirect, CostPath::GpuLoadKernel]
        };
        for (key, path) in merged_keys.iter().zip(paths) {
            assert_eq!(
                *key,
                CostKey::new(path, 3, Representation::Raw, 16, 4).with_dma_ranges(1)
            );
        }
        for copies in [&fragmented, &allocations] {
            let (keys, bytes) = raw_copy_keys(copies, 3, write);
            assert_eq!(bytes, 16);
            for ((key, merged_key), path) in keys.iter().zip(merged_keys).zip(paths) {
                assert_ne!(*key, merged_key);
                assert_eq!(
                    *key,
                    CostKey::new(path, 3, Representation::Raw, 16, 4).with_dma_ranges(4)
                );
            }
        }
        assert_ne!(merged_keys, raw_copy_keys(&contiguous, 3, !write).0);
    }
}
